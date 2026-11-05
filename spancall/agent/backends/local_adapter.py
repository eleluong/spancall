"""Local LLM backend adapter (vLLM / HuggingFace Transformers).

Both ``vllm`` and ``transformers`` are **optional** dependencies (not in
``pyproject.toml``) — the adapter lazy-imports them and raises ``ImportError``
only when a method is actually called, so package import and unit tests never
require the heavy ML stack.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from spancall.agent.backends.base import BaseLLMBackend
from spancall.data.schema import ToolDefinition

_STR_OPEN = '<|"|>'
_STR_CLOSE = '<|"|>'


def _import_vllm() -> Any:
    try:
        import vllm  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "LocalVLLMBackend requires the optional 'vllm' package. "
            "Install it with: pip install vllm"
        ) from exc
    return vllm


def _render_prompt(
    messages: List[Dict[str, Any]],
    chat_template_path: Optional[str],
) -> str:
    """Render ``messages`` through the vendored Gemma-4 chat template.

    When ``chat_template_path`` is None the raw last user message is used
    (ablation / smoke only) so the class still works without a template file.
    """
    if not chat_template_path:
        for m in reversed(messages):
            if m.get("role") == "user" and m.get("content"):
                return str(m["content"])
        return ""
    from jinja2 import Environment, FileSystemLoader  # jinja2 is a hard dep

    template_dir = str(chat_template_path.rsplit("/", 1)[0]) or "."
    template_name = chat_template_path.rsplit("/", 1)[-1]
    env = Environment(
        loader=FileSystemLoader(template_dir), trim_blocks=False, lstrip_blocks=False
    )
    tpl = env.get_template(template_name)
    return tpl.render(messages=messages, add_generation_prompt=True)


class LocalVLLMBackend(BaseLLMBackend):
    """Local inference engine using vLLM for high-throughput evaluation.

    ``generate_tool_call`` runs greedy constrained decoding and parses the
    native tool-call form from ``chat_template_gemma4.jinja``:
    ``<|tool_call>call:NAME{key:value,...}<tool_call|>``. ``value`` uses the
    template's own serialization (strings wrapped in ``<|"|>``, unquoted
    keys), which :func:`_parse_tool_call` round-trips back to Python dicts.
    """

    def __init__(
        self,
        model_path: str,
        chat_template_path: Optional[str] = None,
        max_new_tokens: int = 1024,
        dtype: str = "bfloat16",
        max_context_tokens: int = 32768,
    ) -> None:
        self.model_path = model_path
        self.chat_template_path = chat_template_path
        self.max_new_tokens = int(max_new_tokens)
        self.dtype = dtype
        self.max_context_tokens = int(max_context_tokens)
        self._engine: Optional[Any] = None

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        vllm = _import_vllm()
        self._engine = vllm.LLM(
            model=self.model_path,
            dtype=self.dtype,
            max_model_len=self.max_context_tokens,
        )
        return self._engine

    def generate_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        from vllm import SamplingParams  # type: ignore[import-not-found]
        from spancall.agent.context import ContextBudgetError, estimate_tokens
        import json

        est = estimate_tokens(json.dumps(messages, default=str))
        if est > self.max_context_tokens:
            raise ContextBudgetError(
                f"Context budget exceeded: estimated {est} tokens > {self.max_context_tokens} max"
            )

        prompt = _render_prompt(messages, self.chat_template_path)
        t0 = time.perf_counter()
        params = SamplingParams(temperature=temperature, max_tokens=self.max_new_tokens)
        out = self._get_engine().generate(prompt, params)
        text = out[0].outputs[0].text
        latency_ms = (time.perf_counter() - t0) * 1000
        prompt_tokens = len(getattr(out[0], "prompt_token_ids", []) or [])
        completion_tokens = len(getattr(out[0].outputs[0], "token_ids", []) or [])

        call = _parse_tool_call(text)
        if call is None:
            return {
                "tool_name": None,
                "arguments": None,
                "text": text,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "latency_ms": latency_ms,
            }
        tool_name, parsed = call
        arguments: Dict[str, Any]
        if isinstance(parsed, dict):
            arguments = parsed
        else:
            arguments = {"_raw": str(parsed)}
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "latency_ms": latency_ms,
        }

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        from vllm import SamplingParams  # type: ignore[import-not-found]
        from spancall.agent.context import ContextBudgetError, estimate_tokens
        import json

        est = estimate_tokens(json.dumps(messages, default=str))
        if est > self.max_context_tokens:
            raise ContextBudgetError(
                f"Context budget exceeded: estimated {est} tokens > {self.max_context_tokens} max"
            )

        prompt = _render_prompt(messages, self.chat_template_path)
        params = SamplingParams(temperature=temperature, max_tokens=self.max_new_tokens)
        out = self._get_engine().generate(prompt, params)
        return out[0].outputs[0].text


# --------------------------------------------------------------------------- #
# Native Gemma-4 tool-call parsing (mirror of chat_template_gemma4.jinja)
# --------------------------------------------------------------------------- #


def _coerce_number(tok: str) -> Any:
    try:
        if "." in tok or "e" in tok.lower():
            return float(tok)
        return int(tok)
    except ValueError:
        return tok


def _parse_gemma_value(text: str) -> Tuple[Any, str]:
    """Parse one value in the template's serialization; return ``(value, rest)``."""
    text = text.lstrip()
    if not text:
        return None, ""
    if text.startswith(_STR_OPEN):
        end = text.find(_STR_CLOSE, len(_STR_OPEN))
        if end < 0:
            return text[len(_STR_OPEN):], ""
        return text[len(_STR_OPEN):end], text[end + len(_STR_CLOSE):]
    if text.startswith("{"):
        return _parse_object(text)
    if text.startswith("["):
        return _parse_array(text)
    m = re.match(r"[^,}\]\s]+", text)
    tok = m.group(0) if m else ""
    rest = text[len(tok):]
    if tok == "true":
        return True, rest
    if tok == "false":
        return False, rest
    if tok == "null":
        return None, rest
    return _coerce_number(tok), rest


def _parse_object(text: str) -> Tuple[Dict[str, Any], str]:
    rest = text[1:].lstrip()
    out: Dict[str, Any] = {}
    if rest.startswith("}"):
        return out, rest[1:]
    while True:
        if rest.startswith(_STR_OPEN):
            end = rest.find(_STR_CLOSE, len(_STR_OPEN))
            key = rest[len(_STR_OPEN):end]
            rest = rest[end + len(_STR_CLOSE):]
        else:
            m = re.match(r"[^:,}]+", rest)
            key = (m.group(0).strip() if m else "")
            rest = rest[len(m.group(0)):] if m else rest
        rest = rest.lstrip()
        if rest.startswith(":"):
            rest = rest[1:]
        value, rest = _parse_gemma_value(rest)
        out[key] = value
        rest = rest.lstrip()
        if rest.startswith("}"):
            return out, rest[1:]
        if rest.startswith(","):
            rest = rest[1:].lstrip()


def _parse_array(text: str) -> Tuple[List[Any], str]:
    rest = text[1:].lstrip()
    out: List[Any] = []
    if rest.startswith("]"):
        return out, rest[1:]
    while True:
        value, rest = _parse_gemma_value(rest)
        out.append(value)
        rest = rest.lstrip()
        if rest.startswith("]"):
            return out, rest[1:]
        if rest.startswith(","):
            rest = rest[1:].lstrip()


def _parse_tool_call(text: str) -> Optional[Tuple[str, Any]]:
    """Extract ``(name, parsed_args)`` from one native ``<|tool_call>`` block.

    Handles the template's exact serialization (``call:NAME{...}`` with
    unquoted keys and ``<|"|>``-wrapped strings), returning the *decoded*
    arguments (a dict for ``{...}`` bodies). Returns ``None`` when no tool
    call marker is present (assistant prose response).
    """
    marker = "<|tool_call>call:"
    start = text.find(marker)
    if start < 0:
        return None
    body = text[start + len(marker):]
    end = body.find("<tool_call|>")
    if end < 0:
        return None
    body = body[:end].lstrip()
    brace = body.find("{")
    if brace < 0:
        return body.strip(), ""
    name = body[:brace].strip()
    parsed, rest = _parse_gemma_value(body[brace:])
    return name, parsed if not rest.strip() else json.dumps(parsed)
