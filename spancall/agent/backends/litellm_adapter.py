"""Universal API LLM adapter (via LiteLLM / OpenAI compatible endpoints).

``litellm`` is an **optional** dependency (not in ``pyproject.toml``) — the
adapter degrades gracefully: constructing it raises ``ImportError`` only when
a method is actually called (lazy import), so package import and router/loop
unit tests never require the SDK.

Environment configuration: a repository-root ``.env`` file (or the process
environment) may supply ``BASE_URL``, ``API_KEY`` and ``BASE_LLM_MODEL`` so a
live endpoint can be used without hard-coding credentials.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from spancall.agent.backends.base import BaseLLMBackend
from spancall.data.schema import ToolDefinition


def _load_dotenv_values(path: Union[str, Path] = ".env") -> Dict[str, str]:
    """Minimal ``.env`` parser (no python-dotenv dependency).

    Later lines win over earlier ones; exported ``export KEY=...`` prefixes
    and surrounding quotes are handled. Missing files return ``{}``.
    """
    p = Path(path)
    out: Dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def _import_litellm() -> Any:
    try:
        import litellm  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "UniversalAPIBackend requires the optional 'litellm' package. "
            "Install it with: pip install litellm"
        ) from exc
    return litellm


def _tool_definitions_to_openai(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
    """Convert SpanCall ``ToolDefinition``s into OpenAI function schemas."""
    out: List[Dict[str, Any]] = []
    for t in tools:
        props: Dict[str, Any] = {}
        required: List[str] = []
        for p in t.parameters:
            ptype = "string" if p.param_type == "string" else p.param_type
            spec: Dict[str, Any] = {"type": ptype, "description": p.description}
            if p.enum:
                spec["enum"] = list(p.enum)
            if p.items_type:
                spec["items"] = {"type": "string" if p.items_type == "string" else p.items_type}
            props[p.name] = spec
            if p.required:
                required.append(p.name)
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            }
        )
    return out


class UniversalAPIBackend(BaseLLMBackend):
    """Adapter for hosted LLMs (Gemma, Claude, GPT, Mistral, etc.).

    Uses ``litellm.completion`` with OpenAI-style tool schemas, so any
    provider with a compatible tool-calling endpoint works. ``api_base`` may
    be an OpenAI-compatible proxy URL (vLLM, llama.cpp, etc.).

    Constructor arguments win over a repository-root ``.env`` file, which
    wins over the process environment: ``api_base`` / ``BASE_URL``,
    ``api_key`` / ``API_KEY``, ``model_id`` / ``BASE_LLM_MODEL``.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        env_path: Union[str, Path] = ".env",
        max_context_tokens: int = 32768,
    ) -> None:
        env = _load_dotenv_values(env_path)
        self.model_id = model_id or env.get("BASE_LLM_MODEL") or os.environ.get("BASE_LLM_MODEL") or ""
        self.api_base = api_base or env.get("BASE_URL") or os.environ.get("BASE_URL")
        self.api_key = api_key or env.get("API_KEY") or os.environ.get("API_KEY")
        self.timeout = timeout
        self.max_context_tokens = int(max_context_tokens)

    @classmethod
    def from_env(
        cls,
        env_path: Union[str, Path] = ".env",
        timeout: float = 120.0,
    ) -> "UniversalAPIBackend":
        """Build a backend solely from ``.env`` / environment configuration."""
        return cls(model_id=None, api_base=None, api_key=None, timeout=timeout, env_path=env_path)

    def generate_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        from spancall.agent.context import ContextBudgetError, estimate_tokens
        import json
        est = estimate_tokens(json.dumps(messages, default=str))
        if est > self.max_context_tokens:
            raise ContextBudgetError(
                f"Context budget exceeded: estimated {est} tokens > {self.max_context_tokens} max"
            )

        litellm = _import_litellm()
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if tools:
            kwargs["tools"] = _tool_definitions_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        t0 = time.perf_counter()
        resp = litellm.completion(**kwargs)
        message = resp["choices"][0]["message"]
        usage = resp.get("usage") or {}

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {
                "tool_name": None,
                "arguments": None,
                "text": message.get("content") or "",
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                },
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }

        tc = tool_calls[0]
        fn = tc.get("function") or {}
        arguments = fn.get("arguments")
        if isinstance(arguments, str):
            import json

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        return {
            "tool_name": fn.get("name"),
            "arguments": dict(arguments or {}),
            "call_id": tc.get("id"),
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        from spancall.agent.context import ContextBudgetError, estimate_tokens
        import json
        est = estimate_tokens(json.dumps(messages, default=str))
        if est > self.max_context_tokens:
            raise ContextBudgetError(
                f"Context budget exceeded: estimated {est} tokens > {self.max_context_tokens} max"
            )

        litellm = _import_litellm()
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        resp = litellm.completion(**kwargs)
        return resp["choices"][0]["message"].get("content") or ""
