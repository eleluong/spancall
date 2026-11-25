"""OpenAI-SDK LLM backend adapter (works with any OpenAI-compatible endpoint).

Unlike :class:`spancall.agent.backends.litellm_adapter.UniversalAPIBackend`,
this adapter talks directly to the ``openai`` Python SDK (already used by
``spancall/combine.py`` and ``scripts/eval_llm_local.py``), so live agentic
runs do not require ``litellm``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.backends.litellm_adapter import (
    _load_dotenv_values,
    _tool_definitions_to_openai,
)
from spancall.data.schema import ToolDefinition


def _normalize_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of ``messages`` with OpenAI-valid string ``function.arguments``.

    ``ToolCallRouter.synthesize_injected_call`` stores ``arguments`` as a dict
    for local chat-template compatibility; OpenAI-compatible HTTP APIs require
    ``tool_calls[].function.arguments`` to be a JSON-encoded string.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        msg = dict(m)
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            norm_tcs: List[Dict[str, Any]] = []
            for tc in tcs:
                tc_copy = dict(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    args = fn_copy.get("arguments")
                    if not isinstance(args, str):
                        fn_copy["arguments"] = json.dumps(args or {}, ensure_ascii=False, sort_keys=True)
                    tc_copy["function"] = fn_copy
                norm_tcs.append(tc_copy)
            msg["tool_calls"] = norm_tcs
        out.append(msg)
    return out


class OpenAIAPIBackend(BaseLLMBackend):
    """OpenAI-SDK adapter for hosted or local OpenAI-compatible endpoints."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        env_path: Union[str, Path] = ".env",
        max_context_tokens: int = 32768,
        max_tokens: Optional[int] = None,
        client: Any = None,
    ) -> None:
        env = _load_dotenv_values(env_path)
        self.model_id = (
            model_id
            or env.get("BASE_LLM_MODEL")
            or os.environ.get("BASE_LLM_MODEL")
            or ""
        )
        self.api_base = api_base or env.get("BASE_URL") or os.environ.get("BASE_URL")
        self.api_key = (
            api_key
            or env.get("API_KEY")
            or os.environ.get("API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.timeout = float(timeout)
        self.max_context_tokens = int(max_context_tokens)
        self.max_tokens = max_tokens
        self._client = client

    @classmethod
    def from_env(
        cls,
        env_path: Union[str, Path] = ".env",
        timeout: float = 120.0,
        model_id: Optional[str] = None,
    ) -> "OpenAIAPIBackend":
        return cls(model_id=model_id, env_path=env_path, timeout=timeout)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # lazy optional dependency

            self._client = OpenAI(
                base_url=self.api_base or None,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def generate_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        from spancall.agent.context import ContextBudgetError, estimate_tokens

        est = estimate_tokens(json.dumps(messages, default=str))
        if est > self.max_context_tokens:
            raise ContextBudgetError(
                f"Context budget exceeded: estimated {est} tokens > {self.max_context_tokens} max"
            )

        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": _normalize_openai_messages(messages),
            "temperature": temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = int(self.max_tokens)
        if tools:
            kwargs["tools"] = _tool_definitions_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        t0 = time.perf_counter()
        resp = self._get_client().chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        msg = resp.choices[0].message
        usage_obj = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        }

        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            return {
                "tool_name": None,
                "arguments": None,
                "text": getattr(msg, "content", None) or "",
                "usage": usage,
                "latency_ms": latency_ms,
            }

        tc = tool_calls[0]
        fn = getattr(tc, "function", None)
        raw_args = getattr(fn, "arguments", "{}") if fn is not None else "{}"
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_raw": raw_args}
        elif isinstance(raw_args, dict):
            arguments = dict(raw_args)
        else:
            arguments = {}

        return {
            "tool_name": getattr(fn, "name", None) if fn is not None else None,
            "arguments": arguments,
            "call_id": getattr(tc, "id", None),
            "text": getattr(msg, "content", None) or "",
            "usage": usage,
            "latency_ms": latency_ms,
        }

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        res = self.generate_tool_call(messages, tools=[], temperature=temperature)
        return str(res.get("text") or "")
