"""Public one-call API for combining SpanCall with any LLM.

Quick start::

    from spancall import SpanCallLLM

    caller = SpanCallLLM.from_pretrained(
        "convoicon/spancall-lean-v3-3",
        policy="router_schema",              # see CombinePolicy
        base_url="https://api.openai.com/v1",  # any OpenAI-compatible endpoint
        api_key="sk-...",
        model="gpt-4o-mini",
    )
    out = caller("Flights from Hanoi to Da Nang tomorrow, direct only",
                 tools=[...], now="2026-10-23")
    print(out.tool_name, out.arguments, out.source)   # source: spancall | llm | spancall+llm_fill

Policies (all zero-training, matching the paper):

=====================  ========================================================
``llm_only``           LLM answers every turn (baseline).
``spancall_only``      SpanCall only; abstains are returned as ``hit=False``.
``router``             SpanCall first; full LLM call only on abstention.
``router_schema``      Like ``router`` but tools whose schema declares an
                       ``array``/``object`` parameter go to the LLM (decided
                       after H1 picks the tool; uses no labels).
``hybrid_fill``        On abstention keep H1's tool + grounded scalars and ask
                       the LLM (plain JSON, no ``tools=``) for the rest.
``hybrid_fill_schema`` Same, but also for complex-schema tools.
=====================  ========================================================

This module only orchestrates; the routing/merge logic is shared with the
evaluation code in :mod:`spancall.eval.hybrid_fill`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from spancall.data.schema import ParameterSchema, ToolDefinition, UnifiedSample
from spancall.eval.hybrid_fill import (
    build_hybrid_fill_request,
    call_json_fill_sdk,
    finalize_hybrid_fill_row,
    find_tool_schema,
    is_complex_tool,
    plan_hybrid_fill,
)
from spancall.eval.llm_parity import (
    LLMResponse,
    call_openai_sdk,
    llm_pred_row,
    messages_for,
    openai_tools,
)

__all__ = [
    "CombinePolicy",
    "ToolCallOutcome",
    "SpanCallLLM",
    "coerce_tools",
    "make_openai_client",
]

ToolLike = Union[ToolDefinition, Mapping[str, Any]]


class CombinePolicy(str, Enum):
    LLM_ONLY = "llm_only"
    SPANCALL_ONLY = "spancall_only"
    ROUTER = "router"
    ROUTER_SCHEMA = "router_schema"
    HYBRID_FILL = "hybrid_fill"
    HYBRID_FILL_SCHEMA = "hybrid_fill_schema"


@dataclass
class ToolCallOutcome:
    """Result of one turn. ``arguments`` is a schema-valid dict when ``hit``."""

    hit: bool
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    #: ``spancall`` | ``llm`` | ``spancall+llm_fill`` | ``none``
    source: str = "none"
    policy: str = ""
    llm_called: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    spancall_ms: float = 0.0
    reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def as_tool_call(self) -> Dict[str, Any]:
        """OpenAI-style ``{"name", "arguments"}`` dict (arguments is a dict)."""
        return {"name": self.tool_name, "arguments": dict(self.arguments or {})}


# --------------------------------------------------------------------------- #
# Input coercion
# --------------------------------------------------------------------------- #
_JSON_TYPES = {"float": "number", "int": "integer", "bool": "boolean", "str": "string"}


def _json_type(spec: Mapping[str, Any]) -> str:
    t = spec.get("type", "string")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    t = _JSON_TYPES.get(str(t), str(t))
    if t == "string" and spec.get("format") in ("date", "date-time"):
        return "date"
    return t


def _coerce_tool(t: ToolLike) -> ToolDefinition:
    if isinstance(t, ToolDefinition):
        return t
    fn = t.get("function", t) if isinstance(t, Mapping) else None
    if not isinstance(fn, Mapping) or not fn.get("name"):
        raise TypeError("tool must be a ToolDefinition or an OpenAI-style function dict")
    raw = fn.get("parameters") or []
    params: List[ParameterSchema] = []
    if isinstance(raw, Mapping):  # JSON Schema
        required = set(raw.get("required") or [])
        for name, spec in (raw.get("properties") or {}).items():
            spec = spec or {}
            items = spec.get("items") or {}
            params.append(
                ParameterSchema(
                    name=str(name),
                    description=str(spec.get("description", "")),
                    param_type=_json_type(spec),
                    required=name in required,
                    enum=list(spec["enum"]) if spec.get("enum") is not None else None,
                    default=spec.get("default"),
                    items_type=items.get("type") if isinstance(items, Mapping) else None,
                )
            )
    else:  # SpanCall-native list of param dicts
        params = [ParameterSchema.from_dict(p) for p in raw]
    return ToolDefinition(
        name=str(fn["name"]), description=str(fn.get("description", "")), parameters=params
    )


def coerce_tools(tools: Sequence[ToolLike]) -> List[ToolDefinition]:
    """Accept ``ToolDefinition`` or OpenAI-style function dicts."""
    return [_coerce_tool(t) for t in tools]


def make_openai_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> Any:
    """Build an OpenAI-SDK client; falls back to ``BASE_URL`` / ``API_KEY`` env vars."""
    from openai import OpenAI  # lazy: optional dependency

    return OpenAI(
        base_url=base_url or os.environ.get("BASE_URL") or None,
        api_key=api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _lean_row(res: Any, sid: str) -> Dict[str, Any]:
    calls = [c.as_dict() if hasattr(c, "as_dict") else c for c in (getattr(res, "calls", None) or [])]
    hit = bool(getattr(res, "hit", False))
    return {
        "sample_id": sid,
        "hit": hit,
        "status": getattr(res, "status", "hit" if hit else "abstain"),
        "tool_name": getattr(res, "tool_name", None),
        "payload": getattr(res, "payload", None),
        "payload_partial": dict(getattr(res, "payload_partial", None) or {}),
        "calls": calls,
        "missing_required": list(getattr(res, "missing_required", None) or []),
        "abstain_reason": getattr(res, "abstain_reason", None),
        "confidence": float(getattr(res, "confidence", 0.0) or 0.0),
        "joint_margin": float(getattr(res, "joint_margin", 0.0) or 0.0),
        "latency_ms": float(getattr(res, "latency_ms", 0.0) or 0.0),
    }


class SpanCallLLM:
    """Combine a SpanCall resolver with an OpenAI-compatible LLM under a policy."""

    def __init__(
        self,
        resolver: Any = None,
        llm: Any = None,
        model: Optional[str] = None,
        policy: Union[CombinePolicy, str] = CombinePolicy.ROUTER_SCHEMA,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        fill_max_tokens: Optional[int] = 10000,
        timeout: float = 120.0,
        max_retries: int = 4,
        reasoning_effort: Optional[str] = "medium",
        json_mode: bool = True,
        fill_optional: bool = False,
    ) -> None:
        self.resolver = resolver
        self.llm = llm
        self.model = model
        self.policy = CombinePolicy(policy)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fill_max_tokens = fill_max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.fill_optional = fill_optional

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        *,
        llm: Any = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        policy: Union[CombinePolicy, str] = CombinePolicy.ROUTER_SCHEMA,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> "SpanCallLLM":
        """Load a SpanCall checkpoint (local dir or Hub repo) and attach an LLM.

        Pass ``llm`` (an OpenAI-SDK client) or ``base_url``/``api_key``; with
        neither, ``BASE_URL``/``API_KEY``/``BASE_LLM_MODEL`` env vars are used
        when the policy needs an LLM.
        """
        from spancall.hub import load_resolver

        pol = CombinePolicy(policy)
        resolver = (
            None
            if pol is CombinePolicy.LLM_ONLY
            else load_resolver(path_or_repo, device=device, token=hf_token)
        )
        if pol is not CombinePolicy.SPANCALL_ONLY:
            if llm is None:
                llm = make_openai_client(base_url, api_key)
            if model is None:
                model = os.environ.get("BASE_LLM_MODEL") or None
        return cls(resolver, llm, model, pol, **kwargs)

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        utterance: str,
        tools: Sequence[ToolLike],
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Union[str, date]] = None,
        policy: Optional[Union[CombinePolicy, str]] = None,
    ) -> ToolCallOutcome:
        pol = CombinePolicy(policy) if policy is not None else self.policy
        defs = coerce_tools(tools)
        now_iso = now.isoformat() if isinstance(now, date) else (now or date.today().isoformat())
        sid = "live"
        sample = UnifiedSample(
            sample_id=sid,
            source_dataset="live",
            utterance=utterance,
            tools=defs,
            ground_truth_calls=[],
            recent_context=list(recent_context or []),
            current_time_iso=now_iso,
        )

        if pol is CombinePolicy.LLM_ONLY:
            return self._full_llm(sample, pol, None)

        if self.resolver is None:
            raise ValueError(f"policy {pol.value!r} needs a SpanCall resolver")
        res = self.resolver.try_resolve_candidates(
            defs, utterance, recent_context=recent_context, now=now_iso
        )
        lean = _lean_row(res, sid)

        if pol is CombinePolicy.SPANCALL_ONLY:
            return self._outcome(lean, pol, "spancall" if lean["hit"] else "none", lean)

        if pol in (CombinePolicy.ROUTER, CombinePolicy.ROUTER_SCHEMA):
            complex_tool = pol is CombinePolicy.ROUTER_SCHEMA and is_complex_tool(
                find_tool_schema(sample, lean["tool_name"])
            )
            if lean["hit"] and not complex_tool:
                return self._outcome(lean, pol, "spancall", lean)
            return self._full_llm(sample, pol, lean)

        # hybrid_fill / hybrid_fill_schema
        plan = plan_hybrid_fill(
            sample, lean, mode=pol.value, fill_optional=self.fill_optional
        )
        if plan.action == "spancall_hit":
            return self._outcome(lean, pol, "spancall", lean)
        if plan.action == "full_fallback":
            return self._full_llm(sample, pol, lean)
        messages, _ = build_hybrid_fill_request(sample, plan)
        resp = call_json_fill_sdk(
            self._client(),
            self._model(),
            messages,
            temperature=self.temperature,
            max_tokens=self.fill_max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            json_mode=self.json_mode,
            reasoning_effort=self.reasoning_effort,
        )
        row = finalize_hybrid_fill_row(plan, lean, resp, None, model=self.model)
        return self._outcome(row, pol, "spancall+llm_fill" if row.get("hit") else "none", lean, llm=True)

    # ------------------------------------------------------------------ #
    def _client(self) -> Any:
        if self.llm is None:
            raise ValueError("this policy needs an LLM: pass `llm=` (OpenAI client) or use from_pretrained(base_url=...)")
        return self.llm

    def _model(self) -> str:
        if not self.model:
            raise ValueError("`model` (LLM model name) is required for this policy")
        return self.model

    def _full_llm(
        self, sample: UnifiedSample, pol: CombinePolicy, lean: Optional[Dict[str, Any]]
    ) -> ToolCallOutcome:
        resp: LLMResponse = call_openai_sdk(
            self._client(),
            self._model(),
            messages_for(sample),
            openai_tools(sample),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        row = llm_pred_row(sample.sample_id, resp, model=self.model)
        lean_ms = float((lean or {}).get("latency_ms") or 0.0)
        row["latency_ms"] = lean_ms + float(resp.latency_ms or 0.0)
        return self._outcome(row, pol, "llm" if row["hit"] else "none", lean or {}, llm=True)

    @staticmethod
    def _outcome(
        row: Dict[str, Any],
        pol: CombinePolicy,
        source: str,
        lean: Dict[str, Any],
        llm: bool = False,
    ) -> ToolCallOutcome:
        feats = row.get("features") or {}
        hit = bool(row.get("hit"))
        return ToolCallOutcome(
            hit=hit,
            tool_name=row.get("tool_name") if hit else None,
            arguments=dict(row["payload"]) if hit and row.get("payload") is not None else None,
            source=source,
            policy=pol.value,
            llm_called=llm,
            prompt_tokens=int(feats.get("prompt_tokens") or 0),
            completion_tokens=int(feats.get("completion_tokens") or 0),
            latency_ms=float(row.get("latency_ms") or 0.0),
            spancall_ms=float(lean.get("latency_ms") or 0.0),
            reason=None if hit else (row.get("abstain_reason") or feats.get("llm_error")),
            raw=row,
        )
