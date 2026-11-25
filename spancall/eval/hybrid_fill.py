"""Lightweight JSON Hybrid-Fill evaluation helpers for SpanCall Lean v3.3.

Design principle:
- **100% of tool-name selection is offloaded to SpanCall (H1)** — the LLM never
  sees distractor tools and never uses the heavy OpenAI ``tools=[...]`` wrapper
  (which adds ~200-250 hidden system/schema tokens per call).
- **Grounded scalar parameters are kept from SpanCall** — the LLM is called via
  a minimal plain-text JSON prompt (``tools=[]``) to generate ONLY the missing,
  complex (``array``/``object``), or low-confidence parameters for the already
  selected tool.
- **Single-shot, zero double-fallback** — if SpanCall named a tool, at most one
  lightweight JSON call is made; its parsed fields are merged conservatively
  with SpanCall's grounded scalars without ever triggering a second full LLM call.

Two policies are supported:
1. ``hybrid_fill``: Clean SpanCall hits use 0 LLM tokens; abstentions with a
   predicted tool fill only the missing/uncertain parameters via plain JSON.
2. ``hybrid_fill_schema``: On ``H1``-selected tools declaring ``array``/``object``
   parameters, SpanCall keeps its grounded scalar arguments and the LLM fills
   only the complex (+ missing required) parameters via plain JSON.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from spancall.agent.fill import merge_filled_arguments, unresolved_param_names
from spancall.data.schema import ParameterSchema, ToolDefinition, UnifiedSample
from spancall.eval.llm_parity import (
    LLMResponse,
    _is_retryable_error,
    latency_percentiles,
    merge_with_fallback,
    score_predictions,
    slice_by_complexity,
    token_summary,
)
from spancall.infer.cast import SchemaValidator

HYBRID_MODES = ("hybrid_fill", "hybrid_fill_schema")


@dataclass
class HybridFillPlan:
    """Routing decision for a single sample before any lightweight JSON fill call."""

    sample_id: str
    mode: str
    action: str  # "spancall_hit" | "fill" | "full_fallback"
    tool_name: Optional[str] = None
    schema: Optional[ToolDefinition] = None
    known: Dict[str, Any] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    reason: str = ""


def find_tool_schema(sample: UnifiedSample, tool_name: Optional[str]) -> Optional[ToolDefinition]:
    if not tool_name:
        return None
    for t in sample.tools or ():
        if t.name == tool_name:
            return t
    return None


def complex_param_names(schema: ToolDefinition) -> List[str]:
    """Top-level parameter names that are ``array``, ``object``, or decomposed object roots."""
    names: List[str] = []
    for p in schema.parameters or ():
        if str(getattr(p, "param_type", "")).lower() in ("array", "object"):
            if p.name not in names:
                names.append(p.name)
    for root in (schema.object_paths or {}).keys():
        root_s = str(root)
        if root_s not in names:
            names.append(root_s)
    return names


def is_complex_tool(schema: Optional[ToolDefinition]) -> bool:
    return bool(schema is not None and complex_param_names(schema))


def _missing_required_from_pred(lean_pred: Dict[str, Any]) -> List[str]:
    calls = lean_pred.get("calls") or []
    if calls and isinstance(calls[0], dict):
        mr = calls[0].get("missing_required")
        if mr:
            return [str(x) for x in mr if x]
    mr_top = lean_pred.get("missing_required")
    if mr_top:
        return [str(x) for x in mr_top if x]
    return []


def plan_hybrid_fill(
    sample: UnifiedSample,
    lean_pred: Dict[str, Any],
    *,
    mode: str = "hybrid_fill",
    fill_optional: bool = False,
) -> HybridFillPlan:
    """Decide whether a sample commits directly or runs a lightweight JSON fill."""
    if mode not in HYBRID_MODES:
        raise ValueError(f"Unknown hybrid mode '{mode}'. Expected one of {HYBRID_MODES}.")

    sid = sample.sample_id
    tool_name = lean_pred.get("tool_name")
    schema = find_tool_schema(sample, tool_name)
    miss_req = _missing_required_from_pred(lean_pred)
    hit = bool(lean_pred.get("hit"))

    # If H1 did not select a valid tool at all, fall back to full LLM.
    if schema is None:
        if hit:
            return HybridFillPlan(
                sample_id=sid,
                mode=mode,
                action="spancall_hit",
                tool_name=tool_name,
                known=dict(lean_pred.get("payload") or {}),
                reason="spancall_hit",
            )
        return HybridFillPlan(
            sample_id=sid,
            mode=mode,
            action="full_fallback",
            tool_name=tool_name,
            reason="no_tool_skeleton",
        )

    top_names = [p.name for p in (schema.parameters or ())]
    top_set = set(top_names)

    # Post-H1 schema-aware hybrid fill: on complex tools, keep grounded scalars
    # and ask the LLM (via plain JSON, NO tools wrapper) to fill ONLY the
    # complex params + any missing required scalars.
    if mode == "hybrid_fill_schema":
        c_params = complex_param_names(schema)
        if c_params:
            c_set = set(c_params)
            raw_known = dict(lean_pred.get("payload_partial") or lean_pred.get("payload") or {})
            known_scalar = {
                k: v
                for k, v in raw_known.items()
                if k not in c_set and "." not in str(k) and v is not None
            }
            missing: List[str] = []
            for cp in c_params:
                if cp in top_set and cp not in missing:
                    missing.append(cp)
            for mr in miss_req:
                if mr in top_set and mr not in known_scalar and mr not in missing:
                    missing.append(mr)
            if not missing:
                missing = [p for p in top_names if p not in known_scalar] or list(top_names)
            if not missing:
                return HybridFillPlan(
                    sample_id=sid,
                    mode=mode,
                    action="full_fallback",
                    tool_name=schema.name,
                    schema=schema,
                    reason="zero_param_tool_fallback",
                )
            return HybridFillPlan(
                sample_id=sid,
                mode=mode,
                action="fill",
                tool_name=schema.name,
                schema=schema,
                known=known_scalar,
                missing=missing,
                reason="complex_schema_json_fill",
            )

    # Clean SpanCall hit on all-scalar tool -> 0 LLM tokens!
    if hit:
        return HybridFillPlan(
            sample_id=sid,
            mode=mode,
            action="spancall_hit",
            tool_name=schema.name,
            schema=schema,
            known=dict(lean_pred.get("payload") or {}),
            reason="spancall_hit",
        )

    # Abstention with H1 tool selected: keep grounded params, fill ONLY missing
    # required params; if it was a low-margin/confidence gate abstain (no missing
    # required params), ask the LLM via plain JSON to fill all params of this
    # single selected tool (still zero distractor tools, zero tools=[] wrapper).
    known = {
        k: v
        for k, v in dict(lean_pred.get("payload_partial") or {}).items()
        if v is not None
    }
    miss_opt = None
    if fill_optional:
        miss_opt = [p.name for p in (schema.parameters or ()) if not p.required and p.name not in known]

    if miss_req or miss_opt:
        missing = unresolved_param_names(schema, known, miss_req, miss_opt)
    else:
        missing = [p.name for p in (schema.parameters or ()) if p.required and p.name not in known]
    missing = [m for m in missing if m not in known and m in top_set]

    if not missing:
        if not top_names:
            return HybridFillPlan(
                sample_id=sid,
                mode=mode,
                action="full_fallback",
                tool_name=schema.name,
                schema=schema,
                reason="zero_param_tool_fallback",
            )
        # Low-margin/confidence gate abstention: H1 already picked the tool, so
        # re-fill the tool's parameters in plain JSON without distractor tools.
        return HybridFillPlan(
            sample_id=sid,
            mode=mode,
            action="fill",
            tool_name=schema.name,
            schema=schema,
            known={},
            missing=list(top_names),
            reason="low_conf_json_fill",
        )

    return HybridFillPlan(
        sample_id=sid,
        mode=mode,
        action="fill",
        tool_name=schema.name,
        schema=schema,
        known=known,
        missing=missing,
        reason="abstain_json_fill",
    )


def _format_param_spec(param: ParameterSchema) -> str:
    ptype = str(getattr(param, "param_type", "string") or "string")
    enum_vals = getattr(param, "enum_values", None) or []
    desc = str(getattr(param, "description", "") or "").strip()
    if len(desc) > 80:
        desc = desc[:77].rstrip() + "..."
    spec = f"{param.name} ({ptype}"
    if enum_vals:
        enum_str = "|".join(str(v) for v in enum_vals[:8])
        spec += f"; enum: {enum_str}"
    spec += ")"
    if desc:
        spec += f": {desc}"
    return spec


def build_hybrid_fill_request(
    sample: UnifiedSample,
    plan: HybridFillPlan,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Build a minimal plain-JSON chat prompt with NO ``tools=[...]`` wrapper.

    Returning ``oai_tools = []`` avoids the ~200-250 hidden tokens that hosted
    OpenAI-compatible providers inject whenever ``tools`` is non-empty.
    """
    if plan.action != "fill" or plan.schema is None or not plan.missing:
        raise ValueError(f"Plan for {plan.sample_id} is not a valid fill plan (action={plan.action}).")

    parts: List[str] = [sample.utterance]
    recent = [r for r in (sample.recent_context or []) if isinstance(r, dict)]
    if recent:
        ctx_items = "; ".join(f"{r.get('key', '')}={r.get('value', '')}" for r in recent)
        parts.append(f"Context: {ctx_items}")
    if sample.current_time_iso:
        parts.append(f"Now: {sample.current_time_iso}")

    param_by_name = {p.name: p for p in (plan.schema.parameters or ())}
    specs = [
        _format_param_spec(param_by_name[m]) if m in param_by_name else m
        for m in plan.missing
    ]
    known_line = (
        f"Known: {json.dumps(plan.known, ensure_ascii=False, sort_keys=True)}\n"
        if plan.known
        else ""
    )
    parts.append(
        f"\nTool: {plan.schema.name}\n"
        f"{known_line}"
        f"Extract ONLY these JSON keys: {'; '.join(specs)}\n"
        f"Output ONLY a valid JSON object with those keys (or {{}} if unanswerable)."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON parameter extractor. "
                "Output ONLY a single valid JSON object for the requested keys with no prose."
            ),
        },
        {"role": "user", "content": "\n".join(parts)},
    ]
    return messages, []


def call_json_fill_sdk(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = 10000,
    timeout: float = 120.0,
    max_retries: int = 4,
    base_backoff: float = 1.5,
    json_mode: bool = True,
    reasoning_effort: Optional[str] = "medium",
    extra_body: Optional[Dict[str, Any]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> LLMResponse:
    """Custom JSON-only completion call for HYBRID_FILL (no ``tools`` schema).

    - Uses ``response_format={"type": "json_object"}`` when ``json_mode=True``
      (gracefully retrying without it if a provider rejects ``response_format``).
    - Passes ``reasoning_effort`` (default ``"low"``) so reasoning models emit
      JSON in ``message.content`` instead of exhausting tokens in thinking.
    - Falls back to ``message.reasoning_content`` if ``message.content`` is empty.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    eb = dict(extra_body or {})
    if reasoning_effort:
        eb.setdefault("reasoning_effort", reasoning_effort)
    if eb:
        kwargs["extra_body"] = eb

    last_exc: Optional[Exception] = None
    t0 = time.perf_counter()
    resp = None
    for attempt in range(max(1, int(max_retries) + 1)):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg_l = str(exc).lower()
            # Graceful downgrade if provider rejects response_format or reasoning_effort
            if "response_format" in msg_l and "response_format" in kwargs:
                kwargs.pop("response_format", None)
                continue
            if "reasoning_effort" in msg_l and isinstance(kwargs.get("extra_body"), dict):
                kwargs["extra_body"].pop("reasoning_effort", None)
                if not kwargs["extra_body"]:
                    kwargs.pop("extra_body", None)
                continue
            if attempt >= max_retries or not _is_retryable_error(exc):
                break
            delay = base_backoff * (2 ** attempt) + random.uniform(0.0, 0.5)
            sleep_fn(delay)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    if last_exc is not None or resp is None:
        return LLMResponse(
            latency_ms=latency_ms,
            error=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "empty_response",
        )

    try:
        message = resp.choices[0].message
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        return LLMResponse(latency_ms=latency_ms, error=f"malformed response: {exc}")

    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0

    content = getattr(message, "content", None) or ""
    if not str(content).strip():
        # Reasoning models sometimes place final JSON in reasoning_content if content is empty
        reasoning = getattr(message, "reasoning_content", None) or ""
        if reasoning:
            content = str(reasoning)

    return LLMResponse(
        tool_calls=[],
        text=str(content),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )


def parse_json_fill_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object dict from a plain-text LLM response."""
    raw = str(text or "").strip()
    if not raw:
        return None
    # Strip markdown ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Fallback: locate outermost {...}
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _flatten_matching_dotted(
    filled: Optional[Dict[str, Any]],
    allowed: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """If ``allowed`` expects dotted keys (``a.b``) and the LLM returned nested
    dicts (``{"a": {"b": v}}``), lift matching nested leaves to dotted keys.
    """
    if not filled or not any("." in str(a) for a in allowed):
        return filled
    out = dict(filled)
    for key in allowed:
        key_s = str(key)
        if key_s in out or "." not in key_s:
            continue
        cur: Any = filled
        ok = True
        for part in key_s.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            out[key_s] = cur
    return out


def _extract_fill_arguments(resp: LLMResponse, tool_name: str) -> Optional[Dict[str, Any]]:
    if resp.error:
        return None
    # Support both plain-JSON text responses (primary) and tool_calls (fake/legacy)
    if resp.text:
        parsed = parse_json_fill_text(resp.text)
        if parsed is not None:
            return parsed
    if resp.tool_calls:
        for c in resp.tool_calls:
            if c.get("tool_name") == tool_name and isinstance(c.get("payload"), dict):
                return c["payload"]
        first = resp.tool_calls[0]
        if isinstance(first.get("payload"), dict):
            return first["payload"]
    return None


def _full_fallback_row(
    sample_id: str,
    lean_pred: Dict[str, Any],
    full_llm_pred: Optional[Dict[str, Any]],
    *,
    routed_from: str,
    fill_fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    lean_lat = float(lean_pred.get("latency_ms") or 0.0)
    if full_llm_pred is None:
        return {
            "sample_id": sample_id,
            "hit": False,
            "status": "abstain",
            "tool_name": None,
            "payload": None,
            "calls": [],
            "routed_from": routed_from,
            "latency_ms": lean_lat,
            "features": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "llm_error": "missing_full_llm_fallback",
                "fill_fallback_reason": fill_fallback_reason,
            },
        }
    row = dict(full_llm_pred)
    feats = dict(row.get("features") or {})
    if fill_fallback_reason:
        feats["fill_fallback_reason"] = fill_fallback_reason
    row["features"] = feats
    row["routed_from"] = routed_from
    row["latency_ms"] = lean_lat + float(full_llm_pred.get("latency_ms") or 0.0)
    return row


def finalize_hybrid_fill_row(
    plan: HybridFillPlan,
    lean_pred: Dict[str, Any],
    fill_resp: Optional[LLMResponse],
    full_llm_pred: Optional[Dict[str, Any]],
    *,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble final ``ToolCallEvaluator`` row for one sample (single-shot, no double-call)."""
    if plan.action == "spancall_hit":
        row = dict(lean_pred)
        row["routed_from"] = "spancall_hit"
        return row

    if plan.action == "full_fallback":
        return _full_fallback_row(
            plan.sample_id,
            lean_pred,
            full_llm_pred,
            routed_from=f"{plan.mode}_full_fallback",
            fill_fallback_reason=plan.reason,
        )

    # plan.action == "fill" — single-shot lightweight JSON fill, never double-calls full_llm.
    assert plan.schema is not None
    lean_lat = float(lean_pred.get("latency_ms") or 0.0)
    if fill_resp is None:
        return {
            "sample_id": plan.sample_id,
            "hit": False,
            "status": "abstain",
            "tool_name": None,
            "payload": None,
            "calls": [],
            "abstain_reason": "missing_fill_response",
            "routed_from": f"{plan.mode}_abstain",
            "latency_ms": lean_lat,
            "features": {"prompt_tokens": 0, "completion_tokens": 0, "llm_error": "missing_fill_response"},
        }

    filled_args = _flatten_matching_dotted(
        _extract_fill_arguments(fill_resp, plan.schema.name),
        plan.missing,
    )
    merged = merge_filled_arguments(plan.known, filled_args, plan.missing)
    newly_filled = [k for k in merged if k not in plan.known]

    valid, val_reason = SchemaValidator().validate(merged, plan.schema)
    if fill_resp.error or not newly_filled or not valid:
        abstain_reason = fill_resp.error or (
            "fill_llm_abstain" if not newly_filled else f"schema_invalid:{val_reason}"
        )
        row_abstain: Dict[str, Any] = {
            "sample_id": plan.sample_id,
            "hit": False,
            "status": "abstain",
            "tool_name": None,
            "payload": None,
            "payload_partial": merged or lean_pred.get("payload_partial"),
            "calls": [],
            "abstain_reason": abstain_reason,
            "routed_from": f"{plan.mode}_abstain",
            "latency_ms": lean_lat + float(fill_resp.latency_ms or 0.0),
            "features": {
                "prompt_tokens": int(fill_resp.prompt_tokens or 0),
                "completion_tokens": int(fill_resp.completion_tokens or 0),
                "llm_error": fill_resp.error,
                "known_params": sorted(plan.known.keys()),
                "requested_params": list(plan.missing),
                "fill_valid": False,
            },
        }
        if model:
            row_abstain["model"] = model
        return row_abstain

    call_entry = {
        "tool_name": plan.schema.name,
        "payload": merged,
        "status": "hit",
        "hit": True,
    }
    row: Dict[str, Any] = {
        "sample_id": plan.sample_id,
        "hit": True,
        "status": "hit",
        "tool_name": plan.schema.name,
        "payload": merged,
        "payload_partial": merged,
        "calls": [call_entry],
        "abstain_reason": None,
        "routed_from": plan.mode,
        "latency_ms": lean_lat + float(fill_resp.latency_ms or 0.0),
        "features": {
            "prompt_tokens": int(fill_resp.prompt_tokens or 0),
            "completion_tokens": int(fill_resp.completion_tokens or 0),
            "llm_error": None,
            "known_params": sorted(plan.known.keys()),
            "filled_params": sorted(newly_filled),
            "requested_params": list(plan.missing),
            "fill_valid": True,
        },
    }
    if model:
        row["model"] = model
    return row


def merge_router_schema(
    samples: Sequence[UnifiedSample],
    lean_by_id: Dict[str, Dict[str, Any]],
    llm_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Leak-free post-H1 Router+Schema baseline for direct side-by-side comparison."""
    merged: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        sid = s.sample_id
        lean = lean_by_id.get(sid) or {}
        schema = find_tool_schema(s, lean.get("tool_name"))
        if is_complex_tool(schema) or not lean.get("hit"):
            llm = llm_by_id.get(sid)
            merged[sid] = _full_fallback_row(
                sid,
                lean,
                llm,
                routed_from="router_schema_complex" if is_complex_tool(schema) else "lean_abstain",
            )
        else:
            merged[sid] = lean
    return merged


def compact_system_summary(
    samples: Sequence[UnifiedSample],
    raw_rows: Sequence[Dict[str, Any]],
    preds_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute headline, Normal/Complex, latency, and token metrics for one system."""
    ids = [s.sample_id for s in samples]
    overall = score_predictions(samples, preds_by_id, ids)
    m_f = (overall.get("metrics_by_stratum") or {}).get("F", {})
    cx_slices = slice_by_complexity(raw_rows, ids)
    by_cx: Dict[str, Dict[str, Any]] = {}
    sample_by_id = {s.sample_id: s for s in samples}
    for cx_name, cx_ids in cx_slices.items():
        if not cx_ids:
            continue
        cx_samples = [sample_by_id[i] for i in cx_ids]
        cx_m = score_predictions(cx_samples, preds_by_id, cx_ids)
        cx_f = (cx_m.get("metrics_by_stratum") or {}).get("F", {})
        by_cx[cx_name] = {
            "n": len(cx_ids),
            "n_F": int(cx_f.get("count", 0) or 0),
            "exact_F": float(cx_f.get("exact_payload_match", 0.0) or 0.0),
            "hit_F": float(cx_f.get("hit_rate", 0.0) or 0.0),
            "p_at_hit": float(cx_f.get("precision_at_hit", 0.0) or 0.0),
        }

    lats = [float((preds_by_id.get(sid) or {}).get("latency_ms") or 0.0) for sid in ids]
    toks = token_summary(preds_by_id, ids)
    routed_counts: Dict[str, int] = {}
    for sid in ids:
        rf = str((preds_by_id.get(sid) or {}).get("routed_from") or "spancall_only")
        routed_counts[rf] = routed_counts.get(rf, 0) + 1

    return {
        "n": len(ids),
        "n_F": int(m_f.get("count", 0) or 0),
        "exact_F": float(m_f.get("exact_payload_match", 0.0) or 0.0),
        "param_F1": float(overall.get("param_f1", 0.0) or 0.0),
        "hit_F": float(m_f.get("hit_rate", 0.0) or 0.0),
        "p_at_hit": float(m_f.get("precision_at_hit", 0.0) or 0.0),
        "false_hit_on_A": float(overall.get("false_hit_rate_on_a", 0.0) or 0.0),
        "error_wrong_value": int(overall.get("error_wrong_value", 0) or 0),
        "error_entity_swap": int(overall.get("error_entity_swap", 0) or 0),
        "by_complexity": by_cx,
        "latency_ms": latency_percentiles(lats),
        "tokens": toks,
        "routed_counts": routed_counts,
    }


__all__ = [
    "HYBRID_MODES",
    "HybridFillPlan",
    "find_tool_schema",
    "complex_param_names",
    "is_complex_tool",
    "plan_hybrid_fill",
    "build_hybrid_fill_request",
    "call_json_fill_sdk",
    "parse_json_fill_text",
    "finalize_hybrid_fill_row",
    "merge_router_schema",
    "compact_system_summary",
    "merge_with_fallback",
]
