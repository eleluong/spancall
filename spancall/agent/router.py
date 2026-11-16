"""SpanCall routing engine.

Ref: docs/mechanism.md §11.
Orchestrates speculative fast path before falling back to the LLM.
Supports:
- Modes: LLM_ONLY, SPANCALL_ONLY, ROUTER, HYBRID_FILL, SHADOW
- Integration variants: S_SKIP, S_INJECT, S_HINT
"""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.trace import RoutingDecision, StepTrace
from spancall.data.schema import ToolDefinition

if TYPE_CHECKING:
    from spancall.infer.base import BaseResolver


class RouterMode(str, Enum):
    LLM_ONLY = "llm_only"             # Baseline B0 (Normal Agent)
    SPANCALL_ONLY = "spancall_only"   # Fast-path only (diagnostic)
    ROUTER = "router"                 # Speculative fast path with fallback (System S)
    HYBRID_FILL = "hybrid_fill"       # SpanCall skeleton + LLM fills ONLY missing params
    SHADOW = "shadow"                 # LLM answers, SpanCall runs & logs comparisons


class IntegrationVariant(str, Enum):
    S_SKIP = "s_skip"     # Hit completely replaces LLM call
    S_INJECT = "s_inject" # Hit call + result injected as if LLM called it
    S_HINT = "s_hint"     # On abstain, pass partial spans to LLM context


def _json_dumps(value: Any) -> str:
    """JSON-serialize a tool output for the native chat template."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


class ToolCallRouter:
    """Manages routing between SpanCall fast-path resolver and LLM backend."""

    def __init__(
        self,
        resolver: BaseResolver,
        llm_backend: BaseLLMBackend,
        mode: RouterMode = RouterMode.ROUTER,
        variant: IntegrationVariant = IntegrationVariant.S_SKIP,
        hybrid_fill_optional: bool = False,
    ) -> None:
        self.resolver = resolver
        self.llm_backend = llm_backend
        self.mode = mode
        self.variant = variant
        # HYBRID_FILL asks the LLM about unresolved REQUIRED params by default.
        # Optional params are frequently absent from the utterance by design, so
        # asking for them invites hallucinated values; opt in only to measure it.
        self.hybrid_fill_optional = bool(hybrid_fill_optional)

    def route_turn(
        self,
        utterance: str,
        messages: List[Dict[str, Any]],
        available_tools: List[ToolDefinition],
        session_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Optional[StepTrace]]:
        """Resolves tool call via fast path or delegates to LLM backend.

        Returns:
            ``(result, step_trace)`` where ``result`` is one of:

            * ``{"hit": True, "tool_name", "arguments", ...}`` on a call;
            * ``{"hit": False, ...}`` on abstain / LLM prose.

        The ``messages`` list is **not** mutated; ``s_inject``-style history
        synthesis happens via :meth:`synthesize_injected_call`, which the
        agent loop calls before appending the tool response.
        """
        t0 = time.perf_counter()
        mode = RouterMode(self.mode) if not isinstance(self.mode, RouterMode) else self.mode

        # ---- llm_only: the baseline B0 never touches SpanCall --------------- #
        if mode == RouterMode.LLM_ONLY:
            llm_res = self._llm_call(messages, available_tools, t0)
            result = self._llm_result(llm_res, t0, None)
            trace = self._trace(t0, utterance, RoutingDecision.LLM_FALLBACK, None, llm_res)
            return result, trace

        # ---- Always run SpanCall first (ROUTER / SPANCALL_ONLY / SHADOW) --- #
        res = self.resolver.try_resolve_candidates(
            available_tools, utterance, recent_context=session_context)

        # ---- SPANCALL_ONLY: fast path only, no fallback -------------------- #
        if mode == RouterMode.SPANCALL_ONLY:
            if not res.hit:
                trace = self._trace(t0, utterance, RoutingDecision.SPANCALL_ABSTAIN, res)
                return (
                    {"hit": False, "status": getattr(res, "status", "abstain"),
                     "abstain_reason": res.abstain_reason,
                     "tool_name": res.tool_name, "arguments": None,
                     "payload_partial": dict(getattr(res, "payload_partial", {}) or {}),
                     "missing_required": list(getattr(res, "missing_required", []) or []),
                     "confidence": float(getattr(res, "confidence", 0.0)),
                     "spancall": False},
                    trace,
                )
            trace = self._trace(t0, utterance, RoutingDecision.SPANCALL_HIT, res)
            return (
                {"hit": True, "status": "hit",
                 "tool_name": res.tool_name, "arguments": res.payload,
                 "payload": res.payload,
                 "payload_partial": dict(getattr(res, "payload_partial", None) or res.payload or {}),
                 "missing_required": [],
                 "confidence": float(getattr(res, "confidence", 1.0)),
                 "spancall": True},
                trace,
            )

        # ---- HYBRID_FILL: SpanCall skeleton + LLM fills the gaps only ------ #
        if mode == RouterMode.HYBRID_FILL:
            return self._route_hybrid_fill(utterance, messages, available_tools, res, t0)

        # ---- SHADOW: LLM answers, SpanCall runs + logs comparison ---------- #
        if mode == RouterMode.SHADOW:
            llm_res = self._llm_call(messages, available_tools, t0)
            trace = self._trace(t0, utterance, RoutingDecision.SHADOW_COMPARISON, res, llm_res)
            result = self._llm_result(llm_res, t0, res)
            # ``spancall`` stays False — the *executed* call came from the LLM.
            # The SpanCall hypothesis + comparison telemetry live in the
            # ``spancall_*`` keys below (used by shadow-mode calibration).
            result["spancall_tool"] = res.tool_name
            result["spancall_arguments"] = res.payload
            # Lazy import: spancall.eval imports spancall.agent (circular at
            # module import time), so payload_equal is resolved per call.
            from spancall.eval.toolcall.evaluator import payload_equal  # noqa: PLC0415

            result["spancall_matched"] = bool(
                res.hit and payload_equal(res.payload, llm_res.get("arguments"))
            )
            return result, trace

        # ---- ROUTER (System S): SpanCall first, unmodified-LLM fallback ---- #
        if res.hit:
            trace = self._trace(t0, utterance, RoutingDecision.SPANCALL_HIT, res)
            return (
                {"hit": True, "tool_name": res.tool_name, "arguments": res.payload,
                 "payload": res.payload, "spancall": True,
                 "joint_margin": res.joint_margin, "accept_score": res.accept_score,
                 "features": dict(res.features)},
                trace,
            )

        # SpanCall abstained -> LLM fallback with the UNMODIFIED prompt.
        llm_res = self._llm_call(messages, available_tools, t0)
        trace = self._trace(t0, utterance, RoutingDecision.LLM_FALLBACK, res, llm_res)
        return self._llm_result(llm_res, t0, res), trace

    # ------------------------------------------------------------------ #
    # Mode 3: HYBRID_FILL
    # ------------------------------------------------------------------ #
    def _route_hybrid_fill(
        self,
        utterance: str,
        messages: List[Dict[str, Any]],
        available_tools: List[ToolDefinition],
        res: ResolveResult,
        t0: float,
    ) -> Tuple[Dict[str, Any], Optional[StepTrace]]:
        """SpanCall predicts the call skeleton; the LLM fills ONLY the gaps.

        Three outcomes, in order:

        1. **SpanCall abstained with no tool** -> nothing to build on, so this
           degenerates to the ordinary LLM fallback (mode 2 behaviour).
        2. **SpanCall hit cleanly** -> the LLM is never called.
        3. **Partial**: SpanCall named the tool and grounded some params, but at
           least one required param is unresolved. The LLM is asked *only* for
           those params (see :mod:`spancall.agent.fill`), and its answer is
           merged conservatively: resolved values are never overwritten and
           anything outside the requested set is dropped. The merged argument
           dict then goes through the same schema validation a full SpanCall hit
           would face; if it still fails, we fall back to the ordinary LLM call
           rather than shipping a half-filled payload.

        Asking for a handful of params instead of a whole call is the point: it
        is a shorter completion (fewer tokens, lower latency) and it cannot
        damage the arguments SpanCall already grounded.
        """
        from spancall.agent.fill import (
            build_fill_messages,
            merge_filled_arguments,
            restrict_tool_to_params,
            unresolved_param_names,
        )

        # ---- 1. no skeleton at all ----------------------------------------- #
        if not res.tool_name:
            llm_res = self._llm_call(messages, available_tools, t0)
            trace = self._trace(t0, utterance, RoutingDecision.LLM_FALLBACK, res, llm_res)
            return self._llm_result(llm_res, t0, res), trace

        schema = next((t for t in available_tools if t.name == res.tool_name), None)
        if schema is None:
            llm_res = self._llm_call(messages, available_tools, t0)
            trace = self._trace(t0, utterance, RoutingDecision.LLM_FALLBACK, res, llm_res)
            return self._llm_result(llm_res, t0, res), trace

        # ---- 2. clean SpanCall hit: LLM not needed -------------------------- #
        if res.hit:
            trace = self._trace(t0, utterance, RoutingDecision.SPANCALL_HIT, res)
            return (
                {"hit": True, "tool_name": res.tool_name, "arguments": res.payload,
                 "payload": res.payload, "spancall": True, "hybrid_fill": False,
                 "filled_params": [], "joint_margin": res.joint_margin,
                 "accept_score": res.accept_score, "features": dict(res.features or {})},
                trace,
            )

        known = dict(getattr(res, "payload_partial", None) or {})
        # Only REQUIRED gaps are worth an LLM round trip by default: an optional
        # param SpanCall declined to emit is often genuinely absent from the
        # utterance, and asking for it invents values.
        missing = unresolved_param_names(
            schema,
            known,
            getattr(res, "missing_required", None),
            (getattr(res, "missing_optional", None) if self.hybrid_fill_optional else None),
        )
        missing = [m for m in missing if m not in known]

        # ---- 3a. tool known but the failure is a gate, not a param gap ----- #
        if not missing:
            llm_res = self._llm_call(messages, available_tools, t0)
            trace = self._trace(t0, utterance, RoutingDecision.LLM_FALLBACK, res, llm_res)
            return self._llm_result(llm_res, t0, res), trace

        # ---- 3b. ask the LLM for the gaps only ------------------------------ #
        restricted = restrict_tool_to_params(schema, missing)
        fill_messages = build_fill_messages(messages, schema.name, known, missing)
        fill_res = self._llm_call(fill_messages, [restricted], t0)
        merged = merge_filled_arguments(known, fill_res.get("arguments"), missing)
        newly_filled = [k for k in merged if k not in known]

        from spancall.infer.cast import SchemaValidator

        valid, reason = SchemaValidator().validate(merged, schema)
        if not valid or not newly_filled:
            # A fill that changed nothing, or one that leaves the payload
            # invalid, means the hybrid path contributed nothing. Fall back to
            # the unmodified full LLM call instead of shipping it.
            llm_res = self._llm_call(messages, available_tools, t0)
            trace = self._trace(t0, utterance, RoutingDecision.LLM_FALLBACK, res, llm_res)
            out = self._llm_result(llm_res, t0, res)
            out["hybrid_fill_attempt"] = {
                "missing": list(missing),
                "merged": dict(merged),
                "reason": reason or "nothing_filled",
            }
            return out, trace

        trace = self._trace(t0, utterance, RoutingDecision.HYBRID_FILL, res, fill_res)
        trace.tool_arguments = dict(merged)
        trace.spancall_features.update({
            "hybrid_fill": True,
            "filled_params": list(newly_filled),
            "requested_params": list(missing),
            "known_params": sorted(known),
        })
        return (
            {"hit": True, "tool_name": schema.name, "arguments": dict(merged),
             "payload": dict(merged), "spancall": True, "hybrid_fill": True,
             "filled_params": list(newly_filled),
             "requested_params": list(missing),
             "features": dict(res.features or {}),
             "usage": fill_res.get("usage", {})},
            trace,
        )

    # ------------------------------------------------------------------ #
    # History synthesis for s_inject (Q9.1 byte-identical formats)
    # ------------------------------------------------------------------ #
    def synthesize_injected_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: str = "call_spancall_0",
    ) -> Dict[str, Any]:
        """Build the assistant ``tool_calls`` message that ``s_inject`` appends.

        The shape is byte-identical to what the LLM backend's native tool
        calling emits (OpenAI-style ``tool_calls[].function``), so downstream
        tool responses in the history render exactly like the SSM's own calls
        when run through the native chat template (docs/chat_template_gemma4.jinja).
        """
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": dict(arguments)},
                }
            ],
        }

    def synthesize_tool_response(
        self,
        tool_name: str,
        output: Any,
        call_id: str = "call_spancall_0",
    ) -> Dict[str, Any]:
        """Build the ``role: tool`` message matching the injected call id."""
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": output if isinstance(output, str) else _json_dumps(output),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _llm_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        t0: float,
    ) -> Dict[str, Any]:
        return self.llm_backend.generate_tool_call(messages, tools, temperature=0.0)

    @staticmethod
    def _llm_result(
        llm_res: Dict[str, Any],
        t0: float,
        spancall_res: Optional[ResolveResult],
    ) -> Dict[str, Any]:
        """Normalize a backend result dict into the router result contract."""
        tool_name = llm_res.get("tool_name")
        return {
            "hit": bool(tool_name),
            "tool_name": tool_name,
            "arguments": llm_res.get("arguments"),
            "payload": llm_res.get("arguments"),
            "spancall": False,
            "usage": llm_res.get("usage", {}),
            "latency_ms": llm_res.get("latency_ms", (time.perf_counter() - t0) * 1000),
        }

    @staticmethod
    def _trace(
        t0: float,
        utterance: str,
        decision: RoutingDecision,
        res: Optional[ResolveResult],
        llm_res: Optional[Dict[str, Any]] = None,
    ) -> StepTrace:
        usage = (llm_res or {}).get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
        if tokens_in == 0 and tokens_out == 0:
            tokens_in = int((llm_res or {}).get("tokens_in", 0) or 0)
            tokens_out = int((llm_res or {}).get("tokens_out", 0) or 0)
        features = dict(res.features) if res is not None else {}
        return StepTrace(
            turn_index=0,
            user_input=utterance,
            decision=decision,
            selected_tool=res.tool_name if res is not None else None,
            tool_arguments=res.payload if res is not None else None,
            spancall_features=features,
            llm_tokens_in=tokens_in,
            llm_tokens_out=tokens_out,
            latency_ms=float((llm_res or {}).get("latency_ms", (time.perf_counter() - t0) * 1000)),
        )
