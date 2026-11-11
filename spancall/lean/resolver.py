"""Lean Inference Resolver for SpanCall without abstain machinery.

Provides:
- LeanParameterResolution: resolved parameter detail (with p_abstain=0.0 for eval compatibility).
- LeanResolveResult: clean outcome dataclass (drop-in compatible with ResolveResult).
- LeanResolver: streamlined one-pass inference pipeline:
  render -> tokenize -> candidate masks -> forward (LeanSpanCallModel) ->
  candidate softmax over (0..C-1) -> LeanJointDecoder -> StrictTypeCaster -> SchemaValidator.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from spancall.agent.context import select_recent_context
from spancall.data.align import tooldef_to_template_tool
from spancall.data.schema import ParameterSchema, ToolDefinition
from spancall.infer.cast import SchemaValidator, StrictTypeCaster
from spancall.infer.resolver import FakeTokenizer, _backbone_max_seq_len, _groupkey
from spancall.lean.decode import LeanJointDecoder, LeanValueChoice
from spancall.lean.model import LeanSpanCallModel
from span_template.parser import Entry
from span_template.sanitize import SanitizeError
from span_template.template import (
    RecentRow,
    SpanTemplateOutput,
    build_input_multi,
    tokenize_legend,
    tokenize_offset_mapping,
)


@dataclass
class LeanParameterResolution:
    """Resolved parameter detail in Lean flow."""

    name: str
    value: Any
    confidence: float
    margin: float
    source_type: str
    char_span: Optional[Tuple[int, int]] = None
    p_abstain: float = 0.0


@dataclass
class LeanCallResult:
    """One decoded tool call (R0.4).

    ``LeanResolveResult`` flattens the FIRST call into its own ``tool_name`` /
    ``payload`` / ``parameters`` fields so every existing caller keeps working;
    this type carries the per-call detail when more than one call resolves.
    """

    tool_name: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    parameters: Dict[str, LeanParameterResolution] = field(default_factory=dict)
    joint_margin: float = 0.0
    confidence: float = 0.0
    status: str = "abstain"  # "hit" | "partial" | "abstain"
    hit: bool = False
    missing_required: List[str] = field(default_factory=list)
    call_index: int = 0
    tool_prob: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Serialisable form for prediction records and eval rows."""
        return {
            "call_index": int(self.call_index),
            "tool_name": self.tool_name,
            "payload": dict(self.payload or {}),
            "status": self.status,
            "hit": bool(self.hit),
            "confidence": float(self.confidence),
            "joint_margin": float(self.joint_margin),
            "tool_prob": float(self.tool_prob),
            "missing_required": list(self.missing_required),
        }


@dataclass
class LeanResolveResult:
    """End-to-end outcome of LeanResolver."""

    hit: bool
    tool_name: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    parameters: Dict[str, LeanParameterResolution] = field(default_factory=dict)
    joint_margin: float = 0.0
    accept_score: float = 1.0
    abstain_reason: Optional[str] = None
    status: str = "abstain"  # "hit" | "partial" | "abstain"
    payload_partial: Dict[str, Any] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    error_reason: Optional[str] = None
    # R0.4: one entry per decoded call. Always non-empty once a tool was chosen,
    # with ``calls[0]`` mirrored by the scalar fields above.
    calls: List[LeanCallResult] = field(default_factory=list)


def _call_from_result(
    res: LeanResolveResult, call_index: int, tool_prob: float = 1.0
) -> LeanCallResult:
    """Project a single-call result onto the per-call record (R0.4)."""
    return LeanCallResult(
        tool_name=res.tool_name,
        payload=res.payload,
        parameters=dict(res.parameters),
        joint_margin=float(res.joint_margin),
        confidence=float(res.confidence),
        status=res.status,
        hit=bool(res.hit),
        missing_required=list(res.missing_required),
        call_index=int(call_index),
        tool_prob=float(tool_prob),
    )


def _with_calls(res: LeanResolveResult, tool_prob: float = 1.0) -> LeanResolveResult:
    """Populate ``res.calls`` from ``res`` itself for the single-call paths.

    ``tool_name`` is the marker for "a call was chosen": an abstain that still
    names a tool is a prediction of that tool with an incomplete payload, and
    eval needs it in ``calls`` to score it. An abstain that never picked a tool
    gets an empty list, so ``len(res.calls)`` stays honest.
    """
    res.calls = [_call_from_result(res, 0, tool_prob=tool_prob)] if res.tool_name else []
    return res


def _to_recent_rows(raw: Optional[Sequence[Any]]) -> List[RecentRow]:
    """Convert [{'key': ..., 'value': ...}] dicts or RecentRow instances to template RecentRow."""
    if not raw:
        return []
    rows: List[RecentRow] = []
    for r in raw:
        if isinstance(r, RecentRow):
            rows.append(r)
        elif isinstance(r, dict):
            try:
                rows.append(
                    RecentRow(
                        key=str(r.get("key", "")),
                        value=str(r.get("value", "")),
                        source=str(r.get("source", "recent")),
                    )
                )
            except ValueError:
                continue
        else:
            raise TypeError(f"recent_context rows must be dict/RecentRow, got {type(r).__name__}")
    return rows


def _to_date(val: Any) -> date:
    """Coerce now (date | ISO string | None) to a date."""
    if val is None:
        return date(2026, 1, 1)
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            return date.fromisoformat(val[:10])
        except ValueError:
            return date(2026, 1, 1)
    return date(2026, 1, 1)


class LeanResolver:
    """Streamlined one-pass inference pipeline without abstain machinery."""

    # Capability flag, not a try/except: eval code that wants to pass
    # ``context_schemas`` must check this instead of catching TypeError, because
    # a TypeError silently swallowed would quietly restore the oracle-tool
    # prompt (R0.2) while looking like a successful evaluation run.
    supports_context_schemas: bool = True
    # T1 capability flag, same contract as ``supports_context_schemas``: eval
    # code checks this instead of catching TypeError. A swallowed TypeError
    # would silently drop the instrumentation and report parity as "n/a", which
    # reads like a clean run rather than a missing measurement.
    supports_candidate_parity: bool = True

    def __init__(
        self,
        model: Optional[LeanSpanCallModel] = None,
        encoder: Optional[Any] = None,
        model_path: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        max_seq_len: Optional[int] = None,
        max_span_width: int = 16,
        temperature: float = 1.0,
        decoder: Optional[LeanJointDecoder] = None,
        caster: Optional[StrictTypeCaster] = None,
        validator: Optional[SchemaValidator] = None,
        k_recent: int = 32,
        context_budget_tokens: int = 4096,
        max_cands: Optional[int] = 512,
        max_calls: int = 1,
        call_min_prob: float = 0.25,
        call_min_prob_ratio: float = 0.5,
        candidate_pack_fill: str = "stratified",
        min_confidence: Optional[float] = None,
        min_posterior: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if model is None and encoder is not None:
            model = LeanSpanCallModel(encoder=encoder, max_span_width=max_span_width)
        self.model = model
        self.encoder = encoder or (getattr(model, "backbone", None) if model is not None else None)
        self.tokenizer = tokenizer or FakeTokenizer()
        self.max_seq_len = (
            int(max_seq_len)
            if max_seq_len is not None
            else _backbone_max_seq_len(self.encoder, tokenizer)
        )
        self.max_span_width = int(max_span_width)
        self.temperature = float(temperature)
        # T6: refusal on the decode axis. These are opt-in (None = never refuse)
        # so the default resolver stays byte-identical to the pre-T6 pipeline.
        # They are passed to the decoder through this constructor rather than
        # being mutated afterwards so an injected decoder cannot be left with a
        # stale threshold from a previous experiment.
        self.min_confidence = float(min_confidence) if min_confidence is not None else None
        self.min_posterior = float(min_posterior) if min_posterior is not None else None
        self.decoder = decoder or LeanJointDecoder(
            top_k=3,
            normalize_scores=False,
            min_confidence=self.min_confidence,
            min_posterior=self.min_posterior,
        )
        if decoder is not None:
            # An injected decoder keeps its own other knobs but must honour this
            # resolver's refusal policy; otherwise ``min_confidence=0.4`` here
            # would silently do nothing.
            decoder.min_confidence = self.min_confidence
            decoder.min_posterior = self.min_posterior
        self.caster = caster or StrictTypeCaster()
        self.validator = validator or SchemaValidator()
        self.k_recent = int(k_recent)
        self.context_budget_tokens = int(context_budget_tokens)
        self.max_cands = int(max_cands) if max_cands is not None else None
        # Candidate-retention policy when ``max_cands`` truncates a mask.
        #
        # ``stratified`` (default) reproduces the trainer's retention rule: keep
        # every structured context slot, then spread the remaining budget over
        # the whole utterance. ``head`` is the legacy serve-time behaviour that
        # kept only the prompt's earliest utterance spans — which is why a GT
        # value appearing later in the utterance was never even a candidate at
        # serve time, while it always was during training. Kept selectable so the
        # parity experiment (same model, two pack policies) stays runnable.
        fill = str(candidate_pack_fill or "stratified").lower()
        if fill not in ("head", "stratified"):
            raise ValueError(
                f"candidate_pack_fill must be 'head' or 'stratified', got {candidate_pack_fill!r}"
            )
        self.candidate_pack_fill = fill
        # R0.4: how many tool calls may be emitted. ``max_calls=1`` (default)
        # preserves the single-call contract exactly; a sample that genuinely
        # needs two calls opts in. The count itself is a decision the model does
        # not make -- a dedicated count head is out of scope -- so the second
        # call is admitted only when H1 gives it non-trivial probability.
        self.max_calls = max(int(max_calls), 1)
        self.call_min_prob = float(call_min_prob)
        self.call_min_prob_ratio = float(call_min_prob_ratio)
        # T3: why the LAST ``_tool_scores`` call returned nothing. Eval reads this
        # to attribute an empty ``tool_scores`` row to a specific serve-side cause
        # (no anchors / anchor loss / class mismatch) instead of treating it as a
        # model outcome. Initialised here so the attribute always exists.
        self.last_tool_scores_reason: str = "uninitialized"

        if model_path is not None:
            self.load_checkpoint(model_path)

        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        if self.encoder is not None and hasattr(self.encoder, "eval"):
            self.encoder.eval()

    def load_checkpoint(self, model_path: str) -> None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {model_path}")
        state_path = p / "model.pt" if p.is_dir() else p
        if not state_path.exists():
            raise FileNotFoundError(f"model weights not found: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if self.model is not None:
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            backbone = getattr(self.model, "backbone", None)
            if backbone is not None:
                self.encoder = backbone

    def predict_tool(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Any] = None,
        raw_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Directly evaluate H1 tool selection across candidate tools."""
        if not candidate_schemas:
            return {
                "tool_name": None,
                "tool_idx": -1,
                "tool_prob": 0.0,
                "tool_margin": 0.0,
                "tool_scores": [],
                "gated_hit": False,
                "abstain_reason": "no_candidates",
            }
        if len(candidate_schemas) == 1:
            return {
                "tool_name": candidate_schemas[0].name,
                "tool_idx": 0,
                "tool_prob": 1.0,
                "tool_margin": 1.0,
                "tool_scores": [(candidate_schemas[0].name, 1.0)],
                "gated_hit": True,
                "abstain_reason": None,
            }
        if self.model is None:
            return {
                "tool_name": None,
                "tool_idx": -1,
                "tool_prob": 0.0,
                "tool_margin": 0.0,
                "tool_scores": [],
                "gated_hit": False,
                "abstain_reason": "no_model_loaded",
            }

        eff_budget = min(self.context_budget_tokens, max(64, self.max_seq_len // 2))
        selected_recent, _ = select_recent_context(
            recent_context or [],
            candidate_schemas,
            k_recent=self.k_recent,
            context_budget_tokens=eff_budget,
            tokenizer=self.tokenizer,
        )
        try:
            output = build_input_multi(
                [tooldef_to_template_tool(s) for s in candidate_schemas],
                utterance,
                recent_context=_to_recent_rows(selected_recent),
                now=_to_date(now),
            )
            ids, per_token = tokenize_offset_mapping(output, self.tokenizer, max_len=self.max_seq_len)
            H, _cls_h = self._encode(ids)
        except SanitizeError as exc:
            return {
                "tool_name": None,
                "tool_idx": -1,
                "tool_prob": 0.0,
                "tool_margin": 0.0,
                "tool_scores": [],
                "gated_hit": False,
                "abstain_reason": f"sanitize: {exc}",
                "tool_scores_reason": "sanitize_error",
            }

        choices = self._tool_scores(output, H, per_token)
        if not choices:
            return {
                "tool_name": None,
                "tool_idx": -1,
                "tool_prob": 0.0,
                "tool_margin": 0.0,
                "tool_scores": [],
                "gated_hit": False,
                # T3: the concrete cause, not just "no scores". An eval row can
                # then say "anchor_loss:1of2" instead of blaming H1.
                "abstain_reason": f"no_tool_scores:{self.last_tool_scores_reason}",
                "tool_scores_reason": self.last_tool_scores_reason,
            }

        winner_name, winner_p = choices[0]
        margin = winner_p - (choices[1][1] if len(choices) > 1 else 0.0)
        winner_idx = next(
            (idx for idx, s in enumerate(candidate_schemas) if s.name == winner_name),
            0,
        )
        return {
            "tool_name": winner_name,
            "tool_idx": winner_idx,
            "tool_prob": float(winner_p),
            "tool_margin": float(margin),
            "tool_scores": choices,
            "gated_hit": True,
            "abstain_reason": None,
            # "ok" on the normal path; "anchor_loss:kofN" / "class_mismatch:..."
            # still yields scores but flags a degraded layout (T3).
            "tool_scores_reason": self.last_tool_scores_reason,
        }

    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Any] = None,
        raw_mode: Optional[bool] = None,
    ) -> LeanResolveResult:
        """Score candidate tools via H1 and resolve the top tool."""
        if not candidate_schemas:
            return LeanResolveResult(
                hit=False,
                status="abstain",
                abstain_reason="no_candidates",
                error_reason="no_candidates",
                features={"n_candidates": 0, "empty_candidates": True},
            )
        if len(candidate_schemas) == 1:
            return self.try_resolve(candidate_schemas[0], utterance, recent_context, now)

        if self.model is None or self.tokenizer is None:
            return LeanResolveResult(
                hit=False,
                status="abstain",
                abstain_reason="no_model_or_tokenizer",
                error_reason="no_model_or_tokenizer",
            )

        t0 = time.perf_counter()
        eff_budget = min(self.context_budget_tokens, max(64, self.max_seq_len // 2))
        selected_recent, ctx_features = select_recent_context(
            recent_context or [],
            candidate_schemas,
            k_recent=self.k_recent,
            context_budget_tokens=eff_budget,
            tokenizer=self.tokenizer,
        )

        try:
            output = build_input_multi(
                [tooldef_to_template_tool(s) for s in candidate_schemas],
                utterance,
                recent_context=_to_recent_rows(selected_recent),
                now=_to_date(now),
            )
            ids, per_token = tokenize_offset_mapping(output, self.tokenizer, max_len=self.max_seq_len)
            H, _cls_h = self._encode(ids)
        except Exception as exc:
            reason = f"sanitize: {exc}" if isinstance(exc, SanitizeError) else f"pre_error: {exc}"
            return LeanResolveResult(
                hit=False,
                status="abstain",
                abstain_reason=reason,
                error_reason=reason,
                latency_ms=(time.perf_counter() - t0) * 1000,
                features=dict(ctx_features),
            )
        t_enc = time.perf_counter()

        choices = self._tool_scores(output, H, per_token)
        features = {
            **ctx_features,
            "n_candidates": len(candidate_schemas),
            "pre_ms": (t_enc - t0) * 1000,
            "enc_ms": (time.perf_counter() - t_enc) * 1000,
            "tool_scores": dict(choices),
            "temperature": self.temperature,
            # T3: same attribution as ``predict_tool`` so Part-2 and Part-3 can
            # never disagree about why H1 produced no distribution.
            "tool_scores_reason": self.last_tool_scores_reason,
        }
        if not choices:
            features["n_tool_anchors"] = 0
            features["no_tool_anchors"] = True
            return LeanResolveResult(
                hit=False,
                status="abstain",
                # T3: carry the cause in the reason itself; "no_tool_scores"
                # alone was indistinguishable across four different defects.
                abstain_reason=f"no_tool_scores:{self.last_tool_scores_reason}",
                error_reason=f"no_tool_scores:{self.last_tool_scores_reason}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                features=features,
            )

        winner_name, winner_p = choices[0]
        winner = next((s for s in candidate_schemas if s.name == winner_name), candidate_schemas[0])

        # R0.4: decide how many calls to emit. The winner is always call 0. A
        # further call is admitted only if H1 gives its tool real probability --
        # both an absolute floor and a ratio against the winner -- so a confident
        # single call is never split in two by a rounding-level runner-up.
        admitted: List[Tuple[str, float]] = [(winner_name, float(winner_p))]
        for name, p in choices[1:]:
            if len(admitted) >= self.max_calls:
                break
            if p < self.call_min_prob or p < float(winner_p) * self.call_min_prob_ratio:
                break
            admitted.append((name, float(p)))

        encoded_bundle = (output, H, per_token, ctx_features)
        calls: List[LeanCallResult] = []
        first_res: Optional[LeanResolveResult] = None
        for call_index, (name, p) in enumerate(admitted):
            schema_i = next((s for s in candidate_schemas if s.name == name), None)
            if schema_i is None:
                continue
            part = self._resolve(
                schema_i, utterance, selected_recent, now, encoded_bundle=encoded_bundle
            )
            if first_res is None:
                first_res = part
            calls.append(_call_from_result(part, call_index, tool_prob=p))
        if first_res is None:  # every admitted tool vanished -- impossible, but loud
            first_res = self._resolve(
                winner, utterance, selected_recent, now, encoded_bundle=encoded_bundle
            )
            calls = [_call_from_result(first_res, 0, tool_prob=float(winner_p))]
        res = first_res

        # R5: latency must be ADDITIVE. ``_resolve`` starts its own clock, so its
        # ``latency_ms`` only covers head/decode/post and silently omitted the
        # render + encode + H1 scoring done here. Reported as the sum, and the
        # sub-stages keep their own meaning: pre/enc belong to this frame.
        total_ms = (time.perf_counter() - t0) * 1000
        res.latency_ms = float(res.latency_ms or 0.0) + total_ms
        res.features.update(features)
        # Not every stage name exists in both frames; report the uncovered
        # remainder explicitly rather than letting it vanish into the total.
        res.features["stage_sum_ms"] = float(
            sum(res.features.get(k, 0.0) or 0.0 for k in
                ("pre_ms", "mask_ms", "enc_ms", "heads_ms", "decode_ms", "post_ms"))
        )
        res.features["unattributed_ms"] = float(max(0.0, res.latency_ms - res.features["stage_sum_ms"]))
        res.features["winner_prob"] = float(winner_p)
        res.features["n_calls"] = len(calls)
        res.features["call_probs"] = [round(float(p), 6) for _n, p in admitted[: len(calls)]]
        res.calls = calls
        return res

    def try_resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Any] = None,
        raw_mode: Optional[bool] = None,
        context_schemas: Optional[Sequence[ToolDefinition]] = None,
        call_schemas: Optional[Sequence[ToolDefinition]] = None,
        parity_spans: Optional[Dict[str, Sequence[Tuple[str, str]]]] = None,
        protect_parity_spans: bool = False,
    ) -> LeanResolveResult:
        """Resolve a single tool call.

        Args:
            schema: the tool whose parameters are scored and returned.
            context_schemas: the tools to RENDER into the prompt. Defaults to
                ``[schema]``, which is the oracle-tool setting. Passing
                ``sample.tools`` reproduces the training prompt exactly (every
                native tool present, ``T = 1 + len(native)``). Without this the
                eval prompt carried one ``<|tool|>`` block while training
                carried ``pack_tools``, so H1/H3 saw an anchor layout at eval
                time that they were never trained on (R0.2 / R1b).
            call_schemas: R0.4 oracle multi-call input. When given, ONE call is
                decoded per entry (skipping ``schema`` itself, which is call 0)
                and appended to ``calls``. ``schema``'s own result still
                populates ``tool_name``/``payload``, so single-call callers are
                unaffected.
            parity_spans: T1 candidate-parity hook. ``{param_name:
                [(char_start, char_end), ...]}`` GT spans whose presence in the
                candidate set is reported in ``features["parity_hit"]``.
                Evaluation-only; leave ``None`` in production so the mask
                builder takes its fastest path.
            protect_parity_spans: T2 switch. When True the same spans are also
                forced past a ``max_cands`` truncation, which is the parity fix
                under test. Keep it False when *measuring* the defect — with
                protection on, parity is perfect by construction.
        """
        res = self._resolve(
            schema, utterance, recent_context, now, context_schemas=context_schemas,
            parity_spans=parity_spans, protect_parity_spans=protect_parity_spans,
        )
        extra = [s for s in (call_schemas or ()) if s.name != schema.name]
        if not extra:
            return res
        render = list(context_schemas) if context_schemas else [schema, *extra]
        for other in extra:
            if not any(r.name == other.name for r in render):
                render = [*render, other]
            part = self._resolve(
                other, utterance, recent_context, now, context_schemas=render,
                parity_spans=parity_spans, protect_parity_spans=protect_parity_spans,
            )
            res.calls.append(_call_from_result(part, len(res.calls)))
        res.features["n_calls"] = len(res.calls)
        return res

    def _resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]],
        now: Optional[Any],
        encoded_bundle: Optional[Tuple[SpanTemplateOutput, torch.Tensor, Sequence[Optional[Entry]], Dict[str, Any]]] = None,
        context_schemas: Optional[Sequence[ToolDefinition]] = None,
        parity_spans: Optional[Dict[str, Sequence[Tuple[str, str]]]] = None,
        protect_parity_spans: bool = False,
    ) -> LeanResolveResult:
        # T1/T2 split, deliberately two switches:
        #   ``parity_spans`` alone      -> DIAGNOSIS. Report whether each GT span
        #                                  is inside the candidate set the model
        #                                  was actually given. Protection stays
        #                                  off, so the number is the real miss.
        #   ``protect_parity_spans``    -> FIX under test. Same spans are also
        #                                  forced past ``max_cands`` truncation.
        # Fusing them would make every diagnosis report a trivially perfect
        # parity rate, which is precisely the bug that section 3.3 describes.
        protected_spans = parity_spans if protect_parity_spans else None
        # ``render_schemas`` is what actually goes into the prompt; ``schema`` is
        # only what we score. Keeping them separate is the whole point of R0.2.
        render_schemas: List[ToolDefinition] = list(context_schemas) if context_schemas else [schema]
        if not any(s.name == schema.name for s in render_schemas):
            # Never render a prompt that omits the tool we are about to score.
            render_schemas = [schema, *render_schemas]
        t0 = time.perf_counter()
        if self.model is None or self.tokenizer is None:
            return LeanResolveResult(
                hit=False,
                status="abstain",
                abstain_reason="no_model_or_tokenizer",
                error_reason="no_model_or_tokenizer",
            )

        if encoded_bundle is not None:
            output, H, per_token, ctx_features = encoded_bundle
            # The bundle's tokenization is the one that produced ``per_token``,
            # so rebuild the offsets from it rather than re-encoding: the
            # offsets are a pure function of ``output.text`` and the tokenizer,
            # and _RowCtx identity-checks the list it was built from.
            # R5: the stage clocks must describe THIS call, not the caller's.
            # Previously this branch set ``t_pre = t0`` and THEN immediately
            # overwrote ``t_mask`` before any work, so ``pre_ms`` was 0 and
            # ``mask_ms``/``enc_ms`` were attributed to the wrong stages. The
            # render and the encode both happened in ``try_resolve_candidates``,
            # so the honest report is: pre=0, enc=0, and mask = the work below.
            t_pre = time.perf_counter()
            # Recovering offsets by re-encoding would be a second full pass. The
            # bundle already carries the token count; derive the geometry from
            # the one encode the caller performed.
            offset_map = self.tokenizer(
                output.text, return_offsets_mapping=True, add_special_tokens=False
            )["offset_mapping"]
            cand_masks, _ctx = self._candidate_masks(
                output, output.text, [schema], token_offsets=offset_map,
                protected_spans=protected_spans,
            )
            t_mask = time.perf_counter()
            # In the bundle path the prompt was already rendered by
            # ``try_resolve_candidates``, so reuse ITS anchor count (R9).
            ctx_features = dict(ctx_features)
            ctx_features.setdefault("render_T", sum(
                1 for e in output.entries
                if getattr(e, "kind", None) == "bridge_text"
                and getattr(e, "tool", None) is not None
            ))
            # No encode happened here — it is carried by the bundle.
            t_enc = t_mask
        else:
            # ``eff_budget`` is reported, never silent: R2 was that the resolver
            # quietly halved ``context_budget_tokens`` so inference evicted rows
            # the configured budget said were affordable.
            eff_budget = min(self.context_budget_tokens, max(64, self.max_seq_len // 2))
            selected_recent, ctx_features = select_recent_context(
                recent_context or [],
                render_schemas,
                k_recent=self.k_recent,
                context_budget_tokens=eff_budget,
                tokenizer=self.tokenizer,
            )
            ctx_features = dict(ctx_features)
            ctx_features["recent_budget_tokens"] = int(self.context_budget_tokens)
            ctx_features["recent_budget_effective"] = int(eff_budget)
            ctx_features["recent_budget_clamped"] = bool(eff_budget < int(self.context_budget_tokens))

            try:
                # Render EVERY tool: same grammar, same anchor count as training.
                output = build_input_multi(
                    [tooldef_to_template_tool(s) for s in render_schemas],
                    utterance,
                    recent_context=_to_recent_rows(selected_recent),
                    now=_to_date(now),
                )
                t_pre = time.perf_counter()
                # ONE tokenization: the same offsets feed both the legend mapping
                # and the candidate masks (R4). Previously this encoded the text
                # a second time inside _candidate_masks.
                ids, per_token, token_offsets = tokenize_legend(
                    output, self.tokenizer, max_len=self.max_seq_len
                )
                text = output.text
                cand_masks, _ctx = self._candidate_masks(
                    output, text, [schema], token_offsets=token_offsets,
                    protected_spans=protected_spans,
                )
                t_mask = time.perf_counter()
                H, _cls_h = self._encode(ids)
                t_enc = time.perf_counter()
                # R9: the anchor count actually rendered, so a T mismatch is
                # visible on every eval row instead of being inferred.
                ctx_features["render_T"] = int(sum(
                    1 for e in output.entries
                    if getattr(e, "kind", None) == "bridge_text"
                    and getattr(e, "tool", None) is not None
                ))
            except Exception as exc:
                reason = f"sanitize: {exc}" if isinstance(exc, SanitizeError) else f"render_or_encode_error: {exc}"
                req = [p.name for p in schema.parameters if p.required]
                opt = [p.name for p in schema.parameters if not p.required]
                return LeanResolveResult(
                    hit=False,
                    tool_name=schema.name,
                    status="abstain",
                    missing_required=req,
                    missing_optional=opt,
                    abstain_reason=reason,
                    error_reason=reason,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    features=dict(ctx_features),
                )

        # Run pointer head
        anchors, per_param_scores = self._score_parameters(output, H, per_token, cand_masks, schema=schema)
        t_heads = time.perf_counter()

        if schema.parameters and not anchors:
            reason = "no_param_anchors"
            req = [p.name for p in schema.parameters if p.required]
            opt = [p.name for p in schema.parameters if not p.required]
            return _with_calls(LeanResolveResult(
                hit=False,
                tool_name=schema.name,
                status="abstain",
                missing_required=req,
                missing_optional=opt,
                abstain_reason=reason,
                error_reason=reason,
                latency_ms=(time.perf_counter() - t0) * 1000,
                features=dict(ctx_features),
            ))

        # H3 must only see the anchors of the tool being scored: a distractor's
        # ``<|param|> date`` anchor is not this schema's ``date`` slot (R3).
        if schema.parameters and len(render_schemas) > 1:
            ctx_features = dict(ctx_features)
            ctx_features["n_foreign_param_anchors"] = int(sum(
                1 for e in output.entries
                if getattr(e, "kind", None) == "param_anchor"
                and getattr(e, "tool", None) not in (None, schema.name)
            ))
        param_names = [p.name for p in schema.parameters]
        entity_classes = {p.name: p.entity_class for p in schema.parameters if p.entity_class}
        assignment, delta, chosen, n_valid = self.decoder.decode_detailed(
            per_param_scores, entity_classes=entity_classes, top_k=3
        )
        t_post = time.perf_counter()
        # T6: a param the decoder explicitly refused. Surfaced separately from
        # "missing" so Part-1 can report an honest abstention instead of scoring
        # it as a retrieval miss — those are different failures with different
        # fixes (threshold calibration vs candidate coverage).
        decode_refusals: Dict[str, str] = dict(getattr(self.decoder, "last_refusals", {}) or {})

        margins = self._param_margins(per_param_scores, param_names)

        parameters: Dict[str, LeanParameterResolution] = {}
        payload: Dict[str, Any] = {}
        payload_partial: Dict[str, Any] = {}
        missing_required: List[str] = []
        missing_optional: List[str] = []
        cast_failures: List[str] = []

        for p in schema.parameters:
            choice = chosen.get(p.name)
            raw_ptype = (p.param_type or "string").lower()
            is_nullable = "null" in raw_ptype

            if choice is not None:
                ok, value, reason = self.caster.cast_choice(choice, p, output)
                if ok:
                    span = getattr(choice, "span", None)
                    conf = float(getattr(choice, "prob", 0.0))
                    if value is None:
                        if not p.required:
                            if p.default is not None:
                                payload[p.name] = p.default
                                payload_partial[p.name] = p.default
                            else:
                                missing_optional.append(p.name)
                        elif is_nullable:
                            payload[p.name] = None
                            payload_partial[p.name] = None
                        else:
                            cast_failures.append(f"{p.name}:null_for_required")
                            missing_required.append(p.name)
                    else:
                        payload[p.name] = value
                        payload_partial[p.name] = value

                    parameters[p.name] = LeanParameterResolution(
                        name=p.name,
                        value=value,
                        confidence=conf,
                        margin=float(margins.get(p.name, 0.0)),
                        source_type=str(getattr(choice, "region", "value")),
                        char_span=(int(span[0]), int(span[1])) if span is not None else None,
                        p_abstain=0.0,
                    )
                else:
                    cast_failures.append(f"{p.name}:{reason or 'cast_fail'}")
                    if p.required:
                        missing_required.append(p.name)
                    else:
                        missing_optional.append(p.name)
                    parameters[p.name] = LeanParameterResolution(
                        name=p.name,
                        value=None,
                        confidence=0.0,
                        margin=0.0,
                        source_type="cast_fail",
                        char_span=None,
                        p_abstain=0.0,
                    )
            else:
                if p.required:
                    cast_failures.append(f"missing:{p.name}")
                    missing_required.append(p.name)
                else:
                    if p.default is not None:
                        payload[p.name] = p.default
                        payload_partial[p.name] = p.default
                    else:
                        missing_optional.append(p.name)
                # Distinguish "the decoder refused to pick" from "there was no
                # candidate to pick". Both land in the same missing branch, but
                # only the first is a T6 abstention the caller can act on.
                src = "refused" if p.name in decode_refusals else "missing"
                parameters[p.name] = LeanParameterResolution(
                    name=p.name,
                    value=None,
                    confidence=0.0,
                    margin=0.0,
                    source_type=src,
                    char_span=None,
                    p_abstain=0.0,
                )

        # Note: SchemaValidator.validate signature is validate(payload, tool_schema)
        valid, validation_reason = self.validator.validate(payload, schema)
        if valid:
            payload = self.validator.apply_defaults(payload, schema)

        is_hit = valid and len(missing_required) == 0 and len(cast_failures) == 0
        status = "hit" if is_hit else ("partial" if payload_partial else "abstain")

        if not valid:
            fail_reason: Optional[str] = f"schema_invalid: {validation_reason}"
        elif cast_failures:
            fail_reason = f"cast_failed:{','.join(cast_failures)}"
        else:
            fail_reason = None

        conf_values = [pr.confidence for pr in parameters.values() if pr.value is not None]
        overall_conf = sum(conf_values) / len(conf_values) if conf_values else (1.0 if is_hit else 0.0)

        # T1 candidate-parity telemetry. ``cands_per_param`` is the candidate
        # set size H3 actually chose from; ``cand_pack_fill`` names the retention
        # policy; ``parity_checked``/``parity_hit`` say whether the instrumented
        # GT span survived truncation. A low recall with ``parity_hit`` near 1.0
        # means the model picked the wrong candidate; a low recall with
        # ``parity_hit`` near 0.0 means the GT span was never offerable — the
        # train/serve defect (§3.3), which no amount of training data can fix.
        # NOTE: measured against ``parity_spans`` (the requested set), never
        # against ``protected_spans`` — otherwise the diagnosis would only run
        # when the fix was also on, and would then always report 1.0.
        parity_checked = 0
        parity_hit = 0
        if parity_spans:
            for p_name, spans in parity_spans.items():
                key = self._mask_key(schema.name, p_name)
                m = cand_masks.get(key)
                if m is None or not spans:
                    continue
                have = {
                    (int(getattr(c, "char_start", -1)), int(getattr(c, "char_end", -1)))
                    for c in m["candidates"]
                }
                for cs, ce in spans:
                    parity_checked += 1
                    if (int(cs), int(ce)) in have:
                        parity_hit += 1
        features = {
            **ctx_features,
            "pre_ms": (t_pre - t0) * 1000,
            "mask_ms": (t_mask - t_pre) * 1000,
            "enc_ms": (t_enc - t_mask) * 1000,
            "heads_ms": (t_heads - t_enc) * 1000,
            "decode_ms": (t_post - t_heads) * 1000,
            "post_ms": (time.perf_counter() - t_post) * 1000,
            "temperature": self.temperature,
            "n_valid": int(n_valid),
            "n_params": len(schema.parameters),
            "n_resolved": len(payload_partial),
            "abstain_p": 0.0,
            "cand_pack_fill": self.candidate_pack_fill,
            "cand_cap": self.max_cands if self.max_cands is not None else -1,
            "cands_per_param": {
                self._mask_key(schema.name, k.split("::")[-1]): int(v.get("num_candidates", 0))
                for k, v in cand_masks.items()
                if k.startswith(f"{schema.name}::")
            },
            "parity_checked": int(parity_checked),
            "parity_hit": int(parity_hit),
            "decode_refusals": decode_refusals,
            "n_decode_refusals": len(decode_refusals),
            "min_confidence": self.min_confidence,
            "min_posterior": self.min_posterior,
        }

        return _with_calls(LeanResolveResult(
            hit=bool(is_hit),
            tool_name=schema.name,
            payload=payload if valid else None,
            parameters=parameters,
            joint_margin=float(delta) if valid else 0.0,
            accept_score=overall_conf,
            abstain_reason=fail_reason,
            status=status,
            payload_partial=payload_partial,
            missing_required=missing_required,
            missing_optional=missing_optional,
            confidence=overall_conf,
            latency_ms=(time.perf_counter() - t0) * 1000,
            features=features,
            error_reason=fail_reason,
        ))

    def _encode(self, ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(ids) > self.max_seq_len:
            raise SanitizeError("overflow", f"sequence {len(ids)} tokens > L_max={self.max_seq_len}")
        backbone = self.model.backbone if self.model is not None else self.encoder
        device = next(backbone.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            H = backbone(input_ids, attention_mask)
        if self.model is not None:
            head_p = next(
                (p for n, p in self.model.named_parameters() if not n.startswith(("encoder.", "backbone."))),
                None,
            )
            if head_p is not None and H.dtype != head_p.dtype:
                H = H.to(dtype=head_p.dtype)
        cls_h = H[:, 0]
        return H, cls_h

    @staticmethod
    def _mask_key(tool: Optional[str], param: str) -> str:
        """Qualified mask key. ``param`` alone collides across schemas (R3).

        Two tools may legitimately both declare ``query`` (or ``date``, ``id``).
        Keying by the bare parameter name let the LAST schema silently overwrite
        the first, so H3 scored schema A's anchor against schema B's candidate
        set — a wrong-tool span could win and be cast into A's payload.
        """
        return f"{tool}::{param}" if tool else str(param)

    def _candidate_masks(
        self,
        output: SpanTemplateOutput,
        text: str,
        schemas: Sequence[ToolDefinition],
        token_offsets: Optional[Sequence[Tuple[int, int]]] = None,
        protected_spans: Optional[Dict[str, Sequence[Tuple[str, str]]]] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Any]:
        """Build candidate sets for every parameter of every rendered schema.

        Returns ``(masks, ctx)``. ``masks`` is keyed by ``tool::param`` so a
        shared parameter name across tools cannot collide, and ``ctx`` is the
        per-row :class:`_RowCtx` that every parameter reused — building it once
        here is what removes the K-fold rebuild of the token/entry indexes (R4).

        ``token_offsets`` must be the offsets from the SAME tokenization that
        produced ``output``; when omitted we fall back to a self-consistent local
        encode, but every in-resolver caller passes it.

        ``protected_spans`` is the candidate-parity instrumentation hook (T1):
        ``{mask_key: [(char_start, char_end), ...]}`` names GT spans that must
        survive a ``max_cands`` truncation. Without it, a parity experiment can
        only prove that recall is low, never that the *candidate set* was the
        reason — the GT span could have been dropped before scoring.
        """
        from spancall.model.masks import CandidateMaskBuilder, _RowCtx

        if token_offsets is None:
            enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            token_offsets = list(enc["offset_mapping"])
        offsets = list(token_offsets)

        builder = CandidateMaskBuilder(
            max_seq_len=self.max_seq_len, max_span_width=self.max_span_width
        )
        builder.pack_fill = self.candidate_pack_fill
        entries = output.entries
        max_tok = builder._effective_max_seq_len(offsets)
        ctx = _RowCtx.for_row(entries, offsets, max_tok)

        masks: Dict[str, Dict[str, Any]] = {}
        # Identical parameter signatures across tools produce identical masks, so
        # build once and alias. Keeps the qualified keys without paying twice.
        built: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for schema in schemas:
            for p in schema.parameters:
                # Accept BOTH key forms. ``try_resolve`` documents
                # ``{param_name: spans}`` (that is what the eval harness builds
                # from a sample's ``target_spans``), while the resolver's own
                # internal key is ``tool::param`` (R3: a distractor schema may
                # share a parameter name). Looking up only the qualified key
                # made protection a silent no-op for every caller that followed
                # the documented contract -- the T2 "fix" did nothing at all,
                # and the experiment would have reported "fix has no effect".
                _p_spans = ()
                if protected_spans:
                    _p_spans = (
                        protected_spans.get(self._mask_key(schema.name, p.name))
                        or protected_spans.get(p.name)
                        or ()
                    )
                sig = (
                    p.name,
                    p.param_type,
                    tuple(p.enum) if p.enum else None,
                    p.entity_class,
                    # A protected set changes the retained candidates, so masks
                    # with different protection must not alias.
                    tuple(_p_spans) if protected_spans else None,
                )
                m = built.get(sig)
                if m is None:
                    builder.pack_protected = self._protected_token_spans(
                        ctx, entries, offsets, protected_spans,
                        self._mask_key(schema.name, p.name), p.name,
                    )
                    m = builder.build_param_masks(
                        p, entries, offsets, text, ctx=ctx, max_cands=self.max_cands
                    )
                    built[sig] = m
                masks[self._mask_key(schema.name, p.name)] = m
        builder.pack_protected = None
        return masks, ctx

    @staticmethod
    def _protected_token_spans(
        ctx: Any,
        entries: Sequence[Any],
        offsets: Sequence[Tuple[int, int]],
        protected_spans: Optional[Dict[str, Sequence[Tuple[str, str]]]],
        mask_key: str,
        param_name: Optional[str] = None,
    ) -> Optional[set]:
        """Map GT char spans to token pairs so ``_pack`` can protect them (T1).

        Returns ``None`` when nothing is protected for this mask, which keeps
        the fast path allocation-free for the normal (non-instrumented) case.
        A char span that does not snap onto token boundaries protects nothing:
        the pointer head could never have scored it anyway.

        ``spans`` is resolved from the qualified ``mask_key`` first and the bare
        ``param_name`` second: the public contract for ``try_resolve`` is
        ``{param_name: spans}``, so accepting only ``tool::param`` silently
        disabled protection for every documented caller (T2 was a no-op).
        """
        if not protected_spans:
            return None
        spans = protected_spans.get(mask_key) or (
            protected_spans.get(param_name) if param_name else None
        ) or ()
        if not spans:
            return None
        out: set = set()
        for cs, ce in spans:
            snapped = ctx.snap_char_range(int(cs), int(ce))
            if snapped is not None:
                out.add((int(snapped[0]), int(snapped[1])))
        return out or None

    def _score_parameters(
        self,
        output: SpanTemplateOutput,
        H: torch.Tensor,
        per_token: Sequence[Optional[Entry]],
        cand_masks: Dict[str, Dict[str, Any]],
        schema: Optional[ToolDefinition] = None,
    ) -> Tuple[List[Tuple[str, int]], Dict[str, List[Tuple[Any, float]]]]:
        anchors: List[Tuple[str, int]] = []
        # ``anchor_keys[k]`` mirrors ``anchors[k]`` with the qualified mask key so
        # a distractor tool that shares a parameter name can never supply this
        # schema's candidate set (R3).
        anchor_keys: List[str] = []
        for i, e in enumerate(per_token):
            if e is not None and e.kind == "param_anchor" and e.value != "<|none|>":
                # Anchor ownership: prefer the entry's own tool tag; when the
                # legend is untagged, fall back to the schema we were asked for.
                owner = e.tool if e.tool is not None else (schema.name if schema else None)
                if schema is not None and owner is not None and owner != schema.name:
                    continue
                anchors.append((str(e.text or ""), i))
                anchor_keys.append(self._mask_key(owner, str(e.text or "")))

        param_names = [a[0] for a in anchors]
        device = H.device
        B, _L, _d = H.shape
        K = len(anchors)
        per_param_scores: Dict[str, List[Tuple[Any, float]]] = {
            p.name: [] for p in (schema.parameters if schema else ())
        }
        for n in param_names:
            per_param_scores.setdefault(n, [])
        if K == 0 or self.model is None or self.model.pointer_head is None:
            return anchors, per_param_scores

        max_tok_idx = max(H.size(1) - 1, 0)
        param_pos = torch.tensor([[a[1] for a in anchors]], dtype=torch.long, device=device).clamp(min=0, max=max_tok_idx)
        param_h = H.gather(1, param_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))

        starts_rows: List[List[int]] = []
        ends_rows: List[List[int]] = []
        for key in anchor_keys:
            m = cand_masks.get(key)
            starts_rows.append([a for a, _ in m["cand_index"]] if m else [])
            ends_rows.append([b for _, b in m["cand_index"]] if m else [])
        C = max((len(r) for r in starts_rows), default=0)

        if C == 0:
            return anchors, per_param_scores

        starts = torch.full((B, K, C), -1, dtype=torch.long, device=device)
        ends = torch.full((B, K, C), -1, dtype=torch.long, device=device)
        for k in range(K):
            if starts_rows[k]:
                starts[0, k, : len(starts_rows[k])] = torch.tensor(starts_rows[k], dtype=torch.long, device=device)
                ends[0, k, : len(ends_rows[k])] = torch.tensor(ends_rows[k], dtype=torch.long, device=device)

        with torch.no_grad():
            res = self.model.pointer_head(param_h, H, candidate_masks={"starts": starts, "ends": ends})
            span_logits = res["span_logits"]  # (1, K, C)

        for k, name in enumerate(param_names):
            m = cand_masks.get(anchor_keys[k])
            if not m or not m["candidates"]:
                continue
            cands = m["candidates"]
            n_cands = len(cands)
            raw_logits = span_logits[0, k, :n_cands]
            temp = max(self.temperature, 1e-6)
            probs = torch.softmax(raw_logits / temp, dim=-1)
            if torch.isnan(probs).any():
                probs = torch.zeros_like(probs)
            per_param_scores[name] = [
                (cands[ci], float(probs[ci].detach().item())) for ci in range(n_cands)
            ]

        return anchors, per_param_scores

    def _tool_scores(
        self,
        output: SpanTemplateOutput,
        H: torch.Tensor,
        per_token: Sequence[Optional[Entry]],
    ) -> List[Tuple[str, float]]:
        """H1 softmax over <|tool|> anchors (bridge_text with e.tool != None), descending.

        T3: an empty return used to be indistinguishable from a real H1 result of
        "no anchors", so an eval row with ``tool_scores == []`` looked like a
        model outcome when it was actually a *serve-side* failure. The concrete
        trigger is now recorded in ``self.last_tool_scores_reason`` and surfaced
        by every caller, so "0 scores" can be attributed instead of guessed at.
        """
        self.last_tool_scores_reason = "ok"
        if self.model is None:
            self.last_tool_scores_reason = "no_model"
            return []
        if self.model.tool_head is None:
            self.last_tool_scores_reason = "no_tool_head"
            return []

        positions = [
            i
            for i, e in enumerate(per_token)
            if e is not None and e.kind == "bridge_text" and e.tool is not None
        ]
        if not positions:
            # The rendered prompt carries no ``<|tool|>`` bridge anchor at all.
            # Either the template omitted it or tokenization dropped the owning
            # entry — both are layout defects, never a model abstention.
            self.last_tool_scores_reason = "no_tool_anchors"
            return []
        n_rendered = sum(
            1 for e in output.entries
            if getattr(e, "kind", None) == "bridge_text" and getattr(e, "tool", None) is not None
        )
        if len(positions) != n_rendered:
            # Tokens were lost between parse and tokenize. Recorded, not raised:
            # the surviving anchors still produce a usable distribution, but the
            # mismatch must be visible in eval instead of silently shrinking T.
            self.last_tool_scores_reason = f"anchor_loss:{len(positions)}of{n_rendered}"

        device = H.device
        max_tok_idx = max(H.size(1) - 1, 0)
        pos_t = torch.tensor([positions], dtype=torch.long, device=device).clamp(min=0, max=max_tok_idx)
        tool_h = H.gather(1, pos_t.unsqueeze(-1).expand(-1, -1, H.size(-1)))
        cls_h = H[:, 0]

        with torch.no_grad():
            tl = self.model.tool_head(tool_h, cls_h)  # (1, n_tools)
            tp = torch.softmax(tl / max(self.temperature, 1e-6), dim=-1)

        names = [t.name for t in output.tools]
        if int(tp.size(-1)) != len(names):
            # H1's class count must equal the rendered tool count; a mismatch
            # means the head was trained/loaded against a different tool layout.
            self.last_tool_scores_reason = f"class_mismatch:{int(tp.size(-1))}vs{len(names)}"
        choices = [
            (names[i] if i < len(names) else f"tool{i}", float(tp[0, i].detach().item()))
            for i in range(tp.size(-1))
        ]
        choices.sort(key=lambda c: c[1], reverse=True)
        return choices

    @staticmethod
    def _param_margins(
        per_param_scores: Dict[str, List[Tuple[Any, float]]],
        param_names: Sequence[str],
    ) -> Dict[str, float]:
        """Per-parameter margin = p_top - p_second after R4 value aggregation."""
        margins: Dict[str, float] = {}
        for name in param_names:
            entries = per_param_scores.get(name) or []
            if not entries:
                margins[name] = 0.0
                continue
            raw = [max(float(s), 0.0) for _, s in entries]
            total = sum(raw)
            if total <= 0:
                margins[name] = 0.0
                continue
            probs = [r / total for r in raw]
            groups: Dict[str, float] = {}
            for (cand, _score), prob in zip(entries, probs):
                value = getattr(cand, "canon_value", None)
                groups[_groupkey(value)] = groups.get(_groupkey(value), 0.0) + prob
            ranked = sorted(groups.values(), reverse=True)
            margins[name] = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
        return margins
