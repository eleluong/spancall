"""Inference resolution API for SpanCall ("use").

Ref: docs/mechanism.md §2, §11, docs/full_implementation_plan.md §4.1.3.

The :class:`SpanCallResolver` runs the full one-pass pipeline
``build_input -> tokenize_offset_mapping -> CandidateMaskBuilder ->
Encoder + Heads -> temperature scaling -> ExclusivityJointDecoder ->
StrictTypeCaster -> SchemaValidator -> Gate.evaluate(...)`` and populates
a :class:`ResolveResult` with a wall-clock latency breakdown
(``pre_ms`` / ``mask_ms`` / ``enc_ms`` / ``heads_ms`` / ``decode_ms`` / ``post_ms``).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from spancall.agent.context import select_recent_context
from spancall.data.align import tooldef_to_template_tool
from spancall.data.schema import ParameterSchema, ToolDefinition
from spancall.infer.cast import SchemaValidator, StrictTypeCaster
from spancall.infer.decode import ExclusivityJointDecoder
from spancall.infer.gate import BaseGate, build_gate

try:  # torch is an optional dependency for the pure-python resolver tests
    import torch
except Exception:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]

from span_template.parser import Entry  # noqa: E402
from span_template.sanitize import SanitizeError  # noqa: E402
from span_template.template import (  # noqa: E402
    RecentRow,
    SpanTemplateOutput,
    build_input,
    build_input_multi,
    tokenize_offset_mapping,
)


@dataclass
class ParameterResolution:
    """Resolved parameter detail."""
    name: str
    value: Any
    confidence: float
    margin: float
    source_type: str
    char_span: Optional[tuple[int, int]] = None
    p_abstain: float = 0.0


@dataclass
class ResolveResult:
    """End-to-end outcome of try_resolve."""
    hit: bool
    tool_name: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    parameters: Dict[str, ParameterResolution] = field(default_factory=dict)
    joint_margin: float = 0.0
    accept_score: float = 0.0
    abstain_reason: Optional[str] = None
    latency_ms: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    status: str = "abstain"  # "hit" | "partial" | "abstain"
    payload_partial: Dict[str, Any] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #


def _load_checkpoint(model_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load ``model.pt`` + ``provenance.json`` from a trainer checkpoint dir.

    Returns ``(state_dict, provenance)``; provenance is ``{}`` when absent.
    ``model_path`` may also point directly at a ``.pt`` weights file.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {model_path}")
    if p.is_dir():
        state_path = p / "model.pt"
        prov_path = p / "provenance.json"
    else:
        state_path = p
        prov_path = p.with_suffix(".provenance.json")
    if not state_path.exists():
        raise FileNotFoundError(f"model weights not found: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)  # type: ignore[union-attr]
    provenance: Dict[str, Any] = {}
    if prov_path.exists():
        try:
            with open(prov_path, "r", encoding="utf-8") as fh:
                provenance = json.load(fh)
        except Exception:  # pragma: no cover - non-fatal metadata
            provenance = {}
    return state, provenance


def _default_gate_cfg(thr: Optional[str]) -> Dict[str, Any]:
    """Read gate thresholds + temperature from a calibration JSON (Q8)."""
    cfg: Dict[str, Any] = {"mode": "single_tau", "tau_p": 0.90, "tau_a": 0.85, "tau_delta": 2.0}
    if not thr:
        return cfg
    with open(thr, "r", encoding="utf-8") as fh:
        cal = json.load(fh)
    for key in ("mode", "tau_p", "tau_a", "tau_delta", "temperature", "tau_tool"):
        if key in cal:
            cfg[key] = cal[key]
    return cfg


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #


class SpanCallResolver:
    """High-level one-pass resolver interface.

    The resolver is a thin pipeline over injectable parts so it can be
    smoke-tested with a small random model (no HF download) and so the
    production path can load a trainer checkpoint.

    Args:
        model_path: optional trainer checkpoint dir (``model.pt`` +
            ``provenance.json``) or a bare ``.pt`` weights file.
        calib_thresholds_path: optional JSON with gate thresholds /
            temperature from :mod:`spancall.train.calibrate`.
        encoder: optional injectable encoder (needs ``forward(input_ids,
            attention_mask, token_type_ids=None) -> H``). When model_path is
            given the checkpoint state is loaded on top of ``model`` (or a
            fresh :class:`SpanCallModel` built over ``encoder``).
        model: optional fully-assembled :class:`SpanCallModel` (encoder under
            ``.backbone``, heads under ``.pointer_head`` / ``.abstain_head`` /
            ``.call_accept_head`` / ``.tool_head``).
        tokenizer: optional tokenizer for render + offsets; when None a
            whitespace ``FakeTokenizer`` is used (tests/ablation only).
        temperature: softmax temperature for decoding (Q0); 1.0 = identity.
        gate: optional gate; defaults to ``build_gate()`` from calibration.
        decoder: optional joint decoder; defaults to ``ExclusivityJointDecoder``.
        caster: optional caster; defaults to ``StrictTypeCaster()``.
        validator: optional validator; defaults to ``SchemaValidator()``.
        max_seq_len / max_span_width: model sequence budget for masks.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        calib_thresholds_path: Optional[str] = None,
        *,
        raw_mode: bool = False,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        temperature: float = 1.0,
        gate: Optional[BaseGate] = None,
        decoder: Optional[ExclusivityJointDecoder] = None,
        caster: Optional[StrictTypeCaster] = None,
        validator: Optional[SchemaValidator] = None,
        max_seq_len: Optional[int] = None,
        max_span_width: int = 16,
        k_recent: int = 32,
        context_budget_tokens: int = 16384,
    ) -> None:
        self.raw_mode = bool(raw_mode)
        self.model_path = model_path
        self.calib_thresholds_path = calib_thresholds_path
        # Context window is backbone-driven (single source of truth): derive
        # from the encoder/tokenizer when the caller does not override it.
        # No hardcoded 512-token default anywhere.
        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else (
            _backbone_max_seq_len(encoder, tokenizer)
        )
        self.max_span_width = int(max_span_width)
        self.k_recent = int(k_recent)
        self.context_budget_tokens = int(context_budget_tokens)

        self._gate_cfg = _default_gate_cfg(calib_thresholds_path)
        self.temperature = float(
            self._gate_cfg.get("temperature", temperature) if calib_thresholds_path else temperature
        )

        self.gate = gate or build_gate(
            mode=str(self._gate_cfg.get("mode", "single_tau")),
            tau_p=float(self._gate_cfg.get("tau_p", 0.90)),
            tau_a=float(self._gate_cfg.get("tau_a", 0.85)),
            tau_delta=float(self._gate_cfg.get("tau_delta", 2.0)),
        )
        self.decoder = decoder or ExclusivityJointDecoder(top_k=3, normalize_scores=False)
        self.caster = caster or StrictTypeCaster()
        self.validator = validator or SchemaValidator()
        self.tokenizer = tokenizer or FakeTokenizer()
        self.encoder = encoder
        self.model = model

        # Load checkpoint state when a path is provided.
        if model_path is not None:
            if torch is None:  # pragma: no cover - guarded import
                raise RuntimeError("torch is required to load a checkpoint")
            if self.model is None:
                from spancall.model.model import SpanCallModel

                self.model = SpanCallModel(encoder=self.encoder)
            state, _prov = _load_checkpoint(model_path)
            self.model.load_state_dict(state, strict=False)
            backbone = getattr(self.model, "backbone", None)
            if backbone is not None:
                self.encoder = backbone

        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        if self.encoder is not None and hasattr(self.encoder, "eval"):
            self.encoder.eval()

        # name -> head module cache
        self._heads: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def try_resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Any] = None,
        raw_mode: Optional[bool] = None,
    ) -> ResolveResult:
        """Attempts to resolve a fully-specified tool call in a single forward pass."""
        return self._resolve(schema, utterance, recent_context, now, raw_mode=raw_mode)

    def predict_tool(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Any] = None,
        raw_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Directly evaluate H1 tool selection across candidate tools.

        Returns a dictionary with:
            tool_name: winning tool name
            tool_idx: winning tool index in candidate_schemas
            tool_prob: winning tool softmax probability
            tool_margin: p_top1 - p_top2 margin
            tool_scores: list of (tool_name, prob)
            gated_hit: whether tool_margin > tau_tool (or True in raw_mode)
            abstain_reason: reason if not gated_hit else None
        """
        eff_raw = self.raw_mode if raw_mode is None else bool(raw_mode)
        if not candidate_schemas:
            return {"tool_name": None, "tool_idx": -1, "tool_prob": 0.0, "tool_margin": 0.0,
                    "tool_scores": [], "gated_hit": False, "abstain_reason": "no_candidates"}
        if len(candidate_schemas) == 1:
            return {"tool_name": candidate_schemas[0].name, "tool_idx": 0, "tool_prob": 1.0,
                    "tool_margin": 1.0, "tool_scores": [(candidate_schemas[0].name, 1.0)],
                    "gated_hit": True, "abstain_reason": None}

        eff_budget = min(self.context_budget_tokens, max(64, self.max_seq_len // 2))
        selected_recent, _ = select_recent_context(
            recent_context or [],
            candidate_schemas,
            k_recent=self.k_recent,
            context_budget_tokens=eff_budget,
            tokenizer=self.tokenizer,
        )
        try:
            output = self._build_multi(candidate_schemas, utterance, selected_recent, now)
            ids, per_token = tokenize_offset_mapping(output, self.tokenizer, max_len=self.max_seq_len)
            H, cls_h, _ids, _am = self._encode(ids, len(ids))
        except SanitizeError as exc:
            return {"tool_name": None, "tool_idx": -1, "tool_prob": 0.0, "tool_margin": 0.0,
                    "tool_scores": [], "gated_hit": False, "abstain_reason": f"sanitize: {exc}"}

        choices = self._tool_scores(output, H, per_token)
        if not choices:
            return {"tool_name": None, "tool_idx": -1, "tool_prob": 0.0, "tool_margin": 0.0,
                    "tool_scores": [], "gated_hit": False, "abstain_reason": "no_tool_scores"}

        winner_name, winner_p = choices[0]
        margin = winner_p - (choices[1][1] if len(choices) > 1 else 0.0)
        tau_tool = float(self._gate_cfg.get("tau_tool", 0.0))
        gated_hit = True if eff_raw else (margin > tau_tool)

        # Locate winner_idx by index in candidate_schemas (handles duplicate names safely)
        winner_idx = 0
        for idx, s in enumerate(candidate_schemas):
            if s.name == winner_name:
                winner_idx = idx
                break

        return {
            "tool_name": winner_name,
            "tool_idx": winner_idx,
            "tool_prob": float(winner_p),
            "tool_margin": float(margin),
            "tool_scores": choices,
            "gated_hit": gated_hit,
            "abstain_reason": None if gated_hit else f"tool_margin_below_tau_tool ({margin:.3f} <= {tau_tool})",
        }

    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Any] = None,
        raw_mode: Optional[bool] = None,
    ) -> ResolveResult:
        """Batch-scores candidate schemas and resolves the winner if unambiguous.

        Renders all schemas in one packed sequence (``build_input_multi``),
        runs the H1 tool head over the ``<|tool|>`` anchors, and — when the top
        tool's softmax margin beats ``tau_tool`` (or in raw mode) — delegates to
        the single-schema pipeline for the winning tool.
        """
        eff_raw = self.raw_mode if raw_mode is None else bool(raw_mode)
        if not candidate_schemas:
            # Instrumented for the empty-candidate abstain cohort (H-D3): these rows
            # never reach H1/H3 because the caller supplied no candidate schemas,
            # so the resolver's abstain is a *data/interface* artifact, not model
            # behaviour. Exposing it as a first-class feature makes the diagnosis
            # one grep instead of a re-run (13 persistent abstains in the 1K exp).
            return ResolveResult(
                hit=False, status="abstain", abstain_reason="no_candidates",
                features={"n_candidates": 0, "empty_candidates": True},
            )
        if len(candidate_schemas) == 1:
            return self.try_resolve(candidate_schemas[0], utterance, recent_context, now, raw_mode=eff_raw)
        if torch is None:  # pragma: no cover
            return ResolveResult(hit=False, status="abstain", abstain_reason="torch_unavailable")
        if self.encoder is None:  # pragma: no cover
            return ResolveResult(hit=False, status="abstain", abstain_reason="no_model_loaded")

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
            output = self._build_multi(candidate_schemas, utterance, selected_recent, now)
            ids, per_token = tokenize_offset_mapping(output, self.tokenizer, max_len=self.max_seq_len)
        except SanitizeError as exc:
            return ResolveResult(hit=False, status="abstain", abstain_reason=f"sanitize: {exc}", features=dict(ctx_features))

        try:
            H, cls_h, _ids, _am = self._encode(ids, len(ids))
        except SanitizeError as exc:
            return ResolveResult(hit=False, status="abstain", abstain_reason=f"sanitize: {exc}", features=dict(ctx_features))
        t_enc = time.perf_counter()

        choices = self._tool_scores(output, H, per_token)
        features = {
            **ctx_features,
            "n_candidates": len(candidate_schemas),
            "pre_ms": (t_enc - t0) * 1000,
            "enc_ms": (time.perf_counter() - t_enc) * 1000,
            "tool_scores": dict(choices),
            "temperature": self.temperature,
        }
        if not choices:
            features["n_tool_anchors"] = 0
            features["no_tool_anchors"] = True
            return ResolveResult(hit=False, status="abstain", abstain_reason="no_tool_scores", features=features)
        winner_name, winner_p = choices[0]
        margin = winner_p - (choices[1][1] if len(choices) > 1 else 0.0)
        tau_tool = float(self._gate_cfg.get("tau_tool", 0.0))

        winner = next((s for s in candidate_schemas if s.name == winner_name), candidate_schemas[0])

        res = self._resolve(winner, utterance, selected_recent, now, raw_mode=eff_raw)
        res.features.update(features)

        if not eff_raw and margin <= tau_tool:
            res.hit = False
            res.status = "partial" if res.payload_partial else "abstain"
            res.abstain_reason = f"tool_margin_below_tau_tool ({margin:.3f} <= {tau_tool})"

        return res

    # ------------------------------------------------------------------ #
    # Implementation
    # ------------------------------------------------------------------ #
    def _resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]],
        now: Optional[Any],
        raw_mode: Optional[bool] = None,
    ) -> ResolveResult:
        """Single-schema one-pass resolution (the §4.1.3 chain)."""
        eff_raw = self.raw_mode if raw_mode is None else bool(raw_mode)
        t0 = time.perf_counter()
        if torch is None:  # pragma: no cover
            return ResolveResult(hit=False, abstain_reason="torch_unavailable")
        if self.encoder is None:  # pragma: no cover
            return ResolveResult(hit=False, abstain_reason="no_model_loaded")

        eff_budget = min(self.context_budget_tokens, max(64, self.max_seq_len // 2))
        selected_recent, ctx_features = select_recent_context(
            recent_context or [],
            [schema],
            k_recent=self.k_recent,
            context_budget_tokens=eff_budget,
            tokenizer=self.tokenizer,
        )

        # ---- 1. render + parse (P5) ---------------------------------------
        try:
            output = self._build_single(schema, utterance, selected_recent, now)
        except SanitizeError as exc:
            return self._abstain(schema, f"sanitize: {exc}", t0, ctx_features)
        except Exception as exc:  # pragma: no cover - unexpected render failure
            return self._abstain(schema, f"render_error: {exc}", t0, ctx_features)
        text = output.text
        entries = output.entries
        t_pre = time.perf_counter()

        # ---- 2. tokenize (P6) --------------------------------------------
        try:
            ids, per_token = tokenize_offset_mapping(output, self.tokenizer, max_len=self.max_seq_len)
        except SanitizeError as exc:
            return self._abstain(schema, f"sanitize: {exc}", t0, ctx_features)

        # ---- 3. candidate masks (P4) --------------------------------------
        cand_masks = self._candidate_masks(output, text, [schema])
        t_mask = time.perf_counter()

        # ---- 4. one forward pass ------------------------------------------
        try:
            H, cls_h, _ids, _am = self._encode(ids, len(ids))
        except SanitizeError as exc:
            # overflow beyond the backbone window must abstain, never crash
            return self._abstain(schema, f"sanitize: {exc}", t0, ctx_features)
        t_enc = time.perf_counter()

        anchors, outcome = self._run_heads(output, H, cls_h, per_token, cand_masks)
        t_heads = time.perf_counter()

        # ---- 5. joint decode (Q3) -----------------------------------------
        per_param_scores = outcome["per_param_scores"]           # {name: [(Candidate, p)]}
        param_names = [p.name for p in schema.parameters]
        entity_classes = {p.name: p.entity_class for p in schema.parameters if p.entity_class}
        per_param_abstain_p = outcome.get("per_param_abstain_p", {})

        # Filter out parameters where pointer-level abstain beats all R4-grouped candidates
        active_param_scores: Dict[str, List[Tuple[Any, float]]] = {}
        best_unfiltered: Dict[str, Tuple[Optional[Any], float]] = {}
        abstained_params: set[str] = set()
        for p in schema.parameters:
            entries = per_param_scores.get(p.name) or []
            best_cand, best_raw_p = self._best_group_candidate(entries)
            best_unfiltered[p.name] = (best_cand, best_raw_p)
            p_abs = float(per_param_abstain_p.get(p.name, 0.0))
            if not eff_raw and (not entries or (p_abs > 0.0 and p_abs >= best_raw_p)):
                active_param_scores[p.name] = []
                if entries and p_abs >= best_raw_p:
                    abstained_params.add(p.name)
            elif eff_raw and not entries:
                active_param_scores[p.name] = []
            else:
                active_param_scores[p.name] = entries

        _assignment, delta, chosen, n_valid = self.decoder.decode_detailed(
            active_param_scores, entity_classes=entity_classes, top_k=3)
        t_post = time.perf_counter()

        # per-parameter margins (p_top - p_second after R4 aggregation)
        margins = self._param_margins(per_param_scores, param_names)

        # ---- 6. cast (Q4/Q5) ----------------------------------------------
        parameters: Dict[str, ParameterResolution] = {}
        payload: Dict[str, Any] = {}
        payload_partial: Dict[str, Any] = {}
        missing_required: List[str] = []
        missing_optional: List[str] = []
        cast_failures: List[str] = []
        for p in schema.parameters:
            choice = chosen.get(p.name)
            p_abs = float(per_param_abstain_p.get(p.name, 0.0))
            raw_ptype = (p.param_type or "string").lower()
            is_nullable = "null" in raw_ptype
            if choice is not None:
                ok, value, reason = self.caster.cast_choice(choice, p, output)
                if ok:
                    span = getattr(choice, "span", None)
                    raw_prob = float(getattr(choice, "prob", 0.0))
                    conf = raw_prob * max(0.0, 1.0 - p_abs)
                    if value is None:
                        # Choice was explicitly <|null|>
                        if not p.required:
                            # Optional param: if default is provided materialize it, else omit
                            if p.default is not None:
                                payload[p.name] = p.default
                                payload_partial[p.name] = p.default
                            else:
                                missing_optional.append(p.name)
                        elif is_nullable:
                            payload[p.name] = None
                            payload_partial[p.name] = None
                        else:
                            # Required non-nullable received null
                            cast_failures.append(f"{p.name}:null_for_required")
                            missing_required.append(p.name)
                    else:
                        payload_partial[p.name] = value
                        payload[p.name] = value

                    parameters[p.name] = ParameterResolution(
                        name=p.name,
                        value=value,
                        confidence=conf,
                        margin=float(margins.get(p.name, 0.0)),
                        source_type=str(getattr(choice, "region", "value")),
                        char_span=(int(span[0]), int(span[1])) if span is not None else None,
                        p_abstain=p_abs,
                    )
                else:
                    cast_failures.append(f"{p.name}:{reason or 'cast_fail'}")
                    if p.required:
                        missing_required.append(p.name)
                    else:
                        missing_optional.append(p.name)
                    parameters[p.name] = ParameterResolution(
                        name=p.name,
                        value=None,
                        confidence=0.0,
                        margin=0.0,
                        source_type="cast_fail",
                        char_span=None,
                        p_abstain=p_abs,
                    )
            else:
                if p.required:
                    cast_failures.append(f"missing:{p.name}")
                    missing_required.append(p.name)
                else:
                    missing_optional.append(p.name)
                best_cand, top_score = best_unfiltered.get(p.name, (None, 0.0))
                cand_val = getattr(best_cand, "canon_value", None) if best_cand is not None else None
                cs = getattr(best_cand, "char_start", -1) if best_cand is not None else -1
                ce = getattr(best_cand, "char_end", -1) if best_cand is not None else -1
                c_span = (int(cs), int(ce)) if (cs is not None and ce is not None and cs >= 0 and ce >= cs) else None
                src_type = "abstain" if p.name in abstained_params else "unresolved"
                parameters[p.name] = ParameterResolution(
                    name=p.name,
                    value=cand_val,
                    confidence=float(top_score),
                    margin=0.0,
                    source_type=src_type,
                    char_span=c_span,
                    p_abstain=p_abs,
                )

        # ---- 6b. schema validation (Q6) + defaults ------------------------
        valid, validation_reason = self.validator.validate(payload, schema)
        if valid:
            payload = self.validator.apply_defaults(payload, schema)

        # ---- Continuous confidence (P2) -----------------------------------
        param_probs: Dict[str, float] = {}
        for p in schema.parameters:
            p_abs = parameters[p.name].p_abstain
            if p.name in payload_partial:
                c_k = parameters[p.name].confidence
            else:
                if p.required:
                    c_k = 0.0
                else:
                    c_k = max(p_abs, 1.0 - p_abs)
            param_probs[p.name] = float(c_k)

        required_params = [p for p in schema.parameters if p.required]
        if required_params:
            n_req = len(required_params)
            resolved_req_c = [param_probs[p.name] for p in required_params if p.name in payload_partial]
            n_res_req = len(resolved_req_c)
            min_res_c = min(resolved_req_c) if resolved_req_c else 0.0
            call_confidence = float(min_res_c * (n_res_req / n_req))
        elif schema.parameters:
            resolved_opt_c = [param_probs[p.name] for p in schema.parameters if p.name in payload_partial]
            call_confidence = float(min(resolved_opt_c)) if resolved_opt_c else 0.0
        else:
            call_confidence = 1.0

        # ---- 7. gate (Q8) ---------------------------------------------------
        accept_score = float(outcome.get("accept_score") or 0.0)
        h4_idx = outcome.get("reason_index")
        h4_reason = _H4_REASONS[h4_idx] if h4_idx is not None and 0 <= h4_idx < len(_H4_REASONS) else None

        is_hit, gate_reason = False, None
        if not valid:
            gate_reason = f"schema_invalid: {validation_reason}"
        elif cast_failures:
            gate_reason = f"cast_failed:{','.join(cast_failures)}"
        elif eff_raw:
            is_hit = True
            gate_reason = None
        else:
            is_hit, gate_reason = self.gate.evaluate(
                param_probabilities=param_probs,
                joint_margin=float(delta),
                accept_probability=accept_score,
                is_schema_valid=True,
                risk_tier=schema.risk_tier,
            )
        if not eff_raw and not is_hit and gate_reason is None and h4_reason not in (None, "OK"):
            gate_reason = f"h4:{h4_reason}"

        if is_hit:
            status = "hit"
        elif len(payload_partial) > 0:
            status = "partial"
        else:
            status = "abstain"

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
            "h4_reason": h4_reason,
            "abstain_p": outcome.get("abstain_p"),
        }
        return ResolveResult(
            hit=bool(is_hit),
            status=status,
            tool_name=schema.name,
            payload=payload if valid else None,
            payload_partial=payload_partial,
            missing_required=missing_required,
            missing_optional=missing_optional,
            parameters=parameters,
            confidence=call_confidence,
            joint_margin=float(delta) if valid else 0.0,
            accept_score=accept_score,
            abstain_reason=gate_reason,
            latency_ms=(time.perf_counter() - t0) * 1000,
            features=features,
        )

    # ------------------------------------------------------------------ #
    # Building blocks
    # ------------------------------------------------------------------ #
    @staticmethod
    def _abstain(
        schema: ToolDefinition,
        reason: str,
        t0: float,
        extra_features: Optional[Dict[str, Any]] = None,
    ) -> ResolveResult:
        params = getattr(schema, "parameters", []) or []
        req = [p.name for p in params if getattr(p, "required", True)]
        opt = [p.name for p in params if not getattr(p, "required", True)]
        return ResolveResult(
            hit=False,
            status="abstain",
            tool_name=getattr(schema, "name", None),
            payload=None,
            payload_partial={},
            missing_required=req,
            missing_optional=opt,
            confidence=0.0,
            abstain_reason=reason,
            latency_ms=(time.perf_counter() - t0) * 1000,
            features=dict(extra_features or {}),
        )

    def _build_single(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]],
        now: Optional[Any],
    ) -> SpanTemplateOutput:
        return build_input(
            tooldef_to_template_tool(schema),
            utterance,
            recent_context=_to_recent_rows(recent_context),
            now=_to_date(now),
        )

    def _build_multi(
        self,
        schemas: Sequence[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]],
        now: Optional[Any],
    ) -> SpanTemplateOutput:
        return build_input_multi(
            [tooldef_to_template_tool(s) for s in schemas],
            utterance,
            recent_context=_to_recent_rows(recent_context),
            now=_to_date(now),
        )

    def _encode(self, ids: List[int], _max_len: int) -> Tuple[Any, Any, Any, Any]:
        """One encoder forward over a rendered sequence.

        Returns ``(H, cls_h, input_ids, attention_mask)``. Raises
        ``SanitizeError("overflow")`` when the sequence exceeds the backbone
        context window, so the existing ``try/except SanitizeError`` guards in
        ``_resolve`` / ``try_resolve_candidates`` abstain instead of crashing
        (Phase-1 overflow unification; docs/pipeline_io P6.6).
        """
        n = len(ids)
        if n > self.max_seq_len:
            raise SanitizeError(
                "overflow",
                f"sequence {n} tokens > L_max={self.max_seq_len}",
            )
        device = next(self.encoder.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            H = self.encoder(input_ids, attention_mask)                # (1, L, d)
        if self.model is not None:
            # The encoder lives under ``backbone`` in SpanCallModel (not
            # ``encoder``), and a bf16-trained checkpoint keeps fp32 head
            # params (the trainer runs autocast at train time; the resolver
            # must unify explicitly). Grab the first *head* parameter.
            head_p = next(
                (p for n, p in self.model.named_parameters()
                 if not n.startswith(("encoder.", "backbone."))),
                None,
            )
            if head_p is not None and H.dtype != head_p.dtype:
                H = H.to(dtype=head_p.dtype)
        cls_h = H[:, 0]                                                # (1, 1, d)
        return H, cls_h, input_ids, attention_mask

    def _candidate_masks(
        self,
        output: SpanTemplateOutput,
        text: str,
        schemas: Sequence[ToolDefinition],
    ) -> Dict[str, Dict[str, Any]]:
        """Per-parameter candidate masks over one rendered sequence.

        ``schemas`` are the original :class:`ToolDefinition`s (the mask builder
        consumes :class:`ParameterSchema`, not the template ``Param`` bridge).

        Coordinate space: ``text`` is the *rendered* text, so regex-candidate
        char offsets align with the absolute token offsets from the P6 offset
        mapping (resolves the mask-builder two-text hazard).

        Returns ``{param_name: mask_dict}`` where ``mask_dict`` carries
        ``candidates`` / ``cand_index`` / ``region`` / ``canon_value``.
        """
        from spancall.model.masks import CandidateMaskBuilder

        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        token_offsets: List[Tuple[int, int]] = list(enc["offset_mapping"])
        builder = CandidateMaskBuilder(
            max_seq_len=self.max_seq_len, max_span_width=self.max_span_width)

        entries = output.entries
        masks: Dict[str, Dict[str, Any]] = {}
        for schema in schemas:
            # The mask builder consumes *targetable* legend entries. Tool anchors
            # (<|tool|>/<|param|>) are bridges (non-selectable) tagged per tool;
            # all targetable context/utterance entries are shared and tool-less,
            # so no
            # per-tool filtering is needed for the candidate sets themselves.
            for p in schema.parameters:
                masks[p.name] = builder.build_param_masks(p, entries, token_offsets, text)
        return masks

    def _heads_map(self) -> Dict[str, Any]:
        if self._heads is None:
            m = self.model if self.model is not None else object()
            self._heads = {
                "pointer_head": getattr(m, "pointer_head", None),
                "abstain_head": getattr(m, "abstain_head", None),
                "call_accept_head": getattr(m, "call_accept_head", None),
                "tool_head": getattr(m, "tool_head", None),
                "correctness_head": getattr(m, "correctness_head", None),
            }
        return self._heads

    def _run_heads(
        self,
        output: SpanTemplateOutput,
        H: Any,
        cls_h: Any,
        per_token: Sequence[Optional[Entry]],
        cand_masks: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
        """Run pointer/abstain/accept heads; return ``(anchors, outcome)``.

        ``anchors`` is ``[(param_name, token_index), ...]`` in template order
        (the ``<|param|>`` question anchors); ``outcome`` carries per-param
        ``(Candidate, prob)`` score lists, accept score and H4 reason index.
        """
        # ---- locate <|param|> anchors via per-token ownership --------------
        anchors: List[Tuple[str, int]] = []
        for i, e in enumerate(per_token):
            if e is not None and e.kind == "param_anchor" and e.value != "<|none|>":
                anchors.append((str(e.text or ""), i))
        if not anchors:
            # fallback for heads-only tests: schema order, anchor token 0.
            for schema in output.tools:
                for p in schema.params:
                    anchors.append((p.name, 0))
        param_names = [a[0] for a in anchors]

        device = H.device
        B, _L, _d = H.shape
        K = len(anchors)
        outcome: Dict[str, Any] = {
            "param_names": param_names,
            "anchor_ids": [a[1] for a in anchors],
            "per_param_scores": {n: [] for n in param_names},
            "abstain_p": None,
        }

        heads = self._heads_map()
        acc = heads.get("call_accept_head")
        if acc is not None:
            acc_logit = acc(cls_h)
            outcome["accept_score"] = float(torch.sigmoid(acc_logit).squeeze().detach())

        ptr = heads.get("pointer_head")
        if ptr is None or K == 0:
            return anchors, outcome

        max_tok_idx = max(H.size(1) - 1, 0)
        param_pos = torch.tensor([[a[1] for a in anchors]], dtype=torch.long, device=device).clamp(min=0, max=max_tok_idx)
        param_h = H.gather(1, param_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))  # (1,K,d)

        # ---- utterance pool for H4 -----------------------------------------
        utt_pool = torch.zeros(B, H.size(-1), device=device, dtype=H.dtype)
        # Compute utt_toks using offsets overlapping utterance entries if available
        u_entries = [e for e in output.entries if e.segment == "utterance"]
        if u_entries:
            u_start = min(e.char_start for e in u_entries)
            u_end = max(e.char_end for e in u_entries)
            enc = self.tokenizer(output.text, return_offsets_mapping=True, add_special_tokens=False)
            token_offsets = enc.get("offset_mapping", [])
            utt_toks = [
                i for i, (tcs, tce) in enumerate(token_offsets)
                if i < H.size(1) and tce > u_start and tcs < u_end
            ]
        else:
            utt_toks = [i for i, e in enumerate(per_token)
                        if e is not None and e.segment == "utterance"]
        if utt_toks:
            utt_pool[0] = H[0, utt_toks].mean(0)

        # ---- per-param candidate masks (packed, padded with -1) -----------
        starts_rows: List[List[int]] = []
        ends_rows: List[List[int]] = []
        for name in param_names:
            m = cand_masks.get(name)
            starts_rows.append([a for a, _ in m["cand_index"]] if m else [])
            ends_rows.append([b for _, b in m["cand_index"]] if m else [])
        C = max((len(r) for r in starts_rows), default=0)
        starts = torch.full((B, K, C), -1, dtype=torch.long, device=device)
        ends = torch.full((B, K, C), -1, dtype=torch.long, device=device)
        for k in range(K):
            if starts_rows[k]:
                starts[0, k, : len(starts_rows[k])] = torch.tensor(starts_rows[k], dtype=torch.long, device=device)
                ends[0, k, : len(ends_rows[k])] = torch.tensor(ends_rows[k], dtype=torch.long, device=device)

        p = ptr(param_h, H, candidate_masks={"starts": starts, "ends": ends})
        z = torch.cat([p["abstain_logits"].unsqueeze(-1), p["span_logits"]], dim=-1)  # (1,K,C+1)
        probs = torch.softmax(z / max(self.temperature, 1e-6), dim=-1)
        if torch.isnan(probs).any():  # pragma: no cover - degraded model
            probs = torch.zeros_like(probs)
        abstain_p = probs[..., 0].squeeze(0)                       # (K,)
        span_p = probs[..., 1:]                                    # (1,K,C)
        outcome["abstain_p"] = float(abstain_p.mean().detach()) if K else 0.0
        outcome["per_param_abstain_p"] = {
            name: float(abstain_p[k].detach()) if k < len(abstain_p) else 0.0
            for k, name in enumerate(param_names)
        }

        for k, name in enumerate(param_names):
            m = cand_masks.get(name)
            if not m:
                continue
            cands = m["candidates"]
            outcome["per_param_scores"][name] = [
                (cands[ci], float(span_p[0, k, ci].detach())) for ci in range(len(cands))
            ]

        # ---- H4 reason -----------------------------------------------------
        abst = heads.get("abstain_head")
        if abst is not None and K > 0:
            reason_logits = abst(param_h, cls_h, utt_pool)         # (1,K,R)
            reason_idxs = torch.softmax(reason_logits.detach(), dim=-1).argmax(-1)[0].tolist()
            outcome["per_param_reasons"] = {
                param_names[k]: _H4_REASONS[int(idx)]
                for k, idx in enumerate(reason_idxs)
                if k < len(param_names) and 0 <= int(idx) < len(_H4_REASONS)
            }
            non_ok = [int(idx) for idx in reason_idxs if int(idx) != 0]
            outcome["reason_index"] = non_ok[0] if non_ok else int(reason_idxs[0])

        return anchors, outcome

    def _tool_scores(self, output: SpanTemplateOutput, H: Any, per_token: Sequence[Optional[Entry]]) -> List[Tuple[str, float]]:
        """H1 softmax over ``<|tool|>`` anchors, descending, ``[(name, prob)]``."""
        heads = self._heads_map()
        tool_head = heads.get("tool_head")
        if tool_head is None:
            return []
        positions = [i for i, e in enumerate(per_token)
                     if e is not None and e.kind == "bridge_text" and e.tool is not None]
        if not positions:
            return []
        device = H.device
        tool_h = H.gather(
            1,
            torch.tensor([positions], dtype=torch.long, device=device)
            .unsqueeze(-1).expand(-1, -1, H.size(-1)),
        )
        cls_h = H[:, 0]
        tl = tool_head(tool_h, cls_h)                              # (1, n_tools)
        tp = torch.softmax(tl / max(self.temperature, 1e-6), dim=-1)
        names = [t.name for t in output.tools]
        # bridge_text entries appear in template order == output.tools order.
        choices = [(names[i] if i < len(names) else f"tool{i}", float(tp[0, i].detach()))
                   for i in range(tp.size(-1))]
        choices.sort(key=lambda c: c[1], reverse=True)
        return choices

    @staticmethod
    def _best_group_candidate(
        entries: Sequence[Tuple[Any, float]],
    ) -> Tuple[Optional[Any], float]:
        """Return ``(best_cand, best_group_raw_prob)`` after R4 equal-value aggregation."""
        if not entries:
            return None, 0.0
        group_prob: Dict[str, float] = {}
        group_best_cand: Dict[str, Tuple[Any, float]] = {}
        for cand, score in entries:
            s = max(float(score), 0.0)
            gk = _groupkey(getattr(cand, "canon_value", None))
            group_prob[gk] = group_prob.get(gk, 0.0) + s
            prev = group_best_cand.get(gk)
            if prev is None or s > prev[1]:
                group_best_cand[gk] = (cand, s)
        best_gk = max(group_prob, key=lambda k: group_prob[k])
        return group_best_cand[best_gk][0], group_prob[best_gk]

    @staticmethod
    def _param_margins(
        per_param_scores: Dict[str, List[Tuple[Any, float]]],
        param_names: Sequence[str],
    ) -> Dict[str, float]:
        """Per-parameter margin = p_top − p_second after R4 value aggregation.

        Mirrors the decoder's R4 collapse: softmax over the per-param candidate
        scores, group by canonical value (merging int/float numerics, splitting
        bools), sum mass. Margin is 0.0 when a parameter has < 2 distinct
        aggregated values.
        """
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


def _groupkey(value: Any) -> str:
    """Hashable canonical key matching :func:`spancall.infer.decode._groupkey`."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"num:{float(value)!r}"
    if isinstance(value, (list, tuple)):
        return f"seq:{value!r}"
    return f"any:{value!r}"


_H4_REASONS: Tuple[str, ...] = ("OK", "ABSENT", "AMBIGUOUS", "NEEDS_REASONING", "OFF_WINDOW")


class FakeTokenizer:
    """Char-level tokenizer for smoke tests (no HF dependency).

    Maps each *character* to a single token — one token per char including
    whitespace — so every legend entry boundary (a char offset) coincides with
    a token start/end. This satisfies the P6 boundary-safety contract in
    :func:`span_template.template.tokenize_offset_mapping` for *any* rendered
    text (word-level splitting would straddle entries like ``key=2026-11-02``).

    Provides the ``__call__(..., return_offsets_mapping=True,
    add_special_tokens=False)`` contract; all token ids are 0 (in-vocab pad) —
    hidden states are random-init noise, which is all the smoke path needs.
    """

    def __call__(self, text: str, return_offsets_mapping: bool = False,
                 add_special_tokens: bool = True, **kwargs: Any) -> Dict[str, Any]:
        offsets = [(i, i + 1) for i in range(len(text))]
        out: Dict[str, Any] = {"input_ids": [0] * len(offsets)}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> List[int]:
        return [0] * len(text)


def _to_date(now: Optional[Any]) -> date:
    """Coerce ``now`` (date | ISO string | None) to a ``date`` (R4 determinism)."""
    if now is None:
        return date.today()
    if isinstance(now, date):
        return now
    if isinstance(now, str):
        return date.fromisoformat(now)
    raise TypeError(f"now must be date or ISO string, got {type(now).__name__}")


def _to_recent_rows(recent_context: Optional[List[Dict[str, Any]]]) -> List[RecentRow]:
    """Convert legacy ``[{'key': ..., 'value': ...}]`` rows to template RecentRow."""
    if not recent_context:
        return []
    rows: List[RecentRow] = []
    for row in recent_context:
        if isinstance(row, RecentRow):
            rows.append(row)
        elif isinstance(row, dict):
            rows.append(RecentRow(
                key=str(row.get("key", "")),
                value=str(row.get("value", "")),
                source=str(row.get("source", "recent"))))
        else:
            raise TypeError(f"recent_context rows must be dict/RecentRow, got {type(row).__name__}")
    return rows


# --------------------------------------------------------------------------- #
# Backbone-driven context window (single source of truth for L_max; no 512)
# --------------------------------------------------------------------------- #
def _backbone_max_seq_len(
    encoder: Any = None,
    tokenizer: Any = None,
    fallback: int = 32768,
) -> int:
    """Derive the context window from the encoder/tokenizer, never a literal.

    Priority:
      1. ``encoder.config.max_position_embeddings`` (HF backbone config).
      2. ``encoder.model_max_length`` (a wrapper exposing the tokenizer cap).
      3. ``tokenizer.model_max_length`` (HF AutoTokenizer).
      4. ``fallback`` — the documented Qwen3-0.6B window (32k). No 512 default.

    The resolver stores the derived value in ``self.max_seq_len``; callers may
    still override it explicitly (tests/ablation), which always wins.
    """
    if encoder is not None:
        cfg = getattr(encoder, "config", None)
        if cfg is not None:
            mpe = getattr(cfg, "max_position_embeddings", None)
            if mpe:
                return int(mpe)
        ml = getattr(encoder, "model_max_length", None)
        if ml:
            return int(ml)
    if tokenizer is not None:
        ml = getattr(tokenizer, "model_max_length", None)
        if ml and int(ml) > 0:
            return int(ml)
    return fallback
