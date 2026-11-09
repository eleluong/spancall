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
    build_input,
    build_input_multi,
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
        return date.today()
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            return date.fromisoformat(val[:10])
        except ValueError:
            return date.today()
    return date.today()


class LeanResolver:
    """Streamlined one-pass inference pipeline without abstain machinery."""

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
        k_recent: int = 3,
        context_budget_tokens: int = 256,
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
        self.decoder = decoder or LeanJointDecoder(top_k=3, normalize_scores=False)
        self.caster = caster or StrictTypeCaster()
        self.validator = validator or SchemaValidator()
        self.k_recent = int(k_recent)
        self.context_budget_tokens = int(context_budget_tokens)

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
                "abstain_reason": "no_tool_scores",
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
        }
        if not choices:
            features["n_tool_anchors"] = 0
            features["no_tool_anchors"] = True
            return LeanResolveResult(
                hit=False,
                status="abstain",
                abstain_reason="no_tool_scores",
                error_reason="no_tool_scores",
                latency_ms=(time.perf_counter() - t0) * 1000,
                features=features,
            )

        winner_name, winner_p = choices[0]
        winner = next((s for s in candidate_schemas if s.name == winner_name), candidate_schemas[0])

        res = self._resolve(winner, utterance, selected_recent, now)
        res.features.update(features)
        res.features["winner_prob"] = float(winner_p)
        return res

    def try_resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Any] = None,
        raw_mode: Optional[bool] = None,
    ) -> LeanResolveResult:
        """Resolve a single tool call."""
        return self._resolve(schema, utterance, recent_context, now)

    def _resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]],
        now: Optional[Any],
    ) -> LeanResolveResult:
        t0 = time.perf_counter()
        if self.model is None or self.tokenizer is None:
            return LeanResolveResult(
                hit=False,
                status="abstain",
                abstain_reason="no_model_or_tokenizer",
                error_reason="no_model_or_tokenizer",
            )

        eff_budget = min(self.context_budget_tokens, max(64, self.max_seq_len // 2))
        selected_recent, ctx_features = select_recent_context(
            recent_context or [],
            [schema],
            k_recent=self.k_recent,
            context_budget_tokens=eff_budget,
            tokenizer=self.tokenizer,
        )

        try:
            output = build_input(
                tooldef_to_template_tool(schema),
                utterance,
                recent_context=_to_recent_rows(selected_recent),
                now=_to_date(now),
            )
            t_pre = time.perf_counter()
            ids, per_token = tokenize_offset_mapping(output, self.tokenizer, max_len=self.max_seq_len)
            text = output.text
            cand_masks = self._candidate_masks(output, text, [schema])
            t_mask = time.perf_counter()
            H, _cls_h = self._encode(ids)
            t_enc = time.perf_counter()
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
        anchors, per_param_scores = self._score_parameters(output, H, per_token, cand_masks)
        t_heads = time.perf_counter()

        param_names = [p.name for p in schema.parameters]
        entity_classes = {p.name: p.entity_class for p in schema.parameters if p.entity_class}
        assignment, delta, chosen, n_valid = self.decoder.decode_detailed(
            per_param_scores, entity_classes=entity_classes, top_k=3
        )
        t_post = time.perf_counter()

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
                parameters[p.name] = LeanParameterResolution(
                    name=p.name,
                    value=None,
                    confidence=0.0,
                    margin=0.0,
                    source_type="missing",
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
        }

        return LeanResolveResult(
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
        )

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

    def _candidate_masks(
        self,
        output: SpanTemplateOutput,
        text: str,
        schemas: Sequence[ToolDefinition],
    ) -> Dict[str, Dict[str, Any]]:
        from spancall.model.masks import CandidateMaskBuilder

        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        token_offsets: List[Tuple[int, int]] = list(enc["offset_mapping"])
        builder = CandidateMaskBuilder(
            max_seq_len=self.max_seq_len, max_span_width=self.max_span_width
        )
        entries = output.entries
        masks: Dict[str, Dict[str, Any]] = {}
        for schema in schemas:
            for p in schema.parameters:
                masks[p.name] = builder.build_param_masks(p, entries, token_offsets, text)
        return masks

    def _score_parameters(
        self,
        output: SpanTemplateOutput,
        H: torch.Tensor,
        per_token: Sequence[Optional[Entry]],
        cand_masks: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Tuple[str, int]], Dict[str, List[Tuple[Any, float]]]]:
        anchors: List[Tuple[str, int]] = []
        for i, e in enumerate(per_token):
            if e is not None and e.kind == "param_anchor" and e.value != "<|none|>":
                anchors.append((str(e.text or ""), i))
        if not anchors:
            for schema in output.tools:
                for p in schema.params:
                    anchors.append((p.name, 0))

        param_names = [a[0] for a in anchors]
        device = H.device
        B, _L, _d = H.shape
        K = len(anchors)
        per_param_scores: Dict[str, List[Tuple[Any, float]]] = {n: [] for n in param_names}
        if K == 0 or self.model is None or self.model.pointer_head is None:
            return anchors, per_param_scores

        max_tok_idx = max(H.size(1) - 1, 0)
        param_pos = torch.tensor([[a[1] for a in anchors]], dtype=torch.long, device=device).clamp(min=0, max=max_tok_idx)
        param_h = H.gather(1, param_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))

        starts_rows: List[List[int]] = []
        ends_rows: List[List[int]] = []
        for name in param_names:
            m = cand_masks.get(name)
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
            m = cand_masks.get(name)
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
        """H1 softmax over <|tool|> anchors (bridge_text with e.tool != None), descending."""
        if self.model is None or self.model.tool_head is None:
            return []

        positions = [
            i
            for i, e in enumerate(per_token)
            if e is not None and e.kind == "bridge_text" and e.tool is not None
        ]
        if not positions:
            return []

        device = H.device
        max_tok_idx = max(H.size(1) - 1, 0)
        pos_t = torch.tensor([positions], dtype=torch.long, device=device).clamp(min=0, max=max_tok_idx)
        tool_h = H.gather(1, pos_t.unsqueeze(-1).expand(-1, -1, H.size(-1)))
        cls_h = H[:, 0]

        with torch.no_grad():
            tl = self.model.tool_head(tool_h, cls_h)  # (1, n_tools)
            tp = torch.softmax(tl / max(self.temperature, 1e-6), dim=-1)

        names = [t.name for t in output.tools]
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
