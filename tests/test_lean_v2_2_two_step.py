"""Unit tests for Lean v2.2 (W0 instrumentation, W2 parity/collapse, W3b, W4 two-step)."""

from __future__ import annotations

import torch
import pytest

from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    SpanTarget,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.eval.toolcall.evaluator import ToolCallEvaluator
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver
from spancall.lean.tensorize import tensorize_sample_lean
from spancall.lean.trainer import LeanTrainer
from spancall.model.masks import CandidateMaskBuilder, _RowCtx, classify_parity_miss
from spancall.infer.base import FakeTokenizer


class _FakeEncoder(torch.nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = torch.nn.Linear(1, hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = (input_ids.float() % 17).unsqueeze(-1) / 17.0
        return self.proj(x)


class _ConstantToolHead(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self._logits = logits

    def forward(self, tool_h: torch.Tensor, cls_h: torch.Tensor) -> torch.Tensor:
        b, t, _ = tool_h.shape
        vals = (self._logits + [-10.0] * t)[:t]
        return torch.tensor([vals] * b, dtype=tool_h.dtype, device=tool_h.device)


class _ConstantPointerHead(torch.nn.Module):
    """Forces every parameter to pick the same utterance candidate (collapse) or distinct ones."""

    def __init__(self, target_cand_idx: int = 0) -> None:
        super().__init__()
        self._idx = target_cand_idx

    def forward(self, param_h: torch.Tensor, H: torch.Tensor, candidate_masks: dict) -> dict:
        b, k, c = candidate_masks["starts"].shape
        logits = torch.full((b, k, c), -5.0, dtype=param_h.dtype, device=param_h.device)
        if c > self._idx:
            logits[:, :, self._idx] = 5.0
        return {"span_logits": logits}


def _model() -> LeanSpanCallModel:
    return LeanSpanCallModel(encoder=_FakeEncoder(16), head_dim=16, hidden_dim=16, max_span_width=8)


def _weather_schema() -> ToolDefinition:
    return ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters=[
            ParameterSchema(name="city", param_type="string", description="City name", required=True),
            ParameterSchema(name="country", param_type="string", description="Country name", required=True),
        ],
    )


def _search_schema() -> ToolDefinition:
    return ToolDefinition(
        name="search_docs",
        description="Search documents",
        parameters=[
            ParameterSchema(name="city", param_type="string", description="City name", required=True),
        ],
    )


# --------------------------------------------------------------------------- #
# W4: two-step resolver mode & telemetry
# --------------------------------------------------------------------------- #


def test_resolver_rejects_unknown_resolve_mode() -> None:
    with pytest.raises(ValueError, match="resolve_mode must be"):
        LeanResolver(model=_model(), max_seq_len=512, resolve_mode="three_step")


def test_two_step_mode_rerenders_t1_and_preserves_step_clocks() -> None:
    model = _model()
    model.tool_head_binary = _ConstantToolHead([5.0, -5.0])
    r = LeanResolver(
        model=model,
        max_seq_len=512,
        tool_objective="binary",
        tool_call_threshold=0.5,
        resolve_mode="two_step",
    )
    res = r.try_resolve_candidates(
        [_weather_schema(), _search_schema()],
        utterance="weather in Tokyo Japan",
    )
    assert res.features["resolve_mode"] == "two_step"
    assert res.features["step1_render_T"] == 2
    assert res.features["step2_render_T"] == 1
    assert res.features["step1_enc_ms"] >= 0.0
    assert res.features["step2_enc_ms"] >= 0.0
    assert res.features["enc_ms"] == pytest.approx(
        res.features["step1_enc_ms"] + res.features["step2_enc_ms"], rel=1e-6
    )


def test_two_step_mode_abstains_at_step1_without_running_step2() -> None:
    model = _model()
    model.tool_head_binary = _ConstantToolHead([-4.0, -4.0])
    r = LeanResolver(
        model=model,
        max_seq_len=512,
        tool_objective="binary",
        tool_call_threshold=0.5,
        resolve_mode="two_step",
    )
    res = r.try_resolve_candidates(
        [_weather_schema(), _search_schema()],
        utterance="weather in Tokyo Japan",
    )
    assert res.hit is False
    assert res.status == "abstain"
    assert res.abstain_reason == "tool_below_threshold"
    assert "step2_render_T" not in res.features


# --------------------------------------------------------------------------- #
# W0.1: tool-qualified parity spans in try_resolve_candidates
# --------------------------------------------------------------------------- #


def test_e2e_tool_qualified_parity_spans_match_and_wrong_tool_isolation() -> None:
    model = _model()
    # Pick get_weather (index 0)
    model.tool_head = _ConstantToolHead([5.0, -5.0])
    r = LeanResolver(model=model, max_seq_len=512, tool_objective="ce", resolve_mode="single_pass")
    utt = "weather in Tokyo Japan"
    # "Tokyo" is at [11, 16)
    res_ok = r.try_resolve_candidates(
        [_weather_schema(), _search_schema()],
        utterance=utt,
        parity_spans={"get_weather::city": [(11, 16, "Tokyo")]},
    )
    assert res_ok.features["parity_checked"] == 1
    assert res_ok.features["parity_hit"] == 1
    assert res_ok.features["parity_wrong_tool"] == 0

    # When GT tool was search_docs (sharing param name 'city'), must NOT collide
    res_wrong = r.try_resolve_candidates(
        [_weather_schema(), _search_schema()],
        utterance=utt,
        parity_spans={"search_docs::city": [(11, 16, "Tokyo")]},
    )
    assert res_wrong.features["parity_checked"] == 0
    assert res_wrong.features["parity_wrong_tool"] == 1


# --------------------------------------------------------------------------- #
# W0.2 + W2.3: pointer collapse audit & guard
# --------------------------------------------------------------------------- #


def test_collapse_guard_demotes_utterance_collapse_only() -> None:
    model = _model()
    model.tool_head = _ConstantToolHead([5.0])
    model.pointer_head = _ConstantPointerHead(target_cand_idx=1)
    r_audit = LeanResolver(model=model, max_seq_len=512, collapse_guard=False)
    r_guard = LeanResolver(model=model, max_seq_len=512, collapse_guard=True)

    schema = _weather_schema()
    res_a = r_audit.try_resolve(schema, utterance="Tokyo")
    assert res_a.hit is True
    assert res_a.features["pointer_collapsed"] is True
    assert set(res_a.features["collapsed_params"]) == {"city", "country"}
    assert res_a.features["collapse_gated"] is False

    res_g = r_guard.try_resolve(schema, utterance="Tokyo")
    assert res_g.hit is False
    assert res_g.features["pointer_collapsed"] is True
    assert res_g.features["collapse_gated"] is True
    assert str(res_g.abstain_reason).startswith("pointer_collapse:")


def test_collapse_guard_ignores_shared_null_optional_params() -> None:
    schema = ToolDefinition(
        name="opt_tool",
        description="Optional params",
        parameters=[
            ParameterSchema(name="a", param_type="string", description="Param a", required=False),
            ParameterSchema(name="b", param_type="string", description="Param b", required=False),
        ],
    )
    # Use evaluator to verify non-utterance (null) spans don't trigger collapse_hit
    sample = UnifiedSample(
        sample_id="s1",
        stratum=Stratum.F,
        source_dataset="test",
        utterance="hello",
        tools=[schema],
        ground_truth_calls=[GroundTruthCall(tool_name="opt_tool", arguments={})],
    )
    pred = {
        "sample_id": "s1",
        "hit": True,
        "tool_name": "opt_tool",
        "payload": {},
        "parameters": {
            "a": {"value": None, "source_type": "null", "char_span": [1, 5]},
            "b": {"value": None, "source_type": "null", "char_span": [1, 5]},
        },
        "features": {"pointer_collapsed": False},
    }
    m = ToolCallEvaluator().evaluate([sample], [pred])
    assert m.collapse_hits == 0
    assert m.collapse_hit_rate == 0.0


def test_evaluator_reports_collapse_hit_rate_on_utterance_collapse() -> None:
    schema = _weather_schema()
    sample = UnifiedSample(
        sample_id="s_col",
        stratum=Stratum.F,
        source_dataset="test",
        utterance="USA",
        tools=[schema],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"city": "NYC", "country": "USA"})],
    )
    pred = {
        "sample_id": "s_col",
        "hit": True,
        "tool_name": "get_weather",
        "payload": {"city": "USA", "country": "USA"},
        "parameters": {
            "city": {"value": "USA", "source_type": "utterance", "char_span": [0, 3]},
            "country": {"value": "USA", "source_type": "utterance", "char_span": [0, 3]},
        },
    }
    m = ToolCallEvaluator().evaluate([sample], [pred])
    assert m.collapse_hits == 1
    assert m.collapse_hit_rate == pytest.approx(1.0)
    assert m.metrics_by_stratum["F"]["collapse_hit_rate"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# W0.3: trainer exact_call excludes zero-positive (Stratum-A) rows
# --------------------------------------------------------------------------- #


def test_trainer_evaluate_excludes_zero_positive_rows_from_exact_call() -> None:
    t = LeanTrainer(model=_model(), lr_encoder=1e-3, lr_head=1e-3, device="cpu", use_amp=False)
    B, L, K, C = 2, 12, 2, 4
    batch = {
        "input_ids": torch.randint(0, 50, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "param_positions": torch.tensor([[1, 3], [2, 4]]),
        "cand_starts": torch.randint(0, 4, (B, K, C)),
        "cand_ends": torch.randint(4, 8, (B, K, C)),
        # Row 0 has no supervised params at all (-1 everywhere); Row 1 has supervised params
        "pos_indices": torch.tensor([[[-1, -1], [-1, -1]], [[0, -1], [1, -1]]]),
        "_lean_converted": True,
        "tool_positions": torch.tensor([[0, 5], [0, 6]]),
        "tool_targets": torch.tensor([0, 1]),
    }
    ev = t.evaluate([batch])
    # Exact call is evaluated only on the 1 eligible row (not inflated by row 0)
    assert ev["exact_call"] in (0.0, 1.0)


# --------------------------------------------------------------------------- #
# W2.2: parity miss reason classification
# --------------------------------------------------------------------------- #


def test_classify_parity_miss_reasons() -> None:
    tok = FakeTokenizer()
    text = "weather in New York City tomorrow"
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    ctx = _RowCtx.for_row([], offsets, max_tok=len(offsets))

    p_str = ParameterSchema(name="loc", param_type="string", description="Location")
    p_int = ParameterSchema(name="count", param_type="integer", description="Count")

    assert classify_parity_miss(p_str, 999, 1005, ctx, max_span_width=8) == "never_generated_entrysnap"
    assert classify_parity_miss(p_int, 0, 7, ctx, max_span_width=8) == "never_generated_typefilter"
    assert classify_parity_miss(p_str, 0, len(text), ctx, max_span_width=2) == "never_generated_width"
    assert classify_parity_miss(p_str, 0, 7, ctx, max_span_width=8) == "truncated"


# --------------------------------------------------------------------------- #
# W3b: force_single_tool in tensorize_sample_lean
# --------------------------------------------------------------------------- #


def test_tensorize_force_single_tool_renders_t1() -> None:
    t1 = _weather_schema()
    t2 = _search_schema()
    sample = UnifiedSample(
        sample_id="s_t1",
        stratum=Stratum.F,
        source_dataset="test",
        utterance="weather in Tokyo Japan",
        tools=[t2, t1],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo", "country": "Japan"})],
        target_spans={
            "city": [SpanTarget(param_name="city", target_type="utterance_span", canonical_value="Tokyo", utt_rel_start=11, utt_rel_end=16)],
            "country": [SpanTarget(param_name="country", target_type="utterance_span", canonical_value="Japan", utt_rel_start=17, utt_rel_end=22)],
        },
    )
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=8)
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_tools=3, force_single_tool=True)
    assert row is not None
    assert int(row["tool_positions"].numel()) == 1
    assert int(row["tool_targets"].item()) == 0
