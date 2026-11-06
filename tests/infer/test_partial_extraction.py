"""Tests for partial parameter extraction and continuous confidence gating (Part 4, P1-P6).

Covers:
* P1 — Never drop extracted values: payload_partial, missing_required, missing_optional.
* P2 — Continuous confidence: per-param c_k = p_val * (1 - p_abs), call confidence.
* P3 — Three outcomes: status in ("hit", "partial", "abstain").
* P5 — ToolCallEvaluator partial extraction metrics: param_precision, param_recall, param_f1.
* P6 — Acceptance test suite:
  (a) 2 of 3 params grounded -> payload_partial has 2 correct values, missing_required has 3rd, confidence < fully grounded.
  (b) fully grounded -> status == 'hit', payload valid.
  (c) fully absent -> status == 'abstain', confidence == 0.0.
  (d) monotonicity -> removing a grounded param strictly lowers confidence.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List
import pytest
import torch
from torch import nn

from spancall.data.schema import ParameterSchema, ToolDefinition, UnifiedSample, Stratum, GroundTruthCall
from spancall.eval.toolcall.evaluator import ToolCallEvaluator, payload_equal
from spancall.infer.decode import ValueChoice
from spancall.infer.resolver import ParameterResolution, ResolveResult, SpanCallResolver


class DummyEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        return torch.zeros(B, L, self.hidden_dim, device=input_ids.device)


def _make_schema(num_params: int = 3, num_required: int = 3) -> ToolDefinition:
    params = []
    for i in range(num_params):
        params.append(
            ParameterSchema(
                name=f"param_{i}",
                description=f"Parameter {i}",
                param_type="string",
                required=(i < num_required),
            )
        )
    return ToolDefinition(name="test_tool", description="Test Tool", parameters=params)


def test_p1_resolve_result_fields() -> None:
    res = ResolveResult(
        hit=False,
        status="partial",
        payload=None,
        payload_partial={"param_0": "val0", "param_1": "val1"},
        missing_required=["param_2"],
        missing_optional=[],
        confidence=0.45,
    )
    assert res.hit is False
    assert res.status == "partial"
    assert res.payload is None
    assert res.payload_partial == {"param_0": "val0", "param_1": "val1"}
    assert res.missing_required == ["param_2"]
    assert res.confidence == 0.45


def test_p6_acceptance_suite() -> None:
    """Tests P6 (a), (b), (c), (d)."""
    schema = _make_schema(num_params=3, num_required=3)
    resolver = SpanCallResolver(encoder=DummyEncoder(), max_seq_len=1024)

    # 1. Fully grounded twin (3 of 3 params grounded with high confidence)
    def mock_heads_full(output, H, cls_h, per_token, cand_masks):
        outcome = {
            "per_param_scores": {
                "param_0": [(type("Cand", (), {"canon_value": "A", "char_start": 0, "char_end": 1, "region": "utt"})(), 0.95)],
                "param_1": [(type("Cand", (), {"canon_value": "B", "char_start": 2, "char_end": 3, "region": "utt"})(), 0.92)],
                "param_2": [(type("Cand", (), {"canon_value": "C", "char_start": 4, "char_end": 5, "region": "utt"})(), 0.90)],
            },
            "per_param_abstain_p": {"param_0": 0.05, "param_1": 0.08, "param_2": 0.10},
            "abstain_p": 0.076,
            "accept_score": 0.95,
        }
        return {}, outcome

    resolver._run_heads = mock_heads_full  # type: ignore[assignment]
    res_full = resolver.try_resolve(schema, "A B C")

    # (b) Fully grounded: hit == True, status == 'hit', all 3 in payload
    assert res_full.hit is True
    assert res_full.status == "hit"
    assert res_full.payload == {"param_0": "A", "param_1": "B", "param_2": "C"}
    assert res_full.payload_partial == {"param_0": "A", "param_1": "B", "param_2": "C"}
    assert len(res_full.missing_required) == 0
    conf_full = res_full.confidence
    assert conf_full > 0.70

    # 2. Partially grounded (2 of 3 params grounded, param_2 has candidate spans but abstain wins!)
    def mock_heads_partial(output, H, cls_h, per_token, cand_masks):
        outcome = {
            "per_param_scores": {
                "param_0": [(type("Cand", (), {"canon_value": "A", "char_start": 0, "char_end": 1, "region": "utt"})(), 0.95)],
                "param_1": [(type("Cand", (), {"canon_value": "B", "char_start": 2, "char_end": 3, "region": "utt"})(), 0.92)],
                "param_2": [(type("Cand", (), {"canon_value": "NOISE", "char_start": 0, "char_end": 1, "region": "utt"})(), 0.04)],
            },
            "per_param_abstain_p": {"param_0": 0.05, "param_1": 0.08, "param_2": 0.96},
            "abstain_p": 0.363,
            "accept_score": 0.40,
        }
        return {}, outcome

    resolver._run_heads = mock_heads_partial  # type: ignore[assignment]
    res_partial = resolver.try_resolve(schema, "A B")

    # (a) 2/3 params grounded:
    assert res_partial.hit is False
    assert res_partial.status == "partial"
    assert res_partial.payload is None  # schema missing required param_2
    assert res_partial.payload_partial == {"param_0": "A", "param_1": "B"}
    assert res_partial.missing_required == ["param_2"]
    assert res_partial.parameters["param_2"].source_type == "abstain"
    assert res_partial.parameters["param_2"].value == "NOISE"
    assert res_partial.parameters["param_2"].p_abstain == pytest.approx(0.96)
    # (d) Monotonicity: confidence strictly between 1/3 grounded and 3/3 grounded
    assert 0.0 < res_partial.confidence < conf_full
    assert res_partial.confidence == pytest.approx(0.92 * (2.0 / 3.0))

    # 2b. 1 of 3 params grounded (param_0 grounded, param_1 and param_2 abstain)
    def mock_heads_one(output, H, cls_h, per_token, cand_masks):
        outcome = {
            "per_param_scores": {
                "param_0": [(type("Cand", (), {"canon_value": "A", "char_start": 0, "char_end": 1, "region": "utt"})(), 0.95)],
                "param_1": [(type("Cand", (), {"canon_value": "NOISE", "char_start": 0, "char_end": 1, "region": "utt"})(), 0.05)],
                "param_2": [],
            },
            "per_param_abstain_p": {"param_0": 0.05, "param_1": 0.95, "param_2": 0.99},
            "abstain_p": 0.663,
            "accept_score": 0.20,
        }
        return {}, outcome

    resolver._run_heads = mock_heads_one  # type: ignore[assignment]
    res_one = resolver.try_resolve(schema, "A")
    assert res_one.status == "partial"
    assert res_one.payload_partial == {"param_0": "A"}
    assert set(res_one.missing_required) == {"param_1", "param_2"}
    assert 0.0 < res_one.confidence < res_partial.confidence

    # 3. Fully absent (0 of 3 params grounded)
    def mock_heads_absent(output, H, cls_h, per_token, cand_masks):
        outcome = {
            "per_param_scores": {"param_0": [], "param_1": [], "param_2": []},
            "per_param_abstain_p": {"param_0": 0.99, "param_1": 0.99, "param_2": 0.99},
            "abstain_p": 0.99,
            "accept_score": 0.01,
        }
        return {}, outcome

    resolver._run_heads = mock_heads_absent  # type: ignore[assignment]
    res_absent = resolver.try_resolve(schema, "nothing relevant")

    # (c) Fully absent: status == 'abstain', confidence == 0.0, payload_partial == {}
    assert res_absent.hit is False
    assert res_absent.status == "abstain"
    assert res_absent.payload is None
    assert res_absent.payload_partial == {}
    assert set(res_absent.missing_required) == {"param_0", "param_1", "param_2"}
    assert res_absent.confidence == 0.0


def test_p5_evaluator_partial_metrics() -> None:
    evaluator = ToolCallEvaluator()
    sample = UnifiedSample(
        sample_id="test_1",
        source_dataset="test",
        utterance="Call with 3 params",
        tools=[_make_schema(3, 3)],
        ground_truth_calls=[GroundTruthCall(tool_name="test_tool", arguments={"param_0": "A", "param_1": "B", "param_2": "C"})],
        stratum=Stratum.F,
    )

    # Prediction extracted 2 out of 3 parameters correctly
    pred = {
        "sample_id": "test_1",
        "hit": False,
        "status": "partial",
        "tool_name": "test_tool",
        "payload": None,
        "payload_partial": {"param_0": "A", "param_1": "B"},
        "missing_required": ["param_2"],
        "confidence": 0.60,
    }

    metrics = evaluator.evaluate([sample], [pred])
    assert metrics.exact_payload_match == 0.0
    assert metrics.hit_rate == 0.0
    assert metrics.partial_extraction_rate == 1.0
    # 2 correct out of 2 predicted = precision 1.0
    assert metrics.param_precision == 1.0
    # 2 correct out of 3 ground truth = recall 2/3
    assert metrics.param_recall == pytest.approx(2.0 / 3.0)
    assert metrics.param_f1 == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))


def test_date_iso_payload_equal() -> None:
    # GT from JSONL is ISO string, actual from Caster is date object
    gt = {"departure_date": "2026-10-23"}
    actual = {"departure_date": date(2026, 10, 23)}
    assert payload_equal(gt, actual) is True

    # Reverse
    assert payload_equal(actual, gt) is True
