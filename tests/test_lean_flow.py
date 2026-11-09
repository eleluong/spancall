"""Unit tests for the Lean flow (spancall/lean)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    SpanTarget,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.infer.resolver import FakeTokenizer
from spancall.lean.decode import LeanJointDecoder
from spancall.lean.eval_p1 import (
    EXTRACTABLE_TARGET_TYPES,
    e3_aligner_violation,
    eval_param_extraction_lean,
    param_is_groundable,
)
from spancall.lean.loss import LeanLossContainer, compute_lean_pointer_loss
from spancall.lean.model import LeanPointerHead, LeanSpanCallModel
from spancall.lean.resolver import LeanResolver
from spancall.lean.tensorize import (
    LeanBatchCollate,
    convert_legacy_batch_to_lean,
    tensorize_sample_lean,
)
from spancall.lean.trainer import LeanTrainer
from spancall.model.masks import CandidateMaskBuilder


class TinyDummyEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(1000, hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.emb(input_ids)


def test_lean_pointer_head_shape() -> None:
    head = LeanPointerHead(hidden_dim=64, head_dim=32, max_span_width=8)
    B, K, L, C = 2, 3, 20, 5
    param_h = torch.randn(B, K, 64)
    H = torch.randn(B, L, 64)
    cand_starts = torch.randint(0, 10, (B, K, C))
    cand_ends = cand_starts + torch.randint(1, 4, (B, K, C))

    res = head(param_h, H, candidate_masks={"starts": cand_starts, "ends": cand_ends})
    assert "span_logits" in res
    assert res["span_logits"].shape == (B, K, C)
    assert "abstain_logits" not in res


def test_lean_pointer_loss() -> None:
    z = torch.tensor([
        [[2.0, 0.0, -1.0, -1e4], [0.0, 3.0, 0.0, -1e4]],
        [[0.0, 0.0, 5.0, -1e4], [-1e4, -1e4, -1e4, -1e4]],
    ])
    pos = torch.tensor([
        [[0, -1], [1, -1]],
        [[2, -1], [-1, -1]],
    ])

    loss = compute_lean_pointer_loss(z, pos)
    assert loss.shape == (2, 2)
    assert loss[1, 1] == 0.0
    assert (loss >= 0.0).all()

    container = LeanLossContainer()
    out = container(
        predictions={"span_logits": z},
        targets={"pointer_positives": pos},
    )
    assert "loss" in out
    assert out["loss"] > 0


def test_lean_decoder_greedy_fallback() -> None:
    decoder = LeanJointDecoder(overlap_allowed=False, free_string_overlap_allowed=False)
    cand1 = SimpleNamespace(
        canon_value="value_A", char_start=10, char_end=20, region="utterance", entry_index=0
    )
    cand2 = SimpleNamespace(
        canon_value="value_B", char_start=10, char_end=20, region="utterance", entry_index=0
    )
    # 2 params sharing identical utterance span -> conflict under strict exclusivity
    per_param = {
        "arg1": [(cand1, 0.9)],
        "arg2": [(cand2, 0.8)],
    }
    assignment, delta, chosen, n_valid = decoder.decode_detailed(per_param)
    # Greedy fallback prevents empty return {}
    assert len(assignment) == 2
    assert assignment["arg1"] == "value_A"
    assert assignment["arg2"] == "value_B"


def test_convert_legacy_batch_to_lean_and_idempotency() -> None:
    # Legacy batch: 0 = abstain, 1 = cand 0, 2 = cand 1, -1 = pad
    legacy_pos = torch.tensor([
        [[0, -1], [1, 2]],
        [[-1, -1], [3, -1]],
    ])
    batch = {"pos_indices": legacy_pos}
    lean_batch = convert_legacy_batch_to_lean(batch)
    expected = torch.tensor([
        [[-1, -1], [0, 1]],
        [[-1, -1], [2, -1]],
    ])
    assert torch.equal(lean_batch["pos_indices"], expected)
    # Calling convert_legacy_batch_to_lean a second time must be a no-op (not subtract 1 again!)
    lean_batch_2 = convert_legacy_batch_to_lean(lean_batch)
    assert torch.equal(lean_batch_2["pos_indices"], expected)


def test_tensorize_sample_lean() -> None:
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    schema = ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters=[
            ParameterSchema(name="city", description="City name", param_type="string", required=True),
            ParameterSchema(
                name="unit",
                description="Temperature unit",
                param_type="string",
                required=False,
                default="celsius",
            ),
        ],
    )
    sample = UnifiedSample(
        sample_id="s1",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo"})],
        recent_context=[{"key": "home_city", "value": "Osaka", "source": "recent"}],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_multi_tools=True)
    assert row is not None
    assert row["pos_indices"].shape[0] == 2
    # Both city ("Tokyo" in utterance) and unit (optional omitted -> <|null|>) should have 0-indexed positives >= 0
    assert (row["pos_indices"][0] >= 0).any()
    assert (row["pos_indices"][1] >= 0).any()

    batch = LeanBatchCollate()([row])
    assert "pointer_positives" in batch
    assert batch["cand_starts"].dim() == 3


def test_lean_trainer_train_and_eval() -> None:
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    trainer = LeanTrainer(model=model, lr_encoder=1e-3, lr_head=1e-3, device="cpu", use_amp=False)

    B, L, K, C = 2, 15, 2, 4
    batch = {
        "input_ids": torch.randint(0, 100, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "param_positions": torch.tensor([[1, 3], [2, 4]]),
        "cand_starts": torch.randint(0, 5, (B, K, C)),
        "cand_ends": torch.randint(5, 10, (B, K, C)),
        "pos_indices": torch.tensor([[[0, -1], [1, -1]], [[-1, -1], [2, -1]]]),
        "_lean_converted": True,
        "tool_positions": torch.tensor([[0, 5], [0, 6]]),
        "tool_targets": torch.tensor([0, 1]),
    }

    metrics = trainer.train_step(batch)
    assert "loss" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))
    assert "span_acc" in metrics

    ev = trainer.evaluate([batch])
    assert "eval_span_acc" in ev
    assert "exact_payload_match" in ev


def test_lean_resolver_resolution_and_predict_tool() -> None:
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)

    schema1 = ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters=[
            ParameterSchema(name="city", description="City name", param_type="string", required=True),
            ParameterSchema(
                name="unit",
                description="Temperature unit",
                param_type="string",
                required=False,
                default="celsius",
            ),
        ],
    )
    schema2 = ToolDefinition(
        name="get_time",
        description="Get current time in a city",
        parameters=[
            ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ],
    )

    res = resolver.try_resolve(
        schema1,
        utterance="weather in Tokyo",
        recent_context=[{"key": "last_city", "value": "Kyoto"}],
        now="2026-11-09",
    )
    assert isinstance(res.hit, bool)
    assert res.tool_name == "get_weather"
    assert hasattr(res, "accept_score")
    assert hasattr(res, "abstain_reason")
    for pr in res.parameters.values():
        assert hasattr(pr, "p_abstain")
        assert pr.p_abstain == 0.0

    # Test multi-tool H1 prediction and resolution
    tool_pred = resolver.predict_tool([schema1, schema2], utterance="weather in Tokyo")
    assert tool_pred["tool_name"] in ("get_weather", "get_time")
    assert len(tool_pred["tool_scores"]) == 2
    assert tool_pred["gated_hit"] is True

    res_multi = resolver.try_resolve_candidates([schema1, schema2], utterance="weather in Tokyo")
    assert res_multi.tool_name in ("get_weather", "get_time")


def test_lean_trainer_load_rows_filters_empty_abstain(tmp_path) -> None:
    import json

    t2_path = tmp_path / "t2.jsonl"
    # Row 1: legacy Stratum-F row (pos_indices has 2 -> becomes 1 in Lean)
    r_f = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "param_positions": [1],
        "cand_starts": [[2, 3]],
        "cand_ends": [[2, 3]],
        "pos_indices": [[2, -1]],
        "accept_targets": 1.0,
    }
    # Row 2: legacy Stratum-A row (pos_indices has only 0=abstain -> becomes -1 in Lean)
    r_a = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "param_positions": [1],
        "cand_starts": [[2, 3]],
        "cand_ends": [[2, 3]],
        "pos_indices": [[0, -1]],
        "accept_targets": 0.0,
    }
    with open(t2_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(r_f) + "\n")
        f.write(json.dumps(r_a) + "\n")

    cfg = SimpleNamespace(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        experiment_name="test_lean",
        seed=42,
        gradient_accumulation_steps=1,
        optimization=SimpleNamespace(
            batch_size=2,
            num_epochs=1,
            learning_rate_encoder=1e-3,
            learning_rate_head=1e-3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            mixed_precision="fp32",
        ),
    )
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    trainer = LeanTrainer(model=model, config=cfg, device="cpu", use_amp=False)

    loaded = trainer.load_rows("t2")
    # Only the Stratum-F groundable row should be retained, converted to 0-indexed pos_indices [[1, -1]]
    assert len(loaded) == 1
    assert loaded[0]["pos_indices"] == [[1, -1]]
    assert loaded[0]["_lean_converted"] is True

    # Full train(stages=("T2",)) over tmp_path
    summary = trainer.train(stages=("T2",))
    assert summary["T2"]["status"] == "ok"


# --------------------------------------------------------------------------- #
# Retrieval-first contract: supervision audit + groundable-only eval
# --------------------------------------------------------------------------- #

def _weather_schema(*params: ParameterSchema) -> ToolDefinition:
    return ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters=list(params),
    )


def test_tensorize_classifies_unresolved_target_drop() -> None:
    """A param whose only target is `unresolved` is excluded from pointer loss.

    Under the retrieval-first contract that exclusion is CORRECT (the value must
    be generated), but it must be *classified*, not silently swallowed.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    schema = _weather_schema(
        ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ParameterSchema(name="advice", description="Free-form advice", param_type="string", required=True),
    )
    sample = UnifiedSample(
        sample_id="s-unresolved",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[
            GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo", "advice": "bring an umbrella"})
        ],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
        target_spans={
            "city": [SpanTarget(param_name="city", target_type="utterance_span", canonical_value="Tokyo")],
            "advice": [SpanTarget(param_name="advice", target_type="unresolved")],
        },
    )

    stats: dict = {}
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_multi_tools=True, stats=stats)
    assert row is not None
    # city still supervised; advice dropped with the *expected* reason.
    assert (row["pos_indices"][0] >= 0).any()
    assert stats["param_drop_reasons"] == {"unresolved_target": 1}
    assert row["_param_drop_reasons"] == ["unresolved_target"]


def test_tensorize_flags_match_missed_for_extractable_target() -> None:
    """An *extractable* target the matcher cannot resolve is a bug indicator.

    `match_missed` must be reported separately from `unresolved_target`; it is
    the counter that proves loss is not silently ignoring groundable params.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    schema = _weather_schema(
        ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ParameterSchema(name="secret", description="Hidden token", param_type="string", required=True),
    )
    sample = UnifiedSample(
        sample_id="s-missed",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[
            GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo", "secret": "ZZZ-not-present"})
        ],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
        target_spans={
            "city": [SpanTarget(param_name="city", target_type="utterance_span", canonical_value="Tokyo")],
            # extractable in *type*, but absent from the context -> matcher miss
            "secret": [SpanTarget(param_name="secret", target_type="utterance_span", canonical_value="ZZZ-not-present")],
        },
    )

    stats: dict = {}
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_multi_tools=True, stats=stats)
    assert stats["param_drop_reasons"] == {"match_missed": 1}
    assert row["_param_drop_reasons"] == ["match_missed"]


def test_tensorize_required_param_without_gt_is_not_match_missed() -> None:
    """C3 regression: a targetless param with no GT arg is `absent_from_gt`.

    The old classifier called this `match_missed`, producing a false alarm in
    the tripwire for a param that simply is not part of the call.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    schema = _weather_schema(
        ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ParameterSchema(name="note", description="Free note", param_type="string", required=True),
    )
    sample = UnifiedSample(
        sample_id="s-absent",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
        target_spans={
            "city": [SpanTarget(param_name="city", target_type="utterance_span", canonical_value="Tokyo")],
            # no target for "note" and no GT arg -> benign absence
        },
    )

    stats: dict = {}
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_multi_tools=True, stats=stats)
    assert row is not None
    assert stats["param_drop_reasons"] == {"absent_from_gt": 1}


def test_tensorize_unresolved_only_call_has_no_supervision_bug() -> None:
    """C1 regression: an explicit-null GT param is supervised, not dropped.

    `unit=None` with a `null_ctx` target must map to the deterministic
    `<|null|>` slot, so it must NOT produce a drop reason at all.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    schema = _weather_schema(
        ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ParameterSchema(name="unit", description="Unit", param_type="string", required=False),
    )
    sample = UnifiedSample(
        sample_id="s-null-supervised",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[
            GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo", "unit": None})
        ],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
        target_spans={
            "city": [SpanTarget(param_name="city", target_type="utterance_span", canonical_value="Tokyo")],
            "unit": [SpanTarget(param_name="unit", target_type="null_ctx")],
        },
    )

    stats: dict = {}
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_multi_tools=True, stats=stats)
    assert row is not None
    assert stats.get("param_drop_reasons", {}) == {}
    assert "_param_drop_reasons" not in row
    # both params carry a positive slot
    assert all((p >= 0).any() for p in row["pos_indices"])


def test_tensorize_null_target_uses_single_deterministic_slot() -> None:
    """`null_ctx` supervision resolves to exactly one candidate index."""
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    schema = _weather_schema(
        ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ParameterSchema(name="unit", description="Unit", param_type="string", required=False),
    )
    sample = UnifiedSample(
        sample_id="s-null",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo", "unit": None})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
        target_spans={
            "city": [SpanTarget(param_name="city", target_type="utterance_span", canonical_value="Tokyo")],
            "unit": [SpanTarget(param_name="unit", target_type="null_ctx")],
        },
    )

    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_multi_tools=True)
    assert row is not None
    unit_pos = [pi for pi in row["pos_indices"][1].tolist() if pi >= 0]
    assert len(unit_pos) == 1, f"null_ctx must map to exactly one slot, got {unit_pos}"


def test_param_is_groundable_uses_extractable_target_types() -> None:
    schema = _weather_schema(
        ParameterSchema(name="city", description="c", param_type="string", required=True),
        ParameterSchema(name="advice", description="a", param_type="string", required=True),
    )
    sample = UnifiedSample(
        sample_id="s-ground",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo", "advice": "x"})],
        target_spans={
            "city": [SpanTarget(param_name="city", target_type="utterance_span", canonical_value="Tokyo")],
            "advice": [SpanTarget(param_name="advice", target_type="unresolved")],
        },
    )
    assert "utterance_span" in EXTRACTABLE_TARGET_TYPES
    assert param_is_groundable(sample, "city") is True
    assert param_is_groundable(sample, "advice") is False
    assert param_is_groundable(sample, "missing") is False


class _StubResolution(SimpleNamespace):
    """Minimal LeanParameterResolution stand-in (value/confidence/source_type)."""


class _StubResolveResult(SimpleNamespace):
    pass


class _StubResolver:
    """Resolver stub returning scripted per-param values (keyed by param name)."""

    def __init__(self, scripted: dict) -> None:
        self.scripted = scripted

    def try_resolve(self, tool, utterance, recent_context=None, now=None):
        params = {
            name: _StubResolution(value=value, confidence=0.9, source_type="utterance")
            for name, value in self.scripted.items()
        }
        pred = {k: v.value for k, v in params.items() if v.value is not None}
        return _StubResolveResult(
            parameters=params,
            payload_partial=pred,
            hit=bool(pred),
            tool_name=getattr(tool, "name", ""),
        )


def test_eval_param_extraction_scores_groundable_params_only(tmp_path) -> None:
    """Eval must not punish the fast path for refusing a generation-only param.

    `city` is extractable and predicted correctly; `advice` is `unresolved` and
    refused. Headline extraction metrics must be perfect (1.0), while the legacy
    blended `recall_all` drops to 0.5, and the refusal is booked as
    `refused_unextractable` (NOT as a fast-path miss).
    """
    schema = _weather_schema(
        ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ParameterSchema(name="advice", description="Free-form advice", param_type="string", required=True),
    )
    record = {
        "sample_id": "e1",
        "source_dataset": "test",
        "utterance": "weather in Tokyo",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": [
                    {"name": "city", "description": "City", "param_type": "string", "required": True},
                    {"name": "advice", "description": "Advice", "param_type": "string", "required": True},
                ],
            }
        ],
        "ground_truth_calls": [
            {"tool_name": "get_weather", "arguments": {"city": "Tokyo", "advice": "bring an umbrella"}}
        ],
        "current_time_iso": "2026-11-09",
        "stratum": "F",
        "target_spans": {
            "city": [{"param_name": "city", "target_type": "utterance_span", "canonical_value": "Tokyo"}],
            "advice": [{"param_name": "advice", "target_type": "unresolved"}],
        },
    }
    import json
    with open(tmp_path / "eval_split.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    metrics = eval_param_extraction_lean(
        "eval_split",
        resolver=_StubResolver({"city": "Tokyo"}),
        data_dir=str(tmp_path),
        limit=10,
        max_param_gap_report=5,
    )

    assert metrics["n"] == 1
    # Headline: the extractable param was retrieved correctly.
    assert metrics["extraction_recall"] == 1.0
    assert metrics["extraction_precision"] == 1.0
    assert metrics["extraction_f1"] == 1.0
    # Legacy blended view mixes in the out-of-scope param -> 0.5.
    assert metrics["recall_all"] == 0.5
    # Coverage + refusal accounting.
    assert metrics["fast_path_param_coverage"] == 0.5
    assert metrics["fast_path_call_coverage"] == 0.0
    assert metrics["missed_extractable"] == 0
    assert metrics["refused_unextractable"] == 1
    # Exact match requires the out-of-scope param too -> correctly 0 here.
    assert metrics["exact_payload_match_F"] == 0.0
    assert metrics["exact_payload_match_F_n"] == 0
    assert metrics["stratum_f_n"] == 1
    # S5: the resolver's own hit flag is reported under a non-misleading name.
    assert "hit_rate_F" not in metrics
    assert "resolver_hit_rate_F" in metrics
    # S6: truncation accounting is present.
    assert metrics["truncation"] == {
        "evaluated": 1,
        "total_available": 1,
        "truncated": False,
    }


def _write_split(tmp_path, records, name="eval_split"):
    import json
    with open(tmp_path / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return name


def _weather_record(arguments, target_spans, sample_id="r", stratum="F"):
    return {
        "sample_id": sample_id,
        "source_dataset": "test",
        "utterance": "weather in Tokyo",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": [
                    {"name": "city", "description": "City", "param_type": "string", "required": True},
                    {"name": "unit", "description": "Unit", "param_type": "string?null", "required": False},
                ],
            }
        ],
        "ground_truth_calls": [{"tool_name": "get_weather", "arguments": arguments}],
        "current_time_iso": "2026-11-09",
        "stratum": stratum,
        "target_spans": target_spans,
    }


def test_eval_counts_explicit_null_gt_param_as_groundable(tmp_path) -> None:
    """C1+C2 regression: a null-only call is 100% fast-path coverable.

    Before the fix, `unit=None` was filtered out of the GT universe, so
    coverage was 0.5 and a null-only call could never be "fully extractable".
    """
    name = _write_split(tmp_path, [
        _weather_record(
            {"city": "Tokyo", "unit": None},
            {
                "city": [{"param_name": "city", "target_type": "utterance_span", "canonical_value": "Tokyo"}],
                "unit": [{"param_name": "unit", "target_type": "null_ctx"}],
            },
        )
    ])

    class _NullResolver:
        def try_resolve(self, tool, utterance, recent_context=None, now=None):
            params = {
                "city": _StubResolution(value="Tokyo", confidence=0.9, source_type="utterance"),
                "unit": _StubResolution(value=None, confidence=0.9, source_type="null"),
            }
            return _StubResolveResult(
                parameters=params,
                payload_partial={"city": "Tokyo"},
                hit=True,
                tool_name=getattr(tool, "name", ""),
            )

    metrics = eval_param_extraction_lean(
        name, resolver=_NullResolver(), data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )
    # Both GT params are in scope and both were retrieved correctly.
    assert metrics["fast_path_param_coverage"] == 1.0
    assert metrics["fast_path_call_coverage"] == 1.0
    assert metrics["extraction_recall"] == 1.0
    assert metrics["extraction_precision"] == 1.0
    assert metrics["missed_extractable"] == 0
    assert metrics["pointer_wrong"] == 0
    assert metrics["exact_payload_match_F"] == 1.0
    assert metrics["param_counts"]["total_gt_params"] == 2


def test_eval_counts_null_param_false_positive_as_pointer_wrong(tmp_path) -> None:
    """C4 regression: predicting a value for a null GT param is `pointer_wrong`.

    Previously the param was absent from the GT universe, so the FP silently
    vanished from `pointer_wrong` while still hurting precision.
    """
    name = _write_split(tmp_path, [
        _weather_record(
            {"city": "Tokyo", "unit": None},
            {
                "city": [{"param_name": "city", "target_type": "utterance_span", "canonical_value": "Tokyo"}],
                "unit": [{"param_name": "unit", "target_type": "null_ctx"}],
            },
        )
    ])

    class _BadResolver:
        def try_resolve(self, tool, utterance, recent_context=None, now=None):
            params = {
                "city": _StubResolution(value="Tokyo", confidence=0.9, source_type="utterance"),
                "unit": _StubResolution(value="fahrenheit", confidence=0.9, source_type="utterance"),
            }
            return _StubResolveResult(
                parameters=params,
                payload_partial={"city": "Tokyo", "unit": "fahrenheit"},
                hit=True,
                tool_name=getattr(tool, "name", ""),
            )

    metrics = eval_param_extraction_lean(
        name, resolver=_BadResolver(), data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )
    assert metrics["pointer_wrong"] == 1
    assert metrics["param_counts"]["extractable_fp"] == 1
    # A wrong value is a false POSITIVE, so precision halves but recall holds.
    assert metrics["extraction_precision"] == 0.5
    assert metrics["extraction_recall"] == 1.0


def test_eval_reports_nan_precision_when_nothing_predicted(tmp_path) -> None:
    """S1 regression: zero predictions must be `n/a`, not a fake precision of 1.0."""
    import math
    name = _write_split(tmp_path, [
        _weather_record(
            {"city": "Tokyo"},
            {"city": [{"param_name": "city", "target_type": "utterance_span", "canonical_value": "Tokyo"}]},
        )
    ])

    class _RefusingResolver:
        def try_resolve(self, tool, utterance, recent_context=None, now=None):
            return _StubResolveResult(parameters={}, payload_partial={}, hit=False, tool_name="get_weather")

    metrics = eval_param_extraction_lean(
        name, resolver=_RefusingResolver(), data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )
    assert math.isnan(metrics["extraction_precision"])
    assert metrics["extraction_recall"] == 0.0
    assert math.isnan(metrics["extraction_f1"])
    assert metrics["missed_extractable"] == 1


def test_eval_e3_tripwire_flags_stratum_f_aligner_drift(tmp_path) -> None:
    """E3 regression: a Stratum-F GT arg with no extractable target is drift."""
    name = _write_split(tmp_path, [
        _weather_record(
            {"city": "Tokyo", "unit": None},
            {"city": [{"param_name": "city", "target_type": "utterance_span", "canonical_value": "Tokyo"}]},
            sample_id="drift-1",
        )
    ])

    class _OkResolver:
        def try_resolve(self, tool, utterance, recent_context=None, now=None):
            params = {"city": _StubResolution(value="Tokyo", confidence=0.9, source_type="utterance")}
            return _StubResolveResult(
                parameters=params, payload_partial={"city": "Tokyo"}, hit=True, tool_name="get_weather"
            )

    metrics = eval_param_extraction_lean(
        name, resolver=_OkResolver(), data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )
    # `unit` has a GT arg but no target row at all -> aligner drift.
    assert metrics["e3_aligner_drift"] == 1
    assert metrics["e3_examples"] == ["drift-1:unit:no_target"]


def test_e3_aligner_violation_accepts_clean_and_unresolved_cases() -> None:
    schema = _weather_schema(
        ParameterSchema(name="city", description="c", param_type="string", required=True),
        ParameterSchema(name="advice", description="a", param_type="string", required=True),
    )
    clean = UnifiedSample(
        sample_id="ok",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[
            GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo", "advice": "x"})
        ],
        target_spans={
            "city": [SpanTarget(param_name="city", target_type="utterance_span", canonical_value="Tokyo")],
            "advice": [SpanTarget(param_name="advice", target_type="unresolved")],
        },
    )
    assert e3_aligner_violation(clean, {"city": "Tokyo"}, schema) is None
    # An `unresolved` target is NOT extractable -> violation for that param.
    assert e3_aligner_violation(clean, {"advice": "x"}, schema) == "advice:unresolved"


def test_eval_raises_for_missing_split(tmp_path) -> None:
    """S4 regression: a missing split must be loud, not a silent zero-metric run."""
    with pytest.raises(FileNotFoundError, match="Eval split not found"):
        eval_param_extraction_lean(
            "does_not_exist",
            resolver=_StubResolver({}),
            data_dir=str(tmp_path),
            limit=10,
        )

