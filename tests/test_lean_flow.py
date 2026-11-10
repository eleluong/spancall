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
    PAD_INDEX,
    LeanBatchCollate,
    ToolPool,
    _snap_token_range,
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


# --------------------------------------------------------------------------- #
# H1 vacuity fix: distractor candidate packing, -1 padding, masked tool_acc
# --------------------------------------------------------------------------- #

def _schema(name: str, params: int = 1) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        parameters=[
            ParameterSchema(name=f"p{i}", description=f"param {i}", param_type="string", required=False)
            for i in range(params)
        ],
    )


def test_tool_pool_packs_gt_plus_distractors() -> None:
    """The GT tool must always be present, at a shuffled index, with T == budget."""
    gt = _schema("get_weather", params=2)
    others = [_schema(f"tool_{i}") for i in range(6)]
    pool = ToolPool([gt, *others], seed=7)

    sample = UnifiedSample(
        sample_id="pack-1",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[gt],  # the Hermes reality: exactly ONE native tool
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    cands, gt_idx = pool.candidates_for(sample, gt, pack_tools=3)
    assert len(cands) == 3
    assert cands[gt_idx].name == "get_weather"
    assert len({c.name for c in cands}) == 3  # no duplicate distractor
    assert 0 <= gt_idx < 3


def test_tool_pool_is_deterministic_across_instances() -> None:
    """Same (sample_id, seed) => same distractors, even from a fresh pool object.

    This is what keeps tensorization reproducible inside the fork process pool.
    """
    gt = _schema("get_weather")
    tools = [gt, *[_schema(f"tool_{i}") for i in range(20)]]
    sample = UnifiedSample(
        sample_id="det-1",
        source_dataset="test",
        utterance="weather",
        tools=[gt],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={})],
        stratum=Stratum.F,
    )
    a = ToolPool(tools, seed=42).candidates_for(sample, gt, 4)
    b = ToolPool(tools, seed=42).candidates_for(sample, gt, 4)
    assert [t.name for t in a[0]] == [t.name for t in b[0]]
    assert a[1] == b[1]


def test_tool_pool_degrades_to_single_candidate_when_disabled() -> None:
    """pack_tools=1 must be honest: one candidate, GT at index 0 (the vacuous case)."""
    gt = _schema("get_weather")
    pool = ToolPool([gt, _schema("tool_x")], seed=42)
    sample = UnifiedSample(
        sample_id="single",
        source_dataset="test",
        utterance="weather",
        tools=[gt],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={})],
        stratum=Stratum.F,
    )
    cands, gt_idx = pool.candidates_for(sample, gt, pack_tools=1)
    assert len(cands) == 1 and gt_idx == 0 and cands[0].name == "get_weather"


def test_tensorize_emits_multi_candidate_tool_targets() -> None:
    """With a pool, tensorize must emit T > 1 and an in-range GT index.

    Uses a 1024-token budget: packing 3 tool schemas + the time window exceeds
    512 tokens, and over-length rows are (correctly) dropped by the tensorizer.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=1024, max_span_width=16)
    gt = _schema("get_weather", params=2)
    pool = ToolPool([gt, *[_schema(f"tool_{i}") for i in range(5)]], seed=11)
    sample = UnifiedSample(
        sample_id="t-multi",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[gt],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    row = tensorize_sample_lean(
        sample, tok, mb, max_seq_len=1024, pack_multi_tools=True,
        tool_pool=pool, pack_tools=3,
    )
    assert row is not None
    n_cands = int(row["_n_tool_candidates"])
    assert n_cands == 3, f"expected 3 packed candidates, got {n_cands}"
    assert row["tool_positions"].shape[0] == n_cands
    gt_idx = int(row["tool_targets"])
    assert 0 <= gt_idx < n_cands


def test_collate_pads_missing_tool_targets_with_minus_one() -> None:
    """A row without H1 supervision must be padded with -1, never 0."""
    row_a = {
        "input_ids": torch.tensor([1, 2, 3]),
        "param_positions": torch.tensor([1]),
        "pos_indices": torch.tensor([[0]]),
        "tool_positions": torch.tensor([0, 1]),
        "tool_targets": torch.tensor(1),
        "_lean_converted": True,
    }
    row_b = {  # no tool_positions / tool_targets at all
        "input_ids": torch.tensor([4, 5, 6]),
        "param_positions": torch.tensor([1]),
        "pos_indices": torch.tensor([[0]]),
        "_lean_converted": True,
    }
    batch = LeanBatchCollate()([row_a, row_b])
    assert "tool_targets" in batch, "tool_targets dropped => H1 supervision lost"
    assert batch["tool_targets"].tolist() == [1, -1]
    assert batch["tool_positions"][1].tolist() == [-1, -1]


def test_loss_ignores_padded_tool_targets() -> None:
    """-1 targets must be skipped, not raise, and must not dilute the gradient."""
    loss_fn = LeanLossContainer(lambda_pointer=1.0, lambda_tool=1.0)
    logits = torch.tensor([[0.0, 5.0], [3.0, -3.0]])
    preds = {"tool_logits": logits}
    # Row 0 supervised on tool 1; row 1 padded.
    out_mixed = loss_fn(preds, {"tool_targets": torch.tensor([1, -1])})
    out_row0 = loss_fn({"tool_logits": logits[:1]}, {"tool_targets": torch.tensor([1])})
    assert torch.isfinite(out_mixed["tool"])
    assert torch.allclose(out_mixed["tool"], out_row0["tool"], atol=1e-6)

    # All rows padded => tool term skipped entirely (no NaN poison).
    out_none = loss_fn(preds, {"tool_targets": torch.tensor([-1, -1])})
    assert float(out_none["tool"].item()) == 0.0
    assert torch.isfinite(out_none["loss"])


def test_trainer_tool_acc_is_masked_and_persisted() -> None:
    """tool_acc must ignore padded rows and epoch_eval must carry the H1 fields."""
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
        "tool_positions": torch.tensor([[0, 5, 6], [0, 5, 6]]),
        "tool_targets": torch.tensor([1, -1]),  # second row unsupervised
        "_lean_converted": True,
    }
    metrics = trainer.train_step(batch)
    assert metrics.get("n_tool") == 1.0, "padded row leaked into the tool_acc denominator"
    assert metrics.get("mean_n_candidates") == 3.0

    ev = trainer.evaluate([batch])
    assert ev["n_tool"] == 1.0
    assert "mean_n_candidates" in ev and ev["mean_n_candidates"] == 3.0
    assert "tool_acc" in ev

    # exact_payload_match is ROW-level (denominator = B = 2 rows): a row counts
    # only when ALL its supervised params AND its tool are right. It is
    # therefore a multiple of 1/B and is NOT interchangeable with span_acc,
    # whose denominator is the 3 active (row, param) pairs (multiples of 1/3).
    exact = ev["exact_payload_match"]
    assert 0.0 <= exact <= 1.0
    assert abs(exact * B - round(exact * B)) < 1e-6, (
        f"exact_payload_match={exact} is not row-level over {B} rows; "
        f"it was probably aliased back to span_acc"
    )
    assert abs(ev["span_acc"] * 3 - round(ev["span_acc"] * 3)) < 1e-6
    # ``exact_call`` is the canonical name; the old key stays as a deprecated
    # alias so pre-rename train_log.jsonl files still parse.
    assert ev["exact_call"] == ev["exact_payload_match"]


# --------------------------------------------------------------------------- #
# C2 / C1 / S5 regressions found in review
# --------------------------------------------------------------------------- #

def test_snap_token_range_returns_none_on_miss() -> None:
    """A range past the last token overlaps nothing => None (not a silent 0)."""
    offsets = [(0, 3), (3, 6), (6, 9)]
    assert _snap_token_range(0, 3, offsets, 3) == (0, 0)
    assert _snap_token_range(999, 1005, offsets, 3) is None
    assert _snap_token_range(5, 5, offsets, 3) is None  # degenerate range


def test_pad_index_is_negative_so_live_mask_excludes_it() -> None:
    """C2 contract: PAD_INDEX must be < 0, because callers mask on ``p < 0``."""
    assert PAD_INDEX < 0


def test_tensorize_drops_row_when_gt_tool_anchor_cannot_snap(monkeypatch) -> None:
    """C2: a dead GT anchor must DROP the row, not become ``<|cls|>``.

    ``_snap_token_range`` returns None when the bridge_text tool line cannot be
    mapped onto tokens. The old code fell back to index 0 (``<|cls|>``), which is
    a real token and is never masked (``< 0``), so H1 silently trained against
    the CLS embedding. Now the whole row is dropped because there is no valid
    supervision for it.
    """
    from spancall.lean import tensorize as tz

    real_snap = tz._snap_token_range

    def fake_snap(cs, ce, offsets, max_seq_len):
        # Calls arrive in order: utterance, then <|tool|> anchors, then params.
        # Kill every tool anchor (all calls after the first until the
        # param-anchor phase); the simplest robust rule is "kill every
        # non-utterance call", which makes the GT anchor dead by construction.
        res = real_snap(cs, ce, offsets, max_seq_len)
        if fake_snap.n == 0:
            fake_snap.n += 1
            return res
        fake_snap.n += 1
        return None

    fake_snap.n = 0
    monkeypatch.setattr(tz, "_snap_token_range", fake_snap)

    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=1024, max_span_width=16)
    gt = _schema("get_weather", params=1)
    pool = ToolPool([gt, _schema("tool_a"), _schema("tool_b")], seed=3)
    sample = UnifiedSample(
        sample_id="snap-miss",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[gt],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    row = tensorize_sample_lean(
        sample, tok, mb, max_seq_len=1024, pack_multi_tools=True,
        tool_pool=pool, pack_tools=3,
    )
    assert row is None, (
        "a row whose GT tool anchor cannot snap must be dropped; keeping it "
        "with position 0 trains H1 against <|cls|>"
    )


def test_tensorize_encodes_unsnapped_distractor_as_pad_index(monkeypatch) -> None:
    """C2: a dead DISTRACTOR anchor is -1 (masked), never 0 (live CLS slot)."""
    from spancall.lean import tensorize as tz

    real_snap = tz._snap_token_range

    def fake_snap(cs, ce, offsets, max_seq_len):
        # Only the FIRST tool anchor dies; the utterance and the rest are real.
        res = real_snap(cs, ce, offsets, max_seq_len)
        fake_snap.calls += 1
        if fake_snap.calls == 2:  # call 1 = utterance, call 2 = first <|tool|>
            return None
        return res

    fake_snap.calls = 0
    monkeypatch.setattr(tz, "_snap_token_range", fake_snap)

    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=1024, max_span_width=16)
    gt = _schema("get_weather", params=1)
    pool = ToolPool([gt, _schema("tool_a"), _schema("tool_b")], seed=5)
    sample = UnifiedSample(
        sample_id="snap-partial",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[gt],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    stats: dict = {}
    row = tensorize_sample_lean(
        sample, tok, mb, max_seq_len=1024, pack_multi_tools=True,
        tool_pool=pool, pack_tools=3, stats=stats,
    )
    if row is None:
        # The dead slot happened to be the GT: dropping is the correct policy.
        assert stats.get("tool_snap_misses", 0) >= 1
        return
    tp = row["tool_positions"].tolist()
    assert tp[0] == PAD_INDEX, f"dead anchor encoded as {tp[0]}, expected {PAD_INDEX}"
    assert 0 not in [tp[0]], "index 0 is <|cls|> and is never masked"
    # Live count excludes the dead slot, and the GT index still points at a live
    # anchor (otherwise the row would have been dropped).
    assert int(row["_n_tool_candidates"]) == sum(1 for p in tp if p >= 0)
    assert tp[int(row["tool_targets"])] >= 0
    assert stats.get("tool_snap_misses", 0) >= 1


def test_tool_pool_degrades_honestly_when_smaller_than_pack_tools() -> None:
    """C3: a 2-schema pool cannot produce T=5 — it must return T=2, not lie."""
    gt = _schema("get_weather")
    pool = ToolPool([gt, _schema("tool_a")], seed=42)
    assert len(pool) == 2
    sample = UnifiedSample(
        sample_id="small-pool",
        source_dataset="test",
        utterance="weather",
        tools=[gt],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={})],
        stratum=Stratum.F,
    )
    cands, gt_idx = pool.candidates_for(sample, gt, pack_tools=5)
    assert len(cands) == 2, "pool cannot invent schemas it does not have"
    assert cands[gt_idx].name == "get_weather"
    assert len(pool) < 5, "caller is expected to warn when len(pool) < pack_tools"


def test_tool_pool_reports_name_collisions() -> None:
    """S5: duplicate schema names are counted (and differing ones named)."""
    a = _schema("dup", params=1)
    b = _schema("dup", params=3)  # same name, DIFFERENT definition
    c = _schema("dup", params=1)  # same name, identical definition
    pool = ToolPool([a, b, c, _schema("unique")], seed=42)
    assert len(pool) == 2, "names are deduplicated"
    assert pool.n_duplicates == 2
    assert pool.name_collisions == ["dup"], (
        "only differing definitions are reported as collisions; an identical "
        "duplicate is harmless"
    )
    assert len(pool.by_name["dup"].parameters) == 1, "first definition wins"


def test_trainer_mean_n_candidates_ignores_fully_dead_rows() -> None:
    """C1: a row with all-dead anchors must not dilute mean_n_candidates.

    Review probe: 3 live candidates + 1 all-(-1) row reported 1.5 (the average of
    3 and 0) — which also slipped under the ``< 1.5`` vacuity warning. Only rows
    with >= 1 live anchor may enter the denominator.
    """
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    trainer = LeanTrainer(model=model, lr_encoder=1e-3, lr_head=1e-3, device="cpu", use_amp=False)

    B, L, K, C = 2, 15, 1, 4
    batch = {
        "input_ids": torch.randint(0, 100, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "param_positions": torch.tensor([[1], [2]]),
        "cand_starts": torch.randint(0, 5, (B, K, C)),
        "cand_ends": torch.randint(5, 10, (B, K, C)),
        "pos_indices": torch.tensor([[[0]], [[0]]]),
        # Row 0: three live anchors. Row 1: all anchors dead (padded/MISS).
        "tool_positions": torch.tensor([[0, 5, 6], [-1, -1, -1]]),
        "tool_targets": torch.tensor([1, -1]),
        "_lean_converted": True,
    }
    metrics = trainer.train_step(batch)
    assert metrics.get("mean_n_candidates") == 3.0, (
        f"dead row diluted the mean to {metrics.get('mean_n_candidates')} "
        f"(expected 3.0, the mean over live rows only)"
    )
    ev = trainer.evaluate([batch])
    assert ev["mean_n_candidates"] == 3.0
    assert ev["n_tool"] == 1.0

