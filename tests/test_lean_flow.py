"""Unit tests for the Lean flow (spancall/lean)."""

import math
from datetime import date
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
from spancall.infer.base import FakeTokenizer
from spancall.lean.decode import LeanJointDecoder
from spancall.data.align import tooldef_to_template_tool
from spancall.lean.eval_p1 import (
    EXTRACTABLE_TARGET_TYPES,
    _parity_spans_for_call,
    e3_aligner_violation,
    eval_param_extraction_lean,
    param_is_groundable,
)
from spancall.lean.loss import (
    LeanLossContainer,
    compute_lean_pointer_loss,
    compute_lean_tool_loss,
)
from spancall.lean.model import LeanPointerHead, LeanSpanCallModel
from spancall.lean.resolver import (
    LeanResolver,
    _absolutize_parity_spans,
    _iter_parity_entries,
)
from span_template.template import build_input_multi, resolve_utterance_span
from spancall.lean.tensorize import (
    PAD_INDEX,
    LeanBatchCollate,
    ToolPool,
    _snap_token_range,
    jitter_pack_tools,
    tensorize_sample_lean,
)
from spancall.lean.trainer import (
    LeanTrainer,
    _MicroLossAccumulator,
    apply_lora,
    build_lean_optimizer_groups,
)
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


# ---------------------------------------------------------------------------
# R0.4 / J4 — multi-call decoding
# ---------------------------------------------------------------------------


def _j4_cand(value: str, cs: int, ce: int, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        canon_value=value, char_start=cs, char_end=ce, region="utterance", entry_index=0
    )


def test_decode_calls_allows_two_calls_to_share_one_span() -> None:
    """J4: cross-call span reuse is legal; exclusivity is scoped per call."""
    decoder = LeanJointDecoder(overlap_allowed=False, free_string_overlap_allowed=False)

    # Both calls resolve "city" to the SAME utterance span ("Tokyo"). Under
    # single-call exclusivity that is a conflict, but these are two different
    # tool invocations and reusing the span is exactly right.
    shared = _j4_cand("Tokyo", 10, 15, 0.9)
    per_call = {
        0: {"city": [(shared, 0.9)]},
        1: {"city": [(shared, 0.9)]},
    }
    calls = decoder.decode_calls(per_call)
    assert len(calls) == 2
    assert [c.call_index for c in calls] == [0, 1]
    assert calls[0].values == {"city": "Tokyo"}
    assert calls[1].values == {"city": "Tokyo"}


def test_decode_calls_still_prunes_conflicts_inside_one_call() -> None:
    """J4: intra-call exclusivity must survive the multi-call split."""
    decoder = LeanJointDecoder(overlap_allowed=False, free_string_overlap_allowed=False)
    a = _j4_cand("A", 10, 20, 0.9)
    b = _j4_cand("B", 10, 20, 0.8)
    per_call = {0: {"arg1": [(a, 0.9)], "arg2": [(b, 0.8)]}}

    calls = decoder.decode_calls(per_call)
    assert len(calls) == 1
    single_assignment, _, _, _ = decoder.decode_detailed(per_call[0])
    # Same fallback as the single-call path: greedy top-1 keeps both values.
    assert calls[0].values == single_assignment
    assert calls[0].values == {"arg1": "A", "arg2": "B"}


def test_decode_calls_preserves_positional_alignment_for_empty_calls() -> None:
    """J4: a call with no candidates must still occupy its slot."""
    decoder = LeanJointDecoder()
    per_call = {0: {"q": [(_j4_cand("x", 0, 1, 0.5), 0.5)]}, 1: {}}
    calls = decoder.decode_calls(per_call)
    assert len(calls) == 2
    assert calls[1].call_index == 1
    assert calls[1].values == {}

    # A plain sequence is accepted too, and enumeration order defines the index.
    seq_calls = decoder.decode_calls([per_call[0], per_call[1]])
    assert [c.call_index for c in seq_calls] == [0, 1]
    assert seq_calls[0].values == calls[0].values


def test_decode_calls_single_call_matches_decode_detailed() -> None:
    """J4: n_calls == 1 must be byte-identical to today's fast path."""
    decoder = LeanJointDecoder(top_k=3)
    per_param = {
        "city": [(_j4_cand("Tokyo", 10, 15, 0.7), 0.7), (_j4_cand("Osaka", 20, 25, 0.3), 0.3)],
        "unit": [(_j4_cand("celsius", 30, 37, 0.6), 0.6)],
    }
    values, delta, _chosen, n_nodes = decoder.decode_detailed(per_param)
    calls = decoder.decode_calls({0: per_param})
    assert len(calls) == 1
    assert calls[0].values == values
    assert calls[0].delta == delta
    assert calls[0].n_nodes == n_nodes


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


# ---------------------------------------------------------------------------
# R0.4 / J5 — the resolver reports multiple calls
# ---------------------------------------------------------------------------


def _j5_schemas() -> tuple:
    weather = ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters=[ParameterSchema(name="city", description="City", param_type="string", required=True)],
    )
    hotel = ToolDefinition(
        name="book_hotel",
        description="Book a hotel in a city",
        parameters=[ParameterSchema(name="city", description="City", param_type="string", required=True)],
    )
    return weather, hotel


def test_single_call_result_exposes_one_call_mirroring_the_scalars() -> None:
    """J5: the single-call path must be bit-identical apart from the new list."""
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)
    weather, _hotel = _j5_schemas()

    res = resolver.try_resolve(weather, utterance="weather in Tokyo", now="2026-11-09")
    assert len(res.calls) == 1
    call0 = res.calls[0]
    assert call0.call_index == 0
    assert call0.tool_name == res.tool_name
    assert call0.payload == res.payload
    assert call0.status == res.status
    assert call0.hit == res.hit
    assert call0.confidence == res.confidence


def test_call_schemas_decode_one_call_per_oracle_tool() -> None:
    """J5: the oracle multi-call input yields len(call_schemas) calls."""
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)
    weather, hotel = _j5_schemas()

    res = resolver.try_resolve(
        weather,
        utterance="weather in Tokyo and a hotel in Tokyo",
        now="2026-11-09",
        context_schemas=[weather, hotel],
        call_schemas=[weather, hotel],
    )
    assert [c.tool_name for c in res.calls] == ["get_weather", "book_hotel"]
    assert [c.call_index for c in res.calls] == [0, 1]
    # The scalar fields stay pinned to calls[0] for the pre-R0.4 consumers.
    assert res.tool_name == "get_weather"
    assert res.payload == res.calls[0].payload
    assert res.features["n_calls"] == 2

    # Without call_schemas the extra call is NOT fabricated.
    single = resolver.try_resolve(weather, utterance="weather in Tokyo", now="2026-11-09")
    assert len(single.calls) == 1


def test_try_resolve_candidates_respects_max_calls() -> None:
    """J5: max_calls=1 must never split a confident prediction in two."""
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    weather, hotel = _j5_schemas()

    resolver1 = LeanResolver(model=model, max_seq_len=512, max_calls=1)
    res1 = resolver1.try_resolve_candidates([weather, hotel], utterance="weather in Tokyo")
    assert len(res1.calls) == 1
    assert res1.calls[0].tool_name == res1.tool_name

    # Opting in to two calls can only add, never replace, the winner.
    resolver2 = LeanResolver(
        model=model, max_seq_len=512, max_calls=2, call_min_prob=0.0, call_min_prob_ratio=0.0
    )
    res2 = resolver2.try_resolve_candidates([weather, hotel], utterance="weather in Tokyo")
    assert len(res2.calls) == 2
    assert res2.calls[0].tool_name == res1.calls[0].tool_name
    assert res2.tool_name == res2.calls[0].tool_name
    assert res2.features["n_calls"] == 2
    assert len(res2.features["call_probs"]) == 2


def test_lean_call_result_as_dict_is_serialisable() -> None:
    """J5: prediction records must survive a JSON round-trip."""
    import json

    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)
    weather, _hotel = _j5_schemas()

    res = resolver.try_resolve(weather, utterance="weather in Tokyo", now="2026-11-09")
    record = res.calls[0].as_dict()
    restored = json.loads(json.dumps(record))
    assert restored["call_index"] == 0
    assert restored["tool_name"] == res.tool_name
    assert restored["payload"] == (res.payload or {})


def test_lean_trainer_load_rows_filters_empty_abstain(tmp_path) -> None:
    import json

    t2_path = tmp_path / "t2.jsonl"
    # Row 1: Lean Stratum-F row (0-indexed pos_indices [[1, -1]])
    r_f = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "param_positions": [1],
        "cand_starts": [[2, 3]],
        "cand_ends": [[2, 3]],
        "pos_indices": [[1, -1]],
    }
    # Row 2: Lean unaligned row (pos_indices has only -1)
    r_a = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "param_positions": [1],
        "cand_starts": [[2, 3]],
        "cand_ends": [[2, 3]],
        "pos_indices": [[-1, -1]],
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
    # Only the Stratum-F groundable row should be retained
    assert len(loaded) == 1
    assert loaded[0]["pos_indices"] == [[1, -1]]

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


def test_tensorize_fallback_rescue_is_not_recorded_as_a_drop() -> None:
    """T5 invariant: a drop reason means "this anchor ended with NO positive".

    ``drop_why`` is classified BEFORE the GT-argument fallback (it must be, so
    that ``match_missed`` can suppress the null slot). But when that fallback
    then RESCUES the anchor, the param *is* supervised -- recording a drop
    anyway breaks the audit cell's cross-check
    ``n_zero_positives == len(_param_drop_reasons)`` and, worse, inflates the
    "supervision defects (must be 0)" tripwire with params that were never
    defective. The rescue is still worth counting, on its own channel.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    schema = _weather_schema(
        ParameterSchema(name="city", description="City name", param_type="string", required=True),
    )
    sample = UnifiedSample(
        sample_id="s-rescued",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
        # EMPTY target_spans: the matcher has nothing, so classification says
        # `match_missed`, but the GT-argument fallback finds "Tokyo" in the
        # candidate set and supervises it.
        target_spans={},
    )

    stats: dict = {}
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_multi_tools=True, stats=stats)
    assert row is not None
    # The param is supervised -> no drop recorded.
    assert (row["pos_indices"][0] >= 0).any()
    assert stats.get("param_drop_reasons", {}) == {}
    assert "_param_drop_reasons" not in row
    # ... but the primary-path failure is not erased either.
    assert stats.get("param_rescued_by_fallback", {}) == {"match_missed": 1}


def test_tensorize_null_rescue_is_not_recorded_as_a_drop() -> None:
    """Same invariant for the ``<|null|>`` rescue of an optional param."""
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    schema = _weather_schema(
        ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ParameterSchema(name="unit", description="Unit", param_type="string", required=False),
    )
    sample = UnifiedSample(
        sample_id="s-null-rescued",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[schema],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
        target_spans={
            "city": [SpanTarget(param_name="city", target_type="utterance_span", canonical_value="Tokyo")],
            # "unit" has neither a target nor a GT arg -> absent_from_gt, then
            # rescued into the deterministic <|null|> slot because optional.
        },
    )

    stats: dict = {}
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_multi_tools=True, stats=stats)
    assert row is not None
    assert all((p >= 0).any() for p in row["pos_indices"])
    assert stats.get("param_drop_reasons", {}) == {}
    assert stats.get("param_rescued_by_fallback", {}) == {"absent_from_gt": 1}


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


class _RenderTStubResolver(_StubResolver):
    """Stub that ALSO reports the resolver features R9/R2 need surfaced.

    A stub without ``features`` must leave the metrics at ``n/a`` (unknown), NOT
    at a fabricated T=1 -- that distinction is what the assertions below check.
    """

    def __init__(self, scripted: dict, render_T: int = 1,
                 recent_budget_effective: int = 2048) -> None:
        super().__init__(scripted)
        self.render_T = render_T
        self.recent_budget_effective = recent_budget_effective

    def try_resolve(self, tool, utterance, recent_context=None, now=None, **kwargs):
        res = super().try_resolve(tool, utterance, recent_context, now)
        res.features = {
            "render_T": self.render_T,
            "recent_budget_effective": self.recent_budget_effective,
            "recent_budget_tokens": 4096,
            "recent_budget_clamped": True,
        }
        return res


class _ParityStubResolver(_StubResolver):
    """Stub that records the T1/T2 kwargs the eval harness actually passed.

    T2 is a *fix under test*, so the harness must be able to run BOTH arms.
    Until this test existed, ``protect_parity_spans`` was accepted by the
    resolver but never supplied by ``eval_param_extraction_lean`` -- so the
    experiment could not be run through the eval at all, and the switch was
    dead code from the harness's point of view.
    """

    supports_context_schemas = True
    supports_candidate_parity = True

    def __init__(self, scripted: dict) -> None:
        super().__init__(scripted)
        self.seen: list = []

    def try_resolve(self, tool, utterance, recent_context=None, now=None, **kwargs):
        self.seen.append(dict(kwargs))
        return super().try_resolve(tool, utterance, recent_context, now)


def _parity_split(tmp_path) -> str:
    """Write a one-row split whose `city` GT span is extractable."""
    import json
    record = {
        "sample_id": "p1",
        "source_dataset": "test",
        "utterance": "weather in Tokyo",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": [
                    {"name": "city", "description": "City", "param_type": "string", "required": True},
                ],
            }
        ],
        "ground_truth_calls": [
            {"tool_name": "get_weather", "arguments": {"city": "Tokyo"}}
        ],
        "current_time_iso": "2026-11-09",
        "stratum": "F",
        "target_spans": {
            "city": [{
                "param_name": "city",
                "target_type": "utterance_span",
                "canonical_value": "Tokyo",
                "char_start": 11,
                "char_end": 16,
            }],
        },
    }
    with open(tmp_path / "parity_split.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return "parity_split"


def test_eval_part1_runs_both_parity_arms(tmp_path) -> None:
    """T2: the eval harness must be able to run the FIX arm, not just diagnose.

    ``protect_parity_spans`` is the switch that separates "measure the serve-side
    candidate loss" from "force the GT span past truncation and see how much
    extraction headroom remains". If the harness never forwards it, the second
    arm is unreachable and the experiment in the plan cannot be run at all.
    """
    split = _parity_split(tmp_path)

    diag = _ParityStubResolver({"city": "Tokyo"})
    m_diag = eval_param_extraction_lean(
        split, resolver=diag, data_dir=str(tmp_path), limit=10, max_param_gap_report=5,
    )
    assert diag.seen, "resolver was never called"
    assert diag.seen[0].get("parity_spans"), "T1 spans must be measured by default"
    assert not diag.seen[0].get("protect_parity_spans"), "T1 must NOT protect by default"
    assert m_diag["parity_protection"] is False

    fix = _ParityStubResolver({"city": "Tokyo"})
    m_fix = eval_param_extraction_lean(
        split, resolver=fix, data_dir=str(tmp_path), limit=10,
        max_param_gap_report=5, protect_parity_spans=True,
    )
    assert fix.seen, "resolver was never called"
    # Protection implies measurement: the same spans are handed over AND
    # forced past the candidate cap.
    assert fix.seen[0].get("parity_spans"), "protection must still report parity"
    assert fix.seen[0].get("protect_parity_spans") is True
    assert m_fix["parity_protection"] is True


# ---------------------------------------------------------------------------
# P1 — span ADDRESSING: utterance-relative pairs vs stored absolute offsets
# ---------------------------------------------------------------------------


def _p1_schemas() -> tuple:
    """Two tools with a shared ``city`` param — the shape that shifts offsets."""
    weather = ToolDefinition(
        name="get_weather",
        description="Get weather",
        parameters=[
            ParameterSchema(name="city", description="City", param_type="string", required=True),
        ],
    )
    forecast = ToolDefinition(
        name="get_forecast",
        description="Get forecast",
        parameters=[
            ParameterSchema(name="city", description="City", param_type="string", required=True),
        ],
    )
    return weather, forecast


def _p1_resolver(**kw) -> LeanResolver:
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    return LeanResolver(model=model, max_seq_len=512, **kw)


def _p1_render_absolute(schema, utterance: str, schemas=None) -> tuple:
    """The ABSOLUTE offsets the aligner's render would have stored for ``Tokyo``.

    This is the address the OLD ``_parity_spans_for_call`` emitted: correct inside
    the render that produced it, and wrong inside any other render.
    """
    out = build_input_multi(
        [tooldef_to_template_tool(s) for s in (schemas or [schema])],
        utterance,
        recent_context=[],
        now=date(2026, 11, 9),
    )
    span = resolve_utterance_span(out, 11, 16, expect="Tokyo")
    assert span is not None, "fixture assumption broken: Tokyo not found in render"
    return span


def test_parity_rel_pair_hits_when_the_serve_render_shifts_the_utterance() -> None:
    """P1: the rel pair remains correct under the render the resolver SERVES.

    The defect being fixed: ``char_start``/``char_end`` are absolute in whichever
    render the ALIGNER built. The resolver renders afresh at serve time, and
    adding a second tool pushes every utterance offset to the right, so the stored
    absolute pair lands on the wrong characters and parity reads as a candidate
    loss that never happened. The utterance-relative pair is immune because it
    addresses the utterance TEXT, not the rendered prompt.
    """
    weather, forecast = _p1_schemas()
    utterance = "weather in Tokyo"
    resolver = _p1_resolver()

    single = _p1_render_absolute(weather, utterance, schemas=[weather])
    two_tool = _p1_render_absolute(weather, utterance, schemas=[weather, forecast])
    assert single != two_tool, (
        "fixture assumption broken: packing a distractor tool must shift the "
        "utterance, otherwise this test cannot distinguish the two addressings"
    )

    # Legacy absolute pair, scored against the single-tool render it came from.
    ok = resolver.try_resolve(weather, utterance=utterance, parity_spans={"city": [single]})
    assert ok.features["parity_checked"] == 1
    assert ok.features["parity_hit"] == 1
    assert ok.features["parity_rel_fallback"] == 1, "legacy pair must be flagged as a fallback"
    assert ok.features["parity_rel_resolved"] == 0

    # SAME stored pair, but the resolver now serves the two-tool render.
    stale = resolver.try_resolve(
        weather, utterance=utterance, context_schemas=[weather, forecast],
        parity_spans={"city": [single]},
    )
    assert stale.features["parity_hit"] == 0, (
        "a stored absolute pair must MISS once the serve render shifts -- this is "
        "the H0 measurement defect, and it must remain visible, not be papered over"
    )
    assert stale.features["parity_rel_fallback"] == 1

    # The fix: address the utterance, not the prompt.
    fixed = resolver.try_resolve(
        weather, utterance=utterance, context_schemas=[weather, forecast],
        parity_spans={"city": [(11, 16, "Tokyo")]},
    )
    assert fixed.features["parity_checked"] == 1
    assert fixed.features["parity_hit"] == 1
    assert fixed.features["parity_rel_resolved"] == 1
    assert fixed.features["parity_rel_fallback"] == 0
    assert fixed.features["parity_unresolvable"] == 0


def test_parity_absolute_fallback_is_counted_loudly() -> None:
    """P1: a legacy split must be counted and reported, never silently 'clean'.

    Keeping legacy callers working is necessary; letting them look like a clean
    P1 run is not. Every absolute pair bumps ``parity_rel_fallback`` so the
    printed report can say the number is a lower bound.
    """
    weather, _ = _p1_schemas()
    resolver = _p1_resolver()

    # 2-tuples and char_start/char_end dicts are both the legacy contract.
    for spans in ({"city": [(11, 16)]}, {"city": [{"char_start": 11, "char_end": 16}]}):
        res = resolver.try_resolve(weather, utterance="weather in Tokyo", parity_spans=spans)
        assert res.features["parity_checked"] == 1
        assert res.features["parity_rel_fallback"] == 1
        assert res.features["parity_rel_resolved"] == 0
        assert res.features["parity_unresolvable"] == 0

    # Ambiguity guard: arity decides, so a 3-tuple can never be read as absolute.
    assert _iter_parity_entries([(11, 16)]) == [("abs", 11, 16, None)]
    assert _iter_parity_entries([(11, 16, "Tokyo")]) == [("rel", 11, 16, "Tokyo")]


def test_parity_unresolvable_rel_pair_is_counted_as_a_miss() -> None:
    """P1: a rel pair that cannot map onto the served render is a MISS, not an n/a.

    Dropping it would let a drifted label file report a perfect parity rate -- the
    same accounting mistake as trusting an unresolved label, repeated on the eval
    side. The pair is counted in the denominator and contributes no hit.
    """
    weather, forecast = _p1_schemas()
    resolver = _p1_resolver()

    cases = {
        # The mapped slice does not spell the expected value: the render's text
        # drifted under the label, so the address is not trustworthy.
        "wrong expect": [(11, 16, "Osaka")],
        # The label points past the end of any possible render.
        "out of range": [(9000, 9005, None)],
    }
    for label, spans in cases.items():
        res = resolver.try_resolve(
            weather, utterance="weather in Tokyo", context_schemas=[weather, forecast],
            parity_spans={"city": spans},
        )
        feats = res.features
        assert feats["parity_unresolvable"] == 1, f"{label}: must be counted as unresolvable"
        assert feats["parity_checked"] == 1, f"{label}: must stay in the denominator"
        assert feats["parity_hit"] == 0, f"{label}: an unresolvable pair can never hit"
        assert feats["parity_rel_resolved"] == 0

    # Denominator stays MASK-GATED: a GT param absent from the schema has no
    # candidate set, so counting it would inflate the defect with unmeasurable rows.
    res = resolver.try_resolve(
        weather, utterance="weather in Tokyo", parity_spans={"not_a_param": [(11, 16, "Tokyo")]},
    )
    assert res.features["parity_checked"] == 0


def test_eval_p1_emits_rel_pairs_for_modern_targets_and_legacy_pairs_otherwise() -> None:
    """P1 (emit side): the harness must hand over the render-invariant address.

    Emitting the stored absolute pair is what created the defect, so the eval must
    prefer ``utt_rel_*``. It must ALSO keep emitting the absolute pair when that is
    all a legacy split has -- dropping the row instead would hide it entirely.
    """
    tool = ToolDefinition(
        name="get_weather",
        description="Get weather",
        parameters=[
            ParameterSchema(name="city", description="City", param_type="string", required=True),
        ],
    )
    assert tool.name == "get_weather"
    base = {
        "sample_id": "p1_emit",
        "source_dataset": "test",
        "utterance": "weather in Tokyo",
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "parameters": [
                {"name": "city", "description": "City", "param_type": "string", "required": True},
            ],
        }],
        "ground_truth_calls": [{"tool_name": "get_weather", "arguments": {"city": "Tokyo"}}],
        "current_time_iso": "2026-11-09",
        "stratum": "F",
    }

    modern = UnifiedSample.from_dict({
        **base,
        "target_spans": {"city": [{
            "param_name": "city",
            "target_type": "utterance_span",
            "canonical_value": "Tokyo",
            "char_start": 372,
            "char_end": 377,
            "utt_rel_start": 11,
            "utt_rel_end": 16,
        }]},
    })
    spans = _parity_spans_for_call(modern, 0)["city"]
    assert spans == [(11, 16, "Tokyo")], (
        "a target carrying utt_rel_* must be emitted as a rel triple, not as the "
        "stored absolute pair"
    )

    legacy = UnifiedSample.from_dict({
        **base,
        "target_spans": {"city": [{
            "param_name": "city",
            "target_type": "utterance_span",
            "canonical_value": "Tokyo",
            "char_start": 11,
            "char_end": 16,
        }]},
    })
    assert _parity_spans_for_call(legacy, 0)["city"] == [(11, 16)], (
        "a legacy target must still be emitted (as an absolute pair) so the row is "
        "counted rather than silently dropped"
    )


def test_parity_protection_actually_rescues_a_truncated_gt_span() -> None:
    """P1/T2: protection must genuinely move the GT span past ``max_cands``.

    This arm used to be dead: ``_RowCtx`` exposed no ``snap_char_range``, so every
    protected run raised inside ``_resolve``'s blanket handler and reported
    ``render_or_encode_error`` for the whole row. A regression here would make the
    T2 experiment report zero effect for a reason that has nothing to do with the
    pack policy, so it is asserted directly instead of inferred from a metric.

    The fill policy is pinned to ``head`` on BOTH arms on purpose. Under the
    default ``stratified`` the generation-time stride can legitimately retain a
    late span even with no protection at all (that is P2's fix), so the
    unprotected arm would not reliably demonstrate the defect this test is
    about. ``head`` is the legacy prefix, where a late GT span is unreachable
    unless protection materialises it.
    """
    weather, _ = _p1_schemas()
    utterance = "please look up the current conditions in that far away city of Kobe for me"
    kobe = utterance.index("Kobe")
    spans = {"city": [(kobe, kobe + 4, "Kobe")]}

    for cap in (2, 5):
        diag = _p1_resolver(max_cands=cap, candidate_pack_fill="head").try_resolve(
            weather, utterance=utterance, parity_spans=spans, protect_parity_spans=False,
        )
        assert not str(diag.error_reason or "").startswith("render_or_encode_error")
        assert diag.features["parity_checked"] == 1
        assert diag.features["parity_hit"] == 0, (
            f"cap={cap}: a late GT span is expected to fall outside the truncated "
            "head-filled candidate set -- that is the defect T2 fixes"
        )

        fix = _p1_resolver(max_cands=cap, candidate_pack_fill="head").try_resolve(
            weather, utterance=utterance, parity_spans=spans, protect_parity_spans=True,
        )
        # The T2 arm used to die here with a swallowed "'_RowCtx' object has no
        # attribute 'snap_char_range'" and surface as ``render_or_encode_error``.
        # That is the specific regression to catch; a decode-time reason such as
        # ``schema_invalid`` is legitimate (the encoder here is a dummy, so the
        # pointer's argmax is arbitrary) and says nothing about protection.
        assert not str(fix.error_reason or "").startswith("render_or_encode_error")
        assert fix.features["parity_checked"] == 1
        assert fix.features["parity_hit"] == 1, (
            f"cap={cap}: protection must force the GT span into the candidate set"
        )


def test_parity_absolutizer_accepts_both_address_forms() -> None:
    """P1 unit: the normalizer is the single place arity is interpreted."""
    weather, forecast = _p1_schemas()
    out = build_input_multi(
        [tooldef_to_template_tool(weather), tooldef_to_template_tool(forecast)],
        "weather in Tokyo",
        recent_context=[],
        now=date(2026, 11, 9),
    )

    absolutes, unresolvable, counters = _absolutize_parity_spans(
        out,
        {
            "city": [
                (11, 16, "Tokyo"),                                  # rel -> resolved
                {"utt_rel_start": 11, "utt_rel_end": 16},           # dict rel
                {"char_start": 0, "char_end": 3},                   # dict abs
            ],
            "other": [(11, 16, "Nope")],                            # bad expect
        },
    )
    assert counters["rel_resolved"] == 2
    assert counters["rel_fallback"] == 1
    assert counters["rel_unresolvable"] == 1
    assert unresolvable == {"other": 1}
    assert len(absolutes["city"]) == 3
    assert "other" not in absolutes


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


def test_eval_part1_reports_render_T_telemetry(tmp_path) -> None:
    """R9: Eval Part 1 must report the prompt layout it actually rendered.

    The whole point of the check is that a T=1 oracle-tool eval is *visible*
    rather than inferred, so the metric has to be exact -- and it has to be
    ``n/a`` (not a fabricated 1) when the resolver cannot say.
    """
    record = _weather_record(
        {"city": "Tokyo"},
        {"city": [{"param_name": "city", "target_type": "utterance_span",
                   "canonical_value": "Tokyo"}]},
        sample_id="r-t",
    )
    split = _write_split(tmp_path, [record, dict(record, sample_id="r-t2")])

    metrics = eval_param_extraction_lean(
        split,
        resolver=_RenderTStubResolver({"city": "Tokyo"}, render_T=1),
        data_dir=str(tmp_path),
        limit=10,
        max_param_gap_report=5,
    )
    assert metrics["render_T_rows"] == 2
    assert metrics["mean_render_T"] == 1.0
    assert metrics["render_T_hist"] == {"1": 2}
    # 100% T=1 is the exact layout mismatch the metric exists to expose.
    assert metrics["render_T_t1_share"] == 1.0
    # R2: the clamped budget is reported, not silently applied.
    assert metrics["recent_budget_effective"] == {"2048": 2}


def test_eval_part1_render_T_is_na_when_the_resolver_cannot_report(tmp_path) -> None:
    """An old stub has no ``features``: report unknown, never a fabricated T=1."""
    record = _weather_record({"city": "Tokyo"}, {})
    split = _write_split(tmp_path, [record])

    metrics = eval_param_extraction_lean(
        split,
        resolver=_StubResolver({"city": "Tokyo"}),
        data_dir=str(tmp_path),
        limit=10,
        max_param_gap_report=5,
    )
    assert metrics["render_T_rows"] == 0
    assert metrics["mean_render_T"] != metrics["mean_render_T"]  # NaN
    assert metrics["render_T_hist"] == {}
    assert metrics["recent_budget_effective"] == {}


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


def test_tool_pool_never_truncates_native_tools() -> None:
    """R0.1: ``pack_tools`` is a FLOOR. Five native tools must all survive.

    The old implementation stopped at ``pack_tools`` candidates, so a row that
    genuinely offered five tools trained H1 on an arbitrary three of them while
    the eval prompt rendered all five. H1 was then scored on tools it had never
    been allowed to choose between.
    """
    gt = _schema("pick_one", params=1)
    natives = [_schema("get_weather"), _schema("get_time"), _schema("send_email"),
               _schema("search_web"), _schema("book_table")]
    pool = ToolPool([gt, *natives, _schema("distractor_a")], seed=17)
    sample = UnifiedSample(
        sample_id="native-5",
        source_dataset="test",
        utterance="do the thing",
        tools=[gt, *natives],  # SIX tools offered, pack_tools asks for 3
        ground_truth_calls=[GroundTruthCall(tool_name="pick_one", arguments={"p0": "x"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    cands, gt_idx = pool.candidates_for(sample, gt, pack_tools=3)
    names = {c.name for c in cands}
    for t in [gt, *natives]:
        assert t.name in names, (
            f"native tool {t.name!r} was truncated; pack_tools must never cap"
        )
    assert len(cands) == 6, f"expected all 6 offered tools, got {len(cands)}: {sorted(names)}"
    assert cands[gt_idx].name == "pick_one", "GT must survive and be indexable"


def test_tool_pool_max_tools_ceiling_keeps_the_gt_tool() -> None:
    """The hard ceiling must cut distractors, never the answer itself.

    Truncation happens while the GT tool is still at index 0, so a pathological
    schema dump cannot remove the supervised class. The caller is responsible
    for counting what was cut (``tool_truncated_rows``).
    """
    gt = _schema("pick_one", params=1)
    natives = [_schema(f"native_{i}") for i in range(10)]
    pool = ToolPool([gt, *natives], seed=23)
    sample = UnifiedSample(
        sample_id="ceiling",
        source_dataset="test",
        utterance="do the thing",
        tools=[gt, *natives],
        ground_truth_calls=[GroundTruthCall(tool_name="pick_one", arguments={"p0": "x"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    cands, gt_idx = pool.candidates_for(sample, gt, pack_tools=20, max_tools=4)
    assert len(cands) == 4, f"ceiling not applied: {len(cands)} candidates"
    assert 0 <= gt_idx < len(cands), f"GT index {gt_idx} outside {len(cands)} candidates"
    assert cands[gt_idx].name == "pick_one", "the ceiling dropped the supervised tool"


def test_tensorize_counts_truncation_loudly() -> None:
    """A truncated row must be counted, not silently rendered short."""
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=2048, max_span_width=16)
    gt = _schema("pick_one", params=1)
    natives = [_schema(f"native_{i}") for i in range(8)]
    pool = ToolPool([gt, *natives], seed=31)
    sample = UnifiedSample(
        sample_id="truncate-me",
        source_dataset="test",
        utterance="do the thing",
        tools=[gt, *natives],
        ground_truth_calls=[GroundTruthCall(tool_name="pick_one", arguments={"p0": "x"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    stats: dict = {}
    row = tensorize_sample_lean(
        sample, tok, mb, max_seq_len=2048, tool_pool=pool, pack_tools=20,
        max_tools=4, stats=stats, jitter=False,
    )
    assert row is not None
    assert stats.get("tool_truncated_rows", 0) == 1, (
        "truncation happened but was not reported; a row silently rendered "
        f"fewer tools than the user saw. stats={stats}"
    )
    assert int(row["_n_tool_candidates"]) <= 4


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

    ``jitter=False`` because R1c randomizes the floor per row; this test asserts
    the exact packed count, while ``test_jitter_pack_tools_*`` covers the
    randomization itself.
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
        tool_pool=pool, pack_tools=3, jitter=False,
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
        tool_pool=pool, pack_tools=3, jitter=False,
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
        tool_pool=pool, pack_tools=3, stats=stats, jitter=False,
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


def test_tensorize_max_pos_populates_multiple_positive_slots() -> None:
    """R7: a repeated entity mention must yield MULTIPLE positive slots.

    ``compute_lean_pointer_loss`` implements the marginal log-likelihood
    ``logsumexp(all) - logsumexp(positives)``, which only differs from plain
    single-positive CE when more than one positive slot is populated. The
    tensorizer used to match targets by canonical value and ``break`` on the
    first equal candidate, so a row could never carry two positives even with
    ``max_pos=3`` -- the multi-positive path was dead code in every run.

    "move Tokyo to Tokyo" has the same value at two distinct offsets, so both
    candidates must be supervised.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=1024, max_span_width=16)
    gt = _schema("move_item", params=1)  # optional string param -> free spans
    utterance = "move Tokyo to Tokyo"
    sample = UnifiedSample(
        sample_id="multipos",
        source_dataset="test",
        utterance=utterance,
        tools=[gt],
        ground_truth_calls=[
            GroundTruthCall(tool_name="move_item", arguments={"p0": "Tokyo"})
        ],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    row = tensorize_sample_lean(
        sample, tok, mb, max_seq_len=1024, max_pos=3, pack_multi_tools=True,
    )
    assert row is not None, "utterance fits in 1024 tokens; row must not be dropped"

    pos_row = row["pos_indices"][0].tolist()
    assert len(pos_row) == 3, f"max_pos=3 must expose 3 slots, got {len(pos_row)}"
    live_pos = [p for p in pos_row if p >= 0]
    assert len(live_pos) >= 2, (
        f"only {len(live_pos)} positive slot(s) for a twice-repeated entity; "
        "the value-match break is still collapsing every mention into one slot"
    )
    assert len(set(live_pos)) == len(live_pos), f"duplicate positive slots: {pos_row}"

    # The two positives must be genuinely different token spans, not one span
    # counted twice -- otherwise the loss is unchanged and the test is vacuous.
    starts = row["cand_starts"][0].tolist()
    ends = row["cand_ends"][0].tolist()
    spans = [(starts[p], ends[p]) for p in live_pos]
    assert len(set(spans)) == len(spans), f"positives share one span: {spans}"

    # The multi-positive marginal likelihood must be finite with a real gradient.
    # Shapes are (B, K, C) logits against (B, K, P) positives -- K=1 here.
    logits = torch.randn(1, 1, len(starts), requires_grad=True)
    ptr = torch.tensor([[pos_row]], dtype=torch.long)
    loss = compute_lean_pointer_loss(logits, ptr)
    assert torch.isfinite(loss).all(), f"non-finite multi-positive loss: {loss}"
    assert float(loss.detach().sum()) > 0.0, "loss should be positive for random logits"
    loss.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0, "multi-positive loss has no gradient"


def test_compute_lean_pointer_loss_weights_every_positive() -> None:
    """R7: a second positive must change the loss, and the result is a lower bound.

    Adding a valid alternative can only make the marginal likelihood better, so
    ``loss(2 positives) <= loss(1 positive)`` must hold pointwise for the same
    logits. This is the property that makes multi-positive supervision correct
    rather than merely different.
    """
    logits = torch.tensor([[[1.0, 2.0, 0.5, -1.0]]])  # (B=1, K=1, C=4)
    one = torch.tensor([[[1, -1, -1]]], dtype=torch.long)
    two = torch.tensor([[[1, 2, -1]]], dtype=torch.long)
    loss_one = float(compute_lean_pointer_loss(logits, one).sum())
    loss_two = float(compute_lean_pointer_loss(logits, two).sum())
    assert loss_two <= loss_one + 1e-6, (
        f"adding a positive raised the loss ({loss_two} > {loss_one}); the "
        "marginal likelihood is being computed against the wrong set"
    )
    # With BOTH positives at the top-2 logits, the loss must be strictly smaller.
    assert loss_two < loss_one, "second positive had no effect on the loss"

    # A row with no positives contributes exactly zero (never NaN).
    none = torch.tensor([[[-1, -1, -1]]], dtype=torch.long)
    assert float(compute_lean_pointer_loss(logits, none).sum()) == 0.0


def test_tensorize_capped_candidates_have_no_duplicate_spans() -> None:
    """The candidate cap must emit each ``(tok_start, tok_end)`` span exactly once.

    Regression for a cap block that appended every kept index to BOTH an
    ordered ``pool`` list and the ``keep_set`` membership set. Two consequences
    followed:

    1. the early-stop test ``len(pool) + len(keep_set) >= max_cands`` counted
       each index twice, so it fired at roughly half the real budget; and
    2. the final ``sorted(list(keep_set) + pool[...])`` emitted the same span
       twice, so a ``max_cands=512`` row held ~256 distinct candidates padded
       with ~256 repeats.

    The model then ran a softmax over duplicated columns, and the row kept
    about half the distinct candidate coverage the inference resolver sees --
    a silent train/serve gap pointing the wrong way. ``keep_set`` is now the
    single authoritative membership set.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=1024, max_span_width=16)
    # ``_schema`` builds optional string params -> a free-string candidate set
    # (windowed utterance spans) plus the deterministic <|null|> slot.
    gt = _schema("search_docs", params=1)

    # A long, word-dense utterance yields far more raw candidates than the cap,
    # which is what forces the capping branch to run at all.
    words = [f"w{i}x" for i in range(60)]
    utterance = " ".join(words)

    sample = UnifiedSample(
        sample_id="cap-dedup",
        source_dataset="test",
        utterance=utterance,
        tools=[gt],
        ground_truth_calls=[
            GroundTruthCall(tool_name="search_docs", arguments={"p0": words[20]})
        ],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    cap = 48
    row = tensorize_sample_lean(
        sample, tok, mb, max_seq_len=1024, max_cands=cap, pack_multi_tools=True,
    )
    assert row is not None, "utterance fits in 1024 tokens; row must not be dropped"

    starts = row["cand_starts"][0].tolist()
    ends = row["cand_ends"][0].tolist()
    spans = list(zip(starts, ends))
    # Trailing (-1, -1) entries are padding, not candidates.
    live = [s for s in spans if s != (PAD_INDEX, PAD_INDEX)]

    assert len(live) == cap, (
        f"the cap must fill exactly {cap} columns for an over-long candidate "
        f"set, got {len(live)} (raw utterance yields far more)"
    )
    assert len(live) == len(set(live)), (
        f"capped candidate list repeats spans: {len(live)} columns but only "
        f"{len(set(live))} distinct (tok_start, tok_end) pairs"
    )

    # Capping must never evict the supervised span, and every positive column
    # index must still resolve to a real (non-padding) candidate.
    pos = [p for p in row["pos_indices"][0].tolist() if p >= 0]
    assert pos, "capping dropped the positive span -- supervision would be empty"
    for p in pos:
        assert 0 <= p < len(live), f"positive index {p} out of range for {len(live)} candidates"


def test_jitter_pack_tools_is_deterministic_in_seed_and_sample_id() -> None:
    """R1c: the per-row floor must be reproducible across processes and reruns.

    Tensorization runs inside a fork pool, so the same ``(seed, sample_id)`` must
    yield the same anchor count every time; otherwise the same row would train
    against a different T on every epoch for no reason.
    """
    draws = [jitter_pack_tools(5, "sample-abc", seed=42) for _ in range(20)]
    assert len(set(draws)) == 1, f"non-deterministic floor: {set(draws)}"
    assert jitter_pack_tools(5, "sample-abc", seed=42) == jitter_pack_tools(
        5, "sample-abc", seed=42
    )
    # A different seed must be allowed to (and normally will) differ.
    other = {jitter_pack_tools(8, "sample-abc", seed=s) for s in range(1, 30)}
    assert len(other) > 1, "seed has no effect: the floor is not really randomized"


def test_jitter_pack_tools_range_covers_one_through_the_ceiling() -> None:
    """R1c: the draw is inclusive of BOTH ``min_tools`` and ``pack_tools``.

    T=1 must be reachable, otherwise the oracle-tool eval prompt (exactly one
    rendered tool) stays out of distribution and H1 collapses at serve time.
    No draw may exceed the configured ceiling either.
    """
    hi = 5
    values = {jitter_pack_tools(hi, f"s{i}", seed=7) for i in range(400)}
    assert min(values) == 1, f"T=1 unreachable; draws started at {min(values)}"
    assert max(values) == hi, f"ceiling {hi} unreachable; draws topped at {max(values)}"
    assert values <= set(range(1, hi + 1)), f"draw outside [1, {hi}]: {values}"


def test_jitter_pack_tools_honours_explicit_min_tools() -> None:
    """``jitter_min_tools`` raises the lower bound without touching the ceiling."""
    values = {jitter_pack_tools(6, f"s{i}", seed=3, min_tools=4) for i in range(300)}
    assert values <= set(range(4, 7)), f"draw outside [4, 6]: {values}"
    assert 4 in values and 6 in values


def test_jitter_pack_tools_degenerates_when_floor_meets_min() -> None:
    """A degenerate range (hi <= lo) must return the ceiling, not raise or draw lo-1."""
    assert jitter_pack_tools(1, "s", seed=1, min_tools=1) == 1
    assert jitter_pack_tools(3, "s", seed=1, min_tools=3) == 3
    # min_tools above the ceiling: the ceiling wins, nothing above is invented.
    assert jitter_pack_tools(3, "s", seed=1, min_tools=9) == 3


def test_tensorize_jitter_reaches_t1_and_reports_the_draw() -> None:
    """R1c end-to-end: over many rows the tensorizer must render T=1 sometimes.

    Without jitter, every training row renders exactly ``pack_tools`` anchors and
    the single-tool eval prompt is a shape H1 never saw. The ``jittered_rows``
    and ``render_T_hist`` counters make the realized distribution auditable.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=2048, max_span_width=16)
    gt = _schema("get_weather", params=1)
    pool = ToolPool([gt, *[_schema(f"tool_{i}") for i in range(7)]], seed=13)
    stats: dict = {}
    seen_T: set = set()
    for i in range(40):
        sample = UnifiedSample(
            sample_id=f"jit-{i}",
            source_dataset="test",
            utterance="weather in Tokyo",
            tools=[gt],
            ground_truth_calls=[
                GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})
            ],
            current_time_iso="2026-11-09",
            stratum=Stratum.F,
        )
        row = tensorize_sample_lean(
            sample, tok, mb, max_seq_len=2048, tool_pool=pool, pack_tools=4,
            stats=stats,
        )
        if row is not None:
            seen_T.add(int(row["_n_tool_candidates"]))

    assert stats.get("jittered_rows", 0) == 40, "jitter never ran"
    assert 1 in seen_T, (
        f"no row rendered T=1 over 40 jittered draws; realized T set was {sorted(seen_T)}"
    )
    assert max(seen_T) <= 4, f"jitter exceeded the ceiling: {sorted(seen_T)}"
    hist = stats.get("render_T_hist") or {}
    assert sum(int(v) for v in hist.values()) >= 1, "render_T_hist not populated"


def test_tensorize_without_jitter_renders_exactly_the_floor() -> None:
    """``jitter=False`` is the escape hatch that keeps counts reproducible."""
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=2048, max_span_width=16)
    gt = _schema("get_weather", params=1)
    pool = ToolPool([gt, *[_schema(f"tool_{i}") for i in range(7)]], seed=13)
    stats: dict = {}
    counts = set()
    for i in range(10):
        sample = UnifiedSample(
            sample_id=f"nojit-{i}",
            source_dataset="test",
            utterance="weather in Tokyo",
            tools=[gt],
            ground_truth_calls=[
                GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})
            ],
            current_time_iso="2026-11-09",
            stratum=Stratum.F,
        )
        row = tensorize_sample_lean(
            sample, tok, mb, max_seq_len=2048, tool_pool=pool, pack_tools=4,
            stats=stats, jitter=False,
        )
        if row is not None:
            counts.add(int(row["_n_tool_candidates"]))
    assert counts == {4}, f"jitter=False still varied T: {sorted(counts)}"
    assert stats.get("jittered_rows", 0) == 0
    # T12a: no jitter means no draw, so the histogram must stay absent rather
    # than materialise as an empty-but-present key.
    assert not stats.get("jitter_draw_hist")


def test_tensorize_records_the_jitter_draw_histogram_on_kept_rows() -> None:
    """T12a: ``jittered_rows`` counts DRAWS; the histogram counts SURVIVORS.

    A draw that is later dropped (over-length, dead anchor) must not be counted
    as a row the trainer saw, and a jitter collapsed onto one floor must be
    visible as ``len(hist) == 1`` instead of looking like a healthy run.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=2048, max_span_width=16)
    gt = _schema("get_weather", params=1)
    pool = ToolPool([gt, *[_schema(f"tool_{i}") for i in range(7)]], seed=13)
    stats: dict = {}
    kept = 0
    for i in range(40):
        sample = UnifiedSample(
            sample_id=f"jhist-{i}",
            source_dataset="test",
            utterance="weather in Tokyo",
            tools=[gt],
            ground_truth_calls=[
                GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})
            ],
            current_time_iso="2026-11-09",
            stratum=Stratum.F,
        )
        row = tensorize_sample_lean(
            sample, tok, mb, max_seq_len=2048, tool_pool=pool, pack_tools=4,
            stats=stats,
        )
        if row is not None:
            kept += 1

    hist = stats.get("jitter_draw_hist") or {}
    assert hist, "jitter_draw_hist was never populated on the kept-row exit"
    # Every surviving row contributes exactly one draw.
    assert sum(int(v) for v in hist.values()) == kept
    # And the histograms cannot out-count the draws (rows are only ever dropped).
    assert kept <= int(stats.get("jittered_rows", 0))
    # Floors are keyed by the realised value and bounded by the ceiling.
    assert all(1 <= int(k) <= 4 for k in hist)
    # Over 40 draws the sampler should NOT be degenerate (would mean the interval
    # collapsed to pack_tools, indistinguishable from jitter=False in the render
    # report). This is the exact defect T12a exists to expose.
    assert len(hist) > 1, f"jitter degenerate: every draw landed on {list(hist)}"


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


def test_lean_batch_collate_rank2_missing() -> None:
    """Finding 5: LeanBatchCollate must build rank-2 pad for missing rank-2 key."""
    from spancall.lean.tensorize import LeanBatchCollate

    collate = LeanBatchCollate()
    row1 = {
        "input_ids": torch.tensor([1, 2, 3]),
        "cand_starts": torch.tensor([[0, 1], [2, 3]]),  # shape (2, 2)
        "cand_ends": torch.tensor([[1, 2], [3, 4]]),
    }
    row2 = {
        "input_ids": torch.tensor([4, 5]),
        "cand_starts": None,
        "cand_ends": None,
    }
    batch = collate([row1, row2])
    assert batch["cand_starts"].shape == (2, 2, 2)
    assert (batch["cand_starts"][1] == -1).all()


def test_trainer_tensorize_rows_1d_preservation() -> None:
    """Finding 6: trainer.tensorize_rows must not destroy 1-D supervision tensors."""
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    trainer = LeanTrainer(model=model, lr_encoder=1e-3, lr_head=1e-3, device="cpu", use_amp=False)

    row = {
        "input_ids": [1, 2, 3],
        "cand_starts": [0, 1, 2],  # 1-D with K=1
        "pos_indices": [0],
    }
    tensorized = trainer.tensorize_rows([row])[0]
    assert tensorized["cand_starts"].shape == (1, 3)
    assert tensorized["pos_indices"].shape == (1, 1)

    empty_row = {
        "input_ids": [1, 2],
        "cand_starts": [],  # Empty 1-D
        "pos_indices": [],
    }
    empty_tensorized = trainer.tensorize_rows([empty_row])[0]
    assert empty_tensorized["cand_starts"].shape == (0, 0)
    assert empty_tensorized["pos_indices"].shape == (0, 0)


def test_lean_resolver_no_param_anchors_abstain() -> None:
    """Finding 7: resolver must abstain with no_param_anchors when anchors are missing."""
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)

    schema = ToolDefinition(
        name="test_tool",
        description="Tool with param",
        parameters=[
            ParameterSchema(name="param1", description="A parameter", param_type="string", required=True),
        ],
    )
    # Mock _score_parameters returning empty anchors
    orig_score = resolver._score_parameters
    try:
        resolver._score_parameters = lambda *args, **kwargs: ([], {})
        res = resolver.try_resolve(schema, utterance="test utterance")
        assert res.status == "abstain"
        assert res.abstain_reason == "no_param_anchors"
        assert res.error_reason == "no_param_anchors"
        assert "param1" in res.missing_required
    finally:
        resolver._score_parameters = orig_score


def _two_schemas_sharing_param() -> tuple:
    """Two tools that BOTH declare a parameter named ``query``."""
    a = ToolDefinition(
        name="search_docs",
        description="Search documents",
        parameters=[ParameterSchema(name="query", description="q", param_type="string", required=True)],
    )
    b = ToolDefinition(
        name="search_email",
        description="Search email",
        parameters=[ParameterSchema(name="query", description="q", param_type="string", required=True)],
    )
    return a, b


def test_resolver_renders_all_context_schemas_but_scores_one() -> None:
    """R0.2 + R1b: the prompt must show every tool; only the GT tool is decoded.

    Under the oracle-tool setting eval rendered exactly one ``<|tool|>`` block
    while training rendered ``pack_tools`` of them, so H1/H3 met an anchor
    layout at serve time they had never seen. ``context_schemas`` renders the
    real candidate set and reports the count it actually used.
    """
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)
    assert resolver.supports_context_schemas is True, (
        "the capability flag must exist so eval can gate on it instead of "
        "swallowing a TypeError"
    )

    a, b = _two_schemas_sharing_param()
    res = resolver.try_resolve(
        a, utterance="find the doc about Paris", context_schemas=[a, b],
    )
    # Scored tool is unchanged: context_schemas only affects the PROMPT.
    assert res.tool_name == "search_docs"
    assert set(res.parameters) == {"query"}, (
        f"decoded {sorted(res.parameters)}; a distractor's params must not be "
        "scored into this call's payload"
    )
    assert res.features.get("render_T") == 2, (
        f"render_T={res.features.get('render_T')} but 2 tools were rendered; "
        "the T mismatch would stay invisible"
    )


def test_resolver_always_renders_the_scored_tool() -> None:
    """A ``context_schemas`` list that omits the scored tool must not be trusted."""
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)
    a, b = _two_schemas_sharing_param()
    res = resolver.try_resolve(a, utterance="find doc", context_schemas=[b])
    assert res.tool_name == "search_docs"
    # search_docs got spliced back in, so two blocks are rendered.
    assert res.features.get("render_T") == 2


def test_resolver_reports_effective_recent_budget() -> None:
    """R2: the halved budget must be REPORTED, not applied silently.

    ``eff_budget = min(context_budget_tokens, max(64, max_seq_len // 2))`` used
    to halve the configured budget with no trace in the features, so a run that
    dropped recent rows looked identical to one that simply had few rows.
    """
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512, context_budget_tokens=4096)
    a, _ = _two_schemas_sharing_param()
    res = resolver.try_resolve(a, utterance="find doc", recent_context=[])
    f = res.features
    assert f["recent_budget_tokens"] == 4096, f"configured budget lost: {f}"
    assert f["recent_budget_effective"] == 256, (
        f"effective budget should be max(64, 512//2)=256, got {f}"
    )
    assert f["recent_budget_clamped"] is True, "the clamp must be flagged"


def test_candidate_masks_key_shared_param_names_by_tool() -> None:
    """R3: ``tool_a::query`` and ``tool_b::query`` must be distinct entries.

    Keying masks by bare ``param_name`` made the second schema silently
    overwrite the first, so a distractor tool that happened to share a parameter
    name replaced the supervised tool's candidate set.
    """
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)
    a, b = _two_schemas_sharing_param()

    from datetime import date as _date

    from span_template.template import build_input_multi
    from spancall.data.align import tooldef_to_template_tool

    output = build_input_multi(
        [tooldef_to_template_tool(a), tooldef_to_template_tool(b)],
        "find the doc about Paris",
        now=_date(2026, 11, 9),
    )
    masks, _ctx = resolver._candidate_masks(output, output.text, [a, b])
    assert "search_docs::query" in masks, f"missing qualified key; got {sorted(masks)}"
    assert "search_email::query" in masks, f"missing qualified key; got {sorted(masks)}"
    # Both must have produced a usable candidate set, not one plus a clobbered one.
    for k in ("search_docs::query", "search_email::query"):
        assert masks[k]["candidates"], f"{k} produced no candidates"


def test_resolver_latency_is_additive_across_stages() -> None:
    """R5: ``latency_ms`` must include the render/encode/H1 work done by the caller.

    ``_resolve`` starts its own clock, so its ``latency_ms`` only covered
    head/decode/post. ``try_resolve_candidates`` adds its own frame and reports
    any stage gap explicitly instead of letting it vanish.
    """
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)
    a, b = _two_schemas_sharing_param()
    res = resolver.try_resolve_candidates([a, b], utterance="find the doc about Paris")

    assert res.latency_ms is not None and res.latency_ms > 0.0, "no latency recorded"
    stage_sum = res.features["stage_sum_ms"]
    unattributed = res.features["unattributed_ms"]
    assert stage_sum >= 0.0
    assert unattributed >= 0.0, "sum of stages exceeded total; the math is inverted"
    assert stage_sum <= res.latency_ms + 1e-6, (
        f"stage sum {stage_sum} > total {res.latency_ms}; total is not additive"
    )


def test_tensorize_sanitizes_recent_rows_like_inference() -> None:
    """R8: a ``"user city"`` key must survive as ``"user_city"``, not be dropped.

    Training stringified rows verbatim while serving ran ``sanitize_key``, so a
    key with a space rendered at train time and was renamed (or dropped) at
    serve time -- a silent prompt mismatch.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=2048, max_span_width=16)
    gt = _schema("get_weather", params=1)
    sample = UnifiedSample(
        sample_id="recent-sanitize",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[gt],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})],
        recent_context=[{"key": "user city", "value": "Tokyo", "source": "recent"}],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    stats: dict = {}
    row = tensorize_sample_lean(
        sample, tok, mb, max_seq_len=2048, stats=stats, pack_multi_tools=True,
    )
    assert row is not None, "a sanitizable recent row must not drop the sample"
    assert stats.get("n_recent_rows", 0) == 1, (
        f"the recent row was dropped instead of sanitized: {stats}"
    )
    # Inspect the rows that actually reached the template: the key must be
    # sanitized, not dropped and not passed through verbatim.
    from spancall.lean.tensorize import _recent_rows

    rows = _recent_rows(sample, tools=sample.tools, tokenizer=tok)
    assert len(rows) == 1, f"expected 1 surviving recent row, got {rows}"
    assert rows[0].key == "user_city", (
        f"key rendered as {rows[0].key!r}; training must sanitize like serving"
    )
    assert rows[0].value == "Tokyo"


def test_tensorizer_telemetry_counts_every_T() -> None:
    """R9: ``render_T`` must be recorded on every row, including T=1.

    Counting only T>1 would hide the exact train/serve mismatch the metric
    exists to expose.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=2048, max_span_width=16)
    gt = _schema("get_weather", params=1)
    sample = UnifiedSample(
        sample_id="telemetry",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[gt],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    stats: dict = {}
    row = tensorize_sample_lean(
        sample, tok, mb, max_seq_len=2048, stats=stats, pack_multi_tools=True,
    )
    assert row is not None
    assert stats.get("render_T_rows", 0) == 1, f"T not recorded: {stats}"
    assert stats.get("render_T_sum", 0) == int(row["_n_tool_candidates"]), (
        "the recorded T must match the live anchors actually emitted"
    )
    hist = stats.get("render_T_hist") or {}
    assert sum(int(v) for v in hist.values()) == 1, f"histogram not maintained: {hist}"


def test_pack_multi_tools_flag_does_not_change_the_grammar() -> None:
    """R6: ``pack_multi_tools=False`` must be a no-op now.

    It used to select the SINGLE-tool template, which emits no ``<|none|>``
    anchor and a different anchor layout than the inference path -- training on
    a grammar serving never produced.
    """
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=2048, max_span_width=16)
    gt = _schema("get_weather", params=1)
    sample = UnifiedSample(
        sample_id="r6-grammar",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[gt],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"p0": "Tokyo"})],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    )
    kw = dict(max_seq_len=2048, pack_tools=1, jitter=False)
    on = tensorize_sample_lean(sample, tok, mb, pack_multi_tools=True, **kw)
    off = tensorize_sample_lean(sample, tok, mb, pack_multi_tools=False, **kw)
    assert on is not None and off is not None
    assert torch.equal(on["input_ids"], off["input_ids"]), (
        "pack_multi_tools changed the rendered grammar; the flag must be inert"
    )
    assert torch.equal(on["param_positions"], off["param_positions"])
    assert torch.equal(on["tool_positions"], off["tool_positions"])
    # The multi grammar renders a <|tool|> anchor block even for T=1, which is
    # what the single-tool template never emitted. Both flags must produce it.
    assert on["tool_positions"].numel() == 1, (
        "the unified grammar must emit one <|tool|> anchor for a single tool"
    )
    assert torch.equal(on["tool_targets"], torch.tensor(0))


def test_lean_resolver_candidates_multi_template_and_bundle() -> None:
    """Finding 1: try_resolve_candidates reuses multi-tool encoding and try_resolve uses multi template."""
    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    resolver = LeanResolver(model=model, max_seq_len=512)

    schema1 = ToolDefinition(
        name="tool1",
        description="Tool 1",
        parameters=[ParameterSchema(name="city", description="City name", param_type="string", required=True)],
    )
    schema2 = ToolDefinition(
        name="tool2",
        description="Tool 2",
        parameters=[ParameterSchema(name="item", description="Item name", param_type="string", required=True)],
    )

    # try_resolve on single tool renders with build_input_multi
    res1 = resolver.try_resolve(schema1, utterance="test city Paris")
    assert res1.tool_name == "tool1"

    # try_resolve_candidates resolves in single pass
    res_multi = resolver.try_resolve_candidates([schema1, schema2], utterance="test city Paris")
    assert res_multi.tool_name in ("tool1", "tool2")
    assert "winner_prob" in res_multi.features


# ---------------------------------------------------------------------------
# R0.4 / J2 — the tensorizer emits a call axis
# ---------------------------------------------------------------------------


def _j2_tool(name: str, *params: ParameterSchema) -> ToolDefinition:
    return ToolDefinition(name=name, description=f"{name} tool", parameters=list(params))


def _j2_param(name: str, required: bool = True) -> ParameterSchema:
    return ParameterSchema(name=name, description=name, param_type="string", required=required)


def test_single_call_row_emits_the_legacy_call_axis() -> None:
    """J2: a one-call row must keep every old shape, with the call axis at 0."""
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    tool = _j2_tool("get_weather", _j2_param("city"), _j2_param("unit", required=False))
    sample = UnifiedSample(
        sample_id="j2-single",
        source_dataset="test",
        utterance="weather in Tokyo",
        tools=[tool],
        ground_truth_calls=[GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo"})],
        stratum=Stratum.F,
    )
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512)
    assert row is not None
    # tool_targets stays scalar: F.cross_entropy(tool_logits (B,T), targets (B,)).
    assert row["tool_targets"].dim() == 0
    assert row["call_tool_targets"].shape == (1,)
    assert row["call_param_index"].tolist() == [0, 0]
    assert int(row["n_calls"]) == 1
    assert row["pos_indices"].shape[0] == 2
    assert "_anchor_calls" not in row


def test_two_calls_keep_separate_param_targets_on_a_shared_anchor() -> None:
    """J2: two calls to one tool share the anchor but not the demanded value."""
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    tool = _j2_tool("search", _j2_param("query"))
    sample = UnifiedSample(
        sample_id="j2-two",
        source_dataset="test",
        utterance="search Tokyo and Osaka",
        tools=[tool],
        ground_truth_calls=[
            GroundTruthCall(tool_name="search", arguments={"query": "Tokyo"}),
            GroundTruthCall(tool_name="search", arguments={"query": "Osaka"}),
        ],
        stratum=Stratum.F,
    )
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512)
    assert row is not None
    assert int(row["n_calls"]) == 2
    assert row["call_tool_targets"].shape == (2,)
    # ONE anchor: the two calls share the same (tool, param) question vector.
    assert row["param_positions"].shape == (1,)
    # Both demanded values are valid, so the anchor carries positives from each.
    # The raw count exceeds 2 because one value can surface as several candidate
    # spans; the contract is "at least one positive per demanded value", and the
    # conflict is tagged rather than resolved by silently keeping call 0.
    positives = [p for p in row["pos_indices"][0].tolist() if p >= 0]
    assert len(positives) >= 2
    assert len(set(positives)) == len(positives)  # no duplicated columns
    assert row["_anchor_calls"] == [[0, 1]]
    assert row["_ambiguous_anchor"] == 1


def test_two_calls_to_different_tools_get_one_anchor_each() -> None:
    """J2: anchors are deduplicated by (tool, param), not by param alone."""
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    weather = _j2_tool("get_weather", _j2_param("city"))
    hotel = _j2_tool("book_hotel", _j2_param("city"), _j2_param("nights", required=False))
    sample = UnifiedSample(
        sample_id="j2-two-tools",
        source_dataset="test",
        utterance="weather in Tokyo and book a hotel in Tokyo",
        tools=[weather, hotel],
        ground_truth_calls=[
            GroundTruthCall(tool_name="get_weather", arguments={"city": "Tokyo"}),
            GroundTruthCall(tool_name="book_hotel", arguments={"city": "Tokyo"}),
        ],
        stratum=Stratum.F,
    )
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512, pack_tools=2)
    assert row is not None
    assert int(row["n_calls"]) == 2
    # city@get_weather, city@book_hotel, nights@book_hotel
    assert row["param_positions"].shape == (3,)
    assert sorted(row["call_tool_targets"].tolist()) == sorted(
        {row["tool_targets"].item(), row["call_tool_targets"][1].item()}
    )
    # No anchor is shared, so nothing is ambiguous.
    assert "_ambiguous_anchor" not in row


def test_call_axis_pads_with_pad_index_across_a_batch() -> None:
    """J2: LeanBatchCollate must pad call_tool_targets with -1, not 0."""
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    tool = _j2_tool("search", _j2_param("query"))
    one = UnifiedSample(
        sample_id="j2-b1",
        source_dataset="test",
        utterance="search Tokyo",
        tools=[tool],
        ground_truth_calls=[GroundTruthCall(tool_name="search", arguments={"query": "Tokyo"})],
        stratum=Stratum.F,
    )
    two = UnifiedSample(
        sample_id="j2-b2",
        source_dataset="test",
        utterance="search Tokyo and Osaka",
        tools=[tool],
        ground_truth_calls=[
            GroundTruthCall(tool_name="search", arguments={"query": "Tokyo"}),
            GroundTruthCall(tool_name="search", arguments={"query": "Osaka"}),
        ],
        stratum=Stratum.F,
    )
    rows = [
        tensorize_sample_lean(one, tok, mb, max_seq_len=512),
        tensorize_sample_lean(two, tok, mb, max_seq_len=512),
    ]
    assert all(r is not None for r in rows)
    batch = LeanBatchCollate()([r for r in rows if r is not None])
    assert batch["call_tool_targets"].shape == (2, 2)
    assert batch["n_calls"].tolist() == [1, 2]
    # Row 0 has one call; the second slot is PAD_INDEX, never a fake "tool 0".
    assert batch["call_tool_targets"][0, 1].item() == PAD_INDEX
    # tool_targets keeps its scalar shape through the collator.
    assert batch["tool_targets"].shape == (2,)


# ---------------------------------------------------------------------------
# R0.4 / J3 — set-valued H1 loss
# ---------------------------------------------------------------------------


def test_compute_lean_tool_loss_p1_matches_cross_entropy() -> None:
    """J3: the (B,) path must be bit-identical to plain cross entropy."""
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, -1, 2, -1])

    got = compute_lean_tool_loss(logits, targets)
    want = torch.nn.functional.cross_entropy(logits, targets, ignore_index=PAD_INDEX)
    assert got is not None
    # torch.equal, not torch.allclose: the (B,) branch delegates verbatim, so a
    # bit-identical result is the contract, not merely "close enough".
    assert torch.equal(got, want)


def test_compute_lean_tool_loss_accepts_a_set_of_tools() -> None:
    """J3: any member of the GT set earns credit, so more GT tools => lower loss."""
    torch.manual_seed(1)
    logits = torch.randn(2, 4, requires_grad=True)
    # Row 0: two correct tools (0, 2). Row 1: padded out entirely.
    targets = torch.tensor([[0, 2], [PAD_INDEX, PAD_INDEX]])

    loss_set = compute_lean_tool_loss(logits, targets)
    assert loss_set is not None
    assert torch.isfinite(loss_set)

    # The same row with only tool 0 as correct is a strict subset of the set, so
    # the marginal NLL must be >= the set-valued NLL: adding a correct answer
    # can never make the row look worse.
    subset = torch.tensor([[0, PAD_INDEX], [PAD_INDEX, PAD_INDEX]])
    loss_subset = compute_lean_tool_loss(logits, subset)
    assert loss_subset is not None
    assert float(loss_subset.detach()) >= float(loss_set.detach())

    # Row 1 carries no supervision, so it must not be averaged into the mean:
    # a single-row batch of row 0 alone reports exactly the same loss.
    only_row0 = compute_lean_tool_loss(logits[:1], targets[:1])
    assert only_row0 is not None
    assert torch.allclose(only_row0, loss_set)

    # The set-valued objective is differentiable, and the unsupervised row must
    # contribute zero gradient (otherwise padded rows would drag tool 0 up).
    assert loss_set.requires_grad
    loss_set.backward()
    assert logits.grad is not None
    assert float(logits.grad[1].abs().sum()) == 0.0
    # Both GT tools of row 0 carry gradient: neither is treated as a negative.
    assert float(logits.grad[0, 0]) != 0.0
    assert float(logits.grad[0, 2]) != 0.0


def test_compute_lean_tool_loss_returns_none_without_supervision() -> None:
    """J3: no supervision must skip the term, not report a perfect zero."""
    logits = torch.randn(2, 3)
    assert compute_lean_tool_loss(
        logits, torch.tensor([[PAD_INDEX, PAD_INDEX], [PAD_INDEX, PAD_INDEX]])
    ) is None
    assert compute_lean_tool_loss(logits, torch.tensor([PAD_INDEX, PAD_INDEX])) is None
    assert compute_lean_tool_loss(logits, torch.empty((2, 0), dtype=torch.long)) is None


def test_lean_loss_container_prefers_call_tool_targets() -> None:
    """J3: the call axis wins over the legacy scalar when both are supplied."""
    torch.manual_seed(2)
    logits = torch.randn(2, 4)
    call_targets = torch.tensor([[0, 1], [2, PAD_INDEX]])
    # A deliberately WRONG scalar target: if it were used the loss would differ.
    scalar_targets = torch.tensor([3, 3])

    container = LeanLossContainer(lambda_pointer=0.0, lambda_tool=1.0)
    out = container(
        {"tool_logits": logits},
        {"call_tool_targets": call_targets, "tool_targets": scalar_targets},
    )
    expected = compute_lean_tool_loss(logits, call_targets)
    assert expected is not None
    assert torch.allclose(out["tool"], expected.detach())
    assert torch.allclose(out["loss"], expected.detach())

    # Without the call axis the container must fall back to the scalar path.
    out_legacy = container({"tool_logits": logits}, {"tool_targets": scalar_targets})
    legacy = compute_lean_tool_loss(logits, scalar_targets)
    assert legacy is not None
    assert torch.allclose(out_legacy["tool"], legacy.detach())


def test_train_step_forwards_the_call_axis_to_the_loss() -> None:
    """J3: a tensorized multi-call batch must train through the set-valued loss."""
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=512, max_span_width=16)
    tool = _j2_tool("search", _j2_param("query"))
    sample = UnifiedSample(
        sample_id="j3-two-calls",
        source_dataset="test",
        utterance="search Tokyo and Osaka",
        tools=[tool],
        ground_truth_calls=[
            GroundTruthCall(tool_name="search", arguments={"query": "Tokyo"}),
            GroundTruthCall(tool_name="search", arguments={"query": "Osaka"}),
        ],
        stratum=Stratum.F,
    )
    row = tensorize_sample_lean(sample, tok, mb, max_seq_len=512)
    assert row is not None
    batch = LeanBatchCollate()([row])
    assert batch["call_tool_targets"].shape == (1, 2)

    encoder = TinyDummyEncoder(hidden_dim=32)
    model = LeanSpanCallModel(encoder=encoder, head_dim=16, hidden_dim=32)
    trainer = LeanTrainer(model=model, lr_encoder=1e-3, lr_head=1e-3, device="cpu", use_amp=False)
    metrics = trainer.train_step(batch)
    assert "tool_loss" in metrics
    assert not torch.isnan(torch.tensor(metrics["tool_loss"]))
    # ``tool_acc`` still reads the scalar primary-call target, so it must survive.
    assert "tool_acc" in metrics


# --------------------------------------------------------------------------- #
# R0.4 / J6 — multi-call eval metrics
# --------------------------------------------------------------------------- #

def _j6_two_call_record(extra_second_tool: bool = False) -> dict:
    """One sample with TWO GT calls to different tools, sharing a param name."""
    tools = [
        {
            "name": "get_weather",
            "description": "Weather",
            "parameters": [
                {"name": "city", "description": "City", "param_type": "string", "required": True},
            ],
        },
        {
            "name": "book_hotel",
            "description": "Hotel",
            "parameters": [
                {"name": "city", "description": "City", "param_type": "string", "required": True},
            ],
        },
    ]
    if extra_second_tool:
        tools.append({
            "name": "get_time",
            "description": "Clock",
            "parameters": [
                {"name": "city", "description": "City", "param_type": "string", "required": True},
            ],
        })
    return {
        "sample_id": "j6-two-calls",
        "source_dataset": "test",
        "utterance": "weather in Tokyo and a hotel in Osaka",
        "tools": tools,
        "ground_truth_calls": [
            {"tool_name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"tool_name": "book_hotel", "arguments": {"city": "Osaka"}},
        ],
        "current_time_iso": "2026-11-09",
        "stratum": "F",
        "target_spans": {
            # Call-aware keys: the SAME param name must not collide across calls.
            "0::city": [{"param_name": "city", "target_type": "utterance_span",
                         "canonical_value": "Tokyo"}],
            "1::city": [{"param_name": "city", "target_type": "utterance_span",
                         "canonical_value": "Osaka"}],
        },
    }


class _J6MultiCallResolver:
    """Resolver stub returning one call per requested tool (oracle behaviour)."""

    supports_context_schemas = True

    def __init__(self, values_by_tool: dict, extra_calls: int = 0) -> None:
        self.values_by_tool = values_by_tool
        self.extra_calls = extra_calls
        self.tools_seen: list = []

    def try_resolve(self, tool, utterance, recent_context=None, now=None, **kwargs):
        name = getattr(tool, "name", "")
        self.tools_seen.append(name)
        value = self.values_by_tool.get(name)
        params = (
            {"city": _StubResolution(value=value, confidence=0.9, source_type="utterance")}
            if value is not None else {}
        )
        res = _StubResolveResult(
            parameters=params,
            payload_partial=({k: v.value for k, v in params.items()}),
            hit=bool(params),
            tool_name=name,
        )
        calls = [_StubResolveResult(
            tool_name=name,
            payload=({k: v.value for k, v in params.items()}),
            hit=bool(params),
        )]
        # Simulate the decoder emitting extra calls (a hallucinated over-call).
        for i in range(self.extra_calls):
            calls.append(_StubResolveResult(tool_name=f"extra_{i}", payload={}, hit=False))
        res.calls = calls
        return res


def test_eval_part1_scores_every_gt_call(tmp_path) -> None:
    """J6: a 2-call sample must be scored on BOTH calls, not just ``[0]``.

    Before the fix only ``ground_truth_calls[0]`` was resolved and scored, so the
    13.9% multi-call rows silently reported at most half of their params, and the
    second call's parameters never entered recall/precision at all.
    """
    split = _write_split(tmp_path, [_j6_two_call_record()])
    resolver = _J6MultiCallResolver({"get_weather": "Tokyo", "book_hotel": "Osaka"})

    metrics = eval_param_extraction_lean(
        split, resolver=resolver, data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )

    # Both GT tools were resolved -> both calls were actually evaluated.
    assert resolver.tools_seen == ["get_weather", "book_hotel"]
    # Both calls' params are extractable and both retrieved correctly.
    assert metrics["param_counts"]["total_gt_params"] == 2
    assert metrics["param_counts"]["groundable_gt_params"] == 2
    assert metrics["param_counts"]["extractable_tp"] == 2
    assert metrics["extraction_recall"] == 1.0
    assert metrics["extraction_precision"] == 1.0
    # Set-level exactness: both GT calls match, predicted count agrees.
    assert metrics["call_exact_set_match"] == 1.0
    assert metrics["call_count_acc"] == 1.0


def test_eval_part1_reports_n_calls_histograms(tmp_path) -> None:
    """J6: the call-count distribution of GT and prediction must be reported."""
    split = _write_split(tmp_path, [
        _j6_two_call_record(),
        _weather_record(
            {"city": "Tokyo"},
            {"city": [{"param_name": "city", "target_type": "utterance_span",
                       "canonical_value": "Tokyo"}]},
            sample_id="single",
        ),
    ])
    resolver = _J6MultiCallResolver({"get_weather": "Tokyo", "book_hotel": "Osaka"})

    metrics = eval_param_extraction_lean(
        split, resolver=resolver, data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )

    assert metrics["n_calls_hist"] == {"1": 1, "2": 1}
    assert metrics["n_multi_call_samples"] == 1
    # The single-call row predicts 1 call; the 2-call row predicts 2.
    assert metrics["pred_n_calls_hist"] == {"1": 1, "2": 1}
    assert metrics["call_count_acc"] == 1.0


def test_eval_part1_call_exact_set_match_rejects_an_extra_call(tmp_path) -> None:
    """J6: a hallucinated extra call must break the SET match but not call-0."""
    split = _write_split(tmp_path, [_j6_two_call_record()])
    resolver = _J6MultiCallResolver(
        {"get_weather": "Tokyo", "book_hotel": "Osaka"}, extra_calls=1
    )

    metrics = eval_param_extraction_lean(
        split, resolver=resolver, data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )

    # Every param is still right, so the call-0 headline is unaffected...
    assert metrics["extraction_recall"] == 1.0
    assert metrics["exact_payload_match_F"] == 1.0
    # ...but the model emitted a third call, so the set-level match fails.
    assert metrics["call_exact_set_match"] == 0.0
    assert metrics["call_count_acc"] == 0.0
    assert metrics["pred_n_calls_hist"] == {"3": 1}


def test_eval_part1_multi_call_per_param_rows_are_call_tagged(tmp_path) -> None:
    """J6: two calls sharing a param name must not be summed into one row."""
    split = _write_split(tmp_path, [_j6_two_call_record()])
    resolver = _J6MultiCallResolver({"get_weather": "Tokyo", "book_hotel": "WRONG"})

    metrics = eval_param_extraction_lean(
        split, resolver=resolver, data_dir=str(tmp_path), limit=10, max_param_gap_report=20
    )

    # `city` on call 0 is right, on call 1 is wrong: 1 tp + 1 fp, never 2 tp.
    assert metrics["param_counts"]["extractable_tp"] == 1
    assert metrics["param_counts"]["extractable_fp"] == 1
    assert metrics["call_exact_set_match"] == 0.0
    # The count is still correct (2 GT calls, 2 predicted).
    assert metrics["call_count_acc"] == 1.0


def _j6_same_tool_two_call_record() -> dict:
    """One sample with TWO GT calls to the SAME tool (``get_weather`` twice).

    The call set is a MULTISET: ``[get_weather(Tokyo), get_weather(Osaka)]`` is
    two calls, not one. Part 1 used to dedupe predicted calls by tool name, so a
    perfectly predicted same-tool pair collapsed to a single call and scored
    ``call_count_acc=0`` / ``call_exact_set_match=0``.
    """
    return {
        "sample_id": "j6-same-tool-twice",
        "source_dataset": "test",
        "utterance": "weather in Tokyo and also in Osaka",
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather",
                "parameters": [
                    {"name": "city", "description": "City", "param_type": "string",
                     "required": True},
                ],
            },
        ],
        "ground_truth_calls": [
            {"tool_name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"tool_name": "get_weather", "arguments": {"city": "Osaka"}},
        ],
        "current_time_iso": "2026-11-09",
        "stratum": "F",
        "target_spans": {
            "0::city": [{"param_name": "city", "target_type": "utterance_span",
                         "canonical_value": "Tokyo"}],
            "1::city": [{"param_name": "city", "target_type": "utterance_span",
                         "canonical_value": "Osaka"}],
        },
    }


class _J6SameToolResolver:
    """Resolver stub returning ONE call per resolution, in scripted order.

    Both GT calls request ``get_weather``, so the value cannot be keyed by tool
    name without collapsing the multiset — the values are consumed in call order.
    """

    supports_context_schemas = True

    def __init__(self, values: list) -> None:
        self.values = list(values)
        self.n_resolved = 0

    def try_resolve(self, tool, utterance, recent_context=None, now=None, **kwargs):
        idx = min(self.n_resolved, len(self.values) - 1)
        self.n_resolved += 1
        value = self.values[idx]
        name = getattr(tool, "name", "")
        params = {"city": _StubResolution(value=value, confidence=0.9,
                                          source_type="utterance")}
        payload = {"city": value}
        res = _StubResolveResult(
            parameters=params,
            payload_partial=dict(payload),
            hit=True,
            tool_name=name,
        )
        # One decoded call per resolution — the resolver's ``calls`` mirror.
        res.calls = [_StubResolveResult(tool_name=name, payload=dict(payload), hit=True)]
        return res


def test_eval_part1_counts_two_calls_to_the_same_tool(tmp_path) -> None:
    """J6/I1: same-tool repeated calls must count as TWO calls, not one.

    Regression for the tool-name dedupe in ``eval_param_extraction_lean``: a
    correct ``[get_weather(Tokyo), get_weather(Osaka)]`` prediction used to be
    collapsed to 1 call, so ``call_count_acc`` and ``call_exact_set_match`` read
    0.0 while every parameter was right.
    """
    split = _write_split(tmp_path, [_j6_same_tool_two_call_record()])
    resolver = _J6SameToolResolver(["Tokyo", "Osaka"])

    metrics = eval_param_extraction_lean(
        split, resolver=resolver, data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )

    # Both GT calls were resolved against the same tool.
    assert resolver.n_resolved == 2
    # Both calls' params are right.
    assert metrics["param_counts"]["extractable_tp"] == 2
    assert metrics["extraction_recall"] == 1.0
    # The multiset counts as two: the headline must not collapse it to one.
    assert metrics["n_calls_hist"] == {"2": 1}
    assert metrics["pred_n_calls_hist"] == {"2": 1}
    assert metrics["call_count_acc"] == 1.0
    assert metrics["call_exact_set_match"] == 1.0


class _J6SameToolIdenticalResolver:
    """Returns the SAME single call twice — the hardest multiset case.

    Two GT calls to ``get_weather({"city": "Tokyo"})`` are indistinguishable by
    content, so any dedupe keyed on (tool, payload) wrongly collapses them to
    one call. Only call multiplicity — one call per resolution — is correct.
    """

    supports_context_schemas = True

    def __init__(self, value: str) -> None:
        self.value = value
        self.n_resolved = 0

    def try_resolve(self, tool, utterance, recent_context=None, now=None, **kwargs):
        self.n_resolved += 1
        name = getattr(tool, "name", "")
        params = {"city": _StubResolution(value=self.value, confidence=0.9,
                                          source_type="utterance")}
        payload = {"city": self.value}
        res = _StubResolveResult(
            parameters=params,
            payload_partial=dict(payload),
            hit=True,
            tool_name=name,
        )
        res.calls = [_StubResolveResult(tool_name=name, payload=dict(payload), hit=True)]
        return res


def test_eval_part1_counts_two_identical_same_tool_calls(tmp_path) -> None:
    """Regression: identical arguments must not defeat the multiset count.

    A content-keyed dedupe (tool + payload) still collapsed this case, because
    two identical GT calls produce two identical predictions. The count has to
    follow call multiplicity, not content.
    """
    rec = _j6_same_tool_two_call_record()
    rec["ground_truth_calls"] = [
        {"tool_name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"tool_name": "get_weather", "arguments": {"city": "Tokyo"}},
    ]
    rec["target_spans"] = {
        "0::city": [{"param_name": "city", "target_type": "utterance_span",
                     "canonical_value": "Tokyo"}],
        "1::city": [{"param_name": "city", "target_type": "utterance_span",
                     "canonical_value": "Tokyo"}],
    }
    split = _write_split(tmp_path, [rec])
    resolver = _J6SameToolIdenticalResolver("Tokyo")

    metrics = eval_param_extraction_lean(
        split, resolver=resolver, data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )

    assert resolver.n_resolved == 2
    assert metrics["pred_n_calls_hist"] == {"2": 1}
    assert metrics["call_count_acc"] == 1.0
    assert metrics["call_exact_set_match"] == 1.0


class _J6WholeSetResolver:
    """Decodes the WHOLE call set on every ``try_resolve`` (call_schemas path).

    Each resolution repeats the identical 2-call list, so a naive concatenation
    would report 4 predicted calls for a 2-call sample.
    """

    supports_context_schemas = True

    def __init__(self, pairs) -> None:
        self.pairs = list(pairs)
        self.n_resolved = 0

    def try_resolve(self, tool, utterance, recent_context=None, now=None, **kwargs):
        self.n_resolved += 1
        name = getattr(tool, "name", "")
        value = dict(self.pairs).get(name, "")
        params = {"city": _StubResolution(value=value, confidence=0.9,
                                          source_type="utterance")}
        res = _StubResolveResult(
            parameters=params,
            payload_partial={"city": value},
            hit=bool(value),
            tool_name=name,
        )
        res.calls = [
            _StubResolveResult(tool_name=t, payload={"city": v}, hit=True)
            for t, v in self.pairs
        ]
        return res


def test_eval_part1_does_not_double_count_a_repeated_whole_set(tmp_path) -> None:
    """The whole-set decode shape must be read ONCE, not once per resolution."""
    split = _write_split(tmp_path, [_j6_two_call_record()])
    resolver = _J6WholeSetResolver([("get_weather", "Tokyo"), ("book_hotel", "Osaka")])

    metrics = eval_param_extraction_lean(
        split, resolver=resolver, data_dir=str(tmp_path), limit=10, max_param_gap_report=5
    )

    assert resolver.n_resolved == 2          # both GT tools were resolved
    assert metrics["pred_n_calls_hist"] == {"2": 1}   # not 4
    assert metrics["call_count_acc"] == 1.0
    assert metrics["call_exact_set_match"] == 1.0


# ---------------------------------------------------------------------------
# T12b: the coverage/recall universes must not be conflated
# ---------------------------------------------------------------------------


def _mixed_extractability_split(tmp_path, n: int = 2):
    """Two params per row: ``city`` extractable, ``advice`` unextractable."""
    records = []
    for i in range(n):
        records.append({
            "sample_id": f"u{i}",
            "source_dataset": "test",
            "utterance": "weather in Tokyo",
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "parameters": [
                    {"name": "city", "description": "City", "param_type": "string",
                     "required": True},
                    {"name": "advice", "description": "Advice", "param_type": "string",
                     "required": True},
                ],
            }],
            "ground_truth_calls": [{
                "tool_name": "get_weather",
                "arguments": {"city": "Tokyo", "advice": "umbrella"},
            }],
            "current_time_iso": "2026-11-09",
            "stratum": "F",
            "target_spans": {
                "city": [{"param_name": "city", "target_type": "utterance_span",
                          "canonical_value": "Tokyo"}],
                "advice": [{"param_name": "advice", "target_type": "unresolved"}],
            },
        })
    return _write_split(tmp_path, records)


def test_eval_coverage_and_recall_use_documented_denominators(tmp_path) -> None:
    """``param_coverage`` / ALL gt params, ``extraction_recall`` / extractable only.

    The two denominators are different on purpose. Reporting BOTH counts is what
    makes that auditable instead of a reader having to infer it from source.
    """
    split = _mixed_extractability_split(tmp_path, n=2)
    metrics = eval_param_extraction_lean(
        split, resolver=_StubResolver({"city": "Tokyo"}), data_dir=str(tmp_path),
        limit=10, max_param_gap_report=5,
    )

    assert metrics["fast_path_total_gt_params"] == 4        # 2 rows x 2 params
    assert metrics["fast_path_extractable_gt_params"] == 2  # ``city`` only
    assert metrics["fast_path_unextractable_gt_params"] == 2
    # Partition is exhaustive, so a reader can never get a share > 1.
    assert (metrics["fast_path_extractable_gt_params"]
            + metrics["fast_path_unextractable_gt_params"]
            == metrics["fast_path_total_gt_params"])
    assert metrics["fast_path_param_coverage"] == 0.5       # 2 / 4
    # ``advice`` is refused, which is CORRECT, and must not dent the headline.
    assert metrics["extraction_recall"] == 1.0
    # The all-GT rate: 2 correct retrievals out of 4 GT params.
    assert metrics["fast_path_param_accuracy_all"] == 0.5


def test_eval_strict_recall_never_exceeds_the_headline(tmp_path) -> None:
    """T12b: a WRONG value must lower the strict rate, never the headline.

    ``extraction_recall`` is ``tp / (tp + fn)``: a param answered incorrectly goes
    to ``ex_fp`` and disappears from that denominator, so the headline can read
    1.0 while half the answers are wrong. The strict rate counts wrong answers.
    """
    split = _mixed_extractability_split(tmp_path, n=2)
    # Predicts a WRONG city for both rows: tp=0, wrong=2, fn=0.
    metrics = eval_param_extraction_lean(
        split, resolver=_StubResolver({"city": "Nowhere"}), data_dir=str(tmp_path),
        limit=10, max_param_gap_report=5,
    )

    lenient = metrics["extraction_recall"]
    strict = metrics["fast_path_param_accuracy_scored"]
    assert metrics["pointer_wrong"] == 2
    assert strict == 0.0, f"strict recall with 2 wrong answers must be 0, got {strict}"
    if lenient == lenient:      # not NaN
        assert strict <= lenient


def test_eval_strict_recall_counts_wrong_answers_in_the_denominator(tmp_path) -> None:
    """No wrong answers: strict collapses to the headline value."""
    split = _mixed_extractability_split(tmp_path, n=1)
    metrics = eval_param_extraction_lean(
        split, resolver=_StubResolver({"city": "Tokyo"}), data_dir=str(tmp_path),
        limit=10, max_param_gap_report=5,
    )
    assert metrics["extraction_recall"] == 1.0
    assert metrics["fast_path_param_accuracy_scored"] == 1.0
    # No wrong answers, so the strict universe collapses to the headline one.
    assert metrics["fast_path_params_scored"] == 1


# ---------------------------------------------------------------------------
# T11: config placement must be the one the loader actually reads
# ---------------------------------------------------------------------------


def test_freeze_encoder_stages_loads_from_the_TRAINING_section(tmp_path) -> None:
    """T11: the YAML knob lives under ``training``, not ``model``.

    The notebook generator wrote it under ``model`` for a long time, which
    ``TrainingConfig.from_yaml`` never reads -- so the setting was a SILENT
    no-op and the run trained the whole encoder while claiming otherwise.
    """
    from spancall.train.config import TrainingConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "model:\n"
        "  freeze_encoder_stages: [T9]\n"       # wrong section: must be ignored
        "training:\n"
        "  freeze_encoder_stages: [T0, T1]\n"   # the real one
    )
    cfg = TrainingConfig.from_yaml(yaml_path)
    assert cfg.freeze_encoder_stages == ["T0", "T1"]

    # The wrong-section-only case degrades to the default (no freezing), which is
    # exactly the silent no-op this test exists to prevent from regressing.
    yaml_path.write_text("model:\n  freeze_encoder_stages: [T9]\n")
    cfg = TrainingConfig.from_yaml(yaml_path)
    assert cfg.freeze_encoder_stages == []


# ---------------------------------------------------------------------------
# T12c: logged losses are clamped at the metric boundary only
# ---------------------------------------------------------------------------


def test_log_loss_clamps_negative_but_preserves_nan() -> None:
    """T12c: a -1e-4 bf16 artifact must log as 0.0; a real NaN stays visible."""
    from spancall.lean.trainer import _log_loss

    assert _log_loss(-1e-4) == 0.0
    assert _log_loss(-5.0) == 0.0
    assert _log_loss(0.0) == 0.0
    assert _log_loss(2.5) == 2.5
    # NaN must NOT be scrubbed to 0.0: that would report a diverged run as perfect.
    nan = _log_loss(float("nan"))
    assert nan != nan
    # A non-numeric value degrades to 0.0 rather than raising mid-training.
    assert _log_loss(None) == 0.0


# ---------------------------------------------------------------------------
# T10: LoRA ablation wiring
# ---------------------------------------------------------------------------


def test_lora_config_defaults_are_off_and_round_trip(tmp_path) -> None:
    """T10: LoRA must be OFF by default, and the loader must read its section.

    ``training`` is the section ``TrainingConfig.from_yaml`` actually consults,
    so a knob written anywhere else is a silent no-op.
    """
    from spancall.train.config import TrainingConfig

    # Default: off, and the adapters' own LR is independent of the encoder's.
    assert TrainingConfig().use_lora is False
    assert TrainingConfig().lora_lr > TrainingConfig().optimization.learning_rate_encoder

    yaml_path = tmp_path / "lora.yaml"
    yaml_path.write_text(
        "training:\n"
        "  use_lora: true\n"
        "  lora_r: 8\n"
        "  lora_alpha: 16\n"
        "  lora_dropout: 0.1\n"
        "  lora_lr: 5.0e-4\n"
        "  lora_target_modules: [q_proj, v_proj]\n"
    )
    cfg = TrainingConfig.from_yaml(yaml_path)
    assert cfg.use_lora is True
    assert cfg.lora_r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.1
    assert cfg.lora_lr == 5e-4
    assert cfg.lora_target_modules == ["q_proj", "v_proj"]


def test_build_optimizer_groups_isolates_adapters_at_their_own_lr() -> None:
    """T10: ``lora_*`` params must NOT inherit the layer-decayed encoder LR.

    They live under ``backbone.``, so without the explicit branch they would be
    swept into the deepest encoder group (~2e-5) and never move at the adapter
    LR the config asks for.
    """

    class _Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = 8
            self.emb = nn.Linear(4, 8)
            # ``backbone.backbone.layers`` is one of the paths the builder walks,
            # so this exercises the real layer-decay discovery instead of a stub.
            self.backbone = nn.Module()
            self.backbone.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])

        def forward(self, ids, mask=None, ttype=None):  # pragma: no cover - unused
            return None

    model = LeanSpanCallModel(_Enc())
    # Simulate what peft does: an adapter parameter under the backbone namespace.
    model.backbone.register_parameter("lora_A_probe", nn.Parameter(torch.zeros(2, 2)))

    groups = build_lean_optimizer_groups(
        model, lr_encoder=2e-5, lr_head=1e-4, lora_lr=1e-3,
    )
    by_lr = {round(float(g["lr"]), 10): g for g in groups}
    assert 1e-3 in by_lr, f"no adapter group at lora_lr; lrs={sorted(by_lr)}"

    adapter_groups = [g for g in groups if g["lr"] == 1e-3]
    assert len(adapter_groups) == 1
    names = {
        n for n, p in model.named_parameters()
        if any(p is q for q in adapter_groups[0]["params"])
    }
    assert names == {"backbone.lora_A_probe"}, names

    # Without ``lora_lr`` nothing changes: the probe falls back to the encoder
    # groups, which is the pre-T10 behaviour every existing run depends on.
    legacy = build_lean_optimizer_groups(model, lr_encoder=2e-5, lr_head=1e-4)
    assert all(g["lr"] != 1e-3 for g in legacy)


def test_apply_lora_fails_loudly_when_peft_is_missing(monkeypatch) -> None:
    """T10: a missing ``peft`` must raise, never silently full-fine-tune.

    Silently training a different model than the config asked for is exactly the
    class of defect this investigation already caught with `freeze_encoder_stages`.
    """
    import builtins

    real_import = builtins.__import__

    def _no_peft(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("No module named 'peft'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_peft)

    class _Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = 8
            self.emb = nn.Linear(4, 8)

    with pytest.raises(RuntimeError, match="requires the `peft` package"):
        apply_lora(LeanSpanCallModel(_Enc()), r=4, alpha=8)


def test_trainer_provenance_records_whether_lora_was_used(tmp_path) -> None:
    """T10: provenance.json must distinguish the two ablation arms."""
    import json

    class _Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = 8
            self.emb = nn.Linear(4, 8)

    trainer = LeanTrainer(LeanSpanCallModel(_Enc()), config=None, device="cpu")
    assert trainer.lora_enabled is False
    path = trainer.save_checkpoint(str(tmp_path / "ckpt"))
    meta = json.loads((tmp_path / "ckpt" / "provenance.json").read_text())

    assert meta["use_lora"] is False
    assert meta["lora_param_count"] == 0
    assert path.endswith("model.pt")


def test_micro_loss_is_invariant_to_batch_composition() -> None:
    """Train `mean_loss` and eval `eval_loss` must share ONE scale.

    The bug this pins: `pointer` / `tool` are per-BATCH means, so the old
    `mean of batch means` moved when the SAME rows were merely bucketed
    differently (shuffled train batches vs. length-sorted eval batches).
    Re-bucketing 8 identical rows into 5+3 flipped the reported loss by >1 nat,
    which is what made train and dev losses look like different statistics.
    The slot-weighted (micro) average must be invariant to that regrouping.
    """
    torch.manual_seed(0)
    container = LeanLossContainer(lambda_pointer=1.0, lambda_tool=1.0)

    def _make(bs: int):
        B, K, C, P, T = bs, 3, 25, 2, 12
        span = torch.randn(B, K, C)
        pos = torch.full((B, K, P), -1, dtype=torch.long)
        pos[:, 0, 0] = 1                      # one supervised slot per row
        tool = torch.randn(B, T)
        tgt = torch.full((B,), -1, dtype=torch.long)
        tgt[: max(B // 2, 1)] = 4             # only half the rows carry H1.
        return ({"span_logits": span, "tool_logits": tool},
                {"pointer_positives": pos, "tool_targets": tgt})

    preds, tgts = _make(8)
    a = ({k: v[:5] for k, v in preds.items()}, {k: v[:5] for k, v in tgts.items()})
    b = ({k: v[5:] for k, v in preds.items()}, {k: v[5:] for k, v in tgts.items()})

    one = _MicroLossAccumulator()
    one.add(container(preds, tgts))

    two = _MicroLossAccumulator()
    two.add(container(*a))
    two.add(container(*b))

    lam = (container.lambda_pointer, container.lambda_tool)
    assert abs(one.combined(*lam) - two.combined(*lam)) < 1e-5

    # ...and the legacy statistic is exactly what must NOT be trusted for this.
    legacy = 0.5 * (float(container(*a)["loss"]) + float(container(*b)["loss"]))
    assert abs(float(container(preds, tgts)["loss"]) - legacy) > 1e-3


def test_micro_loss_reports_nan_when_nothing_is_supervised() -> None:
    """An unsupervised batch must not fake a 0.0 loss on the aligned scale."""
    acc = _MicroLossAccumulator()
    assert math.isnan(acc.value("pointer"))
    assert math.isnan(acc.value("tool"))
    assert math.isnan(acc.combined(1.0, 1.0))


