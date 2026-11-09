"""Unit tests for the Lean flow (spancall/lean)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spancall.data.schema import GroundTruthCall, ParameterSchema, Stratum, ToolDefinition, UnifiedSample
from spancall.infer.resolver import FakeTokenizer
from spancall.lean.decode import LeanJointDecoder
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

