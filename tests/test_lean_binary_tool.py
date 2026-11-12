"""v2.1 — the binary H1 tool-selection head (ablation arm of the softmax head).

What these tests protect:

1. **The objective is real, not a relabelling.** ``compute_lean_tool_loss_binary``
   must scatter a called-anchor target into a one-hot and take BCE per anchor, so
   the score of one tool never depends on the other tools' scores. A regression
   to softmax semantics would still return a number, so the numerics are pinned
   against a hand-computed BCE.
2. **Padded anchors stay silent.** Callers mask dead anchors with ``-1e4``;
   ``BCEWithLogits(-1e4, 0) ~ 0`` is what makes the binary arm compatible with
   the existing masking convention. If that ever stopped holding, every padded
   row would inject a huge gradient.
3. **Flag OFF is byte-identical.** No ``tool_binary_*`` key may appear in the
   trainer or loss output unless the flag is on; the shipped softmax arm must
   remain reproducible.
4. **The gate is binary-only.** A sigmoid is an absolute ``P(called)`` so a fixed
   threshold is meaningful; a softmax is relative and must never be thresholded
   here, or stored CE predictions would silently change.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest
import torch
import torch.nn as nn

from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.lean.loss import (
    LeanLossContainer,
    compute_lean_tool_loss_binary,
)
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver
from spancall.lean.trainer import LeanTrainer
from spancall.model.heads import BinaryToolHead
from spancall.train.config import OptimizationConfig


class TinyDummyEncoder(nn.Module):
    """Deterministic stand-in for the backbone.

    Carries a real ``nn.Parameter`` because the resolver's ``_encode`` reads the
    backbone's device from ``next(backbone.parameters())`` -- a parameterless
    stub raises ``StopIteration`` and every resolver call would degrade into a
    ``pre_error``, which is a stub artefact rather than a defect under test.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        B, L = input_ids.shape
        base = torch.arange(L, dtype=torch.float32, device=input_ids.device)
        H = base.view(1, L, 1).expand(B, L, self.hidden_dim)
        return (H * self.scale.view(1, 1, -1)).contiguous()


def _weather_schema() -> ToolDefinition:
    return ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters=[
            ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ],
    )


def _time_schema() -> ToolDefinition:
    return ToolDefinition(
        name="get_time",
        description="Get current time in a city",
        parameters=[
            ParameterSchema(name="city", description="City name", param_type="string", required=True),
        ],
    )


def _model() -> LeanSpanCallModel:
    return LeanSpanCallModel(encoder=TinyDummyEncoder(hidden_dim=32), head_dim=16, hidden_dim=32)


class _ConstantHead(nn.Module):
    """Returns one fixed logit per rendered anchor, broadcast over the batch.

    Used to make the resolver's tool distribution exactly predictable so the
    gate/threshold assertions are about the gate, not about random init.
    """

    def __init__(self, values: List[float]) -> None:
        super().__init__()
        self.values = list(values)

    def forward(self, tool_tokens: torch.Tensor, cls_embedding: Optional[torch.Tensor] = None):
        T = int(tool_tokens.size(1))
        vals = self.values[:T]
        while len(vals) < T:
            vals.append(vals[-1] if vals else 0.0)
        return torch.tensor([vals], dtype=tool_tokens.dtype, device=tool_tokens.device)


# --------------------------------------------------------------------------- #
# 1. head
# --------------------------------------------------------------------------- #
def test_binary_head_emits_one_logit_per_anchor() -> None:
    head = BinaryToolHead(16)
    tool_tokens = torch.randn(3, 5, 16)
    out = head(tool_tokens, torch.randn(3, 16))
    assert out.shape == (3, 5)


def test_binary_head_defaults_cls_to_anchor_mean() -> None:
    """``cls_embedding=None`` must still work: callers/tests may omit it."""
    head = BinaryToolHead(8)
    out = head(torch.randn(2, 4, 8))
    assert out.shape == (2, 4)
    out3 = head(torch.randn(2, 4, 8), torch.randn(2, 1, 8))
    assert out3.shape == (2, 4)


def test_binary_head_has_no_softmax_coupling() -> None:
    """Sum of sigmoids is NOT constrained to 1 — that is the whole point."""
    head = BinaryToolHead(8)
    with torch.no_grad():
        # Make every anchor maximally confident.
        head.mlp[2].bias.fill_(20.0)
        head.mlp[2].weight.zero_()
    p = torch.sigmoid(head(torch.randn(2, 4, 8)))
    assert torch.allclose(p, torch.ones_like(p), atol=1e-4)


# --------------------------------------------------------------------------- #
# 2. loss
# --------------------------------------------------------------------------- #
def test_binary_loss_matches_a_hand_computed_bce_for_scalar_targets() -> None:
    logits = torch.zeros(2, 4)
    targets = torch.tensor([0, 3])
    loss = compute_lean_tool_loss_binary(logits, targets)
    # one-hot row: -log(0.5) at the positive, same at each of the 3 negatives,
    # averaged over the 4 anchors.
    expected = -math.log(0.5)
    assert float(loss) == pytest.approx(expected, rel=1e-6)


def test_binary_loss_accepts_the_multi_call_target_shape() -> None:
    """(B, P) set-valued targets scatter ALL called anchors into the one-hot."""
    logits = torch.zeros(1, 4)
    targets = torch.tensor([[0, 2, -1]])
    loss = compute_lean_tool_loss_binary(logits, targets)
    # two positives, two negatives -> still -log(0.5) per anchor when all are 0.
    assert float(loss) == pytest.approx(-math.log(0.5), rel=1e-6)


def test_binary_loss_masks_padded_rows_and_returns_none_without_supervision() -> None:
    logits = torch.zeros(3, 4)
    all_pad = torch.full((3,), -1, dtype=torch.long)
    assert compute_lean_tool_loss_binary(logits, all_pad) is None
    # One supervised row out of three: the padded rows must not dilute the mean.
    mixed = torch.tensor([0, -1, -1])
    loss = compute_lean_tool_loss_binary(logits, mixed)
    assert float(loss) == pytest.approx(-math.log(0.5), rel=1e-6)


def test_binary_loss_returns_none_for_degenerate_shapes() -> None:
    assert compute_lean_tool_loss_binary(torch.zeros(0, 4), torch.tensor([], dtype=torch.long)) is None
    assert compute_lean_tool_loss_binary(torch.zeros(2), torch.tensor([0, 1])) is None
    # Mismatched row counts cannot be aligned; refuse rather than mis-scatter.
    assert compute_lean_tool_loss_binary(torch.zeros(2, 4), torch.tensor([0, 1, 2])) is None


def test_binary_loss_pos_weight_upweights_the_called_anchor() -> None:
    logits = torch.zeros(1, 4)
    targets = torch.tensor([0])
    plain = float(compute_lean_tool_loss_binary(logits, targets))
    weighted = float(compute_lean_tool_loss_binary(logits, targets, pos_weight=torch.full((4,), 3.0)))
    # 3x on the single positive, 1x on the three negatives, re-averaged.
    expected = (3.0 + 3.0) / 4.0 * -math.log(0.5)
    assert weighted == pytest.approx(expected, rel=1e-6)
    assert weighted > plain


def test_binary_loss_is_silent_on_a_dead_anchor() -> None:
    """A ``-1e4`` dead anchor must neither explode nor dilute the mean.

    The CE path normalises over LIVE anchors only, so a row whose anchor failed
    to snap keeps the same scale. A plain ``mean(-1)`` over both anchors would
    report half the loss and silently make the two arms incomparable.
    """
    logits = torch.tensor([[-1e4, 0.0]])
    targets = torch.tensor([1])
    loss = float(compute_lean_tool_loss_binary(logits, targets))
    assert loss == pytest.approx(-math.log(0.5), rel=1e-6)
    # Sanity: the ALL-dead row is excluded, not scored as a near-zero loss.
    all_dead = torch.tensor([[-1e4, -1e4]])
    assert compute_lean_tool_loss_binary(all_dead, torch.tensor([1])) is None


def test_binary_loss_scale_is_invariant_to_padding_columns() -> None:
    """Two live anchors and two live-plus-pad must report the SAME loss."""
    live = torch.tensor([[0.0, 0.0]])
    padded = torch.tensor([[0.0, 0.0, -1e4, -1e4]])
    t_live = torch.tensor([0])
    t_padded = torch.tensor([0])
    assert float(compute_lean_tool_loss_binary(live, t_live)) == pytest.approx(
        float(compute_lean_tool_loss_binary(padded, t_padded)), rel=1e-6
    )


# --------------------------------------------------------------------------- #
# 3. loss container
# --------------------------------------------------------------------------- #
def _container_preds(**extra: Any) -> dict:
    return {"span_logits": torch.zeros(1, 2, 3), **extra}


def test_container_flag_off_emits_no_binary_keys() -> None:
    """No ``tool_binary_logits`` -> the CE branch, exactly as shipped."""
    c = LeanLossContainer()
    out = c(
        _container_preds(tool_logits=torch.zeros(2, 3)),
        {"pointer_positives": torch.zeros(1, 2, 1, dtype=torch.long), "tool_targets": torch.tensor([0, 1])},
    )
    assert "tool" in out and "tool_sum" in out and "tool_count" in out
    assert not any(k.startswith("tool_binary") for k in out)


def test_container_binary_arm_keeps_ce_reference_only() -> None:
    """In binary mode the CE term is reported but must NOT enter the total."""
    c = LeanLossContainer(tool_objective="binary")
    preds = _container_preds(
        tool_logits=torch.zeros(2, 3),
        tool_binary_logits=torch.zeros(2, 3),
    )
    targets = {
        "pointer_positives": torch.zeros(1, 2, 1, dtype=torch.long),
        "tool_targets": torch.tensor([0, 1]),
    }
    out = c(preds, targets)
    assert "tool" in out  # reference metric still reported
    assert "tool_sum" not in out and "tool_count" not in out  # not micro-averaged
    assert "tool_binary" in out
    # The total is pointer + binary ONLY: the CE term is reported, not summed.
    pointer = float(out["pointer"])
    assert float(out["loss"]) == pytest.approx(pointer + float(out["tool_binary"]), rel=1e-6)
    # And the binary arm trains a DIFFERENT total than the CE arm would.
    ce_total = LeanLossContainer(tool_objective="ce")(preds, targets)
    assert float(ce_total["loss"]) == pytest.approx(pointer + float(ce_total["tool"]), rel=1e-6)


def test_container_rejects_an_unknown_objective() -> None:
    with pytest.raises(ValueError, match="tool_objective must be"):
        LeanLossContainer(tool_objective="softmax-ish")


def test_container_binary_arm_reports_micro_bookkeeping() -> None:
    c = LeanLossContainer(tool_objective="binary")
    out = c(
        _container_preds(tool_binary_logits=torch.zeros(3, 4)),
        {"pointer_positives": torch.zeros(1, 2, 1, dtype=torch.long), "tool_targets": torch.tensor([0, 2, -1])},
    )
    assert float(out["tool_binary_count"]) == 2.0
    assert float(out["tool_binary_sum"]) == pytest.approx(
        float(out["tool_binary"]) * 2.0, rel=1e-6
    )


# --------------------------------------------------------------------------- #
# 4. trainer
# --------------------------------------------------------------------------- #
def _batch() -> dict:
    B, L, K, C = 2, 15, 2, 4
    return {
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


def test_trainer_flag_off_adds_no_binary_metrics() -> None:
    t = LeanTrainer(model=_model(), lr_encoder=1e-3, lr_head=1e-3, device="cpu", use_amp=False)
    assert t.use_binary_tool_head is False
    assert t.loss_container.tool_objective == "ce"
    m = t.train_step(_batch())
    assert "tool_acc" in m
    assert not any(k.startswith("tool_binary") for k in m)
    assert not any(k.startswith("_tool_binary") for k in m)
    ev = t.evaluate([_batch()])
    assert "tool_acc" in ev and "tool_binary_acc" not in ev


def test_trainer_flag_on_emits_binary_metrics_and_trains_the_binary_objective() -> None:
    t = LeanTrainer(
        model=_model(),
        lr_encoder=1e-3,
        lr_head=1e-3,
        device="cpu",
        use_amp=False,
        use_binary_tool_head=True,
    )
    assert t.loss_container.tool_objective == "binary"
    m = t.train_step(_batch())
    # Kill-criterion pair: same argmax rule and the same denominator as tool_acc.
    assert "tool_binary_acc" in m and "tool_acc" in m
    assert m["n_tool_binary"] == m["n_tool"]
    assert "tool_binary_loss" in m
    assert m["_tool_binary_count"] > 0.0
    ev = t.evaluate([_batch()])
    assert "tool_binary_acc" in ev and ev["n_tool_binary"] == ev["n_tool"]
    # The micro loss must be the BINARY objective once that is what trains.
    assert "eval_tool_loss_micro" not in ev or math.isnan(ev["eval_tool_loss_micro"])


def test_trainer_reads_the_flag_from_the_optimization_config() -> None:
    opt = OptimizationConfig()
    opt.use_binary_tool_head = True
    t = LeanTrainer(
        model=_model(),
        config=SimpleNamespace(optimization=opt),
        lr_encoder=1e-3,
        lr_head=1e-3,
        device="cpu",
        use_amp=False,
    )
    assert t.use_binary_tool_head is True
    assert t.loss_container.tool_objective == "binary"


def test_trainer_explicit_arg_overrides_the_config() -> None:
    opt = OptimizationConfig()
    opt.use_binary_tool_head = True
    t = LeanTrainer(
        model=_model(),
        config=SimpleNamespace(optimization=opt),
        lr_encoder=1e-3,
        lr_head=1e-3,
        device="cpu",
        use_amp=False,
        use_binary_tool_head=False,
    )
    assert t.use_binary_tool_head is False
    assert t.loss_container.tool_objective == "ce"


def test_trainer_forces_an_injected_container_into_lock_step() -> None:
    """A CE container + flag on would silently train the OLD head."""
    t = LeanTrainer(
        model=_model(),
        loss_container=LeanLossContainer(tool_objective="ce"),
        lr_encoder=1e-3,
        lr_head=1e-3,
        device="cpu",
        use_amp=False,
        use_binary_tool_head=True,
    )
    assert t.loss_container.tool_objective == "binary"


# --------------------------------------------------------------------------- #
# 5. resolver
# --------------------------------------------------------------------------- #
def test_resolver_rejects_binary_mode_without_a_binary_head() -> None:
    model = _model()
    model.tool_head_binary = None
    with pytest.raises(ValueError, match="tool_head_binary"):
        LeanResolver(model=model, max_seq_len=512, tool_objective="binary")


def test_resolver_rejects_an_unknown_objective() -> None:
    with pytest.raises(ValueError, match="tool_objective must be"):
        LeanResolver(model=_model(), max_seq_len=512, tool_objective="nope")


def test_resolver_ce_path_ignores_the_absolute_threshold() -> None:
    """A low softmax probability must NOT abstain in ``ce`` mode."""
    model = _model()
    model.tool_head = _ConstantHead([0.0, 0.0])
    r = LeanResolver(model=model, max_seq_len=512, tool_objective="ce", tool_call_threshold=0.99)
    pred = r.predict_tool([_weather_schema(), _time_schema()], utterance="weather in Tokyo")
    assert pred["gated_hit"] is True
    assert pred["abstain_reason"] is None
    assert pred["tool_objective"] == "ce"
    assert pred["tool_call_threshold"] is None
    # softmax over 2 equal logits
    assert pred["tool_prob"] == pytest.approx(0.5, rel=1e-5)


def test_resolver_binary_mode_scores_independently_with_sigmoid() -> None:
    model = _model()
    model.tool_head_binary = _ConstantHead([5.0, 5.0])
    r = LeanResolver(model=model, max_seq_len=512, tool_objective="binary", tool_call_threshold=0.5)
    pred = r.predict_tool([_weather_schema(), _time_schema()], utterance="weather in Tokyo")
    # Two independent sigmoids at 5.0, NOT a 0.5/0.5 softmax split.
    assert pred["tool_prob"] == pytest.approx(1.0 / (1.0 + math.exp(-5.0)), rel=1e-5)
    assert sum(s for _n, s in pred["tool_scores"]) > 1.0
    assert pred["gated_hit"] is True
    assert pred["tool_call_threshold"] == pytest.approx(0.5)


def test_resolver_binary_mode_abstains_below_the_absolute_threshold() -> None:
    model = _model()
    model.tool_head_binary = _ConstantHead([-3.0, -3.0])
    r = LeanResolver(model=model, max_seq_len=512, tool_objective="binary", tool_call_threshold=0.5)
    pred = r.predict_tool([_weather_schema(), _time_schema()], utterance="weather in Tokyo")
    assert pred["gated_hit"] is False
    assert pred["abstain_reason"] == "tool_below_threshold"
    assert pred["tool_prob"] < 0.5


def test_resolver_single_candidate_short_circuit_is_identical_in_both_arms() -> None:
    """T == 1 carries no signal in either objective; the gate must not fire."""
    for obj in ("ce", "binary"):
        model = _model()
        if obj == "binary":
            model.tool_head_binary = _ConstantHead([-20.0])
        r = LeanResolver(model=model, max_seq_len=512, tool_objective=obj, tool_call_threshold=0.99)
        pred = r.predict_tool([_weather_schema()], utterance="weather in Tokyo")
        assert pred["gated_hit"] is True
        assert pred["tool_prob"] == 1.0
        assert pred["tool_objective"] == obj


def test_try_resolve_candidates_binary_abstains_below_threshold() -> None:
    model = _model()
    model.tool_head_binary = _ConstantHead([-3.0, -3.0])
    r = LeanResolver(model=model, max_seq_len=512, tool_objective="binary", tool_call_threshold=0.5)
    res = r.try_resolve_candidates([_weather_schema(), _time_schema()], utterance="weather in Tokyo")
    assert res.status == "abstain"
    assert res.abstain_reason == "tool_below_threshold"
    assert res.hit is False
    assert res.features["tool_objective"] == "binary"
    assert res.features["n_calls"] == 0


def test_try_resolve_candidates_ce_never_reports_the_binary_gate() -> None:
    model = _model()
    model.tool_head = _ConstantHead([0.0, 0.0])
    r = LeanResolver(model=model, max_seq_len=512, tool_call_threshold=0.99)
    res = r.try_resolve_candidates([_weather_schema(), _time_schema()], utterance="weather in Tokyo")
    assert res.abstain_reason != "tool_below_threshold"
    assert res.features["tool_objective"] == "ce"
    assert res.features["tool_call_threshold"] is None


# --------------------------------------------------------------------------- #
# 6. provenance
# --------------------------------------------------------------------------- #
def test_provenance_names_the_objective_so_weights_cannot_be_mis_scored(tmp_path) -> None:
    """A checkpoint must say which arm it is; the resolver has to be told too."""
    import json

    for flag in (False, True):
        t = LeanTrainer(
            model=_model(),
            lr_encoder=1e-3,
            lr_head=1e-3,
            device="cpu",
            use_amp=False,
            use_binary_tool_head=flag,
        )
        out = tmp_path / f"ckpt_{flag}"
        t.save_checkpoint(str(out))
        meta = json.loads((out / "provenance.json").read_text())
        assert meta["use_binary_tool_head"] is flag
        assert meta["tool_objective"] == ("binary" if flag else "ce")