"""Tests for :class:`spancall.model.heads` and :class:`spancall.model.loss`.

Verifies (plan §3.2, §3.3; heads.md §H1-H8; docs/remediation_plan D4):
  * per-head output shapes,
  * PointerHead width-bias: equal start/end scores favor the *narrower* span,
  * H3-H4 consistency (abstain logit vs logit(not-OK)) drives gradient agreement,
  * gradient flow reaches each head's parameters,
  * H8 stop-grad: CorrectnessHead never moves encoder / H3 (plan §3.2.5),
  * R4 equal-value invariance: multi-positive pointer loss is invariant to
    positive-set order, and equal targets sum mass,
  * SpanCallLossContainer aggregation (joint weighted total).
"""

from __future__ import annotations

import math

import pytest
import torch

from spancall.model.heads import (
    AbstainHead,
    CallAcceptHead,
    CallCountHead,
    CorrectnessHead,
    PointerHead,
    ToolHead,
)
from spancall.model.loss import SpanCallLossContainer

B, D, DH, L, K, T, C = 3, 16, 8, 12, 4, 3, 5  # batch, hidden, head, seq, params, tools, cands


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def seq() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, L, D)


@pytest.fixture
def param_emb() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, K, D)


@pytest.fixture
def cls_emb(seq: torch.Tensor) -> torch.Tensor:
    return seq[:, 0]


@pytest.fixture
def tool_emb() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(B, T, D)


@pytest.fixture
def cand_masks() -> dict:
    # (B, K, C) starts/ends; all valid (>= 0)
    torch.manual_seed(3)
    s = torch.randint(0, L, (B, K, C))
    e = torch.randint(0, L, (B, K, C))
    e = torch.maximum(e, s)
    return {"starts": s, "ends": e}


# --------------------------------------------------------------------------- #
# H1 / H2
# --------------------------------------------------------------------------- #
def test_tool_head_shape(tool_emb: torch.Tensor, cls_emb: torch.Tensor) -> None:
    out = ToolHead(D)(tool_emb, cls_emb)
    assert out.shape == (B, T), f"expected (B,T) got {out.shape}"


def test_tool_head_no_cls_fallback(tool_emb: torch.Tensor) -> None:
    # cls_embedding=None uses the mean over anchors as a deterministic fallback
    out = ToolHead(D)(tool_emb, None)
    assert out.shape == (B, T)


def test_count_head_shape(cls_emb: torch.Tensor) -> None:
    out = CallCountHead(D, max_calls=4)(cls_emb)
    assert out.shape == (B, 5)


# --------------------------------------------------------------------------- #
# H3 — PointerHead
# --------------------------------------------------------------------------- #
def test_pointer_outputs_and_shapes(
    param_emb: torch.Tensor, seq: torch.Tensor, cand_masks: dict
) -> None:
    head = PointerHead(D, head_dim=DH, max_span_width=4)
    out = head(param_emb, seq, candidate_masks=cand_masks)
    assert out["q"].shape == (B, K, DH)
    assert out["start_logits"].shape == (B, K, L)
    assert out["end_logits"].shape == (B, K, L)
    assert out["abstain_logits"].shape == (B, K)
    assert out["span_logits"].shape == (B, K, C)


def test_pointer_positive_width_prior(
    param_emb: torch.Tensor, seq: torch.Tensor
) -> None:
    """Equal start/end scores favor the narrower span (width bias r(j - i))."""
    head = PointerHead(D, head_dim=DH, max_span_width=4)
    # Widen the width bias embedding so the width term dominates other noise.
    with torch.no_grad():
        head.width_bias.weight.zero_()
        # r(0)=0, r(1)=-0.5, r(2)=-1.0, ... -> negative for non-zero widths
        for w in range(1, head.max_span_width + 1):
            head.width_bias.weight[w, 0] = -0.5 * w
    # Same start and end boundary tokens for candidate 1 (width 0) and
    # candidate 2 (width 0), but candidate 0 uses a wider span.
    s = torch.zeros(B, K, 3, dtype=torch.long)
    e = torch.zeros(B, K, 3, dtype=torch.long)
    s[:, :, 0], e[:, :, 0] = 2, 8   # wide span
    s[:, :, 1], e[:, :, 1] = 3, 3   # narrow (same token)
    s[:, :, 2], e[:, :, 2] = 3, 3   # narrow duplicate
    out = head(param_emb, seq, candidate_masks={"starts": s, "ends": e})
    span = out["span_logits"]  # (B, K, 3)
    # For every (b,k), the two narrow candidates must beat the wide one.
    assert (span[:, :, 1] > span[:, :, 0]).all().item()
    assert (span[:, :, 2] > span[:, :, 0]).all().item()


def test_pointer_padded_slots_are_negative(
    param_emb: torch.Tensor, seq: torch.Tensor
) -> None:
    head = PointerHead(D, head_dim=DH, max_span_width=4)
    s = torch.full((B, K, C), -1, dtype=torch.long)
    e = torch.full((B, K, C), -1, dtype=torch.long)
    s[:, :, 0], e[:, :, 0] = 1, 1
    out = head(param_emb, seq, candidate_masks={"starts": s, "ends": e})
    assert (out["span_logits"][:, :, 0] > out["span_logits"][:, :, 1:].max()).all().item()


# --------------------------------------------------------------------------- #
# H4 — AbstainHead
# --------------------------------------------------------------------------- #
def test_abstain_shape(
    param_emb: torch.Tensor, cls_emb: torch.Tensor, seq: torch.Tensor
) -> None:
    utt = seq.mean(1)
    out = AbstainHead(D)(param_emb, cls_emb, utt)
    assert out.shape == (B, K, 5)


def test_abstain_falls_back_without_utt(
    param_emb: torch.Tensor, cls_emb: torch.Tensor
) -> None:
    out = AbstainHead(D)(param_emb, cls_emb, None)
    assert out.shape == (B, K, 5)


# --------------------------------------------------------------------------- #
# CallAcceptHead + CorrectnessHead
# --------------------------------------------------------------------------- #
def test_call_accept_shape(cls_emb: torch.Tensor) -> None:
    out = CallAcceptHead(D)(cls_emb)
    assert out.shape == (B, 1)


def test_correctness_shape(
    param_emb: torch.Tensor, seq: torch.Tensor
) -> None:
    corr = CorrectnessHead(D, head_dim=DH)
    span = seq[:, :K].mean(1, keepdim=True).expand(B, K, D)  # arbitrary pooled spans
    q = torch.randn(B, K, DH)
    stats = torch.randn(B, K, 8)
    out = corr(span, q, stats)
    assert out.shape == (B, K, 1)


def test_correctness_stop_grad(
    param_emb: torch.Tensor, seq: torch.Tensor
) -> None:
    """H8 must never move the encoder (H) or H3 query (q) — plan §3.2.5.

    Feed a *leaf* sequence embedding (as if it came from a frozen encoder) and
    a *leaf* H3 query; after one backward the leaf inputs must have no grad —
    i.e. the head detached them.
    """
    corr = CorrectnessHead(D, head_dim=DH)
    span = torch.randn(B, K, D, requires_grad=True)      # simulates encoder H pooled
    q = torch.randn(B, K, DH, requires_grad=True)        # simulates H3 query
    stats = torch.randn(B, K, 8)
    target = torch.randint(0, 2, (B, K)).float().unsqueeze(-1)

    logit = corr(span, q, stats)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
    loss.backward()

    assert span.grad is None, "correctness head leaked grad into the encoder"
    assert q.grad is None, "correctness head leaked grad into H3"
    # its own params DO get gradients
    assert corr.mlp[0].weight.grad is not None


# --------------------------------------------------------------------------- #
# R4 equal-value invariance
# --------------------------------------------------------------------------- #
def test_pointer_loss_equal_value_invariance() -> None:
    """Multi-positive loss is order-invariant and sums mass for equal targets (R4)."""
    container = SpanCallLossContainer()
    z = torch.randn(B, K, C)
    # the positive *set* {1, 2}; incoming index order must not change the loss
    p1 = torch.tensor([[[1, 2], [1, 2], [1, 2], [1, 2]]], dtype=torch.long)
    p1 = p1.expand(B, K, -1)
    p_swapped = p1[..., [1, 0]]  # same set, reversed order
    l1 = container.compute_pointer_loss(z, p1)
    l2 = container.compute_pointer_loss(z, p_swapped)
    assert torch.allclose(l1, l2, atol=1e-5), f"{l1} vs {l2}"

    # equal-valued candidates both in the positive set also sum mass symmetrically
    zz = torch.randn(B, K, C)
    p_both_same = torch.tensor([[[1, 1], [1, 1], [1, 1], [1, 1]]], dtype=torch.long).expand(B, K, -1)
    l_dup = container.compute_pointer_loss(zz, p_both_same)
    l_sing = container.compute_pointer_loss(zz, torch.tensor([[[1, -1]]], dtype=torch.long).expand(B, K, -1))
    # two identical positives must not double-count: logsumexp of two equal logits
    # = z_c + log(2), so the loss is the single-positive loss minus log(2).
    assert torch.allclose(l_dup, l_sing - math.log(2.0), atol=1e-4)


def test_pointer_abstain_upweight() -> None:
    container = SpanCallLossContainer(abstain_weight=1.5)
    z = torch.randn(B, K, C)
    # zero everything so the raw (pre-weight) losses for abstain and hit are equal
    zz = torch.zeros_like(z)
    zz[:, :, 0] = 1.0  # abstain logit
    zz[:, :, 1] = 1.0  # hit candidate has the same logit -> equal raw loss
    abstain = torch.tensor([[[0, -1, -1]]]).expand(B, K, -1)
    hit = torch.tensor([[[1,  -1, -1]]]).expand(B, K, -1)
    la = container.compute_pointer_loss(zz, abstain)
    lh = container.compute_pointer_loss(zz, hit)
    assert torch.allclose(la, 1.5 * lh, atol=1e-5), f"{la} vs {lh}"


def test_pointer_masked_rows_zero_grad() -> None:
    container = SpanCallLossContainer()
    z = torch.randn(B, K, C, requires_grad=True)
    masked = torch.full((B, K, 1), -1, dtype=torch.long)
    l = container.compute_pointer_loss(z, masked)
    assert (l == 0).all().item()
    l.sum().backward()
    assert torch.all(z.grad == 0), "masked rows must not contribute gradient"


# --------------------------------------------------------------------------- #
# Gradient flow through the full trainer-used forward
# --------------------------------------------------------------------------- #
def test_joint_loss_gradient_flow(
    param_emb: torch.Tensor, seq: torch.Tensor,
    cls_emb: torch.Tensor, cand_masks: dict,
) -> None:
    """All heads' params receive gradient through SpanCallLossContainer.forward."""
    tool_head = ToolHead(D)
    pointer = PointerHead(D, head_dim=DH, max_span_width=4)
    abstain = AbstainHead(D)
    accept = CallAcceptHead(D)
    corr = CorrectnessHead(D, head_dim=DH)

    pout = pointer(param_emb, seq, candidate_masks=cand_masks)
    z = torch.cat([pout["abstain_logits"].unsqueeze(-1), pout["span_logits"]], -1)
    reason = abstain(param_emb, cls_emb, seq.mean(1))
    acc = accept(cls_emb)
    stats = torch.randn(B, K, 8)
    span = seq[:, :K].mean(1, keepdim=True).expand(B, K, D)
    clog = corr(span, pout["q"], stats)

    preds = {
        "pointer_logits": z,
        "reason_logits": reason,
        "accept_logits": acc,
        "tool_logits": tool_head(torch.randn(B, T, D), cls_emb),
        "corr_logits": clog,
    }
    targs = {
        "pointer_positives": torch.tensor([[[1, 2], [1, 2], [1, 2], [1, 2]]]).expand(B, K, -1),
        "reason_logits": torch.tensor([[1, 1, 3, 0]]).expand(B, K),  # reason TARGET labels
        "accept_targets": torch.ones(B),
        "tool_targets": torch.zeros(B, dtype=torch.long),
    }
    losses = SpanCallLossContainer().forward(preds, targs)
    assert losses["loss"].requires_grad
    losses["loss"].backward()

    for h, name in [
        (tool_head, "tool_head"),
        (pointer, "pointer_head"),
        (abstain, "abstain_head"),
        (accept, "accept_head"),
        (corr, "correctness_head"),
    ]:
        grads = [p.grad is not None for p in h.parameters() if p.requires_grad]
        assert all(grads), f"missing grad in {name}: {grads}"


# --------------------------------------------------------------------------- #
# SpanCallLossContainer aggregation
# --------------------------------------------------------------------------- #
def test_loss_container_returns_all_components() -> None:
    container = SpanCallLossContainer()
    z = torch.randn(B, K, C)
    reason = torch.randn(B, K, 5)
    preds = {
        "pointer_logits": z,
        "reason_logits": reason,
        "accept_logits": torch.randn(B, 1),
        "tool_logits": torch.randn(B, T),
        "corr_logits": torch.randn(B, K, 1),
    }
    targs = {
        "pointer_positives": torch.tensor([[[1, 2], [1, 2], [1, 2], [1, 2]]]).expand(B, K, -1),
        "reason_logits": torch.tensor([[1, 1, 3, 0]]).expand(B, K),
        "accept_targets": torch.ones(B),
        "tool_targets": torch.zeros(B, dtype=torch.long),
    }
    losses = container.forward(preds, targs)
    for key in ["pointer", "accept", "reason", "tool", "corr", "loss"]:
        assert key in losses, f"missing {key}"
    assert losses["loss"].shape == ()
    assert losses["loss"] > 0


def test_loss_pointer_math() -> None:
    """Sanity on the multi-positive softmax: pushing up a positive lowers the loss."""
    container = SpanCallLossContainer()
    z = torch.randn(B, K, C)
    pos = torch.tensor([[[1, -1, -1]]]).expand(B, K, -1)
    l0 = container.compute_pointer_loss(z, pos)
    z_hi = z.clone()
    z_hi[:, :, 1] += 10.0  # boost the single positive candidate
    l1 = container.compute_pointer_loss(z_hi, pos)
    assert (l1 < l0).all().item(), f"expected push-up to lower loss: {l0} -> {l1}"


def test_loss_consistency_penalty() -> None:
    """H3-H4 consistency: margin between abstain logit and logit(not-OK) is penalized."""
    container = SpanCallLossContainer(reason_mse_weight=0.1)
    z = torch.randn(B, K, C)
    abstain_logits = z[..., 0]
    reason = torch.randn(B, K, 5)  # irrelevant; only the OK/not-OK split matters
    preds = {"pointer_logits": z, "reason_logits": reason}
    targs = {
        "pointer_positives": torch.tensor([[[1, 2], [1, 2], [1, 2], [1, 2]]]).expand(B, K, -1),
        "reason_logits": torch.tensor([[1, 1, 3, 0]]).expand(B, K),
    }
    l1 = container.forward(preds, targs)["reason"]
    # Make the abstain logit exactly equal logit(not-OK): the MSE term goes to
    # 0 while the H4 CE term is bit-identical (reason logits untouched).
    ok = reason[..., 0]
    not_ok = torch.logsumexp(reason[..., 1:], -1)
    z2 = z.clone()
    z2[..., 0] = not_ok - ok  # abstain logit == logit(not-OK) for every (b,k)
    preds2 = {"pointer_logits": z2, "reason_logits": reason}
    l2 = container.forward(preds2, targs)["reason"]
    assert l2 < l1, "consistency penalty should shrink when heads agree"


def test_loss_container_weighted_total() -> None:
    container = SpanCallLossContainer(lambda_pointer=2.0, lambda_accept=0.0)
    z = torch.randn(B, K, C)
    preds = {"pointer_logits": z}
    targs = {
        "pointer_positives": torch.tensor([[[1, 2], [1, 2], [1, 2], [1, 2]]]).expand(B, K, -1),
    }
    out = container.forward(preds, targs)
    assert out["loss"] == pytest.approx(2.0 * out["pointer"], rel=1e-3)