"""Lean Loss Container for SpanCall without abstain machinery.

Provides:
- compute_lean_pointer_loss: multi-positive softmax cross-entropy directly over
  candidate spans (B, K, C) without abstain columns or abstain weighting.
- LeanLossContainer: computes joint (tool_loss + pointer_loss).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared sentinel with the tensorizer/collator. Kept as a literal fallback so
# this module has no import cycle with ``tensorize``.
PAD_INDEX: int = -1


def compute_lean_tool_loss(
    tool_logits: torch.Tensor,
    tool_targets: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Set-valued H1 loss: any of the row's GT tools is a correct answer.

    Accepts two target shapes, and the shape decides the objective:

    - ``(B,)`` -- exactly one correct tool per row (the legacy contract). This
      path calls :func:`torch.nn.functional.cross_entropy` verbatim, so its
      value is **identical** to the pre-R0.4 implementation, not merely
      mathematically equivalent.
    - ``(B, P)`` -- up to ``P`` correct tools per row (one per supervised call,
      ``PAD_INDEX`` padded). The negative log marginal likelihood of the target
      SET is ``logsumexp(all tool logits) - logsumexp(target logits)``: any
      member of the set earns full credit, and the gradient still pushes the
      non-chosen GT tools up rather than treating them as negatives.

    Returns ``None`` when no row in the batch has a usable target. Callers must
    then SKIP the tool term entirely instead of adding a zero loss: a batch with
    no H1 supervision must not look like a batch with perfect H1 predictions.
    """
    if tool_logits.dim() != 2 or tool_logits.numel() == 0:
        return None
    C = int(tool_logits.size(-1))
    T = tool_targets
    if not isinstance(T, torch.Tensor) or T.numel() == 0:
        return None

    if T.dim() <= 1:
        flat = T.reshape(-1)
        if int(flat.size(0)) != int(tool_logits.size(0)):
            return None
        has = (flat >= 0) & (flat < C)
        if not bool(has.any()):
            return None
        return F.cross_entropy(tool_logits, flat, ignore_index=PAD_INDEX)

    if T.dim() != 2 or int(T.size(0)) != int(tool_logits.size(0)):
        return None
    valid = (T >= 0) & (T < C)
    has_row = valid.any(-1)
    if not bool(has_row.any()):
        return None

    tgt = tool_logits.gather(-1, T.clamp(min=0, max=max(C - 1, 0)))
    tgt = tgt.masked_fill(~valid, float("-inf"))
    # Rows with no target would make logsumexp(-inf, ...) = -inf and produce
    # NaN in the subtraction; zeroing those slots keeps the row finite and the
    # ``has_row`` factor below excludes it from the mean contribution.
    num = torch.logsumexp(tgt.masked_fill(~has_row.unsqueeze(-1), 0.0), dim=-1)
    den = torch.logsumexp(tool_logits, dim=-1)
    per_row = den - num
    return (per_row * has_row).sum() / has_row.sum().clamp(min=1)


def compute_lean_pointer_loss(
    candidate_logits: torch.Tensor,
    positive_indices: torch.Tensor,
) -> torch.Tensor:
    """Multi-positive softmax cross-entropy over candidate span logits.

    Args:
        candidate_logits: (B, K, C) span logits M(k, c) for c in 0..C-1.
            Invalid/padded candidates should already be masked to -1e4.
        positive_indices: (B, K, P) long tensor with indices in [0, C).
            Slots with value < 0 or >= C are ignored.

    Returns:
        Per-(b, k) loss tensor of shape (B, K). Slots where n_pos == 0
        contribute 0.
    """
    z = candidate_logits
    B, K, C = z.shape
    if K == 0 or C == 0:
        return z.new_zeros((B, K))

    P = positive_indices
    valid = (P >= 0) & (P < C)  # (B, K, P)
    safe = P.clamp(min=0, max=max(C - 1, 0))

    pos_logits = torch.gather(z, -1, safe)  # (B, K, P)
    n_pos = valid.sum(-1)  # (B, K)
    has_pos = (n_pos > 0).unsqueeze(-1)

    # For positions that have positives, mask invalid positive slots to -inf.
    # If a row has no positives at all, keep zeros to avoid NaN in logsumexp.
    pos_logits = torch.where(
        valid,
        pos_logits,
        torch.where(
            has_pos,
            torch.full_like(pos_logits, float("-inf")),
            torch.zeros_like(pos_logits),
        ),
    )

    numerator = torch.logsumexp(pos_logits, dim=-1)   # (B, K)
    denominator = torch.logsumexp(z, dim=-1)           # (B, K)
    losses = denominator - numerator                   # (B, K), >= 0

    # Zero out rows with no positive groundings
    losses = torch.where(n_pos > 0, losses, torch.zeros_like(losses))
    return losses


class LeanLossContainer(nn.Module):
    """Loss module for the Lean flow: H1 Tool Head + H3 Pointer Head."""

    def __init__(
        self,
        lambda_pointer: float = 1.0,
        lambda_tool: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_pointer = lambda_pointer
        self.lambda_tool = lambda_tool

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute the weighted joint loss.

        Args:
            predictions: dict with:
                - "span_logits": (B, K, C)
                - "tool_logits": optional (B, num_tools)
            targets: dict with:
                - "pointer_positives": (B, K, P) long, 0-indexed into 0..C-1
                - "pointer_mask": optional (B, K) float mask
                - "tool_targets": optional (B,) long

        Returns:
            dict containing "loss" and detached scalar components.
        """
        out: Dict[str, Any] = {}
        total_loss = torch.tensor(0.0, device=predictions["span_logits"].device if "span_logits" in predictions else "cpu")

        # 1. Pointer loss (H3)
        if "span_logits" in predictions and "pointer_positives" in targets:
            span_logits = predictions["span_logits"]
            positives = targets["pointer_positives"]
            ptr_per_param = compute_lean_pointer_loss(span_logits, positives)

            if "pointer_mask" in targets:
                ptr_per_param = ptr_per_param * targets["pointer_mask"].float()
                n_active = targets["pointer_mask"].sum().clamp(min=1.0)
            else:
                n_pos = (positives >= 0) & (positives < span_logits.size(-1))
                n_active = (n_pos.sum(-1) > 0).sum().clamp(min=1.0)

            ptr_loss = ptr_per_param.sum() / n_active
            out["pointer"] = ptr_loss.detach()
            total_loss = total_loss + self.lambda_pointer * ptr_loss

        # 2. Tool loss (H1)
        #
        # ``call_tool_targets`` (B, n_calls) is the R0.4 multi-call form and
        # takes precedence when present; ``tool_targets`` (B,) is the legacy
        # single-tool form and is what every pre-existing caller still passes.
        if "tool_logits" in predictions:
            tool_logits = predictions["tool_logits"]
            tool_targets = targets.get("call_tool_targets")
            if tool_targets is None:
                tool_targets = targets.get("tool_targets")
            # PAD_INDEX (-1) is the "no supervised tool" sentinel (rows whose H1
            # anchors the collator padded out, or rows whose GT tool failed to
            # render). ``compute_lean_tool_loss`` masks it; when EVERY row is
            # masked it returns None and the term is skipped rather than added
            # as a zero -- a batch with no H1 supervision must not report a
            # perfect tool loss.
            if tool_targets is not None:
                t_loss = compute_lean_tool_loss(tool_logits, tool_targets)
                if t_loss is not None:
                    out["tool"] = t_loss.detach()
                    total_loss = total_loss + self.lambda_tool * t_loss
                else:
                    out["tool"] = torch.zeros((), device=tool_logits.device)

        out["loss"] = total_loss
        return out

