"""Lean Loss Container for SpanCall without abstain machinery.

Provides:
- compute_lean_pointer_loss: multi-positive softmax cross-entropy directly over
  candidate spans (B, K, C) without abstain columns or abstain weighting.
- compute_lean_tool_loss: softmax cross-entropy (or set-marginal) over the
  rendered ``<|tool|>`` anchors -- the original H1.
- compute_lean_tool_loss_binary: per-anchor BCE, the v2.1 ablation arm. Same
  logits, same targets, different objective.
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

# The trainer masks a dead ``<|tool|>`` anchor with this logit. Restated here as
# a NAMED constant (rather than a magic number) because the binary loss uses it
# to decide which anchors are live, and that choice changes the loss scale.
DEAD_ANCHOR_LOGIT: float = -1e4
_DEAD_ANCHOR_LOGIT: float = DEAD_ANCHOR_LOGIT


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


def _lean_tool_supervision_count(
    tool_logits: torch.Tensor,
    tool_targets: torch.Tensor,
) -> torch.Tensor:
    """Number of rows that actually carry H1 supervision (the loss denominator).

    Mirrors the masking rules of :func:`compute_lean_tool_loss` exactly, so
    ``loss × count`` reconstructs the batch SUM that was averaged into ``loss``.
    Returning the count as a tensor keeps it device-local and differentiable-free.
    """
    C = int(tool_logits.size(-1)) if tool_logits.dim() == 2 else 0
    if tool_logits.dim() != 2 or C == 0 or tool_targets.numel() == 0:
        return torch.zeros((), device=tool_logits.device, dtype=torch.float32)
    if tool_targets.dim() <= 1:
        flat = tool_targets.reshape(-1)
        if int(flat.size(0)) != int(tool_logits.size(0)):
            return torch.zeros((), device=tool_logits.device, dtype=torch.float32)
        has = (flat >= 0) & (flat < C)
        return has.sum().float()
    if int(tool_targets.size(0)) != int(tool_logits.size(0)):
        return torch.zeros((), device=tool_logits.device, dtype=torch.float32)
    valid = (tool_targets >= 0) & (tool_targets < C)
    return valid.any(-1).sum().float()


def compute_lean_tool_loss_binary(
    tool_logits: torch.Tensor,
    tool_targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Binary H1 loss: each rendered anchor is scored as called / not-called.

    Consumes the SAME ``(B, T)`` logit tensor and the SAME targets as
    :func:`compute_lean_tool_loss`; only the objective changes. The one-hot
    target is scattered INSIDE this function, so no new batch key is required and
    the tensorizer is untouched.

    Accepts both target shapes, mirroring the CE path:

    - ``(B,)`` -- one correct anchor per row (legacy contract).
    - ``(B, P)`` -- up to ``P`` correct anchors per row (R0.4 multi-call). Every
      member is a positive, so a two-call row supervises two anchors at target 1
      and the rest at 0. This is the property softmax cannot express cleanly:
      a set of positives is just a one-hot vector with several ones.

    ``pos_weight`` counters the 1-positive-of-T imbalance (a good default is
    ``T - 1``).

    Dead anchors already carry ``-1e4`` from the callers and are EXCLUDED from
    both terms here. Two reasons, and the second is the important one:

    1. ``BCEWithLogits(-1e4, 0) ~ 0``, so they contribute no gradient -- which
       is why the caller needs no extra mask argument.
    2. They must not dilute the denominator either. The CE path normalises over
       live anchors only (``exp(-1e4)`` is 0 in its ``logsumexp``), so a row
       whose anchor failed to snap would otherwise carry a systematically
       SMALLER binary loss than the same row with every anchor live. Averaging
       over a padded column is exactly the kind of silent scale difference that
       makes an ablation uninterpretable.

    Returns ``None`` when no row carries supervision, mirroring the CE contract:
    callers must SKIP the term, never add a zero.
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
        valid = (flat >= 0) & (flat < C)
        has_row = valid
        # Scatter the scalar target into a one-hot (B, C).
        one_hot = torch.zeros_like(tool_logits)
        one_hot[torch.arange(int(flat.size(0)), device=flat.device)[valid], flat[valid]] = 1.0
    else:
        if T.dim() != 2 or int(T.size(0)) != int(tool_logits.size(0)):
            return None
        valid = (T >= 0) & (T < C)
        has_row = valid.any(-1)
        one_hot = torch.zeros_like(tool_logits)
        rows = torch.arange(int(T.size(0)), device=T.device).unsqueeze(-1).expand_as(T)
        # ``valid`` already excludes PAD_INDEX, so the scatter can never write to
        # a negative column; two calls naming the SAME tool are idempotent.
        one_hot[rows[valid], T[valid]] = 1.0

    # Live anchors only: the caller signals "dead" by filling the logit with
    # -1e4, so this threshold identifies exactly the anchors that carry signal.
    live = tool_logits > _DEAD_ANCHOR_LOGIT
    # A row whose every anchor is dead is NOT scoreable: it would otherwise be
    # averaged as a diluted 0.0, i.e. reported as a suspiciously good loss. It is
    # excluded exactly like an unsupervised row (``has_row`` also filters the
    # denominator), mirroring how the CE path treats ``exp(-1e4) == 0``.
    has_row = has_row & live.any(-1)
    if not bool(has_row.any()):
        return None

    per_anchor = F.binary_cross_entropy_with_logits(
        tool_logits, one_hot, reduction="none", pos_weight=pos_weight
    )
    per_anchor = per_anchor * live
    # Average over LIVE anchors inside supervised ROWS only; a row with no
    # supervised anchor (every target -1) contributes nothing to either side.
    per_row = per_anchor.sum(-1) / live.sum(-1).clamp(min=1.0)
    return (per_row * has_row).sum() / has_row.sum().clamp(min=1)


def _lean_tool_binary_supervision_count(
    tool_logits: torch.Tensor,
    tool_targets: torch.Tensor,
) -> torch.Tensor:
    """Rows with binary H1 supervision -- the micro-average denominator.

    Shares the supervision rules with the CE counter, so ``loss x count``
    reconstructs the batch SUM that was averaged, keeping the trainer's
    slot-weighted accumulation valid for both arms.
    """
    return _lean_tool_supervision_count(tool_logits, tool_targets)


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
        lambda_tool_binary: float = 1.0,
        tool_objective: str = "ce",
    ) -> None:
        super().__init__()
        self.lambda_pointer = lambda_pointer
        self.lambda_tool = lambda_tool
        # Weight for the v2.1 binary arm. Kept separate from ``lambda_tool`` so
        # an ablation can track both terms independently.
        self.lambda_tool_binary = lambda_tool_binary
        # Which H1 objective actually TRAINS: ``ce`` (default, the shipped
        # head) or ``binary`` (v2.1). This is deliberately an explicit string
        # rather than "whichever logits happen to be present", because a silent
        # fallback would make a mis-wired ablation look like a successful one.
        obj = str(tool_objective or "ce").lower()
        if obj not in ("ce", "binary"):
            raise ValueError(f"tool_objective must be 'ce' or 'binary', got {tool_objective!r}")
        self.tool_objective = obj

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

        # NOTE on the reported `*_sum` / `*_count` pairs below.
        #
        # `pointer` and `tool` are per-BATCH means (each term is divided by the
        # count inside its own batch). Averaging those means across batches is a
        # mean-of-ratios, so it depends on how rows happen to be bucketed: one
        # long 1-row batch carries the same weight as a full 8-row batch. That is
        # why the trainer's `mean_loss` and `eval_loss` were not on a common scale
        # (docs: train fell 17x while dev rose 32% purely from overfitting, but the
        # absolute gap was partly an accounting artifact).
        #
        # The raw `sum` and `count` are therefore exposed so the trainer can
        # accumulate `sum(sum) / sum(count)` -- a slot-weighted (micro) average
        # that is invariant to batch composition and identical for train and eval.
        # Gradients are unaffected: `ptr_loss` / `t_loss` still normalise per
        # batch, exactly as before.
        if "span_logits" in predictions and "pointer_positives" in targets:
            span_logits = predictions["span_logits"]
            positives = targets["pointer_positives"]
            ptr_per_param = compute_lean_pointer_loss(span_logits, positives)

            if "pointer_mask" in targets:
                ptr_per_param = ptr_per_param * targets["pointer_mask"].float()
                n_active_raw = targets["pointer_mask"].sum()
            else:
                n_pos = (positives >= 0) & (positives < span_logits.size(-1))
                n_active_raw = (n_pos.sum(-1) > 0).sum()

            # `clamp` guards the divisor only; the TRUE count is what gets
            # reported, so a batch with no supervised slots contributes 0 to both
            # the numerator and the denominator of the micro average.
            n_active = n_active_raw.clamp(min=1.0)

            ptr_loss = ptr_per_param.sum() / n_active
            out["pointer"] = ptr_loss.detach()
            out["pointer_sum"] = ptr_per_param.sum().detach()
            out["pointer_count"] = n_active_raw.detach()
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
                    # In the ``binary`` arm the CE value is a REFERENCE metric
                    # only: it is reported so the two objectives can be compared
                    # on identical data, but it must not enter the total (that
                    # would train two competing H1 objectives at once and make
                    # the ablation uninterpretable). ``tool_sum``/``tool_count``
                    # are withheld for the same reason -- they are what the
                    # trainer's micro-average reads.
                    if self.tool_objective != "binary":
                        # Same micro-average bookkeeping as the pointer term:
                        # report the per-batch SUM and the number of supervised
                        # rows so the trainer can weight batches by supervision
                        # instead of by batch count (see the note above).
                        n_tool_sup = _lean_tool_supervision_count(tool_logits, tool_targets)
                        out["tool_sum"] = t_loss.detach() * n_tool_sup
                        out["tool_count"] = n_tool_sup
                        total_loss = total_loss + self.lambda_tool * t_loss
                else:
                    out["tool"] = torch.zeros((), device=tool_logits.device)

        # 3. Binary tool loss (H1-binary, v2.1 ablation arm)
        #
        # Dispatched ONLY on the presence of ``tool_binary_logits``, which the
        # trainer emits only when its flag is on. With the flag off this block is
        # never entered, so the CE path above stays byte-identical and every
        # stored prediction / historical run remains reproducible.
        if self.tool_objective == "binary" and "tool_binary_logits" in predictions:
            bin_logits = predictions["tool_binary_logits"]
            bin_targets = targets.get("call_tool_targets")
            if bin_targets is None:
                bin_targets = targets.get("tool_targets")
            if bin_targets is not None:
                # ``T - 1`` as the positive weight: with T rendered anchors and
                # one called tool the raw BCE is dominated by the negatives, and
                # an un-weighted head learns to emit ~0 everywhere.
                n_anchor = int(bin_logits.size(-1)) if bin_logits.dim() == 2 else 1
                pos_weight = torch.full(
                    (n_anchor,), float(max(n_anchor - 1, 1)), device=bin_logits.device
                )
                b_loss = compute_lean_tool_loss_binary(
                    bin_logits, bin_targets, pos_weight=pos_weight
                )
                if b_loss is not None:
                    out["tool_binary"] = b_loss.detach()
                    n_bin_sup = _lean_tool_binary_supervision_count(bin_logits, bin_targets)
                    out["tool_binary_sum"] = b_loss.detach() * n_bin_sup
                    out["tool_binary_count"] = n_bin_sup
                    total_loss = total_loss + self.lambda_tool_binary * b_loss
                else:
                    out["tool_binary"] = torch.zeros((), device=bin_logits.device)

        out["loss"] = total_loss
        return out

