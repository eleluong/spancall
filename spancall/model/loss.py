"""Loss functions for SpanCall multi-task optimization.

Ref: docs/mechanism.md §8.2, heads.md §4; full_implementation_plan §3.3.

``SpanCallLossContainer`` computes the joint multi-task objective:

1. Multi-positive pointer loss (plan §3.3.1)
   For parameter k with z_k = [M(k,∅), M(k,c_1), ..., M(k,c_{|A_k|})] (abstain
   first, length |A_k|+1) and positive set P_k ⊆ {0..|A_k|}:
       L_k = -(logsumexp_{c∈P_k} z_{k,c} - logsumexp(z_k))
   Abstention rows (P_k == {0}) are up-weighted by ``abstain_weight`` (1.5).
2. Call-accept BCE  (BCEWithLogitsLoss over z_accept vs I[stratum == F]).
3. Abstain-reason CE over 5 classes with weights (0.25,1,1,1,1) plus
   0.1·MSE(logit(¬OK), M(k,∅)) for the H3–H4 consistency tie, where
       logit(¬OK) = logsumexp(z_reason[¬OK]) − z_reason[OK].
4. Tool-head CE (H1) — only when tool logits are provided (pack_tools > 1).
5. Correctness BCE (H8, on-the-fly target):
       y_corr,k = I[argmax_c z_{k,c} ∈ P_k]
   computed with stop-grad (``try_`` detached); the head itself detaches its
   span/query inputs so H8 never moves the encoder or H3.

Total: L = λ_ptr·L_ptr + λ_accept·L_accept + λ_reason·(CE + 0.1·MSE)
            + λ_tool·L_tool + λ_corr·L_corr
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_ABSTAIN_INDEX = 0  # column of M(k,∅) in the concatenated pointer logits


def _logit_not_ok(reason_logits: torch.Tensor) -> torch.Tensor:
    """logit(¬OK) = logsumexp(z_reason[1:]) − z_reason[0]  (H3–H4 consistency)."""
    l_ok = reason_logits[..., _ABSTAIN_INDEX]
    l_not_ok = torch.logsumexp(reason_logits[..., 1:], dim=-1)
    return l_not_ok - l_ok


class SpanCallLossContainer:
    """Computes joint multi-task loss across tool, count, pointer, accept, reason, and correctness heads."""

    def __init__(
        self,
        lambda_tool: float = 1.0,
        lambda_count: float = 0.3,
        lambda_pointer: float = 1.0,
        lambda_accept: float = 0.5,
        lambda_reason: float = 0.5,
        lambda_corr: float = 1.0,
        abstain_weight: float = 1.5,
        reason_class_weights: Sequence[float] = (0.25, 1.0, 1.0, 1.0, 1.0),
        reason_mse_weight: float = 0.1,
    ) -> None:
        self.lambda_tool = lambda_tool
        self.lambda_count = lambda_count
        self.lambda_pointer = lambda_pointer
        self.lambda_accept = lambda_accept
        self.lambda_reason = lambda_reason
        self.lambda_corr = lambda_corr
        self.abstain_weight = abstain_weight
        self.reason_mse_weight = reason_mse_weight
        self._reason_weights: Tuple[float, ...] = tuple(float(w) for w in reason_class_weights)

        self._reason_ce = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(list(reason_class_weights), dtype=torch.float32),
            reduction="none",
        )
        self._bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    # ------------------------------------------------------------------ #
    # Pointer loss
    # ------------------------------------------------------------------ #
    def compute_pointer_loss(
        self,
        candidate_logits: torch.Tensor,
        positive_indices: torch.Tensor,
        abstain_index: int = _ABSTAIN_INDEX,
    ) -> torch.Tensor:
        """Multi-positive softmax cross-entropy (R4: equal-valued targets sum mass).

        Args:
            candidate_logits: (B, K, C) the *full* logit vector z over
                {abstain} ∪ A_k with the abstain logit at column
                ``abstain_index`` (default 0; the trainer prepends
                ``abstain_logits`` to ``span_logits``).
            positive_indices: (B, K, P) long, values in [0, C); each value is
                an index into ``candidate_logits`` (0 = abstain, v>0 = candidate
                v-1). Rows/params masked out of the loss are fully ``-1``.
            abstain_index: column of the abstain logit (must match the position
                used when building ``candidate_logits``).

        Returns:
            Per-(b,k) losses, (B, K); abstention rows (P_k == {0}) up-weighted
            by ``abstain_weight``; fully-masked rows contribute 0.
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
        pos_logits = torch.where(
            valid,
            pos_logits,
            torch.where(
                has_pos,
                torch.full_like(pos_logits, float("-inf")),
                torch.zeros_like(pos_logits),
            ),
        )

        numerator = torch.logsumexp(pos_logits, dim=-1)  # (B, K)
        denominator = torch.logsumexp(z, dim=-1)          # (B, K)
        losses = denominator - numerator                  # (B, K), >= 0

        # abstain rows: valid positives are exactly {abstain_index}
        valid_abstain = valid & (safe == abstain_index)   # (B, K, P)
        is_abstain = (n_pos > 0) & (valid_abstain.sum(-1) == n_pos)  # (B, K)
        weight = torch.where(
            is_abstain,
            torch.full_like(losses, self.abstain_weight),
            torch.ones_like(losses),
        )
        losses = losses * weight
        # fully-masked rows (n_pos == 0) → no gradient
        losses = torch.where(n_pos > 0, losses, torch.zeros_like(losses))
        return losses

    # ------------------------------------------------------------------ #
    # Aggregate
    # ------------------------------------------------------------------ #
    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate the joint weighted loss.

        Expected ``predictions`` keys (only those present contribute):
            pointer_logits: (B, K, C) full z (abstain prepended at col 0), or
                ``span_logits`` (B,K,C) + ``abstain_logits`` (B,K) which are
                combined here.
            accept_logits: (B,) or (B,1).
            reason_logits: (B, K, 5).
            tool_logits: (B, T) — included only when pack_tools > 1.
            corr_logits: (B, K, 1) — included only when the correctness head
                is enabled (stage C).

        Expected ``targets`` keys:
            pointer_positives: (B, K, P) long (see compute_pointer_loss).
            accept_targets: (B,) float.
            reason_targets: (B, K) long; masked entries may be -1.
            reason_mask: optional (B, K) float (default = all valid).
            tool_targets: (B,) long — required iff tool_logits present.
            corr_targets: optional (B, K) float; else computed on the fly.

        Returns dict with "loss" plus per-component scalars.
        """
        out: Dict[str, Any] = {}

        # ---- pointer (H3) -------------------------------------------------
        abstain_logits: Optional[torch.Tensor] = None
        if "pointer_logits" in predictions:
            z = predictions["pointer_logits"]
            abstain_logits = z[..., _ABSTAIN_INDEX]  # M(k,∅), first column
        elif "span_logits" in predictions and "abstain_logits" in predictions:
            abstain_logits = predictions["abstain_logits"]
            z = torch.cat(
                [abstain_logits.unsqueeze(-1), predictions["span_logits"]],
                dim=-1,
            )
        else:
            z = None

        ptr_loss: Optional[torch.Tensor] = None
        if z is not None and "pointer_positives" in targets:
            ptr = self.compute_pointer_loss(z, targets["pointer_positives"])
            if "pointer_mask" in targets:  # optional (B,K) float
                ptr = ptr * targets["pointer_mask"]
            n = ptr.numel() if ptr.numel() else 1
            ptr_loss = ptr.sum() / max(int(n), 1)
            out["pointer"] = ptr_loss.detach()

        # ---- call accept --------------------------------------------------
        accept_loss: Optional[torch.Tensor] = None
        if "accept_logits" in predictions and "accept_targets" in targets:
            _a = predictions["accept_logits"].reshape(-1, 1)
            _t = targets["accept_targets"].reshape(-1, 1).float()
            accept_loss = self._bce(_a, _t).mean()
            out["accept"] = accept_loss.detach()

        # ---- abstain reason (H4) + H3–H4 consistency ----------------------
        reason_loss: Optional[torch.Tensor] = None
        if "reason_logits" in targets and "reason_logits" in predictions:
            rlogits = predictions["reason_logits"]           # (B, K, 5)
            B, K, _ = rlogits.shape
            rt = targets["reason_logits"]                    # (B, K) long
            mask = targets.get("reason_mask")
            if mask is None:
                mask = torch.ones(B, K, dtype=torch.float32, device=rlogits.device)
            else:
                mask = mask.float()

            r_valid = (rt >= 0) & (rt < rlogits.size(-1)) & mask.bool()
            weight = torch.tensor(
                self._reason_weights, dtype=rlogits.dtype, device=rlogits.device
            )
            ce = F.cross_entropy(
                rlogits.reshape(-1, rlogits.size(-1)),
                rt.reshape(-1).clamp(min=0, max=max(rlogits.size(-1) - 1, 0)),
                weight=weight,
                reduction="none",
            ).reshape(B, K)
            ce = torch.where(r_valid, ce, torch.zeros_like(ce))
            n_reason = max(int(r_valid.sum().item()), 1)
            reason_loss = ce.sum() / n_reason

            # H3–H4 consistency: MSE(logit(¬OK), M(k,∅))
            if abstain_logits is not None:
                mse = F.mse_loss(
                    _logit_not_ok(rlogits), abstain_logits, reduction="none"
                )
                mse = torch.where(r_valid, mse, torch.zeros_like(mse))
                reason_loss = (
                    reason_loss + self.reason_mse_weight * (mse.sum() / n_reason)
                )
            out["reason"] = reason_loss.detach()

        # ---- tool (H1) ----------------------------------------------------
        tool_loss: Optional[torch.Tensor] = None
        if "tool_logits" in predictions and "tool_targets" in targets:
            tt = targets["tool_targets"].long()
            if (tt >= 0).any():
                tool_loss = F.cross_entropy(
                    predictions["tool_logits"], tt, ignore_index=-1
                )
                out["tool"] = tool_loss.detach()

        # ---- correctness (H8) ---------------------------------------------
        corr_loss: Optional[torch.Tensor] = None
        if "corr_logits" in predictions and z is not None and "pointer_positives" in targets:
            # on-the-fly target, detached (stop-grad into encoder/H3 already
            # enforced inside CorrectnessHead).
            z_det = z.detach()
            am = z_det.argmax(dim=-1)  # (B, K)
            pos = targets["pointer_positives"]
            valid_pos = (pos >= 0) & (pos < z.size(-1))
            member = ((pos == am.unsqueeze(-1)) & valid_pos).any(dim=-1)  # (B, K)
            n_pos = valid_pos.sum(-1) > 0
            y_corr = (member & n_pos).float()

            clogits = predictions["corr_logits"].reshape(-1, 1)
            y = y_corr.reshape(-1, 1)
            cb = self._bce(clogits, y).reshape(-1)
            cb = cb * n_pos.reshape(-1).float()
            n_corr = max(int(n_pos.sum().item()), 1)
            corr_loss = cb.sum() / n_corr
            out["corr"] = corr_loss.detach()

        # ---- count (H2, spec-only) ----------------------------------------
        count_loss: Optional[torch.Tensor] = None
        if "count_logits" in predictions and "count_targets" in targets:
            count_loss = F.cross_entropy(
                predictions["count_logits"], targets["count_targets"].long()
            )
            out["count"] = count_loss.detach()

        # ---- total --------------------------------------------------------
        active_losses = [
            t for t in (ptr_loss, accept_loss, reason_loss, tool_loss, corr_loss, count_loss)
            if t is not None
        ]
        dev = active_losses[0].device if active_losses else torch.device("cpu")
        total = torch.zeros((), dtype=torch.float32, device=dev)
        if ptr_loss is not None:
            total = total + self.lambda_pointer * ptr_loss
        if accept_loss is not None:
            total = total + self.lambda_accept * accept_loss
        if reason_loss is not None:
            total = total + self.lambda_reason * reason_loss
        if tool_loss is not None:
            total = total + self.lambda_tool * tool_loss
        if corr_loss is not None:
            total = total + self.lambda_corr * corr_loss
        if count_loss is not None:
            total = total + self.lambda_count * count_loss
        out["loss"] = total
        return out
