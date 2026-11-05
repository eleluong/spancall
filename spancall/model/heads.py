"""Prediction heads for SpanCall.

Ref: docs/heads.md §3, docs/mechanism.md §5; full_implementation_plan §3.2.2–3.2.5.
Covers:
- H1: Tool Head (scored per ``<|tool|>``/``<|none|>`` anchor; shared 2-layer MLP)
- H2: Call-Count Head (0..max_calls)
- H3: Pointer Head (boundary-factorized span scorer + width bias + abstain logit)
- H4: Abstain-Reason Head (5 classes: OK/ABSENT/AMBIGUOUS/NEEDS_REASONING/OFF_WINDOW)
- Call-Accept Head (whole-call binary gate from ``<|cls|>``)
- H8: Correctness Head (learned confidence P(top-1 matches GT); stop-grad inputs)

All heads subclass ``torch.nn.Module`` while preserving the existing ``BaseHead``
ABC contract (plan §3.2: "All model modules inherit from torch.nn.Module").

Notation (heads.md §3): B = batch, L = sequence length, d = encoder hidden
size (read from the backbone config, e.g. 768 for ModernBERT-base), d_h = head
width (256 at d=768), K = params per tool, T = packed tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseHead(ABC):
    """Abstract base class for all prediction heads."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# H1 — Tool Head
# --------------------------------------------------------------------------- #
class ToolHead(BaseHead, nn.Module):
    """H1 — Scores candidate tools over ``<|tool|>`` token embeddings.

    A shared 2-layer MLP over ``[H_tau_t; H_CLS]`` emits one logit per anchor
    (t = 1..T) and, when present, the ``<|none|>`` anchor — i.e. a softmax over
    ``{t1..tT, NONE}`` (heads.md §H1).

    When ``pack_tools == 1`` this head is a no-op: it is not called by the
    trainer (docs/heads.md §2, D5). The logit for each anchor:
    u_t = w_tool^T GELU(W [H_tau_t; H_CLS]).
    """

    def __init__(self, hidden_dim: int) -> None:
        # BaseHead is a pure ABC (no __init__); nn.Module.__init__ runs via the MRO.
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tool_token_embeddings: torch.Tensor,
        cls_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-anchor logits.

        Args:
            tool_token_embeddings: H at ``<|tool|>``/``<|none|>`` anchors, (B, T, d).
            cls_embedding: H at ``<|cls|>``, (B, d) or (B, 1, d). When None the
                mean pool over anchors is used as a deterministic fallback.
        Returns:
            (B, T) logits.
        """
        B, T, _ = tool_token_embeddings.shape
        if cls_embedding is None:
            cls = tool_token_embeddings.mean(dim=1, keepdim=True)
        elif cls_embedding.dim() == 2:
            cls = cls_embedding.unsqueeze(1)
        else:
            cls = cls_embedding
        cls = cls.expand(B, T, -1)
        inp = torch.cat([tool_token_embeddings, cls], dim=-1)  # (B, T, 2d)
        return self.mlp(inp).squeeze(-1)  # (B, T)


# --------------------------------------------------------------------------- #
# H2 — Call-Count Head
# --------------------------------------------------------------------------- #
class CallCountHead(BaseHead, nn.Module):
    """H2 — Predicts the number of tool calls needed (0, 1, or multi-call).

    Spec-only head by default (HeadsConfig.count = False). Clause
    ``n > J`` or ``P_count`` low ⇒ abstain (heads.md §H2).
    """

    def __init__(self, hidden_dim: int, max_calls: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_calls = max_calls
        self.proj = nn.Linear(hidden_dim, max_calls + 1)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """Return logits over count classes {0, 1, ..., max_calls}, (B, max_calls+1)."""
        return self.proj(cls_embedding)


# --------------------------------------------------------------------------- #
# H3 — Pointer Head
# --------------------------------------------------------------------------- #
class PointerHead(BaseHead, nn.Module):
    """H3 — Boundary-factorized pointer scoring per parameter.

    Ref: docs/mechanism.md §5, heads.md §H3, plan §3.2.2.

        q_k      = W_q LayerNorm(H_pi_k)          in R^{d_h}
        K_s      = H W_s^T,  K_e = H W_e^T        in R^{B x L x d_h}
        s_{k,i}  = q_k^T K_{s,i} / sqrt(d_h)
        e_{k,j}  = q_k^T K_{e,j} / sqrt(d_h)
        M(k,c)   = s_{k,i} + e_{k,j} + r(j - i)   with r from width_bias,
                                                  bin min(j-i, max_span_width)
        M(k,∅)   = w_∅^T [H_CLS; H_pi_k] + b_∅    (abstain logit)

    ``forward`` returns the full boundary tensors; the trainer/loss gathers
    per-parameter candidate scores via :meth:`score_candidates` (or directly
    from ``start_logits``/``end_logits`` + ``self.width_bias``).
    """

    def __init__(self, hidden_dim: int, head_dim: int = 256, max_span_width: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.max_span_width = max_span_width

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, head_dim)     # W_q: d -> d_h
        self.start_proj = nn.Linear(hidden_dim, head_dim)     # W_s: d -> d_h
        self.end_proj = nn.Linear(hidden_dim, head_dim)       # W_e: d -> d_h
        self.abstain_proj = nn.Linear(2 * hidden_dim, 1)      # w_∅, b_∅ over [CLS; π]
        self.width_bias = nn.Embedding(max_span_width + 1, 1)  # r(j-i), bin'd
        self.scale = head_dim ** 0.5

    def forward(
        self,
        param_token_embeddings: torch.Tensor,
        sequence_embeddings: torch.Tensor,
        candidate_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute boundary scores over the full sequence.

        Args:
            param_token_embeddings: H at ``<|param|>`` anchors π_k, (B, K, d).
            sequence_embeddings: encoder output H, (B, L, d).
            candidate_masks: optional dict with ``"starts"``/``"ends"``
                (B, K, C) long tensors (padded with -1); when provided the
                result also carries gathered ``"span_logits"`` (B, K, C).

        Returns:
            dict with ``q`` (B,K,d_h), ``start_logits`` (B,K,L),
            ``end_logits`` (B,K,L), ``abstain_logits`` (B,K), and, when
            ``candidate_masks`` given, ``span_logits`` (B,K,C).
        """
        q = self.query_proj(self.query_norm(param_token_embeddings))          # (B, K, d_h)
        K_s = self.start_proj(sequence_embeddings)                            # (B, L, d_h)
        K_e = self.end_proj(sequence_embeddings)                              # (B, L, d_h)
        start_logits = torch.matmul(q, K_s.transpose(-1, -2)) / self.scale    # (B, K, L)
        end_logits = torch.matmul(q, K_e.transpose(-1, -2)) / self.scale      # (B, K, L)

        # M(k, ∅) = w_∅^T [H_CLS; H_π_k] + b_∅
        B, K, _ = param_token_embeddings.shape
        cls = sequence_embeddings[:, :1, :].expand(B, K, -1)
        abstain_logits = (
            self.abstain_proj(torch.cat([cls, param_token_embeddings], dim=-1))
            .squeeze(-1)
        )  # (B, K)

        out: Dict[str, torch.Tensor] = {
            "q": q,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "abstain_logits": abstain_logits,
        }
        if candidate_masks is not None:
            out["span_logits"] = self.score_candidates(
                start_logits, end_logits,
                candidate_masks["starts"], candidate_masks["ends"],
            )
        return out

    def score_candidates(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        cand_starts: torch.Tensor,
        cand_ends: torch.Tensor,
    ) -> torch.Tensor:
        """Gather span logits M(k, c) = s_{k,i} + e_{k,j} + r(min(j - i, W_max)).

        Args:
            start_logits / end_logits: (B, K, L).
            cand_starts / cand_ends: (B, K, C) long in [0, L), padded with -1.

        Returns:
            (B, K, C) span logits; padded slots are set to a large negative.
        """
        safe_s = cand_starts.clamp(min=0)
        safe_e = cand_ends.clamp(min=0)
        s_ij = torch.gather(start_logits, -1, safe_s)  # (B, K, C)
        e_ij = torch.gather(end_logits, -1, safe_e)    # (B, K, C)
        width = (cand_ends - cand_starts).clamp(min=0, max=self.max_span_width)
        r = self.width_bias(width).squeeze(-1)         # (B, K, C)
        valid = (cand_starts >= 0) & (cand_ends >= 0)
        scores = s_ij + e_ij + r
        return torch.where(valid, scores, torch.full_like(scores, -1e4))


# --------------------------------------------------------------------------- #
# H4 — Abstain-Reason Head
# --------------------------------------------------------------------------- #
class AbstainHead(BaseHead, nn.Module):
    """H4 — Classifies the reason for parameter or call abstention.

    5 classes (plan §3.2.3): OK, ABSENT, AMBIGUOUS, NEEDS_REASONING, OFF_WINDOW.
    A 2-layer MLP over ``[H_CLS; H_π_k; H̄_utt]`` in R^{3d} (heads.md §H4).

    The H3–H4 twin is enforced by an auxiliary consistency loss in
    ``SpanCallLossContainer`` (MSE between M(k,∅) and logit(¬OK)), not here.
    """

    REASONS: tuple = ("OK", "ABSENT", "AMBIGUOUS", "NEEDS_REASONING", "OFF_WINDOW")

    def __init__(self, hidden_dim: int, num_reasons: int = 5) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasons = num_reasons
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_reasons),
        )

    def forward(
        self,
        param_embeddings: torch.Tensor,
        cls_embedding: torch.Tensor,
        utterance_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return reason logits per parameter.

        Args:
            param_embeddings: H at ``<|param|>`` anchors π_k, (B, K, d).
            cls_embedding: H at ``<|cls|>``, (B, d) or (B, 1, d).
            utterance_embedding: H̄_utt (mean pool over utterance tokens), (B, d)
                or (B, 1, d). When None, the mean over params is used (only a
                deterministic fallback for tests; the trainer always supplies
                the real utterance pool).

        Returns:
            (B, K, num_reasons) logits.
        """
        B, K, _ = param_embeddings.shape

        def _as_bk(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 2:
                return t.unsqueeze(1).expand(B, K, -1)
            return t.expand(B, K, -1)

        cls = _as_bk(cls_embedding)
        if utterance_embedding is None:
            utt = param_embeddings.mean(dim=1, keepdim=True).expand(B, K, -1)
        else:
            utt = _as_bk(utterance_embedding)
        inp = torch.cat([cls, param_embeddings, utt], dim=-1)  # (B, K, 3d)
        return self.mlp(inp)


# --------------------------------------------------------------------------- #
# Call-Accept Head
# --------------------------------------------------------------------------- #
class CallAcceptHead(BaseHead, nn.Module):
    """Whole-call accept head predicting whether call is fully groundable (stratum F).

    ``nn.Sequential(nn.Linear(d, d_h), GELU, nn.Linear(d_h, 1))`` on H_CLS
    (plan §3.2.4). Trained with BCE: target = 1[stratum == F].
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # d_h defaults to d here
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """Return a single binary acceptance logit, (B, 1)."""
        return self.mlp(cls_embedding)


# --------------------------------------------------------------------------- #
# H8 — Correctness Head (learned confidence)
# --------------------------------------------------------------------------- #
class CorrectnessHead(BaseHead, nn.Module):
    """H8 — P(top-1 span exactly matches GT); learned confidence.

    Per (t, k, j) input (plan §3.2.5, heads.md §H8)::

        [h̄_a*; q_k; v_stats] ∈ R^{2 d_h + 8}

    where ``h̄_a* = W_span LayerNorm(mean_{t in a*} H_t.detach())`` uses the
    stop-gradient pooled representation of H3's top-1 span a*, ``q_k`` is
    ``q_k.detach()``, and ``v_stats`` holds 8 fixed features:
    (p_top1, p_top2, margin, entropy, p_abstain, p_ok, span_width_norm, is_ctx).

    Stop-grad is applied *inside* ``forward`` so H8 never moves the encoder or
    H3 (plan §3.2.5: "gradients stopped into the encoder and H3"); its own
    parameters still train via the correctness BCE.
    """

    NUM_STATS: int = 8

    def __init__(self, hidden_dim: int, head_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.span_norm = nn.LayerNorm(hidden_dim)
        self.span_proj = nn.Linear(hidden_dim, head_dim)  # W_span: d -> d_h
        self.mlp = nn.Sequential(
            nn.Linear(head_dim * 2 + self.NUM_STATS, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self,
        span_pooled: torch.Tensor,
        query: torch.Tensor,
        v_stats: torch.Tensor,
    ) -> torch.Tensor:
        """Return P(top-1 matches GT) logit, (B, K, 1).

        Args:
            span_pooled: mean-pooled encoder state over the chosen span a*,
                (B, K, d). Detached inside.
            query: H3 query q_k, (B, K, d_h). Detached inside.
            v_stats: fixed statistics vector, (B, K, 8).
        """
        hbar = self.span_proj(self.span_norm(span_pooled.detach()))  # (B, K, d_h)
        q = query.detach()
        v = v_stats.float()
        inp = torch.cat([hbar, q, v], dim=-1)  # (B, K, 2 d_h + 8)
        return self.mlp(inp)  # (B, K, 1)
