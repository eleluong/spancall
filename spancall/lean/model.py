"""Lean SpanCall Model without abstain machinery.

Provides:
- LeanPointerHead: H3 span scorer over (B, K, C) candidate spans (no abstain logit/projection).
- LeanSpanCallModel: Encoder + H1 ToolHead + LeanPointerHead (no H4 abstain head, no accept head, no correctness head).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from spancall.model.encoder import BaseEncoder
from spancall.model.heads import BaseHead, ToolHead


class LeanPointerHead(BaseHead, nn.Module):
    """Lean H3 — Boundary-factorized span pointer strictly over candidate spans.

    Computes:
        q_k      = W_q LayerNorm(H_pi_k)          in R^{d_h}
        K_s      = H W_s^T,  K_e = H W_e^T        in R^{B x L x d_h}
        s_{k,i}  = q_k^T K_{s,i} / sqrt(d_h)
        e_{k,j}  = q_k^T K_{e,j} / sqrt(d_h)
        M(k,c)   = s_{k,i} + e_{k,j} + r(min(j-i, max_span_width))

    Unlike the legacy PointerHead, this head has NO abstain projection w_∅.
    Logits are strictly (B, K, C) over valid candidates.
    """

    def __init__(self, hidden_dim: int, head_dim: int = 256, max_span_width: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.max_span_width = max_span_width

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, head_dim)      # W_q: d -> d_h
        self.start_proj = nn.Linear(hidden_dim, head_dim)      # W_s: d -> d_h
        self.end_proj = nn.Linear(hidden_dim, head_dim)        # W_e: d -> d_h
        self.width_bias = nn.Embedding(max_span_width + 1, 1)  # r(j-i), bin'd
        self.scale = head_dim ** 0.5

    def forward(
        self,
        param_token_embeddings: torch.Tensor,
        sequence_embeddings: torch.Tensor,
        candidate_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute boundary scores and gather candidate span logits.

        Args:
            param_token_embeddings: H at <|param|> anchors, (B, K, d).
            sequence_embeddings: encoder output H, (B, L, d).
            candidate_masks: dict with "starts"/"ends" (B, K, C) long tensors
                padded with -1.

        Returns:
            dict with:
                "q": (B, K, d_h)
                "start_logits": (B, K, L)
                "end_logits": (B, K, L)
                "span_logits": (B, K, C) when candidate_masks is provided
        """
        q = self.query_proj(self.query_norm(param_token_embeddings))
        K_s = self.start_proj(sequence_embeddings)
        K_e = self.end_proj(sequence_embeddings)
        start_logits = torch.matmul(q, K_s.transpose(-1, -2)) / self.scale
        end_logits = torch.matmul(q, K_e.transpose(-1, -2)) / self.scale

        out: Dict[str, torch.Tensor] = {
            "q": q,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        if candidate_masks is not None:
            out["span_logits"] = self.score_candidates(
                start_logits,
                end_logits,
                candidate_masks["starts"],
                candidate_masks["ends"],
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
            (B, K, C) span logits; invalid/padded slots are set to -1e4.
        """
        B, K, L = start_logits.shape
        if cand_starts.size(-1) == 0 or L == 0:
            return start_logits.new_empty(cand_starts.shape)
        max_l = max(L - 1, 0)
        safe_s = cand_starts.clamp(min=0, max=max_l)
        safe_e = cand_ends.clamp(min=0, max=max_l)
        s_ij = torch.gather(start_logits, -1, safe_s)
        e_ij = torch.gather(end_logits, -1, safe_e)
        width = (cand_ends - cand_starts).clamp(min=0, max=self.max_span_width)
        r = self.width_bias(width).squeeze(-1)
        valid = (cand_starts >= 0) & (cand_ends >= 0) & (cand_starts < L) & (cand_ends < L)
        scores = s_ij + e_ij + r
        return torch.where(valid, scores, torch.full_like(scores, -1e4))


class LeanSpanCallModel(nn.Module):
    """Lean SpanCall model container: Encoder + H1 ToolHead + LeanPointerHead."""

    def __init__(
        self,
        encoder: BaseEncoder,
        head_dim: int = 256,
        max_span_width: int = 16,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim or int(getattr(encoder, "hidden_dim", 0) or 0)
        if not self.hidden_dim:
            raise ValueError(
                "LeanSpanCallModel requires a positive hidden_dim (provide `encoder` "
                "with a `.hidden_dim` attribute or pass `hidden_dim=`)."
            )
        self.head_dim = head_dim
        self.max_span_width = max_span_width

        # Encoder lives under backbone for layer-wise decay discovery
        self.backbone = encoder
        self.tool_head = ToolHead(self.hidden_dim)
        self.pointer_head = LeanPointerHead(
            self.hidden_dim, head_dim=head_dim, max_span_width=max_span_width
        )

