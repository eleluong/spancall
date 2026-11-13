"""Prediction heads for SpanCall.

Ref: docs/heads.md §3, docs/mechanism.md §5.
Covers:
- BaseHead: ABC contract for all prediction heads
- H1: ToolHead (softmax over ``<|tool|>``/``<|none|>`` anchors; shared 2-layer MLP)
- H1-binary: BinaryToolHead (independent per-anchor sigmoid P(this tool is called))

All heads subclass ``torch.nn.Module`` while preserving the ``BaseHead`` ABC
contract. Notation: B = batch, L = sequence length, d = encoder hidden size,
T = packed tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn


class BaseHead(ABC):
    """Abstract base class for all prediction heads."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# H1 — Tool Head (Softmax)
# --------------------------------------------------------------------------- #
class ToolHead(BaseHead, nn.Module):
    """H1 — Scores candidate tools over ``<|tool|>`` token embeddings.

    A shared 2-layer MLP over ``[H_tau_t; H_CLS]`` emits one logit per anchor
    (t = 1..T) and, when present, the ``<|none|>`` anchor — i.e. a softmax over
    ``{t1..tT, NONE}`` (heads.md §3.1).
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
# H1-binary — Independent Per-Anchor Sigmoid Tool Head
# --------------------------------------------------------------------------- #
class BinaryToolHead(BaseHead, nn.Module):
    """H1-binary — Per-anchor P(this tool is called), scored INDEPENDENTLY.

    Same inputs and output shape as :class:`ToolHead` (``(B, T)`` one logit per
    rendered ``<|tool|>`` anchor), so the two are directly comparable arms of the
    same ablation. Only the objective differs: this head is trained with
    ``binary_cross_entropy_with_logits`` against a one-hot of the called anchors
    instead of a softmax cross-entropy.
    """

    def __init__(self, hidden_dim: int) -> None:
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
        """Return per-anchor binary logits, ``(B, T)``."""
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
