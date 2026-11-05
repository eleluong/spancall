"""Assembled SpanCall model: encoder + multi-task heads (plan §3.2, D4).

``SpanCallModel`` is the production container consumed by the trainer and the
inference stack. It wraps the encoder under ``self.backbone`` so
:func:`spancall.train.trainer.build_parameter_groups` can discover the layer
stack for layer-wise LR decay, and it exposes per-head attributes used by
``SpanCallTrainer.train_step`` exactly as named (plan §3.4.2):

    pointer_head      H3 — boundary-factorized span pointer
    abstain_head      H4 — abstain-reason classes (5)
    call_accept_head  whole-call binary accept (always part of the v1 minimal set)
    tool_head         H1 — tool scoring (only with ``HeadsConfig.tool``)
    call_count_head   H2 — call-count (spec-only; ``HeadsConfig.count``)
    correctness_head  H8 — learned confidence (stop-grad; ``HeadsConfig.correctness``)

Heads are constructed from :class:`spancall.train.config.HeadsConfig` (v1
minimal set: H1 + H3 + H4 + H8 + call-accept; H2/H5/H6/H7 only when their flag
is on — docs/remediation_plan.md D4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch.nn as nn

from spancall.model.encoder import BaseEncoder
from spancall.model.heads import (
    AbstainHead,
    CallAcceptHead,
    CallCountHead,
    CorrectnessHead,
    PointerHead,
    ToolHead,
)

if TYPE_CHECKING:  # pragma: no cover
    from spancall.train.config import HeadsConfig

__all__ = ["SpanCallModel"]


class SpanCallModel(nn.Module):
    """Encoder + prediction-head container (trainer and inference entry point)."""

    def __init__(
        self,
        encoder: BaseEncoder,
        heads: Optional["HeadsConfig"] = None,
        head_dim: int = 256,
        max_span_width: int = 16,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if heads is None:
            # local import: keeps the model package independent of train.config
            from spancall.train.config import HeadsConfig

            heads = HeadsConfig()
        self.heads_cfg = heads
        self.hidden_dim = hidden_dim or int(getattr(encoder, "hidden_dim", 0) or 0)
        if not self.hidden_dim:
            raise ValueError(
                "SpanCallModel requires a positive hidden_dim (provide `encoder` "
                "with a `.hidden_dim` attribute or pass `hidden_dim=`)."
            )
        self.head_dim = head_dim
        self.max_span_width = max_span_width

        # Encoder lives under `backbone`: build_parameter_groups matches the
        # "backbone.*" prefix for layer-wise decay and everything else becomes
        # a head-group parameter.
        self.backbone = encoder

        if heads.tool:
            self.tool_head = ToolHead(self.hidden_dim)  # H1
        if heads.count:
            self.call_count_head = CallCountHead(self.hidden_dim)  # H2 (spec-only)
        if heads.pointer:
            self.pointer_head = PointerHead(  # H3
                self.hidden_dim, head_dim=head_dim, max_span_width=max_span_width
            )
        if heads.abstain_reason:
            self.abstain_head = AbstainHead(self.hidden_dim)  # H4
        self.call_accept_head = CallAcceptHead(self.hidden_dim)
        if heads.correctness:
            self.correctness_head = CorrectnessHead(  # H8
                self.hidden_dim, head_dim=head_dim
            )