"""Encoder architectures for SpanCall.

Ref: docs/mechanism.md §4, docs/heads.md §2.
Wraps bidirectional encoder (ModernBERT, DeBERTa-v3) with structural marker tokens.

Implementation notes (plan §3.2.1):
- ``SpanCallTransformerEncoder`` subclasses both ``torch.nn.Module`` and the
  existing ``BaseEncoder`` ABC contract.
- ``backbone`` is an ``AutoModel`` transformer (default
  ``answerdotai/ModernBERT-base``, d = 768).
- ``add_marker_tokens(tokens)`` registers the given markers on the tokenizer,
  resizes the input embedding matrix, and initializes the *new* rows with the
  mean + std of the pre-existing embeddings (so marker embeddings start on
  distribution instead of at the default init).
- ``forward`` returns ``last_hidden_state`` H in R^{B x L x d} with
  ``hidden_dropout_prob`` dropout (dropout is a no-op in eval mode).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class BaseEncoder(ABC):
    """Abstract encoder backbone converting tokenized sequence into contextual representations."""

    @abstractmethod
    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Optional[Any] = None,
    ) -> Any:
        """Computes contextual token representations H in R^{B x L x d}."""
        raise NotImplementedError

    @abstractmethod
    def add_marker_tokens(self, tokens: List[str]) -> int:
        """Register custom marker tokens (<|param|>, <|ctx_time|>, etc.) and resize embeddings."""
        raise NotImplementedError


class SpanCallTransformerEncoder(BaseEncoder, nn.Module):
    """Transformer encoder wrapper supporting ModernBERT / DeBERTa / Qwen backbones.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model id or local path for ``AutoModel.from_pretrained``.
        Defaults to ``Qwen/Qwen3-Embedding-0.6B`` (hidden d = 1024, 32k window;
        long_context_fix.md Phase 2).
    hidden_dropout_prob:
        Dropout applied to the contextual representation H in ``forward``.
    tokenizer:
        Optional tokenizer used by ``add_marker_tokens``. If ``None``,
        ``add_marker_tokens`` raises until a tokenizer is attached (set
        ``self.tokenizer`` afterwards).
    backbone:
        Optional pre-built ``nn.Module`` to inject instead of running
        ``AutoModel.from_pretrained`` (used by tests / fine-grained control).
    bidirectional:
        Force bidirectional attention on decoder-derived backbones (default
        True). Decoder models (Qwen3-0.6B-Base and friends) mask causally by
        default; SpanCall resolves via <|cls|>/<|param|> anchor tokens that must
        read the *whole* sequence, so ``forward`` sweeps every attention module
        clearing ``is_causal`` (Phase 2b unmasked bidirectional attention).
        The ``Qwen3-Embedding`` backbone ships already-bidirectional; the sweep
        is a no-op there and idempotent everywhere.
    from_pretrained_kwargs:
        Extra kwargs forwarded to ``AutoModel.from_pretrained`` (e.g.
        ``torch_dtype``, ``attn_implementation``, ``local_files_only``).
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        hidden_dropout_prob: float = 0.1,
        tokenizer: Any = None,
        backbone: Optional[nn.Module] = None,
        bidirectional: bool = True,
        **from_pretrained_kwargs: Any,
    ) -> None:
        # BaseEncoder is a pure ABC (no __init__); nn.Module.__init__ is next in MRO.
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.hidden_dropout_prob = hidden_dropout_prob
        self.tokenizer: Any = tokenizer
        self.bidirectional = bidirectional

        if backbone is not None:
            self.backbone: nn.Module = backbone
        else:
            self.backbone = AutoModel.from_pretrained(model_name_or_path, **from_pretrained_kwargs)

        # Disable ModernBERT's automatic torch.compile (reference_compile) when
        # present so variable-length batches and resized embeddings do not
        # trigger dynamo recompilation stalls during training/eval.
        if hasattr(getattr(self.backbone, "config", None), "reference_compile"):
            self.backbone.config.reference_compile = False

        if bidirectional:
            self._force_bidirectional()

        self.hidden_dim: int = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(p=hidden_dropout_prob)

        # Long-context guard: the tokenizer window must match the backbone window
        # (HF tokenizers often ship a sentinel ~1e30 or a smaller legacy cap).
        self._sync_tokenizer_max_length()

    def _sync_tokenizer_max_length(self) -> None:
        """Set ``tokenizer.model_max_length`` to the backbone window and assert it."""
        mpe = getattr(getattr(self.backbone, "config", None), "max_position_embeddings", None)
        if self.tokenizer is None or not mpe:
            return
        try:
            self.tokenizer.model_max_length = int(mpe)
        except Exception:  # pragma: no cover - read-only tokenizer stubs
            return
        assert int(self.tokenizer.model_max_length) == int(mpe), (
            f"tokenizer.model_max_length {self.tokenizer.model_max_length} != backbone window {mpe}"
        )

    # ------------------------------------------------------------------ #
    # Bidirectional attention (Phase 2b)
    # ------------------------------------------------------------------ #
    def _force_bidirectional(self) -> None:
        """Clear causal masking on every attention module (idempotent).

        Qwen3-0.6B-Base / Qwen3 decoder backbones ship ``is_causal=True``;
        SpanCall's <|cls|> / <|param|> / <|tool|> anchors must attend to the
        whole 32k sequence, so we sweep ``self.backbone`` and set ``is_causal =
        False`` on every module that exposes it. ``Qwen3-Embedding`` variants
        are already bidirectional — the sweep is a no-op. Calling twice is
        harmless (identity write).
        """
        for module in self.backbone.modules():
            if hasattr(module, "is_causal"):
                try:
                    module.is_causal = False
                except (AttributeError, TypeError):
                    pass
        if hasattr(self.backbone, "config") and hasattr(self.backbone.config, "is_decoder"):
            self.backbone.config.is_decoder = False

    # ------------------------------------------------------------------ #
    # Encoder API
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return contextual token representations H in R^{B x L x d}.

        ``token_type_ids`` is forwarded only when provided (some backbones,
        e.g. ModernBERT, do not use segment embeddings and reject non-None
        segment ids; passing ``None`` matches the backbone defaults exactly).
        """
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**kwargs)
        H = outputs.last_hidden_state
        return self.dropout(H)

    # ------------------------------------------------------------------ #
    # Marker-token registration
    # ------------------------------------------------------------------ #
    def add_marker_tokens(self, tokens: List[str]) -> int:
        """Register marker tokens, resize embeddings, init new rows to mean+std.

        Steps (plan §3.2.1):
        1. ``tokenizer.add_special_tokens`` (deduplicated by the tokenizer).
        2. ``backbone.resize_token_embeddings(len(tokenizer))``.
        3. Initialize the newly inserted embedding rows with
           ``randn * std(existing) + mean(existing)`` so marker embeddings do
           not start at the (uninformative) default init.

        Returns the number of *new* tokens actually added (0 if all markers
        were already present).
        """
        if self.tokenizer is None:
            raise ValueError(
                "SpanCallTransformerEncoder.add_marker_tokens requires a tokenizer; "
                "pass `tokenizer=` to the constructor or set `self.tokenizer`."
            )

        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokens)}
        )
        if added > 0:
            emb = self.backbone.get_input_embeddings()
            old_num = int(emb.num_embeddings)
            # resize_token_embeddings return value varies across transformers
            # versions; read the resized embedding size back instead.
            self.backbone.resize_token_embeddings(len(self.tokenizer))
            emb = self.backbone.get_input_embeddings()
            new_num = int(emb.num_embeddings)
            if new_num > old_num:
                with torch.no_grad():
                    mean = emb.weight[:old_num].mean(dim=0)
                    std = emb.weight[:old_num].std(dim=0)
                    noise = torch.randn(
                        new_num - old_num,
                        emb.weight.size(1),
                        device=emb.weight.device,
                        dtype=emb.weight.dtype,
                    )
                    emb.weight[old_num:new_num] = noise * std + mean
        return added
