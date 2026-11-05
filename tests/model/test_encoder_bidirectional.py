"""Gate tests for Phase 2b — unmasked bidirectional attention on the backbone.

Ref: plans/long_context_fix.md Phase 2 (Option 2). Qwen3-0.6B-Base and other
decoder-derived backbones ship ``is_causal=True`` (causal self-attention).
SpanCall resolves via ``<|cls|>`` / ``<|param|>`` / ``<|tool|>`` anchor tokens that
must read the *whole* sequence, so ``SpanCallTransformerEncoder`` sweeps every
attention module and clears ``is_causal`` (plus ``config.is_decoder``).

These tests use a tiny injected causal backbone with a real
``nn.MultiheadAttention`` so the mechanism is verified deterministically with
no HF download and no dependence on HF attention-implementation internals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from types import SimpleNamespace

from spancall.model.encoder import BaseEncoder, SpanCallTransformerEncoder
from spancall.train.config import ModelConfig


class TinyCausalBackbone(nn.Module):
    """Minimal causal decoder-shaped backbone (``is_causal=True`` attention).

    Mirrors the structural contract of a Qwen3 decoder block: an attention
    module exposing an ``is_causal`` flag plus a ``config`` with
    ``is_decoder=True`` and ``hidden_size``.
    """

    def __init__(self, hidden: int = 16, vocab: int = 64) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, is_decoder=True)
        self.emb = nn.Embedding(vocab, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, 2, batch_first=True)
        # Decoder-derived backbones (Qwen3-Base) start causal.
        self.attn.is_causal = True
        self._forward_calls = 0

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        self._forward_calls += 1
        x = self.ln(self.emb(input_ids))
        key_padding_mask = None if attention_mask is None else (attention_mask == 0)
        # Derive the causal mask explicitly from the swept `is_causal` flag
        # (torch's nn.MultiheadAttention treats a post-hoc `is_causal` attribute
        # inconsistently across versions — see the Phase 2b control test). This
        # mirrors real decoder backbones, where `is_causal` drives an actual
        # causal mask that `_force_bidirectional` disables.
        attn_mask = None
        if getattr(self.attn, "is_causal", False):
            L = x.size(1)
            # Boolean causal mask (True = masked), matching key_padding_mask's
            # bool dtype to avoid torch's mixed-type deprecation warning.
            attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return SimpleNamespace(last_hidden_state=x)


def _run(enc: SpanCallTransformerEncoder, seq: torch.Tensor) -> torch.Tensor:
    attn = torch.ones(1, seq.size(1), dtype=torch.long)
    with torch.no_grad():
        # SpanCallTransformerEncoder.forward already unwraps
        # ``outputs.last_hidden_state`` and returns the raw H tensor.
        return enc.forward(seq, attn)


def test_force_bidirectional_clears_is_causal_on_attention_modules() -> None:
    backbone = TinyCausalBackbone()
    assert backbone.attn.is_causal is True, "sanity: injected backbone is causal"
    enc = SpanCallTransformerEncoder(backbone=backbone, tokenizer=None, bidirectional=True)
    # Sweep cleared the causal flag and the config's decoder shape.
    assert backbone.attn.is_causal is False
    assert backbone.config.is_decoder is False
    assert enc.bidirectional is True


def test_bidirectional_disabled_keeps_causal_flag() -> None:
    backbone = TinyCausalBackbone()
    enc = SpanCallTransformerEncoder(backbone=backbone, tokenizer=None, bidirectional=False)
    assert backbone.attn.is_causal is True
    assert backbone.config.is_decoder is True
    assert enc.bidirectional is False


def test_bidirectional_last_token_perturb_changes_first_hidden_state() -> None:
    """Gate: with unmasking, H[:, 0] must move when only the LAST token changes.

    Under causal attention H[:, 0] cannot depend on the last token; a
    bidirectional model must. This is the Phase 2b acceptance test.
    """
    torch.manual_seed(7)
    backbone = TinyCausalBackbone()
    enc = SpanCallTransformerEncoder(backbone=backbone, tokenizer=None, bidirectional=True)
    enc.eval()

    seq = torch.randint(0, 32, (1, 24))
    seq2 = seq.clone()
    seq2[0, -1] = (seq2[0, -1] + 1) % 32

    H1 = _run(enc, seq)
    H2 = _run(enc, seq2)

    delta_first = (H1[0, 0] - H2[0, 0]).abs().max().item()
    delta_mid = (H1[0, 12] - H2[0, 12]).abs().max().item()
    assert delta_first > 1e-6, "H[:,0] did not react to the last token — still causal"
    assert delta_mid > 1e-6, "mid-sequence representation did not react — still causal"


def test_causal_control_first_hidden_state_is_invariant_to_last_token() -> None:
    """Control: with bidirectional=False the same perturbation leaves H[:,0] fixed."""
    torch.manual_seed(7)
    backbone = TinyCausalBackbone()
    enc = SpanCallTransformerEncoder(backbone=backbone, tokenizer=None, bidirectional=False)
    enc.eval()

    seq = torch.randint(0, 32, (1, 24))
    seq2 = seq.clone()
    seq2[0, -1] = (seq2[0, -1] + 1) % 32

    H1 = _run(enc, seq)
    H2 = _run(enc, seq2)
    delta_first = (H1[0, 0] - H2[0, 0]).abs().max().item()
    assert delta_first < 1e-9, "causal control must keep H[:,0] invariant to the last token"


def test_hidden_dim_auto_derived_from_backbone_config() -> None:
    """SpanCallTransformerEncoder derives d from backbone.config.hidden_size."""
    enc = SpanCallTransformerEncoder(backbone=TinyCausalBackbone(hidden=32), tokenizer=None)
    assert enc.hidden_dim == 32


def test_base_encoder_contract_holds() -> None:
    assert issubclass(SpanCallTransformerEncoder, BaseEncoder)


def test_default_backbone_is_qwen3_embedding_and_window_is_32k() -> None:
    """Phase 2 defaults: Qwen3-Embedding-0.6B with a 32k context window."""
    cfg = ModelConfig()
    assert cfg.backbone_name == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.max_seq_len == 32768
    enc = SpanCallTransformerEncoder
    assert enc.__init__.__defaults__ is not None
    # model_name_or_path is the first default (== the Qwen embedding backbone).
    assert enc.__init__.__defaults__[0] == "Qwen/Qwen3-Embedding-0.6B"

def test_encoder_syncs_tokenizer_model_max_length() -> None:
    """Tokenizer window is forced to the backbone window (no 1e30 sentinel / stale 512)."""
    import torch.nn as nn
    from types import SimpleNamespace
    from spancall.model.encoder import SpanCallTransformerEncoder

    class _BB(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8, max_position_embeddings=32768)

    tok = SimpleNamespace(model_max_length=int(1e30))
    SpanCallTransformerEncoder(tokenizer=tok, backbone=_BB(), bidirectional=False)
    assert tok.model_max_length == 32768
