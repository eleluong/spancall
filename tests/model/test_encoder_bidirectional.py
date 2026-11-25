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


def test_default_backbone_is_modernbert_large_and_window_is_4k() -> None:
    """Default backbone is ModernBERT-large with a 4096 default sequence cap."""
    cfg = ModelConfig()
    assert cfg.backbone_name == "answerdotai/ModernBERT-large"
    assert cfg.max_seq_len == 4096
    enc = SpanCallTransformerEncoder
    assert enc.__init__.__defaults__ is not None
    assert enc.__init__.__defaults__[0] == "answerdotai/ModernBERT-large"

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


# ---------------------------------------------------------------------------
# Marker-token resize: the inference-rebuild regression
# ---------------------------------------------------------------------------
class _ResizableBackbone(nn.Module):
    """Backbone whose embedding matrix can be resized, like a real HF model."""

    def __init__(self, hidden: int = 8, vocab: int = 20) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.emb = nn.Embedding(vocab, hidden)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.emb

    def resize_token_embeddings(self, new_num: int) -> nn.Embedding:
        old = self.emb
        self.emb = nn.Embedding(new_num, old.embedding_dim)
        keep = min(int(old.num_embeddings), int(new_num))
        with torch.no_grad():
            self.emb.weight[:keep] = old.weight[:keep]
        return self.emb


class _TokenizerWithMarkers:
    """A checkpoint-style tokenizer that can already hold some markers.

    This is the case that broke loading the v3.2 Hub export. The repo ships a
    ``tokenizer.json`` with the 15 markers, so ``add_special_tokens`` reports
    ``added == 0`` — while the backbone was just built from base ModernBERT,
    whose embedding matrix is 15 rows shorter.

    ``present`` names the markers already registered at construction; they are
    included in ``len()`` exactly as a real checkpoint tokenizer would include
    them.
    """

    def __init__(self, base_vocab: int = 20, present: tuple = ()) -> None:
        self._base = list(range(base_vocab))
        self._present = list(present)
        self.model_max_length = 4096

    def __len__(self) -> int:
        return len(self._base) + len(self._present)

    def add_special_tokens(self, mapping: dict) -> int:
        wanted = list(mapping.get("additional_special_tokens", []))
        new = [t for t in wanted if t not in self._present]
        self._present.extend(new)
        return len(new)


def test_marker_resize_when_tokenizer_already_holds_them() -> None:
    """``added == 0`` must still resize, or load_state_dict fails on tok_embeddings.

    Regression for::

        size mismatch for backbone.backbone.embeddings.tok_embeddings.weight:
        copying a param with shape torch.Size([50383, 1024]) from checkpoint,
        the shape in current model is torch.Size([50368, 1024]).

    Gating the resize on the tokenizer's "how many did I add" count skipped it
    for an inference rebuild, because a checkpoint tokenizer already lists the
    markers. The invariant is ``num_embeddings == len(tokenizer)``.
    """
    # Base backbone: 20 rows. Checkpoint tokenizer: 20 + 15 markers = 35 entries.
    markers = tuple(f"<|m{i}|>" for i in range(15))
    backbone = _ResizableBackbone(hidden=8, vocab=20)
    tok = _TokenizerWithMarkers(base_vocab=20, present=markers)
    enc = SpanCallTransformerEncoder(backbone=backbone, tokenizer=tok, bidirectional=False)

    added = enc.add_marker_tokens(list(markers))
    assert added == 0, "sanity: every marker is already registered"

    rows = int(enc.backbone.get_input_embeddings().num_embeddings)
    assert rows == len(tok) == 35, (
        f"embeddings still {rows} rows vs tokenizer {len(tok)}; "
        "a checkpoint state_dict would fail to load"
    )


def test_marker_resize_preserves_trained_rows_and_is_idempotent() -> None:
    """Only the appended rows are initialised; a second call is a no-op."""
    backbone = _ResizableBackbone(hidden=8, vocab=20)
    tok = _TokenizerWithMarkers(base_vocab=20, present=("<|cls|>",))
    enc = SpanCallTransformerEncoder(backbone=backbone, tokenizer=tok, bidirectional=False)
    before = enc.backbone.get_input_embeddings().weight[:20].clone()

    enc.add_marker_tokens(["<|cls|>"])
    after = enc.backbone.get_input_embeddings().weight
    assert after.shape[0] == 21
    assert torch.equal(after[:20], before), "pre-existing (trained) rows were overwritten"

    rows_before_second = int(after.shape[0])
    enc.add_marker_tokens(["<|cls|>"])
    assert int(enc.backbone.get_input_embeddings().num_embeddings) == rows_before_second, (
        "a repeat call resized the embeddings again"
    )


def test_marker_resize_shrinks_when_tokenizer_is_smaller() -> None:
    """A tokenizer shorter than the base vocab is honoured too (no silent mismatch)."""
    backbone = _ResizableBackbone(hidden=8, vocab=30)
    tok = _TokenizerWithMarkers(base_vocab=22, present=("<|cls|>",))
    enc = SpanCallTransformerEncoder(backbone=backbone, tokenizer=tok, bidirectional=False)

    enc.add_marker_tokens(["<|cls|>"])
    assert int(enc.backbone.get_input_embeddings().num_embeddings) == 23
