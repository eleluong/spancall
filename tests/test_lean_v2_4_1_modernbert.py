"""Unit tests for SpanCall Lean v2.4.1 ModernBERT-large flow."""

import json
from pathlib import Path

import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from spancall.lean.model import LeanSpanCallModel
from spancall.lean.trainer import apply_lora
from spancall.model.encoder import SpanCallTransformerEncoder


def _tiny_modernbert_cfg() -> ModernBertConfig:
    return ModernBertConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
        reference_compile=True,
    )


def test_encoder_modernbert_disables_compile_and_omits_none_token_type_ids():
    m = ModernBertModel(_tiny_modernbert_cfg())
    enc = SpanCallTransformerEncoder(backbone=m, hidden_dropout_prob=0.0)
    assert enc.backbone.config.reference_compile is False

    ids = torch.tensor([[1, 5, 6, 2]])
    mask = torch.ones_like(ids)
    H = enc(ids, mask)
    assert H.shape == (1, 4, 32)

    lm = LeanSpanCallModel(encoder=enc, head_dim=16, max_span_width=8)
    tool_logits = lm.tool_head(H[:, 1:3], H[:, 0])
    bin_logits = lm.tool_head_binary(H[:, 1:3], H[:, 0])
    assert tool_logits.shape == (1, 2)
    assert bin_logits.shape == (1, 2)


def test_apply_lora_auto_adapts_modernbert_modules():
    pytest.importorskip("peft")
    m = ModernBertModel(_tiny_modernbert_cfg())
    enc = SpanCallTransformerEncoder(backbone=m, hidden_dropout_prob=0.0)
    lm = LeanSpanCallModel(encoder=enc, head_dim=16, max_span_width=8)
    n_lora = apply_lora(lm, r=4, alpha=8)
    assert n_lora > 0


def test_v2_4_1_notebook_has_modernbert_config_and_isolated_dirs():
    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "spancall_lean_v2_4_1_train_eval.ipynb"
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert 'BACKBONE       = "answerdotai/ModernBERT-large"' in all_src
    assert 'cfg["model"]["backbone_name"] = BACKBONE' in all_src
    assert "tensorized_lean_v2_4_1" in all_src
    assert "results_lean_v2_4_1" in all_src

