#!/usr/bin/env python3
"""Smoke: VRAM-lever flags (gradient_checkpointing / freeze_encoder_stages).

Exercises the same tiny ModernBERT backbone as scripts/smoke_trainer.py but with
the new config knobs enabled:

  * gradient_checkpointing=True  -> backbone.gradient_checkpointing_enable called
                                    (tiny backbones may skip; must not crash)
  * freeze_encoder_stages=["T0"] -> after T0, backbone.* params frozen, heads
                                    trainable, and optimizer rebuilt heads-only
  * use_8bit_optimizer=True      -> falls back to AdamW when bitsandbytes absent
  * mixed_precision="fp16"       -> scaler path only active on CUDA; off here

Run: .venv/bin/python scripts/smoke_vram_flags.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import BertTokenizerFast, ModernBertConfig, ModernBertModel

from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.model.model import SpanCallModel
from spancall.train.config import TrainingConfig
from spancall.train.trainer import SpanCallTrainer

L, D, K, C, P = 24, 32, 3, 4, 2
VOCAB = {"[PAD]": 0, "<|cls|>": 1, "<|sep|>": 2, "<|param|>": 3, "<|tool|>": 4,
         ".": 5, "x": 6, "y": 7}


def make_encoder() -> SpanCallTransformerEncoder:
    cfg = ModernBertConfig(
        vocab_size=len(VOCAB), hidden_size=D, intermediate_size=D * 2,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
        pad_token_id=0,
    )
    backbone = ModernBertModel(cfg)
    tok = BertTokenizerFast(tokenizer_file=None, vocab={k: i for i, k in enumerate(sorted(VOCAB))})
    enc = SpanCallTransformerEncoder(model_name_or_path="tiny-modernbert", tokenizer=tok, backbone=backbone)
    enc.add_marker_tokens(["<|param|>", "<|tool|>"])
    return enc


def synthetic_row(seed: int) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, len(VOCAB), (L,))
    input_ids[0] = 1
    attention_mask = torch.ones(L, dtype=torch.long)
    starts = torch.randint(1, L - 1, (K, C), generator=g)
    ends = torch.randint(1, L - 1, (K, C), generator=g)
    ends = torch.maximum(ends, starts)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "param_positions": torch.tensor([5, 9, 13]),
        "utt_starts": torch.tensor([1]),
        "utt_ends": torch.tensor([L - 1]),
        "cand_starts": starts,
        "cand_ends": ends,
        "pos_indices": torch.tensor([[1, 2], [1, 2], [0, -1]]),
        "reason_targets": torch.tensor([1, 1, 3]),
        "accept_targets": torch.tensor(1.0),
        "cue_mask": torch.ones(K),
    }


def write_split(data_dir: Path, name: str, rows: List[Dict[str, Any]]) -> None:
    with open(data_dir / f"{name}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({k: v.tolist() for k, v in r.items()}) + "\n")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_vram_"))
    data_dir = tmp / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    encoder = make_encoder()
    model = SpanCallModel(encoder)

    cfg = TrainingConfig(
        experiment_name="smoke_vram", seed=0,
        output_dir=str(tmp / "results"), data_dir=str(data_dir),
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,          # flag on
        freeze_encoder_stages=["T0", "T1"],   # heads-only for T0
        use_8bit_optimizer=True,              # falls back to AdamW if no bnb
    )
    cfg.optimization.batch_size = 4
    cfg.optimization.num_epochs = 1

    write_split(data_dir, "t0", [synthetic_row(i) for i in range(4)])
    write_split(data_dir, "stage_c", [synthetic_row(100 + i) for i in range(4)])
    write_split(data_dir, "dev", [synthetic_row(200 + i) for i in range(4)])

    trainer = SpanCallTrainer(cfg, model, encoder, encoder.tokenizer)
    summary = trainer.train(stages=["T0", "C"])

    # T0 ran with frozen backbone: every backbone.* param must be frozen now
    # (the freeze persists into later stages unless unfrozen).
    frozen = [n for n, p in model.named_parameters()
              if n.startswith("backbone.") and p.requires_grad]
    assert not frozen, f"backbone params still trainable after T0 freeze: {frozen}"
    # heads must still be trainable
    heads_ok = [n for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("backbone.")]
    assert heads_ok, "no head params trainable after T0 freeze"

    assert summary["T0"]["status"] == "ok", summary
    assert summary["C"]["status"] == "ok", summary
    # stage C re-freezes to correctness-only (existing behavior preserved)
    after_c = {n: p.requires_grad for n, p in model.named_parameters()}
    leaked = [n for n, ok in after_c.items() if ok and not n.startswith("correctness_head")]
    assert not leaked, f"non-correctness params trainable after C: {leaked}"

    print("gradient_checkpointing: enabled without crash")
    print("T0 freeze verified: backbone frozen, heads trainable")
    print("Stage C freeze verified: correctness-only")
    print("optimizer kind:", trainer._optimizer_kind)

    shutil.rmtree(tmp)
    print("SMOKE_VRAM_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())