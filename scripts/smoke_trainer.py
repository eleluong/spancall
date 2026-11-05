#!/usr/bin/env python3
"""Phase 2e smoke: staged SpanCallTrainer end-to-end incl. Stage C freeze path.

Exercises the exact production wiring from ``scripts/train.py`` but with a
tiny config-built ModernBERT backbone (offline-friendly — no hub downloads):

    prepare -> build encoder -> SpanCallModel -> SpanCallTrainer
             -> train(stages=['T0','T1','T2','C']) -> summary.json

Then asserts:
  * every stage runs (no 'skipped' unless data missing), losses are real floats,
  * the Stage C freeze path keeps only ``correctness_head*`` trainable,
  * ``provenance.json`` parses cleanly and embeds the nested HeadsConfig.

Run: .venv/bin/python scripts/smoke_trainer.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# allow `python scripts/smoke_trainer.py` from anywhere in the repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import BertTokenizerFast, ModernBertConfig, ModernBertModel

from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.model.model import SpanCallModel
from spancall.train.config import TrainingConfig
from spancall.train.trainer import SpanCallTrainer

L = 24          # sequence length
D = 32          # hidden dim
K = 3           # params per sample
C = 4           # candidates per param (span_logits width; +1 abstain col)
P = 2           # positives per param
# Qwen-native marker tokens (single atomic tokens, matching the restyled
# template grammar — see span_template/sanitize.SPECIAL_MARKER_TOKENS).
VOCAB = {"[PAD]": 0, "<|cls|>": 1, "<|sep|>": 2, "<|param|>": 3, "<|tool|>": 4,
         ".": 5, "x": 6, "y": 7}


def make_encoder() -> SpanCallTransformerEncoder:
    """Tiny TransformerEncoder over a config-built ModernBert backbone."""
    cfg = ModernBertConfig(
        vocab_size=len(VOCAB),
        hidden_size=D,
        intermediate_size=D * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    backbone = ModernBertModel(cfg)
    tok = BertTokenizerFast(tokenizer_file=None, vocab={k: i for i, k in enumerate(sorted(VOCAB))})
    enc = SpanCallTransformerEncoder(
        model_name_or_path="tiny-modernbert",
        tokenizer=tok,
        backbone=backbone,
    )
    enc.add_marker_tokens(["<|param|>", "<|tool|>"])
    return enc


def synthetic_row(seed: int) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, len(VOCAB), (L,))
    input_ids[0] = 1  # <|cls|>
    attention_mask = torch.ones(L, dtype=torch.long)
    # param anchors + utterance extent
    param_positions = torch.tensor([5, 9, 13])
    utt = torch.tensor([1, L - 1])
    # candidates: (K, C) start/end pairs, -1 padded beyond C
    starts = torch.randint(1, L - 1, (K, C), generator=g)
    ends = torch.randint(1, L - 1, (K, C), generator=g)
    ends = torch.maximum(ends, starts)
    # positives (indices into candidate columns 1..C, 0 = abstain)
    pos = torch.tensor([[1, 2], [1, 2], [0, -1]])  # last param abstains
    reason = torch.tensor([1, 1, 3])                # 0..4
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "param_positions": param_positions,
        "utt_starts": utt[:1],
        "utt_ends": utt[1:],
        "cand_starts": starts,
        "cand_ends": ends,
        "pos_indices": pos,
        "reason_targets": reason,
        "accept_targets": torch.tensor(1.0),  # 0-dim -> collate -> (B,)
        "cue_mask": torch.ones(K),
    }


def write_split(data_dir: Path, name: str, rows: List[Dict[str, Any]]) -> None:
    with open(data_dir / f"{name}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({k: v.tolist() for k, v in r.items()}) + "\n")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_trainer_"))
    data_dir = tmp / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    encoder = make_encoder()
    model = SpanCallModel(encoder)  # default HeadsConfig (H1+H3+H4+H8+accept)

    cfg = TrainingConfig(
        experiment_name="smoke",
        seed=0,
        output_dir=str(tmp / "results"),
        data_dir=str(data_dir),
        gradient_accumulation_steps=1,
    )
    cfg.optimization.batch_size = 4
    cfg.optimization.num_epochs = 1

    write_split(data_dir, "t0", [synthetic_row(i) for i in range(4)])
    write_split(data_dir, "stage_c", [synthetic_row(100 + i) for i in range(4)])
    write_split(data_dir, "dev", [synthetic_row(200 + i) for i in range(4)])

    trainer = SpanCallTrainer(cfg, model, encoder, encoder.tokenizer)
    summary = trainer.train(stages=["T0", "T1", "T2", "C"])

    print("summary:", json.dumps(summary, indent=2, default=str))

    # --- assertions ----------------------------------------------------
    assert summary["T0"]["status"] == "ok" and summary["T0"]["steps"] > 0
    assert isinstance(summary["T0"]["loss"], float)
    # T1/T2 fall back to t1/t2/train, none of which exist -> skipped
    assert summary["T1"]["status"] == "skipped", summary["T1"]
    assert summary["T2"]["status"] == "skipped", summary["T2"]
    # Stage C must run on stage_c.jsonl with the freeze path
    assert summary["C"]["status"] == "ok", summary["C"]

    # Stage C freeze verification: ONLY correctness_head* params trainable
    after = {n: p.requires_grad for n, p in model.named_parameters()}
    leaked = [n for n, ok in after.items() if ok and not n.startswith("correctness_head")]
    frozen_corr = [n for n, ok in after.items() if not ok and n.startswith("correctness_head")]
    assert not leaked, f"non-correctness params still trainable after Stage C: {leaked}"
    assert not frozen_corr, f"correctness params frozen in Stage C: {frozen_corr}"

    # provenance must parse (the bug: nested HeadsConfig corrupted the JSON)
    prov_path = list(Path(tmp / "results").rglob("provenance.json"))
    assert prov_path, "no provenance.json written"
    prov = json.loads(prov_path[0].read_text())
    assert prov["config"]["model"]["heads"]["pointer"] is True
    assert prov["config"]["model"]["heads"]["correctness"] is True
    print("provenance.json OK:", prov["experiment"], prov["stage"], prov["global_step"], "seed", prov["seed"])

    # summary.json contract (mirrors scripts/train.py)
    out_root = Path(tmp / "results") / f"{cfg.experiment_name}_seed{cfg.seed}"
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    assert (out_root / "summary.json").exists()

    cp = list(Path(tmp / "results").rglob("model.pt"))
    print("checkpoints:", [str(p.relative_to(tmp)) for p in cp])
    assert cp, "no model checkpoint persisted"

    shutil.rmtree(tmp)
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())