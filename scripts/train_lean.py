#!/usr/bin/env python3
"""CLI entrypoint for training the Lean SpanCall model (without abstain machinery).

Usage:
    python scripts/train_lean.py --config configs/default.yaml \
        --experiment_name spancall_lean --data_dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# Ensure repo root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spancall.lean.model import LeanSpanCallModel
from spancall.lean.trainer import LeanTrainer
from spancall.train.config import TrainingConfig

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpanCall Lean Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML (TrainingConfig.from_yaml)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment identifier (overrides config.experiment.name)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save artifacts (overrides config.experiment.output_dir)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory holding the tensorized split files (t2.jsonl, dev.jsonl, ...)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="T2",
        help="Comma-separated curriculum stages to run (e.g. T2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (cpu / mps / cuda); default auto-detect",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Manual seed override (config.experiment.seed default)",
    )
    return parser.parse_args(argv)


def build_encoder(cfg: TrainingConfig, tokenizer: Any):
    """Construct the backbone encoder, registering the marker tokens."""
    import torch
    from spancall.model.encoder import SpanCallTransformerEncoder

    pretrained_kw: Dict[str, Any] = {}
    mp = getattr(cfg.optimization, "mixed_precision", "bf16")
    if torch.cuda.is_available() and mp == "bf16":
        pretrained_kw["dtype"] = torch.bfloat16

    enc = SpanCallTransformerEncoder(
        model_name_or_path=cfg.model.backbone_name,
        hidden_dropout_prob=cfg.model.dropout,
        tokenizer=tokenizer,
        **pretrained_kw,
    )
    if tokenizer is not None:
        from span_template.sanitize import SPECIAL_MARKER_TOKENS

        enc.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))
    return enc


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = TrainingConfig.from_yaml(args.config)
    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.seed is not None:
        cfg.seed = int(args.seed)

    print(f"[SpanCall Lean Train] config: {args.config}")
    print(f"[SpanCall Lean Train] backbone={cfg.model.backbone_name}")

    tokenizer = None
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name, use_fast=True)
        except Exception as exc:
            print(f"[SpanCall Lean Train] tokenizer unavailable: {exc}")
            tokenizer = None

    encoder = build_encoder(cfg, tokenizer)

    model = LeanSpanCallModel(
        encoder=encoder,
        head_dim=cfg.model.head_dim,
        max_span_width=cfg.model.max_span_width,
        hidden_dim=encoder.hidden_dim,
    )

    trainer = LeanTrainer(
        model=model,
        config=cfg,
        tokenizer=tokenizer,
        device=args.device or None,
    )

    if tokenizer is not None and cfg.output_dir:
        trainer.save_checkpoint(str(Path(cfg.output_dir) / "model_init"))

    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    summary = trainer.train(stages=stages)

    out_dir = Path(cfg.output_dir) / f"{cfg.experiment_name}_seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"[SpanCall Lean Train] done. summary -> {out_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
