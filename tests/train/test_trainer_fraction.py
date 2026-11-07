"""Tests for the pipeline-verification ``data.fraction`` knob.

``SpanCallTrainer.load_rows`` deterministically subsamples every split when
``config.data.fraction < 1.0`` (the 20%-sanity-run lever — see
``configs/verify20.yaml``). These tests assert the sampling is seeded, stable,
duplicate-free, and that ``1.0`` loads everything.
"""

from __future__ import annotations

import json
import os

from spancall.train.config import TrainingConfig
from spancall.train.trainer import SpanCallTrainer


def _write_split(data_dir: str, split: str, n: int) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, f"{split}.jsonl"), "w") as fh:
        for i in range(n):
            fh.write(json.dumps({"input_ids": [i] * 4, "x": i}) + "\n")


def _trainer(frac: float, seed: int = 42, n: int = 100):
    """A bare trainer wired to a temp data_dir with one split + fraction cfg."""
    import tempfile

    tmp = tempfile.mkdtemp()
    cfg = TrainingConfig.from_yaml("configs/default.yaml")
    cfg.data.fraction = frac
    cfg.seed = seed
    cfg.data_dir = os.path.join(tmp, "tensorized")
    _write_split(cfg.data_dir, "t0", n)

    tr = object.__new__(SpanCallTrainer)
    tr.config = cfg
    return tr, cfg


def test_fraction_one_loads_all_rows() -> None:
    tr, _ = _trainer(1.0, n=100)
    rows = tr.load_rows("t0")
    assert len(rows) == 100


def test_fraction_two_tenths_subsamples_deterministically() -> None:
    tr, _ = _trainer(0.2, n=100)
    rows1 = tr.load_rows("t0")
    rows2 = tr.load_rows("t0")

    assert len(rows1) == 20            # 20% of 100
    assert [r["x"] for r in rows1] == [r["x"] for r in rows2]  # stable across calls
    assert len({r["x"] for r in rows1}) == 20                 # no duplicates


def test_fraction_seed_changes_subsample() -> None:
    tr_a, _ = _trainer(0.2, seed=42, n=100)
    tr_b, _ = _trainer(0.2, seed=7, n=100)
    ids_a = sorted(r["x"] for r in tr_a.load_rows("t0"))
    ids_b = sorted(r["x"] for r in tr_b.load_rows("t0"))
    assert ids_a != ids_b  # different seeds -> different (but valid) subsets


def test_fraction_scales_eval_split_too() -> None:
    tr, cfg = _trainer(0.2, n=50)
    _write_split(cfg.data_dir, "dev", 50)
    assert len(tr.load_rows("dev")) == 10  # eval splits shrink with the same knob


def test_verify20_config_has_fraction() -> None:
    cfg = TrainingConfig.from_yaml("configs/verify20.yaml")
    assert cfg.data.fraction == 0.2