#!/usr/bin/env python3
"""CLI entrypoint for logit calibration and threshold sweep.

Pipeline: load a trained checkpoint + held-out calibration rows produced by the
resolver's ``_run_heads`` (per-parameter probabilities, joint margin, accept
probability, schema validity, risk tier) plus the ground-truth payloads, fit a
temperature, run the Clopper-Pearson-constrained threshold sweep per risk tier,
and persist ``calibration_report.json``.

The report's top-level keys (``mode``, ``tau_p``, ``tau_a``, ``tau_delta``,
``temperature``, ``tau_tool``) are consumed directly by
``spancall.infer.resolver._default_gate_cfg`` via ``--calib_thresholds``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from spancall.train.calibrate import SpanCallCalibrator
    from spancall.train.config import CalibrationConfig, TrainingConfig
    from spancall.train.trainer import SpanCallTrainer
except Exception as exc:  # pragma: no cover - import guard
    print(f"[SpanCall Calibrate] failed to import: {exc}")
    sys.exit(2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpanCall Calibration: temperature scaling + threshold sweep",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="TrainingConfig YAML (model.backbone_name, calibration, gate sections)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trainer checkpoint dir (model.pt + provenance.json) or bare .pt")
    parser.add_argument("--calib_data", type=str, default=None,
                        help="Path to a JSONL of resolver prediction rows "
                             "(one dict per line; keys per calibrate.py module docstring)")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Optional JSONL of {sample_id, arguments} rows. "
                             "When omitted, rows' own 'ground_truth'/'arguments' are used.")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write calibration_report.json "
                             "(default: <checkpoint>/calibration_report.json)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override the fitted temperature (skip L-BFGS fitting).")
    return parser.parse_args(argv)


def _load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ground_truth_map(pred_rows: List[Dict[str, Any]], gt_path: Optional[str]) -> List[Dict[str, Any]]:
    """Build the parallel GT list the calibrator expects.

    Prefer an explicit ``--ground_truth`` JSONL; otherwise fall back to the
    per-row ``ground_truth``/``arguments`` carried inside the prediction rows.
    """
    if gt_path:
        return _load_rows(gt_path)
    out: List[Dict[str, Any]] = []
    for r in pred_rows:
        sid = r.get("sample_id")
        if sid is None:
            continue
        args = r.get("ground_truth") or r.get("arguments")
        if args is None:
            continue
        out.append({"sample_id": sid, "arguments": args})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        train_cfg = TrainingConfig.from_yaml(args.config)
    except Exception as exc:  # pragma: no cover
        print(f"[SpanCall Calibrate] could not load config {args.config}: {exc}")
        return 2

    cal_cfg: CalibrationConfig = train_cfg.calibration
    mode = str(train_cfg.gate.mode)
    cal_cfg.mode = mode

    pred_rows: List[Dict[str, Any]] = []
    if not args.calib_data:
        # Fall back to the canonical calib split on disk.
        cand = Path(train_cfg.data_dir) / "hermes_calib.jsonl"
        if cand.exists():
            args.calib_data = str(cand)
    if args.calib_data:
        pred_rows = _load_rows(args.calib_data)
    if not pred_rows:
        print("[SpanCall Calibrate] no calibration rows found "
              "(pass --calib_data or configure experiment.data_dir)")
        return 2

    # Optional: fit temperature with L-BFGS on the checkpoint's head logits.
    temperature = args.temperature
    if temperature is None and cal_cfg.temperature_scaling and args.checkpoint:
        train_cfg.model.backbone_name = train_cfg.model.backbone_name
        try:
            trainer = SpanCallTrainer(train_cfg)
            trainer.load_checkpoint(args.checkpoint)
            logits = _head_logits(trainer)
            if logits:
                cal = SpanCallCalibrator(cal_cfg)
                temperature = cal.fit_temperature(logits[0], logits[1])
                print(f"[SpanCall Calibrate] fitted temperature = {temperature:.4f}")
        except Exception as exc:
            print(f"[SpanCall Calibrate] temperature fitting skipped ({exc}); using T = 1.0")
            temperature = 1.0
    if temperature is None:
        temperature = 1.0

    gt_rows = _ground_truth_map(pred_rows, args.ground_truth)
    cal = SpanCallCalibrator(cal_cfg)
    report = cal.calibrate(pred_rows, gt_rows, fitted_temperature=float(temperature))

    out_path = args.output
    if not out_path:
        base = Path(args.checkpoint) if args.checkpoint else Path.cwd()
        if base.is_file():
            base = base.parent
        out_path = str(base / "calibration_report.json")
    cal.write_report(report, out_path)

    print(f"[SpanCall Calibrate] wrote {out_path}")
    summary = report["thresholds"]["global"]
    print(f"[SpanCall Calibrate] mode={report['mode']} tau_p={report['tau_p']:.3f} "
          f"tau_a={report['tau_a']:.3f} tau_delta={report['tau_delta']:.2f} "
          f"T={report['temperature']:.3f}")
    print(f"[SpanCall Calibrate] n_calib={report['n_calib']} n_hits={summary['n_hits']:.0f} "
          f"k_correct={summary['k_correct']:.0f} hit_rate={summary['hit_rate']:.3f} "
          f"precision={summary['precision']:.3f} LBD={summary['lbd']:.3f} ECE={report['ece']:.4f}")
    print("[SpanCall Calibrate] OK")
    return 0


def _head_logits(trainer: Any) -> Optional[List[Any]]:
    """Best-effort extraction of (logits, labels) for temperature fitting.

    Overridable; returns None if the checkpoint lacks the metadata to build
    this without a forward pass over calibration data.
    """
    return None


if __name__ == "__main__":
    sys.exit(main())
