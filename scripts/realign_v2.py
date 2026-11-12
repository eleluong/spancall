#!/usr/bin/env python3
"""Re-align the canonical splits with the v2 aligner and gate them.

Why this exists
---------------
The stored GT char offsets in ``data/processed`` were computed by the v1
aligner, which rendered with the **single-tool** ``build_input`` grammar while
every consumer (trainer, resolver, eval) renders with ``build_input_multi``.
The two grammars differ by the ``<|none|>`` line and, once the template was
revised, by ~30 more characters. The measured result on the eval split was
``span_stale_rate = 1.0``: not one stored offset sliced its own canonical
value, so the pointer head was supervised on the wrong characters.

The v2 aligner fixes the cause, not the symptom:

* it renders with the same ``build_input_multi`` grammar as the consumers,
* it emits ``utt_rel_start``/``utt_rel_end`` (offsets relative to the utterance
  entry) alongside the absolute pair, so labels survive tool packing (``T``)
  and any future template revision,
* ``align()`` is now idempotent and rebuilds ``target_spans`` from scratch.

This script is the W1.5 acceptance gate of ``plans/lean_v2_flow_plan.md``: it
re-aligns every split into a NEW directory (never in place), stamps the
template fingerprint that produced the labels, and then audits them. A run that
fails the gate exits non-zero, so a notebook pre-flight can call it and refuse
to train rather than silently fitting stale labels again.

Usage::

    python scripts/realign_v2.py \
        --src_dir data/processed --out_dir data/processed_v2 \
        --gate_split hermes_eval --max_utt_stale_rate 0.02 --max_unresolvable 0.02

The gate is ``utterance_span_stale_rate``, NOT the all-target ``span_stale_rate``:
the latter is ~0.20 even on a perfectly aligned multi-tool split because
``recent_ctx``/``time_ctx``/``enum_ctx``/``boolean_ctx`` targets legitimately
point outside the utterance and are matched by ``canonical_value`` at tensorize
time. ``absolute_stale_rate`` is reported for diagnosis only and is not gated.

Nothing is written when the gate fails unless ``--force`` is given, so a partial
or broken v2 directory never becomes the thing a later run reads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spancall.data.align import ReverseAligner
from spancall.data.partition import _enum_aware_default, sample_to_dict
from spancall.data.schema import UnifiedSample
from span_template.stamp import STAMP_FILENAME, check_stamp, write_stamp

#: Splits the Lean flow reads. ``prepare_report.json`` is copied verbatim so the
#: v2 directory is self-describing (same pack_tools/seed provenance as v1).
SPLIT_FILES = (
    "hermes_train",
    "hermes_dev",
    "hermes_calib",
    "hermes_eval",
    "hermes_test_ood_schema",
    "hermes_test_ood_para",
    "hermes_test_adversarial",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-align canonical splits with the v2 aligner")
    p.add_argument("--src_dir", default="data/processed", help="v1 splits to read")
    p.add_argument("--out_dir", default="data/processed_v2", help="where v2 splits are written")
    p.add_argument(
        "--splits", nargs="+", default=list(SPLIT_FILES),
        help="split basenames (without .jsonl) to re-align",
    )
    p.add_argument("--limit", type=int, default=None, help="cap rows per split (smoke runs)")
    p.add_argument(
        "--gate_split", default="hermes_eval",
        help="split the acceptance gate is evaluated on",
    )
    p.add_argument(
        "--max_utt_stale_rate", type=float, default=0.02,
        help=(
            "gate: fraction of UTTERANCE_SPAN targets that point at NEITHER their "
            "absolute span nor their render-invariant rel pair. Scoped to "
            "utterance targets because recent_ctx/time_ctx/enum_ctx/boolean_ctx "
            "labels legitimately address context rows outside the utterance and "
            "are resolved by canonical_value, so counting them would fail a "
            "correct multi-tool split."
        ),
    )
    p.add_argument(
        "--max_unresolvable", type=float, default=0.02,
        help=(
            "gate: fraction of rel-pair targets whose rel pair does not map onto "
            "the render at all (utterance text drifted). Must be ~0."
        ),
    )
    p.add_argument(
        "--force", action="store_true",
        help="write the v2 files even when the gate fails (inspect, do not train)",
    )
    p.add_argument(
        "--skip_audit", action="store_true",
        help="write files without running the offset audit (debugging only)",
    )
    return p.parse_args()


def _read_split(path: Path, limit: Optional[int]) -> List[UnifiedSample]:
    out: List[UnifiedSample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            out.append(UnifiedSample.from_dict(json.loads(line)))
            if limit is not None and len(out) >= limit:
                break
    return out


def _write_split(samples: List[UnifiedSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(sample_to_dict(s), default=_enum_aware_default) + "\n")


def _summarize(before: List[UnifiedSample], after: List[UnifiedSample]) -> Dict[str, Any]:
    """Per-split label accounting: how many targets carry the rel pair."""
    def _counts(items: List[UnifiedSample]) -> Dict[str, int]:
        n_targets = n_rel = n_span = 0
        for s in items:
            for spans in (s.target_spans or {}).values():
                for t in spans:
                    n_targets += 1
                    if getattr(t, "char_start", None) is not None:
                        n_span += 1
                    if getattr(t, "utt_rel_start", None) is not None:
                        n_rel += 1
        return {"targets": n_targets, "span_targets": n_span, "rel_targets": n_rel}

    a, b = _counts(before), _counts(after)
    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "targets_before": a["targets"],
        "targets_after": b["targets"],
        "rel_targets_before": a["rel_targets"],
        "rel_targets_after": b["rel_targets"],
        "rel_coverage_after": (
            b["rel_targets"] / b["span_targets"] if b["span_targets"] else float("nan")
        ),
    }


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    if not src_dir.is_dir():
        print(f"[realign_v2] source dir not found: {src_dir}", file=sys.stderr)
        sys.exit(2)

    aligner = ReverseAligner(
        fuzzy_matching=False,
        max_date_drift_days=14,
        align_json_mode_keys=True,
    )

    staged: Dict[str, List[UnifiedSample]] = {}
    reports: Dict[str, Any] = {}
    for name in args.splits:
        src = src_dir / f"{name}.jsonl"
        if not src.exists():
            print(f"[realign_v2] skip {name}: {src} not found")
            continue
        before = _read_split(src, args.limit)
        after = [aligner.align(s) for s in before]
        staged[name] = after
        reports[name] = _summarize(before, after)
        r = reports[name]
        print(
            f"[realign_v2] {name}: rows={r['rows_after']} "
            f"targets {r['targets_before']}->{r['targets_after']} "
            f"rel {r['rel_targets_before']}->{r['rel_targets_after']}"
        )

    if not staged:
        print("[realign_v2] nothing re-aligned; check --src_dir/--splits", file=sys.stderr)
        sys.exit(2)

    # ---- acceptance gate ---------------------------------------------------
    #
    # What the gate must NOT be
    # -------------------------
    # ``absolute_stale_rate`` (the absolute pair disagrees with the current
    # render) is NOT a defect signal. A train row packs 2-5 tools while the
    # aligner renders ONE, so its absolute pair is *supposed* to be stale --
    # measuring it as a failure would fail every multi-tool split for being
    # correct. It is still reported, because its size is the measured rot that
    # the rel pair removes.
    #
    # What the gate IS
    # ----------------
    # Every label must land on its own characters through at least ONE of the
    # two addressings (absolute OR rel). If neither works, the pointer head is
    # supervised on the wrong tokens -- exactly the v1 defect. A rel pair that
    # cannot even map onto the render is a different bug (utterance text
    # drifted) and gets its own threshold.
    #
    # The rate is scoped to UTTERANCE_SPAN targets. ``recent_ctx``/``time_ctx``/
    # ``enum_ctx``/``boolean_ctx`` labels deliberately address context rows
    # outside the utterance and ``tensorize`` matches them by
    # ``canonical_value``, so a stale char offset there is not a labelling bug
    # and must not gate the run (measured: 16.4% all-target stale on a
    # perfectly re-aligned train split, 100% of it context targets).
    gate: Dict[str, Any] = {"split": args.gate_split, "evaluated": False, "passed": None}
    if not args.skip_audit:
        gate_name = args.gate_split
        if gate_name not in staged:
            # Fall back to the first staged split rather than skipping the gate:
            # "we could not find the gate split" is not a pass.
            gate_name = next(iter(staged))
            gate["fallback_split"] = gate_name
        # Audit what was just produced, not what is on disk: an in-memory render
        # of the staged rows is the only thing that proves this run is clean.
        from spancall.lean.diagnostics import audit_gt_offsets

        gate.update(_audit_in_memory(staged[gate_name], audit_gt_offsets))
        stale = gate.get("utterance_span_stale_rate")
        unresolvable = gate.get("unresolvable_rate")
        utt_targets = gate.get("utterance_span_targets")

        def _is_num(v: Any) -> bool:
            return isinstance(v, float) and v == v

        # A NaN rate means "nothing was measured", which must never read as a
        # pass: a gate split with zero utterance targets is a corpus problem,
        # not a clean run.
        ok_stale = _is_num(stale) and stale < args.max_utt_stale_rate
        ok_unres = (
            unresolvable is None
            or (_is_num(unresolvable) and unresolvable < args.max_unresolvable)
        )
        passed = bool(ok_stale and ok_unres)
        gate["passed"] = passed
        gate["evaluated"] = True
        gate["thresholds"] = {
            "max_utt_stale_rate": args.max_utt_stale_rate,
            "max_unresolvable": args.max_unresolvable,
        }
        print(
            f"[realign_v2] GATE {gate_name}: utterance_span_stale_rate={stale} "
            f"(< {args.max_utt_stale_rate} over {utt_targets} utterance targets), "
            f"unresolvable_rate={unresolvable} (< {args.max_unresolvable}) "
            f"-> {'PASS' if passed else 'FAIL'}"
        )
        print(
            f"[realign_v2]    reference: repaired absolute drift on this split "
            f"absolute_stale_rate={gate.get('absolute_stale_rate')} "
            f"hist={gate.get('rel_drift_hist')}"
        )
        if not passed and not args.force:
            print(
                "[realign_v2] refusing to write a v2 directory that fails its own gate. "
                "Re-run with --force only to inspect the failure.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ---- write ------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in staged.items():
        _write_split(samples, out_dir / f"{name}.jsonl")
    report_src = src_dir / "prepare_report.json"
    if report_src.exists():
        shutil.copyfile(report_src, out_dir / "prepare_report.json")

    # Stamp the template revision these labels were produced under. A notebook
    # that later trains on this directory can call ``check_stamp`` and refuse to
    # run when the template has moved on -- the exact failure that made v1
    # unreadable.
    stamp_path = write_stamp(out_dir, extra={"source_dir": str(src_dir), "splits": reports})
    verdict = check_stamp(out_dir)
    print(f"[realign_v2] stamp -> {stamp_path} ({verdict.status})")

    gate_path = out_dir / "realign_report.json"
    with open(gate_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"splits": reports, "gate": gate, "stamp": verdict.status},
            fh, indent=2, default=str,
        )
    print(f"[realign_v2] report -> {gate_path}")

    # Sanity: the stamp we just wrote must verify against the tree we wrote it in.
    if not verdict.ok:
        print(f"[realign_v2] stamp verification failed: {verdict.message()}", file=sys.stderr)
        sys.exit(1)
    print(f"[realign_v2] done -> {out_dir}")


def _audit_in_memory(samples: List[UnifiedSample], audit_fn: Any) -> Dict[str, Any]:
    """Run :func:`audit_gt_offsets` over freshly-aligned samples.

    ``audit_gt_offsets`` reads a split from disk by name, which would audit the
    PREVIOUS contents of ``out_dir`` (or nothing at all on a first run). Writing
    a temp file is the smallest honest way to reuse the real audit instead of
    re-implementing its arithmetic in the gate.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        _write_split(samples, Path(tmp) / "gate.jsonl")
        metrics = audit_fn("gate", tmp)
    keep = {
        k: metrics.get(k)
        for k in (
            "span_targets",
            "span_valid",
            "span_valid_via_absolute",
            "span_valid_via_rel",
            "span_with_rel_pair",
            "span_without_rel_pair",
            "rel_pair_unresolvable",
            "span_stale_recoverable",
            "span_stale_lost",
            "span_stale_rate",
            "absolute_stale_rate",
            "unresolvable_rate",
            "utterance_span_targets",
            "utterance_span_valid_rate",
            "utterance_span_stale_rate",
            "utterance_span_unresolvable",
            "by_target_type",
        )
    }
    keep["rel_drift_hist"] = metrics.get("rel_drift_hist")
    return keep


if __name__ == "__main__":
    main()