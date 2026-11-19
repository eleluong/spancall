#!/usr/bin/env python3
"""Build the frozen, fair v3.3 evaluation benchmark (`data/eval_v3_3/`).

Fixes the Hermes vs. xLAM evaluation imbalances discovered in v3.2:
1. Filters 1-call, scoreable (excluding ``A_missing_gt_schema``), and
   Step-0-sanitized rows *before* sampling so both corpora share identical
   selection rules and no downstream ``_clean`` drift can occur.
2. Separates Hermes-only ``F_json`` into ``hermes_eval_f_json.jsonl`` rather
   than mixing it into cross-corpus comparisons.
3. Builds **Tier 1 (Matched Fair Eval)**: ``fair_hermes_eval.jsonl`` and
   ``fair_xlam_eval.jsonl`` with exact 1:1 ``(cohort x complexity)`` cell
   matching across ``{F_func, A_clean, A_adv} x {normal, complex}``.
4. Builds **Tier 2 (Full 1-Call Scoreable Splits)**: all ID, OOD-schema,
   OOD-para, and matched adversarial splits for both Hermes and xLAM.
5. Freezes ``manifest.json`` with SHA256 file hashes, ``sample_id_hash``,
   template stamp, and exact per-cell composition counts.

Usage::

    .venv/bin/python scripts/build_fair_eval_v3_3.py \\
        --data-dir data/processed_v2 \\
        --out-dir data/eval_v3_3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from span_template.sanitize import SanitizeError
from span_template.stamp import template_fingerprint
from span_template.template import RecentRow, build_input_multi
from spancall.data.align import tooldef_to_template_tool
from spancall.data.schema import UnifiedSample
from spancall.eval.llm_parity import (
    corpus_of_row,
    sample_ids_hash,
    sha256_file,
    tool_bucket,
    verify_eval_manifest,
)
from spancall.lean.eval_p1 import classify_eval_cohort, classify_sample_complexity

TIER1_COHORTS: Tuple[str, ...] = ("F_func", "A_clean", "A_adv")
COMPLEXITIES: Tuple[str, ...] = ("normal", "complex")

TIER2_SPLITS: List[Tuple[str, str, bool]] = [
    # (source_split_name, corpus, is_adversarial)
    ("hermes_eval", "hermes", False),
    ("xlam_eval", "xlam", False),
    ("hermes_test_ood_schema", "hermes", False),
    ("xlam_test_ood_schema", "xlam", False),
    ("hermes_test_ood_para", "hermes", False),
    ("xlam_test_ood_para", "xlam", False),
    ("hermes_test_adversarial", "hermes", True),
    ("xlam_test_adversarial", "xlam", True),
]


def _passes_step0_sanitize(row: Dict[str, Any]) -> bool:
    """Verify the row renders cleanly through ``build_input_multi`` without marker injection."""
    try:
        s = UnifiedSample.from_dict(row)
        t_tools = [tooldef_to_template_tool(t) for t in (s.tools or [])]
        recent = [
            RecentRow(
                key=str(r.get("key", "")),
                value=str(r.get("value", "")),
                source=str(r.get("source", "recent")),
            )
            for r in (s.recent_context or [])
            if isinstance(r, dict)
        ]
        clock = date.fromisoformat(s.current_time_iso) if s.current_time_iso else date(2026, 1, 1)
        build_input_multi(t_tools, s.utterance, recent_context=recent, now=clock)
        return True
    except SanitizeError:
        return False
    except Exception:  # noqa: BLE001
        return False


def _load_filtered_rows(
    src_path: Path,
    *,
    single_call_only: bool = True,
    check_step0: bool = True,
) -> Tuple[List[Tuple[str, Dict[str, Any], str, str]], Dict[str, int]]:
    """Load rows, applying 1-call + scoreable + Step-0 sanitize filtering first.

    Returns ``(kept, stats)`` where each kept item is ``(raw_line, parsed_dict, cohort, complexity)``.
    """
    stats = {
        "total_raw": 0,
        "dropped_multi_call": 0,
        "dropped_missing_gt_schema": 0,
        "dropped_step0_veto": 0,
        "kept": 0,
    }
    kept: List[Tuple[str, Dict[str, Any], str, str]] = []
    with open(src_path, "r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            stats["total_raw"] += 1
            d = json.loads(raw)
            if single_call_only and len(d.get("ground_truth_calls") or []) != 1:
                stats["dropped_multi_call"] += 1
                continue
            cohort = classify_eval_cohort(d)
            if cohort == "A_missing_gt_schema":
                stats["dropped_missing_gt_schema"] += 1
                continue
            if check_step0 and not _passes_step0_sanitize(d):
                stats["dropped_step0_veto"] += 1
                continue
            cx = classify_sample_complexity(d)
            kept.append((raw, d, cohort, cx))
            stats["kept"] += 1
    return kept, stats


def _summarize_rows(rows: Sequence[Tuple[str, Dict[str, Any], str, str]]) -> Dict[str, Any]:
    coh = Counter()
    cx = Counter()
    bkt = Counter()
    cell = Counter()
    sids: List[str] = []
    for _, d, c, comp in rows:
        coh[c] += 1
        cx[comp] += 1
        bkt[tool_bucket(len(d.get("tools") or []))] += 1
        cell[f"{c}|{comp}"] += 1
        sids.append(str(d.get("sample_id") or ""))
    return {
        "n_rows": len(rows),
        "sample_id_hash": sample_ids_hash(sids),
        "cohort_counts": dict(sorted(coh.items())),
        "complexity_counts": dict(sorted(cx.items())),
        "tool_bucket_counts": dict(sorted(bkt.items())),
        "cell_counts": dict(sorted(cell.items())),
    }


def _write_split(
    out_dir: Path,
    filename: str,
    rows: Sequence[Tuple[str, Dict[str, Any], str, str]],
    *,
    tier: str,
    corpus: str,
) -> Dict[str, Any]:
    fpath = out_dir / filename
    tmp = fpath.with_suffix(fpath.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for raw, _, _, _ in rows:
            fh.write(raw + "\n")
    os.replace(tmp, fpath)
    meta = _summarize_rows(rows)
    meta["filename"] = filename
    meta["tier"] = tier
    meta["corpus"] = corpus
    meta["sha256"] = sha256_file(fpath)
    return meta


def build_matched_fair_pair(
    hermes_rows: Sequence[Tuple[str, Dict[str, Any], str, str]],
    xlam_rows: Sequence[Tuple[str, Dict[str, Any], str, str]],
    *,
    seed: int = 42,
    fair_limit: Optional[int] = None,
) -> Tuple[
    List[Tuple[str, Dict[str, Any], str, str]],
    List[Tuple[str, Dict[str, Any], str, str]],
    List[Tuple[str, Dict[str, Any], str, str]],
    Dict[str, int],
]:
    """Construct exact 1:1 ``(cohort x complexity)`` matched eval slices.

    Excludes ``F_json`` from both corpora (returned separately as ``f_json_rows``).
    """
    h_cells: Dict[Tuple[str, str], List[Tuple[str, Dict[str, Any], str, str]]] = {}
    x_cells: Dict[Tuple[str, str], List[Tuple[str, Dict[str, Any], str, str]]] = {}
    f_json_rows: List[Tuple[str, Dict[str, Any], str, str]] = []

    for item in hermes_rows:
        _, _, c, cx = item
        if c == "F_json":
            f_json_rows.append(item)
        elif c in TIER1_COHORTS:
            h_cells.setdefault((c, cx), []).append(item)

    for item in xlam_rows:
        _, _, c, cx = item
        if c in TIER1_COHORTS:
            x_cells.setdefault((c, cx), []).append(item)

    # Sort each cell deterministically by sample_id before seeded shuffle
    raw_quotas: Dict[Tuple[str, str], int] = {}
    for c in TIER1_COHORTS:
        for cx in COMPLEXITIES:
            key = (c, cx)
            h_list = sorted(h_cells.get(key, []), key=lambda x: str(x[1].get("sample_id") or ""))
            x_list = sorted(x_cells.get(key, []), key=lambda x: str(x[1].get("sample_id") or ""))
            h_cells[key] = h_list
            x_cells[key] = x_list
            raw_quotas[key] = min(len(h_list), len(x_list))

    total_matched = sum(raw_quotas.values())
    if fair_limit is not None and fair_limit > 0 and fair_limit < total_matched:
        # Scale all cells proportionally while preserving exact 1:1 parity between corpora
        scale = fair_limit / float(total_matched)
        quotas = {k: max(1, int(round(v * scale))) if v > 0 else 0 for k, v in raw_quotas.items()}
        # Trim or top up to hit fair_limit exactly without exceeding raw_quotas
        for k in sorted(quotas):
            quotas[k] = min(quotas[k], raw_quotas[k])
        diff = sum(quotas.values()) - fair_limit
        for k in sorted(quotas, key=lambda kk: quotas[kk], reverse=True):
            if diff == 0:
                break
            if diff > 0 and quotas[k] > 1:
                take = min(diff, quotas[k] - 1)
                quotas[k] -= take
                diff -= take
            elif diff < 0 and quotas[k] < raw_quotas[k]:
                add = min(-diff, raw_quotas[k] - quotas[k])
                quotas[k] += add
                diff += add
    else:
        quotas = raw_quotas

    rng_h = random.Random(int(seed))
    rng_x = random.Random(int(seed) + 17)
    fair_h: List[Tuple[str, Dict[str, Any], str, str]] = []
    fair_x: List[Tuple[str, Dict[str, Any], str, str]] = []
    matched_counts: Dict[str, int] = {}

    for c in TIER1_COHORTS:
        for cx in COMPLEXITIES:
            key = (c, cx)
            q = quotas[key]
            matched_counts[f"{c}|{cx}"] = q
            if q <= 0:
                continue
            hl = list(h_cells[key])
            xl = list(x_cells[key])
            rng_h.shuffle(hl)
            rng_x.shuffle(xl)
            fair_h.extend(hl[:q])
            fair_x.extend(xl[:q])

    rng_h.shuffle(fair_h)
    rng_x.shuffle(fair_x)
    return fair_h, fair_x, f_json_rows, matched_counts


def _stratified_cap_rows(
    rows: Sequence[Tuple[str, Dict[str, Any], str, str]],
    limit: int,
    seed: int,
) -> List[Tuple[str, Dict[str, Any], str, str]]:
    """Deterministic complexity-stratified cap (used for huge adversarial splits)."""
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    by_cx: Dict[str, List[Tuple[str, Dict[str, Any], str, str]]] = {}
    for item in rows:
        by_cx.setdefault(item[3], []).append(item)
    rng = random.Random(int(seed))
    out: List[Tuple[str, Dict[str, Any], str, str]] = []
    total = len(rows)
    remaining = limit
    keys = sorted(by_cx.keys())
    for idx, k in enumerate(keys):
        items = sorted(by_cx[k], key=lambda x: str(x[1].get("sample_id") or ""))
        rng.shuffle(items)
        if idx == len(keys) - 1:
            take = min(remaining, len(items))
        else:
            take = min(int(round(limit * (len(items) / float(total)))), len(items), remaining)
        out.extend(items[:take])
        remaining -= take
    rng.shuffle(out)
    return out


def build_fair_eval_dataset(
    data_dir: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    fair_limit: Optional[int] = None,
    adv_limit: int = 5000,
    check_step0: bool = True,
    quiet: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_meta: Dict[str, Any] = {}
    filter_stats: Dict[str, Dict[str, int]] = {}

    # 1. Load & filter hermes_eval and xlam_eval for Tier 1 + Tier 2
    loaded_tier2: Dict[str, List[Tuple[str, Dict[str, Any], str, str]]] = {}
    for split_name, corpus, is_adv in TIER2_SPLITS:
        src = data_dir / f"{split_name}.jsonl"
        if not src.is_file():
            if not quiet:
                print(f"[v3.3-eval] WARN: missing optional split {src}, skipping", file=sys.stderr)
            continue
        rows, st = _load_filtered_rows(src, single_call_only=True, check_step0=check_step0)
        filter_stats[split_name] = st
        if is_adv and adv_limit > 0:
            rows = _stratified_cap_rows(rows, adv_limit, seed=seed)
        loaded_tier2[split_name] = rows
        out_name = f"full_{split_name}"
        splits_meta[out_name] = _write_split(
            out_dir, f"{out_name}.jsonl", rows, tier="tier2_full", corpus=corpus
        )
        if not quiet:
            print(
                f"[v3.3-eval] Tier 2 {out_name:<32} n={len(rows):>5} "
                f"(raw={st['total_raw']}, multi={st['dropped_multi_call']}, "
                f"missing_schema={st['dropped_missing_gt_schema']}, step0={st['dropped_step0_veto']})"
            )

    if "hermes_eval" not in loaded_tier2 or "xlam_eval" not in loaded_tier2:
        raise FileNotFoundError(
            f"Both hermes_eval.jsonl and xlam_eval.jsonl are required in {data_dir} to build Tier 1 fair eval."
        )

    # 2. Build Tier 1 matched fair eval
    fair_h, fair_x, f_json_rows, matched_counts = build_matched_fair_pair(
        loaded_tier2["hermes_eval"],
        loaded_tier2["xlam_eval"],
        seed=seed,
        fair_limit=fair_limit,
    )
    splits_meta["fair_hermes_eval"] = _write_split(
        out_dir, "fair_hermes_eval.jsonl", fair_h, tier="tier1_fair", corpus="hermes"
    )
    splits_meta["fair_xlam_eval"] = _write_split(
        out_dir, "fair_xlam_eval.jsonl", fair_x, tier="tier1_fair", corpus="xlam"
    )
    combined = list(fair_h) + list(fair_x)
    splits_meta["fair_eval_combined"] = _write_split(
        out_dir, "fair_eval_combined.jsonl", combined, tier="tier1_fair", corpus="combined"
    )
    if f_json_rows:
        splits_meta["hermes_eval_f_json"] = _write_split(
            out_dir, "hermes_eval_f_json.jsonl", f_json_rows, tier="auxiliary", corpus="hermes"
        )

    manifest = {
        "version": "v3.3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "fair_limit": fair_limit,
        "adv_limit": int(adv_limit),
        "single_call_only": True,
        "step0_sanitized": bool(check_step0),
        "source_data_dir": str(data_dir),
        "template_stamp": template_fingerprint(),
        "matched_cell_counts": matched_counts,
        "filter_stats": filter_stats,
        "splits": splits_meta,
    }
    manifest_path = out_dir / "manifest.json"
    tmp_m = manifest_path.with_suffix(".json.tmp")
    with open(tmp_m, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_m, manifest_path)

    # Self-verify before returning
    verify_eval_manifest(out_dir)

    if not quiet:
        print("")
        print("=" * 72)
        print(f"FROZEN v3.3 FAIR EVAL BENCHMARK -> {out_dir}")
        print("=" * 72)
        print(f"  fair_hermes_eval : n={len(fair_h)}  sha256={splits_meta['fair_hermes_eval']['sha256'][:12]}...")
        print(f"  fair_xlam_eval   : n={len(fair_x)}  sha256={splits_meta['fair_xlam_eval']['sha256'][:12]}...")
        print(f"  matched cells    : {matched_counts}")
        if f_json_rows:
            print(f"  hermes_f_json    : n={len(f_json_rows)} (separated from Tier 1)")
        print(f"  manifest         : {manifest_path} (verified OK)")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and freeze the fair v3.3 evaluation dataset.")
    p.add_argument("--data-dir", type=str, default="data/processed_v2",
                   help="Directory holding aligned Hermes + xLAM canonical JSONL splits")
    p.add_argument("--out-dir", type=str, default="data/eval_v3_3",
                   help="Output directory for frozen Tier 1 + Tier 2 eval splits and manifest.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fair-limit", type=int, default=None,
                   help="Optional per-corpus cap for Tier 1 matched fair eval (default: full intersection)")
    p.add_argument("--adv-limit", type=int, default=5000,
                   help="Stratified per-corpus cap for adversarial splits (0 = keep all)")
    p.add_argument("--skip-step0-check", action="store_true",
                   help="Skip Step-0 template render check (faster for synthetic tests)")
    p.add_argument("--verify-only", action="store_true",
                   help="Only verify an existing --out-dir against its manifest.json")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        m = verify_eval_manifest(args.out_dir)
        if not args.quiet:
            print(f"[v3.3-eval] OK: {args.out_dir}/manifest.json verified ({len(m.get('splits') or {})} splits)")
        return 0
    build_fair_eval_dataset(
        Path(args.data_dir),
        Path(args.out_dir),
        seed=args.seed,
        fair_limit=args.fair_limit,
        adv_limit=args.adv_limit,
        check_step0=not args.skip_step0_check,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
