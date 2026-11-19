#!/usr/bin/env python3
"""Unified audited results exporter for SpanCall Lean v3.3.

Ingests:
- Frozen eval manifest & splits (``data/eval_v3_3/manifest.json``)
- SpanCall two-step predictions (``results_lean_v3_3/preds_<split>_twostep.jsonl``)
- Local multi-model LLM predictions (``results_lean_v3_3/llm/<slug>/<split>_preds.jsonl``)
- Training/dev/probe logs (``train_log.jsonl``, ``run_metadata.json``)
- Optional agentic results (``results_lean_v3_3/agentic/*.json``)

Verifies SHA256 hashes and 100% ``sample_id`` coverage (failing loud unless
``--allow-partial`` is explicitly passed), re-scores EVERY raw prediction file
through ``ToolCallEvaluator``, computes paired bootstrap 95% CIs, and emits
all paper & audit artifacts altogether under ``<out_dir>``:

- ``final_report.json``
- ``audit_report.md``
- ``tables/*.csv``
- ``latex/*.tex``

Usage::

    .venv/bin/python scripts/export_v3_3_results.py \\
        --eval-dir data/eval_v3_3 \\
        --results-dir results_lean_v3_3 \\
        --out-dir results_lean_v3_3/export
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spancall.eval.llm_parity import (  # noqa: E402
    BUCKET_LABELS,
    latency_percentiles,
    lean_pred_row,
    load_jsonl,
    load_samples,
    merge_with_fallback,
    score_predictions,
    sha256_file,
    slice_by_complexity,
    slice_by_corpus_bucket,
    theoretical_cost,
    token_summary,
    verify_eval_manifest,
)
from spancall.eval.toolcall.evaluator import ToolCallEvaluator  # noqa: E402

TIER1_SPLITS = ("fair_hermes_eval", "fair_xlam_eval")
TIER2_SPLITS = (
    "full_hermes_eval",
    "full_xlam_eval",
    "full_hermes_test_ood_schema",
    "full_xlam_test_ood_schema",
    "full_hermes_test_ood_para",
    "full_xlam_test_ood_para",
    "full_hermes_test_adversarial",
    "full_xlam_test_adversarial",
)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:.{nd}f}"


def _tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _load_preds_by_id(path: Path, *, is_lean: bool = False) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in load_jsonl(path):
        sid = r.get("sample_id")
        if sid is None:
            continue
        out[str(sid)] = lean_pred_row(r) if is_lean else r
    return out


def _check_coverage(
    split_name: str,
    system_name: str,
    expected_ids: Sequence[str],
    preds_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    exp_set = set(expected_ids)
    got_set = set(preds_by_id.keys())
    missing = sorted(exp_set - got_set)
    extra = sorted(got_set - exp_set)
    return {
        "split": split_name,
        "system": system_name,
        "expected": len(expected_ids),
        "found": len(exp_set & got_set),
        "missing": len(missing),
        "extra": len(extra),
        "ok": (len(missing) == 0 and len(extra) == 0),
        "sample_missing": missing[:5],
        "sample_extra": extra[:5],
    }


def _extract_headline_row(
    split: str,
    system: str,
    metrics: Dict[str, Any],
    preds_by_id: Dict[str, Dict[str, Any]],
    ids: Sequence[str],
    *,
    fallback_rate: Optional[float] = None,
) -> Dict[str, Any]:
    f = (metrics.get("metrics_by_stratum") or {}).get("F", {})
    lats = [float((preds_by_id.get(sid) or {}).get("latency_ms") or 0.0) for sid in ids]
    lat_stats = latency_percentiles(lats)
    tok = token_summary(preds_by_id, ids)
    return {
        "split": split,
        "system": system,
        "n_total": int(metrics.get("total_samples") or len(ids)),
        "n_F": int(f.get("n") or 0),
        "exact_F": float(f.get("exact_payload_match") or 0.0),
        "hit_F": float(f.get("hit_rate") or 0.0),
        "param_F1": float(f.get("param_f1") or 0.0),
        "false_hit_A": float(metrics.get("false_hit_rate_on_a") or 0.0),
        "exact_all": float(metrics.get("exact_payload_match") or 0.0),
        "call_set": float(metrics.get("call_exact_set_match") or 0.0),
        "lat_ms_mean": lat_stats["mean"],
        "lat_ms_p50": lat_stats["p50"],
        "lat_ms_p90": lat_stats["p90"],
        "tok_in_mean": tok["prompt_tokens_mean"],
        "tok_out_mean": tok["completion_tokens_mean"],
        "n_errors": tok["n_errors"],
        "fallback_rate": fallback_rate,
        "theoretical_cost": theoretical_cost(metrics),
    }


def _paired_bootstrap_exact_f(
    samples: Sequence[Any],
    preds_a: Dict[str, Dict[str, Any]],
    preds_b: Dict[str, Dict[str, Any]],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap 95% CI for ``exact_F(A) - exact_F(B)`` on stratum F."""
    ev = ToolCallEvaluator()
    f_samples = [s for s in samples if getattr(s.stratum, "value", str(s.stratum)) == "F"]
    if not f_samples or n_boot <= 0:
        return {"n_F": len(f_samples), "diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}

    vec_a: List[int] = []
    vec_b: List[int] = []
    for s in f_samples:
        pa = preds_a.get(s.sample_id) or {"sample_id": s.sample_id, "hit": False, "payload": None}
        pb = preds_b.get(s.sample_id) or {"sample_id": s.sample_id, "hit": False, "payload": None}
        ma = ev.evaluate([s], [pa])
        mb = ev.evaluate([s], [pb])
        vec_a.append(1 if ma.exact_payload_match >= 0.5 else 0)
        vec_b.append(1 if mb.exact_payload_match >= 0.5 else 0)

    n = len(f_samples)
    obs = (sum(vec_a) - sum(vec_b)) / float(n)
    diffs = [a - b for a, b in zip(vec_a, vec_b)]
    try:
        import numpy as np

        arr = np.asarray(diffs, dtype=np.float32)
        np_rng = np.random.default_rng(int(seed))
        # Chunked vectorized bootstrap (~200x faster than pure Python for 10k resamples)
        chunk = 1000
        means: List[Any] = []
        rem = int(n_boot)
        while rem > 0:
            k = min(chunk, rem)
            idx = np_rng.integers(0, n, size=(k, n))
            means.append(arr[idx].mean(axis=1))
            rem -= k
        boot_arr = np.sort(np.concatenate(means))
        lo_val = float(boot_arr[max(0, int(math.floor(0.025 * len(boot_arr))))])
        hi_val = float(boot_arr[min(len(boot_arr) - 1, int(math.floor(0.975 * len(boot_arr))))])
    except ImportError:
        rng = random.Random(int(seed))
        boot_vals: List[float] = []
        for _ in range(int(n_boot)):
            acc = 0
            for _ in range(n):
                acc += diffs[rng.randrange(n)]
            boot_vals.append(acc / float(n))
        boot_vals.sort()
        lo_val = boot_vals[max(0, int(math.floor(0.025 * len(boot_vals))))]
        hi_val = boot_vals[min(len(boot_vals) - 1, int(math.floor(0.975 * len(boot_vals))))]
    return {
        "n_F": n,
        "diff": obs,
        "ci_lo": lo_val,
        "ci_hi": hi_val,
    }


def _find_train_log(results_dir: Path) -> Optional[Path]:
    direct = results_dir / "train_log.jsonl"
    if direct.is_file():
        return direct
    matches = sorted(results_dir.glob("**/train_log.jsonl"))
    return matches[-1] if matches else None


def _parse_train_log(log_path: Optional[Path]) -> Dict[str, Any]:
    if log_path is None or not log_path.is_file():
        return {"found": False, "epoch_eval": [], "epoch_eval_macro": [], "probe_eval": []}
    rows = load_jsonl(log_path)
    return {
        "found": True,
        "path": str(log_path),
        "epoch_end": [r for r in rows if r.get("event") == "epoch_end"],
        "epoch_eval": [r for r in rows if r.get("event") == "epoch_eval"],
        "epoch_eval_macro": [r for r in rows if r.get("event") == "epoch_eval_macro"],
        "probe_eval": [r for r in rows if r.get("event") == "probe_eval"],
        "early_stop": [r for r in rows if r.get("event") == "early_stop"],
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_latex_headline(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/export_v3_3_results.py — DO NOT EDIT BY HAND",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Split & System & $n_F$ & \texttt{exact\_F} & \texttt{hit\_F} & \texttt{param\_F1} & \texttt{false\_hit\_A} & \texttt{exact\_all} & Lat.\ (ms) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(r['split'])} & {_tex_escape(r['system'])} & {r['n_F']} & "
            f"{_fmt(r['exact_F'])} & {_fmt(r['hit_F'])} & {_fmt(r['param_F1'])} & "
            f"{_fmt(r['false_hit_A'])} & {_fmt(r['exact_all'])} & {_fmt(r['lat_ms_mean'], 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_complexity(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/export_v3_3_results.py — DO NOT EDIT BY HAND",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Split & Complexity & System & $n_F$ & \texttt{exact\_F} & \texttt{hit\_F} & \texttt{param\_F1} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(r['split'])} & {_tex_escape(r['complexity'])} & {_tex_escape(r['system'])} & "
            f"{r['n_F']} & {_fmt(r['exact_F'])} & {_fmt(r['hit_F'])} & {_fmt(r['param_F1'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_buckets(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/export_v3_3_results.py — DO NOT EDIT BY HAND",
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Split & Corpus & Bucket & System & $n_F$ & \texttt{exact\_F} & \texttt{param\_F1} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(r['split'])} & {_tex_escape(r['corpus'])} & {_tex_escape(r['bucket'])} & "
            f"{_tex_escape(r['system'])} & {r['n_F']} & {_fmt(r['exact_F'])} & {_fmt(r['param_F1'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_ood(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/export_v3_3_results.py — DO NOT EDIT BY HAND",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Split & $n$ & $n_F$ & \texttt{exact\_F} & \texttt{hit\_F} & \texttt{param\_F1} & \texttt{false\_hit\_A} & Lat.\ (ms) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(r['split'])} & {r['n_total']} & {r['n_F']} & "
            f"{_fmt(r['exact_F'])} & {_fmt(r['hit_F'])} & {_fmt(r['param_F1'])} & "
            f"{_fmt(r['false_hit_A'])} & {_fmt(r['lat_ms_mean'], 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_audit_md(path: Path, report: Dict[str, Any]) -> None:
    prov = report["provenance"]
    audit = report["audit"]
    lines = [
        "# SpanCall Lean v3.3 — Unified Audit & Paper Export Report",
        "",
        f"- **Generated (UTC):** `{prov['generated_at_utc']}`",
        f"- **Git commit:** `{prov['git_commit']}`",
        f"- **Audit status:** **`{audit['status']}`**",
        f"- **Eval manifest SHA256:** `{prov['manifest_sha256']}`",
        f"- **Template stamp:** `{prov['template_stamp']}`",
        f"- **Models evaluated:** `{', '.join(prov['llm_models']) or 'none'}`",
        "",
        "## 1. Audit Checklist",
        "",
        "| Check | Status | Details |",
        "| :--- | :---: | :--- |",
        f"| Manifest SHA256 integrity | {'PASS' if audit['manifest_verified'] else 'FAIL'} | {len(report['manifest_splits'])} splits verified |",
        f"| 100% `sample_id` coverage | {'PASS' if audit['coverage_all_ok'] else 'FAIL'} | {audit['n_coverage_failures']} failure(s) across {len(audit['coverage_checks'])} files |",
        f"| LLM API error count | {'PASS' if audit['total_llm_errors'] == 0 else 'WARN'} | {audit['total_llm_errors']} total API error(s) |",
        "",
        "## 2. Tier 1 Fair Eval — Headline Results",
        "",
        "| Split | System | $n_F$ | `exact_F` | `hit_F` | `param_F1` | `false_hit_A` | `exact_all` | `lat_ms` (p90) | `tok_in` | `fallback` |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report["tier1_headline"]:
        fb = _fmt(r["fallback_rate"]) if r.get("fallback_rate") is not None else "—"
        lines.append(
            f"| `{r['split']}` | `{r['system']}` | {r['n_F']} | {_fmt(r['exact_F'])} | "
            f"{_fmt(r['hit_F'])} | {_fmt(r['param_F1'])} | {_fmt(r['false_hit_A'])} | "
            f"{_fmt(r['exact_all'])} | {_fmt(r['lat_ms_mean'], 1)} ({_fmt(r['lat_ms_p90'], 1)}) | "
            f"{_fmt(r['tok_in_mean'], 1)} | {fb} |"
        )

    if report["tier1_bootstrap_ci"]:
        lines += [
            "",
            "### Paired Bootstrap 95% CIs (`exact_F` Delta)",
            "",
            "| Split | Comparison | $n_F$ | Delta | 95% CI |",
            "| :--- | :--- | ---: | ---: | :---: |",
        ]
        for c in report["tier1_bootstrap_ci"]:
            lines.append(
                f"| `{c['split']}` | `{c['comparison']}` | {c['n_F']} | "
                f"{c['diff']:+.4f} | `[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]` |"
            )

    if report["tier1_complexity"]:
        lines += [
            "",
            "## 3. Tier 1 Fair Eval — Normal vs. Complex",
            "",
            "| Split | Complexity | System | $n_F$ | `exact_F` | `hit_F` | `param_F1` |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
        ]
        for r in report["tier1_complexity"]:
            lines.append(
                f"| `{r['split']}` | `{r['complexity']}` | `{r['system']}` | {r['n_F']} | "
                f"{_fmt(r['exact_F'])} | {_fmt(r['hit_F'])} | {_fmt(r['param_F1'])} |"
            )

    if report["tier2_full_ood"]:
        lines += [
            "",
            "## 4. Tier 2 Full ID, OOD & Adversarial Generalization (SpanCall v3.3)",
            "",
            "| Split | $n$ | $n_F$ | `exact_F` | `hit_F` | `param_F1` | `false_hit_A` | `lat_ms` |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in report["tier2_full_ood"]:
            lines.append(
                f"| `{r['split']}` | {r['n_total']} | {r['n_F']} | {_fmt(r['exact_F'])} | "
                f"{_fmt(r['hit_F'])} | {_fmt(r['param_F1'])} | {_fmt(r['false_hit_A'])} | "
                f"{_fmt(r['lat_ms_mean'], 1)} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_all_results(
    eval_dir: Path,
    results_dir: Path,
    out_dir: Path,
    *,
    allow_partial: bool = False,
    bootstrap_n: int = 10000,
    seed: int = 42,
    quiet: bool = False,
) -> Dict[str, Any]:
    manifest = verify_eval_manifest(eval_dir)
    manifest_sha = sha256_file(eval_dir / "manifest.json")

    # If Colab exported a snapshot of its eval manifest, verify split SHA256s match
    # local eval_dir so Colab SpanCall predictions and local LLM predictions are
    # guaranteed to have scored byte-identical data.
    snap_manifest_path = results_dir / "eval_manifest.json"
    if snap_manifest_path.is_file():
        snap = json.loads(snap_manifest_path.read_text(encoding="utf-8"))
        for sp_name, sp_meta in (manifest.get("splits") or {}).items():
            snap_sp = (snap.get("splits") or {}).get(sp_name)
            if snap_sp and snap_sp.get("sha256") != sp_meta.get("sha256"):
                raise ValueError(
                    f"[export_v3_3] AUDIT FAILURE: eval_manifest.json in {results_dir} "
                    f"disagrees with {eval_dir}/manifest.json on '{sp_name}' "
                    f"({str(snap_sp.get('sha256'))[:12]} vs {str(sp_meta.get('sha256'))[:12]}). "
                    f"Re-sync data/eval_v3_3/ before exporting."
                )

    # Discover evaluated LLM slugs under results_dir/llm/<slug>/
    llm_root = results_dir / "llm"
    llm_slugs: List[str] = []
    if llm_root.is_dir():
        llm_slugs = sorted(p.name for p in llm_root.iterdir() if p.is_dir())

    coverage_checks: List[Dict[str, Any]] = []
    tier1_headline: List[Dict[str, Any]] = []
    tier1_complexity: List[Dict[str, Any]] = []
    tier1_buckets: List[Dict[str, Any]] = []
    tier1_bootstrap: List[Dict[str, Any]] = []
    tier2_rows: List[Dict[str, Any]] = []
    total_llm_errors = 0

    # ---- Tier 1: Fair Eval (SpanCall + all LLMs + Router Fallbacks) ------- #
    # Cache per-system headline rows by split to compute macro-average
    by_sys_split: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for sp in TIER1_SPLITS:
        if sp not in manifest["splits"]:
            continue
        sp_path = eval_dir / manifest["splits"][sp]["filename"]
        samples, raw_rows = load_samples(sp_path)
        ids = [s.sample_id for s in samples]
        cx_slices = slice_by_complexity(raw_rows, ids)
        bk_slices = slice_by_corpus_bucket(raw_rows, ids)
        sample_by_id = {s.sample_id: s for s in samples}

        systems_preds: List[Tuple[str, Dict[str, Dict[str, Any]], Optional[float]]] = []

        # SpanCall arm
        sc_path = results_dir / f"preds_{sp}_twostep.jsonl"
        lean_preds: Optional[Dict[str, Dict[str, Any]]] = None
        if sc_path.is_file():
            lean_preds = _load_preds_by_id(sc_path, is_lean=True)
            coverage_checks.append(_check_coverage(sp, "spancall_v3_3", ids, lean_preds))
            systems_preds.append(("spancall_v3_3", lean_preds, None))

        # LLM arms + router fallbacks
        for slug in llm_slugs:
            lp = llm_root / slug / f"{sp}_preds.jsonl"
            if not lp.is_file():
                continue
            llm_preds = _load_preds_by_id(lp, is_lean=False)
            coverage_checks.append(_check_coverage(sp, f"llm:{slug}", ids, llm_preds))
            systems_preds.append((f"llm:{slug}", llm_preds, 1.0))
            if lean_preds is not None:
                merged, n_fb = merge_with_fallback(lean_preds, llm_preds, ids)
                fb_rate = n_fb / float(max(len(ids), 1))
                systems_preds.append((f"router:{slug}", merged, fb_rate))

            # Paired bootstrap CIs vs SpanCall
            if lean_preds is not None and bootstrap_n > 0:
                ci_sc = _paired_bootstrap_exact_f(samples, lean_preds, llm_preds, n_boot=bootstrap_n, seed=seed)
                tier1_bootstrap.append({"split": sp, "comparison": f"spancall_v3_3 - llm:{slug}", **ci_sc})
                ci_rt = _paired_bootstrap_exact_f(samples, merged, llm_preds, n_boot=bootstrap_n, seed=seed + 1)
                tier1_bootstrap.append({"split": sp, "comparison": f"router:{slug} - llm:{slug}", **ci_rt})

        for sys_name, preds, fb_rate in systems_preds:
            m = score_predictions(samples, preds, ids=ids)
            hrow = _extract_headline_row(sp, sys_name, m, preds, ids, fallback_rate=fb_rate)
            if sys_name.startswith("llm:"):
                total_llm_errors += int(hrow["n_errors"])
            tier1_headline.append(hrow)
            by_sys_split.setdefault(sys_name, {})[sp] = hrow

            # Complexity breakdown
            for cx_name in ("normal", "complex"):
                cids = cx_slices.get(cx_name) or []
                if not cids:
                    continue
                csamps = [sample_by_id[i] for i in cids]
                cm = score_predictions(csamps, preds, ids=cids)
                cf = (cm.get("metrics_by_stratum") or {}).get("F", {})
                tier1_complexity.append({
                    "split": sp,
                    "complexity": cx_name,
                    "system": sys_name,
                    "n_total": len(cids),
                    "n_F": int(cf.get("n") or 0),
                    "exact_F": float(cf.get("exact_payload_match") or 0.0),
                    "hit_F": float(cf.get("hit_rate") or 0.0),
                    "param_F1": float(cf.get("param_f1") or 0.0),
                })

            # Corpus x tool-count bucket pivot (+ shared 1-5 tool-count-controlled parity)
            shared_by_corpus: Dict[str, List[str]] = {}
            for bk_key in sorted(bk_slices.keys()):
                corpus, bucket = bk_key.split("|", 1)
                bids = bk_slices[bk_key]
                if bucket in ("1-2", "3-5"):
                    shared_by_corpus.setdefault(corpus, []).extend(bids)
                bsamps = [sample_by_id[i] for i in bids]
                bm = score_predictions(bsamps, preds, ids=bids)
                bf = (bm.get("metrics_by_stratum") or {}).get("F", {})
                tier1_buckets.append({
                    "split": sp,
                    "corpus": corpus,
                    "bucket": bucket,
                    "system": sys_name,
                    "n_total": len(bids),
                    "n_F": int(bf.get("n") or 0),
                    "exact_F": float(bf.get("exact_payload_match") or 0.0),
                    "param_F1": float(bf.get("param_f1") or 0.0),
                })
            for corpus, sids in sorted(shared_by_corpus.items()):
                ssamps = [sample_by_id[i] for i in sids]
                sm = score_predictions(ssamps, preds, ids=sids)
                sf = (sm.get("metrics_by_stratum") or {}).get("F", {})
                tier1_buckets.append({
                    "split": sp,
                    "corpus": corpus,
                    "bucket": "shared_1-5",
                    "system": sys_name,
                    "n_total": len(sids),
                    "n_F": int(sf.get("n") or 0),
                    "exact_F": float(sf.get("exact_payload_match") or 0.0),
                    "param_F1": float(sf.get("param_f1") or 0.0),
                })

    # Add unweighted Hermes/xLAM Macro-Average rows to tier1_headline
    for sys_name, sp_map in by_sys_split.items():
        if all(s in sp_map for s in TIER1_SPLITS):
            h_r, x_r = sp_map[TIER1_SPLITS[0]], sp_map[TIER1_SPLITS[1]]
            fb_vals = [r["fallback_rate"] for r in (h_r, x_r) if r.get("fallback_rate") is not None]
            tier1_headline.append({
                "split": "macro_avg",
                "system": sys_name,
                "n_total": h_r["n_total"] + x_r["n_total"],
                "n_F": h_r["n_F"] + x_r["n_F"],
                "exact_F": 0.5 * (h_r["exact_F"] + x_r["exact_F"]),
                "hit_F": 0.5 * (h_r["hit_F"] + x_r["hit_F"]),
                "param_F1": 0.5 * (h_r["param_F1"] + x_r["param_F1"]),
                "false_hit_A": 0.5 * (h_r["false_hit_A"] + x_r["false_hit_A"]),
                "exact_all": 0.5 * (h_r["exact_all"] + x_r["exact_all"]),
                "call_set": 0.5 * (h_r["call_set"] + x_r["call_set"]),
                "lat_ms_mean": 0.5 * (h_r["lat_ms_mean"] + x_r["lat_ms_mean"]),
                "lat_ms_p50": 0.5 * (h_r["lat_ms_p50"] + x_r["lat_ms_p50"]),
                "lat_ms_p90": 0.5 * (h_r["lat_ms_p90"] + x_r["lat_ms_p90"]),
                "tok_in_mean": 0.5 * (h_r["tok_in_mean"] + x_r["tok_in_mean"]),
                "tok_out_mean": 0.5 * (h_r["tok_out_mean"] + x_r["tok_out_mean"]),
                "n_errors": h_r["n_errors"] + x_r["n_errors"],
                "fallback_rate": (sum(fb_vals) / len(fb_vals)) if fb_vals else None,
                "theoretical_cost": None,
            })

    # ---- Tier 2: Full ID, OOD & Adversarial Generalization (SpanCall) ----- #
    for sp in TIER2_SPLITS:
        if sp not in manifest["splits"]:
            continue
        sc_path = results_dir / f"preds_{sp}_twostep.jsonl"
        if not sc_path.is_file():
            continue
        sp_path = eval_dir / manifest["splits"][sp]["filename"]
        samples, _ = load_samples(sp_path)
        ids = [s.sample_id for s in samples]
        lean_preds = _load_preds_by_id(sc_path, is_lean=True)
        coverage_checks.append(_check_coverage(sp, "spancall_v3_3", ids, lean_preds))
        m = score_predictions(samples, lean_preds, ids=ids)
        tier2_rows.append(_extract_headline_row(sp, "spancall_v3_3", m, lean_preds, ids))

    # ---- Audit Gate ------------------------------------------------------- #
    cov_failures = [c for c in coverage_checks if not c["ok"]]
    if cov_failures and not allow_partial:
        msg_lines = ["[export_v3_3] AUDIT FAILURE: incomplete or mismatched sample_id coverage:"]
        for c in cov_failures:
            msg_lines.append(
                f"  - {c['split']} / {c['system']}: expected={c['expected']} found={c['found']} "
                f"missing={c['missing']} extra={c['extra']}"
            )
        msg_lines.append("Pass --allow-partial only for intentional smoke/partial exports.")
        raise ValueError("\n".join(msg_lines))

    # ---- Optional Metadata, Train Log & Agentic Results ------------------- #
    run_meta_path = results_dir / "run_metadata.json"
    run_metadata = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.is_file() else {}
    train_summary = _parse_train_log(_find_train_log(results_dir))

    agentic_dir = results_dir / "agentic"
    agentic_results: Dict[str, Any] = {}
    if agentic_dir.is_dir():
        for jp in sorted(agentic_dir.glob("*.json")):
            agentic_results[jp.stem] = json.loads(jp.read_text(encoding="utf-8"))

    audit_status = "PASS" if (not cov_failures and total_llm_errors == 0) else (
        "PARTIAL" if cov_failures else "PASS_WITH_LLM_ERRORS"
    )

    report: Dict[str, Any] = {
        "version": "v3.3",
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "eval_dir": str(eval_dir),
            "results_dir": str(results_dir),
            "manifest_sha256": manifest_sha,
            "template_stamp": manifest.get("template_stamp"),
            "llm_models": llm_slugs,
            "bootstrap_n": int(bootstrap_n),
        },
        "audit": {
            "status": audit_status,
            "manifest_verified": True,
            "coverage_all_ok": len(cov_failures) == 0,
            "n_coverage_failures": len(cov_failures),
            "total_llm_errors": total_llm_errors,
            "coverage_checks": coverage_checks,
        },
        "manifest_splits": manifest.get("splits") or {},
        "run_metadata": run_metadata,
        "training": train_summary,
        "tier1_headline": tier1_headline,
        "tier1_bootstrap_ci": tier1_bootstrap,
        "tier1_complexity": tier1_complexity,
        "tier1_tool_buckets": tier1_buckets,
        "tier2_full_ood": tier2_rows,
        "agentic": agentic_results,
    }

    # ---- Write All Artifacts Together ------------------------------------- #
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "final_report.json"
    tmp_r = report_path.with_suffix(".json.tmp")
    with open(tmp_r, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
        fh.write("\n")
    os.replace(tmp_r, report_path)

    _write_audit_md(out_dir / "audit_report.md", report)

    tables_dir = out_dir / "tables"
    _write_csv(
        tables_dir / "fair_headline.csv",
        ["split", "system", "n_total", "n_F", "exact_F", "hit_F", "param_F1", "false_hit_A",
         "exact_all", "call_set", "lat_ms_mean", "lat_ms_p50", "lat_ms_p90",
         "tok_in_mean", "tok_out_mean", "n_errors", "fallback_rate", "theoretical_cost"],
        tier1_headline,
    )
    _write_csv(
        tables_dir / "fair_complexity.csv",
        ["split", "complexity", "system", "n_total", "n_F", "exact_F", "hit_F", "param_F1"],
        tier1_complexity,
    )
    _write_csv(
        tables_dir / "fair_tool_buckets.csv",
        ["split", "corpus", "bucket", "system", "n_total", "n_F", "exact_F", "param_F1"],
        tier1_buckets,
    )
    _write_csv(
        tables_dir / "full_ood_generalization.csv",
        ["split", "system", "n_total", "n_F", "exact_F", "hit_F", "param_F1", "false_hit_A",
         "exact_all", "call_set", "lat_ms_mean", "lat_ms_p50", "lat_ms_p90"],
        tier2_rows,
    )
    _write_csv(
        tables_dir / "train_dev_curve.csv",
        ["epoch", "split", "n", "span_acc", "exact_params_only", "exact_call", "tool_acc", "eval_loss"],
        train_summary.get("epoch_eval") or [],
    )

    latex_dir = out_dir / "latex"
    _write_latex_headline(latex_dir / "tab_v3_3_headline.tex", tier1_headline)
    _write_latex_complexity(latex_dir / "tab_v3_3_complexity.tex", tier1_complexity)
    _write_latex_buckets(latex_dir / "tab_v3_3_buckets.tex", tier1_buckets)
    _write_latex_ood(latex_dir / "tab_v3_3_ood.tex", tier2_rows)

    if not quiet:
        print("=" * 72)
        print(f"v3.3 UNIFIED EXPORT COMPLETE -> {out_dir}  [audit={audit_status}]")
        print("=" * 72)
        print(f"  final_report.json : {report_path}")
        print(f"  audit_report.md   : {out_dir / 'audit_report.md'}")
        print(f"  CSVs              : {tables_dir} (5 files)")
        print(f"  LaTeX tables      : {latex_dir} (4 files)")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export all SpanCall v3.3 results altogether with audit verification.")
    p.add_argument("--eval-dir", type=str, default="data/eval_v3_3")
    p.add_argument("--results-dir", type=str, default="results_lean_v3_3")
    p.add_argument("--out-dir", type=str, default="results_lean_v3_3/export")
    p.add_argument("--allow-partial", action="store_true",
                   help="Allow incomplete sample_id coverage (stamps audit_status=PARTIAL)")
    p.add_argument("--bootstrap-n", type=int, default=10000,
                   help="Number of paired bootstrap resamples (0 to skip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        export_all_results(
            Path(args.eval_dir),
            Path(args.results_dir),
            Path(args.out_dir),
            allow_partial=args.allow_partial,
            bootstrap_n=args.bootstrap_n,
            seed=args.seed,
            quiet=args.quiet,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

