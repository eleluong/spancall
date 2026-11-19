#!/usr/bin/env python3
"""Local multi-model LLM evaluator for SpanCall Lean v3.3.

Replaces the single-model Colab LLM baseline notebook with a reproducible,
resumable, multi-model CLI using the OpenAI Python SDK directly.

Key guarantees:
1. **Frozen dataset integrity**: verifies ``data/eval_v3_3/manifest.json``
   SHA256 hashes before calling any model.
2. **Exact fairness parity**: reuses ``spancall.eval.llm_parity`` (same
   OpenAI tool schema conversion, same prompt, same ``ToolCallEvaluator``
   scoring and fallback merge as SpanCall).
3. **Resumable & fault-tolerant**: writes append-only per-row JSONL keyed by
   ``sample_id`` under ``<results_dir>/llm/<slug>/<split>_preds.jsonl`` so
   interrupted runs never waste paid API calls; ``--retry-errors`` re-runs
   only failed rows.
4. **Multi-model ready**: accepts ``--config configs/llm_eval_v3_3.yaml`` or
   ad-hoc ``--model`` / ``--base-url`` / ``--api-key-env`` flags.

Usage::

    # Multi-model via YAML config
    .venv/bin/python scripts/eval_llm_local.py \\
        --config configs/llm_eval_v3_3.yaml

    # Single model ad-hoc
    .venv/bin/python scripts/eval_llm_local.py \\
        --model openai/gpt-oss-120b --slug gpt-oss-120b \\
        --base-url-env LLM_BASE_URL --api-key-env LLM_API_KEY

    # Offline deterministic smoke test (no network, no API key)
    .venv/bin/python scripts/eval_llm_local.py \\
        --slug fake-llm --model fake-llm \\
        --fake-responses /tmp/canned.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spancall.eval.llm_parity import (  # noqa: E402
    call_openai_sdk,
    latency_percentiles,
    lean_pred_row,
    llm_pred_row,
    load_jsonl,
    load_samples,
    merge_with_fallback,
    messages_for,
    openai_tools,
    replay_fake,
    score_predictions,
    token_summary,
    verify_eval_manifest,
)


def _load_dotenv(env_path: Path) -> None:
    """Load simple KEY=VALUE lines from .env without overriding existing env vars."""
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _slugify(model_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(model_name).strip()).strip("-").lower()
    return s or "llm"


def _load_existing_preds(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    by_id: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("sample_id")
            if sid is not None:
                by_id[str(sid)] = row
    return by_id


def _write_ordered_preds(path: Path, ids: Sequence[str], by_id: Dict[str, Dict[str, Any]]) -> None:
    """Write deduped predictions in split order while preserving any extra cached rows.

    Preserving extra ``sample_id``s guarantees that a smoke run with ``--limit N``
    can never truncate previously cached predictions from a full run.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    seen = set()
    with open(tmp, "w", encoding="utf-8") as fh:
        for sid in ids:
            if sid in by_id and sid not in seen:
                fh.write(json.dumps(by_id[sid], default=str) + "\n")
                seen.add(sid)
        for sid, row in by_id.items():
            if sid not in seen:
                fh.write(json.dumps(row, default=str) + "\n")
    os.replace(tmp, path)


def _resolve_model_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    defaults: Dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "concurrency": args.concurrency,
        "max_retries": args.max_retries,
    }
    specs: List[Dict[str, Any]] = []
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        cfg_defaults = dict(cfg.get("defaults") or {})
        cfg_defaults.update({k: v for k, v in defaults.items() if v is not None})
        only = set(args.only or [])
        for entry in cfg.get("models") or []:
            if not isinstance(entry, dict):
                continue
            slug = str(entry.get("slug") or _slugify(entry.get("model", "")))
            if only:
                if slug not in only:
                    continue
            elif not entry.get("enabled", True):
                continue
            merged = dict(cfg_defaults)
            merged.update(entry)
            merged["slug"] = slug
            specs.append(merged)
    else:
        if not args.model:
            raise SystemExit("Pass either --config <yaml> or --model <model_id>.")
        slug = args.slug or _slugify(args.model)
        spec = dict(defaults)
        spec.update({
            "slug": slug,
            "model": args.model,
            "base_url": args.base_url,
            "base_url_env": args.base_url_env,
            "api_key_env": args.api_key_env,
            "default_api_key": args.default_api_key,
        })
        specs.append(spec)
    return specs


def _make_openai_client(spec: Dict[str, Any]) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai package is required: pip install openai") from exc

    api_key_env = spec.get("api_key_env") or "OPENAI_API_KEY"
    api_key = os.environ.get(str(api_key_env)) or spec.get("default_api_key")
    if not api_key:
        raise SystemExit(
            f"[eval_llm_local] {api_key_env} is not set for model '{spec['slug']}'. "
            f"Export {api_key_env} (or put it in .env), or pass --fake-responses."
        )
    base_url = spec.get("base_url")
    if not base_url and spec.get("base_url_env"):
        base_url = os.environ.get(str(spec["base_url_env"])) or None
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def evaluate_model_on_split(
    spec: Dict[str, Any],
    split_name: str,
    split_path: Path,
    out_dir: Path,
    *,
    limit: Optional[int] = None,
    retry_errors: bool = False,
    fake_map: Optional[Dict[str, Dict[str, Any]]] = None,
    spancall_preds_path: Optional[Path] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    samples, _ = load_samples(split_path)
    if limit is not None and limit > 0:
        samples = samples[: int(limit)]
    ids = [s.sample_id for s in samples]

    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{split_name}_preds.jsonl"
    existing = _load_existing_preds(preds_path)

    todo = []
    for s in samples:
        prev = existing.get(s.sample_id)
        if prev is not None:
            has_err = bool((prev.get("features") or {}).get("llm_error"))
            if not (retry_errors and has_err):
                continue
        todo.append(s)

    if not quiet:
        print(
            f"[llm:{spec['slug']}] {split_name}: total={len(samples)}, "
            f"cached={len(samples) - len(todo)}, todo={len(todo)}",
            flush=True,
        )

    client = None if fake_map is not None else (_make_openai_client(spec) if todo else None)
    concurrency = max(1, int(spec.get("concurrency") or 8))
    temperature = float(spec.get("temperature", 0.0))
    max_tokens = spec.get("max_tokens")
    timeout = float(spec.get("timeout", 120.0))
    max_retries = int(spec.get("max_retries", 4))
    extra_body = spec.get("extra_body")

    lock = threading.Lock()
    done_count = 0
    err_count = 0
    t_start = time.perf_counter()

    def _run_one(sample: Any) -> Dict[str, Any]:
        if fake_map is not None:
            resp = replay_fake(fake_map, sample.sample_id)
        else:
            resp = call_openai_sdk(
                client,
                model=str(spec["model"]),
                messages=messages_for(sample),
                tools=openai_tools(sample),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
                extra_body=extra_body,
            )
        return llm_pred_row(sample.sample_id, resp, model=str(spec["model"]))

    if todo:
        with open(preds_path, "a", encoding="utf-8") as append_fh:
            if concurrency == 1 or fake_map is not None:
                for sample in todo:
                    row = _run_one(sample)
                    existing[sample.sample_id] = row
                    append_fh.write(json.dumps(row, default=str) + "\n")
                    append_fh.flush()
                    done_count += 1
                    if (row.get("features") or {}).get("llm_error"):
                        err_count += 1
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                    fut_map = {ex.submit(_run_one, s): s.sample_id for s in todo}
                    for fut in concurrent.futures.as_completed(fut_map):
                        row = fut.result()
                        with lock:
                            existing[row["sample_id"]] = row
                            append_fh.write(json.dumps(row, default=str) + "\n")
                            append_fh.flush()
                            done_count += 1
                            if (row.get("features") or {}).get("llm_error"):
                                err_count += 1
                            if not quiet and (done_count % 50 == 0 or done_count == len(todo)):
                                elapsed = time.perf_counter() - t_start
                                rate = done_count / max(elapsed, 1e-6)
                                print(
                                    f"  [{spec['slug']}:{split_name}] {done_count}/{len(todo)} "
                                    f"({rate:.1f} rows/s, errs={err_count})",
                                    flush=True,
                                )

    # Normalize & deduplicate on disk in exact split order
    _write_ordered_preds(preds_path, ids, existing)

    # Score llm_only (and optional spancall_plus_llm_fallback)
    llm_metrics = score_predictions(samples, existing, ids=ids)
    lats = [float((existing.get(sid) or {}).get("latency_ms") or 0.0) for sid in ids]
    tok = token_summary(existing, ids)
    f_stratum = (llm_metrics.get("metrics_by_stratum") or {}).get("F", {})

    split_summary: Dict[str, Any] = {
        "split": split_name,
        "n_rows": len(ids),
        "partial": bool(limit is not None and limit > 0),
        "preds_path": str(preds_path),
        "llm_only": {
            "exact_F": float(f_stratum.get("exact_payload_match") or 0.0),
            "hit_F": float(f_stratum.get("hit_rate") or 0.0),
            "param_F1": float(f_stratum.get("param_f1") or 0.0),
            "false_hit_A": float(llm_metrics.get("false_hit_rate_on_a") or 0.0),
            "exact_all": float(llm_metrics.get("exact_payload_match") or 0.0),
            "call_set": float(llm_metrics.get("call_exact_set_match") or 0.0),
            "latency_ms": latency_percentiles(lats),
            "tokens": tok,
        },
    }

    if spancall_preds_path is not None and spancall_preds_path.is_file():
        lean_by_id = {str(r["sample_id"]): lean_pred_row(r) for r in load_jsonl(spancall_preds_path) if r.get("sample_id")}
        merged, n_fb = merge_with_fallback(lean_by_id, existing, ids)
        fb_metrics = score_predictions(samples, merged, ids=ids)
        fb_f = (fb_metrics.get("metrics_by_stratum") or {}).get("F", {})
        fb_lats = [float((merged.get(sid) or {}).get("latency_ms") or 0.0) for sid in ids]
        split_summary["spancall_plus_llm_fallback"] = {
            "exact_F": float(fb_f.get("exact_payload_match") or 0.0),
            "hit_F": float(fb_f.get("hit_rate") or 0.0),
            "param_F1": float(fb_f.get("param_f1") or 0.0),
            "false_hit_A": float(fb_metrics.get("false_hit_rate_on_a") or 0.0),
            "exact_all": float(fb_metrics.get("exact_payload_match") or 0.0),
            "fallback_rate": n_fb / float(max(len(ids), 1)),
            "n_fallback": n_fb,
            "latency_ms": latency_percentiles(fb_lats),
        }

    if not quiet:
        lo = split_summary["llm_only"]
        print(
            f"[llm:{spec['slug']}] {split_name} DONE: "
            f"exact_F={lo['exact_F']:.3f} hit_F={lo['hit_F']:.3f} "
            f"param_F1={lo['param_F1']:.3f} false_hit_A={lo['false_hit_A']:.3f} "
            f"lat_mean={lo['latency_ms']['mean']:.1f}ms errs={tok['n_errors']}",
            flush=True,
        )
    return split_summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local multi-model OpenAI-SDK LLM evaluator (SpanCall v3.3).")
    p.add_argument("--config", type=str, default=None, help="YAML config file (e.g. configs/llm_eval_v3_3.yaml)")
    p.add_argument("--only", nargs="+", default=None, help="Only run these model slugs from --config")
    p.add_argument("--eval-dir", type=str, default=None, help="Frozen v3.3 eval dir (default: data/eval_v3_3)")
    p.add_argument("--results-dir", type=str, default=None, help="Results root (default: results_lean_v3_3)")
    p.add_argument("--splits", nargs="+", default=None, help="Manifest splits to evaluate (default: fair_hermes_eval fair_xlam_eval)")
    # Single-model CLI overrides (used when --config is omitted)
    p.add_argument("--model", type=str, default=None, help="Model ID for OpenAI-compatible API")
    p.add_argument("--slug", type=str, default=None, help="Short folder slug for outputs")
    p.add_argument("--base-url", type=str, default=None, help="Explicit OpenAI-compatible base URL")
    p.add_argument("--base-url-env", type=str, default="LLM_BASE_URL", help="Env var holding base URL")
    p.add_argument("--api-key-env", type=str, default="LLM_API_KEY", help="Env var holding API key")
    p.add_argument("--default-api-key", type=str, default=None, help="Fallback key (e.g. EMPTY for local vLLM)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Smoke-test row limit per split (marks summary as partial)")
    p.add_argument("--retry-errors", action="store_true", help="Re-run cached rows that previously errored")
    p.add_argument("--fake-responses", type=str, default=None, help="Offline JSONL of canned responses for CI/smoke tests")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_dotenv(REPO_ROOT / ".env")
    args = parse_args(argv)

    cfg_eval_dir = "data/eval_v3_3"
    cfg_results_dir = "results_lean_v3_3"
    cfg_splits = ["fair_hermes_eval", "fair_xlam_eval"]
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            raw_cfg = yaml.safe_load(fh) or {}
        cfg_eval_dir = str(raw_cfg.get("eval_dir") or cfg_eval_dir)
        cfg_results_dir = str(raw_cfg.get("results_dir") or cfg_results_dir)
        if raw_cfg.get("splits"):
            cfg_splits = list(raw_cfg["splits"])

    eval_dir = Path(args.eval_dir or cfg_eval_dir)
    results_dir = Path(args.results_dir or cfg_results_dir)
    splits = list(args.splits or cfg_splits)

    # Verify frozen manifest SHA256 before making any API call
    manifest = verify_eval_manifest(eval_dir, splits=splits)
    specs = _resolve_model_specs(args)
    if not specs:
        print("[eval_llm_local] no enabled models to run.", file=sys.stderr)
        return 2

    fake_map: Optional[Dict[str, Dict[str, Any]]] = None
    if args.fake_responses:
        fake_map = {str(r.get("sample_id")): r for r in load_jsonl(args.fake_responses)}

    for spec in specs:
        model_dir = results_dir / "llm" / spec["slug"]
        model_dir.mkdir(parents=True, exist_ok=True)
        split_summaries: Dict[str, Any] = {}
        for sp in splits:
            sp_entry = manifest["splits"][sp]
            sp_path = eval_dir / sp_entry["filename"]
            sc_preds = results_dir / f"preds_{sp}_twostep.jsonl"
            split_summaries[sp] = evaluate_model_on_split(
                spec,
                sp,
                sp_path,
                model_dir,
                limit=args.limit,
                retry_errors=args.retry_errors,
                fake_map=fake_map,
                spancall_preds_path=sc_preds if sc_preds.is_file() else None,
                quiet=args.quiet,
            )

        summary_payload = {
            "version": "v3.3",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "slug": spec["slug"],
            "model": spec["model"],
            "eval_dir": str(eval_dir),
            "manifest_template_stamp": manifest.get("template_stamp"),
            "splits": split_summaries,
        }
        sum_path = model_dir / "summary.json"
        tmp = sum_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, sum_path)
        if not args.quiet:
            print(f"[llm:{spec['slug']}] wrote {sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

