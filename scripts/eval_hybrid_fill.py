#!/usr/bin/env python3
"""Resumable HYBRID_FILL & HYBRID_FILL+SCHEMA evaluator for SpanCall Lean v3.3.

Evaluates partial-fill routing on top of frozen Tier 1 SpanCall predictions and
cached full-LLM predictions from ``scripts/eval_llm_local.py``:

- ``hybrid_fill``: SpanCall commits clean hits; on abstentions with a predicted
  tool and unresolved required parameters, the LLM fills ONLY the missing
  parameters via a restricted single-tool schema.
- ``hybrid_fill_schema``: On ``H1``-selected tools with ``array``/``object``
  parameters, SpanCall keeps its grounded scalar arguments and the LLM fills
  ONLY the complex/missing parameters; falls back to full LLM only if no scalar
  skeleton exists or schema validation fails.

Usage::

    # Run both modes for all enabled models in configs/llm_eval_v3_3.yaml
    .venv/bin/python scripts/eval_hybrid_fill.py --config configs/llm_eval_v3_3.yaml

    # Run for one specific model slug (matching results_lean_v3_3/llm/<slug>/)
    .venv/bin/python scripts/eval_hybrid_fill.py \\
        --model openai/gpt-oss-120b --slug openai-gpt-oss-120b \\
        --base-url-env BASE_URL --api-key-env API_KEY

    # Retry only failed fill API calls
    .venv/bin/python scripts/eval_hybrid_fill.py \\
        --model google/gemma-4-26b-a4b-it --slug google-gemma-4-26b-a4b-it \\
        --retry-errors

    # Offline deterministic smoke test (no network, no API key)
    .venv/bin/python scripts/eval_hybrid_fill.py \\
        --slug smoke --model smoke --fake-responses /tmp/fake_fill.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_llm_local import (  # noqa: E402
    _load_dotenv,
    _load_existing_preds,
    _make_openai_client,
    _resolve_model_specs,
    _write_ordered_preds,
)
from spancall.eval.hybrid_fill import (  # noqa: E402
    HYBRID_MODES,
    build_hybrid_fill_request,
    call_json_fill_sdk,
    compact_system_summary,
    finalize_hybrid_fill_row,
    merge_router_schema,
    merge_with_fallback,
    plan_hybrid_fill,
)
from spancall.eval.llm_parity import (  # noqa: E402
    LLMResponse,
    lean_pred_row,
    load_jsonl,
    load_samples,
    replay_fake,
    verify_eval_manifest,
)


def _resp_to_cache_row(sample_id: str, mode: str, missing: Sequence[str], resp: LLMResponse) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "mode": mode,
        "missing": list(missing),
        "tool_calls": list(resp.tool_calls),
        "text": resp.text,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "latency_ms": resp.latency_ms,
        "error": resp.error,
    }


def _cache_row_to_resp(row: Dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        tool_calls=row.get("tool_calls") or [],
        text=str(row.get("text") or ""),
        prompt_tokens=int(row.get("prompt_tokens", 0) or 0),
        completion_tokens=int(row.get("completion_tokens", 0) or 0),
        latency_ms=float(row.get("latency_ms", 0.0) or 0.0),
        error=row.get("error"),
    )


def _load_spancall_preds(results_dir: Path, split_name: str) -> Dict[str, Dict[str, Any]]:
    candidates = [
        results_dir / f"preds_{split_name}_twostep.jsonl",
        results_dir / f"preds_{split_name}.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return {
                str(r["sample_id"]): lean_pred_row(r)
                for r in load_jsonl(path)
                if r.get("sample_id") is not None
            }
    raise FileNotFoundError(f"SpanCall predictions not found for {split_name} in {results_dir}")


def _load_full_llm_preds(results_dir: Path, slug: str, split_name: str) -> Dict[str, Dict[str, Any]]:
    llm_dir = results_dir / "llm" / slug
    path = llm_dir / f"{split_name}_preds.jsonl"
    if not path.is_file():
        return {}
    return _load_existing_preds(path)


def evaluate_hybrid_mode_on_split(
    spec: Dict[str, Any],
    mode: str,
    split_name: str,
    split_path: Path,
    results_dir: Path,
    out_dir: Path,
    *,
    limit: Optional[int] = None,
    retry_errors: bool = False,
    overwrite_cache: bool = False,
    fill_optional: bool = False,
    fake_map: Optional[Dict[str, Dict[str, Any]]] = None,
    quiet: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    samples, raw_rows = load_samples(split_path)
    if limit is not None and limit > 0:
        samples = samples[: int(limit)]
        raw_rows = raw_rows[: int(limit)]
    ids = [s.sample_id for s in samples]

    lean_by_id = _load_spancall_preds(results_dir, split_name)
    full_llm_by_id = _load_full_llm_preds(results_dir, spec["slug"], split_name)

    plans = {
        s.sample_id: plan_hybrid_fill(
            s,
            lean_by_id.get(s.sample_id) or {"sample_id": s.sample_id, "hit": False},
            mode=mode,
            fill_optional=fill_optional,
        )
        for s in samples
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{mode}_{split_name}_fill_cache.jsonl"
    if overwrite_cache and cache_path.is_file():
        cache_path.unlink()
    existing_cache = _load_existing_preds(cache_path)

    fill_samples = [s for s in samples if plans[s.sample_id].action == "fill"]
    todo = []
    for s in fill_samples:
        prev = existing_cache.get(s.sample_id)
        if prev is not None:
            has_err = bool(prev.get("error"))
            if not (retry_errors and has_err):
                continue
        todo.append(s)

    if not quiet:
        n_hit = sum(1 for p in plans.values() if p.action == "spancall_hit")
        n_fb = sum(1 for p in plans.values() if p.action == "full_fallback")
        print(
            f"[hybrid:{spec['slug']}:{mode}] {split_name}: total={len(samples)}, "
            f"spancall_hit={n_hit}, fill_candidates={len(fill_samples)} "
            f"(cached={len(fill_samples) - len(todo)}, todo={len(todo)}), full_fallback={n_fb}",
            flush=True,
        )

    if todo:
        client = None if fake_map is not None else _make_openai_client(spec)
        write_lock = threading.Lock()
        done_count = 0
        err_count = 0

        def _run_one(s: Any) -> Tuple[str, Dict[str, Any]]:
            plan = plans[s.sample_id]
            if fake_map is not None:
                resp = replay_fake(fake_map, s.sample_id)
            else:
                msgs, _ = build_hybrid_fill_request(s, plan)
                resp = call_json_fill_sdk(
                    client,
                    spec["model"],
                    msgs,
                    temperature=float(spec.get("temperature") if spec.get("temperature") is not None else 0.0),
                    max_tokens=spec.get("max_tokens") or 10000,
                    timeout=float(spec.get("timeout") or 120.0),
                    max_retries=int(spec.get("max_retries") if spec.get("max_retries") is not None else 4),
                    json_mode=not bool(spec.get("no_json_mode", False)),
                    reasoning_effort=spec.get("reasoning_effort", "medium"),
                    extra_body=spec.get("extra_body"),
                )
            return s.sample_id, _resp_to_cache_row(s.sample_id, mode, plan.missing, resp)

        concurrency = max(1, int(spec.get("concurrency") or 8))
        with open(cache_path, "a", encoding="utf-8") as append_fh:
            if concurrency == 1 or fake_map is not None:
                for s in todo:
                    sid, crow = _run_one(s)
                    existing_cache[sid] = crow
                    append_fh.write(json.dumps(crow, default=str) + "\n")
                    append_fh.flush()
                    done_count += 1
                    if crow.get("error"):
                        err_count += 1
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futs = [pool.submit(_run_one, s) for s in todo]
                    for fut in concurrent.futures.as_completed(futs):
                        sid, crow = fut.result()
                        with write_lock:
                            existing_cache[sid] = crow
                            append_fh.write(json.dumps(crow, default=str) + "\n")
                            append_fh.flush()
                            done_count += 1
                            if crow.get("error"):
                                err_count += 1
                            if not quiet and (done_count % 50 == 0 or done_count == len(todo)):
                                print(
                                    f"  [{spec['slug']}:{mode}:{split_name}] "
                                    f"{done_count}/{len(todo)} fill calls done (errors={err_count})",
                                    flush=True,
                                )

        _write_ordered_preds(cache_path, [s.sample_id for s in fill_samples], existing_cache)

    # Assemble final merged predictions for all samples
    final_preds: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        sid = s.sample_id
        plan = plans[sid]
        fill_resp = _cache_row_to_resp(existing_cache[sid]) if sid in existing_cache else None
        final_preds[sid] = finalize_hybrid_fill_row(
            plan,
            lean_by_id.get(sid) or {"sample_id": sid, "hit": False},
            fill_resp,
            full_llm_by_id.get(sid),
            model=spec["model"],
        )

    preds_out_path = out_dir / f"{mode}_{split_name}_preds.jsonl"
    _write_ordered_preds(preds_out_path, ids, final_preds)

    summary = compact_system_summary(samples, raw_rows, final_preds)
    return final_preds, summary


def _print_comparison_table(split_name: str, slug: str, rows: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n=== {split_name} ({slug}) ===")
    header = (
        f"{'System':<22} {'exF':>7} {'Nrm':>7} {'Cmp':>7} {'pF1':>7} "
        f"{'falA':>7} {'Lat(ms)':>9} {'Tok':>7} {'LLM%':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, s in rows.items():
        cx = s.get("by_complexity") or {}
        nrm = (cx.get("normal") or {}).get("exact_F", 0.0)
        cmp_ = (cx.get("complex") or {}).get("exact_F", 0.0)
        lat = (s.get("latency_ms") or {}).get("mean", 0.0)
        toks = s.get("tokens") or {}
        tok_mean = float(toks.get("prompt_tokens_mean", 0.0)) + float(toks.get("completion_tokens_mean", 0.0))
        llm_pct = 100.0 * float(toks.get("n_llm_calls", 0)) / max(int(s.get("n", 1)), 1)
        print(
            f"{name:<22} {s['exact_F']:>7.3f} {nrm:>7.3f} {cmp_:>7.3f} {s['param_F1']:>7.3f} "
            f"{s['false_hit_on_A']:>7.3f} {lat:>9.1f} {tok_mean:>7.1f} {llm_pct:>6.1f}%"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HYBRID_FILL and HYBRID_FILL+SCHEMA on v3.3 splits.")
    p.add_argument("--config", type=str, default=None, help="YAML model config (e.g. configs/llm_eval_v3_3.yaml)")
    p.add_argument("--only", nargs="+", default=None, help="Subset of model slugs from --config")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--model-env", type=str, default="BASE_LLM_MODEL")
    p.add_argument("--slug", type=str, default=None, help="Directory slug matching results_dir/llm/<slug>")
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--base-url-env", type=str, default="BASE_URL")
    p.add_argument("--api-key-env", type=str, default="API_KEY")
    p.add_argument("--default-api-key", type=str, default=None)
    p.add_argument("--eval-dir", type=str, default="data/eval_v3_3")
    p.add_argument("--results-dir", type=str, default="results/v3_3")
    p.add_argument("--splits", nargs="+", default=["fair_hermes_eval", "fair_xlam_eval"])
    p.add_argument("--modes", nargs="+", choices=list(HYBRID_MODES), default=list(HYBRID_MODES))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--overwrite-cache", action="store_true", help="Clear existing fill cache before running")
    p.add_argument("--fill-optional", action="store_true")
    p.add_argument("--fake-responses", type=str, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--reasoning-effort", type=str, default="medium", help="Reasoning effort for reasoning models (default: medium; pass 'none' to omit)")
    p.add_argument("--no-json-mode", action="store_true", help="Disable response_format=json_object")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_dotenv(REPO_ROOT / ".env")
    args = parse_args(argv)

    eval_dir = Path(args.eval_dir)
    results_dir = Path(args.results_dir)
    manifest = verify_eval_manifest(eval_dir, args.splits)
    specs = _resolve_model_specs(args)

    fake_map: Optional[Dict[str, Dict[str, Any]]] = None
    if args.fake_responses:
        fake_map = {
            str(r["sample_id"]): r
            for r in load_jsonl(args.fake_responses)
            if r.get("sample_id") is not None
        }

    for spec in specs:
        spec["no_json_mode"] = bool(args.no_json_mode)
        spec["reasoning_effort"] = (
            None
            if args.reasoning_effort is None or str(args.reasoning_effort).lower() == "none"
            else args.reasoning_effort
        )
        slug = spec["slug"]
        out_dir = results_dir / "hybrid_fill" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        model_report: Dict[str, Any] = {
            "slug": slug,
            "model": spec["model"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "splits": {},
        }

        for split_name in args.splits:
            split_path = eval_dir / manifest["splits"][split_name]["filename"]
            samples, raw_rows = load_samples(split_path)
            if args.limit is not None and args.limit > 0:
                samples = samples[: int(args.limit)]
                raw_rows = raw_rows[: int(args.limit)]
            ids = [s.sample_id for s in samples]

            lean_by_id = _load_spancall_preds(results_dir, split_name)
            full_llm_by_id = _load_full_llm_preds(results_dir, slug, split_name)

            split_table: Dict[str, Dict[str, Any]] = {
                "SpanCall": compact_system_summary(samples, raw_rows, lean_by_id),
            }
            if full_llm_by_id:
                router_preds, _ = merge_with_fallback(lean_by_id, full_llm_by_id, ids)
                rs_preds = merge_router_schema(samples, lean_by_id, full_llm_by_id)
                split_table["LLM-only"] = compact_system_summary(samples, raw_rows, full_llm_by_id)
                split_table["Router"] = compact_system_summary(samples, raw_rows, router_preds)
                split_table["Router+Schema"] = compact_system_summary(samples, raw_rows, rs_preds)

            for mode in args.modes:
                _, mode_summary = evaluate_hybrid_mode_on_split(
                    spec,
                    mode,
                    split_name,
                    split_path,
                    results_dir,
                    out_dir,
                    limit=args.limit,
                    retry_errors=args.retry_errors,
                    overwrite_cache=args.overwrite_cache,
                    fill_optional=args.fill_optional,
                    fake_map=fake_map,
                    quiet=args.quiet,
                )
                label = "HYBRID_FILL" if mode == "hybrid_fill" else "HYBRID_FILL+Schema"
                split_table[label] = mode_summary

            model_report["splits"][split_name] = split_table
            if not args.quiet:
                _print_comparison_table(split_name, slug, split_table)

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(model_report, fh, indent=2)
        if not args.quiet:
            print(f"\nWrote {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
