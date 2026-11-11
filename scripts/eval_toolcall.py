#!/usr/bin/env python3
"""CLI entrypoint for offline tool-call evaluation.

Evaluates SpanCall predictions (or the deterministic B1a regex baseline) against
the canonical split JSONL files (e.g. ``hermes_eval.jsonl``) and writes a
metrics report ``results/toolcall_metrics_<split>.json``.

Usage examples:
    .venv/bin/python scripts/eval_toolcall.py --test_data data/processed/hermes_eval.jsonl
    .venv/bin/python scripts/eval_toolcall.py --test_data ... --baseline b1a
    .venv/bin/python scripts/eval_toolcall.py --test_data ... --predictions preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spancall.data.align import tooldef_to_template_tool  # noqa: E402
from spancall.data.schema import Stratum, UnifiedSample  # noqa: E402
from spancall.eval.toolcall.evaluator import B1aBaseline, ToolCallEvaluator  # noqa: E402

try:  # template rendering is optional for the predictions-only path
    from span_template import RecentRow  # noqa: E402
    from span_template.sanitize import SanitizeError  # noqa: E402
    from span_template.template import build_input_multi  # noqa: E402
    _TEMPLATE_AVAILABLE = True
except ImportError:  # pragma: no cover
    RecentRow = None  # type: ignore[assignment,misc]
    SanitizeError = None  # type: ignore[assignment,misc]
    build_input_multi = None  # type: ignore[assignment]
    _TEMPLATE_AVAILABLE = False

# Deterministic fallback clock when a sample omits current_time_iso (R4).
_FALLBACK_NOW = date(2026, 1, 1)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpanCall Offline Tool-Call Evaluation")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to a canonical split JSONL (UnifiedSample records)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Optional JSONL of prediction dicts "
                             "({sample_id, hit, payload, tool_name, accept_probability}). "
                             "When omitted, a random baseline is used.")
    parser.add_argument("--baseline", type=str, default=None, choices=["b1a"],
                        help="Run the B1a deterministic regex baseline instead of model predictions")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write the metrics JSON (default: results/toolcall_metrics_*.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N samples (sanity / smoke use)")
    return parser.parse_args(argv)


def load_samples(path: str) -> List[UnifiedSample]:
    samples: List[UnifiedSample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(UnifiedSample.from_dict(json.loads(line)))
    return samples


def load_predictions(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _random_baseline(samples: List[UnifiedSample]) -> List[Dict[str, Any]]:
    """Deterministic zero-hit baseline (all abstain) for pipeline verification."""
    return [{"sample_id": s.sample_id, "hit": False, "payload": None} for s in samples]


def _render_b1a(s: UnifiedSample, now: Any) -> Optional[Any]:
    """Render the packed multi-tool template for B1a extraction.

    Mirrors the resolver: build the ``SpanTemplateOutput`` over *all* candidate
    tools, with R7 sanitization. Returns ``None`` when the utterance or recent
    values trip marker-injection (the resolver would abstain, so B1a abstains).
    """
    if not _TEMPLATE_AVAILABLE:
        return None
    if not s.tools:
        return None
    try:
        recent = [
            RecentRow(key=str(r.get("key", "")), value=str(r.get("value", "")),
                      source=str(r.get("source", "recent")))
            for r in (s.recent_context or [])
        ]
        clock = date.fromisoformat(s.current_time_iso) if s.current_time_iso else (
            date.fromisoformat(now) if isinstance(now, str) else now)
        return build_input_multi(
            [tooldef_to_template_tool(t) for t in s.tools],
            s.utterance,
            recent_context=recent,
            now=clock or _FALLBACK_NOW,
        )
    except (ValueError, SanitizeError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    samples = load_samples(args.test_data)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        print(f"[SpanCall Eval: ToolCall] no samples in {args.test_data}")
        return 2

    evaluator = ToolCallEvaluator()

    if args.baseline == "b1a":
        baseline = B1aBaseline()
        rows = []
        for s in samples:
            if not s.tools:
                rows.append({"sample_id": s.sample_id, "hit": False, "payload": None, "tool_name": None})
                continue
            output = _render_b1a(s, None)
            if output is None or not getattr(output, "entries", None):
                rows.append({"sample_id": s.sample_id, "hit": False, "payload": None,
                             "tool_name": s.tools[0].name})
                continue
            out = baseline.predict(output, s.tools[0])
            rows.append({
                "sample_id": s.sample_id,
                "hit": out["hit"],
                "payload": out["payload"],
                "tool_name": out["tool_name"],
            })
        metrics = evaluator.evaluate(samples, rows)
        print(f"[SpanCall Eval: ToolCall] B1a baseline on {len(samples)} rows")
    elif args.predictions:
        preds = load_predictions(args.predictions)
        if len(preds) != len(samples):
            print(f"[SpanCall Eval: ToolCall] warning: {len(preds)} predictions vs {len(samples)} samples")
        metrics = evaluator.evaluate(samples, preds)
    else:
        metrics = evaluator.evaluate(samples, _random_baseline(samples))
        print("[SpanCall Eval: ToolCall] no predictions provided — using all-abstain baseline")

    out_path = args.output
    if not out_path:
        stem = Path(args.test_data).stem
        out_path = f"results/toolcall_metrics_{stem}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_metrics_to_dict(metrics), fh, indent=2, default=str)
        fh.write("\n")

    f_metrics = metrics.metrics_by_stratum.get("F", {})
    f_n = int(f_metrics.get("n", 0))
    f_exact = f_metrics.get("exact_payload_match", 0.0)
    f_hit = f_metrics.get("hit_rate", 0.0)
    print(f"[SpanCall Eval: ToolCall] wrote {out_path}")
    print(f"[SpanCall Eval: ToolCall] Stratum F (N={f_n}): exact_payload_match={f_exact:.3f} hit_rate={f_hit:.3f} | "
          f"Stratum A: false_hit_rate={metrics.false_hit_rate_on_a:.4f} | "
          f"Overall exact={metrics.exact_payload_match:.3f} hit={metrics.hit_rate:.3f} ECE={metrics.ece:.4f}")
    # R0.4 / J6: the call-0-only number above cannot see a dropped or invented
    # second call, so surface the set-level view and the call-count histograms.
    print(f"[SpanCall Eval: ToolCall] call set exact={getattr(metrics, 'call_exact_set_match', 0.0):.3f} "
          f"call count acc={getattr(metrics, 'call_count_acc', 0.0):.3f} | "
          f"GT n_calls={getattr(metrics, 'n_calls_hist', {})} "
          f"pred n_calls={getattr(metrics, 'pred_n_calls_hist', {})}")
    print("[SpanCall Eval: ToolCall] OK")
    return 0


def _metrics_to_dict(m: Any) -> Dict[str, Any]:
    return {
        "total_samples": m.total_samples,
        "tool_accuracy": m.tool_accuracy,
        # R0.4 / J6: set-level call metrics. ``getattr`` defaults keep an older
        # metrics object (or a stub) serialisable instead of raising.
        "call_exact_set_match": getattr(m, "call_exact_set_match", 0.0),
        "call_count_acc": getattr(m, "call_count_acc", 0.0),
        "n_calls_hist": getattr(m, "n_calls_hist", {}),
        "pred_n_calls_hist": getattr(m, "pred_n_calls_hist", {}),
        "per_param_accuracy": m.per_param_accuracy,
        "exact_payload_match": m.exact_payload_match,
        "hit_rate": m.hit_rate,
        "precision_at_hit": m.precision_at_hit,
        "false_hit_rate_on_a": m.false_hit_rate_on_a,
        "partial_extraction_rate": getattr(m, "partial_extraction_rate", 0.0),
        "param_precision": getattr(m, "param_precision", 0.0),
        "param_recall": getattr(m, "param_recall", 0.0),
        "param_f1": getattr(m, "param_f1", 0.0),
        "error_wrong_tool": m.error_wrong_tool,
        "error_wrong_value": m.error_wrong_value,
        "error_entity_swap": m.error_entity_swap,
        "error_duplicate_span": m.error_duplicate_span,
        "error_date_misparse": m.error_date_misparse,
        "error_extra_call": m.error_extra_call,
        "error_missing_call": m.error_missing_call,
        "metrics_by_stratum": m.metrics_by_stratum,
        "abstain_reasons": m.abstain_reasons,
        "ece": m.ece,
        "risk_coverage": m.risk_coverage,
        "errors_by_sample": m.errors_by_sample,
    }


if __name__ == "__main__":
    sys.exit(main())
