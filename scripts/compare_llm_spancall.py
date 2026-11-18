#!/usr/bin/env python3
"""Apples-to-apples E2E comparison: a hosted LLM vs the SpanCall fast path.

The user requirement this exists for: *"i need the e2e to be consistent so that i
can compare with a LLM"*. Two parts of that are easy to get wrong, so both are
pinned here:

1. **Same scoring code.** Every system is scored by
   :class:`spancall.eval.toolcall.evaluator.ToolCallEvaluator` on the *same*
   samples, in the same order, with the same ``payload_equal`` /
   ``multi_call_equal`` semantics the notebook uses for Lean predictions.
   Nothing is re-derived here.

2. **Same three systems on every row.** A fast path that abstains is not a
   system on its own -- in production the router falls back to the LLM. So this
   script reports *three* numbers, and the third is the one that is actually
   comparable to `llm_only`:

   ``llm_only``
       The LLM answers every row. Upper bound on quality, worst on cost.
   ``lean_fast_path``
       Only the fast path; an abstain scores as a miss (which is exactly what
       ``exact_payload_match`` already does for ``payload=None``). This is the
       number that must never be quoted as "E2E", because it silently gives
       Lean credit for rows it declined to answer.
   ``lean_plus_fallback``
       The production configuration: Lean when it commits, LLM when it
       abstains. Built as a merged prediction list and scored by the same
       evaluator, so it is directly comparable to ``llm_only`` row for row.

Cost is reported as measured LLM tokens plus the router's theoretical expected
cost. No price table is hardcoded: pass ``--price-in-per-1k`` /
``--price-out-per-1k`` if you want USD, because inventing a per-provider price
would make the cost column quietly wrong for every provider except one.

Keys never live in this file. ``--api_key_env`` names an environment variable;
the script refuses to call out when it is unset, and never echoes the value.

Usage::

    .venv/bin/python scripts/compare_llm_spancall.py \\
        --preds results/lean_v3/preds_eval_twostep.jsonl \\
        --split results/lean_v3/subset_hermes_eval.jsonl \\
        --model qwen3.7-flash --api_key_env OPENAI_API_KEY \\
        --out results/lean_v3/llm_parity.json

Offline smoke test (no network, no key) -- ``--fake-responses`` replays a JSONL
of canned tool calls, which is how the tests exercise the merge/pivot logic::

    .venv/bin/python scripts/compare_llm_spancall.py \\
        --preds preds.jsonl --split subset.jsonl \\
        --fake-responses canned.jsonl --out /tmp/parity.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spancall.data.schema import UnifiedSample  # noqa: E402
from spancall.eval.cost import CostModel  # noqa: E402
from spancall.eval.toolcall.evaluator import ToolCallEvaluator  # noqa: E402

# --------------------------------------------------------------------------- #
# Slice definitions — deliberately identical to the v3 notebook's §4.0b / §5c
# --------------------------------------------------------------------------- #
#: ``(lo, hi, label)``, inclusive. Copied from the notebook's ``TOOL_BUCKET_EDGES``
#: on purpose: a parity table that buckets differently from the notebook's pivot
#: cannot be read side by side with it.
TOOL_BUCKET_EDGES: List[Tuple[int, Optional[int], str]] = [
    (1, 2, "1-2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, None, "11+"),
]

#: Buckets can be sparse (an eval slice may hold no 11+ tool row at all); the
#: pivot always prints every label so a missing cell reads as ``-`` rather than
#: as a row that vanished between runs.
BUCKET_LABELS: List[str] = [b[2] for b in TOOL_BUCKET_EDGES] + ["<1"]


def _tool_bucket(n_tools: Any) -> str:
    """Map a row's distractor-pool size onto its bucket label.

    Buckets are keyed on ``len(sample.tools)`` -- the set the system has to pick
    from -- not on ``len(ground_truth_calls)``. The latter is 1 in a
    ``SINGLE_CALL_ONLY`` split, so bucketing on it would collapse every row into
    one cell and hide the effect distractor count actually has.
    """
    n = int(n_tools or 0)
    for lo, hi, label in TOOL_BUCKET_EDGES:
        if n >= lo and (hi is None or n <= hi):
            return label
    return "<1"


def _corpus_of_row(row: Dict[str, Any]) -> str:
    """Corpus label from ``source_dataset``, falling back to the id prefix."""
    sd = str((row or {}).get("source_dataset") or "")
    if sd.startswith("xlam"):
        return "xlam"
    if sd.startswith("hermes"):
        return "hermes"
    return "xlam" if str((row or {}).get("sample_id") or "").startswith("xlam") else "hermes"


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_samples(path: str) -> Tuple[List[UnifiedSample], List[Dict[str, Any]]]:
    """Load a canonical split, returning both the dataclasses and the raw rows.

    The raw rows are kept because slice labels (``source_dataset``, ``tools``)
    live on the JSON, and re-deriving them from the dataclass would mean
    re-implementing the corpus heuristic twice.
    """
    raw = load_jsonl(path)
    return [UnifiedSample.from_dict(r) for r in raw], raw


# --------------------------------------------------------------------------- #
# LLM side
# --------------------------------------------------------------------------- #
class LLMResponse:
    """One tool-calling completion, in the shape the evaluator wants."""

    __slots__ = ("tool_calls", "text", "prompt_tokens", "completion_tokens", "latency_ms", "error")

    def __init__(
        self,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        text: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        self.tool_calls = list(tool_calls or [])
        self.text = text
        self.prompt_tokens = int(prompt_tokens)
        self.completion_tokens = int(completion_tokens)
        self.latency_ms = float(latency_ms)
        self.error = error


def _openai_tools(sample: UnifiedSample) -> List[Dict[str, Any]]:
    """OpenAI function schemas for one sample's candidate tools.

    Reuses the agent harness's converter verbatim. A second converter here would
    drift from the one the agentic notebook uses, and then the LLM would be
    measured against a different tool description than in production.
    """
    from spancall.agent.backends.litellm_adapter import _tool_definitions_to_openai

    return _tool_definitions_to_openai(list(sample.tools or []))


def _messages_for(sample: UnifiedSample) -> List[Dict[str, str]]:
    """The prompt the LLM sees.

    The fast path receives ``recent_context`` through ``<|ctx_recent|>`` marker
    rows and the clock through the template. The LLM has no such channel, so the
    same information is rendered as plain lines in the user turn. Both systems
    therefore see the same facts -- which is the whole point of a fair
    comparison -- even though their transports differ.
    """
    parts: List[str] = [sample.utterance]
    recent = [r for r in (sample.recent_context or []) if isinstance(r, dict)]
    if recent:
        parts.append("")
        parts.append("Known context:")
        for r in recent:
            parts.append(f"- {r.get('key', '')} = {r.get('value', '')}")
    if sample.current_time_iso:
        parts.append("")
        parts.append(f"Current time: {sample.current_time_iso}")
    parts.append("")
    parts.append("Call the right tool with the correct arguments.")
    return [{"role": "user", "content": "\n".join(parts)}]


def _call_litellm(
    model: str,
    api_key: str,
    api_base: Optional[str],
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    temperature: float,
    timeout: float,
) -> LLMResponse:
    """One tool-calling completion, capturing *every* returned tool call.

    ``UniversalAPIBackend.generate_tool_call`` returns only ``tool_calls[0]``.
    That is fine for the agent loop (which executes one call per turn) but wrong
    here: ``call_exact_set_match`` would then blame the LLM for a second call it
    actually emitted. So the calls are read straight off the response.
    """
    from spancall.agent.backends.litellm_adapter import _import_litellm

    litellm = _import_litellm()
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
        "api_key": api_key,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    t0 = time.perf_counter()
    try:
        resp = litellm.completion(**kwargs)
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
        # Recorded per row instead of raised: a 400 on one malformed schema must
        # not discard the other 1999 rows' worth of paid-for tokens.
        return LLMResponse(latency_ms=(time.perf_counter() - t0) * 1000, error=f"{type(exc).__name__}: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    try:
        message = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        return LLMResponse(latency_ms=latency_ms, error=f"malformed response: {exc}")
    usage = resp.get("usage") or {}
    if not isinstance(usage, dict):
        try:
            usage = usage.model_dump()
        except AttributeError:
            usage = {}

    calls: List[Dict[str, Any]] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                # Keep the raw text so the row scores as a wrong value rather
                # than vanishing from the denominator.
                args = {"_raw": args}
        name = fn.get("name")
        if not name:
            continue
        calls.append({"tool_name": name, "payload": dict(args or {})})

    return LLMResponse(
        tool_calls=calls,
        text=message.get("content") or "",
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        latency_ms=latency_ms,
    )


def _replay_fake(fake: Dict[str, Dict[str, Any]], sample_id: str) -> LLMResponse:
    """Deterministic stand-in for a real completion (offline smoke tests)."""
    row = fake.get(sample_id)
    if row is None:
        return LLMResponse(error="no canned response")
    calls = [
        {"tool_name": c.get("tool_name"), "payload": dict(c.get("arguments") or c.get("payload") or {})}
        for c in (row.get("tool_calls") or [])
        if isinstance(c, dict) and c.get("tool_name")
    ]
    usage = row.get("usage") or {}
    return LLMResponse(
        tool_calls=calls,
        text=str(row.get("text") or ""),
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        latency_ms=float(row.get("latency_ms") or 0.0),
    )


def _llm_pred_row(sample_id: str, resp: LLMResponse) -> Dict[str, Any]:
    """Shape one LLM result as an evaluator prediction.

    ``hit`` is ``bool(tool_calls)``: the LLM "committed" whenever it named a
    tool. That makes ``hit_rate`` mean coverage rather than accuracy for the LLM
    arm -- which is why the printed table labels the columns per system instead
    of reusing Lean's wording.
    """
    calls = [{"tool_name": c["tool_name"], "payload": c.get("payload") or {}, "status": "hit"} for c in resp.tool_calls]
    first = calls[0] if calls else None
    return {
        "sample_id": sample_id,
        "hit": bool(calls),
        "status": "hit" if calls else "abstain",
        "tool_name": first["tool_name"] if first else None,
        "payload": first["payload"] if first else None,
        "payload_partial": first["payload"] if first else None,
        "calls": calls,
        "abstain_reason": None if calls else (resp.error or "no_tool_call"),
        "latency_ms": resp.latency_ms,
        # Usage rides in ``features`` so every downstream token sum reads one
        # place for both arms, and so the emitted LLM JSONL stays re-scorable by
        # ``scripts/eval_toolcall.py`` (it ignores unknown keys).
        "features": {
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "llm_error": resp.error,
        },
    }


def _lean_pred_row(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a Lean ``predict_split`` row without changing its meaning."""
    out = dict(pred)
    out.setdefault("sample_id", pred.get("sample_id"))
    out["hit"] = bool(pred.get("hit"))
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _metrics_dict(m: Any) -> Dict[str, Any]:
    """Serialise metrics exactly as ``scripts/eval_toolcall.py`` does."""
    return {
        "total_samples": m.total_samples,
        "tool_accuracy": m.tool_accuracy,
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


def score(
    samples: Sequence[UnifiedSample],
    preds_by_id: Dict[str, Dict[str, Any]],
    ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Score one system on one slice, aligned by ``sample_id``.

    The evaluator indexes predictions positionally, so the prediction list is
    rebuilt from ``ids`` every time. Passing a shorter list would silently
    re-align every later row onto the wrong sample.
    """
    sel = list(zip(samples, ids)) if ids is not None else [(s, s.sample_id) for s in samples]
    if not sel:
        return {}
    sub_samples = [s for s, _ in sel]
    sub_preds = [
        preds_by_id.get(sid) or {"sample_id": sid, "hit": False, "payload": None}
        for _, sid in sel
    ]
    return _metrics_dict(ToolCallEvaluator().evaluate(sub_samples, sub_preds))


def merge_with_fallback(
    lean_by_id: Dict[str, Dict[str, Any]],
    llm_by_id: Dict[str, Dict[str, Any]],
    ids: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Production routing: Lean when it commits, LLM when it abstains.

    Returned as a plain prediction dict so the *same* evaluator scores it. Doing
    the arithmetic by hand (``lean_exact * h + llm_exact * (1-h)``) would be
    wrong on every non-linear metric -- ``param_f1`` and ``call_exact_set_match``
    both pool numerators and denominators across rows, so they do not decompose
    that way.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    n_fallback = 0
    for sid in ids:
        lean = lean_by_id.get(sid) or {}
        if lean.get("hit"):
            merged[sid] = lean
            continue
        n_fallback += 1
        llm = llm_by_id.get(sid)
        if llm is None:
            merged[sid] = {"sample_id": sid, "hit": False, "payload": None}
            continue
        # The LLM row is adopted as-is; its own ``hit`` value describes the LLM,
        # not the router, and the router did commit to an answer here.
        row = dict(llm)
        row["routed_from"] = "lean_abstain"
        merged[sid] = row
    return merged, n_fallback


# --------------------------------------------------------------------------- #
# Slicing / reporting
# --------------------------------------------------------------------------- #
def _slices(raw_rows: Sequence[Dict[str, Any]], ids: Sequence[str]) -> Dict[str, List[str]]:
    """``{"<corpus>|<bucket>": [sample_id, ...]}`` for every populated cell."""
    by_key: Dict[str, List[str]] = {}
    for raw, sid in zip(raw_rows, ids):
        corpus = _corpus_of_row(raw)
        bucket = _tool_bucket(len((raw or {}).get("tools") or []))
        by_key.setdefault(f"{corpus}|{bucket}", []).append(sid)
    return by_key


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{nd}f}"


def _print_table(
    systems: Dict[str, Dict[str, Any]],
    order: Sequence[str],
) -> None:
    """One row per system over the whole slice."""
    print("")
    print("=" * 96)
    print("E2E PARITY — same samples, same evaluator, same order")
    print("=" * 96)
    header = f"{'system':<22}{'n':>6}{'exact_F':>10}{'hit_F':>9}{'param_F1':>10}{'call_set':>10}{'lat_ms':>10}{'tok_in':>10}{'tok_out':>9}"
    print(header)
    print("-" * len(header))
    for name in order:
        m = systems.get(name) or {}
        if not m:
            print(f"{name:<22}{'-':>6}")
            continue
        f = (m.get("metrics_by_stratum") or {}).get("F", {})
        tim = m.get("timing") or {}
        # Token totals live in ``cost``, not ``timing``: latency is per-row and
        # tokens are per-run, and mixing them made every token column print 0.
        cost = m.get("cost") or {}
        print(
            f"{name:<22}"
            f"{int(m.get('total_samples', 0)):>6}"
            f"{_fmt(f.get('exact_payload_match')):>10}"
            f"{_fmt(f.get('hit_rate')):>9}"
            f"{_fmt(f.get('param_f1')):>10}"
            f"{_fmt(m.get('call_exact_set_match')):>10}"
            f"{_fmt(tim.get('mean_latency_ms'), 1):>10}"
            f"{int(cost.get('prompt_tokens', 0) or 0):>10}"
            f"{int(cost.get('completion_tokens', 0) or 0):>9}"
        )
    print("-" * len(header))
    print("exact_F/hit_F/param_F1 are stratum F. lean_fast_path counts an abstain as")
    print("a miss; lean_plus_fallback is the only Lean row comparable to llm_only.")


def _print_pivot(
    systems: Dict[str, Dict[str, Any]],
    order: Sequence[str],
) -> None:
    """``exact_F`` per (corpus x tool-count bucket), one column per system."""
    pivot = {}
    for name in order:
        pivot[name] = (systems.get(name) or {}).get("pivot") or {}
    if not any(pivot.values()):
        return
    keys = sorted({k for p in pivot.values() for k in p})
    if not keys:
        return
    print("")
    title = "EXACT_F BY CORPUS x TOOL-COUNT BUCKET"
    print(title)
    print("-" * (24 + 12 * len(order)))
    print(f"{'slice':<24}" + "".join(f"{n[:10]:>12}" for n in order))
    for k in keys:
        corpus, bucket = k.split("|", 1)
        label = f"{corpus} {bucket}"
        cells = []
        n = 0
        for name in order:
            cell = pivot[name].get(k)
            if cell:
                cells.append(f"{_fmt(cell.get('exact_payload_match')):>12}")
                if not n:
                    # ``_metrics_dict`` reports the row count as
                    # ``total_samples``; stratum-level ``n`` lives inside
                    # ``metrics_by_stratum`` and is the *stratum* count.
                    n = int(cell.get("total_samples") or 0)
            else:
                # "-" means "no row in this cell", which is not the same as a
                # 0.0 score -- a bucket that no system could answer must not be
                # read as a bucket every system failed.
                cells.append(f"{'-':>12}")
        print(f"{label:<24}" + "".join(cells) + f"   (n={n})")


def _print_cost(systems: Dict[str, Dict[str, Any]], order: Sequence[str]) -> None:
    rows = [(n, (systems.get(n) or {}).get("cost") or {}) for n in order]
    if not any(c for _, c in rows):
        return
    print("")
    print("COST (measured tokens; USD only when --price-in-per-1k / --price-out-per-1k are given)")
    print("-" * 96)
    print(f"{'system':<22}{'llm_calls':>11}{'fallback_rate':>15}{'tok_in':>10}{'tok_out':>9}{'usd':>12}{'theo_cost':>12}")
    for name, c in rows:
        if not c:
            continue
        usd = c.get("usd")
        theo = c.get("theoretical_expected_cost")
        print(
            f"{name:<22}"
            f"{int(c.get('n_llm_calls', 0)):>11}"
            f"{_fmt(c.get('fallback_rate')):>15}"
            f"{int(c.get('prompt_tokens', 0)):>10}"
            f"{int(c.get('completion_tokens', 0)):>9}"
            f"{('-' if usd is None else _fmt(usd, 4)):>12}"
            f"{_fmt(theo, 4):>12}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare a hosted LLM against the SpanCall fast path on the same rows."
    )
    p.add_argument("--preds", type=str, default=None,
                   help="Lean predictions JSONL (preds_<split>.jsonl from predict_split)")
    p.add_argument("--split", type=str, required=True,
                   help="The canonical/subset JSONL the predictions were produced from")
    p.add_argument("--model", type=str, default=None,
                   help="LLM model id passed to litellm (e.g. qwen3.7-flash)")
    p.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY",
                   help="NAME of the env var holding the key (never the key itself)")
    p.add_argument("--api_base_env", type=str, default="",
                   help="Optional NAME of an env var holding an OpenAI-compatible base URL")
    p.add_argument("--limit", type=int, default=None, help="Only the first N rows")
    p.add_argument("--out", type=str, default="results/llm_parity.json",
                   help="Where to write the comparison JSON")
    p.add_argument("--llm-preds-out", type=str, default=None,
                   help="Where to write LLM predictions as JSONL (re-scorable with "
                        "scripts/eval_toolcall.py). Default: <out stem>_llm_preds.jsonl")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Hard cap on LLM calls (cost guard); evaluated rows beyond it "
                        "are dropped from all systems so the comparison stays aligned")
    p.add_argument("--b1a", action="store_true",
                   help="Also score the deterministic B1a regex baseline")
    p.add_argument("--fake-responses", type=str, default=None,
                   help="Offline mode: replay canned completions from this JSONL "
                        "instead of calling a provider (smoke tests / CI)")
    p.add_argument("--price-in-per-1k", type=float, default=None)
    p.add_argument("--price-out-per-1k", type=float, default=None)
    p.add_argument("--quiet", action="store_true", help="Only write the JSON")
    return p.parse_args(argv)


def _b1a_preds(samples: Sequence[UnifiedSample]) -> Dict[str, Dict[str, Any]]:
    """B1a baseline predictions, mirroring ``scripts/eval_toolcall.py``."""
    from datetime import date

    from spancall.eval.toolcall.evaluator import B1aBaseline

    baseline = B1aBaseline()
    out: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        if not s.tools:
            out[s.sample_id] = {"sample_id": s.sample_id, "hit": False, "payload": None, "tool_name": None}
            continue
        try:
            from span_template import RecentRow
            from span_template.template import build_input_multi

            from spancall.data.align import tooldef_to_template_tool

            recent = [
                RecentRow(key=str(r.get("key", "")), value=str(r.get("value", "")),
                          source=str(r.get("source", "recent")))
                for r in (s.recent_context or [])
            ]
            clock = date.fromisoformat(s.current_time_iso) if s.current_time_iso else date(2026, 1, 1)
            output = build_input_multi(
                [tooldef_to_template_tool(t) for t in s.tools],
                s.utterance,
                recent_context=recent,
                now=clock,
            )
        except Exception:  # noqa: BLE001 - a sanitize/marker rejection is an abstain
            out[s.sample_id] = {"sample_id": s.sample_id, "hit": False, "payload": None, "tool_name": None}
            continue
        if output is None or not getattr(output, "entries", None):
            out[s.sample_id] = {"sample_id": s.sample_id, "hit": False, "payload": None,
                                "tool_name": s.tools[0].name}
            continue
        r = baseline.predict(output, s.tools[0])
        out[s.sample_id] = {
            "sample_id": s.sample_id,
            "hit": bool(r["hit"]),
            "payload": r["payload"],
            "payload_partial": r["payload"],
            "tool_name": r["tool_name"],
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.preds and not args.fake_responses and not args.model:
        print("[parity] nothing to compare: pass --preds and/or --model.", file=sys.stderr)
        return 2

    samples, raw_rows = load_samples(args.split)
    if args.limit is not None:
        samples, raw_rows = samples[: args.limit], raw_rows[: args.limit]
    if not samples:
        print(f"[parity] no samples in {args.split}", file=sys.stderr)
        return 2
    ids = [s.sample_id for s in samples]

    # ---- Lean arm -------------------------------------------------------- #
    lean_by_id: Dict[str, Dict[str, Any]] = {}
    n_lean_rows = 0
    if args.preds:
        for row in load_jsonl(args.preds):
            sid = row.get("sample_id")
            if sid is None:
                continue
            lean_by_id[str(sid)] = _lean_pred_row(row)
            n_lean_rows += 1
        missing = [sid for sid in ids if sid not in lean_by_id]
        if missing:
            # Loud, because the most likely cause is scoring a subset against
            # predictions from a different (larger) split -- which would compare
            # Lean's coverage on one population against the LLM's on another.
            print(
                f"[parity] WARNING: {len(missing)}/{len(ids)} split rows have no Lean "
                f"prediction (first: {missing[0]}). Those rows score as Lean abstains.",
                file=sys.stderr,
            )

    # ---- LLM arm --------------------------------------------------------- #
    llm_by_id: Dict[str, Dict[str, Any]] = {}
    n_llm_errors = 0
    if args.fake_responses or args.model:
        fake: Dict[str, Dict[str, Any]] = {}
        if args.fake_responses:
            fake = {str(r.get("sample_id")): r for r in load_jsonl(args.fake_responses)}
        else:
            key_name = args.api_key_env
            api_key = os.environ.get(key_name)
            if not api_key:
                print(
                    f"[parity] {key_name} is not set. This script never reads a key "
                    f"from a file or a flag -- export {key_name} and re-run, or use "
                    f"--fake-responses for an offline smoke run.",
                    file=sys.stderr,
                )
                return 3
            api_base = os.environ.get(args.api_base_env) if args.api_base_env else None

        n_calls = len(samples) if args.max_rows is None else min(len(samples), int(args.max_rows))
        for idx, sample in enumerate(samples[:n_calls]):
            if idx and idx % 25 == 0 and not args.quiet:
                print(f"[parity] LLM {idx}/{n_calls} ...", file=sys.stderr)
            if fake:
                resp = _replay_fake(fake, sample.sample_id)
            else:
                resp = _call_litellm(
                    model=args.model,
                    api_key=api_key,
                    api_base=api_base,
                    messages=_messages_for(sample),
                    tools=_openai_tools(sample),
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
            if resp.error:
                n_llm_errors += 1
            llm_by_id[sample.sample_id] = _llm_pred_row(sample.sample_id, resp)

        if n_calls < len(samples):
            # Trim every arm to the rows the LLM actually covered. Keeping the
            # untested tail would hand Lean free coverage it was never compared on.
            samples, raw_rows, ids = samples[:n_calls], raw_rows[:n_calls], ids[:n_calls]

    # ---- B1a arm --------------------------------------------------------- #
    b1a_by_id = _b1a_preds(samples) if args.b1a else {}

    # ---- Systems --------------------------------------------------------- #
    systems: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def _add(name: str, preds: Dict[str, Dict[str, Any]], extra: Dict[str, Any]) -> None:
        m = score(samples, preds)
        if not m:
            return
        m = dict(m)
        m.update(extra)
        systems[name] = m
        if name not in order:
            order.append(name)

    if llm_by_id:
        tok_in = sum(int((p.get("features") or {}).get("prompt_tokens", 0)) for p in llm_by_id.values())
        tok_out = sum(int((p.get("features") or {}).get("completion_tokens", 0)) for p in llm_by_id.values())
        lat = [float(p.get("latency_ms") or 0.0) for p in llm_by_id.values()]
        cost = {
            "n_llm_calls": len(llm_by_id),
            "fallback_rate": 0.0,
            "prompt_tokens": tok_in,
            "completion_tokens": tok_out,
            "usd": _usd(tok_in, tok_out, args.price_in_per_1k, args.price_out_per_1k),
            "theoretical_expected_cost": None,
        }
        _add("llm_only", llm_by_id, {
            "timing": {"mean_latency_ms": (sum(lat) / len(lat)) if lat else 0.0},
            "cost": cost,
            "n_llm_errors": n_llm_errors,
        })

    if lean_by_id:
        lat = [float(p.get("latency_ms") or 0.0) for p in lean_by_id.values()]
        h = 0.0
        p_at_hit = 0.0
        _add("lean_fast_path", lean_by_id, {
            "timing": {"mean_latency_ms": (sum(lat) / len(lat)) if lat else 0.0},
            "cost": {
                "n_llm_calls": 0,
                "fallback_rate": 1.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "usd": 0.0,
                "theoretical_expected_cost": None,
            },
        })
        f = (systems.get("lean_fast_path", {}).get("metrics_by_stratum") or {}).get("F", {})
        h = float(f.get("hit_rate") or 0.0)
        p_at_hit = float(f.get("precision_at_hit") or 0.0)

        if llm_by_id:
            merged, n_fallback = merge_with_fallback(lean_by_id, llm_by_id, ids)
            n_rows = max(len(ids), 1)
            tok_in = sum(int((p.get("features") or {}).get("prompt_tokens", 0)) for p in llm_by_id.values())
            tok_out = sum(int((p.get("features") or {}).get("completion_tokens", 0)) for p in llm_by_id.values())
            fallback_rate = n_fallback / n_rows
            _add("lean_plus_fallback", merged, {
                "timing": {"mean_latency_ms": (sum(float(p.get("latency_ms") or 0.0) for p in merged.values()) / n_rows)},
                "cost": {
                    "n_llm_calls": n_fallback,
                    # Measured: the LLM prompt/completion totals above are the
                    # cost of answering *every* row, so the routed cost is the
                    # fraction that actually fell through. Labelled as a
                    # prorated estimate rather than passed off as measured.
                    "prompt_tokens": int(round(tok_in * fallback_rate)),
                    "completion_tokens": int(round(tok_out * fallback_rate)),
                    "prompt_tokens_measured_all_rows": tok_in,
                    "completion_tokens_measured_all_rows": tok_out,
                    "fallback_rate": fallback_rate,
                    "usd": _usd(
                        int(round(tok_in * fallback_rate)),
                        int(round(tok_out * fallback_rate)),
                        args.price_in_per_1k,
                        args.price_out_per_1k,
                    ),
                    "theoretical_expected_cost": CostModel().theoretical_expected_cost(h, p_at_hit),
                },
            })

    if b1a_by_id:
        _add("b1a_regex", b1a_by_id, {"timing": {"mean_latency_ms": 0.0}, "cost": {}})

    # ---- Slices (per system) -------------------------------------------- #
    slice_ids = _slices(raw_rows, ids)
    for name, m in systems.items():
        preds = {
            "llm_only": llm_by_id,
            "lean_fast_path": lean_by_id,
            "b1a_regex": b1a_by_id,
        }.get(name)
        if preds is None and name == "lean_plus_fallback":
            preds, _ = merge_with_fallback(lean_by_id, llm_by_id, ids)
        if not preds:
            continue
        pivot: Dict[str, Any] = {}
        for key, sids in slice_ids.items():
            sel = [s for s in samples if s.sample_id in set(sids)]
            if not sel:
                continue
            cell = score(sel, preds, ids=sids)
            if cell:
                pivot[key] = cell
        m["pivot"] = pivot

    # ---- Output ---------------------------------------------------------- #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    llm_preds_path = args.llm_preds_out or str(out_path.with_name(out_path.stem + "_llm_preds.jsonl"))
    if llm_by_id:
        with open(llm_preds_path, "w", encoding="utf-8") as fh:
            for sid in ids:
                if sid in llm_by_id:
                    fh.write(json.dumps(llm_by_id[sid], default=str) + "\n")

    payload = {
        "meta": {
            "split": args.split,
            "preds": args.preds,
            "model": args.model,
            "api_key_env": args.api_key_env if not args.fake_responses else None,
            "fake_responses": args.fake_responses,
            "limit": args.limit,
            "max_rows": args.max_rows,
            "temperature": args.temperature,
            "n_eval_rows": len(ids),
            "n_lean_rows_in_file": n_lean_rows,
            "n_llm_errors": n_llm_errors,
            "price_in_per_1k": args.price_in_per_1k,
            "price_out_per_1k": args.price_out_per_1k,
            "tool_bucket_edges": TOOL_BUCKET_EDGES,
            "systems": order,
        },
        "systems": systems,
        "rows": [
            {
                "sample_id": sid,
                "corpus": _corpus_of_row(raw),
                "n_tools": len((raw or {}).get("tools") or []),
                "bucket": _tool_bucket(len((raw or {}).get("tools") or [])),
                "lean": lean_by_id.get(sid),
                "llm": llm_by_id.get(sid),
                "b1a": b1a_by_id.get(sid),
            }
            for sid, raw in zip(ids, raw_rows)
        ],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")

    if not args.quiet:
        print(f"[parity] {len(ids)} rows · systems: {', '.join(order) or 'none'}")
        _print_table(systems, order)
        _print_pivot(systems, order)
        _print_cost(systems, order)
        print("")
        print(f"[parity] wrote {out_path}")
        if llm_by_id:
            print(f"[parity] LLM predictions -> {llm_preds_path} (re-score: scripts/eval_toolcall.py "
                  f"--test_data {args.split} --predictions {llm_preds_path})")
        if n_llm_errors:
            print(f"[parity] {n_llm_errors} LLM call(s) failed; they score as abstains.", file=sys.stderr)
    return 0


def _usd(tok_in: int, tok_out: int, price_in: Optional[float], price_out: Optional[float]) -> Optional[float]:
    """USD from token counts, or ``None`` when no price was supplied.

    Returning ``None`` (rather than 0.0) is deliberate: a zero would read as
    "free" in a comparison table.
    """
    if price_in is None and price_out is None:
        return None
    return (tok_in / 1000.0) * float(price_in or 0.0) + (tok_out / 1000.0) * float(price_out or 0.0)


if __name__ == "__main__":
    sys.exit(main())