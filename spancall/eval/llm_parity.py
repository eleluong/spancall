"""Shared LLM-vs-SpanCall parity helpers for Lean v3.3.

Provides one canonical implementation of:
- OpenAI tool-schema conversion and fair prompt rendering
- OpenAI-SDK direct calling with bounded retry/backoff
- Prediction shaping (``llm_pred_row`` / ``lean_pred_row``)
- Identical ``ToolCallEvaluator`` scoring and ``SpanCall + LLM fallback`` merging
- Corpus x tool-count bucketing, complexity slicing, and latency/token summaries
- SHA256 manifest verification for frozen ``data/eval_v3_3/`` benchmarks
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from spancall.data.schema import UnifiedSample
from spancall.eval.cost import CostModel
from spancall.eval.toolcall.evaluator import ToolCallEvaluator
from spancall.lean.eval_p1 import classify_eval_cohort, classify_sample_complexity

TOOL_BUCKET_EDGES: List[Tuple[int, Optional[int], str]] = [
    (1, 2, "1-2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, None, "11+"),
]
BUCKET_LABELS: List[str] = [b[2] for b in TOOL_BUCKET_EDGES] + ["<1"]


def tool_bucket(n_tools: Any) -> str:
    """Map candidate tool count ``len(sample.tools)`` onto its bucket label."""
    n = int(n_tools or 0)
    for lo, hi, label in TOOL_BUCKET_EDGES:
        if n >= lo and (hi is None or n <= hi):
            return label
    return "<1"


def corpus_of_row(row: Dict[str, Any]) -> str:
    """Corpus label from ``source_dataset``, falling back to ``sample_id`` prefix."""
    sd = str((row or {}).get("source_dataset") or "")
    if sd.startswith("xlam"):
        return "xlam"
    if sd.startswith("hermes"):
        return "hermes"
    return "xlam" if str((row or {}).get("sample_id") or "").startswith("xlam") else "hermes"


# --------------------------------------------------------------------------- #
# IO & Manifest Integrity
# --------------------------------------------------------------------------- #
def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_ids_hash(sample_ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for sid in sample_ids:
        h.update(str(sid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_samples(path: Path | str) -> Tuple[List[UnifiedSample], List[Dict[str, Any]]]:
    raw = load_jsonl(path)
    return [UnifiedSample.from_dict(r) for r in raw], raw


def verify_eval_manifest(
    eval_dir: Path | str,
    splits: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Verify ``manifest.json`` SHA256 hashes for all (or requested) splits.

    Raises ``FileNotFoundError`` if ``manifest.json`` or any requested split is
    missing, and ``ValueError`` if any file's SHA256 diverges from the manifest.
    """
    edir = Path(eval_dir)
    manifest_path = edir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Eval manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    files_meta = manifest.get("splits") or {}
    check_names = list(splits) if splits is not None else list(files_meta.keys())
    for name in check_names:
        if name not in files_meta:
            raise KeyError(f"Split '{name}' not found in manifest {manifest_path}")
        entry = files_meta[name]
        fpath = edir / entry["filename"]
        if not fpath.is_file():
            raise FileNotFoundError(f"Manifest split file missing: {fpath}")
        actual = sha256_file(fpath)
        expected = entry["sha256"]
        if actual != expected:
            raise ValueError(
                f"SHA256 mismatch for {fpath}: expected {expected[:12]}..., got {actual[:12]}..."
            )
    return manifest


# --------------------------------------------------------------------------- #
# LLM Transport & Shaping
# --------------------------------------------------------------------------- #
class LLMResponse:
    """One tool-calling completion in the shape ``ToolCallEvaluator`` expects."""

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


def openai_tools(sample: UnifiedSample) -> List[Dict[str, Any]]:
    """OpenAI function schemas for one sample's candidate tools."""
    from spancall.agent.backends.litellm_adapter import _tool_definitions_to_openai

    return _tool_definitions_to_openai(list(sample.tools or []))


def messages_for(sample: UnifiedSample) -> List[Dict[str, str]]:
    """Fair user prompt exposing identical utterance, recent_context, and clock."""
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


def _is_retryable_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status in (408, 429, 500, 502, 503, 504):
        return True
    if any(k in name for k in ("ratelimit", "timeout", "connection", "internalserver", "serviceunavailable")):
        return True
    if any(k in msg for k in ("429", "rate limit", "timed out", "temporarily unavailable", "502", "503", "504")):
        return True
    return False


def call_openai_sdk(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: float = 120.0,
    max_retries: int = 4,
    base_backoff: float = 1.5,
    extra_body: Optional[Dict[str, Any]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> LLMResponse:
    """Invoke an OpenAI-SDK-compatible chat completion endpoint with retry/backoff."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if extra_body:
        kwargs["extra_body"] = dict(extra_body)

    last_exc: Optional[Exception] = None
    t0 = time.perf_counter()
    resp = None
    for attempt in range(max(1, int(max_retries) + 1)):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_retries or not _is_retryable_error(exc):
                break
            delay = base_backoff * (2 ** attempt) + random.uniform(0.0, 0.5)
            sleep_fn(delay)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    if last_exc is not None or resp is None:
        return LLMResponse(
            latency_ms=latency_ms,
            error=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "empty_response",
        )

    try:
        choice = resp.choices[0]
        message = choice.message
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        return LLMResponse(latency_ms=latency_ms, error=f"malformed response: {exc}")

    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0

    calls: List[Dict[str, Any]] = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        if fn is None and isinstance(tc, dict):
            fn = tc.get("function") or {}
        name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
        args = getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not name:
            continue
        calls.append({"tool_name": str(name), "payload": dict(args or {})})

    content = getattr(message, "content", None) or ""
    return LLMResponse(
        tool_calls=calls,
        text=str(content),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )


def replay_fake(fake: Dict[str, Dict[str, Any]], sample_id: str) -> LLMResponse:
    """Deterministic stand-in for offline smoke tests / CI."""
    row = fake.get(sample_id)
    if row is None:
        return LLMResponse(error="no canned response")
    if row.get("error"):
        return LLMResponse(
            latency_ms=float(row.get("latency_ms") or 0.0),
            error=str(row["error"]),
        )
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


def llm_pred_row(sample_id: str, resp: LLMResponse, *, model: Optional[str] = None) -> Dict[str, Any]:
    """Shape one LLM completion as a ``ToolCallEvaluator`` prediction row."""
    calls = [
        {"tool_name": c["tool_name"], "payload": c.get("payload") or {}, "status": "hit"}
        for c in resp.tool_calls
    ]
    first = calls[0] if calls else None
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "hit": bool(calls),
        "status": "hit" if calls else "abstain",
        "tool_name": first["tool_name"] if first else None,
        "payload": first["payload"] if first else None,
        "payload_partial": first["payload"] if first else None,
        "calls": calls,
        "abstain_reason": None if calls else (resp.error or "no_tool_call"),
        "latency_ms": resp.latency_ms,
        "features": {
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "llm_error": resp.error,
        },
    }
    if model:
        row["model"] = model
    return row


def lean_pred_row(pred: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(pred)
    out.setdefault("sample_id", pred.get("sample_id"))
    out["hit"] = bool(pred.get("hit"))
    return out


# --------------------------------------------------------------------------- #
# Scoring, Fallback Merge & Summary Statistics
# --------------------------------------------------------------------------- #
def metrics_dict(m: Any) -> Dict[str, Any]:
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


def score_predictions(
    samples: Sequence[UnifiedSample],
    preds_by_id: Dict[str, Dict[str, Any]],
    ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Score one system on one slice, strictly aligned by ``sample_id``."""
    sel = list(zip(samples, ids)) if ids is not None else [(s, s.sample_id) for s in samples]
    if not sel:
        return {}
    sub_samples = [s for s, _ in sel]
    sub_preds = [
        preds_by_id.get(sid) or {"sample_id": sid, "hit": False, "payload": None}
        for _, sid in sel
    ]
    return metrics_dict(ToolCallEvaluator().evaluate(sub_samples, sub_preds))


def merge_with_fallback(
    lean_by_id: Dict[str, Dict[str, Any]],
    llm_by_id: Dict[str, Dict[str, Any]],
    ids: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Production router merge: SpanCall when it commits, LLM when it abstains."""
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
        row = dict(llm)
        row["routed_from"] = "lean_abstain"
        # Total latency on a fallback row is fast-path gate + LLM call
        row["latency_ms"] = float(lean.get("latency_ms") or 0.0) + float(llm.get("latency_ms") or 0.0)
        merged[sid] = row
    return merged, n_fallback


def latency_percentiles(latencies_ms: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(x) for x in latencies_ms if x is not None and math.isfinite(float(x)))
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}

    def _pct(q: float) -> float:
        idx = (len(vals) - 1) * q
        lo = int(math.floor(idx))
        hi = min(len(vals) - 1, lo + 1)
        frac = idx - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "mean": sum(vals) / len(vals),
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "p99": _pct(0.99),
    }


def token_summary(
    preds_by_id: Dict[str, Dict[str, Any]],
    ids: Sequence[str],
) -> Dict[str, Any]:
    tok_in = 0
    tok_out = 0
    n_errors = 0
    n_llm_calls = 0
    for sid in ids:
        p = preds_by_id.get(sid) or {}
        feats = p.get("features") or {}
        pi = int(feats.get("prompt_tokens", 0) or 0)
        po = int(feats.get("completion_tokens", 0) or 0)
        tok_in += pi
        tok_out += po
        if pi > 0 or po > 0 or feats.get("llm_error") is not None or p.get("routed_from") == "lean_abstain":
            n_llm_calls += 1
        if feats.get("llm_error"):
            n_errors += 1
    n = max(len(ids), 1)
    return {
        "prompt_tokens_total": tok_in,
        "completion_tokens_total": tok_out,
        "prompt_tokens_mean": tok_in / n,
        "completion_tokens_mean": tok_out / n,
        "n_llm_calls": n_llm_calls,
        "n_errors": n_errors,
    }


def slice_by_corpus_bucket(
    raw_rows: Sequence[Dict[str, Any]],
    ids: Sequence[str],
) -> Dict[str, List[str]]:
    by_key: Dict[str, List[str]] = {}
    for raw, sid in zip(raw_rows, ids):
        corpus = corpus_of_row(raw)
        bucket = tool_bucket(len((raw or {}).get("tools") or []))
        by_key.setdefault(f"{corpus}|{bucket}", []).append(sid)
    return by_key


def slice_by_complexity(
    raw_rows: Sequence[Dict[str, Any]],
    ids: Sequence[str],
) -> Dict[str, List[str]]:
    by_key: Dict[str, List[str]] = {"normal": [], "complex": []}
    for raw, sid in zip(raw_rows, ids):
        cx = classify_sample_complexity(raw)
        by_key.setdefault(cx, []).append(sid)
    return by_key


def theoretical_cost(metrics: Dict[str, Any]) -> Optional[float]:
    f = (metrics.get("metrics_by_stratum") or {}).get("F", {})
    h = float(f.get("hit_rate") or 0.0)
    p = float(f.get("precision_at_hit") or 0.0)
    return CostModel().theoretical_expected_cost(h, p)


__all__ = [
    "TOOL_BUCKET_EDGES",
    "BUCKET_LABELS",
    "tool_bucket",
    "corpus_of_row",
    "sha256_file",
    "sample_ids_hash",
    "load_jsonl",
    "load_samples",
    "verify_eval_manifest",
    "LLMResponse",
    "openai_tools",
    "messages_for",
    "call_openai_sdk",
    "replay_fake",
    "llm_pred_row",
    "lean_pred_row",
    "metrics_dict",
    "score_predictions",
    "merge_with_fallback",
    "latency_percentiles",
    "token_summary",
    "slice_by_corpus_bucket",
    "slice_by_complexity",
    "theoretical_cost",
    "classify_eval_cohort",
    "classify_sample_complexity",
]

