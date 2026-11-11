"""Offline tool-call evaluator.

Ref: docs/full_implementation_plan.md §4.2, docs/pipeline_io.md Q3–Q9.

Implements ``ToolCallEvaluator(BaseToolCallEvaluator)``:

1. **Canonical payload equality** (``payload_equal``): normalizes key ordering,
   numeric representation (``5 == 5.0``), and strips optional parameters equal
   to schema defaults before deep comparison.
2. **Stratified metrics** grouped by ``Stratum.F`` / ``Stratum.A`` / ``all``:
   exact-payload match, hit rate, precision-at-hit, and the stratum-A safety
   ``false_hit_rate_on_a`` (E7 target <= 1-2%).
3. **7-bucket error taxonomy** over every *false hit*: wrong tool, entity swap,
   duplicate span, date misparse, wrong value, extra call, missing call.
4. **Risk-coverage curve + ECE** over the scored prediction rows.
5. **B1a baseline** (``B1aBaseline``): a deterministic regex/extractor baseline
   working over the identical ``SpanTemplateOutput`` legend (exact enum match,
   single date entry match, single regex numeric match).
"""

from __future__ import annotations

from datetime import date, datetime
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from spancall.data.schema import GroundTruthCall, Stratum, UnifiedSample
from spancall.eval.toolcall.metrics import ToolCallMetrics

# --------------------------------------------------------------------------- #
# Canonical payload equality
# --------------------------------------------------------------------------- #


def _norm_number(v: Any) -> Any:
    """Normalize numeric representations: 5 == 5.0 for number params."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        return int(f) if f.is_integer() else f
    return v


def _deep_equal(a: Any, b: Any) -> bool:
    # 1. Boolean handling (including bool strings and numeric bools)
    if isinstance(a, bool) or isinstance(b, bool):
        if isinstance(a, bool) and isinstance(b, bool):
            return a == b
        bool_val = a if isinstance(a, bool) else b
        other = b if isinstance(a, bool) else a
        if isinstance(other, str):
            low = other.strip().lower()
            if low in ("true", "1", "yes", "t"):
                return bool_val is True
            if low in ("false", "0", "no", "f"):
                return bool_val is False
            return False
        if isinstance(other, (int, float)):
            return bool_val == bool(other)
        return False

    # 2. Date / datetime handling
    if isinstance(a, (date, datetime)) and isinstance(b, str):
        s_a = a.isoformat()
        s_b = b.strip()
        return s_a == s_b or str(a) == s_b or s_b.startswith(s_a) or s_a.startswith(s_b)
    if isinstance(b, (date, datetime)) and isinstance(a, str):
        s_b = b.isoformat()
        s_a = a.strip()
        return s_b == s_a or str(b) == s_a or s_a.startswith(s_b) or s_b.startswith(s_a)
    if isinstance(a, (date, datetime)) and isinstance(b, (date, datetime)):
        return a == b or a.isoformat() == b.isoformat()

    # 3. Numeric string coercion
    if isinstance(a, (int, float)) and isinstance(b, str):
        try:
            val_b = float(b.strip().replace(",", ""))
            return _norm_number(a) == _norm_number(val_b)
        except (ValueError, TypeError):
            return False
    if isinstance(b, (int, float)) and isinstance(a, str):
        try:
            val_a = float(a.strip().replace(",", ""))
            return _norm_number(val_a) == _norm_number(b)
        except (ValueError, TypeError):
            return False

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _norm_number(a) == _norm_number(b)

    if isinstance(a, str) and isinstance(b, str):
        sa = a.strip()
        sb = b.strip()
        if sa == sb:
            return True
        # ISO date/datetime matching: "2026-10-23" matches "2026-10-23T00:00:00"
        if len(sa) >= 10 and len(sb) >= 10 and sa[:10] == sb[:10]:
            if sa.startswith(sb) or sb.startswith(sa):
                return True
        return False

    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[k], b[k]) for k in a)
    return bool(a == b)


def payload_equal(
    expected: Optional[Dict[str, Any]],
    actual: Optional[Dict[str, Any]],
    schema: Optional[Any] = None,
) -> bool:
    """Exact-payload-match equality, tolerant of benign normalizations.

    - Missing optional params in ``actual`` equal to the schema ``default`` are
      materialized before comparison.
    - ``5`` vs ``5.0`` and numeric string coercions compare equal.
    - Symmetric optional-param handling.
    """
    if actual is None:
        return expected is None
    if expected is None:
        return False
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return _deep_equal(expected, actual)
    exp = dict(expected)
    act = dict(actual)
    if schema is not None:
        for p in getattr(schema, "parameters", []):
            name = getattr(p, "name", None)
            if name is None:
                continue
            if not getattr(p, "required", True):
                default_val = getattr(p, "default", None)
                if name not in act and default_val is not None:
                    act[name] = default_val
                if name not in exp and default_val is not None:
                    exp[name] = default_val
                if name not in act and exp.get(name) is None:
                    act[name] = None
                if name not in exp and act.get(name) is None:
                    exp[name] = None
                if name in act and act[name] == default_val and name not in exp:
                    exp[name] = default_val
                if name in exp and exp[name] == default_val and name not in act:
                    act[name] = default_val

    # Symmetric None handling for absent keys
    for k in list(act.keys()):
        if act[k] is None and k not in exp:
            del act[k]
    for k in list(exp.keys()):
        if exp[k] is None and k not in act:
            del exp[k]

    if set(exp.keys()) != set(act.keys()):
        return False
    return all(_deep_equal(exp[k], act[k]) for k in exp)


# --------------------------------------------------------------------------- #
# Error taxonomy
# --------------------------------------------------------------------------- #

def _classify_error(
    sample_id: str,
    predicted: Optional[Dict[str, Any]],
    gt_payload: Optional[Dict[str, Any]],
    predicted_tool: Optional[str],
    gt_tool: Optional[str],
    resolved_params: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return the 7-bucket error categories for one false hit."""
    cats: List[str] = []
    if not gt_payload:
        # GT expected no call (absorbed stratum A / empty arguments): emitting
        # a call is an extra call; emitting nothing is the (correct) abstain.
        if predicted:
            cats.append("error_extra_call")
        return cats
    if predicted is None:
        cats.append("error_missing_call")
        return cats
    if predicted_tool is not None and gt_tool is not None and predicted_tool != gt_tool:
        cats.append("error_wrong_tool")
    exp_keys = set(gt_payload.keys())
    act_keys = set(predicted.keys())

    # entity swap: predicted value of p1 equals GT value of p2 (and vice versa)
    for p1 in exp_keys:
        for p2 in exp_keys:
            if p1 == p2:
                continue
            if p1 in predicted and p2 in gt_payload:
                if _deep_equal(predicted[p1], gt_payload[p2]) and _deep_equal(
                    predicted.get(p2), gt_payload.get(p1)
                ):
                    cats.append("error_entity_swap")
                    break
        else:
            continue
        break

    # duplicate span: two params extracted the same utterance span (GT differ)
    if resolved_params:
        spans: Dict[Tuple[int, int], List[str]] = {}
        for name, r in resolved_params.items():
            span = getattr(r, "char_span", None)
            if span is None:
                continue
            s = (span[0], span[1]) if isinstance(span, (list, tuple)) else span
            spans.setdefault(tuple(s), []).append(name)
        for s, names in spans.items():
            if len(names) > 1 and any(names[0] in gt_payload and n != names[0] and
                                      not _deep_equal(gt_payload.get(names[0]), gt_payload.get(n)) for n in names[1:]):
                cats.append("error_duplicate_span")
                break

    # date misparse: mismatch involves a date/datetime parameter
    if any("date" in str(k).lower() or "time" in str(k).lower() for k in exp_keys | act_keys):
        if not all(_deep_equal(gt_payload.get(k), predicted.get(k)) for k in exp_keys if k in predicted):
            cats.append("error_date_misparse")

    # wrong value: predicted value occurs in the source text but does not match GT
    for k in exp_keys:
        if k in predicted and k in gt_payload and not _deep_equal(predicted[k], gt_payload[k]):
            cats.append("error_wrong_value")
            break

    # extra call: predicted emitted a call but GT expected none
    if not gt_payload and predicted:
        cats.append("error_extra_call")
    # missing call: GT expected parameters but none were provided
    if gt_payload and not predicted:
        cats.append("error_missing_call")
    return cats


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


class BaseToolCallEvaluator(ABC):
    """Abstract evaluator interface for scoring tool-call predictions (plan §4.2).

    Concrete implementations score ``(samples, predictions)`` into a
    :class:`ToolCallMetrics` report. The abstract contract keeps the offline
    evaluator, agentic evaluator, and any backends interchangeable.
    """

    @abstractmethod
    def evaluate(
        self,
        samples: Iterable[UnifiedSample],
        predictions: Iterable[Dict[str, Any]],
    ) -> ToolCallMetrics:
        """Aggregate metrics across all samples."""


class ToolCallEvaluator(BaseToolCallEvaluator):
    """Scores offline predictions against reference ground-truth calls."""

    def evaluate(
        self,
        samples: Iterable[UnifiedSample],
        predictions: Iterable[Dict[str, Any]],
    ) -> ToolCallMetrics:
        """Aggregate metrics across all samples (stratified F / A / all)."""
        m = ToolCallMetrics()
        rows: List[Dict[str, Any]] = []
        samps = list(samples)
        preds = list(predictions)
        for idx, sample in enumerate(samps):
            pred = preds[idx] if idx < len(preds) else {"hit": False}
            gt = self._first_gt(sample)
            row = self._score_one(sample, pred, gt)
            rows.append(row)

        m.total_samples = len(rows)
        self._accumulate(m, rows, stratum=None)

        # R0.4 / J6: set-level call metrics. ``exact_payload_match`` above stays
        # call-0-only so the historical number keeps its meaning; these are the
        # numbers that stop a multi-call row being scored on its first call only.
        n_rows = len(rows)
        m.call_exact_set_match = (
            sum(1.0 for r in rows if r.get("call_set_equal")) / n_rows
        ) if n_rows else 0.0
        m.call_count_acc = (
            sum(1.0 for r in rows if r.get("call_count_equal")) / n_rows
        ) if n_rows else 0.0
        gt_hist: Dict[str, int] = {}
        pred_hist: Dict[str, int] = {}
        for r in rows:
            g = str(int(r.get("n_gt_calls", 0)))
            p = str(int(r.get("n_pred_calls", 0)))
            gt_hist[g] = gt_hist.get(g, 0) + 1
            pred_hist[p] = pred_hist.get(p, 0) + 1
        m.n_calls_hist = dict(sorted(gt_hist.items()))
        m.pred_n_calls_hist = dict(sorted(pred_hist.items()))

        m.metrics_by_stratum = {
            "F": self._stratum_metrics(rows, Stratum.F),
            "A": self._stratum_metrics(rows, Stratum.A),
            "all": self._stratum_metrics(rows, None),
        }
        m.false_hit_rate_on_a = m.metrics_by_stratum["A"].get("false_hit_rate_on_a", 0.0)
        m.abstain_reasons = self._abstain_counts(rows)
        m.ece = self.expected_calibration_error(rows)
        m.risk_coverage = self.compute_risk_coverage_curve(rows)
        m.errors_by_sample = {
            r["sample_id"]: r["errors"] for r in rows if r["errors"]
        }
        return m

    # -- scoring one row ---------------------------------------------------- #

    def _score_one(self, sample: UnifiedSample, pred: Dict[str, Any],
                   gt: Optional[GroundTruthCall]) -> Dict[str, Any]:
        hit = bool(pred.get("hit", False))
        payload = pred.get("payload")
        payload_partial = pred.get("payload_partial")
        if not payload_partial and payload:
            payload_partial = payload
        tool_name = pred.get("tool_name") or pred.get("tool")
        gt_payload = gt.arguments if gt else None
        gt_tool = gt.tool_name if gt else None
        # The schema for ``payload_equal`` must be the GT CALL's tool, not
        # ``sample.tools[0]``. Using the first tool's schema materialised the
        # wrong optional defaults / enum/type coercions whenever the GT call was
        # not the first declared tool (a silent false pass or false fail once a
        # sample declared its tools in any order other than the call order).
        gt_schema = None
        if gt_tool is not None:
            gt_schema = next(
                (t for t in (sample.tools or []) if getattr(t, "name", None) == gt_tool),
                None,
            )
        if gt_schema is None and sample.tools:
            gt_schema = sample.tools[0]
        equal = payload_equal(gt_payload, payload, gt_schema)
        correct = bool(equal and (gt_tool is None or tool_name == gt_tool))

        n_gt_params = 0
        n_pred_params = 0
        n_correct_params = 0
        if gt_payload and isinstance(gt_payload, dict):
            n_gt_params = len(gt_payload)
            if payload_partial and isinstance(payload_partial, dict):
                n_pred_params = len(payload_partial)
                for k, v_gt in gt_payload.items():
                    if k in payload_partial and _deep_equal(payload_partial[k], v_gt):
                        n_correct_params += 1
        elif not gt_payload:
            if payload_partial and isinstance(payload_partial, dict):
                n_pred_params = len(payload_partial)

        has_partial = bool(payload_partial and len(payload_partial) > 0)
        status = pred.get("status", "hit" if hit else ("partial" if has_partial else "abstain"))

        # R0.4 / J6: score the predicted CALL SET, not just the flattened first
        # call. ``payload`` / ``tool_name`` above stay the primary (call-0) view
        # so ``exact_payload_match`` keeps its historical meaning.
        gt_calls = list(getattr(sample, "ground_truth_calls", None) or [])
        pred_calls = self._predicted_call_list(pred)
        call_set_equal, call_count_equal = self.multi_call_equal(
            gt_calls, pred_calls, sample
        )

        # gates (only meaningful when hit)
        gate_correct = bool(hit and correct)
        row = {
            "sample_id": sample.sample_id,
            "stratum": sample.stratum.value,
            "hit": hit,
            "status": status,
            "tool_name": tool_name,
            "gt_tool": gt_tool,
            "payload": payload,
            "payload_partial": payload_partial,
            "gt_payload": gt_payload,
            "payload_equal": equal,
            "correct": correct,
            "gate_correct": gate_correct,
            "confidence": float(pred.get("confidence") or pred.get("accept_probability") or 0.0),
            "n_gt_params": n_gt_params,
            "n_pred_params": n_pred_params,
            "n_correct_params": n_correct_params,
            "has_partial": has_partial,
            "n_gt_calls": len(gt_calls),
            "n_pred_calls": len(pred_calls),
            "call_set_equal": call_set_equal,
            "call_count_equal": call_count_equal,
            "errors": [],
        }
        if hit and not correct:
            row["errors"] = _classify_error(sample.sample_id, payload, gt_payload,
                                            tool_name, gt_tool, pred.get("parameters"))
        return row

    @staticmethod
    def _predicted_call_list(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The predicted calls as ``[{tool_name, payload}, ...]`` (R0.4).

        A prediction written by the Lean ``predict_split`` cell carries ``calls``;
        a legacy single-call prediction has only the flattened ``tool_name`` /
        ``payload``. Reading the flattened fields as exactly one call keeps old
        prediction files scorable instead of pretending they were multi-call.
        """
        calls = pred.get("calls")
        if isinstance(calls, list) and calls:
            out = []
            for c in calls:
                if not isinstance(c, dict):
                    continue
                name = c.get("tool_name")
                if not name:
                    continue
                out.append({"tool_name": name, "payload": c.get("payload") or {}})
            if out:
                return out
        name = pred.get("tool_name") or pred.get("tool")
        if not name:
            return []
        return [{"tool_name": name, "payload": pred.get("payload") or {}}]

    @staticmethod
    def multi_call_equal(
        gt_calls: Sequence[GroundTruthCall],
        pred_calls: Sequence[Dict[str, Any]],
        sample: UnifiedSample,
    ) -> tuple:
        """Set-level call match + call-count agreement (R0.4 / J6).

        Calls are matched as a MULTISET by tool name, then each matched pair's
        payload is compared with :func:`payload_equal` against that tool's own
        schema. Order does not matter: emitting the same two calls in the other
        order is still correct, because a call set has no sequence.
        """
        n_gt = len(gt_calls)
        n_pred = len(pred_calls)
        count_equal = n_gt == n_pred
        if not count_equal:
            return False, False
        if n_gt == 0:
            return True, True

        schemas = {getattr(t, "name", None): t for t in (sample.tools or [])}
        # Group predictions by tool so a tool called twice is matched twice.
        remaining: Dict[str, List[Dict[str, Any]]] = {}
        for c in pred_calls:
            remaining.setdefault(str(c.get("tool_name")), []).append(c)

        for gt in gt_calls:
            name = str(getattr(gt, "tool_name", ""))
            bucket = remaining.get(name) or []
            if not bucket:
                return False, True
            gt_payload = dict(getattr(gt, "arguments", None) or {})
            matched_idx = None
            for i, cand in enumerate(bucket):
                if payload_equal(gt_payload, cand.get("payload") or {}, schemas.get(name)):
                    matched_idx = i
                    break
            if matched_idx is None:
                return False, True
            bucket.pop(matched_idx)
        return True, True

    @staticmethod
    def _first_gt(sample: UnifiedSample) -> Optional[GroundTruthCall]:
        for call in sample.ground_truth_calls:
            return call
        return None

    # -- aggregation -------------------------------------------------------- #

    @staticmethod
    def _stratum_metrics(rows: List[Dict[str, Any]], stratum: Optional[Stratum]) -> Dict[str, float]:
        sel = [r for r in rows if stratum is None or r["stratum"] == stratum.value]
        n = len(sel)
        if n == 0:
            return {
                "n": 0.0,
                "exact_payload_match": 0.0,
                "hit_rate": 0.0,
                "precision_at_hit": 0.0,
                "false_hit_rate_on_a": 0.0,
                "partial_extraction_rate": 0.0,
                "param_precision": 0.0,
                "param_recall": 0.0,
                "param_f1": 0.0,
                "call_exact_set_match": 0.0,
                "call_count_acc": 0.0,
            }
        hits = [r for r in sel if r["hit"]]
        n_hits = len(hits)
        tot_gt = sum(r.get("n_gt_params", 0) for r in sel)
        tot_pred = sum(r.get("n_pred_params", 0) for r in sel)
        tot_corr = sum(r.get("n_correct_params", 0) for r in sel)
        p_prec = (tot_corr / tot_pred) if tot_pred else 0.0
        p_rec = (tot_corr / tot_gt) if tot_gt else 0.0
        p_f1 = (2 * p_prec * p_rec / (p_prec + p_rec)) if (p_prec + p_rec) > 0 else 0.0
        return {
            "n": float(n),
            "exact_payload_match": sum(1.0 for r in sel if r["payload_equal"]) / n,
            "hit_rate": n_hits / n,
            "precision_at_hit": (sum(1.0 for r in hits if r["gate_correct"]) / n_hits) if n_hits else 0.0,
            "false_hit_rate_on_a": (n_hits / n) if (stratum == Stratum.A) else 0.0,
            "partial_extraction_rate": sum(1.0 for r in sel if r.get("has_partial")) / n,
            "param_precision": p_prec,
            "param_recall": p_rec,
            "param_f1": p_f1,
            # R0.4 / J6: set-level call accuracy for the same stratum.
            "call_exact_set_match": sum(1.0 for r in sel if r.get("call_set_equal")) / n,
            "call_count_acc": sum(1.0 for r in sel if r.get("call_count_equal")) / n,
        }

    def _accumulate(self, m: ToolCallMetrics, rows: List[Dict[str, Any]],
                    stratum: Optional[Stratum]) -> None:
        sel = [r for r in rows if stratum is None or r["stratum"] == stratum.value]
        n = len(sel)
        if n == 0:
            return
        m.tool_accuracy = sum(1.0 for r in sel if r["tool_name"] == r["gt_tool"]) / n
        m.per_param_accuracy = sum(1.0 for r in sel if r["payload_equal"]) / n
        m.exact_payload_match = m.per_param_accuracy
        n_hits = sum(1 for r in sel if r["hit"])
        m.hit_rate = n_hits / n
        if n_hits:
            m.precision_at_hit = sum(1.0 for r in sel if r["gate_correct"]) / n_hits
        a_count = sum(1 for r in sel if r["stratum"] == "A")
        m.false_hit_rate_on_a = (
            sum(1 for r in sel if r["stratum"] == "A" and r["hit"]) / a_count
        ) if a_count else 0.0

        tot_gt = sum(r.get("n_gt_params", 0) for r in sel)
        tot_pred = sum(r.get("n_pred_params", 0) for r in sel)
        tot_corr = sum(r.get("n_correct_params", 0) for r in sel)
        m.param_precision = (tot_corr / tot_pred) if tot_pred else 0.0
        m.param_recall = (tot_corr / tot_gt) if tot_gt else 0.0
        m.param_f1 = (2 * m.param_precision * m.param_recall / (m.param_precision + m.param_recall)) if (m.param_precision + m.param_recall) > 0 else 0.0
        m.partial_extraction_rate = sum(1.0 for r in sel if r.get("has_partial")) / n
        for r in sel:
            for err in r["errors"]:
                if err == "error_wrong_tool":
                    m.error_wrong_tool += 1
                elif err == "error_wrong_value":
                    m.error_wrong_value += 1
                elif err == "error_entity_swap":
                    m.error_entity_swap += 1
                elif err == "error_duplicate_span":
                    m.error_duplicate_span += 1
                elif err == "error_date_misparse":
                    m.error_date_misparse += 1
                elif err == "error_extra_call":
                    m.error_extra_call += 1
                elif err == "error_missing_call":
                    m.error_missing_call += 1

    @staticmethod
    def _abstain_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in rows:
            if not r["hit"]:
                reason = "abstained"
                out[reason] = out.get(reason, 0) + 1
        return out

    # -- risk-coverage + ECE ------------------------------------------------ #

    def compute_risk_coverage_curve(
        self,
        samples_or_rows: Sequence[Any],
        scores_and_correctness: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Cumulative risk-coverage over scored rows sorted by confidence."""
        if scores_and_correctness is not None:
            scored = [
                (float(s.get("score") or s.get("confidence") or 0.0), bool(s.get("correct", False)))
                for s in scores_and_correctness
            ]
        else:
            scored = [(r["confidence"], bool(r["correct"])) for r in samples_or_rows]
        scored.sort(key=lambda t: t[0], reverse=True)
        n = len(scored) or 1
        curve: List[Dict[str, Any]] = []
        emitted = errors = 0
        for conf, correct in scored:
            emitted += 1
            if not correct:
                errors += 1
            curve.append({
                "confidence": float(conf),
                "coverage": emitted / n,
                "risk": errors / emitted,
                "precision": (emitted - errors) / emitted,
                "n_emitted": emitted,
                "n_errors": errors,
            })
        return curve

    def expected_calibration_error(
        self,
        rows: Sequence[Dict[str, Any]],
        n_bins: int = 10,
        key: str = "confidence",
    ) -> float:
        """Binned ECE: mean |bin_confidence - bin_accuracy| weighted by size."""
        scored = [(float(r.get(key) or 0.0), bool(r.get("correct", False))) for r in rows]
        if not scored:
            return 0.0
        n = len(scored)
        n_bins = max(1, int(n_bins))
        total = 0.0
        for b in range(n_bins):
            lo = b / n_bins
            hi = (b + 1) / n_bins
            members = [s for s in scored if lo <= s[0] < hi]
            if not members:
                continue
            bin_conf = sum(s[0] for s in members) / len(members)
            bin_acc = sum(1.0 for s in members if s[1]) / len(members)
            total += (len(members) / n) * abs(bin_conf - bin_acc)
        return float(total)

    @staticmethod
    def _sample_stratum(sample: UnifiedSample) -> Stratum:
        return sample.stratum


# --------------------------------------------------------------------------- #
# B1a baseline: deterministic regex extraction over the identical legend
# --------------------------------------------------------------------------- #


class B1aBaseline:
    """Deterministic regex + virtual-context extractor (B1a).

    Walks the same ``SpanTemplateOutput`` legend the resolver sees — no model
    weights — matching enum values exactly, single date entries, and single
    regex numeric matches. Intended as the cheapest fairness lower bound.
    """

    _DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    _NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)")

    def predict(self, output: Any, schema: Any) -> Dict[str, Any]:
        """Return ``{"tool_name", "payload", "hit"}`` for one rendered output."""
        entries = list(getattr(output, "entries", []) or [])
        selectable = [e for e in entries if getattr(e, "selectable", False)]
        text = getattr(output, "text", "")
        tool = getattr(schema, "name", None)
        payload: Dict[str, Any] = {}
        for p in getattr(schema, "parameters", []):
            name = getattr(p, "name", None)
            if name is None:
                continue
            ptype = getattr(p, "param_type", "string")
            enum = list(getattr(p, "enum", None) or [])
            if enum:
                found = self._match_enum(selectable, enum, name)
                if found is not None:
                    payload[name] = found
                continue
            if ptype == "date":
                m = self._DATE_RE.findall(text)
                if len(m) == 1:
                    payload[name] = m[0]
                continue
            if ptype in ("integer", "number"):
                nums = self._NUM_RE.findall(text)
                if len(nums) == 1:
                    payload[name] = _norm_number(nums[0])
                continue
            if ptype == "boolean":
                low = set(getattr(e, "value", "") for e in selectable
                          if isinstance(getattr(e, "value", ""), str))
                low = {str(v).strip().lower() for v in low}
                if "true" in low:
                    payload[name] = True
                elif "false" in low:
                    payload[name] = False
                continue
            # generic string: single selectable entry inside the utterance
            vals = [e.value for e in selectable if getattr(e, "value", None) is not None]
            if len(vals) == 1:
                payload[name] = vals[0]
        return {"tool_name": tool, "payload": payload, "hit": bool(payload)}

    @staticmethod
    def _match_enum(selectable: Sequence[Any], enum: Sequence[Any], name: str) -> Optional[Any]:
        cands = []
        for e in selectable:
            if getattr(e, "value", None) in enum:
                cands.append(e.value)
        if len(cands) == 1:
            return cands[0]
        return None


# --------------------------------------------------------------------------- #
# CLIs / iteration helpers
# --------------------------------------------------------------------------- #


def _iter_samples(samples: Sequence[UnifiedSample]) -> Iterator[UnifiedSample]:
    yield from samples
