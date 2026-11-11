"""Retrieval-first Parameter Extraction Evaluation (Eval Part 1) for Lean Flow.

Ref: docs/param_extraction_eval_coverage.md

Under the retrieval-first contract, param extraction is a **fast path**: it may
only retrieve values that are *present* in the context (utterance, recent
context, time context, enums, booleans, null). Params that need open-ended
generation are out of scope by design — the correct behavior is to refuse
(emit nothing) and route to a slower path.

This module therefore reports **two** recall views instead of one blended
number:

- ``extraction_recall`` / ``extraction_f1`` — scored over params with an
  EXTRACTABLE ground-truth target only. This is the headline accuracy metric
  for the fast path.
- ``recall_all`` — the legacy blended view over every GT argument, kept as a
  diagnostic (it mixes in out-of-scope params and is always <= the extractable
  recall).

Plus the architecture-value metrics the contract implies:

- ``fast_path_param_coverage`` — fraction of GT params that are extractable.
- ``fast_path_call_coverage`` — fraction of samples where *every* GT param is
  extractable (the fast path could handle the whole call).
- ``refused_unextractable`` vs ``missed_extractable`` — split of the old
  ``pointer_empty`` counter into "correct refusal" (routing signal) vs "real
  fast-path failure".
- ``e3_aligner_drift`` — Stratum-F invariant tripwire (Fix E3): every GT arg of
  an "everything should be extractable" sample must carry an extractable
  target. Violations are aligner drift, **not** model misses, so they are
  reported separately instead of being folded into recall.

Ground-truth universe and null handling
---------------------------------------
A GT argument that is explicitly ``null`` is a *real* GT param: the correct
fast-path behavior is to select the deterministic ``<|null|>`` candidate. It is
therefore part of the GT universe (:func:`_gt_params`), of the per-param ``gt``
counts, and of every recall/precision denominator.

Prediction state is likewise null-aware: "predicted null" (the decoder picked a
candidate whose ``source_type`` is ``"null"``) is distinct from "no prediction"
(the resolver produced no value). Conflating the two made a correct null
selection look like a refusal, and made a wrong value on a null GT param
invisible to ``pointer_wrong``.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from spancall.data.schema import Stratum, UnifiedSample

# --------------------------------------------------------------------------- #
# Groundability classification
# --------------------------------------------------------------------------- #

#: Target types the pointer model (fast path) can produce a value for.
EXTRACTABLE_TARGET_TYPES = frozenset({
    "utterance_span",     # value literally in the utterance
    "recent_ctx",         # value from a recent_context row
    "time_ctx",           # date derivable from current_time_iso + clock
    "enum_ctx",           # schema enum value
    "boolean_ctx",        # yes/no phrasing in the utterance
    "null_ctx",           # legitimately null/omitted -> <|null|>
})

#: Target types that require generation/reasoning (out of scope for the fast path).
NON_EXTRACTABLE_TARGET_TYPES = frozenset({
    "unresolved",
})


def param_targets(
    sample: UnifiedSample, param_name: str, call_index: int = 0
) -> List[Any]:
    """SpanTarget list for a param of ONE call (empty when unaligned).

    J6/R0.4: targets are owned by a call, not by a sample. ``call_index=0`` on a
    single-call sample still resolves through the legacy bare keys via
    :meth:`UnifiedSample.targets_for_call`, so every existing caller reads
    exactly what it used to; a multi-call sample must ask for its own call.
    """
    return (sample.targets_for_call(call_index) or {}).get(param_name, [])


def param_is_groundable(
    sample: UnifiedSample, param_name: str, call_index: int = 0
) -> bool:
    """True if the param has at least one extractable ground-truth target."""
    return any(
        t.target_type in EXTRACTABLE_TARGET_TYPES
        for t in param_targets(sample, param_name, call_index)
    )


def classify_param(is_groundable: bool) -> str:
    """Per-param coverage class used for the refusal-vs-failure split.

    - ``extractable``: the GT param has a groundable target — the fast path is
      expected to retrieve it; failure here is a real miss.
    - ``unextractable``: the target is ``unresolved`` or no targets were
      aligned — refusing is correct and routes the call to a slower path.
    """
    return "extractable" if is_groundable else "unextractable"


def _gt_params(gt_args: Dict[str, Any], schema: Any) -> Dict[str, Any]:
    """GT param universe: every schema param with a GT entry, explicit nulls included.

    A GT value of ``None`` means "this param must be null/omitted", which the
    fast path can and must predict via ``<|null|>``. Filtering ``None`` out (as
    the previous implementation did) silently removed those params from
    coverage, from the per-param ``gt`` counts, and from every denominator.
    """
    return {p.name: gt_args.get(p.name) for p in schema.parameters if p.name in gt_args}


def _predicted_calls(res: Any) -> List[Any]:
    """The predicted call list of a resolver result (J6 / R0.4).

    A resolver that decoded the call axis exposes ``calls``; an old stub does
    not. In that case a result naming a tool is exactly ONE call, and a result
    naming none is zero calls — so ``call_count_acc`` stays honest instead of
    assuming the legacy path was multi-call or inventing a call that was never
    predicted.
    """
    calls = getattr(res, "calls", None)
    if calls:
        return [c for c in calls if getattr(c, "tool_name", None)]
    return [res] if getattr(res, "tool_name", None) else []


def _call_identity(call: Any) -> Tuple[str, str]:
    """A content key for one predicted call: ``(tool_name, canonical payload)``."""
    name = str(getattr(call, "tool_name", None) or "")
    payload = getattr(call, "payload", None)
    if payload is None:
        payload = getattr(call, "arguments", None)
    payload = payload or {}
    try:
        serialised = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - exotic payload object
        serialised = repr(payload)
    return name, serialised


def e3_aligner_violation(
    sample: UnifiedSample,
    gt_args: Dict[str, Any],
    schema: Any,
    call_index: int = 0,
) -> Optional[str]:
    """Return a violation string when a Stratum-F GT arg has no extractable target.

    Fix E3 (``docs/param_extraction_eval_coverage.md`` §4): every GT argument of
    a Stratum-F sample must be backed by a target of an
    :data:`EXTRACTABLE_TARGET_TYPES` type. Anything else — no target at all, or
    an ``unresolved`` target — is aligner drift: the param silently disappears
    from pointer supervision and would otherwise be indistinguishable from a
    model miss.
    """
    for pn in _gt_params(gt_args, schema):
        targets = param_targets(sample, pn, call_index)
        if not targets:
            return f"{pn}:no_target"
        bad = sorted({t.target_type for t in targets} - EXTRACTABLE_TARGET_TYPES)
        if bad:
            return f"{pn}:{bad[0]}"
    return None


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #

def _norm(v: Any) -> Any:
    """Normalize a GT/pred value for comparison (None-safe, case tolerant).

    Null-ish strings are folded onto ``None`` so that a ``<|null|>`` selection
    cast to ``"null"``/``"None"`` compares equal to a GT ``null`` argument.
    """
    if v is None:
        return None
    s = str(v).strip()
    low = s.lower()
    if low in ("none", "null"):
        return None
    return low if low in ("true", "false") else s


def _ratio(num: int, den: int) -> float:
    """Ratio that is NaN — not a fake 1.0 — when the denominator is empty."""
    return num / den if den else float("nan")


def _f1(p: float, r: float) -> float:
    """Harmonic mean that propagates NaN instead of inventing a score."""
    if math.isnan(p) or math.isnan(r) or (p + r) <= 0:
        return float("nan")
    return 2 * p * r / (p + r)


def _round(v: float, nd: int = 4) -> float:
    return float("nan") if math.isnan(v) else round(v, nd)


def _fmt(v: Any, nd: int = 4) -> str:
    """Format a metric, rendering NaN as ``n/a``."""
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #

def eval_param_extraction_lean(
    split_name: str,
    resolver: Any,
    data_dir: str,
    limit: Optional[int] = 500,
    max_param_gap_report: int = 20,
) -> Dict[str, Any]:
    """Run retrieval-first param-extraction eval on a canonical split.

    Args:
        split_name: canonical split name (e.g. ``hermes_eval``); the raw
            ``{split_name}.jsonl`` is read from ``data_dir``.
        resolver: object with ``try_resolve(tool, utterance, recent_context,
            current_time_iso)`` (``LeanResolver``).
        data_dir: directory containing ``{split_name}.jsonl``.
        limit: max samples to score (reported via ``truncation``).
        max_param_gap_report: cap on per-param table rows printed.

    Returns:
        A dict of metrics (also printed as a human-readable report).

    Raises:
        FileNotFoundError: if the split file does not exist. Returning ``{}``
            made a missing/misnamed split look like a silent zero-metric run.
    """
    src = os.path.join(data_dir, f"{split_name}.jsonl")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Eval split not found: {src}. Build the canonical split first "
            f"(scripts/prepare_data.py) or point data_dir at the processed dir."
        )

    with open(src, encoding="utf-8") as fh:
        total_available = sum(1 for line in fh if line.strip())

    # --- aggregate counters ---
    total = 0
    stratum_f_total = 0
    stratum_f_exact = 0
    stratum_f_hits = 0

    # Groundable (extractable) param metrics — the headline fast-path quality
    ex_tp = 0
    ex_fp = 0
    ex_fn = 0

    # Legacy blended view (diagnostic)
    all_tp = 0
    all_fp = 0
    all_fn = 0

    extracted = 0                       # samples with >= 1 non-null extracted value
    confs: List[float] = []
    buckets: Counter = Counter()

    # refusal vs failure split (replaces monolithic pointer_empty)
    refused_unextractable = 0
    missed_extractable = 0
    pointer_wrong = 0                   # pred present and != GT (nulls included)

    # coverage accounting
    total_gt_params = 0                 # params present in GT args (nulls included)
    groundable_gt_params = 0            # of those, extractable by the fast path
    calls_all_groundable = 0            # samples where every GT param is extractable

    # J6 / R0.4: multi-call accounting. ``call_count_acc`` compares the number of
    # predicted calls to the number of GT calls; ``call_exact_set_match`` is the
    # set-level "every GT call exact, no extra call" headline that replaces the
    # silently call-0-only reading of a multi-call sample.
    #
    # IMPORTANT — Part 1 runs in the ORACLE-TOOL setting: it resolves exactly one
    # tool per GT call, so it can never predict FEWER calls than GT and never
    # invents a call for a tool it was not asked about. ``call_count_acc`` is
    # therefore bounded above by coverage of the GT calls and measures
    # "did resolving every GT tool produce a call, with no extra calls", NOT a
    # learned count head. Part 3 (``ToolCallEvaluator``) is the place where a
    # genuine predicted-call list is scored.
    n_calls_hist: Counter = Counter()      # GT calls per sample
    pred_n_calls_hist: Counter = Counter()  # predicted calls per sample
    call_count_correct = 0
    call_set_exact = 0                      # all GT calls exact, no extra call
    call_set_exact_F = 0                    # ... restricted to Stratum F

    # R9: the render count T (number of <|tool|> blocks) the RESOLVER actually
    # built. Reported here so the oracle-tool eval layout (T=1) can be compared
    # against the training distribution without guessing from the tensorize cell.
    render_T_hist: Counter = Counter()
    render_T_rows = 0
    recent_budget_effective: Counter = Counter()

    # Fix E3: Stratum-F alignment invariant tripwire
    e3_violations = 0
    e3_examples: List[str] = []

    per_param: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"gt": 0, "tp": 0, "fp": 0, "fn": 0, "extractable": 0, "unextractable": 0}
    )

    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            s = UnifiedSample.from_dict(json.loads(line))
            if not s.ground_truth_calls or not s.tools:
                continue

            gt_calls = list(s.ground_truth_calls)
            n_calls_gt = len(gt_calls)

            # J6/R0.4: score EVERY GT call, not just ``[0]``. Part 1 runs in the
            # oracle-tool setting, so each call is resolved against its own tool
            # (still rendered beside the whole ``s.tools`` set, R0.2). Scoring
            # only call 0 silently under-reported the 13.9% multi-call rows.
            per_call: List[Tuple[int, Any, Any, Any]] = []
            for ci, gt_call in enumerate(gt_calls):
                gt_tool = next((t for t in s.tools if t.name == gt_call.tool_name), None)
                if gt_tool is None:
                    continue
                # R0.2: render the SAME tool set the model was trained on. Under
                # the oracle-tool setting only the GT tool is rendered (T=1),
                # while training packed ``pack_tools`` anchors — a layout H1/H3
                # never saw. The capability flag (not a TypeError fallback) keeps
                # old stubs honest: they simply keep the oracle prompt and the T
                # mismatch is reported, rather than being hidden by a swallowed
                # exception.
                if getattr(resolver, "supports_context_schemas", False):
                    res = resolver.try_resolve(
                        gt_tool, s.utterance, s.recent_context, s.current_time_iso,
                        context_schemas=s.tools,
                    )
                else:
                    res = resolver.try_resolve(
                        gt_tool, s.utterance, s.recent_context, s.current_time_iso
                    )
                per_call.append((ci, gt_call, gt_tool, res))
            if not per_call:
                continue

            total += 1
            n_calls_hist[n_calls_gt] += 1

            # R9/R2 telemetry is per SAMPLE: read it off the first resolution so a
            # 2-call row does not double-count the layout actually rendered.
            res0 = per_call[0][3]
            _feats = getattr(res0, "features", None) or {}
            _rt = _feats.get("render_T")
            if _rt is not None:
                render_T_hist[int(_rt)] += 1
                render_T_rows += 1
            _rb = _feats.get("recent_budget_effective")
            if _rb is not None:
                recent_budget_effective[int(_rb)] += 1

            # J6: predicted call count vs GT call count (R0.4 call-count accuracy).
            #
            # A call set is a MULTISET: two GT calls to the SAME tool are two
            # calls. Deduping by tool name collapsed a perfectly predicted
            # [get_weather, get_weather] down to one call and scored
            # ``call_count_acc=0`` / ``call_exact_set_match=0`` for a correct
            # prediction. Part 3 matches per tool bucket (``multi_call_equal``);
            # Part 1 must not use a weaker rule.
            #
            # Two shapes reach here:
            #  1. a resolution that decoded the whole SET (``calls`` longer than
            #     1, e.g. ``try_resolve(..., call_schemas=...)``) — it repeats
            #     the set once per GT call, so counting every list would
            #     multiply the calls;
            #  2. a resolution that returns the call(s) produced BY that
            #     resolution — the normal Part 1 case. ONE call per resolution
            #     is exactly one predicted call, and two identical calls are
            #     still two calls.
            #
            # So: single-call resolutions always contribute (multiplicity is
            # real and is never deduped — name dedupe used to collapse a
            # same-tool multiset, content dedupe still collapses two identical
            # GT calls). Multi-call lists are unioned with a per-identity
            # MAXIMUM count rather than a sum, which reads a repeated whole-set
            # decode once while keeping any genuine extra call it carries.
            pred_calls: List[Any] = []
            _multi_counts: Dict[Tuple[str, str], int] = {}
            _multi_repr: Dict[Tuple[str, str], Any] = {}
            for _ci, _gc, _gt, r in per_call:
                _cs = _predicted_calls(r)
                if len(_cs) <= 1:
                    pred_calls.extend(_cs)
                    continue
                _local: Dict[Tuple[str, str], int] = {}
                for _c in _cs:
                    _ident = _call_identity(_c)
                    _local[_ident] = _local.get(_ident, 0) + 1
                    _multi_repr.setdefault(_ident, _c)
                for _ident, _n in _local.items():
                    if _n > _multi_counts.get(_ident, 0):
                        _multi_counts[_ident] = _n
            for _ident, _n in _multi_counts.items():
                pred_calls.extend([_multi_repr[_ident]] * _n)
            pred_n_calls_hist[len(pred_calls)] += 1
            if len(pred_calls) == n_calls_gt:
                call_count_correct += 1

            multi_call = n_calls_gt > 1
            sample_extracted = False
            sample_gt_params = 0
            sample_groundable = 0
            call0_exact = False
            all_calls_exact = True

            for k, (ci, gt_call, gt_tool, res) in enumerate(per_call):
                gt_args = dict(gt_call.arguments or {})
                key_prefix = f"{ci}::" if multi_call else ""

                # C1: the GT universe keeps explicitly-null args.
                gt_universe = _gt_params(gt_args, gt_tool)
                norm_gt = {kk: _norm(v) for kk, v in gt_universe.items()}

                # C4: null-aware prediction view — "null" != "no prediction".
                pred_view: Dict[str, Dict[str, Any]] = {}
                pred_args: Dict[str, Any] = {}
                n_non_null_pred = 0
                for p in gt_tool.parameters:
                    pr = res.parameters.get(p.name)
                    if pr is None:
                        continue
                    src_type = str(getattr(pr, "source_type", "") or "")
                    buckets[src_type] += 1
                    confs.append(float(getattr(pr, "confidence", 0.0) or 0.0))
                    is_null = pr.value is None and src_type == "null"
                    present = pr.value is not None or is_null
                    pred_view[p.name] = {
                        "present": present,
                        "is_null": is_null,
                        "value": None if is_null else pr.value,
                    }
                    if present:
                        pred_args[p.name] = None if is_null else pr.value
                        if not is_null:
                            n_non_null_pred += 1

                norm_pred = {kk: _norm(v) for kk, v in pred_args.items()}

                if n_non_null_pred:
                    sample_extracted = True
                exact_this_call = norm_pred == norm_gt
                if k == 0:
                    call0_exact = exact_this_call
                all_calls_exact = all_calls_exact and exact_this_call

                # Fix E3: per-call aligner-drift tripwire.
                if s.stratum == Stratum.F:
                    viol = e3_aligner_violation(s, gt_args, gt_tool, ci)
                    if viol is not None:
                        e3_violations += 1
                        if len(e3_examples) < 10:
                            tag = f"{ci}:" if multi_call else ""
                            e3_examples.append(f"{s.sample_id}:{tag}{viol}")

                # --- coverage / classification per param ---
                param_classes: Dict[str, str] = {}
                n_groundable_this_call = 0
                n_gt_this_call = 0
                for p in gt_tool.parameters:
                    pn = p.name
                    in_gt = pn in gt_universe
                    if in_gt:
                        n_gt_this_call += 1
                        total_gt_params += 1
                    is_groundable = param_is_groundable(s, pn, ci)
                    if in_gt and is_groundable:
                        n_groundable_this_call += 1
                        groundable_gt_params += 1
                    param_classes[pn] = classify_param(is_groundable)
                sample_gt_params += n_gt_this_call
                sample_groundable += n_groundable_this_call

                # --- blended (all-params) scoring, same as the legacy notebook ---
                for kk, v in norm_pred.items():
                    if kk in norm_gt and norm_gt[kk] == v:
                        all_tp += 1
                    else:
                        all_fp += 1
                        if kk in norm_gt:
                            # C4: a wrong value on a null GT param is a bad pointer.
                            pointer_wrong += 1
                for kk in norm_gt:
                    if kk not in norm_pred:
                        all_fn += 1

                # --- single classified pass (S2): headline + refusal split ---
                for p in gt_tool.parameters:
                    pn = p.name
                    cls = param_classes.get(pn, "unextractable")
                    pv = pred_view.get(pn)
                    present = pv is not None and bool(pv["present"])
                    if pn not in norm_gt:
                        # Predicted a param GT never mentions -> precision hit.
                        if present and cls == "extractable":
                            ex_fp += 1
                        continue
                    if cls != "extractable":
                        if not present:
                            refused_unextractable += 1
                        continue
                    gt_v = norm_gt[pn]
                    if not present:
                        ex_fn += 1
                        missed_extractable += 1
                        continue
                    pred_v = None if pv["is_null"] else _norm(pv["value"])
                    if pred_v == gt_v:
                        ex_tp += 1
                    else:
                        ex_fp += 1

                # --- per-param Stratum-F table ---
                # Multi-call rows are tagged ``call_index::param`` so two calls
                # declaring the same param name stay distinguishable instead of
                # being summed into one misleading row.
                if s.stratum == Stratum.F:
                    for p in gt_tool.parameters:
                        pn = p.name
                        st = per_param[key_prefix + pn]
                        cls = param_classes.get(pn, "unextractable")
                        if cls == "extractable":
                            st["extractable"] += 1
                        else:
                            st["unextractable"] += 1
                        if pn in norm_gt:
                            st["gt"] += 1
                        if pn in norm_pred:
                            if pn in norm_gt and norm_pred[pn] == norm_gt[pn]:
                                st["tp"] += 1
                            else:
                                st["fp"] += 1
                        elif pn in norm_gt:
                            st["fn"] += 1

            if sample_extracted:
                extracted += 1
            if sample_gt_params > 0 and sample_groundable == sample_gt_params:
                calls_all_groundable += 1

            # J6: set-level exactness — every GT call matched exactly AND the
            # predicted call count agrees (no missing call, no extra call).
            # ``exact_payload_match_F`` stays call-0-only so its historical
            # meaning is unchanged; this is the new headline that no longer
            # penalises a row merely for having a second call.
            set_exact = (
                all_calls_exact
                and len(per_call) == n_calls_gt
                and len(pred_calls) == n_calls_gt
            )
            if set_exact:
                call_set_exact += 1

            if s.stratum == Stratum.F:
                stratum_f_total += 1
                if call0_exact:
                    stratum_f_exact += 1
                if res0.hit:
                    stratum_f_hits += 1
                if set_exact:
                    call_set_exact_F += 1

            if limit is not None and total >= limit:
                break

    # --- metric computation ---
    ex_p = _ratio(ex_tp, ex_tp + ex_fp)
    ex_r = _ratio(ex_tp, ex_tp + ex_fn)
    all_p = _ratio(all_tp, all_tp + all_fp)
    all_r = _ratio(all_tp, all_tp + all_fn)
    f_exact = _ratio(stratum_f_exact, stratum_f_total)
    f_set_exact = _ratio(call_set_exact, total)
    f_set_exact_F = _ratio(call_set_exact_F, stratum_f_total)
    count_acc = _ratio(call_count_correct, total)
    n_multi_call = sum(v for k, v in n_calls_hist.items() if k > 1)
    f_resolver_hit = _ratio(stratum_f_hits, stratum_f_total)
    e3_rate = _ratio(e3_violations, stratum_f_total)
    mean_conf = sum(confs) / len(confs) if confs else float("nan")
    param_coverage = _ratio(groundable_gt_params, total_gt_params)
    call_coverage = _ratio(calls_all_groundable, total)
    mean_render_T = ((sum(t * c for t, c in render_T_hist.items()) / render_T_rows)
                     if render_T_rows else float("nan"))
    render_T_t1_share = ((render_T_hist.get(1, 0) / render_T_rows)
                         if render_T_rows else float("nan"))
    truncation = {
        "evaluated": total,
        "total_available": total_available,
        "truncated": bool(limit is not None and total >= limit and total_available > total),
    }

    metrics: Dict[str, Any] = {
        "split": split_name,
        "n": total,
        "stratum_f_n": stratum_f_total,
        # Headline fast-path metrics (extractable params only)
        "extraction_precision": _round(ex_p),
        "extraction_recall": _round(ex_r),
        "extraction_f1": _round(_f1(ex_p, ex_r)),
        # Legacy blended view (diagnostic)
        "recall_all": _round(all_r),
        "precision_all": _round(all_p),
        "f1_all": _round(_f1(all_p, all_r)),
        # Architecture-value metrics
        "fast_path_param_coverage": _round(param_coverage),
        "fast_path_call_coverage": _round(call_coverage),
        # Refusal vs failure split
        "refused_unextractable": refused_unextractable,
        "missed_extractable": missed_extractable,
        "pointer_wrong": pointer_wrong,
        "extracted": extracted,
        "exact_payload_match_F": _round(f_exact),
        "exact_payload_match_F_n": stratum_f_exact,
        # S5: this is the resolver's own schema-valid hit flag, NOT GT accuracy.
        "resolver_hit_rate_F": _round(f_resolver_hit),
        "resolver_hit_rate_F_n": stratum_f_hits,
        # J6 / R0.4: multi-call view. ``exact_payload_match_F`` above still
        # scores ONLY call 0 so its historical number is unchanged; these are the
        # set-level metrics that stop silently ignoring every call but the first.
        "call_exact_set_match": _round(f_set_exact),
        "call_exact_set_match_F": _round(f_set_exact_F),
        "call_exact_set_match_n": call_set_exact,
        "call_count_acc": _round(count_acc),
        "call_count_correct_n": call_count_correct,
        "n_calls_hist": {str(k): v for k, v in sorted(n_calls_hist.items())},
        "pred_n_calls_hist": {
            str(k): v for k, v in sorted(pred_n_calls_hist.items())
        },
        "n_multi_call_samples": n_multi_call,
        # Fix E3: aligner-drift tripwire (Stratum F only).
        "e3_aligner_drift": e3_violations,
        "e3_aligner_drift_rate": _round(e3_rate),
        "e3_examples": e3_examples,
        # S6: explicit truncation accounting.
        "truncation": truncation,
        # R9: the prompt layout the resolver actually rendered. A T=1 share of
        # 100% under a trainer that packed distractors is the silent
        # train/serve mismatch this metric exists to expose.
        "mean_render_T": _round(mean_render_T),
        "render_T_hist": {str(k): v for k, v in sorted(render_T_hist.items())},
        "render_T_rows": render_T_rows,
        "render_T_t1_share": _round(render_T_t1_share),
        # R2: the effective recent-context budget (post-clamp), so a silently
        # halved budget is visible instead of inferred.
        "recent_budget_effective": {
            str(k): v for k, v in sorted(recent_budget_effective.items())
        },
        "mean_confidence": _round(mean_conf),
        "param_counts": {
            "extractable_tp": ex_tp,
            "extractable_fp": ex_fp,
            "extractable_fn": ex_fn,
            "all_tp": all_tp,
            "all_fp": all_fp,
            "all_fn": all_fn,
            "total_gt_params": total_gt_params,
            "groundable_gt_params": groundable_gt_params,
        },
        "source_type_distribution": dict(buckets),
    }

    _print_report(metrics, per_param, max_param_gap_report)
    return metrics


def _print_report(
    metrics: Dict[str, Any],
    per_param: Dict[str, Dict[str, int]],
    max_param_gap_report: int,
) -> None:
    tr = metrics["truncation"]
    print(f"=== Retrieval-First Param Extraction (Oracle Tool) on {metrics['split']} "
          f"(N={metrics['n']} of {tr['total_available']} available) ===")
    if tr["truncated"]:
        print(f"  [!] truncated by limit: scored {tr['evaluated']} of {tr['total_available']} records")
    print(f"HEADLINE (extractable params only): Prec={_fmt(metrics['extraction_precision'])} "
          f"Rec={_fmt(metrics['extraction_recall'])} F1={_fmt(metrics['extraction_f1'])}")
    print(f"  -> legacy blended view (all GT args): Prec={_fmt(metrics['precision_all'])} "
          f"Rec={_fmt(metrics['recall_all'])} F1={_fmt(metrics['f1_all'])} (<= headline by construction)")
    gap = metrics["recall_all"] - metrics["extraction_recall"]
    print(f"  -> generation/out-of-scope gap: recall_all - extraction_recall = "
          f"{'n/a' if math.isnan(gap) else f'{gap:+.4f}'}")
    print(f"Exact Payload Match (Stratum F): {_fmt(metrics['exact_payload_match_F'])} "
          f"({metrics['exact_payload_match_F_n']} / {metrics['stratum_f_n']})")
    print(f"Resolver Schema-Valid Hit Rate (Stratum F): {_fmt(metrics['resolver_hit_rate_F'])} "
          f"({metrics['resolver_hit_rate_F_n']} / {metrics['stratum_f_n']})")
    print(f"Extraction Rate (>=1 non-null param): "
          f"{_fmt(_ratio(metrics['extracted'], metrics['n']))}")
    print(f"Mean candidate confidence: {_fmt(metrics['mean_confidence'])}")
    # R9: the prompt layout actually rendered, beside the extraction metrics it
    # is supposed to be comparable to. A T=1 share of 100% means this eval used
    # the oracle-tool layout while training packed distractors (R1b).
    print(f"Prompt layout: mean render_T={_fmt(metrics['mean_render_T'])} "
          f"hist={metrics['render_T_hist']} over {metrics['render_T_rows']} row(s); "
          f"T=1 share={_fmt(metrics['render_T_t1_share'])}")
    _rb = metrics.get("recent_budget_effective") or {}
    if _rb:
        print(f"Recent-context budget (effective, post-clamp): {_rb}")
    # J6 / R0.4: the call-count view. A sample with 2 GT calls used to be scored
    # on call 0 alone; ``call_exact_set_match`` is what stops that being invisible.
    print(f"Call count: GT hist={metrics['n_calls_hist']} "
          f"pred hist={metrics['pred_n_calls_hist']} | "
          f"count acc={_fmt(metrics['call_count_acc'])} "
          f"({metrics['call_count_correct_n']} / {metrics['n']})")
    print(f"Call set exact match: all={_fmt(metrics['call_exact_set_match'])} "
          f"({metrics['call_exact_set_match_n']} / {metrics['n']}) | "
          f"Stratum F={_fmt(metrics['call_exact_set_match_F'])} "
          f"(multi-call samples={metrics['n_multi_call_samples']})")
    print(f"FAST-PATH COVERAGE: {_fmt(metrics['fast_path_param_coverage'])} of GT params are extractable; "
          f"{_fmt(metrics['fast_path_call_coverage'])} of calls fully extractable")
    print(f"Refusal split: refused_unextractable={metrics['refused_unextractable']} "
          f"(correct refusals) | missed_extractable={metrics['missed_extractable']} (real fast-path failures) | "
          f"pointer_wrong={metrics['pointer_wrong']} (pred present, != GT)")
    print(f"E3 ALIGNER DRIFT (Stratum F): {metrics['e3_aligner_drift']} violating sample(s) "
          f"(rate={_fmt(metrics['e3_aligner_drift_rate'])}) — a non-zero value is a DATA bug, not a model miss")
    for ex in metrics["e3_examples"]:
        print(f"    drift: {ex}")

    if per_param:
        print("\n--- Per-parameter extraction (Stratum F only; extractable first, then most false-negatives) ---")
        print(f"{'param':<26}{'gt':>5}{'tp':>5}{'fp':>5}{'fn':>5}{'prec':>7}{'rec':>7}{'f1':>7}  {'cls':<12}{'ext':>4}{'unx':>4}")
        ranked = sorted(
            per_param.items(),
            key=lambda kv: (-kv[1]["extractable"], -kv[1]["fn"], -kv[1]["gt"]),
        )
        for pn, st in ranked[:max_param_gap_report]:
            prec = _ratio(st["tp"], st["tp"] + st["fp"])
            rec = _ratio(st["tp"], st["gt"])
            f1p = _f1(prec, rec)
            cls = "extractable" if st["extractable"] > 0 else "unextractable"
            print(f"{pn:<26}{st['gt']:>5}{st['tp']:>5}{st['fp']:>5}{st['fn']:>5}"
                  f"{_fmt(prec, 3):>7}{_fmt(rec, 3):>7}{_fmt(f1p, 3):>7}  {cls:<12}"
                  f"{st['extractable']:>4}{st['unextractable']:>4}")
    print("================================================================================================")