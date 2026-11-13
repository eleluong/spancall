"""Validation diagnostics for the Lean retrieval-first fast path.

Ref: ``docs/param_extraction_eval_coverage.md``

Part 1 (:mod:`spancall.lean.eval_p1`) reports *headline* numbers. This module
answers the follow-up question a low headline always raises — **why** — with
three independent breakdowns plus the one audit that turned out to explain most
of the gap:

1. :func:`audit_gt_offsets` — *ground-truth span validity.* Every
   ``utterance_span`` target carries a ``(char_start, char_end)`` pair that is
   only meaningful against the render it was produced from. If the render used
   at train/eval time differs by even one character, the stored offsets address
   the wrong text and the pointer is supervised (and parity-scored) against
   garbage. This audit re-renders each sample with the SAME grammar the resolver
   uses and measures how many stored spans still slice their own canonical value.
   It is the difference between "the model picked wrong" and "the label pointed
   at the wrong characters".

2. :func:`analyze_predictions` — reads a ``preds_*.jsonl`` written by
   ``predict_split`` and splits the parameter errors three ways:

   - **source_type precision** (restricted to *extractable* GT params), so a
     ``null`` prediction (often right) is never averaged into an ``utterance``
     prediction (often wrong);
   - **predicted-span region histogram** over ``anchor/ctx`` vs ``utterance`` —
     the pointer-collapse signature is a span that lands in the schema/anchor
     block instead of the user's words;
   - **confidence of correct vs wrong** predictions on the groundable subset, so
     a calibrated-but-weak pointer is distinguishable from an uncalibrated one.

   It also counts *degenerate repeats* (the same value filling several params of
   one row) and *boilerplate hits* (the predicted value equals the first
   utterance token), both of which indicate a collapsed pointer rather than a
   near-miss retrieval.

3. :func:`report` / :func:`run_diagnostics` — printing and one-call driver.

Design constraints
------------------
* **Diagnosis only.** Nothing here changes the resolver, the metrics, or the
  candidate set. These are read-only measurements so a fix can be chosen from
  evidence instead of guessed.
* **No fabricated numbers.** Every rate is ``nan`` when its denominator is
  empty, and every counter is reported alongside the size of the population it
  came from.
* **Same rules as the headline.** Value comparison goes through
  :func:`spancall.lean.eval_p1.norm_value` and groundability through
  :func:`spancall.lean.eval_p1.param_is_groundable`, so this module cannot
  drift into a second, more flattering definition of "correct".
* **Heavy imports are lazy.** ``span_template`` pulls in ``jinja2`` and the
  aligner; they are imported inside the functions so this module is safe to
  import in a torch-free context (tests, CI) as long as the render is injected.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from spancall.data.schema import UnifiedSample
from spancall.lean.eval_p1 import (
    _gt_params,
    norm_value,
    param_is_groundable,
)

__all__ = [
    "audit_gt_offsets",
    "analyze_predictions",
    "run_diagnostics",
    "report",
    "load_split",
    "render_for_sample",
    "classify_span_region",
]

#: ``source_type`` values that assert "no value was produced". Everything else
#: (``utterance``/``recent``/``null``/``bool``/``enum``) is a prediction.
ABSENT_SOURCE_TYPES = frozenset({"missing", "refused"})

#: Counters that are always present in :func:`analyze_predictions` output, even
#: when they never fired. A counter that appears only sometimes forces every
#: reader (report, notebook, test) to guess between "0" and "key missing", and
#: silently reporting nothing is indistinguishable from reporting zero.
_COUNTER_DEFAULTS: Dict[str, int] = {
    "rows": 0,
    "unjoined": 0,
    "joined_by_index": 0,
    "rendered": 0,
    "render_failed": 0,
    "n_multi_call_rows": 0,
    "wrong_tool_rows": 0,
    "empty_payload_hits": 0,
    "gt_tool_missing": 0,
    "gt_params": 0,
    "gt_params_extractable": 0,
    "gt_params_unextractable": 0,
    "gt_param_absent_from_args": 0,
    "extractable_absent": 0,
    "correct": 0,
    "wrong": 0,
    "rows_multi_value": 0,
    "rows_all_values_identical": 0,
    "preds_equal_first_utterance_token": 0,
}

#: Where a predicted char span landed relative to the user's utterance.
REGION_IN_UTTERANCE = "in_utterance"
REGION_BEFORE_UTTERANCE = "before_utterance"   # schema/anchor/ctx block
REGION_AFTER_UTTERANCE = "after_utterance"     # recent rows rendered after
REGION_NO_UTTERANCE = "no_utterance_entry"     # render carried no utterance slice
REGION_NO_SPAN = "no_span"

#: Confidence bucket upper bounds. A pointer that outputs ~0.015 everywhere
#: collapses every parameter into the first bucket, which is itself the signal.
CONF_BUCKETS: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.01)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _ratio(num: int, den: int) -> float:
    """Ratio that is ``nan`` — never a fake 1.0 — when the denominator is empty."""
    return num / den if den else float("nan")


def _round(v: float, nd: int = 4) -> float:
    return float("nan") if isinstance(v, float) and math.isnan(v) else round(v, nd)


def _fmt(v: Any, nd: int = 4) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _bucket_label(conf: float) -> str:
    for edge in CONF_BUCKETS:
        if conf < edge:
            return f"<{edge:g}"
    return f">={CONF_BUCKETS[-2]:g}"


def _bucket_index(label: str) -> int:
    """Sort key for a :func:`_bucket_label` string.

    The labels are ``"<0.02"``, ``"<0.05"``, ... — lexicographic order puts
    ``"<0.1"`` *before* ``"<0.02"``, so a plain ``sorted()`` would print the
    confidence bands shuffled. Map each label back to its position in
    :data:`CONF_BUCKETS` instead.
    """
    for i, edge in enumerate(CONF_BUCKETS):
        if label == f"<{edge:g}":
            return i
    return len(CONF_BUCKETS)  # the overflow band, always last


def _pred_present(detail: Dict[str, Any]) -> bool:
    """True when a predicted parameter asserts a value (null counts as a value).

    Mirrors :mod:`spancall.lean.eval_p1`'s null-aware ``pred_view``: selecting the
    deterministic ``<|null|>`` candidate is a prediction, while ``missing`` /
    ``refused`` / a failed cast is not.
    """
    src = str(detail.get("source_type") or "")
    if src in ABSENT_SOURCE_TYPES or src == "cast_fail":
        return False
    return detail.get("value") is not None or src == "null"


def load_split(data_dir: str, split_name: str, limit: Optional[int] = None) -> List[UnifiedSample]:
    """Load ``{split_name}.jsonl`` as :class:`UnifiedSample` objects, in file order.

    File order matters: it is the order ``predict_split`` consumed, so index
    ``i`` here lines up with the ``i``-th prediction record when the join by
    ``sample_id`` is not available (e.g. an older predictions file).

    Raises:
        FileNotFoundError: a missing split must be loud, not an empty list —
            an empty list silently reports every rate as ``n/a``, which reads
            like a clean run.
    """
    src = os.path.join(data_dir, f"{split_name}.jsonl")
    if not os.path.exists(src):
        raise FileNotFoundError(f"split not found: {src}")
    out: List[UnifiedSample] = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            out.append(UnifiedSample.from_dict(json.loads(line)))
            if limit is not None and len(out) >= limit:
                break
    return out


def _load_preds(preds_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"prediction file not found: {preds_path}")
    out: List[Dict[str, Any]] = []
    with open(preds_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def render_for_sample(
    sample: UnifiedSample,
    *,
    all_tools: bool = True,
) -> Tuple[Any, Tuple[Optional[int], Optional[int]]]:
    """Render a sample with the grammar the RESOLVER uses; return ``(output, utt_span)``.

    The resolver renders every context schema (``context_schemas=sample.tools``),
    which is ``build_input_multi`` — *not* the single-tool ``build_input`` the
    v1 aligner used. Getting this wrong is exactly the class of bug this audit
    exists to catch, so the render is explicit and injectable rather than
    inferred.

    BOTH modes deliberately use ``build_input_multi``: the v2 aligner renders
    its own tool through that same grammar (``tools=[tool]``), so
    ``all_tools=False`` is the only mode that can prove "the aligner and its
    render agree". Rendering it with ``build_input`` would reintroduce the
    ``<|none|>``-line discrepancy the v2 re-alignment exists to remove, and the
    audit would then report a *permanent* stale rate no re-alignment can fix.

    Args:
        sample: the sample to render.
        all_tools: when True (default / resolver behaviour) pack every ambient
            tool; when False render only the GT call's tool, which is what
            :class:`~spancall.data.align.ReverseAligner` does (same grammar,
            one ``<|tool|>`` block).

    Returns:
        ``(SpanTemplateOutput, (char_start, char_end))`` where the span is the
        rendered utterance slice, or ``(None, None)`` values when the render
        carries no selectable utterance entry.
    """
    from datetime import date

    from span_template.template import RecentRow, build_input_multi

    from spancall.data.align import tooldef_to_template_tool

    if not sample.ground_truth_calls or not sample.tools:
        raise ValueError("sample has no GT calls or no tools")
    gt_tool = next(
        (t for t in sample.tools if t.name == sample.ground_truth_calls[0].tool_name),
        None,
    )
    if gt_tool is None:
        raise ValueError(f"GT tool {sample.ground_truth_calls[0].tool_name!r} not among sample.tools")

    recent = [
        RecentRow(
            key=str(r.get("key", "")),
            value=str(r.get("value", "")),
            source=str(r.get("source", "recent")),
        )
        for r in (sample.recent_context or [])
    ]
    now = (
        date.fromisoformat(sample.current_time_iso[:10])
        if sample.current_time_iso
        else date(2026, 1, 1)
    )

    # Same grammar in both modes: ``all_tools`` only changes how many
    # ``<|tool|>`` blocks are packed, never the template. That is what makes
    # ``all_tools=False`` a valid "the aligner agrees with its own render"
    # check -- see the docstring.
    tools = (
        [tooldef_to_template_tool(t) for t in sample.tools]
        if all_tools
        else [tooldef_to_template_tool(gt_tool)]
    )
    output = build_input_multi(
        tools,
        sample.utterance,
        recent_context=recent,
        now=now,
    )

    utt_entries = [e for e in output.entries if e.segment == "utterance" and e.selectable]
    if not utt_entries:
        return output, (None, None)
    return output, (
        min(e.char_start for e in utt_entries),
        max(e.char_end for e in utt_entries),
    )


def classify_span_region(
    char_span: Optional[Sequence[int]],
    utt_span: Tuple[Optional[int], Optional[int]],
    text_len: Optional[int] = None,
) -> str:
    """Classify a predicted char span as in/before/after the rendered utterance.

    A pointer that collapses onto the schema or ``<|ctx_*|>`` block produces
    spans that are *entirely outside* the utterance, which is a different defect
    from picking a wrong word *inside* it. Collapsing them into one "wrong"
    bucket is what hides the collapse.

    Args:
        char_span: ``(char_start, char_end)`` of the predicted candidate.
        utt_span: rendered utterance bounds from :func:`render_for_sample`.
        text_len: length of the rendered text, used only to sanity-check offsets.

    Returns:
        One of the ``REGION_*`` constants.
    """
    if not char_span or len(char_span) < 2:
        return REGION_NO_SPAN
    cs, ce = int(char_span[0]), int(char_span[1])
    if ce < cs:
        cs, ce = ce, cs
    if text_len is not None and (cs < 0 or ce > text_len):
        # An offset past the end of the render cannot be an utterance span.
        return REGION_AFTER_UTTERANCE
    us, ue = utt_span
    if us is None or ue is None:
        return REGION_NO_UTTERANCE
    if ce <= us:
        return REGION_BEFORE_UTTERANCE
    if cs >= ue:
        return REGION_AFTER_UTTERANCE
    return REGION_IN_UTTERANCE


# --------------------------------------------------------------------------- #
# 1. Ground-truth span validity audit
# --------------------------------------------------------------------------- #
def audit_gt_offsets(
    split_name: str,
    data_dir: str,
    limit: Optional[int] = None,
    all_tools: bool = True,
    max_examples: int = 8,
) -> Dict[str, Any]:
    """Check whether stored GT char offsets still address their own value.

    For every ``utterance_span`` target, re-render the sample and compare
    ``render.text[char_start:char_end]`` against the target's ``canonical_value``.
    A mismatch means the label is **stale**: the offsets were computed against a
    different render (different tool packing, different context rows, a different
    template), so the pointer head was supervised on the wrong characters and
    ``candidate_parity`` scores the drift rather than the candidate set.

    When a mismatch is found, the value's TRUE offsets are recovered with the
    aligner's own word-bounded matcher, so the report can state the drift
    in characters instead of just "broken".

    Args:
        split_name: canonical split name (e.g. ``hermes_eval``).
        data_dir: directory holding ``{split_name}.jsonl``.
        limit: max samples audited (``None`` = whole split).
        all_tools: render every ambient tool (resolver behaviour, default) or
            only the GT tool (aligner behaviour). Auditing BOTH is how a
            grammar mismatch between training and serving becomes visible.
        max_examples: cap on the examples carried in the returned dict.

    Returns:
        A dict of counts, rates and example rows (also printed).
    """
    from spancall.data.align import _find_utterance_entry, _word_bounded_matches

    samples = load_split(data_dir, split_name, limit)
    stats: Counter = Counter()
    drift_hist: Counter = Counter()
    # Drift measured for targets whose ABSOLUTE pair is stale but whose newer
    # render-invariant pair still resolves. Kept separate from ``drift_hist``:
    # that one measures "label is broken, here is where the value really is",
    # this one measures "label was repaired by v2, here is how far it had rotted".
    rel_drift_hist: Counter = Counter()
    by_target_type: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "valid": 0,
            "stale_recoverable": 0,
            "stale_lost": 0,
            # Tracked per type because it is only a defect for the types that
            # are supposed to live in the utterance; see the utterance-span
            # scoped gate metrics below.
            "unresolvable": 0,
        }
    )
    examples: List[str] = []
    rendered = 0

    for sample in samples:
        if not sample.ground_truth_calls or not sample.tools:
            stats["samples_without_gt_or_tools"] += 1
            continue
        targets = sample.targets_for_call(0)
        span_targets = [
            (pn, t)
            for pn, ts in targets.items()
            for t in ts
            if getattr(t, "char_start", None) is not None
            and getattr(t, "char_end", None) is not None
        ]
        if not span_targets:
            # No char offsets at all: nothing to audit. Counted so the
            # "rendered" denominator is not silently smaller than "read".
            stats["samples_without_span_targets"] += 1
            continue
        stats["samples_with_span_targets"] += 1
        try:
            output, _utt = render_for_sample(sample, all_tools=all_tools)
        except Exception as exc:  # sanitize / render failure is a data fact, not a crash
            stats["render_failed"] += 1
            if len(examples) < max_examples:
                examples.append(f"{sample.sample_id}: render failed ({type(exc).__name__}: {exc})")
            continue
        rendered += 1
        utt_entry = _find_utterance_entry(output)
        utt_value = str(utt_entry.value) if utt_entry is not None and isinstance(utt_entry.value, str) else ""

        for pname, target in span_targets:
            ttype = str(getattr(target, "target_type", "") or "?")
            stats["span_targets"] += 1
            stats[f"target_type:{ttype}"] += 1
            canon = getattr(target, "canonical_value", None)
            cs, ce = int(target.char_start), int(target.char_end)
            sliced = output.text[cs:ce] if 0 <= cs <= ce <= len(output.text) else ""
            canon_s = str(canon).strip().lower() if canon is not None else None

            # Label-generation accounting happens BEFORE the validity shortcut:
            # a target whose absolute pair is still correct must still be
            # counted in the rel-pair population, otherwise
            # ``absolute_stale_rate`` has a denominator of zero on a perfectly
            # aligned file and reports ``nan`` (which reads as "not measured"
            # even though the answer is "nothing was stale").
            rel_cs = getattr(target, "utt_rel_start", None)
            rel_ce = getattr(target, "utt_rel_end", None)
            has_rel_pair = rel_cs is not None and rel_ce is not None
            stats["span_with_rel_pair" if has_rel_pair else "span_without_rel_pair"] += 1

            if canon_s is not None and sliced.strip().lower() == canon_s:
                stats["valid"] += 1
                stats["valid_via_absolute"] += 1
                by_target_type[ttype]["valid"] += 1
                continue

            # The absolute pair is stale. Before declaring the label dead, try
            # the render-invariant pair: it is the whole point of v2, so an
            # audit that only reads ``char_*`` would report a permanent 100%
            # stale rate on data that is in fact correct.
            rel_span = None
            if has_rel_pair:
                from span_template.template import resolve_utterance_span

                rel_span = resolve_utterance_span(
                    output,
                    int(rel_cs),
                    int(rel_ce),
                    expect=str(canon) if canon is not None else None,
                )
                if rel_span is None:
                    stats["rel_pair_unresolvable"] += 1
                    by_target_type[ttype]["unresolvable"] += 1

            if rel_span is not None:
                stats["valid"] += 1
                stats["valid_via_rel"] += 1
                by_target_type[ttype]["valid"] += 1
                # How far the stored absolute offsets had drifted. This is the
                # measured size of the bug the rel pair removes, and it is the
                # number the spawned notebook can compare v1 vs v2 on.
                rel_drift_hist[rel_span[0] - cs] += 1
                if len(examples) < max_examples:
                    examples.append(
                        f"{sample.sample_id}:{pname} absolute=({cs},{ce}) is STALE but the rel "
                        f"pair recovers ({rel_span[0]},{rel_span[1]}) "
                        f"(drift={rel_span[0] - cs:+d})"
                    )
                continue

            # Broken. Is the value still findable in the current render?
            #
            # Only ``utterance_span`` targets can be recovered from the
            # utterance slice — a ``boolean_ctx``/``recent_ctx`` target legitimately
            # points outside it, so reporting those as "lost" would be wrong.
            recovered = None
            if (
                canon is not None
                and utt_entry is not None
                and utt_value
                and ttype == "utterance_span"
            ):
                occ = _word_bounded_matches(utt_value, str(canon))
                if occ:
                    # Several occurrences are normal for a short value ("2",
                    # "Springfield"). Taking occ[0] would report a huge bogus
                    # negative drift whenever the GT meant the second one, so
                    # take the occurrence nearest the stored offset — the drift
                    # then measures render displacement, which is what we want.
                    best = min(occ, key=lambda o: abs(utt_entry.char_start + o[0] - cs))
                    recovered = (
                        utt_entry.char_start + best[0],
                        utt_entry.char_start + best[1],
                    )

            if recovered is not None:
                stats["stale_recoverable"] += 1
                stats["stale_recoverable_in_utterance"] += 1
                by_target_type[ttype]["stale_recoverable"] += 1
                drift_hist[recovered[0] - cs] += 1
                if len(examples) < max_examples:
                    examples.append(
                        f"{sample.sample_id}:{pname} stored=({cs},{ce}) "
                        f"true=({recovered[0]},{recovered[1]}) drift={recovered[0] - cs:+d} "
                        f"stored_text={sliced!r} canon={str(canon)[:40]!r}"
                    )
            else:
                stats["stale_lost"] += 1
                by_target_type[ttype]["stale_lost"] += 1
                if len(examples) < max_examples:
                    if ttype == "utterance_span":
                        why = "canonical value NOT FOUND in the retained utterance"
                    else:
                        why = "non-utterance target; value not recoverable from the utterance"
                    examples.append(
                        f"{sample.sample_id}:{pname} stored=({cs},{ce}) "
                        f"text={sliced!r} canon={str(canon)[:40]!r} {why}"
                    )

    total = stats["span_targets"]
    usable = stats["valid"] + stats["stale_recoverable"] + stats["stale_lost"]
    # A rel-pair target whose ABSOLUTE pair is also valid never reaches the rel
    # branch above, so ``valid_via_absolute`` counts it. Reporting both is what
    # makes "the rel pair works" distinguishable from "the absolute pair
    # happens to still be right because nothing drifted".
    absolute_stale = stats["valid_via_rel"] + stats["rel_pair_unresolvable"]
    # Restrict the acceptance gate to utterance-span targets (see the comment on
    # the metrics below). ``stale`` here means "carries an offset pair that no
    # longer addresses its own value", which for an utterance target is always
    # a labelling bug -- unlike a ``recent_ctx`` target, whose offset is
    # expected to fall outside the utterance by construction.
    utt_stats = by_target_type.get("utterance_span") or {}
    utt_valid = int(utt_stats.get("valid", 0))
    utt_stale = int(utt_stats.get("stale_recoverable", 0)) + int(
        utt_stats.get("stale_lost", 0)
    )
    utt_unresolvable = int(utt_stats.get("unresolvable", 0))
    utt_total = utt_valid + utt_stale
    metrics: Dict[str, Any] = {
        "split": split_name,
        "render_mode": "all_tools(resolver)" if all_tools else "gt_tool(aligner)",
        # Why `samples_rendered` can be far below `samples_read`: the split is
        # mostly multi-call/negative rows whose CALL 0 carries no char offsets,
        # and this audit only scores call 0. Both numbers are reported so the
        # gap is visible instead of looking like silent skipping.
        "samples_read": len(samples),
        "samples_rendered": rendered,
        "samples_with_span_targets": stats["samples_with_span_targets"],
        "samples_without_span_targets": stats["samples_without_span_targets"],
        "samples_without_gt_or_tools": stats["samples_without_gt_or_tools"],
        "render_failed": stats["render_failed"],
        "span_targets": total,
        "span_valid": stats["valid"],
        "span_stale_recoverable": stats["stale_recoverable"],
        "span_stale_recoverable_in_utterance": stats["stale_recoverable_in_utterance"],
        "span_stale_lost": stats["stale_lost"],
        "span_valid_rate": _round(_ratio(stats["valid"], usable)),
        "span_stale_rate": _round(
            _ratio(stats["stale_recoverable"] + stats["stale_lost"], usable)
        ),
        # --- v2 render-invariance accounting ---------------------------------
        # ``span_with_rel_pair`` / ``span_without_rel_pair`` split the population
        # by label generation. A v2 file must be ~100% rel; a v1 file is 0%, so
        # this is how the audit tells the two datasets apart instead of guessing
        # from the stale rate alone.
        "span_with_rel_pair": stats["span_with_rel_pair"],
        "span_without_rel_pair": stats["span_without_rel_pair"],
        "span_valid_via_absolute": stats["valid_via_absolute"],
        "span_valid_via_rel": stats["valid_via_rel"],
        "rel_pair_unresolvable": stats["rel_pair_unresolvable"],
        # Size of the v1 defect: of the targets that carry the rel pair, how
        # many had a stale absolute pair? This is REPORTING ONLY -- a train row
        # packs 2-5 tools while the aligner renders one, so the absolute pair is
        # legitimately stale there and a high value is not a defect.
        "absolute_stale_rate": _round(_ratio(absolute_stale, stats["span_with_rel_pair"])),
        # The actual gate: of the rel-pair targets, how many could not be mapped
        # onto this render at all (utterance text drifted away from the label)?
        "unresolvable_rate": _round(
            _ratio(stats["rel_pair_unresolvable"], stats["span_with_rel_pair"])
        ),
        "rel_drift_hist": {str(k): v for k, v in sorted(rel_drift_hist.items())},
        # --- utterance-span scoped rates: THE acceptance gate ----------------
        #
        # ``span_stale_rate`` counts every target type, which makes it the wrong
        # gate for a v2 file: ``recent_ctx``/``time_ctx``/``enum_ctx``/
        # ``boolean_ctx`` targets legitimately point OUTSIDE the utterance (a
        # recent-context row, a clock window), so they can never be span-valid
        # and ``tensorize`` resolves them by ``canonical_value`` instead. Scoring
        # them as "stale" makes a perfectly aligned multi-tool split look 16%
        # broken and hides whether the thing v2 fixed is actually fixed.
        #
        # These three rates are restricted to ``target_type == "utterance_span"``,
        # i.e. exactly the population the pointer head is trained on.
        "utterance_span_targets": utt_total,
        "utterance_span_valid_rate": _round(_ratio(utt_valid, utt_total)),
        "utterance_span_stale_rate": _round(_ratio(utt_stale, utt_total)),
        "utterance_span_unresolvable": utt_unresolvable,
        # Drift is reported as a histogram because a single offset (e.g. one
        # dropped/inserted line) explains a "broken everywhere" rate, while a
        # broad spread means the render grammar itself diverged.
        "drift_hist": {str(k): v for k, v in sorted(drift_hist.items())},
        "by_target_type": {k: dict(v) for k, v in sorted(by_target_type.items())},
        "examples": examples,
    }
    return metrics


# --------------------------------------------------------------------------- #
# 2. Prediction error decomposition
# --------------------------------------------------------------------------- #
def analyze_predictions(
    preds_path: str,
    split_name: str,
    data_dir: str,
    limit: Optional[int] = None,
    max_examples: int = 6,
) -> Dict[str, Any]:
    """Decompose parameter errors from a ``preds_*.jsonl`` file.

    Three breakdowns, each answering a different question:

    * ``by_source_type`` — of the *extractable* GT params, how often is a
      prediction correct when its ``source_type`` is ``utterance`` vs ``null``
      vs ``recent``? A high ``null`` precision beside a near-zero ``utterance``
      precision says the deterministic slots were learned and the free-span
      retrieval was not.
    * ``by_region`` — where do predicted spans land? Spans outside the utterance
      are a collapse onto anchors/ctx, a different bug from picking a wrong word.
    * ``by_confidence`` — is the pointer miscalibrated (correct and wrong
      predictions share a confidence band) or merely weak (correct ones are
      confidently higher)?

    Args:
        preds_path: path to the predictions JSONL written by ``predict_split``.
        split_name: split whose raw JSONL is re-read for schemas + targets.
        data_dir: directory holding ``{split_name}.jsonl``.
        limit: max prediction records analysed.
        max_examples: cap on examples per category.

    Returns:
        A nested dict of counts and rates (also printed by :func:`report`).

    Notes:
        Only call 0 is analysed, matching ``exact_payload_match_F``. Multi-call
        rows are counted in ``n_multi_call_rows`` so the reader knows how much
        of the file this excludes.
    """
    preds = _load_preds(preds_path, limit)
    samples = load_split(data_dir, split_name, None)
    by_id = {s.sample_id: s for s in samples}
    by_index = samples

    src_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "wrong": 0, "absent": 0}
    )
    gt_type_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "wrong": 0})
    region_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0})
    conf_correct: List[float] = []
    conf_wrong: List[float] = []
    conf_bucket_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0})

    counts: Counter = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)

    for idx, rec in enumerate(preds):
        # Join rules, deliberately asymmetric:
        #
        # * a record that HAS a sample_id must match it. Falling back to
        #   position would score its prediction against a different sample's
        #   schema and GT — a confident, meaningless precision.
        # * only a legacy record with NO sample_id is joined by position, and
        #   that path is counted so the report can disclose how much used it.
        sid = str(rec.get("sample_id") or "")
        sample: Optional[UnifiedSample]
        if sid:
            sample = by_id.get(sid)
            if sample is None:
                counts["unjoined"] += 1
                if len(examples["unjoined"]) < max_examples:
                    examples["unjoined"].append(
                        f"{sid}: not present in split {split_name!r}"
                    )
                continue
        elif idx < len(by_index):
            sample = by_index[idx]
            counts["joined_by_index"] += 1
        else:
            sample = None
        if sample is None:
            counts["unjoined"] += 1
            continue
        counts["rows"] += 1
        counts[f"status:{rec.get('status')}"] += 1

        if int(rec.get("n_gt_calls") or 0) > 1:
            counts["n_multi_call_rows"] += 1

        # --- boilerplate "hit": empty payload over an all-optional schema ---
        payload = rec.get("payload") or {}
        partial = rec.get("payload_partial") or {}
        if rec.get("hit") and not payload and not partial:
            counts["empty_payload_hits"] += 1
            if len(examples["empty_payload_hit"]) < max_examples:
                examples["empty_payload_hit"].append(
                    f"{sample.sample_id}: hit with no parameter, "
                    f"optional={rec.get('missing_optional')}"
                )

        if not sample.ground_truth_calls or not sample.tools:
            continue
        gt_call = sample.ground_truth_calls[0]
        gt_tool = next((t for t in sample.tools if t.name == gt_call.tool_name), None)
        if gt_tool is None:
            counts["gt_tool_missing"] += 1
            continue
        if rec.get("tool_name") and rec.get("gt_tool") and rec["tool_name"] != rec["gt_tool"]:
            counts["wrong_tool_rows"] += 1

        gt_universe = _gt_params(dict(gt_call.arguments or {}), gt_tool)
        params = rec.get("parameters") or {}

        # --- rendered utterance bounds, for region classification ---
        utt_span: Tuple[Optional[int], Optional[int]] = (None, None)
        text_len: Optional[int] = None
        try:
            output, utt_span = render_for_sample(sample, all_tools=True)
            text_len = len(output.text)
            counts["rendered"] += 1
        except Exception:
            counts["render_failed"] += 1

        # --- pointer-collapse signatures (row level) ---
        present_vals = [
            str(d.get("value"))
            for d in params.values()
            if _pred_present(d) and d.get("value") is not None
        ]
        if len(present_vals) >= 2:
            counts["rows_multi_value"] += 1
            if len(set(present_vals)) == 1:
                counts["rows_all_values_identical"] += 1
                if len(examples["degenerate_repeat"]) < max_examples:
                    examples["degenerate_repeat"].append(
                        f"{sample.sample_id}: all {len(present_vals)} params = {present_vals[0]!r}"
                    )
        utt_words = str(sample.utterance or "").split()
        first_word = utt_words[0] if utt_words else None
        if first_word:
            for d in params.values():
                if _pred_present(d) and isinstance(d.get("value"), str):
                    if d["value"].strip().lower() == first_word.lower():
                        counts["preds_equal_first_utterance_token"] += 1
                        break

        # --- per-parameter scoring, restricted to extractable GT params ---
        for p in gt_tool.parameters:
            pname = p.name
            detail = params.get(pname) or {}
            src = str(detail.get("source_type") or "unpredicted")
            present = bool(detail) and _pred_present(detail)
            groundable = param_is_groundable(sample, pname, 0)
            target_types = sorted(
                {
                    str(getattr(t, "target_type", "") or "?")
                    for t in (sample.targets_for_call(0).get(pname) or [])
                }
            )
            ttype = target_types[0] if target_types else "no_target"

            if pname not in gt_universe:
                counts["gt_param_absent_from_args"] += 1
                continue
            counts["gt_params"] += 1
            if not groundable:
                counts["gt_params_unextractable"] += 1
                continue
            counts["gt_params_extractable"] += 1
            gt_val = norm_value(gt_universe[pname])

            if not present:
                src_stats[src]["absent"] += 1
                gt_type_stats[ttype]["n"] += 1
                counts["extractable_absent"] += 1
                continue

            pred_is_null = src == "null"
            pred_val = None if pred_is_null else norm_value(detail.get("value"))
            correct = pred_val == gt_val

            src_stats[src]["n"] += 1
            src_stats[src]["correct" if correct else "wrong"] += 1
            gt_type_stats[ttype]["n"] += 1
            gt_type_stats[ttype]["correct" if correct else "wrong"] += 1

            region = classify_span_region(detail.get("char_span"), utt_span, text_len)
            counts[f"region:{region}"] += 1
            region_stats[region]["correct" if correct else "wrong"] += 1

            conf = float(detail.get("confidence") or 0.0)
            bucket = _bucket_label(conf)
            conf_bucket_stats[bucket]["correct" if correct else "wrong"] += 1
            if correct:
                conf_correct.append(conf)
                counts["correct"] += 1
            else:
                conf_wrong.append(conf)
                counts["wrong"] += 1
                if len(examples["wrong_with_repeat"]) < max_examples and len(set(present_vals)) == 1 and len(present_vals) >= 2:
                    examples["wrong_with_repeat"].append(
                        f"{sample.sample_id}:{pname} pred={detail.get('value')!r} gt={gt_val!r} "
                        f"src={src} span={detail.get('char_span')} conf={conf:.3f}"
                    )
                if len(examples["wrong_span"]) < max_examples and region != REGION_IN_UTTERANCE:
                    examples["wrong_span"].append(
                        f"{sample.sample_id}:{pname} region={region} "
                        f"pred={detail.get('value')!r} gt={gt_val!r} "
                        f"span={detail.get('char_span')} utt={utt_span}"
                    )

    n_scored = counts["correct"] + counts["wrong"]
    by_source = {
        k: {
            **v,
            "precision": _round(_ratio(v["correct"], v["n"])),
        }
        for k, v in sorted(src_stats.items())
    }
    by_region = {
        k: {**v, "share": _round(_ratio(v["correct"] + v["wrong"], n_scored))}
        for k, v in sorted(region_stats.items())
    }
    # Sorted by band position (low confidence first), not by label string.
    by_conf = {
        k: {
            **v,
            "precision": _round(_ratio(v["correct"], v["correct"] + v["wrong"])),
        }
        for k, v in sorted(conf_bucket_stats.items(), key=lambda kv: _bucket_index(kv[0]))
    }
    by_gt_type = {
        k: {**v, "precision": _round(_ratio(v["correct"], v["n"]))}
        for k, v in sorted(gt_type_stats.items())
    }

    merged_counts = dict(_COUNTER_DEFAULTS)
    merged_counts.update(counts)

    return {
        "preds_path": preds_path,
        "split": split_name,
        "counts": merged_counts,
        "extractable_precision_scored": _round(_ratio(counts["correct"], n_scored)),
        "n_scored": n_scored,
        "by_source_type": by_source,
        "by_gt_target_type": by_gt_type,
        "by_region": by_region,
        "by_confidence": by_conf,
        "confidence_correct": {
            "n": len(conf_correct),
            "mean": _round(sum(conf_correct) / len(conf_correct)) if conf_correct else float("nan"),
            "median": _round(_median(conf_correct)),
        },
        "confidence_wrong": {
            "n": len(conf_wrong),
            "mean": _round(sum(conf_wrong) / len(conf_wrong)) if conf_wrong else float("nan"),
            "median": _round(_median(conf_wrong)),
        },
        "examples": {k: v for k, v in examples.items()},
    }


# --------------------------------------------------------------------------- #
# 3. Printing / driver
# --------------------------------------------------------------------------- #
def report(diag: Dict[str, Any]) -> None:
    """Print :func:`analyze_predictions` / :func:`audit_gt_offsets` results."""
    if "span_targets" in diag:
        _report_offsets(diag)
        return
    _report_predictions(diag)


def _report_offsets(d: Dict[str, Any]) -> None:
    print("=" * 96)
    print(f"GT SPAN VALIDITY AUDIT — {d['split']}  [render={d['render_mode']}]")
    print("=" * 96)
    print(f"  samples read/rendered : {d['samples_read']} / {d['samples_rendered']}"
          f"  (render failures={d['render_failed']})")
    print(f"    of the read: call-0 had span targets={d.get('samples_with_span_targets', 0)},"
          f" no span targets={d.get('samples_without_span_targets', 0)},"
          f" no GT/tools={d.get('samples_without_gt_or_tools', 0)}")
    print(f"  spannable GT targets  : {d['span_targets']}")
    print(f"  span still VALID      : {d['span_valid']}  ({_fmt(d['span_valid_rate'], 4)} of all targets)")
    print(f"  span STALE (value still findable)        : {d['span_stale_recoverable']}")
    print(f"  span STALE (value lost / non-utterance)  : {d['span_stale_lost']}")
    print(f"  => stale rate         : {_fmt(d['span_stale_rate'], 4)}")
    print(f"  => UTTERANCE_SPAN valid rate: {_fmt(d.get('utterance_span_valid_rate'), 4)}"
          "   <- the number that governs the pointer head")
    if "utterance_span_targets" in d:
        # The acceptance gate. ``span_stale_rate`` above counts context targets
        # (recent_ctx / time_ctx / enum_ctx / boolean_ctx), which are resolved by
        # ``canonical_value`` in tensorize and are *expected* to sit outside the
        # utterance, so it is not a defect rate on a multi-tool split.
        print(f"  => UTTERANCE_SPAN stale rate: {_fmt(d.get('utterance_span_stale_rate'), 4)}"
              f"   over {d.get('utterance_span_targets')} utterance targets"
              "   <- v2 ACCEPTANCE GATE: must be < 0.02")
    if "span_with_rel_pair" in d:
        print("\n  --- v2 render-invariance accounting ---")
        print(f"  targets with rel pair : {d['span_with_rel_pair']}"
              f"  (without={d['span_without_rel_pair']})")
        print(f"    valid via absolute  : {d['span_valid_via_absolute']}"
              "   (absolute pair still addresses its own value)")
        print(f"    valid via rel pair  : {d['span_valid_via_rel']}"
              "   (absolute pair STALE, rel pair repaired it)")
        print(f"    rel pair unresolved : {d['rel_pair_unresolvable']}"
              "   (utterance text drifted; label genuinely unusable)")
        print(f"  => absolute-pair stale rate: {_fmt(d.get('absolute_stale_rate'), 4)}"
              "   <- REPORTING ONLY: the share of rel targets whose old absolute"
              " pair had rotted (expected high on multi-tool rows)")
        if d.get("rel_drift_hist"):
            top = sorted(d["rel_drift_hist"].items(), key=lambda kv: -kv[1])[:8]
            print(f"  repaired drift hist   : {top}")
    if d.get("drift_hist"):
        top = sorted(d["drift_hist"].items(), key=lambda kv: -kv[1])[:8]
        print(f"  drift (true-stored)   : {top}")
    for ttype, st in (d.get("by_target_type") or {}).items():
        print(f"    {ttype:<20} valid={st['valid']:<6} stale_recoverable={st['stale_recoverable']:<6}"
              f" stale_lost={st['stale_lost']:<6} unresolvable={st.get('unresolvable', 0):<6}")
    if d.get("examples"):
        print("  examples:")
        for ex in d["examples"]:
            print(f"    - {ex}")
    # Fire on the UTTERANCE-SPAN rate, not the all-target rate: a multi-tool
    # split legitimately carries context targets whose offsets point outside the
    # utterance, so the all-target rate can exceed 0.5 on data that is fine.
    # Payloads produced before the scoped metric existed fall back to the
    # all-target rate so the warning is never silently lost.
    rate = d.get("utterance_span_stale_rate", d.get("span_stale_rate"))
    if isinstance(rate, float) and not math.isnan(rate) and rate > 0.5:
        print("\n  [!] More than half of the stored utterance offsets do NOT slice their own")
        print("      canonical value in the render the resolver actually builds. The pointer")
        print("      head is therefore supervised — and candidate_parity is measured —")
        print("      against the WRONG characters. Fix the offsets before reading any")
        print("      precision/recall number as a statement about model quality.")
    print("=" * 96)


def _report_predictions(d: Dict[str, Any]) -> None:
    c = d["counts"]
    print("=" * 96)
    print(f"PREDICTION ERROR DECOMPOSITION — {d['split']}")
    print(f"  preds: {d['preds_path']}")
    print("=" * 96)
    print(f"  rows joined           : {c.get('rows', 0)}"
          f"  (by index={c.get('joined_by_index', 0)}, unjoined={c.get('unjoined', 0)})")
    for k in sorted(k for k in c if k.startswith("status:")):
        print(f"    {k:<22} {c[k]}")
    print(f"  wrong-tool rows       : {c.get('wrong_tool_rows', 0)}")
    print(f"  empty-payload 'hits'  : {c.get('empty_payload_hits', 0)}"
          "   <- schema-valid with ZERO parameters")
    print(f"  gt params             : {c.get('gt_params', 0)}"
          f"  (extractable={c.get('gt_params_extractable', 0)},"
          f" unextractable={c.get('gt_params_unextractable', 0)})")
    print(f"  scored (correct+wrong): {d['n_scored']}"
          f"  correct={c.get('correct', 0)} wrong={c.get('wrong', 0)}"
          f" absent={c.get('extractable_absent', 0)}")
    print(f"  extractable precision : {_fmt(d['extractable_precision_scored'], 4)}")

    print("\n  --- precision by PREDICTED source_type (extractable GT params only) ---")
    print(f"  {'source_type':<16}{'n':>6}{'correct':>9}{'wrong':>7}{'absent':>8}{'prec':>8}")
    for src, st in sorted(d["by_source_type"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {src:<16}{st['n']:>6}{st['correct']:>9}{st['wrong']:>7}"
              f"{st['absent']:>8}{_fmt(st['precision'], 3):>8}")

    print("\n  --- precision by GT target_type ---")
    for tt, st in sorted(d["by_gt_target_type"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {tt:<20} n={st['n']:<6} prec={_fmt(st['precision'], 3)}")

    print("\n  --- where predicted spans land ---")
    for region, st in sorted(d["by_region"].items(), key=lambda kv: -(kv[1]["correct"] + kv[1]["wrong"])):
        print(f"  {region:<22} correct={st['correct']:<6} wrong={st['wrong']:<6}"
              f" share={_fmt(st['share'], 3)}")
    out_utt = sum(
        st["correct"] + st["wrong"]
        for k, st in d["by_region"].items()
        if k != REGION_IN_UTTERANCE
    )
    if d["n_scored"]:
        print(f"  -> spans OUTSIDE the utterance: {out_utt}/{d['n_scored']}"
              f" ({out_utt / d['n_scored']:.1%})")

    print("\n  --- confidence: correct vs wrong ---")
    cc, cw = d["confidence_correct"], d["confidence_wrong"]
    print(f"  correct: n={cc['n']:<5} mean={_fmt(cc['mean'], 4)} median={_fmt(cc['median'], 4)}")
    print(f"  wrong  : n={cw['n']:<5} mean={_fmt(cw['mean'], 4)} median={_fmt(cw['median'], 4)}")
    print(f"  {'band':<10}{'correct':>9}{'wrong':>8}{'prec':>8}")
    for band, st in d["by_confidence"].items():
        print(f"  {band:<10}{st['correct']:>9}{st['wrong']:>8}{_fmt(st['precision'], 3):>8}")

    print(f"\n  pointer-collapse signatures:")
    print(f"    rows with >=2 values, ALL identical : "
          f"{c.get('rows_all_values_identical', 0)} / {c.get('rows_multi_value', 0)}")
    print(f"    predictions equal to 1st utterance token : "
          f"{c.get('preds_equal_first_utterance_token', 0)}")

    for key, items in (d.get("examples") or {}).items():
        if not items:
            continue
        print(f"\n  examples [{key}]:")
        for ex in items:
            print(f"    - {ex}")
    print("=" * 96)


def run_diagnostics(
    split_name: str,
    data_dir: str,
    preds_path: Optional[str] = None,
    limit: Optional[int] = None,
    do_offsets: bool = True,
    do_predictions: bool = True,
    max_examples: int = 6,
) -> Dict[str, Any]:
    """Run the audits and print them; returns ``{"offsets": ..., "predictions": ...}``.

    Args:
        split_name: canonical split name.
        data_dir: directory holding ``{split_name}.jsonl``.
        preds_path: predictions JSONL. When ``None`` only the offset audit runs.
        limit: max samples/records per audit.
        do_offsets: run :func:`audit_gt_offsets`.
        do_predictions: run :func:`analyze_predictions` (needs ``preds_path``).
        max_examples: cap on example rows.

    Returns:
        Dict with the requested sub-reports (absent keys are omitted, not None,
        so ``report`` can never print an empty section as if it had run).
    """
    out: Dict[str, Any] = {}
    if do_offsets:
        off = audit_gt_offsets(
            split_name, data_dir, limit=limit, all_tools=True, max_examples=max_examples
        )
        _report_offsets(off)
        out["offsets"] = off
    if do_predictions and preds_path:
        pred = analyze_predictions(
            preds_path, split_name, data_dir, limit=limit, max_examples=max_examples
        )
        _report_predictions(pred)
        out["predictions"] = pred
    return out