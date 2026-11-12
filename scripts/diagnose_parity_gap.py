#!/usr/bin/env python3
"""P0 — Candidate-parity MEASUREMENT audit (diagnosis only; no behaviour change).

The v2 run reported `candidate_parity = 0.0806`. That number was computed by
comparing an extractable GT span's **stored absolute** ``char_start/char_end``
against the candidate set the resolver built at serve time. Those offsets are
absolute in whichever render the *aligner* produced; the resolver renders afresh
with its own tool pack, so the utterance moves and the comparison is against the
wrong characters. That is hypothesis **H0** (a measurement defect), and it cannot
be distinguished from a genuine serving gap (**H1** truncation, **H2** candidate
generation) by looking at the old number alone.

This script separates them by computing a 2x2 over every extractable GT span:

                      | stored-absolute hit | re-resolved hit
    stored-absolute   |   (the old metric)  |
    re-resolved       |                     |   (the honest metric)

Decision rule (plan step 2): if re-resolved parity is much higher than
stored-absolute parity, H0 dominates and 0.0806 was mostly mismeasurement. If it
barely moves, H1/H2 dominate and P2 must fix the packer or the generators.

Why this runs WITHOUT a checkpoint
----------------------------------
Candidate masks are a pure function of (render, tokenizer, schema, max_cands).
No weights are involved. So the audit needs no v2 checkpoint and no GPU: it builds
the exact render ``_resolve`` would build and calls the exact mask builder it calls.
The render is produced ONCE per sample and reused for both columns, so the audit
cannot reintroduce the skew it is measuring (that is why ``--all-tools`` renders
every ambient tool, matching ``context_schemas=sample.tools``).

Both columns are computed against the SAME candidate set, which is the only way the
2x2 means anything: if the two columns used different renders, the comparison would
confound addressing with packing.

Attribution for every re-resolved miss (plan step 3)
----------------------------------------------------
* ``truncated`` — the span WAS generated as a candidate but ``max_cands`` evicted
  it. This is H1.
* ``not_generated`` — no builder class produced a covering candidate at all. This
  is H2, and the row records which regions/classes did fire so P2 knows what to fix.
* ``no_utt_entry`` — the render carries no selectable utterance entry, so there is
  nothing to extract. A render-level refusal, not a candidate bug.

Usage::

    python scripts/diagnose_parity_gap.py --split hermes_eval --limit 0
    python scripts/diagnose_parity_gap.py --split hermes_eval --limit 300 --json out.json

``--limit 0`` means "every row". The default split/limit reproduce the eval split
the 0.0806 was measured on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spancall.data.align import tooldef_to_template_tool  # noqa: E402
from spancall.data.schema import UnifiedSample  # noqa: E402
from spancall.lean.eval_p1 import NON_EXTRACTABLE_TARGET_TYPES  # noqa: E402
from spancall.lean.resolver import _to_date, _to_recent_rows  # noqa: E402
from spancall.model.masks import CandidateMaskBuilder, _RowCtx  # noqa: E402
from span_template.template import (  # noqa: E402
    RecentRow,
    build_input_multi,
    resolve_utterance_span,
)

#: Defaults mirror the v2 run: 4096 seq len, 16 max span width, 512 candidate cap,
#: ``stratified`` retention. Kept as flags so the audit can also test H1 directly by
#: re-running with ``--max-cands 0`` (unlimited): if parity jumps, the cap was H1.
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_MAX_SPAN_WIDTH = 16
DEFAULT_MAX_CANDS = 512
DEFAULT_PACK_FILL = "stratified"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed_v2")
    p.add_argument("--split", default="hermes_eval")
    p.add_argument(
        "--limit", type=int, default=0,
        help="rows to audit; 0 = all (default), negative = first |N|",
    )
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--max-span-width", type=int, default=DEFAULT_MAX_SPAN_WIDTH)
    p.add_argument(
        "--max-cands", type=int, default=DEFAULT_MAX_CANDS,
        help="candidate cap; 0 = unlimited (use this to isolate H1 truncation)",
    )
    p.add_argument("--pack-fill", default=DEFAULT_PACK_FILL, choices=("head", "stratified"))
    p.add_argument("--tokenizer", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument(
        "--all-tools", action="store_true", default=True,
        help="render every ambient tool (resolver behaviour, default)",
    )
    p.add_argument(
        "--gt-tool-only", dest="all_tools", action="store_false",
        help="render ONLY the GT call's tool (the aligner's own render). Both columns "
             "then collapse to the aligner's view -- used to prove the addressing fix, "
             "not to measure the serve gap.",
    )
    p.add_argument("--examples", type=int, default=15, help="extra miss examples to keep")
    p.add_argument("--json", default=None, help="write the full report as JSON here")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _load_rows(data_dir: str, split: str, limit: int) -> List[UnifiedSample]:
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"split not found: {path}")
    out: List[UnifiedSample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            s = UnifiedSample.from_dict(json.loads(line))
            if not s.ground_truth_calls or not s.tools:
                continue
            out.append(s)
            if limit and len(out) >= abs(limit):
                break
    return out


def _render(sample: UnifiedSample, all_tools: bool) -> Tuple[Any, List[RecentRow]]:
    """The render ``_resolve`` performs: ``build_input_multi`` over the tool pack.

    ``recent_context`` is converted with the resolver's own ``_to_recent_rows``,
    NOT re-implemented here. That helper drops rows whose key is not a plain
    identifier; a local naive conversion would raise on those and the audit would
    report a render failure for rows the resolver serves fine -- inflating the
    "unmeasurable" bucket with the auditor's own bug.
    """
    gt_tool = next(
        (t for t in sample.tools if t.name == sample.ground_truth_calls[0].tool_name), None,
    )
    if gt_tool is None:
        raise ValueError("GT tool not among sample.tools")
    tools = (
        [tooldef_to_template_tool(t) for t in sample.tools]
        if all_tools
        else [tooldef_to_template_tool(gt_tool)]
    )
    recent = _to_recent_rows(sample.recent_context)
    out = build_input_multi(
        tools, sample.utterance, recent_context=recent, now=_to_date(sample.current_time_iso),
    )
    return out, recent


def _mask_key(tool: str, param: str) -> str:
    return f"{tool}::{param}"


def _build_masks(
    output: Any,
    sample: UnifiedSample,
    tokenizer: Any,
    *,
    max_seq_len: int,
    max_span_width: int,
    max_cands: Optional[int],
    pack_fill: str,
) -> Dict[str, Dict[str, Any]]:
    """Rebuild the candidate sets exactly like ``LeanResolver._candidate_masks``.

    Deliberately a direct call rather than instantiating a resolver: no model, no
    training, and no chance of accidentally scoring anything.
    """
    enc = tokenizer(output.text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = list(enc["offset_mapping"])

    builder = CandidateMaskBuilder(max_seq_len=max_seq_len, max_span_width=max_span_width)
    builder.pack_fill = pack_fill
    entries = output.entries
    max_tok = builder._effective_max_seq_len(offsets)
    ctx = _RowCtx.for_row(entries, offsets, max_tok)

    masks: Dict[str, Dict[str, Any]] = {}
    for schema in sample.tools:
        for p in schema.parameters:
            m = builder.build_param_masks(
                p, entries, offsets, output.text, ctx=ctx, max_cands=max_cands,
            )
            masks[_mask_key(schema.name, p.name)] = m
    return masks


def _covering_regions(cands: Sequence[Any], cs: int, ce: int) -> List[str]:
    """Which candidate regions produced a span that covers ``[cs, ce)``."""
    out = []
    for c in cands:
        if c.char_start <= cs and c.char_end >= ce:
            out.append(getattr(c, "region", "?"))
    return sorted(set(out))


def _targets_for_call(sample: UnifiedSample, call_index: int) -> Dict[str, List[Any]]:
    try:
        return sample.targets_for_call(call_index) or {}
    except Exception:
        return {}


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency guard
        raise SystemExit(f"transformers is required for a real tokenizer: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rows = _load_rows(args.data_dir, args.split, args.limit)
    if not rows:
        raise SystemExit("no usable rows")

    cap = int(args.max_cands) if args.max_cands else None

    # 2x2 cells over extractable GT spans.
    n_extractable = 0
    stored_hit = 0
    reresolved_hit = 0
    both_hit = 0
    stored_only = 0
    reresolved_only = 0
    neither = 0

    # Addressing provenance.
    with_rel_pair = 0
    rel_unresolvable = 0
    legacy_no_rel = 0

    # Attribution for re-resolved misses.
    attribution: Counter = Counter()
    miss_by_type: Dict[str, Counter] = defaultdict(Counter)
    miss_by_region: Counter = Counter()
    truncated_sizes: List[int] = []
    generated_counts: List[int] = []

    # H1 probe: was the span generated before the cap? Rebuild uncapped once per
    # sample whose GT span missed, and see whether it appears.
    truncated_vs_never = Counter()

    examples: List[Dict[str, Any]] = []
    rows_with_utt = 0
    rows_audited = 0

    for s in rows:
        rows_audited += 1
        try:
            output, _recent = _render(s, args.all_tools)
        except Exception:
            attribution["render_failed"] += 1
            continue
        utt_entries = [e for e in output.entries if e.segment == "utterance" and e.selectable]
        has_utt = bool(utt_entries)
        rows_with_utt += int(has_utt)

        masks = _build_masks(
            output, s, tokenizer,
            max_seq_len=args.max_seq_len, max_span_width=args.max_span_width,
            max_cands=cap, pack_fill=args.pack_fill,
        )

        for ci, gt_call in enumerate(s.ground_truth_calls):
            per_param = _targets_for_call(s, ci)
            for pname, targets in per_param.items():
                key = _mask_key(gt_call.tool_name, pname)
                m = masks.get(key)
                # Mask-gated denominator: a GT param absent from the schema has no
                # candidate set, so it cannot contribute a parity verdict.
                if m is None:
                    attribution["no_mask_for_param"] += 1
                    continue
                cands = m["candidates"]
                have = {(int(c.char_start), int(c.char_end)) for c in cands}

                for t in targets:
                    if t.target_type in NON_EXTRACTABLE_TARGET_TYPES:
                        continue
                    abs_cs = getattr(t, "char_start", None)
                    abs_ce = getattr(t, "char_end", None)
                    rel_cs = getattr(t, "utt_rel_start", None)
                    rel_ce = getattr(t, "utt_rel_end", None)
                    if abs_cs is None or abs_ce is None:
                        continue
                    n_extractable += 1

                    stored_ok = (int(abs_cs), int(abs_ce)) in have

                    # Re-resolve the render-invariant pair against THIS render.
                    resolved: Optional[Tuple[int, int]] = None
                    if rel_cs is not None and rel_ce is not None:
                        with_rel_pair += 1
                        expect = getattr(t, "canonical_value", None)
                        resolved = resolve_utterance_span(
                            output, int(rel_cs), int(rel_ce),
                            expect=None if expect is None else str(expect),
                        )
                        if resolved is None:
                            rel_unresolvable += 1
                            attribution["rel_unresolvable"] += 1
                            miss_by_type[str(t.target_type)]["rel_unresolvable"] += 1
                            # An unresolvable label is a miss for the honest metric,
                            # but NOT evidence about the packer: record and move on.
                            stored_hit += int(stored_ok)
                            neither += int(not stored_ok)
                            continue
                    else:
                        legacy_no_rel += 1
                        # No render-invariant address exists; the honest column can
                        # only reuse the stored pair. Flagged in the report.
                        resolved = (int(abs_cs), int(abs_ce))

                    rer_ok = (int(resolved[0]), int(resolved[1])) in have

                    stored_hit += int(stored_ok)
                    reresolved_hit += int(rer_ok)
                    both_hit += int(stored_ok and rer_ok)
                    stored_only += int(stored_ok and not rer_ok)
                    reresolved_only += int(rer_ok and not stored_ok)
                    neither += int(not stored_ok and not rer_ok)

                    if rer_ok:
                        continue

                    # ---- miss attribution (re-resolved column only) ----
                    if not has_utt:
                        attribution["no_utt_entry"] += 1
                        continue
                    generated_counts.append(len(cands))
                    covering = _covering_regions(cands, int(resolved[0]), int(resolved[1]))
                    if covering:
                        # Generated, then evicted by the cap => H1.
                        attribution["truncated"] += 1
                        for r in covering:
                            miss_by_region[r] += 1
                        truncated_sizes.append(len(cands))
                        truncated_vs_never["truncated"] += 1
                    else:
                        # Never generated at all => H2, or the cap evicted the ONLY
                        # covering span. Distinguish by rebuilding uncapped.
                        capped_out = False
                        if cap is not None:
                            full = _build_masks(
                                output, s, tokenizer,
                                max_seq_len=args.max_seq_len,
                                max_span_width=args.max_span_width,
                                max_cands=None, pack_fill=args.pack_fill,
                            ).get(key)
                            if full is not None and _covering_regions(
                                full["candidates"], int(resolved[0]), int(resolved[1])
                            ):
                                capped_out = True
                        if capped_out:
                            attribution["truncated_uncapped_would_hit"] += 1
                            truncated_vs_never["truncated"] += 1
                        else:
                            attribution["not_generated"] += 1
                            truncated_vs_never["never_generated"] += 1
                    miss_by_type[str(t.target_type)]["miss"] += 1
                    if len(examples) < args.examples:
                        examples.append({
                            "sample_id": s.sample_id,
                            "tool": gt_call.tool_name,
                            "param": pname,
                            "param_type": getattr(_schema_param(s, gt_call.tool_name, pname), "param_type", None),
                            "target_type": str(t.target_type),
                            "canonical_value": getattr(t, "canonical_value", None),
                            "stored_abs": [int(abs_cs), int(abs_ce)],
                            "resolved_abs": [int(resolved[0]), int(resolved[1])],
                            "n_candidates": len(cands),
                            "covering_regions": covering,
                            "cand_pack_fill": args.pack_fill,
                            "cand_cap": cap,
                            "span_in_utterance": has_utt,
                            "value_in_text": (
                                None if getattr(t, "canonical_value", None) is None
                                else str(getattr(t, "canonical_value")) in output.text
                            ),
                        })

    def _rate(num: int, den: int) -> Optional[float]:
        return (num / den) if den else None

    return {
        "split": args.split,
        "rows_audited": rows_audited,
        "rows_with_utterance_entry": rows_with_utt,
        "max_cands": cap,
        "pack_fill": args.pack_fill,
        "all_tools": bool(args.all_tools),
        "n_extractable_spans": n_extractable,
        "stored_absolute_parity": _rate(stored_hit, n_extractable),
        "reresolved_parity": _rate(reresolved_hit, n_extractable),
        "delta": (
            _rate(reresolved_hit, n_extractable) - _rate(stored_hit, n_extractable)
            if n_extractable else None
        ),
        "cells": {
            "both_hit": both_hit,
            "stored_only": stored_only,
            "reresolved_only": reresolved_only,
            "neither": neither,
        },
        "addressing": {
            "with_rel_pair": with_rel_pair,
            "legacy_no_rel_pair": legacy_no_rel,
            "rel_unresolvable": rel_unresolvable,
        },
        "attribution": dict(attribution),
        "miss_by_target_type": {k: dict(v) for k, v in miss_by_type.items()},
        "miss_by_covering_region": dict(miss_by_region),
        "capped_candidate_sizes": {
            "n": len(truncated_sizes),
            "median": sorted(truncated_sizes)[len(truncated_sizes) // 2] if truncated_sizes else None,
            "max": max(truncated_sizes) if truncated_sizes else None,
        },
        "truncated_vs_never_generated": dict(truncated_vs_never),
        "examples": examples,
    }


def _schema_param(sample: UnifiedSample, tool_name: str, param: str) -> Any:
    tool = next((t for t in sample.tools if t.name == tool_name), None)
    if tool is None:
        return None
    return next((p for p in tool.parameters if p.name == param), None)


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def report(rep: Dict[str, Any]) -> None:
    n = rep["n_extractable_spans"]
    print("=" * 92)
    print(f"P0 PARITY AUDIT — split={rep['split']}  rows={rep['rows_audited']} "
          f"(with utterance entry: {rep['rows_with_utterance_entry']})")
    print(f"   render: all_tools={rep['all_tools']}   max_cands={rep['max_cands']}   "
          f"pack_fill={rep['pack_fill']}")
    print("=" * 92)
    print(f"extractable GT spans measured: {n}")
    print()
    print(" 2x2 (same candidate set in both columns)")
    print(f"   stored-absolute parity : {_fmt(rep['stored_absolute_parity'])}   "
          f"<- the number 0.0806 came from")
    print(f"   re-resolved parity     : {_fmt(rep['reresolved_parity'])}   "
          f"<- addresses the render actually served")
    print(f"   delta                  : {_fmt(rep['delta'])}")
    c = rep["cells"]
    print(f"     both hit={c['both_hit']}  stored-only={c['stored_only']}  "
          f"re-resolved-only={c['reresolved_only']}  neither={c['neither']}")
    print()
    a = rep["addressing"]
    print(" addressing provenance")
    print(f"   with utt_rel pair      : {a['with_rel_pair']}")
    print(f"   legacy (absolute only) : {a['legacy_no_rel_pair']}  "
          f"<- honest column is a clone of the stored column for these")
    print(f"   rel pair unresolvable  : {a['rel_unresolvable']}  "
          f"<- data/label drift, counted as a miss")
    print()
    print(" attribution of re-resolved misses")
    for k, v in sorted(rep["attribution"].items(), key=lambda kv: -kv[1]):
        tag = {
            "truncated": "H1 — generated, evicted by max_cands",
            "truncated_uncapped_would_hit": "H1 — only an UNCAPPED set contains it",
            "not_generated": "H2 — no builder class produced it",
            "no_utt_entry": "render carried no utterance entry",
            "rel_unresolvable": "label pointed outside the served render",
            "no_mask_for_param": "GT param not in the schema (not counted)",
            "render_failed": "render raised",
        }.get(k, "")
        print(f"   {k:<30} {v:>6}   {tag}")
    print()
    tv = rep["truncated_vs_never_generated"]
    if tv:
        print(f" miss split (H1 vs H2): truncated={tv.get('truncated', 0)}  "
              f"never_generated={tv.get('never_generated', 0)}")
    mb = rep["miss_by_target_type"]
    if mb:
        print(" misses by GT target type:")
        for k, v in sorted(mb.items(), key=lambda kv: -sum(kv[1].values())):
            print(f"   {k:<24} {v}")
    mr = rep["miss_by_covering_region"]
    if mr:
        print(" truncated spans came from these candidate regions:")
        print(f"   {mr}")
    print()
    print(" VERDICT (plan step 2 decision rule)")
    stored = rep["stored_absolute_parity"]
    rer = rep["reresolved_parity"]
    if rer is None:
        print("   no extractable spans — nothing to conclude")
    elif rer >= 0.5 and (stored is None or rer - stored >= 0.2):
        print(f"   H0 DOMINATES: re-resolved {_fmt(rer)} vs stored {_fmt(stored)}. "
              "The 0.0806 was mostly mismeasurement;")
        print("   P2 should NOT be a packer/generator change on the strength of that number.")
    elif rer > (stored or 0.0) + 0.05:
        print(f"   H0 CONTRIBUTES but does not explain it: {_fmt(rer)} vs {_fmt(stored)}.")
        print("   Use the attribution table above to pick H1 (packer) vs H2 (generation).")
    else:
        print(f"   H0 NEGLIGIBLE: {_fmt(rer)} vs {_fmt(stored)}. The gap is real; use the")
        print("   attribution table to fix the packer (H1) or the generator (H2).")
    if rep["examples"]:
        print()
        print(f" sample misses ({len(rep['examples'])})")
        for e in rep["examples"][:10]:
            print(f"   {e['sample_id'][:34]:<34} {e['tool']}.{e['param']:<18} "
                  f"type={e['param_type']} cands={e['n_candidates']} "
                  f"covering={e['covering_regions']} in_text={e['value_in_text']}")
    print("=" * 92)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rep = audit(args)
    if not args.quiet:
        report(rep)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, default=str)
        print(f"[written] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())