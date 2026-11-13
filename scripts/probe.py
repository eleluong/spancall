#!/usr/bin/env python3
"""E0 traffic probe — measure F-share and scope-cut deltas before training (D7).

Runs the *existing* template over hand-labeled call samples, with no model and no
model or gate (this probe's only dependency is ``span_template`` + the
sanitizer).

Outputs (per workload, never aggregated across classes):
  * F-share — fraction of calls fully groundable by the current template.
  * Abstain-reason mix — ABSENT / AMBIGUOUS / NEEDS_REASONING / OFF_WINDOW.
  * Per-scope-cut deltas — F-share lost by removing idioms (D1), multi-word
    enums (D2), recent-context rows, anaphora-beyond-verbatim (flag-gated arms).
  * Retriever R@T — recall at top-N over candidate tool sets (D6 input).

Decision gate (kill floor): F_floor = h_min / 0.6 (default 0.5). F-share is a
coverage *upper bound* on h; below floor => stop and write up the negative result.

Usage:
    python scripts/probe.py --workload data/samples_booking.jsonl \
        --workload-name booking --output results/probe_booking.json \
        --clock.asia_saigon 2026-10-27 --retriever-top-n 5

The flag-gated baseline arms are enabled with --arm-* (see --help). Each arm
re-renders the *same* sample with that restriction toggled off and reports how
many callable samples it converts to/from F.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Standalone script: put the repo root on sys.path so `span_template` resolves
# without requiring an editable install (mirrors how the test suite locates it).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# E0 depends ONLY on span_template + the sanitizer (D7).
from span_template import (
    SanitizeError,
    SpanTemplateOutput,
    Tool,
    Param,
    RecentRow,
    build_input,
)

# Abstain-reason classes (taxonomy shared with spancall.data.align).
ABSENT, AMBIGUOUS, NEEDS_REASONING, OFF_WINDOW = (
    "ABSENT", "AMBIGUOUS", "NEEDS_REASONING", "OFF_WINDOW",
)
ABSTAIN_REASONS: Tuple[str, ...] = (ABSENT, AMBIGUOUS, NEEDS_REASONING, OFF_WINDOW)

# Scope-cut labels (D7: "which restriction consumed the F-share").
SCOPE_CUTS: Tuple[str, ...] = ("idioms", "multi_word_enums", "recent_context", "anaphora")


# --------------------------------------------------------------------------
# Sample I/O
# --------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


@dataclass
class ProbeSample:
    """One hand-labeled call.

    ``gold_params`` maps required param name -> the canonical value that must be
    extractable from the utterance for the call to count as stratum F.
    """

    utterance: str
    tool: Tool
    gold_params: Dict[str, str] = field(default_factory=dict)
    recent_context: Sequence[RecentRow] = ()
    expected_tool: Optional[str] = None   # for retriever R@T
    candidate_tools: Sequence[Tool] = ()  # for retriever R@T (top-N over this)


def read_samples(path: str) -> List[ProbeSample]:
    """Read samples from a JSONL file.

    Each line::
        {"utterance": ..., "tool": {...Tool schema...},
         "gold_params": {param: value}, "recent_context": [{"key","value"}],
         "expected_tool": ..., "candidate_tools": [...Tool schemas...]}

    ``tool``/``candidate_tools`` use the same schema as ``span_template.Tool``
    (``name``, ``description``, ``params: [{name, description, optional, enum}]``).
    """
    samples: List[ProbeSample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            def tool_from_dict(d: Dict[str, Any]) -> Tool:
                params = [
                    Param(
                        name=p.get("name", ""),
                        description=p.get("description", ""),
                        enum=tuple(p.get("enum") or ()),
                        optional=bool(p.get("optional", False)),
                    )
                    for p in d.get("params", [])
                ]
                return Tool(d.get("name", ""), d.get("description", ""), params)

            recent = [
                RecentRow(r.get("key", ""), r.get("value", ""))
                for r in rec.get("recent_context", [])
            ]
            samples.append(
                ProbeSample(
                    utterance=rec.get("utterance", ""),
                    tool=tool_from_dict(rec.get("tool", {})),
                    gold_params={str(k): str(v) for k, v in rec.get("gold_params", {}).items()},
                    recent_context=recent,
                    expected_tool=rec.get("expected_tool"),
                    candidate_tools=[tool_from_dict(t) for t in rec.get("candidate_tools", [])],
                )
            )
    return samples


# --------------------------------------------------------------------------
# Scope-cut flags (D7 flag-gated baseline arms)
# --------------------------------------------------------------------------

@dataclass
class ScopeFlavor:
    """One render flavour, toggling the D1/D2/recent/anaphora cuts.

    The flags mirror the template knobs:
      idioms=False        -> no ``<key> <iso>`` idiom entries  (D1 cut)
      multi_word_enums True-> keep multi-word enum values quoted (D2 fix ON)
      recent_context=()   -> no <|ctx_recent|> rows            (recent cut)
      anaphora_resolved False-> treat non-verbatim references as NOT groundable
                                (anaphora-beyond-verbatim cut)
    """

    name: str
    idioms: bool = True
    multi_word_enums: bool = True
    recent: bool = True
    anaphora: bool = False

    # Toggled-off variant names for the flag-gated arms.
    def without(self, cut: str) -> "ScopeFlavor":
        if cut == "idioms":
            return ScopeFlavor(f"{self.name}-no_idioms", idioms=False, multi_word_enums=self.multi_word_enums, recent=self.recent, anaphora=self.anaphora)
        if cut == "multi_word_enums":
            return ScopeFlavor(f"{self.name}-no_mwe", idioms=self.idioms, multi_word_enums=False, recent=self.recent, anaphora=self.anaphora)
        if cut == "recent_context":
            return ScopeFlavor(f"{self.name}-no_recent", idioms=self.idioms, multi_word_enums=self.multi_word_enums, recent=False, anaphora=self.anaphora)
        if cut == "anaphora":
            return ScopeFlavor(f"{self.name}-no_anaphora", idioms=self.idioms, multi_word_enums=self.multi_word_enums, recent=self.recent, anaphora=False)
        raise ValueError(f"unknown scope cut {cut!r}")


FULL = ScopeFlavor("full")


# --------------------------------------------------------------------------
# Groundability + abstain reasons
# --------------------------------------------------------------------------

# A gold value that looks like an ISO date (YYYY-MM-DD) — probes the closed
# <|ctx_time|> window deterministically.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_fully_groundable(
    out: SpanTemplateOutput, gold: Dict[str, str], anaphora: bool
) -> Tuple[bool, str]:
    """Return (groundable, abstain_reason) for one call under one flavour.

    Groundable == every required gold value is emitted by the template as a
    targetable span: either a selectable ``<|ctx_*|>`` pool entry whose canonical
    value matches, or a verbatim token **within** a selectable span (the whole
    utterance is ONE selectable span, so the check is substring containment, not
    whole-entry equality). The single-tool template does NOT key param values to
    <|param|> anchors (the anchors are bridges), so a gold value missing from the
    target set is the classic abstain: the encoder has nothing to pointer-target.

    Buckets (H4 classes) — assigned by the first matching rule, most actionable
    first; the plan's scope-cut arms (D7) then attribute F-share loss to a
    specific restriction, which is what E0 uses for the decision gate:
      AMBIGUOUS       — the same canonical gold value maps to >1 targetable span,
                        a swap-unsafe tie (swap safety).
      OFF_WINDOW      — the gold value is a date not present in the closed
                        <|ctx_time|> window (range-like idiom / off-window date);
                        deterministic, no reasoning involved.
      NEEDS_REASONING — the gold value is present in the input material
                        (utterance / recent rows) but only recoverable by
                        inference (anaphora / paraphrase), not verbatim.
      ABSENT          — the gold value is not in the input material at all.
    """
    span_texts = [str(e.value) for e in out.targetable()]
    missing = [k for k, v in gold.items() if not any(v in t for t in span_texts)]
    if not missing:
        return True, ""
    # AMBIGUOUS: a required value appears in >1 distinct targetable span (tie).
    if any(sum(1 for t in span_texts if v in t) > 1 for _, v in gold.items()):
        return False, AMBIGUOUS
    # OFF_WINDOW: a missing gold value that looks like an ISO date. The <|ctx_time|>
    # window is closed and bounded (D1), so a date outside it is deterministic:
    # the template cannot surface it, no reasoning involved.
    if any(
        _ISO_DATE_RE.match(str(gold[k])) and not any(str(gold[k]) == t for t in span_texts)
        for k in missing
    ):
        return False, OFF_WINDOW
    # NEEDS_REASONING: present in the input but only via inference (anaphora/
    # paraphrase), e.g. "the flight I picked earlier" for a flight_id value.
    if any(str(gold[k]) in out.utterance for k in missing):
        return False, NEEDS_REASONING
    return False, ABSENT


# --------------------------------------------------------------------------
# Retriever R@T (D6 input)
# --------------------------------------------------------------------------

def retriever_recall_at_top_n(
    samples: Sequence[ProbeSample], top_n: int
) -> Tuple[float, int, int]:
    """Minimal R@T: expected tool in the top-N candidate set.

    In production this is the embedding-retriever recall (D6 option 2); the probe
    accepts pre-computed candidate sets per sample so the number is measured, not
    assumed. Returns (recall, hits, total).
    """
    hits = 0
    total = 0
    for s in samples:
        if s.expected_tool is None or not s.candidate_tools:
            continue
        total += 1
        cand_names = {t.name for t in s.candidate_tools[:top_n]}
        if s.expected_tool in cand_names:
            hits += 1
    if total == 0:
        return float("nan"), 0, 0
    return hits / total, hits, total


# --------------------------------------------------------------------------
# Probe driver
# --------------------------------------------------------------------------

@dataclass
class ProbeReport:
    workload: str
    n: int
    f_share: float                       # F / n
    abstain_mix: Dict[str, int]
    scope_cuts: Dict[str, Dict[str, float]]  # cut -> {F_share, delta}
    rat: Dict[str, Any]                  # R@T + counts
    gate: Dict[str, Any]                 # decision gate result
    details: List[Dict[str, Any]] = field(default_factory=list)  # per-sample rows


def _gate_decision(f_share: float, h_min: float, coverage: float) -> Dict[str, Any]:
    """F_floor = h_min / coverage; default h_min = 0.3, coverage = 0.6 -> 0.5."""
    floor = h_min / coverage
    return {
        "f_floor": round(floor, 4),
        "h_min": h_min,
        "coverage_of_f": coverage,
        "passed": bool(f_share >= floor),
        "notes": (
            "F-share is a coverage upper bound on h, not h itself; "
            f"achievable h ≈ F_share × coverage_of_F = {f_share * coverage:.3f}"
        ),
    }


def run_probe(
    samples: Sequence[ProbeSample],
    *,
    workload: str,
    now: date,
    clock_zone: str = "UTC",
    arms: Sequence[str] = (),
    retriever_top_n: int = 5,
    h_min: float = 0.3,
    coverage_of_f: float = 0.6,
    verbose: bool = False,
) -> ProbeReport:
    """Run E0 over ``samples``.

    Arms are scope cuts (D7 flag-gated baseline arms); each arm toggles exactly
    one restriction off and reports the F-share delta against the full flavour.
    """
    n = len(samples)
    f_count = 0
    abstain_mix: Counter[str] = Counter()
    arm_f: Dict[str, int] = defaultdict(int)

    for i, s in enumerate(samples):
        # Primary render (all D1/D2 fixes on).
        try:
            out = build_input(
                tool=s.tool,
                utterance=s.utterance,
                recent_context=s.recent_context,
                now=now,
                idioms=("tomorrow", "day_after_tomorrow", "tonight") if FULL.idioms else (),
                include_enum=True,
            )
        except SanitizeError as exc:
            abstain_mix[exc.reason] += 1
            if verbose:
                print(f"[{i}] sanitize abstain: {exc.reason}")
            continue
        except Exception as exc:  # e.g. enum pool validation
            abstain_mix[NEEDS_REASONING] += 1
            if verbose:
                print(f"[{i}] render error: {exc}")
            continue

        ok, reason = _is_fully_groundable(out, s.gold_params, anaphora=FULL.anaphora)
        if ok:
            f_count += 1
            arm_f[FULL.name] += 1
        else:
            abstain_mix[reason or NEEDS_REASONING] += 1

        # Flag-gated baseline arms (D7): same sample, one restriction off.
        for cut in arms:
            flavor = FULL.without(cut)
            try:
                arm_out = build_input(
                    tool=s.tool,
                    utterance=s.utterance,
                    recent_context=s.recent_context if flavor.recent else (),
                    now=now,
                    idioms=("tomorrow", "day_after_tomorrow", "tonight") if flavor.idioms else (),
                    include_enum=flavor.multi_word_enums,
                )
            except Exception:
                continue
            arm_ok, _ = _is_fully_groundable(arm_out, s.gold_params, anaphora=flavor.anaphora)
            if arm_ok:
                arm_f[flavor.name] += 1

    f_share = f_count / n if n else 0.0
    scope_cuts: Dict[str, Dict[str, float]] = {}
    for cut in arms:
        flavor = FULL.without(cut)
        arm_f_share = arm_f[flavor.name] / n if n else 0.0
        scope_cuts[cut] = {
            "F_share": round(arm_f_share, 4),
            "delta_vs_full": round(arm_f_share - f_share, 4),
            "n_gained_LOST": arm_f[flavor.name] - f_count,
        }

    recall, hit_n, tot_n = retriever_recall_at_top_n(samples, retriever_top_n)
    rat = {"recall_at_top_n": round(recall, 4) if recall == recall else None, "hits": hit_n, "total": tot_n, "top_n": retriever_top_n}

    gate = _gate_decision(f_share, h_min, coverage_of_f)
    return ProbeReport(
        workload=workload,
        n=n,
        f_share=round(f_share, 4),
        abstain_mix=dict(abstain_mix),
        scope_cuts=scope_cuts,
        rat=rat,
        gate=gate,
    )


def _json_default(o: Any) -> Any:
    if isinstance(o, date):
        return o.isoformat()
    return str(o)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E0 traffic probe (D7): F-share, abstain reasons, scope-cut deltas, R@T",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--workload", required=True, help="JSONL of hand-labeled call samples")
    p.add_argument("--workload-name", default="workload", help="report/workload label")
    p.add_argument("--output", default=None, help="JSON output path (default: <cwd>/results/probe_<name>.json)")
    p.add_argument("--now", default="2026-10-27", help="render date for <|ctx_time|> (caller-resolved user-local date; D3)")
    p.add_argument("--clock-zone", default="UTC", help="IANA zone the --now date was resolved for (D3); informational in the report")
    p.add_argument("--retriever-top-n", type=int, default=5, help="R@T top-N for the schema-selection mode in play (D6)")
    p.add_argument("--h-min", type=float, default=0.3, help="smallest hit rate still giving >= 5x (E6 target)")
    p.add_argument("--coverage-of-f", type=float, default=0.6, help="E2 coverage target over stratum F")
    p.add_argument("--arm", action="append", default=[], dest="arms",
                   choices=list(SCOPE_CUTS), help="extra flag-gated baseline arm(s); repeatable")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.workload):
        print(f"[E0 probe] workload not found: {args.workload}", file=sys.stderr)
        return 2
    now = _parse_date(args.now)
    samples = read_samples(args.workload)
    report = run_probe(
        samples,
        workload=args.workload_name,
        now=now,
        clock_zone=args.clock_zone,
        arms=tuple(args.arms),
        retriever_top_n=args.retriever_top_n,
        h_min=args.h_min,
        coverage_of_f=args.coverage_of_f,
        verbose=args.verbose,
    )
    payload = {
        "workload": report.workload,
        "n": report.n,
        "F_share": report.f_share,
        "abstain_reason_mix": report.abstain_mix,
        "scope_cut_deltas": report.scope_cuts,
        "retriever_RatT": report.rat,
        "decision_gate": report.gate,
        "clock": {"zone": args.clock_zone, "now": now.isoformat()},
    }
    out_path = args.output or os.path.join("results", f"probe_{args.workload_name}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    print(f"[E0 probe] {report.n} samples, F_share={report.f_share:.3f} "
          f"gate_passed={report.gate['passed']}")
    print(f"[E0 probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())