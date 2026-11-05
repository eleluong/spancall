"""Tests for the E0 traffic probe (scripts/probe.py, remediation plan D7).

Verified contract:
- F-share: fraction of samples whose gold params are all emitted as targetable
  spans (utterance tokens or [CTX_*] pool entries) by the *current* template.
- Abstain-reason mix buckets non-F calls into ABSENT / AMBIGUOUS /
  NEEDS_REASONING / OFF_WINDOW.
- Scope-cut deltas: re-render the same samples with one D1/D2/recent/anaphora
  restriction toggled off and report the F-share delta (flag-gated baseline arm).
- Retriever R@T over pre-computed candidate sets (D6 input).
- Decision gate: F_floor = h_min / coverage_of_f (default 0.5); below => stop.
"""

import importlib.util
import os
import sys
import unittest
from datetime import date

from span_template import Param, RecentRow, Tool

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_probe():
    path = os.path.join(_SCRIPTS_DIR, "scripts", "probe.py")
    spec = importlib.util.spec_from_file_location("spancall_probe", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve the module's __dict__
    # (importlib does not do this automatically).
    sys.modules["spancall_probe"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_probe()


def _tool(name="book_flight", params=()):
    return Tool(name, f"{name} description", params)


def _sample(utterance, gold, tool=None, candidates=(), expected=None, recent=()):
    return PROBE.ProbeSample(
        utterance=utterance,
        tool=tool or _tool(),
        gold_params=gold,
        recent_context=recent,
        expected_tool=expected,
        candidate_tools=candidates,
    )


class FShareTest(unittest.TestCase):
    def test_verbatim_gold_is_groundable(self):
        # "VN123" appears verbatim in the utterance => the encoder can pointer it.
        s = _sample("book flight VN123", {"flight_id": "VN123"})
        rep = PROBE.run_probe([s], workload="w", now=date(2026, 10, 27))
        self.assertEqual(rep.f_share, 1.0)
        self.assertEqual(rep.abstain_mix, {})

    def test_missing_gold_abstains(self):
        # booking_ref never appears => ABSENT (nothing to pointer-target).
        s = _sample("check in please", {"booking_ref": "ABC123"})
        rep = PROBE.run_probe([s], workload="w", now=date(2026, 10, 27))
        self.assertEqual(rep.f_share, 0.0)
        self.assertEqual(rep.abstain_mix.get("ABSENT"), 1)

    def test_date_idiom_groundable_via_window(self):
        # "tomorrow" resolves via the D1 idiom -> 2026-10-28 window entry.
        s = _sample("book tomorrow", {"date": "2026-10-28"})
        rep = PROBE.run_probe([s], workload="w", now=date(2026, 10, 27))
        self.assertEqual(rep.f_share, 1.0)

    def test_multi_word_enum_groundable_after_d2(self):
        # "Premium Economy" is a quoted multi-word enum value in <|ctx_enum|>.
        tool = Tool("seat_upgrade", "desc", [Param("cabin", enum=("ECONOMY", "Premium Economy"))])
        s = _sample("upgrade to Premium Economy", {"cabin": "Premium Economy"}, tool=tool)
        rep = PROBE.run_probe([s], workload="w", now=date(2026, 10, 27))
        self.assertEqual(rep.f_share, 1.0)

    def test_anaphora_not_verbatim_abstains(self):
        # "the flight I was looking at" needs reasoning (not verbatim) => non-F.
        s = _sample("book the flight I was looking at", {"flight_id": "VN123"})
        rep = PROBE.run_probe([s], workload="w", now=date(2026, 10, 27))
        self.assertEqual(rep.f_share, 0.0)
        self.assertTrue(rep.abstain_mix)  # bucketed non-empty


class ScopeCutTest(unittest.TestCase):
    def test_multi_word_enum_delta_detects_d2_fix(self):
        # The value is NOT in the utterance — only in the schema's enum pool.
        # With D2 the quoted value is a targetable <|ctx_enum|> entry (F); the arm
        # toggling multi-word enums off drops the pool, making it ABSENT.
        tool = Tool("seat_upgrade", "desc", [Param("cabin", enum=("ECONOMY", "Premium Economy"))])
        s = _sample("upgrade my seat please", {"cabin": "Premium Economy"}, tool=tool)
        rep = PROBE.run_probe(
            [s], workload="w", now=date(2026, 10, 27), arms=("multi_word_enums",)
        )
        self.assertEqual(rep.f_share, 1.0)
        delta = rep.scope_cuts["multi_word_enums"]["delta_vs_full"]
        self.assertEqual(delta, -1.0)
        self.assertEqual(rep.scope_cuts["multi_word_enums"]["F_share"], 0.0)

    def test_recent_context_cut(self):
        # Value lives in a <|ctx_recent|> row: full flavour is groundable; toggling
        # the recent rows off makes the value ABSENT -> F-share delta -1.
        s = _sample(
            "use the flight I picked",
            {"flight_id": "VN123"},
            recent=[RecentRow("picked", "VN123", source="agent")],
        )
        rep = PROBE.run_probe([s], workload="w", now=date(2026, 10, 27), arms=("recent_context",))
        self.assertEqual(rep.f_share, 1.0)
        delta = rep.scope_cuts["recent_context"]["delta_vs_full"]
        self.assertEqual(delta, -1.0)
        self.assertEqual(rep.scope_cuts["recent_context"]["F_share"], 0.0)


class RetrieverRAtTTest(unittest.TestCase):
    def test_top_n_hits_expected_tool(self):
        cands = [
            _tool("cancel_flight"), _tool("check_in"), _tool("book_flight"),
        ]
        s = _sample("book something", {}, expected="book_flight", candidates=cands)
        rep = PROBE.run_probe([s], workload="w", now=date(2026, 10, 27), retriever_top_n=2)
        # book_flight is 3rd; top-2 misses, top-3 hits.
        self.assertEqual(rep.rat["recall_at_top_n"], 0.0)
        rep3 = PROBE.run_probe([s], workload="w", now=date(2026, 10, 27), retriever_top_n=3)
        self.assertEqual(rep3.rat["recall_at_top_n"], 1.0)

    def test_no_candidate_sets_reports_nan(self):
        s = _sample("book something", {})
        rep = PROBE.run_probe([s], workload="w", now=date(2026, 10, 27))
        self.assertIsNone(rep.rat["recall_at_top_n"])


class DecisionGateTest(unittest.TestCase):
    def test_floor_derived_from_cost_model(self):
        gate = PROBE._gate_decision(f_share=0.5, h_min=0.3, coverage=0.6)
        self.assertEqual(gate["f_floor"], 0.5)
        self.assertTrue(gate["passed"])
        gate_low = PROBE._gate_decision(f_share=0.49, h_min=0.3, coverage=0.6)
        self.assertFalse(gate_low["passed"])

    def test_achievable_h_is_f_share_times_coverage(self):
        gate = PROBE._gate_decision(f_share=0.5, h_min=0.3, coverage=0.6)
        self.assertIn("0.300", gate["notes"])


if __name__ == "__main__":
    unittest.main()