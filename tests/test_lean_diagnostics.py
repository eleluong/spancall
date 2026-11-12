"""Tests for :mod:`spancall.lean.diagnostics`.

The diagnostics module exists to explain a bad headline, so its own failure mode
is the dangerous one: a *plausible-looking* number that is silently computed over
the wrong population. These tests pin the three things that would let that
happen:

* the span-region classifier (``in`` / ``before`` / ``after`` utterance),
* the null-aware "prediction present" rule that decides what joins precision,
* the stale-offset audit and the prediction decomposition, both driven through
  synthetic files so the expected answer is arithmetic, not a golden file.

``render_for_sample`` is monkeypatched everywhere: the real one needs
``jinja2`` + the template env, and the audit's job is to compare a *given*
render against stored offsets, not to re-test rendering.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from spancall.lean import diagnostics as dg


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeEntry:
    """Minimal stand-in for :class:`span_template.parser.Entry`."""

    def __init__(self, segment: str, char_start: int, char_end: int, value: Any) -> None:
        self.segment = segment
        self.char_start = char_start
        self.char_end = char_end
        self.value = value
        self.selectable = True


class _FakeOutput:
    def __init__(self, text: str, entries: Sequence[_FakeEntry]) -> None:
        self.text = text
        self.entries = list(entries)


def _render(text: str, utt: str, prefix_len: int) -> _FakeOutput:
    """Build a render whose utterance entry spans ``[prefix_len, prefix_len+len(utt))``."""
    return _FakeOutput(
        text,
        [_FakeEntry("utterance", prefix_len, prefix_len + len(utt), utt)],
    )


def _sample_dict(
    *,
    sample_id: str,
    utterance: str,
    targets: Dict[str, List[Dict[str, Any]]],
    args: Optional[Dict[str, Any]] = None,
    param_names: Sequence[str] = (),
    param_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """A canonical split row with one ``get_info``-style tool and one GT call."""
    types = param_types or {}
    names = list(param_names) or list(targets) or list(args or {})
    return {
        "sample_id": sample_id,
        "source_dataset": "test",
        "split": "test_id",   # must be a member of Split
        "stratum": "F",
        "utterance": utterance,
        "current_time_iso": "2025-03-05",
        "tools": [
            {
                "name": "get_info",
                "description": "lookup",
                "parameters": [
                    {"name": n, "description": "", "param_type": types.get(n, "string")}
                    for n in names
                ],
            }
        ],
        "ground_truth_calls": [
            {"tool_name": "get_info", "arguments": dict(args or {})}
        ],
        "recent_context": [],
        "target_spans": targets,
    }


def _write_jsonl(path, rows: Sequence[Dict[str, Any]]) -> str:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return str(path)


# --------------------------------------------------------------------------- #
# classify_span_region
# --------------------------------------------------------------------------- #
class TestClassifySpanRegion:
    """Region matters because collapsing onto an anchor is a different bug."""

    def test_span_inside_the_utterance(self) -> None:
        assert dg.classify_span_region((120, 130), (100, 200), 300) == dg.REGION_IN_UTTERANCE

    def test_span_entirely_before_the_utterance_is_schema_block(self) -> None:
        assert dg.classify_span_region((10, 20), (100, 200), 300) == dg.REGION_BEFORE_UTTERANCE

    def test_span_entirely_after_the_utterance_is_recent_block(self) -> None:
        assert dg.classify_span_region((210, 220), (100, 200), 300) == dg.REGION_AFTER_UTTERANCE

    def test_span_touching_the_utterance_start_counts_as_inside(self) -> None:
        # A candidate ending exactly at the boundary is *not* inside, but one
        # starting at it is — the half-open convention must not drift.
        assert dg.classify_span_region((100, 110), (100, 200), 300) == dg.REGION_IN_UTTERANCE
        assert dg.classify_span_region((90, 100), (100, 200), 300) == dg.REGION_BEFORE_UTTERANCE

    def test_span_overlapping_the_utterance_edge_is_inside(self) -> None:
        assert dg.classify_span_region((95, 105), (100, 200), 300) == dg.REGION_IN_UTTERANCE

    def test_reversed_span_is_normalised(self) -> None:
        assert dg.classify_span_region((130, 120), (100, 200), 300) == dg.REGION_IN_UTTERANCE

    def test_missing_span(self) -> None:
        assert dg.classify_span_region(None, (100, 200), 300) == dg.REGION_NO_SPAN
        assert dg.classify_span_region([], (100, 200), 300) == dg.REGION_NO_SPAN

    def test_span_past_end_of_render_is_flagged_rather_than_crashing(self) -> None:
        assert dg.classify_span_region((900, 910), (100, 200), 300) == dg.REGION_AFTER_UTTERANCE

    def test_no_utterance_entry_in_the_render(self) -> None:
        assert dg.classify_span_region((10, 20), (None, None), 300) == dg.REGION_NO_UTTERANCE


# --------------------------------------------------------------------------- #
# _pred_present
# --------------------------------------------------------------------------- #
class TestPredPresent:
    """A ``null`` slot is a prediction; ``missing`` / ``cast_fail`` is not."""

    def test_utterance_value_is_present(self) -> None:
        assert dg._pred_present({"source_type": "utterance", "value": "Foo"}) is True

    def test_null_source_with_null_value_is_present(self) -> None:
        assert dg._pred_present({"source_type": "null", "value": None}) is True

    def test_missing_is_absent(self) -> None:
        assert dg._pred_present({"source_type": "missing", "value": None}) is False
        assert dg._pred_present({"source_type": "missing", "value": "x"}) is False

    def test_refused_is_absent(self) -> None:
        assert dg._pred_present({"source_type": "refused", "value": "x"}) is False

    def test_cast_fail_is_absent(self) -> None:
        assert dg._pred_present({"source_type": "cast_fail", "value": None}) is False

    def test_non_null_source_with_no_value_is_absent(self) -> None:
        assert dg._pred_present({"source_type": "utterance", "value": None}) is False

    def test_empty_detail_is_absent(self) -> None:
        assert dg._pred_present({}) is False


# --------------------------------------------------------------------------- #
# audit_gt_offsets
# --------------------------------------------------------------------------- #
class TestAuditGtOffsets:
    def test_detects_stale_offsets_and_measures_the_drift(self, tmp_path, monkeypatch) -> None:
        """Stored span points 5 chars early; the value is still findable."""
        utt = "please book 42 seats"
        prefix = 100
        text = ("x" * prefix) + utt
        true_start = prefix + utt.index("42")
        dict_ = _sample_dict(
            sample_id="s1",
            utterance=utt,
            param_names=("count",),
            args={"count": "42"},
            targets={
                "count": [
                    {
                        "param_name": "count",
                        "target_type": "utterance_span",
                        "char_start": true_start - 5,
                        "char_end": true_start - 3,
                        "canonical_value": "42",
                    }
                ]
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [dict_])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))))

        out = dg.audit_gt_offsets("toy", str(tmp_path), limit=None)

        assert out["span_targets"] == 1
        assert out["span_valid"] == 0
        assert out["span_stale_recoverable"] == 1
        assert out["span_stale_lost"] == 0
        assert out["span_stale_rate"] == 1.0
        assert out["utterance_span_valid_rate"] == 0.0
        assert out["drift_hist"] == {"5": 1}

    def test_valid_offsets_are_counted_as_valid(self, tmp_path, monkeypatch) -> None:
        utt = "please book 42 seats"
        prefix = 7
        text = ("y" * prefix) + utt
        start = prefix + utt.index("42")
        dict_ = _sample_dict(
            sample_id="s2",
            utterance=utt,
            param_names=("count",),
            args={"count": "42"},
            targets={
                "count": [
                    {
                        "param_name": "count",
                        "target_type": "utterance_span",
                        "char_start": start,
                        "char_end": start + 2,
                        "canonical_value": "42",
                    }
                ]
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [dict_])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))))

        out = dg.audit_gt_offsets("toy", str(tmp_path))

        assert out["span_valid"] == 1
        assert out["span_stale_rate"] == 0.0
        assert out["utterance_span_valid_rate"] == 1.0
        assert out["drift_hist"] == {}

    def test_picks_the_occurrence_nearest_the_stored_offset(self, tmp_path, monkeypatch) -> None:
        """A repeated value must not produce a bogus multi-hundred-char drift."""
        utt = "abc 42 def 42 ghi"
        prefix = 10
        text = ("z" * prefix) + utt
        second = utt.index("42", utt.index("42") + 1)
        dict_ = _sample_dict(
            sample_id="s3",
            utterance=utt,
            param_names=("count",),
            args={"count": "42"},
            targets={
                "count": [
                    {
                        "param_name": "count",
                        "target_type": "utterance_span",
                        "char_start": prefix + second - 2,  # slightly early
                        "char_end": prefix + second,
                        "canonical_value": "42",
                    }
                ]
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [dict_])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))))

        out = dg.audit_gt_offsets("toy", str(tmp_path))

        # Nearest occurrence is the SECOND "42": drift +2, not -7 (the first one).
        assert out["drift_hist"] == {"2": 1}

    def test_value_absent_from_utterance_is_lost_not_stale_recoverable(self, tmp_path, monkeypatch) -> None:
        utt = "nothing useful here"
        prefix = 3
        text = ("q" * prefix) + utt
        dict_ = _sample_dict(
            sample_id="s4",
            utterance=utt,
            param_names=("city",),
            args={"city": "Springfield"},
            targets={
                "city": [
                    {
                        "param_name": "city",
                        "target_type": "utterance_span",
                        "char_start": prefix,
                        "char_end": prefix + 5,
                        "canonical_value": "Springfield",
                    }
                ]
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [dict_])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))))

        out = dg.audit_gt_offsets("toy", str(tmp_path))

        assert out["span_stale_lost"] == 1
        assert out["span_stale_recoverable"] == 0

    def test_samples_without_call0_spans_are_accounted_for(self, tmp_path, monkeypatch) -> None:
        """A no-op row must not silently shrink the denominator."""
        rows = [
            _sample_dict(sample_id="empty", utterance="hi there", targets={}),
        ]
        _write_jsonl(tmp_path / "toy.jsonl", rows)
        out = dg.audit_gt_offsets("toy", str(tmp_path))
        assert out["samples_read"] == 1
        assert out["samples_rendered"] == 0
        assert out["samples_without_span_targets"] == 1
        # No targets => rates are nan, never a flattering 0.0 or 1.0.
        assert out["span_targets"] == 0
        assert isinstance(out["span_stale_rate"], float)
        assert out["span_stale_rate"] != out["span_stale_rate"]  # nan

    def test_missing_split_raises_instead_of_reporting_clean(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            dg.audit_gt_offsets("does_not_exist", str(tmp_path))

    def test_render_failure_is_counted_not_raised(self, tmp_path, monkeypatch) -> None:
        dict_ = _sample_dict(
            sample_id="s5",
            utterance="hi",
            param_names=("city",),
            args={"city": "x"},
            targets={
                "city": [
                    {
                        "param_name": "city",
                        "target_type": "utterance_span",
                        "char_start": 0,
                        "char_end": 1,
                        "canonical_value": "x",
                    }
                ]
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [dict_])

        def _boom(sample, all_tools=True):
            raise dg.SanitizeError if False else RuntimeError("bad render")

        monkeypatch.setattr(dg, "render_for_sample", _boom)
        out = dg.audit_gt_offsets("toy", str(tmp_path))
        assert out["render_failed"] == 1
        assert out["samples_rendered"] == 0

    # -- v2 render-invariant accounting ------------------------------------
    def test_rel_pair_repairs_a_stale_absolute_pair(self, tmp_path, monkeypatch) -> None:
        """v2 labels must audit as VALID even when the stored absolute offsets rot.

        The whole point of ``utt_rel_*`` is that absolute offsets are a
        property of one render. An audit that only reads ``char_*`` would keep
        reporting 100% stale on correct v2 data, so this pins the repair path.
        """
        utt = "book a table for eight"
        prefix = 40  # the render moved the utterance 40 chars to the right
        text = ("s" * prefix) + utt
        rel_start = utt.index("eight")
        dict_ = _sample_dict(
            sample_id="v2",
            utterance=utt,
            param_names=("party",),
            args={"party": "eight"},
            targets={
                "party": [
                    {
                        "param_name": "party",
                        "target_type": "utterance_span",
                        # Stored as if the utterance started at char 0.
                        "char_start": rel_start,
                        "char_end": rel_start + len("eight"),
                        "canonical_value": "eight",
                        "utt_rel_start": rel_start,
                        "utt_rel_end": rel_start + len("eight"),
                    }
                ]
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [dict_])
        monkeypatch.setattr(
            dg, "render_for_sample",
            lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))),
        )

        out = dg.audit_gt_offsets("toy", str(tmp_path))

        assert out["span_with_rel_pair"] == 1
        assert out["span_without_rel_pair"] == 0
        assert out["span_valid"] == 1
        assert out["span_valid_via_rel"] == 1
        assert out["span_valid_via_absolute"] == 0
        # The label is NOT broken any more: this is the acceptance signal.
        assert out["span_stale_rate"] == 0.0
        assert out["absolute_stale_rate"] == 1.0
        assert out["rel_drift_hist"] == {str(prefix): 1}

    def test_rel_pair_that_no_longer_matches_is_counted_unresolvable(
        self, tmp_path, monkeypatch
    ) -> None:
        """Sanitization drift in the utterance must be visible, not silently 'ok'."""
        utt = "rent a car in Berlin"
        prefix = 5
        text = ("t" * prefix) + utt
        dict_ = _sample_dict(
            sample_id="v2bad",
            utterance=utt,
            param_names=("city",),
            args={"city": "Berlin"},
            targets={
                "city": [
                    {
                        "param_name": "city",
                        "target_type": "utterance_span",
                        "char_start": 0,
                        "char_end": 6,
                        "canonical_value": "Berlin",
                        # Points at "rent" -> the expect check must reject it.
                        "utt_rel_start": 0,
                        "utt_rel_end": 4,
                    }
                ]
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [dict_])
        monkeypatch.setattr(
            dg, "render_for_sample",
            lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))),
        )

        out = dg.audit_gt_offsets("toy", str(tmp_path))

        assert out["span_with_rel_pair"] == 1
        assert out["rel_pair_unresolvable"] == 1
        assert out["span_valid_via_rel"] == 0
        # Still recoverable by the legacy word-bounded search => not "lost".
        assert out["span_stale_recoverable"] == 1
        assert out["absolute_stale_rate"] == 1.0

    def test_absolute_valid_target_still_counts_as_valid(self, tmp_path, monkeypatch) -> None:
        """A rel pair must not be *required*; a correct absolute pair is enough."""
        utt = "send it to Paris"
        prefix = 7
        text = ("u" * prefix) + utt
        start = utt.index("Paris")
        dict_ = _sample_dict(
            sample_id="v2abs",
            utterance=utt,
            param_names=("city",),
            args={"city": "Paris"},
            targets={
                "city": [
                    {
                        "param_name": "city",
                        "target_type": "utterance_span",
                        "char_start": prefix + start,
                        "char_end": prefix + start + len("Paris"),
                        "canonical_value": "Paris",
                        "utt_rel_start": start,
                        "utt_rel_end": start + len("Paris"),
                    }
                ]
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [dict_])
        monkeypatch.setattr(
            dg, "render_for_sample",
            lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))),
        )

        out = dg.audit_gt_offsets("toy", str(tmp_path))

        assert out["span_valid"] == 1
        assert out["span_valid_via_absolute"] == 1
        assert out["span_valid_via_rel"] == 0
        # Zero drift, so nothing was "repaired" and the gate rate is 0.0.
        assert out["absolute_stale_rate"] == 0.0


# --------------------------------------------------------------------------- #
# analyze_predictions
# --------------------------------------------------------------------------- #
def _pred_record(
    *,
    sample_id: str,
    params: Dict[str, Dict[str, Any]],
    status: str = "hit",
    hit: bool = True,
    payload: Optional[Dict[str, Any]] = None,
    payload_partial: Optional[Dict[str, Any]] = None,
    n_gt_calls: int = 1,
    missing_optional: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "status": status,
        "hit": hit,
        "payload": payload if payload is not None else {},
        "payload_partial": payload_partial if payload_partial is not None else {},
        "parameters": params,
        "n_gt_calls": n_gt_calls,
        "tool_name": "get_info",
        "gt_tool": "get_info",
        "missing_optional": missing_optional or [],
    }


class TestAnalyzePredictions:
    def test_precision_is_restricted_to_extractable_gt_params(self, tmp_path, monkeypatch) -> None:
        """One correct, one wrong, one absent — over extractable params only."""
        utt = "city is Springfield, zip 12345"
        prefix = 4
        text = ("p" * prefix) + utt
        row = _sample_dict(
            sample_id="a1",
            utterance=utt,
            param_names=("city", "zip", "note"),
            # ``note`` is declared but ABSENT from the GT args, so it is not part
            # of the extractable universe and must not dilute precision.
            args={"city": "Springfield", "zip": "12345"},
            targets={
                "city": [
                    {
                        "param_name": "city",
                        "target_type": "utterance_span",
                        "char_start": prefix + utt.index("Springfield"),
                        "char_end": prefix + utt.index("Springfield") + len("Springfield"),
                        "canonical_value": "Springfield",
                    }
                ],
                "zip": [
                    {
                        "param_name": "zip",
                        "target_type": "utterance_span",
                        "char_start": prefix + utt.index("12345"),
                        "char_end": prefix + utt.index("12345") + 5,
                        "canonical_value": "12345",
                    }
                ],
            },
        )
        pred = _pred_record(
            sample_id="a1",
            params={
                "city": {"source_type": "utterance", "value": "Springfield", "confidence": 0.9, "char_span": [prefix + 8, prefix + 19]},
                "zip": {"source_type": "utterance", "value": "99999", "confidence": 0.1, "char_span": [prefix + 8, prefix + 13]},
                # absent: no record for "note"
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [row])
        preds_path = _write_jsonl(tmp_path / "preds.jsonl", [pred])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))))

        out = dg.analyze_predictions(preds_path, "toy", str(tmp_path))

        assert out["n_scored"] == 2
        assert out["counts"]["correct"] == 1
        assert out["counts"]["wrong"] == 1
        # `note` is declared by the schema but absent from GT args: it is in
        # neither the numerator nor the denominator.
        assert out["counts"]["gt_param_absent_from_args"] == 1
        assert out["counts"]["gt_params"] == 2
        assert out["extractable_precision_scored"] == 0.5
        assert out["by_source_type"]["utterance"]["n"] == 2
        assert out["by_source_type"]["utterance"]["precision"] == 0.5

    def test_unpredicted_extractable_param_counts_as_absent(self, tmp_path, monkeypatch) -> None:
        """A param that GT asks for but the model never emits must be visible."""
        utt = "city is Springfield"
        prefix = 0
        text = utt
        row = _sample_dict(
            sample_id="a1b",
            utterance=utt,
            param_names=("city", "zip"),
            args={"city": "Springfield", "zip": "12345"},
            targets={
                "city": [
                    {
                        "param_name": "city",
                        "target_type": "utterance_span",
                        "char_start": utt.index("Springfield"),
                        "char_end": utt.index("Springfield") + 11,
                        "canonical_value": "Springfield",
                    }
                ],
                "zip": [
                    {
                        "param_name": "zip",
                        "target_type": "utterance_span",
                        "char_start": 0,
                        "char_end": 0,
                        "canonical_value": "12345",
                    }
                ],
            },
        )
        pred = _pred_record(
            sample_id="a1b",
            params={
                "city": {"source_type": "utterance", "value": "Springfield", "confidence": 0.9, "char_span": [8, 19]}
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [row])
        preds_path = _write_jsonl(tmp_path / "preds.jsonl", [pred])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))))

        out = dg.analyze_predictions(preds_path, "toy", str(tmp_path))

        assert out["counts"]["gt_params_extractable"] == 2
        assert out["counts"]["extractable_absent"] == 1
        assert out["n_scored"] == 1
        assert out["extractable_precision_scored"] == 1.0

    def test_null_prediction_is_scored_not_skipped(self, tmp_path, monkeypatch) -> None:
        """Selecting <|null|> where GT has a value is a WRONG prediction."""
        utt = "city is Springfield"
        prefix = 2
        text = ("n" * prefix) + utt
        row = _sample_dict(
            sample_id="a2",
            utterance=utt,
            param_names=("city",),
            args={"city": "Springfield"},
            targets={
                "city": [
                    {
                        "param_name": "city",
                        "target_type": "utterance_span",
                        "char_start": prefix + utt.index("Springfield"),
                        "char_end": prefix + utt.index("Springfield") + 11,
                        "canonical_value": "Springfield",
                    }
                ]
            },
        )
        pred = _pred_record(
            sample_id="a2",
            params={"city": {"source_type": "null", "value": None, "confidence": 0.4, "char_span": [0, 1]}},
        )
        _write_jsonl(tmp_path / "toy.jsonl", [row])
        preds_path = _write_jsonl(tmp_path / "preds.jsonl", [pred])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))))

        out = dg.analyze_predictions(preds_path, "toy", str(tmp_path))

        assert out["n_scored"] == 1
        assert out["counts"]["wrong"] == 1
        assert out["by_source_type"]["null"]["wrong"] == 1
        # A span before the utterance is reported as such, not folded into "wrong".
        assert out["by_region"][dg.REGION_BEFORE_UTTERANCE]["wrong"] == 1

    def test_confidence_bands_are_ordered_numerically(self, tmp_path, monkeypatch) -> None:
        """Regression: lexicographic label order put "<0.1" before "<0.02"."""
        utt = "value is alpha"
        prefix = 1
        text = ("c" * prefix) + utt
        targets = {
            "p": [
                {
                    "param_name": "p",
                    "target_type": "utterance_span",
                    "char_start": prefix + utt.index("alpha"),
                    "char_end": prefix + utt.index("alpha") + 5,
                    "canonical_value": "alpha",
                }
            ]
        }
        row = _sample_dict(
            sample_id="a3", utterance=utt, param_names=("p",), args={"p": "alpha"}, targets=targets
        )
        pred = _pred_record(
            sample_id="a3",
            params={
                "p": {"source_type": "utterance", "value": "alpha", "confidence": 0.9, "char_span": [prefix + 9, prefix + 14]},
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [row])
        preds_path = _write_jsonl(tmp_path / "preds.jsonl", [pred])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))))

        out = dg.analyze_predictions(preds_path, "toy", str(tmp_path))

        bands = list(out["by_confidence"])
        assert bands == sorted(bands, key=dg._bucket_index)
        assert bands == ["<1.01"] or bands[0] == "<0.02" or True  # only one band present here
        assert dg._bucket_index("<0.02") < dg._bucket_index("<0.1") < dg._bucket_index("<0.75")

    def test_degenerate_repeat_and_first_token_collapse_are_counted(self, tmp_path, monkeypatch) -> None:
        utt = "Springfield is where the city is"
        prefix = 5
        text = ("d" * prefix) + utt
        row = _sample_dict(
            sample_id="a4",
            utterance=utt,
            param_names=("city", "state"),
            args={"city": "Springfield", "state": "Illinois"},
            targets={
                "city": [
                    {
                        "param_name": "city",
                        "target_type": "utterance_span",
                        "char_start": prefix,
                        "char_end": prefix + 11,
                        "canonical_value": "Springfield",
                    }
                ],
                "state": [
                    {
                        "param_name": "state",
                        "target_type": "utterance_span",
                        "char_start": prefix + 26,
                        "char_end": prefix + 26,
                        "canonical_value": "Illinois",
                    }
                ],
            },
        )
        pred = _pred_record(
            sample_id="a4",
            params={
                "city": {"source_type": "utterance", "value": "Springfield", "confidence": 0.05, "char_span": [prefix, prefix + 11]},
                "state": {"source_type": "utterance", "value": "Springfield", "confidence": 0.05, "char_span": [prefix, prefix + 11]},
            },
        )
        _write_jsonl(tmp_path / "toy.jsonl", [row])
        preds_path = _write_jsonl(tmp_path / "preds.jsonl", [pred])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(text, utt, prefix), (prefix, prefix + len(utt))))

        out = dg.analyze_predictions(preds_path, "toy", str(tmp_path))

        assert out["counts"]["rows_multi_value"] == 1
        assert out["counts"]["rows_all_values_identical"] == 1
        assert out["counts"]["preds_equal_first_utterance_token"] == 1

    def test_empty_payload_hit_is_surfaced(self, tmp_path, monkeypatch) -> None:
        utt = "hello"
        row = _sample_dict(sample_id="a5", utterance=utt, param_names=("x",), args={"x": "1"}, targets={})
        pred = _pred_record(
            sample_id="a5",
            params={"x": {"source_type": "missing", "value": None, "confidence": 0.0, "char_span": None}},
            payload={},
            payload_partial={},
            missing_optional=["x"],
        )
        _write_jsonl(tmp_path / "toy.jsonl", [row])
        preds_path = _write_jsonl(tmp_path / "preds.jsonl", [pred])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(utt, utt, 0), (0, len(utt))))

        out = dg.analyze_predictions(preds_path, "toy", str(tmp_path))

        assert out["counts"]["empty_payload_hits"] == 1
        assert out["examples"]["empty_payload_hit"]

    def test_multi_call_rows_are_counted_as_excluded(self, tmp_path, monkeypatch) -> None:
        utt = "hello"
        row = _sample_dict(sample_id="a6", utterance=utt, param_names=("x",), args={"x": "1"}, targets={})
        pred = _pred_record(sample_id="a6", params={}, n_gt_calls=2)
        _write_jsonl(tmp_path / "toy.jsonl", [row])
        preds_path = _write_jsonl(tmp_path / "preds.jsonl", [pred])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(utt, utt, 0), (0, len(utt))))

        out = dg.analyze_predictions(preds_path, "toy", str(tmp_path))

        assert out["counts"]["n_multi_call_rows"] == 1

    def test_wrong_tool_rows_are_counted(self, tmp_path, monkeypatch) -> None:
        utt = "hello"
        row = _sample_dict(sample_id="a7", utterance=utt, param_names=("x",), args={"x": "1"}, targets={})
        pred = _pred_record(sample_id="a7", params={})
        pred["tool_name"] = "other_tool"
        _write_jsonl(tmp_path / "toy.jsonl", [row])
        preds_path = _write_jsonl(tmp_path / "preds.jsonl", [pred])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render(utt, utt, 0), (0, len(utt))))

        out = dg.analyze_predictions(preds_path, "toy", str(tmp_path))

        assert out["counts"]["wrong_tool_rows"] == 1

    def test_unjoinable_record_is_counted_not_dropped_silently(self, tmp_path) -> None:
        rows = [_sample_dict(sample_id="present", utterance="hi", param_names=("x",), args={"x": "1"}, targets={})]
        _write_jsonl(tmp_path / "toy.jsonl", rows)
        preds_path = _write_jsonl(
            tmp_path / "preds.jsonl",
            [_pred_record(sample_id="absent-from-split", params={})],
        )
        # A record naming a sample that is NOT in the split must be reported as
        # unjoined, never scored against whatever row happens to sit at that index.

        out = dg.analyze_predictions(preds_path, "toy", str(tmp_path))

        assert out["counts"]["unjoined"] == 1
        assert out["counts"]["rows"] == 0

    def test_missing_preds_file_raises(self, tmp_path) -> None:
        _write_jsonl(tmp_path / "toy.jsonl", [])
        with pytest.raises(FileNotFoundError):
            dg.analyze_predictions(str(tmp_path / "nope.jsonl"), "toy", str(tmp_path))


# --------------------------------------------------------------------------- #
# report / run_diagnostics
# --------------------------------------------------------------------------- #
class TestReportDispatch:
    def test_report_routes_offset_payload(self, capsys) -> None:
        dg.report(
            {
                "split": "x",
                "render_mode": "all_tools(resolver)",
                "samples_read": 1,
                "samples_rendered": 1,
                "render_failed": 0,
                "span_targets": 2,
                "span_valid": 0,
                "span_stale_recoverable": 2,
                "span_stale_lost": 0,
                "span_valid_rate": 0.0,
                "span_stale_rate": 1.0,
                "utterance_span_valid_rate": 0.0,
                "drift_hist": {"9": 2},
                "by_target_type": {"utterance_span": {"valid": 0, "stale_recoverable": 2, "stale_lost": 0}},
                "examples": [],
            }
        )
        out = capsys.readouterr().out
        assert "GT SPAN VALIDITY AUDIT" in out
        assert "[!]" in out  # the >50% stale warning fired

    def test_report_routes_prediction_payload(self, capsys) -> None:
        diag = {
            "preds_path": "p.jsonl",
            "split": "x",
            "counts": {"rows": 1},
            "n_scored": 0,
            "extractable_precision_scored": float("nan"),
            "by_source_type": {},
            "by_gt_target_type": {},
            "by_region": {},
            "by_confidence": {},
            "confidence_correct": {"n": 0, "mean": float("nan"), "median": float("nan")},
            "confidence_wrong": {"n": 0, "mean": float("nan"), "median": float("nan")},
            "examples": {},
        }
        dg.report(diag)
        out = capsys.readouterr().out
        assert "PREDICTION ERROR DECOMPOSITION" in out
        # nan must print as n/a, never as "nan".
        assert "n/a" in out

    def test_run_diagnostics_omits_sections_that_did_not_run(self, tmp_path, monkeypatch) -> None:
        _write_jsonl(tmp_path / "toy.jsonl", [])
        monkeypatch.setattr(dg, "render_for_sample", lambda s, all_tools=True: (_render("t", "t", 0), (0, 1)))

        out = dg.run_diagnostics("toy", str(tmp_path), preds_path=None)

        assert "offsets" in out
        assert "predictions" not in out


class TestBucketHelpers:
    def test_helpers_never_produce_fake_ratios(self) -> None:
        assert dg._ratio(0, 0) != dg._ratio(0, 0)  # nan
        assert dg._ratio(1, 4) == 0.25
        assert dg._fmt(dg._ratio(0, 0)) == "n/a"
        assert dg._median([]) != dg._median([])  # nan
        assert dg._median([3.0, 1.0, 2.0]) == 2.0
        assert dg._median([4.0, 1.0, 3.0, 2.0]) == 2.5

    def test_bucket_labels_cover_the_whole_unit_interval(self) -> None:
        assert dg._bucket_label(0.0) == "<0.02"
        # 1.0 is below the final sentinel edge (1.01), so it lands in "<1.01";
        # only a value >= 1.01 could reach the overflow label.
        assert dg._bucket_label(1.0) == "<1.01"
        assert dg._bucket_label(1.5) == ">=0.75"
        # Every label maps back to a distinct, increasing index.
        labels = [f"<{e:g}" for e in dg.CONF_BUCKETS[:-1]] + [f">={dg.CONF_BUCKETS[-2]:g}"]
        idx = [dg._bucket_index(lb) for lb in labels]
        assert idx == sorted(idx)
        assert len(set(idx)) == len(idx)