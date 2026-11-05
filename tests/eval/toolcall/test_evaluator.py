"""Tests for the offline tool-call evaluator (plan §4.2).

Covers (docs/full_implementation_plan.md §4.2, docs/pipeline_io.md Q3–Q9):

* ``payload_equal`` — numeric normalization (5 == 5.0), optional-default
  materialization from schema, key-set mismatch, nested lists/dicts.
* ``_classify_error`` — every bucket of the 7-way taxonomy including the
  now-reachable ``error_extra_call`` and ``error_missing_call`` branches.
* ``ToolCallEvaluator.evaluate`` — stratified F/A/all metrics,
  ``false_hit_rate_on_a`` safety, error aggregation, ``errors_by_sample``,
  ECE, and risk-coverage monotonicity.
* ``B1aBaseline.predict`` — deterministic extraction over a duck-typed
  ``SpanTemplateOutput`` (exact enum, single date, single numeric, boolean).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.eval.toolcall.evaluator import (
    B1aBaseline,
    ToolCallEvaluator,
    _classify_error,
    _deep_equal,
    payload_equal,
)
from spancall.eval.toolcall.metrics import ToolCallMetrics


# --------------------------------------------------------------------------- #
# payload_equal
# --------------------------------------------------------------------------- #


def _tool(*params: ParameterSchema, name: str = "t") -> ToolDefinition:
    return ToolDefinition(name=name, description="d", parameters=list(params))


def test_payload_equal_numeric_normalization() -> None:
    assert payload_equal({"x": 5}, {"x": 5.0}) is True
    assert payload_equal({"x": 5}, {"x": 5.0001}) is False
    # _deep_equal coerces bools via bool(a) == bool(b), so True == 1.
    assert payload_equal({"x": True}, {"x": 1}) is True


def test_payload_equal_none_semantics() -> None:
    assert payload_equal(None, None) is True
    assert payload_equal({"a": 1}, None) is False
    assert payload_equal(None, {"a": 1}) is False


def test_payload_equal_key_set_mismatch() -> None:
    assert payload_equal({"a": 1}, {"a": 1, "b": 2}) is False
    assert payload_equal({"a": 1, "b": 2}, {"a": 1}) is False


def test_payload_equal_key_order_irrelevant() -> None:
    assert payload_equal({"a": 1, "b": [1, {"c": 2}]}, {"b": [1, {"c": 2}], "a": 1}) is True


def test_payload_equal_optional_default_materialization() -> None:
    tool = _tool(
        ParameterSchema(name="a", description="", param_type="string", required=True),
        ParameterSchema(name="b", description="", param_type="integer",
                        required=False, default=42),
    )
    # predicate omits optional-with-default -> materialized on both sides.
    assert payload_equal({"a": "x"}, {"a": "x"}, schema=tool) is True
    assert payload_equal({"a": "x", "b": 42}, {"a": "x"}, schema=tool) is True
    # explicit different value still mismatches.
    assert payload_equal({"a": "x", "b": 7}, {"a": "x"}, schema=tool) is False


def test_payload_equal_scalar_fallback() -> None:
    assert payload_equal("abc", "abc") is True
    assert payload_equal("abc", "abd") is False


# --------------------------------------------------------------------------- #
# _classify_error buckets
# --------------------------------------------------------------------------- #


def test_classify_missing_call() -> None:
    cats = _classify_error("s", None, {"a": 1}, "t", "t")
    assert cats == ["error_missing_call"]


def test_classify_extra_call() -> None:
    # GT expected no call (empty payload), model emitted one.
    cats = _classify_error("s", {"a": 1}, None, "t", None)
    assert "error_extra_call" in cats
    # A stratum with no GT call: hitting is an extra call, not missing.
    assert "error_missing_call" not in cats


def test_classify_wrong_tool() -> None:
    cats = _classify_error("s", {"a": 1}, {"a": 1}, "weather2", "weather")
    assert "error_wrong_tool" in cats
    # exact tool match -> no wrong-tool
    cats = _classify_error("s", {"a": 1}, {"a": 1}, "weather", "weather")
    assert "error_wrong_tool" not in cats


def test_classify_entity_swap() -> None:
    cats = _classify_error(
        "s",
        {"from": "SF", "to": "NY"},
        {"from": "NY", "to": "SF"},
        "flight", "flight",
    )
    assert "error_entity_swap" in cats
    assert "error_wrong_value" in cats


def test_classify_wrong_value_only() -> None:
    cats = _classify_error("s", {"a": 1}, {"a": 2}, "t", "t")
    assert "error_wrong_value" in cats
    assert "error_entity_swap" not in cats


def test_classify_duplicate_span() -> None:
    class Res:
        def __init__(self, span):  # noqa: ANN001
            self.char_span = span

    resolved = {"a": Res((0, 3)), "b": Res((0, 3))}
    cats = _classify_error(
        "s", {"a": "same", "b": "same"}, {"a": "X", "b": "Y"},
        "t", "t", resolved_params=resolved,
    )
    assert "error_duplicate_span" in cats


def test_classify_date_misparse() -> None:
    cats = _classify_error(
        "s", {"departure_date": "2026-10-24"}, {"departure_date": "2026-10-23"},
        "t", "t",
    )
    assert "error_date_misparse" in cats


def test_classify_perfect_hit_returns_no_categories() -> None:
    cats = _classify_error("s", {"a": 1}, {"a": 1}, "t", "t")
    assert cats == []


# --------------------------------------------------------------------------- #
# ToolCallEvaluator.evaluate — stratified metrics
# --------------------------------------------------------------------------- #


def _sample(sid: str, stratum: Stratum, tool: ToolDefinition,
            gt: Optional[GroundTruthCall]) -> UnifiedSample:
    return UnifiedSample(
        sample_id=sid,
        source_dataset="test",
        utterance="book a flight",
        tools=[tool],
        ground_truth_calls=[gt] if gt else [],
        stratum=stratum,
    )


def _weather() -> ToolDefinition:
    return _tool(
        ParameterSchema("city", "", "string", required=True),
        name="weather",
    )


def test_evaluate_stratified_metrics_and_errors() -> None:
    tool = _weather()
    samples = [
        _sample("f-ok", Stratum.F, tool,
                GroundTruthCall("weather", {"city": "SF"})),
        _sample("f-abstain", Stratum.F, tool,
                GroundTruthCall("weather", {"city": "LA"})),
        _sample("a-false-hit", Stratum.A, tool, None),  # nothing should be called
        _sample("a-correct-abstain", Stratum.A, tool, None),
    ]
    predictions = [
        {"hit": True, "tool_name": "weather", "payload": {"city": "SF"},
         "confidence": 0.9},
        {"hit": False, "payload": None, "confidence": 0.2},
        {"hit": True, "tool_name": "weather", "payload": {"city": "NY"},
         "confidence": 0.8},
        {"hit": False, "payload": None, "confidence": 0.1},
    ]
    m = ToolCallEvaluator().evaluate(samples, predictions)
    assert isinstance(m, ToolCallMetrics)
    assert m.total_samples == 4
    assert m.hit_rate == pytest.approx(0.5)
    assert m.precision_at_hit == pytest.approx(0.5)   # 1 of 2 hits correct
    assert m.exact_payload_match == pytest.approx(0.5)  # f-ok + a-correct-abstain
    assert m.error_extra_call == 1
    # the A false hit is an extra call (fixed taxonomy), not a wrong value.
    assert m.error_wrong_value == 0
    assert m.errors_by_sample == {"a-false-hit": ["error_extra_call"]}

    # strata
    F = m.metrics_by_stratum["F"]
    A = m.metrics_by_stratum["A"]
    assert F["n"] == 2.0 and F["hit_rate"] == 0.5
    assert F["precision_at_hit"] == 1.0
    assert A["n"] == 2.0 and A["hit_rate"] == 0.5
    assert A["precision_at_hit"] == 0.0
    # false-hit safety: 1 A-hit / 2 A-rows
    assert m.false_hit_rate_on_a == pytest.approx(0.5)
    assert A["false_hit_rate_on_a"] == pytest.approx(0.5)
    # abstain counting
    assert m.abstain_reasons == {"abstained": 2}


def test_evaluate_all_abstain_is_safe() -> None:
    tool = _weather()
    samples = [
        _sample("f", Stratum.F, tool, GroundTruthCall("weather", {"city": "SF"})),
        _sample("a", Stratum.A, tool, None),
    ]
    predictions = [{"hit": False}, {"hit": False}]
    m = ToolCallEvaluator().evaluate(samples, predictions)
    assert m.hit_rate == 0.0
    assert m.precision_at_hit == 0.0
    assert m.false_hit_rate_on_a == 0.0
    assert m.error_extra_call == 0 and m.error_missing_call == 0
    assert not m.errors_by_sample


def test_evaluate_wrong_tool_counts() -> None:
    tool = _weather()
    samples = [_sample("f", Stratum.F, tool, GroundTruthCall("weather", {"city": "SF"}))]
    predictions = [{"hit": True, "tool_name": "geo", "payload": {"city": "SF"}}]
    m = ToolCallEvaluator().evaluate(samples, predictions)
    assert m.error_wrong_tool == 1
    assert m.tool_accuracy == 0.0  # wrong tool name
    assert m.error_wrong_value == 0  # payload itself matches


def test_evaluate_missing_rows_default_to_abstain() -> None:
    tool = _weather()
    samples = [_sample("f", Stratum.F, tool, GroundTruthCall("weather", {"city": "SF"}))]
    m = ToolCallEvaluator().evaluate(samples, [])  # no predictions provided
    assert m.total_samples == 1
    assert m.hit_rate == 0.0
    assert m.error_missing_call == 0  # abstaining (not a false hit) is safe


def test_expected_calibration_error_synthetic() -> None:
    tool = _weather()
    samples = [
        _sample(f"s{i}", Stratum.F, tool, GroundTruthCall("weather", {"city": "SF"}))
        for i in range(3)
    ]
    predictions = [
        {"hit": True, "tool_name": "weather", "payload": {"city": "SF"}, "confidence": 0.9},
        {"hit": True, "tool_name": "weather", "payload": {"city": "SF"}, "confidence": 0.9},
        {"hit": True, "tool_name": "weather", "payload": {"city": "BOGUS"}, "confidence": 0.1},
    ]
    m = ToolCallEvaluator().evaluate(samples, predictions)
    # bins [0, .5): conf .1/acc 0 -> .1; [.5,1): conf .9/acc 1 -> .1
    # ECE = (1/3)*0.1 + (2/3)*0.1 = 0.1
    assert m.ece == pytest.approx(0.1, abs=1e-9)


def test_risk_coverage_curve_monotone() -> None:
    tool = _weather()
    samples = [
        _sample(f"s{i}", Stratum.F, tool, GroundTruthCall("weather", {"city": "SF"}))
        for i in range(4)
    ]
    predictions = [
        {"hit": True, "tool_name": "weather", "payload": {"city": "SF"}, "confidence": 0.9},
        {"hit": True, "tool_name": "weather", "payload": {"city": "NY"}, "confidence": 0.8},
        {"hit": True, "tool_name": "weather", "payload": {"city": "SF"}, "confidence": 0.7},
        {"hit": True, "tool_name": "weather", "payload": {"city": "LA"}, "confidence": 0.6},
    ]
    m = ToolCallEvaluator().evaluate(samples, predictions)
    curve = m.risk_coverage
    assert len(curve) == 4
    coverages = [c["coverage"] for c in curve]
    assert coverages == sorted(coverages)
    for c in curve:
        assert c["precision"] == pytest.approx(1.0 - c["risk"], abs=1e-12)
    assert curve[0]["risk"] == 0.0
    assert curve[-1]["n_errors"] == 2


def test_eval_package_exports_contract() -> None:
    # the import chain the whole spancall.eval package relies on must hold
    from spancall.eval.toolcall import BaseToolCallEvaluator, ToolCallMetrics  # noqa: F401
    from spancall.eval import BaseToolCallEvaluator as TopLevel  # noqa: F401

    assert ToolCallEvaluator.__mro__[1].__name__ == "BaseToolCallEvaluator"
    assert TopLevel is not None


# --------------------------------------------------------------------------- #
# B1aBaseline — deterministic extraction
# --------------------------------------------------------------------------- #


def _out(text: str, entries: list) -> SimpleNamespace:
    return SimpleNamespace(text=text, entries=entries)


def _sel(value: Any) -> SimpleNamespace:
    return SimpleNamespace(selectable=True, value=value)


def test_b1a_exact_enum_match() -> None:
    schema = _tool(
        ParameterSchema("cabin", "", "string", enum=["ECONOMY", "BUSINESS", "FIRST"]),
        name="book",
    )
    out = _out("book me ECONOMY", [_sel("ECONOMY")])
    res = B1aBaseline().predict(out, schema)
    assert res["hit"] is True
    assert res["payload"] == {"cabin": "ECONOMY"}


def test_b1a_ambiguous_enum_abstains() -> None:
    schema = _tool(
        ParameterSchema("cabin", "", "string", enum=["ECONOMY", "BUSINESS"]),
        name="book",
    )
    out = _out("book me ECONOMY or BUSINESS", [_sel("ECONOMY"), _sel("BUSINESS")])
    res = B1aBaseline().predict(out, schema)
    assert res["hit"] is False  # two candidates -> abstain on that param


def test_b1a_single_date_match() -> None:
    schema = _tool(ParameterSchema("when", "", "date"), name="book")
    out = _out("on 2026-10-23 for two", [])
    res = B1aBaseline().predict(out, schema)
    assert res["payload"] == {"when": "2026-10-23"}


def test_b1a_single_numeric_match() -> None:
    schema = _tool(ParameterSchema("seats", "", "integer"), name="book")
    out = _out("book 42 seats", [])
    res = B1aBaseline().predict(out, schema)
    # B1a keeps the raw regex match string; the resolver caster converts it
    # to an int later on its own path.
    assert res["payload"] == {"seats": "42"}


def test_b1a_boolean_bridge() -> None:
    schema = _tool(ParameterSchema("window", "", "boolean"), name="book")
    out = _out("book true window", [_sel("true")])
    res = B1aBaseline().predict(out, schema)
    assert res["payload"] == {"window": True}


def test_b1a_empty_payload_is_abstain() -> None:
    schema = _tool(ParameterSchema("seats", "", "integer"), name="book")
    out = _out("no numbers here", [])
    res = B1aBaseline().predict(out, schema)
    assert res["hit"] is False and res["payload"] == {}


# --------------------------------------------------------------------------- #
# _deep_equal corner cases
# --------------------------------------------------------------------------- #


def test_deep_equal_corners() -> None:
    assert _deep_equal({"a": 5}, {"a": 5.0}) is True  # numeric normalization
    assert _deep_equal([1, 2], [1, 2.0]) is True
    assert _deep_equal({"a": 1}, {"a": 1, "b": 2}) is False
    assert _deep_equal(3, 3.0) is True
    assert _deep_equal(True, 1) is True  # bool coerced via bool(a) == bool(b)
    assert _deep_equal("1", 1) is False  # strings never coerce to numbers
    assert _deep_equal(None, None) is True