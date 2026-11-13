"""Tests for Q4 provenance mapping + Q5 strict casts + Q6 schema validation.

Covers (docs/pipeline_io.md Q4/Q5/Q6):

* ``StrictTypeCaster.extract_original`` — norm2orig mapping, word-boundary
  snapping (<=1 char partial expands, else ``bad_boundary``), boundary trim,
  and ``bad_offsets`` aborting.
* ``StrictTypeCaster`` Q5 — strict bool/enum/integer (k/m/b)/number/date/
  array/object casts over raw utterance text; ``cast_canonical`` over legend
  values; ``cast_choice`` combining Q4+Q5.
* ``SchemaValidator`` Q6 — default materialization, required/type/enum checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from spancall.data.schema import ParameterSchema, ToolDefinition
from spancall.infer.cast import SchemaValidator, StrictTypeCaster


# --------------------------------------------------------------------------- #
# Q4 fixtures: duck-typed SpanTemplateOutput
# --------------------------------------------------------------------------- #


class _Norm:
    """Duck-type of ``span_template.template.NormalizedText``."""

    def __init__(self, norm: str, orig: str, norm2orig) -> None:
        self.norm = norm
        self.orig = orig
        self.norm2orig = norm2orig


class _Entry:
    """Minimal legend entry with the attributes cast.py reads."""

    def __init__(self, segment: str, char_start: int, char_end: int,
                 value: Any = None, selectable: bool = True) -> None:
        self.segment = segment
        self.char_start = char_start
        self.char_end = char_end
        self.value = value
        self.selectable = selectable


class _Output:
    """Duck-type of ``SpanTemplateOutput`` (needs ``.entries`` + ``.norm``)."""

    def __init__(self, entries, norm: _Norm) -> None:
        self.entries = entries
        self.norm = norm


def _ident_output(utterance: str, utt_start: int = 54) -> _Output:
    """Output with identity normalization, absolute offsets in [utt_start, ...).

    ``norm.norm``/``norm.orig`` hold the *utterance-only* text so relative
    offsets (``char_start - utt_start``) index directly into the utterance
    (Q4 contract), while the legend entry carries the absolute span.
    """
    norm = _Norm(utterance, utterance, list(range(len(utterance) + 1)))
    return _Output([_Entry("utterance", utt_start, utt_start + len(utterance))], norm)


def _p(name: str, ptype: str = "string", *, enum=None, required: bool = True,
       default: Any = None, items_type: Optional[str] = None) -> ParameterSchema:
    return ParameterSchema(
        name=name, description="", param_type=ptype, required=required,
        enum=list(enum) if enum else None, default=default, items_type=items_type,
    )


@pytest.fixture()
def caster() -> StrictTypeCaster:
    return StrictTypeCaster()


# --------------------------------------------------------------------------- #
# Q4 — span -> original text (extract_original)
# --------------------------------------------------------------------------- #


def test_q4_word_bounded_span_extracts_exact_text(caster: StrictTypeCaster) -> None:
    # "cat sat on the mat": "sat" occupies rel [4, 7) -> abs [58, 61).
    out = _ident_output("cat sat on the mat")
    ok, text, reason = caster.extract_original(char_start=58, char_end=61, output=out)
    assert ok and reason is None
    assert text == "sat"


def test_q4_partial_piece_up_to_one_char_snaps_to_word(caster: StrictTypeCaster) -> None:
    # span covers "at" (rel [5, 7)) -> snaps back to the full word "sat".
    out = _ident_output("cat sat on the mat")
    ok, text, reason = caster.extract_original(char_start=59, char_end=61, output=out)
    assert ok and reason is None
    assert text == "sat"


def test_q4_trailing_space_before_word_boundary_is_trimmed(caster: StrictTypeCaster) -> None:
    # span "sat!" (rel [4, 8)): ends at punctuation, not mid-word, so no
    # bad_boundary; _trim() then drops the trailing punctuation -> "sat".
    out = _ident_output("cat sat!")
    ok, text, reason = caster.extract_original(char_start=58, char_end=62, output=out)
    assert ok and reason is None
    assert text == "sat"


def test_q4_boundary_punctuation_is_trimmed(caster: StrictTypeCaster) -> None:
    # "Book a flight  ,  now": span "flight  ," (rel [7, 16)) trims -> "flight".
    out = _ident_output("Book a flight  ,  now")
    ok, text, reason = caster.extract_original(char_start=61, char_end=70, output=out)
    assert ok and reason is None
    assert text == "flight"


def test_q4_partial_piece_longer_than_one_char_abstains(caster: StrictTypeCaster) -> None:
    # "the catalog is here": span "at" (rel [5, 7)) has trailing partial "alog"
    # (4 chars) -> bad_boundary.
    out = _ident_output("the catalog is here")
    ok, text, reason = caster.extract_original(char_start=59, char_end=61, output=out)
    assert not ok
    assert reason == "bad_boundary"


def test_q4_bad_offsets_abstain(caster: StrictTypeCaster) -> None:
    out = _ident_output("cat sat on the mat")
    # char_end <= char_start -> empty span
    ok, _, reason = caster.extract_original(char_start=58, char_end=58, output=out)
    assert not ok and reason == "bad_offsets"
    # span entirely before the utterance region
    ok, _, reason = caster.extract_original(char_start=0, char_end=3, output=out)
    assert not ok and reason == "bad_offsets"
    # span beyond the normalized end
    ok, _, reason = caster.extract_original(char_start=900, char_end=901, output=out)
    assert not ok and reason == "bad_offsets"
    # missing norm (non-rendered output)
    bad = _Output([], None)
    ok, _, reason = caster.extract_original(char_start=58, char_end=61, output=bad)
    assert not ok and reason == "bad_offsets"


def test_q4_cast_choice_uses_legend_value_for_context_regions(caster: StrictTypeCaster) -> None:
    choice = SimpleNamespace(value="ECONOMY", region="enum", span=(10, 20))
    ok, value, reason = caster.cast_choice(choice, _p("cabin", enum=["ECONOMY", "BUSINESS"]))
    assert ok and reason is None
    assert value == "ECONOMY"


def test_q4_cast_choice_maps_utterance_span_then_casts(caster: StrictTypeCaster) -> None:
    # "book 42 seats": "42" occupies rel [5, 7) -> abs [59, 61).
    out = _ident_output("book 42 seats")
    choice = SimpleNamespace(value="42", region="utterance", span=(59, 61))
    ok, value, reason = caster.cast_choice(
        choice, _p("seats", ptype="integer"), output=out
    )
    assert ok and reason is None
    assert value == 42
    assert isinstance(value, int)


def test_q4_cast_choice_propagates_bad_boundary(caster: StrictTypeCaster) -> None:
    out = _ident_output("the catalog is here")
    choice = SimpleNamespace(value="at", region="utterance", span=(59, 61))
    ok, value, reason = caster.cast_choice(choice, _p("word"), output=out)
    assert not ok and reason == "bad_boundary"


# --------------------------------------------------------------------------- #
# Q5 — strict typed casts over raw text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("TRUE", True), ("  True  ", True),
    ("false", False), ("FALSE", False),
])
def test_q5_bool_accepts_exact_forms(caster: StrictTypeCaster, raw: str, expected: bool) -> None:
    ok, value = caster.cast_parameter(raw, _p("flag", ptype="boolean"))
    assert ok and value is expected


@pytest.mark.parametrize("raw", ["yes", "1", "0", "t", "maybe"])
def test_q5_bool_rejects_ambiguous(caster: StrictTypeCaster, raw: str) -> None:
    ok, value = caster.cast_parameter(raw, _p("flag", ptype="boolean"))
    assert not ok and value is None


def test_q5_enum_case_insensitive_returns_canonical(caster: StrictTypeCaster) -> None:
    ok, value = caster.cast_parameter("  first ", _p("cabin", enum=["ECONOMY", "BUSINESS", "FIRST"]))
    assert ok and value == "FIRST"


def test_q5_enum_rejects_out_of_pool(caster: StrictTypeCaster) -> None:
    ok, value = caster.cast_parameter("PREMIUM", _p("cabin", enum=["ECONOMY", "BUSINESS"]))
    assert not ok and value is None


@pytest.mark.parametrize("raw,expected", [
    ("42", 42), ("-17", -17), ("+3", 3), ("1,000", 1000), ("5k", 5000), ("2M", 2_000_000),
])
def test_q5_integer_accepts_strict_forms(caster: StrictTypeCaster, raw: str, expected: int) -> None:
    ok, value = caster.cast_parameter(raw, _p("n", ptype="integer"))
    assert ok and value == expected
    assert isinstance(value, int)


@pytest.mark.parametrize("raw", ["2.5", "1.000", "~500", "five", "1,00", "50%"])
def test_q5_integer_rejects_ambiguous(caster: StrictTypeCaster, raw: str) -> None:
    ok, value = caster.cast_parameter(raw, _p("n", ptype="integer"))
    assert not ok and value is None


@pytest.mark.parametrize("raw,expected", [
    ("3.14", 3.14), ("42", 42.0), (".5", 0.5), ("1.5m", 1_500_000.0), ("2k", 2_000.0),
])
def test_q5_number_accepts_strict_forms(caster: StrictTypeCaster, raw: str, expected: float) -> None:
    ok, value = caster.cast_parameter(raw, _p("x", ptype="number"))
    assert ok and value == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["~100", "1,5", "ten", "0x1F"])
def test_q5_number_rejects_ambiguous(caster: StrictTypeCaster, raw: str) -> None:
    ok, value = caster.cast_parameter(raw, _p("x", ptype="number"))
    assert not ok and value is None


@pytest.mark.parametrize("raw,expected", [
    ("2026-10-23", date(2026, 10, 23)),
    ("23 Oct 2026", date(2026, 10, 23)),
    ("Oct 23, 2026", date(2026, 10, 23)),
])
def test_q5_date_strict_formats(caster: StrictTypeCaster, raw: str, expected: date) -> None:
    ok, value = caster.cast_parameter(raw, _p("when", ptype="date"))
    assert ok and value == expected


@pytest.mark.parametrize("raw", ["2026/10/23", "tomorrow", "Oct 32, 2026", "10-23-2026"])
def test_q5_date_rejects_non_strict(caster: StrictTypeCaster, raw: str) -> None:
    ok, value = caster.cast_parameter(raw, _p("when", ptype="date"))
    assert not ok and value is None


def test_q5_array_elementwise_strict(caster: StrictTypeCaster) -> None:
    ok, value = caster.cast_parameter("a, b, c", _p("tags", ptype="array", items_type="string"))
    assert ok and value == ["a", "b", "c"]
    ok, value = caster.cast_parameter("[1, 2, 3]", _p("ids", ptype="array", items_type="integer"))
    assert ok and value == [1, 2, 3]
    ok, value = caster.cast_parameter(" 1 , 2 ", _p("ids", ptype="array", items_type="integer"))
    assert ok and value == [1, 2]


def test_q5_array_fails_if_any_element_fails(caster: StrictTypeCaster) -> None:
    ok, value = caster.cast_parameter("[1, x]", _p("ids", ptype="array", items_type="integer"))
    assert not ok and value is None
    ok, value = caster.cast_parameter("", _p("ids", ptype="array", items_type="integer"))
    assert not ok and value is None


def test_q5_object_strict_json(caster: StrictTypeCaster) -> None:
    ok, value = caster.cast_parameter('{"a": 1, "b": [1, 2]}', _p("cfg", ptype="object"))
    assert ok and value == {"a": 1, "b": [1, 2]}
    ok, value = caster.cast_parameter("not json", _p("cfg", ptype="object"))
    assert not ok and value is None
    ok, value = caster.cast_parameter("[1, 2]", _p("cfg", ptype="object"))
    assert not ok and value is None


def test_q5_string_collapses_whitespace(caster: StrictTypeCaster) -> None:
    ok, value = caster.cast_parameter("  hello   world  ", _p("msg"))
    assert ok and value == "hello world"
    ok, value = caster.cast_parameter("   ", _p("msg"))
    assert not ok and value is None


def test_q5_cast_canonical_enum_guard(caster: StrictTypeCaster) -> None:
    # enum pool constrains even canonical-looking strings (case-insensitive).
    ok, value = caster.cast_canonical("business", _p("cabin", enum=["ECONOMY", "BUSINESS"]))
    assert ok and value == "BUSINESS"
    ok, value = caster.cast_canonical("FIRST", _p("cabin", enum=["ECONOMY", "BUSINESS"]))
    assert not ok and value is None


def test_q5_cast_canonical_type_checks(caster: StrictTypeCaster) -> None:
    assert caster.cast_canonical(True, _p("b", ptype="boolean")) == (True, True)
    assert caster.cast_canonical(1, _p("b", ptype="boolean")) == (False, None)  # 1 is not bool
    assert caster.cast_canonical(5, _p("i", ptype="integer")) == (True, 5)
    assert caster.cast_canonical(5.0, _p("i", ptype="integer")) == (False, None)
    assert caster.cast_canonical(5, _p("n", ptype="number")) == (True, 5.0)
    ok, value = caster.cast_canonical(date(2026, 1, 1), _p("d", ptype="date"))
    assert ok and value == date(2026, 1, 1)
    assert caster.cast_canonical("2026-01-01", _p("d", ptype="date")) == (False, None)
    assert caster.cast_canonical([1, 2], _p("a", ptype="array")) == (True, [1, 2])
    assert caster.cast_canonical({"k": 1}, _p("o", ptype="object")) == (True, {"k": 1})
    ok, value = caster.cast_canonical(7, _p("s"))
    assert ok and value == "7"
    assert caster.cast_canonical(None, _p("x")) == (True, None)


# --------------------------------------------------------------------------- #
# Q6 — SchemaValidator
# --------------------------------------------------------------------------- #


def _tool(*params: ParameterSchema, name: str = "tool") -> ToolDefinition:
    return ToolDefinition(name=name, description="d", parameters=list(params))


def test_q6_apply_defaults_materializes_optional() -> None:
    tool = _tool(
        _p("a", required=True),
        _p("b", required=False, default=42),
        _p("c", required=False),  # no default -> stays absent
    )
    out = SchemaValidator().apply_defaults({"a": 1}, tool)
    assert out == {"a": 1, "b": 42}


def test_q6_validate_ok() -> None:
    tool = _tool(_p("a", ptype="integer"), _p("b", enum=["X", "Y"]))
    ok, reason = SchemaValidator().validate({"a": 5, "b": "Y"}, tool)
    assert ok and reason is None


def test_q6_validate_missing_required() -> None:
    tool = _tool(_p("a", ptype="integer"), _p("b", ptype="string"))
    ok, reason = SchemaValidator().validate({"a": 5}, tool)
    assert not ok and reason == "missing_required:b"


def test_q6_validate_type_mismatch() -> None:
    tool = _tool(_p("a", ptype="integer"))
    ok, reason = SchemaValidator().validate({"a": "five"}, tool)
    assert not ok and reason == "type_mismatch:a"


def test_q6_validate_enum_mismatch() -> None:
    tool = _tool(_p("b", enum=["X", "Y"]))
    ok, reason = SchemaValidator().validate({"b": "Z"}, tool)
    assert not ok and reason == "enum_mismatch:b"


def test_q6_bool_not_int() -> None:
    tool = _tool(_p("a", ptype="integer"))
    ok, _ = SchemaValidator().validate({"a": True}, tool)
    assert not ok  # bool is not a valid JSON integer

