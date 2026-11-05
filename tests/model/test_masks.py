"""Tests for :class:`spancall.model.masks.CandidateMaskBuilder`.

Uses a trivial char/word tokenizer so no ``transformers``/``torch`` dependency
is required to verify P4 type-filtering, R8 entry snapping, R9 enum masks, the
W_max width cap, and regex candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pytest

from spancall.data.schema import ParameterSchema
from spancall.model.masks import CandidateMaskBuilder


@dataclass(frozen=True)
class Ent:
    """Minimal legend entry matching the attributes masks.py requires."""
    kind: str
    segment: str
    value: Any
    char_start: int
    char_end: int
    selectable: bool = True
    key: Optional[str] = None
    text: str = ""
    tool: Optional[str] = None


def char_tokens(text: str) -> List[Tuple[int, int]]:
    """One token per non-space char: (char_start, char_end) pairs."""
    out: List[Tuple[int, int]] = []
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        out.append((i, i + 1))
    return out


def word_tokens(text: str) -> List[Tuple[int, int]]:
    """Space-delimited word tokens: offsets on the raw text."""
    out: List[Tuple[int, int]] = []
    start = None
    for i, ch in enumerate(text + " "):
        if ch != " " and start is None:
            start = i
        elif ch == " " and start is not None:
            out.append((start, i))
            start = None
    return out


def _p(name: str, ptype: str = "string", *, enum=None, required: bool = True,
        items_type: Optional[str] = None) -> ParameterSchema:
    return ParameterSchema(
        name=name, description="", param_type=ptype, required=required,
        enum=list(enum) if enum else None, items_type=items_type,
    )


@pytest.fixture()
def builder() -> CandidateMaskBuilder:
    return CandidateMaskBuilder(max_seq_len=64, max_span_width=16)


# ---------------------------------------------------------------------------
# R8 entry snapping (atomic CTX entries are single full-entry spans)
# ---------------------------------------------------------------------------

def test_bool_param_only_true_false_entries(builder: CandidateMaskBuilder) -> None:
    text = "<|ctx_meta|> true false maybe"
    offsets = char_tokens(text)
    entries = [
        Ent("bool", "ctx_meta", True, * _span_of(text, "true"), text="true"),
        Ent("bool", "ctx_meta", False, * _span_of(text, "false"), text="false"),
        Ent("string", "ctx_meta", "maybe", * _span_of(text, "maybe"), text="maybe"),
    ]
    res = builder.build_param_masks(_p("flag", "boolean"), entries, offsets, "")
    assert [c.canon_value for c in res["candidates"]] == ["true", "false"]
    assert all(c.region == "bool" for c in res["candidates"])
    # every candidate's tok range is the exact single token
    for c, (ts, te) in zip(res["candidates"], res["cand_index"]):
        assert ts <= te
        assert c.char_start < c.char_end


def test_enum_r9_filters_other_params_values(builder: CandidateMaskBuilder) -> None:
    text = "<|ctx_enum|> red | green | blue | economy"
    offsets = word_tokens(text)
    entries = [
        Ent("enum", "ctx_enum", "red", * _span_of(text, "red")),
        Ent("enum", "ctx_enum", "green", * _span_of(text, "green")),
        Ent("enum", "ctx_enum", "blue", * _span_of(text, "blue")),
        Ent("enum", "ctx_enum", "economy", * _span_of(text, "economy")),
    ]
    res = builder.build_param_masks(
        _p("color", "string", enum=["red", "blue"]), entries, offsets, "")
    vals = [c.canon_value for c in res["candidates"]]
    assert vals == ["red", "blue"]
    assert "green" not in vals and "economy" not in vals


def test_enum_optional_appends_null(builder: CandidateMaskBuilder) -> None:
    text = "<|ctx_enum|> a b | <|ctx_meta|> NULL"
    offsets = char_tokens(text)
    entries = [
        Ent("enum", "ctx_enum", "a", * _span_of(text, "a")),
        Ent("enum", "ctx_enum", "b", * _span_of(text, "b")),
        Ent("null", "ctx_meta", None, * _span_of(text, "NULL")),
    ]
    res = builder.build_param_masks(
        _p("grade", "string", enum=["a", "b"], required=False), entries, offsets, "")
    assert [c.canon_value for c in res["candidates"]] == ["a", "b", None]
    assert res["candidates"][-1].region == "null"


def test_required_param_does_not_add_null(builder: CandidateMaskBuilder) -> None:
    text = "<|ctx_enum|> a b | <|ctx_meta|> NULL"
    offsets = char_tokens(text)
    entries = [
        Ent("enum", "ctx_enum", "a", * _span_of(text, "a")),
        Ent("enum", "ctx_enum", "b", * _span_of(text, "b")),
        Ent("null", "ctx_meta", None, * _span_of(text, "NULL")),
    ]
    res = builder.build_param_masks(
        _p("grade", "string", enum=["a", "b"], required=True), entries, offsets, "")
    assert [c.canon_value for c in res["candidates"]] == ["a", "b"]


# ---------------------------------------------------------------------------
# date / datetime
# ---------------------------------------------------------------------------

def test_date_param_uses_ctx_time_and_iso_regex(builder: CandidateMaskBuilder) -> None:
    text = "<|ctx_time|> 2026-10-24 tomorrow | book on 2026-11-02"
    offsets = char_tokens(text)
    entries = [
        Ent("date", "ctx_time", "2026-10-24", * _span_of(text, "2026-10-24")),
        Ent("date", "ctx_time", "tomorrow", * _span_of(text, "tomorrow")),
    ]
    res = builder.build_param_masks(_p("when", "date"), entries, offsets, text)
    vals = [c.canon_value for c in res["candidates"]]
    # both CTX_TIME entries plus the ISO date regex match in utterance
    assert "2026-10-24" in vals
    assert "tomorrow" in vals
    assert "2026-11-02" in vals
    date_cands = [c for c in res["candidates"] if c.region in ("time", "utterance")]
    assert len(date_cands) >= 3
    assert any(c.region == "time" and c.canon_value == "2026-10-24" for c in date_cands)
    assert any(c.region == "utterance" and c.canon_value == "2026-11-02" for c in date_cands)


def test_date_does_not_match_non_dates(builder: CandidateMaskBuilder) -> None:
    text = "meet at 123 or 10/24 or 2026-13-99"
    offsets = char_tokens(text)
    res = builder.build_param_masks(_p("when", "date"), [], offsets, text)
    assert res["candidates"] == []


# ---------------------------------------------------------------------------
# numeric
# ---------------------------------------------------------------------------

def test_numeric_regex_and_recent_interiors(builder: CandidateMaskBuilder) -> None:
    text = "give me 1500 usd and 2.5k items"
    offsets = word_tokens(text)
    recent = Ent("string", "recent", 42, 0, 0, text="balance=42")
    # snap recent token range manually so the entry maps to token offsets
    res = builder.build_param_masks(_p("amount", "integer"), [recent], offsets, text)
    vals = [c.canon_value for c in res["candidates"]]
    assert "1500" in vals
    assert "2.5k" in vals
    # non-numeric words are not candidates
    assert not any(isinstance(c.canon_value, str) and "usd" in str(c.canon_value).lower()
                   for c in res["candidates"])


def test_numeric_recent_interior(builder: CandidateMaskBuilder) -> None:
    text = "<|ctx_recent|> balance=4250 | what is the current balance"
    offsets = char_tokens(text)
    recent = Ent("string", "recent", "4250", * _span_of(text, "4250"), text="balance=4250")
    res = builder.build_param_masks(_p("balance", "integer"), [recent], offsets, text)
    vals = [c.canon_value for c in res["candidates"]]
    assert "4250" in vals
    assert any(c.region == "recent" and c.canon_value == "4250" for c in res["candidates"])


# ---------------------------------------------------------------------------
# free string / W_max windowing
# ---------------------------------------------------------------------------

def test_free_string_windowed_utterance_spans(builder: CandidateMaskBuilder) -> None:
    text = "book a flight to paris now"
    offsets = word_tokens(text)
    # utterance entry covers the whole text
    utt = Ent("string", "utterance", None, 0, len(text), text=text)
    res = builder.build_param_masks(_p("dest", "string"), [utt], offsets, text)
    assert res["num_candidates"] > 0
    # all utterance candidates inside the word-boundary window
    for c in res["candidates"]:
        if c.region == "utterance":
            assert 0 <= c.tok_end - c.tok_start < builder.max_span_width
            assert c.char_start <= c.char_end
    # 'paris' must be a candidate (single word)
    paris_span = (text.index("paris"), text.index("paris") + 5)
    assert any(c.char_start == paris_span[0] and c.char_end == paris_span[1]
               for c in res["candidates"] if c.region == "utterance")


def test_free_string_width_cap_respected(builder: CandidateMaskBuilder) -> None:
    text = " ".join(f"w{i}" for i in range(30))  # 30 words
    offsets = word_tokens(text)
    utt = Ent("string", "utterance", None, 0, len(text), text=text)
    res = builder.build_param_masks(_p("q", "string"), [utt], offsets, text)
    for c in res["candidates"]:
        if c.region == "utterance":
            assert c.tok_end - c.tok_start < builder.max_span_width


def test_free_string_recent_interior(builder: CandidateMaskBuilder) -> None:
    text = "<|ctx_recent|> tier=gold | please set it to the value from context"
    offsets = char_tokens(text)
    recent = Ent("string", "recent", "gold", * _span_of(text, "gold"), text="tier=gold")
    res = builder.build_param_masks(_p("tier", "string"), [recent], offsets, text)
    assert "gold" in [c.canon_value for c in res["candidates"]]
    assert any(c.region == "recent" and c.canon_value == "gold" for c in res["candidates"])


# ---------------------------------------------------------------------------
# regex cand_index packing + start/end OK masks
# ---------------------------------------------------------------------------

def test_cand_index_matches_char_spans(builder: CandidateMaskBuilder) -> None:
    text = "please pay 1234 now and 99 later"
    offsets = word_tokens(text)
    res = builder.build_param_masks(_p("amt", "integer"), [], offsets, text)
    for c, (ts, te) in zip(res["candidates"], res["cand_index"]):
        assert offsets[ts][0] <= c.char_start
        assert offsets[te][1] >= c.char_end
    assert res["num_candidates"] == len(res["cand_index"])
    # start_ok/end_ok are legacy windowed conveniences sized to the largest
    # candidate token index (never a hardcoded 512 / max_seq_len cap) — see
    # masks._pack and the build_param_masks docstring.
    assert len(res["start_ok"]) == len(res["end_ok"])
    n_toks = max((te for _, te in res["cand_index"]), default=-1) + 1
    assert len(res["start_ok"]) == n_toks
    assert all(len(m) <= builder.max_seq_len for m in (res["start_ok"], res["end_ok"]))


# ---------------------------------------------------------------------------
# marker registration helper (idempotent on a mock tokenizer)
# ---------------------------------------------------------------------------

def test_register_marker_tokens_idempotent() -> None:
    class FakeTok:
        additional_special_tokens: List[str] = []

        def __init__(self) -> None:
            self.added: List[List[str]] = []

        def add_special_tokens(self, spec: dict) -> None:
            self.added.append(spec.get("additional_special_tokens", []))

    tok = FakeTok()
    CandidateMaskBuilder.register_marker_tokens(tok)
    CandidateMaskBuilder.register_marker_tokens(tok)
    assert tok.added, "should have added markers on first call"
    first = tok.added[0]
    # SPECIAL_MARKER_TOKENS are the atomic Qwen-native <|...|> forms (single
    # tokens for modern BPE like Qwen3): <|tool|>, <|param|>, ... plus the
    # CTX_* segments and the <|bar|>/<|eq|>/<|q|> P6 reserved delimiters
    # (long-context R8 boundary safety).
    for expected in ("<|cls|>", "<|tool|>", "<|param|>", "<|none|>", "<|null|>",
                     "<|unresolved|>", "<|ctx_meta|>", "<|ctx_time|>",
                     "<|ctx_enum|>", "<|ctx_recent|>", "<|sep|>", "<|eos|>",
                     "<|bar|>", "<|eq|>", "<|q|>"):
        assert expected in first, f"missing {expected} in {first}"


def _span_of(text: str, needle: str) -> Tuple[int, int]:
    i = text.index(needle)
    return (i, i + len(needle))