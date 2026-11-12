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
from spancall.model.masks import CandidateMaskBuilder, TokenIndex


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
# T4 candidate retention policy under a max_cands truncation
# ---------------------------------------------------------------------------

PACK_TEXT = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"


def _pack_masks(builder: CandidateMaskBuilder, max_cands: Optional[int]):
    """Build a free-string param over ``PACK_TEXT`` with the whole text as utterance."""
    offsets = word_tokens(PACK_TEXT)
    utt = Ent("string", "utterance", None, 0, len(PACK_TEXT), text=PACK_TEXT)
    return builder.build_param_masks(
        _p("q", "string"), [utt], offsets, PACK_TEXT, max_cands=max_cands
    )


def test_pack_head_fill_is_the_legacy_prefix(builder: CandidateMaskBuilder) -> None:
    """``pack_fill='head'`` must reproduce the pre-T4 behaviour exactly.

    The T4 rewrite is only safe if the legacy policy is still reachable and
    unchanged: it is the control arm of the train/serve comparison.
    """
    builder.pack_fill = "head"
    res = _pack_masks(builder, max_cands=3)
    assert res["num_candidates"] == 3
    # Head retention keeps the EARLIEST spans in document order.
    starts = [c.tok_start for c in res["candidates"]]
    assert starts == sorted(starts)
    assert max(starts) < 4, f"head fill kept a late span: {starts}"


def test_pack_protected_span_survives_truncation(builder: CandidateMaskBuilder) -> None:
    """A protected (GT) span must never be evicted by ``max_cands``.

    This is the actual T2 fix: parity protection is only meaningful if the span
    survives the cap. It also guards the ``tool::param`` vs bare-param key
    mismatch that previously made protection a silent no-op.
    """
    # NOTE: word_tokens yields CHAR offsets; candidates carry TOKEN indices.
    # With this one-token-per-word tokenizer the last word is token index n-1.
    last = len(word_tokens(PACK_TEXT)) - 1   # "juliet" -> (9, 9)
    target = (last, last)
    builder.pack_fill = "head"               # would definitely drop a late span
    builder.pack_protected = {target}
    res = _pack_masks(builder, max_cands=3)
    kept = {(c.tok_start, c.tok_end) for c in res["candidates"]}
    assert target in kept, f"protected span evicted by max_cands; kept={sorted(kept)}"


def test_pack_result_stays_sorted_by_document_position(
    builder: CandidateMaskBuilder,
) -> None:
    """Priority reordering must not leak out as a non-monotone candidate list.

    The pointer head's geometry and the legacy ``start_ok``/``end_ok`` masks both
    assume candidates ascend by token index; an unsorted list silently corrupts
    them rather than raising.
    """
    last = len(word_tokens(PACK_TEXT)) - 1
    target = (last, last)
    builder.pack_fill = "stratified"
    builder.pack_protected = {target}
    res = _pack_masks(builder, max_cands=4)
    starts = [c.tok_start for c in res["candidates"]]
    assert starts == sorted(starts), f"candidate order not monotone: {starts}"
    # The protected span survives under the OTHER fill policy too.
    assert target in {(c.tok_start, c.tok_end) for c in res["candidates"]}


def test_pack_protected_span_off_word_bound_survives(
    builder: CandidateMaskBuilder,
) -> None:
    """A GT span whose snapped start is NOT a word-bound token must survive.

    The generator only *spontaneously* emits starts at word boundaries, but the
    resolver protects GT spans by snapping their CHAR range onto tokens. Word
    bounds are derived from the joined entry text, so they need not coincide
    with the rendered coordinates a GT span was measured in — a protected start
    can therefore sit off the word-bound set. Widening the budget alone left
    those spans un-materialised, i.e. protection was still a silent no-op.
    """
    offsets = char_tokens(PACK_TEXT)          # one token per non-space char
    utt = Ent("string", "utterance", None, 0, len(PACK_TEXT), text=PACK_TEXT)
    # "ravo": begins inside the word "bravo", ends on a word boundary.
    cs = PACK_TEXT.index("bravo") + 1
    target = TokenIndex(offsets, len(offsets)).snap_char_range(cs, cs + 4)
    assert target is not None

    # Control: head truncation without protection drops the span entirely.
    builder.pack_protected = None
    builder.pack_fill = "head"
    control = builder.build_param_masks(
        _p("q", "string"), [utt], offsets, PACK_TEXT, max_cands=3
    )
    assert target not in {(c.tok_start, c.tok_end) for c in control["candidates"]}

    builder.pack_protected = {target}
    res = builder.build_param_masks(
        _p("q", "string"), [utt], offsets, PACK_TEXT, max_cands=3
    )
    kept = {(c.tok_start, c.tok_end) for c in res["candidates"]}
    assert target in kept, f"off-bound protected span evicted; kept={sorted(kept)}"
    starts = [c.tok_start for c in res["candidates"]]
    assert starts == sorted(starts), f"candidate order not monotone: {starts}"


def test_pack_without_protection_still_respects_the_cap(
    builder: CandidateMaskBuilder,
) -> None:
    """Production path: ``pack_protected`` is None -> no protection, cap honoured."""
    builder.pack_protected = None
    res = _pack_masks(builder, max_cands=2)
    assert res["num_candidates"] == 2


def test_free_utterance_generation_honours_the_stratified_policy(
    builder: CandidateMaskBuilder,
) -> None:
    """P2: ``pack_fill='stratified'`` must be applied at GENERATION time too.

    ``_pack`` implements the retention policy, but it can only choose among the
    candidates the generator materialised. The generator used to stop as soon as
    it had ``cap`` pairs, i.e. it was always head-first, so ``_pack`` only ever
    saw the surviving head prefix and could not spread the retained set. The
    resolver meanwhile reported ``cand_pack_fill='stratified'`` in its features:
    the telemetry named a policy that was not in force.

    Measured on ``hermes_eval`` (200 rows, 285 spans, cap 512): parity 0.0772
    with the head-first generator vs 0.2175 once the stride is real.
    """
    builder.pack_protected = None
    builder.pack_fill = "head"
    head = _pack_masks(builder, max_cands=4)
    head_starts = [c.tok_start for c in head["candidates"]]

    builder.pack_fill = "stratified"
    strat = _pack_masks(builder, max_cands=4)
    strat_starts = [c.tok_start for c in strat["candidates"]]

    assert strat["num_candidates"] == head["num_candidates"] == 4
    # Both stay monotone: document order is a contract, not a property of the policy.
    assert strat_starts == sorted(strat_starts)
    # And the policies must actually differ -- the regression was that they did not.
    assert strat_starts != head_starts, (
        "stratified retention produced the head prefix; the generation-time cap is "
        "overriding pack_fill again"
    )
    # Stratified spreads across the whole utterance, so it reaches past the head.
    assert max(strat_starts) > max(head_starts)


def test_stratified_generation_reaches_a_span_the_head_policy_cannot(
    builder: CandidateMaskBuilder,
) -> None:
    """The end-to-end payload of the P2 fix, on a single deterministic row.

    A late span is reachable under ``stratified`` and unreachable under ``head``
    at the same cap. This is the mechanism the parity number depends on, asserted
    directly so a future refactor of the enumeration cannot silently restore
    head-first truncation.

    The utterance is deliberately long (60 words). The stride samples
    ``int(i * n / rem)`` over the *pair* universe, and early starts contribute
    more pairs than late ones, so with a tiny utterance a cap of 3 legitimately
    lands in the first few starts. The property under test -- retention spans the
    document instead of its head -- only has teeth at a size comparable to the
    real rows (median 2.6k pairs per free-string parameter).
    """
    text = " ".join(f"w{i}" for i in range(60))
    offsets = word_tokens(text)
    utt = Ent("string", "utterance", None, 0, len(text), text=text)
    late_third = (2 * len(offsets)) // 3

    builder.pack_protected = None
    builder.pack_fill = "head"
    head = builder.build_param_masks(
        _p("q", "string"), [utt], offsets, text, max_cands=8
    )
    head_starts = [c.tok_start for c in head["candidates"]]
    assert max(head_starts) < late_third, (
        f"head fill unexpectedly reached the last third: {head_starts}"
    )

    builder.pack_fill = "stratified"
    strat = builder.build_param_masks(
        _p("q", "string"), [utt], offsets, text, max_cands=8
    )
    strat_starts = [c.tok_start for c in strat["candidates"]]
    assert max(strat_starts) >= late_third, (
        "the stratified policy must retain a span in the last third of the "
        f"utterance, the head prefix it replaces cannot: got {strat_starts}"
    )
    assert strat_starts == sorted(strat_starts)
    assert len(strat_starts) == 8


def test_retain_pairs_never_exceeds_the_cap_and_keeps_protected(
    builder: CandidateMaskBuilder,
) -> None:
    """``_retain_pairs`` contract: protected first, cap respected, no duplicates."""
    pairs = [(i, i) for i in range(20)]
    n_prot = 2                                       # first two are "protected"
    for fill in ("head", "stratified"):
        builder.pack_fill = fill
        kept = builder._retain_pairs(pairs, 5, n_prot)
        assert len(kept) == 5, f"{fill}: cap not honoured"
        assert len(set(kept)) == 5, f"{fill}: duplicate pairs retained"
        assert kept[:n_prot] == pairs[:n_prot], f"{fill}: protected pairs not first"
    # No cap -> the whole universe, unchanged order.
    builder.pack_fill = "stratified"
    assert builder._retain_pairs(pairs, None, n_prot) == pairs
    # The universe already fits -> no policy applies.
    assert builder._retain_pairs(pairs, 20, n_prot) == pairs


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