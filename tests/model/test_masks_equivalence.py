"""Equivalence tests for the ``masks.py`` performance rewrite.

The candidate builder was rewritten for speed along three axes:

1. ``_entry_tok_range`` (per entry, full linear token scan) -> ``TokenIndex``
   (bisect over token char starts/ends).
2. ``legend_entries.index(entry)`` (O(E) *whole-dataclass* comparison per
   candidate) -> ``EntryIndex`` (O(1) identity map).
3. The ``W=16`` windowed pair set was generated in full and then truncated to
   ``max_cands`` by the caller -> the cap is now passed down and the surplus
   pairs are never built.

Each of those is only acceptable if the output is *identical*, because
``spancall/lean/trainer.py`` re-reads the tensorized JSONL from disk. These
tests pin that: the reference implementations below are verbatim copies of the
pre-optimisation code, and every test asserts the new path reproduces them
exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import pytest

from spancall.data.schema import ParameterSchema
from spancall.model.masks import CandidateMaskBuilder, TokenIndex


# ---------------------------------------------------------------------------
# Reference (pre-optimisation) implementations
# ---------------------------------------------------------------------------

def ref_snap(
    cs: int, ce: int, token_offsets: Sequence[Tuple[int, int]], max_seq_len: int
) -> Optional[Tuple[int, int]]:
    """Original ``_entry_tok_range`` body: full linear scan, first/last overlap."""
    first = last = None
    for i, (tcs, tce) in enumerate(token_offsets):
        if i >= max_seq_len:
            break
        if tce <= cs or tcs >= ce:
            continue
        if first is None:
            first = i
        last = i
    if first is None:
        return None
    return (first, last)


# ---------------------------------------------------------------------------
# 1. TokenIndex.snap_char_range vs the linear scan
# ---------------------------------------------------------------------------

def _random_offsets(rng: random.Random, n: int, *, degenerate: bool) -> List[Tuple[int, int]]:
    """Contiguous token offsets, optionally with zero-width/degenerate tokens."""
    out: List[Tuple[int, int]] = []
    pos = 0
    for _ in range(n):
        width = rng.choice([0, 0, 1, 1, 2, 3, 5]) if degenerate else rng.choice([1, 1, 2, 3, 5])
        out.append((pos, pos + width))
        pos += width if degenerate else max(width, 1)
    return out


@pytest.mark.parametrize("seed", range(6))
def test_token_index_matches_linear_scan(seed: int) -> None:
    rng = random.Random(seed)
    degenerate = seed % 2 == 1
    checked = 0
    for _ in range(300):
        n = rng.randint(0, 40)
        offsets = _random_offsets(rng, n, degenerate=degenerate)
        total = offsets[-1][1] if offsets else 1
        # max_seq_len is randomly None-ish (n), exact, or a truncating cap
        cap = rng.choice([n, n, max(n - rng.randint(0, 3), 0), n + 5])
        tindex = TokenIndex(offsets, cap)
        for _ in range(12):
            cs = rng.randint(0, max(total, 1))
            ce = cs + rng.randint(0, 8)
            expected = ref_snap(cs, ce, offsets, cap)
            got = tindex.snap_char_range(cs, ce)
            assert got == expected, (
                f"snap mismatch cs={cs} ce={ce} cap={cap} offsets={offsets!r}: "
                f"expected {expected} got {got}"
            )
            checked += 1
    assert checked > 0


def test_token_index_ignores_degenerate_boundary_tokens() -> None:
    # Two zero-width tokens precede the real span; the linear scan also skipped
    # them (``tce <= cs``), so both must report token 2 onward.
    offsets = [(0, 0), (0, 0), (0, 4), (4, 9)]
    assert TokenIndex(offsets, 4).snap_char_range(0, 9) == (2, 3)
    assert ref_snap(0, 9, offsets, 4) == (2, 3)


def test_token_index_respects_max_seq_len() -> None:
    offsets = [(i, i + 1) for i in range(10)]
    # span reaches token 9 but the cap is 5 -> clamped, matching the scan's break
    assert TokenIndex(offsets, 5).snap_char_range(0, 10) == (0, 4)
    assert ref_snap(0, 10, offsets, 5) == (0, 4)


# ---------------------------------------------------------------------------
# 2. EntryIndex.of vs legend_entries.index
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ent:
    kind: str
    segment: str
    value: Any
    char_start: int
    char_end: int
    selectable: bool = True
    key: Optional[str] = None
    text: str = ""
    tool: Optional[str] = None


def test_entry_index_matches_list_index() -> None:
    from spancall.model.masks import EntryIndex

    entries = [
        Ent("string", "utterance", None, 0, 10, text="hello there"),
        Ent("string", "utterance", None, 11, 20, text="paris"),
        Ent("bool", "ctx_meta", True, 21, 25, text="true"),
        Ent("null", "ctx_meta", None, 26, 32, text="<|null|>"),
    ]
    eidx = EntryIndex(entries)
    for i, e in enumerate(entries):
        assert eidx.of(e) == i
        assert eidx.of(e) == entries.index(e)
    # duplicate-valued entries: ``list.index`` returns the FIRST match
    twin = Ent("string", "utterance", None, 11, 20, text="paris")
    assert eidx.of(twin) == entries.index(twin) == 1


def test_entry_index_absent_entry_is_minus_one() -> None:
    from spancall.model.masks import EntryIndex

    entries = [Ent("string", "utterance", None, 0, 5, text="abc")]
    missing = Ent("string", "utterance", None, 99, 102, text="zzz")
    assert EntryIndex(entries).of(missing) == -1


def test_entry_index_for_char_matches_linear_scan() -> None:
    from spancall.model.masks import EntryIndex

    def ref_for_char(entries: Sequence[Any], offset: int) -> int:
        for i, e in enumerate(entries):
            if e.char_start <= offset < e.char_end:
                return i
        return -1

    entries = [
        Ent("string", "utterance", None, 0, 10, text="aaaaaaaaaa"),
        Ent("string", "utterance", None, 10, 20, text="bbbbbbbbbb"),
        Ent("null", "ctx_meta", None, 20, 30, text="<|null|>"),
    ]
    eidx = EntryIndex(entries)
    for offset in range(-2, 34):
        assert eidx.for_char(offset) == ref_for_char(entries, offset), offset


# ---------------------------------------------------------------------------
# 3. max_cands cap applied during generation == truncation afterwards
# ---------------------------------------------------------------------------

def _words(n: int) -> Tuple[str, List[Tuple[int, int]]]:
    text = " ".join(f"w{i}" for i in range(n))
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for w in text.split(" "):
        offsets.append((pos, pos + len(w)))
        pos += len(w) + 1
    return text, offsets


def _p(name: str, ptype: str = "string", *, required: bool = True) -> ParameterSchema:
    return ParameterSchema(
        name=name, description="", param_type=ptype, required=required,
        enum=None, items_type=None,
    )


def test_capped_generation_matches_post_hoc_truncation() -> None:
    """The in-generator cap must reproduce ``masks["candidates"][:max_cands]``."""
    text, offsets = _words(40)
    utt = Ent("string", "utterance", None, 0, len(text), text=text)
    builder = CandidateMaskBuilder(max_seq_len=None, max_span_width=16)

    full = builder.build_param_masks(_p("q", "string"), [utt], offsets, text, max_cands=None)
    total = full["num_candidates"]
    assert total > 8, "need a candidate set large enough to exercise the cap"

    for cap in (0, 1, 2, 5, 8, 13, total - 1, total, total + 5):
        capped = builder.build_param_masks(
            _p("q", "string"), [utt], offsets, text, max_cands=cap
        )
        expect = full["candidates"][: max(0, cap)]
        assert capped["candidates"] == expect, f"cap={cap}"
        assert capped["num_candidates"] == len(expect), f"cap={cap}"
        assert capped["cand_index"] == [(c.tok_start, c.tok_end) for c in expect]
        assert capped["region"] == [c.region for c in expect]
        assert capped["canon_value"] == [c.canon_value for c in expect]


def test_shared_row_ctx_matches_per_param_rebuild() -> None:
    """A row-level ``_RowCtx`` must give byte-identical masks to rebuilding it."""
    from spancall.model.masks import _RowCtx

    text, offsets = _words(20)
    utt = Ent("string", "utterance", None, 0, len(text), text=text)
    recent = Ent("string", "recent", "w3", *offsets[3], text="w3=w3")
    entries = [utt, recent]
    builder = CandidateMaskBuilder(max_seq_len=None, max_span_width=16)

    row_ctx = _RowCtx.for_row(entries, offsets, len(offsets))
    for schema in (_p("q", "string"), _p("n", "integer"), _p("b", "boolean", required=False)):
        shared = builder.build_param_masks(schema, entries, offsets, text, ctx=row_ctx)
        fresh = builder.build_param_masks(schema, entries, offsets, text, ctx=None)
        assert shared["cand_index"] == fresh["cand_index"], schema.name
        assert shared["region"] == fresh["region"], schema.name
        assert shared["canon_value"] == fresh["canon_value"], schema.name
        assert shared["start_ok"] == fresh["start_ok"], schema.name
        assert shared["end_ok"] == fresh["end_ok"], schema.name


def test_stale_row_ctx_is_rejected_not_misused() -> None:
    """A ctx built for another row's tokenisation must be rebuilt, not reused."""
    from spancall.model.masks import _RowCtx

    text, offsets = _words(10)
    utt = Ent("string", "utterance", None, 0, len(text), text=text)
    builder = CandidateMaskBuilder(max_seq_len=None, max_span_width=16)

    stale = _RowCtx.for_row([utt], offsets, len(offsets))
    other_text, other_offsets = _words(14)
    other_utt = Ent("string", "utterance", None, 0, len(other_text), text=other_text)

    got = builder.build_param_masks(
        _p("q", "string"), [other_utt], other_offsets, other_text, ctx=stale
    )
    fresh = builder.build_param_masks(
        _p("q", "string"), [other_utt], other_offsets, other_text, ctx=None
    )
    assert got["cand_index"] == fresh["cand_index"]