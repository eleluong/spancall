"""Candidate masks and constraint filtering (P4) for SpanCall.

Ref: docs/mechanism.md §3.4, §5.2; docs/pipeline_io.md P4; full_implementation_plan §3.1.
Rules:
- R8: Context token snapping (pointers only to valid entry bounds)
- R9: Parameter-specific enum boundaries (prevent cross-enum bleeding)
- Type pre-filtering: restricts candidate spans by schema type

The builder is *deterministic* and torch-free: candidate generation operates on
the rendered-text legend entries and character/token offsets, so it can be unit
tested without the ``model`` extra. GPU representation helpers return plain
ints/lists; callers may wrap them in tensors.

Candidate record (pipeline_io.md P4)::

    Candidate { tok_start, tok_end,     # inclusive token range
                char_start, char_end,   # in the RENDERED text
                region,                 # utterance | recent | time | enum | bool | null
                canon_value,            # canonical value for equal-value aggregation
                entry_index }           # index into legend_entries (or -1 for regex spans)
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from spancall.data.schema import ParameterSchema

# ---------------------------------------------------------------------------
# Regex candidates (P4: integer/number/date/format pre-filtering)
# ---------------------------------------------------------------------------

# Numeric: optional sign, digit groups with separators, optional decimal, optional k/m/b suffix.
_NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?[kKmMbB]?")

# Strict date patterns only (P4): ISO (2026-10-24), dotted US (10/24/2026, 10-24-2026),
# and 24 Oct 2026 / October 24, 2026 forms.
_DATE_RE = re.compile(
    r"(?<!\d)"
    r"(?:(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])"
    r"|(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}"
    r"|(?:0?[1-9]|[12]\d|3[01])\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:19|20)\d{2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:0?[1-9]|[12]\d|3[01]),?\s+(?:19|20)\d{2})"
    r"(?!\d)"
)

# email / url / phone / id (P4 "format" params)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(r"https?://[^\s|]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


@dataclass(frozen=True)
class Candidate:
    """One valid candidate span for a parameter."""

    tok_start: int  # inclusive
    tok_end: int  # inclusive
    char_start: int  # in rendered text
    char_end: int  # in rendered text
    region: str  # utterance | recent | time | enum | bool | null
    canon_value: Any
    entry_index: int = -1  # index into legend_entries, or -1 for regex-derived spans


# ---------------------------------------------------------------------------
# Legend entry shape (duck-typed: any object with these attrs is accepted)
# ---------------------------------------------------------------------------

_ENTRY_ATTRS = ("kind", "segment", "value", "char_start", "char_end",
                "selectable", "key", "text", "tool")


def _entry_like(e: Any) -> bool:
    return all(hasattr(e, a) for a in _ENTRY_ATTRS)


class TokenIndex:
    """Per-row lookup structure over ``offset_mapping``.

    Why this exists: ``_entry_tok_range`` was a full linear scan of every token
    *per entry*, and ``build_param_masks`` re-snapped every targetable entry once
    **per parameter**. With K params, E entries and n tokens that is
    ``O(K · E · n)`` per row — at ``MAX_SEQ_LEN=4096`` this dominates row cost.
    The scan only ever needs the first and last token overlapping a char range,
    which two bisects over the sorted token char-starts answer directly.

    The structure is built once per row and is read-only afterwards, so it is
    safe to share across every parameter of that row.
    """

    __slots__ = ("offsets", "starts", "ends", "max_seq_len", "_n")

    def __init__(
        self,
        token_offsets: Sequence[Tuple[int, int]],
        max_seq_len: Optional[int] = None,
    ) -> None:
        self.offsets = token_offsets
        # Token char-starts are non-decreasing for every fast tokenizer, and
        # offset_mapping is ordered by token index, so this sort is a no-op but
        # guards the bisect invariant if a caller ever hands us something odd.
        self.starts = [tcs for tcs, _ in token_offsets]
        self.ends = [tce for _, tce in token_offsets]
        self._n = len(token_offsets)
        self.max_seq_len = self._n if max_seq_len is None else min(int(max_seq_len), self._n)

    def snap_char_range(self, cs: int, ce: int) -> Optional[Tuple[int, int]]:
        """First/last token index overlapping ``[cs, ce)``, or None.

        A token ``i`` overlaps when ``ends[i] > cs and starts[i] < ce`` — the
        exact predicate of the linear scan this replaces. Because ``starts`` and
        ``ends`` are both non-decreasing, the overlapping set is precisely the
        index interval ``[first, last]`` where ``first`` is the smallest index
        with ``ends[i] > cs`` and ``last`` the largest with ``starts[i] < ce``:

        * ``i >= first`` implies ``ends[i] >= ends[first] > cs``
        * ``i <= last`` implies ``starts[i] <= starts[last] < ce``

        so the two bisects reproduce the scan exactly — including for zero-width
        tokens (``starts[i] == ends[i]``, which do overlap an interior range) and
        for empty or inverted ranges (``ce <= cs``, which still overlap a token
        that strictly straddles them). Neither case needs special handling.
        """
        if self._n == 0:
            return None
        # Smallest i with ends[i] > cs. ends is non-decreasing.
        first = bisect_right(self.ends, cs)
        # Largest j with starts[j] < ce  ==  (bisect_left(starts, ce) - 1).
        last = bisect_left(self.starts, ce) - 1
        # Clamp to the snapping window and to real token bounds (the linear scan
        # simply stopped collecting past max_seq_len, which is this clamp).
        hi = self.max_seq_len - 1
        if first > hi:
            return None
        if last > hi:
            last = hi
        if first > last:
            return None
        return (first, last)

    def window(self, cs: int, ce: int, max_tok: Optional[int] = None) -> Tuple[int, int]:
        """``[lo, hi)`` token-index bounds of the tokens overlapping ``[cs, ce)``.

        Same predicate as :meth:`snap_char_range`, but returned as bounds so a
        caller can slice instead of scanning the whole row. ``hi`` is exclusive
        and already clamped to ``max_tok``. An empty range (``lo >= hi``) means
        no token overlaps.

        Why it matters: ``_free_recent_spans`` re-enumerated *every* token of the
        row once per recent entry — ``O(E · n)`` per parameter. Restricting to
        the tokens that actually overlap each entry's char range makes it
        ``O(entry_span)``, and that function was 96% of the row cost.
        """
        if self._n == 0:
            return (0, 0)
        hi_cap = self.max_seq_len if max_tok is None else min(int(max_tok), self.max_seq_len)
        lo = bisect_right(self.ends, cs)
        hi = bisect_left(self.starts, ce)
        if lo < 0:
            lo = 0
        if hi > hi_cap:
            hi = hi_cap
        if lo > hi_cap:
            lo = hi_cap
        return (lo, hi if hi > lo else lo)

    def snap_entry(self, entry: Any) -> Optional[Tuple[int, int]]:
        return self.snap_char_range(entry.char_start, entry.char_end)


def _entry_tok_range(
    entry: Any,
    token_offsets: Sequence[Tuple[int, int]],
    max_seq_len: int,
) -> Optional[Tuple[int, int]]:
    """R8 snapping: first/last token covering [char_start, char_end].

    Handles both exact-boundary tokenizers and byte-level BPE tokenizers (e.g.
    Qwen3) where the first token of an entry may include a leading space
    (``tcs < cs < tce``) or the last token may include trailing whitespace/quote.

    Kept as a thin adapter over :class:`TokenIndex` for callers that hold bare
    offsets; the per-row path reuses one ``TokenIndex`` instead.
    """
    return TokenIndex(token_offsets, max_seq_len).snap_entry(entry)


class EntryIndex:
    """Per-row legend lookup: entry object -> its index, and char -> owning entry.

    Why this exists: candidates carry ``entry_index``, and every candidate used
    to compute it with ``legend_entries.index(entry)`` — a linear scan that
    compares *whole dataclasses* (every field of every entry) and is therefore
    ``O(E · field_cost)`` **per candidate**. With up to ``max_cands`` candidates
    per parameter that was the second-largest per-row cost after snapping.

    ``identity`` lookup is what the call sites actually mean: the entries handed
    to ``build_param_masks`` are the very objects stored in the list, so an
    identity map is both exact and O(1). The dataclass-``==`` fallback is only
    consulted for a copy that is not present by identity, preserving the old
    semantics for callers that pass equals-but-not-identical entries.
    """

    __slots__ = ("entries", "_by_id", "_by_value", "_starts")

    def __init__(self, entries: Sequence[Any]) -> None:
        self.entries = list(entries)
        self._by_id: Dict[int, int] = {}
        self._starts: List[int] = []
        self._by_value: Optional[Dict[Any, int]] = None
        for i, e in enumerate(self.entries):
            self._by_id.setdefault(id(e), i)
            # ``char_start`` is non-decreasing in the rendered legend, which is
            # what makes ``for_char`` bisectable.
            self._starts.append(int(getattr(e, "char_start", 0)))

    def _value_map(self) -> Dict[Any, int]:
        # Built on first miss only: a dict keyed by an unhashable dataclass
        # would raise, and the common path never needs it.
        if self._by_value is None:
            self._by_value = {}
            for i, e in enumerate(self.entries):
                try:
                    self._by_value.setdefault(e, i)
                except TypeError:  # unhashable entry: no fallback possible
                    continue
        return self._by_value

    def of(self, entry: Any) -> int:
        """Index of ``entry`` in the legend, or -1 when it is not present."""
        i = self._by_id.get(id(entry))
        if i is not None:
            return i
        try:
            return self._value_map().get(entry, -1)
        except TypeError:
            return -1

    def for_char(self, char_offset: int) -> int:
        """Index of the entry owning ``char_offset`` (``[start, end)``), or -1."""
        # Largest i with starts[i] <= char_offset.
        i = bisect_right(self._starts, char_offset) - 1
        while i >= 0:
            e = self.entries[i]
            if e.char_start <= char_offset < e.char_end:
                return i
            # Walk back over entries that start at the same offset or earlier
            # but do not contain it; bounded in practice (legend entries nest
            # at most a few levels deep).
            if i > 0 and self._starts[i - 1] == self._starts[i]:
                i -= 1
                continue
            return -1
        return -1


class _RowCtx:
    """Per-row work-sharing bundle threaded through ``build_param_masks``.

    Carries the two O(1) indexes plus the entry->token snap cache, so the
    ``K`` parameter passes of a single row reuse one another's work instead of
    each rebuilding it (the previous behaviour: K full linear re-snaps of every
    legend entry).
    """

    __slots__ = ("tindex", "eidx", "snaps", "max_tok", "offsets")

    def __init__(
        self,
        token_index: TokenIndex,
        entry_index: EntryIndex,
        max_tok: int,
        offsets: Any = None,
    ) -> None:
        self.tindex = token_index
        self.eidx = entry_index
        self.max_tok = max_tok
        # Kept only as an identity token so a caller-supplied context can be
        # rejected when it belongs to a different row's tokenization.
        self.offsets = offsets
        self.snaps: Dict[int, Optional[Tuple[int, int]]] = {}

    @classmethod
    def for_row(
        cls,
        legend_entries: Sequence[Any],
        token_offsets: Sequence[Tuple[int, int]],
        max_tok: int,
    ) -> "_RowCtx":
        """Build the shared indexes for one row once, for reuse by every param."""
        return cls(
            TokenIndex(token_offsets, max_tok),
            EntryIndex(legend_entries),
            max_tok,
            token_offsets,
        )

    def matches(self, token_offsets: Any, max_tok: int) -> bool:
        """True when this context was built for exactly this row's tokenization."""
        return self.offsets is token_offsets and self.max_tok == max_tok

    def snap(self, entry: Any) -> Optional[Tuple[int, int]]:
        """Entry token span, memoised by entry IDENTITY for this row."""
        key = id(entry)
        if key in self.snaps:
            return self.snaps[key]
        r = self.tindex.snap_entry(entry)
        self.snaps[key] = r
        return r


def _normalize_param_type(raw_type: Optional[str]) -> Tuple[str, bool]:
    """Normalize JSON-schema parameter type (including union strings like ``"['string', 'null']"``).

    Returns ``(primary_type, is_nullable)``.
    """
    raw = (raw_type or "string").strip().lower()
    is_nullable = "null" in raw
    if raw.startswith("["):
        for candidate in ("boolean", "integer", "number", "date", "datetime", "time", "array", "object", "string"):
            if candidate in raw:
                return candidate, is_nullable
    return raw, is_nullable


class CandidateMaskBuilder:
    """Build deterministic, type-filtered candidate sets per parameter (P4)."""

    def __init__(self, max_seq_len: Optional[int] = None, max_span_width: int = 16) -> None:
        # ``None`` (default) derives the window from the *actual* token count at
        # build time — there is no hardcoded 512-token cap (long-context fix).
        # ``_effective_max_seq_len`` resolves None -> len(token_offsets).
        self.max_seq_len = max_seq_len
        self.max_span_width = max_span_width

    def _effective_max_seq_len(self, token_offsets: Sequence[Tuple[int, int]]) -> int:
        """Window actually used for snapping: explicit cap or the real token count."""
        return self.max_seq_len if self.max_seq_len is not None else len(token_offsets)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_param_masks(
        self,
        param_schema: ParameterSchema,
        legend_entries: List[Any],
        token_offsets: List[Tuple[int, int]],
        utterance_text: str,
        ctx: Optional["_RowCtx"] = None,
        max_cands: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Construct the candidate set A_k for one parameter.

        Args:
            param_schema: the parameter's JSON-schema-derived definition.
            legend_entries: parsed legend entries (``parser.Entry``-like).
            token_offsets: ``offset_mapping`` from tokenization
                (list of ``(char_start, char_end)``, one per token).
            utterance_text: the raw user utterance text (for regex spans).
            ctx: pre-built :class:`_RowCtx` for this row. Supplying it is what
                makes the ``K``-parameter loop cheap: the per-entry token snap,
                the entry→index map and the entry char index are then computed
                once per row instead of once per parameter. Silently rebuilt if
                it was made for a different tokenization, so a stale context can
                never mis-snap.
            max_cands: stop generating once this many candidates exist. The cap
                used to be applied *after* ``_free_utterance_spans`` had already
                enumerated the whole ``W=16`` windowed pair set, so it saved no
                work at all; capping here means the surplus pairs are never
                built. The final list is still clamped to this length, which
                keeps the output byte-identical to the old post-hoc truncation.

        Returns a dict with:
            candidates: list[Candidate] — the ordered A_k (abstain ∅ excluded;
                the abstain slot is always unmasked and handled by the head).
            cand_index: list[Tuple[int, int]] — packed (tok_start, tok_end).
            start_ok / end_ok: boolean masks over tokens (windowed form) for
                utterance spans, else None. Note these are *legacy* conveniences;
                the primary form is ``zip(candidates, cand_index)``.
            region: list[str] region tags aligned to candidates.
            canon_value: list aligned to candidates.
        """
        max_tok = self._effective_max_seq_len(token_offsets)
        if ctx is None or not ctx.matches(token_offsets, max_tok):
            ctx = _RowCtx.for_row(legend_entries, token_offsets, max_tok)
        eidx = ctx.eidx

        cands: List[Candidate] = []
        ptype, _is_nullable = _normalize_param_type(param_schema.param_type)
        enum_vals = set(param_schema.enum) if param_schema.enum else None

        # --- split entries by what they can feed -----------------------------
        targetable = [e for e in legend_entries if e.selectable]
        ctx_time = [e for e in targetable if e.segment == "ctx_time"]
        ctx_enum = [e for e in targetable if e.segment == "ctx_enum"]
        ctx_bool = [e for e in targetable if e.segment == "ctx_meta"
                    and str(e.kind).lower() == "bool"]
        ctx_null = [e for e in targetable if e.segment == "ctx_meta"
                    and str(e.kind).lower() in ("null", "unresolved")]
        recent_vals = [e for e in targetable if e.segment == "recent"]
        utterance_entries = [e for e in targetable if e.segment == "utterance"]
        u_bounds: Optional[Tuple[int, int]] = (
            (min(e.char_start for e in utterance_entries), max(e.char_end for e in utterance_entries))
            if utterance_entries
            else None
        )

        # Entry -> token span, resolved ONCE per row through ``ctx`` and shared
        # by every parameter. Previously each of the K params re-ran a full
        # linear token scan for every targetable entry: O(K·E·n) per row.
        tok_by_entry: Dict[int, Tuple[int, int]] = {}
        for e in targetable:
            r = ctx.snap(e)
            if r is not None:
                tok_by_entry[id(e)] = r

        # --- boolean ---------------------------------------------------------
        if ptype == "boolean":
            for e in ctx_bool:
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "bool", str(e.value).lower(), eidx.of(e)))
            cands.extend(self._null_candidates(ctx_null, eidx, tok_by_entry, param_schema))
            return self._pack(cands, max_cands)

        # --- enum (R9: strict membership) ------------------------------------
        if enum_vals is not None and ptype in ("string", "integer", "number"):
            for e in ctx_enum:
                if e.value not in enum_vals:
                    continue
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "enum", e.value, eidx.of(e)))
            cands.extend(self._null_candidates(ctx_null, eidx, tok_by_entry, param_schema))
            return self._pack(cands, max_cands)

        # --- date / datetime -------------------------------------------------
        if ptype in ("date", "datetime", "time"):
            for e in ctx_time:
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "time", e.value, eidx.of(e)))
            cands.extend(self._regex_candidates(
                _DATE_RE, utterance_text, token_offsets, "utterance", "time",
                char_bounds=u_bounds, max_tok=max_tok))
            # date recent values (whole leaf entries + regex sub-spans in multi-word recent rows)
            for e in recent_vals:
                if isinstance(e.value, str) and _DATE_RE.fullmatch(e.value.strip()):
                    r = tok_by_entry.get(id(e))
                    if r is not None:
                        cands.append(Candidate(
                            r[0], r[1], e.char_start, e.char_end, "recent", e.value.strip(), eidx.of(e)))
            cands.extend(self._recent_regex_candidates(
                _DATE_RE, recent_vals, utterance_text, token_offsets, eidx,
                existing=cands, max_tok=max_tok))
            cands.extend(self._null_candidates(ctx_null, eidx, tok_by_entry, param_schema))
            return self._pack(cands, max_cands)

        # --- numeric ---------------------------------------------------------
        if ptype in ("integer", "number"):
            cands.extend(self._regex_candidates(
                _NUMERIC_RE, utterance_text, token_offsets, "utterance", "numeric",
                char_bounds=u_bounds, max_tok=max_tok))
            # numeric recent value interiors (whole leaf entries + regex sub-spans in multi-word recent rows)
            for e in recent_vals:
                if not self._looks_numeric(e.value):
                    continue
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "recent", e.value, eidx.of(e)))
            cands.extend(self._recent_regex_candidates(
                _NUMERIC_RE, recent_vals, utterance_text, token_offsets, eidx,
                existing=cands, max_tok=max_tok))
            cands.extend(self._null_candidates(ctx_null, eidx, tok_by_entry, param_schema))
            return self._pack(cands, max_cands)

        # --- format params (email/url/phone/id) ------------------------------
        if ptype in ("email", "url", "phone", "id", "identifier", "uuid"):
            regex = {"email": _EMAIL_RE, "url": _URL_RE,
                     "phone": _PHONE_RE}.get(ptype, _EMAIL_RE)
            cands.extend(self._regex_candidates(
                regex, utterance_text, token_offsets, "utterance", ptype,
                char_bounds=u_bounds, max_tok=max_tok))
            # recent value interiors + regex sub-spans inside multi-word recent rows
            for e in recent_vals:
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "recent", e.value, eidx.of(e)))
            cands.extend(self._recent_regex_candidates(
                regex, recent_vals, utterance_text, token_offsets, eidx,
                existing=cands, max_tok=max_tok))
            cands.extend(self._null_candidates(ctx_null, eidx, tok_by_entry, param_schema))
            return self._pack(cands, max_cands)

        # --- free string (windowed utterance spans ∪ recent interiors & sub-spans) ---
        recent_cands: List[Candidate] = []
        for e in recent_vals:
            r = tok_by_entry.get(id(e))
            if r is None:
                continue
            recent_cands.append(Candidate(
                r[0], r[1], e.char_start, e.char_end, "recent", e.value, eidx.of(e)))
        free_recent = self._free_recent_spans(
            recent_vals, token_offsets, eidx, utterance_text, existing=recent_cands,
            max_tok=max_tok, tindex=ctx.tindex)
        null_cands = self._null_candidates(ctx_null, eidx, tok_by_entry, param_schema)

        # Reserved tail for structured context entries (recent + null) so that
        # capping free utterance spans never evicts these high-priority candidates.
        reserved_tail = len(recent_cands) + len(free_recent) + len(null_cands)
        remain = None if max_cands is None else max(0, int(max_cands) - len(cands) - reserved_tail)

        cands.extend(self._free_utterance_spans(
            utterance_entries, tok_by_entry, token_offsets, eidx, utterance_text,
            max_tok=max_tok, cap=remain, tindex=ctx.tindex))
        cands.extend(recent_cands)
        cands.extend(free_recent)
        cands.extend(null_cands)
        return self._pack(cands, max_cands)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pack(self, cands: List[Candidate], max_cands: Optional[int] = None) -> Dict[str, Any]:
        # ``max_cands`` is re-applied here as the single authoritative clamp.
        # If truncation is required, preserve the <|null|> slot (if present)
        # by drawing truncation budget from non-null candidates first.
        if max_cands is not None and len(cands) > max_cands:
            cap = max(0, int(max_cands))
            null_cands = [c for c in cands if getattr(c, "region", "") == "null"]
            other_cands = [c for c in cands if getattr(c, "region", "") != "null"]
            cands = other_cands[: max(0, cap - len(null_cands))] + null_cands
        # window for the legacy start_ok/end_ok convenience masks: the largest
        # token index among all candidates (never a hardcoded 512).
        seq_len = max((c.tok_end for c in cands), default=-1) + 1
        start_ok: List[bool] = [False] * seq_len
        end_ok: List[bool] = [False] * seq_len
        for c in cands:
            if c.region == "utterance":
                start_ok[c.tok_start] = True
                end_ok[c.tok_end] = True
        return {
            "candidates": cands,
            "cand_index": [(c.tok_start, c.tok_end) for c in cands],
            "start_ok": start_ok,
            "end_ok": end_ok,
            "region": [c.region for c in cands],
            "canon_value": [c.canon_value for c in cands],
            "max_span_width": self.max_span_width,
            "num_candidates": len(cands),
        }

    def _null_candidates(
        self,
        ctx_null: List[Any],
        eidx: EntryIndex,
        tok_by_entry: Dict[int, Tuple[int, int]],
        param_schema: ParameterSchema,
    ) -> List[Candidate]:
        """Optional or nullable params get the <|null|> entry appended (plan §3.1, P4)."""
        _, is_nullable = _normalize_param_type(param_schema.param_type)
        if param_schema.required and not is_nullable:
            return []
        out: List[Candidate] = []
        for e in ctx_null:
            r = tok_by_entry.get(id(e))
            if r is None:
                continue
            out.append(Candidate(
                r[0], r[1], e.char_start, e.char_end, "null", None, eidx.of(e)))
        return out

    def _regex_candidates(
        self,
        regex: re.Pattern,
        text: str,
        token_offsets: Sequence[Tuple[int, int]],
        region: str,
        canon_prefix: str,
        char_bounds: Optional[Tuple[int, int]] = None,
        max_tok: Optional[int] = None,
    ) -> List[Candidate]:
        """Regex matches → char ranges → (start,end) token pairs (P4)."""
        out: List[Candidate] = []
        if not text:
            return out
        if max_tok is None:
            max_tok = self._effective_max_seq_len(token_offsets)
        s_lo = char_bounds[0] if char_bounds is not None else 0
        s_hi = min(char_bounds[1], len(text)) if char_bounds is not None else len(text)
        if s_hi <= s_lo:
            return out
        for m in regex.finditer(text, s_lo, s_hi):
            cs, ce = m.start(), m.end()
            r = _char_range_to_tokens(cs, ce, token_offsets, max_tok)
            if r is None:
                continue
            value = text[cs:ce]
            out.append(Candidate(
                r[0], r[1], cs, ce, region, value, -1))
        return out

    def _recent_regex_candidates(
        self,
        regex: re.Pattern,
        recent_entries: List[Any],
        rendered_text: str,
        token_offsets: Sequence[Tuple[int, int]],
        eidx: EntryIndex,
        existing: Optional[Sequence[Candidate]] = None,
        max_tok: Optional[int] = None,
    ) -> List[Candidate]:
        """Regex sub-spans inside recent_context rows (e.g. numbers/dates/emails in conversation rows)."""
        out: List[Candidate] = []
        seen = {(c.tok_start, c.tok_end, c.region) for c in (existing or ())}
        if max_tok is None:
            max_tok = self._effective_max_seq_len(token_offsets)
        for e in recent_entries:
            if e.char_end <= e.char_start:
                continue
            entry_slice = (
                rendered_text[e.char_start:e.char_end]
                if len(rendered_text) >= e.char_end
                else str(e.value or "")
            )
            if not entry_slice:
                continue
            e_idx = eidx.of(e)
            for m in regex.finditer(entry_slice):
                cs = e.char_start + m.start()
                ce = e.char_start + m.end()
                r = _char_range_to_tokens(cs, ce, token_offsets, max_tok)
                if r is None:
                    continue
                key = (r[0], r[1], "recent")
                if key in seen:
                    continue
                seen.add(key)
                out.append(Candidate(
                    r[0], r[1], cs, ce, "recent", m.group(0), e_idx))
        return out

    def _free_utterance_spans(
        self,
        utterance_entries: List[Any],
        tok_by_entry: Dict[int, Tuple[int, int]],
        token_offsets: Sequence[Tuple[int, int]],
        eidx: EntryIndex,
        utterance_text: str = "",
        max_tok: Optional[int] = None,
        cap: Optional[int] = None,
        tindex: Optional[TokenIndex] = None,
    ) -> List[Candidate]:
        """All token pairs wholly inside the utterance, windowed by W_max (W_max=16).

        Only word-aligned boundaries are emitted (P4: "word-aligned"), so we keep
        tokens whose char range starts at a word boundary in text. This mirrors
        ReverseAligner's ``_word_bounded_matches`` behavior for consistency.

        ``cap`` bounds how many candidates this function may produce. The
        windowed enumeration is O(n·W) and the caller always truncated the
        result to ``max_cands`` afterwards; passing that headroom down means the
        surplus pairs are never materialised at all.
        """
        out: List[Candidate] = []
        if not utterance_entries or cap == 0:
            return out
        # build the full utterance char range from entry extremes
        u_start = min(e.char_start for e in utterance_entries)
        u_end = max(e.char_end for e in utterance_entries)
        # word boundary positions: _entry_text joins entries with ' ', producing
        # offsets relative to the joined string — translate them into the
        # *rendered* (absolute) coordinate space used by token_offsets.
        text = self._entry_text(utterance_entries)
        word_bounds = {u_start + b for b in _word_boundaries(text)}

        # tokens overlapping utterance range (clamped to [u_start, u_end] for BPE
        # tokens that include leading space or trailing newline)
        if max_tok is None:
            max_tok = self._effective_max_seq_len(token_offsets)
        tindex = tindex or TokenIndex(token_offsets, max_tok)
        # Only the tokens overlapping the utterance range, found by bisect rather
        # than by scanning the row (same overlap predicate, so same pairs).
        lo, hi = tindex.window(u_start, u_end, max_tok)
        in_toks = [
            (i, (max(token_offsets[i][0], u_start), min(token_offsets[i][1], u_end)))
            for i in range(lo, hi)
            if max(token_offsets[i][0], u_start) < min(token_offsets[i][1], u_end)
        ]
        starts = [i for i, (tcs, tce) in in_toks if tcs in word_bounds or tcs == 0]
        tok_map = dict(in_toks)
        cand_set: Dict[Tuple[int, int], Candidate] = {}
        # O(n·W) instead of O(n²): a candidate end j only counts within
        # max_span_width of the start i, so scan the bounded window [i, i+W).
        # in_toks is a contiguous subset of token indices — dict lookup is O(1).
        full = cap is None
        for i in starts:
            if not full and len(cand_set) >= cap:
                break
            tcs_i, _ = tok_map[i]
            for j in range(i, min(i + self.max_span_width, max_tok)):
                if j not in tok_map:
                    continue
                tce_j = tok_map[j][1]
                if tce_j in word_bounds or tce_j == u_end:
                    cs_i, ce_j = tcs_i, tce_j
                    cval: Optional[str] = None
                    if len(utterance_text) >= ce_j:
                        while cs_i < ce_j and utterance_text[cs_i].isspace():
                            cs_i += 1
                        while ce_j > cs_i and utterance_text[ce_j - 1].isspace():
                            ce_j -= 1
                        if cs_i >= ce_j:
                            continue
                        cval = utterance_text[cs_i:ce_j]
                    cand = Candidate(
                        i, j, cs_i, ce_j, "utterance",
                        cval, eidx.for_char(cs_i))
                    cand_set[(i, j)] = cand
                    if not full and len(cand_set) >= cap:
                        return list(cand_set.values())
        # also add single-token spans not starting on a word bound (whole tokens)
        # (only when the cap still has room — this pass only ever *adds* spans
        # that the loop above skipped, so it cannot reorder existing output)
        if full or len(cand_set) < cap:
            for i, (tcs, tce) in in_toks:
                if not full and len(cand_set) >= cap:
                    break
                if (i, i) not in cand_set:
                    cs_i, ce_j = tcs, tce
                    cval = None
                    if len(utterance_text) >= ce_j:
                        while cs_i < ce_j and utterance_text[cs_i].isspace():
                            cs_i += 1
                        while ce_j > cs_i and utterance_text[ce_j - 1].isspace():
                            ce_j -= 1
                        if cs_i >= ce_j:
                            continue
                        cval = utterance_text[cs_i:ce_j]
                    cand_set[(i, i)] = Candidate(
                        i, i, cs_i, ce_j, "utterance", cval, eidx.for_char(cs_i))
        return list(cand_set.values())

    def _free_recent_spans(
        self,
        recent_entries: List[Any],
        token_offsets: Sequence[Tuple[int, int]],
        eidx: EntryIndex,
        rendered_text: str = "",
        existing: Optional[Sequence[Candidate]] = None,
        max_tok: Optional[int] = None,
        tindex: Optional[TokenIndex] = None,
    ) -> List[Candidate]:
        """Word-aligned sub-span candidates inside multi-word recent_context entries."""
        out: List[Candidate] = []
        seen = {(c.tok_start, c.tok_end, c.region) for c in (existing or ())}
        if max_tok is None:
            max_tok = self._effective_max_seq_len(token_offsets)
        tindex = tindex or TokenIndex(token_offsets, max_tok)
        for e in recent_entries:
            if e.char_end <= e.char_start:
                continue
            entry_slice = (
                rendered_text[e.char_start:e.char_end]
                if len(rendered_text) >= e.char_end
                else str(e.value or "")
            )
            if " " not in entry_slice.strip():
                continue
            e_idx = eidx.of(e)
            word_bounds = {e.char_start + b for b in _word_boundaries(entry_slice)}
            # Only the tokens overlapping this entry's char range — the old code
            # enumerated every token of the row once per entry (O(E·n)); the
            # bisect window makes it O(span). The overlap predicate is identical
            # (``tce > char_start and tcs < char_end``), so the pairs are the same.
            lo, hi = tindex.window(e.char_start, e.char_end, max_tok)
            in_toks = [
                (i, (max(token_offsets[i][0], e.char_start), min(token_offsets[i][1], e.char_end)))
                for i in range(lo, hi)
                if max(token_offsets[i][0], e.char_start) < min(token_offsets[i][1], e.char_end)
            ]
            tok_map = dict(in_toks)
            starts = [i for i, (tcs, tce) in in_toks if tcs in word_bounds or tcs == e.char_start]
            # O(n·W) bounded-window scan (mirror of _free_utterance_spans)
            for i in starts:
                tcs_i, _ = tok_map[i]
                for j in range(i, min(i + self.max_span_width, max_tok)):
                    if j not in tok_map:
                        continue
                    tce_j = tok_map[j][1]
                    if tce_j in word_bounds or tce_j == e.char_end:
                        key = (i, j, "recent")
                        if key in seen:
                            continue
                        cs_i, ce_j = tcs_i, tce_j
                        if len(rendered_text) >= ce_j:
                            while cs_i < ce_j and rendered_text[cs_i].isspace():
                                cs_i += 1
                            while ce_j > cs_i and rendered_text[ce_j - 1].isspace():
                                ce_j -= 1
                            sub_val = rendered_text[cs_i:ce_j]
                        else:
                            sub_val = entry_slice[cs_i - e.char_start:ce_j - e.char_start].strip()
                        if not sub_val:
                            continue
                        seen.add(key)
                        out.append(Candidate(
                            i, j, cs_i, ce_j, "recent", sub_val, e_idx))
        return out

    @staticmethod
    def _entry_text(entries: Sequence[Any]) -> str:
        if not entries:
            return ""
        return " ".join(e.text or "" for e in entries)

    @staticmethod
    def _looks_numeric(value: Any) -> bool:
        if value is None:
            return False
        s = str(value).strip()
        return bool(_NUMERIC_RE.fullmatch(s))

    @staticmethod
    def register_marker_tokens(tokenizer: Any) -> None:
        """Register Qwen-native marker tokens as additional special tokens.

        Uses ``span_template.sanitize.SPECIAL_MARKER_TOKENS`` — the atomic
        ``<|...|>`` forms ``<|cls|>``, ``<|tool|>``, ``<|param|>``,
        ``<|ctx_meta|>`` and the P6 reserved delimiters
        ``<|bar|>``/``<|eq|>``/``<|q|>`` — so BPE tokenizers (Qwen3, 32k
        window) keep each structural marker a single token instead of splitting
        bracket/angle halves (docs/pipeline_io.md P6.3, long-context R8
        validation gate). Requires a tokenizer supporting ``add_special_tokens``
        (transformers AutoTokenizer).
        """
        from span_template.sanitize import SPECIAL_MARKER_TOKENS  # local import: optional dep

        special = list(getattr(tokenizer, "additional_special_tokens", None) or [])
        existing = set(special)
        missing = [t for t in SPECIAL_MARKER_TOKENS if t not in existing]
        # also ensure they're in all_special_tokens (idempotent)
        if missing:
            tokenizer.add_special_tokens({"additional_special_tokens": missing})


def _word_boundaries(text: str) -> set:
    """Char offsets at word / non-word transitions in ``text`` (consistent with ReverseAligner)."""
    bounds = {0, len(text)}
    for i, ch in enumerate(text):
        if not (ch.isalnum() or ch == "_"):
            bounds.add(i)
            bounds.add(i + 1)
    return bounds


def _char_range_to_tokens(
    cs: int, ce: int, token_offsets: Sequence[Tuple[int, int]], max_seq_len: Optional[int] = None
) -> Optional[Tuple[int, int]]:
    """Map an absolute char range to the (start,end) token pair covering it.

    ``max_seq_len=None`` means no cap (the full ``token_offsets`` window).
    """
    first = last = None
    for i, (tcs, tce) in enumerate(token_offsets):
        if max_seq_len is not None and i >= max_seq_len:
            break
        # token overlaps the char range
        if tce > cs and tcs < ce:
            if first is None:
                first = i
            last = i
    if first is None:
        return None
    return (first, last)


# NOTE: the old module-level ``_entry_index`` (``legend_entries.index(entry)``,
# O(E) whole-dataclass equality per candidate) and ``_entry_index_for_char``
# (O(E) linear scan) were removed deliberately. Both are superseded by
# :class:`EntryIndex`, which resolves the same answers in O(1) / O(log E) from a
# per-row index. Do not reintroduce them: candidate ``entry_index`` is written
# for every candidate, so an O(E) lookup there is quadratic in the legend size.
