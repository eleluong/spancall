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


def _entry_tok_range(
    entry: Any,
    token_offsets: Sequence[Tuple[int, int]],
    max_seq_len: int,
) -> Optional[Tuple[int, int]]:
    """R8 snapping: first/last token wholly inside [char_start, char_end].

    Returns None when the entry has no fully-inside token (should not happen
    for legend entries; the template boundary check guarantees alignment).
    """
    cs, ce = entry.char_start, entry.char_end
    first = last = None
    for i, (tcs, tce) in enumerate(token_offsets):
        if i >= max_seq_len:
            break
        if tcs >= cs and tce <= ce:
            if first is None:
                first = i
            last = i
    if first is None:
        return None
    return (first, last)


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
    ) -> Dict[str, Any]:
        """Construct the candidate set A_k for one parameter.

        Args:
            param_schema: the parameter's JSON-schema-derived definition.
            legend_entries: parsed legend entries (``parser.Entry``-like).
            token_offsets: ``offset_mapping`` from tokenization
                (list of ``(char_start, char_end)``, one per token).
            utterance_text: the raw user utterance text (for regex spans).

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
        cands: List[Candidate] = []
        ptype = (param_schema.param_type or "string").lower()
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

        # token offsets grouped per entry for utterance windowing
        tok_by_entry: Dict[int, Tuple[int, int]] = {}

        def snap(e: Any) -> Optional[Tuple[int, int]]:
            r = _entry_tok_range(e, token_offsets, self._effective_max_seq_len(token_offsets))
            if r is not None:
                tok_by_entry[id(e)] = r
            return r

        for e in targetable:
            snap(e)

        # --- boolean ---------------------------------------------------------
        if ptype == "boolean":
            for e in ctx_bool:
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "bool", str(e.value).lower(), _entry_index(legend_entries, e)))
            return self._pack(cands)

        # --- enum (R9: strict membership) ------------------------------------
        if enum_vals is not None and ptype in ("string", "integer", "number"):
            for e in ctx_enum:
                if e.value not in enum_vals:
                    continue
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "enum", e.value, _entry_index(legend_entries, e)))
            cands.extend(self._null_candidates(ctx_null, legend_entries, tok_by_entry, param_schema))
            return self._pack(cands)

        # --- date / datetime -------------------------------------------------
        if ptype in ("date", "datetime", "time"):
            for e in ctx_time:
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "time", e.value, _entry_index(legend_entries, e)))
            cands.extend(self._regex_candidates(
                _DATE_RE, utterance_text, token_offsets, "utterance", "time"))
            # date recent values (whole leaf entries + regex sub-spans in multi-word recent rows)
            for e in recent_vals:
                if isinstance(e.value, str) and _DATE_RE.fullmatch(e.value.strip()):
                    r = tok_by_entry.get(id(e))
                    if r is not None:
                        cands.append(Candidate(
                            r[0], r[1], e.char_start, e.char_end, "recent", e.value.strip(), _entry_index(legend_entries, e)))
            cands.extend(self._recent_regex_candidates(
                _DATE_RE, recent_vals, utterance_text, token_offsets, legend_entries, existing=cands))
            cands.extend(self._null_candidates(ctx_null, legend_entries, tok_by_entry, param_schema))
            return self._pack(cands)

        # --- numeric ---------------------------------------------------------
        if ptype in ("integer", "number"):
            cands.extend(self._regex_candidates(
                _NUMERIC_RE, utterance_text, token_offsets, "utterance", "numeric"))
            # numeric recent value interiors (whole leaf entries + regex sub-spans in multi-word recent rows)
            for e in recent_vals:
                if not self._looks_numeric(e.value):
                    continue
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "recent", e.value, _entry_index(legend_entries, e)))
            cands.extend(self._recent_regex_candidates(
                _NUMERIC_RE, recent_vals, utterance_text, token_offsets, legend_entries, existing=cands))
            cands.extend(self._null_candidates(ctx_null, legend_entries, tok_by_entry, param_schema))
            return self._pack(cands)

        # --- format params (email/url/phone/id) ------------------------------
        if ptype in ("email", "url", "phone", "id", "identifier", "uuid"):
            regex = {"email": _EMAIL_RE, "url": _URL_RE,
                     "phone": _PHONE_RE}.get(ptype, _EMAIL_RE)
            cands.extend(self._regex_candidates(
                regex, utterance_text, token_offsets, "utterance", ptype))
            # recent value interiors + regex sub-spans inside multi-word recent rows
            for e in recent_vals:
                r = tok_by_entry.get(id(e))
                if r is None:
                    continue
                cands.append(Candidate(
                    r[0], r[1], e.char_start, e.char_end, "recent", e.value, _entry_index(legend_entries, e)))
            cands.extend(self._recent_regex_candidates(
                regex, recent_vals, utterance_text, token_offsets, legend_entries, existing=cands))
            cands.extend(self._null_candidates(ctx_null, legend_entries, tok_by_entry, param_schema))
            return self._pack(cands)

        # --- free string (windowed utterance spans ∪ recent interiors & sub-spans) ---
        cands.extend(self._free_utterance_spans(
            utterance_entries, tok_by_entry, token_offsets, legend_entries, utterance_text))
        for e in recent_vals:
            r = tok_by_entry.get(id(e))
            if r is None:
                continue
            cands.append(Candidate(
                r[0], r[1], e.char_start, e.char_end, "recent", e.value, _entry_index(legend_entries, e)))
        cands.extend(self._free_recent_spans(
            recent_vals, token_offsets, legend_entries, utterance_text, existing=cands))
        cands.extend(self._null_candidates(ctx_null, legend_entries, tok_by_entry, param_schema))
        return self._pack(cands)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pack(self, cands: List[Candidate]) -> Dict[str, Any]:
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
        legend_entries: List[Any],
        tok_by_entry: Dict[int, Tuple[int, int]],
        param_schema: ParameterSchema,
    ) -> List[Candidate]:
        """Optional params get the <|null|> entry appended (plan §3.1, P4)."""
        if param_schema.required:
            return []
        out: List[Candidate] = []
        for e in ctx_null:
            r = tok_by_entry.get(id(e))
            if r is None:
                continue
            out.append(Candidate(
                r[0], r[1], e.char_start, e.char_end, "null", None, _entry_index(legend_entries, e)))
        return out

    def _regex_candidates(
        self,
        regex: re.Pattern,
        text: str,
        token_offsets: Sequence[Tuple[int, int]],
        region: str,
        canon_prefix: str,
    ) -> List[Candidate]:
        """Regex matches → char ranges → (start,end) token pairs (P4)."""
        out: List[Candidate] = []
        if not text:
            return out
        for m in regex.finditer(text):
            cs, ce = m.start(), m.end()
            r = _char_range_to_tokens(cs, ce, token_offsets, self._effective_max_seq_len(token_offsets))
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
        legend_entries: List[Any],
        existing: Optional[Sequence[Candidate]] = None,
    ) -> List[Candidate]:
        """Regex sub-spans inside recent_context rows (e.g. numbers/dates/emails in conversation rows)."""
        out: List[Candidate] = []
        seen = {(c.tok_start, c.tok_end, c.region) for c in (existing or ())}
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
            e_idx = _entry_index(legend_entries, e)
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
        legend_entries: List[Any],
        utterance_text: str = "",
    ) -> List[Candidate]:
        """All token pairs wholly inside the utterance, windowed by W_max (W_max=16).

        Only word-aligned boundaries are emitted (P4: "word-aligned"), so we keep
        tokens whose char range starts at a word boundary in text. This mirrors
        ReverseAligner's ``_word_bounded_matches`` behavior for consistency.
        """
        out: List[Candidate] = []
        if not utterance_entries:
            return out
        # build the full utterance char range from entry extremes
        u_start = min(e.char_start for e in utterance_entries)
        u_end = max(e.char_end for e in utterance_entries)
        # word boundary positions: _entry_text joins entries with ' ', producing
        # offsets relative to the joined string — translate them into the
        # *rendered* (absolute) coordinate space used by token_offsets.
        text = self._entry_text(utterance_entries)
        word_bounds = {u_start + b for b in _word_boundaries(text)}

        # tokens inside utterance range (window = explicit cap or actual count)
        max_tok = self._effective_max_seq_len(token_offsets)
        in_toks = [
            (i, (tcs, tce))
            for i, (tcs, tce) in enumerate(token_offsets)
            if i < max_tok and tcs >= u_start and tce <= u_end
        ]
        starts = [i for i, (tcs, tce) in in_toks if tcs in word_bounds or tcs == 0]
        tok_map = dict(in_toks)
        cand_set: Dict[Tuple[int, int], Candidate] = {}
        for i in starts:
            tcs_i, _ = tok_map[i]
            for j, (_tcs_j, tce_j) in in_toks:
                if j < i or (j - i) >= self.max_span_width:
                    continue
                if tce_j in word_bounds or tce_j == u_end:
                    cval = (
                        utterance_text[tcs_i:tce_j].strip()
                        if len(utterance_text) >= tce_j
                        else None
                    )
                    cand = Candidate(
                        i, j, tcs_i, tce_j, "utterance",
                        cval, _entry_index_for_char(legend_entries, tcs_i))
                    cand_set[(i, j)] = cand
        # also add single-token spans not starting on a word bound (whole tokens)
        for i, (tcs, tce) in in_toks:
            if (i, i) not in cand_set:
                cval = (
                    utterance_text[tcs:tce].strip()
                    if len(utterance_text) >= tce
                    else None
                )
                cand_set[(i, i)] = Candidate(
                    i, i, tcs, tce, "utterance", cval, _entry_index_for_char(legend_entries, tcs))
        return list(cand_set.values())

    def _free_recent_spans(
        self,
        recent_entries: List[Any],
        token_offsets: Sequence[Tuple[int, int]],
        legend_entries: List[Any],
        rendered_text: str = "",
        existing: Optional[Sequence[Candidate]] = None,
    ) -> List[Candidate]:
        """Word-aligned sub-span candidates inside multi-word recent_context entries."""
        out: List[Candidate] = []
        seen = {(c.tok_start, c.tok_end, c.region) for c in (existing or ())}
        max_tok = self._effective_max_seq_len(token_offsets)
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
            e_idx = _entry_index(legend_entries, e)
            word_bounds = {e.char_start + b for b in _word_boundaries(entry_slice)}
            in_toks = [
                (i, (tcs, tce))
                for i, (tcs, tce) in enumerate(token_offsets)
                if i < max_tok and tcs >= e.char_start and tce <= e.char_end
            ]
            tok_map = dict(in_toks)
            starts = [i for i, (tcs, tce) in in_toks if tcs in word_bounds or tcs == e.char_start]
            for i in starts:
                tcs_i, _ = tok_map[i]
                for j, (_tcs_j, tce_j) in in_toks:
                    if j < i or (j - i) >= self.max_span_width:
                        continue
                    if tce_j in word_bounds or tce_j == e.char_end:
                        key = (i, j, "recent")
                        if key in seen:
                            continue
                        sub_val = (
                            rendered_text[tcs_i:tce_j].strip()
                            if len(rendered_text) >= tce_j
                            else entry_slice[tcs_i - e.char_start:tce_j - e.char_start].strip()
                        )
                        if not sub_val:
                            continue
                        seen.add(key)
                        out.append(Candidate(
                            i, j, tcs_i, tce_j, "recent", sub_val, e_idx))
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
    """Char offsets that start or end a space-delimited token in ``text``."""
    bounds = {0, len(text)}
    for i, ch in enumerate(text):
        if ch.isspace():
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


def _entry_index(legend_entries: Sequence[Any], entry: Any) -> int:
    try:
        return legend_entries.index(entry)
    except ValueError:
        return -1


def _entry_index_for_char(legend_entries: Sequence[Any], char_offset: int) -> int:
    """Entry index owning a char offset (for regex-free utterance spans)."""
    for i, e in enumerate(legend_entries):
        if e.char_start <= char_offset < e.char_end:
            return i
    return -1
