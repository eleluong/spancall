"""Reverse alignment of ground-truth call arguments against candidate spans and contexts.

Ref: docs/mechanism.md §8.1, docs/full_implementation_plan.md §2.1.
Identifies whether all arguments can be grounded (Stratum F) or requires
abstention (Stratum A), and emits **multi-positive** :class:`SpanTarget` labels
per parameter (R4: every equal-valued candidate — e.g. a `<|ctx_time|>` date entry
reached by both ``+1d`` and ``tomorrow`` — is a positive, never a chosen one).

Design:

- **Synthetic clock (anti-leakage):** when ``sample.current_time_iso`` is set it
  is the base date; otherwise a deterministic date is derived from
  ``hash(sample_id)``. Never anchored to the GT date, except when the utterance
  carries a relative idiom (``tomorrow``, ``day after tomorrow``, ``tonight``,
  ``this_<dow>``, ``next_<dow>``, ``in k days``) and no timestamp is recorded:
  then ``base = gt_date - idiom_offset`` and ``current_time_iso`` is stored so
  the relative date resolves to ``<|ctx_time|>`` (proposal §5.1).
- **Legend matching:** every parameter is grounded against the rendered
  :class:`SpanTemplateOutput.entries` (``<|null|>``/``true``/``false`` in
  <|ctx_meta|>, ``date`` in <|ctx_time|>, ``enum`` in <|ctx_enum|>, ``string`` in
  the utterance and quoted ``string`` rows in <|ctx_recent|>) — *no* string-regex
  heuristics over raw text.
- **Stratum & H4 reason:** every required param resolved -> ``Stratum.F`` with
  ``h4_reasons[param] == "OK"``; otherwise ``Stratum.A`` and each unaligned
  parameter gets an H4 class (``ABSENT`` | ``OFF_WINDOW`` | ``AMBIGUOUS`` |
  ``NEEDS_REASONING`` per docs/heads.md §H4).
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from span_template.template import (
    Param,
    RecentRow,
    SpanTemplateOutput,
    Tool,
    build_input,
)

from spancall.data.base import BaseAligner
from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    SpanTarget,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)

# ---------------------------------------------------------------------------
# H4 reason vocabulary (docs/heads.md §H4)
# ---------------------------------------------------------------------------
H4_REASONS = ("OK", "ABSENT", "AMBIGUOUS", "NEEDS_REASONING", "OFF_WINDOW")

# ---------------------------------------------------------------------------
# Deterministic synthetic-clock helpers
# ---------------------------------------------------------------------------

_SYNTH_START = date(2025, 1, 1)
_SYNTH_END = date(2027, 12, 31)
_SYNTH_SPAN_DAYS = (_SYNTH_END - _SYNTH_START).days + 1

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?$"
)

# Relative date idioms handled by build_time_window (span_template/template.py).
_IDIOM_OFFSETS = {
    "today": 0,
    "tonight": 0,
    "tomorrow": 1,
    "day after tomorrow": 2,
    "day_after_tomorrow": 2,
}
_WEEKDAY_NAMES = {
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
    "friday": 5, "saturday": 6, "sunday": 7,
    "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7,
}
_WEEKDAY_TOKEN_RE = re.compile(
    r"\b(?:this|next)_(" + "|".join(_WEEKDAY_NAMES) + r")\b"
)
_IN_K_DAYS_RE = re.compile(r"\bin\s+(\d+)\s+days?\b", re.IGNORECASE)


def _synthetic_base(sample_id: str) -> date:
    """Deterministic synthetic date in 2025-01-01..2027-12-31 seeded by id."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    n = int.from_bytes(digest[:4], "big")
    return _SYNTH_START + timedelta(days=n % _SYNTH_SPAN_DAYS)


def _parse_iso_datetime(value: Any) -> Optional[date]:
    """Extract the date from an ISO timestamp (``2026-11-02T...`` or ``YYYY-MM-DD``)."""
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    if isinstance(value, str):
        m = _ISO_DATETIME_RE.match(value.strip())
        if m:
            return date.fromisoformat(m.group(1))
        m = _ISO_DATE_RE.match(value.strip())
        if m:
            # whole-string ISO date (no capture group) -> the full match IS the date
            return date.fromisoformat(m.group(0))
    return None


def _resolve_base_date(sample: UnifiedSample, gt_dates: Sequence[date]) -> Tuple[date, str]:
    """Compute the anti-leakage base date + whether it is synthetic.

    Returns ``(base_date, current_time_iso)`` where ``current_time_iso`` is the
    value to persist on the sample (synthetic clock assignment is recorded so
    re-alignment after augmentation reuses the *same* clock for the same row).
    """
    if sample.current_time_iso:
        parsed = _parse_iso_datetime(sample.current_time_iso)
        if parsed is not None:
            return parsed, sample.current_time_iso

    # Relative idiom in the utterance? Solve base = gt_date - offset.
    utt_lower = sample.utterance.lower()
    idiom_offset: Optional[int] = None
    for idiom, offset in _IDIOM_OFFSETS.items():
        if idiom in utt_lower:
            idiom_offset = offset
            break
    if idiom_offset is None:
        m = _IN_K_DAYS_RE.search(utt_lower)
        if m:
            try:
                idiom_offset = int(m.group(1))
            except ValueError:
                idiom_offset = None
    if idiom_offset is not None and gt_dates:
        base = gt_dates[0] - timedelta(days=idiom_offset)
        iso = base.isoformat()
        return base, iso

    base = _synthetic_base(sample.sample_id)
    return base, base.isoformat()


def _candidate_weekday_delta(utterance: str, base: date) -> Optional[int]:
    """If the utterance contains ``this_<dow>`` / ``next_<dow>`` resolve the
    delta applied to ``base`` that the time window would emit, else None.

    ``this_<dow>`` maps to ``(target - base.isoweekday()) % 7``; ``next_<dow>``
    is strictly future (1..7 days), mirroring ``span_template.template._weekday``.
    Returns None when no weekday idiom is present.
    """
    m = _WEEKDAY_TOKEN_RE.search(utterance.lower())
    if not m:
        return None
    prefix, _, name = m.group(0).partition("_")
    target = _WEEKDAY_NAMES[name]
    delta = (target - base.isoweekday()) % 7
    if prefix == "next" and delta == 0:
        delta = 7
    return delta


def _is_colliding_gt_date(gt: Any, base: date, idiom_delta: Optional[int]) -> bool:
    """Whether a GT value is plausibly a relative idiom anchored to ``base``.

    Used only to choose the priority order of the time-window collision search:
    when the utterance says ``tomorrow`` and the GT date equals ``base + 1`` we
    prefer the idiom entry even though it collides with the ``+1d`` offset entry.
    """
    day = _parse_iso_datetime(gt)
    if day is None or idiom_delta is None:
        return False
    return (day - base).days == idiom_delta


def _resolve_gt_dates(gt_call: GroundTruthCall) -> List[date]:
    out: List[date] = []
    for v in (gt_call.arguments or {}).values():
        day = _parse_iso_datetime(v)
        if day is not None:
            out.append(day)
    return out


def _date_from_relative(entry: Any, base: date) -> Optional[int]:
    """Return ``days - base`` for a <|ctx_time|> entry; None for non-relative keys."""
    key = entry.key or ""
    if key == "today" or key == "tonight":
        return 0
    if key == "tomorrow":
        return 1
    if key == "day_after_tomorrow":
        return 2
    m = re.fullmatch(r"\+(\\d+)d", key)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    m = _WEEKDAY_TOKEN_RE.fullmatch(key)
    if m:
        prefix, _, name = m.group(0).partition("_")
        target = _WEEKDAY_NAMES[name]
        delta = (target - base.isoweekday()) % 7
        if prefix == "next" and delta == 0:
            delta = 7
        return delta
    return None


# ---------------------------------------------------------------------------
# Utility: utterance word-boundary matching (Case E)
# ---------------------------------------------------------------------------

_UNBOUNDED_RE = re.compile(r"[^\s,.;:!?\"'()\[\]{}<>]+$")


def _word_bounded_matches(text: str, needle: str) -> List[Tuple[int, int]]:
    """All non-overlapping, word-bounded occurrences of ``needle`` in ``text``.

    Word boundary: the char before/after the match must be a non-alphanumeric /
    non-underscore char (so ``"5"`` never matches inside ``"15"``). Case is
    preserved; matching is exact on the rendered (normalized) utterance.
    """
    if not needle:
        return []
    out: List[Tuple[int, int]] = []
    start = 0
    n = len(text)
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        before = text[i - 1] if i > 0 else " "
        after = text[i + len(needle)] if i + len(needle) < n else " "
        if not _is_word_char(before) and not _is_word_char(after):
            out.append((i, i + len(needle)))
        start = i + 1
    return out


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


# ---------------------------------------------------------------------------
# json-mode: infer a ParameterSchema from a GT arguments dict
# ---------------------------------------------------------------------------


def _infer_schema_from_arguments(
    gt_call: GroundTruthCall,
    tool_def: ToolDefinition,
) -> Tuple[ToolDefinition, bool]:
    """Synthesize ``ParameterSchema`` entries from ``gt_call.arguments`` keys via JSON types.

    Returns ``(tool_def, changed)`` — ``changed=True`` means the tool gained the
    inferred schema (used when the synthetic ``json_output`` tool has no params).
    Python scalar types map to schema types: ``str`` -> string, ``int`` -> integer,
    ``float`` -> number, ``bool`` -> boolean, ``list`` -> array, ``dict`` -> object.
    """
    if tool_def.parameters:
        return tool_def, False
    args = gt_call.arguments or {}
    params: List[ParameterSchema] = []
    for key, value in args.items():
        if isinstance(value, bool):
            ptype, items_type = "boolean", None
        elif isinstance(value, int) and not isinstance(value, bool):
            ptype, items_type = "integer", None
        elif isinstance(value, float):
            ptype, items_type = "number", None
        elif isinstance(value, str):
            ptype, items_type = "string", None
        elif isinstance(value, list):
            ptype, items_type = "array", _element_type(value)
        elif isinstance(value, dict):
            ptype, items_type = "object", None
        else:
            ptype, items_type = "string", None
        params.append(
            ParameterSchema(
                name=str(key),
                description="",
                param_type=ptype,
                required=True,
                items_type=items_type,
            )
        )
    if not params:
        return tool_def, False
    tool_def = ToolDefinition(
        name=tool_def.name,
        description=tool_def.description,
        parameters=params,
        risk_tier=tool_def.risk_tier,
    )
    return tool_def, True


def _element_type(values: Sequence[Any]) -> Optional[str]:
    for v in values:
        if isinstance(v, bool):
            return "boolean"
        if isinstance(v, int):
            return "integer"
        if isinstance(v, float):
            return "number"
        if isinstance(v, str):
            return "string"
    return None


# ---------------------------------------------------------------------------
# Schema -> span_template.Tool conversion
# ---------------------------------------------------------------------------


def tooldef_to_template_tool(t: ToolDefinition) -> Tool:
    """Convert a :class:`ToolDefinition` to a :class:`span_template.Tool`."""
    params: List[Param] = []
    for p in t.parameters:
        kind = _param_kind(p)
        params.append(
            Param(
                name=p.name,
                description=p.description,
                kind=kind,
                optional=not p.required,
                required=p.required,
                enum=[str(v) for v in (p.enum or [])],
                default=p.default,
                has_default=p.default is not None,
            )
        )
    return Tool(name=t.name, description=t.description, params=params)


def _param_kind(p: ParameterSchema) -> str:
    """Map a ParameterSchema type to a span_template.Param.kind token."""
    if p.param_type == "boolean":
        return "bool"
    if p.param_type == "integer":
        return "int"
    if p.param_type == "number":
        return "float"
    if p.param_type == "array":
        if p.items_type == "integer":
            return "List[int]"
        if p.items_type == "number":
            return "List[float]"
        if p.items_type == "boolean":
            return "List[bool]"
        return "List[str]"
    return "str"


def _reason_key(call_index: int, param_name: str, multi_call: bool) -> str:
    """Key space for ``target_spans`` / ``h4_reasons`` (R0.4).

    Multi-call samples qualify by call so two calls declaring the same parameter
    name do not overwrite each other. Single-call samples keep the bare name so
    every existing reader sees byte-identical output.
    """
    return f"{int(call_index)}::{param_name}" if multi_call else param_name


# ---------------------------------------------------------------------------
# The aligner
# ---------------------------------------------------------------------------


class ReverseAligner(BaseAligner):
    """Engine for aligning tool arguments to utterance spans and context items."""

    def __init__(
        self,
        fuzzy_matching: bool = False,
        max_date_drift_days: int = 14,
        align_json_mode_keys: bool = True,
    ) -> None:
        self.fuzzy_matching = fuzzy_matching
        self.max_date_drift_days = max_date_drift_days
        self.align_json_mode_keys = align_json_mode_keys

    # ------------------------------------------------------------------ main
    def align(self, sample: UnifiedSample) -> UnifiedSample:
        """Ground every GT argument into ``sample.target_spans`` + stratum/H4 tags."""
        sample = _copy_sample(sample)
        sample.stratum = Stratum.A
        sample.metadata.setdefault("h4_reasons", {})
        sample.metadata.setdefault("list_sizes", {})

        if not sample.ground_truth_calls:
            sample.metadata["abstain_reason"] = "no_gt_call"
            return sample

        # json_mode key inference (phase-1 spec §2.1 step 4): synthesize a
        # ParameterSchema from the JSON keys so extractable fields supervise.
        if self.align_json_mode_keys:
            sample = self._infer_json_mode_schemas(sample)

        # base date: recorded timestamp wins; else deterministic synthetic clock,
        # solved against a relative idiom when the utterance carries one.
        gt_dates = _resolve_gt_dates(sample.ground_truth_calls[0])
        base, iso = _resolve_base_date(sample, gt_dates)
        if not sample.current_time_iso:
            sample.current_time_iso = iso

        reason_for: Dict[str, Optional[str]] = {}
        all_grounded = True
        # R0.4: when a sample carries more than one GT call, span keys and H4
        # reasons are qualified with the call index so two calls that declare
        # the same parameter name (e.g. both ask for ``query``) stop colliding.
        # Single-call samples keep bare keys, which is what every existing
        # reader (augment, diagnose scripts, tensorize) expects.
        multi_call = len(sample.ground_truth_calls) > 1
        for call_index, gt_call in enumerate(sample.ground_truth_calls):
            tool_def = _find_tool(sample.tools, gt_call.tool_name)
            if tool_def is None:
                # call tool not among the ambient tools -> ungroundable (Stratum A)
                for pname in gt_call.arguments:
                    reason_for[_reason_key(call_index, pname, multi_call)] = "OFF_WINDOW"
                all_grounded = False
                continue
            per_param, ok = self._align_call(
                sample, gt_call, tool_def, base,
                call_index=call_index, multi_call=multi_call,
            )
            for pname, reason in per_param.items():
                reason_for[_reason_key(call_index, pname, multi_call)] = reason
            all_grounded = all_grounded and ok

        if all_grounded:
            sample.stratum = Stratum.F
            for pname in list(reason_for):
                if reason_for.get(pname) in (None, "OK"):
                    reason_for[pname] = "OK"
        else:
            sample.metadata["abstain_reason"] = "ungrounded_required"

        sample.metadata["h4_reasons"] = reason_for
        return sample

    # ------------------------------------------------------------- internals
    def _infer_json_mode_schemas(self, sample: UnifiedSample) -> UnifiedSample:
        """Replace empty-param tools used by json-mode rows with inferred schemas."""
        new_tools = list(sample.tools)
        for gt in sample.ground_truth_calls:
            idx = next(
                (i for i, t in enumerate(new_tools) if t.name == gt.tool_name),
                None,
            )
            if idx is None:
                continue
            updated, changed = _infer_schema_from_arguments(gt, new_tools[idx])
            if changed:
                new_tools[idx] = updated
        sample.tools = new_tools
        return sample

    def _align_call(
        self,
        sample: UnifiedSample,
        gt_call: GroundTruthCall,
        tool_def: ToolDefinition,
        base: date,
        call_index: int = 0,
        multi_call: bool = False,
    ) -> Tuple[Dict[str, Optional[str]], bool]:
        """Align one GT call; returns ``({param: reason}, all_grounded)``.

        ``call_index``/``multi_call`` route every emitted target into the
        call-qualified key space (R0.4). They default to the legacy behaviour so
        direct callers and single-call corpora are byte-identical to before.
        """
        # Always render so deterministic <|ctx_time|>/<|ctx_enum|> pools are available.
        try:
            output = self._render(sample, tool_def, base)
        except Exception as exc:  # SanitizeError / ValueError -> abstain (plan §2.1 step 2)
            reasons: Dict[str, Optional[str]] = {}
            for p in tool_def.parameters:
                reasons[p.name] = "NEEDS_REASONING" if p.required else None
            reasons["__render_error__"] = str(exc)
            return reasons, False

        params = tool_def.parameters or []
        names = {p.name for p in params}
        args = gt_call.arguments or {}
        reasons: Dict[str, Optional[str]] = {}
        ok = True
        for p in params:
            if p.name not in args or args[p.name] is None:
                # Case A: omitted / None
                if not p.required:
                    reasons[p.name] = "OK"
                    self._emit(sample, p.name, [self._null_target(p)],
                               call_index=call_index, multi_call=multi_call)
                else:
                    reasons[p.name] = "ABSENT"
                    self._emit(sample, p.name, [self._unresolved_target(p)],
                               call_index=call_index, multi_call=multi_call)
                    ok = False
                continue
            targets, p_ok, reason = self._align_value(
                p, args[p.name], output, base, sample
            )
            if targets:
                self._emit(sample, p.name, targets,
                           call_index=call_index, multi_call=multi_call)
            if not p_ok:
                ok = False
                reasons[p.name] = reason or "NEEDS_REASONING"
            else:
                reasons[p.name] = "OK"

        # params mentioned in args but absent from the tool schema -> ungroundable
        for pname in args:
            if pname not in names:
                reasons[pname] = "OFF_WINDOW"
                ok = False
        return reasons, ok

    def _render(
        self,
        sample: UnifiedSample,
        tool_def: ToolDefinition,
        base: date,
    ) -> SpanTemplateOutput:
        """Render the single-tool template for the target tool + recent context."""
        tool = tooldef_to_template_tool(tool_def)
        recent = [RecentRow(key=r["key"], value=r["value"], source=r.get("source", "recent"))
                  for r in sample.recent_context]
        return build_input(
            tool=tool,
            utterance=sample.utterance,
            recent_context=recent,
            now=base,
        )

    # -------------------------------------------------- per-value alignment
    def _align_value(
        self,
        p: ParameterSchema,
        val: Any,
        output: SpanTemplateOutput,
        base: date,
        sample: UnifiedSample,
    ) -> Tuple[List[SpanTarget], bool, Optional[str]]:
        """Align one argument to zero-or-more SpanTargets.

        Returns ``(targets, ok, reason)`` — ``ok=False`` when the required
        parameter could not be grounded; ``reason`` is the H4 class.
        """
        # normalize the parameter value for comparison
        if isinstance(val, bool):
            s_val = str(val).lower()
        else:
            s_val = str(val).strip()

        # Case B: boolean
        if p.param_type == "boolean" or isinstance(val, bool):
            matches = [
                e for e in output.entries
                if e.segment == "ctx_meta" and e.kind == "bool"
                and str(e.value).lower() == (str(val).lower() if isinstance(val, str) else s_val)
            ]
            if matches:
                return (
                    [self._target(p, e, "boolean_ctx", bool(val)) for e in matches],
                    True,
                    None,
                )
            return self._fail(p, "AMBIGUOUS")

        # Case C: enum param
        if p.enum:
            s_lower = str(val).strip().lower()
            matches = [
                e for e in output.entries
                if e.segment == "ctx_enum" and e.kind == "enum"
                and str(e.value).strip().lower() == s_lower
            ]
            if matches:
                return (
                    [self._target(p, e, "enum_ctx", val) for e in matches],
                    True,
                    None,
                )
            # enum value genuinely not in the pool -> ungroundable
            return self._fail(p, "OFF_WINDOW")

        # Case D: date/datetime value
        if _parse_iso_datetime(val) is not None or (
            p.param_type in ("date", "datetime")
        ):
            return self._align_date(p, val, output, base)

        # Case F: array
        if p.param_type == "array" and isinstance(val, list):
            return self._align_array(p, val, output, base, sample)

        # Case E: scalar string / integer / number
        # -- recent-context first (most specific: whole-value equality on a
        #    selectable quoted string row, or a word-bounded substring)
        recent_matches = self._match_recent(s_val, output, p)
        if recent_matches:
            return recent_matches, True, None

        # -- utterance verbatim (word-bounded, non-overlapping)
        utt = _find_utterance_entry(output)
        if utt is not None:
            if self.fuzzy_matching:
                occ = self._fuzzy_occ(utt, s_val)
            else:
                occ = _word_bounded_matches(utt.value if isinstance(utt.value, str) else "", s_val)
            if occ:
                targets = [
                    SpanTarget(
                        param_name=p.name,
                        target_type="utterance_span",
                        char_start=utt.char_start + rel_start,
                        char_end=utt.char_start + rel_end,
                        canonical_value=val,
                    )
                    for rel_start, rel_end in occ
                ]
                return targets, True, None

        # scalar params may also come from enum-less <|ctx_enum|>? No — Case C above.
        return self._fail(p, "OFF_WINDOW")

    def _align_date(
        self,
        p: ParameterSchema,
        val: Any,
        output: SpanTemplateOutput,
        base: date,
    ) -> Tuple[List[SpanTarget], bool, Optional[str]]:
        """Case D: find ALL colliding <|ctx_time|> entries plus verbatim literals in
        the utterance (R4 — every equal-valued entry is a positive)."""
        day = _parse_iso_datetime(val)
        if day is None:
            return self._fail(p, "OFF_WINDOW")
        iso = day.isoformat()
        entries = [e for e in output.entries if e.segment == "ctx_time" and e.kind == "date"]
        targets: List[SpanTarget] = []
        for e in entries:
            if str(e.value) == iso:
                targets.append(self._target(p, e, "time_ctx", iso))
        if not targets and self.max_date_drift_days > 0:
            for e in entries:
                rel = _date_from_relative(e, base)
                if rel is not None and abs((day - base).days - rel) <= self.max_date_drift_days:
                    targets.append(self._target(p, e, "time_ctx", iso))
                    break  # one drift proxy positive; no fabrication of multiple

        # utterance verbatim literal (YYYY-MM-DD) matches
        utt = _find_utterance_entry(output)
        if utt is not None and isinstance(utt.value, str):
            occ = _word_bounded_matches(utt.value, iso)
            for rel_start, rel_end in occ:
                targets.append(
                    SpanTarget(
                        param_name=p.name,
                        target_type="utterance_span",
                        char_start=utt.char_start + rel_start,
                        char_end=utt.char_start + rel_end,
                        canonical_value=iso,
                    )
                )
        if targets:
            return targets, True, None
        return self._fail(p, "OFF_WINDOW")

    def _align_array(
        self,
        p: ParameterSchema,
        val: List[Any],
        output: SpanTemplateOutput,
        base: date,
        sample: UnifiedSample,
    ) -> Tuple[List[SpanTarget], bool, Optional[str]]:
        """Case F: align every element via Cases B–E; all elements must align."""
        all_targets: List[SpanTarget] = []
        for v_i in val:
            sub_p = _element_param(p, v_i)
            targets, ok, reason = self._align_value(sub_p, v_i, output, base, sample)
            if not ok or not targets:
                sample.metadata.setdefault("list_sizes", {})[p.name] = len(val)
                return self._fail(p, reason or "OFF_WINDOW")
            all_targets.extend(targets)
        sample.metadata.setdefault("list_sizes", {})[p.name] = len(val)
        return all_targets, True, None

    # ----------------------------------------------------------- recent ctx
    def _match_recent(
        self,
        s_val: str,
        output: SpanTemplateOutput,
        p: ParameterSchema,
    ) -> List[SpanTarget]:
        """Find selectable recent-context rows whose value equals or word-bounded-
        contains ``s_val``; every collision is a positive (R4)."""
        out: List[SpanTarget] = []
        for e in output.entries:
            if e.segment != "recent" or not e.selectable or e.kind != "string":
                continue
            e_val = str(e.value)
            if e_val == s_val:
                out.append(self._target(p, e, "recent_ctx", e.value, ctx_key=e.key))
                continue
            if not self.fuzzy_matching and _word_bounded_matches(e_val, s_val):
                # whole-value equality is preferred; substring matches are a
                # fallback only when the substring is a full word inside the value
                if _is_full_token_in(e_val, s_val):
                    out.append(self._target(p, e, "recent_ctx", e.value, ctx_key=e.key))
        return out

    # -------------------------------------------------------------- helpers
    def _target(
        self,
        p: ParameterSchema,
        e: Any,
        target_type: str,
        canonical_value: Any,
        ctx_key: Optional[str] = None,
    ) -> SpanTarget:
        return SpanTarget(
            param_name=p.name,
            target_type=target_type,
            char_start=e.char_start,
            char_end=e.char_end,
            ctx_key=ctx_key,
            canonical_value=canonical_value,
        )

    def _null_target(self, p: ParameterSchema) -> SpanTarget:
        return SpanTarget(
            param_name=p.name,
            target_type="null_ctx",
            canonical_value=None,
        )

    def _unresolved_target(self, p: ParameterSchema) -> SpanTarget:
        return SpanTarget(
            param_name=p.name,
            target_type="unresolved",
            canonical_value=None,
        )

    def _fail(self, p: ParameterSchema, reason: str) -> Tuple[List[SpanTarget], bool, str]:
        return [self._unresolved_target(p)], False, reason

    def _emit(
        self,
        sample: UnifiedSample,
        pname: str,
        targets: List[SpanTarget],
        call_index: int = 0,
        multi_call: bool = False,
    ) -> None:
        """Record one parameter's targets under its (possibly call-qualified) key.

        Multi-call samples key by ``f"{call_index}::{pname}"``; single-call
        samples keep the bare ``pname`` so every existing reader is untouched
        (R0.4, ``UnifiedSample.targets_for_call``).
        """
        key = _reason_key(call_index, pname, multi_call)
        sample.target_spans.setdefault(key, []).extend(targets)

    def _fuzzy_occ(self, utt: Any, s_val: str) -> List[Tuple[int, int]]:
        """Case-insensitive fallback used when ``fuzzy_matching=True``."""
        text = utt.value if isinstance(utt.value, str) else ""
        low = text.lower()
        needle = s_val.lower()
        out: List[Tuple[int, int]] = []
        start = 0
        while True:
            i = low.find(needle, start)
            if i < 0:
                break
            before = text[i - 1] if i > 0 else " "
            after = text[i + len(needle)] if i + len(needle) < len(text) else " "
            if not _is_word_char(before) and not _is_word_char(after):
                out.append((i, i + len(needle)))
            start = i + 1
        return out


def _find_tool(tools: Sequence[ToolDefinition], name: str) -> Optional[ToolDefinition]:
    for t in tools:
        if t.name == name:
            return t
    return None


def _find_utterance_entry(output: SpanTemplateOutput) -> Optional[Any]:
    for e in output.entries:
        if e.segment == "utterance" and e.selectable:
            return e
    return None


def _is_full_token_in(text: str, token: str) -> bool:
    """True iff ``token`` occurs in ``text`` as a word-bounded token.

    Mirrors ``_word_bounded_matches`` semantics so value-mentions like ``alice``
    inside ``alice@example.com`` qualify (R4 — every collision is a positive),
    while ``"5"`` never matches inside ``"15"``.
    """
    return bool(_word_bounded_matches(text, token))


def _element_param(p: ParameterSchema, v: Any) -> ParameterSchema:
    """Sub-schema for one array element (keeps the parent's name/enum/entity)."""
    ptype = p.items_type or _infer_scalar_type(v)
    return ParameterSchema(
        name=p.name,
        description=p.description,
        param_type=ptype,
        required=True,
        enum=p.enum,
        items_type=None,
        entity_class=p.entity_class,
    )


def _infer_scalar_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str) and _ISO_DATE_RE.match(v):
        return "date"
    return "string"


def _copy_sample(sample: UnifiedSample) -> UnifiedSample:
    """Deep-ish copy: fresh target_spans/metadata dicts so re-alignment (e.g.
    after augmentation) never mutates the caller's instance."""
    from dataclasses import replace

    return replace(
        sample,
        target_spans=dict(sample.target_spans),
        metadata=dict(sample.metadata),
    )
