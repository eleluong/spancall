"""SpanCall input template — render (Jinja) → parse (marker grammar) → legend.

This is the so-called "inverted chat template": instead of only producing text
for a generative model to continue, we render the §3.1 template into the exact
token sequence the encoder consumes, AND parse that same text into the
extraction legend (entries + canonical values) so the pointer head gets span
supervision without string-regex heuristics.

Markers use the Qwen-native ``<|...|>`` convention — atomic special tokens
registered on the tokenizer (``span_template.sanitize.SPECIAL_MARKER_TOKENS``)
matching Qwen3's own ``<|im_start|>``-style ids, so a modern BPE never
byte-splits a structural marker (docs/pipeline_io.md P6.3, long-context R8
validation gate).

Render and parse share `template.jinja` / `template_multi.jinja` as the single
source of truth:
  1. build_input(tool, utterance, ...) renders the single-tool template
  2. build_input_multi(tools, utterance, ...) renders the packed multi-tool
     template (pipeline_io P5, heads.md §2) with the <|none|> anchor
  3. span_template.parser.parse(text) recovers the legend from the text

Sanitization (pipeline_io P0/P1, proposal R7) runs *before* rendering: marker
injection / empty / too-long inputs raise ``SanitizeError`` so the router can
abstain; clean inputs render with the normalized text AND a ``norm2orig`` map so
Q4 can always return the user's exact original characters.

The result (`SpanTemplateOutput`) carries both `text` and `entries`; a tool call
is considered "context-ready" only when every required param maps to a
selectable target (or the optional <|null|> / <|unresolved|> handles it) — see
`gate_tool_call` for the all-or-nothing whole-call rule.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from jinja2 import Environment, FileSystemLoader

from .parser import Entry, ParseResult, parse
from .sanitize import (
    SanitizeError,
    check_recent_value,
    sanitize_utterance,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent
_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), trim_blocks=False,
                   lstrip_blocks=False)


def quote_value(v: str) -> str:
    """R1 quoting: `\\` -> `\\\\`, `"` -> `\\"`, wrapped in double quotes."""
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


_env.globals["quote"] = quote_value


def _enum_token(v: str) -> str:
    """Render one enum value for <|ctx_enum|> (D2/R5).

    Values containing internal whitespace are emitted quoted (R1) so the pool
    cannot fragment; `|` is rejected at ingestion (validate_enums) and therefore
    never appears here even inside quotes. Single-token values stay unquoted.
    """
    return quote_value(v) if _ENUM_WS_RE.search(v) else v


_env.filters["enum_token"] = _enum_token

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_ENUM_BAD_RE = re.compile(r"\|")  # the separator '|' is forbidden even quoted (R5/D2)
_ENUM_WS_RE = re.compile(r"\s")   # whitespace requires quoting (D2)


# ---------------------------------------------------------------------------
# Interpreted input dataclasses (feed the template context)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    name: str
    description: str = ""
    kind: str = "str"            # str | int | float | bool | List[str]
    optional: bool = False
    required: bool = True        # explicit; optional may be required=False
    enum: Sequence[str] = ()     # candidate values fed into <|ctx_enum|>
    default: Any = None          # JSON Schema default applied deterministically (Q6.2)
    has_default: bool = False    # distinguish default=None from "no default"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""
    params: Sequence[Param] = ()


@dataclass(frozen=True)
class TimeEntry:
    key: str
    iso: str


@dataclass(frozen=True)
class RecentRow:
    """A key/value in recent context (agent working state)."""

    key: str
    value: str
    source: str = "recent"

    def __post_init__(self) -> None:
        if not _KEY_RE.match(self.key):
            raise ValueError(
                f"RecentRow key {self.key!r} must be a plain identifier "
                "(use value for anything with spaces)"
            )
        if check_recent_value(self.key):
            raise ValueError(f"RecentRow key {self.key!r} contains a reserved marker")
        if check_recent_value(self.value):
            raise ValueError(
                f"RecentRow value for {self.key!r} contains a reserved marker (R7)")


@dataclass(frozen=True)
class NormalizedText:
    """P0 result: normalized text + the offset map back to the user's original."""

    orig: str
    norm: str
    norm2orig: Sequence[int]


@dataclass
class SpanTemplateOutput:
    """Result of `build_input`: the rendered text + the parsed legend."""

    text: str
    entries: List[Entry]
    tool: Tool
    utterance: str
    time_window: Sequence[TimeEntry]
    recent_context: Sequence[RecentRow]
    norm: Optional[NormalizedText] = None   # P0 offsets (set when sanitized)
    tools: Sequence[Tool] = ()              # multi-tool packing (P5)

    # -- legend accessors ---------------------------------------------------
    def targetable(self) -> Iterable[Entry]:
        return (e for e in self.entries if e.selectable)

    def entry_at(self, offset: int, segment: Optional[str] = None) -> Optional[Entry]:
        # `_parse` is a lazy property (no hasattr guard needed — always present).
        return self._parse.locate(offset, segment)

    def tokenize(self, tokenizer: Any) -> Tuple[List[int], List[Entry]]:
        """Char→token mapping via offset mapping; raises on boundary violations.

        Returns (token_ids, per_token_entry) using the tokenizer's
        ``tokenize`` + ``convert_tokens_to_ids``. For a production offset-mapping
        path with boundary-safety checks, use `tokenize_offset_mapping`.
        """
        ids: List[int] = []
        per_token: List[Entry] = []
        for e in self.entries:
            piece = self.text[e.char_start : e.char_end]
            if not piece:
                continue
            toks = tokenizer.tokenize(piece)
            ids.extend(tokenizer.convert_tokens_to_ids(toks) if toks else [])
            per_token.extend([e] * len(toks))
        return ids, per_token

    @property
    def _parse(self) -> ParseResult:
        if getattr(self, "__parse", None) is None:
            self.__parse = parse(self.text)
        return self.__parse

    def __repr__(self) -> str:
        return (f"SpanTemplateOutput(text_len={len(self.text)}, "
                f"entries={len(self.entries)}, tools={[t.name for t in self.tools]})")


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def build_time_window(
    base: date,
    window_days: Sequence[int] = (1, 2, 3, 7),
    weekdays: Sequence[str] = ("next_mon", "this_fri"),
    idioms: Sequence[str] = ("tomorrow", "day_after_tomorrow", "tonight"),
    fmt: str = "%Y-%m-%d",
) -> List[TimeEntry]:
    """Deterministic <|ctx_time|> window (bounded, configurable — §3.1 R4).

    `today`, `+kd` offsets, weekday entries, and a **closed idiom map** (D1,
    docs/remediation_plan.md §3 D1): `tomorrow` (= `+1d`),
    `day_after_tomorrow` (= `+2d`), `tonight` (= today's date, date-params only
    — a time-of-day value cannot be satisfied by any <|ctx_time|> date entry, so a
    datetime param abstains at the caster for it). Weekday tokens are
    `this_<dow>` (= current week's weekday; on the same weekday it resolves to
    *today*) or `next_<dow>` (strictly future, i.e. 1–7 days ahead).

    The vocabulary stays closed: range-like idioms ("next week", "this weekend")
    are deliberately **off-window** — they need canonicalization semantics the
    template does not claim; the param abstains to the LLM.
    """
    out: List[TimeEntry] = [TimeEntry("today", base.strftime(fmt))]
    for off in sorted(set(window_days)):
        out.append(TimeEntry(f"+{off}d", (base + timedelta(days=off)).strftime(fmt)))
    for name in weekdays:
        out.append(TimeEntry(name, _weekday(base, name).strftime(fmt)))
    for idiom in idioms:
        out.append(_idiom_entry(base, idiom, fmt))
    return out


def _idiom_entry(base: date, idiom: str, fmt: str) -> TimeEntry:
    """Resolve a closed-vocabulary relative-date idiom to a dated entry (D1).

    Entries are separate (`tomorrow` and `+1d` are distinct keys that may share a
    canonical date value) — the R4 equal-value aggregation at inference and the
    multi-positive training labels fold colliding values, never the keys.
    """
    if idiom == "tomorrow":
        return TimeEntry(idiom, (base + timedelta(days=1)).strftime(fmt))
    if idiom == "day_after_tomorrow":
        return TimeEntry(idiom, (base + timedelta(days=2)).strftime(fmt))
    if idiom == "tonight":
        return TimeEntry(idiom, base.strftime(fmt))
    raise ValueError(
        f"Unsupported idiom token {idiom!r}; supported: "
        "tomorrow | day_after_tomorrow | tonight")


def _weekday(base: date, token: str) -> date:
    prefix, _, name = token.rpartition("_")
    table = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}
    if name not in table:
        raise ValueError(f"Unsupported weekday token {token!r}; "
                         f"supported: this_<dow> | next_<dow>")
    target = table[name]
    delta = (target - base.isoweekday()) % 7
    if prefix == "this":
        pass  # 0..6 days ahead; same weekday => today
    elif prefix == "next":
        if delta == 0:
            delta = 7  # "next" always strictly future
    else:
        raise ValueError(f"Unsupported weekday token {token!r}; use this_<dow>|next_<dow>")
    return base + timedelta(days=delta)


def validate_enums(tools: Sequence[Tool]) -> None:
    """R5/D2: enum values must be joinable by ' | ' without ambiguity.

    - The separator `|` is rejected anywhere — even inside a quoted value,
      because a quoted entry containing ` | ` could still fragment the pool
      (R5 stays strict there).
    - Leading/trailing whitespace of the whole value is rejected (a dangling
      space next to the separator is ambiguous in the rendered pool).
    - Internal whitespace is allowed; such values are rendered quoted (see
      `_enum_token`) so the pool remains unambiguous.

    Enforced by `validate_enums` before rendering (proposal R5 / R5-D2).
    """
    for t in tools:
        for p in t.params:
            for v in p.enum:
                if not isinstance(v, str) or not v.strip():
                    raise ValueError(
                        f"tool {t.name!r} param {p.name!r}: enum value {v!r} must be "
                        "a non-empty string (R5/D2)")
                if v != v.strip():
                    raise ValueError(
                        f"tool {t.name!r} param {p.name!r}: enum value {v!r} has "
                        "leading/trailing whitespace (R5/D2)")
                if _ENUM_BAD_RE.search(v):
                    raise ValueError(
                        f"tool {t.name!r} param {p.name!r}: enum value {v!r} contains "
                        "the '|' separator, forbidden even quoted (R5/D2)")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def build_input(
    tool: Tool,
    utterance: str,
    *,
    recent_context: Sequence[RecentRow] = (),
    now: date,
    time_window: Optional[Sequence[TimeEntry]] = None,
    window_days: Sequence[int] = (1, 2, 3, 7),
    weekday_tokens: Sequence[str] = ("next_mon", "this_fri"),
    idioms: Sequence[str] = ("tomorrow", "day_after_tomorrow", "tonight"),
    include_enum: bool = True,
    sanitize: bool = True,
) -> SpanTemplateOutput:
    """Render + parse the *single-tool* SpanCall template (proposal §3.1).

    Renders ``template.jinja`` — no ``<|none|>`` anchor: that belongs only to the
    packed multi-tool form `build_input_multi` (pipeline_io P5, heads.md §2).

    Args:
        tool: candidate tool schema (selected by the planner).
        utterance: current user message (turn N).
        recent_context: optional rows from agent working state (R6).
        now: system clock date for <|ctx_time|>. Required — rendering must be
             deterministic (no wall-clock fallback) so cached renders and
             byte-exact fixtures are reproducible (proposal §3.1, R4).
        time_window: prebuilt window; else built from window_days/weekday_tokens.
        window_days, weekday_tokens: <|ctx_time|> window params (R4).
        include_enum: include <|ctx_enum|> (R5).
        sanitize: run P0/P1 (default True; raises ``SanitizeError`` on injection).

    Raises:
        SanitizeError: marker injection / empty / too-long utterance (R7).
        ValueError: enum pool contains a value with whitespace or '|' (R5).
    """
    norm_utt, norm = _normalize(utterance, sanitize)
    validate_enums([tool])

    if time_window is None:
        time_window = build_time_window(now,
                                        window_days=window_days,
                                        weekdays=weekday_tokens,
                                        idioms=idioms)
    enums: List[str] = []
    if include_enum:
        enums = sorted({v for p in tool.params for v in p.enum})

    template = _env.get_template("template.jinja")
    rendered = template.render(
        tool=tool,
        utterance=norm_utt,
        time_window=time_window,
        enums=enums,
        recent_context=recent_context,
    )

    parsed = parse(rendered)
    return SpanTemplateOutput(
        text=rendered,
        entries=parsed.entries,
        tool=tool,
        utterance=norm_utt,
        time_window=list(time_window),
        recent_context=list(recent_context),
        norm=norm,
        tools=[tool],
    )


def _normalize(utterance: str, sanitize: bool) -> Tuple[str, Optional[NormalizedText]]:
    """P0/P1 before rendering (shared by single/multi build paths).

    Returns (normalized_text, offset_map_or_None). When ``sanitize`` is True any
    marker-injection / empty / too-long input raises ``SanitizeError`` so the
    router abstains (R7); clean input yields the normalized form plus ``norm2orig``
    for Q4 span -> original-text provenance.
    """
    if not sanitize:
        return utterance, None
    res = sanitize_utterance(utterance)
    res.check()
    return res.norm, NormalizedText(orig=utterance, norm=res.norm,
                                    norm2orig=res.norm2orig)


def build_input_multi(
    tools: Sequence[Tool],
    utterance: str,
    *,
    recent_context: Sequence[RecentRow] = (),
    now: date,
    time_window: Optional[Sequence[TimeEntry]] = None,
    window_days: Sequence[int] = (1, 2, 3, 7),
    weekday_tokens: Sequence[str] = ("next_mon", "this_fri"),
    idioms: Sequence[str] = ("tomorrow", "day_after_tomorrow", "tonight"),
    include_enum: bool = True,
    sanitize: bool = True,
) -> SpanTemplateOutput:
    """Render + parse the multi-tool packed template (pipeline_io P5, heads.md §2).

    Packs up to `len(tools)` <|tool|> blocks, a <|none|> anchor, shared context
    and utterance. Same sanitization and enum-pool validation as `build_input`.

    Args:
        now: system clock date for <|ctx_time|>. Required — deterministic
             rendering (no wall-clock fallback), same guarantee as `build_input`.
    """
    tools = list(tools)
    if not tools:
        raise ValueError("build_input_multi requires at least one tool")

    # P0/P1: sanitize the utterance (marker injection -> abstain).
    norm_utt, norm = _normalize(utterance, sanitize)

    # R7 also applies to recent-context rows (validated in RecentRow.__post_init__).
    validate_enums(tools)

    if time_window is None:
        time_window = build_time_window(now,
                                        window_days=window_days,
                                        weekdays=weekday_tokens,
                                        idioms=idioms)
    enums: List[str] = []
    if include_enum:
        enums = sorted({v for t in tools for p in t.params for v in p.enum})

    template = _env.get_template("template_multi.jinja")
    rendered = template.render(
        tools=tools,
        utterance=norm_utt,
        time_window=time_window,
        enums=enums,
        recent_context=recent_context,
    )

    parsed = parse(rendered)
    return SpanTemplateOutput(
        text=rendered,
        entries=parsed.entries,
        tool=tools[0],
        utterance=norm_utt,
        time_window=list(time_window),
        recent_context=list(recent_context),
        norm=norm,
        tools=tools,
    )


def tokenize_offset_mapping(
    output: SpanTemplateOutput,
    tokenizer: Any,
    max_len: Optional[int] = None,
) -> Tuple[List[int], List[Optional[Entry]]]:
    """P6 production path: tokenize with `return_offsets_mapping` + boundary check.

    Returns (token_ids, per_token_entry) where per_token_entry[i] is the legend
    entry whose char span the token is **wholly** inside, or None for
    bridge/delimiter tokens. Raises ``SanitizeError("bad_boundary")`` if any
    entry boundary falls mid-token — the template must be fixed, never the model
    silently mis-snapped (pipeline_io P6.3, Q4 provenance).

    ``max_len`` is the *optional* overflow cap. When ``None`` (the default) no
    tokenize-time cap is applied: the authoritative overflow guard is the
    resolver's backbone-driven ``L_max`` (``SpanCallResolver._encode``), so the
    template layer never hardcodes a window (pipeline_io P6.6, long-context fix).
    """
    enc = tokenizer(output.text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    per_token: List[Optional[Entry]] = [None] * len(ids)

    # ownership map: an entry claims ONLY its char_start. Registering char_end
    # too would mis-attribute a token starting on a shared boundary (entry A
    # ending exactly where entry B starts) to the ending entry, because
    # setdefault keeps the first writer.
    bounds: Dict[int, Entry] = {}
    for e in output.entries:
        bounds.setdefault(e.char_start, e)

    for i, (cs, ce) in enumerate(offsets):
        # a token is inside an entry iff it starts at an entry boundary
        if cs in bounds:
            per_token[i] = bounds[cs]

    # boundary-safety: every entry's char_start must coincide with a token start,
    # and char_end with a token end (no straddling BPE token).
    token_starts = {cs for cs, _ in offsets}
    token_ends = {ce for _, ce in offsets}
    for e in output.entries:
        if e.char_start not in token_starts or e.char_end not in token_ends:
            raise SanitizeError(
                "bad_boundary",
                f"entry {e.kind}@{e.char_start}:{e.char_end} straddles a token",
            )

    if max_len is not None and len(ids) > max_len:
        raise SanitizeError("overflow", f"{len(ids)} tokens > L_max={max_len}")
    return ids, per_token


# ---------------------------------------------------------------------------
# Whole-call gating (all-or-nothing, §3.4)
# ---------------------------------------------------------------------------


class GateResult:
    __slots__ = ("hit", "payload", "missing_required", "abstained")

    def __init__(self, hit: bool, payload: Dict[str, Any],
                 missing_required: Sequence[str] = (),
                 abstained: bool = True):
        self.hit = hit
        self.payload = payload
        self.missing_required = missing_required
        self.abstained = abstained

    def __repr__(self) -> str:
        return f"GateResult(hit={self.hit}, payload={self.payload}, " \
               f"missing_required={list(self.missing_required)})"


def gate_tool_call(
    output: SpanTemplateOutput,
    resolve: Callable[[Param], Tuple[bool, Any]],
) -> GateResult:
    """Decide, all-or-nothing, whether SpanCall can emit the full payload.

    Args:
        output: the rendered+parsed input (legend).
        resolve: a function param -> (resolved: bool, value). For a real system
                 this is the pointer-head mapping over `output.targetable()`;
                 for tests, a stance function.

    Returns:
        GateResult with hit=True when every REQUIRED param of every tool
        resolved; otherwise abstain (whole call) with the list of missing
        required params. For multi-tool packs (`build_input_multi`) this is a
        whole-call gate across ALL tools — no partial emission (pipeline_io Q8).
        Payload keys are namespaced ``<tool>.<param>`` when more than one tool
        is present, so same-named params across tools cannot collide.

    Schema defaults (Q6.2): an absent *optional* param carrying a declared
    `default` is materialized deterministically from the schema — the encoder
    never invents one. An absent optional without a default stays omitted.
    """
    # Whole-call gating (Q8): EVERY tool in the pack must fully resolve, or the
    # whole call abstains (no partial emission). `build_input` sets
    # tools=[tool]; `build_input_multi` sets tools=[t1, ...].
    tools = list(output.tools) or [output.tool]
    multi = len(tools) > 1
    payload: Dict[str, Any] = {}
    missing: List[str] = []
    for t in tools:
        for p in t.params:
            key = f"{t.name}.{p.name}" if multi else p.name
            ok, value = resolve(p)
            if not ok:
                if p.required:
                    missing.append(key)
                    continue
                elif p.has_default:
                    payload[key] = p.default      # deterministic schema default (Q6.2)
                else:
                    continue                       # optional + no default -> omit
            else:
                payload[key] = value
    if missing:
        return GateResult(hit=False, payload=payload, missing_required=missing, abstained=True)
    return GateResult(hit=True, payload=payload, abstained=False)


# ---------------------------------------------------------------------------
# Render-vs-spec consistency (proposal §3.1.1)
# ---------------------------------------------------------------------------

# The documented rendering contract (proposal §3.1.1) for the single-tool form.
# Used by the spec-diff test to guarantee doc <> code never drift.
RENDERING_CONTRACT: str = """<|cls|>
<|tool|> {{ tool.name }} ({{ tool.description }})
<|param|> {{ p.name }} ({{ p.description }})
<|sep|>
<|ctx_meta|> <|null|> | true | false | <|unresolved|>
<|ctx_time|> today={{ iso }} | +1d={{ iso }} | ...
<|sep|>
{{ utterance }}
<|sep|>
<|ctx_recent|> key="quoted value" | ...
<|eos|>"""