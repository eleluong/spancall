"""Parser for the SpanCall marker grammar.

Parses the *rendered* output of `template.jinja` / `template_multi.jinja` back
into the extraction legend (entries with absolute char offsets, values, segment,
selectable). This is the inverse of rendering: render → text, parse ← text. Both
sides must agree on the markers defined in the templates; the grammar lives here.

Markers use the Qwen-native ``<|...|>`` convention — atomic special tokens
registered on the tokenizer (``span_template.sanitize.SPECIAL_MARKER_TOKENS``) so
a modern BPE never byte-splits a structural marker (docs/pipeline_io.md P6.3,
long-context R8 validation gate). The sanitizer's R7 injection regex also catches
the legacy bracketed ``[X]`` forms as reserved, but the *parser* only understands
the native ``<|X|>`` grammar rendered by the templates.

Entry kinds:
  - param_anchor / bridge_text : schema anchors (bridge, never target)
  - enum / date / bool / null / unresolved : CTX targets (selectable)
  - string : user utterance tokens or recent-context quoted values (selectable)
  - key : recent-context / time-window key (bridge, "mention -> memory" bridge)

For multi-tool packing (pipeline_io P5), every schema anchor is tagged with the
tool it belongs to (`Entry.tool`), so the pointer head can key queries by
``(tool, param)``.

Contract: for every entry, out.text[entry.char_start:entry.char_end] == entry.text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .sanitize import MARKER_TOKENS  # single authoritative vocabulary (R7, P6)

# Marker grammar — must match template.jinja / template_multi.jinja. The
# ``<|...|>`` forms are the Qwen-native atomic special tokens; the parser is
# deliberately strict here (it never falls back to the legacy ``[X]`` forms).
TOOL_RE = re.compile(r"<\|tool\|>\s+([^\s(]+)")
PARAM_RE = re.compile(r"<\|param\|>\s+([^\s(]+)")
NONE_RE = re.compile(r"<\|none\|>")
META_BODY_RE = re.compile(
    r"<\|ctx_meta\|>\s*(?P<body><\|(?:null|unresolved)\|>\s*"
    r"\|\s*true\s*\|\s*false\s*\|\s*<\|(?:null|unresolved)\|>)"
)
# Time-window keys (R4): allow a leading '+' for offset keys ("+1d") as well as
# plain identifiers ("today", "next_mon"). The builder's closed vocabulary (R4)
# bounds keys; the parser is permissive but exact on the date value.
TIME_ENTRY_RE = re.compile(r"(?P<key>\+?[A-Za-z0-9_.-]+)=(?P<value>\d{4}-\d{2}-\d{2})")
ENUM_BODY_RE = re.compile(r"<\|ctx_enum\|>\s*(?P<values>.*)$")
# A quoted pool entry: `"..."` with R1 escapes. An enum value is quoted only
# when it contains internal whitespace (D2/R5); `|` never appears even quoted.
_ENUM_QUOTED_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
RECENT_ENTRY_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)=(?P<value>\"(?:[^\"\\]|\\.)*\")")

_SEP = " | "


@dataclass
class Entry:
    """One atomic unit in the legend."""

    kind: str                    # param_anchor | bridge_text | enum | date | bool
                                 #   | null | unresolved | string | key
    segment: str                 # schema | ctx_meta | ctx_time | ctx_enum | utterance | recent
    value: object                # canonical value this span yields
    char_start: int
    char_end: int
    selectable: bool
    key: Optional[str] = None    # for key/value pairs (date/string): the key string
    text: str = ""
    tool: Optional[str] = None   # owning tool name (multi-tool packing, P5)

    def span(self) -> Tuple[int, int]:
        return (self.char_start, self.char_end)


@dataclass
class ParseResult:
    """Parsed legend + the raw text (should equal the render output)."""

    text: str
    entries: List[Entry] = field(default_factory=list)

    def targetable(self) -> List[Entry]:
        return [e for e in self.entries if e.selectable]

    def locate(self, offset: int, segment: Optional[str] = None) -> Optional[Entry]:
        for e in self.entries:
            if e.char_start <= offset < e.char_end and (segment is None or e.segment == segment):
                return e
        return None


def parse(text: str) -> ParseResult:
    """Parse a rendered SpanCall template into the legend (absolute offsets)."""
    result = ParseResult(text=text)
    line_offset = 0
    current_tool: Optional[str] = None
    for line in text.splitlines():
        m_tool = TOOL_RE.search(line)
        if m_tool:
            current_tool = m_tool.group(1)
        elif NONE_RE.search(line):
            # The <|none|> anchor belongs to no tool; everything after it
            # (shared [CTX] context, utterance, recent rows) is tool-less too.
            current_tool = None
        _parse_line(result, line, line_offset, current_tool)
        line_offset += len(line) + 1  # +1 for the '\n'
    return result


def _make_add(result: ParseResult, line_offset: int, tool: Optional[str]):
    """Build an add helper that records entries with ABSOLUTE offsets.

    text must be the raw substring of the line (so the char-offset contract
    holds); value is the canonical value (may differ, e.g. unescaped).
    """
    def add(rel_start: int, text: str, value: object, segment: str,
            selectable: bool, key: Optional[str] = None,
            kind: str = "string") -> None:
        start = line_offset + rel_start
        end = start + len(text)
        result.entries.append(Entry(
            kind=kind, segment=segment, value=value,
            char_start=start, char_end=end, selectable=selectable,
            key=key, text=text, tool=tool,
        ))
    return add


def _parse_line(result: ParseResult, line: str, line_offset: int,
                tool: Optional[str]) -> None:
    """Parse one line of rendered text into entries."""
    stripped = line.strip()
    if not stripped:
        return  # blank lines carry no legend entries (defensive: no zero-length)
    add = _make_add(result, line_offset, tool)

    # Special marker lines — always bridges.
    if stripped.startswith("<|cls|>"):
        return
    if stripped.startswith("<|sep|>") or stripped.startswith("<|eos|>"):
        return

    # Tool line: "<|tool|> name (desc)" — bridge anchor tagged with the tool.
    m = TOOL_RE.search(line)
    if m:
        add(m.start(1), m.group(1), m.group(1), "schema", False,
            kind="bridge_text")
        return

    # "<|none|>" anchor line — bridge (multi-tool packing, pipeline_io P5).
    if NONE_RE.search(line):
        for m in NONE_RE.finditer(line):
            add(m.start(), m.group(0), "<|none|>", "schema", False, kind="param_anchor")
        return

    # Param line(s): "<|param|> name (desc)" — question anchors (bridge).
    if PARAM_RE.search(line):
        for m in PARAM_RE.finditer(line):
            add(m.start(1), m.group(1), m.group(1), "schema", False,
                kind="param_anchor")
        return

    # CTX_META line: "<|ctx_meta|> <|null|> | true | false | <|unresolved|>"
    m = META_BODY_RE.search(line)
    if m:
        body = m.group("body")
        base = line.index(body)
        for token in body.split(_SEP):
            token = token.strip()
            if token.startswith("<|") and token.endswith("|>"):
                kind = "null" if token == "<|null|>" else "unresolved"
                idx = line.find(token, base)
                if idx >= 0:
                    add(idx, token, token, "ctx_meta", True, kind=kind)
                base = idx + len(token) if idx >= 0 else base
            elif token in ("true", "false"):
                idx = line.find(token, base)
                if idx >= 0:
                    add(idx, token, token, "ctx_meta", True, kind="bool")
                base = idx + len(token) if idx >= 0 else base
        return

    # CTX_TIME line: "<|ctx_time|> key=2026-10-23 | ..."
    if stripped.startswith("<|ctx_time|>"):
        body = line[len("<|ctx_time|>"):]
        base = line.index("<|ctx_time|>") + len("<|ctx_time|>")
        for m in TIME_ENTRY_RE.finditer(body):
            key, value = m.group("key"), m.group("value")
            idx = line.find(m.group(0), base)
            if idx < 0:
                continue
            add(idx, key, key, "ctx_time", False, key=key, kind="key")
            add(idx + len(key) + 1, value, value, "ctx_time", True, key=key, kind="date")
        return

    # CTX_ENUM line: "<|ctx_enum|> v1 | "v2 with space" | ..."
    m = ENUM_BODY_RE.search(line)
    if m:
        base = line.index(m.group(0))
        body = m.group("values")
        idx = 0
        n = len(body)
        while idx < n:
            # strip the ' | ' separators (and any stray whitespace): skip
            # spaces, then an optional '|', then spaces again.
            while idx < n and body[idx] == " ":
                idx += 1
            if idx < n and body[idx] == "|":
                idx += 1
                while idx < n and body[idx] == " ":
                    idx += 1
            if idx >= n:
                break
            q = _ENUM_QUOTED_RE.match(body, idx)
            if q:
                raw = q.group(0)
                rel = line.find(raw, base)
                add(rel, raw, _unescape(raw), "ctx_enum", True, kind="enum")
                base = rel + len(raw)
                idx = q.end()
                continue
            # plain token: up to the next separator boundary
            end = body.find(_SEP, idx)
            if end < 0:
                end = n
            raw = body[idx:end].strip()
            rel = line.find(raw, base)
            if rel >= 0:
                add(rel, raw, raw, "ctx_enum", True, kind="enum")
            base = rel + len(raw) if rel >= 0 else base
            idx = end
        return

    # User utterance line — all tokens targetable (selectable).
    if not stripped.startswith("<|ctx_recent|>"):
        add(0, stripped, stripped, "utterance", True, kind="string")
        return

    # CTX_RECENT line: "<|ctx_recent|> key="escaped value" | ..."
    if stripped.startswith("<|ctx_recent|>"):
        body = line[len("<|ctx_recent|>"):]
        base = line.index("<|ctx_recent|>") + len("<|ctx_recent|>")
        for m in RECENT_ENTRY_RE.finditer(body):
            key, quoted = m.group("key"), m.group("value")
            idx = line.find(m.group(0), base)
            if idx < 0:
                continue
            add(idx, key, key, "recent", False, key=key, kind="key")
            # value interior (skip '=' and the surrounding quotes)
            # key spans [idx, idx+len(key)); '=' at idx+len(key); '"' at +1
            interior_start = idx + len(key) + 2
            raw = quoted[1:-1]
            add(interior_start, raw, _unescape(quoted), "recent", True,
                key=key, kind="string")
        return


def _unescape(quoted: str) -> str:
    return quoted[1:-1].replace('\\"', '"').replace("\\\\", "\\")