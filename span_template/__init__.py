"""SpanCall input template package (docs/proposal.md §3.1).

Renders `template.jinja` / `template_multi.jinja` (Jinja chat-like templates) and
parses the marker grammar back into the extraction legend — the *inverse* of a
chat template: text for the encoder + targets/values for the pointer head, from
one source. Sanitization (proposal R7, pipeline_io P0/P1) runs before render so
marker injection / empty / too-long inputs abstain deterministically.

Typical use:

    from span_template import Tool, Param, RecentRow, build_input

    out = build_input(
        tool=Tool("book_flight", "Book a selected flight",
                  params=[Param("flight_id", "Selected flight identifier"),
                          Param("email", "Notification email", optional=True)]),
        utterance="Book that flight, send the receipt to my work email",
        recent_context=[RecentRow("picked", "VN123", source="agent")],
        now=date(2026, 10, 23),
    )
    text     = out.text                      # what the encoder consumes
    targets  = list(out.targetable())        # candidate span values + offsets

    from span_template import gate_tool_call
    result = gate_tool_call(out, resolve_fn)  # all-or-nothing whole-call gate

Multi-tool packing (pipeline_io P5, heads.md §2):

    from span_template import build_input_multi
    out = build_input_multi([tool_a, tool_b], "search flights tomorrow")
"""

from .parser import Entry, ParseResult, parse
from .sanitize import (
    MARKER_TOKENS,
    SPECIAL_MARKER_TOKENS,
    SanitizeError,
    SanitizeResult,
    sanitize_utterance,
    check_recent_value,
)
from .template import (
    GateResult,
    NormalizedText,
    Param,
    RecentRow,
    SpanTemplateOutput,
    TimeEntry,
    Tool,
    build_input,
    build_input_multi,
    build_time_window,
    gate_tool_call,
    tokenize_offset_mapping,
    validate_enums,
)

__all__ = [
    "Entry",
    "GateResult",
    "MARKER_TOKENS",
    "SPECIAL_MARKER_TOKENS",
    "NormalizedText",
    "Param",
    "ParseResult",
    "RecentRow",
    "SanitizeError",
    "SanitizeResult",
    "SpanTemplateOutput",
    "TimeEntry",
    "Tool",
    "build_input",
    "build_input_multi",
    "build_time_window",
    "check_recent_value",
    "gate_tool_call",
    "parse",
    "sanitize_utterance",
    "tokenize_offset_mapping",
    "validate_enums",
]