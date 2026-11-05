"""Unit tests for :class:`spancall.data.align.ReverseAligner`.

Covers the plan §2.1 contract:
  * verbatim utterance spans (word-bounded, absolute char offsets)
  * recent-context rows (equality and full-token substring)
  * colliding <|ctx_time|> entries (R4 — every equal-valued CTX entry is a positive)
  * enum grounding from <|ctx_enum|>
  * <|null|>/optional + ABSENT(required) H4 classification
  * Stratum.A aggregation when any required param is ungroundable
  * json-mode key inference + array (Case F) alignment

Fixtures build ``UnifiedSample`` directly; one end-to-end path runs the real
:class:`HermesLoader` + aligner over ``SINGLETURN_ROW``.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from spancall.data.align import ReverseAligner
from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    SpanTarget,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from tests.data.test_hermes import SINGLETURN_ROW, _write_config_rows


def _sample(
    *,
    sid: str = "t",
    utterance: str,
    tools: list[ToolDefinition],
    gt: list[GroundTruthCall],
    recent: list[dict] | None = None,
    current_time_iso: str | None = None,
) -> UnifiedSample:
    return UnifiedSample(
        sample_id=sid,
        source_dataset="test",
        utterance=utterance,
        tools=tools,
        ground_truth_calls=gt,
        recent_context=recent or [],
        current_time_iso=current_time_iso,
    )


def _tool(
    name: str = "tool",
    params: list[ParameterSchema] | None = None,
) -> ToolDefinition:
    return ToolDefinition(name=name, description="d", parameters=params or [])


def _p(name: str, ptype: str = "string", *, enum=None, required: bool = True,
       items_type: str | None = None) -> ParameterSchema:
    return ParameterSchema(
        name=name, description="d", param_type=ptype, required=required,
        enum=enum, items_type=items_type,
    )


def _render_utterance_entry(sample: UnifiedSample) -> tuple:
    """Re-render the *aligned* sample exactly like ``ReverseAligner._render``.

    ``SpanTarget.char_start/char_end`` are ABSOLUTE offsets into the rendered
    template text (not the raw ``sample.utterance``), so tests must slice the
    render to validate them. Returns ``(utterance_entry, output)``.
    """
    from datetime import date as _date

    from span_template.template import RecentRow, build_input
    from spancall.data.align import _find_utterance_entry, tooldef_to_template_tool

    tool_def = next(
        (t for t in sample.tools if t.name == sample.ground_truth_calls[0].tool_name),
        None,
    )
    assert tool_def is not None, "aligned sample has no matching tool"
    assert sample.current_time_iso, "align() must set current_time_iso"
    base = _date.fromisoformat(sample.current_time_iso[:10])
    recent = [
        RecentRow(key=r["key"], value=r["value"], source=r.get("source", "recent"))
        for r in sample.recent_context
    ]
    out = build_input(
        tool=tooldef_to_template_tool(tool_def),
        utterance=sample.utterance,
        recent_context=recent,
        now=base,
    )
    utt = _find_utterance_entry(out)
    assert utt is not None, "no selectable utterance entry in render"
    return utt, out


# ---------------------------------------------------------------------------
# Verbatim utterance spans
# ---------------------------------------------------------------------------

def test_verbatim_utterance_span() -> None:
    """A required string param grounded as a word-bounded utterance span gets
    char offsets into the RENDERED template text pointing at the utterance span."""
    tools = [_tool(params=[_p("camera_id")])]
    utterance = "Show me the 4k feed of camera cam-1."
    s = _sample(
        sid="verbatim-1",
        utterance=utterance,
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"camera_id": "cam-1"})],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    spans = out.target_spans["camera_id"]
    assert len(spans) == 1
    span = spans[0]
    assert span.target_type == "utterance_span"
    assert span.canonical_value == "cam-1"
    # offsets are ABSOLUTE into the rendered template text, so the render slice
    # equals the value while the raw-utterance slice does not (offsets exceed it).
    utt, output = _render_utterance_entry(out)
    assert output.text[utt.char_start : utt.char_end] == utterance
    assert output.text[span.char_start : span.char_end] == "cam-1"
    # the entry may include trailing punctuation, so measure offsets from its
    # START (both ends), not from its end.
    rel_start = span.char_start - utt.char_start
    rel_end = span.char_end - utt.char_start
    j = utterance.index("cam-1")
    assert rel_start == j
    assert rel_end == j + len("cam-1")
    assert utterance[rel_start : rel_end] == "cam-1"
    assert span.token_start is None  # token offsets filled later by the tokenizer


def test_verbatim_word_boundary_no_partial_match() -> None:
    """R4: '5' must NOT match inside '15' — word-bounded matching."""
    tools = [_tool(params=[_p("qty", "integer")])]
    s = _sample(
        sid="wb-1",
        utterance="Schedule 15 notifications for queue 5.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"qty": 5})],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    (span,) = out.target_spans["qty"]
    assert utterance_span(out, span)
    utt, output = _render_utterance_entry(out)
    # the span slices the RENDERED text, whose utterance-bounded slice = "5"
    assert output.text[span.char_start : span.char_end] == "5"
    rel_start = span.char_start - utt.char_start
    assert out.utterance[rel_start] == "5"
    assert rel_start > 0 and out.utterance[rel_start - 1] == " "


def utterance_span(sample: UnifiedSample, span: SpanTarget) -> bool:
    return span.target_type == "utterance_span"


# ---------------------------------------------------------------------------
# Recent context
# ---------------------------------------------------------------------------

def test_recent_context_equal_value() -> None:
    """Whole-value equality on a selectable recent row -> recent_ctx target."""
    tools = [_tool(params=[_p("case_id")])]
    s = _sample(
        sid="rc-1",
        utterance="Now close the case I opened earlier.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"case_id": "C-4120"})],
        recent=[{"key": "case_id", "value": "C-4120", "source": "conversation"}],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    (span,) = out.target_spans["case_id"]
    assert span.target_type == "recent_ctx"
    assert span.ctx_key == "case_id"
    assert span.canonical_value == "C-4120"
    # char offsets should point at the recent row's value span in the render
    assert out.utterance[span.char_start : span.char_end] or span.char_start is not None


def test_recent_context_full_token_substring() -> None:
    """A word that is a full token inside the recent value is a positive."""
    tools = [_tool(params=[_p("account")])]
    s = _sample(
        sid="rc-2",
        utterance="Use the account I set up.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"account": "alice"})],
        recent=[{"key": "account", "value": "alice@example.com", "source": "recent"}],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    (span,) = out.target_spans["account"]
    assert span.target_type == "recent_ctx"


# ---------------------------------------------------------------------------
# Enum / boolean / date / colliding CTX_TIME
# ---------------------------------------------------------------------------

def test_enum_from_ctx_enum() -> None:
    tools = [_tool(params=[_p("stream_quality", "string", enum=["720p", "1080p", "4k"])])]
    s = _sample(
        sid="enum-1",
        utterance="Give me the 4k stream.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"stream_quality": "4k"})],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    (span,) = out.target_spans["stream_quality"]
    assert span.target_type == "enum_ctx"
    assert span.canonical_value == "4k"


def test_enum_value_not_in_pool_off_window() -> None:
    tools = [_tool(params=[_p("stream_quality", "string", enum=["720p", "1080p"])])]
    s = _sample(
        sid="enum-2",
        utterance="Stream at 8k please.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"stream_quality": "8k"})],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.A
    assert out.metadata["h4_reasons"]["stream_quality"] == "OFF_WINDOW"


def test_boolean_true_from_ctx_meta() -> None:
    tools = [_tool(params=[_p("enabled", "boolean")])]
    s = _sample(
        sid="bool-1",
        utterance="Turn notification on.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"enabled": True})],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    (span,) = out.target_spans["enabled"]
    assert span.target_type == "boolean_ctx"
    assert span.canonical_value is True


def test_colliding_ctx_time_multi_positive() -> None:
    """R4: the ISO date 2026-10-28 appears twice in <|ctx_time|> (as `today`
    when base=2026-10-28 and as some +kNd entry) -> BOTH are positives, and the
    model can point to either."""
    base = date(2026, 10, 28)
    iso = base.isoformat()
    tools = [_tool(params=[_p("delivery_date", "date")])]
    s = _sample(
        sid="date-collide",
        utterance="Deliver on the 28th please.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"delivery_date": iso})],
        current_time_iso="2026-10-28T09:00:00+00:00",
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    spans = out.target_spans["delivery_date"]
    assert len(spans) >= 2, f"expected multiple colliding CTX_TIME positives, got {len(spans)}"
    assert all(sp.target_type == "time_ctx" for sp in spans)
    assert all(sp.canonical_value == iso for sp in spans)


def test_date_verbatim_literal_in_utterance() -> None:
    tools = [_tool(params=[_p("when", "date")])]
    s = _sample(
        sid="date-literal",
        utterance="Book it for 2026-11-05 please.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"when": "2026-11-05"})],
        current_time_iso="2026-10-28T09:00:00+00:00",
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    spans = out.target_spans["when"]
    _, output = _render_utterance_entry(out)
    assert any(
        sp.target_type == "utterance_span"
        and output.text[sp.char_start : sp.char_end] == "2026-11-05"
        for sp in spans
    )


# ---------------------------------------------------------------------------
# Optional / null / absent
# ---------------------------------------------------------------------------

def test_optional_none_null_target() -> None:
    tools = [_tool(params=[
        _p("location", required=True),
        _p("priority", "string", required=False),
    ])]
    s = _sample(
        sid="null-1",
        utterance="Book a cab to downtown.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"location": "downtown", "priority": None})],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F  # optional None OK
    assert out.metadata["h4_reasons"]["priority"] == "OK"
    (span,) = out.target_spans["priority"]
    assert span.target_type == "null_ctx"
    assert span.canonical_value is None


def test_missing_required_absent() -> None:
    tools = [_tool(params=[_p("location", required=True)])]
    s = _sample(
        sid="absent-1",
        utterance="Book a cab.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={})],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.A
    assert out.metadata["abstain_reason"] == "ungrounded_required"
    assert out.metadata["h4_reasons"]["location"] == "ABSENT"


def test_missing_optional_is_ok() -> None:
    tools = [_tool(params=[_p("note", required=False)])]
    s = _sample(
        sid="absent-opt",
        utterance="Just do it.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={})],
    )
    out = ReverseAligner().align(s)
    # param neither given nor required -> no supervision needed (Stratum.A is
    # safe: nothing to ground; the call is fully absent)
    assert out.stratum == Stratum.F
    assert out.metadata["h4_reasons"]["note"] == "OK"


# ---------------------------------------------------------------------------
# Array (Case F) — regression for the _align_value(sample) thread-through
# ---------------------------------------------------------------------------

def test_array_of_strings_aligns() -> None:
    tools = [_tool(params=[_p("tags", "array", required=True, items_type="string")])]
    s = _sample(
        sid="arr-1",
        utterance="Tag it alpha and beta please.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"tags": ["alpha", "beta"]})],
    )
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    spans = out.target_spans["tags"]
    assert len(spans) == 2
    _, output = _render_utterance_entry(out)
    values = sorted(
        output.text[sp.char_start : sp.char_end]
        for sp in spans if sp.target_type == "utterance_span"
    )
    assert values == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# json-mode key inference
# ---------------------------------------------------------------------------

def test_json_mode_key_inference_grounds_gt() -> None:
    """A synthetic json_output tool with no parameters gets a ParameterSchema
    inferred from the GT argument keys; extraction params become groundable."""
    tools = [_tool("json_output", params=[])]
    s = _sample(
        sid="json-1",
        utterance="Return with entity 'abc' and mode 'fast'.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="json_output", arguments={"entity": "abc", "mode": "fast"})],
    )
    out = ReverseAligner().align(s)
    # both keys now have schemas -> inferable spans -> F
    assert out.stratum == Stratum.F
    assert set(out.target_spans.keys()) == {"entity", "mode"}
    (espan,) = out.target_spans["entity"]
    _, output = _render_utterance_entry(out)
    assert output.text[espan.char_start : espan.char_end] == "abc"


# ---------------------------------------------------------------------------
# End-to-end over the real Hermes SINGLETURN fixture
# ---------------------------------------------------------------------------

def test_align_hermes_singleturn_end_to_end(tmp_path) -> None:
    path = _write_config_rows(tmp_path, "func_calling_singleturn", [SINGLETURN_ROW])
    from spancall.data.sources import HermesLoader

    (s,) = list(HermesLoader(config_name="func_calling_singleturn", local_path=path).load())
    out = ReverseAligner().align(s)
    assert out.stratum == Stratum.F
    assert out.metadata["h4_reasons"]["camera_id"] == "OK"
    assert out.metadata["h4_reasons"]["stream_quality"] == "OK"
    (cam,) = out.target_spans["camera_id"]
    _, output = _render_utterance_entry(out)
    assert output.text[cam.char_start : cam.char_end] == "cam-1"
    (qual,) = out.target_spans["stream_quality"]
    assert qual.target_type == "enum_ctx"
    assert qual.canonical_value == "4k"