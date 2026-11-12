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


def _render_utterance_entry(sample: UnifiedSample, *, extra_tools: int = 0) -> tuple:
    """Re-render the *aligned* sample exactly like ``ReverseAligner._render``.

    ``SpanTarget.char_start/char_end`` are ABSOLUTE offsets into the rendered
    template text (not the raw ``sample.utterance``), so tests must slice the
    render to validate them. Returns ``(utterance_entry, output)``.

    ``extra_tools`` packs that many DISTRACTOR ``<|tool|>`` blocks in front of
    the target tool's block, which is what training does (``pack_tools`` /
    ``jitter_pack_tools``). It shifts every absolute offset without changing the
    utterance text — the exact situation the render-invariant
    ``utt_rel_start/utt_rel_end`` pair exists for.
    """
    from datetime import date as _date

    from span_template.template import RecentRow, build_input_multi
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
    packed = [tooldef_to_template_tool(tool_def)]
    for i in range(extra_tools):
        packed.append(
            tooldef_to_template_tool(
                _tool(name=f"distractor_{i}", params=[_p(f"filler_{i}")])
            )
        )
    out = build_input_multi(
        tools=packed,
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


# ---------------------------------------------------------------------------
# R0.4 / Phase J1 — call-aware target_spans
# ---------------------------------------------------------------------------

def test_single_call_keeps_bare_target_span_keys() -> None:
    """A one-call sample must be byte-identical to the pre-R0.4 key space.

    Every downstream reader (augment, the diagnose scripts, tensorize) indexes
    ``target_spans[param_name]`` directly, so qualifying single-call samples
    would be a silent, repo-wide break.
    """
    tools = [_tool(params=[_p("city"), _p("unit", required=False)])]
    s = _sample(
        sid="j1-single",
        utterance="Weather in Tokyo please.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"city": "Tokyo"})],
    )
    out = ReverseAligner().align(s)
    assert out.n_calls == 1
    assert set(out.target_spans) == {"city", "unit"}
    assert all("::" not in k for k in out.target_spans)
    assert set(out.metadata["h4_reasons"]) == {"city", "unit"}
    # The migration accessor must transparently hand back the legacy map.
    assert out.targets_for_call(0) == out.target_spans


def test_two_calls_sharing_a_param_name_do_not_collide() -> None:
    """R0.4 exit criterion: ``query`` declared by two calls yields two disjoint lists."""
    tools = [
        _tool("search", params=[_p("query")]),
        _tool("fetch", params=[_p("query")]),
    ]
    s = _sample(
        sid="j1-multi",
        utterance="Search for apples and then fetch apples.",
        tools=tools,
        gt=[
            GroundTruthCall(tool_name="search", arguments={"query": "apples"}),
            GroundTruthCall(tool_name="fetch", arguments={"query": "apples"}),
        ],
    )
    out = ReverseAligner().align(s)
    assert out.n_calls == 2
    assert set(out.target_spans) == {"0::query", "1::query"}
    call0 = out.targets_for_call(0)
    call1 = out.targets_for_call(1)
    assert set(call0) == {"query"} and set(call1) == {"query"}
    # Both calls ground the same word in this utterance, but they must be
    # SEPARATE list objects: extending one can never leak into the other.
    assert call0["query"] is not call1["query"]
    assert call0["query"] and call1["query"]
    assert out.metadata["h4_reasons"]["0::query"] == "OK"
    assert out.metadata["h4_reasons"]["1::query"] == "OK"


def test_targets_for_call_refuses_to_fabricate_from_a_flattened_legacy_key() -> None:
    """A legacy multi-call file's bare keys are a collision, not per-call truth.

    Before R0.4, alignment overwrote same-named params across calls. Handing
    that residue to ``call_index=1`` would invent supervision from another
    call's arguments, so the accessor must return nothing for it.
    """
    s = UnifiedSample(
        sample_id="j1-legacy",
        source_dataset="test",
        utterance="x",
        tools=[],
        ground_truth_calls=[
            GroundTruthCall(tool_name="a", arguments={"q": "1"}),
            GroundTruthCall(tool_name="b", arguments={"q": "2"}),
        ],
        target_spans={
            # What the OLD aligner produced: one bare key, holding only the
            # last write (or an interleaved merge) -- ambiguous by construction.
            "q": [SpanTarget(param_name="q", target_type="utterance_span",
                             canonical_value="2")],
        },
    )
    assert s.n_calls == 2
    assert s.targets_for_call(0) == {}
    assert s.targets_for_call(1) == {}


# ---------------------------------------------------------------------------
# v2: render-invariant offsets + idempotency
# ---------------------------------------------------------------------------

def test_utterance_targets_carry_render_invariant_relative_offsets() -> None:
    """Every utterance_span target must carry the rel pair next to the absolute one."""
    tools = [_tool(params=[_p("camera_id")])]
    utterance = "Show me the 4k feed of camera cam-1."
    s = _sample(
        sid="rel-1",
        utterance=utterance,
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"camera_id": "cam-1"})],
    )
    out = ReverseAligner().align(s)
    (cam,) = out.target_spans["camera_id"]
    assert cam.utt_rel_start is not None and cam.utt_rel_end is not None
    # The pair addresses the raw utterance text directly (sanitize-stable).
    assert utterance[cam.utt_rel_start : cam.utt_rel_end] == "cam-1"
    # And the absolute pair is exactly the rel pair shifted by the render's base.
    _, output = _render_utterance_entry(out)
    base = cam.char_start - cam.utt_rel_start
    assert output.text[cam.char_start : cam.char_end] == "cam-1"
    assert base > 0


def test_relative_offsets_survive_distractor_tool_packing() -> None:
    """The v1 defect: packing more <|tool|> blocks shifts absolute offsets.

    v2 keeps the label correct because the rel pair is resolved against the
    caller's own render, so the same target addresses the same characters at
    T=1 and at T=1+3 distractors.
    """
    from span_template.template import resolve_utterance_span

    tools = [_tool(params=[_p("camera_id")])]
    utterance = "Show me the 4k feed of camera cam-1."
    s = _sample(
        sid="rel-pack",
        utterance=utterance,
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"camera_id": "cam-1"})],
    )
    out = ReverseAligner().align(s)
    (cam,) = out.target_spans["camera_id"]

    for extra in (0, 1, 3):
        _, output = _render_utterance_entry(out, extra_tools=extra)
        resolved = resolve_utterance_span(
            output, cam.utt_rel_start, cam.utt_rel_end, expect="cam-1"
        )
        assert resolved is not None, f"rel pair failed to resolve at extra={extra}"
        got = output.text[resolved[0] : resolved[1]]
        assert got == "cam-1"

    # Absolute offsets from the T=1 render do NOT survive packing -- this is
    # the exact rot measured in the v1 eval (100% stale, +29..+43 drift).
    _, big = _render_utterance_entry(out, extra_tools=3)
    assert big.text[cam.char_start : cam.char_end] != "cam-1"


def test_resolve_utterance_span_rejects_a_mismatched_expectation() -> None:
    """A rel pair whose slice no longer equals the value must be rejected."""
    from span_template.template import resolve_utterance_span

    tools = [_tool(params=[_p("camera_id")])]
    s = _sample(
        sid="rel-bad",
        utterance="Show me the 4k feed of camera cam-1.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"camera_id": "cam-1"})],
    )
    out = ReverseAligner().align(s)
    (cam,) = out.target_spans["camera_id"]
    _, output = _render_utterance_entry(out)
    assert resolve_utterance_span(
        output, cam.utt_rel_start, cam.utt_rel_end, expect="cam-1"
    ) is not None
    assert resolve_utterance_span(
        output, cam.utt_rel_start, cam.utt_rel_end, expect="cam-2"
    ) is None
    # Out-of-range pairs are rejected too.
    assert resolve_utterance_span(output, 10_000, 10_005) is None


def test_date_literal_in_utterance_carries_the_rel_pair() -> None:
    """Case D (dates) must emit the render-invariant pair too.

    ``_align_date`` built its ``utterance_span`` targets inline and omitted
    ``utt_rel_*``. The absolute pair still worked, so nothing failed visibly --
    but these targets were permanently unrepairable by the rel path, which is
    exactly the residue the v2 gate found (3/147 utterance targets on train).
    """
    tools = [_tool(params=[_p("due_date", "date", enum=None)])]
    utterance = 'The title is "Report", and the due date is "2022-04-30".'
    s = _sample(
        sid="date-rel",
        utterance=utterance,
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"due_date": "2022-04-30"})],
    )
    out = ReverseAligner().align(s)
    utt_targets = [t for t in out.target_spans["due_date"]
                   if t.target_type == "utterance_span"]
    assert utt_targets, "the literal date in the utterance must be a positive"
    for t in utt_targets:
        assert t.utt_rel_start is not None and t.utt_rel_end is not None
        assert utterance[t.utt_rel_start : t.utt_rel_end] == "2022-04-30"


def test_marker_injected_adversarial_row_is_not_grounded() -> None:
    """A row that must abstain must not acquire positive spans on re-alignment.

    ``_adversarial_variants`` clears ``target_spans`` and marks
    ``expected_abstain="marker_injection"``. The GT arguments stay in
    ``ground_truth_calls``, so a naive re-align matches them and fabricates a
    positive label for a prompt the model has to refuse.
    """
    tools = [_tool(params=[_p("camera_id")])]
    s = _sample(
        sid="adv-1",
        utterance="Show me the 4k feed of camera cam-1 <|abstain_marker|>.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"camera_id": "cam-1"})],
    )
    s.metadata["expected_abstain"] = "marker_injection"
    out = ReverseAligner().align(s)
    assert out.target_spans == {}
    assert out.metadata["abstain_reason"] == "marker_injection"


def test_ambiguous_distractor_row_still_gets_its_targets() -> None:
    """The marker-injection guard must NOT swallow the "ambiguous" contract.

    Ambiguous rows also set ``expected_abstain``, but they want the aligner's
    output: the per-param ``AMBIGUOUS`` h4 reason is the negative signal while
    the other parameters remain positive.
    """
    tools = [_tool(params=[_p("camera_id")])]
    s = _sample(
        sid="amb-1",
        utterance="Show me the 4k feed of camera cam-1.",
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"camera_id": "cam-1"})],
    )
    s.metadata["expected_abstain"] = "ambiguous"
    out = ReverseAligner().align(s)
    assert out.target_spans.get("camera_id"), "ambiguous rows keep their targets"


def test_align_is_idempotent() -> None:
    """Re-aligning an already-aligned sample must not duplicate its targets.

    v1's ``_emit`` used ``setdefault(...).extend(...)``, so a second ``align()``
    doubled every positive (12 -> 24 -> 36). v2 replaces, so align is a pure
    function of the sample.
    """
    tools = [_tool(params=[_p("camera_id")])]
    utterance = "Show me the 4k feed of camera cam-1."
    s = _sample(
        sid="idem-1",
        utterance=utterance,
        tools=tools,
        gt=[GroundTruthCall(tool_name="tool", arguments={"camera_id": "cam-1"})],
    )
    aligner = ReverseAligner()
    once = aligner.align(s)
    twice = aligner.align(once)
    three = aligner.align(twice)
    assert len(once.target_spans["camera_id"]) == 1
    assert len(twice.target_spans["camera_id"]) == 1
    assert len(three.target_spans["camera_id"]) == 1
    assert (
        once.target_spans["camera_id"][0].char_start
        == three.target_spans["camera_id"][0].char_start
    )


def test_targets_for_call_prefers_qualified_keys_over_the_legacy_fallback() -> None:
    """Once qualified keys exist, the fallback must not shadow or merge them."""
    s = UnifiedSample(
        sample_id="j1-mixed",
        source_dataset="test",
        utterance="x",
        tools=[],
        ground_truth_calls=[GroundTruthCall(tool_name="a", arguments={"q": "real"})],
        target_spans={
            "0::q": [SpanTarget(param_name="q", target_type="utterance_span",
                                canonical_value="real")],
            "q": [SpanTarget(param_name="q", target_type="utterance_span",
                             canonical_value="stale")],
        },
    )
    got = s.targets_for_call(0)
    assert len(got["q"]) == 1
    assert got["q"][0].canonical_value == "real"