"""Tests for lightweight plain-JSON HYBRID_FILL and HYBRID_FILL+SCHEMA evaluation."""

from __future__ import annotations

from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.eval.hybrid_fill import (
    build_hybrid_fill_request,
    finalize_hybrid_fill_row,
    parse_json_fill_text,
    plan_hybrid_fill,
)
from spancall.eval.llm_parity import LLMResponse


def _make_sample() -> UnifiedSample:
    scalar_tool = ToolDefinition(
        name="calculate_mortgage",
        description="Compute monthly payment",
        parameters=[
            ParameterSchema(name="interest_rate", param_type="number", description="APR", required=True),
            ParameterSchema(name="term", param_type="integer", description="Years", required=True),
            ParameterSchema(name="loan_amount", param_type="number", description="Principal", required=True),
        ],
    )
    complex_tool = ToolDefinition(
        name="filter_products",
        description="Filter catalog items",
        parameters=[
            ParameterSchema(name="category", param_type="string", description="Category", required=True),
            ParameterSchema(name="tags", param_type="array", description="Filter tags", required=True),
        ],
    )
    return UnifiedSample(
        sample_id="s1",
        source_dataset="hermes",
        utterance="Mortgage at 3.5% for 30 years on 500000",
        tools=[scalar_tool, complex_tool],
        ground_truth_calls=[
            GroundTruthCall(
                tool_name="calculate_mortgage",
                arguments={"interest_rate": 3.5, "term": 30, "loan_amount": 500000},
            )
        ],
        stratum=Stratum.F,
    )


def test_build_request_uses_plain_json_with_no_tools_wrapper() -> None:
    sample = _make_sample()
    lean_pred = {
        "sample_id": "s1",
        "hit": False,
        "tool_name": "calculate_mortgage",
        "payload_partial": {"interest_rate": 3.5, "term": 30},
        "calls": [{"tool_name": "calculate_mortgage", "missing_required": ["loan_amount"]}],
        "latency_ms": 50.0,
    }
    plan = plan_hybrid_fill(sample, lean_pred, mode="hybrid_fill")
    assert plan.action == "fill"
    assert plan.known == {"interest_rate": 3.5, "term": 30}
    assert plan.missing == ["loan_amount"]

    msgs, oai_tools = build_hybrid_fill_request(sample, plan)
    assert oai_tools == []  # NO OpenAI tools=[...] wrapper overhead!
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "loan_amount" in msgs[1]["content"]
    assert "calculate_mortgage" in msgs[1]["content"]


def test_parse_json_fill_text_handles_plain_and_fenced() -> None:
    assert parse_json_fill_text('{"loan_amount": 500000}') == {"loan_amount": 500000}
    assert parse_json_fill_text('```json\n{"tags": ["a", "b"]}\n```') == {"tags": ["a", "b"]}
    assert parse_json_fill_text('Here is the JSON: {"x": 1} done.') == {"x": 1}
    assert parse_json_fill_text("") is None


def test_finalize_merges_plain_json_conservatively() -> None:
    sample = _make_sample()
    lean_pred = {
        "sample_id": "s1",
        "hit": False,
        "tool_name": "calculate_mortgage",
        "payload_partial": {"interest_rate": 3.5, "term": 30},
        "calls": [{"tool_name": "calculate_mortgage", "missing_required": ["loan_amount"]}],
        "latency_ms": 50.0,
    }
    plan = plan_hybrid_fill(sample, lean_pred, mode="hybrid_fill")
    fill_resp = LLMResponse(
        text='{"interest_rate": 99.0, "loan_amount": 500000, "extra": "ignored"}',
        prompt_tokens=60,
        completion_tokens=10,
        latency_ms=150.0,
    )
    row = finalize_hybrid_fill_row(plan, lean_pred, fill_resp, full_llm_pred=None, model="test-llm")
    assert row["hit"] is True
    assert row["payload"] == {"interest_rate": 3.5, "term": 30, "loan_amount": 500000}
    assert row["latency_ms"] == 200.0
    assert row["routed_from"] == "hybrid_fill"
    assert row["features"]["filled_params"] == ["loan_amount"]


def test_hybrid_fill_schema_offloads_tool_and_keeps_scalars() -> None:
    sample = _make_sample()
    lean_pred = {
        "sample_id": "s1",
        "hit": True,
        "tool_name": "filter_products",
        "payload": {"category": "shoes", "tags": "wrong_scalar"},
        "payload_partial": {"category": "shoes", "tags": "wrong_scalar"},
        "calls": [{"tool_name": "filter_products", "missing_required": []}],
        "latency_ms": 40.0,
    }
    plan = plan_hybrid_fill(sample, lean_pred, mode="hybrid_fill_schema")
    assert plan.action == "fill"
    assert plan.known == {"category": "shoes"}
    assert plan.missing == ["tags"]

    fill_resp = LLMResponse(
        text='```json\n{"tags": ["running", "sale"]}\n```',
        prompt_tokens=70,
        completion_tokens=12,
        latency_ms=180.0,
    )
    row = finalize_hybrid_fill_row(plan, lean_pred, fill_resp, full_llm_pred=None, model="test-llm")
    assert row["hit"] is True
    assert row["payload"] == {"category": "shoes", "tags": ["running", "sale"]}
    assert row["routed_from"] == "hybrid_fill_schema"


def test_low_margin_gate_abstain_still_offloads_tool_selection_to_h1() -> None:
    sample = _make_sample()
    lean_pred = {
        "sample_id": "s1",
        "hit": False,
        "tool_name": "calculate_mortgage",
        "payload_partial": {"interest_rate": 3.5, "term": 30, "loan_amount": 500000},
        "calls": [{"tool_name": "calculate_mortgage", "missing_required": []}],
        "latency_ms": 50.0,
    }
    plan = plan_hybrid_fill(sample, lean_pred, mode="hybrid_fill")
    assert plan.action == "fill"
    assert plan.reason == "low_conf_json_fill"
    assert plan.missing == ["interest_rate", "term", "loan_amount"]


def test_single_shot_never_double_calls_full_llm_on_abstain_or_invalid() -> None:
    sample = _make_sample()
    lean_pred = {
        "sample_id": "s1",
        "hit": False,
        "tool_name": "calculate_mortgage",
        "payload_partial": {"interest_rate": 3.5, "term": 30},
        "calls": [{"tool_name": "calculate_mortgage", "missing_required": ["loan_amount"]}],
        "latency_ms": 50.0,
    }
    plan = plan_hybrid_fill(sample, lean_pred, mode="hybrid_fill")
    fill_resp = LLMResponse(text="{}", prompt_tokens=60, completion_tokens=5, latency_ms=120.0)
    full_llm = {
        "sample_id": "s1",
        "hit": True,
        "tool_name": "calculate_mortgage",
        "payload": {"interest_rate": 3.5, "term": 30, "loan_amount": 500000},
        "latency_ms": 500.0,
        "features": {"prompt_tokens": 400, "completion_tokens": 50},
    }
    row = finalize_hybrid_fill_row(plan, lean_pred, fill_resp, full_llm_pred=full_llm)
    assert row["hit"] is False
    assert row["routed_from"] == "hybrid_fill_abstain"
    assert row["latency_ms"] == 170.0
    assert row["features"]["prompt_tokens"] == 60
    assert row["features"]["completion_tokens"] == 5


def test_finalize_flattens_nested_json_when_missing_uses_dotted_keys() -> None:
    dotted_tool = ToolDefinition(
        name="analyze_site",
        description="Analyze site",
        parameters=[
            ParameterSchema(name="ctx.species", param_type="string", description="Species", required=True),
            ParameterSchema(name="ctx.note", param_type="string", description="Note", required=True),
        ],
    )
    sample = UnifiedSample(
        sample_id="s_nested",
        source_dataset="hermes",
        utterance="Northern Cardinals affected by noise",
        tools=[dotted_tool],
        ground_truth_calls=[],
        stratum=Stratum.F,
    )
    lean_pred = {
        "sample_id": "s_nested",
        "hit": False,
        "tool_name": "analyze_site",
        "payload_partial": {"ctx.species": "Northern Cardinals"},
        "calls": [{"tool_name": "analyze_site", "missing_required": ["ctx.note"]}],
        "latency_ms": 30.0,
    }
    plan = plan_hybrid_fill(sample, lean_pred, mode="hybrid_fill")
    fill_resp = LLMResponse(
        text='{"ctx": {"note": "noise"}}',
        prompt_tokens=50,
        completion_tokens=8,
        latency_ms=100.0,
    )
    row = finalize_hybrid_fill_row(plan, lean_pred, fill_resp, full_llm_pred=None)
    assert row["hit"] is True
    assert row["payload"] == {"ctx.species": "Northern Cardinals", "ctx.note": "noise"}
