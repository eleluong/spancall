"""Tests for Phase 3 & Phase 4 long-context agent loop and trajectory grounding.

Validates:
- flatten_tool_output: JSON and nested dict/list decomposition into leaf rows.
- rows_from_messages: trajectory conversion into R6 rows with active utterance excluded.
- select_recent_context: R7 rejection, deduplication, relevance scoring, and whole-row eviction.
- trim_messages: atomic turn grouping (tool_calls paired with tool responses) and token budget.
- ReActAgentLoop multi-turn trajectory grounding: mid-turn SpanCall hit using prior tool output.
- Defense-in-depth ContextBudgetError on backends.
- 10k-token transcript safety.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pytest

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.backends.litellm_adapter import UniversalAPIBackend
from spancall.agent.context import (
    ContextBudgetError,
    flatten_tool_output,
    latest_user_utterance,
    rows_from_messages,
    select_recent_context,
    trim_messages,
)
from spancall.agent.loop import ReActAgentLoop
from spancall.agent.router import IntegrationVariant, RouterMode, ToolCallRouter
from spancall.agent.tools.base import BaseToolEnvironment, ToolExecutionResult
from spancall.agent.trace import ExecutionTrace, RoutingDecision
from spancall.data.schema import ParameterSchema, ToolDefinition
from spancall.infer.base import ResolveResult


# --------------------------------------------------------------------------- #
# Unit tests: flatten_tool_output
# --------------------------------------------------------------------------- #
def test_flatten_tool_output_nested_json() -> None:
    data = {
        "flights": [
            {"flight_id": "VN123", "price": 250},
            {"flight_id": "VN456", "price": 300},
        ],
        "status": "ok",
    }
    rows = flatten_tool_output(0, "search_flights", data)
    keys = [r["key"] for r in rows]
    vals = [r["value"] for r in rows]

    assert "t0_search_flights_flights.0.flight_id" in keys
    assert "VN123" in vals
    assert "t0_search_flights_flights.0.price" in keys
    assert "250" in vals
    assert "t0_search_flights_flights.1.flight_id" in keys
    assert "VN456" in vals
    assert "t0_search_flights_status" in keys
    assert "ok" in vals
    for r in rows:
        assert r["source"] == "tool_output"


def test_flatten_tool_output_string_json() -> None:
    raw_json = '{"user": {"id": "U789", "name": "Alice"}}'
    rows = flatten_tool_output(1, "lookup_user", raw_json)
    keys = {r["key"]: r["value"] for r in rows}
    assert keys.get("t1_lookup_user_user.id") == "U789"
    assert keys.get("t1_lookup_user_user.name") == "Alice"


def test_flatten_tool_output_unstructured_string_no_vmax_cap() -> None:
    long_text = "Detailed error diagnostic report: " + ("x" * 2000)
    rows = flatten_tool_output(2, "diagnostic", long_text)
    assert len(rows) == 1
    assert rows[0]["key"] == "t2_diagnostic"
    assert rows[0]["value"] == long_text


# --------------------------------------------------------------------------- #
# Unit tests: rows_from_messages & latest_user_utterance
# --------------------------------------------------------------------------- #
def test_latest_user_utterance() -> None:
    msgs = [
        {"role": "user", "content": "Initial request"},
        {"role": "assistant", "content": "Thinking..."},
        {"role": "tool", "content": '{"result": 1}'},
        {"role": "user", "content": "Follow-up question"},
    ]
    assert latest_user_utterance(msgs) == "Follow-up question"
    assert latest_user_utterance(msgs[:3]) == "Initial request"
    assert latest_user_utterance([]) == ""


def test_rows_from_messages_excludes_active_user() -> None:
    msgs = [
        {"role": "user", "content": "Initial instruction"},
        {"role": "assistant", "content": "Sure, searching."},
        {"role": "tool", "name": "search", "content": '{"flight_id": "FL01"}'},
        {"role": "user", "content": "Book that flight"},
    ]
    rows = rows_from_messages(msgs, exclude_last_user=True)
    # The active user message "Book that flight" must NOT be in recent_context rows
    vals = [r["value"] for r in rows]
    assert "Book that flight" not in vals
    assert "Initial instruction" in vals
    assert "Sure, searching." in vals
    assert "FL01" in vals


# --------------------------------------------------------------------------- #
# Unit tests: select_recent_context
# --------------------------------------------------------------------------- #
def test_select_recent_context_r7_marker_rejection() -> None:
    tools = [ToolDefinition(name="book", description="book flight")]
    rows = [
        {"key": "safe_key", "value": "safe_val"},
        {"key": "bad_key", "value": "malicious <|tool|> injection"},
        {"key": "bad_key2", "value": "malicious [TOOL] injection"},
        {"key": "normal", "value": "normal_val"},
    ]
    selected, features = select_recent_context(rows, tools=tools)
    vals = [r["value"] for r in selected]
    assert "safe_val" in vals
    assert "normal_val" in vals
    assert not any("injection" in v for v in vals)


def test_select_recent_context_deduplication() -> None:
    tools = [ToolDefinition(name="book", description="book flight")]
    rows = [
        {"key": "k1", "value": "v1"},
        {"key": "k1", "value": "v1"},
        {"key": "k2", "value": "v2"},
    ]
    selected, features = select_recent_context(rows, tools=tools)
    assert len(selected) == 2
    assert selected[0] == {"key": "k1", "value": "v1"}
    assert selected[1] == {"key": "k2", "value": "v2"}


def test_select_recent_context_whole_row_budget_eviction() -> None:
    tools = [ToolDefinition(name="lookup", description="lookup details")]
    # Create several rows with distinct values
    rows = [
        {"key": f"key_{i}", "value": f"value_payload_data_{i}" * 10}
        for i in range(20)
    ]
    # Set a budget that can only hold a few rows (~60 tokens per row)
    selected, features = select_recent_context(
        rows,
        tools=tools,
        k_recent=4,
        context_budget_tokens=200,
    )
    assert features["truncated"] is True
    assert len(selected) < 20
    # Eviction must be whole-row from oldest: newest rows are preserved
    assert selected[-1]["key"] == "key_19"
    for r in selected:
        assert not r["value"].endswith("...")  # Never cut mid-value


# --------------------------------------------------------------------------- #
# Unit tests: trim_messages
# --------------------------------------------------------------------------- #
def test_trim_messages_preserves_atomic_tool_call_groups() -> None:
    # Build a conversation with system, initial user, several tool turns, and final user
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Task instruction: book trip."},
    ]
    for i in range(5):
        call_id = f"call_{i}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "search"}}],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": "search",
            "content": f"Results for step {i}: " + ("data " * 100),
        })
    messages.append({"role": "user", "content": "Now confirm everything."})

    trimmed, count = trim_messages(
        messages,
        max_context_tokens=150,  # tight budget forcing eviction
        keep_last_tool_turns=1,
    )
    assert count > 0
    # Pinned messages must remain
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["content"] == "Task instruction: book trip."
    assert trimmed[-1]["content"] == "Now confirm everything."

    # Invariant: every assistant tool_calls message must be immediately followed by its tool response
    for idx, msg in enumerate(trimmed):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            assert idx + 1 < len(trimmed)
            assert trimmed[idx + 1].get("role") == "tool"
            assert trimmed[idx + 1].get("tool_call_id") == msg["tool_calls"][0]["id"]


# --------------------------------------------------------------------------- #
# Integration: Multi-turn loop with fast-path trajectory grounding
# --------------------------------------------------------------------------- #
class MultiTurnTrajectoryEnv(BaseToolEnvironment):
    """Env where search yields flight FL-888, and book requires FL-888."""

    def __init__(self) -> None:
        self.booked_flight: Optional[str] = None
        self.search_called = False

    def get_available_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_flights",
                description="Search available flights",
                parameters=[
                    ParameterSchema(name="origin", description="departure", param_type="string"),
                    ParameterSchema(name="destination", description="arrival", param_type="string"),
                ],
            ),
            ToolDefinition(
                name="book_flight",
                description="Book a flight",
                parameters=[ParameterSchema(name="flight_id", description="id", param_type="string")],
            ),
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        if tool_name == "search_flights":
            self.search_called = True
            return ToolExecutionResult(
                success=True,
                output={"status": "available", "flight_id": "FL-888", "price": 450},
            )
        if tool_name == "book_flight":
            self.booked_flight = str(arguments.get("flight_id"))
            return ToolExecutionResult(
                success=True,
                output={"confirmation": "CONF-888", "status": "confirmed"},
            )
        return ToolExecutionResult(success=False, output=f"Unknown tool: {tool_name}")

    def reset(self) -> None:
        self.booked_flight = None
        self.search_called = False

    def evaluate_task_success(self) -> bool:
        return self.booked_flight == "FL-888"


class ScriptedTrajectoryResolver:
    """Simulates SpanCall resolving turn 0 (search) and turn 1 (book via recent_context)."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Any = None,
    ) -> ResolveResult:
        self.calls.append({
            "utterance": utterance,
            "recent_context": recent_context,
            "tools": candidate_schemas,
        })
        # Turn 0: initial user request
        if "Find flights" in utterance and not recent_context:
            return ResolveResult(
                hit=True,
                tool_name="search_flights",
                payload={"origin": "SFO", "destination": "JFK"},
                joint_margin=2.0,
                accept_score=0.98,
            )

        # Turn 1: user says "Book that flight"; grounded via recent_context
        if recent_context:
            for r in recent_context:
                if r.get("value") == "FL-888":
                    return ResolveResult(
                        hit=True,
                        tool_name="book_flight",
                        payload={"flight_id": "FL-888"},
                        joint_margin=2.5,
                        accept_score=0.99,
                    )
        return ResolveResult(hit=False, abstain_reason="no_grounding")


class DummyLLM(BaseLLMBackend):
    def generate_tool_call(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"tool_name": None, "arguments": None, "text": "noop"}

    def generate_response(self, *args: Any, **kwargs: Any) -> str:
        return "noop"


def test_agent_loop_mid_turn_hit_grounded_in_trajectory() -> None:
    env = MultiTurnTrajectoryEnv()
    resolver = ScriptedTrajectoryResolver()
    router = ToolCallRouter(
        resolver=resolver,  # type: ignore[arg-type]
        llm_backend=DummyLLM(),
        mode=RouterMode.ROUTER,
        variant=IntegrationVariant.S_SKIP,
    )
    loop = ReActAgentLoop(
        router=router,
        environment=env,
        max_turns=3,
    )
    trace = loop.run_task("Find flights from SFO to JFK", "task-traj-1")

    # In turn 0, search_flights was called via SpanCall.
    # In turn 1, book_flight succeeded on the fast path because the trajectory
    # tool output (FL-888) was fed into recent_context!
    assert trace.task_completed is True
    assert trace.task_success is True
    assert env.booked_flight == "FL-888"
    assert len(trace.steps) == 2
    assert trace.steps[0].decision == RoutingDecision.SPANCALL_HIT
    assert trace.steps[1].decision == RoutingDecision.SPANCALL_HIT
    assert trace.total_spancall_calls == 2
    assert trace.total_llm_calls == 0

    # Verify that turn 1 resolver call saw the recent_context containing FL-888
    turn1_call = resolver.calls[1]
    assert turn1_call["utterance"] == "Find flights from SFO to JFK"
    recent_vals = [r["value"] for r in (turn1_call["recent_context"] or [])]
    assert "FL-888" in recent_vals


# --------------------------------------------------------------------------- #
# Backend safety & 10k-token transcript safety
# --------------------------------------------------------------------------- #
def test_backend_context_budget_error_defense_in_depth() -> None:
    backend = UniversalAPIBackend(
        model_id="test-model",
        api_key="fake-key",
        max_context_tokens=50,
    )
    giant_messages = [
        {"role": "user", "content": "data " * 200}
    ]
    with pytest.raises(ContextBudgetError):
        backend.generate_tool_call(giant_messages, tools=[])


def test_multiline_recent_values_and_key_sanitization_roundtrip() -> None:
    """Multi-line tool outputs/conversation turns and leading-digit keys must not corrupt parser lines."""
    from datetime import date
    from span_template.template import Param, RecentRow, Tool, build_input

    messages = [
        {"role": "user", "content": "Line 1 of user context\nLine 2 of user context\r\nLine 3"},
        {"role": "tool", "name": "0_search", "content": "Unstructured line A\nUnstructured line B\tTabbed"},
        {"role": "user", "content": "Active utterance"},
    ]
    rows = rows_from_messages(messages, exclude_last_user=True)
    tools = [ToolDefinition(name="book", description="book flight")]
    selected, _ = select_recent_context(
        rows + [{"key": "99_numeric_start", "value": "val\nwith\nnewlines"}],
        tools=tools,
    )
    t_rows = [RecentRow(key=r["key"], value=r["value"], source=r.get("source", "recent")) for r in selected]
    tpl_tool = Tool(name="book", description="book flight", params=[Param(name="id", description="id")])
    out = build_input(tpl_tool, "Active utterance", recent_context=t_rows, now=date(2026, 11, 5))
    recent_entries = [e for e in out.entries if e.segment == "recent" and e.selectable]
    assert len(recent_entries) == 3
    assert any("Line 1 of user context Line 2 of user context Line 3" == e.value for e in recent_entries)
    assert any("Unstructured line A Unstructured line B Tabbed" == e.value for e in recent_entries)
    assert any("val with newlines" == e.value for e in recent_entries)


def test_candidate_mask_builder_recent_regex_and_subspans() -> None:
    """CandidateMaskBuilder must extract numbers, dates, emails, and sub-spans from multi-word recent rows."""
    from datetime import date
    from span_template.template import Param, RecentRow, Tool, build_input
    from spancall.model.masks import CandidateMaskBuilder

    tpl_tool = Tool(
        name="book",
        description="book",
        params=[
            Param(name="price", description="price"),
            Param(name="when", description="date"),
            Param(name="contact", description="email"),
            Param(name="city", description="city"),
        ],
    )
    out = build_input(
        tpl_tool,
        "Book it for me please",
        recent_context=[
            RecentRow(
                key="turn_0_assistant",
                value="Found flight to San Francisco on 2026-11-05 for 450 USD, contact agent@airline.com",
            )
        ],
        now=date(2026, 11, 5),
    )
    # Word-level offset mapping on rendered text
    offsets = []
    start = None
    for idx, ch in enumerate(out.text + " "):
        if not ch.isspace() and start is None:
            start = idx
        elif ch.isspace() and start is not None:
            offsets.append((start, idx))
            start = None

    builder = CandidateMaskBuilder()
    num_masks = builder.build_param_masks(
        ParameterSchema(name="price", description="price", param_type="integer"),
        out.entries,
        offsets,
        out.text,
    )
    assert any(c.region == "recent" and c.canon_value == "450" for c in num_masks["candidates"])

    date_masks = builder.build_param_masks(
        ParameterSchema(name="when", description="date", param_type="date"),
        out.entries,
        offsets,
        out.text,
    )
    assert any(c.region == "recent" and c.canon_value == "2026-11-05" for c in date_masks["candidates"])

    email_masks = builder.build_param_masks(
        ParameterSchema(name="contact", description="email", param_type="email"),
        out.entries,
        offsets,
        out.text,
    )
    assert any(c.region == "recent" and c.canon_value == "agent@airline.com" for c in email_masks["candidates"])

    str_masks = builder.build_param_masks(
        ParameterSchema(name="city", description="city", param_type="string"),
        out.entries,
        offsets,
        out.text,
    )
    assert any(c.region == "recent" and c.canon_value == "San Francisco" for c in str_masks["candidates"])


def test_strict_type_caster_non_utterance_string_values() -> None:
    """StrictTypeCaster.cast_choice must cast string values from recent/time/bool regions."""
    from datetime import date
    from types import SimpleNamespace
    from spancall.infer.cast import StrictTypeCaster

    caster = StrictTypeCaster()
    ok_i, val_i, _ = caster.cast_choice(
        SimpleNamespace(value="450", region="recent", span=(10, 13)),
        ParameterSchema(name="price", description="p", param_type="integer"),
    )
    assert ok_i and val_i == 450

    ok_d, val_d, _ = caster.cast_choice(
        SimpleNamespace(value="2026-11-05", region="time", span=(10, 20)),
        ParameterSchema(name="when", description="d", param_type="date"),
    )
    assert ok_d and val_d == date(2026, 11, 5)

    ok_b, val_b, _ = caster.cast_choice(
        SimpleNamespace(value="true", region="bool", span=(10, 14)),
        ParameterSchema(name="flag", description="b", param_type="boolean"),
    )
    assert ok_b and val_b is True


def test_resolver_10k_token_transcript_hit_or_abstain_never_crash() -> None:
    """End-to-end LeanResolver with a >10k-token recent transcript budgets cleanly and never crashes."""
    import torch
    import torch.nn as nn
    from spancall.lean.resolver import LeanResolver

    class TinyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = 8
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            B, L = input_ids.shape
            return torch.zeros(B, L, 8, device=input_ids.device)

    schema = ToolDefinition(
        name="book_flight",
        description="Book a flight by ID",
        parameters=[ParameterSchema(name="flight_id", description="flight id", param_type="string")],
    )
    # Build >10k tokens of recent_context (~50k chars)
    huge_recent = [
        {"key": f"turn_{i}_tool", "value": f"Flight option FL-{i:04d} with detailed itinerary notes " + ("x" * 200)}
        for i in range(200)
    ]
    resolver = LeanResolver(
        encoder=TinyEncoder(),
        max_seq_len=8192,
        context_budget_tokens=2048,
        k_recent=16,
    )
    res = resolver.try_resolve(schema, "Please book flight FL-0199", recent_context=huge_recent)
    assert isinstance(res, ResolveResult)
    assert res.features.get("truncated") is True
    assert res.features.get("n_recent_rows", 0) > 0
    assert res.features.get("recent_tokens", 0) <= 2048
