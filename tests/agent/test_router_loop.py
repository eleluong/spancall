"""Tests for :class:`spancall.agent.router.ToolCallRouter` and
:class:`spancall.agent.loop.ReActAgentLoop` (plan §4.3, mechanism §11).

Covers:
- All four ``RouterMode`` values (llm_only / spancall_only / router / shadow)
  against a fake resolver + fake LLM backend.
- ``IntegrationVariant.S_INJECT`` history synthesis: the loop appends an
  OpenAI-style assistant ``tool_calls`` message + ``role:tool`` result so the
  conversation is byte-identical in shape to a native LLM tool call.
- Multi-turn booking task with ``MockBookingEnvironment`` (search -> book).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.loop import ReActAgentLoop
from spancall.agent.router import IntegrationVariant, RouterMode, ToolCallRouter
from spancall.agent.tools.base import BaseToolEnvironment
from spancall.agent.tools.mock import MockBookingEnvironment
from spancall.agent.trace import ExecutionTrace, RoutingDecision, StepTrace
from spancall.data.schema import ToolDefinition
from spancall.infer.base import ResolveResult


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeResolver:
    """Stand-in for a BaseResolver with a scripted sequence of outcomes."""

    def __init__(self, outcomes: List[ResolveResult]) -> None:
        self.outcomes = list(outcomes)
        self.calls: List[Dict[str, Any]] = []

    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Any = None,
    ) -> ResolveResult:
        self.calls.append(
            {"utterance": utterance, "recent_context": recent_context, "tools": candidate_schemas}
        )
        if not self.outcomes:
            return ResolveResult(hit=False, abstain_reason="no_more_outcomes")
        return self.outcomes.pop(0)


class FakeLLMBackend(BaseLLMBackend):
    """Scripted backend: returns preset tool-call dicts in sequence."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self.responses:
            return {"tool_name": None, "arguments": None, "text": ""}
        return self.responses.pop(0)

    def generate_response(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        return "I can help you with that."


def _hit(tool: str, args: Dict[str, Any], **kw: Any) -> ResolveResult:
    return ResolveResult(
        hit=True,
        tool_name=tool,
        payload=dict(args),
        joint_margin=kw.pop("joint_margin", 2.5),
        accept_score=kw.pop("accept_score", 0.97),
        features=kw.pop("features", {"source": "fake"}),
    )


def _miss(reason: str = "low_confidence", tool: Optional[str] = None) -> ResolveResult:
    return ResolveResult(hit=False, abstain_reason=reason, tool_name=tool)


def _make_router(
    resolve_outcomes: List[ResolveResult],
    llm_responses: List[Dict[str, Any]],
    mode: RouterMode = RouterMode.ROUTER,
    variant: IntegrationVariant = IntegrationVariant.S_SKIP,
) -> ToolCallRouter:
    return ToolCallRouter(
        resolver=FakeResolver(resolve_outcomes),  # type: ignore[arg-type]
        llm_backend=FakeLLMBackend(llm_responses),
        mode=mode,
        variant=variant,
    )


_TOOLS = [ToolDefinition(name="book_flight", description="book a flight")]


# --------------------------------------------------------------------------- #
# Router modes
# --------------------------------------------------------------------------- #
def test_router_mode_llm_only_never_touches_resolver() -> None:
    router = _make_router(
        resolve_outcomes=[_hit("book_flight", {"flight_id": "F001"})],
        llm_responses=[{"tool_name": "book_flight", "arguments": {"flight_id": "F002"}}],
        mode=RouterMode.LLM_ONLY,
    )
    result, step = router.route_turn("Book a flight", [], _TOOLS)
    assert result["hit"] is True
    assert result["spancall"] is False
    assert result["tool_name"] == "book_flight"
    assert result["arguments"] == {"flight_id": "F002"}
    assert step.decision == RoutingDecision.LLM_FALLBACK
    # The resolver must never have been called.
    assert router.resolver.calls == []  # type: ignore[attr-defined]


def test_router_mode_spancall_only_hit() -> None:
    router = _make_router(
        resolve_outcomes=[_hit("book_flight", {"flight_id": "F001"})],
        llm_responses=[],
        mode=RouterMode.SPANCALL_ONLY,
    )
    result, step = router.route_turn("Book flight F001", [], _TOOLS)
    assert result["hit"] is True
    assert result["spancall"] is True
    assert result["tool_name"] == "book_flight"
    assert result["arguments"] == {"flight_id": "F001"}
    assert step.decision == RoutingDecision.SPANCALL_HIT
    # No LLM call in spancall_only.
    assert router.llm_backend.calls == []  # type: ignore[attr-defined]


def test_router_mode_spancall_only_abstain_no_fallback() -> None:
    router = _make_router(
        resolve_outcomes=[_miss("low_confidence")],
        llm_responses=[],
        mode=RouterMode.SPANCALL_ONLY,
    )
    result, step = router.route_turn("Something unclear", [], _TOOLS)
    assert result["hit"] is False
    assert result["abstain_reason"] == "low_confidence"
    assert step.decision == RoutingDecision.SPANCALL_ABSTAIN
    assert router.llm_backend.calls == []  # type: ignore[attr-defined]


def test_router_mode_router_hit_skips_llm() -> None:
    router = _make_router(
        resolve_outcomes=[_hit("book_flight", {"flight_id": "F001"}, joint_margin=1.2)],
        llm_responses=[{"tool_name": "book_flight", "arguments": {"flight_id": "OTHER"}}],
        mode=RouterMode.ROUTER,
    )
    result, step = router.route_turn("Book F001", [], _TOOLS)
    assert result["hit"] is True
    assert result["spancall"] is True
    assert result["joint_margin"] == 1.2
    assert step.decision == RoutingDecision.SPANCALL_HIT
    # LLM must not be consulted on a SpanCall hit.
    assert router.llm_backend.calls == []  # type: ignore[attr-defined]


def test_router_mode_router_abstain_falls_back_to_unmodified_llm() -> None:
    router = _make_router(
        resolve_outcomes=[_miss("margin_below_tau")],
        llm_responses=[{"tool_name": "book_flight", "arguments": {"flight_id": "F003"}}],
        mode=RouterMode.ROUTER,
    )
    result, step = router.route_turn(
        "Please book flight F003 for me",
        [{"role": "user", "content": "Please book flight F003 for me"}],
        _TOOLS,
    )
    assert result["hit"] is True
    assert result["spancall"] is False
    assert result["tool_name"] == "book_flight"
    assert step.decision == RoutingDecision.LLM_FALLBACK
    # The LLM receives the messages unchanged (no prompt mutation).
    llm_call = router.llm_backend.calls[0]  # type: ignore[attr-defined]
    assert llm_call["messages"] == [{"role": "user", "content": "Please book flight F003 for me"}]


def test_router_mode_shadow_runs_both_and_records_comparison() -> None:
    llm_res = {
        "tool_name": "book_flight",
        "arguments": {"flight_id": "F001"},
        "usage": {"prompt_tokens": 41, "completion_tokens": 12},
    }
    router = _make_router(
        resolve_outcomes=[_hit("book_flight", {"flight_id": "F001"})],
        llm_responses=[llm_res],
        mode=RouterMode.SHADOW,
    )
    result, step = router.route_turn("Book F001", [], _TOOLS)
    # Shadow executes the LLM call.
    assert result["hit"] is True
    assert result["spancall"] is False
    assert result["tool_name"] == "book_flight"
    # SpanCall comparison telemetry is attached.
    assert result["spancall_tool"] == "book_flight"
    assert result["spancall_arguments"] == {"flight_id": "F001"}
    assert result["spancall_matched"] is True
    assert step.decision == RoutingDecision.SHADOW_COMPARISON
    assert step.llm_tokens_in == 41
    assert step.llm_tokens_out == 12
    assert step.spancall_features == {"source": "fake"}


# --------------------------------------------------------------------------- #
# History synthesis (s_inject)
# --------------------------------------------------------------------------- #
def test_synthesize_injected_call_openai_shape() -> None:
    router = _make_router([], [])
    msg = router.synthesize_injected_call("book_flight", {"flight_id": "F001"}, "call_spancall_0")
    assert msg == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_spancall_0",
                "type": "function",
                "function": {"name": "book_flight", "arguments": {"flight_id": "F001"}},
            }
        ],
    }


def test_synthesize_tool_response_matches_call_id() -> None:
    router = _make_router([], [])
    msg = router.synthesize_tool_response("book_flight", {"booking_ref": "BR-1001"}, "call_spancall_0")
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_spancall_0"
    assert msg["name"] == "book_flight"
    assert msg["content"] == '{"booking_ref": "BR-1001"}'


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #
def test_loop_llm_only_multi_turn_books_flight() -> None:
    llm = FakeLLMBackend(
        [
            {"tool_name": "search_flights", "arguments": {"origin": "SFO", "destination": "JFK", "date": "2026-11-05"}},
            {"tool_name": "book_flight", "arguments": {"flight_id": "F001"}},
        ]
    )
    env = MockBookingEnvironment(required_flight_id="F001")
    loop = ReActAgentLoop(
        router=ToolCallRouter(resolver=FakeResolver([]), llm_backend=llm, mode=RouterMode.LLM_ONLY),  # type: ignore[arg-type]
        environment=env,
        max_turns=6,
    )
    trace = loop.run_task("Find and book a SFO->JFK flight on 2026-11-05", "task-llm-1")
    assert isinstance(trace, ExecutionTrace)
    assert trace.task_completed is True
    assert trace.task_success is True
    assert env.bookings[0]["flight_id"] == "F001"
    # Two tool steps in llm_only -> both are LLM_FALLBACK.
    assert [s.decision for s in trace.steps] == [
        RoutingDecision.LLM_FALLBACK,
        RoutingDecision.LLM_FALLBACK,
    ]
    assert trace.total_llm_calls == 2
    assert trace.total_spancall_calls == 0


def test_loop_router_s_inject_spancall_hit_synthesizes_history() -> None:
    env = MockBookingEnvironment(required_flight_id="F001")
    router = _make_router(
        resolve_outcomes=[_hit("book_flight", {"flight_id": "F001"})],
        llm_responses=[],
        mode=RouterMode.ROUTER,
        variant=IntegrationVariant.S_INJECT,
    )
    loop = ReActAgentLoop(router=router, environment=env, max_turns=4)
    trace = loop.run_task("Book flight F001", "task-inj-1")

    assert trace.task_completed is True
    assert trace.task_success is True
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.decision == RoutingDecision.SPANCALL_HIT
    assert step.selected_tool == "book_flight"
    assert step.tool_arguments == {"flight_id": "F001"}
    assert step.tool_execution_success is True
    assert step.tool_output == '{"booking_ref": "BR-1001"}'
    assert step.spancall_features["injected_call_id"] == "call_spancall_0"
    # s_inject: an OpenAI-style assistant tool_calls message is in the history.
    assert trace.steps[0].tool_arguments == {"flight_id": "F001"}
    assert trace.total_spancall_calls == 1


def test_loop_router_s_skip_no_synthetic_assistant_message() -> None:
    """s_skip replaces the LLM call entirely; history has only the tool result."""
    env = MockBookingEnvironment(required_flight_id="F001")
    router = _make_router(
        resolve_outcomes=[_hit("book_flight", {"flight_id": "F001"})],
        llm_responses=[],
        mode=RouterMode.ROUTER,
        variant=IntegrationVariant.S_SKIP,
    )
    loop = ReActAgentLoop(router=router, environment=env, max_turns=4)
    trace = loop.run_task("Book flight F001", "task-skip-1")
    assert trace.task_success is True
    step = trace.steps[0]
    assert step.decision == RoutingDecision.SPANCALL_HIT
    # s_skip does not synthesize the assistant call message; the tool response
    # carries a default call id.
    assert "injected_call_id" not in step.spancall_features


def test_loop_abstain_fallback_appends_llm_prose() -> None:
    router = _make_router(
        resolve_outcomes=[_miss("ambiguous")],
        llm_responses=[{"tool_name": None, "arguments": None, "text": "Could you clarify?"}],
        mode=RouterMode.ROUTER,
        variant=IntegrationVariant.S_SKIP,
    )
    env = MockBookingEnvironment(required_flight_id="F001")
    loop = ReActAgentLoop(router=router, environment=env, max_turns=2)
    trace = loop.run_task("hmm", "task-abs-1")
    assert trace.steps[0].decision == RoutingDecision.LLM_FALLBACK
    assert not trace.task_completed


def test_loop_respects_max_turns_cap() -> None:
    # Resolver always misses, LLM always returns prose without tool calls.
    router = _make_router(
        resolve_outcomes=[_miss("always_miss")],
        llm_responses=[{"tool_name": None, "arguments": None, "text": "Let me think…"}],
        mode=RouterMode.ROUTER,
        variant=IntegrationVariant.S_SKIP,
    )
    env = MockBookingEnvironment(required_flight_id="F001")
    loop = ReActAgentLoop(router=router, environment=env, max_turns=3)
    trace = loop.run_task("Book flight F001", "task-cap-1")
    assert len(trace.steps) == 3
    assert trace.task_completed is False
    assert trace.task_success is False


def test_loop_empty_environment_session_context_default() -> None:
    """A minimal env without session_context still works (default [] on base)."""
    from spancall.agent.tools.base import ToolExecutionResult

    class MinimalEnv(BaseToolEnvironment):
        def get_available_tools(self) -> List[ToolDefinition]:
            return [ToolDefinition(name="noop", description="no-op tool")]

        def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
            return ToolExecutionResult(success=True, output={"ok": True})

        def reset(self) -> None:
            pass

        def evaluate_task_success(self) -> bool:
            return True

    router = _make_router(
        resolve_outcomes=[_hit("noop", {})],
        llm_responses=[],
        mode=RouterMode.ROUTER,
        variant=IntegrationVariant.S_SKIP,
    )
    loop = ReActAgentLoop(router=router, environment=MinimalEnv(), max_turns=2)
    trace = loop.run_task("do nothing", "task-min-1")
    assert trace.task_success is True
    # The router saw the default (empty) session context.
    assert router.resolver.calls[0]["recent_context"] == []  # type: ignore[attr-defined]