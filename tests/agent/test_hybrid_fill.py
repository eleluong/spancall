"""Tests for the v3 mode-3 integration: SpanCall skeleton + LLM fill.

Covers :mod:`spancall.agent.fill` (pure merge/prompt helpers) and the
``RouterMode.HYBRID_FILL`` branch of :class:`spancall.agent.router.ToolCallRouter`.

The properties under test are the ones that make the mode *safe*, not merely
functional:

* the LLM is never asked when SpanCall already hit cleanly;
* the LLM's prompt physically excludes already-resolved params;
* a resolved SpanCall value can never be overwritten by the LLM;
* parameters the LLM was not asked for are dropped, not merged;
* an unhelpful fill falls back to the ordinary LLM call instead of shipping a
  half-filled payload.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.fill import (
    build_fill_messages,
    merge_filled_arguments,
    param_leaves,
    restrict_tool_to_params,
    unresolved_param_names,
)
from spancall.agent.router import RouterMode, ToolCallRouter
from spancall.agent.trace import RoutingDecision
from spancall.data.schema import ParameterSchema, ToolDefinition
from spancall.infer.base import ResolveResult


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="book_flight",
        description="Book a flight",
        parameters=[
            ParameterSchema(name="flight_id", description="id", param_type="string"),
            ParameterSchema(name="passenger", description="name", param_type="string"),
            ParameterSchema(
                name="seat", description="seat", param_type="string", required=False
            ),
        ],
    )


class ScriptedBackend(BaseLLMBackend):
    """Backend returning preset dicts and recording the prompts it received."""

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

    def generate_response(
        self, messages: List[Dict[str, Any]], temperature: float = 0.0
    ) -> str:
        return ""


class FixedResolver:
    def __init__(self, result: ResolveResult) -> None:
        self.result = result
        self.calls = 0

    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Any = None,
    ) -> ResolveResult:
        self.calls += 1
        return self.result


def _router(res: ResolveResult, backend: BaseLLMBackend, **kwargs: Any) -> ToolCallRouter:
    return ToolCallRouter(
        resolver=FixedResolver(res),  # type: ignore[arg-type]
        llm_backend=backend,
        mode=RouterMode.HYBRID_FILL,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_param_leaves_includes_dotted_object_leaves() -> None:
    tool = ToolDefinition(
        name="search",
        description="",
        parameters=[ParameterSchema(name="query", description="", param_type="string")],
        object_paths={"filters": ["filters.color", "filters.size"]},
    )
    leaves = param_leaves(tool)
    assert "query" in leaves
    assert "filters" in leaves
    assert "filters.color" in leaves


def test_unresolved_prefers_resolver_lists_over_payload_derivation() -> None:
    names = unresolved_param_names(
        _tool(),
        payload_partial={"flight_id": "F001"},
        missing_required=["passenger"],
        missing_optional=["seat"],
    )
    # Only what the resolver reported, and never a param it already grounded.
    assert names == ["passenger", "seat"]


def test_unresolved_derives_from_payload_when_lists_are_empty() -> None:
    names = unresolved_param_names(_tool(), {"flight_id": "F001"})
    assert set(names) == {"passenger", "seat"}


def test_restrict_tool_marks_requested_params_required() -> None:
    restricted = restrict_tool_to_params(_tool(), ["seat"])
    assert [p.name for p in restricted.parameters] == ["seat"]
    assert restricted.parameters[0].required is True


def test_build_fill_messages_states_known_values_are_immutable() -> None:
    msgs = build_fill_messages([], "book_flight", {"flight_id": "F001"}, ["passenger"])
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert "book_flight" in content
    assert "F001" in content
    assert "passenger" in content
    assert "DO NOT change" in content


def test_merge_never_overwrites_resolved_and_drops_unrequested() -> None:
    merged = merge_filled_arguments(
        {"flight_id": "F001"},
        {"flight_id": "WRONG", "passenger": "Alice", "injected": "x", "seat": None},
        allowed=["passenger", "seat"],
    )
    assert merged == {"flight_id": "F001", "passenger": "Alice"}


# --------------------------------------------------------------------------- #
# Router branch
# --------------------------------------------------------------------------- #
def test_clean_hit_never_calls_the_llm() -> None:
    backend = ScriptedBackend([])
    res = ResolveResult(hit=True, tool_name="book_flight", payload={"flight_id": "F001"})
    result, step = _router(res, backend).route_turn("book F001", [], [_tool()])
    assert result["hit"] is True
    assert result["hybrid_fill"] is False
    assert backend.calls == []
    assert step.decision == RoutingDecision.SPANCALL_HIT


def test_partial_fill_asks_only_for_missing_params_and_merges() -> None:
    backend = ScriptedBackend([{"tool_name": "book_flight", "arguments": {"passenger": "Alice"}}])
    res = ResolveResult(
        hit=False,
        tool_name="book_flight",
        status="partial",
        payload_partial={"flight_id": "F001"},
        missing_required=["passenger"],
        missing_optional=["seat"],
        abstain_reason="missing_required:passenger",
    )
    result, step = _router(res, backend).route_turn("book F001 for Alice", [], [_tool()])

    assert result["hit"] is True
    assert result["hybrid_fill"] is True
    assert result["arguments"] == {"flight_id": "F001", "passenger": "Alice"}
    assert result["filled_params"] == ["passenger"]
    assert result["spancall"] is True
    assert step.decision == RoutingDecision.HYBRID_FILL

    # The LLM saw a restricted schema: only the missing REQUIRED param.
    sent_tools = backend.calls[0]["tools"]
    assert [t.name for t in sent_tools] == ["book_flight"]
    assert [p.name for p in sent_tools[0].parameters] == ["passenger"]


def test_fill_that_adds_nothing_falls_back_to_full_llm_call() -> None:
    backend = ScriptedBackend([
        {"tool_name": "book_flight", "arguments": {}},  # useless fill
        {"tool_name": "book_flight", "arguments": {"flight_id": "F001", "passenger": "Bob"}},
    ])
    res = ResolveResult(
        hit=False,
        tool_name="book_flight",
        payload_partial={"flight_id": "F001"},
        missing_required=["passenger"],
    )
    result, step = _router(res, backend).route_turn("book it for Bob", [], [_tool()])

    assert result["hit"] is True
    assert result.get("hybrid_fill") is not True
    assert step.decision == RoutingDecision.LLM_FALLBACK
    assert len(backend.calls) == 2
    # The fallback is the ORDINARY prompt, not the fill prompt.
    assert backend.calls[1]["messages"] == []


def test_no_tool_from_spancall_degrades_to_llm_only() -> None:
    backend = ScriptedBackend([{"tool_name": "book_flight", "arguments": {"flight_id": "F9"}}])
    res = ResolveResult(hit=False, tool_name=None, abstain_reason="tool_below_threshold")
    result, step = _router(res, backend).route_turn("book something", [], [_tool()])
    assert result["hit"] is True
    assert step.decision == RoutingDecision.LLM_FALLBACK
    assert len(backend.calls) == 1


def test_optional_params_not_requested_by_default() -> None:
    backend = ScriptedBackend([{"tool_name": "book_flight", "arguments": {"passenger": "Alice"}}])
    res = ResolveResult(
        hit=False,
        tool_name="book_flight",
        payload_partial={"flight_id": "F001"},
        missing_required=["passenger"],
        missing_optional=["seat"],
    )
    result, _ = _router(res, backend).route_turn("book F001 for Alice", [], [_tool()])
    assert result["requested_params"] == ["passenger"]
    assert "seat" not in result["arguments"]


def test_hybrid_fill_optional_flag_requests_optionals_too() -> None:
    backend = ScriptedBackend([
        {"tool_name": "book_flight", "arguments": {"passenger": "Alice", "seat": "12A"}}
    ])
    res = ResolveResult(
        hit=False,
        tool_name="book_flight",
        payload_partial={"flight_id": "F001"},
        missing_required=["passenger"],
        missing_optional=["seat"],
    )
    result, _ = _router(res, backend, hybrid_fill_optional=True).route_turn(
        "book F001 for Alice in 12A", [], [_tool()]
    )
    assert set(result["requested_params"]) == {"passenger", "seat"}
    assert result["arguments"]["seat"] == "12A"