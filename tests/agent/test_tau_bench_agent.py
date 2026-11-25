"""Verification tests for Stage 1 (agent router/loop upgrades) and Stage 2 (Simple τ-bench Retail)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.loop import ReActAgentLoop
from spancall.agent.router import IntegrationVariant, RouterMode, ToolCallRouter
from spancall.agent.tools.tau_retail import (
    TAU_RETAIL_SYSTEM_PROMPT,
    ScriptedUserSimulator,
    SimpleTauRetailEnvironment,
    build_simple_tau_retail_suite,
    make_tau_retail_tools,
)
from spancall.agent.trace import RoutingDecision
from spancall.data.schema import ParameterSchema, ToolDefinition
from spancall.infer.base import NoOpResolver, ResolveResult

from scripts.eval_tau_bench import (
    DeterministicTauRetailBackend,
    main as eval_tau_main,
    run_simple_tau_retail,
)


class _ScriptedResolver:
    """Returns a scripted sequence of ResolveResult objects."""

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
        self.calls.append({"utterance": utterance, "recent_context": recent_context})
        if not self.outcomes:
            return ResolveResult(hit=False, abstain_reason="exhausted")
        return self.outcomes.pop(0)


class _ScriptedBackend(BaseLLMBackend):
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
            return {"tool_name": None, "arguments": None, "text": "Done."}
        return self.responses.pop(0)

    def generate_response(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        return "Done."


# --------------------------------------------------------------------------- #
# Stage 1: RouterMode.ROUTER_SCHEMA, duplicate guard, risk_tier gate, prose
# --------------------------------------------------------------------------- #
def test_router_schema_delegates_array_tool_to_llm() -> None:
    tools = make_tau_retail_tools()
    resolver = _ScriptedResolver([
        ResolveResult(
            hit=True,
            tool_name="return_delivered_order_items",
            payload={"order_id": "#W2002", "item_ids": ["item_401_white"], "payment_method_id": "gift_card_4412001"},
        )
    ])
    backend = _ScriptedBackend([
        {
            "tool_name": "return_delivered_order_items",
            "arguments": {"order_id": "#W2002", "item_ids": ["item_401_white"], "payment_method_id": "gift_card_4412001"},
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }
    ])
    router = ToolCallRouter(resolver=resolver, llm_backend=backend, mode=RouterMode.ROUTER_SCHEMA)  # type: ignore[arg-type]
    res, step = router.route_turn("Return item_401_white", [], tools)
    assert res["spancall"] is False
    assert step is not None
    assert step.decision == RoutingDecision.LLM_FALLBACK
    assert step.spancall_features["router_gate_reason"] == "complex_schema"
    assert len(backend.calls) == 1


def test_router_risk_tier_gate_delegates_high_risk_write_tool() -> None:
    tools = make_tau_retail_tools()
    resolver = _ScriptedResolver([
        ResolveResult(
            hit=True,
            tool_name="cancel_pending_order",
            payload={"order_id": "#W1001", "reason": "ordered by mistake"},
        )
    ])
    backend = _ScriptedBackend([
        {"tool_name": None, "arguments": None, "text": "Please confirm you want to cancel #W1001."}
    ])
    router = ToolCallRouter(
        resolver=resolver,  # type: ignore[arg-type]
        llm_backend=backend,
        mode=RouterMode.ROUTER_SCHEMA,
        skip_risk_tiers=("high",),
    )
    res, step = router.route_turn("Cancel #W1001", [], tools)
    assert res["hit"] is False
    assert res["text"] == "Please confirm you want to cancel #W1001."
    assert step is not None
    assert step.spancall_features["router_gate_reason"] == "risk_tier:high"


def test_router_duplicate_call_guard_prevents_loop() -> None:
    tools = make_tau_retail_tools()
    dup_hit = ResolveResult(
        hit=True,
        tool_name="get_order_details",
        payload={"order_id": "#W1001"},
    )
    resolver = _ScriptedResolver([dup_hit, dup_hit])
    backend = _ScriptedBackend([
        {"tool_name": None, "arguments": None, "text": "Order #W1001 is pending."}
    ])
    router = ToolCallRouter(
        resolver=resolver,  # type: ignore[arg-type]
        llm_backend=backend,
        mode=RouterMode.ROUTER_SCHEMA,
        variant=IntegrationVariant.S_INJECT,
    )
    env = SimpleTauRetailEnvironment(
        goal_checker=lambda db, h: any(c["tool_name"] == "get_order_details" for c in h)
    )
    loop = ReActAgentLoop(
        router=router,
        environment=env,
        max_turns=5,
        prevent_duplicate_calls=True,
    )
    trace = loop.run_task(
        "Check order #W1001",
        "dup_guard_test",
        user_simulator=ScriptedUserSimulator([]),
    )
    assert trace.task_success is True
    assert len(trace.steps) == 2
    assert trace.steps[0].decision == RoutingDecision.SPANCALL_HIT
    assert trace.steps[1].decision == RoutingDecision.LLM_FALLBACK
    assert trace.steps[1].spancall_features["router_gate_reason"] == "duplicate_call"


def test_shadow_mode_flags_false_hit_on_prose() -> None:
    tools = make_tau_retail_tools()
    resolver = _ScriptedResolver([
        ResolveResult(hit=True, tool_name="get_order_details", payload={"order_id": "#W1001"})
    ])
    backend = _ScriptedBackend([
        {"tool_name": None, "arguments": None, "text": "Your order is pending."}
    ])
    router = ToolCallRouter(resolver=resolver, llm_backend=backend, mode=RouterMode.SHADOW)  # type: ignore[arg-type]
    res, step = router.route_turn("Thanks!", [], tools)
    assert res["hit"] is False
    assert res["text"] == "Your order is pending."
    assert res["shadow_false_hit_on_prose"] is True
    assert step is not None


# --------------------------------------------------------------------------- #
# Stage 2: Full Simple τ-bench Retail suite (10 tasks) + CLI trace export
# --------------------------------------------------------------------------- #
def test_simple_tau_retail_all_10_tasks_succeed_with_deterministic_backend() -> None:
    summaries, traces_by_mode = run_simple_tau_retail(
        resolver=NoOpResolver(),
        backend=DeterministicTauRetailBackend(),
        modes=["llm_only", "router_schema", "shadow"],
        skip_risk_tiers=("high",),
    )
    assert len(summaries) == 3
    for s in summaries:
        assert s["tasks"] == 10
        assert s["success_rate"] == 1.0, f"Mode {s['mode']} failed: {s}"
        assert s["diagnostic_buckets"].get("tool_exec_error", 0) == 0
    assert len(traces_by_mode["router_schema"]) == 10


def test_openai_adapter_normalizes_injected_tool_call_arguments_to_json_str() -> None:
    from spancall.agent.backends.openai_adapter import _normalize_openai_messages

    raw_msgs = [
        {"role": "user", "content": "Check #W1001"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_order_details", "arguments": {"order_id": "#W1001"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "name": "get_order_details", "content": "{}"},
    ]
    norm = _normalize_openai_messages(raw_msgs)
    args_val = norm[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args_val, str)
    assert json.loads(args_val) == {"order_id": "#W1001"}
    # Original input is not mutated
    assert isinstance(raw_msgs[1]["tool_calls"][0]["function"]["arguments"], dict)


def test_hybrid_fill_schema_preserves_grounded_scalars_and_enforces_risk_tier_on_partial() -> None:
    tools = make_tau_retail_tools()
    # 1) Complex tool hit: keeps grounded scalar order_id & payment_method_id, asks LLM only for item_ids
    resolver_complex = _ScriptedResolver([
        ResolveResult(
            hit=True,
            tool_name="return_delivered_order_items",
            payload={
                "order_id": "#W2002",
                "item_ids": "bad_scalar",
                "payment_method_id": "gift_card_4412001",
            },
        )
    ])
    backend_complex = _ScriptedBackend([
        {
            "tool_name": "return_delivered_order_items",
            "arguments": {"item_ids": ["item_401_white"]},
            "usage": {"prompt_tokens": 30, "completion_tokens": 10},
        }
    ])
    router_complex = ToolCallRouter(
        resolver=resolver_complex,  # type: ignore[arg-type]
        llm_backend=backend_complex,
        mode=RouterMode.HYBRID_FILL_SCHEMA,
    )
    res, step = router_complex.route_turn("Return item_401_white from #W2002", [], tools)
    assert res["hit"] is True
    assert res["hybrid_fill"] is True
    assert res["arguments"] == {
        "order_id": "#W2002",
        "item_ids": ["item_401_white"],
        "payment_method_id": "gift_card_4412001",
    }
    assert step is not None
    assert step.decision == RoutingDecision.HYBRID_FILL

    # 2) Partial high-risk write tool: risk_tier gate forces full LLM fallback (never hybrid-fills)
    resolver_risk = _ScriptedResolver([
        ResolveResult(
            hit=False,
            tool_name="cancel_pending_order",
            payload_partial={"order_id": "#W1001"},
            missing_required=["reason"],
        )
    ])
    backend_risk = _ScriptedBackend([
        {"tool_name": None, "arguments": None, "text": "Please confirm cancellation."}
    ])
    router_risk = ToolCallRouter(
        resolver=resolver_risk,  # type: ignore[arg-type]
        llm_backend=backend_risk,
        mode=RouterMode.HYBRID_FILL_SCHEMA,
        skip_risk_tiers=("high",),
    )
    res2, step2 = router_risk.route_turn("Cancel #W1001", [], tools)
    assert res2["hit"] is False
    assert res2["text"] == "Please confirm cancellation."
    assert step2 is not None
    assert step2.spancall_features["router_gate_reason"] == "risk_tier:high"


def test_overflow_telemetry_fires_on_context_overflow_abstain_and_seq_tokens() -> None:
    resolver = _ScriptedResolver([
        ResolveResult(
            hit=False,
            abstain_reason="context_overflow: 4200 tokens > L_max=4096",
            features={"context_overflow_seq": True},
        )
    ])
    backend = _ScriptedBackend([
        {"tool_name": None, "arguments": None, "text": "Handled by LLM."}
    ])
    router = ToolCallRouter(resolver=resolver, llm_backend=backend, mode=RouterMode.ROUTER_SCHEMA)  # type: ignore[arg-type]
    env = SimpleTauRetailEnvironment()
    loop = ReActAgentLoop(router=router, environment=env, max_turns=3)
    trace = loop.run_task("Long prompt", "overflow_test", user_simulator=ScriptedUserSimulator([]))
    assert trace.steps[0].spancall_features.get("context_overflow_seq") is True


def test_duplicate_guard_allows_reread_after_state_mutation() -> None:
    """After a mutation sets state_changed=True, re-reading get_order_details must not be blocked."""
    get_ord = ResolveResult(hit=True, tool_name="get_order_details", payload={"order_id": "#W1001"})
    cancel_ord = ResolveResult(
        hit=True,
        tool_name="cancel_pending_order",
        payload={"order_id": "#W1001", "reason": "ordered by mistake"},
    )
    resolver = _ScriptedResolver([get_ord, cancel_ord, get_ord, ResolveResult(hit=False, abstain_reason="done")])
    backend = _ScriptedBackend([{"tool_name": None, "arguments": None, "text": "Verified cancelled."}])
    router = ToolCallRouter(
        resolver=resolver,  # type: ignore[arg-type]
        llm_backend=backend,
        mode=RouterMode.ROUTER_SCHEMA,
        variant=IntegrationVariant.S_INJECT,
    )
    env = SimpleTauRetailEnvironment(
        goal_checker=lambda db, h: db["orders"]["#W1001"]["status"] == "cancelled"
    )
    loop = ReActAgentLoop(router=router, environment=env, max_turns=6, prevent_duplicate_calls=True)
    trace = loop.run_task("Cancel and re-check #W1001", "reread_test", user_simulator=ScriptedUserSimulator([]))
    assert trace.task_success is True
    # Turn 0: get_order_details (hit), Turn 1: cancel_pending_order (hit, state_changed=True),
    # Turn 2: get_order_details AGAIN (hit, NOT blocked by duplicate_call!)
    assert trace.steps[0].decision == RoutingDecision.SPANCALL_HIT
    assert trace.steps[1].decision == RoutingDecision.SPANCALL_HIT
    assert trace.steps[2].decision == RoutingDecision.SPANCALL_HIT
    assert trace.steps[2].selected_tool == "get_order_details"


def test_task_08_fails_if_cancel_is_attempted_on_delivered_order() -> None:
    task_08 = next(t for t in build_simple_tau_retail_suite() if t.task_id == "tau_retail_08_cannot_cancel_delivered")
    env = SimpleTauRetailEnvironment(goal_checker=task_08.goal_checker)
    env.execute("get_order_details", {"order_id": "#W1002"})
    assert env.evaluate_task_success() is True
    # Attempting cancel_pending_order violates policy even though the DB stays "delivered"
    env.execute("cancel_pending_order", {"order_id": "#W1002", "reason": "no longer needed"})
    assert env.evaluate_task_success() is False


def test_simple_tau_retail_spancall_fast_path_reduces_llm_calls() -> None:
    """When SpanCall resolves read-only lookups, LLM calls drop while preserving 100% success."""
    tasks = build_simple_tau_retail_suite()[:2]  # 2 lookup-chain tasks
    # Task 1: find_user_id_by_name_zip -> get_order_details -> abstain (LLM prose)
    # Task 2: find_user_id_by_email -> get_user_details -> abstain (LLM prose)
    resolver = _ScriptedResolver([
        ResolveResult(
            hit=True,
            tool_name="find_user_id_by_name_zip",
            payload={"first_name": "Yusuf", "last_name": "Rossi", "zip": "10003"},
        ),
        ResolveResult(hit=True, tool_name="get_order_details", payload={"order_id": "#W1001"}),
        ResolveResult(hit=False, abstain_reason="done"),
        ResolveResult(
            hit=True,
            tool_name="find_user_id_by_email",
            payload={"email": "mei.chen@example.com"},
        ),
        ResolveResult(hit=True, tool_name="get_user_details", payload={"user_id": "mei_chen_4412"}),
        ResolveResult(hit=False, abstain_reason="done"),
    ])
    summaries, _ = run_simple_tau_retail(
        resolver=resolver,
        backend=DeterministicTauRetailBackend(),
        modes=["llm_only", "router_schema"],
        tasks=tasks,
    )
    llm_only = next(s for s in summaries if s["mode"] == "llm_only")
    router_schema = next(s for s in summaries if s["mode"] == "router_schema")
    assert llm_only["success_rate"] == 1.0
    assert router_schema["success_rate"] == 1.0
    assert router_schema["total_spancall_hits"] == 4
    assert router_schema["total_llm_calls"] < llm_only["total_llm_calls"]


def test_eval_tau_bench_cli_writes_traces_and_summary(tmp_path: Path) -> None:
    rc = eval_tau_main([
        "--backend", "deterministic",
        "--modes", "llm_only", "router_schema",
        "--limit", "3",
        "--skip-high-risk",
        "--out-dir", str(tmp_path),
    ])
    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 2
    assert all(row["success_rate"] == 1.0 for row in summary)
    trace_lines = (tmp_path / "traces_router_schema.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(trace_lines) == 3
