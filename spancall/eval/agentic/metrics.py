"""Agentic task evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from spancall.agent.trace import ExecutionTrace, RoutingDecision


@dataclass
class AgentMetrics:
    """End-to-end multi-turn agent evaluation metrics."""
    total_tasks: int = 0
    task_success_rate: float = 0.0
    average_turns_per_task: float = 0.0
    tool_call_accuracy_per_step: float = 0.0
    recovery_rate_after_bad_call: float = 0.0

    # Fast-path utilization
    spancall_hit_rate: float = 0.0
    wrong_hit_rate: float = 0.0
    side_effecting_error_count: int = 0

    # Resource efficiency.
    # NOTE: cross-mode latency/token savings are NOT reported here — a
    # single-mode aggregate has no baseline to reduce against (per-step sums
    # already compose the trace totals). The paired B0-vs-S comparison lives
    # in ``ComparativeStudyResult`` (speedup_ratio / token_savings_percent).
    total_llm_calls_saved: int = 0


def aggregate_metrics(traces: List[ExecutionTrace]) -> AgentMetrics:
    """Aggregate task-level metrics from a list of execution traces.

    Mirrors the axes in docs/architecture.md §8.2:
    - task success (end-state check), per-step call accuracy, turns/task,
      recovery after a bad call *within the same trace*
    - fast-path share (SpanCall hit rate over all turns where SpanCall ran)
    - wrong-hit rate + side-effecting error count
    """
    m = AgentMetrics()
    m.total_tasks = len(traces)
    if not traces:
        return m

    success = sum(1 for t in traces if t.task_success)
    m.task_success_rate = success / len(traces)

    total_turns = sum(len(t.steps) for t in traces)
    m.average_turns_per_task = total_turns / len(traces) if traces else 0.0

    # --- per-step tool-call accuracy + same-trace recovery ------------------- #
    tool_steps = 0
    successful_tool_steps = 0
    bad_calls = 0
    recovered_after_bad = 0
    for t in traces:
        # Recovery is defined per trace: a failure in one task never counts as
        # recovered by a later success in a *different* task.
        seen_bad_call = False
        for step in t.steps:
            if step.selected_tool is None:
                continue
            tool_steps += 1
            if step.tool_execution_success:
                successful_tool_steps += 1
                if seen_bad_call:
                    recovered_after_bad += 1
            else:
                bad_calls += 1
                seen_bad_call = True
    m.tool_call_accuracy_per_step = (
        successful_tool_steps / tool_steps if tool_steps else 0.0
    )
    m.recovery_rate_after_bad_call = (
        recovered_after_bad / bad_calls if bad_calls else 0.0
    )

    # --- fast-path utilization ----------------------------------------------- #
    spancall_eligible = 0
    spancall_hits = 0
    for t in traces:
        for step in t.steps:
            d = step.decision
            if d in (RoutingDecision.SPANCALL_HIT, RoutingDecision.SPANCALL_ABSTAIN):
                spancall_eligible += 1
                if d == RoutingDecision.SPANCALL_HIT:
                    spancall_hits += 1
    m.spancall_hit_rate = spancall_hits / spancall_eligible if spancall_eligible else 0.0

    wrong_hits = sum(
        1
        for t in traces
        for step in t.steps
        if step.decision == RoutingDecision.SPANCALL_HIT
        and not step.tool_execution_success
    )
    m.wrong_hit_rate = wrong_hits / spancall_hits if spancall_hits else 0.0
    m.side_effecting_error_count = sum(
        1
        for t in traces
        for step in t.steps
        if not step.tool_execution_success
        and (step.spancall_features or {}).get("risk_tier") == "side_effecting"
    )

    # --- resource savings ---------------------------------------------------- #
    baseline_llm_calls = sum(t.total_llm_calls + t.total_spancall_calls for t in traces)
    actual_llm_calls = sum(t.total_llm_calls for t in traces)
    m.total_llm_calls_saved = max(0, baseline_llm_calls - actual_llm_calls)

    return m


__all__ = ["AgentMetrics", "aggregate_metrics"]
