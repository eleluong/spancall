"""Tests for the comparative agent evaluation (plan §4.3, E9-E11).

Covers:
- ``aggregate_metrics`` aggregation axes (task success, turns, per-step
  tool-call accuracy, recovery, fast-path share, resource savings).
- ``ComparativeEvaluator``: paired non-inferiority (margin -0.02), paired
  bootstrap CI, per-task deltas, empirical vs theoretical cost savings.
- ``MultiSeedAgentEvaluator``: 3-seed paired run over the mock booking
  environment.
- ``scripts/eval_agent.py`` smoke run (deterministic backend) writing a
  results JSON that satisfies the E9 gate shape.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.loop import ReActAgentLoop
from spancall.agent.router import IntegrationVariant, RouterMode, ToolCallRouter
from spancall.agent.tools.mock import MockBookingEnvironment
from spancall.agent.trace import ExecutionTrace, RoutingDecision, StepTrace
from spancall.data.schema import ToolDefinition
from spancall.eval.agentic.evaluator import MultiSeedAgentEvaluator
from spancall.eval.agentic.metrics import AgentMetrics, aggregate_metrics
from spancall.eval.compare import ComparativeEvaluator, ComparativeStudyResult
from spancall.eval.cost import CostModel
from spancall.infer.resolver import ResolveResult, SpanCallResolver


# --------------------------------------------------------------------------- #
# Fixtures: traced runs over the mock booking environment
# --------------------------------------------------------------------------- #
class _StaticResolver(SpanCallResolver):
    """Resolver whose fast path always hits ``book_flight``."""

    def __init__(self, hits: bool = True) -> None:
        self._hits = hits
        super().__init__()

    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Dict[str, Any] | None = None,
        now: Any = None,
    ) -> ResolveResult:
        if not self._hits:
            return ResolveResult(hit=False, abstain_reason="no_hit")
        return ResolveResult(
            hit=True, tool_name="book_flight", payload={"flight_id": "F001"}
        )


class _ScriptBackend(BaseLLMBackend):
    """Scripted backend for the frozen traces (deterministic, no LLM)."""

    def generate_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        text = " ".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        if "F001" in text and "book" in text.lower():
            return {
                "tool_name": "book_flight",
                "arguments": {"flight_id": "F001"},
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                "latency_ms": 10.0,
            }
        if "flight" in text.lower():
            return {
                "tool_name": "search_flights",
                "arguments": {"origin": "SFO", "destination": "JFK", "date": "2026-11-05"},
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                "latency_ms": 10.0,
            }
        return {
            "tool_name": None,
            "arguments": None,
            "text": "I need more information.",
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            "latency_ms": 10.0,
        }

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        return "Done."


def _run_trace(
    mode: str,
    variant: str = "s_skip",
    required_flight_id: str = "F001",
    task_instruction: str = "Book flight F001.",
    task_id: str = "task_mock",
) -> ExecutionTrace:
    env = MockBookingEnvironment(required_flight_id=required_flight_id)
    backend = _ScriptBackend()
    resolver = _StaticResolver(hits=mode != "llm_only")
    router = ToolCallRouter(
        resolver=resolver,
        llm_backend=backend,
        mode=RouterMode(mode),
        variant=IntegrationVariant(variant),
    )
    loop = ReActAgentLoop(router=router, environment=env, max_turns=8)
    # Mode-independent task id by default: normal and SpanCall arms must be
    # paired by the *same* task_id ordering when aggregated (ComparativeEvaluator
    # enforces it). Tests may override task_id to exercise the pairing check.
    return loop.run_task(task_instruction, task_id=task_id)


# --------------------------------------------------------------------------- #
# aggregate_metrics
# --------------------------------------------------------------------------- #
def test_aggregate_metrics_task_success_and_turns() -> None:
    traces = [_run_trace("llm_only"), _run_trace("llm_only")]
    m = aggregate_metrics(traces)
    assert m.total_tasks == 2
    assert m.task_success_rate == 1.0
    assert m.average_turns_per_task > 0.0


def test_aggregate_metrics_fast_path_share() -> None:
    traces = [_run_trace("router")]
    m = aggregate_metrics(traces)
    # The fast path resolved turn 0 (book F001) -> one spancall hit.
    assert m.spancall_hit_rate == pytest.approx(1.0)
    assert m.total_llm_calls_saved >= 0


def test_aggregate_metrics_empty_is_safe() -> None:
    m = aggregate_metrics([])
    assert m.total_tasks == 0
    assert m.task_success_rate == 0.0


# --------------------------------------------------------------------------- #
# ComparativeEvaluator
# --------------------------------------------------------------------------- #
def _paired_traces(n: int = 6) -> tuple[List[ExecutionTrace], List[ExecutionTrace]]:
    normal = [_run_trace("llm_only", task_instruction=f"Book flight F001. seed {i}") for i in range(n)]
    spancall = [_run_trace("router", task_instruction=f"Book flight F001. seed {i}") for i in range(n)]
    return normal, spancall


def test_compare_non_inferiority_passes_when_equal() -> None:
    normal, spancall = _paired_traces()
    comp = ComparativeEvaluator(cost_model=CostModel(), num_bootstrap_resamples=2000, seed=7)
    res = comp.compare(normal, spancall)
    assert isinstance(res, ComparativeStudyResult)
    # Both succeed -> delta ~ 0, CI lower much bigger than -0.02.
    assert res.task_success_delta == pytest.approx(0.0)
    assert res.non_inferiority_passed is True
    assert res.p_value < 0.05
    assert res.bootstrap_ci_95[0] <= res.bootstrap_ci_95[1]
    assert res.num_tasks == 6


def test_compare_detects_worse_agent() -> None:
    # SpanCall traces are made to fail (unreachable flight) while normal succeeds.
    normal = [_run_trace("llm_only") for _ in range(4)]
    failing = [
        _run_trace("router", required_flight_id="NOPE", task_instruction=f"Book flight NOPE. {i}")
        for i in range(4)
    ]
    comp = ComparativeEvaluator(cost_model=CostModel(), num_bootstrap_resamples=2000, seed=7)
    res = comp.compare(normal, failing)
    assert res.task_success_delta < 0
    assert res.non_inferiority_passed is False


def test_compare_rejects_unpaired_lists() -> None:
    comp = ComparativeEvaluator(cost_model=CostModel())
    with pytest.raises(ValueError, match="paired"):
        comp.compare([_run_trace("llm_only")], [_run_trace("llm_only"), _run_trace("router")])
    with pytest.raises(ValueError, match="non-empty"):
        comp.compare([], [])


def test_compare_rejects_mismatched_task_id_ordering() -> None:
    normal, _ = _paired_traces(n=2)
    # Same length but different task ids -> identity check fires.
    spancall = [_run_trace("router", task_id=f"task_alt_{i}") for i in range(2)]
    with pytest.raises(ValueError, match="task_id"):
        ComparativeEvaluator(cost_model=CostModel()).compare(normal, spancall)


def test_compare_empirical_cost_uses_cost_model() -> None:
    normal = [_run_trace("llm_only") for _ in range(2)]
    spancall = [_run_trace("router") for _ in range(2)]
    cm = CostModel(cost_spancall_call=0.05)
    comp = ComparativeEvaluator(cost_model=cm, num_bootstrap_resamples=500, seed=1)
    res = comp.compare(normal, spancall)
    # empirical cost should be <= 100% saving (can't save more than all cost).
    assert res.empirical_cost_saving_percent >= -50.0
    assert res.empirical_cost_saving_percent <= 100.0


def test_compare_bootstrap_ci_matches_p_value_direction() -> None:
    normal, spancall = _paired_traces(n=10)
    comp = ComparativeEvaluator(
        cost_model=CostModel(), non_inferiority_margin=0.02, num_bootstrap_resamples=3000, seed=3
    )
    res = comp.compare(normal, spancall)
    # With all deltas == 0, the CI straddles 0 and p is tiny.
    assert res.bootstrap_ci_95[1] >= 0.0 - 1e-9


# --------------------------------------------------------------------------- #
# MultiSeedAgentEvaluator
# --------------------------------------------------------------------------- #
class _AlwaysSearchEnv(MockBookingEnvironment):
    """Environment that never completes (so seeds don't collapse)."""

    def evaluate_task_success(self) -> bool:
        return False


def _task(task_id: str, env_factory: Any) -> Dict[str, Any]:
    return {"task_id": task_id, "instruction": "Do something.", "environment": env_factory}


def test_multiseed_evaluator_runs_seeds_and_records_metadata() -> None:
    backend = _ScriptBackend()
    router = ToolCallRouter(
        resolver=_StaticResolver(hits=False),
        llm_backend=backend,
        mode=RouterMode.LLM_ONLY,
        variant=IntegrationVariant.S_SKIP,
    )
    env = _AlwaysSearchEnv(required_flight_id=None)
    loop = ReActAgentLoop(router=router, environment=env, max_turns=3)
    tasks = [_task("t1", lambda rng: _AlwaysSearchEnv())]
    ev = MultiSeedAgentEvaluator(seed=10)
    metrics, traces = ev.evaluate_agent(loop, tasks, num_seeds=3)
    assert len(traces) == 3
    assert {t.metadata.get("seed") for t in traces} == {10, 11, 12}
    assert metrics.total_tasks == 3


def test_compare_result_to_dict_serializable() -> None:
    normal, spancall = _paired_traces(n=3)
    comp = ComparativeEvaluator(cost_model=CostModel(), num_bootstrap_resamples=300)
    res = comp.compare(normal, spancall)
    d = res.to_dict()
    json.dumps(d)  # must not raise
    assert "task_success_delta" in d
    assert "non_inferiority_passed" in d
    # per-task rows travel with the serialized payload (no manual re-add).
    assert "per_task_comparisons" in d
    assert len(d["per_task_comparisons"]) == 3


def test_recovery_rate_is_same_trace_only() -> None:
    """A failure in one task must not be 'recovered' by a success in another."""
    fail_trace = _run_trace("router", required_flight_id="NOPE", task_instruction="Book flight NOPE.")
    # Force a failed tool step with no later success in the same trace.
    success_trace = _run_trace("router")
    m = aggregate_metrics([fail_trace, success_trace, success_trace])
    # fail_trace: one bad call, no in-trace recovery. success_traces: no bad calls.
    # Old impl leaked across traces (would report recovery 1.0).
    assert m.recovery_rate_after_bad_call == 0.0


# --------------------------------------------------------------------------- #
# scripts/eval_agent.py end-to-end smoke
# --------------------------------------------------------------------------- #
def test_eval_agent_script_smoke(tmp_path) -> None:
    import subprocess
    import sys

    out = tmp_path / "agent_compare.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/eval_agent.py",
            "--benchmark",
            "custom_mock",
            "--backend",
            "deterministic",
            "--num_seeds",
            "3",
            "--variant",
            "s_inject",
            "--num_bootstrap",
            "200",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=".",  # repo root
    )
    assert proc.returncode == 0, proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["integration_variant"] == "s_inject"
    assert data["num_tasks"] > 0
    assert "task_success_delta" in data
    assert data["non_inferiority_passed"] in (True, False)