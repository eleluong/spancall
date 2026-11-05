"""Comparative evaluation: Agent + Tool Span (SpanCall) vs Normal Agent.

Rigorous paired statistical comparison:
- Non-inferiority test on task success rate (margin delta, e.g. -2%)
- Paired bootstrap confidence intervals
- Empirical cost model validation vs docs/mechanism.md §1

Protocol (docs/architecture.md §8.3, full_implementation_plan.md §4.3):
same LLM, tools and tasks; only the router differs. ``normal_agent_traces``
(S) and ``spancall_agent_traces`` (B0) must be **paired** 1-to-1 per task
(identical ``task_id`` ordering), aggregated over >= 3 seeds.

The non-inferiority test is one-sided: H0 = SpanCall is worse than the normal
agent by more than ``margin`` (0.02 = 2 points). We pass when the lower bound
of the 95% bootstrap CI on the mean paired delta is >= -margin.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from spancall.agent.trace import ExecutionTrace
from spancall.eval.agentic.metrics import AgentMetrics
from spancall.eval.agentic.metrics import aggregate_metrics
from spancall.eval.cost import CostMetrics, CostModel


@dataclass
class ComparativeStudyResult:
    """Formal research comparison between baseline normal agent and SpanCall agent."""
    baseline_metrics: AgentMetrics
    spancall_metrics: AgentMetrics
    task_success_delta: float = 0.0
    non_inferiority_passed: bool = False
    bootstrap_ci_95: Tuple[float, float] = (0.0, 0.0)
    p_value: float = 1.0

    # Resource differences
    speedup_ratio: float = 1.0
    token_savings_percent: float = 0.0
    theoretical_cost_saving_percent: float = 0.0
    empirical_cost_saving_percent: float = 0.0

    # Explains the accounting basis when the theoretical and empirical cost
    # series disagree (e.g. an all-abstain resolver: the formula charges the
    # per-call attempt cost c_s at hit_rate=0 while the empirical side only
    # bills executed fast-path/LLM calls).
    theoretical_cost_note: str = ""

    per_task_comparisons: List[Dict[str, Any]] = field(default_factory=list)

    # Number of tasks / seeds actually compared.
    num_tasks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable payload for ``results/`` artifacts."""
        return {
            "baseline_metrics": _metrics_to_dict(self.baseline_metrics),
            "spancall_metrics": _metrics_to_dict(self.spancall_metrics),
            "task_success_delta": self.task_success_delta,
            "non_inferiority_passed": self.non_inferiority_passed,
            "bootstrap_ci_95": list(self.bootstrap_ci_95),
            "p_value": self.p_value,
            "speedup_ratio": self.speedup_ratio,
            "token_savings_percent": self.token_savings_percent,
            "theoretical_cost_saving_percent": self.theoretical_cost_saving_percent,
            "empirical_cost_saving_percent": self.empirical_cost_saving_percent,
            "theoretical_cost_note": self.theoretical_cost_note,
            "num_tasks": self.num_tasks,
            "per_task_comparisons": self.per_task_comparisons,
        }


class ComparativeEvaluator:
    """Orchestrates head-to-head evaluation between Normal Agent and Agent + SpanCall."""

    def __init__(
        self,
        cost_model: CostModel,
        non_inferiority_margin: float = 0.02,
        num_bootstrap_resamples: int = 10000,
        seed: int = 42,
    ) -> None:
        self.cost_model = cost_model
        self.non_inferiority_margin = non_inferiority_margin
        self.num_bootstrap_resamples = int(num_bootstrap_resamples)
        self.seed = seed

    def compare(
        self,
        normal_agent_traces: List[ExecutionTrace],
        spancall_agent_traces: List[ExecutionTrace],
    ) -> ComparativeStudyResult:
        """Performs paired statistical analysis across identical task environments.

        Args:
            normal_agent_traces: B0 (normal agent) traces, one per task per seed.
            spancall_agent_traces: S (Agent + SpanCall) traces, paired 1-to-1
                with ``normal_agent_traces`` in the same (task, seed) order.

        Raises:
            ValueError: when the lists are empty or not paired 1-to-1.
        """
        if not normal_agent_traces or not spancall_agent_traces:
            raise ValueError("compare() requires non-empty trace lists")
        if len(normal_agent_traces) != len(spancall_agent_traces):
            raise ValueError(
                f"trace lists must be paired 1-to-1; got "
                f"{len(normal_agent_traces)} vs {len(spancall_agent_traces)}"
            )
        normal_ids = [t.task_id for t in normal_agent_traces]
        spancall_ids = [t.task_id for t in spancall_agent_traces]
        if normal_ids != spancall_ids:
            raise ValueError(
                "trace lists must be paired by identical task_id ordering; "
                f"normal={normal_ids!r} != spancall={spancall_ids!r}"
            )

        n = len(normal_agent_traces)
        baseline_metrics = aggregate_metrics(normal_agent_traces)
        spancall_metrics = aggregate_metrics(spancall_agent_traces)

        # ---- paired per-task deltas -------------------------------------- #
        per_task: List[Dict[str, Any]] = []
        deltas: List[float] = []
        speedups: List[float] = []
        for i, (b, s) in enumerate(zip(normal_agent_traces, spancall_agent_traces)):
            b_success = 1.0 if b.task_success else 0.0
            s_success = 1.0 if s.task_success else 0.0
            delta = s_success - b_success
            deltas.append(delta)
            speedup = _safe_speedup(b.total_latency_ms, s.total_latency_ms)
            speedups.append(speedup)
            per_task.append(
                {
                    "task_id": s.task_id or b.task_id or f"task_{i}",
                    "baseline_success": b_success,
                    "spancall_success": s_success,
                    "delta": delta,
                    "baseline_latency_ms": b.total_latency_ms,
                    "spancall_latency_ms": s.total_latency_ms,
                    "speedup_ratio": speedup,
                    "baseline_llm_calls": b.total_llm_calls,
                    "spancall_llm_calls": s.total_llm_calls,
                    "spancall_hits": s.total_spancall_calls,
                }
            )

        mean_delta = sum(deltas) / n
        ci_lower, ci_upper, p_value = self._paired_bootstrap(deltas)

        # ---- resource deltas --------------------------------------------- #
        speedup_ratio = _safe_speedup(
            sum(t.total_latency_ms for t in normal_agent_traces),
            sum(t.total_latency_ms for t in spancall_agent_traces),
        )
        baseline_tokens = sum(
            t.total_tokens_prompt + t.total_tokens_completion for t in normal_agent_traces
        )
        spancall_tokens = sum(
            t.total_tokens_prompt + t.total_tokens_completion for t in spancall_agent_traces
        )
        token_savings_percent = _pct_change(baseline_tokens, spancall_tokens)

        # Theoretical cost model validation: feed the measured hit-rate and
        # precision-at-hit of the SpanCall fast path into the cost formula and
        # compare against the empirical per-task cost reduction.
        metrics = self._fast_path_metrics(spancall_agent_traces)
        h, p_hit = metrics
        theoretical = self.cost_model.theoretical_expected_cost(h, p_hit)
        theoretical_cost_saving_percent = (1.0 - theoretical) * 100.0

        baseline_cost = self._empirical_cost(normal_agent_traces, self.cost_model.c_s)
        spancall_cost = self._empirical_cost(spancall_agent_traces, self.cost_model.c_s)
        empirical_cost_saving_percent = _pct_change(baseline_cost, spancall_cost)

        theoretical_cost_note = (
            ""
            if h > 0.0
            else (
                f"hit_rate=0: theoretical formula charges the per-call attempt "
                f"cost c_s={self.cost_model.c_s} on every turn (E[cost] = "
                f"{1.0 + self.cost_model.c_s:.3f} vs baseline 1.0), while the "
                f"empirical side only bills executed fast-path/LLM calls "
                f"(ROUTER-mode abstains bill nothing). See docs/mechanism.md §1."
            )
        )

        return ComparativeStudyResult(
            baseline_metrics=baseline_metrics,
            spancall_metrics=spancall_metrics,
            task_success_delta=mean_delta,
            non_inferiority_passed=bool(ci_lower >= -self.non_inferiority_margin),
            bootstrap_ci_95=(ci_lower, ci_upper),
            p_value=p_value,
            speedup_ratio=speedup_ratio,
            token_savings_percent=token_savings_percent,
            theoretical_cost_saving_percent=theoretical_cost_saving_percent,
            empirical_cost_saving_percent=empirical_cost_saving_percent,
            theoretical_cost_note=theoretical_cost_note,
            per_task_comparisons=per_task,
            num_tasks=n,
        )

    def _paired_bootstrap(
        self,
        deltas: Sequence[float],
    ) -> Tuple[float, float, float]:
        """Paired bootstrap CI + one-sided non-inferiority p-value.

        Resamples the *paired* delta vector with replacement
        (``num_bootstrap_resamples`` times); the 95% CI is the 2.5/97.5
        percentiles of resample means. The one-sided p-value is the fraction
        of resample means below ``-margin`` (i.e. the probability, under the
        observed data, that SpanCall is worse than the agent by more than the
        margin); lower is stronger evidence of non-inferiority.
        """
        n = len(deltas)
        if n == 0:
            return 0.0, 0.0, 1.0
        rng = random.Random(self.seed)
        margin = self.non_inferiority_margin
        means: List[float] = []
        below: int = 0
        for _ in range(self.num_bootstrap_resamples):
            total = 0.0
            for _ in range(n):
                total += deltas[rng.randrange(n)]
            mean = total / n
            means.append(mean)
            if mean < -margin:
                below += 1
        means.sort()
        lo = means[int(0.025 * len(means))]
        hi = means[int(0.975 * len(means))]
        p_value = below / len(means)
        return lo, hi, p_value

    @staticmethod
    def _fast_path_metrics(traces: Sequence[ExecutionTrace]) -> Tuple[float, float]:
        """(hit_rate, precision_at_hit) over all SpanCall-eligible turns.

        A turn counts as a fast-path candidate when the trace enabled the
        fast path (``spancall_calls`` ran). Among candidates, hit_rate is the
        share of turns SpanCall actually resolved; precision_at_hit is the
        share of those hits that executed successfully in the environment.
        """
        candidates = 0
        hits = 0
        hit_success = 0
        for t in traces:
            for step in t.steps:
                if step.decision.value in ("spancall_hit", "spancall_abstain"):
                    candidates += 1
                    if step.decision.value == "spancall_hit":
                        hits += 1
                        if step.tool_execution_success:
                            hit_success += 1
        if candidates == 0:
            return 0.0, 0.0
        hit_rate = hits / candidates
        precision = (hit_success / hits) if hits else 0.0
        return hit_rate, precision

    @staticmethod
    def _empirical_cost(traces: Sequence[ExecutionTrace], cost_spancall: float = 0.07) -> float:
        """Normalized empirical cost: LLM calls cost 1, SpanCall calls cost c_s.

        Mirrors the CostModel formula's normalization (c_L = 1). Per task the
        cost is ``c_s * spancall_calls + 1 * llm_calls``.
        """
        total = 0.0
        for t in traces:
            total += t.total_llm_calls + t.total_spancall_calls * cost_spancall
        return total


def _metrics_to_dict(m: AgentMetrics) -> Dict[str, Any]:
    d = {k: getattr(m, k) for k in _AGENT_METRIC_FIELDS}
    return d


_AGENT_METRIC_FIELDS: Tuple[str, ...] = (
    "total_tasks",
    "task_success_rate",
    "average_turns_per_task",
    "tool_call_accuracy_per_step",
    "recovery_rate_after_bad_call",
    "spancall_hit_rate",
    "wrong_hit_rate",
    "side_effecting_error_count",
    "total_llm_calls_saved",
)


def _safe_speedup(baseline_ms: float, spancall_ms: float) -> float:
    if baseline_ms <= 0 or spancall_ms <= 0:
        return 1.0
    return baseline_ms / spancall_ms


def _pct_change(baseline: float, new: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - new) / baseline * 100.0


# Re-export so callers can build CostMetrics when measuring.
__all__ = [
    "ComparativeEvaluator",
    "ComparativeStudyResult",
    "CostMetrics",
]
