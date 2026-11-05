"""Multi-turn agent evaluator interface."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Tuple

from spancall.agent.loop import BaseAgentLoop
from spancall.agent.trace import ExecutionTrace
from spancall.eval.agentic.metrics import AgentMetrics, aggregate_metrics


class BaseAgentEvaluator(ABC):
    """Executes agent evaluation on multi-turn benchmarks (e.g., tau-bench)."""

    @abstractmethod
    def evaluate_agent(
        self,
        agent_loop: BaseAgentLoop,
        task_suite: List[Dict[str, Any]],
        num_seeds: int = 3,
    ) -> Tuple[AgentMetrics, List[ExecutionTrace]]:
        """Runs test tasks across multiple random seeds and aggregates task-level metrics."""
        raise NotImplementedError


class MultiSeedAgentEvaluator(BaseAgentEvaluator):
    """Runs a task suite across multiple seeds and aggregates paired traces.

    Each task dict carries at least:
        - ``task_id``: unique identifier.
        - ``instruction``: the user instruction handed to ``run_task``.
        - ``environment``: a :class:`BaseToolEnvironment` instance (or a
          zero-arg factory) to bind to the loop for a run.

    The seed controls environment stochasticity (via a shared RNG) and is
    recorded in trace ``metadata`` so paired runs (normal vs SpanCall) line
    up 1-to-1 when the caller alternates seeds.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)

    def evaluate_agent(
        self,
        agent_loop: BaseAgentLoop,
        task_suite: List[Dict[str, Any]],
        num_seeds: int = 3,
    ) -> Tuple[AgentMetrics, List[ExecutionTrace]]:
        traces: List[ExecutionTrace] = []
        for seed_offset in range(max(1, int(num_seeds))):
            seed = self.seed + seed_offset
            rng = random.Random(seed)
            for task in task_suite:
                env = task["environment"]
                if callable(env):
                    env = env(rng)  # factory gets the seed RNG
                # Bind the loop to this task's environment for the run.
                if hasattr(agent_loop, "environment") and agent_loop.environment is not env:
                    agent_loop.environment = env  # type: ignore[attr-defined]
                instruction = task.get("instruction", task.get("task_instruction", ""))
                trace = agent_loop.run_task(str(instruction), task_id=str(task["task_id"]))
                trace.metadata["seed"] = seed
                traces.append(trace)
        return aggregate_metrics(traces), traces
