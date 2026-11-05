"""Evaluation subpackage for SpanCall."""

from spancall.eval.cost import CostMetrics, CostModel
from spancall.eval.toolcall import BaseToolCallEvaluator, ToolCallMetrics
from spancall.eval.agentic import (
    AgentMetrics,
    BaseAgentEvaluator,
    MultiSeedAgentEvaluator,
    aggregate_metrics,
)
from spancall.eval.compare import ComparativeEvaluator, ComparativeStudyResult

__all__ = [
    "AgentMetrics",
    "BaseAgentEvaluator",
    "MultiSeedAgentEvaluator",
    "aggregate_metrics",
    "BaseToolCallEvaluator",
    "ComparativeEvaluator",
    "ComparativeStudyResult",
    "CostMetrics",
    "CostModel",
    "ToolCallMetrics",
]
