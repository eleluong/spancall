"""Telemetry and execution trace schemas for agentic evaluation and shadow mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RoutingDecision(str, Enum):
    SPANCALL_HIT = "spancall_hit"
    SPANCALL_ABSTAIN = "spancall_abstain"
    LLM_FALLBACK = "llm_fallback"
    SHADOW_COMPARISON = "shadow_comparison"


@dataclass
class StepTrace:
    """Telemetry recorded for a single agent turn."""
    turn_index: int
    user_input: str
    decision: RoutingDecision
    selected_tool: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    ground_truth_arguments: Optional[Dict[str, Any]] = None
    spancall_features: Dict[str, Any] = field(default_factory=dict)
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    latency_ms: float = 0.0
    tool_execution_success: bool = True
    tool_output: Optional[str] = None
    context_trimmed: int = 0


@dataclass
class ExecutionTrace:
    """Full task trajectory trace."""
    task_id: str
    environment_name: str
    mode: str  # "normal_agent", "spancall_agent", "shadow"
    integration_variant: str  # "s_skip", "s_inject", "s_hint"
    steps: List[StepTrace] = field(default_factory=list)
    task_completed: bool = False
    task_success: bool = False
    total_latency_ms: float = 0.0
    total_tokens_prompt: int = 0
    total_tokens_completion: int = 0
    total_spancall_calls: int = 0
    total_llm_calls: int = 0
    total_context_trimmed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
