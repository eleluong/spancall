"""Offline tool-call evaluation metrics and error taxonomy.

Ref: docs/mechanism.md §14, docs/proposal.md §6.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolCallMetrics:
    """Tool-call evaluation metrics reported across strata."""
    # Core performance
    total_samples: int = 0
    tool_accuracy: float = 0.0
    per_param_accuracy: float = 0.0
    exact_payload_match: float = 0.0

    # Multi-call (R0.4 / J6). ``exact_payload_match`` stays call-0-only so its
    # historical meaning is unchanged; these score the predicted call SET.
    call_exact_set_match: float = 0.0
    call_count_acc: float = 0.0
    n_calls_hist: Dict[str, int] = field(default_factory=dict)
    pred_n_calls_hist: Dict[str, int] = field(default_factory=dict)

    # Partial extraction performance (P5)
    partial_extraction_rate: float = 0.0
    param_precision: float = 0.0
    param_recall: float = 0.0
    param_f1: float = 0.0

    # Gating and coverage
    hit_rate: float = 0.0
    precision_at_hit: float = 0.0
    false_hit_rate_on_a: float = 0.0  # Crucial safety check on stratum A
    collapse_hits: int = 0
    collapse_hit_rate: float = 0.0

    # Error taxonomy
    error_wrong_tool: int = 0
    error_wrong_value: int = 0
    error_entity_swap: int = 0
    error_duplicate_span: int = 0
    error_date_misparse: int = 0
    error_extra_call: int = 0
    error_missing_call: int = 0

    # Stratified breakdown
    metrics_by_stratum: Dict[str, Dict[str, float]] = field(default_factory=dict)
    abstain_reasons: Dict[str, int] = field(default_factory=dict)

    # Calibration + error detail
    ece: float = 0.0
    risk_coverage: List[Dict[str, Any]] = field(default_factory=list)
    errors_by_sample: Dict[str, List[str]] = field(default_factory=dict)
