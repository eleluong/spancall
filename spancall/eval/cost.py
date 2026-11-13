"""Cost and compute accounting model.

Ref: docs/mechanism.md §1.
Formula:
  E[cost] = c_s + (1 - h) * c_L + h * (1 - p) * (c_w + c_L)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostMetrics:
    """Detailed compute and financial metrics per call or task."""
    encoder_tflops: float = 0.0
    llm_tflops: float = 0.0
    total_tflops: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    estimated_usd: float = 0.0


class CostModel:
    """Empirical cost and FLOPs calculator."""

    def __init__(
        self,
        cost_spancall_call: float = 0.07,
        cost_llm_call: float = 1.0,
        cost_wrong_call_penalty: float = 1.0,
    ) -> None:
        self.c_s = cost_spancall_call
        self.c_L = cost_llm_call
        self.c_w = cost_wrong_call_penalty

    def theoretical_expected_cost(self, hit_rate: float, precision_at_hit: float) -> float:
        """Computes expected normalized cost per call."""
        h = hit_rate
        p = precision_at_hit
        return self.c_s + (1.0 - h) * self.c_L + h * (1.0 - p) * (self.c_w + self.c_L)

    def compute_break_even_hit_rate(self, precision_at_hit: float = 0.95) -> float:
        """Finds minimum hit rate where fast path saves compute."""
        p = precision_at_hit
        denominator = self.c_L - (1.0 - p) * (self.c_w + self.c_L)
        if denominator <= 0:
            return 1.0
        return self.c_s / denominator
