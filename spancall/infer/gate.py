"""Whole-call confidence gating.

Ref: docs/mechanism.md §7.3, docs/remediation_plan.md D4.

v1 default (D4): **single-τ gate** — hit iff every parameter's top-1 probability
is at least τ, the call-level accept probability is at least τ_a, and the schema
validates. The joint-margin check (tau_delta) is *dropped from the v1 decision*;
it stays available as a calibration signal and in the ``conjunction`` ablation.

    hit <=> min_k p_k >= tau_p   AND   accept >= tau_a   AND   validate()

`gate.mode` (configs/default.yaml) selects the implementation; this module is the
single place the mode switch lives (D4):
  - ``single_tau``  -> SingleTauGate            (default)
  - ``conjunction`` -> ConjunctionGate          (ablation; 8-feature style)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseGate(ABC):
    """Abstract whole-call confidence gate deciding hit vs abstain."""

    @abstractmethod
    def evaluate(
        self,
        param_probabilities: Dict[str, float],
        joint_margin: float,
        accept_probability: float,
        is_schema_valid: bool,
        risk_tier: str = "read_only",
    ) -> Tuple[bool, Optional[str]]:
        """Determines whether to trigger fast-path hit or abstain to LLM.

        Returns:
            (is_hit, rejection_reason_if_abstain)
        """
        raise NotImplementedError


class SingleTauGate(BaseGate):
    """v1 minimal gate (D4): min_k p_k >= tau_p AND accept >= tau_a AND validate().

    The joint margin is accepted but deliberately not part of the decision in v1
    (the docs' whole-call gate is over-specified before its ablations exist); it
    is recorded on the rejection reason for telemetry when present.
    """

    def __init__(
        self,
        tau_p: float = 0.90,
        tau_a: float = 0.85,
        tau_delta: float = 2.0,   # informational in single-τ mode
    ) -> None:
        self.tau_p = tau_p
        self.tau_a = tau_a
        self.tau_delta = tau_delta

    def evaluate(
        self,
        param_probabilities: Dict[str, float],
        joint_margin: float,
        accept_probability: float,
        is_schema_valid: bool,
        risk_tier: str = "read_only",
    ) -> Tuple[bool, Optional[str]]:
        if not param_probabilities:
            return False, "no_params"
        min_p = min(param_probabilities.values())
        if min_p < self.tau_p:
            return False, f"min_p_below_tau ({min_p:.3f} < {self.tau_p})"
        if accept_probability < self.tau_a:
            return False, f"accept_below_tau_a ({accept_probability:.3f} < {self.tau_a})"
        if not is_schema_valid:
            return False, "schema_invalid"
        return True, None


class ConjunctionGate(BaseGate):
    """Calibrated conjunction baseline gate (8-feature style, heads.md §5).

    hit <=> min_k p_k >= tau_p AND Delta_joint >= tau_delta AND a >= tau_a AND
    validate(). Kept as an ablation behind ``gate.mode: conjunction``; the joint
    margin and 8-feature machinery are NOT the v1 default (D4).
    """

    def __init__(
        self,
        tau_p: float = 0.90,
        tau_delta: float = 2.0,
        tau_a: float = 0.85,
    ) -> None:
        self.tau_p = tau_p
        self.tau_delta = tau_delta
        self.tau_a = tau_a

    def evaluate(
        self,
        param_probabilities: Dict[str, float],
        joint_margin: float,
        accept_probability: float,
        is_schema_valid: bool,
        risk_tier: str = "read_only",
    ) -> Tuple[bool, Optional[str]]:
        if not param_probabilities:
            return False, "no_params"
        min_p = min(param_probabilities.values())
        if min_p < self.tau_p:
            return False, f"min_p_below_tau ({min_p:.3f} < {self.tau_p})"
        if joint_margin < self.tau_delta:
            return False, f"joint_margin_below_tau_delta ({joint_margin:.3f} < {self.tau_delta})"
        if accept_probability < self.tau_a:
            return False, f"accept_below_tau_a ({accept_probability:.3f} < {self.tau_a})"
        if not is_schema_valid:
            return False, "schema_invalid"
        return True, None


def build_gate(
    mode: str = "single_tau",
    tau_p: float = 0.90,
    tau_delta: float = 2.0,
    tau_a: float = 0.85,
) -> BaseGate:
    """Factory for the configured gate (D4: this module owns the mode switch).

    Args:
        mode: ``single_tau`` (default) or ``conjunction``.
    """
    if mode == "single_tau":
        return SingleTauGate(tau_p=tau_p, tau_a=tau_a, tau_delta=tau_delta)
    if mode == "conjunction":
        return ConjunctionGate(tau_p=tau_p, tau_delta=tau_delta, tau_a=tau_a)
    raise ValueError(f"unknown gate.mode {mode!r} (expected single_tau | conjunction)")
