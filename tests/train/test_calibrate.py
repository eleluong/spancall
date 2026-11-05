"""Tests for :class:`spancall.train.calibrate.SpanCallCalibrator`.

Covers (docs/full_implementation_plan.md §3.5, docs/mechanism.md §7.4):

* ``clopper_pearson_lower_bound`` — known analytical values, edge cases
  (k=0, n=0), monotonicity, cross-check against ``scipy.stats.beta.ppf``, and
  the pure-Python ``_beta_ppf`` fallback.
* ``fit_temperature`` — L-BFGS over log T: identity when inputs are absent,
  T > 1 when correct logits are overconfident, T < 1 when underconfident,
  T ≈ 1 on a flat/neutral target, and the 1e-3 floor clamp.
* ``sweep_thresholds`` — Clopper-Pearson honesty (100% precision on 10 rows
  does *not* license a 0.95 gate), hit-rate maximization under constraint,
  conservative tie-break at equal hit rate, and the n_hits==0/empty fallback.
* ``thresholds_per_tier`` — per-tier targets (0.95/0.97/0.99) produce
  different constraint results at the same n, plus the conservative fallback
  for empty tiers.
* ``calibrate``/``write_report`` — top-level keys round-trip into
  ``resolver._default_gate_cfg`` (mode/tau_p/tau_a/tau_delta/temperature/
  tau_tool).
* ``risk_coverage_curve`` / ``expected_calibration_error`` — monotone
  coverage, risk = 1 - precision, ECE on synthetic confidences.
"""

from __future__ import annotations

import json
import math

import pytest

from spancall.train.calibrate import (
    SpanCallCalibrator,
    TIER_TARGETS,
    _beta_ppf,
    _incbeta,
    _log_gamma,
    payloads_equal,
)
from spancall.train.config import CalibrationConfig

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - scipy-free test environments
    _scipy_stats = None


def _cal(target: float = 0.95, confidence: float = 0.95, mode: str = "single_tau",
         temp: bool = True) -> SpanCallCalibrator:
    return SpanCallCalibrator(
        CalibrationConfig(
            target_precision=target, confidence_level=confidence, mode=mode,
            temperature_scaling=temp,
        )
    )


def _row(sample_id: str, *, payload, probs: dict | None = None, accept: float = 0.99,
         schema_valid: bool = True, margin: float = 1.0, tier: str = "read_only") -> dict:
    return {
        "sample_id": sample_id,
        "tool_name": "t",
        "payload": payload,
        "param_probabilities": probs if probs is not None else {"a": 1.0},
        "accept_probability": accept,
        "schema_valid": schema_valid,
        "joint_margin": margin,
        "risk_tier": tier,
    }


def _gt(*rows: tuple) -> list[dict]:
    return [{"sample_id": r[0], "arguments": r[1]} for r in rows]


# --------------------------------------------------------------------------- #
# Clopper-Pearson lower bound
# --------------------------------------------------------------------------- #


def test_lbd_knowledge_values() -> None:
    cal = _cal()
    # LBD(90, 100; 0.95) ≈ 0.836282 (textbook + prior smoke value).
    assert cal.clopper_pearson_lower_bound(90, 100, 0.95) == pytest.approx(0.836282, abs=1e-5)
    # LBD(10, 10; 0.95) = 0.05 ** (1/10) ≈ 0.7411
    v = cal.clopper_pearson_lower_bound(10, 10, 0.95)
    assert v == pytest.approx(0.05 ** 0.1, abs=1e-4)
    assert 0.65 < v < 0.75


def test_lbd_edges() -> None:
    cal = _cal()
    assert cal.clopper_pearson_lower_bound(0, 50) == 0.0
    assert cal.clopper_pearson_lower_bound(5, 0) == 0.0
    assert cal.clopper_pearson_lower_bound(0, 0) == 0.0
    # higher confidence -> lower bound shrinks
    low_conf = cal.clopper_pearson_lower_bound(45, 50, 0.90)
    high_conf = cal.clopper_pearson_lower_bound(45, 50, 0.99)
    assert low_conf > high_conf > 0.0


def test_lbd_monotone_in_k() -> None:
    cal = _cal()
    a = cal.clopper_pearson_lower_bound(40, 50)
    b = cal.clopper_pearson_lower_bound(50, 50)
    assert a < b


@pytest.mark.skipif(_scipy_stats is None, reason="scipy not installed")
def test_lbd_matches_scipy() -> None:
    cal = _cal()
    for k, n in [(1, 10), (5, 20), (48, 50), (90, 100), (3, 3)]:
        want = float(_scipy_stats.beta.ppf(0.05, k, n - k + 1))
        assert cal.clopper_pearson_lower_bound(k, n, 0.95) == pytest.approx(want, abs=1e-10)


def test_pure_python_fallback_matches() -> None:
    got = _beta_ppf(0.05, 90.0, 100.0 - 90.0 + 1.0)
    assert got == pytest.approx(0.836282, abs=1e-3)
    # _incbeta sanity: I_0 = 0, I_1 = 1, and the CDF of Beta(1,1) is the identity.
    assert _incbeta(0.0, 1.0, 1.0) == 0.0
    assert _incbeta(1.0, 1.0, 1.0) == 1.0
    assert _incbeta(0.25, 1.0, 1.0) == pytest.approx(0.25, abs=1e-9)
    # _log_gamma(1) == 0 and gamma(n+1) == n!
    assert _log_gamma(1.0) == pytest.approx(0.0, abs=1e-12)
    assert math.exp(_log_gamma(6.0)) == pytest.approx(120.0, rel=1e-9)
    with pytest.raises(ValueError):
        _log_gamma(0.0)


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #


def test_fit_temperature_identity_without_inputs() -> None:
    # No pairs to fit -> identity regardless of torch availability.
    assert _cal().fit_temperature(None, None) == 1.0


def test_fit_temperature_overconfident_with_label_noise_cools() -> None:
    import torch

    # 70/30 class-0/1 labels with logits [3, 0]: the empirical frequency is
    # p=0.7 but the model always predicts class 0 with confidence sigmoid(3)
    # ~= 0.953 -> overconfident relative to the observed rate. NLL is
    # minimized by cooling: T* = 3 / ln(0.7/0.3) ~= 3.54 > 1.
    logits = torch.tensor([[3.0, 0.0]] * 100)
    labels = torch.tensor([0] * 70 + [1] * 30, dtype=torch.long)
    t = _cal().fit_temperature(logits, labels)
    assert t > 1.5


def test_fit_temperature_underconfident_sinks_below_one() -> None:
    import torch

    # 95/5 class-0/1 labels with underconfident logits [0.3, 0]: the model
    # predicts class 0 at sigmoid(0.3) ~= 0.574 but the empirical rate is 0.95
    # -> underconfident. NLL is minimized by sharpening:
    # T* = 0.3 / ln(0.95/0.05) ~= 0.102 < 1.
    logits = torch.tensor([[0.3, 0.0]] * 100)
    labels = torch.tensor([0] * 95 + [1] * 5, dtype=torch.long)
    t = _cal().fit_temperature(logits, labels)
    assert 0.0 < t < 0.8


def test_fit_temperature_neutral_target_identity() -> None:
    import torch

    # Flat logits with 50/50 soft labels: NLL == log(2) for every T -> T ~= 1.
    logits = torch.tensor([[[0.0, 0.0]] * 64])
    labels = torch.full((64, 2), 0.5)
    t = _cal().fit_temperature(logits, labels)
    assert 0.7 < t < 1.4


def test_fit_temperature_floor_clamp() -> None:
    import torch

    # All-correct two-class input: NLL is minimized by T -> 0, so the 1e-3
    # floor clamp is the active lower bound and T must land on it.
    logits = torch.tensor([[3.0, 0.0]] * 64)
    labels = torch.zeros(64, dtype=torch.long)
    t = _cal().fit_temperature(logits, labels)
    assert 0.0 < t < 1.0
    assert t >= 1e-3


# --------------------------------------------------------------------------- #
# Payload equality (shared gate signal)
# --------------------------------------------------------------------------- #


def test_payloads_equal_pure() -> None:
    assert payloads_equal({"a": 1}, {"a": 1}) is True
    assert payloads_equal({"a": 1}, {"a": 2}) is False
    assert payloads_equal({"a": 1}, None) is False
    assert payloads_equal(None, None) is False  # both None is not a hit
    assert payloads_equal({"a": [1, 2]}, {"a": [1, 2]}) is True
    assert payloads_equal({"a": {"b": 1}}, {"a": {"b": 1}}) is True


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #


def test_sweep_clopper_pearson_honesty_small_n() -> None:
    # 10/10 correct with target 0.95: max possible LBD(10, 10) = 0.741 < 0.95,
    # so NO config may claim the constraint, even at 100% precision.
    rows = [
        _row(f"s{i}", payload={"a": 1}, probs={"a": 0.99}, accept=0.99)
        for i in range(10)
    ]
    gt = _gt(*[(f"s{i}", {"a": 1}) for i in range(10)])
    res = _cal(target=0.95).sweep_thresholds(rows, gt)
    assert res["constraint_met"] is False
    assert res["k_correct"] == 10.0
    assert res["n_hits"] == 10.0
    assert res["precision"] == 1.0
    assert res["hit_rate"] == 1.0
    # highest-LBD config among ties is the first (most permissive) grid point.
    assert res["tau_p"] == 0.50
    assert res["tau_a"] == 0.50
    assert res["lbd"] == pytest.approx(0.05 ** 0.1, abs=1e-4)


def test_sweep_maximizes_hit_rate_under_constraint() -> None:
    # 8 rows strong (0.99), 2 rows weaker (0.60), all correct, target 0.50.
    # A tau_p <= 0.60 gate passes all 10 (LBD(10,10)=0.741 >= 0.5, met);
    # a stricter gate passes only 8. Hit-rate maximization must pick the
    # permissive config, tie-broken to the most conservative thresholds that
    # still pass everything.
    rows = [
        _row(f"s{i}", payload={"a": 1}, probs={"a": 0.99}, accept=0.99)
        for i in range(8)
    ] + [
        _row(f"s{i}", payload={"a": 1}, probs={"a": 0.60}, accept=0.60)
        for i in range(8, 10)
    ]
    gt = _gt(*[(f"s{i}", {"a": 1}) for i in range(10)])
    res = _cal(target=0.50).sweep_thresholds(rows, gt)
    assert res["constraint_met"] is True
    assert res["n_hits"] == 10.0
    assert res["k_correct"] == 10.0
    assert res["tau_p"] == 0.60   # max threshold keeping all 10
    assert res["tau_a"] == 0.60
    assert res["lbd"] >= 0.50


def test_sweep_conservative_tie_break_at_equal_hit_rate() -> None:
    # All rows identical and strong: every grid config passes all 6 with the
    # same LBD => the sweep picks the most conservative thresholds.
    rows = [
        _row(f"s{i}", payload={"a": 1}, probs={"a": 0.99}, accept=0.99)
        for i in range(6)
    ]
    gt = _gt(*[(f"s{i}", {"a": 1}) for i in range(6)])
    res = _cal(target=0.50).sweep_thresholds(rows, gt)
    assert res["constraint_met"] is True
    assert res["n_hits"] == 6.0
    assert res["k_correct"] == 6.0
    assert res["tau_p"] == 0.99  # max on grid
    assert res["tau_a"] == 0.98  # max on grid


def test_sweep_unmet_constraint_picks_highest_lbd() -> None:
    # Only the correct row survives any gate (wrong rows at probs/accept 0.30):
    # every config claims n_hits=1, k=1 with LBD(1,1)=0.05. Target 0.95 is
    # never met; the winner is the first grid point (most permissive).
    rows = [
        _row("s0", payload={"a": 1}, probs={"a": 0.99}, accept=0.99),
        _row("s1", payload={"a": 77}, probs={"a": 0.30}, accept=0.30),
        _row("s2", payload={"a": 77}, probs={"a": 0.30}, accept=0.30),
    ]
    gt = _gt(("s0", {"a": 1}), ("s1", {"a": 2}), ("s2", {"a": 3}))
    res = _cal(target=0.95).sweep_thresholds(rows, gt)
    assert res["constraint_met"] is False
    assert res["n_hits"] == 1.0
    assert res["k_correct"] == 1.0
    assert res["precision"] == 1.0
    assert res["lbd"] == pytest.approx(0.05, abs=1e-6)  # LBD(1,1) = alpha
    assert res["tau_p"] == 0.50 and res["tau_a"] == 0.50


def test_sweep_nhits_zero_fallback() -> None:
    # No row can ever pass (schema invalid everywhere) -> fallback dict.
    rows = [_row("s0", payload=None, schema_valid=False)]
    gt = _gt(("s0", {"a": 1}))
    res = _cal().sweep_thresholds(rows, gt)
    assert res["constraint_met"] is False
    assert res["n_hits"] == 0.0
    assert res["n_total"] == 1.0
    assert res["hit_rate"] == 0.0 and res["precision"] == 0.0


def test_sweep_empty_inputs() -> None:
    res = _cal().sweep_thresholds([], [])
    assert res["constraint_met"] is False
    assert res["n_total"] == 0.0
    assert res["hit_rate"] == 0.0


def test_sweep_conjunction_uses_joint_margin() -> None:
    rows = [
        _row(f"s{i}", payload={"a": 1}, probs={"a": 0.99}, accept=0.99, margin=0.5)
        for i in range(5)
    ]
    gt = _gt(*[(f"s{i}", {"a": 1}) for i in range(5)])
    res = _cal(target=0.50, mode="conjunction").sweep_thresholds(rows, gt)
    assert res["mode"] == "conjunction"
    # margin 0.5 passes tau_delta in {0.0, 0.25, 0.5}; among constraint-met
    # ties the sweep prefers the MOST conservative (largest tp+ta+td) point.
    assert res["tau_delta"] == 0.5
    assert res["n_hits"] == 5.0


# --------------------------------------------------------------------------- #
# Per-risk-tier thresholds
# --------------------------------------------------------------------------- #


def test_thresholds_per_tier_applies_per_tier_targets() -> None:
    # 60 all-correct rows per tier: LBD(60,60;0.95)=0.0513**0 ??? = 0.951 >= 0.95
    # (read_only met), but < 0.97 (idempotent) and < 0.99 (side_effecting):
    # the same data earns different gating guarantees per tier. This is the
    # exact honesty contract the plan mandates.
    rows = [
        _row(f"{tier}-{i}", payload={"a": 1}, probs={"a": 0.99}, accept=0.99,
             tier=tier)
        for tier in TIER_TARGETS for i in range(60)
    ]
    gt = _gt(*[(r["sample_id"], {"a": 1}) for r in rows])
    out = _cal().thresholds_per_tier(rows, gt)
    assert set(out.keys()) == set(TIER_TARGETS.keys())
    # read_only target 0.95 is met at n=60 (LBD ~= 0.9513): pick most
    # conservative thresholds that pass all 60.
    assert out["read_only"]["constraint_met"] is True
    assert out["read_only"]["n_hits"] == 60.0
    assert out["read_only"]["tau_p"] == 0.99 and out["read_only"]["tau_a"] == 0.98
    # idempotent (0.97) and side_effecting (0.99) are NOT met at n=60.
    assert out["idempotent"]["constraint_met"] is False
    assert out["idempotent"]["n_hits"] == 60.0
    assert out["side_effecting"]["constraint_met"] is False
    assert out["side_effecting"]["n_hits"] == 60.0


def test_thresholds_per_tier_empty_tier_fallback() -> None:
    cal = _cal()
    rows = [_row("s0", payload={"a": 1}, probs={"a": 0.99}, accept=0.99, tier="read_only")]
    gt = _gt(("s0", {"a": 1}))
    out = cal.thresholds_per_tier(rows, gt)
    # only read_only has rows; the other two tiers get conservative fallbacks.
    assert out["idempotent"]["constraint_met"] is False
    assert out["idempotent"]["n_total"] == 0.0
    assert out["side_effecting"]["tau_p"] == 0.95
    assert out["side_effecting"]["tau_a"] == 0.90
    assert out["side_effecting"]["tau_delta"] == 0.0  # single_tau mode


# --------------------------------------------------------------------------- #
# Risk-coverage + ECE
# --------------------------------------------------------------------------- #


def test_risk_coverage_curve_monotone_and_risk_definition() -> None:
    cal = _cal()
    rows = [
        _row("s0", payload={"a": 1}, accept=0.9),
        _row("s1", payload={"a": 2}, accept=0.8),
        _row("s2", payload={"a": 3}, accept=0.7),
    ]
    gt = _gt(("s0", {"a": 1}), ("s1", {"a": 2}), ("s2", {"a": 9}))
    curve = cal.risk_coverage_curve(rows, gt)
    assert len(curve) == 3
    coverages = [c["coverage"] for c in curve]
    assert coverages == sorted(coverages)
    for c in curve:
        assert c["precision"] == pytest.approx(1.0 - c["risk"], abs=1e-12)
    assert curve[0]["risk"] == 0.0  # highest-conf row is correct
    assert curve[-1]["n_errors"] == 1


def test_ece_synthetic_confidences() -> None:
    cal = _cal()
    rows = [
        _row("s0", payload={"a": 1}, accept=0.9),
        _row("s1", payload={"a": 2}, accept=0.9),
        _row("s2", payload={"a": 3}, accept=0.1),
    ]
    gt = _gt(("s0", {"a": 1}), ("s1", {"a": 2}), ("s2", {"a": 3}))
    ece = cal.expected_calibration_error(rows, gt, n_bins=2)
    # bin [0, .5): conf 0.1/acc 1.0 -> |0.1-1.0|*1/3; bin [.5, 1): 0.9/1.0 -> |0.9-1.0|*2/3
    assert ece == pytest.approx(0.9 * (1 / 3) + 0.1 * (2 / 3), abs=1e-9)


def test_ece_empty() -> None:
    assert _cal().expected_calibration_error([], []) == 0.0


# --------------------------------------------------------------------------- #
# calibrate() + report round-trip into resolver gate cfg
# --------------------------------------------------------------------------- #


def test_calibrate_report_round_trips_into_resolver_gate() -> None:
    from spancall.infer.resolver import _default_gate_cfg

    rows = [
        _row(f"s{i}", payload={"a": 1}, probs={"a": 0.99}, accept=0.99)
        for i in range(5)
    ]
    gt = _gt(*[(f"s{i}", {"a": 1}) for i in range(5)])
    cal = _cal(target=0.95)
    report = cal.calibrate(rows, gt, fitted_temperature=0.87)
    assert report["mode"] == "single_tau"
    assert set(("tau_p", "tau_a", "tau_delta", "temperature", "tau_tool")) <= set(report)
    assert report["n_calib"] == 5
    assert report["thresholds"]["per_tier"]["read_only"]["n_hits"] == 5.0
    assert isinstance(report["risk_coverage"], list) and report["risk_coverage"]

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "calibration_report.json")
        cal.write_report(report, path)
        with open(path) as fh:
            reloaded = json.load(fh)
        cfg = _default_gate_cfg(path)
    # resolver reads exactly the keys the report carries
    assert cfg["mode"] == report["mode"]
    assert cfg["tau_p"] == report["tau_p"]
    assert cfg["tau_a"] == report["tau_a"]
    assert cfg["tau_delta"] == report["tau_delta"]
    assert cfg["temperature"] == report["temperature"]
    assert cfg["tau_tool"] == report["tau_tool"]
    assert reloaded["n_calib"] == 5


def test_default_gate_cfg_without_report() -> None:
    from spancall.infer.resolver import _default_gate_cfg

    cfg = _default_gate_cfg(None)
    assert cfg == {"mode": "single_tau", "tau_p": 0.90, "tau_a": 0.85, "tau_delta": 2.0}


def test_read_calib_rows_roundtrip() -> None:
    import tempfile
    from pathlib import Path

    rows = [_row("s0", payload={"a": 1}), _row("s1", payload={"a": 2})]
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "rows.jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        got = _cal().read_calib_rows(path)
    assert len(got) == 2
    assert got[0]["sample_id"] == "s0"
    assert got[1]["payload"] == {"a": 2}