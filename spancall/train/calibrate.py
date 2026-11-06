"""Confidence calibration and threshold sweep engine.

Ref: docs/mechanism.md §7.4, docs/full_implementation_plan.md §3.5.

- Temperature scaling (L-BFGS) on held-out logits (``fit_temperature``)
- Clopper-Pearson exact binomial lower bound (scipy, with a pure-Python
  regularized-incomplete-beta fallback when scipy is absent)
- Threshold sweep (tau_p, tau_Delta, tau_a) over a held-out *paraphrase-
  disjoint* calibration set (``sweep_thresholds``)
- Enforces per-risk-tier precision via the Clopper-Pearson LBD:
  read_only 0.95, idempotent 0.97, side_effecting 0.99
- Emits ``calibration_report.json`` (temperatures, thresholds per tier,
  full risk-coverage curve, ECE).
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from spancall.train.config import CalibrationConfig

try:  # torch is required for L-BFGS temperature scaling only
    import torch
    from torch import nn
    from torch.nn import functional as F  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - torch-less test environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - pure-Python fallback kept live
    _scipy_stats = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Constant tables
# --------------------------------------------------------------------------- #

TIER_TARGETS: Dict[str, float] = {
    "read_only": 0.95,
    "idempotent": 0.97,
    "side_effecting": 0.99,
}

_TAU_P_GRID: List[float] = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50..0.99
_TAU_A_GRID: List[float] = [round(0.50 + 0.02 * i, 2) for i in range(25)]  # 0.50..0.98
_TAU_D_GRID: List[float] = [round(0.0 + 0.25 * i, 2) for i in range(21)]  # 0.0..5.0


# --------------------------------------------------------------------------- #
# Pure-Python regularized incomplete beta (Clopper-Pearson fallback)
# --------------------------------------------------------------------------- #


def _log_gamma(x: float) -> float:
    """Lanczos log-gamma (g = 7, n = 9) — used by the beta fallback."""
    if x <= 0.0:
        raise ValueError(f"log_gamma requires x > 0, got {x!r}")
    if x < 0.5:
        return math.log(math.pi) - math.log(math.sin(math.pi * x)) - _log_gamma(1.0 - x)
    c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    x -= 1.0
    t = x + 7.5
    a = c[0]
    for i in range(1, 9):
        a += c[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _incbeta(x: float, a: float, b: float, iters: int = 600, eps: float = 1e-11) -> float:
    """Regularized incomplete beta I_x(a, b) via the power series, mirrored
    onto the complement when x is large (fast convergence for both tails)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    use_comp = x > (a + 1.0) / (a + b + 2.0)
    if use_comp:
        return 1.0 - _incbeta(1.0 - x, b, a, iters=iters, eps=eps)
    lbeta = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta)
    term = 1.0  # n = 0 term
    total = term
    for n in range(1, iters):
        term *= ((a + b + n - 1.0) / (a + n)) * x
        total += term
        if abs(term) < eps:
            break
    return front * total / a


def _beta_ppf(p: float, a: float, b: float, tol: float = 1e-10) -> float:
    """Inverse regularized incomplete beta by bisection (fallback only)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _incbeta(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Payload equality + gate decisions (shared with eval in Phase 3e)
# --------------------------------------------------------------------------- #


def payloads_equal(expected: Optional[Dict[str, Any]], actual: Optional[Dict[str, Any]]) -> bool:
    """Exact-payload-match equality used to label emitted calls as correct.

    Both payloads must be non-``None`` and equal as dicts (deep equality).
    """
    if expected is None or actual is None:
        return False
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return bool(expected == actual)
    if set(expected.keys()) != set(actual.keys()):
        return False
    return all(_deep_eq(expected[k], actual[k]) for k in expected)


def _deep_eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (date, datetime)) and isinstance(b, str):
        s_a, s_b = a.isoformat(), b.strip()
        return s_a == s_b or str(a) == s_b or s_b.startswith(s_a) or s_a.startswith(s_b)
    if isinstance(b, (date, datetime)) and isinstance(a, str):
        s_b, s_a = b.isoformat(), a.strip()
        return s_b == s_a or str(b) == s_a or s_a.startswith(s_b) or s_b.startswith(s_a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a.keys()) == set(b.keys()) and all(_deep_eq(a[k], b[k]) for k in a)
    return bool(a == b)


def _gate_pass(
    row: Dict[str, Any],
    tau_p: float,
    tau_a: float,
    tau_delta: float,
    mode: str = "single_tau",
) -> bool:
    """Reproduce the resolver's gate decision from raw calibration signals.

    Mirrors ``SingleTauGate.evaluate`` / ``ConjunctionGate`` semantics so the
    sweep searches exactly the space the deployed resolver will use.
    """
    probs = row.get("param_probabilities") or {}
    if not probs:
        return False
    if min(probs.values()) < tau_p:
        return False
    if not row.get("schema_valid", True):
        return False
    if (row.get("accept_probability") or 0.0) < tau_a:
        return False
    if mode == "conjunction" and (row.get("joint_margin") or 0.0) < tau_delta:
        return False
    return True


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class BaseCalibrator(ABC):
    """Abstract interface for calibrating model logits and gating thresholds."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config

    @abstractmethod
    def fit_temperature(self, val_logits: Any, val_labels: Any) -> float:
        """Learn optimal temperature scaling parameter via NLL minimization."""
        raise NotImplementedError

    @abstractmethod
    def sweep_thresholds(
        self,
        calib_predictions: List[Dict[str, Any]],
        calib_ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Find optimal gating thresholds (tau_p, tau_delta, tau_a) guaranteeing target precision."""
        raise NotImplementedError

    @abstractmethod
    def clopper_pearson_lower_bound(
        self,
        k_successes: int,
        n_trials: int,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Clopper-Pearson exact binomial lower confidence bound."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# SpanCallCalibrator
# --------------------------------------------------------------------------- #


class SpanCallCalibrator(BaseCalibrator):
    """L-BFGS temperature scaling + Clopper-Pearson-constrained threshold sweep.

    Prediction-row contract (each element of ``calib_predictions``)::

        {
          "sample_id": str,
          "tool_name": str,
          "payload": dict | None,            # emitted when the gate passed
          "param_probabilities": {param: p}, # per-param confidences
          "joint_margin": float,             # Delta_joint (Q3)
          "accept_probability": float,       # CallAcceptHead sigmoid
          "schema_valid": bool,              # Q6 result
          "risk_tier": str,                  # read_only | idempotent | side_effecting
        }

    ``calib_ground_truths`` is a parallel list (or ``{sample_id: arguments}``
    mapping) with keys ``sample_id``/``arguments`` (the reference payload).
    Rows with no reference emission are labeled *wrong* when the gate emits.
    """

    # -- helpers ------------------------------------------------------------ #

    def clopper_pearson_lower_bound(
        self,
        k_successes: int,
        n_trials: int,
        confidence: float = 0.95,
    ) -> float:
        """Exact binomial lower confidence bound for the hit rate.

        .. math:: LBD(k, n; α) = Beta^{-1}(α; k, n - k + 1)   (0 when k = 0)

        Uses ``scipy.stats.beta.ppf`` when available, else a pure-Python
        regularized-incomplete-beta bisection.
        """
        if n_trials <= 0:
            return 0.0
        k = int(k_successes)
        if k <= 0:
            return 0.0
        alpha = 1.0 - confidence
        if _scipy_stats is not None:
            return float(_scipy_stats.beta.ppf(alpha, k, n_trials - k + 1))
        return float(_beta_ppf(alpha, float(k), float(n_trials - k + 1)))

    # -- temperature scaling (L-BFGS) --------------------------------------- #

    def fit_temperature(self, val_logits: Any, val_labels: Any) -> float:
        """Learn the scalar temperature T > 0 minimizing NLL (L-BFGS over log T).

        ``val_logits``/``val_labels`` may be a single ``(logits, labels)`` pair
        or a list of pairs so H3 pointer logits and H8/CallAccept logits can be
        fit jointly with one shared temperature (plan §3.5.1). Labels may be
        hard class indices ``[N]`` or one-hot soft targets ``[N, C]``.

        Returns ``1.0`` (identity) when torch is unavailable.
        """
        if torch is None or F is None:  # pragma: no cover - torch-less env
            return 1.0
        pairs = _normalize_pairs(val_logits, val_labels)
        if not pairs:
            return 1.0
        log_t = nn.Parameter(torch.tensor(0.0))
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)  # type: ignore[attr-defined]

        def closure() -> torch.Tensor:  # type: ignore[no-any-return]
            opt.zero_grad()
            t = torch.clamp(log_t.exp(), min=1e-3)
            losses: List[torch.Tensor] = []
            for lo, la in pairs:
                if la.dtype.is_floating_point and la.shape[-1] == lo.shape[-1]:
                    lp = F.log_softmax(lo / t, dim=-1)
                    losses.append(-(la * lp).sum(dim=-1).mean())
                else:
                    losses.append(F.cross_entropy(lo / t, la))
            loss = torch.stack(losses).mean()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            return float(torch.clamp(log_t.exp(), min=1e-3).item())

    # -- single-tier threshold sweep --------------------------------------- #

    def sweep_thresholds(
        self,
        calib_predictions: List[Dict[str, Any]],
        calib_ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Grid-search (tau_p, tau_a[, tau_delta]) maximizing hit rate subject
        to ``ClopperPearsonLBD(k_correct, n_hits; confidence) >= target``.

        Returns the winning configuration plus hit-rate/LBD diagnostics.
        """
        mode = str(getattr(self.config, "mode", "single_tau"))
        target = float(getattr(self.config, "target_precision", 0.95))
        confidence = float(getattr(self.config, "confidence_level", 0.95))
        preds, gt = _align(calib_predictions, calib_ground_truths)

        jm_grid = _TAU_D_GRID if mode == "conjunction" else [0.0]
        best: Optional[Dict[str, float]] = None
        for tp in _TAU_P_GRID:
            for ta in _TAU_A_GRID:
                for td in jm_grid:
                    n_hits, k_correct = _evaluate_gate(preds, gt, tp, ta, td, mode)
                    if n_hits == 0:
                        lbd = 0.0
                    else:
                        lbd = self.clopper_pearson_lower_bound(k_correct, n_hits, confidence)
                    if best is None or _better(best, tp, ta, td, n_hits, k_correct, lbd, target):
                        best = {
                            "tau_p": float(tp),
                            "tau_a": float(ta),
                            "tau_delta": float(td),
                            "mode": mode,
                            "n_hits": float(n_hits),
                            "k_correct": float(k_correct),
                            "n_total": float(len(preds)),
                            "hit_rate": (n_hits / len(preds)) if preds else 0.0,
                            "precision": (k_correct / n_hits) if n_hits else 0.0,
                            "lbd": float(lbd),
                            "constraint_met": bool(n_hits > 0 and lbd >= target),
                        }
        if best is None:
            best = {
                "tau_p": 0.95, "tau_a": 0.90, "tau_delta": 2.0, "mode": mode,
                "n_hits": 0.0, "k_correct": 0.0, "n_total": float(len(preds)),
                "hit_rate": 0.0, "precision": 0.0, "lbd": 0.0, "constraint_met": False,
            }
        return best

    # -- per-risk-tier thresholds ------------------------------------------- #

    def thresholds_per_tier(
        self,
        calib_predictions: List[Dict[str, Any]],
        calib_ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute thresholds for each risk tier at its own target precision."""
        preds, gt = _align(calib_predictions, calib_ground_truths)
        out: Dict[str, Dict[str, float]] = {}
        for tier, target in TIER_TARGETS.items():
            cfg = _tier_config(self.config, target)
            tier_preds = [r for r in preds if (r.get("risk_tier") or "read_only") == tier]
            tier_gt = [
                {"sample_id": r["sample_id"], "arguments": gt.get(r["sample_id"])}
                for r in tier_preds
            ]
            if not tier_preds:
                out[tier] = _fallback_thresholds(self.config, target, 0.0)
                continue
            sub = SpanCallCalibrator(cfg)
            res = sub.sweep_thresholds(tier_preds, tier_gt)
            out[tier] = res
        return out

    # -- risk-coverage curve + ECE ------------------------------------------ #

    def risk_coverage_curve(
        self,
        calib_predictions: List[Dict[str, Any]],
        calib_ground_truths: List[Dict[str, Any]],
        key: str = "accept_probability",
    ) -> List[Dict[str, Any]]:
        """Cumulative risk (1 - precision) vs coverage sorting by confidence.

        Walk the calibration set in decreasing confidence; every step emits one
        more row; ``risk`` is the fraction of emitted rows that are incorrect.
        """
        preds, gt = _align(calib_predictions, calib_ground_truths)
        scored: List[Tuple[float, bool]] = []
        for r in preds:
            conf = float(r.get(key) or 0.0)
            correct = payloads_equal(gt.get(r["sample_id"]), r.get("payload"))
            scored.append((conf, correct))
        scored.sort(key=lambda t: t[0], reverse=True)
        curve: List[Dict[str, Any]] = []
        n = len(scored)
        emitted = 0
        errors = 0
        for conf, correct in scored:
            emitted += 1
            if not correct:
                errors += 1
            curve.append({
                "confidence": float(conf),
                "coverage": emitted / n if n else 1.0,
                "risk": errors / emitted if emitted else 0.0,
                "precision": (emitted - errors) / emitted if emitted else 0.0,
                "n_emitted": emitted,
                "n_errors": errors,
            })
        return curve

    def expected_calibration_error(
        self,
        calib_predictions: List[Dict[str, Any]],
        calib_ground_truths: List[Dict[str, Any]],
        n_bins: int = 10,
        key: str = "accept_probability",
    ) -> float:
        """Standard binned ECE: mean |bin_confidence - bin_accuracy| weighted by size."""
        preds, gt = _align(calib_predictions, calib_ground_truths)
        scored: List[Tuple[float, bool]] = []
        for r in preds:
            conf = float(r.get(key) or 0.0)
            correct = payloads_equal(gt.get(r["sample_id"]), r.get("payload"))
            scored.append((conf, correct))
        if not scored:
            return 0.0
        n = len(scored)
        n_bins = max(1, int(n_bins))
        total = 0.0
        for b in range(n_bins):
            lo = b / n_bins
            hi = (b + 1) / n_bins
            members = [s for s in scored if lo <= s[0] < hi]
            if not members:
                continue
            bin_conf = sum(s[0] for s in members) / len(members)
            bin_acc = sum(1.0 for s in members if s[1]) / len(members)
            total += (len(members) / n) * abs(bin_conf - bin_acc)
        return float(total)

    # -- orchestration + report --------------------------------------------- #

    def calibrate(
        self,
        calib_predictions: List[Dict[str, Any]],
        calib_ground_truths: List[Dict[str, Any]],
        fitted_temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Full pipeline → the ``calibration_report.json`` payload.

        Top-level keys match what ``resolver._default_gate_cfg`` reads so the
        report can be passed directly as ``calib_thresholds_path``.
        """
        mode = str(getattr(self.config, "mode", "single_tau"))
        best = self.sweep_thresholds(calib_predictions, calib_ground_truths)
        per_tier = self.thresholds_per_tier(calib_predictions, calib_ground_truths)
        curve = self.risk_coverage_curve(calib_predictions, calib_ground_truths)
        ece = self.expected_calibration_error(calib_predictions, calib_ground_truths)
        report: Dict[str, Any] = {
            "mode": mode,
            "tau_p": float(best["tau_p"]),
            "tau_a": float(best["tau_a"]),
            "tau_delta": float(best["tau_delta"]),
            "temperature": float(fitted_temperature),
            "tau_tool": 0.0,
            "target_precision": float(getattr(self.config, "target_precision", 0.95)),
            "confidence_level": float(getattr(self.config, "confidence_level", 0.95)),
            "n_calib": len(calib_predictions),
            "thresholds": {
                "global": best,
                "per_tier": per_tier,
            },
            "risk_coverage": curve,
            "ece": float(ece),
        }
        return report

    def write_report(self, report: Dict[str, Any], path: Union[str, Path]) -> None:
        """Persist the calibration report as pretty JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
            fh.write("\n")

    def read_calib_rows(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load calibration rows from a JSONL file (each line one dict)."""
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _align(
    preds: Sequence[Dict[str, Any]],
    gt: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[Dict[str, Any]]]]:
    """Return (preds, {sample_id: arguments-or-None}) aligned by sample_id.

    ``gt`` entries are ``{"sample_id": ..., "arguments": ...}`` dicts; the whole
    mapping may also be passed as ``{sample_id: arguments}`` directly.
    """
    gt_map: Dict[str, Optional[Dict[str, Any]]] = {}
    if isinstance(gt, dict):
        for sid, arguments in gt.items():
            gt_map[str(sid)] = arguments if isinstance(arguments, dict) else None
        return list(preds), gt_map
    for g in gt:
        if not isinstance(g, dict):
            continue
        sid = g.get("sample_id")
        if sid is not None:
            gt_map[str(sid)] = g.get("arguments")
    return list(preds), gt_map


def _normalize_pairs(val_logits: Any, val_labels: Any) -> List[Tuple[Any, Any]]:
    """Normalize fit_temperature inputs into [(logits_tensor, labels_tensor)]."""
    if val_logits is None or val_labels is None:
        return []

    def _t(x: Any) -> Any:
        if torch is not None and torch.is_tensor(x):
            return x.detach().float()
        return torch.tensor(x, dtype=torch.float32)

    def _l(x: Any) -> Any:
        if torch is not None and torch.is_tensor(x):
            return x.detach()
        if isinstance(x, list) and x and isinstance(x[0], list):
            return torch.tensor(x, dtype=torch.float32)
        return torch.tensor(x)

    pairs: List[Tuple[Any, Any]] = []
    if isinstance(val_logits, (list, tuple)) and val_logits and isinstance(val_logits[0], (list, tuple)):
        # list of (logits, labels) pairs
        for lo, la in val_logits:
            pairs.append((_t(lo), _l(la)))
    else:
        pairs.append((_t(val_logits), _l(val_labels)))
    return [(lo, la) for lo, la in pairs if lo.numel() and la.numel() > 0]


def _evaluate_gate(
    preds: List[Dict[str, Any]],
    gt_map: Dict[str, Optional[Dict[str, Any]]],
    tau_p: float,
    tau_a: float,
    tau_delta: float,
    mode: str,
) -> Tuple[int, int]:
    """Count (n_hits, k_correct) for one (tau_p, tau_a, tau_delta) point."""
    n_hits = 0
    k_correct = 0
    for r in preds:
        if not _gate_pass(r, tau_p, tau_a, tau_delta, mode):
            continue
        n_hits += 1
        if payloads_equal(gt_map.get(r["sample_id"]), r.get("payload")):
            k_correct += 1
    return n_hits, k_correct


def _better(
    best: Dict[str, float],
    tp: float,
    ta: float,
    td: float,
    n_hits: int,
    k_correct: int,
    lbd: float,
    target: float,
) -> bool:
    """Selection rule: prefer constraint-satisfying configs, then higher hit
    rate, then (among ties) the more conservative thresholds."""
    b_met = best.get("constraint_met")
    met = bool(n_hits > 0 and lbd >= target)
    if met != b_met:
        return met
    if met:
        if n_hits > best.get("n_hits", 0):
            return True
        if n_hits == best.get("n_hits", 0):
            return (tp + ta + (td if td else 0.0)) > (
                best.get("tau_p", 0.0) + best.get("tau_a", 0.0) + best.get("tau_delta", 0.0)
            )
        return False
    # neither satisfies: pick the highest LBD (most trustworthy), tie-break on hit rate
    if lbd > best.get("lbd", -1.0):
        return True
    if lbd == best.get("lbd", -1.0):
        return n_hits > best.get("n_hits", 0)
    return False


def _tier_config(base: Any, target: float) -> CalibrationConfig:
    """Config copy with the tier's target precision and caller's mode/confidence."""
    mode = getattr(base, "mode", "single_tau")
    return CalibrationConfig(
        target_precision=float(target),
        confidence_level=float(getattr(base, "confidence_level", 0.95)),
        temperature_scaling=bool(getattr(base, "temperature_scaling", True)),
        mode=str(mode),
    )


def _fallback_thresholds(base: Any, target: float, n: float) -> Dict[str, Any]:
    """Defaults emitted for tiers with no calibration rows (conservative)."""
    mode = getattr(base, "mode", "single_tau")
    return {
        "tau_p": 0.95,
        "tau_a": 0.90,
        "tau_delta": 2.0 if mode == "conjunction" else 0.0,
        "mode": mode,
        "n_hits": 0.0,
        "k_correct": 0.0,
        "n_total": n,
        "hit_rate": 0.0,
        "precision": 0.0,
        "lbd": 0.0,
        "constraint_met": False,
    }
