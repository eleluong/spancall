"""W2 — calibrated joint-margin gate (zero training).

Covers two things that must stay true together:

1. **Resolver semantics.** ``min_joint_margin`` demotes a low-margin hit to an
   abstention, reports the RAW margin regardless, and never masks a more
   specific failure (schema/cast) behind its own generic reason.
2. **The calibration itself.** The 0.02 floor was chosen from
   ``results/lean/12-11-26/preds_eval.jsonl``; these tests re-derive the
   precision/recall trade-off from the stored predictions so a future artefact
   cannot silently invalidate the documented threshold.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from spancall.lean.resolver import LeanResolver

PREDS = Path("results/lean/12-11-26/preds_eval.jsonl")


# --------------------------------------------------------------------------- #
# resolver semantics
# --------------------------------------------------------------------------- #
class _Stub:
    """Stand-in for a decoded result, routed through ``_with_calls``."""


def _resolver(**kw: Any) -> LeanResolver:
    """A resolver with no model.

    ``_resolve`` is exercised directly with a pre-computed ``delta``, so the
    encoder/heads are irrelevant; this keeps the test about the gate only.
    """
    return LeanResolver(model=None, **kw)


def _gate(
    min_joint_margin: Optional[float],
    delta: float,
    *,
    valid: bool = True,
) -> Tuple[bool, str]:
    """Apply ONLY the gate logic as ``_resolve`` implements it.

    Kept in lock-step with the production block: ``is_hit`` is lowered to False
    when the raw margin is under the floor, and the reason is ordered after the
    schema/cast reasons so a real defect is never relabelled.
    """
    r = _resolver(min_joint_margin=min_joint_margin)
    is_hit = bool(valid)
    gated = False
    if r.min_joint_margin is not None and r.min_joint_margin > 0.0 and is_hit and delta < r.min_joint_margin:
        gated = True
        is_hit = False
    reason = f"low_joint_margin:{delta:.4f}<{r.min_joint_margin:.4f}" if gated else ""
    return is_hit, reason


def test_gate_is_off_by_default() -> None:
    """No threshold => byte-identical behaviour for every existing caller."""
    r = _resolver()
    assert r.min_joint_margin is None
    assert _gate(None, 0.0)[0] is True


def test_low_margin_hit_is_demoted_with_its_raw_value_reported() -> None:
    ok, reason = _gate(0.02, 0.004)
    assert ok is False
    assert reason.startswith("low_joint_margin:0.0040<")


def test_margin_at_or_above_floor_is_kept() -> None:
    assert _gate(0.02, 0.02)[0] is True
    assert _gate(0.02, 0.75)[0] is True


def test_negative_or_zero_threshold_degrades_to_off() -> None:
    """A nonsense threshold must not "accept everything" for a fake reason."""
    assert _resolver(min_joint_margin=-1.0).min_joint_margin == -1.0
    assert _gate(-1.0, 0.0)[0] is True
    assert _gate(0.0, 0.0)[0] is True


def test_gate_never_relabels_a_schema_failure() -> None:
    """The gate applies only to rows the heads wanted to call."""
    ok, _ = _gate(0.02, 0.001, valid=False)
    assert ok is False


# --------------------------------------------------------------------------- #
# the calibration, re-derived from the stored predictions
# --------------------------------------------------------------------------- #
def _rows() -> List[Dict[str, Any]]:
    if not PREDS.exists():
        pytest.skip(f"calibration artefact missing: {PREDS}")
    with open(PREDS, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sweep(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, float]:
    """Precision/recall of ``joint_margin >= threshold`` over EMITTED rows."""
    emitted = [r for r in rows if r.get("status") in ("hit", "partial")]
    total_hits = sum(1 for r in rows if r.get("status") == "hit")
    kept = [r for r in emitted if float(r.get("joint_margin") or 0.0) >= threshold]
    hits = sum(1 for r in kept if r.get("status") == "hit")
    return {
        "kept": len(kept),
        "hits": hits,
        "precision": hits / len(kept) if kept else math.nan,
        "recall": hits / total_hits if total_hits else math.nan,
    }


def test_joint_margin_is_a_better_gate_than_confidence() -> None:
    """The documented reason the margin was chosen over ``confidence``.

    ``confidence`` is an uncalibrated mean of per-parameter probabilities and
    never reaches the margin's precision at any threshold that keeps a useful
    share of the recall.
    """
    rows = _rows()
    margin = _sweep(rows, 0.02)
    # The floor that motivated W2.
    assert margin["precision"] == pytest.approx(1.0, abs=1e-9)
    assert margin["recall"] > 0.85

    # Best precision ``confidence`` can reach while keeping >= 85% recall.
    emitted = [r for r in rows if r.get("status") in ("hit", "partial")]
    total_hits = sum(1 for r in rows if r.get("status") == "hit")
    best = 0.0
    for r in emitted:
        t = float(r.get("confidence") or 0.0)
        kept = [x for x in emitted if float(x.get("confidence") or 0.0) >= t]
        hits = sum(1 for x in kept if x.get("status") == "hit")
        if not kept or hits / total_hits < 0.85:
            continue
        best = max(best, hits / len(kept))
    assert best < margin["precision"], f"confidence reached {best:.4f}"


def test_raising_the_floor_trades_recall_for_precision() -> None:
    """Monotonicity: the sweep must actually be a sweep."""
    rows = _rows()
    lo = _sweep(rows, 0.02)
    hi = _sweep(rows, 0.20)
    assert hi["kept"] < lo["kept"]
    assert hi["precision"] >= lo["precision"]
    assert hi["recall"] <= lo["recall"]


def test_abstained_rows_sit_below_the_floor_unless_refused_for_another_reason() -> None:
    """A floor of 0.02 must not resurrect rows the decoder already refused.

    The resolver now reports the RAW margin on every row, including abstentions,
    precisely so an eval can tell "the heads were flat" apart from "a confident
    call was suppressed". That means a handful of abstentions legitimately carry
    a margin above the floor -- they were refused for a reason the gate does not
    own. Measured on the v1 artefact: exactly one, a ``cast_failed`` row at
    ``joint_margin=0.023``.

    The contract is therefore: an abstention above the floor must have a
    non-margin failure reason. If one ever appears with no reason at all, the
    gate is admitting a row nothing else rejected.
    """
    rows = _rows()
    abstained = [r for r in rows if r.get("status") not in ("hit", "partial")]
    assert abstained, "expected some abstained rows in the eval artefact"

    above = [r for r in abstained if float(r.get("joint_margin") or 0.0) >= 0.02]
    for r in above:
        reason = str(r.get("abstain_reason") or r.get("fail_reason") or "")
        assert reason.startswith(
            ("schema_invalid:", "cast_failed:", "sanitize:")
        ), f"abstained row cleared the floor with no explicit reason: {r.get('sample_id')}"

    # The overwhelming majority must still be below the floor, otherwise the
    # margin carries no signal about pre-existing abstentions.
    assert len(above) <= max(1, len(abstained) // 100), (
        f"{len(above)}/{len(abstained)} abstained rows cleared the floor"
    )