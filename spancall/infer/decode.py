"""Joint assignment decoder enforcing span exclusivity and computing margins.

Ref: docs/mechanism.md §6, docs/pipeline_io.md Q3, full_implementation_plan §4.1.

``ExclusivityJointDecoder`` implements, in order:

1. **R4 equal-value aggregation** — candidates are grouped by canonical value:
   ``p_k(v) = Σ_{c ∈ A_k : canon(c)=v} p_k(c)`` and each value is represented
   by its highest-scoring span (so equal-valued textual duplicates share mass,
   exactly as in the multi-positive softmax loss).
2. **Top-K distinct values** per parameter (default ``K = 3``), tied
   deterministically (lexicographic by candidate tuple).
3. **Conflict-pruned Cartesian search (DFS)** — a partial assignment is invalid
   as soon as two parameters pick overlapping utterance spans whose entity
   classes match, or two free-utterance strings overlap (unless allowed).
4. **Best + second-best ranking**: ``S(a) = Σ_k log p_k(c_k)``;
   ``delta_joint = S(a*) − S(a²)``. Ties resolve lexicographically, so a
   genuine tie yields ``delta_joint = 0`` (and a conservative gate reject).
"""

from __future__ import annotations

import itertools  # noqa: F401  (kept for introspection / future pruning utilities)
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Abstract contract
# --------------------------------------------------------------------------- #


class BaseJointDecoder(ABC):
    """Abstract joint assignment search across candidate spans."""

    @abstractmethod
    def decode(
        self,
        candidate_scores_per_param: Dict[str, List[Tuple[Any, float]]],
        entity_classes: Optional[Dict[str, str]] = None,
        top_k: int = 3,
    ) -> Tuple[Dict[str, Any], float]:
        """Finds best non-colliding assignment and computes margin to runner-up assignment.

        Returns:
            (best_assignment, delta_joint)
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Public result types
# --------------------------------------------------------------------------- #


@dataclass
class ValueChoice:
    """One aggregated canonical value for a parameter (post-R4 collapse).

    ``prob`` is the R4 mass ``p_k(v) = Σ_{c: canon(c)=v} p_k(c)``; ``span`` is
    the character span of the highest-scoring candidate in the group.
    """

    value: Any
    prob: float
    span: Optional[Tuple[int, int]] = None
    region: str = "value"
    entry_index: int = -1
    entity: Optional[str] = None  # resolved from entity_classes per param

    @property
    def log_p(self) -> float:
        """log p_k(v), clamped away from log(0)."""
        return math.log(max(self.prob, 1e-12))


def _groupkey(value: Any) -> str:
    """Hashable canonical key; numerics merge across int/float (R4)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"num:{float(value)!r}"
    if isinstance(value, (list, tuple)):
        return f"seq:{value!r}"
    return f"any:{value!r}"


def _sortkey(value: Any) -> Tuple[str, str]:
    """Deterministic, comparable ordering key for tie-breaking."""
    if value is None:
        return ("null", "")
    if isinstance(value, bool):
        return ("bool", str(value).lower())
    if isinstance(value, (int, float)):
        return ("num", repr(float(value)))
    if isinstance(value, (list, tuple)):
        return ("seq", repr(value))
    return ("any", repr(value))


# --------------------------------------------------------------------------- #
# Concrete decoder
# --------------------------------------------------------------------------- #


class ExclusivityJointDecoder(BaseJointDecoder):
    """Equal-value aggregation + top-K + exclusive joint assignment (Q3).

    Args:
        top_k: number of distinct candidate values kept per parameter.
        overlap_allowed: master switch; when True, span exclusivity is disabled
            entirely (ablation).
        free_string_overlap_allowed: allow two *untyped* utterance strings to
            share an overlapping span (e.g. same free-text quoted twice).
        allow_nested: allow a strictly-nested utterance span inside another of
            the same entity class (partial overlap still rejected).
        normalize_scores: softmax-normalize raw scores per parameter before R4
            aggregation (False when inputs are already calibrated probabilities).
        max_nodes: safety cap on the number of partial assignments explored by
            the DFS (default well above the 3^8 = 6561 worst case).
    """

    def __init__(
        self,
        top_k: int = 3,
        overlap_allowed: bool = False,
        free_string_overlap_allowed: bool = False,
        allow_nested: bool = False,
        normalize_scores: bool = True,
        max_nodes: int = 100_000,
    ) -> None:
        self.top_k = int(top_k)
        self.overlap_allowed = overlap_allowed
        self.free_string_overlap_allowed = free_string_overlap_allowed
        self.allow_nested = allow_nested
        self.normalize_scores = normalize_scores
        self.max_nodes = int(max_nodes)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def decode(
        self,
        candidate_scores_per_param: Dict[str, List[Tuple[Any, float]]],
        entity_classes: Optional[Dict[str, str]] = None,
        top_k: int = 3,
    ) -> Tuple[Dict[str, Any], float]:
        """Return ``(best_assignment, delta_joint)`` (base contract)."""
        best, delta, _, _ = self.decode_detailed(
            candidate_scores_per_param, entity_classes=entity_classes, top_k=top_k
        )
        return best, delta

    def decode_detailed(
        self,
        candidate_scores_per_param: Dict[str, List[Tuple[Any, float]]],
        entity_classes: Optional[Dict[str, str]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float, Dict[str, ValueChoice], int]:
        """Detailed joint decode.

        Returns:
            ``(best_assignment, delta_joint, chosen_per_param, n_valid)`` where
            ``chosen_per_param`` maps each parameter name to its chosen
            :class:`ValueChoice` (value, aggregated prob, span) and ``n_valid``
            is the number of conflict-free full assignments explored.
        """
        k = int(top_k) if top_k is not None else self.top_k
        entity_classes = entity_classes or {}

        per_param: Dict[str, List[ValueChoice]] = {}
        for name, entries in candidate_scores_per_param.items():
            cands = [_coerce_entry(e) for e in entries]
            choices = self._collapse(cands, k)
            for v in choices:
                v.entity = entity_classes.get(name)
            per_param[name] = choices

        names = list(per_param.keys())
        ranked = self._search(names, per_param)
        if not ranked:
            return {}, 0.0, {}, 0

        best_score, _, best_combo = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else best_score
        delta = best_score - second_score
        assignment = {name: v.value for name, v in zip(names, best_combo)}
        chosen = {name: v for name, v in zip(names, best_combo)}
        return assignment, delta, chosen, len(ranked)

    # ------------------------------------------------------------------ #
    # R4 aggregation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax(scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    def _collapse(self, cands: Sequence["_Cand"], k: int) -> List[ValueChoice]:
        """Group by canonical value, sum probability mass, keep top-K (R4)."""
        if not cands:
            return []
        if self.normalize_scores:
            probs = self._softmax([c.score for c in cands])
        else:
            probs = [max(float(c.score), 0.0) for c in cands]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

        groups: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []  # first-seen order, for stable iteration
        for c, p in zip(cands, probs):
            gk = _groupkey(c.value)
            g = groups.get(gk)
            if g is None:
                g = {
                    "value": c.value,
                    "prob": 0.0,
                    "best_score": float("-inf"),
                    "span": None,
                    "region": c.region,
                    "entry_index": -1,
                }
                groups[gk] = g
                order.append(gk)
            g["prob"] += p
            if c.score > g["best_score"]:
                g["best_score"] = c.score
                if c.char_start is not None and c.char_end is not None:
                    g["span"] = (c.char_start, c.char_end)
                g["region"] = c.region
                g["entry_index"] = c.entry_index

        choices = [
            ValueChoice(
                value=groups[gk]["value"],
                prob=groups[gk]["prob"],
                span=groups[gk]["span"],
                region=groups[gk]["region"],
                entry_index=groups[gk]["entry_index"],
            )
            for gk in order
        ]
        choices.sort(key=lambda v: (-v.prob, _sortkey(v.value)))
        return choices[:k]

    # ------------------------------------------------------------------ #
    # Conflict-pruned exhaustive search
    # ------------------------------------------------------------------ #
    def _search(
        self,
        names: List[str],
        per_param: Dict[str, List[ValueChoice]],
    ) -> List[Tuple[float, Tuple[Tuple[str, str], ...], Tuple[ValueChoice, ...]]]:
        """DFS over per-param top-K lists, pruning on span conflicts.

        Returns ranked ``(score, tie_key, combo)`` entries, best first.
        """
        ranked: List[Tuple[float, Tuple[Tuple[str, str], ...], Tuple[ValueChoice, ...]]] = []
        current: List[ValueChoice] = []
        nodes = 0

        def dfs(idx: int, score: float) -> None:
            nonlocal nodes
            nodes += 1
            if nodes > self.max_nodes:
                raise ValueError(
                    f"joint decode exceeded {self.max_nodes} partial assignments; "
                    "increase max_nodes or reduce candidate sizes"
                )
            if idx == len(names):
                ranked.append(
                    (
                        score,
                        tuple(_sortkey(v.value) for v in current),
                        tuple(current),
                    )
                )
                return
            name = names[idx]
            for v in per_param[name]:
                if any(self._conflict(v, prev) for prev in current):
                    continue
                current.append(v)
                dfs(idx + 1, score + v.log_p)
                current.pop()

        dfs(0, 0.0)
        ranked.sort(key=lambda r: (-r[0], r[1]))
        return ranked

    # ------------------------------------------------------------------ #
    # Exclusivity constraint
    # ------------------------------------------------------------------ #
    def _conflict(self, v1: ValueChoice, v2: ValueChoice) -> bool:
        """True when the two chosen values cannot co-exist in one call (Q3)."""
        if self.overlap_allowed:
            return False
        if v1.region != "utterance" or v2.region != "utterance":
            return False
        if v1.span is None or v2.span is None:
            return False
        s1, e1 = v1.span
        s2, e2 = v2.span
        if not (s1 < e2 and s2 < e1):  # half-open overlap
            return False
        if v1.entity != v2.entity:
            return False  # distinct semantic classes may share text
        if v1.entity is None:
            # both free utterance strings: shared text only when allowed
            return not self.free_string_overlap_allowed
        if self.allow_nested and self._strictly_nested(v1.span, v2.span):
            return False
        return True

    @staticmethod
    def _strictly_nested(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        s1, e1 = span1
        s2, e2 = span2
        return (s1 < s2 and e2 < e1) or (s2 < s1 and e1 < e2)


# --------------------------------------------------------------------------- #
# Candidate coercion (duck-typed; accepts model.Candidate or any object with
# canon_value/char_start/char_end/region/entry_index, or plain scalars).
# --------------------------------------------------------------------------- #


@dataclass
class _Cand:
    value: Any
    score: float
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    region: str = "value"
    entry_index: int = -1


def _coerce_entry(entry: Any) -> _Cand:
    """Turn one ``(candidate, score)`` entry into a normalized ``_Cand``."""
    if not (isinstance(entry, (tuple, list)) and len(entry) == 2
            and isinstance(entry[1], (int, float))):
        raise TypeError(
            "candidate entries must be (candidate, score) pairs, got "
            f"{type(entry).__name__}"
        )
    obj, score = entry
    score = float(score)

    if hasattr(obj, "canon_value"):
        return _Cand(
            value=obj.canon_value,
            score=score,
            char_start=getattr(obj, "char_start", None),
            char_end=getattr(obj, "char_end", None),
            region=getattr(obj, "region", "value"),
            entry_index=getattr(obj, "entry_index", -1),
        )
    # plain scalar candidate (string / None / number)
    return _Cand(value=obj, score=score)
