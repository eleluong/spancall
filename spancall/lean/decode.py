"""Lean Joint Assignment Decoder.

Implements:
1. R4 value aggregation across candidate spans.
2. Conflict-pruned DFS over top-K candidate values.
3. Robust fallback: if conflict-pruned search finds no conflict-free assignment,
   it falls back to greedy top-1 selection per parameter instead of wiping out
   the assignment (preventing empty payload collapse).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class LeanValueChoice:
    """One aggregated canonical value for a parameter (post-R4 collapse)."""

    value: Any
    prob: float
    span: Optional[Tuple[int, int]] = None
    region: str = "value"
    entry_index: int = -1
    entity: Optional[str] = None

    @property
    def log_p(self) -> float:
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


@dataclass
class LeanCallDecode:
    """One decoded call for the R0.4 multi-call path.

    ``values`` holds that call's ``param -> canonical value`` assignment. Calls
    are decoded independently, so two calls may legitimately resolve to the same
    span (e.g. the same ``city`` for a weather tool and a hotel tool) -- that is
    not a conflict, it is the correct answer for both.
    """

    call_index: int
    values: Dict[str, Any]
    delta: float
    chosen: Dict[str, LeanValueChoice]
    n_nodes: int = 0


@dataclass
class _Cand:
    value: Any
    score: float
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    region: str = "value"
    entry_index: int = -1


def _coerce_entry(entry: Any) -> _Cand:
    """Turn one (candidate, score) entry into a normalized _Cand."""
    if not (isinstance(entry, (tuple, list)) and len(entry) == 2 and isinstance(entry[1], (int, float))):
        raise TypeError(f"candidate entries must be (candidate, score) pairs, got {type(entry).__name__}")
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
    return _Cand(value=obj, score=score)


class LeanJointDecoder:
    """Lean joint decoder with R4 aggregation and resilient greedy fallback."""

    def __init__(
        self,
        top_k: int = 3,
        overlap_allowed: bool = False,
        free_string_overlap_allowed: bool = True,
        allow_nested: bool = True,
        normalize_scores: bool = True,
        max_nodes: int = 50_000,
        min_confidence: Optional[float] = None,
        min_posterior: Optional[float] = None,
    ) -> None:
        self.top_k = int(top_k)
        self.overlap_allowed = overlap_allowed
        self.free_string_overlap_allowed = free_string_overlap_allowed
        self.allow_nested = allow_nested
        self.normalize_scores = normalize_scores
        self.max_nodes = int(max_nodes)
        # T6: refusal on the DECODE axis. Lean has no abstain head, so a param
        # with a flat, low-score candidate distribution used to be force-assigned
        # its argmax regardless — every param got a value and every call looked
        # confident. These thresholds let the decoder emit "no choice" instead:
        #   ``min_confidence`` — the winning candidate's own probability must
        #                        clear this floor.
        #   ``min_posterior``  — the winner must beat the runner-up by at least
        #                        this much (a mass-tied distribution has no winner).
        # Both default to None (never refuse), so existing callers and every
        # current test keep byte-identical behaviour. Refusal is opt-in because
        # it changes recall semantics and must be measured against the fixed
        # candidate set (T2) before being enabled in production.
        self.min_confidence = float(min_confidence) if min_confidence is not None else None
        self.min_posterior = float(min_posterior) if min_posterior is not None else None
        # T6: last refusal reasons, keyed by parameter name. Initialised here (not
        # only inside ``decode_detailed``) so a resolver can read it before the
        # first decode without an AttributeError.
        self.last_refusals: Dict[str, str] = {}

    def _should_refuse(self, choices: Sequence["LeanValueChoice"]) -> Tuple[bool, str]:
        """T6: decide whether a param has a defensible winner at all.

        Returns ``(refuse, reason)``. A parameter with NO candidates is not a
        refusal — that is a coverage outcome the caller already reports — so
        only a *populated* candidate set can trigger this.
        """
        if not choices:
            return False, ""
        best = float(getattr(choices[0], "prob", 0.0) or 0.0)
        if self.min_confidence is not None and best < self.min_confidence:
            return True, "below_min_confidence"
        if self.min_posterior is not None and len(choices) > 1:
            second = float(getattr(choices[1], "prob", 0.0) or 0.0)
            if (best - second) < self.min_posterior:
                return True, "indistinguishable_from_runner_up"
        return False, ""

    def decode(
        self,
        candidate_scores_per_param: Dict[str, List[Tuple[Any, float]]],
        entity_classes: Optional[Dict[str, str]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float]:
        best, delta, _, _ = self.decode_detailed(
            candidate_scores_per_param, entity_classes=entity_classes, top_k=top_k
        )
        return best, delta

    def decode_detailed(
        self,
        candidate_scores_per_param: Dict[str, List[Tuple[Any, float]]],
        entity_classes: Optional[Dict[str, str]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float, Dict[str, LeanValueChoice], int]:
        k = int(top_k) if top_k is not None else self.top_k
        entity_classes = entity_classes or {}

        per_param: Dict[str, List[LeanValueChoice]] = {}
        self.last_refusals: Dict[str, str] = {}
        for name, entries in candidate_scores_per_param.items():
            cands = [_coerce_entry(e) for e in entries]
            choices = self._collapse(cands, k)
            for v in choices:
                v.entity = entity_classes.get(name)
            # T6: drop the winner entirely when the candidate distribution does
            # not support one. An empty list makes the param fall into the
            # caller's "missing" branch (the honest outcome) instead of the
            # force-assign-argmax path that made every call look confident.
            refuse, reason = self._should_refuse(choices)
            if refuse:
                self.last_refusals[name] = reason
                choices = []
            per_param[name] = choices

        names = list(per_param.keys())
        active_names = [name for name in names if per_param[name]]
        if not active_names:
            return {}, 0.0, {}, 0

        ranked = self._search(active_names, per_param)
        if not ranked:
            # Resilient greedy fallback: instead of failing to empty dict {},
            # select the top-1 choice for each active parameter.
            best_combo = [per_param[name][0] for name in active_names]
            assignment = {name: v.value for name, v in zip(active_names, best_combo)}
            chosen = {name: v for name, v in zip(active_names, best_combo)}
            return assignment, 0.0, chosen, 0

        best_score, _, best_combo = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else best_score
        delta = best_score - second_score
        assignment = {name: v.value for name, v in zip(active_names, best_combo)}
        chosen = {name: v for name, v in zip(active_names, best_combo)}
        return assignment, delta, chosen, len(ranked)

    def decode_calls(
        self,
        per_param_scores_by_call: Any,
        entity_classes: Optional[Dict[str, str]] = None,
        top_k: Optional[int] = None,
    ) -> List[LeanCallDecode]:
        """Decode every call independently (R0.4).

        ``per_param_scores_by_call`` is either a mapping ``{call_index: {param:
        [(candidate, score), ...]}}`` (the tensorizer's ``call_param_index``
        layout, where index ``k`` of the flat anchor list owns one call) or a
        plain sequence of such per-param dicts, in which case enumeration order
        defines ``call_index``.

        Each call runs the unchanged single-call search, so exclusivity is
        scoped **within** a call and never across calls: two calls resolving to
        the same span is the correct answer for e.g. "weather in Tokyo and a
        hotel in Tokyo", and pruning it would zero out a whole call.

        Returns one :class:`LeanCallDecode` per call, ordered by ``call_index``.
        A call with no non-empty parameter set is still represented (with empty
        ``values``) so ``len(result)`` always equals the number of calls asked
        for -- callers rely on position to align with ``call_tool_targets``.
        """
        if isinstance(per_param_scores_by_call, dict):
            items: List[Tuple[int, Any]] = sorted(
                per_param_scores_by_call.items(), key=lambda kv: int(kv[0])
            )
        else:
            items = list(enumerate(per_param_scores_by_call))

        out: List[LeanCallDecode] = []
        for ci, per_param in items:
            values, delta, chosen, n_nodes = self.decode_detailed(
                per_param or {}, entity_classes=entity_classes, top_k=top_k
            )
            out.append(
                LeanCallDecode(
                    call_index=int(ci),
                    values=values,
                    delta=float(delta),
                    chosen=chosen,
                    n_nodes=int(n_nodes),
                )
            )
        return out

    @staticmethod
    def _softmax(scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        if total <= 0.0:
            return [1.0 / len(scores)] * len(scores)
        return [e / total for e in exps]

    def _collapse(self, cands: Sequence[_Cand], k: int) -> List[LeanValueChoice]:
        """Group candidates by canonical value, sum probabilities, keep top-K."""
        if not cands:
            return []
        if self.normalize_scores:
            probs = self._softmax([c.score for c in cands])
        else:
            probs = [max(float(c.score), 0.0) for c in cands]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(cands)] * len(cands)

        groups: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
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
            LeanValueChoice(
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

    def _search(
        self,
        names: List[str],
        per_param: Dict[str, List[LeanValueChoice]],
    ) -> List[Tuple[float, Tuple[Tuple[str, str], ...], Tuple[LeanValueChoice, ...]]]:
        ranked: List[Tuple[float, Tuple[Tuple[str, str], ...], Tuple[LeanValueChoice, ...]]] = []
        current: List[LeanValueChoice] = []
        nodes = 0

        def dfs(idx: int, score: float) -> None:
            nonlocal nodes
            nodes += 1
            if nodes > self.max_nodes:
                return
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

    def _conflict(self, v1: LeanValueChoice, v2: LeanValueChoice) -> bool:
        if self.overlap_allowed:
            return False
        if v1.region != "utterance" or v2.region != "utterance":
            return False
        if v1.span is None or v2.span is None:
            return False
        s1, e1 = v1.span
        s2, e2 = v2.span
        if not (s1 < e2 and s2 < e1):
            return False
        if v1.entity != v2.entity:
            return False
        if v1.entity is None:
            return not self.free_string_overlap_allowed
        if self.allow_nested and self._strictly_nested(v1.span, v2.span):
            return False
        return True

    @staticmethod
    def _strictly_nested(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        s1, e1 = span1
        s2, e2 = span2
        return (s1 < s2 and e2 < e1) or (s2 < s1 and e1 < e2)

