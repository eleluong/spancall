"""Schema-disjoint train/dev/eval splitting for UnifiedSample records.

Ref: docs/mechanism.md §8.4 ("Hold out schemas, not just utterances") and
docs/architecture.md §4.3 step 5. The docs REQUIRE zero-shot OOD-schema
evaluation, so partition must run on **whole schema groups** (frozenset of
tool names in a sample), never on individual utterances.

This module implements the missing ``partition_by_schema`` contract from
``spancall.data.mix.DatasetMixer`` as a standalone function so the other
loaders (xlam/glaive/toolbench/bfcl/tau_bench) can reuse it later.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from spancall.data.schema import Split, UnifiedSample

DEFAULT_RATIOS: Dict[str, float] = {
    "train": 0.80,
    "dev": 0.10,
    "eval": 0.10,
}

# The "eval" requested split maps to the documented in-distribution test split.
_SPLIT_BY_NAME: Dict[str, Split] = {
    "train": Split.TRAIN,
    "dev": Split.DEV,
    "eval": Split.TEST_ID,
}
_NAME_BY_SPLIT: Dict[Split, str] = {v: k for k, v in _SPLIT_BY_NAME.items()}


def _enum_aware_default(obj: Any) -> Any:
    """JSON serializer fallback that handles Enum + dataclass values."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sample_to_dict(sample: UnifiedSample) -> Dict[str, Any]:
    """Serialize a UnifiedSample to a JSON-safe dict (enums -> their values)."""
    return asdict(sample)


def schema_key(sample: UnifiedSample) -> Tuple[str, ...]:
    """Deterministic key for the schema of a sample: sorted tool names."""
    return tuple(sorted(tool.name for tool in sample.tools)) or ("__no_tools__",)


def gt_tool_key(sample: UnifiedSample) -> Tuple[str, ...]:
    """Deterministic key for the ground-truth tool(s) of a sample.

    Used for corpora like xLAM where each row packs a subset of overlapping
    candidate tools (so grouping by all candidate tools creates a single
    connected component), ensuring every GT tool is disjoint across splits.
    """
    names = tuple(sorted({c.tool_name for c in (sample.ground_truth_calls or []) if c.tool_name}))
    return names or schema_key(sample)


def partition_schema_disjoint(
    samples: Sequence[UnifiedSample],
    ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Tuple[Dict[Split, List[UnifiedSample]], Dict[str, Any]]:
    """Split samples into train/dev/eval keeping whole schema groups together.

    Args:
        samples: unified records (all configs merged, or a single config).
        ratios: ``{"train": 0.8, "dev": 0.1, "eval": 0.1}`` (default). Any
            subset of the three keys is allowed; the rest get ratio 0.
        seed: RNG seed for deterministic assignment.

    Returns:
        A ``(splits, report)`` tuple: ``splits`` maps ``Split`` -> list of
        samples; ``report`` carries per-config counts, per-split counts, the
        schema-overlap check, F-stratum share per split, and a warning when the
        schema-disjoint guarantee could not be satisfied (too few schemas).
    """
    if ratios is None:
        ratios = dict(DEFAULT_RATIOS)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("ratios must sum to > 0")

    # Normalize ratios over the requested splits (missing keys -> 0).
    requested = ["train", "dev", "eval"]
    norm: Dict[str, float] = {k: ratios.get(k, 0.0) / total for k in requested}
    split_names = [_SPLIT_BY_NAME[k] for k in requested]

    # Group samples by schema key; keep insertion order for determinism.
    groups: Dict[Tuple[str, ...], List[UnifiedSample]] = {}
    for s in samples:
        groups.setdefault(schema_key(s), []).append(s)

    distinct_schemas = len(groups)
    rng = random.Random(seed)
    assigned: Dict[Split, List[UnifiedSample]] = {k: [] for k in split_names}

    report: Dict[str, Any] = {
        "seed": seed,
        "ratios": norm,
        "total_samples": len(samples),
        "distinct_schemas": distinct_schemas,
        "schema_disjoint": True,
        "warning": None,
        "per_config": {},
        "per_split": {},
        "f_stratum_share": {},
    }

    if distinct_schemas < 3:
        # Too few schemas to split disjointly: fall back to utterance-level
        # assignment and say so loudly (the docs demand it stays visible).
        report["schema_disjoint"] = False
        report["warning"] = (
            f"Only {distinct_schemas} distinct schema(s) — cannot guarantee "
            "schema-disjoint splits; fell back to utterance-level assignment."
        )
        order = list(samples)
        rng.shuffle(order)
        for s in order:
            chosen = _weighted_split(split_names, norm, rng)
            assigned[chosen].append(s)
    else:
        # Assign whole schema groups to the split with the largest remaining
        # deficit, processing groups LARGEST-FIRST. Heavy-tailed schema sizes
        # (e.g. json_output 1488 rows, ExpertQAExtractor 1464 rows) must be
        # placed early against the largest deficit so the tail of tiny groups
        # can fine-tune the ratios; smallest-first would dump the tail into one
        # split and overshoot the ratios by thousands of rows.
        # Shuffle first (seed-deterministic), then stable-sort by size so ties
        # break randomly but big groups always precede small ones.
        keys = list(groups.keys())
        rng.shuffle(keys)
        ordered_keys = sorted(keys, key=lambda k: -len(groups[k]))
        counts: Dict[Split, int] = {k: 0 for k in split_names}
        for key in ordered_keys:
            deficits = {
                n: (norm[_NAME_BY_SPLIT[n]] * len(samples) - counts[n])
                for n in split_names
            }
            chosen = max(split_names, key=lambda n: deficits[n])
            assigned[chosen].extend(groups[key])
            counts[chosen] += len(groups[key])

    # ---- statistics -------------------------------------------------------
    for sp, items in assigned.items():
        report["per_split"][sp.value] = len(items)
        f_count = sum(1 for s in items if s.stratum == s.stratum.F)
        report["f_stratum_share"][sp.value] = f_count / len(items) if items else 0.0
    for s in samples:
        cfg = s.metadata.get("config", "unknown")
        report["per_config"][cfg] = report["per_config"].get(cfg, 0) + 1

    return assigned, report


def _weighted_split(
    split_names: List[Split],
    ratios: Dict[str, float],
    rng: random.Random,
    ratio_by_split: Optional[Dict[Split, float]] = None,
) -> Split:
    """Pick a split proportionally, using ``ratio_by_split`` (preferred), else
    ``ratios`` keyed by legacy name (train/dev/eval) or by :class:`Split`.

    Supports arbitrary split sets (e.g. the 4-way train/dev/calib/
    test_ood_schema partition) by treating ``ratios`` as direct ``Split``-keyed
    weights whenever the name-keyed lookup does not apply.
    """
    if ratio_by_split is None:
        ratio_by_split = {}
        for name in split_names:
            if name in ratios:
                ratio_by_split[name] = ratios[name]
            else:
                ratio_by_split[name] = ratios[_NAME_BY_SPLIT[name]]
    r = rng.random()
    acc = 0.0
    for name in split_names:
        acc += ratio_by_split[name]
        if r <= acc:
            return name
    return split_names[-1]


def partition_schema_groups(
    samples: Sequence[UnifiedSample],
    targets: Dict[str, float],
    seed: int = 42,
    group_by: str = "schema",
) -> Tuple[Dict[Split, List[UnifiedSample]], Dict[str, Any]]:
    """Generalized schema-disjoint partition over *arbitrary* target splits.

    The default 3-way :func:`partition_schema_disjoint` is hardcoded to
    train/dev/eval; the full pipeline
    needs 4 schema-held-out buckets — ``train`` 80%, ``dev`` 5%, ``calib`` 5%,
    ``test_ood_schema`` 10% — so dev/calib and OOD-schema evaluation stay
    schema-disjoint from training. This function shares the largest-first
    greedy assignment (heavy-tailed schema groups land first against the
    largest deficit; the tail fine-tunes the ratios).

    Args:
        samples: unified records.
        targets: ``{Split.value (or name): ratio}`` — e.g.
            ``{"train": 0.80, "dev": 0.05, "calib": 0.05,
                "test_ood_schema": 0.10}``. Ratios are normalized to sum 1.
        seed: RNG seed for deterministic assignment.
        group_by: ``"schema"`` (sorted candidate tool names, default) or
            ``"gt_tool"`` (sorted ground-truth tool names, recommended for
            multi-tool-candidate corpora like xLAM).

    Returns:
        ``(splits, report)``: ``splits`` maps ``Split`` -> samples; ``report``
        carries counts, schema-overlap status and a fallback warning.
    """
    if not targets:
        raise ValueError("targets must be non-empty")
    if group_by not in ("schema", "gt_tool"):
        raise ValueError(f"Unknown group_by={group_by!r}; expected 'schema' or 'gt_tool'")
    key_fn = gt_tool_key if group_by == "gt_tool" else schema_key
    # Normalize ratios over the requested splits (key by Split).
    total = sum(float(v) for v in targets.values())
    norm: Dict[Split, float] = {}
    for name, ratio in targets.items():
        sp = Split(name) if name in {s.value for s in Split} else _lookup_split_name(name)
        if sp is None:
            raise ValueError(f"Unknown split target {name!r}; expected one of "
                             f"{[s.value for s in Split]}")
        norm[sp] = float(ratio) / total
    requested = list(norm.keys())

    groups: Dict[Tuple[str, ...], List[UnifiedSample]] = {}
    for s in samples:
        groups.setdefault(key_fn(s), []).append(s)
    distinct_schemas = len(groups)
    rng = random.Random(seed)
    assigned: Dict[Split, List[UnifiedSample]] = {k: [] for k in requested}

    report: Dict[str, Any] = {
        "seed": seed,
        "ratios": {k.value: v for k, v in norm.items()},
        "total_samples": len(samples),
        "distinct_schemas": distinct_schemas,
        "schema_disjoint": True,
        "warning": None,
        "per_split": {},
        "f_stratum_share": {},
    }

    if distinct_schemas < len(requested):
        report["schema_disjoint"] = False
        report["warning"] = (
            f"Only {distinct_schemas} distinct schema(s) — cannot guarantee "
            "schema-disjoint splits; fell back to utterance-level assignment."
        )
        order = list(samples)
        rng.shuffle(order)
        for s in order:
            assigned[
                _weighted_split(requested, norm, rng, ratio_by_split=norm)
            ].append(s)
    else:
        # largest-first greedy against the largest remaining deficit (see the
        # docstring of partition_schema_disjoint for the heavy-tail rationale)
        keys = list(groups.keys())
        rng.shuffle(keys)
        ordered_keys = sorted(keys, key=lambda k: -len(groups[k]))
        counts: Dict[Split, int] = {k: 0 for k in requested}
        for key in ordered_keys:
            deficits = {
                n: (norm[n] * len(samples) - counts[n])
                for n in requested
            }
            chosen = max(requested, key=lambda n: deficits[n])
            assigned[chosen].extend(groups[key])
            counts[chosen] += len(groups[key])

    for sp, items in assigned.items():
        report["per_split"][sp.value] = len(items)
        f_count = sum(1 for s in items if s.stratum == s.stratum.F)
        report["f_stratum_share"][sp.value] = f_count / len(items) if items else 0.0
    return assigned, report


def _lookup_split_name(name: str) -> Optional[Split]:
    """Resolve a human split name (``train``, ``dev``, ``eval``, ``calib``,
    ``test_ood_schema``, ``test_id``, ``test_ood_para``, ``test_adversarial``)
    to a :class:`Split` member, else None."""
    lowered = name.lower()
    if lowered == "eval":
        return Split.TEST_ID
    for member in Split:
        if member.value == lowered:
            return member
    return None


# Reverse map for user-facing file names: the requested "eval" split is stored
# as Split.TEST_ID internally, but written as `<prefix>_eval.jsonl`.
_FILENAME_BY_SPLIT: Dict[Split, str] = {
    Split.TRAIN: "train",
    Split.DEV: "dev",
    Split.CALIB: "calib",
    Split.TEST_ID: "eval",
    Split.TEST_OOD_SCHEMA: "test_ood_schema",
    Split.TEST_OOD_PARA: "test_ood_para",
    Split.TEST_ADVERSARIAL: "test_adversarial",
}


def write_splits_jsonl(
    splits: Dict[Split, List[UnifiedSample]],
    output_dir: str,
    prefix: str = "hermes",
) -> Dict[str, str]:
    """Write each split to ``<output_dir>/<prefix>_<name>.jsonl``.

    The serialized row is ``asdict(UnifiedSample)`` (enums become values via
    the default encoder at write time). User-facing names: ``train``/``dev``/
    ``eval``/``calib``/``test_ood_schema``/``test_ood_para``/``test_adversarial``
    (the eval split is stored as :data:`Split.TEST_ID` internally).
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for sp, items in splits.items():
        name = _FILENAME_BY_SPLIT.get(sp, sp.value)
        path = os.path.join(output_dir, f"{prefix}_{name}.jsonl")
        with open(path, "w") as fh:
            for s in items:
                fh.write(json.dumps(sample_to_dict(s), default=_enum_aware_default) + "\n")
        paths[sp.value] = path
    return paths