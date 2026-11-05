#!/usr/bin/env python3
"""CLI entrypoint for dataset ingestion, alignment, augmentation, mixing, and
JSONL emission into the 7 canonical SpanCall splits.

Pipeline (docs/full_implementation_plan.md §2 and docs/architecture.md §4.3):

1. Read ``data.sources`` from the YAML config — one entry per data source.
2. For each ``hermes`` source entry, instantiate :class:`HermesLoader` per
   configured dataset config ("setup") and ingest into ``UnifiedSample``.
3. Reverse-align every GT argument -> ``target_spans`` + stratum F/A + H4
   reasons (:class:`ReverseAligner`).
4. Augment (deterministic, seed-fixed): shuffling, paraphrase, swaps +
   distractors, R7 adversarial rows (:class:`DataAugmenter`).
5. Partition **schema-disjoint** into train/dev/calib/test_ood_schema (docs
   require zero-shot OOD-schema eval — mechanism §8.4, plan §2.3).
6. Finalize into the 7 canonical splits (:class:`DatasetMixer.finalize_splits`):
   train (55/20/10/15 mix) / dev / calib / test_id / test_ood_schema /
   test_ood_para / test_adversarial.
7. Write ``hermes_{train,dev,calib,eval,test_ood_schema,test_ood_para,
   test_adversarial}.jsonl`` + ``prepare_report.json`` under ``--output_dir``.

Example::

    python scripts/prepare_data.py --sources hermes --output_dir data/processed
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from spancall.data.align import ReverseAligner
from spancall.data.augment import DataAugmenter
from spancall.data.mix import DatasetMixer
from spancall.data.partition import write_splits_jsonl
from spancall.data.schema import Split, Stratum, UnifiedSample
from spancall.data.sources import HERMES_CONFIGS, HermesLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpanCall Data Preparation Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--sources", nargs="+", default=["hermes"], help="Data source names to ingest")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for unified JSONL")
    parser.add_argument("--skip_augment", action="store_true", help="Disable augmentation (natural-only pipeline)")
    parser.add_argument("--skip_mix", action="store_true", help="Disable the 55/20/10/15 stratified train mix")
    parser.add_argument("--seed", type=int, default=None, help="Override the pipeline seed (default: config data.schema_split.seed)")
    return parser.parse_args()


def _load_sources_from_yaml(config_path: str) -> List[Dict[str, object]]:
    with open(config_path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    return (data.get("data") or {}).get("sources") or []


def ingest_hermes(source_cfg: Dict[str, object]) -> List[UnifiedSample]:
    """Ingest all configured hermes configs into UnifiedSample records."""
    hub_id = str(source_cfg.get("hub_id") or "NousResearch/hermes-function-calling-v1")
    local_path = source_cfg.get("local_path")
    limit = source_cfg.get("limit")
    configs = source_cfg.get("configs") or HERMES_CONFIGS
    if not isinstance(configs, list):
        raise TypeError(f"data.sources[hermes].configs must be a list, got {type(configs).__name__}")

    out: List[UnifiedSample] = []
    for cfg_name in configs:
        loader = HermesLoader(
            data_path_or_hub_id=hub_id,
            config_name=str(cfg_name),
            local_path=str(local_path) if local_path else None,
            limit=int(limit) if limit else None,
        )
        samples = list(loader.load())
        out.extend(samples)
        print(
            f"[SpanCall Data] hermes/{cfg_name}: seen={loader.stats['rows_seen']} "
            f"yielded={loader.stats['rows_yielded']} "
            f"gt_calls={loader.stats['parsed_gt_calls']} "
            f"json_rows={loader.stats['json_mode_rows']} "
            f"dropped={loader.stats['dropped_no_gt'] + loader.stats['dropped_no_conversation'] + loader.stats['dropped_no_human_turn']}"
        )
    return out


def _stratum_mix_from_yaml(cfg_yaml: Dict[str, Any]) -> Dict[str, float]:
    """Read the 55/20/10/15 mix shares from ``data.stratum_mix``."""
    mix = (cfg_yaml.get("data") or {}).get("stratum_mix") or {}
    return {
        "share_natural_f": float(mix.get("natural_f", 0.55)),
        "share_paraphrase_f": float(mix.get("paraphrase_f", 0.20)),
        "share_swap_distractors": float(mix.get("swap_distractor_f", 0.10)),
        "share_abstention_a": float(mix.get("abstention_a", 0.15)),
    }


def build_report(
    splits: Dict[Split, List[UnifiedSample]],
    align_stats: Dict[str, int],
    aug_stats: Dict[str, int],
    pack_tools: int,
    seed: int,
) -> Dict[str, Any]:
    """Assemble a machine-readable prepare_report.json."""
    per_split: Dict[str, Any] = {}
    for sp, items in splits.items():
        n = len(items)
        per_split[sp.value] = {
            "count": n,
            "stratum_f": sum(1 for s in items if s.stratum == Stratum.F),
            "stratum_a": sum(1 for s in items if s.stratum == Stratum.A),
            "aug_types": dict(Counter(s.metadata.get("aug_type") for s in items)),
            "h4_reasons": dict(
                Counter(
                    r
                    for s in items
                    for r in (s.metadata.get("h4_reasons") or {}).values()
                )
            ),
        }
        # schema disjointness of the four schema-held-out splits (by schema key)
        if sp in (Split.TRAIN, Split.DEV, Split.CALIB, Split.TEST_OOD_SCHEMA):
            from spancall.data.partition import schema_key

            per_split[sp.value]["distinct_schemas"] = len({schema_key(s) for s in items})
    return {
        "pipeline": "load -> align -> augment -> schema_partition -> finalize",
        "pack_tools": pack_tools,
        "seed": seed,
        "align": align_stats,
        "augmentation": aug_stats,
        "splits": per_split,
    }


def main() -> None:
    args = parse_args()
    source_cfgs = _load_sources_from_yaml(args.config)
    hermes_cfgs = [c for c in source_cfgs if isinstance(c, dict) and c.get("name") == "hermes"]
    if not hermes_cfgs:
        print(f"[SpanCall Data] No 'hermes' source found in {args.config}; nothing to do.", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r") as fh:
        cfg_yaml = yaml.safe_load(fh) or {}

    all_samples: List[UnifiedSample] = []
    for cfg in hermes_cfgs:
        all_samples.extend(ingest_hermes(cfg))
    print(f"[SpanCall Data] Total ingested: {len(all_samples)} UnifiedSample records")

    # ---- seeds/shares from config -----------------------------------------
    schema_split = (cfg_yaml.get("data") or {}).get("schema_split") or {}
    seed = args.seed if args.seed is not None else int(schema_split.get("seed", 42))
    pack_tools = int((cfg_yaml.get("data") or {}).get("pack_tools", 1))
    mix_kwargs = _stratum_mix_from_yaml(cfg_yaml)

    # ---- 1. reverse alignment ---------------------------------------------
    aligner = ReverseAligner(
        fuzzy_matching=False,
        max_date_drift_days=14,
        align_json_mode_keys=True,
    )
    aligned: List[UnifiedSample] = []
    for s in all_samples:
        a = aligner.align(s)
        aligned.append(a)
    align_stats: Dict[str, int] = {
        "total": len(aligned),
        "stratum_f": sum(1 for s in aligned if s.stratum == Stratum.F),
        "stratum_a": sum(1 for s in aligned if s.stratum == Stratum.A),
        "h4_ok": sum(
            1
            for s in aligned
            if all(r == "OK" for r in (s.metadata.get("h4_reasons") or {}).values())
        ),
        "h4_absent": sum(
            1
            for s in aligned
            if "ABSENT" in (s.metadata.get("h4_reasons") or {}).values()
        ),
        "h4_off_window": sum(
            1
            for s in aligned
            if "OFF_WINDOW" in (s.metadata.get("h4_reasons") or {}).values()
        ),
        "h4_ambiguous": sum(
            1
            for s in aligned
            if "AMBIGUOUS" in (s.metadata.get("h4_reasons") or {}).values()
        ),
        "h4_needs_reasoning": sum(
            1
            for s in aligned
            if "NEEDS_REASONING" in (s.metadata.get("h4_reasons") or {}).values()
        ),
    }
    f_share = align_stats["stratum_f"] / align_stats["total"] if align_stats["total"] else 0.0
    print(
        f"[SpanCall Data] Aligned: F={align_stats['stratum_f']} A={align_stats['stratum_a']} "
        f"F_share={f_share:.3f} (E0 gate floor >= 0.50)"
    )

    # ---- per-config F-share (E0 gate, plan line 366) ------------------------
    per_config: Dict[str, Dict[str, int]] = {}
    for s in aligned:
        cfg = str(s.metadata.get("config") or "unknown")
        entry = per_config.setdefault(cfg, {"F": 0, "A": 0})
        entry[str(s.stratum.value)] = entry.get(str(s.stratum.value), 0) + 1
    align_stats["per_config"] = per_config
    print("[SpanCall Data] Per-config F-share (E0 gate: F_share >= 0.50 on function-calling configs):")
    for cfg in sorted(per_config):
        f = per_config[cfg]["F"]
        a = per_config[cfg]["A"]
        share = f / (f + a) if (f + a) else 0.0
        flag = "PASS" if share >= 0.50 else "---"
        print(f"  {cfg:<28} F={f:>6} A={a:>6} F_share={share:.3f}  {flag}")

    # ---- 2. augmentation ----------------------------------------------------
    aug_stats: Dict[str, int] = {"total_augmented": 0}
    if args.skip_augment:
        augmented = aligned
        print("[SpanCall Data] Augmentation skipped (--skip_augment)")
    else:
        augmenter = DataAugmenter(
            enable_shuffling=True,
            enable_paraphrase=True,
            enable_swaps=True,
            enable_distractors=True,
            enable_adversarial_markers=True,
            seed=seed,
        )
        augmented = list(aligned)
        aug_by_type: Counter = Counter()
        for s in aligned:
            variants = augmenter.augment(s)
            aug_by_type.update(v.metadata.get("aug_type") for v in variants)
            augmented.extend(variants)
        aug_stats["total_augmented"] = len(augmented) - len(aligned)
        aug_stats["by_type"] = dict(aug_by_type)
        print(f"[SpanCall Data] Augmentation produced {aug_stats['total_augmented']} variants "
              f"{dict(aug_by_type)}")

    # ---- 3. schema-disjoint partition (train/dev/calib/test_ood_schema) ----
    mixer = DatasetMixer(pack_tools=pack_tools, seed=seed, **mix_kwargs)
    schema_splits = mixer.partition_by_schema(
        augmented,
        test_schema_ratio=float(schema_split.get("test_ood_schema_ratio", 0.10)),
        calib_ratio=float(schema_split.get("calib_ratio", 0.05)),
        dev_ratio=float(schema_split.get("dev_ratio", 0.05)),
        seed=seed,
    )
    print(
        f"[SpanCall Data] Schema partition: "
        + ", ".join(f"{k.value}={len(v)}" for k, v in schema_splits.items())
    )

    # ---- 4. finalize into 7 canonical splits --------------------------------
    augmenter_final = DataAugmenter(
        enable_shuffling=False,
        enable_paraphrase=True,
        enable_swaps=False,
        enable_distractors=False,
        enable_adversarial_markers=True,
        seed=seed,
    ) if not args.skip_augment else None
    final = mixer.finalize_splits(
        schema_splits,
        train_mix=not args.skip_mix,
        augmenter=augmenter_final,
        seed=seed,
    )
    for sp in Split:
        n = len(final.get(sp, []))
        f_n = sum(1 for s in final.get(sp, []) if s.stratum == Stratum.F)
        print(f"[SpanCall Data] final/{sp.value}: n={n} F={f_n} A={n - f_n}")

    # ---- 5. write JSONL + report -------------------------------------------
    paths = write_splits_jsonl(final, args.output_dir, prefix="hermes")
    report = build_report(final, align_stats, aug_stats, pack_tools, seed)
    report_path = os.path.join(args.output_dir, "prepare_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"[SpanCall Data] Split files -> {sorted(paths.values())}")
    print(f"[SpanCall Data] Report -> {report_path}")
    print(f"[SpanCall Data] Done.")


if __name__ == "__main__":
    main()
