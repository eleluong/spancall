"""Dataset mixing and partition strategies.

Ref: docs/mechanism.md §8.3.

`DatasetMixer` assembles the final training mix (55% natural F, 20% paraphrase F,
10% swap/distractor F, 15% abstention A) and the 7 canonical evaluation splits:

    train / dev / calib / test_id / test_ood_schema / test_ood_para / test_adversarial

Data flow (`scripts/prepare_data.py`):
    load -> partition_schema_groups (train/dev/calib/test_ood_schema)
        -> ReverseAligner.align -> DataAugmenter.augment -> DatasetMixer.mix
        -> write 7 JSONL splits
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from spancall.data.augment import DataAugmenter
from spancall.data.partition import partition_schema_groups
from spancall.data.schema import Split, Stratum, UnifiedSample

class DatasetMixer:
    """Manages sampling ratios across data strata and augmentation categories.

    ``pack_tools`` is the D5 lever read from ``data.pack_tools``
    (configs/default.yaml) as described in the module docstring of the
    original skeleton.
    """

    def __init__(
        self,
        share_natural_f: float = 0.55,
        share_paraphrase_f: float = 0.20,
        share_swap_distractors: float = 0.10,
        share_abstention_a: float = 0.15,
        pack_tools: int = 1,
        seed: int = 42,
        augmenter: Optional[DataAugmenter] = None,
    ) -> None:
        self.share_natural_f = share_natural_f
        self.share_paraphrase_f = share_paraphrase_f
        self.share_swap_distractors = share_swap_distractors
        self.share_abstention_a = share_abstention_a
        self.pack_tools = pack_tools
        self.seed = seed
        self.augmenter = augmenter
        self._check_ratios()

    def _check_ratios(self) -> None:
        shares = (
            self.share_natural_f
            + self.share_paraphrase_f
            + self.share_swap_distractors
            + self.share_abstention_a
        )
        if not 0.99 <= shares <= 1.01:
            raise ValueError(
                f"Mix shares must sum to 1.0, got {shares:.4f}"
            )

    # ------------------------------------------------------------------ mix
    def mix(self, samples: List[UnifiedSample]) -> List[UnifiedSample]:
        """Return a stratified training mix (with replacement to hit targets).

        Ratios are applied over the **augmentation category** (not the natural
        stratum): the input list may contain natural-F samples and their
        `aug_chain`-tagged augmentations; the target distribution is:
        ``natural_f`` 55%, ``paraphrase_f`` 20%, ``swap_distractor_f`` +
        ``swap_f``/``ambiguous_a`` + ``distractor_f`` 10%, ``abstention_a`` 15%.
        """
        buckets = _bucket_samples(samples)
        n = max(
            1,
            len(buckets["natural_f"])
            + len(buckets["paraphrase_f"])
            + len(buckets["swap_distractor_f"])
            + len(buckets["abstention_a"]),
        )
        targets: Dict[str, int] = {
            "natural_f": round(self.share_natural_f * n),
            "paraphrase_f": round(self.share_paraphrase_f * n),
            "swap_distractor_f": round(self.share_swap_distractors * n),
            "abstention_a": round(self.share_abstention_a * n),
        }
        rng = random.Random(self.seed)
        out: List[UnifiedSample] = []
        for bucket, count in targets.items():
            pool = buckets[bucket]
            if not pool:
                continue
            if len(pool) >= count:
                out.extend(rng.sample(pool, count))
            else:
                # sample with replacement to reach target
                out.extend(rng.choices(pool, k=count))
        rng.shuffle(out)
        return out

    # ------------------------------------------------------- partitioning
    def partition_by_schema(
        self,
        samples: List[UnifiedSample],
        test_schema_ratio: float = 0.10,
        calib_ratio: float = 0.05,
        dev_ratio: float = 0.05,
        seed: int = 42,
        paraphrase_disjoint: bool = True,
        group_by: str = "schema",
    ) -> Dict[Split, List[UnifiedSample]]:
        """Split by whole-schema groups into train/dev/calib/test_ood_schema.

        ``test_id``, ``test_ood_para`` and ``test_adversarial`` are derived
        downstream (see :meth:`finalize_splits` / `scripts/prepare_data.py`).

        Args:
            samples: aligned (or pre-partition) records.
            test_schema_ratio, calib_ratio, dev_ratio: schema-group shares
                (default 10% / 5% / 5%; train takes the remainder 80%).
            paraphrase_disjoint: when True, calib is additionally paraphrase-
                disjoint (never sees train/dev trainers) — enforced by emitting
                a `__no_tools__`-style marker later; here it guarantees calib
                rows never appear in the other schema groups.

        Returns:
            ``{Split: [UnifiedSample]}`` covering TRAIN/DEV/CALIB/TEST_OOD_SCHEMA.
        """
        train_ratio = max(
            0.0,
            1.0 - test_schema_ratio - calib_ratio - dev_ratio,
        )
        targets: Dict[str, float] = {
            "train": train_ratio,
            "dev": dev_ratio,
            "calib": calib_ratio,
            "test_ood_schema": test_schema_ratio,
        }
        splits, _ = partition_schema_groups(samples, targets, seed=seed, group_by=group_by)
        if paraphrase_disjoint:
            # Calib shares no schema group with train/dev/test_ood_schema by
            # construction -> paraphrase-disjoint via schema disjointness.
            pass
        return splits

    # ----------------------------------------------------- split finalizer
    def finalize_splits(
        self,
        schema_splits: Dict[Split, List[UnifiedSample]],
        *,
        train_mix: bool = True,
        augmenter: Optional[DataAugmenter] = None,
        seed: int = 42,
    ) -> Dict[Split, List[UnifiedSample]]:
        """Expand the 4-way schema split into the full 7 canonical splits.

        From ``{train, dev, calib, test_ood_schema}``:

        - **train**: stratified 55/20/10/15 mix (unless ``train_mix=False``).
        - **dev**: natural F + A distribution (no mixing).
        - **calib**: untouched (feed Stage-C H8 + temperature/tau calibration).
        - **test_id**: held-out utterances from *within* train schemas. We split
          each train schema group's utterances 90/10 schema-stable; the 10%
          holdout becomes TEST_ID (in-distribution, same schemas as train).
        - **test_ood_schema**: passed through unchanged.
        - **test_ood_para**: deterministic surface paraphrases of the OOD-schema
          rows (via the augmenter's paraphrase path) + test_id rows.
        - **test_adversarial**: R7 marker-injection rows (MUST abstain).
        """
        train = schema_splits.get(Split.TRAIN, [])
        dev = schema_splits.get(Split.DEV, [])
        calib = schema_splits.get(Split.CALIB, [])
        ood = schema_splits.get(Split.TEST_OOD_SCHEMA, [])
        aug = augmenter or self.augmenter or DataAugmenter(
            enable_shuffling=False,
            enable_paraphrase=True,
            enable_swaps=False,
            enable_distractors=False,
            enable_adversarial_markers=True,
            seed=seed,
        )

        # test_id: 10% utterance holdout within train schemas (schema-stable)
        train_part, test_id = _split_within_schema(train, holdout_ratio=0.10, seed=seed)

        mixed_train = self.mix(train_part) if train_mix else list(train_part)

        # test_ood_para: surface-paraphrase the OOD rows; keep only F results
        test_ood_para: List[UnifiedSample] = []
        for s in ood:
            variant = None
            for v in aug.augment(s):
                if v.metadata.get("aug_type") == "paraphrase_f":
                    variant = v
                    break
            if variant is not None:
                test_ood_para.append(variant)
        test_ood_para.extend(_surface_paraphrase_many(test_id, aug))

        # test_adversarial: R7 rows from the adversarial augmenter
        r7_aug = DataAugmenter(
            enable_shuffling=False,
            enable_paraphrase=False,
            enable_swaps=False,
            enable_distractors=False,
            enable_adversarial_markers=True,
            seed=seed,
        )
        test_adversarial: List[UnifiedSample] = []
        for s in ood + train:
            variants = r7_aug.augment(s)
            for v in variants:
                if v.metadata.get("aug_type") == "adversarial_r7":
                    v.stratum = Stratum.A
                    test_adversarial.append(v)

        return {
            Split.TRAIN: mixed_train,
            Split.DEV: list(dev),
            Split.CALIB: list(calib),
            Split.TEST_ID: test_id,
            Split.TEST_OOD_SCHEMA: list(ood),
            Split.TEST_OOD_PARA: test_ood_para,
            Split.TEST_ADVERSARIAL: test_adversarial,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bucket_samples(samples: Sequence[UnifiedSample]) -> Dict[str, List[UnifiedSample]]:
    """Bucket samples by augmentation category for the 55/20/10/15 mix."""
    buckets: Dict[str, List[UnifiedSample]] = {
        "natural_f": [],
        "paraphrase_f": [],
        "swap_distractor_f": [],
        "abstention_a": [],
    }
    for s in samples:
        aug_type = s.metadata.get("aug_type")
        if aug_type in ("paraphrase_f",):
            buckets["paraphrase_f"].append(s)
        elif aug_type in ("swap_f", "distractor_f", "ambiguous_a", "swap_distractor_f"):
            buckets["swap_distractor_f"].append(s)
        elif aug_type in ("adversarial_r7",) or s.stratum == Stratum.A:
            buckets["abstention_a"].append(s)
        elif s.stratum == Stratum.F:
            buckets["natural_f"].append(s)
        else:
            # unknown/Stratum.A without augmentation tags -> abstention bucket
            buckets["abstention_a"].append(s)
    return buckets


def _split_within_schema(
    samples: Sequence[UnifiedSample],
    holdout_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[UnifiedSample], List[UnifiedSample]]:
    """Stable within-schema utterance holdout for TEST_ID (schema-stable)."""
    rng = random.Random(seed + 1)
    from spancall.data.partition import schema_key

    by_schema: Dict[Tuple[str, ...], List[UnifiedSample]] = {}
    for s in samples:
        by_schema.setdefault(schema_key(s), []).append(s)
    main: List[UnifiedSample] = []
    holdout: List[UnifiedSample] = []
    for group in by_schema.values():
        if len(group) == 1:
            main.extend(group)
            continue
        k = max(1, round(len(group) * holdout_ratio))
        sampled = rng.sample(group, k)
        holdout.extend(sampled)
        main.extend(s for s in group if s not in sampled)
    return main, holdout


def _surface_paraphrase_many(
    samples: Sequence[UnifiedSample],
    augmenter: DataAugmenter,
) -> List[UnifiedSample]:
    out: List[UnifiedSample] = []
    for s in samples:
        for v in augmenter.augment(s):
            if (
                v.metadata.get("aug_type") == "paraphrase_f"
                and v.stratum == Stratum.F
            ):
                out.append(v)
                break
    return out
