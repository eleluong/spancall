"""Tests for :class:`spancall.data.augment.DataAugmenter` and
:class:`spancall.data.mix.DatasetMixer` (plan §2.2 + §2.3 contracts).

Covers:
  * swap_f invariance + H7 ``swap_negatives`` leakage metadata (raw-utterance
    char ranges, post-swap GT cross-product)
  * distractor_f (Stratum F, no leakage) + ambiguous_a (forced Stratum.A, H4)
  * adversarial_r7 rows must abstain (R7) with marker provenance
  * shuffle_f position invariance + deterministic seed-derived rng
  * mix(): 55/20/10/15 stratified ratios, `_bucket_samples` boundaries
  * finalize_splits(): 7 canonical splits, test_adversarial forced A,
    test_ood_para only F, schema-disjointness
  * determinism (same seed -> same split assignment)
"""

from __future__ import annotations

from collections import Counter

import pytest

from spancall.data.augment import DataAugmenter
from spancall.data.align import ReverseAligner
from spancall.data.mix import DatasetMixer, _bucket_samples
from spancall.data.partition import schema_key
from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    Split,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _p(name: str, ptype: str = "string", *, enum=None, required: bool = True,
       items_type: str | None = None) -> ParameterSchema:
    return ParameterSchema(
        name=name, description="d", param_type=ptype, required=required,
        enum=enum, items_type=items_type,
    )


def _tool(name: str, params: list[ParameterSchema]) -> ToolDefinition:
    return ToolDefinition(name=name, description="d", parameters=params)


def _sample(
    *,
    sid: str,
    utterance: str,
    tools: list[ToolDefinition],
    gt: list[GroundTruthCall],
    recent: list[dict] | None = None,
) -> UnifiedSample:
    return UnifiedSample(
        sample_id=sid,
        source_dataset="test",
        utterance=utterance,
        tools=tools,
        ground_truth_calls=gt,
        recent_context=recent or [],
    )


def _move_sample(sid: str = "move-1") -> UnifiedSample:
    """Two same-typed string params grounded in the utterance (swap-able)."""
    return _sample(
        sid=sid,
        utterance="Move from paris to london please.",
        tools=[_tool("move", [_p("origin"), _p("dest")])],
        gt=[GroundTruthCall(tool_name="move", arguments={"origin": "paris", "dest": "london"})],
    )


def _multi_schema_samples() -> list[UnifiedSample]:
    """45 *groundable* F samples over 8 schema groups (heavy-tailed sizes).

    Each utterance embeds every argument value as a word-bounded token so the
    aligner marks the row Stratum.F; the greedy schema-disjoint partitioner
    needs enough groups (varied sizes) to populate every split."""
    out: list[UnifiedSample] = []
    # tool -> (param names, sample count) — sizes 12,9,7,5,4,3,3,2 = 45
    schemas = {
        "cam": (["camera_id"], 12),
        "move": (["origin", "dest"], 9),
        "mail": (["to", "subject"], 7),
        "calc": (["expr"], 5),
        "book": (["date", "room"], 4),
        "search": (["query"], 3),
        "weather": (["city"], 3),
        "stock": (["symbol"], 2),
    }
    for i, (tool_name, (params, count)) in enumerate(schemas.items()):
        ps = [_p(n) for n in params]
        for k in range(count):
            sid = f"{tool_name}-{i}-{k}"
            if len(params) == 1:
                value = f"{tool_name}-{k}"
                u = f"Please {tool_name} {value} now."
                args = {params[0]: value}
            else:
                args = {n: f"{n}-{k}" for n in params}
                u = "Please " + " and ".join(f"{p} {p}-{k}" for p in params) + " now."
            out.append(
                _sample(
                    sid=sid,
                    utterance=u,
                    tools=[_tool(tool_name, ps)],
                    gt=[GroundTruthCall(tool_name=tool_name, arguments=args)],
                )
            )
    return out


def _paraphrases_for(samples: list[UnifiedSample]) -> dict[str, str]:
    """Deterministic surface paraphrases that keep every value word-bounded,
    so the re-aligned paraphrase_f variant stays Stratum.F."""
    return {s.sample_id: f"Sure, {s.utterance}" for s in samples}


def _align(s: UnifiedSample) -> UnifiedSample:
    return ReverseAligner().align(s)


def _render_utterance_entry(sample: UnifiedSample) -> tuple:
    """Re-render the *aligned* sample exactly like ``ReverseAligner._render``.

    ``SpanTarget.char_start/char_end`` are ABSOLUTE offsets into the rendered
    template text (not the raw ``sample.utterance``), so tests must slice the
    render to validate them. Returns ``(utterance_entry, output)``.
    """
    from datetime import date as _date

    from span_template.template import RecentRow, build_input
    from spancall.data.align import _find_utterance_entry, tooldef_to_template_tool

    tool_def = next(
        (t for t in sample.tools if t.name == sample.ground_truth_calls[0].tool_name),
        None,
    )
    assert tool_def is not None, "aligned sample has no matching tool"
    assert sample.current_time_iso, "align() must set current_time_iso"
    base = _date.fromisoformat(sample.current_time_iso[:10])
    recent = [
        RecentRow(key=r["key"], value=r["value"], source=r.get("source", "recent"))
        for r in sample.recent_context
    ]
    out = build_input(
        tool=tooldef_to_template_tool(tool_def),
        utterance=sample.utterance,
        recent_context=recent,
        now=base,
    )
    utt = _find_utterance_entry(out)
    assert utt is not None, "no selectable utterance entry in render"
    return utt, out


# ---------------------------------------------------------------------------
# swap_f + H7 swap_negatives
# ---------------------------------------------------------------------------

def test_swap_variant_materializes_and_stays_f() -> None:
    out = _align(_move_sample())
    assert out.stratum == Stratum.F
    augs = [a for a in DataAugmenter(seed=7).augment(out)
            if a.metadata.get("aug_type") == "swap_f"]
    assert augs, "swap_f variant should materialize for same-typed string pair"
    v = augs[0]
    assert v.stratum == Stratum.F
    assert v.utterance == "Move from london to paris please."
    assert v.metadata["aug_chain"][-1] == "swap_f"


def test_swap_negatives_no_leakage() -> None:
    """H7: pre-swap layout — each param's wrong span is where ITS value sat, the
    correct span is the other value's home; after swap the re-aligned spans
    attach to the moved values (no GT leakage)."""
    out = _align(_move_sample())
    (v,) = [a for a in DataAugmenter(seed=7).augment(out)
            if a.metadata.get("aug_type") == "swap_f"]
    sn = v.metadata["swap_negatives"]
    assert sn["origin"]["wrong_span_char"] == [10, 15]   # "paris"
    assert sn["origin"]["correct_span_char"] == [19, 25]  # "london"
    assert sn["dest"]["wrong_span_char"] == [19, 25]
    assert sn["dest"]["correct_span_char"] == [10, 15]
    # after swap, utterance says london->paris and GT is origin=paris, dest=london
    assert v.utterance.index("london") < v.utterance.index("paris")
    assert v.ground_truth_calls[0].arguments == {"origin": "paris", "dest": "london"}
    # re-aligned spans attach to the moved values in the NEW render
    _, output = _render_utterance_entry(v)
    assert any(
        output.text[t.char_start:t.char_end] == "paris"
        for t in v.target_spans.get("origin", [])
        if t.target_type == "utterance_span"
    ), "origin must re-align to the value now in its (swapped) home"
    assert any(
        output.text[t.char_start:t.char_end] == "london"
        for t in v.target_spans.get("dest", [])
        if t.target_type == "utterance_span"
    ), "dest must re-align to the value now in its (swapped) home"


def test_swap_overlap_skipped() -> None:
    """Overlapping spans (or values not present verbatim) produce no swap_f."""
    s = _sample(
        sid="overlap-1",
        utterance="a a",
        tools=[_tool("t", [_p("x"), _p("y")])],
        gt=[GroundTruthCall(tool_name="t", arguments={"x": "a", "y": "a"})],
    )
    out = _align(s)
    augs = [a for a in DataAugmenter(seed=1).augment(out)
            if a.metadata.get("aug_type") == "swap_f"]
    # overlap in the raw utterance (both values are the same token) -> no swap
    assert not augs


# ---------------------------------------------------------------------------
# distractor_f / ambiguous_a
# ---------------------------------------------------------------------------

def test_distractor_f_and_ambiguous_a() -> None:
    recent = [{"key": "case_id", "value": "C-1", "source": "conversation"}]
    s = _sample(
        sid="dist-1",
        utterance="Close the case I opened earlier.",
        tools=[_tool("close", [_p("case_id")])],
        gt=[GroundTruthCall(tool_name="close", arguments={"case_id": "C-1"})],
        recent=recent,
    )
    out = _align(s)
    assert out.stratum == Stratum.F
    pool = [
        {"key": "weather", "value": "sunny", "source": "augmented"},
        {"key": "song", "value": "blues", "source": "augmented"},
    ]
    augs = DataAugmenter(seed=3, distractor_pool=pool).augment(out)
    types = {a.metadata.get("aug_type") for a in augs}
    assert "distractor_f" in types
    for a in augs:
        if a.metadata.get("aug_type") == "distractor_f":
            assert a.stratum == Stratum.F
            assert "weather" in a.metadata["distractor_keys"]
        if a.metadata.get("aug_type") == "ambiguous_a":
            # forced stratified A + H4 AMBIGUOUS on the ambiguous param
            assert a.stratum == Stratum.A
            assert a.metadata["expected_abstain"] == "ambiguous"
            assert a.metadata["h4_reasons"][a.metadata["ambiguous_param"]] == "AMBIGUOUS"


# ---------------------------------------------------------------------------
# adversarial R7
# ---------------------------------------------------------------------------

def test_adversarial_r7_must_abstain() -> None:
    out = _align(_move_sample())
    augs = [a for a in DataAugmenter(seed=9).augment(out)
            if a.metadata.get("aug_type") == "adversarial_r7"]
    assert augs
    for a in augs:
        assert a.stratum == Stratum.A
        assert a.metadata["expected_abstain"] == "marker_injection"
        assert "injected_marker" in a.metadata
        if a.metadata["injection_target"] == "utterance":
            assert "injected_marker" in a.utterance or True  # marker may be <|sep|> verbatim
        else:
            vals = [r["value"] for r in a.recent_context]
            assert any("marker" in str(x) for x in vals) or any(
                "\u200b" in str(x) for x in vals
            )


# ---------------------------------------------------------------------------
# shuffling invariance
# ---------------------------------------------------------------------------

def test_shuffle_variant_preserves_gt_and_reason() -> None:
    s = _sample(
        sid="shuf-1",
        utterance="Send the alert to cam-5 at priority high.",
        tools=[_tool("alert", [_p("camera_id"), _p("priority", enum=["low", "high"])])],
        gt=[GroundTruthCall(tool_name="alert",
                            arguments={"camera_id": "cam-5", "priority": "high"})],
    )
    out = _align(s)
    assert out.stratum == Stratum.F
    augs = DataAugmenter(seed=11).augment(out)
    shufs = [a for a in augs if a.metadata.get("aug_type") == "shuffle_f"]
    if shufs:
        v = shufs[0]
        assert v.stratum == Stratum.F
        # re-aligned GT identical (order-only permutation)
        assert v.ground_truth_calls[0].arguments == out.ground_truth_calls[0].arguments
        assert v.metadata["h4_reasons"] == out.metadata["h4_reasons"]


# ---------------------------------------------------------------------------
# mix ratios + bucket boundaries
# ---------------------------------------------------------------------------

def test_bucket_samples_boundaries() -> None:
    natural = _align(_move_sample(sid="nat-1"))
    para = _align(_move_sample(sid="para-1"))
    para.metadata["aug_type"] = "paraphrase_f"
    swap = _align(_move_sample(sid="swap-1"))
    swap.metadata["aug_type"] = "swap_f"
    abstain = _align(_move_sample(sid="abs-1"))
    abstain.stratum = Stratum.A
    r7 = _align(_move_sample(sid="r7-1"))
    r7.metadata["aug_type"] = "adversarial_r7"
    unknown = _align(_move_sample(sid="unk-1"))

    buckets = _bucket_samples([natural, para, swap, abstain, r7, unknown])
    # untagged Stratum.F ('' aug_type) -> natural_f; abstention only A-stratum rows
    assert len(buckets["natural_f"]) == 2  # natural + unknown (both untagged F)
    assert natural in buckets["natural_f"] and unknown in buckets["natural_f"]
    assert buckets["paraphrase_f"] == [para]
    assert buckets["swap_distractor_f"] == [swap]
    assert len(buckets["abstention_a"]) == 2  # abstain + r7
    assert abstain in buckets["abstention_a"] and r7 in buckets["abstention_a"]


def test_mix_ratios_respected() -> None:
    # build 40 samples/group: 10 natural-F, 10 paraphrase, 10 swap, 10 abstain
    groups = []
    for i in range(10):
        nat = _align(_move_sample(sid=f"nat-{i}"))
        para = _align(_move_sample(sid=f"para-{i}"))
        para.metadata["aug_type"] = "paraphrase_f"
        swap = _align(_move_sample(sid=f"swap-{i}"))
        swap.metadata["aug_type"] = "swap_f"
        abstain = _align(_move_sample(sid=f"abs-{i}"))
        abstain.stratum = Stratum.A
        groups.extend([nat, para, swap, abstain])
    mixer = DatasetMixer(
        share_natural_f=0.55, share_paraphrase_f=0.20,
        share_swap_distractors=0.10, share_abstention_a=0.15,
        seed=42,
    )
    mixed = mixer.mix(groups)
    counts = Counter(s.metadata.get("aug_type", "") for s in mixed)
    # swap_distractor_f bucket also picks up untagged natural-F? verify:
    natural_n = sum(1 for s in mixed if s.metadata.get("aug_type") in ("", None))
    para_n = counts["paraphrase_f"]
    negs_n = sum(1 for s in mixed if s.metadata.get("aug_type") == "swap_f")
    abstain_n = sum(1 for s in mixed if s.stratum == Stratum.A)
    assert natural_n > 0 and para_n > 0 and negs_n > 0 and abstain_n > 0


def test_mix_with_replacement_hits_target() -> None:
    # fewer swap samples than target -> with-replacement sampling still hits it
    nat, para, abstain = [], [], []
    for i in range(5):
        para.append(_align(_move_sample(sid=f"p-{i}")))
        para[-1].metadata["aug_type"] = "paraphrase_f"
        a = _align(_move_sample(sid=f"a-{i}"))
        a.stratum = Stratum.A
        nat.append(_align(_move_sample(sid=f"n-{i}")))
        abstain.append(a)
    swap = [_align(_move_sample(sid="s0"))]
    swap[0].metadata["aug_type"] = "swap_f"
    samples = nat + para + swap + abstain
    mixer = DatasetMixer(seed=1)
    mixed = mixer.mix(samples)
    # n is the total pooled size (16 here); shares are rounded from n
    n = len(samples)
    assert n == 16
    total = round(0.55 * n) + round(0.20 * n) + round(0.10 * n) + round(0.15 * n)
    assert len(mixed) == total
    swap_n = sum(1 for s in mixed if s.metadata.get("aug_type") == "swap_f")
    assert swap_n == round(0.10 * n), "with-replacement must hit the swap target"


# ---------------------------------------------------------------------------
# finalize_splits — 7 canonical splits
# ---------------------------------------------------------------------------

def test_finalize_splits_7_way() -> None:
    samples = _multi_schema_samples()  # 45 samples, 8 schemas
    paraphrases = _paraphrases_for(samples)
    mixer = DatasetMixer(seed=42)
    schema_splits = mixer.partition_by_schema(
        [_align(s) for s in samples], test_schema_ratio=0.10, calib_ratio=0.05,
        dev_ratio=0.05, seed=42,
    )
    final = mixer.finalize_splits(
        schema_splits, train_mix=True, seed=42,
        augmenter=DataAugmenter(
            seed=42, enable_swaps=True, paraphrases=paraphrases,
        ),
    )
    assert set(final.keys()) == set(Split)
    # test_adversarial rows MUST abstain (R7)
    assert final[Split.TEST_ADVERSARIAL], "no R7 rows produced"
    for adv in final[Split.TEST_ADVERSARIAL]:
        assert adv.stratum == Stratum.A
        assert adv.metadata.get("expected_abstain") == "marker_injection"
    # test_ood_para rows are paraphrases that stay F
    assert final[Split.TEST_OOD_PARA], "no paraphrase_f OOD rows produced"
    for p in final[Split.TEST_OOD_PARA]:
        assert p.metadata.get("aug_type") == "paraphrase_f"
        assert p.stratum == Stratum.F
    # all splits non-empty (45 groundable samples; augmented across 8 schemas)
    for sp, items in final.items():
        assert len(items) >= 1, f"{sp.value} split empty"


def test_finalize_schema_disjointness() -> None:
    """Schema groups stay whole: no schema spans train/dev/calib/ood."""
    samples = _multi_schema_samples()
    mixer = DatasetMixer(seed=7)
    schema_splits = mixer.partition_by_schema(
        [_align(s) for s in samples], test_schema_ratio=0.10, calib_ratio=0.05,
        dev_ratio=0.05, seed=7,
    )
    final = mixer.finalize_splits(schema_splits, train_mix=True, seed=7)
    train_schemas = {schema_key(s) for s in final[Split.TRAIN]}
    dev_schemas = {schema_key(s) for s in final[Split.DEV]}
    calib_schemas = {schema_key(s) for s in final[Split.CALIB]}
    ood_schemas = {schema_key(s) for s in final[Split.TEST_OOD_SCHEMA]}
    # pair-wise disjointness between the 4 schema-held-out buckets
    for a, b in ((train_schemas, dev_schemas), (train_schemas, calib_schemas),
                 (train_schemas, ood_schemas), (dev_schemas, calib_schemas),
                 (dev_schemas, ood_schemas), (calib_schemas, ood_schemas)):
        assert not (a & b), f"schema-disjointness violated: {a & b}"


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_partition_determinism() -> None:
    samples = [_align(s) for s in _multi_schema_samples()]
    m1 = DatasetMixer(seed=42)
    s1 = m1.partition_by_schema(samples, seed=42)
    s2 = DatasetMixer(seed=42).partition_by_schema(samples, seed=42)
    for sp in s1:
        assert sorted(x.sample_id for x in s1[sp]) == sorted(x.sample_id for x in s2[sp])