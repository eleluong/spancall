"""Unit tests for SpanCall Lean v2.4 bottleneck fixes (W1-W3)."""

from types import SimpleNamespace
import torch
from spancall.lean.decode import LeanJointDecoder
from spancall.lean.loss import (
    LeanLossContainer,
    _lean_tool_binary_supervision_count,
    compute_lean_tool_loss_binary,
)
from spancall.model.masks import CandidateMaskBuilder


def _cand(val, cs, ce, region="utterance"):
    return SimpleNamespace(
        canon_value=val,
        char_start=cs,
        char_end=ce,
        region=region,
        entry_index=-1,
    )


def test_masks_utterance_survives_huge_recent_and_filters_json_punct():
    """Long <|recent|> history must not starve <|utt|> spans out of max_cands=64,
    and JSON syntax fragments in recent_context must be filtered."""
    builder = CandidateMaskBuilder(max_seq_len=512, max_span_width=8)
    builder.pack_fill = "stratified"

    # Simulate a long JSON-like recent entry followed by a short utterance.
    recent_str = 'action {"cuisine": "Italian", "location": "New York", "notes": "alpha beta gamma delta epsilon"}'
    utt_str = "Book a table in San Francisco tomorrow"
    text = f"{recent_str} | {utt_str}"
    r_start, r_end = 0, len(recent_str)
    u_start = len(recent_str) + 3
    u_end = len(text)

    # Whitespace-level token offsets
    offsets = []
    pos = 0
    for tok in text.split(" "):
        idx = text.index(tok, pos)
        offsets.append((idx, idx + len(tok)))
        pos = idx + len(tok)

    entries = [
        SimpleNamespace(segment="recent", kind="recent_val", value=recent_str, text=recent_str, char_start=r_start, char_end=r_end, selectable=True),
        SimpleNamespace(segment="utterance", kind="utt_text", value=utt_str, text=utt_str, char_start=u_start, char_end=u_end, selectable=True),
    ]
    param = SimpleNamespace(name="city", param_type="string", required=True, enum=None)
    res = builder.build_param_masks(param, entries, offsets, text, max_cands=64)

    regions = res["region"]
    vals = res["canon_value"]
    assert "utterance" in regions, "Utterance candidates were starved by recent_context"
    # 7-word utterance has 7*8/2 = 28 spans total — all 28 must survive!
    assert regions.count("utterance") == 28, f"Expected all 28 utterance slots, got {regions.count('utterance')}"
    assert "San Francisco" in vals
    for reg, v in zip(regions, vals):
        if reg == "recent" and v != recent_str:
            assert not any(ch in str(v) for ch in ('"', "{", "}", "[", "]", "\\")), (
                f"JSON syntax fragment leaked into recent candidates: {v!r}"
            )


def test_decoder_prevents_cross_param_exact_span_collapse():
    """Sibling parameters in one call must not collapse onto the same span,
    even across different entity classes or recent_context."""
    decoder = LeanJointDecoder(top_k=5, free_string_overlap_allowed=False, normalize_scores=False)

    # Case 1: distinct entity classes (title vs year) where both rank "2010" #1
    scores_movie = {
        "title": [(_cand("2010", 30, 34), 0.9), (_cand("Inception", 10, 19), 0.8)],
        "year": [(_cand(2010, 30, 34), 0.95), (_cand(2009, 40, 44), 0.1)],
    }
    assign, _delta, _chosen, _n = decoder.decode_detailed(
        scores_movie, entity_classes={"title": "title", "year": "year"}
    )
    assert assign["year"] == 2010
    assert assign["title"] == "Inception"

    # Case 2: recent region free strings (side1 vs side2)
    scores_recent = {
        "side1": [(_cand(4.0, 5, 6, region="recent"), 0.9), (_cand(3.0, 2, 3, region="recent"), 0.7)],
        "side2": [(_cand(4.0, 5, 6, region="recent"), 0.95), (_cand(3.0, 2, 3, region="recent"), 0.6)],
    }
    assign_r, _, _, _ = decoder.decode_detailed(scores_recent)
    assert assign_r["side1"] != assign_r["side2"]
    assert {assign_r["side1"], assign_r["side2"]} == {3.0, 4.0}


def test_binary_tool_loss_supervises_stratum_a_abstain_rows():
    """abstain_mask=True rows must receive all-zero BCE supervision and count
    toward the micro-average denominator."""
    logits = torch.tensor([[2.0, -2.0], [2.0, 2.0]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0, -1], dtype=torch.long)
    abstain_mask = torch.tensor([False, True], dtype=torch.bool)

    loss_without = compute_lean_tool_loss_binary(logits, targets, abstain_mask=None)
    loss_with = compute_lean_tool_loss_binary(logits, targets, abstain_mask=abstain_mask)
    assert loss_without is not None and loss_with is not None
    # Row 1 has high positive logits (+2.0) against all-zero target, so loss_with > loss_without
    assert float(loss_with.item()) > float(loss_without.item())

    cnt = _lean_tool_binary_supervision_count(logits, targets, abstain_mask=abstain_mask)
    assert int(cnt.item()) == 2

    container = LeanLossContainer(lambda_pointer=1.0, lambda_tool_binary=1.0, tool_objective="binary")
    out = container(
        {"tool_binary_logits": logits},
        {"tool_targets": targets, "abstain_mask": abstain_mask},
    )
    assert int(out["tool_binary_count"].item()) == 2
    out["loss"].backward()
    # Gradient on row 1 (abstain) should push both logits DOWN (positive grad on logits)
    assert (logits.grad[1] > 0).all()


def test_tensorize_abstain_sample_packs_distractors_and_trainer_retains(tmp_path):
    """Stratum-A rows tensorized with include_abstain=True must pack distractors
    from tool_pool and survive LeanTrainer.load_rows when use_binary_tool_head=True."""
    import json
    from spancall.data.schema import ParameterSchema, ToolDefinition, UnifiedSample
    from spancall.lean.tensorize import ToolPool, tensorize_sample_lean
    from spancall.lean.trainer import LeanTrainer

    class _CharTokenizer:
        def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
            words = text.split(" ")
            ids, offsets = [], []
            pos = 0
            for idx, w in enumerate(words):
                start = text.index(w, pos)
                end = start + len(w)
                ids.append((idx % 90) + 5)
                offsets.append((start, end))
                pos = end
            return {"input_ids": ids, "offset_mapping": offsets}

    t1 = ToolDefinition(
        name="get_weather",
        description="Get weather",
        parameters=[ParameterSchema(name="city", description="City", param_type="string", required=True)],
    )
    t2 = ToolDefinition(
        name="book_flight",
        description="Book flight",
        parameters=[ParameterSchema(name="dest", description="Destination", param_type="string", required=True)],
    )
    pool = ToolPool([t1, t2], seed=42)
    sample = UnifiedSample(
        sample_id="abstain-1",
        source_dataset="hermes",
        utterance="Just saying hello today",
        tools=[t1],
        ground_truth_calls=[],
    )
    stats = {}
    row = tensorize_sample_lean(
        sample=sample,
        tokenizer=_CharTokenizer(),
        mask_builder=None,
        max_seq_len=512,
        tool_pool=pool,
        pack_tools=2,
        jitter=False,
        include_abstain=True,
        stats=stats,
    )
    assert row is not None
    assert int(row["is_abstain"].item()) == 1
    assert int(row["tool_targets"].item()) == -1
    assert int(row["_n_tool_candidates"]) == 2
    assert stats.get("abstain_rows") == 1

    # Verify LeanTrainer.load_rows keeps is_abstain=1 only when use_binary_tool_head=True
    p = tmp_path / "abstain.jsonl"
    serial = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in row.items()
    }
    p.write_text(json.dumps(serial) + "\n")

    dummy_trainer_bin = SimpleNamespace(
        config=SimpleNamespace(data_dir=str(tmp_path), data=None, seed=42),
        use_binary_tool_head=True,
    )
    dummy_trainer_ce = SimpleNamespace(
        config=SimpleNamespace(data_dir=str(tmp_path), data=None, seed=42),
        use_binary_tool_head=False,
    )
    rows_bin = LeanTrainer.load_rows(dummy_trainer_bin, "abstain")
    rows_ce = LeanTrainer.load_rows(dummy_trainer_ce, "abstain")
    assert len(rows_bin) == 1
    assert len(rows_ce) == 0

