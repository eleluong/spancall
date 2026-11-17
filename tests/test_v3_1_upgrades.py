"""Unit and parity tests for SpanCall Lean v3.1 upgrades."""
from __future__ import annotations

from datetime import date

import torch
from datetime import date

import torch.nn as nn

from span_template.template import build_input_multi, tokenize_offset_mapping
from spancall.data.align import ReverseAligner, _infer_schema_from_arguments, tooldef_to_template_tool
from spancall.data.flatten import iter_balanced_spans, safe_parse_literal
from spancall.data.schema import GroundTruthCall, ParameterSchema, Stratum, ToolDefinition, UnifiedSample
from spancall.infer.cast import StrictTypeCaster
from spancall.lean.decode import LeanJointDecoder
from spancall.lean.loss import LeanLossContainer
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver
from spancall.model.masks import CandidateMaskBuilder
from spancall.train.config import OptimizationConfig


# ---------------------------------------------------------------------------
# 1. Array Mechanism Upgrades
# ---------------------------------------------------------------------------

def test_array_mask_emits_bracket_and_numeric_candidates() -> None:
    tool = ToolDefinition(
        name="matrix_multiply",
        description="Multiply matrices",
        parameters=[
            ParameterSchema(name="matrix_a", description="", param_type="array", required=True, items_type="number"),
            ParameterSchema(name="numbers", description="", param_type="array", required=False, items_type="number"),
        ],
    )
    utt = "Multiply [[1, 2], [3, 4]] and numbers [2.1, 3.4, 4.6]."
    out = build_input_multi([tooldef_to_template_tool(tool)], utt, now=date(2026, 1, 1))

    # Simple char-level tokenizer stub
    offsets = [(i, i + 1) for i in range(len(out.text))]
    builder = CandidateMaskBuilder(max_seq_len=len(offsets), max_span_width=16)

    m_mat = builder.build_param_masks(tool.parameters[0], out.entries, offsets, out.text)
    vals_mat = [c.canon_value for c in m_mat["candidates"]]
    assert "[[1, 2], [3, 4]]" in vals_mat
    assert "[2.1, 3.4, 4.6]" in vals_mat
    assert "2.1" in vals_mat

    m_num = builder.build_param_masks(tool.parameters[1], out.entries, offsets, out.text)
    assert any(c.region == "null" for c in m_num["candidates"])


def test_array_align_nested_and_empty_lists() -> None:
    tool = ToolDefinition(
        name="matrix_multiply",
        description="Multiply matrices",
        parameters=[
            ParameterSchema(name="matrix_a", description="", param_type="array", required=True, items_type="number"),
            ParameterSchema(name="tags", description="", param_type="array", required=False, items_type="string"),
        ],
    )
    sample = UnifiedSample(
        sample_id="xlam::mat1",
        source_dataset="xlam-60k",
        utterance="Multiply [[1, 2], [3, 4]].",
        tools=[tool],
        ground_truth_calls=[GroundTruthCall(tool_name="matrix_multiply", arguments={"matrix_a": [[1, 2], [3, 4]], "tags": []})],
    )
    aligned = ReverseAligner().align(sample)
    assert aligned.stratum == Stratum.F
    mat_targs = aligned.targets_for_call(0)["matrix_a"]
    assert len(mat_targs) == 1
    assert mat_targs[0].target_type == "utterance_span"
    assert mat_targs[0].canonical_value == [[1, 2], [3, 4]]
    assert aligned.targets_for_call(0)["tags"][0].target_type == "null_ctx"


def test_select_list_choices_doc_order_and_no_container_leak() -> None:
    decoder = LeanJointDecoder(normalize_scores=False)

    def _cand(val, cs, ce):
        return type("C", (), {"canon_value": val, "char_start": cs, "char_end": ce, "region": "utterance", "entry_index": -1})()

    # broccoli has higher score than chicken, but chicken appears earlier in utterance
    entries = [
        (_cand("broccoli", 20, 28), 0.35),
        (_cand("chicken", 10, 17), 0.30),
        (_cand("Chicken Tomato Garlic", 10, 38), 0.28),  # overlaps chicken & broccoli -> must be pruned
        (_cand("cheese", 30, 36), 0.25),
    ]
    sel = decoder.select_list_choices(entries)
    assert [c.value for c in sel] == ["chicken", "broccoli", "cheese"]

    # Whole-list bracketed literal winner must not union extra elements
    lit_entries = [
        (_cand("[[1, 2], [3, 4]]", 10, 26), 0.80),
        (_cand("1", 12, 13), 0.25),
    ]
    sel_lit = decoder.select_list_choices(lit_entries)
    assert [c.value for c in sel_lit] == ["[[1, 2], [3, 4]]"]

    # Supports > 8 elements (up to 32)
    many = [(_cand(f"item_{i}", i * 10, i * 10 + 6), 0.08) for i in range(12)]
    sel_many = decoder.select_list_choices(many)
    assert len(sel_many) == 12


def test_cast_array_nested_and_quoted_literals() -> None:
    caster = StrictTypeCaster()
    p_mat = ParameterSchema(name="m", description="", param_type="array", items_type="number")
    ok, val = caster.cast_parameter("[[1, 2], [3, 4]]", p_mat)
    assert ok and val == [[1, 2], [3, 4]]

    p_str = ParameterSchema(name="items", description="", param_type="array", items_type="string")
    ok2, val2 = caster.cast_parameter('["a, b", "c"]', p_str)
    assert ok2 and val2 == ["a, b", "c"]


# ---------------------------------------------------------------------------
# 2. Object Mechanism Upgrades
# ---------------------------------------------------------------------------

def test_json_mode_infers_dotted_object_leaves() -> None:
    gt = GroundTruthCall(
        tool_name="json_output",
        arguments={"filters": {"color": "red", "size": 10}, "query": "shoes"},
    )
    tool = ToolDefinition(name="json_output", description="", parameters=[])
    new_tool, changed = _infer_schema_from_arguments(gt, tool)
    assert changed
    names = [p.name for p in new_tool.parameters]
    assert names == ["filters.color", "filters.size", "query"]
    assert new_tool.object_paths == {"filters": ["filters.color", "filters.size"]}


def test_literal_object_align_mask_and_cast() -> None:
    tool = ToolDefinition(
        name="merge_dictionaries",
        description="Merge two dicts",
        parameters=[
            ParameterSchema(name="dict1", description="", param_type="object", required=True),
            ParameterSchema(name="dict2", description="", param_type="object", required=True),
        ],
    )
    utt = "Merge {'a': 1, 'b': 2} and {'b': 3, 'c': 4}."
    sample = UnifiedSample(
        sample_id="xlam::m1",
        source_dataset="xlam-60k",
        utterance=utt,
        tools=[tool],
        ground_truth_calls=[GroundTruthCall(tool_name="merge_dictionaries", arguments={"dict1": {"a": 1, "b": 2}, "dict2": {"b": 3, "c": 4}})],
    )
    aligned = ReverseAligner().align(sample)
    assert aligned.stratum == Stratum.F
    targs = aligned.targets_for_call(0)
    assert targs["dict1"][0].target_type == "utterance_span"
    assert targs["dict1"][0].canonical_value == {"a": 1, "b": 2}
    assert targs["dict2"][0].utt_rel_start is not None

    out = build_input_multi([tooldef_to_template_tool(tool)], utt, now=date(2026, 1, 1))
    offsets = [(i, i + 1) for i in range(len(out.text))]
    builder = CandidateMaskBuilder(max_seq_len=len(offsets), max_span_width=16)
    m = builder.build_param_masks(tool.parameters[0], out.entries, offsets, out.text)
    assert "{'a': 1, 'b': 2}" in [c.canon_value for c in m["candidates"]]

    caster = StrictTypeCaster()
    ok, parsed = caster.cast_parameter("{'a': 1, 'b': 2}", tool.parameters[0])
    assert ok and parsed == {"a": 1, "b": 2}


def test_upgrade_opaque_object_schemas_legacy_split() -> None:
    from scripts.realign_v2 import _upgrade_opaque_object_schemas

    tool = ToolDefinition(
        name="calc_loan",
        description="",
        parameters=[ParameterSchema(name="terms", description="", param_type="object", required=True)],
    )
    s1 = UnifiedSample(
        sample_id="h1",
        source_dataset="hermes",
        utterance="Loan 500000 for 30 years.",
        tools=[tool],
        ground_truth_calls=[GroundTruthCall(tool_name="calc_loan", arguments={"terms": {"amount": 500000, "years": 30}})],
    )
    n_upg = _upgrade_opaque_object_schemas([s1])
    assert n_upg == 1
    assert s1.tools[0].object_paths == {"terms": ["terms.amount", "terms.years"]}
    aligned = ReverseAligner().align(s1)
    assert aligned.stratum == Stratum.F


# ---------------------------------------------------------------------------
# 3. Loss Re-Weighting
# ---------------------------------------------------------------------------

def test_loss_defaults_and_focal_multi_pos_weighting() -> None:
    opt = OptimizationConfig()
    assert opt.lambda_pointer == 2.0
    assert opt.lambda_tool == 0.2
    assert opt.pointer_focal_gamma == 1.0
    assert opt.multi_pos_slot_weight == 1.5

    logits = torch.tensor([[[5.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])  # slot 0 easy, slot 1 hard + multi-pos
    pos = torch.tensor([[[0, -1], [0, 1]]])
    base_loss = LeanLossContainer(lambda_pointer=1.0, lambda_tool=1.0)
    v31_loss = LeanLossContainer(
        lambda_pointer=2.0,
        lambda_tool=0.2,
        pointer_focal_gamma=1.0,
        multi_pos_slot_weight=1.5,
    )
    out_base = base_loss({"span_logits": logits}, {"pointer_positives": pos})
    out_v31 = v31_loss({"span_logits": logits}, {"pointer_positives": pos})
    # Logged raw CE stays identical for cross-version comparison
    assert torch.allclose(out_base["pointer"], out_v31["pointer"])
    assert torch.allclose(out_base["pointer_sum"], out_v31["pointer_sum"])
    assert out_v31["loss"] != out_base["loss"]


# ---------------------------------------------------------------------------
# 4. Batched / Parallel Evaluation Parity
# ---------------------------------------------------------------------------

class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, return_offsets_mapping: bool = False, add_special_tokens: bool = False):
        ids = [ord(ch) % 127 + 1 for ch in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


class _TinyEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(256, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.emb(input_ids))
        return h * attention_mask.unsqueeze(-1).float()


def test_batched_eval_matches_sequential_resolver() -> None:
    torch.manual_seed(0)
    enc = _TinyEncoder(hidden_dim=16)
    model = LeanSpanCallModel(encoder=enc, head_dim=8, max_span_width=8, hidden_dim=16)
    tok = _TinyTokenizer()
    resolver = LeanResolver(model=model, encoder=enc, tokenizer=tok, max_seq_len=512, max_span_width=8, resolve_mode="two_step")

    t1 = ToolDefinition(name="get_weather", description="Weather", parameters=[ParameterSchema(name="city", description="", param_type="string", required=True)])
    t2 = ToolDefinition(name="book_flight", description="Flight", parameters=[ParameterSchema(name="dest", description="", param_type="string", required=True)])

    samples = [
        ([t1, t2], "Check weather in Tokyo please", None, None),
        ([t1, t2], "Book a flight to Paris tomorrow", None, None),
        ([t1], "Weather in London", None, None),
    ]

    seq_tools = [resolver.predict_tool(*it) for it in samples]
    bat_tools = resolver.predict_tools_batch(samples, batch_size=2, num_workers=2)
    for st, bt in zip(seq_tools, bat_tools):
        assert st["tool_name"] == bt["tool_name"]
        assert abs(st["tool_prob"] - bt["tool_prob"]) < 1e-5

    cand_items = [
        {"candidate_schemas": c, "utterance": u, "recent_context": r, "now": n}
        for c, u, r, n in samples
    ]
    seq_res = [resolver.try_resolve_candidates(c, u, r, n) for c, u, r, n in samples]
    bat_res = resolver.resolve_candidates_batch(cand_items, batch_size=2, num_workers=2)
    for sr, br in zip(seq_res, bat_res):
        assert sr.tool_name == br.tool_name
        assert sr.payload == br.payload
        assert sr.hit == br.hit

    oracle_items = [
        {"schema": t1, "utterance": "Weather in London", "recent_context": None, "now": None, "context_schemas": [t1]},
        {"schema": t2, "utterance": "Flight to Paris", "recent_context": None, "now": None, "context_schemas": [t2]},
    ]
    seq_or = [resolver.try_resolve(it["schema"], it["utterance"], context_schemas=it["context_schemas"]) for it in oracle_items]
    bat_or = resolver.resolve_oracle_batch(oracle_items, batch_size=2, num_workers=2)
    for so, bo in zip(seq_or, bat_or):
        assert so.tool_name == bo.tool_name
        assert so.payload == bo.payload
