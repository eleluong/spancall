"""Unit tests for the Salesforce xLAM-60k loader and gt_tool partitioning."""

from __future__ import annotations

import json
from pathlib import Path

from spancall.data.align import ReverseAligner
from spancall.data.partition import partition_schema_groups
from spancall.data.schema import Split, Stratum
from spancall.data.sources.xlam import XLAMLoader, map_xlam_type

SINGLE_CALL_ROW = {
    "id": 1,
    "query": "Can you fetch the details for 'ethereum' on the blockchain?",
    "tools": json.dumps([
        {
            "name": "peers",
            "description": "Retrieves company peers.",
            "parameters": {
                "symbol": {"description": "Stock symbol", "type": "str", "default": ""},
            },
        },
        {
            "name": "web_chain_details",
            "description": "Fetch blockchain details.",
            "parameters": {
                "chain_slug": {"description": "Blockchain slug", "type": "str"},
                "include_stats": {"description": "Include stats", "type": "bool, optional"},
                "limit": {"description": "Max items", "type": "Optional[int]"},
            },
        },
    ]),
    "answers": json.dumps([
        {"name": "web_chain_details", "arguments": {"chain_slug": "ethereum"}},
    ]),
}

MULTI_CALL_ROW = {
    "id": 0,
    "query": "Where can I find live giveaways for beta access and games?",
    "tools": json.dumps([
        {
            "name": "live_giveaways_by_type",
            "description": "Retrieve live giveaways.",
            "parameters": {
                "type": {"description": "Giveaway type", "type": "str", "default": "game"},
            },
        },
    ]),
    "answers": json.dumps([
        {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
        {"name": "live_giveaways_by_type", "arguments": {"type": "game"}},
    ]),
}


def test_map_xlam_type() -> None:
    assert map_xlam_type("str") == ("string", None, False)
    assert map_xlam_type("int, optional") == ("integer", None, True)
    assert map_xlam_type("Optional[float]") == ("number", None, True)
    assert map_xlam_type("bool, optional") == ("boolean", None, True)
    assert map_xlam_type("List[int]") == ("array", "integer", False)
    assert map_xlam_type("Tuple[float, float]") == ("array", "number", False)
    assert map_xlam_type("Dict") == ("object", None, False)
    assert map_xlam_type("Callable[[float], float]") == ("string", None, False)


def test_xlam_loader_single_call_only(tmp_path: Path) -> None:
    src = tmp_path / "xlam.json"
    src.write_text(json.dumps([SINGLE_CALL_ROW, MULTI_CALL_ROW]))

    loader = XLAMLoader(local_path=str(src), single_call_only=True)
    samples = list(loader.load())

    assert len(samples) == 1
    assert loader.stats["rows_seen"] == 2
    assert loader.stats["rows_yielded"] == 1
    assert loader.stats["dropped_multi_call"] == 1

    s = samples[0]
    assert s.sample_id == "xlam::1"
    assert s.source_dataset == "xlam-60k"
    assert s.metadata["config"] == "xlam"
    assert s.metadata["n_calls"] == 1
    assert [t.name for t in s.tools] == ["peers", "web_chain_details"]

    chain_tool = s.tools[1]
    by_name = {p.name: p for p in chain_tool.parameters}
    assert by_name["chain_slug"].required is True
    assert by_name["chain_slug"].param_type == "string"
    assert by_name["include_stats"].required is False
    assert by_name["include_stats"].param_type == "boolean"
    assert by_name["limit"].required is False
    assert by_name["limit"].param_type == "integer"

    peers_tool = s.tools[0]
    assert peers_tool.parameters[0].required is False
    assert peers_tool.parameters[0].default == ""


def test_xlam_loader_allows_multi_call_when_disabled(tmp_path: Path) -> None:
    src = tmp_path / "xlam.jsonl"
    src.write_text(json.dumps(SINGLE_CALL_ROW) + "\n" + json.dumps(MULTI_CALL_ROW) + "\n")

    loader = XLAMLoader(local_path=str(src), single_call_only=False)
    samples = list(loader.load())
    assert len(samples) == 2
    assert len(samples[1].ground_truth_calls) == 2


def test_xlam_aligner_grounds_single_call(tmp_path: Path) -> None:
    src = tmp_path / "xlam.json"
    src.write_text(json.dumps([SINGLE_CALL_ROW]))
    sample = next(XLAMLoader(local_path=str(src)).load())

    aligned = ReverseAligner().align(sample)
    assert aligned.stratum == Stratum.F
    assert aligned.target_spans["chain_slug"][0].canonical_value == "ethereum"


def test_partition_gt_tool_disjoint_across_shared_candidates(tmp_path: Path) -> None:
    rows = []
    for i, gt in enumerate(["tool_a", "tool_a", "tool_b", "tool_c", "tool_d"]):
        rows.append({
            "id": i,
            "query": f"Run {gt} with val_{i}",
            "tools": json.dumps([
                {"name": "shared_distractor", "description": "", "parameters": {}},
                {"name": gt, "description": "", "parameters": {"x": {"type": "str"}}},
            ]),
            "answers": json.dumps([{"name": gt, "arguments": {"x": f"val_{i}"}}]),
        })
    src = tmp_path / "xlam.json"
    src.write_text(json.dumps(rows))
    samples = list(XLAMLoader(local_path=str(src)).load())

    splits, report = partition_schema_groups(
        samples,
        {"train": 0.5, "dev": 0.25, "test_ood_schema": 0.25},
        seed=42,
        group_by="gt_tool",
    )
    assert report["schema_disjoint"] is True
    seen_gt = {}
    for sp, items in splits.items():
        for s in items:
            gt_name = s.ground_truth_calls[0].tool_name
            if gt_name in seen_gt:
                assert seen_gt[gt_name] == sp
            seen_gt[gt_name] = sp


def test_v2_6_notebook_includes_xlam_mix_and_eval() -> None:
    nb_path = Path(__file__).resolve().parents[2] / "notebooks" / "spancall_lean_v2_6_train_eval.ipynb"
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 26
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "summary_report.md" in all_src
    assert "XLAM_TRAIN_ROWS   = 15000" in all_src
    assert "XLAM_DEV_ROWS     = 1000" in all_src
    assert "SINGLE_CALL_ONLY  = True" in all_src
    assert '["hermes_train.jsonl", "xlam_train.jsonl"]' in all_src
    assert '["hermes_dev.jsonl", "xlam_dev.jsonl"]' in all_src
    assert '"xlam_eval"' in all_src
    assert '"xlam_test_ood_schema"' in all_src
    assert 'CONFIG_PATH = "configs/colab_lean_v2_6.yaml"' in all_src


