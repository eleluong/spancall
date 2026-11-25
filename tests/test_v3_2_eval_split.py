"""Tests for Normal vs Complex evaluation partitioning."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    SpanTarget,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.lean.eval_p1 import (
    classify_sample_complexity,
    eval_param_extraction_lean,
    is_complex_param,
)


def test_is_complex_param() -> None:
    assert not is_complex_param(ParameterSchema(name="city", description="", param_type="string"))
    assert not is_complex_param(ParameterSchema(name="count", description="", param_type="integer"))
    assert not is_complex_param(ParameterSchema(name="flag", description="", param_type="boolean"))
    assert not is_complex_param(ParameterSchema(name="opt", description="", param_type=["string", "null"]))
    assert is_complex_param(ParameterSchema(name="tags", description="", param_type="array", items_type="string"))
    assert is_complex_param(ParameterSchema(name="cfg", description="", param_type="object"))
    assert is_complex_param({"name": "filters.color", "param_type": "string"}, {"filters": ["filters.color"]})


def test_classify_sample_complexity_dict_and_sample() -> None:
    normal_dict = {
        "sample_id": "s1",
        "utterance": "weather in Paris",
        "tools": [
            {
                "name": "get_weather",
                "description": "",
                "parameters": [{"name": "city", "description": "", "param_type": "string", "required": True}],
            }
        ],
        "ground_truth_calls": [{"tool_name": "get_weather", "arguments": {"city": "Paris"}}],
    }
    complex_array_dict = {
        "sample_id": "s2",
        "utterance": "search recipes with chicken and garlic",
        "tools": [
            {
                "name": "search_recipes",
                "description": "",
                "parameters": [{"name": "ingredients", "description": "", "param_type": "array", "required": True}],
            }
        ],
        "ground_truth_calls": [{"tool_name": "search_recipes", "arguments": {"ingredients": ["chicken", "garlic"]}}],
    }
    complex_obj_dict = {
        "sample_id": "s3",
        "utterance": "filter red",
        "tools": [
            {
                "name": "filter_items",
                "description": "",
                "parameters": [{"name": "filters.color", "description": "", "param_type": "string", "required": True}],
                "object_paths": {"filters": ["filters.color"]},
            }
        ],
        "ground_truth_calls": [{"tool_name": "filter_items", "arguments": {"filters": {"color": "red"}}}],
    }
    assert classify_sample_complexity(normal_dict) == "normal"
    assert classify_sample_complexity(complex_array_dict) == "complex"
    assert classify_sample_complexity(complex_obj_dict) == "complex"

    s_norm = UnifiedSample.from_dict(normal_dict)
    s_comp = UnifiedSample.from_dict(complex_array_dict)
    assert classify_sample_complexity(s_norm) == "normal"
    assert classify_sample_complexity(s_comp) == "complex"


def test_eval_param_extraction_lean_by_complexity(tmp_path: Path) -> None:
    split_path = tmp_path / "mini_eval.jsonl"
    rows = [
        {
            "sample_id": "norm1",
            "utterance": "get weather in Paris",
            "stratum": "F",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "",
                    "parameters": [{"name": "city", "description": "", "param_type": "string", "required": True}],
                }
            ],
            "ground_truth_calls": [{"tool_name": "get_weather", "arguments": {"city": "Paris"}}],
            "target_spans": {
                "city": [{"param_name": "city", "target_type": "utterance_span", "char_start": 15, "char_end": 20, "canonical_value": "Paris"}]
            },
        },
        {
            "sample_id": "comp1",
            "utterance": "tags a and b",
            "stratum": "F",
            "tools": [
                {
                    "name": "tag_items",
                    "description": "",
                    "parameters": [{"name": "tags", "description": "", "param_type": "array", "items_type": "string", "required": True}],
                }
            ],
            "ground_truth_calls": [{"tool_name": "tag_items", "arguments": {"tags": ["a", "b"]}}],
            "target_spans": {
                "tags": [{"param_name": "tags", "target_type": "utterance_span", "char_start": 5, "char_end": 6, "canonical_value": "a"}]
            },
        },
    ]
    split_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    class _StubResolver:
        supports_candidate_parity = False
        supports_context_schemas = False

        def try_resolve(self, tool, utterance, recent_context, current_time_iso):
            if tool.name == "get_weather":
                params = {"city": SimpleNamespace(value="Paris", source_type="utterance", confidence=0.9)}
                return SimpleNamespace(hit=True, parameters=params, calls=[], features={})
            params = {"tags": SimpleNamespace(value=["a"], source_type="utterance", confidence=0.5)}
            return SimpleNamespace(hit=False, parameters=params, calls=[], features={})

    m = eval_param_extraction_lean(str(split_path), resolver=_StubResolver(), data_dir=str(tmp_path), quiet=True)
    assert "by_complexity" in m
    norm = m["by_complexity"]["normal"]
    comp = m["by_complexity"]["complex"]
    assert norm["stratum_f_n"] == 1
    assert norm["exact_payload_match_F"] == 1.0
    assert norm["extraction_f1"] == 1.0
    assert comp["stratum_f_n"] == 1
    assert comp["exact_payload_match_F"] == 0.0
    assert comp["tp"] == 0
    assert comp["wrong"] == 1


def test_eval_param_extraction_lean_decomposed_object_leaves(tmp_path: Path) -> None:
    split_path = tmp_path / "obj_eval.jsonl"
    row = {
        "sample_id": "obj1",
        "utterance": "filter color red",
        "stratum": "F",
        "tools": [
            {
                "name": "filter_items",
                "description": "",
                "parameters": [{"name": "filters.color", "description": "", "param_type": "string", "required": True}],
                "object_paths": {"filters": ["filters.color"]},
            }
        ],
        "ground_truth_calls": [{"tool_name": "filter_items", "arguments": {"filters": {"color": "red"}}}],
        "target_spans": {
            "filters.color": [{"param_name": "filters.color", "target_type": "utterance_span", "char_start": 13, "char_end": 16, "canonical_value": "red"}]
        },
    }
    split_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    class _ObjResolver:
        supports_candidate_parity = False
        supports_context_schemas = False

        def try_resolve(self, tool, utterance, recent_context, current_time_iso):
            params = {"filters.color": SimpleNamespace(value="red", source_type="utterance", confidence=0.95)}
            return SimpleNamespace(hit=True, parameters=params, calls=[], features={})

    m = eval_param_extraction_lean(str(split_path), resolver=_ObjResolver(), data_dir=str(tmp_path), quiet=True)
    comp = m["by_complexity"]["complex"]
    assert comp["total_gt_params"] == 1
    assert comp["groundable_gt_params"] == 1
    assert comp["tp"] == 1
    assert comp["extraction_f1"] == 1.0
    assert comp["exact_payload_match_F"] == 1.0



