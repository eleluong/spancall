"""End-to-end tests for Lean v2.7 object-param decomposition (Steps 2-5).

Covers:
1. Hermes & xLAM schema parsing for nested ``properties`` (depth <= 3 vs > 3,
   free-form objects, required inheritance, ``object_paths``).
2. ``ReverseAligner`` flattening nested GT arguments and emitting dotted leaf
   targets, plus Case G (``OBJECT_UNDECOMPOSED``) for opaque objects.
3. ``span_template.template.gate_payload`` multi-tool ``::`` namespacing with
   dotted leaf names.
4. ``LeanResolver`` reassembling dotted leaves back into nested dicts after
   casting and schema validation.
"""
from __future__ import annotations

import json
import unittest
from datetime import date

from span_template.template import Param, Tool, build_input_multi, gate_tool_call
from spancall.data.align import ReverseAligner
from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.data.sources.hermes import _tool_from_json_schema
from spancall.data.sources.xlam import _parse_xlam_tools
from spancall.infer.base import FakeTokenizer
from spancall.lean.resolver import LeanResolver


class TestSchemaObjectDecomposition(unittest.TestCase):
    def test_hermes_schema_decomposes_nested_properties(self):
        params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string", "enum": ["red", "blue"]},
                        "size": {"type": "integer", "default": 10},
                        "nested": {
                            "type": "object",
                            "properties": {
                                "flag": {"type": "boolean"},
                            },
                            "required": ["flag"],
                        },
                    },
                    "required": ["color", "nested"],
                },
                "freeform": {"type": "object"},
            },
            "required": ["query", "filters"],
        }
        tool = _tool_from_json_schema("search", "Search items", params)
        by_name = {p.name: p for p in tool.parameters}

        self.assertIn("query", by_name)
        self.assertIn("filters.color", by_name)
        self.assertIn("filters.size", by_name)
        self.assertIn("filters.nested.flag", by_name)
        self.assertIn("freeform", by_name)
        self.assertNotIn("filters", by_name)

        self.assertTrue(by_name["filters.color"].required)
        self.assertEqual(by_name["filters.color"].enum, ["red", "blue"])
        self.assertFalse(by_name["filters.size"].required)
        self.assertEqual(by_name["filters.size"].default, 10)
        self.assertTrue(by_name["filters.nested.flag"].required)
        self.assertEqual(by_name["filters.nested.flag"].param_type, "boolean")
        self.assertEqual(by_name["freeform"].param_type, "object")

        self.assertEqual(
            tool.object_paths,
            {"filters": ["filters.color", "filters.nested.flag", "filters.size"]},
        )

    def test_hermes_schema_caps_depth_at_3(self):
        params = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "properties": {
                                "c": {
                                    "type": "object",
                                    "properties": {
                                        "d": {"type": "string"},
                                    },
                                },
                            },
                        }
                    },
                }
            },
            "required": ["a"],
        }
        tool = _tool_from_json_schema("deep_tool", "Deep", params)
        by_name = {p.name: p for p in tool.parameters}
        # Depth 3 ("a.b.c") is the cap, so a.b.c stays an opaque object rather than a.b.c.d
        self.assertIn("a.b.c", by_name)
        self.assertEqual(by_name["a.b.c"].param_type, "object")
        self.assertNotIn("a.b.c.d", by_name)

    def test_xlam_schema_decomposes_nested_properties(self):
        raw_tools = json.dumps([
            {
                "name": "filter_products",
                "description": "Filter products",
                "parameters": {
                    "filters": {
                        "type": "dict",
                        "properties": {
                            "brand": {"type": "str"},
                            "max_price": {"type": "int, optional"},
                        },
                    },
                    "raw_meta": {"type": "dict, optional"},
                },
            }
        ])
        tools = _parse_xlam_tools(raw_tools)
        self.assertIsNotNone(tools)
        tool = tools[0]
        by_name = {p.name: p for p in tool.parameters}
        self.assertIn("filters.brand", by_name)
        self.assertIn("filters.max_price", by_name)
        self.assertIn("raw_meta", by_name)
        self.assertTrue(by_name["filters.brand"].required)
        self.assertFalse(by_name["filters.max_price"].required)
        self.assertEqual(by_name["filters.max_price"].param_type, "integer")
        self.assertEqual(by_name["raw_meta"].param_type, "object")
        self.assertEqual(
            tool.object_paths,
            {"filters": ["filters.brand", "filters.max_price"]},
        )


class TestAlignerAndTemplateObjectDecomposition(unittest.TestCase):
    def test_aligner_flattens_gt_and_grounds_object_leaves(self):
        tool = _tool_from_json_schema(
            "search",
            "Search",
            {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "properties": {
                            "color": {"type": "string"},
                            "size": {"type": "integer"},
                        },
                        "required": ["color", "size"],
                    }
                },
                "required": ["filters"],
            },
        )
        sample = UnifiedSample(
            sample_id="obj-align-1",
            source_dataset="test",
            utterance="Find red shoes in size 10",
            tools=[tool],
            ground_truth_calls=[
                GroundTruthCall(
                    tool_name="search",
                    arguments={"filters": {"color": "red", "size": 10}},
                )
            ],
        )
        aligned = ReverseAligner().align(sample)
        self.assertEqual(aligned.stratum, Stratum.F)
        self.assertIn("filters.color", aligned.target_spans)
        self.assertIn("filters.size", aligned.target_spans)
        self.assertEqual(aligned.target_spans["filters.color"][0].canonical_value, "red")
        self.assertEqual(aligned.target_spans["filters.size"][0].canonical_value, 10)
        self.assertEqual(aligned.metadata["h4_reasons"]["filters.color"], "OK")
        self.assertEqual(aligned.metadata["h4_reasons"]["filters.size"], "OK")

    def test_aligner_marks_opaque_object_as_object_undecomposed(self):
        tool = ToolDefinition(
            name="raw_call",
            description="Raw",
            parameters=[
                ParameterSchema(
                    name="payload",
                    description="Free-form",
                    param_type="object",
                    required=True,
                )
            ],
        )
        sample = UnifiedSample(
            sample_id="obj-opaque-1",
            source_dataset="test",
            utterance="Send payload",
            tools=[tool],
            ground_truth_calls=[
                GroundTruthCall(tool_name="raw_call", arguments={"payload": {"a": 1}})
            ],
        )
        aligned = ReverseAligner().align(sample)
        self.assertEqual(aligned.stratum, Stratum.A)
        self.assertEqual(aligned.metadata["h4_reasons"]["payload"], "OBJECT_UNDECOMPOSED")

    def test_gate_payload_uses_double_colon_in_multi_tool_pack(self):
        t1 = Tool(
            name="search",
            description="Search",
            params=[Param(name="filters.color", description="Color", kind="str", required=True)],
        )
        t2 = Tool(
            name="other",
            description="Other",
            params=[Param(name="filters.color", description="Color", kind="str", required=False)],
        )
        out_multi = build_input_multi([t1, t2], "red", now=date(2026, 11, 15))
        res = gate_tool_call(out_multi, lambda p: (p.name == "filters.color", "red"))
        self.assertTrue(res.hit)
        self.assertIn("search::filters.color", res.payload)
        self.assertIn("other::filters.color", res.payload)

        out_single = build_input_multi([t1], "red", now=date(2026, 11, 15))
        res_single = gate_tool_call(out_single, lambda p: (True, "red"))
        self.assertTrue(res_single.hit)
        self.assertIn("filters.color", res_single.payload)


class TestResolverObjectReassembly(unittest.TestCase):
    def test_resolver_reassembles_dotted_leaves_into_nested_dict(self):
        tool = _tool_from_json_schema(
            "search",
            "Search",
            {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "properties": {
                            "color": {"type": "string"},
                            "size": {"type": "integer"},
                        },
                        "required": ["color", "size"],
                    }
                },
                "required": ["filters"],
            },
        )
        tok = FakeTokenizer()
        resolver = LeanResolver(model=object(), encoder=None, tokenizer=tok, max_seq_len=512)
        resolver._encode = lambda ids: (None, None)
        resolver._score_parameters = lambda *a, **kw: (["filters.color", "filters.size"], {})
        out = build_input_multi(
            [
                Tool(
                    name=tool.name,
                    description=tool.description,
                    params=[
                        Param(name="filters.color", description="", kind="str", required=True),
                        Param(name="filters.size", description="", kind="int", required=True),
                    ],
                )
            ],
            "Find red shoes in size 10",
            now=date(2026, 11, 15),
        )
        # Stub model scores so "red" and "10" win for filters.color and filters.size
        from spancall.lean.decode import LeanValueChoice

        utt_entry = next(e for e in out.entries if e.segment == "utterance")
        red_s = utt_entry.char_start + utt_entry.value.index("red")
        red_e = red_s + 3
        ten_s = utt_entry.char_start + utt_entry.value.index("10")
        ten_e = ten_s + 2

        chosen = {
            "filters.color": LeanValueChoice(
                value="red",
                prob=0.95,
                span=(red_s, red_e),
                region="utterance",
            ),
            "filters.size": LeanValueChoice(
                value="10",
                prob=0.92,
                span=(ten_s, ten_e),
                region="utterance",
            ),
        }
        res = None
        orig_decode = resolver.decoder.decode_detailed

        def _fake_decode(*args, **kwargs):
            return {"filters.color": "red", "filters.size": "10"}, 0.5, chosen, 1

        resolver.decoder.decode_detailed = _fake_decode
        try:
            res = resolver.try_resolve(
                tool,
                "Find red shoes in size 10",
                now=date(2026, 11, 15),
            )
        finally:
            resolver.decoder.decode_detailed = orig_decode
        self.assertTrue(res.hit)
        self.assertEqual(res.payload, {"filters": {"color": "red", "size": 10}})
        self.assertEqual(res.payload_partial, {"filters": {"color": "red", "size": 10}})


if __name__ == "__main__":
    unittest.main()
