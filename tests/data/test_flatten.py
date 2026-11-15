"""Tests for Lean v2.7 object-param decomposition helpers.

Covers `spancall/data/flatten.py` (flatten/unflatten round trips, declared vs
inferred mode, ambiguity guards) and the `ToolDefinition.object_paths`
serialization contract in `spancall/data/schema.py`.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict, is_dataclass

from spancall.data.flatten import (
    OBJECT_MAX_DEPTH,
    derive_object_paths,
    flatten_args,
    get_path,
    is_decomposable,
    object_paths_from_params,
    split_flat_payload,
    unflatten_args,
    validate_object_paths,
)
from spancall.data.schema import ParameterSchema, ToolDefinition


def _p(name: str, ptype: str = "string", **kw) -> ParameterSchema:
    return ParameterSchema(name=name, description="", param_type=ptype, **kw)


class TestDeriveObjectPaths(unittest.TestCase):
    def test_groups_leaves_by_root(self):
        paths = derive_object_paths(["filters.color", "filters.size", "query"])
        self.assertEqual(paths, {"filters": ["filters.color", "filters.size"]})

    def test_ignores_undotted_names(self):
        self.assertEqual(derive_object_paths(["query", "limit"]), {})

    def test_nested_root_uses_first_segment(self):
        paths = derive_object_paths(["a.b.c", "a.b.d"])
        self.assertEqual(paths, {"a": ["a.b.c", "a.b.d"]})

    def test_from_parameter_schemas(self):
        params = [_p("filters.color"), _p("filters.size"), _p("query")]
        self.assertEqual(object_paths_from_params(params), derive_object_paths(
            ["filters.color", "filters.size", "query"]))


class TestValidateObjectPaths(unittest.TestCase):
    def test_clean_schema_has_no_problems(self):
        self.assertEqual(validate_object_paths(["filters.color", "query"]), [])

    def test_flags_literal_root_shadowing_dotted_leaves(self):
        problems = validate_object_paths(["filters", "filters.color"])
        self.assertTrue(any("ambiguous namespace" in p for p in problems))

    def test_flags_undeclared_leaf(self):
        problems = validate_object_paths(
            ["filters.color"], {"filters": ["filters.size"]}
        )
        self.assertTrue(any("not a declared param" in p for p in problems))

    def test_flags_leaf_owned_by_another_root(self):
        problems = validate_object_paths(
            ["filters.color"], {"filters": ["other.color"]}
        )
        self.assertTrue(any("whose root is not" in p for p in problems))


class TestFlattenInferred(unittest.TestCase):
    def test_nested_object_becomes_dotted_leaves(self):
        flat = flatten_args({"filters": {"color": "red", "size": 10}, "query": "x"})
        self.assertEqual(flat, {"filters.color": "red", "filters.size": 10, "query": "x"})

    def test_round_trip_is_exact(self):
        original = {"filters": {"color": "red", "size": 10}, "query": "x"}
        self.assertEqual(unflatten_args(flatten_args(original)), original)

    def test_depth_cap_keeps_deep_tree_whole(self):
        deep = {"a": {"b": {"c": {"d": 1}}}}
        flat = flatten_args(deep)
        # a.b.c.d would be 4 segments > OBJECT_MAX_DEPTH (3): subtree kept whole.
        self.assertEqual(flat, {"a.b.c": {"d": 1}})
        self.assertEqual(OBJECT_MAX_DEPTH, 3)

    def test_depth_cap_exactly_at_limit_flattens(self):
        flat = flatten_args({"a": {"b": {"c": 1}}})
        self.assertEqual(flat, {"a.b.c": 1})

    def test_empty_object_is_preserved(self):
        self.assertEqual(flatten_args({"filters": {}}), {"filters": {}})
        self.assertEqual(unflatten_args({"filters": {}}), {"filters": {}})

    def test_non_mapping_values_pass_through(self):
        flat = flatten_args({"tags": ["a", "b"], "n": 3, "ok": True})
        self.assertEqual(flat, {"tags": ["a", "b"], "n": 3, "ok": True})

    def test_keys_with_dots_are_not_renamed(self):
        # A literal key containing '.' cannot be expressed as a path segment;
        # the subtree is kept whole instead of being silently rewritten.
        flat = flatten_args({"odd": {"a.b": 1}})
        self.assertEqual(flat, {"odd": {"a.b": 1}})
        self.assertEqual(unflatten_args(flat), {"odd": {"a.b": 1}})

    def test_none_leaf_value_is_kept(self):
        flat = flatten_args({"filters": {"color": None}})
        self.assertEqual(flat, {"filters.color": None})
        self.assertEqual(unflatten_args(flat), {"filters": {"color": None}})


class TestFlattenDeclared(unittest.TestCase):
    def test_only_declared_roots_are_split(self):
        flat = flatten_args(
            {"filters": {"color": "red"}, "other": {"x": 1}},
            {"filters": ["filters.color"]},
        )
        self.assertEqual(flat, {"filters.color": "red", "other": {"x": 1}})

    def test_round_trip_with_declared_paths(self):
        original = {"filters": {"color": "red", "size": 10}, "query": "x"}
        paths = {"filters": ["filters.color", "filters.size"]}
        flat = flatten_args(original, paths)
        self.assertEqual(unflatten_args(flat, paths), original)

    def test_undeclared_sibling_keys_survive_round_trip(self):
        # Schema drift: the payload carries a key the schema does not describe.
        original = {"filters": {"color": "red", "brand_new": 1}}
        paths = {"filters": ["filters.color"]}
        flat = flatten_args(original, paths)
        self.assertIn("filters.brand_new", flat)
        self.assertEqual(unflatten_args(flat, paths), original)

    def test_dotted_literal_key_survives_when_root_undeclared(self):
        original = {"weird.key": 1, "filters": {"color": "red"}}
        paths = {"filters": ["filters.color"]}
        flat = flatten_args(original, paths)
        self.assertEqual(flat, {"weird.key": 1, "filters.color": "red"})
        self.assertEqual(unflatten_args(flat, paths), original)

    def test_scalar_where_object_declared_is_kept_whole(self):
        flat = flatten_args({"filters": "not-an-object"}, {"filters": ["filters.color"]})
        self.assertEqual(flat, {"filters": "not-an-object"})


class TestUnflatten(unittest.TestCase):
    def test_conflicting_literal_and_leaf_raises_in_strict_mode(self):
        with self.assertRaises(ValueError):
            unflatten_args({"filters": "scalar", "filters.color": "red"})

    def test_non_strict_mode_lets_literal_win(self):
        out = unflatten_args({"filters": "scalar", "filters.color": "red"}, strict=False)
        self.assertEqual(out["filters"], "scalar")

    def test_missing_leaf_is_simply_absent(self):
        out = unflatten_args({"filters.color": "red"})
        self.assertEqual(out, {"filters": {"color": "red"}})

    def test_null_leaf_is_present_not_dropped(self):
        out = unflatten_args({"filters.color": None})
        self.assertEqual(out, {"filters": {"color": None}})


class TestSplitFlatPayload(unittest.TestCase):
    def test_separates_leaves_from_whole_roots(self):
        leaves, whole = split_flat_payload(
            {"filters.color": "red", "query": "x"}, {"filters": ["filters.color"]}
        )
        self.assertEqual(leaves, {"filters.color": "red"})
        self.assertEqual(whole, {"query": "x"})


class TestHelpers(unittest.TestCase):
    def test_get_path_reads_nested(self):
        self.assertEqual(get_path({"a": {"b": 2}}, "a.b"), 2)

    def test_get_path_returns_missing_sentinel(self):
        from spancall.data.flatten import _MISSING

        self.assertIs(get_path({"a": {}}, "a.b"), _MISSING)

    def test_is_decomposable_depth_rules(self):
        self.assertTrue(is_decomposable({"a": 1}))
        self.assertTrue(is_decomposable({"a": {"b": {"c": 1}}}))
        self.assertFalse(is_decomposable({"a": {"b": {"c": {"d": 1}}}}))
        self.assertFalse(is_decomposable({}))
        self.assertFalse(is_decomposable("string"))


class TestToolDefinitionObjectPaths(unittest.TestCase):
    def test_default_is_none_and_back_compat(self):
        tool = ToolDefinition(name="t", description="d", parameters=[_p("q")])
        self.assertIsNone(tool.object_paths)
        self.assertFalse(tool.has_object_paths())

    def test_from_dict_reads_object_paths(self):
        tool = ToolDefinition.from_dict({
            "name": "t",
            "description": "d",
            "parameters": [{"name": "filters.color", "description": "", "param_type": "string"}],
            "object_paths": {"filters": ["filters.color"]},
        })
        self.assertEqual(tool.object_paths, {"filters": ["filters.color"]})
        self.assertTrue(tool.has_object_paths())

    def test_from_dict_derives_paths_for_legacy_records(self):
        tool = ToolDefinition.from_dict({
            "name": "t",
            "description": "d",
            "parameters": [
                {"name": "filters.color", "description": "", "param_type": "string"},
                {"name": "filters.size", "description": "", "param_type": "integer"},
                {"name": "query", "description": "", "param_type": "string"},
            ],
        })
        self.assertEqual(tool.object_paths["filters"], ["filters.color", "filters.size"])

    def test_from_dict_refuses_ambiguous_derivation(self):
        tool = ToolDefinition.from_dict({
            "name": "t",
            "description": "d",
            "parameters": [
                {"name": "filters", "description": "", "param_type": "object"},
                {"name": "filters.color", "description": "", "param_type": "string"},
            ],
        })
        self.assertIsNone(tool.object_paths)

    def test_object_paths_round_trips_through_asdict(self):
        tool = ToolDefinition(
            name="t", description="d",
            parameters=[_p("filters.color")],
            object_paths={"filters": ["filters.color"]},
        )
        self.assertTrue(is_dataclass(tool))
        rebuilt = ToolDefinition.from_dict(asdict(tool))
        self.assertEqual(rebuilt.object_paths, tool.object_paths)


if __name__ == "__main__":
    unittest.main()