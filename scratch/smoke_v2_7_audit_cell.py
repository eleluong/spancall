"""Execute the v2.7 notebook §4b audit cell against synthetic data.

The build-time cell guard only proves the audit cell *compiles* and that its
names resolve. This harness proves it *behaves*: it materialises a tiny
``eval.jsonl`` split (one object-bearing tool, one array-bearing tool), a
matching ``preds_eval.jsonl`` with one deliberately-wrong object leaf and one
order-scrambled array, then execs the real cell source in a namespace that
mimics the notebook's globals.

Run: .venv/bin/python scratch/smoke_v2_7_audit_cell.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NOTEBOOK = "notebooks/spancall_lean_v2_7_train_eval.ipynb"


def _load_cells():
    with open(NOTEBOOK) as fh:
        nb = json.load(fh)
    return ["".join(c["source"]) for c in nb["cells"]]


def _find_audit_cell(sources):
    for src in sources:
        if src.lstrip().startswith("# 4b. Object-decomposition audit"):
            return src
    raise SystemExit("audit cell not found in notebook")


def _tool(name, params, object_paths=None):
    return {
        "name": name,
        "description": f"{name} tool",
        "parameters": params,
        "risk_tier": "read_only",
        "object_paths": object_paths,
    }


def _p(name, ptype, required=True):
    return {"name": name, "description": name, "param_type": ptype, "required": required}


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="v27audit_")
    data_dir = os.path.join(tmp, "processed")
    results_dir = os.path.join(tmp, "results")
    os.makedirs(data_dir)
    os.makedirs(results_dir)

    # --- Tool A: object decomposed into two scalar leaves -------------------
    tool_a = _tool(
        "search_catalog",
        [_p("query", "string"), _p("filters.color", "string"), _p("filters.size", "integer")],
        object_paths={"filters": ["filters.color", "filters.size"]},
    )
    # --- Tool B: object left opaque (no object_paths) ----------------------
    tool_b = _tool(
        "make_reservation",
        [_p("party", "object"), _p("tags", "array")],
        object_paths=None,
    )

    samples = [
        {
            "sample_id": "s_obj",
            "source_dataset": "synthetic",
            "utterance": "red size 10",
            "split": "test_id",
            "tools": [tool_a],
            "ground_truth_calls": [
                {"tool_name": "search_catalog",
                 "arguments": {"query": "red", "filters": {"color": "red", "size": 10}}}
            ],
        },
        {
            "sample_id": "s_arr",
            "source_dataset": "synthetic",
            "utterance": "party of two",
            "split": "test_id",
            "tools": [tool_b],
            "ground_truth_calls": [
                {"tool_name": "make_reservation",
                 "arguments": {"party": {"adults": 2}, "tags": ["a", "b"]}}
            ],
        },
    ]
    with open(os.path.join(data_dir, "eval.jsonl"), "w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")

    preds = [
        {   # one object leaf wrong ("blue" instead of "red")
            "sample_id": "s_obj",
            "gt_args": {"query": "red", "filters": {"color": "red", "size": 10}},
            "payload_partial": {"query": "red", "filters.color": "blue", "filters.size": 10},
            "hit": False,
        },
        {   # array order scrambled, same set -> the known v2.6 FP mode
            "sample_id": "s_arr",
            "gt_args": {"party": {"adults": 2}, "tags": ["a", "b"]},
            "payload_partial": {"party": {"adults": 2}, "tags": ["b", "a"]},
            "hit": False,
        },
    ]
    with open(os.path.join(results_dir, "preds_eval.jsonl"), "w") as fh:
        for p in preds:
            fh.write(json.dumps(p) + "\n")

    audit_src = _find_audit_cell(_load_cells())
    ns = {
        "__name__": "v27_audit_smoke",
        "DATA_DIR": data_dir,
        "RESULTS_DIR": results_dir,
        "EVAL_SPLIT": "eval",
        "SUBSET_EVAL_PATH": os.path.join(data_dir, "eval.jsonl"),
        "os": os,
    }
    print("=" * 78)
    print("EXECUTING REAL §4b CELL SOURCE")
    print("=" * 78)
    exec(compile(audit_src, "<audit-cell>", "exec"), ns)

    # ---------------------------------------------------------------- asserts
    stats = ns["_schema_stats"]
    tot, ok = ns["_type_tot"], ns["_type_ok"]
    failures = []

    print("\n" + "=" * 78)
    print("ASSERTIONS")
    print("=" * 78)

    def check(label, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        if not cond:
            failures.append(label)

    check("one object root decomposed", stats["object_roots_decomposed"] == 1,
          f"got {stats['object_roots_decomposed']}")
    check("two scalar leaves emitted", stats["decomposed_leaf_params"] == 2,
          f"got {stats['decomposed_leaf_params']}")
    check("one tool still opaque", stats["tools_with_opaque_object"] == 1,
          f"got {stats['tools_with_opaque_object']}")
    check("object-leaf kind scored", tot["object-leaf"] == 2, f"got {tot['object-leaf']}")
    check("object-leaf exact == 1 (size only)", ok["object-leaf"] == 1,
          f"got {ok['object-leaf']}")
    check("array kind scored", tot["array"] == 1, f"got {tot['array']}")
    check("array exact == 0 (order-scrambled)",
          ok["array"] == 0 and tot["array"] == 1, f"got {ok['array']}/{tot['array']}")
    check("string kind scored", tot["string"] == 1, f"got {tot['string']}")

    # Both rows fail *only* on complex params (s_obj on an object-leaf, s_arr on
    # an array), so both belong in the residual list. A row failing on a scalar
    # would be absent -- that is the property being tested.
    residual = {r["sample_id"]: r for r in ns["_residual"]}
    check("both complex-only rows flagged", set(residual) == {"s_obj", "s_arr"},
          f"got {sorted(residual)}")
    for _sid, _want in (("s_obj", {"object-leaf"}), ("s_arr", {"array"})):
        _r = residual.get(_sid)
        check(f"{_sid} kinds are complex-only", _r is not None and _r["kinds"] == sorted(_want),
              str(_r["kinds"]) if _r else "missing")
        if _r:
            check(f"{_sid} records the wrong param", bool(_r["bad"]), str(_r["bad"]))

    # --- control: a scalar-only failure must NOT appear in the residual list --
    saved = open(os.path.join(results_dir, "preds_eval.jsonl")).read()
    try:
        with open(os.path.join(results_dir, "preds_eval.jsonl"), "w") as fh:
            fh.write(json.dumps({
                "sample_id": "s_obj",
                "gt_args": {"query": "red", "filters": {"color": "red", "size": 10}},
                "payload_partial": {"query": "WRONG",
                                    "filters.color": "red", "filters.size": 10},
                "hit": False,
            }) + "\n")
        ns_ctl = {
            "__name__": "v27_audit_smoke_ctl",
            "DATA_DIR": data_dir,
            "RESULTS_DIR": results_dir,
            "EVAL_SPLIT": "eval",
            "SUBSET_EVAL_PATH": os.path.join(data_dir, "eval.jsonl"),
            "os": os,
        }
        exec(compile(audit_src, "<audit-cell-ctl>", "exec"), ns_ctl)
    finally:
        with open(os.path.join(results_dir, "preds_eval.jsonl"), "w") as fh:
            fh.write(saved)

    check("scalar-only failure excluded from residual",
          ns_ctl["_residual"] == [],
          f"got {[r['sample_id'] for r in ns_ctl['_residual']]}")

    check("reassembly roundtrip counted", ns["_roundtrip_tot"] == 1,
          f"got {ns['_roundtrip_tot']}")
    check("reassembly roundtrip FAILED for wrong leaf",
          ns["_roundtrip_ok"] == 0, f"got {ns['_roundtrip_ok']}")

    print("\n" + "=" * 78)
    if failures:
        print(f"SMOKE FAILED — {len(failures)} assertion(s): {failures}")
        return 1
    print("SMOKE PASSED — audit cell behaves as designed on synthetic data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())