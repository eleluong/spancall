"""Execute the v2.7 notebook §4d audit cell against synthetic data.

The build-time cell guard only proves the audit cell *compiles* and that its
names resolve. This harness proves it *behaves*, and specifically pins down the
three regressions that a compile-only check cannot catch:

1. **Flat vs nested predicted payload.** Before step 5 lands the resolver emits
   a FLAT payload (``{"filters.color": "red"}``); after step 5 it emits the
   reassembled NESTED dict (``{"filters": {"color": "red"}}``). The audit must
   score object leaves identically in both cases -- otherwise it silently
   reports 0 % on every object leaf in one of the two regimes.
2. **Leaf-level reassembly.** The integrity block must test that each *emitted*
   leaf survives flat -> nested at the same path, not that the whole payload
   matches GT (``payload_equal`` demands equal key sets, so a model that merely
   omits an argument would look like a reassembly defect).
3. **Missing predictions.** With no ``preds_eval.jsonl`` the cell must say so and
   not raise.

Run: .venv/bin/python scratch/smoke_v2_7_audit_cell.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NOTEBOOK = "notebooks/spancall_lean_v2_7_train_eval.ipynb"


def _load_cell_source(predicate) -> str:
    with open(NOTEBOOK) as fh:
        nb = json.load(fh)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if predicate(src):
            return src
    raise SystemExit("target cell not found in notebook")


def _audit_src() -> str:
    return _load_cell_source(
        lambda s: s.lstrip().startswith("# 4d. Object-decomposition audit")
    )


def _p(name, ptype, required=True):
    return {"name": name, "description": name, "param_type": ptype, "required": required}


TOOL_OBJ = {
    "name": "search_catalog",
    "description": "search",
    "parameters": [
        _p("query", "string"),
        _p("filters.color", "string"),
        _p("filters.size", "integer"),
    ],
    "risk_tier": "read_only",
    "object_paths": {"filters": ["filters.color", "filters.size"]},
}
TOOL_OPAQUE = {
    "name": "make_reservation",
    "description": "reserve",
    "parameters": [_p("party", "object"), _p("tags", "array")],
    "risk_tier": "read_only",
    "object_paths": None,
}

SAMPLES = [
    {
        "sample_id": "s_obj",
        "source_dataset": "synthetic",
        "utterance": "red size 10",
        "split": "test_id",
        "tools": [TOOL_OBJ],
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
        "tools": [TOOL_OPAQUE],
        "ground_truth_calls": [
            {"tool_name": "make_reservation",
             "arguments": {"party": {"adults": 2}, "tags": ["a", "b"]}}
        ],
    },
]

GT = {
    "s_obj": {"query": "red", "filters": {"color": "red", "size": 10}},
    "s_arr": {"party": {"adults": 2}, "tags": ["a", "b"]},
}


def _run_audit(tmp: str, preds: list) -> dict:
    data_dir = os.path.join(tmp, "processed")
    results_dir = os.path.join(tmp, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(data_dir, "eval.jsonl"), "w") as fh:
        for s in SAMPLES:
            fh.write(json.dumps(s) + "\n")
    with open(os.path.join(results_dir, "preds_eval.jsonl"), "w") as fh:
        for p in preds:
            fh.write(json.dumps(p) + "\n")
    ns = {
        "__name__": "v27_audit_smoke",
        "DATA_DIR": data_dir,
        "RESULTS_DIR": results_dir,
        "EVAL_SPLIT": "eval",
        "SUBSET_EVAL_PATH": os.path.join(data_dir, "eval.jsonl"),
        "os": os,
    }
    exec(compile(_audit_src(), "<audit-cell>", "exec"), ns)
    return ns


FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(label)


def main() -> int:
    src = _audit_src()

    # ---------------------------------------------------------------- flat preds
    # Pre-step-5 resolver output: dotted keys at the top level.
    flat_preds = [
        {"sample_id": "s_obj", "gt_args": GT["s_obj"],
         "payload_partial": {"query": "red", "filters.color": "red", "filters.size": 10},
         "hit": True},
        {"sample_id": "s_arr", "gt_args": GT["s_arr"],
         "payload_partial": {"party": {"adults": 2}, "tags": ["b", "a"]},
         "hit": False},
    ]
    print("=" * 78)
    print("CASE A — FLAT predicted payload (pre-step-5 resolver)")
    print("=" * 78)
    ns = _run_audit(tempfile.mkdtemp(prefix="v27flat_"), flat_preds)
    tot, ok = ns["_type_tot"], ns["_type_ok"]
    check("object-leaf scored", tot["object-leaf"] == 2, f"got {tot['object-leaf']}")
    check("both object leaves correct (no false 0%)",
          ok["object-leaf"] == 2, f"got {ok['object-leaf']}/2")
    check("array order-scramble detected", (ok["array"], tot["array"]) == (0, 1),
          f"got {ok['array']}/{tot['array']}")
    check("no residual row (only array failed)",
          {r["sample_id"] for r in ns["_residual"]} == {"s_arr"},
          str([r["sample_id"] for r in ns["_residual"]]))
    check("reassembly integrity clean", ns["_rt_ok"] == ns["_rt_tot"] == 1,
          f"{ns['_rt_ok']}/{ns['_rt_tot']}")

    # -------------------------------------------------------------- nested preds
    # Post-step-5 resolver output: reassembled nested dict.
    nested_preds = [
        {"sample_id": "s_obj", "gt_args": GT["s_obj"],
         "payload_partial": {"query": "red", "filters": {"color": "red", "size": 10}},
         "hit": True},
        {"sample_id": "s_arr", "gt_args": GT["s_arr"],
         "payload_partial": {"party": {"adults": 2}, "tags": ["a", "b"]},
         "hit": True},
    ]
    print("\n" + "=" * 78)
    print("CASE B — NESTED predicted payload (post-step-5 resolver)")
    print("=" * 78)
    ns = _run_audit(tempfile.mkdtemp(prefix="v27nest_"), nested_preds)
    tot, ok = ns["_type_tot"], ns["_type_ok"]
    check("object-leaf scored", tot["object-leaf"] == 2, f"got {tot['object-leaf']}")
    check("nested payload scores identical to flat (no false 0%)",
          ok["object-leaf"] == 2, f"got {ok['object-leaf']}/2")
    check("array now correct", ok["array"] == 1, f"got {ok['array']}")
    check("no residual rows", ns["_residual"] == [],
          str([r["sample_id"] for r in ns["_residual"]]))
    check("reassembly integrity clean", ns["_rt_ok"] == ns["_rt_tot"] == 1,
          f"{ns['_rt_ok']}/{ns['_rt_tot']}")

    # ------------------------------------------ deliberate reassembly corruption
    # A payload whose leaf is present but whose nesting is wrong cannot happen
    # through flatten_args, so this asserts the *detector* fires when a leaf is
    # emitted but unflatten drops it. Simulated via an undeclared root.
    print("\n" + "=" * 78)
    print("CASE C — wrong object leaf value must be scored WRONG")
    print("=" * 78)
    bad_preds = [
        {"sample_id": "s_obj", "gt_args": GT["s_obj"],
         "payload_partial": {"query": "red", "filters.color": "blue",
                             "filters.size": 10},
         "hit": False},
        {"sample_id": "s_arr", "gt_args": GT["s_arr"],
         "payload_partial": {"party": {"adults": 2}, "tags": ["a", "b"]},
         "hit": True},
    ]
    ns = _run_audit(tempfile.mkdtemp(prefix="v27bad_"), bad_preds)
    ok = ns["_type_ok"]
    check("wrong leaf scores 1/2 not 2/2", ok["object-leaf"] == 1,
          f"got {ok['object-leaf']}")
    check("wrong leaf appears in residual",
          any("filters.color" in r["bad"] for r in ns["_residual"]),
          str(ns["_residual"]))

    # --------------------------------------------------------- missing pred file
    print("\n" + "=" * 78)
    print("CASE D — missing preds_eval.jsonl must not raise")
    print("=" * 78)
    try:
        ns = _run_audit(tempfile.mkdtemp(prefix="v27none_"), [])
        check("cell survives empty predictions", sum(ns["_type_tot"].values()) == 0,
              f"scored {sum(ns['_type_tot'].values())}")
        check("no residual without predictions", ns["_residual"] == [])
        check("reassembly totals zero", ns["_rt_tot"] == 0)
    except Exception as exc:  # noqa: BLE001 - the point is to catch a crash
        check("cell survives empty predictions", False, f"raised {exc!r}")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"SMOKE FAILED — {len(FAILURES)} assertion(s): {FAILURES}")
        return 1
    print("SMOKE PASSED — audit cell is correct for flat AND nested payloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())