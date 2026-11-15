"""Execute the v2.7 notebook §4c / §4e / §4f eval cells against synthetic data.

The build-time cell guard only proves each cell *compiles* and that its names
resolve; it cannot see a wrong comparison. This harness pins down the defect the
v2.7 review found in the inherited v2.6 eval cells:

**Flat vs nested key space.** ``gt_args`` is stored NESTED
(``{"filters": {"color": "red"}}``) while ``payload_partial`` carries the
decomposed DOTTED form (``{"filters.color": "red"}``). A raw
``payload_equal(gt_args, payload_partial)`` therefore returns False on every
object-bearing call, regardless of model quality, and the downstream stage
attribution blames the pointer head (``STEP2B_POINTER_RANK_MISS``) for what is
actually a correct reassembly. Fix: flatten both sides through the declared
``object_paths`` before comparing (``_V27_FIXES`` in the v2.7 builder).

Four assertions:
1. the raw comparison really is broken (guards against the fix being a no-op);
2. §4e buckets a reassembly-correct row as ``EXACT_HIT``, not STEP2B;
3. §4f does not report a reassembly-correct row as a precision FP;
4. §4c's waterfall advances past "All Params Matched" for that same row.

Cells 23 / 25 are executed verbatim from the notebook (they are self-contained
after the fix). Cell 19 cannot be executed wholesale - it calls ``predict_split``
and needs a trained resolver - so only its ``_waterfall`` function is sliced out
and run with injected globals.

Run: .venv/bin/python scratch/smoke_v2_7_eval_cells.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spancall.data.schema import UnifiedSample  # noqa: E402
from spancall.eval.toolcall.evaluator import payload_equal  # noqa: E402
from spancall.lean.eval_p1 import classify_eval_cohort  # noqa: E402

NOTEBOOK = "notebooks/spancall_lean_v2_7_train_eval.ipynb"


# --------------------------------------------------------------------------- #
# Notebook cell loading
# --------------------------------------------------------------------------- #
def _cells() -> list:
    with open(NOTEBOOK) as fh:
        return ["".join(c["source"]) for c in json.load(fh)["cells"]]


def _cell_starting_with(prefix: str) -> str:
    for src in _cells():
        if src.lstrip().startswith(prefix):
            return src
    raise SystemExit(f"cell starting with {prefix!r} not found in {NOTEBOOK}")


def _step4_src() -> str:
    """§4e — stage attribution (starts with the json/os/textwrap import line)."""
    for src in _cells():
        if "STEP2B_POINTER_RANK_MISS" in src and "buckets = defaultdict(list)" in src:
            return src
    raise SystemExit("§4e stage-attribution cell not found")


def _step5_src() -> str:
    """§4f — param-precision FP deep dive."""
    for src in _cells():
        if "FP reasons:" in src and "fp_rows" in src:
            return src
    raise SystemExit("§4f FP deep-dive cell not found")


def _waterfall_src(step3: str) -> str:
    """Slice ``def _waterfall(...)`` out of §4c by indentation."""
    lines = step3.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("def _waterfall("))
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][:1] in (" ", "\t", "#")):
        end += 1
    return "\n".join(lines[start:end])


# --------------------------------------------------------------------------- #
# Synthetic fixture: one object-bearing call, correctly predicted
# --------------------------------------------------------------------------- #
TOOL = {
    "name": "search_catalog",
    "description": "search",
    "parameters": [
        {"name": "query", "description": "q", "param_type": "string"},
        {"name": "filters.color", "description": "c", "param_type": "string"},
        {"name": "filters.size", "description": "s", "param_type": "integer"},
    ],
    "risk_tier": "read_only",
    "object_paths": {"filters": ["filters.color", "filters.size"]},
}

SUBSET_ROW = {
    "sample_id": "s_obj",
    "source_dataset": "synthetic",
    "utterance": "red ones, size 10",
    "split": "test_id",
    "stratum": "F",
    "tools": [TOOL],
    "ground_truth_calls": [
        {"tool_name": "search_catalog",
         "arguments": {"query": "red",
                       "filters": {"color": "red", "size": 10}}}
    ],
}

# NESTED GT vs DOTTED prediction -> the v2.7 key-space collision.
GT_NESTED = {"query": "red", "filters": {"color": "red", "size": 10}}
PAYLOAD_DOTTED = {"query": "red", "filters.color": "red", "filters.size": 10}

BASE_PRED = {
    "sample_id": "s_obj",
    "gt_tool": "search_catalog",
    "tool_name": "search_catalog",
    "gt_args": GT_NESTED,
    "payload_partial": PAYLOAD_DOTTED,
    "payload": PAYLOAD_DOTTED,
    "hit": True,
    "abstain_reason": None,
    "joint_margin": 0.5,
    "features": {"parity_checked": 3, "parity_hit": 3},
    "parameters": {
        "query": {"value": "red", "confidence": 0.9, "margin": 0.4,
                  "source_type": "utt_span", "char_span": [0, 3]},
        "filters.color": {"value": "red", "confidence": 0.85, "margin": 0.3,
                          "source_type": "utt_span", "char_span": [0, 3]},
        "filters.size": {"value": 10, "confidence": 0.8, "margin": 0.25,
                         "source_type": "utt_span", "char_span": [16, 18]},
    },
}


def _write_fixture(tmp: str, preds: list) -> dict:
    results_dir = os.path.join(tmp, "results")
    os.makedirs(results_dir, exist_ok=True)
    subset_path = os.path.join(results_dir, "subset_eval.jsonl")
    with open(subset_path, "w") as fh:
        fh.write(json.dumps(SUBSET_ROW) + "\n")
    for name in ("preds_eval.jsonl", "preds_eval_twostep.jsonl"):
        with open(os.path.join(results_dir, name), "w") as fh:
            for p in preds:
                fh.write(json.dumps(p, default=str) + "\n")
    return {"RESULTS_DIR": results_dir, "SUBSET_EVAL_PATH": subset_path}


def _run_cell(src: str, env: dict, label: str) -> dict:
    ns = {"__name__": f"v27_smoke_{label}", "os": os, **env}
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        exec(compile(src, f"<{label}>", "exec"), ns)
    ns["_stdout"] = buf.getvalue()
    return ns


FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(label)


# --------------------------------------------------------------------------- #
def main() -> int:
    cells = _cells()
    step3 = next(s for s in cells if "def _waterfall(" in s)
    step4_src, step5_src = _step4_src(), _step5_src()

    sample = UnifiedSample.from_dict(SUBSET_ROW)
    tool = next(t for t in sample.tools if t.name == "search_catalog")
    paths = dict(tool.object_paths or {})

    print("=" * 78)
    print("CASE 0 — the bug is real (guards against a no-op fix)")
    print("=" * 78)
    raw = payload_equal(GT_NESTED, PAYLOAD_DOTTED, tool)
    check("raw nested-vs-dotted comparison is False", raw is False, f"got {raw}")
    check("cohort is F_func", classify_eval_cohort(SUBSET_ROW) == "F_func",
          classify_eval_cohort(SUBSET_ROW))

    import spancall.data.flatten as _flat
    gt_flat = _flat.flatten_args(GT_NESTED, paths)
    pred_flat = _flat.flatten_args(PAYLOAD_DOTTED, paths)
    flattened = payload_equal(gt_flat, pred_flat, tool)
    check("flattened comparison is True", flattened is True, f"got {flattened}")
    check("flattening produced dotted leaves",
          set(gt_flat) == {"query", "filters.color", "filters.size"}, str(sorted(gt_flat)))

    # ---------------------------------------------------------------- §4e
    print("\n" + "=" * 78)
    print("CASE 1 — §4e must bucket a reassembly-correct row as EXACT_HIT")
    print("=" * 78)
    env = _write_fixture(tempfile.mkdtemp(prefix="v27e5_"), [BASE_PRED])
    ns = _run_cell(step4_src, env, "step4")
    buckets = ns["buckets"]
    check("EXACT_HIT == 1", len(buckets["EXACT_HIT"]) == 1,
          f"got {len(buckets['EXACT_HIT'])}")
    check("not blamed on STEP2B", len(buckets["STEP2B_POINTER_RANK_MISS"]) == 0,
          f"got {len(buckets['STEP2B_POINTER_RANK_MISS'])}")
    check("not blamed on STEP2C", len(buckets["STEP2C_GATE_OR_SCHEMA_REJECT"]) == 0,
          f"got {len(buckets['STEP2C_GATE_OR_SCHEMA_REJECT'])}")

    print("\n  -- negative control: a wrong leaf must still be attributed")
    bad = json.loads(json.dumps(BASE_PRED))
    bad["payload_partial"] = {"query": "red", "filters.color": "blue",
                              "filters.size": 10}
    bad["hit"] = False
    env = _write_fixture(tempfile.mkdtemp(prefix="v27e5b_"), [bad])
    buckets = _run_cell(step4_src, env, "step4_bad")["buckets"]
    check("wrong leaf -> STEP2B (pointer rank miss)",
          len(buckets["STEP2B_POINTER_RANK_MISS"]) == 1,
          str({k: len(v) for k, v in buckets.items()}))

    # ---------------------------------------------------------------- §4f
    print("\n" + "=" * 78)
    print("CASE 2 — §4f must not report a correct row as a precision FP")
    print("=" * 78)
    env = _write_fixture(tempfile.mkdtemp(prefix="v27f5_"), [BASE_PRED])
    ns = _run_cell(step5_src, env, "step5")
    check("fp_rows empty", ns["fp_rows"] == [], f"got {len(ns['fp_rows'])}")
    check("FP reason counter empty", not ns["fp_reason_counter"],
          str(dict(ns["fp_reason_counter"])))

    print("\n  -- §4f is also self-contained: no PREDS_PATH/raw_map from §4e")
    check("no NameError running §4f alone", "_preds" in ns and len(ns["_preds"]) == 1,
          f"_preds={len(ns.get('_preds') or [])}")
    check("§4f resolved PREDS_PATH itself",
          os.path.basename(ns["PREDS_PATH"]) == "preds_eval_twostep.jsonl",
          ns["PREDS_PATH"])

    print("\n  -- negative control: a wrong value is still an FP with a reason")
    # ``fp_reason_counter`` is built from the stored per-param ``parameters``
    # block (the decoder's own view), not from the payload, so the corruption
    # has to be present there for a real resolving run -- mirroring what the
    # notebook actually sees.
    bad_payload = json.loads(json.dumps(BASE_PRED))
    bad_payload["payload"] = {"query": "red", "filters.color": "blue",
                              "filters.size": 10}
    bad_payload["parameters"]["filters.color"]["value"] = "blue"
    bad_payload["hit"] = False
    env = _write_fixture(tempfile.mkdtemp(prefix="v27f5b_"), [bad_payload])
    ns = _run_cell(step5_src, env, "step5_bad")
    check("wrong value -> 1 FP row", len(ns["fp_rows"]) == 1,
          f"got {len(ns['fp_rows'])}")
    check("FP reason is wrong_value (dotted leaf matched to flat GT)",
          ns["fp_reason_counter"].get("wrong_value") == 1,
          str(dict(ns["fp_reason_counter"])))
    check("flat GT stashed on the row for the refusal sweep",
          (ns["fp_rows"][0].get("_gt_flat") or {}).get("filters.color") == "red",
          str(ns["fp_rows"][0].get("_gt_flat")))
    check("no spurious param_not_in_gt (key spaces line up)",
          not ns["fp_reason_counter"].get("param_not_in_gt"),
          str(dict(ns["fp_reason_counter"])))

    # ---------------------------------------------------------------- §4c
    print("\n" + "=" * 78)
    print("CASE 3 — §4c waterfall must advance past 'All Params Matched'")
    print("=" * 78)
    tmp = tempfile.mkdtemp(prefix="v27e3_")
    env = _write_fixture(tmp, [BASE_PRED])
    wf_ns = {
        "__name__": "v27_smoke_waterfall",
        "os": os,
        "json": json,  # imported by the §4c cell header, not by _waterfall itself
        "payload_equal": payload_equal,
        "_flat": _flat,
        "classify_eval_cohort": classify_eval_cohort,
        "UnifiedSample": UnifiedSample,
        "_raw_sub": {"s_obj": (SUBSET_ROW, sample)},
    }
    exec(compile(_waterfall_src(step3), "<waterfall>", "exec"), wf_ns)
    preds_path = os.path.join(env["RESULTS_DIR"], "preds_eval_twostep.jsonl")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        wf_ns["_waterfall"](preds_path, "smoke")
    out = buf.getvalue()
    # Parse the funnel rows: "  -> Step 2b: <name padded to 44> 1 / 1 (...)".
    stages = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"->\s+Step\s+(\S+?):.*?(\d+)\s*/\s*\d+", out)
    }
    check("waterfall reached Step 2b with 1 row", stages.get("2b") == 1, str(stages))
    check("waterfall reached Step 2c with 1 row", stages.get("2c") == 1, str(stages))
    ordered = [stages.get(k) for k in ("0", "1", "2a", "2b", "2c")]
    check("no stage regressed below the previous one",
          all(a >= b for a, b in zip(ordered, ordered[1:])), str(stages))

    print("\n  -- negative control: a wrong leaf must stop at Step 2b")
    bad_wf = json.loads(json.dumps(BASE_PRED))
    bad_wf["payload_partial"] = {"query": "red", "filters.color": "blue",
                                 "filters.size": 10}
    bad_wf["hit"] = False
    tmp_bad = tempfile.mkdtemp(prefix="v27e3b_")
    env_bad = _write_fixture(tmp_bad, [bad_wf])
    with contextlib.redirect_stdout(io.StringIO()) as buf2:
        wf_ns["_waterfall"](
            os.path.join(env_bad["RESULTS_DIR"], "preds_eval_twostep.jsonl"), "bad"
        )
    bad_stages = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"->\s+Step\s+(\S+?):.*?(\d+)\s*/\s*\d+", buf2.getvalue())
    }
    check("Step 2a still passes on the wrong leaf", bad_stages.get("2a") == 1,
          str(bad_stages))
    check("Step 2b drops the wrong leaf to 0", bad_stages.get("2b") == 0,
          str(bad_stages))
    check("Step 2c therefore also 0", bad_stages.get("2c") == 0, str(bad_stages))

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"SMOKE FAILED — {len(FAILURES)} assertion(s): {FAILURES}")
        return 1
    print("SMOKE PASSED — §4c/§4e/§4f score nested and dotted payloads identically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())