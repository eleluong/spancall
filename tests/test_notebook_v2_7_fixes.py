"""Regression tests for the v2.7 notebook semantic fixes.

The build-time cell guard proves every cell *compiles* and that its names
resolve. It cannot see a wrong comparison, so three classes of defect reached a
generated notebook and stayed there until review:

1. **Flat vs nested key space.** ``gt_args`` is stored NESTED
   (``{"filters": {"color": "red"}}``) while ``payload_partial``/``parameters``
   carry the decomposed DOTTED form (``{"filters.color": "red"}``). A raw
   ``payload_equal(gt_args, payload_partial)`` is False on every object-bearing
   call regardless of model quality, and the stage attribution then blames the
   pointer head for a correct reassembly.
2. **``schema=None``.** ``payload_equal`` only materialises optional defaults
   when a schema is passed (``spancall/eval/toolcall/evaluator.py``), so a
   correct row that merely omitted an optional param was counted as an FP.
3. **Un-gated silent failures.** ``LeanResolver(model_path=None)`` skips
   ``load_checkpoint``, so a run with no checkpoint scores randomly initialised
   weights; and the §1 offset gate audited only eval+dev while §5 reports on
   OOD/adversarial splits.

These tests read the *generated* notebook, so they cover whatever the builder
actually wrote rather than the builder's intentions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "spancall_lean_v2_7_train_eval.ipynb"
)


def _sources() -> List[str]:
    with open(NOTEBOOK) as fh:
        return ["".join(c["source"]) for c in json.load(fh)["cells"]]


def _code_cells() -> List[str]:
    with open(NOTEBOOK) as fh:
        nb = json.load(fh)
    return [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    ]


def _cell_containing(*needles: str) -> str:
    for src in _code_cells():
        if all(n in src for n in needles):
            return src
    raise AssertionError(f"no cell contains all of {needles!r}")


@pytest.fixture(scope="module")
def notebook_exists() -> None:
    if not NOTEBOOK.is_file():
        pytest.skip(f"{NOTEBOOK} not built; run scripts/build_lean_v2_7_colab_notebook.py")


# ---------------------------------------------------------------------------
# 1. No payload_equal call may compare raw nested GT to a dotted payload
# ---------------------------------------------------------------------------
def _call_args(src: str, callee: str) -> List[str]:
    """Top-level argument strings of every ``callee(...)`` call in ``src``.

    Balanced-paren scan rather than a regex: the arguments themselves contain
    calls (``_flat.flatten_args(p.get("payload") or {}, _paths)``), so a
    non-greedy ``\\(.*?\\)`` would stop at the first inner ``)`` and inspect a
    fragment instead of the argument list.
    """
    out: List[str] = []
    needle = callee + "("
    pos = 0
    while True:
        start = src.find(needle, pos)
        if start < 0:
            return out
        i = start + len(needle)
        depth = 1
        while i < len(src) and depth:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
            i += 1
        out.append(src[start + len(needle) : i - 1])
        pos = i


def _split_args(args: str) -> List[str]:
    """Split an argument list on top-level commas only."""
    parts, depth, cur = [], 0, ""
    for ch in args:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def test_no_payload_equal_call_passes_schema_none(notebook_exists) -> None:
    """``schema=None`` silently disables optional-default materialisation."""
    offenders = []
    for src in _code_cells():
        for args in _call_args(src, "payload_equal"):
            parts = _split_args(args)
            if len(parts) >= 3 and parts[2] == "None":
                line = src[: src.find(args)].count("\n") + 1
                offenders.append(f"line {line}: {args.strip()[:80]}")
    assert not offenders, "payload_equal called with schema=None:\n" + "\n".join(offenders)


def test_every_payload_equal_call_flattens_both_sides(notebook_exists) -> None:
    """Both compared operands must reach the call in the flat key space.

    A dotted leaf (``filters.color``) can never equal the nested dict it came
    from, so a comparison that does not flatten the GT *and* the prediction is
    wrong by construction, no matter how good the model is.
    """
    sites = 0
    for src in _code_cells():
        for args in _call_args(src, "payload_equal"):
            parts = _split_args(args)
            assert len(parts) >= 3, f"payload_equal called with {len(parts)} arg(s): {args!r}"
            sites += 1
            for operand in parts[:2]:
                if re.fullmatch(r"\w+", operand):
                    # Named variable: it must be assigned from flatten_args here.
                    assert re.search(rf"^\s*{operand} = .*flatten_args\(", src, re.M), (
                        f"payload_equal operand {operand!r} is not produced by "
                        f"flatten_args in this cell; nested GT vs dotted payload "
                        f"will never match"
                    )
                else:
                    assert "flatten_args" in operand, (
                        f"payload_equal operand is compared un-flattened: {operand[:100]!r}"
                    )
    assert sites >= 3, f"expected at least 3 payload_equal call sites, found {sites}"


def test_waterfall_uses_declared_object_paths(notebook_exists) -> None:
    src = _cell_containing("def _waterfall(", "All Params Matched")
    assert "flatten_args" in src
    assert "object_paths" in src


def test_stage_attribution_flattens_partial_and_full_payload(notebook_exists) -> None:
    src = _cell_containing("STEP2B_POINTER_RANK_MISS")
    assert "_part_flat" in src, "partial payload is compared un-flattened"
    assert 'flatten_args(p.get("payload")' in src, "full payload is compared un-flattened"


# ---------------------------------------------------------------------------
# 2. §4f must be self-contained and reuse the flat GT for the refusal sweep
# ---------------------------------------------------------------------------
def test_fp_deep_dive_is_self_contained(notebook_exists) -> None:
    """§4f used to close over PREDS_PATH/raw_map from §4e -> NameError on rerun."""
    src = _cell_containing("FP reasons:", "fp_rows")
    assert re.search(r'^PREDS_PATH = ', src, re.M), "§4f does not resolve PREDS_PATH itself"
    assert "raw_map" in src and "preds_eval_twostep.jsonl" in src
    assert "run 4c first" in src, "§4f does not guard a missing predictions file"


def test_fp_deep_dive_uses_flat_gt_everywhere(notebook_exists) -> None:
    src = _cell_containing("FP reasons:", "fp_rows")
    assert 'p["_gt_flat"] = _gt_flat' in src
    # Every downstream read of the GT args inside §4f must use the flat copy,
    # otherwise dotted predicted param names never line up.
    reads = re.findall(r'p\.get\("gt_args"\)', src)
    assert len(reads) <= 1, (
        "§4f reads raw gt_args after flattening; the refusal sweep and the dump "
        f"must use _gt_flat (found {len(reads)} raw reads)"
    )
    assert src.count('p.get("_gt_flat")') >= 3


# ---------------------------------------------------------------------------
# 3. Silent-failure gates
# ---------------------------------------------------------------------------
def test_missing_checkpoint_is_fatal(notebook_exists) -> None:
    src = _cell_containing("CHECKPOINT_DIR", "LeanResolver(")
    assert "if CHECKPOINT_DIR is None:" in src
    assert "UN TRAINED" in src.upper() or "UNTRAINED" in src.upper()


def test_offset_gate_covers_the_splits_section5_reports(notebook_exists) -> None:
    """Gate B used to audit eval+dev only, while §5 also reports OOD/adversarial."""
    src = _cell_containing("[Gate B / Audit]", "audit_gt_offsets")
    assert "TEST_SPLITS" in src, "Gate B does not iterate the OOD/adversarial splits"


def test_stamp_provenance_collision_is_reported(notebook_exists) -> None:
    src = _cell_containing("[Gate A / Stamp]")
    assert "prepare_data.py re-stamped" in src


def test_decomposition_gate_is_warning_level(notebook_exists) -> None:
    """Gate C must inform, not abort: plan steps 2-3 (label side) are pending."""
    src = _cell_containing("[Gate C / Decompose]")
    gate = src[src.index("# Pre-Flight Gate C") :]
    assert "raise SystemExit" not in gate, "Gate C must not abort a run"
    assert "WARN" in gate


def test_data_cell_imports_what_it_calls(notebook_exists) -> None:
    src = _cell_containing("[Gate A / Stamp]")
    assert "subprocess" in src.split("\n\n")[0] or "import subprocess" in src
    assert "sys" in src


# ---------------------------------------------------------------------------
# 4. Schema-coverage counters are per distinct tool, not per occurrence
# ---------------------------------------------------------------------------
def test_schema_view_counters_are_deduped_per_tool(notebook_exists) -> None:
    src = _cell_containing("Object Decomposition Coverage")
    assert "_seen_tools" in src, "schema coverage counted per (sample, tool) occurrence"


# ---------------------------------------------------------------------------
# 5. No v2.6 path artifacts survived the version bump
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "artifact",
    ["processed_v2/", "tensorized_lean_v2_6", "results_lean_v2_6", "colab_lean_v2_6"],
)
def test_no_v2_6_path_artifacts(notebook_exists, artifact: str) -> None:
    blob = "\n".join(_sources())
    assert artifact not in blob, f"stale v2.6 path artifact {artifact!r} in the v2.7 notebook"