"""Regression tests for the LLM-baseline *population contract*.

The bug these tests exist to prevent
------------------------------------
`spancall_lean_v3_2_llm_baseline.ipynb` compares SpanCall against a hosted LLM on
"the same rows". The rows are defined by `build_stratified_eval_subset(...,
limit=EVAL_LIMIT)`, and `EVAL_LIMIT` must equal the value the training notebook
used (2000), because the frozen prediction files were produced from that sample.

That contract was broken in two places at once:

1. the cell passed `limit=None`, sampling the WHOLE split (9,101 xLAM rows) -
   a superset of the paper's sample, so the shape of the output looked correct;
2. the subset was cached on `os.path.exists()`, so once a wrong `_clean` file
   existed, every later run reused it even after the argument was fixed.

The failure surfaced three cells later as

    [spancall:xlam_eval] 69/910 subset rows have NO prediction (first: xlam::4997::r7)

which blames the predictions and tells the reader to re-run inference - the one
action that cannot possibly help. These tests pin all three things:

* `EVAL_LIMIT` equals the training notebook's value;
* the subset-building call passes `limit=EVAL_LIMIT` and is NOT cached;
* `_validate_coverage` distinguishes "wrong population" from "interrupted run"
  and words the first one so it stops sending readers to re-infer.

They read the *generated* notebook, so they assert what shipped rather than what
the builder intended.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
LLM_NB = ROOT / "notebooks" / "spancall_lean_v3_2_llm_baseline.ipynb"
TRAIN_NB = ROOT / "notebooks" / "spancall_lean_v3_2_train_eval.ipynb"
BUILDER = ROOT / "scripts" / "build_llm_baseline_notebook.py"

# The paper's population sizes, from the v3.2 run.
V32_CLEAN_ROWS = {"hermes_eval": 1659, "xlam_eval": 2000}


def _code_cells(path: Path) -> List[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _find_cell(marker: str) -> str:
    """The one code cell containing `marker`.

    Located by content, not by index: the setup section is split into §0a
    (bootstrap magics) and §0b (pure-Python config), and any future cell that
    gets inserted would otherwise silently re-point these tests at the wrong
    cell while still passing.
    """
    hits = [src for src in _code_cells(LLM_NB) if marker in src]
    assert len(hits) == 1, (
        f"expected exactly one code cell containing {marker!r}, found {len(hits)}"
    )
    return hits[0]


def _config_cell() -> str:
    """§0b: the configuration cell that defines EVAL_LIMIT and `_limit_rows`."""
    return _find_cell("EVAL_LIMIT = 2000")


def _data_cell() -> str:
    """§1: stratify -> single-call filter -> Step-0 sanitize."""
    return _find_cell("SUBSET_INFO[_sp] = build_stratified_eval_subset(")


def _infer_cell() -> str:
    """§3: load the frozen SpanCall predictions and check coverage."""
    return _find_cell("def _validate_coverage")


def _code_only(src: str) -> str:
    """Drop `#` comments so prose about a retired setting is not read as code."""
    return "\n".join(l.split("#", 1)[0] for l in src.splitlines())


# ---------------------------------------------------------------------------
# The limit is the population definition, and it must match the training run
# ---------------------------------------------------------------------------

def test_eval_limit_is_defined_in_the_config_cell() -> None:
    assert re.search(r"^EVAL_LIMIT\s*=\s*(\d+)", _config_cell(), re.M), (
        "the config cell (§0b) must define EVAL_LIMIT: it is the population "
        "definition, not a sampling convenience, and it needs to be visible next "
        "to the other knobs."
    )


def test_eval_limit_matches_the_training_notebook() -> None:
    """The whole comparison rests on this equality.

    Two notebooks that sample different populations cannot be compared, and
    nothing in the metrics would reveal it - both would print a well-formed
    table over a different base.
    """
    m_llm = re.search(r"^EVAL_LIMIT\s*=\s*(\d+)", _config_cell(), re.M)
    assert m_llm, "EVAL_LIMIT missing from the LLM baseline notebook"
    train_src = "\n".join(_code_cells(TRAIN_NB))
    m_train = re.search(r"^EVAL_LIMIT\s*=\s*(\d+)", train_src, re.M)
    assert m_train, "EVAL_LIMIT missing from the training notebook"

    assert int(m_llm.group(1)) == int(m_train.group(1)), (
        f"LLM baseline samples limit={m_llm.group(1)} but the training notebook "
        f"(which produced the frozen predictions) used limit={m_train.group(1)}. "
        f"These must be equal or the two arms are scored on different rows."
    )


def test_subset_is_built_with_limit_eval_limit_not_none() -> None:
    """`limit=None` samples the whole split - a superset of the paper's sample."""
    code = _code_only(_data_cell())
    call = re.search(
        r"build_stratified_eval_subset\((.*?)\)\s*\n", code, re.S
    )
    assert call, "cell 1 must call build_stratified_eval_subset()"
    args = call.group(1)

    assert re.search(r"\blimit\s*=\s*EVAL_LIMIT\b", args), (
        "the subset must be sampled with `limit=EVAL_LIMIT`. Passing `limit=None` "
        "samples the whole split (9,101 xLAM rows vs the paper's 2,000), which is "
        "a different population that the frozen predictions cannot cover."
    )
    assert not re.search(r"\blimit\s*=\s*None\b", args), (
        "`limit=None` is the exact bug this notebook was fixed for"
    )


def test_subset_is_not_cached_on_file_existence() -> None:
    """`if not os.path.exists(_sub)` is a cache with no invalidation key.

    A `_clean` file produced with a different `EVAL_LIMIT` would be reused
    silently, and the mismatch would only appear as a coverage error later -
    after the reader had already been told to re-run inference.
    """
    code = _code_only(_data_cell())
    guarded = re.search(
        r"if\s+not\s+os\.path\.exists\(\s*_sub\s*\)\s*:", code
    )
    assert guarded is None, (
        "the stratified subset must be rebuilt unconditionally: caching it on "
        "file existence lets a subset built with a different EVAL_LIMIT survive "
        "the fix and keep producing a population-mismatch error."
    )


def test_single_call_filter_also_runs_unconditionally() -> None:
    code = _code_only(_data_cell())
    # The filter call must not sit under an `if not os.path.exists(...)` branch.
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if "_single_call_filter(_sub" in line:
            indent = len(line) - len(line.lstrip())
            # Walk back to the nearest enclosing guard at a lower indent.
            for prev in reversed(lines[:i]):
                if not prev.strip():
                    continue
                prev_indent = len(prev) - len(prev.lstrip())
                if prev_indent < indent:
                    assert "os.path.exists" not in prev, (
                        "the single-call filter is cached behind a file-existence "
                        "check; a stale multi-call subset would then survive."
                    )
                    break
            return
    pytest.fail("cell 1 no longer calls _single_call_filter()")


# ---------------------------------------------------------------------------
# The coverage guard must name the RIGHT cause
# ---------------------------------------------------------------------------

def _validate_coverage_fn():
    """Extract the shipped `_validate_coverage` and its `_limit_rows` dependency."""
    c0, c3 = _config_cell(), _infer_cell()
    limit_src = c0[c0.index("def _limit_rows"):c0.index("# Print shapes")]
    start = c3.index("def _validate_coverage")
    end = c3.index("def run_spancall_inference")
    ns: Dict[str, Any] = {
        "os": __import__("os"),
        "json": json,
        "EVAL_LIMIT": 2000,
        "LLM_LIMIT_FRACTION": 1.0,
    }
    exec(compile(limit_src, "<limit>", "exec"), ns)
    exec(compile(c3[start:end], "<cov>", "exec"), ns)
    return ns["_validate_coverage"]


def _row(sid: str) -> Dict[str, Any]:
    return {
        "sample_id": sid,
        "stratum": "F",
        "utterance": "weather please",
        "tools": [{"name": "get_weather", "description": "d", "parameters": []}],
        "ground_truth_calls": [{"tool_name": "get_weather", "arguments": {"city": "Hanoi"}}],
    }


def _write(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_coverage_passes_when_populations_match(tmp_path: Path) -> None:
    validate = _validate_coverage_fn()
    sub = tmp_path / "subset_xlam_eval_clean.jsonl"
    rows = [_row(f"x{i}") for i in range(20)]
    _write(sub, rows)
    validate("xlam_eval", str(sub), {r["sample_id"]: {} for r in rows})


def test_population_mismatch_is_reported_as_a_sampling_bug(tmp_path: Path) -> None:
    """A subset larger than EVAL_LIMIT proves a different sampling call.

    Sampling caps at `limit` and the later stages only filter, so this size is
    unreachable for a correct build. It must NOT be reported as a missing
    prediction, because re-running inference cannot fix it.
    """
    validate = _validate_coverage_fn()
    sub = tmp_path / "subset_xlam_eval_clean.jsonl"
    rows = [_row(f"x{i}") for i in range(3000)]      # > EVAL_LIMIT
    _write(sub, rows)
    preds = {r["sample_id"]: {} for r in rows[:2000]}

    with pytest.raises(SystemExit) as exc:
        validate("xlam_eval", str(sub), preds)

    msg = str(exc.value)
    assert "POPULATION MISMATCH" in msg, f"wrong diagnosis: {msg}"
    assert "EVAL_LIMIT" in msg, "the message must name the setting to fix"
    assert "re-running inference" in msg.lower(), (
        "the message must say explicitly that re-inference will not help"
    )
    assert "will NOT help" in msg


def test_interrupted_inference_keeps_the_original_message(tmp_path: Path) -> None:
    """Right population, a few rows absent -> the row-level message is correct."""
    validate = _validate_coverage_fn()
    sub = tmp_path / "subset_xlam_eval_clean.jsonl"
    rows = [_row(f"x{i}") for i in range(100)]
    _write(sub, rows)
    preds = {r["sample_id"]: {} for r in rows[:90]}

    with pytest.raises(SystemExit) as exc:
        validate("xlam_eval", str(sub), preds)

    msg = str(exc.value)
    assert "10/100 subset rows have NO prediction" in msg
    assert "POPULATION MISMATCH" not in msg
    assert "x9" in msg, "the message should name a missing row"


# ---------------------------------------------------------------------------
# The builder must not reintroduce `limit=None`
# ---------------------------------------------------------------------------

def test_builder_source_passes_limit_eval_limit() -> None:
    """Guard the generator too: the .ipynb is generated and could drift."""
    src = BUILDER.read_text(encoding="utf-8")
    assert "limit=EVAL_LIMIT" in src, (
        "scripts/build_llm_baseline_notebook.py must sample with limit=EVAL_LIMIT"
    )
    assert "limit=None" not in _code_only(src).replace("`limit=None`", ""), (
        "the builder must not pass limit=None to build_stratified_eval_subset"
    )


def test_persisted_metadata_records_the_population(tmp_path) -> None:
    """`llm_parity.json` must say which rows were eligible, not just how many scored."""
    cell7 = _find_cell('"n_rows_sampled"')
    assert '"eval_limit"' in cell7, (
        "meta must record eval_limit: a reader cannot tell a 2000-row sample from "
        "a 9101-row one by looking at n_rows alone"
    )
    assert '"n_rows_sampled"' in cell7, (
        "meta must record the pre-filter sampled count so the population is auditable"
    )