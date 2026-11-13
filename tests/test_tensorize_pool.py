"""Tests for the tensorize pool in the generated Lean notebook.

The tensorize cell is *string data* inside ``scripts/build_lean_v2_colab_notebook.py``
and only becomes code once the notebook runs, so nothing in the normal test
suite exercised its scheduling logic. That logic is where the interesting
failure modes live:

* results silently reordered (they must be applied in INPUT order even though
  they complete out of order -- the split files are read back by index-assuming
  callers and a scrambled split would be an invisible data bug),
* unbounded task submission (pickling an entire 50k-line split into the queue),
* ``need`` being ignored so a sampled split reads the whole corpus.

These tests run the REAL cell source against a thread-backed stand-in pool:
threads need no pickling, so the scheduling logic is tested in isolation, while
the production fork wiring stays the notebook's business.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "spancall_lean_v2_train_eval.ipynb"
)

# Markers that bracket the pool machinery inside the tensorize cell.
_START = "_WINDOW = "
_END = "groups = defaultdict(list)"


def _tensorize_cell_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src = src if isinstance(src, str) else "".join(src)
        if "def _drain_ordered" in src:
            return src
    raise AssertionError("tensorize cell not found in the notebook")


def _load_pool_module() -> Dict[str, Any]:
    """Exec only the pool helpers out of the real cell source."""
    src = _tensorize_cell_source()
    start = src.index(_START)
    end = src.index(_END)
    ns: Dict[str, Any] = {
        "queue": queue,
        "time": time,
        "json": json,
        "Counter": Counter,
        # Only touched on the `executor is None` branch, which these tests
        # deliberately avoid: they inject their own pool.
        "mp": None,
        "ProcessPoolExecutor": None,
    }
    exec(compile(src[start:end], str(NOTEBOOK), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def pool_mod() -> Dict[str, Any]:
    return _load_pool_module()


def _install(mod: Dict[str, Any], fn) -> Dict[str, Any]:
    """Point the module's task symbol at a test double."""
    mod["_tensorize_lean_task"] = fn
    return mod


def test_results_are_applied_in_input_order_despite_out_of_order_completion(
    pool_mod,
) -> None:
    """Later rows finish FIRST; the output must still read 0,1,2,...

    The sleep is inverted against the index so completion order is exactly
    reversed. Without the reorder buffer the file order would be reversed too.
    """
    n = 40

    def slow_reversed(args):
        raw_line, _pack = args
        i = int(raw_line)
        time.sleep((n - i) * 0.0015)
        return f"row{i}", {}, {}

    _install(pool_mod, slow_reversed)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows, drops, errors, info = pool_mod["_tensorize_many"](
            [str(i) for i in range(n)], True, None, 8, executor=pool
        )

    assert rows == [f"row{i}" for i in range(n)]
    assert info["pooled"] is True
    assert info["workers"] == 8
    assert not drops and not errors


def test_inflight_tasks_are_bounded_by_the_window(pool_mod) -> None:
    """Submitting a huge split must not queue the whole thing at once."""
    lock = threading.Lock()
    live = [0]
    peak = [0]

    def counting(args):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.004)
        with lock:
            live[0] -= 1
        return "r", {}, {}

    _install(pool_mod, counting)
    window = 6
    with ThreadPoolExecutor(max_workers=4) as pool:
        pool_mod["_drain_ordered"](
            pool,
            [str(i) for i in range(120)],
            True,
            lambda res: None,
            lambda: False,
            rows_len=lambda: 0,
            need=120,
            workers=4,
            window=window,
        )

    # Never more than one window outstanding in the parent's hands.
    assert peak[0] <= window, f"peak in-flight {peak[0]} exceeded window {window}"


def test_submission_is_budgeted_by_need(pool_mod) -> None:
    """A sampled split must not pickle the whole corpus before stopping.

    Regression: with only a window-sized cap, ``need=5`` over 500 lines still
    submitted all 500 tasks. The budget cap fixes that -- but it must stay
    generous enough that DROPPED rows (which return None) can still be made up.
    """
    seen: List[int] = []
    lock = threading.Lock()

    def half_dropped(args):
        raw_line, _pack = args
        with lock:
            seen.append(int(raw_line))
        # Every other row is unusable, so half the tasks yield nothing.
        return (None if int(raw_line) % 2 else f"row{raw_line}"), {}, {}

    _install(pool_mod, half_dropped)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows, _drops, _errors, _info = pool_mod["_tensorize_many"](
            [str(i) for i in range(500)], True, 5, 4, executor=pool
        )

    assert len(rows) == 5
    # 5 usable rows at a 50% accept rate needs ~10 tasks; the guard allows a
    # small overshoot but nowhere near the 500-line split.
    assert len(seen) <= 5 * 2 + max(2 * 4, 16), f"submitted {len(seen)} of 500"


def test_early_stop_honours_need(pool_mod) -> None:
    """Once `need` rows exist, the remaining input must not be processed."""
    seen: List[int] = []
    lock = threading.Lock()

    def recording(args):
        raw_line, _pack = args
        with lock:
            seen.append(int(raw_line))
        time.sleep(0.001)
        return f"row{raw_line}", {}, {}

    _install(pool_mod, recording)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows, _drops, _errors, _info = pool_mod["_tensorize_many"](
            [str(i) for i in range(500)], True, 5, 2, executor=pool
        )

    assert len(rows) == 5
    # Every task yields a row, so nothing needs to be made up for drops and the
    # run stops at exactly need + guard, guard = max(2*workers, 16) = 16 here.
    # Exact, not "<=": the budget is charged against SUBMITTED-but-not-APPLIED
    # (submit - nxt). Charging it against in-flight tasks alone let one extra
    # task per applied row slip through and overshot by need (22 for need=5).
    assert len(seen) == 5 + max(2 * 2, 16), (
        f"need=5 processed {len(seen)} lines, expected exactly need + guard"
    )
    assert len(seen) < 500 / 10, "budget cap did not stop the run early"


def test_budget_counts_buffered_results_not_just_inflight(pool_mod) -> None:
    """A finished-but-unapplied result still occupies a budget slot.

    Regression guard for the exact off-by-need the window cap had: when the cap
    tracked in-flight tasks only, every applied row freed a slot that was then
    refilled, so ``need=5`` submitted 22 tasks. Here completion is made fast and
    submission is observed directly.
    """
    submitted: List[int] = []
    lock = threading.Lock()

    def instant(args):
        raw_line, _pack = args
        with lock:
            submitted.append(int(raw_line))
        return f"row{raw_line}", {}, {}

    _install(pool_mod, instant)
    need, workers = 4, 3
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows, _d, _e, _i = pool_mod["_tensorize_many"](
            [str(i) for i in range(400)], True, need, workers, executor=pool
        )

    assert len(rows) == need
    assert len(submitted) == need + max(2 * workers, 16)


def test_drops_and_errors_are_aggregated(pool_mod) -> None:
    """The audit counters must survive the trip back from the pool."""

    def messy(args):
        raw_line, _pack = args
        i = int(raw_line)
        if i % 2:
            return None, {"match_missed": 1}, {}
        return f"row{i}", {"absent_from_gt": 2, "match_missed": 3}, {"ValueError": 1}

    _install(pool_mod, messy)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows, drops, errors, _info = pool_mod["_tensorize_many"](
            [str(i) for i in range(10)], True, None, 3, executor=pool
        )

    assert rows == [f"row{i}" for i in range(0, 10, 2)]
    # 5 kept rows contribute 2 each; the 5 dropped rows contribute 0.
    assert drops["absent_from_gt"] == 10
    # 5 dropped x1 + 5 kept x3.
    assert drops["match_missed"] == 5 + 15
    # Only the kept rows report the error.
    assert errors["ValueError"] == 5


def test_single_process_path_needs_no_pool(pool_mod) -> None:
    """workers=1 must fall through to the inline loop, in order."""
    _install(pool_mod, lambda args: (f"row{args[0]}", {}, {}))
    rows, _drops, _errors, info = pool_mod["_tensorize_many"](
        [str(i) for i in range(7)], True, None, 1, executor=None
    )
    assert rows == [f"row{i}" for i in range(7)]
    assert info["pooled"] is False
    assert info["workers"] == 1


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_matches_a_sequential_reference_with_random_drops(pool_mod, seed) -> None:
    """Differential test against the obvious sequential implementation.

    Randomised accept/drop patterns and randomised sleeps (so completion order
    is scrambled) must still produce exactly the sequential result: same rows in
    the same order, same drop and error totals. This is the check that would
    have caught an off-by-one in the reorder buffer or a lost audit counter,
    which are the two ways this restructure could silently corrupt a split.
    """
    import random

    rng = random.Random(seed)
    n = rng.randint(1, 60)
    need = rng.choice([None, rng.randint(1, n)])
    lines = [str(i) for i in range(n)]
    accept = [rng.random() < 0.65 for _ in range(n)]
    drop_key = "absent_from_gt"
    delay = [rng.random() * 0.004 for _ in range(n)]   # scrambles completion order

    def task(args):
        raw_line, _pack = args
        i = int(raw_line)
        time.sleep(delay[i])
        if not accept[i]:
            return None, {drop_key: 1}, {}
        return f"row{i}", {drop_key: 3}, {}

    # --- reference: strictly sequential ---
    ref_rows, ref_drops = [], Counter()
    for line in lines:
        row, drops, _err = task((line, True))
        for k, v in drops.items():
            ref_drops[k] += v
        if row is not None and (need is None or len(ref_rows) < need):
            ref_rows.append(row)
        if need is not None and len(ref_rows) >= need:
            break

    # --- actual: the reorder-buffered pool path ---
    _install(pool_mod, task)
    with ThreadPoolExecutor(max_workers=rng.randint(2, 6)) as pool:
        rows, drops, errors, _info = pool_mod["_tensorize_many"](
            lines, True, need, 4, executor=pool
        )

    assert rows == ref_rows
    assert dict(drops) == dict(ref_drops)
    assert not errors


# ---------------------------------------------------------------------------
# Static pins on the cell text: the parts that cannot be unit-tested here
# ---------------------------------------------------------------------------

def test_worker_parses_the_raw_line_inside_the_child() -> None:
    """The task must take JSONL TEXT and parse it itself.

    Parsing in the parent is what made the whole corpus parse single-threaded
    before any fork happened.
    """
    src = _tensorize_cell_source()
    assert "raw_line, pack_multi = args" in src
    assert "d = json.loads(raw_line)" in src


def test_rows_are_serialized_in_the_child_not_the_parent() -> None:
    """The worker returns a JSON string; the parent only joins and writes."""
    src = _tensorize_cell_source()
    assert "return json.dumps(_serialize(row)), drops, errors" in src
    # Parent-side write must be a plain join, with no per-row json.dumps.
    assert 'fh.write("\\n".join(out_rows))' in src
    assert "fh.write(json.dumps(r)" not in src


def test_one_pool_is_created_outside_the_split_loop() -> None:
    """A per-split fork re-warmed the tokenizer for every split."""
    src = _tensorize_cell_source()
    pool_pos = src.index("_shared_ex = ProcessPoolExecutor")
    loop_pos = src.index("for (in_name, pack_multi), out_names in groups.items():")
    assert pool_pos < loop_pos, "the shared pool must be created before the loop"
    assert "executor=_shared_ex" in src


def test_fork_fallback_is_announced_not_silent() -> None:
    """A quiet degrade to 1 vCPU is the 'it just seems slow' failure mode."""
    src = _tensorize_cell_source()
    assert "fork context unavailable" in src
    assert "SINGLE-PROCESS" in src


def test_task_forwards_max_tools_and_jitter_to_the_tensorizer() -> None:
    """R0.1/R1c: the notebook must actually USE the new capacity knobs.

    The repo-side defaults are useless if the generated cell keeps calling
    ``tensorize_sample_lean`` without them -- that is exactly how a repo fix
    fails to reach the notebook.
    """
    src = _tensorize_cell_source()
    assert "max_tools=MAX_TOOLS" in src
    assert "jitter=JITTER_PACK_TOOLS" in src


def test_render_T_histogram_crosses_the_process_boundary() -> None:
    """R9: a fork child cannot mutate a parent dict, so T rides the drops channel."""
    src = _tensorize_cell_source()
    assert 'f"render_T={int(_t)}"' in src
    # The parent must decode it into a real histogram, not print raw channel keys.
    assert "render_hist" in src
    assert '"render_T_hist"' in src
    assert 'k.startswith("render_T=")' in src


def test_truncation_and_jitter_are_counted_not_swallowed() -> None:
    """R0.1: dropped native tools must be counted; R1c: jittered rows too."""
    src = _tensorize_cell_source()
    assert "tool_truncated_rows" in src
    assert "jittered_rows" in src
    assert "[!]" in src and "DROPPED from the prompt" in src


BUILDER = Path(__file__).resolve().parents[1] / "scripts" / "build_lean_v2_colab_notebook.py"


def test_part3_forensics_surfaces_render_T_from_persisted_features() -> None:
    """R9: the Part-3 tables must report the prompt layout, not just hit rates.

    ``preds_eval.jsonl`` records already carry ``features`` (``_features_detail``
    keeps every key but ``recent_selected``/``tool_scores_all``), and the resolver
    sets ``features["render_T"]`` on every resolve path, so the forensics cell can
    histogram it without re-running inference.
    """
    src = BUILDER.read_text(encoding="utf-8")
    assert "PROMPT LAYOUT (render_T)" in src
    assert 'feats.get("render_T")' in src
    assert "single-class there" in src  # the T=1 vacuity warning must survive
    assert "recent_budget_effective" in src
    # It must read the persisted features, never invent a T when the key is absent.
    assert "render_T not reported by this resolver build" in src
    assert "from collections import Counter, defaultdict" in src