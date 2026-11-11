"""Tests for the notebook build-time cell guard.

``scripts/build_lean_colab_notebook.py`` emits each cell as its own Jupyter
scope. ``nbformat.validate()`` only checks the JSON envelope, so a cell that
*uses* a name it never imports ships to Colab and raises ``NameError`` on the
first run — which is exactly how the ``cand_hist: Dict[int, int]`` bug reached
the notebook (``typing`` was imported in a later cell, invisible across scopes).

These tests pin the guard's semantics against the cases that matter, in
particular that it models Jupyter scoping correctly rather than using a regex:

* a name bound by an EARLIER cell resolves (sequential execution),
* a name bound by a LATER cell does not,
* a genuine undefined name inside a function body is caught.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_lean_colab_notebook.py"


def _load_guard():
    """Import the guard straight out of the build script without running it."""
    src = BUILD_SCRIPT.read_text(encoding="utf-8")
    start = src.index("def _cell_source")
    end = src.rindex("_guard_code_cells(cells)")
    ns: Dict[str, Any] = {}
    exec(compile(src[start:end], str(BUILD_SCRIPT), "exec"), ns)
    return ns["_guard_code_cells"]


@pytest.fixture(scope="module")
def guard():
    return _load_guard()


def cell(source: str) -> Dict[str, Any]:
    return {"cell_type": "code", "source": source}


def _blocks(guard, cells: List[Dict[str, Any]]) -> bool:
    try:
        guard(cells)
    except SystemExit:
        return True
    return False


# ---------------------------------------------------------------------------
# The bug this guard exists to prevent
# ---------------------------------------------------------------------------

def test_catches_the_original_dict_annotation_bug(guard) -> None:
    """The exact regression: annotate with Dict, import typing in a later cell."""
    assert _blocks(guard, [
        cell("cand_hist: Dict[int, int] = {}\n"),
        cell("from typing import Dict\n"),
    ]), "guard must reject a name imported only in a LATER cell"


def test_accepts_the_import_in_the_using_cell(guard) -> None:
    assert not _blocks(guard, [
        cell("from typing import Dict\ncand_hist: Dict[int, int] = {}\n"),
    ])


# ---------------------------------------------------------------------------
# Jupyter execution semantics
# ---------------------------------------------------------------------------

def test_earlier_cell_bindings_resolve(guard) -> None:
    assert not _blocks(guard, [cell("X = 1\n"), cell("print(X)\n")])


def test_later_cell_bindings_do_not_resolve(guard) -> None:
    assert _blocks(guard, [cell("print(Y)\n"), cell("Y = 2\n")])


def test_builtins_resolve(guard) -> None:
    assert not _blocks(guard, [cell("print(len([1, 2]), sorted({3, 1}))\n")])


def test_closure_over_cell_level_global_resolves(guard) -> None:
    # ``json`` is bound at module level of the same cell and used inside a def;
    # Python resolves it at call time, so this is legal.
    assert not _blocks(guard, [
        cell("import json\ndef f(x):\n    return json.dumps(x)\nf({})\n"),
    ])


def test_undefined_name_inside_function_is_caught(guard) -> None:
    assert _blocks(guard, [cell("def f():\n    return nope_undefined\n")])


def test_lambda_parameter_is_not_flagged(guard) -> None:
    assert not _blocks(guard, [cell("g = lambda a, b: a + b\nprint(g(1, 2))\n")])


def test_comprehension_variable_does_not_leak(guard) -> None:
    """Python 3 comprehension scopes are real scopes; ``c`` is undefined after."""
    assert _blocks(guard, [cell("found = [c for c in range(3)]\nprint(c)\n")])


def test_class_body_and_method_resolve(guard) -> None:
    assert not _blocks(guard, [
        cell(
            "class A:\n"
            "    K = 3\n"
            "    def m(self, x):\n"
            "        return x + self.K\n"
            "print(A().m(1))\n"
        ),
    ])


# ---------------------------------------------------------------------------
# Non-Python cells and syntax errors
# ---------------------------------------------------------------------------

def test_ipython_cells_are_skipped(guard) -> None:
    assert not _blocks(guard, [cell("!pip install -q foo\n")])
    assert not _blocks(guard, [cell("%matplotlib inline\n")])


def test_syntax_error_blocks(guard) -> None:
    assert _blocks(guard, [cell("def f(:\n    pass\n")])


def test_markdown_cells_are_ignored(guard) -> None:
    assert not _blocks(guard, [{"cell_type": "markdown", "source": "not Python at all"}])


# ---------------------------------------------------------------------------
# T10: the LoRA ablation cell must route knobs to the TRAINING section
# ---------------------------------------------------------------------------

ABLATION_CELL_MARKER = "# ---- T10: LoRA vs full-FT ablation"


def _ablation_cell_source() -> str:
    """Load the T10 ablation cell out of the generated notebook, if present."""
    import json

    nb_path = Path(BUILD_SCRIPT).resolve().parents[1] / "notebooks" / "spancall_lean_train_eval_test.ipynb"
    if not nb_path.exists():
        pytest.skip("generated notebook missing; run scripts/build_lean_colab_notebook.py")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for c in nb.get("cells", []):
        src = c.get("source", "")
        src = src if isinstance(src, str) else "".join(src)
        if src.startswith(ABLATION_CELL_MARKER):
            return src
    pytest.skip("T10 ablation cell not found in the generated notebook")


def _run_ablation_cell(tmp_path, monkeypatch) -> Dict[str, Any]:
    """Execute the ablation cell with the flag OFF and return its namespace."""
    import json

    import yaml

    src = _ablation_cell_source()
    cfg_path = tmp_path / "colab.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "x", "seed": 42},
                "model": {"max_seq_len": 4096},
                "training": {"gradient_accumulation_steps": 4},
                "optimization": {"num_epochs": 5},
                "data": {"pack_tools": 4, "fraction": 0.2},
            }
        ),
        encoding="utf-8",
    )
    ns: Dict[str, Any] = {
        "RESULTS_DIR": str(tmp_path),
        "TENSOR_DIR": str(tmp_path / "t"),
        "DEVICE": "cpu",
        "SEED": 42,
        "NUM_EPOCHS": 5,
        "TRAIN_FRACTION": 0.2,
        "PACK_TOOLS": 4,
        "CONFIG_PATH": str(cfg_path),
    }
    exec(compile(src, "<t10 ablation cell>", "exec"), ns)
    assert ns["RUN_LORA_ABLATION"] is False, "the ablation must be opt-in"
    return ns


def test_lora_ablation_cell_is_default_off_and_writes_training_section(tmp_path) -> None:
    """T10/T11 regression: LoRA knobs under ``model:`` are a silent no-op."""
    import yaml

    ns = _run_ablation_cell(tmp_path, None)
    base = yaml.safe_load(open(ns["CONFIG_PATH"]))

    lora_path = ns["_write_arm_config"](base, True, "arm_lora", str(tmp_path / "out"))
    written = yaml.safe_load(open(lora_path))

    # The knobs must land in `training`, the only section TrainingConfig reads.
    assert written["training"]["use_lora"] is True
    assert written["training"]["lora_r"] == ns["LORA_R"]
    assert written["training"]["lora_alpha"] == ns["LORA_ALPHA"]
    assert written["training"]["lora_lr"] == ns["LORA_LR"]
    assert (written.get("model") or {}).get("use_lora") is None, (
        "use_lora written under `model:` would be a silent no-op"
    )

    # And they must actually be READ by the real loader.
    from spancall.train.config import TrainingConfig

    cfg = TrainingConfig.from_yaml(lora_path)
    assert cfg.use_lora is True
    assert cfg.lora_r == ns["LORA_R"]
    assert cfg.lora_lr == ns["LORA_LR"]

    ft_path = ns["_write_arm_config"](base, False, "arm_ft", str(tmp_path / "out"))
    assert TrainingConfig.from_yaml(ft_path).use_lora is False


def test_lora_ablation_arms_differ_only_in_use_lora(tmp_path) -> None:
    """The A/B is only interpretable if every other knob is identical."""
    import yaml

    ns = _run_ablation_cell(tmp_path, None)
    base = yaml.safe_load(open(ns["CONFIG_PATH"]))
    ft = yaml.safe_load(open(ns["_write_arm_config"](base, False, "ft", str(tmp_path))))
    lora = yaml.safe_load(open(ns["_write_arm_config"](base, True, "lora", str(tmp_path))))

    assert ft["training"]["use_lora"] is False
    assert lora["training"]["use_lora"] is True
    for section in ("optimization", "data", "model"):
        assert ft.get(section) == lora.get(section), f"{section} must not differ between arms"
    assert ft["optimization"]["num_epochs"] == ns["LORA_ABLATION_EPOCHS"]
    assert lora["optimization"]["num_epochs"] == ns["LORA_ABLATION_EPOCHS"]


def test_lora_ablation_summary_survives_a_missing_arm(tmp_path) -> None:
    """An arm that crashed mid-run must report n/a, not raise or invent zeros."""
    import math

    ns = _run_ablation_cell(tmp_path, None)
    summary = ns["_arm_summary"](str(tmp_path / "absent"), "absent")

    assert summary["lora_param_count"] is None
    for key in ("final_train_loss", "dev_span_acc", "dev_exact_call", "dev_tool_acc"):
        assert math.isnan(summary[key]), f"{key} must be NaN ('never measured'), not 0.0"