"""The LLM-baseline notebook must have a builder, and the builder must win.

``notebooks/spancall_lean_v3_2_llm_baseline.ipynb`` is the notebook that replaces
the ``B1a`` regex straw man in the paper, so its protocol is reviewable Python
(``scripts/build_llm_baseline_notebook.py``) rather than a JSON cell blob. These
tests lock the same contract as ``tests/test_v3_2_train_eval_builder.py``:

1. the builder parses and declares ``CELLS``;
2. ``--check`` passes on the artifact on disk and **fails when a cell drifts**;
3. a rebuild is lossless for cell text, ids, metadata *and executed outputs*.

(3) is load-bearing here in a way it is not for the other notebooks: this file is
matched by ``notebooks/*_llm_*.ipynb`` in ``.gitignore``. The notebook is never
committed, so after a paid run the executed outputs are the *only* copy of what
the hosted model actually returned. A naive builder that re-emitted blank cells
would delete the evidence, and git would show nothing.

The build-time guard is asserted too: it is the reason the setup section is split
into §0a (``!git clone`` / ``%pip``, skipped by the guard) and §0b (plain Python,
checked). Without the split, every cell after it would report an unbound
``PROJECT_ROOT``.

Tests that need the artifact skip when it is absent (it is gitignored), matching
``tests/test_notebook_v3_fixes.py``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_llm_baseline_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "spancall_lean_v3_2_llm_baseline.ipynb"

_skip_no_builder = pytest.mark.skipif(
    not BUILDER.is_file(), reason="LLM baseline builder missing"
)
_skip_no_notebook = pytest.mark.skipif(
    not NOTEBOOK.is_file(),
    reason=f"{NOTEBOOK.name} not built; run scripts/build_llm_baseline_notebook.py",
)


def _load_builder() -> Any:
    """Import the builder module without running ``__main__``.

    Deliberately from its real path, never from a temp copy: ``build()`` resolves
    the shared cell guard relative to ``__file__``, so a copied module would have
    no ``build_lean_v2_6_colab_notebook.py`` next to it. Tests that need to write
    somewhere else pass ``out_path=`` instead.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_build_llm_baseline", BUILDER)
    assert spec and spec.loader, f"cannot load {BUILDER}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_fingerprint(path: Path) -> List[Dict[str, Any]]:
    """What a rebuild must preserve, per cell, independent of JSON formatting."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "cell_type": cell["cell_type"],
            "source": "".join(cell["source"]).rstrip("\n"),
            "id": cell.get("id"),
            "execution_count": cell.get("execution_count"),
            "outputs": cell.get("outputs") or [],
        }
        for cell in nb["cells"]
    ]


def _fake_output(text: str) -> Dict[str, Any]:
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": [text],
    }


@_skip_no_builder
def test_builder_exists_and_parses() -> None:
    """The source of truth for the LLM-baseline notebook is checked in as Python."""
    import ast

    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "build" in names, "builder has no build() entry point"

    # ``CELLS`` is annotated (``CELLS: List[Tuple[str, str]] = [...]``), so it is
    # an AnnAssign, not an Assign — check both or the assertion is vacuous.
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    assert "CELLS" in assigned, "builder must declare CELLS so the notebook text is readable"


@_skip_no_builder
def test_builder_has_check_mode() -> None:
    """A builder without ``--check`` cannot detect drift, which is its whole job."""
    import inspect

    module = _load_builder()
    assert "check" in inspect.signature(module.build).parameters, (
        "build() must accept check=True so CI can assert the artifact matches"
    )


@_skip_no_builder
def test_cells_are_kind_source_pairs() -> None:
    """``CELLS`` uses the repo's ``(kind, source)`` shape, not ad-hoc dicts."""
    module = _load_builder()
    assert module.CELLS, "CELLS is empty"
    for entry in module.CELLS:
        assert isinstance(entry, tuple) and len(entry) == 2, entry
        kind, src = entry
        assert kind in (module.CODE, module.MD), kind
        assert isinstance(src, str) and src.strip(), "empty cell source"


@_skip_no_builder
def test_setup_is_split_so_the_guard_can_see_the_config() -> None:
    """§0a uses IPython magics; §0b must be plain Python and re-derive the root.

    The shared guard skips any cell containing ``!``/``%`` lines, so a single
    combined setup cell would make ``PROJECT_ROOT`` invisible to every later
    cell's name check.
    """
    module = _load_builder()
    code = [src for kind, src in module.CELLS if kind == module.CODE]

    def _magic_lines(src: str) -> List[str]:
        """Lines the shared guard treats as IPython, i.e. ``!``/``%`` at column 0."""
        return [
            line for line in src.splitlines()
            if line.lstrip().startswith(("!", "%"))
        ]

    boot = [s for s in code if _magic_lines(s)]
    assert len(boot) == 1, "expected exactly one bootstrap cell with IPython magics"
    assert any(line.lstrip().startswith("%pip") for line in _magic_lines(boot[0]))
    assert "!git clone" in boot[0], "the bootstrap cell should clone the repo"

    config = [s for s in code if "EVAL_LIMIT = " in s]
    assert len(config) == 1, "expected exactly one configuration cell"
    assert not _magic_lines(config[0]), (
        "the configuration cell must stay pure Python: one magic line anywhere in "
        "it makes the guard skip the whole cell, and then every later cell is "
        "reported as using an unbound PROJECT_ROOT"
    )
    assert "PROJECT_ROOT = Path.cwd().resolve()" in config[0], (
        "the configuration cell must re-derive PROJECT_ROOT: §0a is skipped by "
        "the guard, so its bindings are not visible to later cells"
    )


@_skip_no_builder
@_skip_no_notebook
def test_builder_check_passes_on_a_fresh_notebook() -> None:
    """``--check`` against the artifact on disk must exit 0."""
    module = _load_builder()
    module.build(check=True)  # raises SystemExit if stale


@_skip_no_builder
@_skip_no_notebook
def test_builder_check_fails_when_the_notebook_drifts(tmp_path: Path) -> None:
    """A cell edited outside the builder must be reported, not tolerated."""
    target = tmp_path / NOTEBOOK.name
    drifted = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in drifted["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = ["# edited outside the builder\n"] + cell["source"]
            break
    target.write_text(json.dumps(drifted), encoding="utf-8")

    module = _load_builder()
    with pytest.raises(SystemExit) as exc:
        module.build(out_path=target, check=True)
    assert "stale" in str(exc.value), "drift must be reported as stale, not swallowed"


@_skip_no_builder
@_skip_no_notebook
def test_rebuild_is_lossless(tmp_path: Path) -> None:
    """A rebuild must reproduce the executed artifact exactly.

    The notebook is gitignored, so executed outputs exist nowhere else. This test
    seeds an output + ``execution_count`` into a copy and asserts they survive a
    rebuild, which is what a run of this notebook would have produced.
    """
    target = tmp_path / NOTEBOOK.name
    shutil.copy2(NOTEBOOK, target)

    seeded = json.loads(target.read_text(encoding="utf-8"))
    for cell in seeded["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["execution_count"] = 3
        cell["outputs"] = [_fake_output("seeded evidence\n")]
        break
    target.write_text(json.dumps(seeded), encoding="utf-8")
    before = _cell_fingerprint(target)

    module = _load_builder()
    module.build(out_path=target)
    after = _cell_fingerprint(target)

    assert after == before, "rebuild changed cell text, ids or executed outputs"
    assert any(c["outputs"] for c in after), "rebuilt notebook lost the executed output"
    assert any(c["execution_count"] == 3 for c in after), "execution_count was dropped"


@_skip_no_builder
def test_outputs_are_matched_by_content_not_position() -> None:
    """An edited cell must not inherit another cell's outputs on rebuild.

    Keying records on the cell index would graft output A onto cell B as soon as a
    cell is inserted or reordered — silently, and only visible as a wrong number
    in the paper.
    """
    module = _load_builder()
    kind, src = module.CELLS[1]
    other_kind, other_src = module.CELLS[2]
    assert kind == module.CODE and other_kind == module.CODE

    record = {
        "cell_type": "code",
        "source": [src],
        "execution_count": 1,
        "outputs": [_fake_output("from cell 1\n")],
        "metadata": {},
    }
    # Rebuild with a CELLS list whose first code cell is a *different* cell.
    original = module.CELLS
    try:
        module.CELLS = [(module.MD, "# title"), (module.CODE, other_src), (module.CODE, src)]
        cells = module._cell_objects({src.rstrip("\n"): [record]}, [])
    finally:
        module.CELLS = original

    assert cells[1]["outputs"] == [], "an unrelated cell inherited executed output"
    assert [o["text"] for o in cells[2]["outputs"]] == [["from cell 1\n"]], (
        "the record was not reattached to the cell whose source it matches"
    )


@_skip_no_builder
def test_guard_rejects_an_undeclared_name() -> None:
    """A cell that reads a name nothing binds must fail the build.

    Asserted against the guard directly, not through ``build(check=True)``: the
    latter would also exit non-zero because the on-disk artifact is stale, which
    would make this test pass for the wrong reason.
    """
    module = _load_builder()
    guard = module._guard()

    with pytest.raises(SystemExit):
        guard([new_cell("print(UNDECLARED_SETTING_SETTING)\n")])

    # And a name bound by an EARLIER cell is accepted, so the test is not just
    # asserting that the guard rejects everything.
    guard(
        [
            new_cell("SETTING = 2000\n"),
            new_cell("print(SETTING)\n"),
        ]
    )


def new_cell(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "source": source,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    }