#!/usr/bin/env python3
"""Offline smoke test for the `DOWNLOAD_RESULTS` block of
`notebooks/spancall_lean_v3_2_train_eval.ipynb` (§5, the final eval cell).

This notebook is HAND-WRITTEN - there is no builder and no build-time cell
guard - so unlike the other two v3.2 notebooks its code is never compiled until
someone runs it. This is the only check that the download block executes.

The block is sliced out of the shipped cell and run verbatim with
`google.colab.files` stubbed. What it must get right:

  * ONE `files.download()` call. Chrome blocks the 2nd+ automatic download, so a
    per-file loop would silently deliver only the first artifact.
  * `hf_export/` EXCLUDED. §7 already pushed those weights to the Hub; re-shipping
    them turns a zip of results into a multi-GB download that times out.
  * the zip written NEXT TO `results_lean_v3_2/`, never inside it - otherwise the
    walk would archive the archive it is writing.
  * a re-run REBUILDS. An append would duplicate every entry and grow the file
    on each run.
  * relative paths preserved, so `agentic/agentic_mode_table.json` unpacks back
    into the same layout.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch"))

import nbformat  # noqa: E402

# Shared stubs: one definition of "pretend to be Colab" across all transport tests.
from smoke_llm_baseline_upload import _fake_colab, _no_colab, _StubFiles  # noqa: E402

NB = REPO / "notebooks" / "spancall_lean_v3_2_train_eval.ipynb"

_ANCHOR = "if DOWNLOAD_RESULTS:"


def _block_source() -> str:
    """The shipped download block, sliced from the cell that contains it."""
    nb = nbformat.read(NB, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source)
        if _ANCHOR in src:
            return src[src.index(_ANCHOR):]
    raise AssertionError(
        f"no cell in {NB.name} contains `{_ANCHOR}`; the download block is missing"
    )


def _seed(results_dir: Path) -> None:
    (results_dir / "agentic").mkdir(parents=True, exist_ok=True)
    (results_dir / "agentic" / "agentic_mode_table.json").write_text("{}")
    (results_dir / "preds_eval.jsonl").write_text('{"sample_id": "x"}\n')
    (results_dir / "tensorize_stats.json").write_text("{}")
    (results_dir / "nested").mkdir()
    (results_dir / "nested" / "deep.json").write_text("{}")
    # The Hub export: big, already published, and must not be re-shipped.
    (results_dir / "hf_export").mkdir()
    (results_dir / "hf_export" / "model.safetensors").write_bytes(b"x" * 1024)
    (results_dir / "hf_export" / "nested").mkdir()
    (results_dir / "hf_export" / "nested" / "big.bin").write_bytes(b"y" * 1024)


def _run(results_dir: Path, *, enabled: bool) -> tuple[str, _StubFiles]:
    ns = {"RESULTS_DIR": str(results_dir), "DOWNLOAD_RESULTS": enabled, "os": os}
    files_obj = _StubFiles()
    buf = io.StringIO()
    with _fake_colab(files_obj):
        with contextlib.redirect_stdout(buf):
            exec(compile(_block_source(), "<train_eval_download>", "exec"), ns)
    return buf.getvalue(), files_obj


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_te_download_"))
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    try:
        results_dir = tmp / "results_lean_v3_2"
        _seed(results_dir)
        zip_path = tmp / "results_lean_v3_2.zip"

        # ================================================================== #
        # 1. flag off -> inert
        # ================================================================== #
        print("\n[case] DOWNLOAD_RESULTS = False")
        out, files_obj = _run(results_dir, enabled=False)
        check(files_obj.n_download == 0, "flag off -> nothing downloaded")
        check(not zip_path.exists(), "flag off -> no zip created")

        # ================================================================== #
        # 2. flag on outside Colab -> no crash, no zip
        # ================================================================== #
        print("\n[case] DOWNLOAD_RESULTS = True, NOT in Colab")
        ns = {"RESULTS_DIR": str(results_dir), "DOWNLOAD_RESULTS": True, "os": os}
        buf = io.StringIO()
        with _no_colab():
            with contextlib.redirect_stdout(buf):
                exec(compile(_block_source(), "<train_eval_download>", "exec"), ns)
        check(not zip_path.exists(), "outside Colab -> no zip created")
        check("not Colab" in buf.getvalue(), "outside Colab -> the reason is printed")

        # ================================================================== #
        # 3. flag on in Colab -> one download, right contents
        # ================================================================== #
        print("\n[case] DOWNLOAD_RESULTS = True, in Colab")
        out, files_obj = _run(results_dir, enabled=True)
        check(files_obj.n_download == 1,
              f"EXACTLY ONE download call (got {files_obj.n_download}) - Chrome blocks the 2nd")
        check(len(files_obj.paths) == 1, "the one call carried exactly one path")
        check(files_obj.paths and files_obj.paths[0] == str(zip_path),
              "the zip sits NEXT TO results_lean_v3_2/, not inside it")
        check(zip_path.exists() and zipfile.is_zipfile(zip_path),
              "the downloaded artifact is a real zip")
        check(zip_path.stat().st_size > 0, "the zip is not empty")

        names = set(zipfile.ZipFile(zip_path).namelist())
        check("agentic/agentic_mode_table.json" in names,
              "kept agentic/agentic_mode_table.json")
        check("preds_eval.jsonl" in names, "kept the top-level preds")
        check("tensorize_stats.json" in names, "kept the top-level stats")
        check("nested/deep.json" in names, "preserved nested relative paths")
        check(not any(n.startswith("hf_export") for n in names),
              "EXCLUDED hf_export/ entirely")
        check(not any("model.safetensors" in n for n in names),
              "did not re-ship the published weights")
        check(not any(n.startswith("results_lean_v3_2/") for n in names),
              "no redundant top-directory prefix inside the archive")
        check(not any(n.endswith(".zip") for n in names),
              "the archive does not contain itself")

        # ================================================================== #
        # 4. re-run rebuilds rather than appends
        # ================================================================== #
        print("\n[case] second run with the previous zip still on disk")
        out2, files_obj2 = _run(results_dir, enabled=True)
        check(files_obj2.n_download == 1, "second run also makes exactly one download")
        entries = zipfile.ZipFile(zip_path).namelist()
        names2 = set(entries)
        check(names2 == names,
              "second run rebuilt the same set (nothing appended, nothing doubled)")
        # A written-in-append-mode archive would hold every name twice; the
        # entry COUNT versus the distinct-name count is what catches that.
        check(len(entries) == len(names2),
              f"second run left no duplicate entries ({len(entries)} entries, "
              f"{len(names2)} distinct)")

        print()
        if failures:
            print(f"FAILED: {len(failures)} check(s)")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("ALL TRAIN_EVAL DOWNLOAD CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())