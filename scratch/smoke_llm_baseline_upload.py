#!/usr/bin/env python3
"""Offline smoke test for the UPLOAD path of cell 3 in
`notebooks/spancall_lean_v3_2_llm_baseline.ipynb`.

`files.upload()` cannot be called outside Colab, so it is stubbed. Everything
else - the whitelist, the move, the sha256 provenance, `_find_existing_preds`,
`_validate_coverage` - is the SHIPPED cell 3, read out of the notebook. A stub
that let the cell drift from the shipped code would be worthless, so the source
is executed verbatim.

The failure this test exists to prevent: `files.upload()` writes into the CWD,
which after `os.chdir(PROJECT_ROOT)` is the git checkout - a directory
`_find_existing_preds` does NOT search. If the file is not moved, the cell prints
"Saved ...", finds nothing, and silently RE-INFERS, replacing the paper's
SpanCall column with a fresh one.

This test never chdir-s. `PROJECT_ROOT` is passed as a plain string and the
"uploaded" bytes are written under it directly, which is precisely where the
real widget would have put them.

Checks:
  1. A whitelisted upload is MOVED out of `PROJECT_ROOT` into `results_lean_v3_2/`.
  2. Its sha256 is recorded in `UPLOADED_PROVENANCE` and surfaced in
     `SPANCALL_SOURCE`, so the parity JSON can name the exact bytes scored.
  3. A non-whitelisted filename is refused and left alone.
  4. Flag off -> no upload widget, no move, no sha256.
  5. Flag on outside Colab -> no move, and the user is told why.
  6. A cancelled widget (empty payload, or an exception) does not kill the cell.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

NB = REPO / "notebooks" / "spancall_lean_v3_2_llm_baseline.ipynb"


# --------------------------------------------------------------------------- #
# shipped-source extraction (identical slicing to smoke_llm_baseline_reuse.py)
# --------------------------------------------------------------------------- #
def _limit_rows_from_notebook() -> str:
    """The REAL `_limit_rows` from cell 0, so this test cannot drift from it."""
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = "".join(code[0]["source"])
    start = src.index("def _limit_rows")
    return src[start:src.index("# Print shapes")]


def _cell3_source() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return "".join(code[3]["source"])


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _sub_row(sid: str, stratum: str = "F", tool: str = "get_weather") -> dict:
    return {
        "sample_id": sid,
        "stratum": stratum,
        "utterance": "weather please",
        "tools": [{"name": tool, "description": "d", "parameters": []}],
        "ground_truth_calls": [{"tool_name": tool, "arguments": {"city": "Hanoi"}}],
    }


def _pred(sid: str, hit: bool, tool: str, stratum: str = "F") -> dict:
    return {
        "sample_id": sid,
        "stratum": stratum,
        "hit": hit,
        "tool_name": tool if hit else None,
        "payload": {"city": "Hanoi"} if hit else None,
        "latency_ms": 12.5,
        "status": "hit" if hit else "abstain",
    }


def _pred_bytes(ids) -> bytes:
    return "".join(json.dumps(_pred(s, True, "get_weather")) + "\n" for s in ids).encode()


class _StubFiles:
    """Stand-in for `google.colab.files`; records what the cell asked for."""

    def __init__(self, payload=None, raises: Exception | None = None) -> None:
        self.payload = payload or {}
        self.raises = raises
        self.n_upload = 0
        self.n_download = 0
        self.paths: list[str] = []     # every path handed to `download()`, in order

    def upload(self):
        self.n_upload += 1
        if self.raises is not None:
            raise self.raises
        return dict(self.payload)

    def download(self, path):          # not used by cell 3; the download test reads `paths`
        self.n_download += 1
        self.paths.append(str(path))


@contextlib.contextmanager
def _fake_colab(files_obj):
    """`from google.colab import files` succeeds only inside this block."""
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files = files_obj
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules["google"] = google
    sys.modules["google.colab"] = colab
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@contextlib.contextmanager
def _no_colab():
    """`from google.colab import files` raises ImportError inside this block.

    A `None` entry in `sys.modules` makes the import raise ImportError rather
    than falling through to a real `google` package, which would otherwise make
    the "outside Colab" case depend on the developer's site-packages.
    """
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules["google.colab"] = None
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# --------------------------------------------------------------------------- #
# cell execution
# --------------------------------------------------------------------------- #
def _run_cell3(
    root: Path,
    split: str,
    subset_dir: Path,
    results_out: Path,
    *,
    env_flag: str,
    files_obj: _StubFiles | None,
    upload_to_cwd: dict[str, bytes] | None = None,
) -> tuple[dict, str]:
    """Exec the shipped cell 3; return (namespace, captured stdout).

    `upload_to_cwd` emulates the widget's side effect: `files.upload()` writes
    the bytes into the CWD, which is why the cell has to move them. It is passed
    separately from the stub's return value so a test can prove that the RETURN
    value is what drives the loop, not the side effect.

    `UPLOAD_V3_2_PREDS` MUST be set as an env var, not in `ns`: cell 3 reads the
    knob with `os.environ.get(...)` and rebinds the name, so an `ns` entry is
    overwritten before it is ever read. Setting the env var is also what a Colab
    user actually does, so this is the honest test of the documented switch.
    """
    for name, blob in (upload_to_cwd or {}).items():
        (root / name).write_bytes(blob)

    ns = {
        "PROJECT_ROOT": str(root),
        "RESULTS_DIR": str(results_out),
        "V3_2_RESULTS_DIR": None,          # exercises the in-repo default
        "EVAL_SPLITS": [split],
        "SUBSET_PATHS": {split: str(subset_dir / f"subset_{split}_clean.jsonl")},
        "LLM_LIMIT_FRACTION": 1.0,
        "EVAL_BATCH_SIZE": 32,
        "EVAL_WORKERS": 4,
        "os": os, "json": json,
    }
    exec(compile(_limit_rows_from_notebook(), "<_limit_rows>", "exec"), ns)

    saved_flag = os.environ.get("UPLOAD_V3_2_PREDS")
    os.environ["UPLOAD_V3_2_PREDS"] = env_flag
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(_cell3_source(), "<cell3>", "exec"), ns)
    finally:
        if saved_flag is None:
            os.environ.pop("UPLOAD_V3_2_PREDS", None)
        else:
            os.environ["UPLOAD_V3_2_PREDS"] = saved_flag
    return ns, buf.getvalue()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_upload_"))
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    try:
        subset_dir = tmp / "subsets"
        results_out = tmp / "results_lean_v3_2_llm"
        ids = ["h1", "h2", "h3"]
        _write_jsonl(
            subset_dir / "subset_hermes_eval_clean.jsonl",
            [_sub_row(s) for s in ids],
        )
        blob = _pred_bytes(ids)
        want_sha = hashlib.sha256(blob).hexdigest()

        # ================================================================== #
        # 1. whitelisted upload is moved into results_lean_v3_2/ + sha256
        # ================================================================== #
        print("\n[case] whitelisted upload, flag on, in Colab")
        root = tmp / "proj_upload"
        root.mkdir()
        files_obj = _StubFiles(
            payload={"preds_eval_twostep.jsonl": blob, "notes.md": b"# not a preds file\n"}
        )
        with _fake_colab(files_obj):
            ns, out = _run_cell3(
                root, "hermes_eval", subset_dir, results_out,
                env_flag="1", files_obj=files_obj,
                upload_to_cwd={"preds_eval_twostep.jsonl": blob, "notes.md": b"x"},
            )

        v32 = root / "results_lean_v3_2"
        check(files_obj.n_upload == 1, "the widget was shown exactly once")
        check(not (root / "preds_eval_twostep.jsonl").exists(),
              "uploaded preds were MOVED out of PROJECT_ROOT (the git checkout)")
        check((v32 / "preds_eval_twostep.jsonl").exists(),
              "uploaded preds landed in results_lean_v3_2/ where the lookup searches")
        check((v32 / "preds_eval_twostep.jsonl").read_bytes() == blob,
              "the moved bytes are the uploaded bytes")
        check(not (v32 / "notes.md").exists(),
              "a non-whitelisted filename was NOT copied in")
        check((root / "notes.md").exists(),
              "the non-whitelisted file was left alone, not deleted")
        check("IGNORED notes.md" in out, "the refused name is reported loudly")

        prov = ns["UPLOADED_PROVENANCE"]
        check(prov.get("hermes_eval", {}).get("sha256") == want_sha,
              "UPLOADED_PROVENANCE recorded the true sha256 of the bytes scored")
        source = ns["SPANCALL_SOURCE"]["hermes_eval"]
        check(source.startswith("reused:"),
              f"hermes marked reused, not recomputed -> {source[:48]}...")
        check(f"sha256={want_sha[:12]}" in source,
              "SPANCALL_SOURCE carries the sha256 prefix into llm_parity.json")
        check(len(ns["SPANCALL_PREDS"]["hermes_eval"]) == len(ids),
              "the uploaded rows are what got scored")

        # ================================================================== #
        # 2. flag off -> no widget at all
        # ================================================================== #
        print("\n[case] flag off, file already on disk")
        root = tmp / "proj_off"
        root.mkdir()
        (root / "results_lean_v3_2").mkdir()
        _write_jsonl(root / "results_lean_v3_2" / "preds_eval_twostep.jsonl",
                     [_pred(s, True, "get_weather") for s in ids])
        files_obj = _StubFiles()
        with _fake_colab(files_obj):
            ns, out = _run_cell3(
                root, "hermes_eval", subset_dir, results_out,
                env_flag="0", files_obj=files_obj,
            )
        check(files_obj.n_upload == 0, "flag off -> upload widget never shown")
        check("[upload] off" in out, "flag off -> says so")
        check(ns["UPLOADED_PROVENANCE"] == {}, "flag off -> no provenance recorded")
        check("sha256=" not in ns["SPANCALL_SOURCE"]["hermes_eval"],
              "a plain on-disk reuse is not mislabelled as an upload")

        # ================================================================== #
        # 3. flag on outside Colab -> no widget, no move
        # ================================================================== #
        print("\n[case] flag on, NOT in Colab")
        root = tmp / "proj_nocolab"
        root.mkdir()
        (root / "results_lean_v3_2").mkdir()
        _write_jsonl(root / "results_lean_v3_2" / "preds_eval_twostep.jsonl",
                     [_pred(s, True, "get_weather") for s in ids])
        files_obj = _StubFiles(payload={"preds_eval_twostep.jsonl": blob})
        with _no_colab():
            ns, out = _run_cell3(
                root, "hermes_eval", subset_dir, results_out,
                env_flag="1", files_obj=files_obj,
            )
        check(files_obj.n_upload == 0,
              "outside Colab the widget is never called (it would raise)")
        check("not Colab" in out, "outside Colab the user is told why nothing happened")
        check(ns["UPLOADED_PROVENANCE"] == {}, "outside Colab no provenance recorded")
        check(ns["SPANCALL_SOURCE"]["hermes_eval"].startswith("reused:"),
              "the existing on-disk file is still reused")

        # ================================================================== #
        # 4. cancelled widget (empty payload) -> cell survives
        # ================================================================== #
        print("\n[case] widget cancelled (returns {})")
        root = tmp / "proj_cancel"
        root.mkdir()
        (root / "results_lean_v3_2").mkdir()
        _write_jsonl(root / "results_lean_v3_2" / "preds_eval_twostep.jsonl",
                     [_pred(s, True, "get_weather") for s in ids])
        files_obj = _StubFiles(payload={})
        with _fake_colab(files_obj):
            ns, _ = _run_cell3(
                root, "hermes_eval", subset_dir, results_out,
                env_flag="1", files_obj=files_obj,
            )
        check(files_obj.n_upload == 1, "the widget was shown")
        check(ns["UPLOADED_PROVENANCE"] == {}, "cancel moved nothing and recorded nothing")
        check(ns["SPANCALL_SOURCE"]["hermes_eval"].startswith("reused:"),
              "cancel falls back to the on-disk file instead of re-inferring")

        # ================================================================== #
        # 5. widget raises -> caught, cell survives
        # ================================================================== #
        print("\n[case] widget raises")
        root = tmp / "proj_raises"
        root.mkdir()
        (root / "results_lean_v3_2").mkdir()
        _write_jsonl(root / "results_lean_v3_2" / "preds_eval_twostep.jsonl",
                     [_pred(s, True, "get_weather") for s in ids])
        files_obj = _StubFiles(raises=RuntimeError("frontend went away"))
        with _fake_colab(files_obj):
            ns, out = _run_cell3(
                root, "hermes_eval", subset_dir, results_out,
                env_flag="1", files_obj=files_obj,
            )
        check("cancelled" in out, "the exception is reported, not swallowed silently")
        check(ns["SPANCALL_SOURCE"]["hermes_eval"].startswith("reused:"),
              "a dead widget does not turn a reuse into a paid re-inference")

        # ================================================================== #
        # 6. guard interaction: an upload still has to pass coverage
        # ================================================================== #
        print("\n[case] upload that does not cover the subset is still rejected")
        root = tmp / "proj_short"
        root.mkdir()
        short_blob = _pred_bytes(ids[:2])          # h3 deliberately missing
        files_obj = _StubFiles(payload={"preds_eval_twostep.jsonl": short_blob})
        try:
            with _fake_colab(files_obj):
                _run_cell3(
                    root, "hermes_eval", subset_dir, results_out,
                    env_flag="1", files_obj=files_obj,
                    upload_to_cwd={"preds_eval_twostep.jsonl": short_blob},
                )
        except SystemExit as e:
            check("have NO prediction" in str(e),
                  "a short upload fails the coverage guard instead of scoring a subset")
        else:
            check(False, "a short upload must raise SystemExit, not pass silently")

        print()
        if failures:
            print(f"FAILED: {len(failures)} check(s)")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("ALL UPLOAD-PATH CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())