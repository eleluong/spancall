#!/usr/bin/env python3
"""Offline smoke test for the TRANSPORT block of cell 7 in
`notebooks/spancall_lean_v3_2_llm_baseline.ipynb`: the browser zip download and
the optional Hugging Face push.

The epilogue is sliced out of the shipped cell and executed verbatim, with
`google.colab.files` stubbed. The `files.download()` call cannot happen outside
Colab, so everything below the stub is what is under test.

The failure this test exists to prevent is subtler than "no download":

  * `files.download()` called once per artifact -> Chrome blocks the 2nd+
    automatic download, so the user gets the first file and believes the rest
    were saved. No exception, no warning, silent partial delivery.
  * `*.ckpt-*.jsonl` included -> those rows carry `_fingerprint` and possibly
    `llm_error`; once off the VM a half-failed run reads as a result.
  * the staging directory left behind, or the zip appended to instead of
    rebuilt -> the next run archives its own predecessor.

Checks:
  1. Flag off -> no download, and the message says how to turn it on.
  2. Flag on outside Colab -> no download, and the reason is printed.
  3. Flag on in Colab -> EXACTLY ONE download call, of a real zip.
  4. The zip holds `llm_parity.json` and the per-split preds.
  5. The zip excludes `*.ckpt-*`, `*.bak-*`, `*.tmp*`, `subset_*` and the
     staging directory itself.
  6. The staging directory is gone after the call (the zip is the artifact).
  7. A second run does not append: the archive is rebuilt, not grown.

The Hub push is exercised against a fake `HfApi`, so nothing leaves the machine.
What it must get right:

  8. Off by default, and it says so.
  9. No token -> skipped, not crashed, and nothing uploaded.
 10. Writes ONLY under `eval/llm_baseline/<run_tag>/` - never the repo root,
     which would clobber `model.pt`, `resolve_config.json` and the model card.
 11. `<run_tag>` distinguishes a 10% smoke run from the 100% run, so a smoke
     push cannot overwrite the numbers the paper cites.
 12. A missing `llm_parity.json` refuses the push instead of publishing
     predictions with no table.
 13. Both transports ship the SAME file set (one `_ship_files` whitelist).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import zipfile
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch"))

# Reuse the Colab stubs from the upload test: one definition of "pretend to be
# Colab" for both tests, so the two cannot drift apart.
from smoke_llm_baseline_upload import _fake_colab, _no_colab, _StubFiles  # noqa: E402

NB = REPO / "notebooks" / "spancall_lean_v3_2_llm_baseline.ipynb"

# The first marker of the transport tail. If the block is ever moved or renamed
# this fails loudly instead of testing nothing.
_ANCHOR = "DOWNLOAD_RESULTS = os.environ.get("

# The shared whitelist sits between the payload and the download block, and BOTH
# transports call it, so the slice has to resume here rather than at `_ANCHOR`.
_SHIP = "def _ship_files("


class _FakeApi:
    """Records `upload_file` calls; stands in for `huggingface_hub.HfApi`."""

    def __init__(self, token=None, fail=None):
        self.token = token
        self.fail = fail
        self.uploads = []      # list of (path_in_repo, commit_message)
        self.created = []      # list of repo_id

    def create_repo(self, repo_id, repo_type=None, exist_ok=False):
        self.created.append(repo_id)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id,
                    repo_type=None, commit_message=None):
        if self.fail is not None:
            raise self.fail
        self.uploads.append((path_in_repo, commit_message))


@contextmanager
def _fake_hf_api(fake):
    """Make `from huggingface_hub import HfApi` resolve to `fake`.

    `sys.modules[...]` injection rather than patching an attribute: the cell does
    a fresh `from ... import HfApi` inside the try-block, so only a module-level
    substitution is actually observed.
    """
    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = lambda token=None: fake
    fake.token = fake.token
    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = mod
    try:
        yield fake
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved


def _cell7_source() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return "".join(code[7]["source"])


def _epilogue_source() -> str:
    """The shipped transport code, verbatim.

    Two regions, concatenated around the table:
      * the HEAD - everything before `payload = {`, which holds `_out`, `RUN_TAG`,
        `HUB_EVAL_PREFIX` and the revision lookups the push block reads. It is
        pure (string formatting and `globals().get`), so executing it is safe and
        keeps the test from hard-coding a run-tag format the cell could change.
      * the TAIL - from `def _ship_files(...)` on: the shared whitelist plus the
        download and push blocks. Slicing at `_ANCHOR`` would drop `_ship_files`,
        and then both transports look broken for the wrong reason.

    `payload = { ... }` and the printed table are skipped: they need `SYSTEMS` /
    `COMPLEXITY` / `SUBSET_PATHS`, which is a different test's job
    (`smoke_llm_baseline_e2e.py`) and would only add imports here.
    """
    src = _cell7_source()
    for marker, why in (
        (_ANCHOR, "the download epilogue is missing"),
        (_SHIP, "the shared `_ship_files` whitelist is missing"),
        ("payload = {", "can't split the head off the table"),
    ):
        if marker not in src:
            raise AssertionError(f"cell 7 has no `{marker}` block; {why}")
    head = src[:src.index("payload = {")]
    tail = src[src.index(_SHIP):]
    if _ANCHOR not in tail:
        raise AssertionError(
            "the download block sits ABOVE `_ship_files`; the slice would drop it"
        )
    return head + "\n" + tail


# Values the epilogue needs from earlier cells. Supplied explicitly so the test
# can assert on the run tag instead of inheriting whatever the developer's shell
# happens to export.
_RUN_DEFAULTS = {
    "HF_REPO_ID": "convoicon/spancall-lean-v3-2",
    "LLM_MODEL": "qwen3.7-flash",
    "LLM_LIMIT_FRACTION": 1.0,
    "SEED": 42,
}


def _run_epilogue(results_dir: Path, *, env_flag: str, files_obj: _StubFiles,
                  push_flag: str = "0", token: str | None = None,
                  **over) -> tuple[dict, str]:
    ns = {"RESULTS_DIR": str(results_dir), "os": os}
    ns.update(_RUN_DEFAULTS)
    ns.update(over)

    saved = {k: os.environ.get(k) for k in
             ("DOWNLOAD_RESULTS", "PUSH_RESULTS_TO_HUB", "HF_TOKEN",
              "HUGGING_FACE_HUB_TOKEN")}
    os.environ["DOWNLOAD_RESULTS"] = env_flag
    os.environ["PUSH_RESULTS_TO_HUB"] = push_flag
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(k, None)
    if token is not None:
        os.environ["HF_TOKEN"] = token
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(_epilogue_source(), "<epilogue>", "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return ns, buf.getvalue()


def _seed(results_dir: Path) -> None:
    """A results dir with one of every filename class the filter must judge."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "llm_parity.json").write_text('{"n_F": 1110}', encoding="utf-8")
    for name in (
        "preds_hermes_eval_llm.jsonl",                 # keep
        "preds_xlam_eval_llm.jsonl",                   # keep
        "preds_hermes_eval_llm.ckpt-20261118.jsonl",   # drop: resume state
        "preds_xlam_eval_llm.bak-20261118.jsonl",      # drop: previous run
        "preds_hermes_eval_llm.tmp.jsonl",             # drop: partial write
        "subset_hermes_eval_clean.jsonl",              # drop: rebuildable
        "notes.txt",                                   # drop: not an artifact
    ):
        (results_dir / name).write_text('{"sample_id": "x"}\n', encoding="utf-8")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_download_"))
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    try:
        # ================================================================== #
        # 1. flag off
        # ================================================================== #
        print("\n[case] flag off")
        results_dir = tmp / "res_off"
        _seed(results_dir)
        files_obj = _StubFiles()
        with _fake_colab(files_obj):
            _, out = _run_epilogue(results_dir, env_flag="0", files_obj=files_obj)
        check(files_obj.n_download == 0, "flag off -> nothing downloaded")
        check("[download] off" in out, "flag off -> says so")
        check("DOWNLOAD_RESULTS=1" in out, "flag off -> tells the user how to turn it on")
        check(not (results_dir / "llm_baseline_artifacts").exists(),
              "flag off -> no staging directory created")

        # ================================================================== #
        # 2. flag on, not in Colab
        # ================================================================== #
        print("\n[case] flag on, NOT in Colab")
        results_dir = tmp / "res_nocolab"
        _seed(results_dir)
        files_obj = _StubFiles()
        with _no_colab():
            _, out = _run_epilogue(results_dir, env_flag="1", files_obj=files_obj)
        check(files_obj.n_download == 0, "outside Colab -> nothing downloaded")
        check("not Colab" in out, "outside Colab -> the reason is printed")
        check(not (results_dir / "llm_baseline_artifacts").exists(),
              "outside Colab -> no staging directory left behind")

        # ================================================================== #
        # 3-6. flag on, in Colab
        # ================================================================== #
        print("\n[case] flag on, in Colab")
        results_dir = tmp / "res_on"
        _seed(results_dir)
        files_obj = _StubFiles()
        with _fake_colab(files_obj):
            ns, out = _run_epilogue(results_dir, env_flag="1", files_obj=files_obj)

        check(files_obj.n_download == 1,
              f"EXACTLY ONE download call (got {files_obj.n_download}) - Chrome blocks the 2nd")
        check(len(files_obj.paths) == 1, "the one call carried exactly one path")

        zip_path = Path(files_obj.paths[0]) if files_obj.paths else None
        check(zip_path is not None and zip_path.exists(),
              f"the downloaded path exists -> {zip_path}")
        check(zip_path is not None and zipfile.is_zipfile(zip_path),
              "the downloaded artifact is a real zip")
        check(zip_path is not None and zip_path.stat().st_size > 0,
              "the zip is not empty")

        names = set(zipfile.ZipFile(zip_path).namelist()) if zip_path else set()
        check("llm_parity.json" in names, "zip holds llm_parity.json")
        check("preds_hermes_eval_llm.jsonl" in names, "zip holds the hermes preds")
        check("preds_xlam_eval_llm.jsonl" in names, "zip holds the xlam preds")

        check(not any(".ckpt-" in n for n in names),
              "zip EXCLUDES *.ckpt-* (resume state would read as a result)")
        check(not any(".bak-" in n for n in names), "zip excludes *.bak-*")
        check(not any(".tmp" in n for n in names), "zip excludes *.tmp*")
        check(not any(n.startswith("subset_") for n in names),
              "zip excludes subset_* (rebuildable, and not an artifact)")
        check("notes.txt" not in names, "zip excludes unrelated files")
        check(not any(n.startswith("llm_baseline_artifacts") for n in names),
              "zip does not contain the staging directory itself")

        check(not (results_dir / "llm_baseline_artifacts").exists(),
              "staging directory removed after the call (the zip IS the artifact)")
        check("[download] bundling 3 file(s)" in out,
              "the printed bundle count matches the 3 kept files")

        # ================================================================== #
        # 7. a second run rebuilds rather than appends
        # ================================================================== #
        print("\n[case] second run with the previous zip still on disk")
        files_obj2 = _StubFiles()
        with _fake_colab(files_obj2):
            _, out2 = _run_epilogue(results_dir, env_flag="1", files_obj=files_obj2)
        check(files_obj2.n_download == 1, "second run also makes exactly one download")
        names2 = set(zipfile.ZipFile(files_obj2.paths[0]).namelist())
        check(names2 == names,
              "second run rebuilt the same file set (nothing appended, nothing doubled)")
        check(not any(n.startswith("llm_baseline_artifacts") for n in names2),
              "second run did not sweep the previous zip into the new one")

        # ================================================================== #
        # 8-13. Hub push
        # ================================================================== #
        print("\n[case] hub push, flag off")
        _, out = _run_epilogue(results_dir, env_flag="0", files_obj=_StubFiles())
        check("[hub-push] off" in out, "push is off by default and says so")

        print("\n[case] hub push, flag on but no token")
        fake = _FakeApi()
        with _fake_hf_api(fake):
            _, out = _run_epilogue(results_dir, env_flag="0", files_obj=_StubFiles(),
                                   push_flag="1", token=None)
        check(fake.uploads == [], "no token -> nothing uploaded")
        check("SKIPPED" in out, "no token -> skipped loudly, not crashed")

        print("\n[case] hub push, flag on with a token")
        fake = _FakeApi()
        with _fake_hf_api(fake):
            ns, out = _run_epilogue(results_dir, env_flag="0", files_obj=_StubFiles(),
                                    push_flag="1", token="hf_fake")
        check(len(fake.uploads) == 3,
              f"uploaded the 3 shippable files (got {len(fake.uploads)})")
        paths = [p for p, _ in fake.uploads]
        check(all(p.startswith("eval/llm_baseline/") for p in paths),
              "every upload is under eval/llm_baseline/")
        check(not any(p.count("/") < 3 for p in paths),
              "no upload lands at the repo ROOT (would clobber model.pt / card)")
        check(all(p.endswith(("llm_parity.json", "_llm.jsonl")) for p in paths),
              "only artifacts uploaded, never weights or resolve_config.json")
        check(not any(".ckpt-" in p or ".bak-" in p or ".tmp" in p for p in paths),
              "resume/backup state excluded from the push too")
        check(all("qwen3.7-flash-full-seed42" in p for p in paths),
              f"run tag carries model+fraction+seed -> {paths[0]}")
        check(ns["HUB_EVAL_PREFIX"] == "eval/llm_baseline/qwen3.7-flash-full-seed42",
              "HUB_EVAL_PREFIX matches the uploaded paths")
        msgs = [m for _, m in fake.uploads]
        check(all("full" in m and "seed=42" in m for m in msgs),
              "commit messages record fraction and seed")
        check(all("HEAD" in m for m in msgs),
              "commit messages name the model revision (HEAD when unpinned)")

        print("\n[case] hub push distinguishes a smoke run from the full run")
        fake_smoke = _FakeApi()
        with _fake_hf_api(fake_smoke):
            ns_smoke, _ = _run_epilogue(results_dir, env_flag="0", files_obj=_StubFiles(),
                                        push_flag="1", token="hf_fake",
                                        LLM_LIMIT_FRACTION=0.1)
        smoke_paths = [p for p, _ in fake_smoke.uploads]
        check(all("f0.1" in p for p in smoke_paths),
              f"a 10% run lands in its own folder -> {smoke_paths[0]}")
        check(ns_smoke["HUB_EVAL_PREFIX"] != ns["HUB_EVAL_PREFIX"],
              "smoke and full runs CANNOT overwrite each other")

        print("\n[case] hub push refuses when llm_parity.json is missing")
        partial = tmp / "res_partial"
        (partial / "nested").mkdir(parents=True)
        (partial / "preds_hermes_eval_llm.jsonl").write_text('{"sample_id": "x"}\n')
        fake = _FakeApi()
        with _fake_hf_api(fake):
            _, out = _run_epilogue(partial, env_flag="0", files_obj=_StubFiles(),
                                   push_flag="1", token="hf_fake")
        check(fake.uploads == [],
              "no parity table -> nothing published (evidence without the conclusion)")
        check("llm_parity.json is missing" in out, "and the refusal is explained")

        print("\n[case] hub push failure does not fail the cell")
        fake = _FakeApi(fail=RuntimeError("403 Forbidden"))
        with _fake_hf_api(fake):
            _, out = _run_epilogue(results_dir, env_flag="0", files_obj=_StubFiles(),
                                   push_flag="1", token="hf_fake")
        check("FAILED" in out and "403" in out, "the API error is reported")
        check("local artifacts are intact" in out,
              "and the message reassures that the local files survive")
        check((results_dir / "llm_parity.json").exists(),
              "a failed push did not remove the local artifact")

        print("\n[case] one whitelist for both transports")
        # The invariant `_ship_files` exists to enforce: the folder on the Hub and
        # the zip in the browser must be the same set. If they diverge, a reviewer
        # citing the Hub folder is looking at different evidence than the author
        # who checked the zip.
        fake = _FakeApi()
        with _fake_hf_api(fake):
            _run_epilogue(results_dir, env_flag="0", files_obj=_StubFiles(),
                          push_flag="1", token="hf_fake")
        pushed = sorted(os.path.basename(p) for p, _msg in fake.uploads)
        # `_fake_colab` yields None, so hold the stub instead of the context value.
        dl_files = _StubFiles()
        with _fake_colab(dl_files):
            _run_epilogue(results_dir, env_flag="1", files_obj=dl_files)
        with zipfile.ZipFile(dl_files.paths[0]) as zf:
            zipped = sorted(zf.namelist())
        check(pushed == zipped,
              f"push and zip ship the same files (push={pushed}, zip={zipped})")

        print()
        if failures:
            print(f"FAILED: {len(failures)} check(s)")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("ALL DOWNLOAD+PUBLISH-PATH CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())