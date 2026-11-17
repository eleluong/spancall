#!/usr/bin/env python3
"""Execute the *shipped* agentic cell 0b under stubbed stdin.

The notebook is the artifact that runs, so the resolution logic is exercised by
slicing it out of `notebooks/spancall_lean_v3_2_agentic.ipynb` rather than by
re-implementing it here. No network, no Hugging Face, no LLM: every external
call in the cell is a name lookup or a prompt.

Checks, in the order the user would hit them:

1. env vars win over `.env`, and no LLM prompt is asked;
2. `.env` is used when the environment is empty;
3. a blank/EOF prompt leaves credentials empty and prints no secret;
4. a typed key is used, is not echoed, and is reported only as ``<set>``;
5. the v3.2 defaults are what the notebook claims (repo id, results dir,
   max_cands).
"""
from __future__ import annotations

import builtins
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "spancall_lean_v3_2_agentic.ipynb"

_failures: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        _failures.append(label)


def _code_cells() -> list[str]:
    with open(NB, encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def _config_cell() -> str:
    """The cell that resolves credentials (it defines ``_prompt``)."""
    for src in _code_cells():
        if "def _prompt(" in src:
            return src
    raise AssertionError("no cell defines _prompt - credential block is missing")


def _config_slice() -> str:
    """From the ``ENV = ...`` line to the end of the cell.

    Everything before that is imports plus ``PROJECT_ROOT`` discovery, which is
    supplied by the fixture namespace instead: re-running it would ``chdir`` the
    test process.
    """
    src = _config_cell()
    lines = src.splitlines()
    start = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("ENV = {**_load_dotenv_values")
    )
    return "\n".join(lines[start:]) + "\n"


def _run(env_vars: dict, answers: list[str], dotenv: dict | None = None):
    """Exec the shipped block with a fake stdin and a fake `.env` reader.

    Only ``input()`` is stubbed: the cell prompts with plain ``input()`` for every
    value, secrets included, because ``getpass`` reads ``/dev/tty`` and would
    block under ``nbconvert --execute``.
    """
    asked: list[str] = []
    typed_iter = iter(answers)
    printed: list[str] = []

    def fake_input(prompt: str = "") -> str:
        asked.append(f"input:{prompt}")
        try:
            return next(typed_iter)
        except StopIteration:
            raise EOFError from None

    ns = {
        "__name__": "agentic_credentials_smoke",
        "os": os,
        "json": json,
        # `PROJECT_ROOT` is a Path in the notebook; str() on it is used below.
        "PROJECT_ROOT": Path("/tmp/fake_root"),
        "_load_dotenv_values": lambda _p: dict(dotenv or {}),
        "torch": type("_T", (), {"cuda": type("_C", (), {"is_available": staticmethod(lambda: False)}),
                                 "backends": type("_B", (), {"mps": type("_M", (), {"is_available": staticmethod(lambda: False)})})}),
        "print": lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
    }

    saved_env = dict(os.environ)
    saved_input = builtins.input
    try:
        for k in ("HF_TOKEN", "HUGGING_FACE_TOKEN", "SPANCALL_HF_REPO_ID", "BASE_LLM_MODEL",
                  "BASE_URL", "LLM_BASE_URL", "API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
            os.environ.pop(k, None)
        os.environ.update(env_vars)

        builtins.input = fake_input
        exec(compile(_config_slice(), "<agentic_cell_0b>", "exec"), ns)
    finally:
        builtins.input = saved_input
        os.environ.clear()
        os.environ.update(saved_env)

    return ns, asked, "\n".join(printed)


def main() -> int:
    print("=" * 72)
    print("AGENTIC v3.2 - CREDENTIAL BLOCK SMOKE")
    print("=" * 72)

    # -- 1. environment wins over .env, no prompts -------------------------- #
    ns, asked, out = _run(
        {
            "HF_TOKEN": "hf_from_env",
            "BASE_LLM_MODEL": "gpt-4o-mini",
            "BASE_URL": "https://example.test/v1",
            "API_KEY": "sk-from-env",
        },
        answers=[],
        dotenv={"API_KEY": "sk-from-dotenv", "BASE_URL": "https://dotenv.test"},
    )
    check(ns["HF_TOKEN"] == "hf_from_env", "1a env HF_TOKEN wins")
    check(ns["LLM_API_KEY"] == "sk-from-env", "1b env API_KEY wins over .env")
    check(ns["LLM_BASE_URL"] == "https://example.test/v1", "1c env BASE_URL wins over .env")
    check(ns["LLM_MODEL_ID"] == "gpt-4o-mini", "1d env BASE_LLM_MODEL honoured")
    check(not asked, f"1e no prompt when everything is set (got {asked})")
    check("hf_from_env" not in out, "1f HF token never printed")
    check("sk-from-env" not in out, "1g API key never printed")
    check("HF token present  = <set>" in out, "1h HF token reported as <set>")
    check("LLM key present   = <set>" in out, "1i LLM key reported as <set>")

    # -- 2. .env fills the gaps --------------------------------------------- #
    ns, asked, out = _run(
        {},
        answers=[],
        dotenv={
            "HF_TOKEN": "hf_dotenv",
            "API_KEY": "sk-dotenv",
            "BASE_URL": "https://dotenv.test/v1",
            "BASE_LLM_MODEL": "claude-x",
        },
    )
    check(ns["LLM_API_KEY"] == "sk-dotenv", "2a .env API_KEY used")
    check(ns["LLM_BASE_URL"] == "https://dotenv.test/v1", "2b .env BASE_URL used")
    check(ns["LLM_MODEL_ID"] == "claude-x", "2c .env BASE_LLM_MODEL used")
    check(ns["HF_TOKEN"] == "hf_dotenv", "2d .env HF_TOKEN used")
    check(not asked, f"2e no prompt when .env is complete (got {asked})")

    # -- 3. nothing supplied -> prompt, Enter skips -------------------------- #
    ns, asked, out = _run({}, answers=["", "", "", ""])
    check(ns["LLM_API_KEY"] == "", "3a blank prompt leaves API key empty")
    check(ns["LLM_BASE_URL"] == "", "3b blank prompt leaves base URL empty")
    check(ns["LLM_MODEL_ID"] == "google/gemma-4", "3c model falls back to the documented default")
    check(ns["HF_TOKEN"] is None, "3d blank HF prompt leaves HF_TOKEN as None")
    check(len(asked) == 4, f"3e four prompts asked (got {len(asked)}: {asked})")
    check(all(a.startswith("input:") for a in asked),
          "3f every prompt goes through input() (getpass cannot be driven by stdin)")
    check("(none - public repos only)" in out, "3g missing HF token reported honestly")
    check("(none - deterministic backend)" in out, "3h missing LLM key reported honestly")

    # -- 4. typed key is used, not echoed ----------------------------------- #
    # Prompt order is fixed by the cell: HF token -> model id -> base URL ->
    # API key. Supplying them out of order would put the secret into the *base
    # URL* slot, which is printed on purpose.
    ns, asked, out = _run({}, answers=["", "my-model", "", "sk-typed-secret"])
    check(ns["LLM_API_KEY"] == "sk-typed-secret", "4a typed API key is used")
    check("sk-typed-secret" not in out, "4b typed API key is never echoed into output")
    check("LLM key present   = <set>" in out, "4c typed key reported only as <set>")
    check(ns["LLM_MODEL_ID"] == "my-model", "4d typed model id is used")
    check(ns["LLM_BASE_URL"] == "", "4e skipped base URL stays empty (not defaulted)")

    # -- 5. v3.2 defaults --------------------------------------------------- #
    ns, _asked, _out = _run({}, answers=["", "", "", ""])
    check(ns["HF_REPO_ID"] == "convoicon/spancall-lean-v3-2", "5a default repo is the v3.2 export")
    check("results_lean_v3_2" in ns["RESULTS_DIR"], "5b results dir is v3.2")
    check(ns["LOCAL_CHECKPOINT_DIR"].endswith("results_lean_v3_2/hf_export"),
          "5c local checkpoint path is v3.2")
    check(ns["USE_LOCAL_CHECKPOINT"] is False, "5d local checkpoint stays opt-in")
    check(ns["DEVICE"] == "cpu" and ns["SEED"] == 42 and ns["MAX_TURNS"] == 8,
          "5e runtime constants survive the credential block")

    # -- 6. repo override --------------------------------------------------- #
    ns, _asked, _out = _run({"SPANCALL_HF_REPO_ID": "me/own-model"}, answers=["", "", "", ""])
    check(ns["HF_REPO_ID"] == "me/own-model", "6a SPANCALL_HF_REPO_ID overrides the default")

    # -- 7. max_cands is pinned, not inherited ------------------------------ #
    load_src = next(s for s in _code_cells() if "MAX_CANDS_FALLBACK" in s)
    check(re.search(r"MAX_CANDS_FALLBACK\s*=\s*1024", load_src), "7a max_cands fallback is 1024")
    check(re.search(r"max_cands=int\(_cfg\.get\(\"max_cands\"\) or MAX_CANDS_FALLBACK\)", load_src),
          "7b max_cands is passed to LeanResolver")
    check(re.search(r'collapse_guard=bool\(_cfg\.get\("collapse_guard", True\)\)', load_src),
          "7c collapse_guard defaults to True")
    check(re.search(r'prevent_param_collapse=bool\(_cfg\.get\("prevent_param_collapse", True\)\)', load_src),
          "7d prevent_param_collapse defaults to True")

    print("=" * 72)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for f in _failures:
            print("  - " + f)
        return 1
    print("ALL AGENTIC CREDENTIAL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())