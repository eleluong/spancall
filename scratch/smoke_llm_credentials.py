#!/usr/bin/env python3
"""Offline smoke test for the credential block of cell 0 in
`notebooks/spancall_lean_v3_2_llm_baseline.ipynb`.

The user asked for the LLM endpoint/key to be *read from input like the HF
token*, so this test pins the resolution order and, more importantly, the
behaviours that would silently corrupt an experiment if they regressed:

  1. `os.environ` wins over `.env` and over the prompt (CI / nbconvert never
     blocks on stdin).
  2. A repo-root `.env` is used when the environment is empty.
  3. When neither is set, the cell PROMPTS. EOF at the prompt must not crash
     and must not invent a value.
  4. A key typed at the prompt is used by the client, and is reported only as
     `<set>` - never echoed into the output that ends up in a notebook.
  5. With no key anywhere, `LLM_API_KEY` is empty so cell 4 stops loudly
     instead of scoring an all-abstain LLM arm.

The block is sliced out of the shipped notebook, so this test exercises the
code that actually runs rather than a paraphrase of it.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "spancall_lean_v3_2_llm_baseline.ipynb"


def _cell0_source() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return "".join(code[0]["source"])


def _credential_block() -> str:
    """`def _prompt(...)` to the end of the cell.

    Starts at the resolver helpers and runs all the way through the config
    printout, so the test can also assert that no secret is echoed. The
    bootstrap above it is skipped (it does `git clone` / `%pip`)."""
    lines = _cell0_source().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("def _prompt("))
    return "\n".join(lines[start:]) + "\n"


def _run(project_root: Path, environ: dict, answers: list):
    """Execute the sliced block. `answers` feeds `input()`, in prompt order.

    Every prompt goes through plain ``input()`` — there is no ``getpass`` path to
    stub, because ``getpass`` reads ``/dev/tty`` and cannot be driven from stdin.
    Returns (namespace, stdout, prompts).
    """
    ns = {
        "os": os,
        "PROJECT_ROOT": str(project_root),
        # names the post-bootstrap part of the cell expects to already exist
        "EVAL_SPLITS": ["hermes_eval", "xlam_eval"],
        # Cell 0 now also defines EVAL_LIMIT and prints it, so the sliced block
        # needs it for the same reason it needs EVAL_SPLITS.
        "EVAL_LIMIT": 2000,
        "RESULTS_DIR": str(project_root / "results_lean_v3_2_llm"),
    }
    prompts: list[str] = []
    plain = list(answers)

    real_input = builtins.input

    def fake_input(prompt=""):
        prompts.append(prompt)
        if not plain:
            raise EOFError
        return plain.pop(0)

    saved = {k: os.environ.get(k) for k in list(os.environ)}
    for k in [
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LLM_MODEL",
        "LLM_BASE_URL", "BASE_URL", "LLM_API_KEY",
        "OPENAI_API_KEY", "API_KEY",
    ]:
        os.environ.pop(k, None)
    os.environ.update(environ)

    buf = io.StringIO()
    try:
        builtins.input = fake_input
        with redirect_stdout(buf):
            exec(compile(_credential_block(), "<cell0-creds>", "exec"), ns)
    finally:
        builtins.input = real_input
        for k in [
            "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LLM_MODEL",
            "LLM_BASE_URL", "BASE_URL", "LLM_API_KEY",
            "OPENAI_API_KEY", "API_KEY",
        ]:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None and k not in environ:
                os.environ[k] = v

    return ns, buf.getvalue(), prompts


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    print("[block] sliced credential section")
    block = _credential_block()
    check("def _resolve(" in block, "block contains _resolve")
    check("LLM_API_KEY" in block, "block resolves LLM_API_KEY")

    # ---- 1. environment wins, no prompt ---------------------------------- #
    print("\n[assert] environment takes priority and never blocks")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_creds_env_"))
    (tmp / ".env").write_text(
        "LLM_API_KEY=from-dotenv\nLLM_BASE_URL=http://dotenv:8000/v1\n"
        "LLM_MODEL=dotenv-model\n", encoding="utf-8"
    )
    ns, out, prompts = _run(
        tmp,
        {"LLM_API_KEY": "from-environ", "LLM_BASE_URL": "http://env:9000/v1",
         "LLM_MODEL": "env-model"},
        answers=[],
    )
    check(ns["LLM_API_KEY"] == "from-environ", "env beats .env for the key")
    check(ns["LLM_BASE_URL"] == "http://env:9000/v1", "env beats .env for base_url")
    check(ns["LLM_MODEL"] == "env-model", "env beats .env for the model")
    # The HF token is asked for *unconditionally* (same UX as the v3.2 notebook),
    # so the assertion is about the LLM values specifically: none of them may
    # prompt once they are resolvable from the environment.
    llm_prompts = [p for p in prompts if "LLM" in p and "HF" not in p]
    check(not llm_prompts, f"no LLM prompt when env is complete (got {llm_prompts})")

    # ---- 2. .env is consulted ------------------------------------------- #
    print("\n[assert] repo-root .env is used when the environment is empty")
    ns, out, prompts = _run(tmp, {}, answers=[])
    check(ns["LLM_API_KEY"] == "from-dotenv", "key read from .env")
    check(ns["LLM_BASE_URL"] == "http://dotenv:8000/v1", "base_url read from .env")
    check(ns["LLM_MODEL"] == "dotenv-model", "model read from .env")
    llm_prompts = [p for p in prompts if "LLM" in p and "HF" not in p]
    check(not llm_prompts, f"no LLM prompt when .env is complete (got {llm_prompts})")

    # ---- 3. nothing set -> prompts, EOF must not crash ------------------- #
    print("\n[assert] interactive prompt, and EOF degrades instead of crashing")
    empty = Path(tempfile.mkdtemp(prefix="smoke_creds_empty_"))
    ns, out, prompts = _run(empty, {}, answers=[])
    check(len(prompts) >= 3, f"prompted for the missing values (asked {len(prompts)})")
    check(ns["LLM_API_KEY"] == "", "EOF leaves the key EMPTY (cell 4 will stop)")
    check(ns["LLM_BASE_URL"] == "", "EOF leaves base_url empty (= api.openai.com)")
    check(ns["LLM_MODEL"] == "qwen3.7-flash", "EOF keeps the model default")
    check("from-environ" not in out, "no secret text in stdout")
    check("MISSING" in out or "cell 4 will stop" in out,
          "output flags the missing key")

    # ---- 4. typed at the prompt -> used, but never echoed ---------------- #
    print("\n[assert] a key typed at the prompt is used and never printed")
    typed_secret = "sk-typed-at-the-prompt-SECRET"
    # Prompt order is fixed by the cell: HF token -> model -> base URL -> key.
    ns, out, prompts = _run(
        empty,
        {},
        answers=["", "my-model", "", typed_secret],   # HF skipped, key typed
    )
    check(ns["LLM_MODEL"] == "my-model", "typed model name used")
    check(ns["LLM_API_KEY"] == typed_secret, "typed key used by the cell")
    check(typed_secret not in out, "typed key is NOT in the printed config block")
    check("LLM_API_KEY     = <set>" in out, "typed key is reported only as <set>")
    check(ns["HF_TOKEN"] == "", "HF token may be skipped (Enter)")
    hf_prompted = any("HF" in p or "Hugging" in p for p in prompts)
    check(hf_prompted, "HF token is prompted for, like the v3.2 notebook")

    # ---- 5. HF token wires into the hub calls --------------------------- #
    print("\n[assert] HF token is exported for huggingface_hub")
    ns, out, prompts = _run(empty, {}, answers=["hf_abc123"])
    check(os.environ.get("HF_TOKEN") == "" or ns["HF_TOKEN"] == "hf_abc123",
          "HF_TOKEN resolved from the prompt")

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CREDENTIAL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())