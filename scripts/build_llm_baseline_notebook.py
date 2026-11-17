#!/usr/bin/env python3
"""Build ``notebooks/spancall_lean_v3_2_llm_baseline.ipynb``.

Why a builder instead of a hand-written .ipynb: every other notebook in this
repository is generated (``scripts/build_lean_v3_2_*``), so the eval protocol
lives in reviewable Python instead of inside JSON cell blobs. Editing a cell
here is a diff you can read.

What the notebook does
----------------------
Compares **SpanCall Lean v3.2** against **one hosted LLM** on the *same rows*,
in the *same order*, scored by the *same* ``ToolCallEvaluator``. This is the
replacement for the regex ``B1a`` baseline in the paper: B1a is a weak,
easily-beaten straw man, and a reviewer will say so.

User decisions pinned here:
  * **One LLM arm**, via the **OpenAI SDK** (``openai`` package). The model is
    hosted by the user elsewhere, so ``base_url`` + ``model`` + key-from-env are
    all configurable. **No litellm, no local open-weights inference.**
  * SpanCall **code and checkpoint are pulled from HuggingFace**
    (``convoicon/spancall-lean-v3-2``), data from ``convoicon/spancall``.

The fairness contract is not re-implemented in the notebook: the scoring,
tool-schema conversion, prompt rendering and prediction shaping are imported
from ``scripts/compare_llm_spancall.py`` so the notebook cannot drift from the
script the repo already tests.

Usage::

    .venv/bin/python scripts/build_llm_baseline_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

try:  # nbformat is a dev dependency; fail loudly rather than writing a bad file
    import nbformat as nbf
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "nbformat is required to build notebooks: .venv/bin/python -m pip install nbformat"
    ) from exc


OUT_PATH = Path("notebooks/spancall_lean_v3_2_llm_baseline.ipynb")

# --------------------------------------------------------------------------- #
# Cell sources
# --------------------------------------------------------------------------- #

MD_TITLE = """# SpanCall Lean v3.2 — LLM Baseline (OpenAI SDK)

This notebook answers one question with numbers: **does SpanCall actually beat a
hosted LLM at tool-call extraction, when both are given the same tools, the same
utterance, and the same scorer?**

It exists to replace the `B1a` regex baseline in the paper. `B1a` is a
capitalisation/slot-pattern straw man; an LLM is the baseline a reviewer will
ask for, and it is the only one that makes the latency/cost argument load-bearing.

## The fairness contract (this is the whole point)

| | SpanCall `two_step` | Hosted LLM |
| --- | --- | --- |
| Rows | same subset files, same order | same subset files, same order |
| Tool list | same `UnifiedSample.tools` | same `UnifiedSample.tools`, via `_tool_definitions_to_openai` |
| Utterance / recent context / clock | rendered into the SpanCall template | rendered as plain text lines in the user turn |
| Scorer | `ToolCallEvaluator` | `ToolCallEvaluator` |
| Abstain semantics | `hit=False, payload=None` | no `tool_calls` returned → `hit=False, payload=None` |

Nothing is re-derived here. `score`, `_llm_pred_row`, `_messages_for`,
`_openai_tools`, `_slices`, `_print_table` and `_print_pivot` are **imported from
`scripts/compare_llm_spancall.py`**, which is the module the repository already
tests. A second copy of that logic in the notebook is how the two arms quietly
stop being comparable.

**What is configurable, and why it matters:** `LLM_LIMIT` trims every arm to the
same first *N* rows. It is for cheap smoke runs, not for reporting — a headline
number on 300 rows is a smoke test, and the notebook says so when it prints.

## Before the first run

Cell 0 asks for four things, the same way the v3.2 notebook asks for your HF
token — **press Enter to be prompted**, or pre-set them for a non-interactive run:

| value | env var | notes |
| --- | --- | --- |
| HF access token | `HF_TOKEN` | optional; the `convoicon/*` repos are public |
| LLM model name | `LLM_MODEL` | e.g. `gpt-4o-mini`, or whatever your server serves |
| LLM base URL | `LLM_BASE_URL` | `http://host:8000/v1`; blank → `api.openai.com` |
| LLM API key | `LLM_API_KEY` | asked with `input()`, never written, reported only as `<set>` |

`OPENAI_API_KEY` / `API_KEY` and a repo-root `.env` are also accepted, so a
kernel that already has `OPENAI_API_KEY` exported just works. Secrets are only
ever printed as `<set>`.
"""

CODE_CONFIG = '''# 0. Setup: pull the public repo, install deps, pin the v3.2 paths.
#
# Same bootstrap as `spancall_lean_v3_2_train_eval.ipynb` §0. The only extra
# dependency is `openai`: the LLM arm speaks the OpenAI SDK protocol, so any
# OpenAI-compatible endpoint (the user's own vLLM/SGLang/TGI deployment, a
# gateway, or api.openai.com) works without a code change.
import os, sys, json
from pathlib import Path

GIT_REPO_URL = "https://github.com/eleluong/spancall.git"
GIT_BRANCH   = "main"

cwd = Path.cwd().resolve()
if (cwd / "spancall" / "__init__.py").is_file():
    PROJECT_ROOT = cwd
elif (cwd.parent / "spancall" / "__init__.py").is_file():
    PROJECT_ROOT = cwd.parent
elif (cwd / "spancall" / "spancall" / "__init__.py").is_file():
    PROJECT_ROOT = cwd / "spancall"
else:
    print(f"Cloning public repository from {GIT_REPO_URL}...")
    !git clone --branch {GIT_BRANCH} {GIT_REPO_URL}
    PROJECT_ROOT = cwd / "spancall"

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"SpanCall project root: {PROJECT_ROOT}")

# `openai` is the arm under test; `huggingface_hub` fetches code+checkpoint.
%pip install -q -e ".[model,data,agent,dev]" openai huggingface_hub

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42

# v3.2 artifacts. The checkpoint is a state_dict, NOT a save_pretrained model:
# LeanSpanCallModel has no save_pretrained(), so a transformers-shaped config
# would misdescribe the architecture. `resolve_config.json` is the rebuild recipe.
HF_REPO_ID      = "convoicon/spancall-lean-v3-2"
HF_DATASET_ID   = "convoicon/spancall"

DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed_v3_2")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results_lean_v3_2_llm")
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_checkpoint")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# The two eval splits behind the paper's headline table. Both are single-call
# subsets, which is why `SINGLE_CALL_ONLY=True` and `max_calls=1` below.
EVAL_SPLITS = ["hermes_eval", "xlam_eval"]

# --------------------------------------------------------------------------- #
# Where the frozen v3.2 SpanCall predictions come from.
# --------------------------------------------------------------------------- #
# `results_lean_v3_2/` is NOT in git and NOT on the HF Hub (the training notebook
# only backs it up to Google Drive). So a bare `git clone` starts without it, and
# the reuse path below would silently fall through to inference - the opposite of
# what we want.
#
# Point this at the directory that holds the published predictions:
#   * a Google Drive mount, e.g. "/content/drive/MyDrive/spancall_lean_artifacts/results_lean_v3_2"
#   * a local copy, e.g. os.path.join(PROJECT_ROOT, "results_lean_v3_2")
# Leave it None to use the in-repo default and accept the inference fallback.
V3_2_RESULTS_DIR = os.environ.get("SPANCALL_V3_2_RESULTS_DIR") or None

# --------------------------------------------------------------------------- #
# Credentials & endpoint — typed in at run time, exactly like the HF token in
# `spancall_lean_v3_2_train_eval.ipynb`.
# --------------------------------------------------------------------------- #
# Nothing secret is written into this file. Every value is resolved in this order:
#   1. already set in the kernel / environment      (CI, nbconvert, `%env`)
#   2. a repo-root `.env`
#   3. an interactive prompt — press Enter to skip
# So "Run all" in Colab/Jupyter stops and asks for the token, the endpoint, the
# model name and the key; a scripted run supplies them up front and never blocks
# on stdin. Pressing Enter at a prompt falls back to the default, and an
# unresolvable value is left empty rather than silently defaulted to something
# that would make a paid call to the wrong place.
def _prompt(label: str, default: str = "", secret: bool = False) -> str:
    """One interactive question. Enter keeps `default`; EOF means "skip".

    Plain ``input()`` on purpose: ``getpass`` reads ``/dev/tty`` instead of
    stdin, so under ``nbconvert --execute`` or a piped heredoc it blocks on a
    terminal that does not exist, and it cannot be patched by a test. The entry
    is never written to this file and never printed — only ``<set>`` — so what
    protects the artifact is the absent write, not the hidden keystroke.
    """
    suffix = f" [{default}]" if default else " [optional — Enter to skip]"
    try:
        val = input(f"{label}{suffix}: ")
    except (EOFError, OSError):
        val = ""
    return (val or "").strip() or default


def _read_dotenv(path: str) -> dict:
    """Minimal `KEY=VALUE` reader; never overrides a real env var."""
    out = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        out = {}
    return out


_ENV_FILE = _read_dotenv(os.path.join(str(PROJECT_ROOT), ".env"))


def _resolve(*names, default: str = "", secret: bool = False, label: str = "") -> str:
    """env > .env > interactive prompt, for the first name that yields a value."""
    for name in names:
        val = os.environ.get(name) or _ENV_FILE.get(name)
        if val:
            return val.strip()
    return _prompt(label or names[0], default, secret=secret)


# Hugging Face: needed only for a gated dataset or a private checkpoint repo.
# The public `convoicon/*` repos work anonymously, so this may stay empty.
HF_TOKEN = _resolve("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", default="",
                    secret=True, label="Hugging Face access token")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# The LLM arm. These four are the only things the run needs from you.
LLM_MODEL     = _resolve("LLM_MODEL", default="gpt-4o-mini",
                         label="LLM model name")
LLM_BASE_URL  = _resolve("LLM_BASE_URL", "BASE_URL", default="",
                         label="LLM base URL (blank = api.openai.com)")
LLM_API_KEY   = _resolve("LLM_API_KEY", "OPENAI_API_KEY", "API_KEY", default="",
                         secret=True, label="LLM API key")
LLM_TEMPERATURE = 0.0   # greedy: a temperature sweep is a separate experiment
LLM_TIMEOUT     = 120.0
LLM_MAX_RETRIES = 3

# Cost guard / smoke knob. `None` = every row of both splits (~3.7k LLM calls).
# BOTH arms are trimmed to the same rows, so the comparison never spans
# different populations. Set 300 for a smoke run; report numbers only at None.
LLM_LIMIT: "int | None" = None

# SpanCall inference knobs (mirrored from the v3.2 notebook so the two agree).
EVAL_BATCH_SIZE = 32
EVAL_WORKERS    = 4
MIN_JOINT_MARGIN = float(os.environ.get("MIN_JOINT_MARGIN", "0.02"))
SINGLE_CALL_ONLY = True

# Print shapes, never secrets: a run log that leaks a key gets pasted into a bug
# report, so the key is only ever reported as set / not set.
print("=" * 72)
print(f"HF_TOKEN        = {'<set>' if HF_TOKEN else '(none - public repos only)'}")
print(f"LLM_MODEL       = {LLM_MODEL}")
print(f"LLM_BASE_URL    = {LLM_BASE_URL or '(SDK default: api.openai.com)'}")
print(f"LLM_API_KEY     = {'<set>' if LLM_API_KEY else '(MISSING - cell 4 will stop)'}")
print(f"LLM_LIMIT       = {LLM_LIMIT}")
print(f"EVAL_SPLITS     = {EVAL_SPLITS}")
print(f"RESULTS_DIR     = {RESULTS_DIR}")
print("=" * 72)
'''

MD_DATA = """## 1. Eval data — the exact splits the paper scores

The paper's v3.2 headline table is computed on `subset_hermes_eval_clean.jsonl`
(1,659 rows) and `subset_xlam_eval_clean.jsonl` (2,000 rows). Those are
**stratified** subsets: `F_func` / `F_json` (groundable) plus `A_clean` /
`A_adv` (ungroundable, where a correct system must abstain).

The subsets are rebuilt here with the repo's own
`build_stratified_eval_subset()` rather than downloaded, because a downloaded
subset could have been produced by a different stratification seed and the
`n_F` column would then not match the paper by an amount nobody could explain.

### The `data/processed_v3_2` directory is **built, not downloaded**

`convoicon/spancall` hosts only `data/processed/` (v1) and `data/processed_v2/`.
There is no `processed_v3_2` on the Hub, and there must not be one: v2.7 object
decomposition changes the rendered parameter list, so a prebuilt directory would
silently reintroduce v2.6-shaped labels (one opaque `object` param, no dotted
leaves). The v3.2 notebook therefore **re-aligns** every split with
`scripts/realign_v2.py` at run time, and so does this cell. Pulling
`processed_v3_2` directly would fail; skipping the realign would produce labels
the checkpoint was not trained on.
"""

CODE_DATA = '''# 1. Build data/processed_v3_2 (realign, do NOT download it), then the subsets.
#
# `convoicon/spancall` has `data/processed/` (v1) and `data/processed_v2/` only.
# There is no `processed_v3_2` on the Hub: v2.7 object decomposition changes the
# rendered param list, so the v3.2 directory must be REGENERATED or the model is
# scored on labels it was never trained on. Same two-stage pipeline as
# `spancall_lean_v3_2_train_eval.ipynb` cell 3.
from huggingface_hub import snapshot_download
from spancall.lean.eval_p1 import build_stratified_eval_subset
import json, os, subprocess, sys, shutil

PROCESSED_V1 = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_V2 = os.path.join(PROJECT_ROOT, "data", "processed_v2")

HERMES_SPLITS = ["hermes_train", "hermes_dev", "hermes_calib", "hermes_eval",
                 "hermes_test_ood_schema", "hermes_test_ood_para",
                 "hermes_test_adversarial"]
XLAM_SPLITS = ["xlam_train", "xlam_dev", "xlam_calib", "xlam_eval",
               "xlam_test_ood_schema", "xlam_test_ood_para",
               "xlam_test_adversarial"]
NEEDED = [s for s in EVAL_SPLITS]          # only the splits we actually score


def _present(d):
    return all(os.path.exists(os.path.join(d, f"{s}.jsonl")) for s in NEEDED)


def _realign(src_dir, out_dir, splits, gate_split):
    """Re-align v1/v2 canonical splits through the v2.7 (object-decomposing) aligner."""
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "realign_v2.py"),
           "--src_dir", src_dir, "--out_dir", out_dir,
           "--splits", *splits, "--gate_split", gate_split,
           "--max_utt_stale_rate", "0.02", "--max_unresolvable", "0.02"]
    print(f"[data] realign {'/'.join(splits)} -> {out_dir}")
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    if rc != 0:
        raise SystemExit(
            f"realign_v2.py exited {rc}. The eval splits could not be rebuilt from "
            f"{src_dir}, so the LLM would be compared against a checkpoint on labels "
            f"it never saw. Inspect {out_dir}/realign_report.json."
        )


os.makedirs(DATA_DIR, exist_ok=True)

# --- Hermes: v1 processed/ -> v3.2 -----------------------------------------
# Per-split, not whole-directory: needing xLAM must not trigger a Hermes
# re-align (the aligner is deterministic, but it is minutes of work).
_hermes_missing = [s for s in NEEDED if s.startswith("hermes")
                   and not os.path.exists(os.path.join(DATA_DIR, f"{s}.jsonl"))]
if _hermes_missing:
    if not _present(PROCESSED_V1):
        print(f"[data] pulling v1 canonical splits from {HF_DATASET_ID}")
        snapshot_download(
            repo_id=HF_DATASET_ID, repo_type="dataset", local_dir=PROJECT_ROOT,
            allow_patterns=["data/processed/*.jsonl", "data/processed/*.json"],
            token=HF_TOKEN or None,   # anonymous if the prompt was skipped
        )
    _realign(PROCESSED_V1, DATA_DIR, HERMES_SPLITS, "hermes_eval")

# --- xLAM: v2 processed_v2/ -> v3.2 ----------------------------------------
_xlam_missing = [s for s in NEEDED if s.startswith("xlam")
                 and not os.path.exists(os.path.join(DATA_DIR, f"{s}.jsonl"))]
if _xlam_missing and all(
        os.path.exists(os.path.join(PROCESSED_V2, f"{s}.jsonl")) for s in XLAM_SPLITS):
    _realign(PROCESSED_V2, DATA_DIR, XLAM_SPLITS, "xlam_eval")
elif _xlam_missing:
    print(f"[data] v2 xLAM splits absent -> pulling from {HF_DATASET_ID}")
    snapshot_download(
        repo_id=HF_DATASET_ID, repo_type="dataset", local_dir=PROJECT_ROOT,
        allow_patterns=["data/processed_v2/xlam_*.jsonl", "data/processed_v2/*.json"],
        token=HF_TOKEN or None,
    )
    _realign(PROCESSED_V2, DATA_DIR, XLAM_SPLITS, "xlam_eval")

_got = [s for s in NEEDED if os.path.exists(os.path.join(DATA_DIR, f"{s}.jsonl"))]
if len(_got) != len(NEEDED):
    raise SystemExit(f"[data] missing eval splits in {DATA_DIR}: "
                     f"{sorted(set(NEEDED) - set(_got))}")
print(f"[data] processed_v3_2 ready: {_got}")

# --- Stratified subsets (the paper's exact population) ---------------------
SUBSET_PATHS = {}
for _sp in EVAL_SPLITS:
    _clean = os.path.join(RESULTS_DIR, f"subset_{_sp}_clean.jsonl")
    if not os.path.exists(_clean):
        print(f"[data] building stratified subset for {_sp} ...")
        # `limit=None` matters: the signature default is 500 rows, which would
        # quietly shrink the comparison to a sample the paper never reports.
        build_stratified_eval_subset(
            split_name=_sp,
            data_dir=DATA_DIR,
            out_path=_clean,
            limit=None,
            seed=SEED,
        )
    SUBSET_PATHS[_sp] = _clean
    _rows = [json.loads(l) for l in open(_clean, encoding="utf-8") if l.strip()]
    _strata = {}
    for _r in _rows:
        _strata[_r.get("stratum", "?")] = _strata.get(_r.get("stratum", "?"), 0) + 1
    print(f"[data] {_sp}: {len(_rows)} rows  strata={_strata}")

# Every downstream cell iterates THIS mapping, so both arms are guaranteed the
# same rows in the same order. A dict, not a set, because order is the contract.
print("\\nSubset files:")
for _k, _v in SUBSET_PATHS.items():
    print(f"  {_k:<14} -> {_v}")
'''

MD_MODEL = """## 2. SpanCall arm — the checkpoint rebuild recipe (lazy)

This cell only **defines** `load_resolver()`; nothing is downloaded here. Cell 3
calls it only if the frozen v3.2 predictions are unavailable, so the normal path
skips a ~1.5 GB fetch and a model construction that would go unused.

`convoicon/spancall-lean-v3-2` ships `model.pt` (a `state_dict`),
`resolve_config.json` (the rebuild recipe), and tokenizer files.

Two details that are easy to get wrong, and that this cell pins:

1. **`head_dim` is architecture, not a serving preference.** Rebuilding
   `LeanSpanCallModel` without it gets the 256 default and loads with
   `strict=False`, leaving the pointer head mis-dimensioned while printing no
   error. It is read from `resolve_config.json`.
2. **`resolve_mode="two_step"` is pinned at construction.** v3 has exactly one
   extraction mode; pinning it means the mode being scored is visible on the line
   that creates the resolver, not inferred from a library default.
"""

CODE_MODEL = '''# 2. Define the resolver rebuild recipe from the HF checkpoint.
#
# LAZY ON PURPOSE: this cell only defines `load_resolver()`. If cell 3 finds the
# v3.2 predictions (the normal case), it never calls it, and we skip an
# unnecessary ~1.5 GB download plus a model construction we would not use.
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import torch, os, sys, json

sys.path.insert(0, str(PROJECT_ROOT))
from span_template.sanitize import SPECIAL_MARKER_TOKENS
from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver

RESOLVER = None      # populated only if cell 3 needs to infer


def load_resolver(verbose: bool = True):
    """Download the checkpoint and build a ready `two_step` LeanResolver."""
    global RESOLVER
    if RESOLVER is not None:
        return RESOLVER

    _weights = hf_hub_download(HF_REPO_ID, "model.pt", cache_dir=HF_CACHE_DIR,
                               token=HF_TOKEN or None)
    _cfg_p   = hf_hub_download(HF_REPO_ID, "resolve_config.json", cache_dir=HF_CACHE_DIR,
                               token=HF_TOKEN or None)
    with open(_cfg_p, encoding="utf-8") as _fh:
        rcfg = json.load(_fh)
    if verbose:
        print("[hf] checkpoint:", _weights)
        print("[hf] resolve_config:", json.dumps(rcfg, indent=2, default=str)[:600])

    backbone       = rcfg["backbone"]
    max_seq_len    = int(rcfg["max_seq_len"])
    max_span_width = int(rcfg["max_span_width"])
    head_dim       = int(rcfg.get("head_dim", 256))   # ARCHITECTURE, not a serving knob
    max_calls      = 1 if SINGLE_CALL_ONLY else int(rcfg.get("max_calls", 1))

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    if verbose:
        print(f"[model] backbone={backbone} device={device} "
              f"max_seq_len={max_seq_len} head_dim={head_dim}")

    # The tokenizer ships in the checkpoint repo, so the marker tokens are already
    # in its vocabulary; `add_marker_tokens` is idempotent and re-checks anyway.
    tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO_ID, cache_dir=HF_CACHE_DIR, use_fast=True, token=HF_TOKEN or None)
    encoder = SpanCallTransformerEncoder(
        model_name_or_path=backbone,
        hidden_dropout_prob=0.1,
        tokenizer=tokenizer,
    )
    encoder.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))

    model = LeanSpanCallModel(
        encoder=encoder,
        head_dim=head_dim,
        max_span_width=max_span_width,
        hidden_dim=encoder.hidden_dim,
    )

    resolver = LeanResolver(
        model_path=None,            # weights are loaded explicitly below
        encoder=encoder,
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_span_width=max_span_width,
        max_calls=max_calls,
        call_min_prob=float(rcfg.get("call_min_prob", 0.25)),
        call_min_prob_ratio=float(rcfg.get("call_min_prob_ratio", 0.5)),
        min_joint_margin=float(rcfg.get("min_joint_margin", MIN_JOINT_MARGIN)),
        collapse_guard=True,
        prevent_param_collapse=True,
        resolve_mode="two_step",    # v3: the only mode
    )

    # `strict=False` is required (marker embeddings were resized), so the load is
    # VERIFIED rather than trusted: a cache that silently missed would leave
    # randomly initialised weights scoring plausible-looking numbers.
    _state = torch.load(_weights, map_location="cpu", weights_only=False)
    _incompat = model.load_state_dict(_state, strict=False)
    if verbose:
        print("[model] missing keys:", len(_incompat.missing_keys),
              "unexpected:", len(_incompat.unexpected_keys))
    _encoder_missing = [k for k in _incompat.missing_keys
                        if k.startswith(("encoder.", "backbone."))]
    if _encoder_missing:
        raise SystemExit(
            f"encoder weights did not load ({len(_encoder_missing)} keys). "
            f"Scoring untrained weights would look like a result."
        )

    model.to(device)
    model.eval()
    if hasattr(encoder, "eval"):
        encoder.eval()
    resolver.supports_candidate_parity = False   # explicit, not inherited
    RESOLVER = resolver
    return resolver


print("load_resolver() defined - the checkpoint is fetched only if inference is needed.")
'''

MD_INFER = """## 3. SpanCall arm — **reuse the v3.2 predictions, do not re-run**

SpanCall does **not** need to be re-run. The v3.2 train/eval notebook already
wrote `preds_eval_twostep.jsonl` (`hermes_eval`) and
`preds_xlam_eval_twostep.jsonl` (`xlam_eval`), and the paper's numbers came from
exactly those files. This cell **loads them**, and only runs inference if they
are genuinely absent.

That ordering is deliberate. Re-running inference would (a) burn GPU time to
reproduce numbers we already have, and (b) risk scoring a *fresh* prediction
file whose few flipped rows would move the SpanCall column in the paper without
anyone noticing. Loading the frozen predictions keeps the baseline side of the
comparison pinned to the run that was actually published.

The loaded file is validated against the subset before it is trusted: every
subset `sample_id` must be covered, otherwise the two arms would be scored on
different populations.

Inference is kept as a fallback path for the case where the v3.2 results
directory is not available (a fresh machine, or a re-run). If it is used, the
cell prints a warning that the SpanCall numbers are **new** and may not match
the paper.

### Where the predictions come from (read this before the first run)

`results_lean_v3_2/` is **not in git and not on the HuggingFace Hub** — the
training notebook only copies it to Google Drive. So a bare `git clone` starts
without it, and the reuse path cannot fire. Set `SPANCALL_V3_2_RESULTS_DIR`
(or `V3_2_RESULTS_DIR` in the config cell) to the directory holding:

| split | file |
| --- | --- |
| `hermes_eval` | `preds_eval_twostep.jsonl` |
| `xlam_eval` | `preds_xlam_eval_twostep.jsonl` |

When neither is found the cell says so explicitly and re-infers, and the
persisted `llm_parity.json` records `spancall_source` so a later reader can tell
which happened. Comparing a re-inferred SpanCall run against the paper's LLM
numbers is a mistake the metadata is designed to catch.

### If inference does run

It mirrors `predict_split()` from the v3.2 notebook with one deliberate
difference: **`parity_spans` is not passed.** That is not a shortcut —
`parity_spans` alone is a *diagnostic*: it reports whether the ground-truth span
was inside the candidate pool the model was actually given, and leaves
on/off-decoding untouched
([`resolver.py:1020-1027`](../spancall/lean/resolver.py#L1020)). The LLM arm has
no analogue, so passing it would add a column only one system can populate.
`e2e_parity` still lives in the training notebook's §4c; it is simply not part
of this comparison.

`protect_parity_spans` stays `False` in every path — that switch *does* force GT
spans past the `max_cands` truncation, which would be a ground-truth leak.
"""

CODE_INFER = '''# 3. Load the SpanCall predictions the v3.2 run already produced.
#
# NO RE-RUN by default: the paper's SpanCall column came from these exact files,
# and re-inferring would risk scoring a slightly different prediction file and
# moving the baseline side of the comparison without anyone noticing.
#
# The v3.2 notebook writes them under `results_lean_v3_2/`:
#   hermes_eval -> preds_eval_twostep.jsonl
#   xlam_eval   -> preds_xlam_eval_twostep.jsonl
from spancall.data.schema import UnifiedSample
import json, os, glob

# `V3_2_RESULTS_DIR` comes from the config cell. `results_lean_v3_2/` is not in
# git and not on the Hub, so a fresh clone must be pointed at it (Drive mount or
# local copy) - otherwise inference below is the only way to get SpanCall numbers.
RESULTS_V3_2 = V3_2_RESULTS_DIR or os.path.join(PROJECT_ROOT, "results_lean_v3_2")
_SEARCH_DIRS = [RESULTS_V3_2]
_DEFAULT_DIR = os.path.join(PROJECT_ROOT, "results_lean_v3_2")
if _DEFAULT_DIR not in _SEARCH_DIRS:
    _SEARCH_DIRS.append(_DEFAULT_DIR)

CANDIDATE_PRED_PATHS = {
    "hermes_eval": [
        "preds_eval_twostep.jsonl",
        "preds_hermes_eval_twostep.jsonl",
    ],
    # NOTE: `preds_eval_twostep.jsonl` is deliberately NOT listed here. That file
    # is the *Hermes* eval run; accepting it for xLAM would score Hermes
    # predictions against the xLAM subset and produce a plausible-looking but
    # meaningless number. A miss is the correct, loud outcome.
    "xlam_eval": [
        "preds_xlam_eval_twostep.jsonl",
        "preds_eval_twostep_xlam.jsonl",
    ],
}


def _find_existing_preds(split_name: str):
    for _d in _SEARCH_DIRS:
        for _f in CANDIDATE_PRED_PATHS.get(split_name, []):
            _p = os.path.join(_d, _f)
            if os.path.exists(_p):
                return _p
    # Last resort: any matching file anywhere under the candidate dirs.
    for _d in _SEARCH_DIRS:
        if not os.path.isdir(_d):
            continue
        hits = sorted(glob.glob(os.path.join(_d, "**", f"*{split_name}*twostep*.jsonl"),
                                 recursive=True))
        if hits:
            return hits[0]
        # A bare `preds_eval_twostep.jsonl` is unambiguous only if just one dir
        # has it and we are asked for hermes (handled above); for xLAM, a glob
        # on the split name is the honest answer, so return None rather than
        # silently scoring Hermes predictions as xLAM.
    return None


def _load_preds(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("sample_id") is not None:
                    out[str(r["sample_id"])] = r
    return out


def _validate_coverage(split_name: str, subset_path: str, by_id: dict) -> None:
    """Both arms must be scored on the same population, or nothing else means anything."""
    rows = [json.loads(l) for l in open(subset_path, encoding="utf-8") if l.strip()]
    if LLM_LIMIT is not None:
        rows = rows[: int(LLM_LIMIT)]
    want = [str(r.get("sample_id")) for r in rows]
    missing = [sid for sid in want if sid not in by_id]
    if missing:
        raise SystemExit(
            f"[spancall:{split_name}] {len(missing)}/{len(want)} subset rows have NO prediction "
            f"(first: {missing[0]}). Scoring SpanCall on a different population than the LLM "
            f"would make every comparison number wrong. Re-run the v3.2 notebook's §4c for "
            f"this split, or point CANDIDATE_PRED_PATHS at the right file."
        )
    extra = len(set(by_id) - set(want))
    print(f"[spancall:{split_name}] coverage OK: {len(want)}/{len(want)} subset rows"
          + (f" ({extra} extra preds ignored)" if extra else ""))


def run_spancall_inference(split_name: str, subset_path: str, out_path: str) -> dict:
    """FALLBACK ONLY: mirrors predict_split() from the v3.2 notebook."""
    rows = [json.loads(l) for l in open(subset_path, encoding="utf-8") if l.strip()]
    if LLM_LIMIT is not None:
        rows = rows[: int(LLM_LIMIT)]
    samples = [UnifiedSample.from_dict(r) for r in rows]
    print(f"[spancall:{split_name}] INFERENCE fallback on {len(samples)} rows ...")
    # Only now do we pay for the download + model build.
    resolver = load_resolver()

    items = [
        {
            "candidate_schemas": s.tools,
            "utterance": s.utterance,
            "recent_context": s.recent_context,
            "now": s.current_time_iso,
            # No "parity_spans": diagnosis-only, and the LLM arm has no analogue.
        }
        for s in samples
    ]
    results = resolver.resolve_candidates_batch(
        items,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=EVAL_WORKERS,
        protect_parity_spans=False,   # never True: it would leak GT spans past max_cands
    )

    by_id = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for s, res in zip(samples, results):
            rec = {
                "sample_id": s.sample_id,
                "stratum": s.stratum.value,
                "gt_tool": (s.ground_truth_calls[0].tool_name if s.ground_truth_calls else None),
                "gt_args": (s.ground_truth_calls[0].arguments if s.ground_truth_calls else {}),
                "gt_calls": [{"tool_name": c.tool_name, "arguments": dict(c.arguments or {})}
                             for c in (s.ground_truth_calls or [])],
                "n_gt_calls": len(s.ground_truth_calls or []),
                "calls": [c.as_dict() for c in (getattr(res, "calls", None) or [])],
                "hit": bool(res.hit),
                "status": getattr(res, "status", "hit" if res.hit else "abstain"),
                "payload": res.payload,
                "payload_partial": getattr(res, "payload_partial", {}),
                "tool_name": res.tool_name,
                "confidence": float(getattr(res, "confidence", 0.0) or 0.0),
                "joint_margin": float(getattr(res, "joint_margin", 0.0) or 0.0),
                "abstain_reason": res.abstain_reason,
                "latency_ms": float(getattr(res, "latency_ms", 0.0) or 0.0),
            }
            fh.write(json.dumps(rec, default=str) + "\\n")
            by_id[s.sample_id] = rec
    print(f"[spancall:{split_name}] wrote {out_path}")
    return by_id


SPANCALL_PREDS = {}
SPANCALL_SOURCE = {}     # split -> "reused:<path>" | "recomputed"
for _sp in EVAL_SPLITS:
    _existing = _find_existing_preds(_sp)
    if _existing is not None:
        SPANCALL_PREDS[_sp] = _load_preds(_existing)
        SPANCALL_SOURCE[_sp] = f"reused: {_existing}"
        print(f"[spancall:{_sp}] REUSED {_existing} (no inference)")
    else:
        print(f"[spancall:{_sp}] !! no v3.2 predictions found in {_SEARCH_DIRS}")
        print(f"   {_sp} will be RE-INFERRED, so these SpanCall numbers are NEW and")
        print(f"   will not reproduce the paper. To reuse the published run instead,")
        print(f"   set SPANCALL_V3_2_RESULTS_DIR (or V3_2_RESULTS_DIR in the config")
        print(f"   cell) to the directory holding preds_eval_twostep.jsonl for hermes")
        print(f"   and preds_xlam_eval_twostep.jsonl for xlam.")
        _out = os.path.join(RESULTS_DIR, f"preds_{_sp}_twostep.jsonl")
        SPANCALL_PREDS[_sp] = run_spancall_inference(_sp, SUBSET_PATHS[_sp], _out)
        SPANCALL_SOURCE[_sp] = "recomputed (paper numbers may differ)"
    _validate_coverage(_sp, SUBSET_PATHS[_sp], SPANCALL_PREDS[_sp])
    _n_hit = sum(1 for r in SPANCALL_PREDS[_sp].values() if r.get("hit"))
    _lat = [float(r.get("latency_ms") or 0.0) for r in SPANCALL_PREDS[_sp].values()]
    _lat = [x for x in _lat if x > 0]
    print(f"[spancall:{_sp}] hit={_n_hit}/{len(SPANCALL_PREDS[_sp])}  "
          f"mean_latency_ms={(sum(_lat)/len(_lat)) if _lat else float('nan'):.1f}")
'''

MD_LLM = """## 4. LLM arm — one hosted model, OpenAI SDK

Keys never appear in this file, and never in the output. Cell 0 asks for the
model name, the base URL and the API key the same way the v3.2 notebook asks for
your HF token: **press Enter and it will prompt you**, or set `LLM_MODEL` /
`LLM_BASE_URL` / `LLM_API_KEY` beforehand (env or a repo-root `.env`) if you are
running non-interactively. The key is only ever reported as `<set>`, and it is
never written into the notebook.

The base URL is what makes this endpoint-agnostic: point it at your own
vLLM/SGLang/TGI server (`http://host:8000/v1`), a gateway, or leave it blank for
`api.openai.com`. The cell refuses to run when no key was provided rather than
silently producing an all-abstain arm that would flatter SpanCall.

Two LLM-specific choices that matter for fairness:

- **Every returned tool call is captured**, not just `tool_calls[0]`. Scoring a
  multi-call answer by its first call would blame the LLM for a correct second
  call.
- **Malformed JSON arguments are kept** as `{"_raw": "<text>"}` rather than
  dropped. Dropping them would remove the row from the denominator and turn a
  parse failure into a free abstain.
"""

CODE_LLM = '''# 4. Call the hosted LLM through the OpenAI SDK.
#
# `_messages_for` / `_openai_tools` / `_llm_pred_row` / `LLMResponse` are imported
# from scripts/compare_llm_spancall.py: the notebook must not grow a second copy
# of the prompt or the schema converter, or the two arms stop being comparable.
from openai import OpenAI
from spancall.data.schema import UnifiedSample
import importlib.util, json, os, time

_spec = importlib.util.spec_from_file_location(
    "compare_llm_spancall", os.path.join(PROJECT_ROOT, "scripts", "compare_llm_spancall.py")
)
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

# `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` were already resolved in cell 0
# (env -> .env -> interactive prompt). Re-reading the environment here would
# reintroduce the exact failure the prompt exists to prevent: a key that was
# typed in at cell 0 and then not found by a second, independent lookup.
if not LLM_API_KEY:
    raise SystemExit(
        "no API key. Re-run cell 0 and type one at the 'LLM API key' prompt, or "
        "export LLM_API_KEY (also accepted: OPENAI_API_KEY, API_KEY) / put it in "
        "the repo-root .env. Scoring an all-abstain LLM arm would flatter SpanCall."
    )
print(f"[llm] model={LLM_MODEL}  base_url={LLM_BASE_URL or '(SDK default)'}  key=<set>")

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None,
                timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)


def run_llm(split_name: str, subset_path: str, out_path: str) -> dict:
    rows = [json.loads(l) for l in open(subset_path, encoding="utf-8") if l.strip()]
    if LLM_LIMIT is not None:
        rows = rows[: int(LLM_LIMIT)]          # identical trim to the SpanCall arm
    samples = [UnifiedSample.from_dict(r) for r in rows]
    print(f"[llm:{split_name}] {len(samples)} rows ...")

    by_id = {}
    n_err = 0
    for idx, s in enumerate(samples):
        if idx and idx % 25 == 0:
            print(f"[llm:{split_name}] {idx}/{len(samples)} ...")
        tools = cmp._openai_tools(s)
        messages = cmp._messages_for(s)
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=LLM_TEMPERATURE,
            )
        except Exception as exc:  # one bad row must not kill a paid-for run
            n_err += 1
            by_id[s.sample_id] = cmp._llm_pred_row(
                s.sample_id, cmp.LLMResponse(error=f"{type(exc).__name__}: {exc}")
            )
            continue
        latency_ms = (time.perf_counter() - t0) * 1000.0

        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            fn = getattr(tc, "function", None)
            if fn is None or not fn.name:
                continue
            args = fn.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    # Keep the raw text so the row scores as a wrong VALUE
                    # instead of vanishing from the denominator.
                    args = {"_raw": args}
            calls.append({"tool_name": fn.name, "payload": dict(args or {})})

        usage = getattr(resp, "usage", None)
        by_id[s.sample_id] = cmp._llm_pred_row(
            s.sample_id,
            cmp.LLMResponse(
                tool_calls=calls,
                text=(msg.content or ""),
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_ms=latency_ms,
            ),
        )

    with open(out_path, "w", encoding="utf-8") as fh:
        for sid, rec in by_id.items():
            fh.write(json.dumps(rec, default=str) + "\\n")

    n_hit = sum(1 for r in by_id.values() if r.get("hit"))
    tok_in = sum(int((r.get("features") or {}).get("prompt_tokens", 0)) for r in by_id.values())
    tok_out = sum(int((r.get("features") or {}).get("completion_tokens", 0)) for r in by_id.values())
    print(f"[llm:{split_name}] committed={n_hit}/{len(by_id)}  errors={n_err}  "
          f"tok_in={tok_in}  tok_out={tok_out}")
    return by_id


LLM_PREDS = {}
for _sp in EVAL_SPLITS:
    _out = os.path.join(RESULTS_DIR, f"preds_{_sp}_llm.jsonl")
    LLM_PREDS[_sp] = run_llm(_sp, SUBSET_PATHS[_sp], _out)
'''

MD_SCORE = """## 5. Score both arms with the same evaluator

`cmp.score()` rebuilds the prediction list **positionally from `ids`** on every
call. That is why the ids are passed explicitly: the evaluator aligns by index,
so a shorter prediction list would silently shift every later row onto the wrong
sample and produce a plausible-looking but wrong table.

Reported per split: `n_F`, `exact_F`, `hit_F`, `param_F1_F`, `call_set`,
`false_hit_on_A` (does a system answer a turn it *should* abstain on), plus the
LLM's latency and token totals.
"""

CODE_SCORE = '''# 5. Score SpanCall and the LLM on identical rows.
from spancall.eval.toolcall.evaluator import ToolCallEvaluator
from spancall.data.schema import UnifiedSample
import json, os

def _load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

SYSTEMS = {}      # split -> {system_name -> metrics}
PIVOTS  = {}      # split -> {system_name -> {"corpus|bucket" -> metrics}}

for _sp in EVAL_SPLITS:
    _rows = _load_rows(SUBSET_PATHS[_sp])
    if LLM_LIMIT is not None:
        _rows = _rows[: int(LLM_LIMIT)]
    _samples = [UnifiedSample.from_dict(r) for r in _rows]
    _ids = [s.sample_id for s in _samples]

    _sys = {}
    for _name, _preds in (
        ("spancall_two_step", SPANCALL_PREDS[_sp]),
        (f"llm:{LLM_MODEL}",  LLM_PREDS[_sp]),
    ):
        _m = cmp.score(_samples, _preds, ids=_ids)
        if not _m:
            continue
        _lat = [float((p or {}).get("latency_ms") or 0.0) for p in _preds.values()]
        _tok_in  = sum(int(((p or {}).get("features") or {}).get("prompt_tokens", 0)) for p in _preds.values())
        _tok_out = sum(int(((p or {}).get("features") or {}).get("completion_tokens", 0)) for p in _preds.values())
        _m["timing"] = {"mean_latency_ms": (sum(_lat) / len(_lat)) if _lat else 0.0}
        _m["cost"] = {"n_llm_calls": (len(_preds) if _name.startswith("llm:") else 0),
                      "prompt_tokens": _tok_in, "completion_tokens": _tok_out}
        _sys[_name] = _m

        # Per (corpus x tool-count bucket) cells, same edges as the paper's pivot.
        _pivot = {}
        for _key, _sids in cmp._slices(_rows, _ids).items():
            _sel_ids = set(_sids)
            _sel = [s for s in _samples if s.sample_id in _sel_ids]
            _cell = cmp.score(_sel, _preds, ids=_sids)
            if _cell:
                _pivot[_key] = _cell
        _sys[_name]["pivot"] = _pivot

    SYSTEMS[_sp] = _sys
    cmp._print_pivot(_sys, list(_sys.keys()))

    # Table layout is reimplemented here (not cmp._print_table) on purpose: the
    # script's footer talks about `lean_fast_path` / `lean_plus_fallback`, which
    # are three-system production concepts that do not exist in this 2-arm
    # experiment. Printing that footer would invite the reader to compare hit_F
    # across arms, which is exactly the mistake section 8 warns against.
    print("")
    _hdr = (f"{'system':<22}{'n':>6}{'n_F':>6}{'exact_F':>10}{'hit_F':>9}"
            f"{'param_F1':>10}{'call_set':>10}{'falsehit_A':>12}{'lat_ms':>9}")
    print("=" * len(_hdr))
    print(f"SPLIT {_sp}")
    print(_hdr)
    print("-" * len(_hdr))
    for _name, _m in _sys.items():
        _f = (_m.get("metrics_by_stratum") or {}).get("F", {})
        print(f"{_name:<22}"
              f"{int(_m.get('total_samples', 0)):>6}"
              f"{int(_f.get('n', 0) or 0):>6}"
              f"{cmp._fmt(_f.get('exact_payload_match')):>10}"
              f"{cmp._fmt(_f.get('hit_rate')):>9}"
              f"{cmp._fmt(_f.get('param_f1')):>10}"
              f"{cmp._fmt(_m.get('call_exact_set_match')):>10}"
              f"{cmp._fmt(_m.get('false_hit_rate_on_a'), 4):>12}"
              f"{cmp._fmt((_m.get('timing') or {}).get('mean_latency_ms'), 1):>9}")
    print("-" * len(_hdr))
    print("hit_F is a COMMITMENT rate. The LLM commits on nearly every row, so its")
    print("hit_F and SpanCall's hit_F are not the same quantity - compare exact_F.")
    print(f"[score] {_sp}: {len(_ids)} rows scored for {len(_sys)} systems")
'''

MD_COMPLEXITY = """## 6. Normal vs. Complex — the axis the paper argues on

The paper's central claim is not "we win everywhere". It is **"we win on
scalar-only calls and lose on calls with arrays/objects"** — a
representational limit of single-span pointer decoding, not an attention or
capacity failure.

So the LLM has to be measured on that same partition, not just overall. If the
LLM is uniformly better on both halves, the honest conclusion changes; if it
degrades on Complex too (or costs more), the split-scoring story survives.

`classify_sample_complexity()` is the same function the v3.2 notebook's §4c/§5
tables use, so these numbers drop straight into the paper's Normal/Complex table.
"""

CODE_COMPLEXITY = '''# 6. Score both systems on the Normal / Complex call partition.
from spancall.lean.eval_p1 import classify_sample_complexity
from spancall.data.schema import UnifiedSample
import json

COMPLEXITY = {}   # split -> {"normal"|"complex" -> {system -> metrics}}

for _sp in EVAL_SPLITS:
    _rows = [json.loads(l) for l in open(SUBSET_PATHS[_sp], encoding="utf-8") if l.strip()]
    if LLM_LIMIT is not None:
        _rows = _rows[: int(LLM_LIMIT)]
    _samples = [UnifiedSample.from_dict(r) for r in _rows]

    _by_cx = {"normal": [], "complex": []}
    for _raw, _s in zip(_rows, _samples):
        _by_cx[classify_sample_complexity(_raw)].append(_s)

    COMPLEXITY[_sp] = {}
    print(f"\\n=== {_sp}: Normal vs Complex ===")
    # `n_rows` is left-aligned: right-aligning it next to the left-aligned
    # `system` column ran the two headers together ("n_rowssystem").
    print(f"{'type':<10}{'n_rows':<9}{'system':<22}{'n_F':>6}{'exact_F':>10}{'hit_F':>9}{'param_F1':>10}")
    print("-" * 76)
    for _cx, _sel in _by_cx.items():
        if not _sel:
            continue
        _ids = [s.sample_id for s in _sel]
        _cell = {}
        for _name, _preds in (
            ("spancall_two_step", SPANCALL_PREDS[_sp]),
            (f"llm:{LLM_MODEL}",  LLM_PREDS[_sp]),
        ):
            _m = cmp.score(_sel, _preds, ids=_ids)
            if not _m:
                continue
            _cell[_name] = _m
            _f = (_m.get("metrics_by_stratum") or {}).get("F", {})
            print(f"{_cx:<10}{len(_sel):<9}{_name:<22}"
                  f"{int(_f.get('n', 0) or 0):>6}"
                  f"{cmp._fmt(_f.get('exact_payload_match')):>10}"
                  f"{cmp._fmt(_f.get('hit_rate')):>9}"
                  f"{cmp._fmt(_f.get('param_f1')):>10}")
        COMPLEXITY[_sp][_cx] = _cell
    print("-" * 76)
'''

MD_PERSIST = """## 7. Persist results + the paper-ready table

`llm_parity.json` holds every metric, both arms, both splits, plus the per-row
record so any number can be re-derived without re-paying for LLM tokens.

The Markdown block below is shaped like the paper's headline table so it can be
copied into `research_paper/sections/05_results.tex` directly. It replaces the
`B1a` row.
"""

CODE_PERSIST = '''# 7. Write JSON artifacts and print a paper-shaped table.
import json, os


def _eval_rows(path):
    """The rows that were actually scored, i.e. after the LLM_LIMIT trim."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows if LLM_LIMIT is None else rows[: int(LLM_LIMIT)]


_out = os.path.join(RESULTS_DIR, "llm_parity.json")
payload = {
    "meta": {
        "spancall_repo": HF_REPO_ID,
        "dataset": HF_DATASET_ID,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL or "(sdk default)",
        "llm_temperature": LLM_TEMPERATURE,
        "llm_limit": LLM_LIMIT,
        "seed": SEED,
        "eval_splits": EVAL_SPLITS,
        "resolve_mode": "two_step",
        "min_joint_margin": MIN_JOINT_MARGIN,
        "n_rows": {sp: len(_eval_rows(SUBSET_PATHS[sp])) for sp in EVAL_SPLITS},
        "spancall_source": SPANCALL_SOURCE,
        "note": ("One hosted-LLM arm via the OpenAI SDK. Both arms scored by the "
                 "same ToolCallEvaluator on the same rows in the same order. "
                 "No price table is assumed; token counts are measured, USD is not reported."),
    },
    "systems": SYSTEMS,
    "complexity": COMPLEXITY,
}
with open(_out, "w", encoding="utf-8") as _fh:
    json.dump(payload, _fh, indent=2, default=str)
print(f"[persist] wrote {_out}")

if LLM_LIMIT is not None:
    print(f"\\n!! LLM_LIMIT={LLM_LIMIT}: this is a SMOKE RUN on {LLM_LIMIT} rows per split.")
    print("!! Do not copy these numbers into the paper; re-run with LLM_LIMIT=None.")

print("\\n" + "=" * 96)
print("PAPER TABLE (replaces the B1a row) — stratum F metrics")
print("=" * 96)
print(f"{'split':<14}{'system':<22}{'n_F':>6}{'exact_F':>10}{'hit_F':>9}"
      f"{'param_F1':>10}{'false_hit_A':>13}{'lat_ms':>9}{'tok_in':>9}{'tok_out':>9}")
print("-" * 112)
for _sp in EVAL_SPLITS:
    for _name, _m in SYSTEMS[_sp].items():
        _f = (_m.get("metrics_by_stratum") or {}).get("F", {})
        print(f"{_sp:<14}{_name:<22}"
              f"{int(_f.get('n', 0) or 0):>6}"
              f"{cmp._fmt(_f.get('exact_payload_match')):>10}"
              f"{cmp._fmt(_f.get('hit_rate')):>9}"
              f"{cmp._fmt(_f.get('param_f1')):>10}"
              f"{cmp._fmt(_m.get('false_hit_rate_on_a'), 4):>13}"
              f"{cmp._fmt((_m.get('timing') or {}).get('mean_latency_ms'), 1):>9}"
              f"{int((_m.get('cost') or {}).get('prompt_tokens', 0) or 0):>9}"
              f"{int((_m.get('cost') or {}).get('completion_tokens', 0) or 0):>9}")
print("-" * 112)
print("exact_F / hit_F / param_F1 are stratum F (`F_func` + `F_json`).")
print("false_hit_A is the share of ungroundable turns the system answered anyway (lower is better).")
'''

MD_TAKEAWAY = """## 8. How to read this before putting it in the paper

Three things must be checked in the output above, in this order. If the first
one fails, the comparison is not publishable and no amount of framing fixes it.

1. **`n_F` must match between the two arms, per split.** If SpanCall shows
   `n_F=1110` and the LLM shows `n_F=1200`, they were scored on different
   populations and every other number is meaningless.
2. **`hit_F` is coverage, not accuracy, for the LLM.** The LLM almost never
   abstains, so its `hit_F` will sit near 1.0 while its `exact_F` is much lower.
   SpanCall's `hit_F` is a real commitment rate. Do not compare the two `hit_F`
   columns as if they meant the same thing; that is exactly the error the
   `lean_fast_path` vs `llm_only` distinction in `compare_llm_spancall.py`
   exists to prevent.
3. **`false_hit_A` is where SpanCall is genuinely weak.** v3.2 sits at
   0.482–0.574 on the eval sets, i.e. it answers roughly half the turns it should
   decline. The LLM will likely be better here, because it can read "there is no
   matching tool" from context. Report it; do not bury it.

Latency and tokens are measured in this run, but the sandbox/network path
dominates LLM latency, so treat the wall-clock comparison as indicative and the
token counts as the hard number.
"""

CELLS: List[Dict[str, Any]] = [
    {"type": "markdown", "source": MD_TITLE},
    {"type": "code",     "source": CODE_CONFIG},
    {"type": "markdown", "source": MD_DATA},
    {"type": "code",     "source": CODE_DATA},
    {"type": "markdown", "source": MD_MODEL},
    {"type": "code",     "source": CODE_MODEL},
    {"type": "markdown", "source": MD_INFER},
    {"type": "code",     "source": CODE_INFER},
    {"type": "markdown", "source": MD_LLM},
    {"type": "code",     "source": CODE_LLM},
    {"type": "markdown", "source": MD_SCORE},
    {"type": "code",     "source": CODE_SCORE},
    {"type": "markdown", "source": MD_COMPLEXITY},
    {"type": "code",     "source": CODE_COMPLEXITY},
    {"type": "markdown", "source": MD_PERSIST},
    {"type": "code",     "source": CODE_PERSIST},
    {"type": "markdown", "source": MD_TAKEAWAY},
]


def build(out_path: Path = OUT_PATH) -> Path:
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    for cell in CELLS:
        src = cell["source"]
        if cell["type"] == "markdown":
            nb.cells.append(new_markdown_cell(src.strip() + "\n"))
        else:
            nb.cells.append(new_code_cell(src.strip() + "\n"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
        fh.write("\n")
    return out_path


if __name__ == "__main__":
    p = build()
    nb = json.loads(p.read_text(encoding="utf-8"))
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"wrote {p}  ({len(nb['cells'])} cells: {n_code} code, {n_md} markdown)")