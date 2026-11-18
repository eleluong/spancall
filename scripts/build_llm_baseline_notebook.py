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

**What is configurable, and why it matters:** `LLM_LIMIT_FRACTION` trims every
arm to the same *fraction* of each split. It defaults to `0.1`, so the notebook
is a ~10% smoke run out of the box (≈366 paid calls instead of ≈3,659); set it
to `1.0` for the paper run. It is not a `rows[:N]` slice: the stratified subset
writes one contiguous block per cohort, so a head-slice would be 100% `F_func`
and would never exercise abstention. The trim is cohort-proportional, so the
smoke run's metrics stay comparable to the full run's.

## Before the first run

Cell 0 asks for four things, the same way the v3.2 notebook asks for your HF
token — **press Enter to be prompted**, or pre-set them for a non-interactive run:

| value | env var | notes |
| --- | --- | --- |
| HF access token | `HF_TOKEN` | optional; the `convoicon/*` repos are public |
| LLM model name | `LLM_MODEL` | e.g. `qwen3.7-flash`, or whatever your server serves |
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

# Pin the weights, or the numbers describe a moving target. A model repo's HEAD
# advances on every §7 re-push, so `HF_REPO_ID` names a repo, not a model. Set
# this to a commit sha (or tag) to make the eval reproducible; leave `None` to
# follow HEAD, in which case cell 7 records the sha it actually scored.
HF_REVISION     = None

# Publish the artifacts under `eval/llm_baseline/<run_tag>/` in HF_REPO_ID.
# OFF by default: a push is public and effectively permanent, and an eval folder
# that quietly accumulates a row per smoke run is worse than no folder at all.
PUSH_RESULTS_TO_HUB = False

DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed_v3_2")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results_lean_v3_2_llm")
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_checkpoint")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# The two eval splits behind the paper's headline table. Both are single-call
# subsets, which is why `SINGLE_CALL_ONLY=True` and `max_calls=1` below.
EVAL_SPLITS = ["hermes_eval", "xlam_eval"]

# Rows to SAMPLE from each split before any filtering. Must equal EVAL_LIMIT in
# `spancall_lean_v3_2_train_eval.ipynb`, which is 2000.
#
# This is not a tunable: it is the POPULATION DEFINITION. The frozen prediction
# files (`preds_eval_twostep.jsonl` = 1,659 rows, `preds_xlam_eval_twostep.jsonl`
# = 2,000 rows) were produced by sampling with `limit=2000`, so a subset built
# with a different limit is a different set of rows. `limit=None` samples the
# WHOLE split (9,101 xLAM rows) and then the frozen predictions cover only the
# first ~22% of it - the coverage check then fails with "N subset rows have NO
# prediction", which is correct but describes a self-inflicted wound: the two
# arms were pointed at different populations before either ran.
#
# `build_stratified_eval_subset()` allocates `int(round(target_n * ratio))` per
# cohort from a seed-shuffled per-cohort list, so `limit=2000` is deterministic
# and its output is a SUBSET of what `limit=None` would emit - which is exactly
# why a 9,101-row `_clean` file looks plausible and is wrong.
EVAL_LIMIT = 2000

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
LLM_MODEL     = _resolve("LLM_MODEL", default="qwen3.7-flash",
                         label="LLM model name")
LLM_BASE_URL  = _resolve("LLM_BASE_URL", "BASE_URL", default="",
                         label="LLM base URL (blank = api.openai.com)")
LLM_API_KEY   = _resolve("LLM_API_KEY", "OPENAI_API_KEY", "API_KEY", default="",
                         secret=True, label="LLM API key")
LLM_TEMPERATURE = 0.0   # greedy: a temperature sweep is a separate experiment
LLM_TIMEOUT     = 120.0
LLM_MAX_RETRIES = 3

# Cost guard / smoke knob, expressed as a FRACTION of each split.
#   1.0  -> every row of both splits (~3.7k LLM calls) = the paper run
#   0.1  -> ~10% of the rows (~366 LLM calls) = "does the pipeline work?" run
# Report numbers only at 1.0. Both arms are trimmed by the SAME function, so the
# comparison can never span different populations.
LLM_LIMIT_FRACTION = float(os.environ.get("LLM_LIMIT_FRACTION", "0.1"))

# Checkpointing. The LLM arm is the only part of this notebook that spends money,
# so every completed row is appended to disk before the next call goes out.
#   LLM_RESUME=1  -> a good run is reused instead of re-bought. Cell 4 then costs
#                    nothing and re-scores an older run for free.
#   LLM_RESUME=0  -> force a fresh run (the old artifacts are backed up, not lost).
# Only rows whose run fingerprint matches exactly are reused; see cell 4.
LLM_RESUME = os.environ.get("LLM_RESUME", "1").strip().lower() not in ("0", "false", "no")
LLM_CHECKPOINT_EVERY = int(os.environ.get("LLM_CHECKPOINT_EVERY", "25"))

# split -> how the paid rows in `preds_{split}_llm.jsonl` were obtained. Filled
# by cell 4 and persisted by cell 7: a resumed table is a real result, but a
# reader must be able to see that its rows came from a checkpoint rather than
# from the session that produced the table. Declared here so cell 7 does not
# depend on cell 4 having run.
LLM_SOURCE = {}

# SpanCall inference knobs (mirrored from the v3.2 notebook so the two agree).
EVAL_BATCH_SIZE = 32
EVAL_WORKERS    = 4
MIN_JOINT_MARGIN = float(os.environ.get("MIN_JOINT_MARGIN", "0.02"))
SINGLE_CALL_ONLY = True


def _limit_rows(rows, fraction=None):
    """Take `fraction` of each cohort, preserving file order. Deterministic.

    A plain `rows[:int(len(rows) * fraction)]` is NOT a valid sample here.
    `build_stratified_eval_subset()` emits ONE CONTIGUOUS BLOCK PER COHORT
    (F_func, then F_json, then A_clean, then A_adv), so the first 10% of the
    file is ~100% F_func: `false_hit_rate_on_a` would have an empty
    denominator and the smoke run would never exercise abstention at all -
    it would silently validate only the easy half of the task.

    Sampling per cohort keeps the mix (and therefore every metric) comparable
    to the full run, at a tenth of the tokens.
    """
    frac = LLM_LIMIT_FRACTION if fraction is None else float(fraction)
    rows = list(rows)
    if frac >= 1.0:
        return rows
    if frac <= 0.0:
        raise SystemExit("LLM_LIMIT_FRACTION must be > 0 (use 1.0 for the full run)")
    # Imported here, not at cell top: this runs after `%pip install -e .`, and
    # keeping the import local keeps cell 0 cheap.
    from spancall.lean.eval_p1 import classify_eval_cohort

    total = {}
    for _r in rows:
        _k = classify_eval_cohort(_r)
        total[_k] = total.get(_k, 0) + 1
    quota = {_k: max(1, int(round(_n * frac))) for _k, _n in total.items()}

    out, used = [], {}
    for _r in rows:
        _k = classify_eval_cohort(_r)
        if used.get(_k, 0) < quota[_k]:
            out.append(_r)
            used[_k] = used.get(_k, 0) + 1
    return out


# Print shapes, never secrets: a run log that leaks a key gets pasted into a bug
# report, so the key is only ever reported as set / not set.
print("=" * 72)
print(f"HF_TOKEN        = {'<set>' if HF_TOKEN else '(none - public repos only)'}")
print(f"LLM_MODEL       = {LLM_MODEL}")
print(f"LLM_BASE_URL    = {LLM_BASE_URL or '(SDK default: api.openai.com)'}")
print(f"LLM_API_KEY     = {'<set>' if LLM_API_KEY else '(MISSING - cell 4 will stop)'}")
print(f"LLM_LIMIT_FRACTION = {LLM_LIMIT_FRACTION}")
print(f"LLM_RESUME      = {LLM_RESUME}   (checkpoint every {LLM_CHECKPOINT_EVERY} rows)")
print(f"EVAL_SPLITS     = {EVAL_SPLITS}")
print(f"EVAL_LIMIT      = {EVAL_LIMIT}   (rows sampled per split; must match v3.2)")
print(f"RESULTS_DIR     = {RESULTS_DIR}")
print("=" * 72)
if LLM_LIMIT_FRACTION < 1.0:
    print(f"SMOKE MODE: scoring {LLM_LIMIT_FRACTION:.0%} of each split (~10% of the")
    print("token cost). These numbers validate the pipeline; they are NOT the paper's.")
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

### Three stages, matching §4.0 of the training notebook

| stage | output | what it does |
| --- | --- | --- |
| 1. stratify | `subset_{sp}.jsonl` | `build_stratified_eval_subset(limit=EVAL_LIMIT)`, seed-controlled, cohort-proportional |
| 2. single-call | `subset_{sp}.jsonl` (in place) | keep rows with exactly one GT call, because `SINGLE_CALL_ONLY` pins `max_calls=1` |
| 3. Step-0 gate | `subset_{sp}_clean.jsonl` | drop rows `build_input_multi()` cannot render; **this file is scored** |

The training notebook scores the `_clean` file, so this notebook does too.

All three stages matter for fairness, and each one was missing here at some point:

- **Stage 1's `limit`** — `EVAL_LIMIT` is the **population definition**, not a
  sampling convenience. The training notebook samples `limit=2000` and writes
  those rows to `preds_*_twostep.jsonl`; a subset sampled with any other limit
  (in particular `None`, which means *the whole split*) describes different rows.
  That mismatch is silent until the coverage check runs, and then it looks like
  missing predictions rather than like the sampling bug it is. `EVAL_LIMIT` in
  cell 0 must stay at `2000`.
- **Stage 2** — with `max_calls=1`, SpanCall *cannot* emit a second call. Scoring
  it on multi-call rows charges it with a structural miss, while the LLM (which
  returns several calls happily) is treated generously on the same rows. The
  paper's 1,659 / 2,000-row populations are already single-call; without this
  filter the LLM here would be scored on a larger, different population.
- **Stage 3** — the gate vetoed 0 rows in the published run, which is exactly why
  a silent divergence would go unnoticed until the data changed.

### The subset is rebuilt every run, never reused from disk

Stage 1-3 are unconditional. Caching them on `os.path.exists()` was a cache with
no invalidation key: a `subset_xlam_eval_clean.jsonl` left behind by a build with
a different `EVAL_LIMIT` is silently reused, and the population mismatch only
surfaces three cells later as "N subset rows have NO prediction". Rebuilding is
deterministic (fixed seed) and costs one line scan plus ~2k renders, so the cache
bought nothing and could only ever hide a stale sample.

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
#
# TWO stages, exactly as the v3.2 notebook does it (its §4.0). Stage 1 is the
# stratified sample; stage 2 is the **Step-0 pre-encode sanitize gate**, which
# drops rows whose tool schemas or utterance cannot be rendered by
# `build_input_multi()` at all. The training notebook writes the ungated sample
# to `subset_{sp}.jsonl` and the gated one to `subset_{sp}_clean.jsonl`, and
# then scores the `_clean` file.
#
# Skipping stage 2 here would be a fairness bug, not a shortcut: SpanCall's own
# headline numbers come from the post-gate file, so an ungated LLM arm would be
# scored on rows SpanCall was never allowed to attempt (and on rows whose
# schemas it cannot even render). The published run vetoed 0 rows, which is
# exactly why nobody would notice the divergence until it mattered.
from span_template.template import RecentRow, build_input_multi
from span_template.sanitize import SanitizeError
from spancall.data.align import tooldef_to_template_tool
from spancall.lean.eval_p1 import classify_eval_cohort
from spancall.data.schema import UnifiedSample as _US


def _step0_sanitize_gate(sub_path: str, clean_path: str, split_name: str):
    """Drop rows `build_input_multi()` rejects; report per cohort, like §4.0."""
    stats = {c: {"n": 0, "vetoed": 0} for c in ("F_func", "F_json", "A_clean", "A_adv")}
    keep = []
    with open(sub_path, encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            d = json.loads(raw)
            coh = classify_eval_cohort(d)
            st = stats.setdefault(coh, {"n": 0, "vetoed": 0})
            st["n"] += 1
            s = _US.from_dict(d)
            vetoed = False
            try:
                recent = [
                    RecentRow(key=str(r.get("key", "")), value=str(r.get("value", "")),
                              source=str(r.get("source", "recent")))
                    for r in (s.recent_context or [])
                ]
                build_input_multi(
                    [tooldef_to_template_tool(t) for t in (s.tools or [])],
                    s.utterance,
                    recent_context=recent,
                )
            except SanitizeError:
                vetoed = True            # rendered form would be unsafe to encode
            except Exception:
                vetoed = False           # NOT a sanitize veto: keep the row
            if vetoed:
                st["vetoed"] += 1
            else:
                keep.append(raw)

    with open(clean_path, "w", encoding="utf-8") as fh:
        for raw in keep:
            fh.write(raw + "\\n")

    print(f"[data] Step-0 sanitize gate ({split_name}):")
    for c in ("F_func", "F_json", "A_clean", "A_adv"):
        st = stats.get(c) or {"n": 0, "vetoed": 0}
        rate = (st["vetoed"] / st["n"]) if st["n"] else 0.0
        print(f"         {c:<8}: vetoed={st['vetoed']:>3} / {st['n']:<4} ({rate:.1%})")
    print(f"         kept {len(keep)} rows -> {clean_path}")
    if len(keep) == 0:
        raise SystemExit(f"[data] the sanitize gate vetoed every row of {split_name}")
    return len(keep)


def _single_call_filter(sub_path: str, split_name: str) -> int:
    """Keep only rows with exactly one ground-truth call, in place.

    `SINGLE_CALL_ONLY = True` above sets `max_calls=1` on the resolver, so
    SpanCall can never emit a second call. Scoring it on multi-call rows would
    charge it with a structural miss it is not allowed to fix, while the LLM -
    which happily returns several calls - would be scored generously on the same
    rows. The training notebook applies this filter (its §4.0), so the paper's
    1,659 / 2,000-row populations are already single-call; skipping it here
    would score a different, larger population than the paper reports.
    """
    with open(sub_path, encoding="utf-8") as fh:
        keep = [l for l in fh
                if l.strip() and len(json.loads(l).get("ground_truth_calls") or []) == 1]
    with open(sub_path, "w", encoding="utf-8") as fh:
        fh.writelines(keep)
    print(f"[data] [SINGLE_CALL_ONLY] {split_name}: kept {len(keep)} single-call rows")
    return len(keep)


# The subset is rebuilt on EVERY run, unconditionally.
#
# It used to be `if not os.path.exists(_sub)`, which is a cache with no
# invalidation key: a `subset_xlam_eval_clean.jsonl` left behind by an earlier
# version of this cell is silently reused. That is exactly how the
# "69/910 subset rows have NO prediction" failure happened - an earlier build
# passed `limit=None`, sampling all 9,101 xLAM rows instead of the paper's
# 2,000, and every later run happily reused the 9,101-row file while the frozen
# predictions still covered only 2,000. The population mismatch then surfaced as
# a coverage error three cells later, blaming the predictions for a subset bug.
#
# Rebuilding is cheap (a line scan of the split + ~2k renders) and the realign
# above stays cached, so there is nothing worth caching here. `build_stratified
# _eval_subset` is seed-deterministic, so an unconditional rebuild is a no-op
# when nothing changed and a repair when something did.
SUBSET_PATHS = {}
SUBSET_INFO = {}
for _sp in EVAL_SPLITS:
    _sub = os.path.join(RESULTS_DIR, f"subset_{_sp}.jsonl")          # ungated
    _clean = os.path.join(RESULTS_DIR, f"subset_{_sp}_clean.jsonl")  # gated, SCORED
    print(f"[data] building stratified subset for {_sp} (limit={EVAL_LIMIT}) ...")
    # `limit=EVAL_LIMIT`, NOT `None` and NOT the 500 default. This argument
    # defines the population: the frozen v3.2 predictions were produced by
    # sampling `limit=EVAL_LIMIT`, so any other value makes the two arms
    # incomparable before a single row is scored.
    SUBSET_INFO[_sp] = build_stratified_eval_subset(
        split_name=_sp,
        data_dir=DATA_DIR,
        out_path=_sub,
        limit=EVAL_LIMIT,
        seed=SEED,
    )
    # Order matters and matches the training notebook: sample -> single-call
    # -> sanitize gate. The `_clean` file must be the SAME population the
    # paper's SpanCall numbers were computed on.
    if SINGLE_CALL_ONLY:
        _single_call_filter(_sub, _sp)
    _step0_sanitize_gate(_sub, _clean, _sp)
    SUBSET_PATHS[_sp] = _clean
    _rows = [json.loads(l) for l in open(_clean, encoding="utf-8") if l.strip()]
    _strata = {}
    for _r in _rows:
        _strata[_r.get("stratum", "?")] = _strata.get(_r.get("stratum", "?"), 0) + 1

    # SINGLE-CALL ASSERTION. The filter above is now unconditional, so this can
    # only fire if the filter itself regressed. It stays because the failure it
    # catches - scoring a resolver with `max_calls=1` on rows that ask for two
    # calls - is invisible in every metric except as a mysterious hit-rate dip.
    if SINGLE_CALL_ONLY:
        _multi = sum(1 for _r in _rows
                     if len(_r.get("ground_truth_calls") or []) != 1)
        if _multi:
            raise SystemExit(
                f"[data] {_clean} holds {_multi} rows that are not single-call, but "
                f"SINGLE_CALL_ONLY=True. The single-call filter did not run, so this "
                f"cell would compare the arms on a population the paper never used."
            )

    # POPULATION REPORT. `sampled_total` and `clean rows` are printed TOGETHER
    # with the v3.2 reference counts because the mismatch that caused the
    # coverage failure was invisible in either number alone: 9,101 xLAM rows is
    # a perfectly valid subset, just not the one the predictions describe.
    _sampled = SUBSET_INFO[_sp]["sampled_total"]
    if _sp == "xlam_eval" and _sampled != 2000:
        print(f"[data] !! {_sp}: sampled {_sampled} rows, but the frozen v3.2 "
              f"predictions cover 2000. EVAL_LIMIT must be 2000.")
    _n_scored = len(_limit_rows(_rows))
    print(f"[data] {_sp}: sampled {_sampled} -> single-call+gate {len(_rows)} clean rows"
          f"  strata={_strata}")
    print(f"[data] {_sp}: {_n_scored} rows will be scored "
          f"(LLM_LIMIT_FRACTION={LLM_LIMIT_FRACTION})")
    print(f"[data] {_sp}: v3.2 reference = "
          f"{'1659' if _sp == 'hermes_eval' else '2000'} rows"
          f" (must match, or the frozen predictions describe another population)")

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
import torch, os, sys, json, re

sys.path.insert(0, str(PROJECT_ROOT))
from span_template.sanitize import SPECIAL_MARKER_TOKENS
from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver

RESOLVER = None      # populated only if cell 3 needs to infer
# The commit sha the weights came from. `None` until `load_resolver()` runs, but
# always bound so cell 7 can read it whether or not inference happened.
MODEL_REVISION_RESOLVED = None


def load_resolver(verbose: bool = True):
    """Download the checkpoint and build a ready `two_step` LeanResolver."""
    global RESOLVER
    if RESOLVER is not None:
        return RESOLVER

    _weights = hf_hub_download(HF_REPO_ID, "model.pt", cache_dir=HF_CACHE_DIR,
                               revision=HF_REVISION, token=HF_TOKEN or None)
    _cfg_p   = hf_hub_download(HF_REPO_ID, "resolve_config.json", cache_dir=HF_CACHE_DIR,
                               revision=HF_REVISION, token=HF_TOKEN or None)

    # The exact commit behind the weights, read back from the cache path with no
    # extra network call: `hf_hub_download` stores under `snapshots/<sha>/`.
    # Cell 7 records this in `meta.model_revision`, so a reader can tell which
    # weights a table belongs to even after the repo moves on.
    global MODEL_REVISION_RESOLVED
    _m = re.search(r"[\\/]snapshots[\\/]([0-9a-f]{7,40})[\\/]", _weights)
    MODEL_REVISION_RESOLVED = _m.group(1) if _m else (HF_REVISION or "(unknown)")
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

    # The tokenizer ships in the checkpoint repo, so its vocabulary ALREADY holds
    # the 15 marker tokens (50383 entries) while the backbone we just built from
    # base ModernBERT has 50368 embedding rows. `add_marker_tokens` reconciles the
    # two by comparing sizes rather than by counting how many tokens the
    # tokenizer newly registered - that count is 0 here, and gating the resize on
    # it left the matrix 15 rows short and made `load_state_dict` fail with a
    # `tok_embeddings.weight` size mismatch.
    tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO_ID, revision=HF_REVISION, cache_dir=HF_CACHE_DIR, use_fast=True,
        token=HF_TOKEN or None)
    encoder = SpanCallTransformerEncoder(
        model_name_or_path=backbone,
        hidden_dropout_prob=0.1,
        tokenizer=tokenizer,
    )
    _added = encoder.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))
    _rows = int(encoder.backbone.get_input_embeddings().num_embeddings)
    if verbose:
        print(f"[model] markers newly registered={_added}  "
              f"embedding_rows={_rows}  tokenizer_len={len(tokenizer)}")
    if _rows != len(tokenizer):
        raise SystemExit(
            f"embedding rows ({_rows}) != tokenizer length ({len(tokenizer)}); "
            f"the checkpoint state_dict cannot load into this model."
        )

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

### Two different failures, and only one of them is about predictions

The coverage check distinguishes them, because the fix is completely different
and the old message conflated them:

| symptom | cause | fix |
| --- | --- | --- |
| subset file is **larger than `EVAL_LIMIT`** | cell 1 sampled a different population (e.g. `limit=None` → all 9,101 xLAM rows) | set `EVAL_LIMIT=2000` and re-run cell 1 |
| subset ≤ `EVAL_LIMIT`, a few rows absent | inference was interrupted | re-run §4c of the v3.2 notebook |

A clean subset can never exceed `EVAL_LIMIT`: `build_stratified_eval_subset()`
caps at `limit` and the single-call + sanitize stages only remove rows. So
"larger than `EVAL_LIMIT`" is a *proof* that the subset came from a different
sampling call. Re-running inference, or repointing `CANDIDATE_PRED_PATHS`, cannot
fix that case — and the original message said to do exactly that, which is how a
one-argument bug turns into hours of wasted GPU time.

Inference is kept as a fallback path for the case where the v3.2 results
directory is not available (a fresh machine, or a re-run). If it is used, the
cell prints a warning that the SpanCall numbers are **new** and may not match
the paper.

### Where the predictions come from (read this before the first run)

`results_lean_v3_2/` is **not in git and not on the HuggingFace Hub** — the
training notebook only copies it to Google Drive. So a bare `git clone` starts
without it, and the reuse path cannot fire. There are **three** ways to supply
it, in the order the cell tries them:

| # | how | when to use it |
| --- | --- | --- |
| 1 | `SPANCALL_V3_2_RESULTS_DIR` (or `V3_2_RESULTS_DIR` in the config cell) pointing at a directory that already holds the files | local copy; Drive already mounted |
| 2 | **browser upload, `UPLOAD_V3_2_PREDS=1`** | a fresh Colab VM with no Drive and no clone history — nothing to mount, nothing to find |
| 3 | re-infer (fallback) | you actually want new SpanCall numbers |

The two files, by exact name — the upload is whitelisted against this table and
so is the on-disk search, so the two paths cannot disagree:

| split | file |
| --- | --- |
| `hermes_eval` | `preds_eval_twostep.jsonl` |
| `xlam_eval` | `preds_xlam_eval_twostep.jsonl` |

Why the upload is opt-in and why it *moves* the file: `files.upload()` is a
blocking widget (it would halt "Run all"), and it writes into the current
directory — which by then is the git checkout, **not** a directory this cell
searches. A file left there would produce the worst possible outcome: the user
sees "Saved …", the cell prints "no v3.2 predictions found", and the SpanCall
column is silently re-inferred. The upload therefore moves a whitelisted file
into `results_lean_v3_2/`, and every uploaded file gets a `sha256` recorded so
`llm_parity.json` can state which bytes it scored.

Filename whitelisting cannot catch a Hermes file uploaded under the xLAM name —
the `sample_id` coverage check below can, and it runs on the uploaded file too.

When none of the three paths is available the cell says so explicitly and
re-infers, and the persisted `llm_parity.json` records `spancall_source` so a
later reader can tell which happened. Comparing a re-inferred SpanCall run
against the paper's LLM numbers is a mistake the metadata is designed to catch.

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
import json, os, glob, shutil, hashlib, time

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


def _validate_coverage(split_name: str, subset_path: str, by_id: dict,
                       pred_path: str = None) -> None:
    """Both arms must be scored on the same population, or nothing else means anything.

    A missing prediction is ALWAYS fatal, but it has two very different causes and
    the old message could not tell them apart:

      1. the predictions are for the right rows and some are genuinely absent
         (an interrupted inference run) -> re-run §4c;
      2. the SUBSET is the wrong population, so the predictions were never meant
         to cover it (this is what `limit=None` produced: 9,101 xLAM rows vs the
         2,000 the frozen file covers) -> `EVAL_LIMIT` is wrong.

    The second case used to be reported as case 1, which sends the reader off to
    re-infer millions of tokens for a bug that lives in one argument. The check
    below distinguishes them by SIZE: if the predictions cover far fewer rows than
    the subset asks for while also being almost entirely a subset of it, the
    population was resampled, not truncated.
    """
    all_rows = [json.loads(l) for l in open(subset_path, encoding="utf-8") if l.strip()]
    rows = _limit_rows(all_rows)      # same cohort-proportional trim as every other cell
    want = [str(r.get("sample_id")) for r in rows]
    missing = [sid for sid in want if sid not in by_id]
    if missing:
        # Two causes, and the difference is decisive:
        #
        #  * the SUBSET is the wrong population (built with a different
        #    `EVAL_LIMIT`), so the frozen predictions were never meant to cover
        #    it. Signature: the subset was drawn from a LARGER pool than the
        #    paper's, so more rows survived sampling+filtering than EVAL_LIMIT.
        #  * inference was INTERRUPTED. Signature: the subset is a legitimate
        #    population (<= EVAL_LIMIT rows, since sampling caps it and the
        #    single-call/sanitize stages only ever remove rows); only some of its
        #    rows are absent from the prediction file.
        #
        # `len(all_rows) > _eval_limit` is exact, not a heuristic: sampling can
        # never emit more than `limit` rows, and both later stages only filter.
        # A clean subset larger than EVAL_LIMIT therefore PROVES it came from a
        # different sampling call - no amount of re-inference can fix that.
        #
        # `globals().get`, not a bare name: the offline smoke test exec's this
        # cell with a namespace that only carries the evaluation names, so a
        # direct reference would raise NameError there and only there. The value
        # still comes from the notebook at run time.
        _eval_limit = globals().get("EVAL_LIMIT")
        if _eval_limit and len(all_rows) > _eval_limit:
            raise SystemExit(
                f"[spancall:{split_name}] POPULATION MISMATCH, not a missing prediction. "
                f"The subset file holds {len(all_rows)} rows, which is more than "
                f"EVAL_LIMIT={_eval_limit}: it was sampled from a larger pool than the "
                f"paper's, so {len(missing)} of the {len(want)} scored rows were never "
                f"going to have a prediction (first: {missing[0]}). The frozen v3.2 "
                f"file covers the {_eval_limit}-row stratified subset only, so re-running "
                f"inference or repointing CANDIDATE_PRED_PATHS will NOT help. Set "
                f"EVAL_LIMIT={_eval_limit} in cell 0 and re-run cell 1."
            )
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
    rows = _limit_rows(rows)          # identical trim to the LLM arm
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
    # WALL CLOCK is measured here and reported SEPARATELY from ``latency_ms``.
    # The two answer different questions: ``latency_ms`` is the amortised
    # per-sample cost (comparable to the LLM's per-call time), while this
    # wall clock is the throughput actually achieved on whatever hardware ran
    # the cell. Conflating them is what let the old per-row timer look honest
    # while scaling with the batch size.
    _wall_t0 = time.perf_counter()
    results = resolver.resolve_candidates_batch(
        items,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=EVAL_WORKERS,
        protect_parity_spans=False,   # never True: it would leak GT spans past max_cands
    )
    _wall_s = time.perf_counter() - _wall_t0
    print(f"[spancall:{split_name}] wall clock {_wall_s:.2f}s for {len(results)} rows "
          f"= {len(results) / _wall_s:.1f} rows/sec (batch_size={EVAL_BATCH_SIZE}, "
          f"workers={EVAL_WORKERS})")

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


# --------------------------------------------------------------------------- #
# Optional Colab upload: a THIRD way to supply the frozen v3.2 predictions.
#
# The two existing ways are a Drive mount (`SPANCALL_V3_2_RESULTS_DIR`) and a
# local copy. Both need a filesystem the runtime can already see, which a fresh
# Colab VM does not have. This adds a browser upload without weakening either
# guarantee:
#
#  * OPT-IN (`UPLOAD_V3_2_PREDS=1`). `files.upload()` is a blocking widget: under
#    "Run all" it stops the kernel until the user clicks. Making it the default
#    would turn "Run all" into "Run all, then wait forever".
#  * WHITELISTED BY NAME against `CANDIDATE_PRED_PATHS`. That dict is also what
#    `_find_existing_preds` searches, so the upload path and the lookup path
#    cannot disagree about which filename belongs to which split. Anything else
#    is refused loudly rather than dropped next to the clone.
#  * MOVED into `RESULTS_V3_2`. `files.upload()` writes to the CWD, which after
#    `os.chdir(PROJECT_ROOT)` is the git checkout - NOT a directory
#    `_find_existing_preds` searches. Without the move the user would see
#    "Saved ..." and the very next lines would print "no v3.2 predictions found"
#    and RE-INFER, silently replacing the paper's SpanCall column with a fresh
#    one. That is the divergence this whole cell is written to prevent.
#  * CONTENT IS STILL VALIDATED BY `_validate_coverage` below. A Hermes file
#    uploaded under the xLAM name cannot be caught by a filename whitelist; the
#    sample_id coverage check is the thing that actually fails, and it is the
#    same check the on-disk paths go through.
#  * PROVENANCE is recorded as a sha256 prefix, so `llm_parity.json` can say
#    which bytes it scored long after the Colab runtime is gone.
# --------------------------------------------------------------------------- #
UPLOADED_PROVENANCE = {}


def _sha256(path: str) -> str:
    _h = hashlib.sha256()
    with open(path, "rb") as _fh:
        for _chunk in iter(lambda: _fh.read(1 << 20), b""):
            _h.update(_chunk)
    return _h.hexdigest()


UPLOAD_V3_2_PREDS = os.environ.get("UPLOAD_V3_2_PREDS", "0").strip().lower() not in (
    "", "0", "false", "no")
try:                                   # absent outside Colab / nbconvert / CI
    from google.colab import files as _colab_files
    IN_COLAB = True
except ImportError:
    _colab_files = None
    IN_COLAB = False

if not UPLOAD_V3_2_PREDS:
    print("[upload] off (set UPLOAD_V3_2_PREDS=1 to upload predictions in the browser)")
elif not IN_COLAB:
    print("[upload] UPLOAD_V3_2_PREDS=1 but this is not Colab -> using files already "
          "on disk")
else:
    _WANTED = {f for _fs in CANDIDATE_PRED_PATHS.values() for f in _fs}
    print(f"[upload] pick {sorted(_WANTED)} (Ctrl-C / Cancel to skip)")
    try:
        _uploaded = _colab_files.upload() or {}
    except Exception as _exc:          # a cancelled widget must not kill the cell
        _uploaded = {}
        print(f"[upload] cancelled ({type(_exc).__name__}); continuing with disk state")

    _dest = RESULTS_V3_2
    os.makedirs(_dest, exist_ok=True)
    for _fn, _blob in _uploaded.items():
        if _fn not in _WANTED:
            print(f"[upload] IGNORED {_fn} ({len(_blob)} bytes): not one of "
                  f"{sorted(_WANTED)}. Rename it and re-run if it is a v3.2 preds file.")
            continue
        _src = os.path.join(str(PROJECT_ROOT), _fn)     # CWD, because of the chdir
        _dst = os.path.join(_dest, _fn)
        if os.path.exists(_dst):
            print(f"[upload] replacing existing {_dst}")
        shutil.move(_src, _dst)
        _split_of = next((_sp for _sp, _fns in CANDIDATE_PRED_PATHS.items() if _fn in _fns),
                         None)
        if _split_of is not None:
            UPLOADED_PROVENANCE[_split_of] = {
                "path": _dst, "bytes": len(_blob), "sha256": _sha256(_dst)}
        print(f"[upload] {_fn} -> {_dst}  ({len(_blob)} bytes, "
              f"sha256={UPLOADED_PROVENANCE.get(_split_of, {}).get('sha256', '?')[:12]})")


SPANCALL_PREDS = {}
SPANCALL_SOURCE = {}     # split -> "reused:<path>" | "recomputed"
for _sp in EVAL_SPLITS:
    _existing = _find_existing_preds(_sp)
    if _existing is not None:
        SPANCALL_PREDS[_sp] = _load_preds(_existing)
        # Which BYTES were scored, not just which path. An uploaded file has no
        # git history and no Hub revision behind it, so a sha256 prefix in
        # `meta.spancall_source` is the only thing that lets a reader verify
        # later that the paper's SpanCall column came from the published run.
        _prov = UPLOADED_PROVENANCE.get(_sp)
        SPANCALL_SOURCE[_sp] = (
            f"reused: {_existing}"
            + (f" sha256={_prov['sha256'][:12]}" if _prov else "")
        )
        print(f"[spancall:{_sp}] REUSED {_existing} (no inference)"
              + (f"  sha256={_prov['sha256'][:12]} (uploaded)" if _prov else ""))
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

### Checkpointing: a good run is saved, a broken one costs one row

This is the only cell that spends money, so it is written to survive an
interruption and to *not* re-buy work it already has.

- Every completed row is appended to a checkpoint file and `fsync`-ed
  **before the next call goes out**, so a crash, a Colab timeout or a closed
  laptop can lose at most one paid call. The merged
  `preds_{split}_llm.jsonl` is refreshed atomically every
  `LLM_CHECKPOINT_EVERY` rows, so a partial run is still scoreable.
- Re-running the cell with `LLM_RESUME=1` (the default) **resumes**: rows
  already paid for are read from the checkpoint and only the missing ones are
  fetched. A complete checkpoint prints
  `COMPLETE run reused - 0 API calls, $0 spent`, and cells 5-7 re-score it for
  free. That is how a good run becomes the run you report.
- Reuse is gated on a **fingerprint** of the model, base URL, temperature, the
  row set, and every rendered prompt + tool schema. Change any of those and the
  fingerprint changes, so the notebook starts a separate checkpoint instead of
  splicing answers from two different prompts into one table. A 10% smoke run
  and the 100% paper run therefore never share a checkpoint.
- **Failed calls are not treated as complete.** An error row keeps its place in
  the denominator but is retried on the next run rather than being frozen into
  the table as a free abstention.
- `LLM_RESUME=0` forces a fresh run; the previous artifacts are renamed to
  `*.bak-<timestamp>` rather than deleted, so a bad knob cannot destroy good
  data.

If some rows are still missing after a run, the cell raises rather than scoring
a short prediction list - the evaluator aligns predictions to rows
positionally, so a missing row would silently shift every later row onto the
wrong sample.

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


def _fingerprint(split_name: str, samples) -> str:
    """Identity of one exact paid run.

    Reuse is only safe if nothing that changes a completion changed: the model,
    the endpoint, the temperature, the row set, and the rendered prompt + tool
    schema of every row. Hashing the prompt is what catches a template edit -
    without it, a resumed run would silently splice answers produced by two
    different prompts into a single table.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps({"model": LLM_MODEL, "base_url": LLM_BASE_URL,
                         "temperature": LLM_TEMPERATURE,
                         "split": split_name}, sort_keys=True).encode("utf-8"))
    for s in samples:
        h.update(str(s.sample_id).encode("utf-8"))
        h.update(json.dumps({"m": cmp._messages_for(s), "t": cmp._openai_tools(s)},
                            sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def _ckpt_path(out_path: str, fp: str) -> str:
    """One checkpoint per fingerprint, so a smoke run can never poison a paper run."""
    return out_path[: -len(".jsonl")] + f".ckpt-{fp}.jsonl"


def _is_complete(rec: dict) -> bool:
    """A row is reusable unless the call itself failed.

    Transient failures are deliberately NOT frozen: an error row is re-attempted
    on the next run rather than baked into the table as a free abstention.
    """
    return not (rec.get("features") or {}).get("llm_error")


def _load_ckpt(path: str, fp: str) -> dict:
    """Rows already paid for under `fp`. Ignores a torn final line."""
    good = {}
    if not os.path.exists(path):
        return good
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # killed mid-write; that row is simply re-bought
            if rec.get("_fingerprint") == fp and rec.get("sample_id") is not None:
                good[str(rec["sample_id"])] = rec
    return good


def _write_jsonl_atomic(path: str, rows) -> None:
    """Write via tmp + rename, so a crash never leaves a half-written artifact."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, default=str) + "\\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run_llm(split_name: str, subset_path: str, out_path: str) -> dict:
    rows = [json.loads(l) for l in open(subset_path, encoding="utf-8") if l.strip()]
    rows = _limit_rows(rows)          # identical trim to the SpanCall arm
    samples = [UnifiedSample.from_dict(r) for r in rows]
    print(f"[llm:{split_name}] {len(samples)} rows ...")

    fp = _fingerprint(split_name, samples)
    ckpt = _ckpt_path(out_path, fp)
    order = [str(s.sample_id) for s in samples]

    def _snapshot(by_id):
        # Strip the internal key ON EVERY write, not just the final one: a
        # mid-run crash leaves the artifact in exactly the same shape as a
        # clean finish, and a copy per row means the returned `by_id` rows are
        # never mutated behind the caller's back.
        out = []
        for _sid in order:
            if _sid in by_id:
                _r = dict(by_id[_sid])
                _r.pop("_fingerprint", None)
                out.append(_r)
        return out

    if not LLM_RESUME:
        # Never destroy a good run just because someone flipped the knob.
        _stamp = time.strftime("%Y%m%d-%H%M%S")
        for _p in (ckpt, out_path):
            if os.path.exists(_p):
                os.replace(_p, f"{_p}.bak-{_stamp}")
                print(f"[llm:{split_name}] LLM_RESUME=0 -> kept previous run as "
                      f"{os.path.basename(_p)}.bak-{_stamp}")

    by_id = _load_ckpt(ckpt, fp)
    _have = {sid for sid, r in by_id.items() if _is_complete(r)}
    _todo = [s for s in samples if str(s.sample_id) not in _have]
    if by_id:
        print(f"[llm:{split_name}] checkpoint {os.path.basename(ckpt)}: "
              f"{len(_have)}/{len(samples)} rows already paid for, {len(_todo)} to fetch")
    if not _todo:
        LLM_SOURCE[split_name] = (f"checkpoint reuse ({len(_have)}/{len(samples)} rows, "
                                 f"0 API calls this run, fp={fp})")
        print(f"[llm:{split_name}] COMPLETE run reused - 0 API calls, $0 spent")
    else:
        LLM_SOURCE[split_name] = (f"api ({len(_todo)} fetched this run, "
                                 f"{len(_have)} resumed, fp={fp})")
        print(f"[llm:{split_name}] fingerprint {fp}  ckpt -> {ckpt}")

    n_err = 0
    _since = 0
    # Append + fsync per row: ~1ms against a ~1s completion, so the cost of being
    # safe is unmeasurable and a crash can never lose more than one paid call.
    with open(ckpt, "a", encoding="utf-8") as _ck:
        for idx, s in enumerate(_todo):
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
                rec = cmp._llm_pred_row(
                    s.sample_id, cmp.LLMResponse(error=f"{type(exc).__name__}: {exc}")
                )
            else:
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
                rec = cmp._llm_pred_row(
                    s.sample_id,
                    cmp.LLMResponse(
                        tool_calls=calls,
                        text=(msg.content or ""),
                        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                        latency_ms=latency_ms,
                    ),
                )

            rec["_fingerprint"] = fp
            by_id[str(s.sample_id)] = rec
            _ck.write(json.dumps(rec, default=str) + "\\n")
            _ck.flush()
            os.fsync(_ck.fileno())

            _since += 1
            if _since >= LLM_CHECKPOINT_EVERY:
                _since = 0
                # Refresh the merged artifact mid-run so the work is scoreable
                # even if the process dies before the final write.
                _write_jsonl_atomic(out_path, _snapshot(by_id))
                print(f"[llm:{split_name}] {idx + 1}/{len(_todo)} new rows "
                      f"({len(by_id)}/{len(samples)} total) -> {os.path.basename(out_path)}")

    out_rows = _snapshot(by_id)    # clean rows, in subset order

    _missing = [sid for sid in order if sid not in by_id]
    if _missing:
        raise SystemExit(
            f"[llm:{split_name}] {len(_missing)}/{len(order)} rows have NO prediction "
            f"(first: {_missing[0]}). A partial row set would shift every later row in "
            f"the positional scorer, so it is refused. Re-run cell 4: it resumes from "
            f"{ckpt} and only fetches what is missing."
        )

    _write_jsonl_atomic(out_path, out_rows)

    n_hit = sum(1 for r in out_rows if r.get("hit"))
    tok_in = sum(int((r.get("features") or {}).get("prompt_tokens", 0)) for r in out_rows)
    tok_out = sum(int((r.get("features") or {}).get("completion_tokens", 0)) for r in out_rows)
    print(f"[llm:{split_name}] committed={n_hit}/{len(out_rows)}  errors={n_err}  "
          f"tok_in={tok_in}  tok_out={tok_out}")
    if n_err:
        # Surface the distinct provider messages: an auth/quota/model-name
        # problem fails EVERY row with the same text, and a zero-metric LLM row
        # is the downstream symptom, not the cause.
        _msgs = {}
        for r in out_rows:
            _e = (r.get("features") or {}).get("llm_error")
            if _e:
                _msgs[_e] = _msgs.get(_e, 0) + 1
        for _m, _c in sorted(_msgs.items(), key=lambda kv: -kv[1])[:5]:
            print(f"[llm:{split_name}] !! provider error x{_c}: {_m[:200]}")
    _left = sum(1 for r in out_rows if not _is_complete(r))
    if _left:
        print(f"[llm:{split_name}] !! {_left} row(s) failed and stay checkpointed as "
              f"incomplete. Re-run cell 4 to retry only those (the "
              f"{len(out_rows) - _left} good rows are reused for free).")
    print(f"[llm:{split_name}] saved -> {out_path}  (resume state: {ckpt})")
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

### Latency is reported as a distribution, not a mean

Both arms record `latency_ms` per row, so the notebook prints mean / p50 / p90 /
p95 / p99 / max per function call, then p90 again per **corpus × tool-count
bucket**. A mean answers "which arm is slower on average"; a paper claim about
serving cost needs the tail, because that is what a p99 SLO is written against.

Percentiles are **nearest-rank** (no interpolation): p90 is the smallest observed
latency such that at least 90% of calls are at or below it. Exact, so the two
arms on identical rows are directly comparable, and a 231-row smoke run cannot
invent a percentile from a fitted distribution.

Rows with `latency_ms == 0` are **excluded**, and `n_timed` says how many
survived. Zero means the call was never timed — an errored LLM response, or a
reused SpanCall prediction written before timings were recorded. Counting those
as 0 ms would make a broken arm look *faster*, which is the one error direction
that flatters a baseline.
"""

CODE_SCORE = '''# 5. Score SpanCall and the LLM on identical rows.
from spancall.eval.toolcall.evaluator import ToolCallEvaluator
from spancall.data.schema import UnifiedSample
import json, os, math

def _load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _lat_stats(preds, ids=None):
    """Latency distribution per function call (one call per row: SINGLE_CALL_ONLY).

    Percentiles use the nearest-rank definition: p90 is the smallest observed
    latency such that at least 90% of the calls are at or below it. With a few
    hundred rows this is what a reader means by "p90", and it is exact (no
    interpolation), so two systems on the same rows are directly comparable.

    Zero-latency rows are dropped and `n_timed` reports the survivors. Those are
    rows the system never timed: an errored LLM call, or a reused SpanCall
    prediction written before timings were recorded. Averaging them in as 0ms
    would make a failing arm look FAST, which is the one direction of error that
    flatters the baseline.
    """
    # Prediction dicts are keyed by str(sample_id) (see cell 4 / _load_preds),
    # while `ids` carries the raw sample_id. Normalise both: on an int-keyed
    # split a naive `_sid not in ids` would match nothing and report n_timed=0
    # for a perfectly good arm.
    _keep = {str(x) for x in ids} if ids is not None else None
    vals = []
    for _sid, _p in preds.items():
        if _keep is not None and str(_sid) not in _keep:
            continue
        _v = float((_p or {}).get("latency_ms") or 0.0)
        if _v > 0:
            vals.append(_v)
    if not vals:
        return {"n_timed": 0}
    vals.sort()

    def _pct(q):
        k = max(1, math.ceil(q / 100.0 * len(vals)))
        return vals[min(k, len(vals)) - 1]

    return {
        "n_timed": len(vals),
        "mean": sum(vals) / len(vals),
        "p50": _pct(50),
        "p90": _pct(90),
        "p95": _pct(95),
        "p99": _pct(99),
        "max": vals[-1],
    }


def _print_latency(sys_metrics, split_name):
    """Two views: the distribution per arm, then p90 per corpus x tool bucket."""
    _hdr = (f"{'system':<22}{'n_timed':>8}{'mean':>9}{'p50':>8}"
            f"{'p90':>8}{'p95':>8}{'p99':>8}{'max':>9}")
    print("\\nLATENCY PER FUNCTION CALL (ms) - " + split_name)
    print("=" * len(_hdr))
    print(_hdr)
    print("-" * len(_hdr))
    for _name, _m in sys_metrics.items():
        _t = _m.get("timing") or {}
        if not _t.get("n_timed"):
            print(f"{_name:<22}{'0':>8}   (nothing timed; see the error rows)")
            continue
        print(f"{_name:<22}{_t['n_timed']:>8}"
              f"{_t['mean']:>9.1f}{_t['p50']:>8.1f}{_t['p90']:>8.1f}"
              f"{_t['p95']:>8.1f}{_t['p99']:>8.1f}{_t['max']:>9.1f}")
    print("-" * len(_hdr))
    print("Nearest-rank percentiles over positive per-row latencies.")
    print("LLM latency = one network round trip to the hosted endpoint.")
    print("SpanCall latency = AMORTISED per-sample cost: its own render/mask,")
    print("  plus its share of the batched backbone forward (batch_ms / rows),")
    print("  plus head scoring and decode. It is NOT the wall time of the batch,")
    print("  so it is comparable across models but understates GPU throughput.")
    print("  For throughput, see the rows/sec line above, which is wall clock.")


def _print_latency_pivot(sys_metrics, rows, ids):
    """p90 latency per (corpus x tool-count bucket), the paper's slice axis.

    A single p90 hides the fact that both arms slow down as the candidate list
    grows. If SpanCall's p90 grows with the tool count and the LLM's does not,
    that is a real finding; if both are flat, the latency claim is not
    slice-dependent and should not be presented as if it were.
    """
    _slices = cmp._slices(rows, ids)
    if not _slices:
        return
    _names = list(sys_metrics.keys())
    _w = max(len(k) for k in _slices)
    print("\\np90 LATENCY BY CORPUS x TOOL-COUNT BUCKET (ms)")
    print("-" * (_w + 8 + 16 * len(_names)))
    print(f"{'slice':<{_w}}{'n':>8}" + "".join(f"{n[:14]:>16}" for n in _names))
    for _key, _sids in _slices.items():
        _sel = set(_sids)
        _cells = []
        for _n in _names:
            _t = (sys_metrics[_n].get("latency_by_slice") or {}).get(_key) or {}
            _cells.append(f"{_t['p90']:.1f}" if _t.get("n_timed") else "n/a")
        print(f"{_key:<{_w}}{len(_sids):>8}" + "".join(f"{c:>16}" for c in _cells))
    print("-" * (_w + 8 + 16 * len(_names)))
    print("Cells marked n/a had no timed call on those rows.")


SYSTEMS = {}      # split -> {system_name -> metrics}
PIVOTS  = {}      # split -> {system_name -> {"corpus|bucket" -> metrics}}

for _sp in EVAL_SPLITS:
    _rows = _limit_rows(_load_rows(SUBSET_PATHS[_sp]))
    _samples = [UnifiedSample.from_dict(r) for r in _rows]
    _ids = [s.sample_id for s in _samples]
    _slices = cmp._slices(_rows, _ids)

    _sys = {}
    for _name, _preds in (
        ("spancall_two_step", SPANCALL_PREDS[_sp]),
        (f"llm:{LLM_MODEL}",  LLM_PREDS[_sp]),
    ):
        _m = cmp.score(_samples, _preds, ids=_ids)
        if not _m:
            continue
        _tok_in  = sum(int(((p or {}).get("features") or {}).get("prompt_tokens", 0)) for p in _preds.values())
        _tok_out = sum(int(((p or {}).get("features") or {}).get("completion_tokens", 0)) for p in _preds.values())
        _tim = _lat_stats(_preds, ids=set(_ids))
        # `mean_latency_ms` is kept as the key the persisted artifact and the
        # paper table already read; the percentiles ride alongside it.
        _m["timing"] = dict(_tim, mean_latency_ms=_tim.get("mean", 0.0))
        _m["cost"] = {"n_llm_calls": (len(_preds) if _name.startswith("llm:") else 0),
                      "prompt_tokens": _tok_in, "completion_tokens": _tok_out}
        _sys[_name] = _m

        # Per (corpus x tool-count bucket) cells, same edges as the paper's pivot.
        _pivot = {}
        for _key, _sids in _slices.items():
            _sel_ids = set(_sids)
            _sel = [s for s in _samples if s.sample_id in _sel_ids]
            _cell = cmp.score(_sel, _preds, ids=_sids)
            if _cell:
                _pivot[_key] = _cell
        _sys[_name]["pivot"] = _pivot
        _sys[_name]["latency_by_slice"] = {
            _key: _lat_stats(_preds, ids=set(_sids)) for _key, _sids in _slices.items()
        }

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

    # Latency gets its own table: a mean hides the tail, and the tail is the
    # whole point of comparing a local decoder with a network round trip.
    _print_latency(_sys, _sp)
    _print_latency_pivot(_sys, _rows, _ids)
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
    _rows = _limit_rows(
        [json.loads(l) for l in open(SUBSET_PATHS[_sp], encoding="utf-8") if l.strip()]
    )
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

### Getting the artifacts off the VM

`DOWNLOAD_RESULTS=1` bundles `llm_parity.json` and the per-split
`preds_{split}_llm.jsonl` files into **one** zip and hands it to the browser.
One zip, not one download per file: Chrome blocks the second automatic
download, so a per-file loop would deliver only the first artifact and then look
like it had worked.

Two deliberate exclusions:

- `*.ckpt-*.jsonl` — checkpoint rows carry `_fingerprint` and any `llm_error`,
  so a half-failed run would read as a result once it left the VM. Cell 4 is the
  resume mechanism; the merged `preds_*.jsonl` is the result.
- `subset_*.jsonl` — large, and fully rebuildable by cell 1.

### Publishing next to the weights: `PUSH_RESULTS_TO_HUB`

`DOWNLOAD_RESULTS` gets the artifacts to one browser. `PUSH_RESULTS_TO_HUB=1`
puts them **where the model lives**, under `eval/llm_baseline/<run_tag>/` in
`HF_REPO_ID` — a folder a third party can cite and fetch without asking you.

Three properties are deliberate:

- **It never writes to the repo root.** `path_in_repo` keeps the eval tree
  disjoint from `model.pt`, `resolve_config.json` and the model card §7
  generates. A root-level push would overwrite the card, and would make
  `snapshot_download()` hand prediction rows to anyone who only wanted weights.
- **`<run_tag>` contains the fraction and the seed**, e.g.
  `eval/llm_baseline/qwen3.7-flash-f0.1-seed42/`. A smoke run lands in its own
  folder and can never overwrite the 100% run the paper cites. `llm_parity.json`
  carries `meta.run_tag` and `meta.hub_eval_prefix`, so a table found later
  points back at its own location.
- **The commit message names the model revision.** Every push records which
  weights the numbers describe.

### Pinning the weights (`HF_REVISION`)

A model repo's `HEAD` moves on every §7 re-push, so `HF_REPO_ID` names a repo,
not a model. The two are easy to conflate and expensive to get wrong: an eval
scored against one commit and cited against `main` is unreproducible.

- `HF_REVISION = "<commit sha or tag>"` makes the run reproducible. Do this for
  anything you intend to publish.
- Leaving it `None` follows `HEAD`. That is fine for a smoke run, and cell 7
  still records the sha it actually scored — read back from the snapshot cache
  path, so no extra network call — into `meta.spancall_revision`. The field is
  `(HEAD, unpinned)` only when inference never ran, i.e. when cell 3 reused the
  frozen predictions and no checkpoint was ever downloaded.

### Bringing the paid rows back

The upload in cell 3 covers only the **SpanCall** side. A fresh Colab runtime
has no LLM checkpoint, so re-running cell 4 at `LLM_LIMIT_FRACTION=1.0`
**re-buys every row**. If you already have a finished `preds_{split}_llm.jsonl`,
copy it into `RESULTS_DIR` before running cell 4: the fingerprint check then
matches and cell 4 prints `COMPLETE run reused - 0 API calls, $0 spent`.

If you published a previous run, you can also fetch it back by revision instead
of by hand:

```python
from huggingface_hub import snapshot_download
snapshot_download(HF_REPO_ID, repo_type="model", revision=REV,
                  allow_patterns="eval/llm_baseline/*/*.jsonl")
```
"""

CODE_PERSIST = '''# 7. Write JSON artifacts and print a paper-shaped table.
import json, os


def _eval_rows(path):
    """The rows that were actually scored, i.e. after the fraction trim."""
    return _limit_rows([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])


_out = os.path.join(RESULTS_DIR, "llm_parity.json")

# `<model>/<fraction>/<seed>`: the tag a run is published under. It carries the
# fraction on purpose - a 10% smoke run and the 100% run must not collide in
# `eval/llm_baseline/`, or the smoke numbers would overwrite the paper's.
RUN_TAG = "{}-{}-seed{}".format(
    LLM_MODEL.replace("/", "_"),
    ("full" if LLM_LIMIT_FRACTION >= 1.0 else "f{:g}".format(LLM_LIMIT_FRACTION)),
    SEED,
)
HUB_EVAL_PREFIX = "eval/llm_baseline/{}".format(RUN_TAG)

# `globals().get`, not a bare name: this cell is exec'd by the offline smoke test
# with a namespace that only holds the evaluation names, so a direct reference to
# a cell-1/cell-2 global would raise NameError there and only there. Both names
# are still resolved from the notebook at run time, which is what matters.
_rev_resolved = globals().get("MODEL_REVISION_RESOLVED")
_rev_pinned = globals().get("HF_REVISION")

payload = {
    "meta": {
        "spancall_repo": HF_REPO_ID,
        # WHICH weights these numbers describe. `spancall_repo` alone is a repo,
        # not a model: §7 re-pushes move HEAD, so without the sha this table
        # could silently belong to weights nobody can reproduce.
        "spancall_revision": _rev_resolved or _rev_pinned or "(HEAD, unpinned)",
        "spancall_revision_pinned": bool(_rev_pinned),
        "hub_eval_prefix": HUB_EVAL_PREFIX,
        "run_tag": RUN_TAG,
        "dataset": HF_DATASET_ID,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL or "(sdk default)",
        "llm_temperature": LLM_TEMPERATURE,
        "llm_limit_fraction": LLM_LIMIT_FRACTION,
        # The sampling limit that DEFINED the population. Recorded next to the
        # results because a reader comparing against the paper needs to know
        # which rows were eligible: `EVAL_LIMIT=None` would have produced the
        # same table shape on a 4.5x larger base, and nothing in the metrics
        # would say so.
        #
        # `globals().get` for the same reason as `_rev_resolved` below: this cell
        # is exec'd by the offline smoke test with a namespace that carries only
        # the evaluation names.
        "eval_limit": globals().get("EVAL_LIMIT"),
        "seed": SEED,
        "eval_splits": EVAL_SPLITS,
        "resolve_mode": "two_step",
        "min_joint_margin": MIN_JOINT_MARGIN,
        "n_rows": {sp: len(_eval_rows(SUBSET_PATHS[sp])) for sp in EVAL_SPLITS},
        "n_rows_sampled": {
            sp: (globals().get("SUBSET_INFO") or {}).get(sp, {}).get("sampled_total")
            for sp in EVAL_SPLITS
        },
        "spancall_source": SPANCALL_SOURCE,
        "llm_source": LLM_SOURCE,
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

if LLM_LIMIT_FRACTION < 1.0:
    print(f"\\n!! LLM_LIMIT_FRACTION={LLM_LIMIT_FRACTION}: SMOKE RUN on "
          f"{LLM_LIMIT_FRACTION:.0%} of each split "
          f"({payload['meta']['n_rows']}).")
    print("!! Do not copy these numbers into the paper; re-run with "
          "LLM_LIMIT_FRACTION=1.0.")

print("\\n" + "=" * 96)
print("PAPER TABLE (replaces the B1a row) — stratum F metrics")
print("=" * 96)
print(f"{'split':<14}{'system':<22}{'n_F':>6}{'exact_F':>10}{'hit_F':>9}"
      f"{'param_F1':>10}{'false_hit_A':>13}{'lat_ms':>9}{'p90_ms':>9}"
      f"{'tok_in':>9}{'tok_out':>9}")
print("-" * 121)
for _sp in EVAL_SPLITS:
    for _name, _m in SYSTEMS[_sp].items():
        _f = (_m.get("metrics_by_stratum") or {}).get("F", {})
        _t = _m.get("timing") or {}
        print(f"{_sp:<14}{_name:<22}"
              f"{int(_f.get('n', 0) or 0):>6}"
              f"{cmp._fmt(_f.get('exact_payload_match')):>10}"
              f"{cmp._fmt(_f.get('hit_rate')):>9}"
              f"{cmp._fmt(_f.get('param_f1')):>10}"
              f"{cmp._fmt(_m.get('false_hit_rate_on_a'), 4):>13}"
              f"{cmp._fmt(_t.get('mean'), 1):>9}"
              f"{cmp._fmt(_t.get('p90'), 1):>9}"
              f"{int((_m.get('cost') or {}).get('prompt_tokens', 0) or 0):>9}"
              f"{int((_m.get('cost') or {}).get('completion_tokens', 0) or 0):>9}")
print("-" * 121)
print("exact_F / hit_F / param_F1 are stratum F (`F_func` + `F_json`).")
print("false_hit_A is the share of ungroundable turns the system answered anyway (lower is better).")
print(f"lat_ms / p90_ms are per-call latency over {LLM_LIMIT_FRACTION:.0%} of each split.")
print("  LLM lat_ms = network round trip; SpanCall lat_ms = amortised per-sample")
print("  cost (own preprocess + batch_forward/rows + heads + decode). Same unit,")
print("  same population, so the two columns are directly comparable.")

# --------------------------------------------------------------------------- #
# What counts as a shipment artifact.
#
# One definition, used by BOTH transports below, because two copies of a
# whitelist is how a file ends up in the zip but not in the Hub folder (or the
# reverse) with nobody noticing.
#
# `*.ckpt-*` is excluded on purpose: those rows carry `_fingerprint` and any
# `llm_error`, so a half-failed run would read as a finished result once it left
# the VM. `*.bak-*` / `*.tmp*` are previous or partial writes for the same
# reason.
# --------------------------------------------------------------------------- #
def _ship_files(results_dir: str) -> list:
    keep = ["llm_parity.json"]
    for _f in sorted(os.listdir(results_dir)):
        if not (_f.startswith("preds_") and _f.endswith(".jsonl")):
            continue
        if ".ckpt-" in _f or ".bak-" in _f or ".tmp" in _f:
            continue                              # resume state, not a result
        keep.append(_f)
    return [_f for _f in keep if os.path.isfile(os.path.join(results_dir, _f))]


# --------------------------------------------------------------------------- #
# Optional Colab download: get the artifacts off the VM in ONE click.
#
# Two things this must NOT do:
#  * call `files.download()` once per artifact - Chrome blocks the 2nd+ automatic
#    download, so the user would receive only whatever came first;
#  * leave the zip behind - the next run would archive its own predecessor.
#
# `IN_COLAB` is deliberately RE-DETECTED here instead of inherited from cell 3.
# This cell is exec'd directly by the offline smoke test with a hand-built
# namespace, so relying on a name bound in another cell would be a NameError
# there and only there.
# --------------------------------------------------------------------------- #
DOWNLOAD_RESULTS = os.environ.get("DOWNLOAD_RESULTS", "0").strip().lower() not in (
    "", "0", "false", "no")
try:
    from google.colab import files as _colab_files
    _in_colab = True
except ImportError:
    _colab_files = None
    _in_colab = False

if not DOWNLOAD_RESULTS:
    print(f"\\n[download] off. Artifacts are on disk at {RESULTS_DIR}")
    print("[download] set DOWNLOAD_RESULTS=1 (Colab) to fetch them as one zip.")
elif not _in_colab:
    print(f"\\n[download] DOWNLOAD_RESULTS=1 but this is not Colab -> artifacts stay at "
          f"{RESULTS_DIR}")
else:
    import shutil as _shutil
    _bundle = os.path.join(RESULTS_DIR, "llm_baseline_artifacts")
    _shutil.rmtree(_bundle, ignore_errors=True)   # never archive a previous bundle
    os.makedirs(_bundle, exist_ok=True)

    _keep = _ship_files(RESULTS_DIR)
    for _f in _keep:
        _shutil.copy2(os.path.join(RESULTS_DIR, _f), os.path.join(_bundle, _f))
    print(f"\\n[download] bundling {len(_keep)} file(s): {_keep}")

    _zip = _shutil.make_archive(_bundle, "zip", root_dir=_bundle)
    print(f"[download] {_zip}  {os.path.getsize(_zip) / 1e6:.2f} MB")
    _shutil.rmtree(_bundle, ignore_errors=True)   # the zip is the artifact
    _colab_files.download(_zip)                   # exactly ONE browser download
    print(f"[download] handed to the browser. Local copy kept at {_zip}")


# --------------------------------------------------------------------------- #
# Optional Hub push: publish the artifacts NEXT TO the weights they describe.
#
# A model repo has a property the zip does not: it is citable, and the numbers
# are only meaningful for one specific revision of one specific model. So the
# results go to `eval/llm_baseline/<run_tag>/` inside HF_REPO_ID rather than to
# a repo root, for three reasons:
#
#  * `path_in_repo` keeps the eval tree disjoint from `model.pt`,
#    `resolve_config.json` and the generated model card. A root-level push would
#    overwrite the card §7 just wrote, and would make `snapshot_download()` pull
#    prediction files for anyone who only wanted weights.
#  * `<run_tag>` embeds the model name, the fraction and the seed. A
#    `LLM_LIMIT_FRACTION=0.1` smoke run therefore lands somewhere else and can
#    never overwrite the 100% run the paper cites.
#  * the commit message names the revision the table was scored against, so the
#    repo's own commit log records the evidence chain.
#
# OFF by default. A push is public and effectively permanent; making it implicit
# would let an exploratory run publish under the paper's dataset name.
#
# The token is READ FROM THE ENVIRONMENT, not from `HF_TOKEN` in the namespace:
# cell 1 already exported it there, and a bare name would be a NameError when
# this cell is exec'd with a hand-built namespace.
# --------------------------------------------------------------------------- #
_env = os.environ
PUSH_RESULTS_TO_HUB = _env.get("PUSH_RESULTS_TO_HUB", "0").strip().lower() not in (
    "", "0", "false", "no")

if not PUSH_RESULTS_TO_HUB:
    print("\\n[hub-push] off. Set PUSH_RESULTS_TO_HUB=1 to publish under "
          f"{HUB_EVAL_PREFIX}/.")
else:
    _tok = _env.get("HF_TOKEN") or _env.get("HUGGING_FACE_HUB_TOKEN")
    _ship = _ship_files(RESULTS_DIR)
    print(f"\\n[hub-push] target {HF_REPO_ID} :: {HUB_EVAL_PREFIX}/ "
          f"({len(_ship)} file(s): {_ship})")
    if not _tok:
        print("[hub-push] no HF token in the environment -> SKIPPED (files stay local).")
    elif "llm_parity.json" not in _ship:
        # Pushing predictions without the table would publish the evidence and
        # lose the conclusion; refuse rather than create a half-useful folder.
        print("[hub-push] llm_parity.json is missing -> SKIPPED. Run this cell fully "
              "first; a partial push is worse than none.")
    else:
        try:
            from huggingface_hub import HfApi
            _api = HfApi(token=_tok)
            _api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
            for _f in _ship:
                _api.upload_file(
                    path_or_fileobj=os.path.join(RESULTS_DIR, _f),
                    path_in_repo=f"{HUB_EVAL_PREFIX}/{_f}",
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                    commit_message=(
                        f"eval: LLM baseline {RUN_TAG} "
                        f"(model={_rev_resolved or _rev_pinned or 'HEAD'}, "
                        f"fraction={LLM_LIMIT_FRACTION}, seed={SEED})"
                    ),
                )
                print(f"[hub-push] uploaded {_f}")
            print(f"[hub-push] done -> https://huggingface.co/{HF_REPO_ID}"
                  f"/tree/main/{HUB_EVAL_PREFIX}")
            print("[hub-push] cite it with "
                  f"`revision={_rev_resolved or _rev_pinned or 'HEAD'}` — "
                  "this folder describes those weights, not the repo's future HEAD.")
        except Exception as _exc:
            # Deliberately non-fatal: the local artifacts are already written, so
            # a network or permission failure must not mark the run as failed.
            print(f"[hub-push] FAILED ({type(_exc).__name__}: {_exc})")
            print(f"[hub-push] local artifacts are intact at {RESULTS_DIR}")
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

### On the latency numbers

4. **Report p90, not just the mean — and say what each one measures.** The two
   figures are not the same quantity. SpanCall's `latency_ms` is an **amortised
   per-sample cost**: its own render/mask/tokenize work, plus its share of the
   batched backbone forward (`batch_forward_ms / rows_in_batch`), plus head
   scoring and decode. It deliberately does NOT include the queue wait behind
   the other rows in its batch — a per-row `perf_counter()` delta would, and
   that error grows linearly with the batch size, which is why an earlier
   version of this notebook reported latencies many times too high. The LLM's
   `latency_ms` is a network round trip to a hosted endpoint plus queueing.
   That asymmetry is the *finding*, not a flaw in the measurement, but it has to
   be stated or a reviewer will read "850 ms" as model quality. The honest
   second number is **throughput** (wall-clock rows/sec), printed alongside.
5. **Check `n_timed` before quoting any percentile.** If `n_timed` is 0 the arm
   made no timed call (see the error rows above), and if it is well below `n`,
   some rows were never timed. A p90 over 3 surviving rows is not a p90.
6. **Compare p90 against p50 on both arms.** A similar mean with a much worse
   p90 means the tail — retries, long candidate lists, provider throttling — is
   carrying the difference. That is a scheduling claim, not a modelling claim,
   and it belongs in the discussion rather than the results table.

Token counts remain the hard number for cost, since they do not depend on where
the notebook was run; wall clock does, and a Colab CPU path will flatter
SpanCall's p90. Re-measure on the serving hardware before quoting latency.
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