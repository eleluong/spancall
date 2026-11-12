#!/usr/bin/env python3
"""Generate notebooks/spancall_lean_train_eval_test.ipynb via nbformat (valid JSON guaranteed)."""
from pathlib import Path
import json

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parent / ".." / "notebooks" / "spancall_lean_train_eval_test.ipynb"

# --------------------------------------------------------------------------
# Cell sources
# --------------------------------------------------------------------------

MARKDOWN_TITLE = """# SpanCall Lean Flow — Train / Eval / Test Notebook (No Abstain Machinery)

This notebook runs the **standalone SpanCall Lean workflow** (`spancall/lean/`):
**git clone/pull → env + HF dataset pull → 0-indexed Lean tensorize → T2 training (`LeanSpanCallModel` + `LeanLossContainer`) → `LeanResolver` prediction → offline eval (Part 1 / Part 2 / Part 3) → OOD/adversarial test splits**.

### What Lean Flow Changes vs. Legacy SpanCall
- **Model (`LeanSpanCallModel`)**: Drops `AbstainHead` (H4), `CallAcceptHead`, `CorrectnessHead`, and `abstain_proj` column-0 on the pointer head. Keeps only `backbone` + `ToolSelectionHead` (H1) + `LeanPointerHead` (H3 over `0..C-1` real candidates + `<|null|>`).
- **Loss (`LeanLossContainer`)**: Pure $L = \\lambda_{\\text{ptr}} L_{\\text{ptr}} + \\lambda_{\\text{tool}} L_{\\text{tool}}$ with multi-positive softmax over `0..C-1`.
- **Decoder (`LeanJointDecoder`)**: Always selects the best valid non-overlapping span assignment (with R4 equal-value aggregation + greedy top-1 fallback so overlapping utterance candidates never wipe out parameters).
- **Resolver (`LeanResolver`)**: Direct argmax decoding + type casting (`P9`) + schema validation (`P10`) — no calibration or confidence/abstain gates needed.
"""

MARKDOWN_SETUP = """---
## 0. Setup — Pull Code from Git & Install

Clones the public SpanCall repository (`https://github.com/eleluong/spancall.git`) or pulls the latest updates from `origin/main` if already cloned (e.g. running in Google Colab), enters the repository directory, adds it to `sys.path`, and installs dependencies in editable mode."""

CODE_SETUP = '''# 0a. Pull latest code from public GitHub repository and install dependencies
import os, sys
from pathlib import Path

GIT_REPO_URL = "https://github.com/eleluong/spancall.git"
GIT_BRANCH   = "main"

# Locate or clone repository
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

# Change directory to project root and ensure it is on sys.path
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pull latest code from git
if (PROJECT_ROOT / ".git").is_dir():
    print(f"Pulling latest code from {GIT_BRANCH}...")
    !git pull origin {GIT_BRANCH} || echo "git pull note: continuing with local files"

print(f"SpanCall project root: {PROJECT_ROOT}")

# Install package in editable mode along with optional dependencies
%pip install -q -e ".[model,data,agent,dev]" huggingface_hub'''

MARKDOWN_CONFIG_INTRO = """### 0b. Configuration & Environment

Reads `.env` (or environment variables) for `DATASET_NAME` (`convoicon/spancall`) and `HUGGING_FACE_TOKEN`, auto-discovers `PROJECT_ROOT`, and configures isolated Lean output paths (`data/tensorized_lean` and `results_lean`) so Lean artifacts never overwrite legacy runs.
"""

CODE_CONFIG = '''import os, sys
from pathlib import Path

# Auto-discover project root (works whether cwd is repo root, notebooks/, or /content/spancall)
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "spancall" / "__init__.py").is_file():
    if (PROJECT_ROOT.parent / "spancall" / "__init__.py").is_file():
        PROJECT_ROOT = PROJECT_ROOT.parent
    elif (PROJECT_ROOT / "spancall" / "spancall" / "__init__.py").is_file():
        PROJECT_ROOT = PROJECT_ROOT / "spancall"
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spancall.agent.backends.litellm_adapter import _load_dotenv_values

ENV = {**_load_dotenv_values(PROJECT_ROOT / ".env"), **os.environ}
HF_DATASET_ID = ENV.get("DATASET_NAME", "convoicon/spancall")
HF_TOKEN      = ENV.get("HUGGING_FACE_TOKEN") or ENV.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# --- optional Colab Drive persistence ---
DRIVE_MOUNT   = False
DRIVE_OUT     = "/content/drive/MyDrive/spancall_lean_artifacts"

# --- data / checkpoints / eval (isolated from legacy directories) ---
DATA_DIR          = str(PROJECT_ROOT / "data" / "processed")
TENSOR_DIR        = str(PROJECT_ROOT / "data" / "tensorized_lean")
RESULTS_DIR       = str(PROJECT_ROOT / "results_lean")
LEGACY_RESULTS_DIR = str(PROJECT_ROOT / "results")
EVAL_SPLIT        = "hermes_eval"          # in-domain eval (TEST_ID)
TEST_SPLITS       = ["hermes_test_ood_schema", "hermes_test_ood_para", "hermes_test_adversarial"]

import torch
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# SINGLE SOURCE OF TRUTH - Lean Flow configuration
# ============================================================================

EXPERIMENT_NAME = "spancall_lean_colab"
SEED            = 42

# --- model / tensorization budget ---
BACKBONE       = "Qwen/Qwen3-Embedding-0.6B"
MAX_SEQ_LEN    = 4096
MAX_SPAN_WIDTH = 16
MAX_CANDS      = 512

import yaml
_cfg_yaml = {}
try:
    _cfg_yaml = yaml.safe_load(open("configs/default.yaml")) or {}
except Exception as _cfg_exc:  # noqa: BLE001 - reported, never silent
    print(f"[!] configs/default.yaml unreadable ({type(_cfg_exc).__name__}: "
          f"{_cfg_exc}); max_pos/max_tools/jitter fall back to code defaults.")
MAX_POS = int(((_cfg_yaml.get("optimization") or {}).get("max_pos", 3)) or 3)

# R0.1: ``pack_tools`` is a FLOOR (T = max(pack_tools, 1 + len(native tools))),
# never a cap -- a native tool must never be silently dropped from the prompt.
# ``MAX_TOOLS`` is the one knob allowed to reduce T, and it is an emergency
# brake for VRAM, so it is None (off) by default.
# R1c: ``JITTER_PACK_TOOLS`` draws T uniformly from [1, floor] per row, which is
# what puts the oracle-tool eval layout (T=1) back in the training distribution.
_data_cfg = _cfg_yaml.get("data") or {}
MAX_TOOLS = _data_cfg.get("max_tools")
MAX_TOOLS = None if MAX_TOOLS in (None, "", 0) else int(MAX_TOOLS)
JITTER_PACK_TOOLS = bool(_data_cfg.get("jitter_pack_tools", True))

# Lean Flow only needs multi-tool `t2` (train) and `dev` (held-out eval)
SPLIT_INPUTS = {
    "t2":  "hermes_train.jsonl",
    "dev": "hermes_dev.jsonl",
}

SAMPLE_LIMITS = {"t2": None, "dev": None}
DATA_FRACTION = 1.0
TENSORIZE_WORKERS = max(1, min(os.cpu_count() or 1, 16))

EVAL_LIMIT     = 500
TEST_LIMIT     = 200
FORCE_HF_DOWNLOAD = False

print("=" * 72)
print("LEAN FLOW DATA CONFIG")
print("=" * 72)
print(f"EXPERIMENT_NAME   = {EXPERIMENT_NAME}  SEED = {SEED}")
print(f"BACKBONE          = {BACKBONE}")
print(f"MAX_SEQ_LEN       = {MAX_SEQ_LEN}  MAX_SPAN_WIDTH = {MAX_SPAN_WIDTH}  MAX_POS = {MAX_POS}")
print(f"MAX_TOOLS         = {MAX_TOOLS}  (None = no ceiling; pack_tools is a floor, never a cap)")
print(f"JITTER_PACK_TOOLS = {JITTER_PACK_TOOLS}  (per-row T draw from [1, floor] so T=1 is in-distribution)")
for k, v in SPLIT_INPUTS.items():
    print(f"  {k:<9} <- {v}   (limit={SAMPLE_LIMITS.get(k)})")
print(f"DATA_FRACTION     = {DATA_FRACTION}")
print(f"TENSORIZE_WORKERS = {TENSORIZE_WORKERS}")
print(f"EVAL_LIMIT        = {EVAL_LIMIT}  TEST_LIMIT = {TEST_LIMIT}")
print(f"DATA_DIR          = {DATA_DIR}")
print(f"TENSOR_DIR        = {TENSOR_DIR}")
print(f"RESULTS_DIR       = {RESULTS_DIR}")
print(f"DEVICE            = {DEVICE}")
'''

MARKDOWN_ENV = """---
## 1. Runtime & Package Check

Verify GPU/device availability and confirm `spancall.lean` and `span_template` import cleanly."""

CODE_GPU = """!nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo 'No NVIDIA GPU detected (using MPS / CPU fallback)'"""

CODE_IMPORTS_CHECK = '''import spancall
import spancall.lean as lean
import span_template
import transformers
import datasets
import huggingface_hub
from spancall.lean.eval_p1 import (
    EXTRACTABLE_TARGET_TYPES,
    NON_EXTRACTABLE_TARGET_TYPES,
    e3_aligner_violation,
    eval_param_extraction_lean,
)

print("spancall     :", spancall.__file__)
print("spancall.lean:", lean.__file__)
print("span_template:", span_template.__file__)
print("transformers :", transformers.__version__, "| datasets:", datasets.__version__)
print("retrieval-first eval ready; extractable target types:", sorted(EXTRACTABLE_TARGET_TYPES))
print("out-of-scope target types:", sorted(NON_EXTRACTABLE_TARGET_TYPES),
      "| E3 tripwire:", callable(e3_aligner_violation))'''

MARKDOWN_DRIVE = """### 1b. (Optional) Google Drive mount (Colab only)

Mounted artifacts under `DRIVE_OUT` keep results between Colab sessions."""

CODE_DRIVE = '''if DRIVE_MOUNT:
    import shutil
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs(DRIVE_OUT, exist_ok=True)
    print('drive mounted:', DRIVE_OUT)'''

MARKDOWN_DATA_INTRO = """---
## 2. Dataset — Load via Hugging Face (`convoicon/spancall`)

Loads the 7 canonical SpanCall splits (`hermes_train`, `hermes_dev`, `hermes_calib`, `hermes_eval`, `hermes_test_ood_schema`, `hermes_test_ood_para`, `hermes_test_adversarial`) from `HF_DATASET_ID` (`convoicon/spancall`) on the Hugging Face Hub."""

CODE_DATA_HF = '''from huggingface_hub import snapshot_download
from datasets import load_dataset

CANONICAL_SPLITS = [
    "hermes_train.jsonl",
    "hermes_dev.jsonl",
    "hermes_calib.jsonl",
    "hermes_eval.jsonl",
    "hermes_test_ood_schema.jsonl",
    "hermes_test_ood_para.jsonl",
    "hermes_test_adversarial.jsonl",
]

os.makedirs(DATA_DIR, exist_ok=True)
missing = [f for f in CANONICAL_SPLITS if not os.path.exists(os.path.join(DATA_DIR, f))]

if FORCE_HF_DOWNLOAD or missing:
    print(f"Downloading splits from Hugging Face dataset: {HF_DATASET_ID} ...")
    snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        local_dir=str(PROJECT_ROOT),
        allow_patterns=["data/processed/*.jsonl", "data/processed/*.json"],
        token=HF_TOKEN,
    )

print("Canonical splits in DATA_DIR:")
for f in sorted(os.listdir(DATA_DIR)):
    p = os.path.join(DATA_DIR, f)
    if os.path.isfile(p):
        print(f"  {f:<34} ({os.path.getsize(p):>12,} bytes)")

hf_ds = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATA_DIR, "hermes_train.jsonl"),
        "dev":   os.path.join(DATA_DIR, "hermes_dev.jsonl"),
        "eval":  os.path.join(DATA_DIR, "hermes_eval.jsonl"),
    },
)
print("\\nHugging Face DatasetDict loaded:", hf_ds)'''

MARKDOWN_TENSOR_INTRO = """---
## 3. Lean Tensorizer — `UnifiedSample` → 0-Indexed Lean Trainer Rows

Uses [`spancall.lean.tensorize.tensorize_sample_lean`](../spancall/lean/tensorize.py) directly:
- **0-indexed `pos_indices` in `[0, C)`** (no column-0 abstain slot; `-1` is padding).
- Omitted optional/nullable parameters on Stratum-F rows point at a **deterministic** `<|null|>` candidate slot (resolved per param, never a best-effort region search).
- Rows without any positive groundable target (`has_positive_targets(row) == False`, e.g., empty Stratum-A rows) are filtered out cleanly so every training step has active pointer + tool supervision.

### Param supervision audit (retrieval-first contract)

A Stratum-F param with no matched positive receives **no pointer loss term** (the loss zeroes rows where `n_pos == 0`). That silent exclusion is only correct for some reasons, so the tensorizer classifies **every** drop into a 5-way taxonomy:

| drop reason | meaning | severity |
|---|---|---|
| `unresolved_target` | GT target type is `unresolved` — the value must be generated/transformed, not retrieved | **expected**: out of scope for the fast path |
| `absent_from_gt` | no aligned target and no GT arg — the param is not part of this call | **benign** |
| `match_missed` | an *extractable* target existed but the span matcher failed to find it | **bug**: a groundable param is silently unsupervised — must be driven to ~0 |
| `unaligned_target` | a target row exists but carries neither `canonical_value` nor `char_start` | **bug**: aligner produced an unusable target |
| `null_slot_missing` | GT is null / `null_ctx` but the mask builder emitted no `<|null|>` candidate | **bug**: cannot supervise a known-null answer |

Explicit `null` GT arguments are supervised: they map to the deterministic `<|null|>` slot. The cell below prints every reason per split, flags the two defect classes, and ends with three tripwires — `match_missed total`, `aligner/null-slot defects total`, and `tensorize exceptions total` — all of which must be `0`. Exceptions raised inside the fork worker are now **counted by class** instead of being silently swallowed into a smaller training set.
"""

CODE_TENSOR_SETUP = '''import json, os, random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import torch
from transformers import AutoTokenizer

from span_template import SPECIAL_MARKER_TOKENS
from spancall.data.schema import UnifiedSample
from spancall.model.masks import CandidateMaskBuilder
from spancall.lean.tensorize import ToolPool, tensorize_sample_lean, has_positive_targets

tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_MARKER_TOKENS)})
mask_builder = CandidateMaskBuilder(max_seq_len=MAX_SEQ_LEN, max_span_width=MAX_SPAN_WIDTH)

# ---- H1 candidate breadth -------------------------------------------------
# Hermes rows ship ONE tool each, so packing only s.tools renders a single
# <|tool|> anchor: tool_logits is (B, 1), argmax is always 0, tool_targets is
# always 0 => tool_acc = 1.0 for ANY weights and CE over one class = 0 (H1 gets
# no gradient). PACK_TOOLS > 1 packs real distractor schemas from the corpus so
# H1 becomes an actual discrimination task.
import yaml
# Guarded read: a missing/unparsable config must not kill the notebook, and a
# silent fallback of 1 would reintroduce the vacuous single-candidate H1.
PACK_TOOLS = 1
try:
    with open(os.path.join(str(PROJECT_ROOT), "configs", "default.yaml")) as _fh:
        _default_cfg = yaml.safe_load(_fh) or {}
    PACK_TOOLS = int((_default_cfg.get("data") or {}).get("pack_tools", 1) or 1)
except Exception as _exc:  # noqa: BLE001 - reported, never silent
    print(f"[!] configs/default.yaml unreadable ({type(_exc).__name__}: {_exc}); "
          f"defaulting PACK_TOOLS=1 -> H1 will be a single-class softmax (CE == 0).")
TOOL_POOL = None  # bound (once) in the tensorize cell, before forking
print("tokenizer ready; vocab =", len(tokenizer), "| MAX_SEQ_LEN =", MAX_SEQ_LEN,
      "| PACK_TOOLS =", PACK_TOOLS)'''

CODE_TENSOR_RUN = '''# ---- tensorize splits into Lean 0-indexed format (parallel fork pool) ----
# ``typing`` is imported HERE, in the cell that uses it. Notebook cells do not
# share scope, so the ``from typing import ...`` that appears in the resolver
# cell further down is invisible to this one: annotating ``cand_hist: Dict[...]``
# without this line raises NameError at runtime.
from typing import Any, Dict, List, Optional
import json
import math
import os
import queue
import random
import time

def _iter_raw_lines(path: str):
    """Yield the raw JSONL TEXT of each non-empty line.

    Deliberately does NOT ``json.loads`` here: parsing in the parent would make
    the whole corpus parse single-threaded before any fork happens (the parent
    wasted one full core on this on Colab's 2 vCPUs) and would hold every row
    dict in RAM at once. The fork workers parse their own line instead.
    """
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line

def _iter_raw_dicts(path: str):
    """Parsed variant, for consumers that need dicts in the parent."""
    for line in _iter_raw_lines(path):
        yield json.loads(line)

def _fraction_subset(raw_lines, frac: float, seed: int):
    """Deterministic fraction of the raw lines (still unparsed text)."""
    raw_lines = list(raw_lines)
    if frac >= 1.0:
        return raw_lines
    rng = random.Random(int(seed))
    n = max(int(len(raw_lines) * frac), 1)
    keep = set(rng.sample(range(len(raw_lines)), min(n, len(raw_lines))))
    return [d for i, d in enumerate(raw_lines) if i in keep]

def _serialize(row):
    """JSON-serializable view of a tensorized row.

    CONTRACT: the worker returns ``json.dumps(_serialize(row))`` and the parent
    writes it verbatim. ``spancall/lean/trainer.py:573 load_rows`` re-parses
    these files, so the bytes must stay identical to the old parent-side
    ``json.dumps(r)``. Do not add kwargs here.
    """
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}

def _tensorize_lean_task(args):
    """Tensorize one sample; returns (json_line_or_None, drop_reasons, errors).

    ``args`` is ``(raw_jsonl_text, pack_multi)``, NOT a parsed dict: the child
    does its own ``json.loads``, so the parent never pays a single-threaded
    corpus parse and never holds the whole split in RAM. Returning an already
    serialized line (rather than a dict) likewise moves ``json.dumps`` off the
    parent, which was the only serial tail left after the pool finished.

    The serialized form is byte-identical to ``json.dumps(_serialize(row))``
    because ``spancall/lean/trainer.py:573`` re-reads these files; see the
    contract note on ``_serialize`` below.

    The drop-reason audit travels back through the return value because the
    fork pool cannot mutate a parent dict. It is reported even for rows that
    are filtered out (all params dropped => no positives), which is exactly
    the case that used to vanish silently.

    Exceptions are NO LONGER swallowed. The exception class name is counted
    and returned so a tokenizer/schema mismatch shows up as a number instead
    of silently shrinking the training set.
    """
    raw_line, pack_multi = args
    drops = {}
    errors = {}
    try:
        d = json.loads(raw_line)
        s = UnifiedSample.from_dict(d)
        ts_stats = {}
        row = tensorize_sample_lean(
            s,
            tokenizer=tokenizer,
            mask_builder=mask_builder,
            max_seq_len=MAX_SEQ_LEN,
            max_cands=MAX_CANDS,
            max_pos=int(MAX_POS),
            pack_multi_tools=pack_multi,
            stats=ts_stats,
            # Read the pool from the MODULE GLOBAL, not the task tuple. Under the
            # "fork" context every child already inherits the parent's globals,
            # so passing the pool in the tuple only pickled the entire schema
            # corpus once per row (measured: 4.2 GB of redundant pickling on a
            # 50k-row split). Rebind ``TOOL_POOL``/``PACK_TOOLS_EPH`` before
            # calling _tensorize_many; the fallback path reads the same globals.
            tool_pool=TOOL_POOL,
            pack_tools=PACK_TOOLS_EPH,
            # R0.1: an explicit, loud ceiling that can only REDUCE T. ``None``
            # (the default) means "never drop a native tool"; setting it is an
            # emergency VRAM brake, and the truncation is counted below.
            max_tools=MAX_TOOLS,
            # R1c: draw T per row from [1, floor] so the oracle-tool eval layout
            # (T=1) is inside the training distribution instead of out of it.
            jitter=JITTER_PACK_TOOLS,
        )
        # L1 audit: classify Stratum-F params that ended up with NO positive.
        #   unresolved_target -> needs generation/transform; OUT OF SCOPE for the
        #                        retrieval-first fast path (correct exclusion).
        #   unaligned_target  -> target row carried neither value nor span (data bug).
        #   null_slot_missing -> answer is null but no <|null|> candidate exists (bug).
        #   absent_from_gt    -> param is not part of this call (benign).
        #   match_missed      -> groundable target existed but the matcher failed
        #                        => a supervision bug, silently excluded from loss.
        drops = dict(ts_stats.get("param_drop_reasons", {}))
        # A reason is only recorded when the anchor ended with NO positive. Params
        # that the fallback RESCUED are supervised, so they must not be counted as
        # drops -- but they are still evidence that the primary path failed, and
        # the child's stats dict cannot cross the fork boundary. Encode them on
        # the same synthetic-key channel as ``render_T=<T>`` below so the parent
        # can flag a high rescue rate instead of losing it.
        for _why, _cnt in (ts_stats.get("param_rescued_by_fallback") or {}).items():
            _k = f"rescued:{_why}"
            drops[_k] = drops.get(_k, 0) + int(_cnt)
        # Anchor-snap misses must be visible: a bridge_text tool line the
        # tokenizer could not map back to tokens means that H1 candidate was
        # emitted as PAD_INDEX (-1) and is invisible to the tool head.
        # This MUST be merged AFTER the rebind above: writing it into ``drops``
        # before ``drops = dict(ts_stats[...])`` discarded it unconditionally,
        # so the downstream tool_snap_miss warnings could never fire.
        if ts_stats.get("tool_snap_misses"):
            drops["tool_snap_miss"] = drops.get("tool_snap_miss", 0) + int(
                ts_stats["tool_snap_misses"]
            )
        # R9: the render-T histogram must survive the return trip. The fork pool
        # cannot mutate a parent dict, and ``drops`` is the only counter channel
        # that already crosses the process boundary, so encode the histogram as
        # synthetic ``render_T=<T>`` keys and decode it in the parent.
        for _t, _cnt in (ts_stats.get("render_T_hist") or {}).items():
            _k = f"render_T={int(_t)}"
            drops[_k] = drops.get(_k, 0) + int(_cnt)
        # R0.1: a truncated row is a model-visible capability loss, not a benign
        # data quirk -- count it on the same channel so it cannot vanish.
        if ts_stats.get("tool_truncated_rows"):
            drops["tool_truncated_rows"] = drops.get("tool_truncated_rows", 0) + int(
                ts_stats["tool_truncated_rows"]
            )
        # R1c: how many rows actually had their floor jittered.
        if ts_stats.get("jittered_rows"):
            drops["jittered_rows"] = drops.get("jittered_rows", 0) + int(
                ts_stats["jittered_rows"]
            )
        # T12a: the DRAW histogram, keyed on the floor each surviving row was
        # actually built with. ``jittered_rows`` above only proves a draw
        # HAPPENED; it cannot distinguish "drew uniformly over [min, pack]" from
        # "drew pack_tools every single row" (which is what a mis-wired
        # ``jitter_min_tools`` looks like, and it would leave the T=1 eval layout
        # just as absent as jitter=False). Encode on the synthetic-key channel,
        # same as ``render_T=<T>``, because a nested dict cannot cross the fork.
        for _floor, _cnt in (ts_stats.get("jitter_draw_hist") or {}).items():
            _k = f"jitter_draw={int(_floor)}"
            drops[_k] = drops.get(_k, 0) + int(_cnt)
        if row is None or not has_positive_targets(row):
            return None, drops, errors
        # Serialize inside the child: the parent then only concatenates text.
        return json.dumps(_serialize(row)), drops, errors
    except Exception as exc:  # noqa: BLE001 - counted, not swallowed
        cls = type(exc).__name__
        errors[cls] = errors.get(cls, 0) + 1
        return None, drops, errors

# In-flight task cap. Bounds both the reorder buffer and parent RAM; large
# enough that a straggler row cannot starve the pool, small enough that we do
# not pickle 50k lines into the queue at once.
_WINDOW = 2048

def _drain_ordered(pool, raw, pack_multi: bool, absorb, done, rows_len, need,
                   workers: int = 1, window: int = _WINDOW):
    """Submit in input order, apply in input order, without head-of-line stalls.

    ``ProcessPoolExecutor.map`` yields strictly in submission order, so ONE slow
    row parks every already-finished result behind it and idles the other
    workers (head-of-line blocking). Here results are queued as they complete
    and applied only when every earlier index has been applied -- identical
    output order, but no worker waits on a straggler.

    The completion queue is fed by futures' done callbacks, so the parent does
    no polling: one blocking ``get`` per row, O(1) each. (An earlier version
    called ``wait`` on the whole in-flight set each iteration, which allocated a
    window-sized set per row and burned real parent CPU.)

    Two separate bounds keep submission small:

    * ``window`` caps outstanding tasks so the reorder buffer stays bounded.
    * the *budget* (``need - rows``) caps them by how many rows we can still
      use. Without it a sampled split (``need=200`` of 50k lines) happily
      submitted and pickled the whole corpus before the early stop was ever
      consulted; a ``need=5`` run would still ship 500 tasks to the pool.
      ``guard`` exists because rows can be DROPPED: we may need to consume
      more tasks than ``need`` remaining to still collect ``need`` rows, and
      the cap self-corrects after every completion, so it never deadlocks.

    The budget is charged against ``submit - nxt`` -- tasks SUBMITTED but not
    yet APPLIED -- not against in-flight tasks alone. A finished result parked
    in the reorder buffer is still a row we may be about to collect, so counting
    only in-flight tasks let one extra task per applied row slip past the cap
    (``need=5`` shipped 22 tasks against a 21-task budget).
    """
    n = len(raw)
    guard = max(2 * int(workers), 16)
    finished_q = queue.SimpleQueue()   # (index, future) in COMPLETION order
    buffer = {}                        # index -> result, ready but not applicable yet
    nxt = 0                            # next index whose result may be applied
    submit = 0                         # next index to submit
    inflight = 0
    while nxt < n and not done():
        budget = max(0, need - rows_len())
        cap = min(window, budget + guard)
        while submit < n and (submit - nxt) < cap and not done():
            fut = pool.submit(_tensorize_lean_task, (raw[submit], pack_multi))
            # Bind the index NOW; a closure over `submit` would see its final value.
            fut.add_done_callback(lambda f, i=submit: finished_q.put((i, f)))
            submit += 1
            inflight += 1
        if inflight == 0:
            break
        idx, fut = finished_q.get()
        inflight -= 1
        buffer[idx] = fut.result()
        while nxt in buffer and not done():
            absorb(buffer.pop(nxt))
            nxt += 1

def _tensorize_many(raw, pack_multi: bool, need, workers: int, executor=None):
    """Tensorize raw JSONL lines; returns (rows, drops, errors, info).

    ``rows`` holds pre-serialized JSON strings in input order.
    ``executor`` lets the caller fork ONE pool for every split; without it a
    throwaway pool is created (and torn down) per split.
    """
    if need is None:
        need = len(raw)
    rows = []
    drops_total = Counter()
    errors_total = Counter()
    info = {"workers": 1, "pooled": False}

    def _absorb(res):
        line, drops, errors = res
        for k, v in (drops or {}).items():
            drops_total[k] += v
        for k, v in (errors or {}).items():
            errors_total[k] += v
        if line is not None and len(rows) < need:
            rows.append(line)

    def _done():
        return len(rows) >= need

    ctx = None
    if executor is None and workers > 1:
        try:
            ctx = mp.get_context("fork")
        except Exception as exc:  # noqa: BLE001
            # Silent degradation to 1 vCPU is the failure mode that makes this
            # cell look "just slow" on Colab. Say it out loud.
            print(f"  [!] fork context unavailable ({type(exc).__name__}: {exc}); "
                  f"tensorizing SINGLE-PROCESS with {workers} requested workers")
            ctx = None
    elif executor is None:
        print(f"  [!] TENSORIZE_WORKERS={workers} -> tensorizing SINGLE-PROCESS")

    if executor is not None or ctx is not None:
        pool = executor
        owned = pool is None
        try:
            if owned:
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
            info["workers"] = int(getattr(pool, "_max_workers", workers) or workers)
            info["pooled"] = True
            _drain_ordered(pool, raw, pack_multi, _absorb, _done,
                           rows_len=lambda: len(rows), need=need,
                           workers=info["workers"])
            return rows, drops_total, errors_total, info
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"  [!] process pool failed ({type(exc).__name__}: {exc}); "
                  f"falling back to single-process")
            info["workers"] = 1
            info["pooled"] = False
            rows = []
            drops_total = Counter()
            errors_total = Counter()
        finally:
            if owned and pool is not None:
                pool.shutdown(wait=True)
    for line in raw:
        _absorb(_tensorize_lean_task((line, pack_multi)))
        if _done():
            break
    return rows, drops_total, errors_total, info

groups = defaultdict(list)
for out_name, in_name in SPLIT_INPUTS.items():
    pack_multi = True
    groups[(in_name, pack_multi)].append(out_name)

os.makedirs(TENSOR_DIR, exist_ok=True)
stats = {}
drop_summary = {}
error_summary = {}

# ---- ONE distractor pool, built from TRAIN rows only ----------------------
# Two independent reasons this is a single pool hoisted out of the split loop:
#  1. LEAKAGE: building the pool per split seeded the dev/eval distractor set
#     with dev/eval schemas, i.e. held-out schemas influenced the very
#     candidate list the model is scored on. Train-only removes that channel
#     (a dev row's GT tool still comes from the row itself, never the pool).
#  2. The pool is a plain Python object. Binding it ONCE to the module global
#     below lets the fork pool inherit it by reference: it never enters the
#     task tuple, so it is never re-pickled per row.
POOL_SOURCE = SPLIT_INPUTS.get("t2") or next(iter(SPLIT_INPUTS.values()))
_pool_path = os.path.join(DATA_DIR, POOL_SOURCE)
if PACK_TOOLS > 1:
    if not os.path.exists(_pool_path):
        print(f"[!] tool-pool source missing ({_pool_path}) -> H1 distractors "
              f"disabled; tool_acc will be a single-class artifact.")
        TOOL_POOL = None
    else:
        TOOL_POOL = ToolPool.from_row_dicts(_iter_raw_dicts(_pool_path), seed=SEED)
        print(f"[tool-pool] built once from {POOL_SOURCE}: {len(TOOL_POOL)} "
              f"distinct schemas (pack_tools={PACK_TOOLS})")
        if len(TOOL_POOL) < PACK_TOOLS:
            print(f"[!VACUOUS-H1] pool holds {len(TOOL_POOL)} schemas but "
                  f"pack_tools={PACK_TOOLS}: no row can reach T={PACK_TOOLS}. "
                  f"Lower data.pack_tools or ingest more schemas.")
        if TOOL_POOL.n_duplicates:
            print(f"[tool-pool] {TOOL_POOL.n_duplicates} duplicate schema name(s) "
                  f"collapsed; {len(TOOL_POOL.name_collisions)} of them had a "
                  f"DIFFERING definition (first kept): "
                  f"{TOOL_POOL.name_collisions[:10]}")
else:
    TOOL_POOL = None
    print("[tool-pool] pack_tools=1 -> no distractors (H1 will be vacuous: "
          "CE == 0, tool_acc == 1.0 for any weights).")
PACK_TOOLS_EPH = int(PACK_TOOLS) if TOOL_POOL is not None else 1

# ---- ONE pool for ALL splits ---------------------------------------------
# Forking per split re-warmed the interpreter and re-inherited the tokenizer
# every time; on Colab most of a short split's wall-clock was that startup.
# One pool is forked once (inheriting TOOL_POOL/tokenizer by reference) and
# shared by every (in_name, pack_multi) group below.
_fork_ctx = None
_shared_ex = None
if TENSORIZE_WORKERS > 1:
    try:
        _fork_ctx = mp.get_context("fork")
    except Exception as _exc:  # noqa: BLE001 - surfaced, never silent
        print(f"[!] fork context unavailable ({type(_exc).__name__}: {_exc}) -> "
              f"tensorize will run SINGLE-PROCESS on one core; expect a large "
              f"slowdown versus Colab's {os.cpu_count()} vCPU(s).")
if _fork_ctx is not None:
    _shared_ex = ProcessPoolExecutor(max_workers=TENSORIZE_WORKERS, mp_context=_fork_ctx)
    print(f"[tensorize] one fork pool for all splits: {TENSORIZE_WORKERS} worker(s) "
          f"(cpu_count={os.cpu_count()})")

per_split_timing = {}
try:
    for (in_name, pack_multi), out_names in groups.items():
        src = os.path.join(DATA_DIR, in_name)
        if not os.path.exists(src):
            for on in out_names:
                print(f"SKIP {on}: {src} missing")
            continue
        t_read = time.perf_counter()
        raw = _fraction_subset(_iter_raw_lines(src), DATA_FRACTION, SEED)
        t_read = time.perf_counter() - t_read
        need = None
        for on in out_names:
            lim = SAMPLE_LIMITS.get(on)
            if lim is not None:
                need = lim if need is None else max(need, lim)
        t_pool = time.perf_counter()
        rows, drops, errors, tinfo = _tensorize_many(
            raw, pack_multi, need, TENSORIZE_WORKERS, executor=_shared_ex,
        )
        t_pool = time.perf_counter() - t_pool
        per_split_timing[in_name] = {
            "read_lines_s": round(t_read, 3),
            "tensorize_s": round(t_pool, 3),
            "workers": tinfo["workers"],
            "pooled": tinfo["pooled"],
            "rows_per_s": (round(len(rows) / t_pool, 1) if t_pool > 0 else None),
        }
        print(f"[tensorize] {in_name}: {len(rows)} rows in {t_pool:.1f}s "
              f"({len(rows) / t_pool:.1f} rows/s, {tinfo['workers']} worker(s), "
              f"read {t_read:.2f}s)")
        missed = int(drops.get("match_missed", 0))
        unresolved = int(drops.get("unresolved_target", 0))
        drop_summary[in_name] = dict(drops)
        if errors:
            error_summary[in_name] = dict(errors)
        for on in out_names:
            limit = SAMPLE_LIMITS.get(on)
            out_rows = rows if limit is None else rows[:limit]
            out_path = os.path.join(TENSOR_DIR, f"{on}.jsonl")
            t_write = time.perf_counter()
            with open(out_path, "w") as fh:
                # Rows arrive pre-serialized from the workers: no json.dumps
                # on the parent's single thread.
                fh.write("\\n".join(out_rows))
                if out_rows:
                    fh.write("\\n")
            per_split_timing[in_name]["write_s"] = round(
                time.perf_counter() - t_write, 3)
            # Stats need dicts; parse the kept rows only (not the whole split).
            stats_rows = [json.loads(x) for x in out_rows]
            # LIVE anchors only (>= 0). A row with zero live anchors carries NO
            # H1 supervision, so counting it as "1 candidate" would dilute the
            # mean toward the vacuity threshold and hide a real problem.
            # n_rows_h1 is the honest denominator: rows H1 can learn from.
            cand_counts = [
                int(r.get("_n_tool_candidates", 0)) for r in stats_rows
                if int(r.get("_n_tool_candidates", 0)) > 0
            ]
            cand_hist: Dict[int, int] = {}
            for c in cand_counts:
                cand_hist[c] = cand_hist.get(c, 0) + 1
            mean_cands = ((sum(cand_counts) / len(cand_counts))
                          if cand_counts else float("nan"))
            n_rows_h1 = len(cand_counts)
            n_rows_no_h1 = len(stats_rows) - n_rows_h1
            # R9: the RENDER count T, beside mean_n_candidates. These measure
            # different things and both matter: T is how many <|tool|> blocks the
            # prompt actually carries (the train/serve layout axis), while
            # n_candidates is how many of those anchors survived tokenization.
            # ``tool_positions`` is serialized as a plain list, so len() is the
            # exact T the trainer will see for this row.
            render_hist: Dict[int, int] = {}
            for r in stats_rows:
                tp = r.get("tool_positions")
                if not tp:
                    continue
                render_hist[len(tp)] = render_hist.get(len(tp), 0) + 1
            n_render_rows = sum(render_hist.values())
            mean_render_T = ((sum(t * c for t, c in render_hist.items())
                              / n_render_rows) if n_render_rows else float("nan"))
            t1_share = (render_hist.get(1, 0) / n_render_rows) if n_render_rows else float("nan")
            # T9: the ANALYTIC expectation for the T=1 share, not a hardcoded
            # complaint. ``jitter_pack_tools`` draws each row's floor uniformly
            # from ``[1, hi]``, so exactly one of those ``hi`` buckets is T=1 and
            # the share you should SEE is ~1/hi. The old check asserted only
            # "T=1 happened at all", which passed for a share 7x below the
            # expectation (446/9830 = 4.5% where 1/3 = 33% was due) — an
            # unrepresentative draw reads exactly like a healthy one.
            expected_t1 = ((1.0 / max(int(PACK_TOOLS_EPH), 1)) if JITTER_PACK_TOOLS else 0.0)
            # Binomial tolerance: 4 sd around the expected share is wide enough
            # to ignore sampling noise and tight enough to catch the observed
            # 4.5%-vs-33% class of defect.
            t1_sd = (math.sqrt(max(expected_t1 * (1.0 - expected_t1), 1e-12)
                               / max(n_render_rows, 1)) if n_render_rows else float("nan"))
            t1_off = bool(
                n_render_rows >= 30
                and t1_sd == t1_sd
                and abs(t1_share - expected_t1) > 4.0 * t1_sd
            )
            n_truncated = int(drops.get("tool_truncated_rows", 0))
            n_jittered = int(drops.get("jittered_rows", 0))
            # Phase I: capacity/layout channels. These are COUNTERS, not drop
            # reasons — lumping them into ``param_drop_other`` would print
            # "[note] dropped_over_length=412" as if a parameter had lost its
            # supervision, which is a different (and worse) failure.
            n_over_length = int(drops.get("dropped_over_length", 0))
            n_ambiguous = int(drops.get("multi_call_ambiguous_anchors", 0))
            n_recent_trunc = int(drops.get("recent_truncated_rows", 0))
            n_recent_rows = int(drops.get("n_recent_rows", 0))
            # T12a: decode the synthetic ``jitter_draw=<floor>`` channel back into
            # a real histogram. This is the POSITIVE control for the T=1 check
            # below: render_T_hist says what layout was produced, jitter_draw_hist
            # says what floor was DRAWN, so a gap between them localises the bug
            # to the draw (here) rather than to rendering (render_T there).
            draw_hist: Dict[int, int] = {}
            for _k, _v in drops.items():
                if _k.startswith("jitter_draw="):
                    try:
                        draw_hist[int(_k[len("jitter_draw="):])] = int(_v)
                    except (TypeError, ValueError):
                        pass
            n_draws = sum(draw_hist.values())
            _CHANNEL = (
                "tool_truncated_rows", "jittered_rows", "dropped_over_length",
                "multi_call_ambiguous_anchors", "recent_truncated_rows",
                "n_recent_rows", "tool_snap_misses", "param_snap_misses",
            )
            # T5: params the fallback rescued. NOT drops (they are supervised),
            # so they must not reach ``param_drop_other``, but they must be
            # reported: a high rescue rate means the primary matcher is failing.
            rescued = {k[len("rescued:"):]: int(v) for k, v in drops.items()
                       if k.startswith("rescued:")}
            param_other = {
                k: v for k, v in drops.items()
                if k not in ("match_missed", "unresolved_target")
                and k not in _CHANNEL
                and not k.startswith("render_T=")
                and not k.startswith("rescued:")
                and not k.startswith("jitter_draw=")
            }
            stats[on] = {
                "rows": len(stats_rows),
                "filtered_or_dropped": len(raw) - len(rows),
                "raw_after_fraction": len(raw),
                "fraction": DATA_FRACTION,
                "pack_multi": pack_multi,
                "limit": limit,
                "pack_tools": PACK_TOOLS_EPH,
                "tool_pool_size": len(TOOL_POOL) if TOOL_POOL is not None else 0,
                "rows_with_h1_supervision": n_rows_h1,
                "rows_without_h1_supervision": n_rows_no_h1,
                # NaN (not 0.0) when H1 has nothing to score: 0.0 reads as a
                # real measurement and silently passes any `< 1.5` check.
                "mean_n_candidates": (round(mean_cands, 3)
                                      if mean_cands == mean_cands else None),
                "n_candidates_hist": {str(k): v for k, v in sorted(cand_hist.items())},
                # R9: rendered T (one <|tool|> block per candidate tool).
                "mean_render_T": (round(mean_render_T, 3)
                                  if mean_render_T == mean_render_T else None),
                "render_T_hist": {str(k): v for k, v in sorted(render_hist.items())},
                "render_T_t1_share": (round(t1_share, 4)
                                      if t1_share == t1_share else None),
                # T9: what the layout SHOULD have produced, so a future reader
                # can tell "T=1 is a 4.5% accident" from "T=1 is 1/3 by design".
                "render_T_t1_expected_share": round(expected_t1, 4),
                "render_T_t1_share_off": t1_off,
                "jitter_pack_tools": bool(JITTER_PACK_TOOLS),
                # T12a: draws that did NOT jitter are still draws. If
                # ``jittered_rows`` is 0 while ``jitter_draw_hist`` only has the
                # ceiling bucket, the sampler degenerated to pack_tools -- which
                # is indistinguishable from jitter=False in the render_T report.
                "jittered_rows": n_jittered,
                "jitter_draw_hist": {str(k): v for k, v in sorted(draw_hist.items())},
                "jitter_draw_rows": n_draws,
                "jitter_draw_distinct_floors": len(draw_hist),
                "max_tools": MAX_TOOLS,
                "tool_truncated_rows": n_truncated,
                "param_drop_match_missed": missed,
                "param_drop_unresolved": unresolved,
                "param_drop_other": param_other,
                # T5: supervised anyway (fallback rescued them) -- reported so a
                # failing primary matcher cannot hide behind the fallback.
                "param_rescued_by_fallback": rescued,
                # Phase I: the numeric delta report reads these five channels
                # directly. Every one of them is a "the model never saw this"
                # counter, so none may hide inside param_drop_other.
                "dropped_over_length": n_over_length,
                "multi_call_ambiguous_anchors": n_ambiguous,
                "n_recent_rows": n_recent_rows,
                "recent_truncated_rows": n_recent_trunc,
                "tool_snap_misses": int(drops.get("tool_snap_misses", 0)),
                "param_snap_misses": int(drops.get("param_snap_misses", 0)),
                "tensorize_errors": dict(errors),
                "timing": per_split_timing.get(in_name, {}),
            }
            print(f"wrote {on}.jsonl: {len(stats_rows)} groundable rows "
                  f"({len(raw) - len(rows)} ungroundable/Stratum-A/over-length "
                  f"filtered)")
            if n_rows_h1 == 0:
                print(f"  [!VACUOUS-H1] {on}: NO row carries an H1 anchor at all "
                      f"(pack_tools={PACK_TOOLS_EPH}, pool="
                      f"{len(TOOL_POOL) if TOOL_POOL is not None else 0}). tool_acc "
                      f"is undefined here, not 1.0 — check the tensorize errors above.")
            elif mean_cands < 1.5:
                print(f"  [!VACUOUS-H1] {on}: mean candidates/row={mean_cands:.2f} "
                      f"over {n_rows_h1} H1 rows "
                      f"(hist={dict(sorted(cand_hist.items()))}) -> tool_acc is an "
                      f"artifact and H1 gets ~zero gradient. Check pack_tools/pool.")
            else:
                print(f"  [tool] {on}: mean candidates/row={mean_cands:.2f} over "
                      f"{n_rows_h1} H1 rows "
                      f"(hist={dict(sorted(cand_hist.items()))}) -> H1 is a real "
                      f"{int(round(mean_cands))}-way discrimination task.")
            if n_rows_no_h1:
                print(f"  [i] {on}: {n_rows_no_h1} row(s) have no live H1 anchor "
                      f"(no candidates packed, or the GT anchor failed to snap).")
            # R9: T is the layout axis, reported beside candidate breadth.
            if n_render_rows:
                print(f"  [render] {on}: mean render_T={mean_render_T:.2f} over "
                      f"{n_render_rows} row(s) "
                      f"(hist={dict(sorted(render_hist.items()))}, "
                      f"T=1 share={t1_share:.1%} vs expected "
                      f"{expected_t1:.1%}, jittered_rows={n_jittered})")
                # T9: compare the OBSERVED share against the analytic one. The
                # pre-T9 check only asked "did T=1 ever happen", so a draw 7x
                # below expectation passed silently.
                if JITTER_PACK_TOOLS and t1_share == 0.0:
                    print(f"  [!] {on}: T=1 never occurred although jitter is ON — "
                          f"every row would still be out-of-distribution for the "
                          f"oracle-tool eval layout. Check jitter_pack_tools wiring.")
                elif t1_off:
                    print(f"  [!] {on}: T=1 share={t1_share:.1%} but {expected_t1:.1%} "
                          f"was expected (pack_tools={PACK_TOOLS_EPH}, "
                          f"jitter={JITTER_PACK_TOOLS}, sd={t1_sd:.3f}). The draw is "
                          f"not representative of [1, {PACK_TOOLS_EPH}]: the oracle-tool "
                          f"eval layout (T=1) is UNDER-represented in training (R1c).")
                elif not JITTER_PACK_TOOLS and t1_share == 0.0:
                    print(f"  [i] {on}: T=1 never occurred and jitter is OFF "
                          f"(JITTER_PACK_TOOLS=False) -> the oracle-tool eval prompt "
                          f"(T=1) is OUT of the training distribution (R1c).")
                # T12a: the DRAW side of the same question. render_T_hist above is
                # the layout that was RENDERED; this is the floor that was DRAWN.
                # A jitter that degenerated to the ceiling shows up here and
                # nowhere else, because the rendered T still looks like a legal
                # pack_tools row.
                if n_draws:
                    print(f"  [jitter] {on}: {n_draws} row(s) drew a floor "
                          f"(hist={dict(sorted(draw_hist.items()))}, "
                          f"{len(draw_hist)} distinct floor(s) of "
                          f"{max(int(PACK_TOOLS_EPH), 1)} possible)")
                    if JITTER_PACK_TOOLS and len(draw_hist) == 1:
                        _only = next(iter(draw_hist))
                        print(f"  [!] {on}: jitter is ON but every draw landed on "
                              f"the SAME floor ({_only}). The sampler is "
                              f"degenerate -- jitter_min_tools/pack_tools wiring "
                              f"collapsed the interval. Training sees one layout "
                              f"only, so the T=1 eval prompt stays OOD (R1c).")
                    elif JITTER_PACK_TOOLS and 1 not in draw_hist:
                        print(f"  [!] {on}: no row drew floor=1, so the minimum "
                              f"packing (the oracle-tool eval layout) was never "
                              f"sampled. Check data.jitter_min_tools.")
                elif JITTER_PACK_TOOLS:
                    print(f"  [!] {on}: jitter is ON but NO row reported a draw "
                          f"(jitter_draw_hist empty). Either no row survived "
                          f"tensorization or the stats channel is not wired -- "
                          f"this is the silent-no-op signature.")
            if n_truncated:
                print(f"  [!] {on}: tool_truncated_rows={n_truncated} (max_tools="
                      f"{MAX_TOOLS}) — native tools were DROPPED from the prompt, so "
                      f"H1 can never pick them. Raise data.max_tools or unset it.")
        if missed:
            print(f"  [!] MATCH_MISSED={missed} on {in_name}: extractable targets the matcher "
                  f"failed on -> these params get NO pointer supervision (real bug).")
        n_rescued = sum(rescued.values())
        if n_rescued:
            print(f"  [note] rescued_by_fallback={n_rescued} on {in_name} "
                  f"{rescued} -> the param IS supervised (not a drop), but the "
                  f"primary target/matcher path missed it. A share near the "
                  f"groundable-param count means the GT-argument fallback is "
                  f"carrying the alignment, and `match_missed` should not be read "
                  f"as zero just because the drop counter is.")
        # ``render_T=<T>`` histogram keys and the capacity channel counters are
        # NOT drop reasons: they are already reported above, and printing them as
        # "[note] render_T=3=412" would bury the real audit lines.
        other = {k: v for k, v in drops.items()
                 if k not in ("match_missed", "unresolved_target")
                 and k not in _CHANNEL
                 and not k.startswith("render_T=")
                 and not k.startswith("rescued:")}
        for k in sorted(other):
            label = "BUG" if k in ("unaligned_target", "null_slot_missing") else "note"
            print(f"  [{label}] {k}={other[k]} on {in_name}")
        # Phase I: capacity channels printed explicitly. Each is a row/param the
        # model never got to learn from, which is the only reason this section
        # exists — silence here would read as "nothing was lost".
        if n_over_length:
            print(f"  [!] dropped_over_length={n_over_length} on {in_name}: rows "
                  f"longer than max_seq_len, dropped whole. If this is a large "
                  f"share, raise data.max_seq_len — those samples are unseen.")
        if n_ambiguous:
            print(f"  [i] multi_call_ambiguous_anchors={n_ambiguous} on {in_name}: "
                  f"anchors two calls share, so the pointer loss treats BOTH "
                  f"demands as valid (R0.4 honest-failure count, not a bug).")
        if n_recent_trunc:
            print(f"  [i] recent_truncated_rows={n_recent_trunc} on {in_name}: rows "
                  f"whose recent block exceeded the token budget and was trimmed.")
        if drops.get("tool_snap_misses"):
            print(f"  [!] tool_snap_miss={drops['tool_snap_misses']} on {in_name}: "
                  f"<|tool|> bridge lines the tokenizer could not map to tokens. Those "
                  f"candidates are PAD_INDEX(-1), i.e. invisible to H1.")
        if drops.get("param_snap_misses"):
            print(f"  [!] param_snap_miss={drops['param_snap_misses']} on {in_name}: "
                  f"param anchors that failed to snap; those parameters lose all "
                  f"pointer supervision for the row.")
        if unresolved:
            print(f"  [i] unresolved_target={unresolved} on {in_name}: needs "
                  f"generation/transform; out of scope for the retrieval-first fast "
                  f"path (correct exclusion).")
        for cls_name in sorted(errors):
            print(f"  [!!] tensorize raised {cls_name} x{errors[cls_name]} on {in_name} "
                  f"-> samples silently lost from training. Investigate.")
finally:
    if _shared_ex is not None:
        _shared_ex.shutdown(wait=True)
        print("[tensorize] shared fork pool closed")

print("\\nLean tensorize summary:", json.dumps(stats, indent=2))
print("\\nTensorize timing (per input split):", json.dumps(per_split_timing, indent=2))
print("\\nParam supervision audit (Stratum-F params dropped from pointer loss):",
      json.dumps(drop_summary, indent=2))
if error_summary:
    print("\\nTensorize exceptions (samples lost from training):",
          json.dumps(error_summary, indent=2))
missed_total = sum(d.get("match_missed", 0) for d in drop_summary.values())
print(f"match_missed total = {missed_total} (must be ~0; >0 means the span matcher "
      f"is losing groundable targets).")
bug_drops = sum(
    d.get("unaligned_target", 0) + d.get("null_slot_missing", 0)
    for d in drop_summary.values()
)
print(f"aligner/null-slot defects total = {bug_drops} (must be ~0; >0 means target "
      f"rows are unusable or the <|null|> candidate is missing).")
error_total = sum(v for e in error_summary.values() for v in e.values())
print(f"tensorize exceptions total = {error_total} (must be 0; >0 means samples "
      f"silently vanished from the training set).")

fingerprint = {
    "experiment_name": EXPERIMENT_NAME,
    "seed": SEED,
    "backbone": BACKBONE,
    "max_seq_len": MAX_SEQ_LEN,
    "max_span_width": MAX_SPAN_WIDTH,
    "max_pos": MAX_POS,
    "data_fraction": DATA_FRACTION,
    "split_inputs": SPLIT_INPUTS,
    "sample_limits": SAMPLE_LIMITS,
    "lean_flow": True,
}
fp_path = os.path.join(TENSOR_DIR, "config_fingerprint.json")
with open(fp_path, "w") as fh:
    json.dump(fingerprint, fh, indent=2, sort_keys=True)
print("wrote config fingerprint ->", fp_path)
'''

MARKDOWN_TENSOR_SANITY = """### Sanity: verify 0-indexed positive targets on `dev.jsonl`

Also re-runs the drop-reason audit over the *serialized* rows (`_param_drop_reasons`) to confirm the per-row classification survived serialization, and cross-checks that every param with zero positives has a recorded reason (no silent drops)."""

CODE_TENSOR_SANITY = '''import json, os
from collections import Counter
dev_path = os.path.join(TENSOR_DIR, "dev.jsonl")
if os.path.exists(dev_path):
    with open(dev_path) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    if not rows:
        print("dev.jsonl found but is empty (0 rows)")
    else:
        # In Lean format, positive candidate indices are 0-indexed (pi >= 0; -1 is padding)
        n_hit = sum(
            1 for r in rows
            if any(pi >= 0 for p_pos in r.get("pos_indices", []) for pi in p_pos)
        )
        print(f"dev.jsonl: {len(rows)} rows; rows with ≥1 positive target (pi >= 0): {n_hit} ({n_hit/len(rows):.1%})")
        r = rows[0]
        print("sample row keys:", sorted(r.keys()))
        c_len = len(r["cand_starts"][0]) if r.get("cand_starts") and len(r["cand_starts"]) > 0 else 0
        print("input_ids:", len(r.get("input_ids", [])), "seq_len; params K =", len(r.get("param_positions", [])),
              "cand (K,C) =", len(r.get("cand_starts", [])), "x", c_len)

        # ---- H1 trainability probe: T = number of <|tool|> candidate anchors
        # per row. T == 1 => softmax over a single class => CE == 0 (zero
        # gradient, H1 never trains) and argmax(-1) == 0 == gt => tool_acc is a
        # vacuous 1.0000 for ANY weights. T > 1 => H1 is a real task. This is
        # THE number that explains a "perfect" tool_acc, so it is printed on
        # its face instead of hidden behind the accuracy.
        def _tool_positions(r):
            """Unwrap tool_positions; the row may have been serialized as [[...]]."""
            tp = r.get("tool_positions") or []
            if tp and isinstance(tp[0], list):
                return tp[0]
            return tp

        tool_rows = [r for r in rows if "tool_positions" in r]
        t_hist = Counter()
        n_dead_slots = 0
        for r in tool_rows:
            # LIVE anchors only: a slot that failed to snap is PAD_INDEX (-1) and
            # is masked out of the logits, so it is NOT a choice H1 can make.
            # Counting raw list length would report T=3 for a row with one live
            # anchor plus two dead slots and claim H1 is a 3-way task.
            tp = _tool_positions(r)
            live = sum(1 for p in tp if int(p) >= 0)
            n_dead_slots += len(tp) - live
            t_hist[live] += 1
        n_tool_rows = len(tool_rows)
        n_h1_rows = sum(v for k, v in t_hist.items() if k > 0)
        mean_t = (sum(k * v for k, v in t_hist.items()) / n_h1_rows) if n_h1_rows else 0.0
        n_multi = sum(v for k, v in t_hist.items() if k > 1)
        print(f"H1 live candidates/row T: mean={mean_t:.2f} over {n_h1_rows} trainable rows; "
              f"hist={dict(sorted(t_hist.items()))}; dead (unsnapped) slots={n_dead_slots}")
        if n_tool_rows:
            print(f"  rows with T>1 (real discrimination): {n_multi}/{n_h1_rows} "
                  f"({(n_multi/max(n_h1_rows, 1)):.1%})")
        else:
            print("  no tool-supervised rows at all")
        if n_dead_slots:
            print(f"  [!] {n_dead_slots} <|tool|> anchor(s) failed to snap to tokens and "
                  f"are PAD_INDEX(-1) (invisible to H1). Non-zero means the bridge_text "
                  f"offset mapping and the tokenizer disagree.")
        if n_h1_rows and mean_t < 1.5:
            print("[!VACUOUS-H1] mean T < 1.5: tool_acc=1.0000 is an artifact — the H1 "
                  "softmax has a single class, CE is identically 0, and H1 receives NO "
                  "gradient. Set data.pack_tools > 1 in configs/default.yaml and rebuild "
                  "the tool pool (ToolPool.from_row_dicts).")
        if tool_rows:
            # tool_targets must be a valid index into [0, T) AND must point at a
            # LIVE anchor; -1 is the collate padding sentinel and must NOT appear
            # pre-collate. A target pointing at a dead slot is unbounded H1
            # supervision (the logit is -1e4, so CE explodes instead of training).
            bad_t = 0
            bad_examples = []
            gt_hist = Counter()
            for r in tool_rows:
                gt = r.get("tool_targets")
                if isinstance(gt, list):
                    gt = gt[0] if gt else -1
                tp = _tool_positions(r)
                gi = int(gt) if gt is not None else -1
                ok = 0 <= gi < len(tp) and int(tp[gi]) >= 0
                if not ok:
                    bad_t += 1
                    if len(bad_examples) < 5:
                        bad_examples.append((r.get("sample_id", "?"), gi, len(tp)))
                gt_hist[gi] += 1
            print(f"  tool_targets in-range AND live: {n_tool_rows - bad_t}/{n_tool_rows} "
                  f"(must equal rows; out-of-range/dead targets are silently wrong H1 "
                  f"supervision)")
            if bad_t:
                print(f"  [!] bad tool_targets examples (sample_id, gt, T): {bad_examples}")
            print(f"  gt tool index distribution (should be ~uniform if shuffled): "
                  f"{dict(sorted(gt_hist.items()))}")

        # ---- L1 audit on serialized rows: every supervised-clean param must be
        # explained. A param with all -1 positives and NO drop reason would be a
        # silent loss-hole. Reasons are split into EXPECTED (repairable only by
        # a slower path) vs BUG (must be ~0):
        #   expected : unresolved_target (needs generation), absent_from_gt
        #   bugs     : match_missed, unaligned_target, null_slot_missing
        reason_counts = Counter()
        unexplained = 0
        n_audited_rows = 0
        for r in rows:
            if "_param_drop_reasons" not in r:
                continue           # not a Stratum-F row: no audit applies
            n_audited_rows += 1
            reasons = r.get("_param_drop_reasons") or []
            reason_counts.update(reasons)
            n_zero_pos = sum(1 for p_pos in r.get("pos_indices", []) if not any(pi >= 0 for pi in p_pos))
            if n_zero_pos > len(reasons):
                unexplained += n_zero_pos - len(reasons)
        print(f"param drop reasons (dev, {n_audited_rows} audited Stratum-F rows):",
              dict(reason_counts) or "{}")
        print(f"params with zero positives and no recorded reason: {unexplained} "
              f"(must be 0 — otherwise loss silently ignores a groundable param)")
        EXPECTED_DROPS = ("unresolved_target", "absent_from_gt")
        BUG_DROPS = ("match_missed", "unaligned_target", "null_slot_missing")
        n_expected = sum(reason_counts.get(k, 0) for k in EXPECTED_DROPS)
        print(f"  expected exclusions (fast path out of scope): {n_expected} "
              f"{ {k: reason_counts[k] for k in EXPECTED_DROPS if reason_counts.get(k)} }")
        n_bugs = sum(reason_counts.get(k, 0) for k in BUG_DROPS)
        print(f"  supervision defects (must be 0): {n_bugs} "
              f"{ {k: reason_counts[k] for k in BUG_DROPS if reason_counts.get(k)} }")
        if reason_counts.get("match_missed"):
            print(f"[!] match_missed={reason_counts['match_missed']} on dev — matcher is losing "
                  f"extractable targets; check target_spans/token offsets.")
        if reason_counts.get("null_slot_missing"):
            print(f"[!] null_slot_missing={reason_counts['null_slot_missing']} on dev — the mask "
                  f"builder emitted no <|null|> candidate for a null GT param.")
        if reason_counts.get("unaligned_target"):
            print(f"[!] unaligned_target={reason_counts['unaligned_target']} on dev — target rows "
                  f"carry neither canonical_value nor char_start.")
else:
    print("dev.jsonl not found — run the tensorizer cell first")'''

MARKDOWN_TRAIN = """---
## 4. Training — Lean Flow (`scripts/train_lean.py`)

Runs [`scripts/train_lean.py`](../scripts/train_lean.py) on stage `T2` with `LeanSpanCallModel` and `LeanLossContainer` ($L = \\lambda_{\\text{ptr}} L_{\\text{ptr}} + \\lambda_{\\text{tool}} L_{\\text{tool}}$).

**Refined recipe (bs=128 effective).** The previous run used `bs=8` with `accum=4`, i.e. an *effective batch of 32* on ~3.9k rows → only ~122 optimizer steps/epoch. Dev `exact_call` peaked at epoch 2 of 5 and then decayed monotonically while train loss fell 17×. The knobs below target that:

- `effective_batch_size=128` is authoritative; `accum` is **derived** from `micro_bs`, so the two can never disagree.
- `min_lr_ratio=0.10` floors the cosine decay — decaying to ~1e-9 left the final epochs doing nothing but overfitting.
- `early_stopping_patience=1` + `restore_best_on_exit=True` stop the run at the dev minimum and put the **best** epoch's weights back, so the stage checkpoint is no longer the overfit final epoch.

**Loss scale (train ↔ eval alignment).** `mean_loss` (train) and `eval_loss` (dev) are now both **slot-weighted micro averages** — every supervised param slot / supervised row contributes once, regardless of how rows were bucketed. Previously each was a mean of per-batch means, so a 1-row batch produced by the token cap weighed as much as a full batch and the two series were on different scales. The legacy numbers remain available as `mean_loss_batchmean` / `eval_loss_batchmean`."""

CODE_COLAB_CFG = '''import yaml, os

TRAIN_FRACTION = 1.0

with open("configs/default.yaml") as fh:
    cfg = yaml.safe_load(fh)

cfg["experiment"]["name"] = EXPERIMENT_NAME
cfg["experiment"]["seed"] = SEED
cfg["experiment"]["output_dir"] = RESULTS_DIR
cfg["experiment"]["data_dir"] = TENSOR_DIR

NUM_EPOCHS = 5
cfg["optimization"]["num_epochs"] = NUM_EPOCHS
# MICRO batch. Rows per OPTIMIZER step is EFFECTIVE_BATCH_SIZE and the
# accumulation factor is derived from the two, so they cannot disagree.
# The old recipe (bs=8, accum=4 => effective 32) gave ~122 optimizer steps/epoch
# on ~3.9k rows; dev exact_call then peaked at epoch 2/5 and decayed while train
# loss fell 17x. A larger effective batch reduces that memorisation pressure.
MICRO_BATCH_SIZE = 32
EFFECTIVE_BATCH_SIZE = 128
cfg["optimization"]["batch_size"] = MICRO_BATCH_SIZE
cfg["optimization"]["effective_batch_size"] = EFFECTIVE_BATCH_SIZE
cfg["optimization"]["max_tokens_per_batch"] = 4096
cfg["optimization"]["optimizer"] = "adamw"
cfg["optimization"]["scheduler"] = "cosine_warmup"
# LR floor: cosine-to-zero left the last epochs at ~1e-9 LR, i.e. pure overfitting.
cfg["optimization"]["min_lr_ratio"] = 0.10
# Early stop on the first eval split and restore the best epoch's weights so the
# stage checkpoint is not the (empirically worst) final epoch.
cfg["optimization"]["early_stopping_patience"] = 1
cfg["optimization"]["restore_best_on_exit"] = True
cfg["optimization"]["mixed_precision"] = "bf16"
# Accumulation is DERIVED from effective_batch_size // micro_batch_size; this value
# is only a fallback for a config that lacks effective_batch_size.
cfg["training"]["gradient_accumulation_steps"] = max(
    int(round(EFFECTIVE_BATCH_SIZE / MICRO_BATCH_SIZE)), 1)
cfg["training"]["gradient_checkpointing"] = True
cfg["training"]["eval_splits"] = ["dev"]
cfg["training"]["eval_limit"] = 500
# IMPORTANT: ``freeze_encoder_stages`` belongs to the TRAINING section, not
# ``model`` — it is a field of ``TrainingConfig``. Writing it under ``model``
# made the setting a silent no-op: ``TrainingConfig.from_dict`` never looked
# there, so the notebook's knob did nothing at all.
cfg["training"]["freeze_encoder_stages"] = []
cfg["model"]["max_seq_len"] = MAX_SEQ_LEN

# fraction is a TRAINER-side subsample; pack_tools is a TENSORIZER-side knob, so
# it must be pinned to the value the tensorize cell ACTUALLY used or the recorded
# config would disagree with the data on disk. That effective value is
# PACK_TOOLS_EPH (1 when no pool could be built), not the raw config request.
_EFFECTIVE_PACK_TOOLS = int(globals().get("PACK_TOOLS_EPH", PACK_TOOLS) or 1)
cfg.setdefault("data", {})["fraction"] = TRAIN_FRACTION
cfg["data"]["pack_tools"] = _EFFECTIVE_PACK_TOOLS

out = "configs/colab_lean.yaml"
with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh)
print(f"wrote {out} (epochs={NUM_EPOCHS}, micro_bs={MICRO_BATCH_SIZE}, "
      f"effective_bs={EFFECTIVE_BATCH_SIZE}, accum={cfg['training']['gradient_accumulation_steps']}, "
      f"optimizer={cfg['optimization']['optimizer']}, lr_floor={cfg['optimization']['min_lr_ratio']}, "
      f"early_stop_patience={cfg['optimization']['early_stopping_patience']}, bf16=True, "
      f"data_fraction={TRAIN_FRACTION}, pack_tools={_EFFECTIVE_PACK_TOOLS})")
print(f"loss scale: train mean_loss and dev eval_loss are now BOTH slot-weighted "
      f"(micro averages), so the two series are directly comparable. The legacy "
      f"mean-of-batch-means is still logged as mean_loss_batchmean / "
      f"eval_loss_batchmean for continuity with older runs.")
if _EFFECTIVE_PACK_TOOLS < 2:
    print("[!VACUOUS-H1] effective pack_tools < 2: H1 is a single-class softmax "
          "(CE == 0). Any tool_acc printed as 1.000 is an artifact.")
CONFIG_PATH = out
'''

CODE_TRAIN_RUN = '''# Direct in-notebook Lean training execution
from scripts.train_lean import main as train_lean_main

train_lean_main([
    "--config", CONFIG_PATH,
    "--experiment_name", EXPERIMENT_NAME,
    "--output_dir", RESULTS_DIR,
    "--data_dir", TENSOR_DIR,
    "--stages", "T2",
    "--device", DEVICE,
    "--seed", str(SEED),
])'''

MARKDOWN_TRAIN_TELEMETRY = """### Training Telemetry (Epoch Curves & Lean Loss Components)

Replays `results_lean/{exp}_seed{seed}/train_log.jsonl`:
- Per-stage summary (rows, epochs, final loss, steps)
- Per-epoch held-out eval curve (`span_acc`, `exact_call`, plus the H1 columns
  `tool_acc` / `n_tool` / `cands/row`)
- Per-epoch mean train loss
- LR + loss-component trajectory (`pointer`, `tool`)

`exact_call` (formerly `exact_payload_match`) is **row-level**: a row counts only
if every supervised param AND the selected tool are correct. `span_acc` is
per-(row, param) and is therefore not bounded by `exact_call`. A `tool_acc` of
`1.000` with `cands/row = 1.0` is a **single-class softmax artifact** (CE == 0,
H1 receives no gradient), flagged as `VACUOUS H1` below."""

CODE_TRAIN_TELEMETRY = '''# ---- training telemetry: replay train_log.jsonl into readable tables ----
import json, os, math
from collections import defaultdict

def _num(rec, key):
    """Return a float, or None for missing / None / NaN.

    NaN is reported as an em-dash, never formatted as a real number: a NaN means
    "never measured" and must not read as 0.000 (or pass a `< 1.5` threshold).
    """
    v = rec.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None

def _load_train_log():
    p = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_seed{SEED}", "train_log.jsonl")
    if not os.path.exists(p):
        print(f"train_log.jsonl not found at {p}")
        return []
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]

train_log = _load_train_log()
if train_log:
    stages = defaultdict(list)
    for rec in train_log:
        stages[rec.get("stage", "?")].append(rec)

    print("=" * 78)
    print("LEAN TRAIN LOG — STAGE SUMMARIES")
    print("=" * 78)
    for st, recs in stages.items():
        starts = [r for r in recs if r.get("event") == "stage_start"]
        ends = [r for r in recs if r.get("event") == "stage_end"]
        if ends:
            e = ends[-1]
            s = starts[-1] if starts else {}
            print(f"  {st:<6} rows={s.get('rows','?'):<7} epochs={s.get('num_epochs','?'):<4} "
                  f"final_loss={e.get('loss', 0):.4f}  steps={e.get('steps','?')}")

    print("\\n" + "=" * 78)
    print("PER-EPOCH HELD-OUT EVAL CURVE (epoch_eval events)")
    print("=" * 78)
    evals = [r for r in train_log if r.get("event") == "epoch_eval"]
    if evals:
        # H1 columns (tool_acc / n_tool / cands) are printed alongside the
        # pointer metrics so a vacuous single-candidate 1.0000 is immediately
        # readable: tool_acc only means something when cands/row > 1. Missing
        # and NaN values render as "—" rather than as a plausible 0.000.
        print(f"  {'stage':<7}{'ep':>3}{'n':>5}{'span_acc':>10}{'exact_call':>12}"
              f"{'tool_acc':>10}{'n_tool':>8}{'cands/row':>11}")
        for r in evals:
            span = _num(r, "span_acc")
            exact = _num(r, "exact_call")
            tool_acc = _num(r, "tool_acc")
            n_tool = _num(r, "n_tool")
            cands = _num(r, "mean_n_candidates")
            sp = f"{span:>10.3f}" if span is not None else f"{'—':>10}"
            ex = f"{exact:>12.3f}" if exact is not None else f"{'—':>12}"
            ta = f"{tool_acc:>10.3f}" if tool_acc is not None else f"{'—':>10}"
            nt = f"{int(n_tool):>8}" if n_tool is not None else f"{'—':>8}"
            cd = f"{cands:>11.2f}" if cands is not None else f"{'—':>11}"
            if cands is None:
                vac = "  <-- H1 NOT EVALUATED"
            elif cands < 1.5:
                vac = "  <-- VACUOUS H1"
            else:
                vac = ""
            print(f"  {str(r.get('stage','?')):<7}{r.get('epoch','?'):>3}{r.get('n','?'):>5}"
                  f"{sp}{ex}{ta}{nt}{cd}{vac}")
        _cand = [_num(r, "mean_n_candidates") for r in evals]
        _cand = [c for c in _cand if c is not None]
        if _cand and max(_cand) < 1.5:
            print("\\n  [!] H1 WARNING: every eval epoch had < 1.5 candidates/row, so the")
            print("      tool_acc column above is a single-class softmax artifact (CE == 0,")
            print("      H1 gets no gradient). Set data.pack_tools > 1 and rebuild the pool.")
        elif not _cand:
            print("\\n  [!] H1 WARNING: no epoch recorded candidates/row, so tool_acc is not")
            print("      interpretable. Re-generate the notebook so epoch_eval emits it.")
    else:
        print("  no epoch_eval events")

    # ------------------------------------------------------------------
    # ALIGNED TRAIN vs EVAL LOSS
    # Both numbers are now SLOT-WEIGHTED micro averages (sum of per-slot
    # losses / number of supervised slots), so they live on ONE scale and
    # the `gap` column is meaningful. Previously `mean_loss` was a mean of
    # per-batch means over shuffled train batches while eval loss was a mean
    # over length-sorted batches, which made the two numbers incomparable.
    # The `legacy` columns reproduce the old batch-mean values.
    # ------------------------------------------------------------------
    print("\\n" + "=" * 78)
    print("ALIGNED TRAIN vs EVAL LOSS (slot-weighted micro averages)")
    print("=" * 78)
    ends = [r for r in train_log if r.get("event") == "epoch_end"]
    if ends:
        print(f"  {'stage':<8}{'ep':>4}{'train_loss':>12}{'dev_loss':>11}"
              f"{'gap':>10}{'ptr':>9}{'tool':>9}{'train_bm':>10}{'dev_bm':>9}")
        for r in ends:
            ep = r.get("epoch", "?")
            tr = _num(r, "mean_loss")
            tr_bm = _num(r, "mean_loss_batchmean")
            ptr = _num(r, "pointer_loss")
            tool = _num(r, "tool_loss")
            # The same-epoch, same-stage dev row carries the aligned eval loss.
            dev_row = next(
                (e for e in evals
                 if e.get("stage") == r.get("stage") and e.get("epoch") == ep),
                None,
            )
            dv = _num(dev_row, "eval_loss") if dev_row is not None else None
            dv_bm = _num(dev_row, "eval_loss_batchmean") if dev_row is not None else None
            gap = (dv - tr) if (dv is not None and tr is not None) else None
            tr_s = f"{tr:>12.4f}" if tr is not None else f"{'—':>12}"
            dv_s = f"{dv:>11.4f}" if dv is not None else f"{'—':>11}"
            gap_s = f"{gap:>+10.4f}" if gap is not None else f"{'—':>10}"
            ptr_s = f"{ptr:>9.4f}" if ptr is not None else f"{'—':>9}"
            tool_s = f"{tool:>9.4f}" if tool is not None else f"{'—':>9}"
            trbm_s = f"{tr_bm:>10.4f}" if tr_bm is not None else f"{'—':>10}"
            dvbm_s = f"{dv_bm:>9.4f}" if dv_bm is not None else f"{'—':>9}"
            print(f"  {str(r.get('stage','?')):<8}{str(ep):>4}"
                  f"{tr_s}{dv_s}{gap_s}{ptr_s}{tool_s}{trbm_s}{dvbm_s}")
        print("\\n  gap = dev_loss - train_loss on the SAME (slot-weighted) scale.")
        print("  A gap that GROWS after the first epochs is genuine overfitting;")
        print("  ptr/tool columns split the train loss into its two heads.")
        print("  train_bm/dev_bm are the LEGACY batch-mean values (mean of ratios);")
        print("  they are kept only for continuity and are composition-sensitive.")
        rounds = [(_num(r, "mean_loss"),
                   _num(next((e for e in evals
                              if e.get("stage") == r.get("stage")
                              and e.get("epoch") == r.get("epoch")), None),
                         "eval_loss")) for r in ends]
        rounds = [(t, d) for t, d in rounds if t is not None and d is not None]
        if len(rounds) >= 2:
            first_gap = rounds[0][1] - rounds[0][0]
            best_i = min(range(len(rounds)), key=lambda i: rounds[i][1])
            print(f"\\n  first-epoch gap={first_gap:+.4f}; "
                  f"dev_loss is lowest at epoch {ends[best_i].get('epoch')} "
                  f"({rounds[best_i][1]:.4f}).")
    else:
        print("  no epoch_end events")

    print("\\n" + "=" * 78)
    print("LEAN LOSS-COMPONENT TRAJECTORY (train events, downsampled to ~40 rows)")
    print("=" * 78)
    trs = [r for r in train_log if r.get("event") == "train"]
    if trs:
        step = max(1, len(trs) // 40)
        print(f"  {'stage':<8}{'step':>6}{'loss':>10}{'pointer':>10}{'tool':>10}{'lr':>12}")
        for i, r in enumerate(trs):
            if i % step == 0 or i == len(trs) - 1:
                print(f"  {str(r.get('stage','?')):<8}{r.get('stage_step','?'):>6}{r.get('loss', 0):>10.4f}"
                      f"{r.get('pointer', 0):>10.4f}{r.get('tool', 0):>10.4f}{r.get('lr', 0):>12.2e}")
    else:
        print("  no train events")
else:
    print("No training log found — run the training cell first.")
'''

MARKDOWN_LORA_ABLATION = """### (Optional) T10 — LoRA vs Full-Fine-Tune Ablation

Default **OFF**. Set `RUN_LORA_ABLATION = True` (and `!pip install peft` in cell 0a) to train
both arms back-to-back with the **same seed, fraction, epochs, and data**, and print a
comparison table:

| arm | backbone | adapter params |
|---|---|---|
| `full_ft` | fully trainable (current behaviour) | 0 |
| `lora` | frozen + low-rank adapters at `lora_lr` (~1e-3) | `r`, `alpha`, targets |

Motivation (investigation §10.2–10.4): at the 0.2 data fraction the two heads are effectively
the only thing being trained, so the encoder can never re-align its representation to the
pointer vocabulary. LoRA gives the backbone a cheap low-rank path to move. Prediction: near
identical `exact_call`, better epoch-tail `train_loss` for LoRA.

LoRA knobs are **`TrainingConfig` fields and therefore live under the `training:` YAML
section** — writing them under `model:` is silently ignored (the same class of bug as
`freeze_encoder_stages`). Enabling LoRA without `peft` **raises**, never silently
full-fine-tunes."""

CODE_LORA_ABLATION = '''# ---- T10: LoRA vs full-FT ablation (opt-in; default OFF) ----
import os, json, math, glob, yaml

RUN_LORA_ABLATION = False          # flip to True to actually run the ablation
LORA_ABLATION_EPOCHS = int(globals().get("NUM_EPOCHS", 5))  # identical for BOTH arms
LORA_R = 16
LORA_ALPHA = 32                       # alpha = 2r
LORA_DROPOUT = 0.05
LORA_LR = 1e-3                        # adapters get their OWN group, not lr_encoder
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _write_arm_config(base_cfg, use_lora, arm_name, arm_out):
    """Materialise one arm's YAML. LoRA knobs go under ``training``, never ``model``."""
    cfg = json.loads(json.dumps(base_cfg))          # yaml-safe deep copy
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = arm_name
    cfg["experiment"]["output_dir"] = arm_out
    cfg.setdefault("training", {})
    # T11 lesson: these are TrainingConfig fields. Under `model` they are a no-op.
    cfg["training"]["use_lora"] = bool(use_lora)
    cfg["training"]["lora_r"] = int(LORA_R)
    cfg["training"]["lora_alpha"] = int(LORA_ALPHA)
    cfg["training"]["lora_dropout"] = float(LORA_DROPOUT)
    cfg["training"]["lora_lr"] = float(LORA_LR)
    cfg["training"]["lora_target_modules"] = list(LORA_TARGET_MODULES)
    cfg.setdefault("optimization", {})
    cfg["optimization"]["num_epochs"] = int(LORA_ABLATION_EPOCHS)
    cfg.setdefault("data", {})
    cfg["data"]["fraction"] = TRAIN_FRACTION
    cfg["data"]["pack_tools"] = int(globals().get("PACK_TOOLS_EPH", PACK_TOOLS) or 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"lora_ablation_{arm_name}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    return path


def _arm_summary(arm_root, exp_name):
    """Read back one arm's tail-of-training dev metrics + adapter provenance."""
    out = {
        "final_train_loss": float("nan"),
        "dev_span_acc": float("nan"),
        "dev_exact_call": float("nan"),
        "dev_tool_acc": float("nan"),
        "lora_param_count": None,
    }
    log_p = os.path.join(arm_root, f"{exp_name}_seed{SEED}", "train_log.jsonl")
    recs = []
    if os.path.exists(log_p):
        with open(log_p) as fh:
            recs = [json.loads(l) for l in fh if l.strip()]
    evals = [r for r in recs if r.get("event") == "epoch_eval" and r.get("split") == "dev"]
    if evals:
        last = evals[-1]
        for src, dst in (("span_acc", "dev_span_acc"),
                         ("exact_call", "dev_exact_call"),
                         ("tool_acc", "dev_tool_acc")):
            try:
                out[dst] = float(last.get(src))
            except (TypeError, ValueError):
                out[dst] = float("nan")
    ends = [r for r in recs if r.get("event") == "epoch_end"]
    if ends:
        try:
            out["final_train_loss"] = float(ends[-1].get("mean_loss"))
        except (TypeError, ValueError):
            pass
    provs = sorted(glob.glob(os.path.join(arm_root, "**", "provenance.json"), recursive=True))
    if provs:
        with open(provs[-1]) as fh:
            out["lora_param_count"] = json.load(fh).get("lora_param_count")
    return out


arms = [("full_ft", False), ("lora", True)]

if RUN_LORA_ABLATION:
    try:
        import peft  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "RUN_LORA_ABLATION=True requires the `peft` package. Run "
            "`!pip install peft` in cell 0a, or leave the flag False. (T10: "
            "enabling LoRA must fail loudly, never silently full-fine-tune.)"
        )
    from scripts.train_lean import main as train_lean_main

    with open(CONFIG_PATH) as fh:
        base_cfg = yaml.safe_load(fh)

    results = {}
    for arm_name, use_lora in arms:
        arm_root = os.path.join(RESULTS_DIR, "lora_ablation", arm_name)
        cfg_path = _write_arm_config(base_cfg, use_lora, f"lora_ablation_{arm_name}", arm_root)
        print("=" * 78)
        print(f"[T10] arm={arm_name}  use_lora={use_lora}  out={arm_root}")
        print("=" * 78)
        train_lean_main([
            "--config", cfg_path,
            "--experiment_name", f"lora_ablation_{arm_name}",
            "--output_dir", arm_root,
            "--data_dir", TENSOR_DIR,
            "--stages", "T2",
            "--device", DEVICE,
            "--seed", str(SEED),
        ])
        results[arm_name] = _arm_summary(arm_root, f"lora_ablation_{arm_name}")

    print("=" * 78)
    print("T10 LORA ABLATION - dev comparison (identical seed / fraction / epochs)")
    print("=" * 78)
    hdr = (f"{'arm':<10}{'use_lora':>10}{'adapters':>12}{'train_loss':>12}"
           f"{'span_acc':>10}{'exact_call':>12}{'tool_acc':>10}")
    print(hdr)
    print("-" * len(hdr))

    def _fmt(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(v) else f"{v:.4f}"

    for arm_name, use_lora in arms:
        m = results[arm_name]
        ap = m.get("lora_param_count")
        print(f"{arm_name:<10}{str(use_lora):>10}{(ap if ap is not None else '-'):>12}"
              f"{_fmt(m['final_train_loss']):>12}{_fmt(m['dev_span_acc']):>10}"
              f"{_fmt(m['dev_exact_call']):>12}{_fmt(m['dev_tool_acc']):>10}")
    print()
    print("Read: the arms differ ONLY in whether the backbone got adapters. An")
    print("unchanged `exact_call` with a lower LoRA train-loss tail is the §10.2")
    print("prediction (bottleneck is the pointer vocabulary, not capacity). A dev")
    print("gain WITHOUT a loss-tail gain is noise -- check the seeded floor first.")
else:
    print("[T10] LoRA ablation skipped (RUN_LORA_ABLATION = False).")
    print("      Wiring + config knobs are live (training.use_lora / lora_r /")
    print("      lora_alpha / lora_dropout / lora_lr / lora_target_modules); only")
    print("      the GPU run is deferred. Enable the flag + `!pip install peft`.")
'''

MARKDOWN_PICK_CP = """### Pick the Best Lean Checkpoint

Selects `results_lean/{exp}_seed{seed}/best_checkpoint` (or the latest stage checkpoint)."""

CODE_PICK_CP = '''import glob, os, json

exp_root = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_seed{SEED}")

def _latest_stage_cp():
    best_cp = os.path.join(exp_root, "best_checkpoint")
    if os.path.exists(os.path.join(best_cp, "model.pt")):
        return best_cp, "best_curve"
    for stage in ["T2", "t2"]:
        cps = sorted(glob.glob(os.path.join(exp_root, stage, "checkpoints", "**", "model.pt"), recursive=True))
        if cps:
            return os.path.dirname(cps[-1]), stage
    return None, None

ALLOW_STALE_CHECKPOINT = False
fp_path = os.path.join(TENSOR_DIR, "config_fingerprint.json")
fingerprint = {
    "experiment_name": EXPERIMENT_NAME,
    "seed": SEED,
    "backbone": BACKBONE,
    "max_seq_len": MAX_SEQ_LEN,
    "max_span_width": MAX_SPAN_WIDTH,
    "max_pos": MAX_POS,
    "data_fraction": DATA_FRACTION,
    "split_inputs": SPLIT_INPUTS,
    "sample_limits": SAMPLE_LIMITS,
    "lean_flow": True,
}
config_ok = True
if os.path.exists(fp_path):
    with open(fp_path) as fh:
        saved = json.load(fh)
    config_ok = (saved == fingerprint)
    if not config_ok:
        print("WARNING: TENSOR_DIR fingerprint does NOT match current config.")
else:
    print("WARNING: no config_fingerprint.json in TENSOR_DIR (run the tensorize cell first).")
    config_ok = False

CHECKPOINT_DIR = None
cp_dir, cp_stage = _latest_stage_cp()
model_init = os.path.join(RESULTS_DIR, "model_init")

if cp_dir is not None and (config_ok or ALLOW_STALE_CHECKPOINT):
    CHECKPOINT_DIR = cp_dir
    print(f"picked stage-{cp_stage} checkpoint @ {CHECKPOINT_DIR}")
elif os.path.isdir(model_init) and config_ok:
    CHECKPOINT_DIR = model_init
    print("no staged checkpoint; using model_init @", CHECKPOINT_DIR)
else:
    all_m = sorted(glob.glob(os.path.join(RESULTS_DIR, "**", "model.pt"), recursive=True))
    if all_m and (config_ok or ALLOW_STALE_CHECKPOINT):
        CHECKPOINT_DIR = os.path.dirname(all_m[-1])
        print("fallback checkpoint @", CHECKPOINT_DIR)
    else:
        print("no usable checkpoint found")

print("CHECKPOINT_DIR =", CHECKPOINT_DIR)
if CHECKPOINT_DIR:
    print("  contains:", sorted(os.listdir(CHECKPOINT_DIR)))
'''

MARKDOWN_PREDICT_INTRO = """---
## 5. Inference & Evaluation (`LeanResolver`)

Loads `LeanSpanCallModel` + `LeanResolver` and evaluates across the same 3 decoupled parts as the legacy notebook:
1. **Eval Part 1 — Parameter Extraction Evaluation (Oracle / Given GT Tool)**: `resolver.try_resolve(gt_tool, ...)`
2. **Eval Part 2 — Function (Tool) Prediction Evaluation**: `resolver.predict_tool(candidates, ...)`
3. **Eval Part 3 — End-to-End Function Calling Evaluation**: `resolver.try_resolve_candidates(s.tools, ...)` vs `B1a` (and vs Legacy SpanCall if present).
"""

CODE_RESOLVER = '''import json, os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from span_template import SPECIAL_MARKER_TOKENS
from spancall.data.schema import UnifiedSample
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver
from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.train.config import TrainingConfig

cfg = TrainingConfig.from_yaml(CONFIG_PATH)
tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name, use_fast=True)
encoder = SpanCallTransformerEncoder(
    model_name_or_path=cfg.model.backbone_name,
    hidden_dropout_prob=cfg.model.dropout,
    tokenizer=tokenizer,
)
encoder.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))
model = LeanSpanCallModel(
    encoder=encoder,
    head_dim=cfg.model.head_dim,
    max_span_width=cfg.model.max_span_width,
    hidden_dim=encoder.hidden_dim,
)

# R0.4: the call budget MUST come from config. A resolver left on the default
# ``max_calls=1`` can never report more than one predicted call, so every
# multi-call metric computed from its predictions (``pred_n_calls_hist``,
# ``call_exact_set_match``) is pinned to the single-call share by construction
# — a structural cap that looks like model failure in the run log.
MAX_CALLS = int(getattr(cfg.data, "max_calls", 1))
CALL_MIN_PROB = float(getattr(cfg.data, "call_min_prob", 0.25))
CALL_MIN_PROB_RATIO = float(getattr(cfg.data, "call_min_prob_ratio", 0.5))

resolver = LeanResolver(
    model_path=CHECKPOINT_DIR,
    encoder=encoder,
    model=model,
    tokenizer=tokenizer,
    max_seq_len=cfg.model.max_seq_len,
    max_span_width=cfg.model.max_span_width,
    max_calls=MAX_CALLS,
    call_min_prob=CALL_MIN_PROB,
    call_min_prob_ratio=CALL_MIN_PROB_RATIO,
)
print(f"resolver call budget: max_calls={MAX_CALLS} "
      f"call_min_prob={CALL_MIN_PROB} call_min_prob_ratio={CALL_MIN_PROB_RATIO}")
if MAX_CALLS <= 1:
    print("  [!] max_calls=1: multi-call predictions are structurally impossible;")
    print("      call_exact_set_match / pred_n_calls_hist will be capped at the")
    print("      single-call share. Set data.max_calls in the YAML to measure them.")
if DEVICE in ("cuda", "mps"):
    model.to(DEVICE)
model.eval()
if encoder is not None and hasattr(encoder, "eval"):
    encoder.eval()
print("LeanResolver ready (checkpoint =", CHECKPOINT_DIR, ")")'''

MARKDOWN_EVAL_PART1 = """---
## 5a. Evaluation Part 1 — Lean Parameter Extraction (Oracle Tool Given, Retrieval-First)

Evaluates `resolver.try_resolve(gt_tool, ...)` directly on ground-truth tools. Because Lean Flow has no abstain head or gate vetoes, `res.payload_partial` and `res.parameters[p].value` directly reflect the decoded pointer predictions (with R4 equal-value aggregation and greedy top-1 fallback).

**Retrieval-first contract** (`docs/param_extraction_eval_coverage.md`): the pointer head can only *retrieve* a value that already exists in the context (utterance, recent context, time, enum, boolean, null). Params whose ground truth is `unresolved` (needs generation/transformation) are **out of scope** — refusing them is correct behaviour and routes the call to a slower path. So this cell reports:

- **extraction_recall / precision / F1** — headline quality, scored over *extractable* params only (the fast path's actual job). `n/a` means the denominator was empty (no predictions / no params), **not** a perfect score.
- **fast_path_param_coverage / fast_path_call_coverage** — how much of the workload the fast path can serve at all.
- **refused_unextractable vs missed_extractable** — splits the old `pointer_empty` bucket into *correct refusals* (routing signal) and *real fast-path failures* (the number to drive down).
- **recall_all** — legacy blended view over every GT arg, kept as a diagnostic only (always ≤ extractable recall by construction).
- **pointer_wrong** — predicted a value where GT expects a different value *or* `null`. Null-param false positives are counted here (they used to be invisible).
- **exact_payload_match_F** — full-call match; **resolver_hit_rate_F** is the resolver's own schema-validity flag and is *not* GT accuracy (kept as a separate, clearly named diagnostic).
- **E3 aligner drift** — Stratum-F samples whose GT args lack an extractable target. Non-zero is a **data** bug (aligner drift), not a model miss; it is printed with up to 10 example `sample_id:param` pairs.
- **truncation** — `evaluated / total_available` so a `limit`-capped run is never mistaken for a full-split number.

Explicit `null` GT arguments are part of the GT universe (the fast path should select `<|null|>`), so they count in coverage and every denominator.

Implementation lives in `spancall/lean/eval_p1.py` so it is unit-tested, not notebook-local."""

CODE_EVAL_PART1 = '''import math
from spancall.lean.eval_p1 import eval_param_extraction_lean

def _m(key):
    """Read a required metric; NaN renders as n/a (empty denominator, not 1.0)."""
    v = p1_metrics[key]              # KeyError is intentional: no silent fallback
    return "n/a" if isinstance(v, float) and math.isnan(v) else v

# Retrieval-first quality metrics live in spancall/lean/eval_p1.py (unit-tested).
# NOTE: raises FileNotFoundError if EVAL_SPLIT was never built — loud by design.
p1_metrics = eval_param_extraction_lean(
    EVAL_SPLIT,
    resolver=resolver,
    data_dir=DATA_DIR,
    limit=EVAL_LIMIT,
    max_param_gap_report=25,
)

tr = p1_metrics["truncation"]
print("\\n--- Fast-path value proposition ---")
print(f"split                  = {p1_metrics['split']}  "
      f"(scored {tr['evaluated']} / {tr['total_available']} available"
      f"{', TRUNCATED' if tr['truncated'] else ''})")
print(f"extraction_recall        = {_m('extraction_recall')}")
print(f"extraction_precision     = {_m('extraction_precision')}")
print(f"extraction_f1            = {_m('extraction_f1')}")
print(f"fast_path_param_coverage = {_m('fast_path_param_coverage')}")
print(f"fast_path_call_coverage  = {_m('fast_path_call_coverage')}")
print(f"correct refusals         = {_m('refused_unextractable')} "
      f"| real fast-path misses = {_m('missed_extractable')} "
      f"| pointer_wrong = {_m('pointer_wrong')}")
print(f"E3 aligner drift         = {_m('e3_aligner_drift')} sample(s) "
      f"(>0 is a DATA bug, not a model miss)")

# Phase I / J6: the call axis and the prompt layout, both read straight off the
# metrics dict so a multi-call or T=1 run is legible without re-deriving it.
print(f"\\n--- Call axis & layout (J6 / R0.4) ---")
print(f"call_exact_set_match     = {_m('call_exact_set_match')} "
      f"({_m('call_exact_set_match_n')} / {_m('n')} row(s), all calls exact)")
print(f"exact_payload_match_F    = {_m('exact_payload_match_F')}  <- CALL 0 ONLY; "
      f"compare against the set number above, not in place of it")
print(f"call_count_acc           = {_m('call_count_acc')} "
      f"({_m('call_count_correct_n')} / {_m('n')})")
print(f"n_calls (GT)             = {_m('n_calls_hist')}  "
      f"multi-call rows={_m('n_multi_call_samples')}")
print(f"n_calls (predicted)      = {_m('pred_n_calls_hist')}")
print(f"mean_render_T            = {_m('mean_render_T')} "
      f"hist={_m('render_T_hist')} T=1 share={_m('render_T_t1_share')}")
_rb = p1_metrics.get("recent_budget_effective") or {}
if _rb:
    print(f"recent_budget_effective  = {_rb}")
if _m("render_T_t1_share") == 1.0:
    print("  [!] every Part-1 prompt rendered T=1: the layouts here are the "
          "oracle-tool ones, so any multi-call claim must be read against "
          "Part 3 as well.")

# T1/T2: candidate parity is THE split between "the model picked wrong" and
# "the right answer was never offerable". Printing it only as a raw key in
# p1_metrics forced the reader to remember which of the two failure modes a
# given number implies, so state it explicitly.
print("\\n--- Candidate parity (T1 diagnosis / T2 fix) ---")
_pc = p1_metrics.get("candidate_parity_checked") or 0
_arm = "PROTECTED (fix under test)" if p1_metrics.get("parity_protection") else "DIAGNOSIS (protection off)"
print(f"arm                    = {_arm}")
if _pc:
    _cp = p1_metrics["candidate_parity"]
    print(f"candidate_parity       = {_m('candidate_parity')} "
          f"({p1_metrics['candidate_parity_hit']} / {_pc} extractable GT spans offered)")
    print(f"candidate_miss_rate    = {_m('candidate_miss_rate')} "
          f"| not offered = {p1_metrics['candidate_parity_undetermined']} undetermined")
    print(f"cands_per_param        = {_m('cands_per_param_hist')} "
          f"fill={_m('cand_pack_fill')} cap={_m('cand_cap')}")
    if not p1_metrics.get("parity_protection") and _cp == _cp and _cp < 0.99:
        print(f"  [!] {1 - _cp:.1%} of extractable GT spans were NEVER in the served "
              f"candidate set. That is a serve-side defect, not model quality: "
              f"no amount of training fixes it. Re-run Part 1 with "
              f"protect_parity_spans=True to measure the headroom the fix recovers.")
    elif p1_metrics.get("parity_protection") and _cp == _cp and _cp < 0.99:
        print(f"  [!] protection is ON but parity is still {_cp:.1%}: the GT span is "
              f"missing for a reason OTHER than max_cands truncation (region/type "
              f"filter, or a target whose char offsets do not survive tokenization).")
    else:
        print("  [i] parity is ~1.0: every extractable GT span was offerable, so "
              "extraction_recall now measures model quality rather than mask coverage.")
else:
    print("candidate_parity       = n/a (no instrumented GT span in this split)")'''

MARKDOWN_EVAL_PART2 = """---
## 5b. Evaluation Part 2 — Lean Function (Tool) Prediction (H1)

Evaluates `resolver.predict_tool(candidates, ...)` across ≥ 2 candidate schemas per sample."""

CODE_EVAL_PART2 = '''import json, os
from collections import Counter, defaultdict
from spancall.data.schema import UnifiedSample

def eval_tool_prediction_lean(split_name: str, limit: Optional[int] = 500,
                              distractor_pool: Optional[dict] = None):
    src = os.path.join(DATA_DIR, f"{split_name}.jsonl")
    if not os.path.exists(src):
        print(f"File not found: {src}")
        return

    rows = []
    if distractor_pool is None:
        distractor_pool = {}
        with open(src) as fh:
            for line in fh:
                if not line.strip():
                    continue
                s = UnifiedSample.from_dict(json.loads(line))
                rows.append(s)
                for t in s.tools or []:
                    distractor_pool.setdefault(t.name, t)
    else:
        with open(src) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows.append(UnifiedSample.from_dict(json.loads(line)))
    distractor_names = sorted(distractor_pool)

    total = 0
    multi_total = 0
    multi_correct = 0
    single_total = 0
    single_correct = 0
    augmented = 0
    probs = []
    margins = []
    empty_scores = 0
    # T3: attribute every empty-``tool_scores`` row to a concrete serve-side
    # cause. Without this bucket the old "75 empty rows" line was unfalsifiable:
    # it could be no anchors, anchor loss in tokenization, or a head/layout
    # mismatch, and those have completely different fixes.
    empty_reasons = Counter()
    # Phase I: T is the layout axis H1 actually discriminated over. A mean of
    # 1.0 makes ``tool_acc`` a single-class softmax artifact, so it must be
    # printed beside the accuracy, never left to be inferred.
    cand_counts = Counter()
    wrong_as = Counter()
    per_tool = defaultdict(lambda: {"n": 0, "correct": 0, "wrong_as": Counter()})

    for s in rows:
        if not s.ground_truth_calls or not s.tools:
            continue
        gt_name = s.ground_truth_calls[0].tool_name
        native_multi = len(s.tools) > 1

        candidates = list(s.tools)
        names = {t.name for t in candidates}
        grew = False
        if gt_name not in names and gt_name in distractor_pool:
            candidates.append(distractor_pool[gt_name])
            names.add(gt_name)
            grew = True
        if len(names) < 2:
            distractor = next((distractor_pool[n] for n in distractor_names if n != gt_name), None)
            if distractor is None:
                continue
            candidates.append(distractor)
            grew = True
        if grew:
            augmented += 1

        pred = resolver.predict_tool(candidates, s.utterance, s.recent_context, s.current_time_iso)
        pred_name = pred.get("tool_name")
        is_correct = bool(pred_name) and pred_name == gt_name
        total += 1
        probs.append(pred.get("tool_prob", 0.0))
        margins.append(pred.get("tool_margin", 0.0))
        if not pred.get("tool_scores"):
            empty_scores += 1
            empty_reasons[str(pred.get("tool_scores_reason") or "unknown")] += 1
        # Phase I: count how many candidate schemas H1 actually chose between.
        _scores = pred.get("tool_scores") or {}
        cand_counts[len(_scores)] += 1

        if native_multi:
            multi_total += 1
            multi_correct += int(is_correct)
        else:
            single_total += 1
            single_correct += int(is_correct)
        if pred_name and pred_name != gt_name:
            wrong_as[pred_name] += 1
        tt = per_tool[gt_name]
        tt["n"] += 1
        tt["correct"] += int(is_correct)
        if pred_name and pred_name != gt_name:
            tt["wrong_as"][pred_name] += 1

        if limit is not None and total >= limit:
            break

    acc_all = (single_correct + multi_correct) / max(1, total)
    acc_multi = multi_correct / max(1, multi_total)
    acc_single = single_correct / max(1, single_total)
    mean_p = sum(probs) / max(1, len(probs))
    mean_m = sum(margins) / max(1, len(margins))

    mean_cands = (sum(c * n for c, n in cand_counts.items()) / max(1, total))
    print(f"=== Lean Tool Prediction (H1) on {split_name} (N={total}) — >=2 candidates ===")
    print(f"Top-1 Tool Accuracy (Overall): {acc_all:.4f} ({(single_correct + multi_correct)}/{total})")
    print(f"  Multi-tool subset: {acc_multi:.4f} ({multi_correct}/{multi_total})")
    print(f"  Single+distractor subset: {acc_single:.4f} ({single_correct}/{single_total})")
    print(f"  mean_n_candidates: {mean_cands:.2f} "
          f"(hist={dict(sorted(cand_counts.items()))})")
    if total and mean_cands < 1.5:
        print("  [!VACUOUS-H1] mean_n_candidates < 1.5: H1 had ONE class per row, so "
              "the accuracy above is an artifact, not tool-selection skill.")
    print(f"Mean Winner Tool Probability: {mean_p:.4f} | Mean Margin: {mean_m:.4f}")
    print(f"Rows with empty tool_scores: {empty_scores}")
    if empty_scores:
        print(f"  empty tool_scores by cause: {dict(empty_reasons.most_common())}")
    print(f"Rows where candidates were augmented beyond s.tools: {augmented}")
    print(f"Most common wrong picks: {wrong_as.most_common(8)}")

    if per_tool:
        print("\\n--- Per-tool H1 accuracy ---")
        print(f"{'tool':<30}{'n':>6}{'acc':>8}{'wrong_as':<40}")
        for tname, st in sorted(per_tool.items(), key=lambda kv: (-kv[1]["n"], kv[1]["correct"])):
            acc_t = st["correct"] / max(1, st["n"])
            wrong_str = ", ".join(f"{w}x{v}" for w, v in st["wrong_as"].most_common(3))
            print(f"{tname:<30}{st['n']:>6}{acc_t:>8.3f}  {wrong_str}")
    print("==================================================================")

eval_tool_prediction_lean(EVAL_SPLIT, limit=EVAL_LIMIT)'''

CODE_PREDICT_SPLIT = '''def _param_detail(pr) -> Dict[str, Any]:
    return {
        "value": getattr(pr, "value", None),
        "confidence": float(getattr(pr, "confidence", 0.0) or 0.0),
        "margin": float(getattr(pr, "margin", 0.0) or 0.0),
        "source_type": getattr(pr, "source_type", None),
        "char_span": list(pr.char_span) if getattr(pr, "char_span", None) else None,
        "p_abstain": 0.0,
    }


def _features_detail(features) -> Dict[str, Any]:
    feats = dict(features or {})
    if "tool_scores" in feats and isinstance(feats["tool_scores"], dict):
        top = sorted(feats["tool_scores"].items(), key=lambda kv: -float(kv[1]))[:3]
        feats["tool_scores"] = {k: float(v) for k, v in top}
    return {k: v for k, v in feats.items()
            if k not in ("recent_selected", "tool_scores_all")}


def predict_split(split_name: str, out_path: str, limit: Optional[int] = None,
                  detailed: bool = True) -> None:
    # T8: ONE definition of "a predicted call", shared with Eval Part 1. Imported
    # here (not copy-pasted) so the two parts cannot drift apart again.
    from spancall.lean.eval_p1 import predicted_call_count as _n_predicted_calls
    src = os.path.join(DATA_DIR, f"{split_name}.jsonl")
    sub_src = os.path.join(RESULTS_DIR, f"subset_{split_name}.jsonl")
    if not os.path.exists(src):
        print(f"no source {src} — skipped")
        return
    preds: List[Dict[str, Any]] = []
    subset_samples: List[Dict[str, Any]] = []
    n = 0
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw_dict = json.loads(line)
            s = UnifiedSample.from_dict(raw_dict)
            res = resolver.try_resolve_candidates(s.tools, s.utterance, s.recent_context, s.current_time_iso)
            rec = {
                "sample_id": s.sample_id,
                "stratum": s.stratum.value,
                "gt_tool": (s.ground_truth_calls[0].tool_name if s.ground_truth_calls else None),
                "gt_args": (s.ground_truth_calls[0].arguments if s.ground_truth_calls else {}),
                # R0.4 / J6: keep the WHOLE gt call list so the evaluator can score
                # the predicted call SET instead of only the flattened first call.
                "gt_calls": [{"tool_name": c.tool_name, "arguments": dict(c.arguments or {})}
                             for c in (s.ground_truth_calls or [])],
                "n_gt_calls": len(s.ground_truth_calls or []),
                # R0.4: the decoded call list. ``tool_name``/``payload`` above stay
                # the primary (call-0) view, so single-call consumers are unchanged.
                # T8: ``n_pred_calls`` uses the SAME rule as Part 1, via the
                # shared helper — a call with no tool name asserts nothing, so it
                # is not a prediction. Counting raw ``len(res.calls)`` made Part 3
                # report abstained decodes as calls and disagree with Part 1 on
                # the very same resolver output.
                "calls": [c.as_dict() for c in (getattr(res, "calls", None) or [])],
                "n_pred_calls": _n_predicted_calls(res),
                "hit": bool(res.hit),
                "status": getattr(res, "status", "hit" if res.hit else "partial"),
                "payload": res.payload,
                "payload_partial": getattr(res, "payload_partial", {}),
                "missing_required": getattr(res, "missing_required", []),
                "missing_optional": getattr(res, "missing_optional", []),
                "tool_name": res.tool_name,
                "confidence": float(getattr(res, "confidence", res.accept_score) or 0.0),
                "accept_probability": float(res.accept_score or 0.0),
                "joint_margin": float(getattr(res, "joint_margin", 0.0) or 0.0),
                "abstain_reason": res.abstain_reason,
                "latency_ms": float(getattr(res, "latency_ms", 0.0) or 0.0),
            }
            if detailed:
                rec["parameters"] = {p: _param_detail(pr) for p, pr in (res.parameters or {}).items()}
                rec["features"] = _features_detail(getattr(res, "features", {}))
            preds.append(rec)
            subset_samples.append(raw_dict)
            n += 1
            if limit is not None and n >= limit:
                break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for p in preds:
            fh.write(json.dumps(p, default=str) + "\\n")
    with open(sub_src, "w") as fh:
        for s_dict in subset_samples:
            fh.write(json.dumps(s_dict, default=str) + "\\n")
    n_hit = sum(1 for p in preds if p["hit"])
    print(f"predicted {n} rows → {out_path} (hit={n_hit}) and saved subset → {sub_src}")

predict_split(EVAL_SPLIT, os.path.join(RESULTS_DIR, "preds_eval.jsonl"), limit=EVAL_LIMIT)'''

MARKDOWN_SAMPLES_INTRO = """---
## 5c. Sample-Prediction Forensics

Inspects per-sample predictions in `preds_eval.jsonl`: hits, wrong-tool picks, and partial/schema-invalid payloads."""

CODE_SAMPLES_FORENSICS = '''import json, os, textwrap
from collections import Counter

PREDS_PATH = os.path.join(RESULTS_DIR, "preds_eval.jsonl")
if not os.path.exists(PREDS_PATH):
    print("preds_eval.jsonl not found — run the predict_split cell first")
else:
    with open(PREDS_PATH) as fh:
        preds = [json.loads(l) for l in fh if l.strip()]
    subs = {}
    sub_path = os.path.join(RESULTS_DIR, f"subset_{EVAL_SPLIT}.jsonl")
    if os.path.exists(sub_path):
        from spancall.data.schema import UnifiedSample
        with open(sub_path) as fh:
            for l in fh:
                if l.strip():
                    d = json.loads(l)
                    subs[d.get("sample_id")] = UnifiedSample.from_dict(d)

    def _utt(sid):
        s = subs.get(sid)
        return textwrap.shorten(s.utterance, width=160, placeholder=" …") if s else "(n/a)"

    def _verdict(p):
        if p.get("tool_name") and p.get("gt_tool") and p["tool_name"] != p["gt_tool"]:
            return "WRONG_TOOL"
        if p.get("hit"):
            return "HIT"
        return "PARTIAL"

    verdicts = Counter(_verdict(p) for p in preds)
    print(f"Loaded {len(preds)} Lean prediction records from {PREDS_PATH}")
    print("Verdict distribution:")
    for v, c in verdicts.most_common():
        print(f"  {v:<12}{c:>5}  ({c / max(1, len(preds)):.1%})")

    hits = [p for p in preds if p.get("hit")]
    print("\\n" + "=" * 72)
    print(f"SAMPLE HITS ({len(hits)} total; showing up to 5)")
    print("=" * 72)
    for p in hits[:5]:
        print(f"\\n--- {p.get('sample_id')} [{p.get('tool_name')}] conf={p.get('confidence'):.3f} ---")
        print(f"  utt : {_utt(p.get('sample_id'))}")
        print(f"  gt  : {json.dumps(p.get('gt_args'), default=str)}")
        print(f"  pred: {json.dumps(p.get('payload'), default=str)}")

    wrong = [p for p in preds if _verdict(p) == "WRONG_TOOL"]
    if wrong:
        print("\\n" + "=" * 72)
        print(f"WRONG-TOOL MISSES ({len(wrong)} total; showing up to 5)")
        print("=" * 72)
        for p in wrong[:5]:
            print(f"\\n--- {p.get('sample_id')} → predicted {p.get('tool_name')} (gt={p.get('gt_tool')}) ---")
            print(f"  utt : {_utt(p.get('sample_id'))}")
            if "features" in p and "tool_scores" in (p.get("features") or {}):
                print(f"  H1 scores: {p['features']['tool_scores']}")

    part = [p for p in preds if _verdict(p) == "PARTIAL"]
    if part:
        print("\\n" + "=" * 72)
        print(f"PARTIAL / SCHEMA-INVALID MISSES ({len(part)} total; showing up to 5)")
        print("=" * 72)
        for p in part[:5]:
            print(f"\\n--- {p.get('sample_id')} [{p.get('tool_name')}] reason={p.get('abstain_reason')} ---")
            print(f"  utt : {_utt(p.get('sample_id'))}")
            print(f"  gt  : {json.dumps(p.get('gt_args'), default=str)}")
            print(f"  pred: {json.dumps(p.get('payload_partial'), default=str)}")'''

MARKDOWN_EVAL_INTRO = """---
## 6. General Function Calling Evaluation (Eval Part 3)

Runs [`scripts/eval_toolcall.py`](../scripts/eval_toolcall.py) on identical subsets (`subset_{EVAL_SPLIT}.jsonl`):
1. **B1a baseline** (`--baseline b1a`)
2. **SpanCall Lean** (`--predictions preds_eval.jsonl`)
3. *(Automatic if present)* **Legacy SpanCall** (`results/toolcall_metrics_{EVAL_SPLIT}_model.json`) for direct comparison."""

CODE_EVAL_B1A = '''import subprocess, sys
subprocess.run([
    sys.executable, "scripts/eval_toolcall.py",
    "--test_data", f"{RESULTS_DIR}/subset_{EVAL_SPLIT}.jsonl",
    "--baseline", "b1a",
    "--output", f"{RESULTS_DIR}/toolcall_metrics_{EVAL_SPLIT}_b1a.json",
], check=True)'''

CODE_EVAL_MODEL = '''import subprocess, sys
subprocess.run([
    sys.executable, "scripts/eval_toolcall.py",
    "--test_data", f"{RESULTS_DIR}/subset_{EVAL_SPLIT}.jsonl",
    "--predictions", f"{RESULTS_DIR}/preds_eval.jsonl",
    "--output", f"{RESULTS_DIR}/toolcall_metrics_{EVAL_SPLIT}_lean.json",
], check=True)'''

MARKDOWN_CMP = """### Headline Comparison Table (B1a vs Legacy vs Lean Flow)"""

CODE_CMP = '''import json, os

def _load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)

rows = []
for label, path in [
    ("B1a (regex)", os.path.join(RESULTS_DIR, f"toolcall_metrics_{EVAL_SPLIT}_b1a.json")),
    ("SpanCall (legacy)", os.path.join(LEGACY_RESULTS_DIR, f"toolcall_metrics_{EVAL_SPLIT}_model.json")),
    ("SpanCall (LEAN)", os.path.join(RESULTS_DIR, f"toolcall_metrics_{EVAL_SPLIT}_lean.json")),
]:
    m = _load_metrics(path)
    if m is None:
        continue
    m_f = m.get("metrics_by_stratum", {}).get("F", {})
    rows.append({
        "system": label,
        "n_F": int(m_f.get("n", 0)),
        "exact_F": round(m_f.get("exact_payload_match", 0.0), 3),
        # R0.4 / J6: set-level headline. ``exact_F`` above stays call-0-only, so
        # a two-call row can be "exact" while its SET is wrong (dropped/extra call).
        "set_exact_F": round(m_f.get("call_exact_set_match", 0.0), 3),
        "count_acc": round(m.get("call_count_acc", 0.0), 3),
        "hit_F": round(m_f.get("hit_rate", 0.0), 3),
        "param_F1_F": round(m_f.get("param_f1", 0.0), 3),
        "false_hit_on_A": round(m.get("false_hit_rate_on_a", 0.0), 4),
        "exact_all": round(m.get("exact_payload_match", 0.0), 3),
        "set_exact_all": round(m.get("call_exact_set_match", 0.0), 3),
        "hit_all": round(m.get("hit_rate", 0.0), 3),
        "n_calls": m.get("n_calls_hist", {}),
    })

if rows:
    headers = ["system", "n_F", "exact_F", "set_exact_F", "count_acc", "hit_F",
               "param_F1_F", "false_hit_on_A", "exact_all", "set_exact_all",
               "hit_all", "n_calls"]
    print(" | ".join(headers))
    print("-|" * len(headers))
    for r in rows:
        print(" | ".join(str(r[h]) for h in headers))
else:
    print("no metric files found yet — run the eval_toolcall cells above")'''

MARKDOWN_EVAL_DETAIL_INTRO = """### Detailed Eval Forensics (Error Taxonomy & Per-Tool E2E)"""

CODE_EVAL_DETAIL = '''import json, os
from collections import Counter, defaultdict

def _err_buckets(m):
    return {k: m.get(k, 0) for k in [
        "error_wrong_tool", "error_wrong_value", "error_entity_swap",
        "error_duplicate_span", "error_date_misparse", "error_extra_call", "error_missing_call"]}

m_path = os.path.join(RESULTS_DIR, f"toolcall_metrics_{EVAL_SPLIT}_lean.json")
if not os.path.exists(m_path):
    print("Lean metrics file not found — run the eval_toolcall cell first")
else:
    with open(m_path) as fh:
        m = json.load(fh)

    print("=" * 72)
    print(f"ERROR TAXONOMY — {EVAL_SPLIT} (SpanCall Lean)")
    print("=" * 72)
    errs = _err_buckets(m)
    tot_err = sum(errs.values())
    for k, v in sorted(errs.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"  {k:<22}{v:>5}  ({v / max(1, tot_err):.1%} of errors)")

    p_path = os.path.join(RESULTS_DIR, "preds_eval.jsonl")
    sub_path = os.path.join(RESULTS_DIR, f"subset_{EVAL_SPLIT}.jsonl")
    preds, gt_map = [], {}
    if os.path.exists(p_path):
        with open(p_path) as fh:
            preds = [json.loads(l) for l in fh if l.strip()]
    if os.path.exists(sub_path):
        from spancall.eval.toolcall.evaluator import payload_equal
        from spancall.data.schema import UnifiedSample
        with open(sub_path) as fh:
            for l in fh:
                if l.strip():
                    s = UnifiedSample.from_dict(json.loads(l))
                    if s.ground_truth_calls:
                        gt_map[s.sample_id] = s
        for p in preds:
            s = gt_map.get(p.get("sample_id"))
            if s is not None and s.ground_truth_calls:
                gt = s.ground_truth_calls[0]
                p["gt_tool"] = gt.tool_name
                p["payload_equal"] = payload_equal(gt.arguments or {}, p.get("payload"),
                                                   s.tools[0] if s.tools else None)

    print("\\n" + "=" * 72)
    print("PER-TOOL E2E (SpanCall Lean)")
    print("=" * 72)
    per_tool = defaultdict(lambda: {"n": 0, "hit": 0, "exact": 0})
    for p in preds:
        tname = p.get("gt_tool") or p.get("tool_name")
        per_tool[tname]["n"] += 1
        per_tool[tname]["hit"] += int(bool(p.get("hit")))
        per_tool[tname]["exact"] += int(bool(p.get("payload_equal")))
    print(f"{'tool':<30}{'n':>6}{'hit':>6}{'hit%':>7}{'exact':>7}{'exact%':>8}")
    for tname, st in sorted(per_tool.items(), key=lambda kv: -kv[1]["n"]):
        if st["n"] == 0:
            continue
        print(f"{tname:<30}{st['n']:>6}{st['hit']:>6}{st['hit'] / st['n']:>7.2f}"
              f"{st['exact']:>7}{st['exact'] / st['n']:>8.2f}")

    # R9: the render count T the resolver actually built per row, straight off the
    # persisted ``features`` block. Without this the eval table cannot be read:
    # a T=1 layout makes H1 a single-class softmax whose accuracy is 1.0 for ANY
    # weights, so a high ``hit`` rate there is not evidence of tool selection.
    print("\\n" + "=" * 72)
    print("PROMPT LAYOUT (render_T) & RECENT BUDGET — SpanCall Lean")
    print("=" * 72)
    rt_hist: Counter = Counter()
    rb_hist: Counter = Counter()
    n_rt = 0
    for p in preds:
        feats = p.get("features") or {}
        rt = feats.get("render_T")
        if rt is not None:
            try:
                rt_hist[int(rt)] += 1
                n_rt += 1
            except (TypeError, ValueError):
                pass
        rb = feats.get("recent_budget_effective")
        if rb is not None:
            try:
                rb_hist[int(rb)] += 1
            except (TypeError, ValueError):
                pass

    if not n_rt:
        print("  render_T not reported by this resolver build — layout UNKNOWN")
        print("  (re-run the predict cell with a resolver that sets features['render_T'])")
    else:
        mean_T = sum(t * c for t, c in rt_hist.items()) / n_rt
        t1_share = rt_hist.get(1, 0) / n_rt
        print(f"  mean render_T = {mean_T:.2f} over {n_rt} row(s)")
        print(f"  hist          = {dict(sorted(rt_hist.items()))}")
        print(f"  T=1 share     = {t1_share:.1%}")
        if t1_share == 1.0:
            print("  [!] EVERY eval prompt rendered exactly ONE tool block. H1's softmax")
            print("      is single-class there: tool_acc = 1.0 regardless of the weights,")
            print("      so the hit rates above are NOT evidence of tool-selection skill.")
        elif t1_share == 0.0:
            print("  [!] T=1 never occurred in eval although training jitters T down to 1.")
            print("      Eval is then stricter than training on the T axis (R1).")
    if rb_hist:
        print(f"  recent_budget_effective = {dict(sorted(rb_hist.items()))}")
    else:
        print("  recent_budget_effective not reported (no clamp telemetry on these rows)")

    # R0.4 / J6: the call-axis view. ``exact_payload_match`` scores ONLY call 0,
    # so on the multi-call rows it silently forgives a dropped or invented second
    # call. These lines make that visible instead of trusting the headline.
    print("\\n" + "=" * 72)
    print("CALL SET (multi-call supervision, R0.4) — SpanCall Lean")
    print("=" * 72)
    gt_hist: Counter = Counter()
    pred_hist: Counter = Counter()
    n_set_rows = 0
    n_set_exact = 0
    for p in preds:
        n_gt = p.get("n_gt_calls")
        n_pred = p.get("n_pred_calls")
        if n_gt is None:
            n_gt = len(p.get("gt_calls") or [])
        if n_pred is None:
            n_pred = len(p.get("calls") or [])
        gt_hist[int(n_gt)] += 1
        pred_hist[int(n_pred)] += 1
        n_set_rows += 1
        n_set_exact += int(int(n_gt) == int(n_pred))
    multi_share = (sum(c for k, c in gt_hist.items() if k > 1) / n_set_rows) if n_set_rows else 0.0
    print(f"  GT n_calls histogram  = {dict(sorted(gt_hist.items()))}")
    print(f"  pred n_calls histogram= {dict(sorted(pred_hist.items()))}")
    print(f"  multi-call share      = {multi_share:.1%} of {n_set_rows} row(s)")
    print(f"  call_count_acc (from these rows)  = {n_set_exact / max(1, n_set_rows):.3f}")
    # R0.4: the resolver's call budget is a STRUCTURAL ceiling, not model quality.
    # With max_calls=1 no row can predict a second call, so a lower SET headline
    # is the cap talking, not a dropped call.
    max_pred_calls = max(pred_hist) if pred_hist else 0
    if multi_share > 0 and max_pred_calls <= 1:
        print("  [!] STRUCTURAL CAP: every row predicted <=1 call while "
              f"{multi_share:.1%} of rows are multi-call.")
        print("      The resolver was built with max_calls=1, so the call axis was")
        print("      never decoded. This number measures configuration, not the model.")
        print("      Set data.max_calls > 1 in the YAML and rebuild the resolver.")
    set_exact_metric = m.get("call_exact_set_match")
    if set_exact_metric is not None:
        print(f"  call_exact_set_match (evaluator)  = {set_exact_metric:.3f}")
        print(f"  exact_payload_match (call-0 only) = {m.get('exact_payload_match', 0.0):.3f}")
        if multi_share > 0 and set_exact_metric < m.get("exact_payload_match", 0.0):
            print("  [!] the SET headline is BELOW the call-0 headline: at least one")
            print("      multi-call row matched call 0 but lost a call in the set.")
    else:
        print("  call_exact_set_match not present — re-run eval_toolcall.py (R0.4)")'''

MARKDOWN_TEST_INTRO = """---
## 7. Test — OOD / Adversarial Splits

Evaluates `LeanResolver` and `B1a` on the three unseen test splits (`hermes_test_ood_schema`, `hermes_test_ood_para`, `hermes_test_adversarial`)."""

CODE_TEST_PREDS = '''for split in TEST_SPLITS:
    predict_split(split, os.path.join(RESULTS_DIR, f"preds_{split}.jsonl"), limit=TEST_LIMIT)'''

CODE_TEST_EVAL = '''import subprocess, sys

def _run_eval(split: str, label: str, extra: list) -> None:
    out = f"{RESULTS_DIR}/toolcall_metrics_{split}_{label}.json"
    data_file = f"{RESULTS_DIR}/subset_{split}.jsonl"
    if not os.path.exists(data_file):
        data_file = f"{DATA_DIR}/{split}.jsonl"
    cmd = [
        sys.executable, "scripts/eval_toolcall.py",
        "--test_data", data_file,
        *extra,
        "--output", out,
    ]
    subprocess.run(cmd, check=False)
    print(f"eval {split} [{label}] -> {out}")

for split in TEST_SPLITS:
    _run_eval(split, "b1a", ["--baseline", "b1a"])
    _run_eval(split, "lean", ["--predictions", f"{RESULTS_DIR}/preds_{split}.jsonl"])

print("test eval done")'''

CODE_TEST_TABLE = '''import json, os

print(f"{'split':<26}{'system':<16}{'n_F':>6}{'exact_F':>9}{'hit_F':>8}{'param_F1':>9}{'exact_all':>10}")
for split in TEST_SPLITS + [EVAL_SPLIT]:
    candidates = [
        ("b1a", os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_b1a.json")),
        ("legacy (raw)", os.path.join(LEGACY_RESULTS_DIR, f"toolcall_metrics_{split}_model.json")),
        ("lean", os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_lean.json")),
    ]
    for label, path in candidates:
        m = _load_metrics(path)
        if not m:
            continue
        m_f = m.get("metrics_by_stratum", {}).get("F", {})
        n_f = int(m_f.get("n", 0))
        exact_f = m_f.get("exact_payload_match", 0.0)
        hit_f = m_f.get("hit_rate", 0.0)
        pf1_f = m_f.get("param_f1", 0.0)
        exact_all = m.get("exact_payload_match", 0.0)
        print(f"{split:<26}{label:<16}{n_f:>6}"
              f"{exact_f:>9.3f}{hit_f:>8.3f}"
              f"{pf1_f:>9.3f}{exact_all:>10.3f}")'''

MARKDOWN_PERSIST = """---
## 8. Persist to Drive (Optional)"""

CODE_PERSIST = '''if DRIVE_MOUNT:
    import shutil, os
    os.makedirs(DRIVE_OUT, exist_ok=True)
    for name, src in [
        ("data_tensorized_lean", TENSOR_DIR),
        ("results_lean", RESULTS_DIR),
    ]:
        if os.path.isdir(src):
            dst = os.path.join(DRIVE_OUT, name)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"mirrored {src} → {dst}")
    print("\\nLean artifacts persisted to", DRIVE_OUT)'''

MARKDOWN_TESTS = """---
## 9. Test Suite (Lean Flow + Repo Unit Tests)

Runs `tests/test_lean_flow.py` and the full unit test suite to confirm everything passes."""

CODE_TESTS = """!python -m pytest tests/test_lean_flow.py tests -q --disable-warnings --no-header 2>&1 | tail -25"""

CODE_SUMMARY = '''print(f"""
== SpanCall Lean Flow pipeline complete ==

What ran:
  0. git clone/pull + editable pip install
  1. env setup (.env loaded) + spancall.lean import check
  2. HF dataset ({HF_DATASET_ID}) → {DATA_DIR}
  3. 0-indexed Lean tensorizer → {TENSOR_DIR}
  4. Lean T2 training (scripts/train_lean.py) → {RESULTS_DIR}
  5. LeanResolver eval Part 1 (retrieval-first param extraction) & Part 2 (H1 Tool Selection)
  6. Eval Part 3 (End-to-End Function Calling: B1a vs Legacy vs Lean) on {EVAL_SPLIT}
  7. OOD / Adversarial test splits: {TEST_SPLITS}
  8. Optional Drive mirror @ {DRIVE_OUT}
  9. Lean Flow + repo unit tests
""")'''

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

cells = [
    new_markdown_cell(MARKDOWN_TITLE),
    new_markdown_cell(MARKDOWN_SETUP),
    new_code_cell(CODE_SETUP),
    new_markdown_cell(MARKDOWN_CONFIG_INTRO),
    new_code_cell(CODE_CONFIG),
    new_markdown_cell(MARKDOWN_ENV),
    new_code_cell(CODE_GPU),
    new_code_cell(CODE_IMPORTS_CHECK),
    new_markdown_cell(MARKDOWN_DRIVE),
    new_code_cell(CODE_DRIVE),
    new_markdown_cell(MARKDOWN_DATA_INTRO),
    new_code_cell(CODE_DATA_HF),
    new_markdown_cell(MARKDOWN_TENSOR_INTRO),
    new_code_cell(CODE_TENSOR_SETUP),
    new_code_cell(CODE_TENSOR_RUN),
    new_markdown_cell(MARKDOWN_TENSOR_SANITY),
    new_code_cell(CODE_TENSOR_SANITY),
    new_markdown_cell(MARKDOWN_TRAIN),
    new_code_cell(CODE_COLAB_CFG),
    new_code_cell(CODE_TRAIN_RUN),
    new_markdown_cell(MARKDOWN_TRAIN_TELEMETRY),
    new_code_cell(CODE_TRAIN_TELEMETRY),
    new_markdown_cell(MARKDOWN_LORA_ABLATION),
    new_code_cell(CODE_LORA_ABLATION),
    new_markdown_cell(MARKDOWN_PICK_CP),
    new_code_cell(CODE_PICK_CP),
    new_markdown_cell(MARKDOWN_PREDICT_INTRO),
    new_code_cell(CODE_RESOLVER),
    new_markdown_cell(MARKDOWN_EVAL_PART1),
    new_code_cell(CODE_EVAL_PART1),
    new_markdown_cell(MARKDOWN_EVAL_PART2),
    new_code_cell(CODE_EVAL_PART2),
    new_code_cell(CODE_PREDICT_SPLIT),
    new_markdown_cell(MARKDOWN_SAMPLES_INTRO),
    new_code_cell(CODE_SAMPLES_FORENSICS),
    new_markdown_cell(MARKDOWN_EVAL_INTRO),
    new_code_cell(CODE_EVAL_B1A),
    new_code_cell(CODE_EVAL_MODEL),
    new_markdown_cell(MARKDOWN_CMP),
    new_code_cell(CODE_CMP),
    new_markdown_cell(MARKDOWN_EVAL_DETAIL_INTRO),
    new_code_cell(CODE_EVAL_DETAIL),
    new_markdown_cell(MARKDOWN_TEST_INTRO),
    new_code_cell(CODE_TEST_PREDS),
    new_code_cell(CODE_TEST_EVAL),
    new_code_cell(CODE_TEST_TABLE),
    new_markdown_cell(MARKDOWN_PERSIST),
    new_code_cell(CODE_PERSIST),
    new_markdown_cell(MARKDOWN_TESTS),
    new_code_cell(CODE_TESTS),
    new_code_cell(CODE_SUMMARY),
]

# --------------------------------------------------------------------------
# Build-time guard: compile every code cell and resolve names
# --------------------------------------------------------------------------
# nbformat.validate() only checks the *JSON envelope*. The cell bodies are
# never compiled, so a name that is used but never imported in that cell ships
# to Colab and raises NameError on the first run -- exactly the ``Dict[int, int]``
# bug that shipped in the tensorize cell. Each cell is its own module scope in
# Jupyter, so "imported in a later cell" does NOT help and must not be treated
# as resolving.
#
# Scope analysis uses stdlib ``symtable`` (real lexical scoping, not a regex
# guess) and carries the set of names bound by *earlier* cells forward, which is
# precisely Jupyter's execution semantics.

def _cell_source(cell) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _is_ipython_cell(src: str) -> bool:
    """True for cells containing ``!shell`` / ``%magic`` lines (not valid Python)."""
    for line in src.splitlines():
        head = line.lstrip()
        if head.startswith(("!", "%", "%%")):
            return True
    return False


def _scope_walk(tbl, enclosing, available, unresolved, is_module: bool) -> None:
    """Collect names referenced by ``tbl`` that resolve to nothing visible."""
    local = set()
    for sym in tbl.get_symbols():
        if sym.is_assigned() or sym.is_imported() or sym.is_parameter() or sym.is_namespace():
            local.add(sym.get_name())
    visible = local | enclosing
    for sym in tbl.get_symbols():
        name = sym.get_name()
        if name in visible:
            continue
        # ``is_free`` means "bound in an enclosing *function* scope", which
        # ``visible`` already accounts for; if it were wrong, compile() would
        # have raised first.
        if sym.is_free():
            continue
        if not (sym.is_referenced() or sym.is_declared_global()):
            continue
        if name in available:
            continue
        unresolved.append(name)
    for child in tbl.get_children():
        _scope_walk(child, visible, available, unresolved, is_module=False)


def _guard_code_cells(cells) -> int:
    import builtins
    import symtable

    # Names injected by Jupyter/Colab or provided by the kernel at runtime.
    available = set(dir(builtins)) | {
        "__name__", "__file__", "__doc__", "__builtins__", "__spec__",
        "display", "get_ipython", "In", "Out", "exit", "quit", "_", "__", "___",
    }
    problems = []
    checked = skipped = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = _cell_source(cell)
        if _is_ipython_cell(src):
            skipped += 1
            continue
        first = src.strip().splitlines()[0] if src.strip() else ""
        label = first[:72]
        try:
            compile(src, "<generated cell>", "exec")
        except SyntaxError as exc:
            problems.append(f"SYNTAX  [{label}]\n        {exc}")
            continue
        checked += 1
        table = symtable.symtable(src, "<generated cell>", "exec")
        # Module-level binds of THIS cell are visible to closures defined in it,
        # so fold them in before resolving references (Python resolves globals at
        # call time, not at def time).
        module_bound = {
            sym.get_name()
            for sym in table.get_symbols()
            if sym.is_assigned() or sym.is_imported() or sym.is_parameter() or sym.is_namespace()
        }
        cell_available = available | module_bound
        unresolved: list = []
        _scope_walk(table, frozenset(), cell_available, unresolved, is_module=True)
        for name in sorted(set(unresolved)):
            problems.append(f"NAME    {name!r} is used but never bound in this cell\n        [{label}]")
        available |= module_bound

    if problems:
        print(f"\n!!! {len(problems)} notebook cell problem(s) found — refusing to write !!!\n")
        for p in problems:
            print("  " + p)
        print(
            "\nEach cell is its own Jupyter scope: a name imported in a LATER cell "
            "does not resolve. Add the import to the cell that uses it.\n"
        )
        raise SystemExit(1)
    print(f"cell guard: {checked} code cells compile, all names resolve ({skipped} IPython cell(s) skipped)")
    return checked


_guard_code_cells(cells)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
nb.metadata["colab"] = {"gpuType": "T4", "toc_visible": True, "provenance": []}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)

# verify parse + json round trip
with open(OUT, "r") as fh:
    json.load(fh)
nb2 = nbformat.read(OUT, as_version=4)
nbformat.validate(nb2)
print(f"wrote {OUT.resolve()} — {len(cells)} cells, valid nbformat 4 JSON")

