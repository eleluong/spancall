#!/usr/bin/env python3
"""Build ``notebooks/spancall_lean_v3_2_train_eval.ipynb``.

v3.2 was previously maintained directly as notebook JSON, which meant every fix
landed as an unreviewable blob edit. This module is the source of truth: the
cell bodies below are the notebook, and ``python scripts/build_lean_v3_2_train_eval_notebook.py``
regenerates the artifact from them.

Why the cell text is inlined rather than sliced-and-patched from an earlier
builder: v3.2 diverged from v3.1 in 13 of 19 code cells (~20% of characters and
several cells differ almost entirely), so a rename-plus-override layer would
have been a fake abstraction. Each cell here is written out in full and can be
read as ordinary Python.

The build enforces the same contract as every other builder in ``scripts/``:

* every code cell must compile and every free name must resolve in notebook
  order (``_guard_code_cells``), and
* the result must be valid nbformat 4 JSON.

Usage::

    python scripts/build_lean_v3_2_train_eval_notebook.py           # write the notebook
    python scripts/build_lean_v3_2_train_eval_notebook.py --check   # fail if out of date
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / ".." / "notebooks" / "spancall_lean_v3_2_train_eval.ipynb"

# ``CODE``/``MD`` just name the cell type so the list below reads top-to-bottom.
CODE = "code"
MD = "markdown"

# (cell_type, source) in notebook order.
CELLS: List[Tuple[str, str]] = [
    (MD, r'''# SpanCall Lean Flow **v3.2** — Normal vs. Complex Evaluation & xLAM Tool-Extraction Benchmark

End-to-end training and evaluation notebook for **v3.2**, building on v3.1:

- **Normal vs. Complex type separation.** Every extraction & E2E benchmark (§4b, §4c, §4d, §4e, §5, §6) separates **Normal (scalar)** types (`string`, `integer`, `number`, `boolean`, `date`, `enum`) from **Complex** types (`array`, `object`, `object-leaf`) at both parameter and call levels.
- **Full xLAM + Hermes tool-extraction benchmark.** §4.0–§4e now evaluate both `hermes_eval` and `xlam_eval` (Step 1 `H1`, Step 2 `H3` oracle extraction, Step 3 E2E waterfall, per-type audit, and stage-attributed forensics).
- **Dual-mode array, object decomposition, pointer-favored loss (`H3` > `H1`), and batched eval** carried forward from v3.1.

**Paths:** `data/processed_v3_2`, `data/tensorized_lean_v3_2`, `results_lean_v3_2`.'''),
    (MD, r'''---
## 0. Setup & Configuration

Clones or updates the SpanCall repository, installs dependencies in editable mode, and configures the v3 paths (`data/tensorized_lean_v3_2`, `results_lean_v3_2`), the mid-training probe knobs, the tool-count bucket edges, and the HF export target (prompted via `input()`). Nothing here requires a key — the HF push is guarded later.'''),
    (CODE, r'''# 0a. Pull latest code from public GitHub repository and install dependencies
import os, sys
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

if (PROJECT_ROOT / ".git").is_dir():
    print(f"Pulling latest code from {GIT_BRANCH}...")
    !git pull origin {GIT_BRANCH} || echo "git pull note: continuing with local files"

print(f"SpanCall project root: {PROJECT_ROOT}")
%pip install -q -e ".[model,data,agent,dev]" huggingface_hub'''),
    (CODE, r'''# 0b. Configuration & Environment (Full Data, v3.2)
import os, subprocess, sys
from pathlib import Path
import torch
import yaml

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "spancall" / "__init__.py").is_file():
    if (PROJECT_ROOT.parent / "spancall" / "__init__.py").is_file():
        PROJECT_ROOT = PROJECT_ROOT.parent
    elif (PROJECT_ROOT / "spancall" / "spancall" / "__init__.py").is_file():
        PROJECT_ROOT = PROJECT_ROOT / "spancall"
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HF_DATASET_ID = "convoicon/spancall"

DRIVE_MOUNT       = False
DRIVE_OUT         = "/content/drive/MyDrive/spancall_lean_artifacts"

# Optional browser download of the results at the end of the eval cell, for a
# runtime with no Drive mount - or when you just want the files on your laptop.
# Off by default: `files.download()` needs a live front-end, and the cell that
# does the downloading is also the one a scripted nbconvert run executes.
DOWNLOAD_RESULTS  = False

DATA_DIR          = str(PROJECT_ROOT / "data" / "processed_v3_2")
V1_DATA_DIR       = str(PROJECT_ROOT / "data" / "processed")
TENSOR_DIR        = str(PROJECT_ROOT / "data" / "tensorized_lean_v3_2")
RESULTS_DIR       = str(PROJECT_ROOT / "results_lean_v3_2")
EVAL_SPLIT        = "hermes_eval"
EVAL_SPLITS       = ["hermes_eval", "xlam_eval"]
TEST_SPLITS       = [
    "hermes_test_ood_schema",
    "hermes_test_ood_para",
    "hermes_test_adversarial",
    "xlam_eval",
    "xlam_test_ood_schema",
]

# xLAM-60k (1-call) mixing knobs:
#   0     -> Hermes-only baseline (Run A)
#   5000  -> Hermes + 5k xLAM (Run B)
#   15000 -> Hermes + 15k xLAM (Run C, default)
#   None  -> Hermes + full xLAM train
XLAM_TRAIN_ROWS   = 15000
XLAM_DEV_ROWS     = 1000
SINGLE_CALL_ONLY  = True

MAX_UTT_STALE_RATE = 0.02
MAX_UNRESOLVABLE   = 0.02
ALLOW_STALE_LABELS = False
FORCE_REALIGN      = False
FORCE_HF_DOWNLOAD  = False
MIN_JOINT_MARGIN   = 0.02

DEVICE          = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
EXPERIMENT_NAME = "spancall_lean_modernbert"
SEED            = 42

BACKBONE       = "answerdotai/ModernBERT-large"
MAX_SEQ_LEN    = 4096
MAX_SPAN_WIDTH = 16
MAX_CANDS      = 1024

_cfg_yaml = {}
try:
    with open("configs/default.yaml") as _fh:
        _cfg_yaml = yaml.safe_load(_fh) or {}
except Exception as _cfg_exc:
    print(f"[!] configs/default.yaml unreadable ({type(_cfg_exc).__name__}: {_cfg_exc})")

MAX_POS = int(((_cfg_yaml.get("optimization") or {}).get("max_pos", 3)) or 3)
_data_cfg = _cfg_yaml.get("data") or {}
MAX_TOOLS = _data_cfg.get("max_tools")
MAX_TOOLS = None if MAX_TOOLS in (None, "", 0) else int(MAX_TOOLS)
JITTER_PACK_TOOLS = bool(_data_cfg.get("jitter_pack_tools", True))

SPLIT_INPUTS = {
    "t2":  ["hermes_train.jsonl", "xlam_train.jsonl"],
    "dev": ["hermes_dev.jsonl", "xlam_dev.jsonl"],
}
SAMPLE_LIMITS     = {"t2": None, "dev": None}
DATA_FRACTION = 1.0
TENSORIZE_WORKERS = max(1, min(os.cpu_count() or 1, 16))
EVAL_LIMIT        = 2000
TEST_LIMIT        = 600
EVAL_BATCH_SIZE   = int(os.environ.get("EVAL_BATCH_SIZE", "8"))   # inference only; small on purpose, see note
EVAL_WORKERS      = 4
# NOTE (latency): SpanCall's per-row `latency_ms` is an AMORTISED cost
# (batch_forward_ms / rows_in_batch + per-row work), so it shrinks as the batch
# grows and a batch of 32 flatters it. Evaluation therefore uses a small batch
# (4-8) to stay closer to serving conditions. Training batch sizes
# (MICRO_BATCH_SIZE / EFFECTIVE_BATCH_SIZE) are unaffected. Predictions can also
# differ by a row or two between batch sizes (bf16 + padding), so report every
# number from ONE run at ONE batch size.

# ---- Lean v3: mid-training probe eval (T13) -------------------------------
# A SMALL, fixed set scored every PROBE_EVERY_STEPS optimiser steps so
# convergence / overfit is visible mid-run instead of once per epoch.
# Sources are the *dev* corpora, filtered to rows that did NOT land in
# `dev.jsonl` (see the tensorize cell) and never the `hermes_eval` slice --
# the probe must not leak into the number the run is judged on.
PROBE_SPLIT       = "probe"
PROBE_N           = 300
PROBE_EVERY_STEPS = 1000
PROBE_PERCENT     = 0.10   # fallback cadence: every 10% of optimiser steps
PROBE_LIMIT       = 300
PROBE_SOURCES     = ["xlam_dev.jsonl", "hermes_dev.jsonl"]

# ---- Lean v3: Hugging Face Hub (interactive input) -----------------------
# Press Enter at any prompt to keep the default (or skip optional token).
# Non-interactive runs (e.g. nbconvert / CI) catch EOFError and use defaults.
def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else " [optional — Enter to skip]"
    try:
        val = input(f"{label}{suffix}: ").strip()
    except (EOFError, OSError):
        val = ""
    return val or default

HF_TOKEN = _prompt("Hugging Face access token")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

HF_REPO_ID      = _prompt("HF model repo ID", "convoicon/spancall-lean-v3-2")
HF_PUSH_PRIVATE = False

# ---- Lean v3: tool-count buckets for the eval pivot ----------------------
# Buckets are keyed on ``len(sample.tools)`` -- the distractor-pool size H1
# must actually choose from -- NOT on the number of GT calls, which moves
# when SINGLE_CALL_ONLY filtering is applied and would make the pivot's row
# definitions depend on which filter ran rather than on the row itself.
TOOL_BUCKET_EDGES = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, None, "11+")]


def _tool_bucket(n_tools) -> str:
    """Map a row's tool count onto its bucket label."""
    n = int(n_tools or 0)
    for _lo, _hi, _label in TOOL_BUCKET_EDGES:
        if n >= _lo and (_hi is None or n <= _hi):
            return _label
    return "<1"


def _corpus_of_row(row) -> str:
    """Corpus label for a raw row (source_dataset, then sample_id prefix)."""
    _sd = str((row or {}).get("source_dataset") or "")
    if _sd.startswith("xlam"):
        return "xlam"
    if _sd.startswith("hermes"):
        return "hermes"
    return "xlam" if str((row or {}).get("sample_id") or "").startswith("xlam") else "hermes"

print("=" * 72)
print("LEAN FLOW v3.2 CONFIG (HERMES + xLAM-60K 1-CALL)")
print("=" * 72)
print(f"EXPERIMENT_NAME   = {EXPERIMENT_NAME}  SEED = {SEED}")
print(f"BACKBONE          = {BACKBONE}")
print(f"MAX_SEQ_LEN       = {MAX_SEQ_LEN}  MAX_SPAN_WIDTH = {MAX_SPAN_WIDTH}  MAX_POS = {MAX_POS}")
print(f"DATA_FRACTION     = {DATA_FRACTION}  TENSORIZE_WORKERS = {TENSORIZE_WORKERS}")
print(f"XLAM_TRAIN_ROWS   = {XLAM_TRAIN_ROWS}  XLAM_DEV_ROWS = {XLAM_DEV_ROWS}  SINGLE_CALL_ONLY = {SINGLE_CALL_ONLY}")
print(f"EVAL_LIMIT        = {EVAL_LIMIT}  TEST_LIMIT = {TEST_LIMIT}")
print(f"PROBE             = {PROBE_SPLIT} n={PROBE_N} every={PROBE_EVERY_STEPS} "
      f"opt-steps (limit={PROBE_LIMIT})")
print(f"HF_REPO_ID        = {HF_REPO_ID}  (push guarded on an HF token)")
print(f"DATA_DIR          = {DATA_DIR}")
print(f"TENSOR_DIR        = {TENSOR_DIR}")
print(f"RESULTS_DIR       = {RESULTS_DIR}")
print(f"DEVICE            = {DEVICE}")'''),
    (CODE, r'''# 0c. Runtime, GPU & Package Check (+ Optional Colab Drive Mount / Download)
!nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo 'No NVIDIA GPU detected (using MPS / CPU fallback)'

import spancall
import spancall.lean as lean
import span_template
import transformers
import datasets

print("spancall     :", spancall.__file__)
print("spancall.lean:", lean.__file__)
print("span_template:", span_template.__file__)
print("transformers :", transformers.__version__, "| datasets:", datasets.__version__)

if DRIVE_MOUNT:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_OUT, exist_ok=True)
    print("Drive mounted:", DRIVE_OUT)'''),
    (MD, r'''---
## 1. Dataset (`data/processed_v3_2/`) & Pre-Flight Gate

Downloads the raw splits from Hugging Face if missing, runs `scripts/realign_v2.py` to produce render-invariant `utt_rel_start`/`utt_rel_end` offsets, and verifies the template stamp (`Gate A`) and `utterance_span` offset validity (`Gate B`).'''),
    (CODE, r'''from huggingface_hub import snapshot_download
from span_template.stamp import check_stamp
from spancall.lean.diagnostics import audit_gt_offsets
import json, os, subprocess, sys

CANONICAL_SPLITS = [
    "hermes_train.jsonl",
    "hermes_dev.jsonl",
    "hermes_calib.jsonl",
    "hermes_eval.jsonl",
    "hermes_test_ood_schema.jsonl",
    "hermes_test_ood_para.jsonl",
    "hermes_test_adversarial.jsonl",
]

XLAM_SPLITS = [
    "xlam_train.jsonl",
    "xlam_dev.jsonl",
    "xlam_calib.jsonl",
    "xlam_eval.jsonl",
    "xlam_test_ood_schema.jsonl",
    "xlam_test_ood_para.jsonl",
    "xlam_test_adversarial.jsonl",
]

os.makedirs(DATA_DIR, exist_ok=True)
_missing_v2 = [f for f in CANONICAL_SPLITS + XLAM_SPLITS if not os.path.exists(os.path.join(DATA_DIR, f))]
if _missing_v2:
    # v2.7: never pull a prebuilt processed_v3_2. Object decomposition changes the
    # rendered param list, so a stale directory would train on v2.7-shaped labels
    # (one opaque `object` param, no dotted leaves) and every object number in
    # section 4d would be 0 for a reason that has nothing to do with the model.
    # Hermes splits are rebuilt by the realign block below; xLAM by prepare_data.
    print(f"[v3.2] {len(_missing_v2)} of {len(CANONICAL_SPLITS) + len(XLAM_SPLITS)} split(s) "
          f"missing in {DATA_DIR}")
    print("       no prebuilt v2 download is used in v2.7 - every split is rebuilt below.")

_already = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in CANONICAL_SPLITS)
_stamp_ok = check_stamp(DATA_DIR).ok if _already else False
_decomp_ok = False
if _already:
    _eval_p = os.path.join(DATA_DIR, f"{EVAL_SPLIT}.jsonl")
    with open(_eval_p) as _fh_d:
        for _i_d, _l_d in enumerate(_fh_d):
            if _i_d >= 300:
                break
            if '"object_paths":' in _l_d and '"object_paths": null' not in _l_d:
                _decomp_ok = True
                break
if FORCE_REALIGN or not _already or not _stamp_ok or not _decomp_ok:
    os.makedirs(V1_DATA_DIR, exist_ok=True)
    missing_v1 = [f for f in CANONICAL_SPLITS if not os.path.exists(os.path.join(V1_DATA_DIR, f))]
    if missing_v1:
        snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            local_dir=str(PROJECT_ROOT),
            allow_patterns=["data/processed/*.jsonl", "data/processed/*.json"],
            token=HF_TOKEN,
        )
    print(f"Re-aligning v1 -> {DATA_DIR} ...")
    _realign = [
        sys.executable, os.path.join(str(PROJECT_ROOT), "scripts", "realign_v2.py"),
        "--src_dir", V1_DATA_DIR,
        "--out_dir", DATA_DIR,
        "--gate_split", EVAL_SPLIT,
        "--max_utt_stale_rate", str(MAX_UTT_STALE_RATE),
        "--max_unresolvable", str(MAX_UNRESOLVABLE),
    ]
    if ALLOW_STALE_LABELS:
        _realign.append("--force")
    _rc = subprocess.run(_realign, cwd=str(PROJECT_ROOT)).returncode
    if _rc != 0:
        raise SystemExit(f"realign_v2 exited {_rc}. Inspect data/processed_v3_2/realign_report.json.")
else:
    print(f"v2 splits ready in {DATA_DIR}")

# xLAM-60k (1-call) splits: download public xlam_*.jsonl from convoicon/spancall
# and re-align through the v2.7 aligner (avoids the gated Salesforce/xlam-60k
# hub download unless the snapshot lacks them).
_missing_xlam = [f for f in XLAM_SPLITS if not os.path.exists(os.path.join(DATA_DIR, f))]
if _missing_xlam and (XLAM_TRAIN_ROWS is None or XLAM_TRAIN_ROWS > 0):
    _xlam_src_dir = os.path.join(str(PROJECT_ROOT), "data", "processed_v2")
    _missing_xlam_src = [f for f in XLAM_SPLITS if not os.path.exists(os.path.join(_xlam_src_dir, f))]
    if _missing_xlam_src:
        print(f"Downloading xLAM splits from {HF_DATASET_ID} ...")
        snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            local_dir=str(PROJECT_ROOT),
            allow_patterns=["data/processed_v2" + "/xlam_*.jsonl", "data/processed_v2" + "/*.json"],
            token=HF_TOKEN,
        )
    if all(os.path.exists(os.path.join(_xlam_src_dir, f)) for f in XLAM_SPLITS):
        print(f"Re-aligning xLAM splits -> {DATA_DIR} ...")
        _xlam_realign = [
            sys.executable, os.path.join(str(PROJECT_ROOT), "scripts", "realign_v2.py"),
            "--src_dir", _xlam_src_dir,
            "--out_dir", DATA_DIR,
            "--splits", *[os.path.splitext(f)[0] for f in XLAM_SPLITS],
            "--gate_split", "xlam_eval",
            "--max_utt_stale_rate", str(MAX_UTT_STALE_RATE),
            "--max_unresolvable", str(MAX_UNRESOLVABLE),
        ]
        if ALLOW_STALE_LABELS:
            _xlam_realign.append("--force")
        _xrc = subprocess.run(_xlam_realign, cwd=str(PROJECT_ROOT)).returncode
        if _xrc != 0:
            raise SystemExit(f"realign_v2 (xlam) exited {_xrc}. Inspect data/processed_v3_2/realign_report.json.")
    else:
        print(f"Preparing xLAM-60k (1-call) splits into {DATA_DIR} via scripts/prepare_data.py ...")
        _prep_rc = subprocess.run([
            sys.executable, os.path.join(str(PROJECT_ROOT), "scripts", "prepare_data.py"),
            "--sources", "xlam",
            "--output_dir", DATA_DIR,
        ], cwd=str(PROJECT_ROOT)).returncode
        if _prep_rc != 0:
            raise SystemExit(f"prepare_data.py --sources xlam exited {_prep_rc} (set HF_TOKEN if Salesforce/xlam-function-calling-60k is gated)")
        # v2.7 provenance note: template_stamp.json is directory-level, so the
        # stamp now attests the xLAM splits prepare_data just wrote, not the
        # Hermes splits realigned above. Gate A below still validates the
        # template REVISION (the thing that invalidated v1), nothing more.
        print("[Gate A / note] prepare_data.py re-stamped " + DATA_DIR + " -> Hermes splits are no longer attested by that stamp.")

# Pre-Flight Gate A: template stamp
_verdict = check_stamp(DATA_DIR)
print(f"[Gate A / Stamp] {_verdict.status}: {_verdict.message()}")
if not _verdict.ok and not ALLOW_STALE_LABELS:
    raise SystemExit("PRE-FLIGHT FAILED (template stamp): " + _verdict.message())

# Pre-Flight Gate B: offset audit on every split this notebook later
# reports on, not just eval+dev. The OOD / adversarial splits come from
# the same aligner, so an offset defect there has to fail here rather
# than surface as a mysteriously low section-5 score.
_audit_splits = [EVAL_SPLIT, "hermes_dev"]
for _osp in list(TEST_SPLITS) + ["xlam_dev"]:
    if _osp in _audit_splits:
        continue
    if os.path.exists(os.path.join(DATA_DIR, f"{_osp}.jsonl")):
        _audit_splits.append(_osp)
for _split in _audit_splits:
    _m = audit_gt_offsets(_split, DATA_DIR)
    _stale = _m.get("utterance_span_stale_rate")
    _unres = _m.get("unresolvable_rate")
    _n_utt = int(_m.get("utterance_span_targets") or 0)
    if _n_utt == 0 and "adversarial" in _split:
        _ok = True  # marker-injection splits have target_spans={} by contract
    else:
        _ok = (
            isinstance(_stale, float) and _stale == _stale and _stale < MAX_UTT_STALE_RATE
            and ((_unres is None) or (isinstance(_unres, float) and _unres == _unres and _unres < MAX_UNRESOLVABLE))
        )
    print(f"[Gate B / Audit] {_split:<16} utterance_span_stale_rate={_stale} "
          f"({_m.get('utterance_span_targets')} targets), unresolvable_rate={_unres} -> {'PASS' if _ok else 'FAIL'}")
    if not _ok and not ALLOW_STALE_LABELS:
        raise SystemExit(f"PRE-FLIGHT FAILED on {_split}")

# Pre-Flight Gate C (v2.7, warning-only): is object decomposition present?
# If the labels are still v2.6-shaped (one opaque `object` param, no dotted
# leaves) then every 'object-leaf' figure in section 4d is structurally 0
# and the v2.7 question cannot be answered by this run. Warn loudly instead
# of aborting, because the label-side work is plan steps 2-3.
from spancall.data.schema import UnifiedSample as _C_Sample
from spancall.data.flatten import PATH_SEP as _C_SEP
_c_roots = _c_leaves = _c_opaque = 0
_c_path = os.path.join(DATA_DIR, f"{EVAL_SPLIT}.jsonl")
if os.path.exists(_c_path):
    with open(_c_path) as _fh_c:
        for _i_c, _line_c in enumerate(_fh_c):
            if _i_c >= 300:
                break
            if not _line_c.strip():
                continue
            _s_c = _C_Sample.from_dict(json.loads(_line_c))
            for _t_c in _s_c.tools:
                _c_roots += len(_t_c.object_paths or {})
                for _p_c in _t_c.parameters:
                    if _C_SEP in _p_c.name:
                        _c_leaves += 1
                    elif (_p_c.param_type or "").lower() == "object":
                        _c_opaque += 1
if _c_roots + _c_leaves + _c_opaque == 0:
    print("[Gate C / Decompose] no object-bearing tool in "
          + EVAL_SPLIT + " - nothing to check.")
else:
    _c_ok = _c_roots > 0 and _c_leaves >= _c_opaque
    print(f"[Gate C / Decompose] roots={_c_roots} leaves={_c_leaves} opaque={_c_opaque} -> {'PASS' if _c_ok else 'WARN'}")
    if not _c_ok:
        print("  [!] labels look v2.6-shaped for object params: opaque `object`")
        print("      params outnumber dotted leaves. Section 4d will report zero")
        print("      object-leaf slots until plan steps 2-3 land (nested schema")
        print("      parsing + aligner object case). Do not read 4d as model quality.")
        print("      Hint: if reusing an older directory, set FORCE_REALIGN = True.")'''),
    (MD, r'''---
## 2. Lean Tensorizer — Hermes + xLAM-60k (1-Call) Mix

Tensorizes `hermes_train.jsonl` + `xlam_train.jsonl` (up to `XLAM_TRAIN_ROWS`) into `t2.jsonl` and `hermes_dev.jsonl` + `xlam_dev.jsonl` (up to `XLAM_DEV_ROWS`) into `dev.jsonl` using a shared multiprocessing fork pool.'''),
    (CODE, r'''from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json, math, multiprocessing as mp, os, queue, random, time
import yaml
from transformers import AutoTokenizer

from span_template import SPECIAL_MARKER_TOKENS
from spancall.data.schema import UnifiedSample
from spancall.model.masks import CandidateMaskBuilder
from spancall.lean.tensorize import ToolPool, tensorize_sample_lean, has_positive_targets

tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_MARKER_TOKENS)})
mask_builder = CandidateMaskBuilder(max_seq_len=MAX_SEQ_LEN, max_span_width=MAX_SPAN_WIDTH)

# Lean v3: PACK_TOOLS = 6 with JITTER_PACK_TOOLS = True draws the per-row
# floor from [1, 6] (uniform mean 3.5; ~4.0 tools/row after native multi-tool
# rows), giving H1 harder distractor sets while keeping T=1 in-distribution.
PACK_TOOLS = 6

TOOL_POOL = None
PACK_TOOLS_EPH = 1

def _iter_raw_lines(path: str):
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line

def _iter_raw_dicts(path: str):
    for line in _iter_raw_lines(path):
        yield json.loads(line)

def _fraction_subset(raw_lines, frac: float, seed: int):
    raw_lines = list(raw_lines)
    if frac >= 1.0:
        return raw_lines
    rng = random.Random(int(seed))
    n = max(int(len(raw_lines) * frac), 1)
    keep = set(rng.sample(range(len(raw_lines)), min(n, len(raw_lines))))
    return [d for i, d in enumerate(raw_lines) if i in keep]

def _serialize(row):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}

def _tensorize_lean_task(args):
    raw_line, pack_multi = args
    drops: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    try:
        d = json.loads(raw_line)
        s = UnifiedSample.from_dict(d)
        ts_stats: Dict[str, Any] = {}
        row = tensorize_sample_lean(
            s,
            tokenizer=tokenizer,
            mask_builder=mask_builder,
            max_seq_len=MAX_SEQ_LEN,
            max_cands=MAX_CANDS,
            max_pos=int(MAX_POS),
            pack_multi_tools=pack_multi,
            stats=ts_stats,
            tool_pool=TOOL_POOL,
            pack_tools=PACK_TOOLS_EPH,
            max_tools=MAX_TOOLS,
            jitter=JITTER_PACK_TOOLS,
            include_abstain=True,
        )
        drops = dict(ts_stats.get("param_drop_reasons", {}))
        for _t, _cnt in (ts_stats.get("render_T_hist") or {}).items():
            _k = f"render_T={int(_t)}"
            drops[_k] = drops.get(_k, 0) + int(_cnt)
        if row is None or (not has_positive_targets(row) and not bool(row.get("is_abstain", 0))):
            return None, drops, errors
        return json.dumps(_serialize(row)), drops, errors
    except Exception as exc:
        cls = type(exc).__name__
        errors[cls] = errors.get(cls, 0) + 1
        return None, drops, errors

_WINDOW = 2048

def _drain_ordered(pool, raw, pack_multi: bool, absorb, done, rows_len, need,
                   workers: int = 1, window: int = _WINDOW):
    n = len(raw)
    guard = max(2 * int(workers), 16)
    finished_q: queue.SimpleQueue = queue.SimpleQueue()
    buffer: Dict[int, Any] = {}
    nxt = 0
    submit = 0
    inflight = 0
    while nxt < n and not done():
        budget = max(0, need - rows_len())
        cap = min(window, budget + guard)
        while submit < n and (submit - nxt) < cap and not done():
            fut = pool.submit(_tensorize_lean_task, (raw[submit], pack_multi))
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
    if need is None:
        need = len(raw)
    rows: List[str] = []
    drops_total: Counter = Counter()
    errors_total: Counter = Counter()

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

    if executor is not None:
        _drain_ordered(executor, raw, pack_multi, _absorb, _done,
                       rows_len=lambda: len(rows), need=need, workers=workers)
        return rows, drops_total, errors_total

    for line in raw:
        _absorb(_tensorize_lean_task((line, pack_multi)))
        if _done():
            break
    return rows, drops_total, errors_total

def _filter_single_call_lines(raw_lines):
    if not SINGLE_CALL_ONLY:
        return list(raw_lines), 0
    kept = []
    dropped = 0
    for line in raw_lines:
        d = json.loads(line)
        if len(d.get("ground_truth_calls") or []) == 1:
            kept.append(line)
        else:
            dropped += 1
    return kept, dropped

def _cap_lines(raw_lines, cap: Optional[int], seed: int):
    raw_lines = list(raw_lines)
    if cap is None or cap >= len(raw_lines):
        return raw_lines
    if cap <= 0:
        return []
    rng = random.Random(int(seed))
    keep = set(rng.sample(range(len(raw_lines)), cap))
    return [d for i, d in enumerate(raw_lines) if i in keep]

os.makedirs(TENSOR_DIR, exist_ok=True)
_t2_sources = SPLIT_INPUTS.get("t2") or next(iter(SPLIT_INPUTS.values()))
if isinstance(_t2_sources, str):
    _t2_sources = [_t2_sources]
if PACK_TOOLS > 1:
    _pool_dicts = []
    for _src_file in _t2_sources:
        _src_path = os.path.join(DATA_DIR, _src_file)
        if not os.path.exists(_src_path):
            continue
        if _src_file.startswith("xlam_"):
            if XLAM_TRAIN_ROWS is not None and XLAM_TRAIN_ROWS <= 0:
                continue
            for _l in _cap_lines(_iter_raw_lines(_src_path), XLAM_TRAIN_ROWS, SEED):
                _pool_dicts.append(json.loads(_l))
        else:
            _pool_dicts.extend(_iter_raw_dicts(_src_path))
    if _pool_dicts:
        TOOL_POOL = ToolPool.from_row_dicts(_pool_dicts, seed=SEED)
        print(f"[tool-pool] built from {_t2_sources}: {len(TOOL_POOL)} schemas "
              f"(collisions={len(TOOL_POOL.name_collisions)}, pack_tools={PACK_TOOLS})")
PACK_TOOLS_EPH = int(PACK_TOOLS) if TOOL_POOL is not None else 1

_fork_ctx = None
_shared_ex = None
if TENSORIZE_WORKERS > 1:
    try:
        _fork_ctx = mp.get_context("fork")
        _shared_ex = ProcessPoolExecutor(max_workers=TENSORIZE_WORKERS, mp_context=_fork_ctx)
    except Exception as _exc:
        print(f"[!] fork pool unavailable ({_exc}); running single-process")

try:
    for out_name, in_spec in SPLIT_INPUTS.items():
        in_files = [in_spec] if isinstance(in_spec, str) else list(in_spec)
        raw: List[str] = []
        per_src_counts: Dict[str, int] = {}
        for in_name in in_files:
            src = os.path.join(DATA_DIR, in_name)
            if not os.path.exists(src):
                per_src_counts[in_name] = 0
                continue
            filtered_lines, n_multi_drop = _filter_single_call_lines(_iter_raw_lines(src))
            if in_name.startswith("xlam_"):
                cap = XLAM_TRAIN_ROWS if out_name == "t2" else XLAM_DEV_ROWS
                part = _cap_lines(filtered_lines, cap, SEED)
            else:
                part = _fraction_subset(filtered_lines, DATA_FRACTION, SEED)
            per_src_counts[in_name] = len(part)
            if n_multi_drop:
                print(f"  [{out_name}/{in_name}] SINGLE_CALL_ONLY dropped {n_multi_drop} multi-call rows")
            raw.extend(part)
        if len(in_files) > 1:
            _shuf_rng = random.Random(int(SEED))
            _shuf_rng.shuffle(raw)
        limit = SAMPLE_LIMITS.get(out_name)
        t0 = time.perf_counter()
        rows, drops, errors = _tensorize_many(raw, True, limit, TENSORIZE_WORKERS, executor=_shared_ex)
        dt = time.perf_counter() - t0
        out_path = os.path.join(TENSOR_DIR, f"{out_name}.jsonl")
        with open(out_path, "w") as fh:
            fh.write("\n".join(rows) + ("\n" if rows else ""))
        render_hist = {k.split("=")[1]: v for k, v in drops.items() if k.startswith("render_T=")}
        print(f"[tensorize] {out_name:<4} <- {per_src_counts}: {len(rows):>6} rows in {dt:.1f}s "
              f"({len(rows)/max(dt, 1e-6):.1f} rows/s) | render_T={render_hist} "
              f"| match_missed={drops.get('match_missed', 0)} | errors={dict(errors)}")
    # --- Lean v3: build the probe split (small, fixed, DISJOINT from dev) ---
    # The probe is what LeanTrainer scores every PROBE_EVERY_STEPS optimiser
    # steps. It must be (a) small, so that cadence is affordable, and (b) NOT
    # a subset of dev.jsonl -- otherwise the probe curve would just re-measure
    # the rows the per-epoch dev eval already reports, and a plateau would be
    # indistinguishable from a re-scored flat dev set.
    _dev_ids = set()
    _dev_tensor_path = os.path.join(TENSOR_DIR, "dev.jsonl")
    if os.path.exists(_dev_tensor_path):
        with open(_dev_tensor_path, "r") as _fh_dev:
            for _l in _fh_dev:
                if _l.strip():
                    _dev_ids.add(json.loads(_l).get("sample_id"))
    _probe_raw: List[str] = []
    _probe_cap = max(PROBE_N * 3, PROBE_N)
    for _pf in PROBE_SOURCES:
        if len(_probe_raw) >= _probe_cap:
            break
        _pf_path = os.path.join(DATA_DIR, _pf)
        if not os.path.exists(_pf_path):
            print(f"  [probe] source {_pf} missing - skipped")
            continue
        for _l in _iter_raw_lines(_pf_path):
            if json.loads(_l).get("sample_id") in _dev_ids:
                continue
            _probe_raw.append(_l)
            if len(_probe_raw) >= _probe_cap:
                break
    random.Random(int(SEED) + 977).shuffle(_probe_raw)
    _prows, _pdrops, _perrors = _tensorize_many(
        _probe_raw, True, PROBE_N, TENSORIZE_WORKERS, executor=_shared_ex
    )
    _probe_out = os.path.join(TENSOR_DIR, f"{PROBE_SPLIT}.jsonl")
    with open(_probe_out, "w") as _fh_p:
        _fh_p.write("\n".join(_prows) + ("\n" if _prows else ""))
    _probe_ids = {
        json.loads(_l).get("sample_id") for _l in _prows if _l.strip()
    } - {None}
    print(f"[probe] {len(_prows)} rows -> {_probe_out} "
          f"(sources={PROBE_SOURCES}, overlap_with_dev={len(_probe_ids & _dev_ids)}, "
          f"render_drops={dict(_pdrops)}, errors={dict(_perrors)})")
    if not _prows:
        print("[probe] EMPTY - mid-training probe eval will not run. Check that "
              "PROBE_SOURCES rows survive tensorization.")
finally:
    if _shared_ex is not None:
        _shared_ex.shutdown(wait=True)

fingerprint = {
    "experiment_name": EXPERIMENT_NAME,
    "seed": SEED,
    "backbone": BACKBONE,
    "max_seq_len": MAX_SEQ_LEN,
    "max_span_width": MAX_SPAN_WIDTH,
    "max_pos": MAX_POS,
    "data_fraction": DATA_FRACTION,
    "xlam_train_rows": XLAM_TRAIN_ROWS,
    "xlam_dev_rows": XLAM_DEV_ROWS,
    "split_inputs": SPLIT_INPUTS,
    "sample_limits": SAMPLE_LIMITS,
    "lean_flow": True,
}
fp_path = os.path.join(TENSOR_DIR, "config_fingerprint.json")
with open(fp_path, "w") as fh:
    json.dump(fingerprint, fh, indent=2, sort_keys=True)
print("wrote config fingerprint ->", fp_path)'''),
    (MD, r'''---
## 3. Training — Lean Flow (`scripts/train_lean.py`)

Trains `LeanSpanCallModel` (`ModernBERT-large` + `ToolSelectionHead` H1 + `LeanPointerHead` H3) on stage `T2` with full data (`TRAIN_FRACTION = 1.0`), effective batch size 128, `bf16`, and best-checkpoint restoration.

v3 adds two things to the training loop itself, both configured here and implemented in `LeanTrainer.run_stage`:

- **Mid-training probe eval.** A small probe split is scored every `PROBE_EVERY_STEPS` optimiser steps (`probe_percent` is the fallback cadence when no step count is set) and written to `train_log.jsonl` as `event="probe_eval"`. It fires only on an accumulation-window boundary, so the weights being scored are exactly the ones the next micro-batch trains from — scoring mid-window would report a mix of stale and updated weights.
- **Detailed step logging.** `train` events (total loss, pointer loss, tool/tool_binary loss, LR, elapsed time) are written to `train_log.jsonl` and printed live every `total_steps // 100` steps, then summarized in §3b.'''),
    (CODE, r'''import os, yaml
from scripts.train_lean import main as train_lean_main

TRAIN_FRACTION = 1.0
NUM_EPOCHS = 3
MICRO_BATCH_SIZE = 32
EFFECTIVE_BATCH_SIZE = 128

with open("configs/default.yaml") as fh:
    cfg = yaml.safe_load(fh)

cfg["experiment"]["name"] = EXPERIMENT_NAME
cfg["experiment"]["seed"] = SEED
cfg["experiment"]["output_dir"] = RESULTS_DIR
cfg["experiment"]["data_dir"] = TENSOR_DIR

cfg["optimization"]["num_epochs"] = NUM_EPOCHS
cfg["optimization"]["batch_size"] = MICRO_BATCH_SIZE
cfg["optimization"]["effective_batch_size"] = EFFECTIVE_BATCH_SIZE
cfg["optimization"]["max_tokens_per_batch"] = 4096
cfg["optimization"]["optimizer"] = "adamw"
cfg["optimization"]["scheduler"] = "cosine_warmup"
cfg["optimization"]["min_lr_ratio"] = 0.10
cfg["optimization"]["early_stopping_patience"] = 1
cfg["optimization"]["restore_best_on_exit"] = True
cfg["optimization"]["mixed_precision"] = "bf16"

cfg["training"]["gradient_accumulation_steps"] = max(int(round(EFFECTIVE_BATCH_SIZE / MICRO_BATCH_SIZE)), 1)
cfg["training"]["gradient_checkpointing"] = False
cfg["training"]["eval_splits"] = ["dev"]
cfg["training"]["eval_limit"] = 2000

# Lean v3: mid-training probe eval. Counted in OPTIMISER steps, so it
# fires many times per epoch rather than once -- that is the whole point
# of the amendment ("run eval each 1k ... with a small test set"). The
# per-epoch dev eval above stays, but it now only has to track train/val
# loss, which is cheap.
cfg["training"]["probe_split"] = PROBE_SPLIT
cfg["training"]["probe_every_steps"] = PROBE_EVERY_STEPS
cfg["training"]["probe_percent"] = PROBE_PERCENT
cfg["training"]["probe_limit"] = PROBE_LIMIT
cfg["training"]["freeze_encoder_stages"] = []

cfg["model"]["backbone_name"] = BACKBONE
cfg["model"]["max_seq_len"] = MAX_SEQ_LEN

_EFFECTIVE_PACK_TOOLS = int(globals().get("PACK_TOOLS_EPH", PACK_TOOLS) or 1)
cfg.setdefault("data", {})["fraction"] = TRAIN_FRACTION
cfg["data"]["pack_tools"] = _EFFECTIVE_PACK_TOOLS

CONFIG_PATH = "configs/colab_lean_v3_2.yaml"
with open(CONFIG_PATH, "w") as fh:
    yaml.safe_dump(cfg, fh)
print(f"wrote {CONFIG_PATH} (backbone={BACKBONE}, epochs={NUM_EPOCHS}, "
      f"micro_bs={MICRO_BATCH_SIZE}, effective_bs={EFFECTIVE_BATCH_SIZE}, "
      f"data_fraction={TRAIN_FRACTION}, pack_tools={_EFFECTIVE_PACK_TOOLS})")

train_lean_main([
    "--config", CONFIG_PATH,
    "--experiment_name", EXPERIMENT_NAME,
    "--output_dir", RESULTS_DIR,
    "--data_dir", TENSOR_DIR,
    "--stages", "T2",
    "--device", DEVICE,
    "--seed", str(SEED),
])'''),
    (CODE, r'''# 3b. Training Telemetry, Best Checkpoint Selection & LeanResolver Init
import glob, json, math, os
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer

from span_template import SPECIAL_MARKER_TOKENS
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver
from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.train.config import TrainingConfig

def _num(rec, key):
    if rec is None:
        return None
    v = rec.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f

def _fmt_cell(v, width, fmt=".3f"):
    return f"{'—':>{width}}" if v is None else f"{v:>{width}{fmt}}"

exp_root = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_seed{SEED}")
log_path = os.path.join(exp_root, "train_log.jsonl")
if os.path.exists(log_path):
    with open(log_path) as fh:
        train_log = [json.loads(l) for l in fh if l.strip()]
    evals = [r for r in train_log if r.get("event") == "epoch_eval"]
    ends  = [r for r in train_log if r.get("event") == "epoch_end"]
    print("=" * 84)
    print("PER-EPOCH HELD-OUT DEV CURVE")
    print("=" * 84)
    print(f"  {'ep':>3}{'n':>6}{'span_acc':>10}{'params_ok':>11}{'exact_call':>12}"
          f"{'exact_T1':>10}{'exact_T2+':>10}{'tool_all':>10}{'tool_T2+':>10}{'train_loss':>12}{'dev_loss':>10}")
    for r in evals:
        ep = r.get("epoch", "?")
        end_r = next((e for e in ends if e.get("epoch") == ep), None)
        print(f"  {ep:>3}{int(_num(r, 'n_span') or _num(r, 'n') or 0):>6}"
              f"{_fmt_cell(_num(r, 'span_acc'), 10)}"
              f"{_fmt_cell(_num(r, 'exact_params_only'), 11)}"
              f"{_fmt_cell(_num(r, 'exact_call'), 12)}"
              f"{_fmt_cell(_num(r, 'exact_call_t1'), 10)}"
              f"{_fmt_cell(_num(r, 'exact_call_multi'), 10)}"
              f"{_fmt_cell(_num(r, 'tool_acc'), 10)}"
              f"{_fmt_cell(_num(r, 'tool_acc_multi'), 10)}"
              f"{_fmt_cell(_num(end_r, 'mean_loss'), 12, '.4f')}"
              f"{_fmt_cell(_num(r, 'eval_loss'), 10, '.4f')}")

# Lean v3: step-level training log + mid-training probe curve.
# Read straight from train_log.jsonl so this cell stays self-contained.
_probe_log = os.path.join(exp_root, "train_log.jsonl")
_train_step_recs = []
_probe_recs = []
if os.path.exists(_probe_log):
    with open(_probe_log) as _fh_pb:
        for _l in _fh_pb:
            if not _l.strip():
                continue
            _rec = json.loads(_l)
            _ev = _rec.get("event")
            if _ev == "train":
                _train_step_recs.append(_rec)
            elif _ev == "probe_eval":
                _probe_recs.append(_rec)
if _train_step_recs:
    print("\n" + "=" * 84)
    print(f"TRAINING STEP LOG ({len(_train_step_recs)} logged checkpoints)")
    print("=" * 84)
    print(f"  {'ep':>3}{'opt_step':>10}{'loss':>10}{'ptr':>10}"
          f"{'tool':>10}{'tool_bin':>10}{'lr':>11}{'elapsed':>10}")
    _stride = max(len(_train_step_recs) // 30, 1)
    _shown = _train_step_recs[::_stride]
    if _train_step_recs[-1] is not _shown[-1]:
        _shown.append(_train_step_recs[-1])
    for _tr in _shown:
        _el = _num(_tr, "elapsed_s")
        _el_str = f"{_el:>9.0f}s" if _el is not None else f"{'—':>10}"
        print(f"  {_tr.get('epoch', '?'):>3}"
              f"{int(_tr.get('step') or 0):>10}"
              f"{_fmt_cell(_num(_tr, 'loss'), 10, '.4f')}"
              f"{_fmt_cell(_num(_tr, 'pointer'), 10, '.4f')}"
              f"{_fmt_cell(_num(_tr, 'tool'), 10, '.4f')}"
              f"{_fmt_cell(_num(_tr, 'tool_binary'), 10, '.4f')}"
              f"{_fmt_cell(_num(_tr, 'lr'), 11, '.2e')}"
              f"{_el_str}")
if _probe_recs:
    print("\n" + "=" * 84)
    print(f"MID-TRAINING PROBE CURVE ('{PROBE_SPLIT}', n<={PROBE_LIMIT}, "
          f"every {PROBE_EVERY_STEPS} opt-steps)")
    print("=" * 84)
    print(f"  {'opt_step':>9}{'ep':>4}{'n':>6}{'exact_call':>12}"
          f"{'span_acc':>10}{'tool_acc':>10}{'loss':>10}")
    for _pr in _probe_recs:
        print(f"  {int(_pr.get('global_step') or 0):>9}{_pr.get('epoch', '?'):>4}"
              f"{int(_pr.get('n') or 0):>6}"
              f"{_fmt_cell(_num(_pr, 'exact_call'), 12)}"
              f"{_fmt_cell(_num(_pr, 'span_acc'), 10)}"
              f"{_fmt_cell(_num(_pr, 'tool_acc'), 10)}"
              f"{_fmt_cell(_num(_pr, 'eval_loss'), 10, '.4f')}")
    _pb_best = max(_probe_recs, key=lambda r: float(r.get("exact_call") or -1.0))
    print(f"  [peak] opt_step={_pb_best.get('global_step')} "
          f"exact_call={float(_pb_best.get('exact_call') or 0.0):.4f}")
else:
    print(f"\n[probe] no probe_eval records in {_probe_log} - "
          "probe eval was disabled for this run.")

best_cp = os.path.join(exp_root, "best_checkpoint")
if os.path.exists(os.path.join(best_cp, "model.pt")):
    CHECKPOINT_DIR = best_cp
else:
    cps = sorted(glob.glob(os.path.join(exp_root, "**", "model.pt"), recursive=True))
    CHECKPOINT_DIR = os.path.dirname(cps[-1]) if cps else None
if CHECKPOINT_DIR is None:
    # LeanResolver(model_path=None) skips load_checkpoint entirely, so the
    # eval below would score randomly initialised weights and still print
    # plausible-looking numbers. Refuse instead.
    raise SystemExit(
        "no model.pt under " + exp_root + ": training persisted no checkpoint. "
        "LeanResolver(model_path=None) would run UNTRAINED weights, so every "
        "metric in sections 3-5 would be noise."
    )
print("\nCHECKPOINT_DIR =", CHECKPOINT_DIR)

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

MAX_CALLS = 1 if SINGLE_CALL_ONLY else int(getattr(cfg.data, "max_calls", 1))
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
    min_joint_margin=MIN_JOINT_MARGIN,
    collapse_guard=True,
    prevent_param_collapse=True,
    # Lean v3: THE extraction mode, pinned at construction even though
    # `LeanResolver` now defaults to `two_step`. Pinning it means the
    # mode this notebook scores is visible on the line that creates the
    # resolver, not inferred from a library default.
    resolve_mode="two_step",
)
if DEVICE in ("cuda", "mps"):
    model.to(DEVICE)
model.eval()
if hasattr(encoder, "eval"):
    encoder.eval()
print("LeanResolver ready.")'''),
    (MD, r'''---
## 4. Step-by-Step Evaluation (`hermes_eval` & `xlam_eval`)

### 4.0 Stratified Benchmark Slices & Step 0 (Pre-Encode Sanitize Gate)

Builds the shared deterministic stratified slices (`subset_hermes_eval.jsonl` and `subset_xlam_eval.jsonl` across `F_func`, `F_json`, `A_clean`, `A_adv`), reports corpus x tool-count and **Normal vs. Complex** composition, and runs the Step 0 pre-encode marker-injection check.'''),
    (CODE, r'''import json, os
from collections import Counter
from span_template.template import RecentRow, build_input_multi
from span_template.sanitize import SanitizeError
from spancall.data.schema import UnifiedSample
from spancall.data.align import tooldef_to_template_tool
from spancall.lean.eval_p1 import (
    build_stratified_eval_subset,
    classify_eval_cohort,
)
try:
    from spancall.lean.eval_p1 import (
        COMPLEX_PARAM_TYPES,
        is_complex_param,
        classify_sample_complexity,
    )
except ImportError:
    COMPLEX_PARAM_TYPES = frozenset({"array", "object"})

    def is_complex_param(param, object_paths=None) -> bool:
        if object_paths:
            pname = str(getattr(param, "name", None) or (param.get("name") if isinstance(param, dict) else "") or "")
            root = pname.split(".", 1)[0]
            if any(str(leaf) == pname for leaf in (object_paths.get(root) or ())):
                return True
        raw_pt = getattr(param, "param_type", None) if not isinstance(param, dict) else param.get("param_type")
        if isinstance(raw_pt, (list, tuple)):
            return any(str(t).strip().lower() in COMPLEX_PARAM_TYPES for t in raw_pt)
        return str(raw_pt or "string").strip().lower() in COMPLEX_PARAM_TYPES

    def classify_sample_complexity(sample_or_dict) -> str:
        if isinstance(sample_or_dict, dict):
            tools = sample_or_dict.get("tools") or []
            gt_calls = sample_or_dict.get("ground_truth_calls") or []
        else:
            tools = getattr(sample_or_dict, "tools", None) or []
            gt_calls = getattr(sample_or_dict, "ground_truth_calls", None) or []
        tool_by_name = {
            str(t.get("name") if isinstance(t, dict) else getattr(t, "name", "")): t
            for t in tools if (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
        }
        for call in gt_calls:
            cname = call.get("tool_name") if isinstance(call, dict) else getattr(call, "tool_name", None)
            cargs = (call.get("arguments") if isinstance(call, dict) else getattr(call, "arguments", None)) or {}
            if any(isinstance(v, (list, dict)) for v in cargs.values()):
                return "complex"
            tdef = tool_by_name.get(str(cname or "")) or (tools[0] if tools else None)
            if tdef is None:
                continue
            obj_paths = (tdef.get("object_paths") if isinstance(tdef, dict) else getattr(tdef, "object_paths", None)) or {}
            params = (tdef.get("parameters") if isinstance(tdef, dict) else getattr(tdef, "parameters", None)) or []
            pmap = {
                str(p.get("name") if isinstance(p, dict) else getattr(p, "name", "")): p
                for p in params if (p.get("name") if isinstance(p, dict) else getattr(p, "name", None))
            }
            for arg_name in cargs:
                p = pmap.get(str(arg_name))
                if (p is not None and is_complex_param(p, obj_paths)) or str(arg_name) in obj_paths:
                    return "complex"
        return "normal"

# Fallback if Cell 3 was executed before EVAL_SPLITS was defined
if "EVAL_SPLITS" not in globals():
    EVAL_SPLITS = ["hermes_eval", "xlam_eval"]

ACTIVE_EVAL_SPLITS = [
    sp for sp in EVAL_SPLITS
    if os.path.exists(os.path.join(DATA_DIR, f"{sp}.jsonl"))
]
if not ACTIVE_EVAL_SPLITS:
    ACTIVE_EVAL_SPLITS = [EVAL_SPLIT]

SUBSET_EVAL_PATHS = {}
CLEAN_EVAL_PATHS = {}

_comp_rows = 0
_comp_bucket = Counter()
_comp_corpus = Counter()
_comp_grid = Counter()
_cx_grid = Counter()

for _sp in ACTIVE_EVAL_SPLITS:
    _sub_p = os.path.join(RESULTS_DIR, f"subset_{_sp}.jsonl")
    _cln_p = os.path.join(RESULTS_DIR, f"subset_{_sp}_clean.jsonl")
    _slice_info = build_stratified_eval_subset(
        _sp,
        data_dir=DATA_DIR,
        out_path=_sub_p,
        limit=EVAL_LIMIT,
        seed=SEED,
    )
    if SINGLE_CALL_ONLY:
        with open(_sub_p, encoding="utf-8") as _fh:
            _one_call_lines = [
                _l for _l in _fh
                if _l.strip() and len((json.loads(_l).get("ground_truth_calls") or [])) == 1
            ]
        with open(_sub_p, "w", encoding="utf-8") as _fh:
            _fh.writelines(_one_call_lines)
        print(f"[SINGLE_CALL_ONLY] filtered {_sp} slice -> {len(_one_call_lines)} 1-call rows")
    print(f"=== 4.0 Stratified Benchmark Slice ({_sp}) ===")
    print("  Available by cohort:", _slice_info["available_by_cohort"])
    print("  Sampled   by cohort:", _slice_info["sampled_by_cohort"], f"(total={_slice_info['sampled_total']})")

    _step0_stats = {c: {"n": 0, "vetoed": 0} for c in ("F_func", "F_json", "A_clean", "A_adv")}
    _clean_lines = []
    with open(_sub_p, encoding="utf-8") as _fh:
        for _line in _fh:
            _raw = _line.strip()
            if not _raw:
                continue
            _d = json.loads(_raw)
            _bk = _tool_bucket(len(_d.get("tools") or []))
            _cp = _corpus_of_row(_d)
            _cx = classify_sample_complexity(_d)
            _comp_rows += 1
            _comp_bucket[_bk] += 1
            _comp_corpus[_cp] += 1
            _comp_grid[(_cp, _bk)] += 1
            _cx_grid[(_sp, _cx)] += 1

            _cohort = classify_eval_cohort(_d)
            _s = UnifiedSample.from_dict(_d)
            _st = _step0_stats.setdefault(_cohort, {"n": 0, "vetoed": 0})
            _st["n"] += 1
            _vetoed = False
            try:
                _recent = [
                    RecentRow(key=str(r.get("key", "")), value=str(r.get("value", "")), source=str(r.get("source", "recent")))
                    for r in (_s.recent_context or [])
                ]
                build_input_multi([tooldef_to_template_tool(t) for t in (_s.tools or [])], _s.utterance, recent_context=_recent)
            except SanitizeError:
                _vetoed = True
            except Exception:
                _vetoed = False
            if _vetoed:
                _st["vetoed"] += 1
            else:
                _clean_lines.append(_raw)

    with open(_cln_p, "w", encoding="utf-8") as _fh:
        for _l in _clean_lines:
            _fh.write(_l + "\n")

    print(f"=== Step 0: Pre-Encode Sanitize Gate ({_sp}) ===")
    for _c in ("F_func", "F_json", "A_clean", "A_adv"):
        _st = _step0_stats.get(_c, {"n": 0, "vetoed": 0})
        _rate = (_st["vetoed"] / _st["n"]) if _st["n"] else 0.0
        print(f"  {_c:<8}: vetoed={_st['vetoed']:>3} / {_st['n']:<3} ({_rate:.1%})")
    print(f"  Non-vetoed clean rows -> {_cln_p} ({len(_clean_lines)} rows)\n")

    SUBSET_EVAL_PATHS[_sp] = _sub_p
    CLEAN_EVAL_PATHS[_sp] = _cln_p

# Backwards-compatible aliases pointing to primary EVAL_SPLIT
SUBSET_EVAL_PATH = SUBSET_EVAL_PATHS[EVAL_SPLIT]
CLEAN_EVAL_PATH = CLEAN_EVAL_PATHS[EVAL_SPLIT]

print("=== 4.0b Evaluation-Set Composition (corpus x tool-count bucket) ===")
_comp_buckets = [b[2] for b in TOOL_BUCKET_EDGES]
print(f"{'corpus':<10}" + "".join(f"{_b:>9}" for _b in _comp_buckets) + f"{'total':>9}")
print("-" * (10 + 9 * (len(_comp_buckets) + 1)))
for _cp in ("hermes", "xlam"):
    _vals = [_comp_grid.get((_cp, _b), 0) for _b in _comp_buckets]
    print(f"{_cp:<10}" + "".join(f"{_v:>9}" for _v in _vals) + f"{_comp_corpus.get(_cp, 0):>9}")
print(f"{'total':<10}" + "".join(f"{_comp_bucket.get(_b, 0):>9}" for _b in _comp_buckets) + f"{_comp_rows:>9}")

print("\n=== 4.0c Evaluation-Set Complexity (Normal Scalar vs Complex Array/Object) ===")
print(f"{'split':<18}{'normal':>9}{'complex':>9}{'total':>9}{'complex%':>10}")
print("-" * 55)
for _sp in ACTIVE_EVAL_SPLITS:
    _n_norm = _cx_grid.get((_sp, "normal"), 0)
    _n_comp = _cx_grid.get((_sp, "complex"), 0)
    _tot = _n_norm + _n_comp
    print(f"{_sp:<18}{_n_norm:>9}{_n_comp:>9}{_tot:>9}{(_n_comp / max(1, _tot)):>9.1%}")'''),
    (MD, r'''---
### 4a. Step 1 in Isolation — Tool Selection (`H1`)

Scores `resolver.predict_tool` on native multi-tool rows (`len(s.tools) >= 2`), single-tool rows augmented with random pool distractors (`T=3`), and measures `F_func` vs `A_clean` score/margin separation.'''),
    (CODE, r'''import json, os, random, zlib
from collections import Counter, defaultdict
from spancall.data.schema import UnifiedSample
from spancall.lean.eval_p1 import classify_eval_cohort

def _pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return s[idx]

def eval_tool_prediction_step1(subset_path: str, pool_path: str, pack_k: int = 3):
    with open(subset_path, encoding="utf-8") as fh:
        raw_dicts = [json.loads(l) for l in fh if l.strip()]

    pool = {}
    if os.path.exists(pool_path):
        with open(pool_path, encoding="utf-8") as fh:
            for l in fh:
                if not l.strip():
                    continue
                for t in (json.loads(l).get("tools") or []):
                    if t.get("name") and t["name"] not in pool:
                        pool[t["name"]] = UnifiedSample.from_dict(
                            {"sample_id": "p", "utterance": "", "tools": [t], "ground_truth_calls": []}
                        ).tools[0]
    pool_names = sorted(pool)

    native_stats = {"F_func": [0, 0], "A_clean": [0, 0], "other": [0, 0]}
    by_t = defaultdict(lambda: [0, 0])
    aug_stats = [0, 0]
    probs_by_cohort = defaultdict(list)
    margins_by_cohort = defaultdict(list)
    wrong_pairs = Counter()

    prepared = []
    for d in raw_dicts:
        cohort = classify_eval_cohort(d)
        s = UnifiedSample.from_dict(d)
        if not s.ground_truth_calls or not s.tools:
            continue
        gt_name = s.ground_truth_calls[0].tool_name
        native_multi = len(s.tools) >= 2
        candidates = list(s.tools)
        if not native_multi and pool_names:
            seen = {t.name for t in candidates}
            salt = zlib.crc32(f"step1::{SEED}::{s.sample_id}".encode("utf-8"))
            rng = random.Random(salt)
            for pick in rng.sample(pool_names, min(len(pool_names), pack_k + 3)):
                if len(candidates) >= pack_k:
                    break
                if pick not in seen:
                    candidates.append(pool[pick])
                    seen.add(pick)
            rng.shuffle(candidates)
        if len(candidates) < 2:
            continue
        prepared.append((cohort, s, gt_name, native_multi, candidates))

    if hasattr(resolver, "predict_tools_batch"):
        preds_step1 = resolver.predict_tools_batch(
            [(cands, s.utterance, s.recent_context, s.current_time_iso) for _, s, _, _, cands in prepared],
            batch_size=EVAL_BATCH_SIZE, num_workers=EVAL_WORKERS,
        )
    else:
        preds_step1 = [
            resolver.predict_tool(cands, s.utterance, s.recent_context, s.current_time_iso)
            for _, s, _, _, cands in prepared
        ]

    for (cohort, s, gt_name, native_multi, candidates), pred in zip(prepared, preds_step1):
        pred_name = pred.get("tool_name")
        ok = bool(pred_name) and pred_name == gt_name

        bucket = cohort if cohort in native_stats else "other"
        if native_multi:
            native_stats[bucket][1] += 1
            native_stats[bucket][0] += int(ok)
            t_key = str(len(candidates)) if len(candidates) <= 4 else "5+"
            by_t[t_key][1] += 1
            by_t[t_key][0] += int(ok)
        else:
            aug_stats[1] += 1
            aug_stats[0] += int(ok)

        probs_by_cohort[cohort].append(float(pred.get("tool_prob") or 0.0))
        margins_by_cohort[cohort].append(float(pred.get("tool_margin") or 0.0))
        if pred_name and not ok:
            wrong_pairs[(gt_name, pred_name)] += 1

    print("=== Step 1a: Native Multi-Tool Accuracy (len(s.tools) >= 2) ===")
    tot_c = sum(v[0] for v in native_stats.values())
    tot_n = sum(v[1] for v in native_stats.values())
    print(f"  Overall native multi-tool: {tot_c / max(1, tot_n):.4f} ({tot_c}/{tot_n})")
    for bk in ("F_func", "A_clean", "other"):
        c, n = native_stats[bk]
        if n:
            print(f"    {bk:<8}: {c / n:.4f} ({c}/{n})")
    print("  By native candidate count T:", {k: f"{v[0]}/{v[1]} ({v[0]/v[1]:.3f})" for k, v in sorted(by_t.items())})

    print("\n=== Step 1b: Single-Tool + Random Pool Distractors (T=3) ===")
    if aug_stats[1]:
        print(f"  Single+random distractors: {aug_stats[0] / aug_stats[1]:.4f} ({aug_stats[0]}/{aug_stats[1]})")

    print("\n=== Step 1c: Score & Margin Separation (F_func vs A_clean) ===")
    for bk in ("F_func", "A_clean"):
        ps = probs_by_cohort.get(bk, [])
        ms = margins_by_cohort.get(bk, [])
        if ps:
            print(f"  {bk:<8} (n={len(ps):>3}) | tool_prob p10={_pct(ps,0.1):.3f} p50={_pct(ps,0.5):.3f} p90={_pct(ps,0.9):.3f} "
                  f"| margin p10={_pct(ms,0.1):.3f} p50={_pct(ms,0.5):.3f} p90={_pct(ms,0.9):.3f}")
    if wrong_pairs:
        print("  Top confusions (gt -> pred):", [(f"{g} -> {p}", cnt) for (g, p), cnt in wrong_pairs.most_common(6)])

for _sp in ACTIVE_EVAL_SPLITS:
    print("\n" + "=" * 72)
    print(f"4a. Step 1 Tool Selection (H1) — {_sp}")
    print("=" * 72)
    eval_tool_prediction_step1(CLEAN_EVAL_PATHS[_sp], os.path.join(DATA_DIR, f"{_sp}.jsonl"))'''),
    (MD, r'''---
### 4b. Step 2 in Isolation — Parameter Extraction (`H3`, Oracle Tool Given)

Scores `H3` (`resolver.try_resolve(gt_tool, ...)`) under the two-step render (`T=1`) with **Oracle Candidates** (`protect_parity_spans=True`, gates OFF) on `F_func` for both `hermes_eval` and `xlam_eval`, split into **Overall**, **Normal (scalar)**, and **Complex (`array`/`object`)**.'''),
    (CODE, r'''import json, math, os
from spancall.lean.eval_p1 import eval_param_extraction_lean
from spancall.lean.resolver import LeanResolver

def _fmt_p1(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)

_r_2b = LeanResolver(
    model=model, encoder=encoder, tokenizer=tokenizer,
    max_seq_len=cfg.model.max_seq_len, max_span_width=cfg.model.max_span_width,
    max_calls=MAX_CALLS, call_min_prob=CALL_MIN_PROB, call_min_prob_ratio=CALL_MIN_PROB_RATIO,
    min_joint_margin=0.0, collapse_guard=False, prevent_param_collapse=False,
    resolve_mode="two_step",
)

_step2b_results = {}
print("=== Step 2b: Oracle Cands (T=1, protect=T, gates=OFF, F_func) — Normal vs Complex ===")
print(f"{'Split / Slice':<34}{'n_F':>6}{'Parity':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Coll_F':>8}{'Exact_F':>9}")
print("-" * 89)

for _sp in ACTIVE_EVAL_SPLITS:
    _mf = eval_param_extraction_lean(
        CLEAN_EVAL_PATHS[_sp], resolver=_r_2b, data_dir=DATA_DIR, limit=None,
        measure_candidate_parity=True, protect_parity_spans=True,
        context_mode="single", stratum_filter="F", exclude_tools=["json_output"], quiet=True,
        batch_size=EVAL_BATCH_SIZE, num_workers=EVAL_WORKERS,
    )
    _bc = _mf.get("by_complexity")
    if not _bc:
        # Fallback when running against a pre-v3.2 spancall.lean.eval_p1 checkout:
        # partition CLEAN_EVAL_PATHS[_sp] into normal and complex files and score each.
        _bc = {}
        with open(CLEAN_EVAL_PATHS[_sp], encoding="utf-8") as _fh_c:
            _cln_rows = [json.loads(l) for l in _fh_c if l.strip()]
        for _ck in ("normal", "complex"):
            _sub_rows = [r for r in _cln_rows if classify_sample_complexity(r) == _ck]
            if not _sub_rows:
                continue
            _tmp_p = os.path.join(RESULTS_DIR, f"subset_{_sp}_clean_{_ck}.jsonl")
            with open(_tmp_p, "w", encoding="utf-8") as _fh_w:
                for _r in _sub_rows:
                    _fh_w.write(json.dumps(_r, default=str) + "\n")
            _cm = eval_param_extraction_lean(
                _tmp_p, resolver=_r_2b, data_dir=DATA_DIR, limit=None,
                measure_candidate_parity=True, protect_parity_spans=True,
                context_mode="single", stratum_filter="F", exclude_tools=["json_output"], quiet=True,
                batch_size=EVAL_BATCH_SIZE, num_workers=EVAL_WORKERS,
            )
            _bc[_ck] = {
                "stratum_f_n": _cm.get("stratum_f_n", 0),
                "total_gt_params": _cm.get("fast_path_total_gt_params", 0),
                "candidate_parity": _cm.get("candidate_parity"),
                "extraction_precision": _cm.get("extraction_precision"),
                "extraction_recall": _cm.get("extraction_recall"),
                "extraction_f1": _cm.get("extraction_f1"),
                "pointer_collapse_rate_F": _cm.get("pointer_collapse_rate_F"),
                "exact_payload_match_F": _cm.get("exact_payload_match_F"),
            }
        _mf["by_complexity"] = _bc
    _step2b_results[_sp] = _mf
    print(f"{f'{_sp} (ALL)':<34}{int(_mf['stratum_f_n']):>6}{_fmt_p1(_mf['candidate_parity']):>8}"
          f"{_fmt_p1(_mf['extraction_precision']):>8}{_fmt_p1(_mf['extraction_recall']):>8}"
          f"{_fmt_p1(_mf['extraction_f1']):>8}{_fmt_p1(_mf['pointer_collapse_rate_F']):>8}"
          f"{_fmt_p1(_mf['exact_payload_match_F']):>9}")
    for _ck, _clabel in (("normal", "Normal (scalar)"), ("complex", "Complex (arr/obj)")):
        _cm = _bc.get(_ck) or {}
        if not _cm.get("total_gt_params") and not _cm.get("stratum_f_n"):
            continue
        _cpar = _fmt_p1(_cm["candidate_parity"]) if "candidate_parity" in _cm else "—"
        _ccol = _fmt_p1(_cm["pointer_collapse_rate_F"]) if "pointer_collapse_rate_F" in _cm else "—"
        print(f"  {f'|- {_clabel}':<32}{int(_cm.get('stratum_f_n', 0)):>6}{_cpar:>8}"
              f"{_fmt_p1(_cm.get('extraction_precision')):>8}{_fmt_p1(_cm.get('extraction_recall')):>8}"
              f"{_fmt_p1(_cm.get('extraction_f1')):>8}{_ccol:>8}"
              f"{_fmt_p1(_cm.get('exact_payload_match_F')):>9}")'''),
    (MD, r'''---
### 4c. Step 3 — Composed E2E (`two_step`), Stage Waterfall & Headline Table

Runs E2E inference on `hermes_eval` and `xlam_eval` in `two_step` mode, prints the stage-by-stage waterfall funnel, and reports `B1a (regex)` vs `SpanCall Lean (two_step)` overall and split by **Normal (scalar)** vs. **Complex (`array`/`object`)**.'''),
    (CODE, r'''import json, os, shutil, subprocess, sys, time
from scripts.eval_toolcall import main as eval_toolcall_main
from typing import Any, Dict, List, Optional
from spancall.data.schema import UnifiedSample
import spancall.data.flatten as _flat
from spancall.eval.toolcall.evaluator import payload_equal
from spancall.lean.eval_p1 import (
    _parity_spans_for_call,
    classify_eval_cohort,
    predicted_call_count as _n_predicted_calls,
)

def _param_detail(pr) -> Dict[str, Any]:
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
    return {k: v for k, v in feats.items() if k not in ("recent_selected", "tool_scores_all")}

def predict_split(split_name: str, out_path: str, limit: Optional[int] = None,
                  detailed: bool = True, protect_parity_spans: bool = False,
                  resolve_mode: Optional[str] = None, src_path: Optional[str] = None) -> None:
    sub_src = os.path.join(RESULTS_DIR, f"subset_{split_name}.jsonl")
    src = src_path or (sub_src if (split_name in ACTIVE_EVAL_SPLITS and os.path.exists(sub_src)) else os.path.join(DATA_DIR, f"{split_name}.jsonl"))
    if not os.path.exists(src):
        print(f"no source {src} — skipped")
        return
    preds: List[Dict[str, Any]] = []
    subset_samples: List[Dict[str, Any]] = []
    n = 0
    parity_checked = 0
    parity_hit = 0
    collapse_hits = 0
    _supports_parity = bool(getattr(resolver, "supports_candidate_parity", False))
    _raw_items = []
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw_dict = json.loads(line)
            s = UnifiedSample.from_dict(raw_dict)
            _p_spans = None
            if _supports_parity and s.ground_truth_calls:
                _gt0 = s.ground_truth_calls[0]
                _raw_ps = _parity_spans_for_call(s, 0)
                if _raw_ps:
                    _p_spans = {f"{_gt0.tool_name}::{k}": v for k, v in _raw_ps.items()}
            _raw_items.append((raw_dict, s, _p_spans))
            if limit is not None and len(_raw_items) >= limit:
                break

    if _raw_items and hasattr(resolver, "resolve_candidates_batch") and resolve_mode in (None, "two_step"):
        _batch_in = [
            {
                "candidate_schemas": s.tools,
                "utterance": s.utterance,
                "recent_context": s.recent_context,
                "now": s.current_time_iso,
                "parity_spans": _p_spans if _supports_parity else None,
            }
            for _, s, _p_spans in _raw_items
        ]
        # WALL CLOCK, reported separately from ``latency_ms``. They answer
        # different questions: ``latency_ms`` is the AMORTISED per-sample cost
        # (its own preprocess + its share of the batched forward + heads +
        # decode), while this is the throughput the hardware actually delivered.
        # Reporting them as one number is what let the old per-row timer look
        # honest while it actually scaled with the batch size.
        _wall_t0 = time.perf_counter()
        _batch_res = resolver.resolve_candidates_batch(
            _batch_in,
            batch_size=EVAL_BATCH_SIZE,
            num_workers=EVAL_WORKERS,
            protect_parity_spans=protect_parity_spans,
        )
        _wall_s = time.perf_counter() - _wall_t0
    else:
        _wall_t0 = time.perf_counter()
        _batch_res = [
            (
                resolver.try_resolve_candidates(
                    s.tools, s.utterance, s.recent_context, s.current_time_iso,
                    parity_spans=_p_spans, protect_parity_spans=protect_parity_spans,
                    resolve_mode=resolve_mode,
                )
                if _supports_parity
                else resolver.try_resolve_candidates(s.tools, s.utterance, s.recent_context, s.current_time_iso)
            )
            for _, s, _p_spans in _raw_items
        ]
        _wall_s = time.perf_counter() - _wall_t0

    for (raw_dict, s, _p_spans), res in zip(_raw_items, _batch_res):
            _feats = getattr(res, "features", None) or {}
            parity_checked += int(_feats.get("parity_checked", 0) or 0)
            parity_hit += int(_feats.get("parity_hit", 0) or 0)
            if res.hit and _feats.get("pointer_collapsed"):
                collapse_hits += 1
            rec = {
                "sample_id": s.sample_id,
                "stratum": s.stratum.value,
                "gt_tool": (s.ground_truth_calls[0].tool_name if s.ground_truth_calls else None),
                "gt_args": (s.ground_truth_calls[0].arguments if s.ground_truth_calls else {}),
                "gt_calls": [{"tool_name": c.tool_name, "arguments": dict(c.arguments or {})}
                             for c in (s.ground_truth_calls or [])],
                "n_gt_calls": len(s.ground_truth_calls or []),
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for p in preds:
            fh.write(json.dumps(p, default=str) + "\n")
    if os.path.abspath(src) != os.path.abspath(sub_src):
        with open(sub_src, "w") as fh:
            for s_dict in subset_samples:
                fh.write(json.dumps(s_dict, default=str) + "\n")
    n_hit = sum(1 for p in preds if p["hit"])
    _par_str = f"{parity_hit}/{parity_checked} ({parity_hit / parity_checked:.3f})" if parity_checked else "n/a"
    print(f"predicted {n} rows ({resolve_mode or 'default'}) -> {out_path} (hit={n_hit}, collapse_hits={collapse_hits}, e2e_parity={_par_str})")
    # Latency vs throughput, both printed. In LATENCY, compare against the LLM's
    # per-call time; in THROUGHPUT, compare against its wall clock. ``latency_ms``
    # is the SUM of the resolver's stage timers (pre/mask/enc/heads/decode/post),
    # where ``enc_ms`` is the amortised batch-forward share, so it does not grow
    # with the batch size the way a per-row perf_counter delta would.
    _lat = [float(p.get("latency_ms") or 0.0) for p in preds]
    _lat_pos = [x for x in _lat if x > 0.0]
    if _lat_pos and n:
        _mean_ms = sum(_lat_pos) / len(_lat_pos)
        print(f"  latency: mean={_mean_ms:.1f} ms/row (amortised per-sample cost, "
              f"n_timed={len(_lat_pos)}/{n})")
        print(f"  throughput: {n / _wall_s:.1f} rows/sec wall clock "
              f"({_wall_s:.1f}s for {n} rows at batch_size={EVAL_BATCH_SIZE}, "
              f"workers={EVAL_WORKERS})")
        _stage_keys = ("pre_ms", "mask_ms", "enc_ms", "heads_ms", "decode_ms", "post_ms")
        _bad = [
            p["sample_id"] for p in preds
            if float(p.get("latency_ms") or 0.0) > 0.0
            and abs(
                float(p["latency_ms"])
                - sum(float((p.get("features") or {}).get(k, 0.0) or 0.0) for k in _stage_keys)
            ) > 1e-3
        ]
        print(f"  additivity: {n - len(_bad)}/{n} rows have latency_ms == stage_sum"
              + (f" (first mismatch: {_bad[0]})" if _bad else ""))

def _load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)

def _eval_complexity_split(split_name: str, subset_path: str, preds_path: str, sys_label: str = "lean_twostep") -> Dict[str, Any]:
    """Score Normal-only vs Complex-bearing subsets via eval_toolcall_main."""
    with open(subset_path, encoding="utf-8") as _fh:
        _raw_rows = [json.loads(l) for l in _fh if l.strip()]
    with open(preds_path, encoding="utf-8") as _fh:
        _pred_map = {p["sample_id"]: p for p in (json.loads(l) for l in _fh if l.strip())}
    _out_by_cx = {}
    for _cx in ("normal", "complex"):
        _crows = [r for r in _raw_rows if classify_sample_complexity(r) == _cx]
        if not _crows:
            continue
        _csub = os.path.join(RESULTS_DIR, f"subset_{split_name}_{_cx}.jsonl")
        _cpred = os.path.join(RESULTS_DIR, f"preds_{split_name}_{_cx}_{sys_label}.jsonl")
        with open(_csub, "w", encoding="utf-8") as _fh:
            for _r in _crows:
                _fh.write(json.dumps(_r, default=str) + "\n")
        with open(_cpred, "w", encoding="utf-8") as _fh:
            for _r in _crows:
                _p = _pred_map.get(_r["sample_id"])
                if _p is not None:
                    _fh.write(json.dumps(_p, default=str) + "\n")
        _cout = os.path.join(RESULTS_DIR, f"toolcall_metrics_{split_name}_{_cx}_{sys_label}.json")
        eval_toolcall_main(["--test_data", _csub, "--predictions", _cpred, "--output", _cout])
        _out_by_cx[_cx] = _load_metrics(_cout)
    return _out_by_cx

def _waterfall(preds_path: str, sub_path: str, label: str) -> None:
    _raw_sub = {}
    with open(sub_path) as _fh:
        for _l in _fh:
            if _l.strip():
                _d = json.loads(_l)
                _raw_sub[_d["sample_id"]] = (_d, UnifiedSample.from_dict(_d))
    with open(preds_path) as _fh:
        _ps = [json.loads(l) for l in _fh if l.strip()]
    _f  = [p for p in _ps if classify_eval_cohort(_raw_sub[p["sample_id"]][0]) == "F_func"]
    _ac = [p for p in _ps if classify_eval_cohort(_raw_sub[p["sample_id"]][0]) == "A_clean"]
    _aa = [p for p in _ps if classify_eval_cohort(_raw_sub[p["sample_id"]][0]) == "A_adv"]

    n_tot = len(_f)
    s0 = [p for p in _f if not str(p.get("abstain_reason") or "").startswith("sanitize")]
    s1 = [p for p in s0 if p.get("tool_name") == p.get("gt_tool")]
    s2a = [p for p in s1 if int((p.get("features") or {}).get("parity_checked") or 0) == int((p.get("features") or {}).get("parity_hit") or 0)]
    s2b_exact = []
    for p in s2a:
        _s = _raw_sub[p["sample_id"]][1]
        _gt_t = next((t for t in _s.tools if t.name == p["gt_tool"]), None)
        _paths = dict(getattr(_gt_t, "object_paths", None) or {})
        _gt_flat = _flat.flatten_args(p.get("gt_args") or {}, _paths)
        _pred_flat = _flat.flatten_args(p.get("payload_partial") or {}, _paths)
        if payload_equal(_gt_flat, _pred_flat, _gt_t):
            s2b_exact.append(p)
    s2c_hit = [p for p in s2b_exact if p.get("hit")]

    print(f"\n--- Waterfall Funnel [{label}] on F_func (N={n_tot}) ---")
    for _name, _cnt in [
        ("Total F_func", n_tot),
        ("-> Step 0: Sanitize OK", len(s0)),
        ("-> Step 1: Correct Tool", len(s1)),
        ("-> Step 2a: All GT Spans in C=512", len(s2a)),
        ("-> Step 2b: All Params Matched (partial)", len(s2b_exact)),
        ("-> Step 2c: Schema & Gates Hit (E2E Exact)", len(s2c_hit)),
    ]:
        print(f"  {_name:<44} {_cnt:>4} / {n_tot} ({_cnt / max(1, n_tot):>6.1%})")
    _ac_hit = sum(1 for p in _ac if p.get("hit"))
    _aa_san = sum(1 for p in _aa if str(p.get("abstain_reason") or "").startswith("sanitize"))
    print(f"  [A_clean] correct abstain: {len(_ac) - _ac_hit}/{len(_ac)} ({1.0 - _ac_hit / max(1, len(_ac)):.1%}) | "
          f"[A_adv] sanitize-vetoed: {_aa_san}/{len(_aa)} ({_aa_san / max(1, len(_aa)):.1%})")

EVAL_PREDS_PATHS = {}
_complexity_eval_metrics = {}

for _sp in ACTIVE_EVAL_SPLITS:
    _sub_sp = SUBSET_EVAL_PATHS[_sp]
    _pred_sp = os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl") if _sp == EVAL_SPLIT else os.path.join(RESULTS_DIR, f"preds_{_sp}_twostep.jsonl")
    predict_split(_sp, _pred_sp, limit=None, src_path=_sub_sp)
    if _sp != EVAL_SPLIT:
        shutil.copyfile(_pred_sp, os.path.join(RESULTS_DIR, f"preds_{_sp}.jsonl"))
    EVAL_PREDS_PATHS[_sp] = _pred_sp
    _waterfall(_pred_sp, _sub_sp, f"{_sp} / two_step")

    for _args, _out in [
        (["--baseline", "b1a"], f"toolcall_metrics_{_sp}_b1a.json"),
        (["--predictions", _pred_sp], f"toolcall_metrics_{_sp}_lean_twostep.json"),
    ]:
        eval_toolcall_main([
            "--test_data", _sub_sp,
            *_args,
            "--output", os.path.join(RESULTS_DIR, _out),
        ])
    _complexity_eval_metrics[_sp] = _eval_complexity_split(_sp, _sub_sp, _pred_sp, "lean_twostep")

_p_two = EVAL_PREDS_PATHS[EVAL_SPLIT]

print("\n=== Headline Comparison Table (hermes_eval & xlam_eval) ===")
_headers = ["split", "system", "n_F", "exact_F", "set_exact_F", "count_acc", "hit_F", "collapse_F", "param_F1_F", "false_hit_on_A", "exact_all"]
print(" | ".join(_headers))
print("-|" * len(_headers))
for _sp in ACTIVE_EVAL_SPLITS:
    for _label, _fn in [
        ("B1a (regex)", f"toolcall_metrics_{_sp}_b1a.json"),
        ("SpanCall Lean (two_step)", f"toolcall_metrics_{_sp}_lean_twostep.json"),
    ]:
        _m = _load_metrics(os.path.join(RESULTS_DIR, _fn))
        if not _m:
            continue
        _mf = _m.get("metrics_by_stratum", {}).get("F", {})
        _row = {
            "split": _sp,
            "system": _label,
            "n_F": int(_mf.get("n", 0)),
            "exact_F": round(_mf.get("exact_payload_match", 0.0), 3),
            "set_exact_F": round(_mf.get("call_exact_set_match", 0.0), 3),
            "count_acc": round(_m.get("call_count_acc", 0.0), 3),
            "hit_F": round(_mf.get("hit_rate", 0.0), 3),
            "collapse_F": round(_mf.get("collapse_hit_rate", 0.0), 3),
            "param_F1_F": round(_mf.get("param_f1", 0.0), 3),
            "false_hit_on_A": round(_m.get("false_hit_rate_on_a", 0.0), 4),
            "exact_all": round(_m.get("exact_payload_match", 0.0), 3),
        }
        print(" | ".join(str(_row[h]) for h in _headers))

print("\n=== E2E Breakdown by Complexity: Normal (Scalar) vs. Complex (Array/Object) ===")
print(f"{'split':<16}{'type':<10}{'n_F':>6}{'exact_F':>9}{'hit_F':>8}{'param_F1_F':>11}{'false_hit_A':>12}{'exact_all':>10}")
print("-" * 82)
for _sp in ACTIVE_EVAL_SPLITS:
    for _cx in ("normal", "complex"):
        _m = (_complexity_eval_metrics.get(_sp) or {}).get(_cx)
        if not _m:
            continue
        _mf = _m.get("metrics_by_stratum", {}).get("F", {})
        print(f"{_sp:<16}{_cx:<10}{int(_mf.get('n', 0)):>6}"
              f"{_mf.get('exact_payload_match', 0.0):>9.3f}"
              f"{_mf.get('hit_rate', 0.0):>8.3f}"
              f"{_mf.get('param_f1', 0.0):>11.3f}"
              f"{_m.get('false_hit_rate_on_a', 0.0):>12.4f}"
              f"{_m.get('exact_payload_match', 0.0):>10.3f}")'''),
    (MD, r'''---
### 4d. Object-Decomposition & Complexity Audit (`hermes_eval` & `xlam_eval`)

Quantifies schema decomposition coverage, **Per-Param-Type Exact Match** grouped into **Normal (scalar)** vs. **Complex (`array`/`object`/`object-leaf`)** parameters, and residual examples failing only on complex parameters.'''),
    (CODE, r'''# 4d. Object-decomposition & Normal-vs-Complex param audit across ACTIVE_EVAL_SPLITS
import json, os
from collections import Counter
from typing import Any, Dict, List

import spancall.data.flatten as _flat
from spancall.data.schema import UnifiedSample
from spancall.eval.toolcall.evaluator import _deep_equal

def _is_object_leaf(name: str, object_paths: Dict[str, List[str]]) -> bool:
    root = str(name).split(_flat.PATH_SEP, 1)[0]
    return bool(object_paths) and any(
        str(leaf) == str(name) for leaf in (object_paths.get(root) or ())
    )

def _gather_samples(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        print(f"no source {path} - skipped")
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows

for _sp in ACTIVE_EVAL_SPLITS:
    print("=" * 84)
    print(f"4d. AUDIT — {_sp}")
    print("=" * 84)
    _schema_stats = Counter()
    _seen_tools: set = set()
    _dup_tools = 0
    for _raw in _gather_samples(os.path.join(DATA_DIR, f"{_sp}.jsonl")):
        _s = UnifiedSample.from_dict(_raw)
        for _t in _s.tools:
            if _t.name in _seen_tools:
                _dup_tools += 1
                continue
            _seen_tools.add(_t.name)
            _opaque = [p.name for p in _t.parameters if (p.param_type or "").lower() == "object"]
            _leaves = [p.name for p in _t.parameters if _is_object_leaf(p.name, _t.object_paths or {})]
            if _opaque:
                _schema_stats["tools_with_opaque_object"] += 1
                _schema_stats["opaque_object_params"] += len(_opaque)
            if _leaves:
                _schema_stats["tools_with_decomposed_object"] += 1
                _schema_stats["decomposed_leaf_params"] += len(_leaves)
            if _t.object_paths:
                _schema_stats["object_roots_decomposed"] += len(_t.object_paths)

    print("=== Object Decomposition Coverage (schema level, distinct tools) ===")
    print(f"  distinct tool schemas scanned    : {len(_seen_tools)} ({_dup_tools} repeated occurrences ignored)")
    print(f"  object roots decomposed          : {_schema_stats['object_roots_decomposed']}")
    print(f"  scalar leaf params emitted       : {_schema_stats['decomposed_leaf_params']}")
    print(f"  tools with decomposed objects    : {_schema_stats['tools_with_decomposed_object']}")
    print(f"  tools still carrying an opaque object param : {_schema_stats['tools_with_opaque_object']}")
    print(f"  opaque object params (routed to slow path)  : {_schema_stats['opaque_object_params']}")

    _PRED_PATH = EVAL_PREDS_PATHS.get(_sp, "")
    _preds_by_id: Dict[str, Dict[str, Any]] = {}
    if _PRED_PATH and os.path.exists(_PRED_PATH):
        for _p in _gather_samples(_PRED_PATH):
            _preds_by_id[_p["sample_id"]] = _p

    _type_tot = Counter()
    _type_ok = Counter()
    _group_tot = Counter()
    _group_ok = Counter()
    _residual: List[Dict[str, Any]] = []

    if not _preds_by_id:
        print(f"\n[!] no predictions at {_PRED_PATH} - run 4c first. Per-type table skipped.")
        continue

    for _raw in _gather_samples(SUBSET_EVAL_PATHS[_sp]):
        _sid = _raw.get("sample_id")
        _p = _preds_by_id.get(_sid)
        if _p is None or not _p.get("gt_args"):
            continue
        _s = UnifiedSample.from_dict(_raw)
        if not _s.ground_truth_calls:
            continue
        _gt_call = _s.ground_truth_calls[0]
        _gt_tool = next((t for t in _s.tools if t.name == _gt_call.tool_name), None)
        if _gt_tool is None:
            continue
        _paths = dict(_gt_tool.object_paths or {})
        _gt_full = _flat.flatten_args(_gt_call.arguments or {}, _paths)
        _pred_raw = _p.get("payload_partial") or {}
        _pred_args = _flat.flatten_args(_pred_raw, _paths)

        _bad_names: List[str] = []
        for _param in _gt_tool.parameters:
            _name = _param.name
            if _name not in _gt_full:
                continue
            if _is_object_leaf(_name, _paths):
                _kind = "object-leaf"
            else:
                _kind = (_param.param_type or "string").lower()
            _grp = "complex" if (_kind == "object-leaf" or _kind in COMPLEX_PARAM_TYPES) else "normal"
            _type_tot[_kind] += 1
            _group_tot[_grp] += 1
            _ok = _name in _pred_args and _deep_equal(_gt_full[_name], _pred_args[_name])
            if _ok:
                _type_ok[_kind] += 1
                _group_ok[_grp] += 1
            else:
                _bad_names.append(_name)

        if _bad_names:
            _kinds = set()
            for _n in _bad_names:
                _pp = next((q for q in _gt_tool.parameters if q.name == _n), None)
                _kinds.add("object-leaf" if _is_object_leaf(_n, _paths)
                           else ((_pp.param_type or "string").lower() if _pp else "unknown"))
            if _kinds <= {"object-leaf", "array", "object"}:
                _residual.append({"sample_id": _sid, "tool": _gt_call.tool_name,
                                  "bad": _bad_names, "kinds": sorted(_kinds),
                                  "gt": _gt_full, "pred": _pred_args})

    print(f"\n=== Per-Param-Type Exact Match ({_sp}) ===")
    print(f"{'group':<10}{'param_type':<18}{'n':>6}{'exact':>9}{'rate':>8}")
    print("-" * 51)
    for _k in sorted(_type_tot, key=lambda x: -_type_tot[x]):
        _n, _ok = _type_tot[_k], _type_ok[_k]
        _grp = "complex" if (_k == "object-leaf" or _k in COMPLEX_PARAM_TYPES) else "normal"
        _rate = f"{_ok / _n:>8.3f}" if _n else f"{'n/a':>8}"
        print(f"{_grp:<10}{_k:<18}{_n:>6}{_ok:>9}{_rate}")
    print("-" * 51)
    for _grp in ("normal", "complex"):
        _gn, _gok = _group_tot[_grp], _group_ok[_grp]
        _grate = f"{_gok / _gn:>8.3f}" if _gn else f"{'n/a':>8}"
        print(f"{_grp.upper():<10}{'ALL_' + _grp.upper():<18}{_gn:>6}{_gok:>9}{_grate}")

    print(f"\n=== Residual Rows Failing Only On Complex Params ({len(_residual)} total; showing up to 3) ===")
    for _r in _residual[:3]:
        print(f"  {_r['sample_id']}  tool={_r['tool']}  kinds={_r['kinds']}")
        for _n in _r["bad"][:4]:
            print(f"    [WRONG] {_n:<26} gt={_r['gt'].get(_n)!r:<36} pred={_r['pred'].get(_n)!r}")'''),
    (MD, r'''---
### 4e. Stage-Attributed Error Forensics

Groups `F_func` misses by the exact stage where they failed (`Step 1 WRONG_TOOL`, `Step 2a PARITY_MISS`, `Step 2b POINTER_RANK_MISS`, `Step 2c GATE_OR_SCHEMA_REJECT`).'''),
    (CODE, r'''import json, os, textwrap
from collections import defaultdict
from spancall.data.schema import UnifiedSample
import spancall.data.flatten as _flat
from spancall.eval.toolcall.evaluator import payload_equal
from spancall.lean.eval_p1 import classify_eval_cohort

for _sp in ACTIVE_EVAL_SPLITS:
    PREDS_PATH = EVAL_PREDS_PATHS.get(_sp, os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl"))
    _sub_path = SUBSET_EVAL_PATHS[_sp]
    if not (os.path.exists(PREDS_PATH) and os.path.exists(_sub_path)):
        continue
    with open(PREDS_PATH) as fh:
        preds = [json.loads(l) for l in fh if l.strip()]
    subs, raw_map = {}, {}
    with open(_sub_path) as fh:
        for l in fh:
            if l.strip():
                d = json.loads(l)
                raw_map[d["sample_id"]] = d
                subs[d["sample_id"]] = UnifiedSample.from_dict(d)

    buckets = defaultdict(list)
    buckets_cx = {"normal": defaultdict(int), "complex": defaultdict(int)}
    for p in preds:
        sid = p.get("sample_id")
        if sid not in raw_map or classify_eval_cohort(raw_map[sid]) != "F_func":
            continue
        _cx = classify_sample_complexity(raw_map[sid])
        feats = p.get("features") or {}
        if str(p.get("abstain_reason") or "").startswith("sanitize"):
            _stage = "STEP0_SANITIZE"
        elif p.get("tool_name") != p.get("gt_tool"):
            _stage = "STEP1_WRONG_TOOL"
        elif int(feats.get("parity_hit") or 0) < int(feats.get("parity_checked") or 0):
            _stage = "STEP2A_PARITY_MISS"
        else:
            s = subs[sid]
            gt_t = next((t for t in s.tools if t.name == p["gt_tool"]), None)
            _paths = dict(getattr(gt_t, "object_paths", None) or {})
            _gt_flat = _flat.flatten_args(p.get("gt_args") or {}, _paths)
            _part_flat = _flat.flatten_args(p.get("payload_partial") or {}, _paths)
            partial_ok = payload_equal(_gt_flat, _part_flat, gt_t)
            if p.get("hit") and payload_equal(
                _gt_flat, _flat.flatten_args(p.get("payload") or {}, _paths), gt_t
            ):
                _stage = "EXACT_HIT"
            elif not partial_ok:
                _stage = "STEP2B_POINTER_RANK_MISS"
            else:
                _stage = "STEP2C_GATE_OR_SCHEMA_REJECT"
        buckets[_stage].append(p)
        buckets_cx[_cx][_stage] += 1

    tot_f = sum(len(v) for v in buckets.values())
    tot_norm = sum(buckets_cx["normal"].values())
    tot_comp = sum(buckets_cx["complex"].values())
    print(f"\nF_func Stage Attribution ({_sp}, N={tot_f} | normal={tot_norm}, complex={tot_comp}):")
    for k in ("EXACT_HIT", "STEP0_SANITIZE", "STEP1_WRONG_TOOL", "STEP2A_PARITY_MISS", "STEP2B_POINTER_RANK_MISS", "STEP2C_GATE_OR_SCHEMA_REJECT"):
        c = len(buckets[k])
        cn = buckets_cx["normal"][k]
        cc = buckets_cx["complex"][k]
        print(f"  {k:<30}{c:>4} ({c / max(1, tot_f):>5.1%}) | "
              f"normal={cn:>4} ({cn / max(1, tot_norm):>5.1%}) | "
              f"complex={cc:>4} ({cc / max(1, tot_comp):>5.1%})")'''),
    (MD, r'''---
### 4f. Param-Precision FP Deep Dive (diagnostic)

Dumps `F_func` rows that reached Step 2 (right tool, parity OK) but produced a wrong payload, with per-parameter `confidence` / `margin` / `source_type` / `char_span` detail. Purpose: test the hypothesis that the decoder force-assigns argmax values to params whose value is absent or ambiguous (T6 refusal off: `min_confidence` / `min_posterior` are `None`, and the greedy fallback in `decode_detailed` returns top-1 with `delta=0`).

Also sweeps per-param confidence / margin thresholds offline over the stored `parameters` detail to estimate the precision/recall tradeoff of enabling per-param refusal.'''),
    (CODE, r'''import json, os
from collections import Counter, defaultdict
from spancall.data.schema import UnifiedSample
import spancall.data.flatten as _flat
from spancall.eval.toolcall.evaluator import payload_equal
from spancall.lean.eval_p1 import classify_eval_cohort

N_DUMP = 20  # number of FP rows to print in full

# v2.7: self-contained. This cell used to close over PREDS_PATH and
# raw_map left behind by 4e, so running or re-running it alone raised
# NameError; and a stale PREDS_PATH from an earlier run silently scored
# the wrong file. Recompute both here, same precedence as 4e.
PREDS_PATH = EVAL_PREDS_PATHS.get(EVAL_SPLIT, os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl"))
if not os.path.exists(PREDS_PATH):
    raise SystemExit(f"no predictions at {PREDS_PATH} - run 4c first.")
raw_map, subs = {}, {}
if os.path.exists(SUBSET_EVAL_PATH):
    with open(SUBSET_EVAL_PATH) as _fh:
        for _l in _fh:
            if not _l.strip():
                continue
            _d = json.loads(_l)
            raw_map[_d["sample_id"]] = _d
            subs[_d["sample_id"]] = UnifiedSample.from_dict(_d)
else:
    print(f"[!] no eval subset at {SUBSET_EVAL_PATH} - cohort filter skipped.")

with open(PREDS_PATH) as fh:
    _preds = [json.loads(l) for l in fh if l.strip()]

fp_rows = []
fp_reason_counter = Counter()
conf_hist, margin_hist = [], []
delta_zero_rows = 0

for p in _preds:
    sid = p.get("sample_id")
    if sid not in raw_map or classify_eval_cohort(raw_map[sid]) != "F_func":
        continue
    feats = p.get("features") or {}
    if str(p.get("abstain_reason") or "").startswith("sanitize"):
        continue
    if p.get("tool_name") != p.get("gt_tool"):
        continue
    if int(feats.get("parity_hit") or 0) < int(feats.get("parity_checked") or 0):
        continue  # candidate-generation miss, not a precision problem
    # v2.7: pass the REAL tool schema (None skips optional-default
    # materialisation, so a correct row that simply omitted an optional
    # param was counted as a precision FP) and flatten both sides into the
    # declared object key space (nested GT vs dotted payload).
    _s = subs.get(sid)
    _gt_t = None
    if _s is not None:
        _gt_t = next((t for t in _s.tools if t.name == p.get("gt_tool")), None)
    _paths = dict(getattr(_gt_t, "object_paths", None) or {})
    _gt_flat = _flat.flatten_args(p.get("gt_args") or {}, _paths)
    p["_gt_flat"] = _gt_flat  # flat GT, reused by the refusal sweep below
    if p.get("hit") and payload_equal(
        _gt_flat, _flat.flatten_args(p.get("payload") or {}, _paths), _gt_t
    ):
        continue
    # This row passed Step 0/1/2a but failed payload equality -> precision FP.
    if abs(float(p.get("joint_margin") or 0.0)) < 1e-9:
        delta_zero_rows += 1
    params = p.get("parameters") or {}
    gt_args = p.get("_gt_flat") or {}  # flat, so dotted leaves line up
    for pn, pr in params.items():
        conf_hist.append(float(pr.get("confidence") or 0.0))
        margin_hist.append(float(pr.get("margin") or 0.0))
        pv = pr.get("value")
        if pn not in gt_args:
            fp_reason_counter["param_not_in_gt"] += 1
        elif gt_args[pn] != pv and gt_args[pn] is not None:
            fp_reason_counter["wrong_value"] += 1
        elif gt_args[pn] is None and pv is not None:
            fp_reason_counter["should_be_null"] += 1
    fp_rows.append(p)

print(f"F_func Step-2 precision-FP rows: {len(fp_rows)} / {sum(1 for p in _preds if p.get('sample_id') in raw_map and classify_eval_cohort(raw_map[p['sample_id']]) == 'F_func')}")
print(f"  joint_margin == 0 (greedy fallback / no runner-up) rows: {delta_zero_rows}")
print(f"  FP reasons: {dict(fp_reason_counter)}")
if conf_hist:
    conf_hist.sort(); margin_hist.sort()
    def _q(a, q): return a[min(len(a) - 1, int(q * len(a)))]
    print(f"  predicted-param confidence: p10={_q(conf_hist, 0.10):.3f} p50={_q(conf_hist, 0.50):.3f} p90={_q(conf_hist, 0.90):.3f}")
    print(f"  predicted-param margin:      p10={_q(margin_hist, 0.10):.3f} p50={_q(margin_hist, 0.50):.3f} p90={_q(margin_hist, 0.90):.3f}")

for p in fp_rows[:N_DUMP]:
    print("-" * 100)
    print(f"{p['sample_id']}  tool={p.get('tool_name')}  hit={p.get('hit')}  "
          f"joint_margin={float(p.get('joint_margin') or 0.0):.4f}")
    print(f"  utterance: {(raw_map[p['sample_id']].get('utterance') or '')[:220]!r}")
    print(f"  gt_args:   {json.dumps(p.get('gt_args') or {}, default=str)[:400]}")
    print(f"  payload:   {json.dumps(p.get('payload') or {}, default=str)[:400]}")
    for pn, pr in (p.get("parameters") or {}).items():
        gt_v = (p.get("_gt_flat") or {}).get(pn, "<NOT_IN_GT>")
        flag = "OK" if gt_v == pr.get("value") else ("NULL?" if gt_v is None else "WRONG")
        print(f"    [{flag:<6}] {pn:<28} pred={str(pr.get('value'))[:60]!s:<62} gt={str(gt_v)[:40]:<42} conf={float(pr.get('confidence') or 0.0):.3f} margin={float(pr.get('margin') or 0.0):.3f} src={pr.get('source_type')} span={pr.get('char_span')}")

# Offline refusal sweep: what would per-param thresholding have done?
if conf_hist or margin_hist:
    print("\n=== Offline Per-Param Refusal Sweep (stored predictions, no re-inference) ===")
    print(f"{'conf>=':>7}{'margin>=':>9}{'dropped':>9}{'dropped_OK':>12}{'dropped_bad':>12}")
    for c_thr in (0.0, 0.3, 0.5, 0.7):
        for m_thr in (0.0, 0.05, 0.1, 0.2):
            dropped = ok = bad = 0
            for p in fp_rows:
                gt_args = p.get("_gt_flat") or {}
                for pn, pr in (p.get("parameters") or {}).items():
                    if float(pr.get("confidence") or 0.0) < c_thr or float(pr.get("margin") or 0.0) < m_thr:
                        dropped += 1
                        if pn in gt_args and gt_args[pn] == pr.get("value"):
                            ok += 1  # would have removed a CORRECT param
                        else:
                            bad += 1  # would have removed a wrong/spurious param
            print(f"{c_thr:>7.2f}{m_thr:>9.2f}{dropped:>9d}{ok:>12d}{bad:>12d}")
    print("(dropped_bad >> dropped_OK at the operating point => per-param refusal (T6) would recover precision)")'''),
    (MD, r'''---
## 5. OOD / Adversarial Test Splits & Artifact Persistence

Evaluates `LeanResolver` and `B1a` on `hermes_test_ood_schema`, `hermes_test_ood_para`, and `hermes_test_adversarial`, then gets the artifacts off the VM in whichever way the runtime allows.

Two independent, both-off-by-default options, set in §0b:

- **`DRIVE_MOUNT = True`** — mirrors `data/tensorized_lean_v3_2/` and `results_lean_v3_2/` to `DRIVE_OUT` on Google Drive. Best for a long run you may resume in another session.
- **`DOWNLOAD_RESULTS = True`** — zips `results_lean_v3_2/` and hands it to the browser as **one** download. Best when you just want the files on your laptop, and the only option with no Drive mount at all. `hf_export/` is excluded because §7 already pushed it to the Hub.

Neither is the default: `files.download()` needs a live front-end, and this cell also runs under `nbconvert --execute`.'''),
    (CODE, r'''import json, os, shutil, subprocess, sys
from spancall.lean.eval_p1 import build_stratified_eval_subset

for split in TEST_SPLITS:
    if split in ACTIVE_EVAL_SPLITS:
        # Already sliced, predicted, and scored in Section 4c
        continue
    sub_test_path = os.path.join(RESULTS_DIR, f"subset_{split}.jsonl")
    build_stratified_eval_subset(
        split,
        data_dir=DATA_DIR,
        out_path=sub_test_path,
        limit=TEST_LIMIT,
        seed=SEED,
    )
    if SINGLE_CALL_ONLY:
        with open(sub_test_path, encoding="utf-8") as _fh:
            _one_call_lines = [
                _l for _l in _fh
                if _l.strip() and len((json.loads(_l).get("ground_truth_calls") or [])) == 1
            ]
        with open(sub_test_path, "w", encoding="utf-8") as _fh:
            _fh.writelines(_one_call_lines)
    predict_split(
        split,
        os.path.join(RESULTS_DIR, f"preds_{split}.jsonl"),
        limit=None,
        src_path=sub_test_path,
    )
    data_file = sub_test_path
    for label, extra in [("b1a", ["--baseline", "b1a"]), ("lean", ["--predictions", os.path.join(RESULTS_DIR, f"preds_{split}.jsonl")])]:
        eval_toolcall_main([
            "--test_data", data_file,
            *extra,
            "--output", os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_{label}.json"),
        ])

_all_eval_and_test = list(dict.fromkeys(list(TEST_SPLITS) + [EVAL_SPLIT]))

print(f"\n{'split':<26}{'system':<12}{'n_F':>6}{'exact_F':>9}{'hit_F':>8}{'param_F1':>9}{'exact_all':>10}")
print("-" * 80)
for split in _all_eval_and_test:
    lean_lbl = "lean_twostep" if split in ACTIVE_EVAL_SPLITS else "lean"
    for label in ("b1a", lean_lbl):
        m = _load_metrics(os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_{label}.json"))
        if not m:
            continue
        m_f = m.get("metrics_by_stratum", {}).get("F", {})
        print(f"{split:<26}{label:<12}{int(m_f.get('n', 0)):>6}"
              f"{m_f.get('exact_payload_match', 0.0):>9.3f}{m_f.get('hit_rate', 0.0):>8.3f}"
              f"{m_f.get('param_f1', 0.0):>9.3f}{m.get('exact_payload_match', 0.0):>10.3f}")

# Normal vs. Complex breakdown across all test + eval splits
print(f"\n=== Test & Eval Splits: Normal (Scalar) vs. Complex (Array/Object) ===")
print(f"{'split':<26}{'type':<10}{'n_F':>6}{'exact_F':>9}{'hit_F':>8}{'param_F1':>9}{'exact_all':>10}")
print("-" * 78)
for split in _all_eval_and_test:
    if split in _complexity_eval_metrics:
        _cx_map = _complexity_eval_metrics[split]
    else:
        _sub_p = os.path.join(RESULTS_DIR, f"subset_{split}.jsonl")
        _pred_p = os.path.join(RESULTS_DIR, f"preds_{split}.jsonl")
        if not (os.path.exists(_sub_p) and os.path.exists(_pred_p)):
            continue
        _cx_map = _eval_complexity_split(split, _sub_p, _pred_p, "lean")
    for _cx in ("normal", "complex"):
        _m = _cx_map.get(_cx)
        if not _m:
            continue
        _mf = _m.get("metrics_by_stratum", {}).get("F", {})
        print(f"{split:<26}{_cx:<10}{int(_mf.get('n', 0)):>6}"
              f"{_mf.get('exact_payload_match', 0.0):>9.3f}{_mf.get('hit_rate', 0.0):>8.3f}"
              f"{_mf.get('param_f1', 0.0):>9.3f}{_m.get('exact_payload_match', 0.0):>10.3f}")

if DRIVE_MOUNT:
    os.makedirs(DRIVE_OUT, exist_ok=True)
    for name, src in [("data_tensorized_lean_v3_2", TENSOR_DIR), ("results_lean_v3_2", RESULTS_DIR)]:
        if os.path.isdir(src):
            dst = os.path.join(DRIVE_OUT, name)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"mirrored {src} -> {dst}")

# --------------------------------------------------------------------------- #
# Optional browser download of `results_lean_v3_2/`.
#
# This is the notebook that PRODUCES the predictions the LLM-baseline notebook
# consumes, so the artifact has to be able to leave the VM. Drive is one way
# (above); this is the other, and it is the only one that works on a runtime
# with no Drive mount.
#
# Three constraints, all learned from `files.download()`'s actual behaviour:
#  * ONE zip, one `download()` call. Chrome blocks the second automatic
#    download, so a per-file loop silently delivers only the first file.
#  * Absolute `RESULTS_DIR` only. Everything here runs after `os.chdir`, so a
#    relative path would resolve inside the git clone instead.
#  * `hf_export/` is excluded. It is already pushed to the Hub by section 7,
#    so re-shipping a multi-hundred-MB state_dict through a browser adds
#    nothing but a failed download.
#
# `google.colab` is imported under a guard: it does not exist locally, under
# `nbconvert --execute`, or in CI, and a bare import would raise there.
# --------------------------------------------------------------------------- #
if DOWNLOAD_RESULTS:
    try:
        from google.colab import files as _colab_files
        _in_colab = True
    except ImportError:
        _colab_files = None
        _in_colab = False
    if not _in_colab:
        print(f"[download] DOWNLOAD_RESULTS=True but this is not Colab -> ")
        print(f"[download] artifacts stay at {RESULTS_DIR}")
    else:
        import zipfile as _zipfile
        _zip_path = os.path.join(os.path.dirname(RESULTS_DIR), "results_lean_v3_2.zip")
        if os.path.exists(_zip_path):
            os.remove(_zip_path)          # never append to a previous archive
        _n_files = 0
        with _zipfile.ZipFile(_zip_path, "w", _zipfile.ZIP_DEFLATED) as _zf:
            for _root, _dirs, _files in os.walk(RESULTS_DIR):
                if os.path.basename(_root) == "hf_export":
                    _dirs[:] = []          # already on the Hub; do not re-ship it
                    continue
                for _f in _files:
                    _src = os.path.join(_root, _f)
                    _zf.write(_src, os.path.relpath(_src, RESULTS_DIR))
                    _n_files += 1
        print(f"[download] {_zip_path}  {os.path.getsize(_zip_path) / 1e6:.2f} MB  "
              f"({_n_files} file(s), hf_export/ excluded)")
        _colab_files.download(_zip_path)    # exactly ONE browser download
        print(f"[download] handed to the browser. Local copy kept at {_zip_path}")'''),
    (MD, r'''---
### 5b. Per-Corpus Evaluation (Hermes vs xLAM)

Splits every evaluation subset by `source_dataset` (Hermes vs xLAM) and reruns `scripts/eval_toolcall.py` per corpus, so Hermes-only and xLAM-only metrics are reported separately instead of blended. Runs after all subsets and predictions exist (eval split + all test splits).'''),
    (CODE, r'''import json, os, subprocess, sys

def _corpus_of(row):
    sd = str(row.get("source_dataset") or "")
    if sd.startswith("xlam"):
        return "xlam"
    if sd.startswith("hermes"):
        return "hermes"
    # Fallback: sample_id prefix ("xlam::..." vs hermes-style ids)
    return "xlam" if str(row.get("sample_id") or "").startswith("xlam") else "hermes"

_corpus_eval_pairs = []  # (split, corpus, label, metrics_path)

for split in TEST_SPLITS + [EVAL_SPLIT]:
    sub_path = SUBSET_EVAL_PATH if (split == EVAL_SPLIT and os.path.exists(SUBSET_EVAL_PATH)) \
        else os.path.join(RESULTS_DIR, f"subset_{split}.jsonl")
    if not os.path.exists(sub_path):
        continue
    with open(sub_path, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    by_corpus = {"hermes": [], "xlam": []}
    for r in rows:
        by_corpus[_corpus_of(r)].append(r)
    if split == EVAL_SPLIT:
        # Lean v3: exactly one Lean arm, labelled `lean_twostep` so a
        # stale `_lean.json` from a pre-v3 run can never be picked up.
        _pred_specs = [
            ("b1a", None),
            ("lean_twostep", os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")),
        ]
    else:
        _pred_specs = [("b1a", None), ("lean", os.path.join(RESULTS_DIR, f"preds_{split}.jsonl"))]
    for corpus, crows in by_corpus.items():
        if not crows:
            continue
        csub = os.path.join(RESULTS_DIR, f"subset_{split}_{corpus}.jsonl")
        with open(csub, "w", encoding="utf-8") as fh:
            for r in crows:
                fh.write(json.dumps(r, default=str) + "\n")
        ids = {r["sample_id"] for r in crows}
        for label, pred_path in _pred_specs:
            cpred = None
            if pred_path is not None:
                if not os.path.exists(pred_path):
                    continue
                with open(pred_path, encoding="utf-8") as fh:
                    _pred_rows = [json.loads(l) for l in fh if l.strip()]
                _keep = [p for p in _pred_rows if p.get("sample_id") in ids]
                if not _keep:
                    continue
                cpred = os.path.join(RESULTS_DIR, f"preds_{split}_{corpus}_{label}.jsonl")
                with open(cpred, "w", encoding="utf-8") as fh:
                    for p in _keep:
                        fh.write(json.dumps(p, default=str) + "\n")
            out_json = os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_{corpus}_{label}.json")
            _cmd = ["--test_data", csub]
            if label == "b1a":
                _cmd += ["--baseline", "b1a"]
            else:
                _cmd += ["--predictions", cpred]
            _cmd += ["--output", out_json]
            eval_toolcall_main(_cmd)
            _corpus_eval_pairs.append((split, corpus, label, out_json))

print(f"\n=== Per-Corpus Evaluation (Hermes vs xLAM) ===")
print(f"{'split':<26}{'corpus':<9}{'system':<14}{'n_F':>6}{'exact_F':>9}{'hit_F':>8}{'param_F1':>9}{'exact_all':>10}")
print("-" * 90)
for split, corpus, label, out_json in _corpus_eval_pairs:
    m = _load_metrics(out_json)
    if not m:
        continue
    m_f = m.get("metrics_by_stratum", {}).get("F", {})
    print(f"{split:<26}{corpus:<9}{label:<14}{int(m_f.get('n', 0)):>6}"
          f"{m_f.get('exact_payload_match', 0.0):>9.3f}{m_f.get('hit_rate', 0.0):>8.3f}"
          f"{m_f.get('param_f1', 0.0):>9.3f}{m.get('exact_payload_match', 0.0):>10.3f}")'''),
    (MD, r'''---
### 5c. Tool-Count Bucket Pivot (Hermes vs xLAM)

Re-runs `scripts/eval_toolcall.py` per **tool-count bucket** (`1-2`, `3-5`, `6-10`, `11+`) on `hermes_eval` and `xlam_eval`, then prints one pivot.

The bucket is keyed on `len(sample.tools)` — the distractor-pool size H1 must actually choose from — **not** on the number of ground-truth calls. That distinction matters: the call count moves when `SINGLE_CALL_ONLY` filtering is applied, which would make the pivot's row definitions depend on a filter rather than on the row. Bucketing on `len(tools)` keeps each bucket well defined no matter what ran before it.'''),
    (CODE, r'''# 5c. Tool-count bucket pivot (per split, per bucket, both systems)
import json, os, subprocess, sys
from collections import defaultdict
from scripts.eval_toolcall import main as eval_toolcall_main

# Default scope is the two slices that each carry ONE corpus, so the pivot's
# rows are genuinely (corpus x bucket) instead of a blended slice re-filtered
# after the fact. Flip PIVOT_ALL_SPLITS to cover every test split at roughly
# four times the subprocess count.
PIVOT_ALL_SPLITS = False
PIVOT_SPLITS = (TEST_SPLITS + [EVAL_SPLIT]) if PIVOT_ALL_SPLITS else [EVAL_SPLIT, "xlam_eval"]
BUCKETS = [b[2] for b in TOOL_BUCKET_EDGES]

_pivot = []
for _split in PIVOT_SPLITS:
    _sub = SUBSET_EVAL_PATH if (_split == EVAL_SPLIT and os.path.exists(SUBSET_EVAL_PATH)) \
        else os.path.join(RESULTS_DIR, f"subset_{_split}.jsonl")
    if not os.path.exists(_sub):
        print(f"[5c] no subset for {_split} - skipped (run section 4 / 5 first)")
        continue
    with open(_sub, encoding="utf-8") as _fh:
        _rows = [json.loads(l) for l in _fh if l.strip()]
    _by_bucket = defaultdict(list)
    for _r in _rows:
        _by_bucket[_tool_bucket(len(_r.get("tools") or []))].append(_r)
    if _split == EVAL_SPLIT:
        _pred_specs = [("b1a", None),
                       ("lean", os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl"))]
    else:
        _pred_specs = [("b1a", None),
                       ("lean", os.path.join(RESULTS_DIR, f"preds_{_split}.jsonl"))]
    for _bucket in BUCKETS:
        _brows = _by_bucket.get(_bucket) or []
        if not _brows:
            continue
        _bsub = os.path.join(RESULTS_DIR, f"subset_{_split}_{_bucket}.jsonl")
        with open(_bsub, "w", encoding="utf-8") as _fh:
            for _r in _brows:
                _fh.write(json.dumps(_r, default=str) + "\n")
        _ids = {_r["sample_id"] for _r in _brows}
        for _label, _ppath in _pred_specs:
            _bp = None
            if _ppath is not None:
                if not os.path.exists(_ppath):
                    continue
                with open(_ppath, encoding="utf-8") as _fh:
                    _pr = [json.loads(l) for l in _fh if l.strip()]
                _keep = [p for p in _pr if p.get("sample_id") in _ids]
                if not _keep:
                    continue
                _bp = os.path.join(RESULTS_DIR, f"preds_{_split}_{_bucket}_{_label}.jsonl")
                with open(_bp, "w", encoding="utf-8") as _fh:
                    for _p in _keep:
                        _fh.write(json.dumps(_p, default=str) + "\n")
            _out = os.path.join(RESULTS_DIR, f"toolcall_metrics_{_split}_{_bucket}_{_label}.json")
            _argv = ["--test_data", _bsub]
            _argv += ["--baseline", "b1a"] if _label == "b1a" else ["--predictions", _bp]
            _argv += ["--output", _out]
            eval_toolcall_main(_argv)
            _pivot.append((_split, _corpus_of_row(_brows[0]), _bucket, _label, _out))

print("\n=== 5c. exact_F by tool-count bucket (corpus x bucket) ===")
print(f"{'split':<26}{'corpus':<9}{'bucket':<8}{'system':<7}{'n_F':>6}{'exact_F':>9}"
      f"{'hit_F':>8}{'param_F1':>9}")
print("-" * 84)
for _split, _corpus, _bucket, _label, _out in _pivot:
    _m = _load_metrics(_out)
    if not _m:
        continue
    _mf = _m.get("metrics_by_stratum", {}).get("F", {})
    print(f"{_split:<26}{_corpus:<9}{_bucket:<8}{_label:<7}{int(_mf.get('n', 0)):>6}"
          f"{_mf.get('exact_payload_match', 0.0):>9.3f}{_mf.get('hit_rate', 0.0):>8.3f}"
          f"{_mf.get('param_f1', 0.0):>9.3f}")
if not _pivot:
    print("  no bucket rows - no subset was available.")'''),
    (MD, r'''---
## 6. Final Run Summary Report

Aggregates the configuration, tensorized dataset counts, best training epoch, `hermes_eval` headline metrics, and all OOD/cross-corpus test results into a single summary printed to stdout and saved as `summary_report.json` / `summary_report.md`.'''),
    (CODE, r'''import json, os

def _count_jsonl(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())

def _safe_round(v, ndigits: int = 4):
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None

_t2_n = _count_jsonl(os.path.join(TENSOR_DIR, "t2.jsonl"))
_dev_n = _count_jsonl(os.path.join(TENSOR_DIR, "dev.jsonl"))

_best_epoch = None
_all_epochs = []
_train_log_file = os.path.join(exp_root, "train_log.jsonl")
if os.path.exists(_train_log_file):
    with open(_train_log_file, encoding="utf-8") as _fh:
        _recs = [json.loads(l) for l in _fh if l.strip()]
    _evals = [r for r in _recs if r.get("event") == "epoch_eval"]
    _ends = {r.get("epoch"): r for r in _recs if r.get("event") == "epoch_end"}
    for _r in _evals:
        _ep = _r.get("epoch")
        _end = _ends.get(_ep, {})
        _entry = {
            "epoch": _ep,
            "span_acc": _safe_round(_r.get("span_acc")),
            "exact_params_only": _safe_round(_r.get("exact_params_only")),
            "exact_call": _safe_round(_r.get("exact_call")),
            "tool_acc": _safe_round(_r.get("tool_acc")),
            "eval_loss": _safe_round(_r.get("eval_loss")),
            "train_loss": _safe_round(_end.get("mean_loss")),
        }
        _all_epochs.append(_entry)
    if _all_epochs:
        _best_epoch = max(
            _all_epochs,
            key=lambda e: (e["exact_call"] if e["exact_call"] is not None else -1.0),
        )

_headline = {}
for _label, _fn in [
    ("B1a (regex)", f"toolcall_metrics_{EVAL_SPLIT}_b1a.json"),
    ("SpanCall Lean (two_step)", f"toolcall_metrics_{EVAL_SPLIT}_lean_twostep.json"),
]:
    _m = _load_metrics(os.path.join(RESULTS_DIR, _fn))
    if not _m:
        continue
    _mf = _m.get("metrics_by_stratum", {}).get("F", {})
    _headline[_label] = {
        "n_F": int(_mf.get("n", 0)),
        "exact_F": _safe_round(_mf.get("exact_payload_match"), 3),
        "hit_F": _safe_round(_mf.get("hit_rate"), 3),
        "collapse_F": _safe_round(_mf.get("collapse_hit_rate"), 3),
        "param_F1_F": _safe_round(_mf.get("param_f1"), 3),
        "false_hit_on_A": _safe_round(_m.get("false_hit_rate_on_a"), 4),
        "exact_all": _safe_round(_m.get("exact_payload_match"), 3),
    }

_split_results = {}
for _sp in TEST_SPLITS + [EVAL_SPLIT]:
    _split_results[_sp] = {}
    # Lean v3: the eval split's only Lean artifact is `_lean_twostep`, so
    # reading `_lean` there would silently drop the eval split from the
    # summary. Test splits still use the `_lean` label written by section 5.
    _labels = ("b1a", "lean_twostep") if _sp in ACTIVE_EVAL_SPLITS else ("b1a", "lean")
    for _lbl in _labels:
        _m = _load_metrics(os.path.join(RESULTS_DIR, f"toolcall_metrics_{_sp}_{_lbl}.json"))
        if not _m:
            continue
        _mf = _m.get("metrics_by_stratum", {}).get("F", {})
        _split_results[_sp][_lbl] = {
            "n_F": int(_mf.get("n", 0)),
            "exact_F": _safe_round(_mf.get("exact_payload_match"), 3),
            "hit_F": _safe_round(_mf.get("hit_rate"), 3),
            "param_F1": _safe_round(_mf.get("param_f1"), 3),
            "false_hit_on_A": _safe_round(_m.get("false_hit_rate_on_a"), 4),
            "exact_all": _safe_round(_m.get("exact_payload_match"), 3),
        }

_summary = {
    "config": {
        "experiment_name": EXPERIMENT_NAME,
        "backbone": BACKBONE,
        "seed": SEED,
        "single_call_only": SINGLE_CALL_ONLY,
        "xlam_train_rows": XLAM_TRAIN_ROWS,
        "xlam_dev_rows": XLAM_DEV_ROWS,
        "max_calls": MAX_CALLS,
        "min_joint_margin": MIN_JOINT_MARGIN,
        "checkpoint_dir": CHECKPOINT_DIR,
    },
    "tensorized_rows": {"t2": _t2_n, "dev": _dev_n},
    "best_epoch": _best_epoch,
    "hermes_eval_headline": _headline,
    "complexity_eval": {
        _sp: {
            _cx: {
                "n_F": int((_cm.get("metrics_by_stratum", {}).get("F", {})).get("n", 0)),
                "exact_F": _safe_round((_cm.get("metrics_by_stratum", {}).get("F", {})).get("exact_payload_match"), 3),
                "hit_F": _safe_round((_cm.get("metrics_by_stratum", {}).get("F", {})).get("hit_rate"), 3),
                "param_F1": _safe_round((_cm.get("metrics_by_stratum", {}).get("F", {})).get("param_f1"), 3),
            }
            for _cx, _cm in (_cx_map or {}).items() if _cm
        }
        for _sp, _cx_map in (_complexity_eval_metrics or {}).items()
    },
    "split_results": _split_results,
}

_summary_json = os.path.join(RESULTS_DIR, "summary_report.json")
with open(_summary_json, "w", encoding="utf-8") as _fh:
    json.dump(_summary, _fh, indent=2)

_md_lines = [
    f"# SpanCall Lean v3.2 Run Summary ({EXPERIMENT_NAME})",
    "",
    "## Configuration & Data",
    f"- **Backbone**: `{BACKBONE}` | **Seed**: `{SEED}` | **Single-Call Only**: `{SINGLE_CALL_ONLY}`",
    f"- **xLAM Mix Caps**: `XLAM_TRAIN_ROWS={XLAM_TRAIN_ROWS}`, `XLAM_DEV_ROWS={XLAM_DEV_ROWS}`",
    f"- **Tensorized Rows**: `t2={_t2_n}`, `dev={_dev_n}`",
    f"- **Checkpoint**: `{CHECKPOINT_DIR}`",
    "",
    "## Best Dev Epoch",
]
if _best_epoch:
    _md_lines.append(
        f"- Epoch `{_best_epoch['epoch']}`: `exact_call={_best_epoch['exact_call']}`, "
        f"`tool_acc={_best_epoch['tool_acc']}`, `span_acc={_best_epoch['span_acc']}`, "
        f"`exact_params_only={_best_epoch['exact_params_only']}`, `eval_loss={_best_epoch['eval_loss']}`"
    )
else:
    _md_lines.append("- No training log found.")

_md_lines.extend([
    "",
    f"## Headline Comparison ({EVAL_SPLIT})",
    "| System | n_F | exact_F | hit_F | collapse_F | param_F1_F | false_hit_on_A | exact_all |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
])
for _sys_name, _vals in _headline.items():
    _md_lines.append(
        f"| {_sys_name} | {_vals['n_F']} | {_vals['exact_F']} | {_vals['hit_F']} | "
        f"{_vals['collapse_F']} | {_vals['param_F1_F']} | {_vals['false_hit_on_A']} | {_vals['exact_all']} |"
    )

_md_lines.extend([
    "",
    "## All Test Splits (two_step Lean vs B1a)",
    "| Split | System | n_F | exact_F | hit_F | param_F1 | false_hit_on_A | exact_all |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
])
for _sp, _by_sys in _split_results.items():
    for _lbl, _vals in _by_sys.items():
        _md_lines.append(
            f"| {_sp} | {_lbl} | {_vals['n_F']} | {_vals['exact_F']} | {_vals['hit_F']} | "
            f"{_vals['param_F1']} | {_vals['false_hit_on_A']} | {_vals['exact_all']} |"
        )

if _complexity_eval_metrics:
    _md_lines.extend([
        "",
        "## Complexity Breakdown (Normal Scalar vs. Complex Array/Object)",
        "| Split | Type | n_F | exact_F | hit_F | param_F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for _sp, _cx_map in _complexity_eval_metrics.items():
        for _cx in ("normal", "complex"):
            _cm = (_cx_map or {}).get(_cx)
            if not _cm:
                continue
            _cmf = _cm.get("metrics_by_stratum", {}).get("F", {})
            _md_lines.append(
                f"| {_sp} | {_cx} | {int(_cmf.get('n', 0))} | "
                f"{_safe_round(_cmf.get('exact_payload_match'), 3)} | "
                f"{_safe_round(_cmf.get('hit_rate'), 3)} | "
                f"{_safe_round(_cmf.get('param_f1'), 3)} |"
            )

_summary_md_text = "\n".join(_md_lines) + "\n"
_summary_md = os.path.join(RESULTS_DIR, "summary_report.md")
with open(_summary_md, "w", encoding="utf-8") as _fh:
    _fh.write(_summary_md_text)

print(_summary_md_text)
print(f"Saved summary -> {_summary_json} & {_summary_md}")'''),
    (MD, r'''---
## 7. Hugging Face Export & (Guarded) Push

Writes a self-describing export directory and uploads it **only when an HF token is available**.

- `model.pt` — `state_dict` of the trained `LeanSpanCallModel`. Note it is a plain `nn.Module` with **no `save_pretrained()`**, so a `state_dict` plus the rebuild recipe is the honest artifact; emitting a transformers-shaped `config.json` would misdescribe the architecture.
- tokenizer files, `resolve_config.json` (backbone, `max_seq_len`, `max_span_width`, `max_calls`, gate thresholds, `resolve_mode="two_step"`, template stamp) and `provenance.json`.
- `README.md` — an auto-generated model card carrying this run's headline metrics.

Without a token the export is still written locally and the push is skipped, so this cell is a no-op on CI and on local smoke runs.'''),
    (CODE, r'''# 7. HF export + guarded push (model card generated from this run)
import json, os, shutil
import torch

HF_EXPORT_DIR = os.path.join(RESULTS_DIR, "hf_export")
shutil.rmtree(HF_EXPORT_DIR, ignore_errors=True)
os.makedirs(HF_EXPORT_DIR, exist_ok=True)

# LeanSpanCallModel is a plain nn.Module with NO save_pretrained(). The export
# is therefore a state_dict plus everything a consumer needs to rebuild the SAME
# resolver: tokenizer files, resolve_config.json, provenance. Writing a
# transformers-style config.json would be a lie about the architecture.
# Parameters are moved to CPU so the artifact is device-independent.
torch.save(
    {k: v.detach().cpu() for k, v in model.state_dict().items()},
    os.path.join(HF_EXPORT_DIR, "model.pt"),
)
tokenizer.save_pretrained(HF_EXPORT_DIR)

_resolve_cfg = {
    "kind": "spancall_lean",
    "version": "v3.2",
    "flow": "lean",
    "backbone": BACKBONE,
    "max_seq_len": int(MAX_SEQ_LEN),
    "max_span_width": int(MAX_SPAN_WIDTH),
    # ``head_dim`` is part of the architecture, not a serving preference: a
    # consumer that rebuilds LeanSpanCallModel without it gets the 256 default
    # and a state_dict that loads with ``strict=False`` while silently leaving
    # the pointer head mis-dimensioned. Export it so the rebuild is exact.
    "head_dim": int(cfg.model.head_dim),
    "max_calls": int(MAX_CALLS),
    "call_min_prob": float(CALL_MIN_PROB),
    "call_min_prob_ratio": float(CALL_MIN_PROB_RATIO),
    "min_joint_margin": float(MIN_JOINT_MARGIN),
    # v3: the single extraction mode, written into the artifact so a consumer
    # never has to infer it. LeanResolver's own default is two_step as of v3,
    # but the export is what a third-party reload actually reads, so the mode
    # is recorded here rather than left to a default that could be revised.
    "resolve_mode": "two_step",
    "collapse_guard": True,
    "prevent_param_collapse": True,
    "single_call_only": bool(SINGLE_CALL_ONLY),
    "marker_tokens": list(SPECIAL_MARKER_TOKENS),
    "experiment_name": EXPERIMENT_NAME,
    "seed": int(SEED),
}
_stamp_path = os.path.join(DATA_DIR, "template_stamp.json")
if os.path.exists(_stamp_path):
    with open(_stamp_path, encoding="utf-8") as _fh:
        _resolve_cfg["template_stamp"] = json.load(_fh)
with open(os.path.join(HF_EXPORT_DIR, "resolve_config.json"), "w", encoding="utf-8") as _fh:
    json.dump(_resolve_cfg, _fh, indent=2, default=str)

_prov = {
    "checkpoint_dir": CHECKPOINT_DIR,
    "best_epoch": (_best_epoch or {}).get("epoch") if isinstance(_best_epoch, dict) else None,
    "eval_split": EVAL_SPLIT,
    "eval_limit": int(EVAL_LIMIT),
    "test_limit": int(TEST_LIMIT),
    "probe_split": PROBE_SPLIT,
    "probe_every_steps": int(PROBE_EVERY_STEPS),
    "probe_limit": int(PROBE_LIMIT),
    "headline": _headline,
    "split_results": _split_results,
}
_prov_src = os.path.join(RESULTS_DIR, "summary_report.json")
if os.path.exists(_prov_src):
    shutil.copyfile(_prov_src, os.path.join(HF_EXPORT_DIR, "summary_report.json"))
with open(os.path.join(HF_EXPORT_DIR, "provenance.json"), "w", encoding="utf-8") as _fh:
    json.dump(_prov, _fh, indent=2, default=str)

_card = [
    "---",
    "tags: [spancall, tool-calling, speculative-decoding]",
    "---",
    "",
    f"# SpanCall Lean v3.2 — `{EXPERIMENT_NAME}`",
    "",
    "Bidirectional-encoder tool-call decoder (ModernBERT-large + H1 tool head + H3 "
    "pointer head) that emits a complete tool call in one forward pass instead of "
    "generating JSON autoregressively.",
    "",
    f"- **Backbone**: `{BACKBONE}`  |  **max_seq_len**: `{MAX_SEQ_LEN}`",
    "- **Extraction mode**: `two_step` (the only mode in v3)",
    f"- **Single-call only**: `{SINGLE_CALL_ONLY}`  |  **min_joint_margin**: `{MIN_JOINT_MARGIN}`",
    "",
    "## How to load",
    "",
    "```python",
    "from huggingface_hub import hf_hub_download",
    "from transformers import AutoTokenizer",
    "",
    f'tok = AutoTokenizer.from_pretrained("{HF_REPO_ID}")',
    f'cfg = json.load(open(hf_hub_download("{HF_REPO_ID}", "resolve_config.json")))',
    f"weights = hf_hub_download('{HF_REPO_ID}', 'model.pt')",
    "# rebuild SpanCallTransformerEncoder + LeanSpanCallModel exactly as in the",
    "# notebook, load_state_dict(torch.load(weights)), then:",
    "#   LeanResolver(..., resolve_mode=cfg['resolve_mode'])",
    "```",
    "",
    f"## Headline metrics (`{EVAL_SPLIT}`, limit={EVAL_LIMIT})",
    "",
    "| System | n_F | exact_F | hit_F | param_F1_F | exact_all |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for _sys, _v in (_headline or {}).items():
    _card.append(
        f"| {_sys} | {_v.get('n_F')} | {_v.get('exact_F')} | {_v.get('hit_F')} | "
        f"{_v.get('param_F1_F')} | {_v.get('exact_all')} |"
    )
if not (_headline or {}):
    _card.append("| (no metrics - run sections 4-6 first) | | | | | |")
_card += ["", "_Auto-generated by `notebooks/spancall_lean_v3_2_train_eval.ipynb`._", ""]
with open(os.path.join(HF_EXPORT_DIR, "README.md"), "w", encoding="utf-8") as _fh:
    _fh.write("\n".join(_card))

print(f"[hf] export ready -> {HF_EXPORT_DIR}")
for _fname in sorted(os.listdir(HF_EXPORT_DIR)):
    print(f"     {_fname}")

# Guarded push: token comes from interactive input (HF_TOKEN) or os.environ.
_hf_token = HF_TOKEN or os.environ.get("HF_TOKEN")
if not _hf_token:
    print("[hf] no HF token (HUGGING_FACE_TOKEN / HF_TOKEN) -> push SKIPPED; export is local only.")
elif not HF_REPO_ID:
    print("[hf] HF_REPO_ID is empty -> push SKIPPED.")
else:
    from huggingface_hub import HfApi

    _api = HfApi(token=_hf_token)
    _api.create_repo(
        repo_id=HF_REPO_ID, repo_type="model", private=HF_PUSH_PRIVATE, exist_ok=True
    )
    _api.upload_folder(
        repo_id=HF_REPO_ID,
        repo_type="model",
        folder_path=HF_EXPORT_DIR,
        commit_message=f"SpanCall Lean v3.2 ({EXPERIMENT_NAME}, seed={SEED})",
    )
    print(f"[hf] pushed -> https://huggingface.co/{HF_REPO_ID}")'''),
]


def _guard() -> Any:
    """Load ``_guard_code_cells`` from the v2.6 builder (the shared guard)."""
    v26 = HERE / "build_lean_v2_6_colab_notebook.py"
    src = v26.read_text(encoding="utf-8")
    start = src.index("def _cell_source")
    end = src.rindex("_guard_code_cells(cells)")
    ns: Dict[str, Any] = {"__name__": "build_lean_v3_2_guard"}
    exec(compile(src[start:end], str(v26), "exec"), ns)
    return ns["_guard_code_cells"]


def _executed_records(out_path: Path) -> Dict[str, Dict[str, Any]]:
    """Capture executed code cells already on disk, keyed by index *and* content.

    This notebook is the record of the run whose numbers the paper quotes: it
    ships with 19 executed code cells. A rebuild that silently dropped those
    outputs would delete the evidence, so each cell's record is re-attached when
    its text has not changed. Keying on content as well as index means an edited
    cell never inherits a stale output from a different cell.
    """
    if not out_path.exists():
        return {}
    disk = json.loads(out_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for idx, cell in enumerate(disk.get("cells", [])):
        if cell.get("cell_type") != CODE:
            continue
        if not cell.get("outputs") and cell.get("execution_count") is None:
            continue
        key = f"{idx}:{''.join(cell.get('source', [])).rstrip(chr(10))}"
        out[key] = cell
    return out


def _cell_objects(
    executed: Dict[str, Dict[str, Any]] | None = None,
    existing: List[Dict[str, Any]] | None = None,
) -> List[Any]:
    executed = executed or {}
    existing = existing or []
    out: List[Any] = []
    for idx, (kind, src) in enumerate(CELLS):
        body = src if src.endswith("\n") else src + "\n"
        record = executed.get(f"{idx}:{src.rstrip(chr(10))}")
        if record:
            # ``from_dict`` rebuilds real Output objects; assigning the raw dicts
            # would pass a later json.dumps but blow up inside nbformat.write.
            cell = nbformat.from_dict({**record, "source": body})
        else:
            cell = new_code_cell(body) if kind == CODE else new_markdown_cell(body)
        # Keep the cell id when the cell at this index is unchanged: a rebuild of
        # untouched cells should not churn ids (they show up in notebook diffs).
        if idx < len(existing) and existing[idx].get("id"):
            if "".join(existing[idx].get("source", [])).rstrip("\n") == src.rstrip("\n"):
                cell["id"] = existing[idx]["id"]
        out.append(cell)
    return out


def build(out_path: Path = OUT, check: bool = False) -> Path:
    """Write (or verify) the notebook. ``check=True`` never mutates the file."""
    out_path = Path(out_path)
    executed = _executed_records(out_path)
    previous = (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    )
    cells = _cell_objects(executed, previous.get("cells") or [])
    _guard()(cells)

    nb = new_notebook(cells=cells)
    # Carry over the environment metadata the kernel wrote, so the rebuilt
    # artifact stays identical to the notebook that produced the run.
    for key, value in (previous.get("metadata") or {}).items():
        nb.metadata[key] = value
    nb.metadata.setdefault("kernelspec", {
        "display_name": "Python 3", "language": "python", "name": "python3",
    })
    nb.metadata.setdefault("language_info", {"name": "python", "version": "3.11"})
    nb.metadata.setdefault("colab", {"gpuType": "T4", "toc_visible": True, "provenance": []})

    if check:
        current = json.loads(out_path.read_text(encoding="utf-8"))
        # nbformat normalises trailing newlines, so compare on stripped text:
        # the builder's contract is the cell *content*, not its final newline.
        on_disk = [(c["cell_type"], "".join(c["source"]).rstrip("\n")) for c in current["cells"]]
        expected = [(c["cell_type"], c["source"].rstrip("\n")) for c in cells]
        if on_disk != expected:
            raise SystemExit(
                f"{out_path} is stale: {sum(1 for a, b in zip(on_disk, expected) if a != b)} "
                f"cell(s) differ, {len(on_disk)} on disk vs {len(expected)} in the builder. "
                f"Run: python {Path(__file__).name}"
            )
        print(f"{out_path} matches the builder ({len(cells)} cells)")
        return out_path

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)

    # Validate what actually hit the disk, not the object we hoped to write.
    written = json.loads(out_path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.reads(json.dumps(written), as_version=4))
    print(f"wrote {out_path} - {len(cells)} cells, valid nbformat 4 JSON")
    return out_path


if __name__ == "__main__":
    build(check="--check" in sys.argv[1:])
