#!/usr/bin/env python3
"""Generate notebooks/spancall_train_eval_test.ipynb via nbformat (valid JSON guaranteed)."""
from pathlib import Path
import json

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parent / ".." / "notebooks" / "spancall_train_eval_test.ipynb"

# --------------------------------------------------------------------------
# Cell sources
# --------------------------------------------------------------------------

MARKDOWN_TITLE = """# SpanCall — Train / Eval / Test Notebook

This notebook runs the **full SpanCall workflow**: **git clone/pull → env + HF dataset pull → tensorize → staged training (T0/T1/T2/C) → prediction → calibration → offline eval → test splits**.

> The public code repository (`https://github.com/eleluong/spancall.git`) is cloned or pulled from Git, dependencies are installed, and canonical dataset splits are pulled automatically from the Hugging Face Hub (`DATASET_NAME` in `.env`, default `convoicon/spancall`) when not already present in `DATA_DIR`.
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

Reads `.env` (or environment variables) for `DATASET_NAME` (`convoicon/spancall`), `HUGGING_FACE_TOKEN`, and LLM credentials (`BASE_URL`, `API_KEY`, `BASE_LLM_MODEL`), auto-discovers `PROJECT_ROOT` so `spancall` and `span_template` import directly, and configures device & paths.
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
DRIVE_OUT     = "/content/drive/MyDrive/spancall_artifacts"

# --- data / checkpoints / eval ---
DATA_DIR      = str(PROJECT_ROOT / "data" / "processed")
TENSOR_DIR    = str(PROJECT_ROOT / "data" / "tensorized")
RESULTS_DIR   = str(PROJECT_ROOT / "results")
EVAL_SPLIT    = "hermes_eval"          # in-domain eval (TEST_ID)
TEST_SPLITS   = ["hermes_test_ood_schema", "hermes_test_ood_para", "hermes_test_adversarial"]

import torch
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# SINGLE SOURCE OF TRUTH - every "what data to process / train on" knob.
# Everything below is defined BEFORE any download, filtering, or tensorization
# runs, so changing a knob NEVER requires re-downloading or re-processing data
# that was built under a different config. Canonical splits in DATA_DIR are
# never rewritten by later cells.
# ============================================================================

EXPERIMENT_NAME = "spancall_colab"
SEED            = 42                 # seeds ALL deterministic subsampling
                                     # (must equal trainer seed for reproducibility)

# --- model / tensorization budget ---
BACKBONE       = "Qwen/Qwen3-Embedding-0.6B"   # encoder + tokenizer backbone
MAX_SEQ_LEN    = 4096                # rows whose tokenized input exceeds this are
                                     # dropped during tensorization (reported in the
                                     # summary - NEVER written back over source splits)
MAX_SPAN_WIDTH = 16                  # candidate span width cap (mask builder)
# T3: MAX_POS is now driven by configs/default.yaml (optimization.max_pos).
# It fingerprints tensorized rows (pos_indices pad columns), so changing it
# requires re-tensorizing — keeping the single source of truth in the config
# prevents the trainer (max_pos) and the tensorizer (MAX_POS) from drifting.
import yaml
try:
    _cfg_yaml = yaml.safe_load(open("configs/default.yaml")) or {}
    MAX_POS   = int((_cfg_yaml.get("optimization") or {}).get("max_pos", 3))
except Exception:
    MAX_POS   = 3                   # fallback: legacy multi-positive default

# --- which canonical split feeds each trainer split ---
SPLIT_INPUTS = {
    "t0":       "hermes_train.jsonl",
    "t1":       "hermes_train.jsonl",
    "t2":       "hermes_train.jsonl",
    "stage_c":  "hermes_calib.jsonl",
    "dev":      "hermes_dev.jsonl",
    "calib":    "hermes_calib.jsonl",
}

# --- hard caps on samples tensorized per split (None = all) ---
SAMPLE_LIMITS = {"t0": None, "t1": None, "t2": None, "stage_c": None, "dev": None, "calib": None}

# --- how much of each split to tensorize ---
# Applied BEFORE tensorization via a seeded row selection that mirrors
# SpanCallTrainer.load_rows exactly (random.Random(SEED), n = N*frac,
# sorted rng.sample) so a 0.2 run tensorizes 20% instead of tensorizing 100%
# and only then training on 20%. When DATA_FRACTION < 1.0 the trainer must run
# with data.fraction = 1.0 (see cell §4) because the tensorized files ALREADY
# are the subset.
DATA_FRACTION = 1.0                  # 1.0 = full data (B1); <1.0 = seeded quick run

# --- parallel tensorization ---
TENSORIZE_WORKERS = max(1, min(os.cpu_count() or 1, 16))  # fork pool size for the
                                         # tensorize loop. Output is deterministic
                                         # regardless of this value (rows keep input order).

# --- eval / test sample caps at inference time ---
EVAL_LIMIT     = 500                 # param-extraction / tool-pred / e2e eval
TEST_LIMIT     = 200                 # predictions per OOD/adversarial test split

# --- dataset re-pull ---
FORCE_HF_DOWNLOAD = False            # True -> always re-pull canonical splits from HF

# ============================================================================
# CONFIG SUMMARY - the run's data contract is explicit up front.
# ============================================================================
print("=" * 72)
print("DATA CONFIG (single source of truth, defined before any processing)")
print("=" * 72)
print(f"EXPERIMENT_NAME = {EXPERIMENT_NAME}  SEED = {SEED}")
print(f"BACKBONE        = {BACKBONE}")
print(f"MAX_SEQ_LEN     = {MAX_SEQ_LEN}  MAX_SPAN_WIDTH = {MAX_SPAN_WIDTH}  MAX_POS = {MAX_POS}")
print("SPLIT_INPUTS:")
for k, v in SPLIT_INPUTS.items():
    print(f"  {k:<9} <- {v}   (limit={SAMPLE_LIMITS.get(k)})")
print(f"DATA_FRACTION   = {DATA_FRACTION}  (seeded row selection BEFORE tensorization)")
print(f"TENSORIZE_WORKERS = {TENSORIZE_WORKERS}  (parallel tensorize pool)")
print(f"EVAL_LIMIT      = {EVAL_LIMIT}  TEST_LIMIT = {TEST_LIMIT}")
print(f"FORCE_HF_DOWNLOAD = {FORCE_HF_DOWNLOAD}")

print(f"PROJECT_ROOT  = {PROJECT_ROOT}")
print(f"HF_DATASET_ID = {HF_DATASET_ID} (token={'set' if HF_TOKEN else 'unset'})")
print(f"DATA_DIR      = {DATA_DIR}")
print(f"TENSOR_DIR    = {TENSOR_DIR}")
print(f"DEVICE        = {DEVICE}")
'''

MARKDOWN_ENV = """---
## 1. Runtime & Package Check

Verify GPU/device availability and confirm `spancall` and `span_template` import cleanly."""

CODE_GPU = """!nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo 'No NVIDIA GPU detected (using MPS / CPU fallback)'"""

CODE_IMPORTS_CHECK = '''import spancall
import span_template
import transformers
import datasets
import huggingface_hub

print("spancall     :", spancall.__file__)
print("span_template:", span_template.__file__)
print("transformers :", transformers.__version__, "| datasets:", datasets.__version__)'''

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

Loads the 7 canonical SpanCall splits (`hermes_train`, `hermes_dev`, `hermes_calib`, `hermes_eval`, `hermes_test_ood_schema`, `hermes_test_ood_para`, `hermes_test_adversarial`) from `HF_DATASET_ID` (`convoicon/spancall`) on the Hugging Face Hub using `huggingface_hub.snapshot_download` (or `datasets.load_dataset`).

- If `DATA_DIR` already contains the `.jsonl` splits locally, they are used directly (set `FORCE_HF_DOWNLOAD = True` to re-pull from HF).
- You can also inspect any split directly as a Hugging Face `Dataset` via `datasets.load_dataset(HF_DATASET_ID, data_files=...)`."""

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

# Load via Hugging Face `datasets` to verify hub/local dataset integration
hf_ds = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATA_DIR, "hermes_train.jsonl"),
        "dev":   os.path.join(DATA_DIR, "hermes_dev.jsonl"),
        "calib": os.path.join(DATA_DIR, "hermes_calib.jsonl"),
        "eval":  os.path.join(DATA_DIR, "hermes_eval.jsonl"),
    },
)
print("\\nHugging Face DatasetDict loaded:", hf_ds)'''

MARKDOWN_TENSOR_INTRO = """---
## 3. The Tensorizer — `UnifiedSample` → trainer rows (closes the P7 gap)

The trainer consumes **tensorized rows** (`t0/t1/t2/stage_c/dev/calib.jsonl`) with this exact per-row shape (see [`spancall/train/trainer.py`](../spancall/train/trainer.py) batch-format docstring):

```
input_ids, attention_mask   (L,) long
param_positions             (K,) long in [0, L)          # <|param|> anchor positions
utt_starts, utt_ends        (1,) long                    # utterance extent
cand_starts, cand_ends      (K, C) long, -1 padded       # candidate spans
pos_indices                 (K, P) long, -1 padded       # positive cand cols (0 = abstain)
reason_targets              (K,) long 0..4, -1 masked    # H4 reason classes
accept_targets              ()  float                    # stratum F → 1.0, A → 0.0
cue_mask                    (K,) float                   # pointer mask (all ones)
tool_positions              (T,) long in [0, L)          # <|tool|> anchor positions (multi-tool)
tool_targets                ()  long                     # ground-truth tool index (0..T-1)
```

The cell below builds those rows per sample by:

1. Loading the tokenizer + marker tokens (same as `scripts/train.py`).
2. For multi-tool training/dev (`t2` and `dev`), rendering candidate tools with `build_input_multi` to produce `tool_positions` and `tool_targets`. For single-tool stages (`t0`/`t1`), rendering `build_input(gt_tool)`.
3. Tokenizing with `return_offsets_mapping` and BPE-aware entry snapping (`_snap`).
4. Building `CandidateMaskBuilder.build_param_masks` per param (P4) over the rendered text and offsets.
5. Matching ground-truth positive spans by verifying `canonical_value` against candidate canonical values.
6. Capping candidates to `MAX_C = 512` by sampling hard negatives near target spans plus a uniform stride (no answer-position leakage).
7. Writing full splits without destructive downsampling."""

CODE_TENSOR_SETUP = '''import json, os, math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from span_template import SPECIAL_MARKER_TOKENS, Entry, RecentRow, build_input
from spancall.data.align import tooldef_to_template_tool
from spancall.data.schema import Stratum, UnifiedSample
from spancall.model.masks import CandidateMaskBuilder

H4_REASONS = ("OK", "ABSENT", "AMBIGUOUS", "NEEDS_REASONING", "OFF_WINDOW")
STRATUM_ABSTAIN = {Stratum.A}                       # absent/unresolvable → abstain row
# D2 fix: when a Stratum-F parameter has NO matched positive candidate, the old
# behavior silently labeled it abstain (pos_cols=[0]) — corrupting ~15% of F
# params into fake abstain targets (notebook H-D2 CONFIRMED, silent_rate=15.4%).
# MASK_F_UNMATCHED=1 turns those rows into fully-masked (-1) targets that are
# excluded from the pointer loss (has_pos=False) instead of teaching H3 to abstain.
MASK_F_UNMATCHED = 1


# BACKBONE / MAX_SEQ_LEN / MAX_SPAN_WIDTH come from cell §0b (single config source)
tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_MARKER_TOKENS)})
mask_builder = CandidateMaskBuilder(max_seq_len=MAX_SEQ_LEN, max_span_width=MAX_SPAN_WIDTH)
print('tokenizer ready; vocab =', len(tokenizer), '| MAX_SEQ_LEN =', MAX_SEQ_LEN)'''

CODE_TENSOR_HELPERS = '''def _now_iso(s: UnifiedSample) -> date:
    """Deterministic clock from the sample (R4): current_time_iso else fallback."""
    if s.current_time_iso:
        try:
            return date.fromisoformat(s.current_time_iso)
        except ValueError:
            pass
    return date(2026, 1, 1)


def _recent_rows(s: UnifiedSample) -> List[RecentRow]:
    rows_out: List[RecentRow] = []
    for r in (s.recent_context or []):
        try:
            rows_out.append(RecentRow(key=str(r.get("key", "")), value=str(r.get("value", "")),
                                      source=str(r.get("source", "recent"))))
        except ValueError:
            continue
    return rows_out


def _h4(s: UnifiedSample, param: str, default: str = "OK") -> int:
    reasons = (s.metadata.get("h4_reasons") or {}) if s.metadata else {}
    val = reasons.get(param, default)
    return H4_REASONS.index(val) if val in H4_REASONS else 0



def _snap(cs: int, ce: int, offsets: List[tuple]) -> Optional[tuple]:
    """BPE-aware snapping: first/last token overlapping [cs, ce)."""
    if ce <= cs:
        return None
    first = last = None
    for i, (tcs, tce) in enumerate(offsets):
        if i >= MAX_SEQ_LEN:
            break
        if tce <= tcs:
            continue
        if tce > cs and tcs < ce:
            if first is None:
                first = i
            last = i
    return (first, last) if first is not None else None


def _param_anchors(entries, offsets: List[tuple], n_params: int) -> List[int]:
    """Template-order <|param|> anchor token indexes."""
    anchors: List[int] = []
    for e in entries:
        if getattr(e, "kind", None) != "param_anchor" or getattr(e, "value", None) == "<|none|>":
            continue
        r = _snap(e.char_start, e.char_end, offsets)
        anchors.append(r[0] if r is not None else 0)
    return (anchors + [0] * n_params)[:n_params]




def tensorize_sample(
    s: UnifiedSample,
    tokenizer,
    mask_builder,
    pack_multi_tools: bool = False,
) -> Optional[Dict[str, Any]]:
    """Build one trainer row from a UnifiedSample. Supports multi-tool packing for H1 tool selection."""
    if not s.tools or not s.ground_truth_calls:
        return None
    gt_tool_name = s.ground_truth_calls[0].tool_name
    gt_idx = next((i for i, t in enumerate(s.tools) if t.name == gt_tool_name), None)
    if gt_idx is None:
        return None
    gt_tool = s.tools[gt_idx]
    if not gt_tool.parameters:
        return None

    from span_template import build_input_multi

    # Always pack when requested (even for 1 tool) so every row of a split has
    # tool_positions/tool_targets -> uniform keys for Trainer.collate.
    use_multi = pack_multi_tools
    try:
        if use_multi:
            out = build_input_multi(
                [tooldef_to_template_tool(t) for t in s.tools],
                s.utterance,
                recent_context=_recent_rows(s),
                now=_now_iso(s),
            )
        else:
            out = build_input(
                tooldef_to_template_tool(gt_tool),
                s.utterance,
                recent_context=_recent_rows(s),
                now=_now_iso(s),
            )
    except Exception:
        return None

    enc = tokenizer(out.text, return_offsets_mapping=True, add_special_tokens=False)
    ids: List[int] = enc["input_ids"]
    offsets: List[tuple] = enc["offset_mapping"]
    if len(ids) > MAX_SEQ_LEN:
        return None

    # Utterance extent
    utt_entries = [e for e in out.entries if getattr(e, "segment", None) == "utterance"]
    if not utt_entries:
        return None
    u_cs = min(e.char_start for e in utt_entries)
    u_ce = max(e.char_end for e in utt_entries)
    usnap = _snap(u_cs, u_ce, offsets)
    if usnap is None:
        return None
    utt_starts, utt_ends = usnap

    # Tool anchors (H1)
    tool_positions: List[int] = []
    if use_multi:
        t_entries = [e for e in out.entries if getattr(e, "kind", None) == "bridge_text" and getattr(e, "tool", None) is not None]
        for e in t_entries:
            tsnap = _snap(e.char_start, e.char_end, offsets)
            tool_positions.append(tsnap[0] if tsnap else 0)

    # Param anchors: slice to gt_tool's parameters
    K = len(gt_tool.parameters)
    if use_multi:
        t_entries = [e for e in out.entries if getattr(e, "kind", None) == "bridge_text" and getattr(e, "tool", None) is not None]
        s_cs = t_entries[gt_idx].char_start
        s_ce = t_entries[gt_idx + 1].char_start if gt_idx + 1 < len(t_entries) else min(e.char_start for e in utt_entries)
        p_entries = [
            e for e in out.entries
            if getattr(e, "kind", None) == "param_anchor"
            and getattr(e, "value", None) != "<|none|>"
            and s_cs <= e.char_start < s_ce
        ]
        anchors = []
        for e in p_entries:
            psnap = _snap(e.char_start, e.char_end, offsets)
            anchors.append(psnap[0] if psnap else 0)
        anchors = (anchors + [0] * K)[:K]
    else:
        anchors = _param_anchors(out.entries, offsets, K)

    cand_start_rows: List[List[int]] = []
    cand_end_rows: List[List[int]] = []
    pos_rows: List[List[int]] = []
    reasons: List[int] = []
    cue: List[float] = []

    MAX_C = 512

    for idx, param in enumerate(gt_tool.parameters):
        masks = mask_builder.build_param_masks(param, out.entries, offsets, out.text)
        cands = list(masks["candidates"])

        targets = (s.target_spans or {}).get(param.name, [])
        matched_cis: List[int] = []
        for t in targets:
            if t.target_type == "unresolved":
                continue
            if t.target_type == "null_ctx":
                for ci, c in enumerate(cands):
                    if c.region == "null" and ci not in matched_cis:
                        matched_cis.append(ci)
                        break
                continue
            found = False
            # Verify canonical value first to avoid shifted char offsets
            if t.canonical_value is not None:
                t_str = str(t.canonical_value).strip().lower()
                for ci, c in enumerate(cands):
                    if c.canon_value is not None:
                        c_str = str(c.canon_value).strip().lower()
                        if c.canon_value == t.canonical_value or c_str == t_str:
                            if ci not in matched_cis:
                                matched_cis.append(ci)
                            found = True
                            break
            if not found and t.char_start is not None:
                for ci, c in enumerate(cands):
                    if c.char_start == t.char_start and c.char_end == t.char_end:
                        if ci not in matched_cis:
                            matched_cis.append(ci)
                        found = True
                        break

        # Fallback to direct GT argument matching if target_spans missed (recovers silent abstains)
        if not matched_cis and s.stratum == Stratum.F and gt_tool.name in (s.ground_truth_calls[0].tool_name if s.ground_truth_calls else ""):
            gt_args = dict(s.ground_truth_calls[0].arguments or {}) if s.ground_truth_calls else {}
            gt_val = gt_args.get(param.name)
            if gt_val is not None:
                for ci, c in enumerate(cands):
                    if c.canon_value is not None:
                        c_val = str(c.canon_value).strip().lower()
                        g_val = str(gt_val).strip().lower()
                        if c.canon_value == gt_val or c_val == g_val:
                            if ci not in matched_cis:
                                matched_cis.append(ci)

        # Candidate capping: hard negatives near positive spans + uniform stride
        if len(cands) > MAX_C:
            keep_pos = matched_cis[:MAX_POS]
            keep_set = set(keep_pos)
            pool = []
            for pi in keep_pos:
                for delta in range(-15, 16):
                    ni = pi + delta
                    if 0 <= ni < len(cands) and ni not in keep_set:
                        pool.append(ni)
                        keep_set.add(ni)
            rem = MAX_C - len(keep_set)
            if rem > 0:
                stride = max(1, len(cands) // rem)
                for ci in range(0, len(cands), stride):
                    if ci not in keep_set:
                        pool.append(ci)
                        keep_set.add(ci)
                    if len(pool) + len(keep_pos) >= MAX_C:
                        break
            kept_indices = sorted(keep_pos + pool[: MAX_C - len(keep_pos)])
            idx_map = {old_ci: new_ci for new_ci, old_ci in enumerate(kept_indices)}
            cands = [cands[ci] for ci in kept_indices]
            matched_cis = [idx_map[ci] for ci in keep_pos if ci in idx_map]

        cand_start_rows.append([c.tok_start for c in cands])
        cand_end_rows.append([c.tok_end for c in cands])

        _max_pos = int(MAX_POS)
        pos_cols = [ci + 1 for ci in matched_cis[:_max_pos]]
        if s.stratum in STRATUM_ABSTAIN:
            pos_cols = [0]                          # genuine abstain target (row is A)
        elif not pos_cols and MASK_F_UNMATCHED:
            pos_cols = []                           # D2 fix: exclude from loss entirely
        elif not pos_cols:
            pos_cols = [0]                          # old corrupting behavior (opt-out)

        pos_rows.append((pos_cols + [-1] * _max_pos)[:_max_pos])
        reasons.append(_h4(s, param.name))
        cue.append(0.0 if not pos_cols else 1.0)    # fully-masked row → zeroed via pointer_mask

    C = max((len(c) for c in cand_start_rows), default=0)
    C = max(min(C, MAX_C), 1)
    for i in range(len(cand_start_rows)):
        if len(cand_start_rows[i]) < C:
            cand_start_rows[i] += [-1] * (C - len(cand_start_rows[i]))
            cand_end_rows[i] += [-1] * (C - len(cand_end_rows[i]))
        else:
            cand_start_rows[i] = cand_start_rows[i][:C]
            cand_end_rows[i] = cand_end_rows[i][:C]

    row: Dict[str, Any] = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.ones(len(ids), dtype=torch.long),
        "param_positions": torch.tensor(anchors, dtype=torch.long),
        "utt_starts": torch.tensor([utt_starts], dtype=torch.long),
        "utt_ends": torch.tensor([utt_ends], dtype=torch.long),
        "cand_starts": torch.tensor(cand_start_rows, dtype=torch.long),
        "cand_ends": torch.tensor(cand_end_rows, dtype=torch.long),
        "pos_indices": torch.tensor(pos_rows, dtype=torch.long),
        "reason_targets": torch.tensor(reasons, dtype=torch.long),
        "accept_targets": torch.tensor(1.0 if s.stratum == Stratum.F else 0.0),
        "cue_mask": torch.tensor(cue, dtype=torch.float),
    }
    if tool_positions:
        row["tool_positions"] = torch.tensor(tool_positions, dtype=torch.long)
        row["tool_targets"] = torch.tensor(gt_idx, dtype=torch.long)
    return row'''

CODE_TENSOR_RUN = '''# ---- tensorize all splits: config-driven, NON-destructive + PARALLEL ----
# Canonical splits under DATA_DIR are never mutated: rows are length-filtered
# (len(ids) > MAX_SEQ_LEN -> tensorize_sample returns None) and fractioned
# (seeded row selection mirroring SpanCallTrainer.load_rows) here in-memory,
# with drop counts reported below.
#
# W1: the per-sample tensorize loop runs in a fork-based ProcessPoolExecutor
# (TENSORIZE_WORKERS workers). Each worker inherits the already-loaded
# tokenizer + mask_builder at fork time -> zero per-worker re-loading cost.
# If fork is unavailable or the pool raises, it falls back to an identical
# sequential loop (same input order -> same output rows).
#
# W2: out-splits sharing the same (input split, packing mode) are tensorized
# ONCE and written to every member. hermes_train's single-tool form feeds both
# t0 and t1, so that file is processed twice total (single-tool + multi-tool)
# instead of three times.

import json, os, random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from spancall.data.schema import UnifiedSample

def _iter_raw_lines(path: str):
    """Yield raw JSON dicts from a .jsonl split (no dataclass construction)."""
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

def _fraction_subset(raw_lines, frac: float, seed: int):
    """Seeded index selection contractually identical to trainer.load_rows:
    rng = random.Random(seed); n = max(int(N*frac), 1);
    keep sorted(rng.sample(range(N), min(n, N))). Applied to RAW lines so a
    <1.0 fraction run saves the expensive tensorize work too."""
    raw_lines = list(raw_lines)
    if frac >= 1.0:
        return raw_lines
    rng = random.Random(int(seed))
    n = max(int(len(raw_lines) * frac), 1)
    keep = set(rng.sample(range(len(raw_lines)), min(n, len(raw_lines))))
    return [d for i, d in enumerate(raw_lines) if i in keep]

def _serialize(row):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}

def _tensorize_task(args):
    """Tensorize one raw sample -> serialized row dict, or None (dropped).
    Module-level so the fork pool can reference it; tokenizer / mask_builder /
    tensorize_sample / _serialize are inherited from the parent at fork time."""
    d, pack_multi = args
    try:
        s = UnifiedSample.from_dict(d)
        row = tensorize_sample(s, tokenizer, mask_builder, pack_multi_tools=pack_multi)
        if row is None:
            return None
        return _serialize(row)
    except Exception:
        return None

_CHUNK = 256  # tasks per submitted batch (bounded memory + early-limit stop)

def _tensorize_many(raw, pack_multi: bool, need, workers: int):
    """Tensorize `raw` in a fork pool (or sequentially as fallback), preserving
    input order. Stops early once `need` serialized rows are produced so
    SAMPLE_LIMITS still bounds runtime."""
    if need is None:
        need = len(raw)
    rows = []
    ctx = None
    if workers > 1:
        try:
            ctx = mp.get_context("fork")
        except Exception:
            ctx = None
    if ctx is not None:
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                for start in range(0, len(raw), _CHUNK):
                    batch = raw[start:start + _CHUNK]
                    for res in ex.map(_tensorize_task, [(d, pack_multi) for d in batch]):
                        if res is not None:
                            rows.append(res)
                            if len(rows) >= need:
                                return rows
            return rows
        except Exception:
            rows = []  # pool failed -> sequential fallback below
    for d in raw:
        res = _tensorize_task((d, pack_multi))
        if res is not None:
            rows.append(res)
            if len(rows) >= need:
                break
    return rows

# Group out-splits by (input split, packing mode) so shared work runs once.
# In the unified B-single pipeline, all train and dev rows use uniform multi-tool packing.
groups = defaultdict(list)
for out_name, in_name in SPLIT_INPUTS.items():
    pack_multi = out_name in ("t0", "t1", "t2", "dev")
    groups[(in_name, pack_multi)].append(out_name)

os.makedirs(TENSOR_DIR, exist_ok=True)
stats = {}

for (in_name, pack_multi), out_names in groups.items():
    src = os.path.join(DATA_DIR, in_name)
    if not os.path.exists(src):
        for on in out_names:
            print(f"SKIP {on}: {src} missing")
        continue
    raw = _fraction_subset(_iter_raw_lines(src), DATA_FRACTION, SEED)
    # need = largest per-member limit so shared work satisfies the tightest cap
    need = None
    for on in out_names:
        lim = SAMPLE_LIMITS.get(on)
        if lim is not None:
            need = lim if need is None else max(need, lim)
    rows = _tensorize_many(raw, pack_multi, need, TENSORIZE_WORKERS)
    mode = "multi-tool" if pack_multi else "single-tool"
    for on in out_names:
        limit = SAMPLE_LIMITS.get(on)
        out_rows = rows if limit is None else rows[:limit]
        out_path = os.path.join(TENSOR_DIR, f"{on}.jsonl")
        with open(out_path, "w") as fh:
            for r in out_rows:
                fh.write(json.dumps(r) + "\\n")
        stats[on] = {
            "rows": len(out_rows),
            "dropped": len(raw) - len(rows),   # over processed prefix (exact when no limit)
            "shared_with": [o for o in out_names if o != on] or None,
            "raw_after_fraction": len(raw),
            "fraction": DATA_FRACTION,
            "pack_multi": pack_multi,
            "limit": limit,
        }
        print(f"wrote {on}.jsonl: {len(out_rows)} rows ({len(raw) - len(rows)} dropped,"
              f" {len(raw)} raw after fraction {DATA_FRACTION}, {mode})")

print("\\ntensorize summary:", json.dumps(stats, indent=2))

# ---- config fingerprint: guards checkpoint selection against stale runs ----
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
}
# NOTE: TENSORIZE_WORKERS is intentionally NOT in the fingerprint — output rows
# are identical regardless of pool size, so checkpoint validity is unaffected.
fp_path = os.path.join(TENSOR_DIR, "config_fingerprint.json")
with open(fp_path, "w") as fh:
    json.dump(fingerprint, fh, indent=2, sort_keys=True)
print("wrote config fingerprint ->", fp_path)
'''

MARKDOWN_TENSOR_SANITY = """### Sanity: dev split tensorizes with computable positives

Quick diagnostic so you can see the ratio of fully-grounded rows before spending GPU time on training."""

CODE_TENSOR_SANITY = '''import json, os
dev_path = os.path.join(TENSOR_DIR, "dev.jsonl")
if os.path.exists(dev_path):
    with open(dev_path) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    if not rows:
        print("dev.jsonl found but is empty (0 rows)")
    else:
        # A positive target is a candidate span index > 0 (0 is abstain, -1 is padding)
        n_hit = sum(
            1 for r in rows
            if any(pi > 0 for p_pos in r.get("pos_indices", []) for pi in p_pos)
        )
        print(f"dev.jsonl: {len(rows)} rows; rows with ≥1 positive target: {n_hit} ({n_hit/len(rows):.1%})")
        r = rows[0]
        print("sample row keys:", sorted(r.keys()))
        c_len = len(r["cand_starts"][0]) if r.get("cand_starts") and len(r["cand_starts"]) > 0 else 0
        print("input_ids:", len(r.get("input_ids", [])), "seq_len; params K =", len(r.get("param_positions", [])),
              "cand (K,C) =", len(r.get("cand_starts", [])), "x", c_len)
else:
    print("dev.jsonl not found — run the tensorizer cell first")'''

MARKDOWN_TRAIN = """---
## 4. Training — staged curriculum `T0 → T1 → T2 → C`

Runs [`scripts/train.py`](../scripts/train.py) against the tensorized splits.

- `--data_dir` → `TENSOR_DIR` above.
- `--config` → `configs/colab.yaml` (5 epochs for full convergence, batch_size 8, gradient_accumulation_steps 4, effective batch 32, bf16 mixed precision, gradient checkpointing, unfreezing all encoder stages)."""

CODE_COLAB_CFG = '''import yaml, os

# EXPERIMENT_NAME / SEED / DATA_FRACTION / BACKBONE / MAX_SEQ_LEN are defined
# (once) in cell §0b. DATA_FRACTION is already applied as a seeded row subset
# DURING tensorization (cell §3), so the trainer consumes 100% of the
# tensorized files. Set TRAIN_FRACTION < 1.0 HERE only if you re-tensorized
# with DATA_FRACTION = 1.0 and want a second, trainer-side subsample for an
# even quicker sanity run.
TRAIN_FRACTION = 1.0

with open("configs/default.yaml") as fh:
    cfg = yaml.safe_load(fh)

cfg["experiment"]["name"] = EXPERIMENT_NAME
cfg["experiment"]["seed"] = SEED
cfg["experiment"]["output_dir"] = RESULTS_DIR
cfg["experiment"]["data_dir"] = TENSOR_DIR

# Train for 5 epochs (user requirement: at least 3, recommended 5)
NUM_EPOCHS = 5
cfg["optimization"]["num_epochs"] = NUM_EPOCHS
cfg["optimization"]["batch_size"] = 8
cfg["optimization"]["mixed_precision"] = "bf16"
# abstain_weight intentionally NOT overridden here: the shipped H-T2 recipe
# (0.5) flows through from configs/default.yaml (docs/param_recall_diagnosis.md §7b).
cfg["training"]["gradient_accumulation_steps"] = 4
cfg["training"]["gradient_checkpointing"] = True
cfg["training"]["eval_splits"] = ["dev"]      # A6: per-epoch held-out test evaluation curve
cfg["training"]["eval_limit"] = 500
cfg["training"]["resolver_eval_split"] = "dev"  # e2e resolver eval hook (T1-gate vs post-eval divergence)
cfg["training"]["resolver_eval_limit"] = 40
cfg["model"]["freeze_encoder_stages"] = []
cfg["model"]["max_seq_len"] = MAX_SEQ_LEN          # from §0b

# fraction: tensorized files already reflect DATA_FRACTION (see cell §3)
cfg.setdefault("data", {})["fraction"] = TRAIN_FRACTION

out = "configs/colab.yaml"
with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh)
print(f"wrote {out} (epochs={NUM_EPOCHS}, bs=8, accum=4, bf16=True, freeze=[], data_fraction={TRAIN_FRACTION})")
CONFIG_PATH = out
'''

CODE_TRAIN_RUN = '''# Direct in-notebook training execution (provides live logging and full native tracebacks)
from scripts.train import main as train_main

train_main([
    "--config", CONFIG_PATH,
    "--experiment_name", EXPERIMENT_NAME,
    "--output_dir", RESULTS_DIR,
    "--data_dir", TENSOR_DIR,
    "--stages", "T2",
    "--device", DEVICE,
    "--seed", str(SEED),
])'''

MARKDOWN_TRAIN_TELEMETRY = """### Training telemetry (epoch curves & loss components)

`run_stage` writes a JSON-lines log to `results/{exp}_seed{seed}/train_log.jsonl` (`stage_start` / `train` / `epoch_end` / `epoch_eval` / `stage_end` events). This cell replays it into directly readable tables:

- per-stage summary (rows, epochs, final loss, steps)
- per-epoch held-out eval curve (`span_acc`, `exact_payload_match`, `abstain_acc`) — the A6 gate used for checkpoint selection
- per-epoch mean train loss
- LR + loss-component trajectory (pointer / accept / reason / tool / corr)"""

CODE_TRAIN_TELEMETRY = '''# ---- training telemetry: replay train_log.jsonl into readable tables ----
import json, os
from collections import defaultdict

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
    print("TRAIN LOG — STAGE SUMMARIES")
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
        print(f"  {'stage':<8}{'epoch':>6}{'n':>6}{'span_acc':>10}{'exact_payload':>15}{'abstain_acc':>13}")
        for r in evals:
            print(f"  {str(r.get('stage','?')):<8}{r.get('epoch','?'):>6}{r.get('n','?'):>6}"
                  f"{r.get('span_acc', 0):>10.3f}{r.get('exact_payload_match', 0):>15.3f}"
                  f"{r.get('abstain_acc', 0):>13.3f}")
    else:
        print("  no epoch_eval events (held-out eval not enabled for these stages)")

    print("\\n" + "=" * 78)
    print("PER-EPOCH MEAN TRAIN LOSS (epoch_end events)")
    print("=" * 78)
    ends = [r for r in train_log if r.get("event") == "epoch_end"]
    if ends:
        print(f"  {'stage':<8}{'epoch':>6}{'mean_loss':>12}")
        for r in ends:
            print(f"  {str(r.get('stage','?')):<8}{r.get('epoch','?'):>6}{r.get('mean_loss', 0):>12.4f}")
    else:
        print("  no epoch_end events")

    print("\\n" + "=" * 78)
    print("LOSS-COMPONENT TRAJECTORY (train events, downsampled to ~40 rows)")
    print("=" * 78)
    trs = [r for r in train_log if r.get("event") == "train"]
    if trs:
        step = max(1, len(trs) // 40)
        print(f"  {'stage':<8}{'step':>6}{'loss':>9}{'pointer':>10}{'accept':>10}{'reason':>10}{'tool':>10}{'corr':>10}{'lr':>12}")
        for i, r in enumerate(trs):
            if i % step == 0 or i == len(trs) - 1:
                print(f"  {str(r.get('stage','?')):<8}{r.get('stage_step','?'):>6}{r.get('loss', 0):>9.4f}"
                      f"{r.get('pointer', 0):>10.4f}{r.get('accept', 0):>10.4f}{r.get('reason', 0):>10.4f}"
                      f"{r.get('tool', 0):>10.4f}{r.get('corr', 0):>10.4f}{r.get('lr', 0):>12.2e}")
    else:
        print("  no train events")
else:
    print("No training log found — run the training cell first.")
'''

MARKDOWN_PICK_CP = """### Pick the best checkpoint

The trainer writes checkpoints per stage under `results/{exp}_seed{seed}/{stage}/checkpoints/`. We pick:

1. The **stage-C checkpoint** if it exists (full curriculum, frozen-encoder H8), else
2. the **last stage with a checkpoint** (T2 → T1 → T0), else
3. the `model_init` dir saved by `scripts/train.py`.

`CHECKPOINT_DIR` is then used by prediction/calibration/eval."""

CODE_PICK_CP = '''import glob, os, json

exp_root = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_seed{SEED}")

def _latest_stage_cp():
    # A7: Best-checkpoint-by-epoch-curve
    best_cp = os.path.join(exp_root, "best_checkpoint")
    if os.path.exists(os.path.join(best_cp, "model.pt")):
        return best_cp, "best_curve"
    for stage in ["T2", "T1", "t2", "t1", "C", "T0"]:
        cps = sorted(glob.glob(os.path.join(exp_root, stage, "checkpoints", "**", "model.pt"), recursive=True))
        if cps:
            return os.path.dirname(cps[-1]), stage
    return None, None

# ---- stale-artifact guard: refuse checkpoints trained under a different config ----
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
}
config_ok = True
if os.path.exists(fp_path):
    with open(fp_path) as fh:
        saved = json.load(fh)
    config_ok = (saved == fingerprint)
    if not config_ok:
        print("WARNING: TENSOR_DIR fingerprint does NOT match current config.")
        print("  saved  :", json.dumps(saved, sort_keys=True))
        print("  current:", json.dumps(fingerprint, sort_keys=True))
else:
    print("WARNING: no config_fingerprint.json in TENSOR_DIR (run the tensorize cell first).")
    config_ok = False

CHECKPOINT_DIR = None
cp_dir, cp_stage = _latest_stage_cp()
model_init = os.path.join(RESULTS_DIR, "model_init")

if cp_dir is not None and config_ok:
    CHECKPOINT_DIR = cp_dir
    print(f"picked stage-{cp_stage} checkpoint @ {CHECKPOINT_DIR}")
elif cp_dir is not None and not config_ok and ALLOW_STALE_CHECKPOINT:
    CHECKPOINT_DIR = cp_dir
    print(f"WARNING: using STALE stage-{cp_stage} checkpoint @ {CHECKPOINT_DIR} (config mismatch)")
elif os.path.isdir(model_init) and config_ok:
    CHECKPOINT_DIR = model_init
    print("no staged checkpoint; using model_init @", CHECKPOINT_DIR)
else:
    all_m = sorted(glob.glob(os.path.join(RESULTS_DIR, "**", "model.pt"), recursive=True))
    if all_m and (config_ok or ALLOW_STALE_CHECKPOINT):
        CHECKPOINT_DIR = os.path.dirname(all_m[-1])
        print("fallback checkpoint @", CHECKPOINT_DIR)
    else:
        print("no usable checkpoint (config fingerprint mismatch and ALLOW_STALE_CHECKPOINT=False)")

print("CHECKPOINT_DIR =", CHECKPOINT_DIR)
if CHECKPOINT_DIR:
    print("  contains:", sorted(os.listdir(CHECKPOINT_DIR)))
'''

MARKDOWN_PREDICT_INTRO = """---
## 5. Inference & Evaluation

To evaluate correctly without conflating failure modes, the evaluation is broken into three distinct, separate parts:
1. **Eval Part 1 — Parameter Extraction Evaluation (Oracle / Given GT Tool)**: Tests parameter extraction (`resolver.try_resolve(gt_tool, ...)`) independently of tool selection. Runs in **uncalibrated diagnostics mode**: it scores the raw argmax values from `res.parameters[p].value` (populated even when the abstain head vetoes a param) instead of `payload_partial`, so it isolates pointer-head quality and reports a veto/decomposition breakdown (`pointer correct-but-vetoed` vs `pointer wrong` vs `pointer empty`) — no calibration, no gate.
2. **Eval Part 2 — Function (Tool) Prediction Evaluation**: Tests H1 tool selection (`resolver.predict_tool(...)`) across candidate tool schemas independently. Runs in **uncalibrated diagnostics mode**: it always scores ≥ 2 candidates (GT tool guaranteed present; a same-corpus distractor is appended for single-tool rows) so the single-candidate shortcut in `predict_tool` never fires — reporting raw top-1 accuracy, mean winner probability and margin, with no gating.
3. **Eval Part 3 — End-to-End Function Calling Evaluation (General)**: Tests full function calling (`resolver.try_resolve_candidates(s.tools, ...)`: H1 tool selection + H3/H4 parameter extraction & gating) vs the `B1a` baseline on identical data subsets across `hermes_eval` and the 3 OOD/adversarial splits (`hermes_test_ood_schema`, `hermes_test_ood_para`, `hermes_test_adversarial`).
"""

CODE_RESOLVER = '''import json, os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from span_template import SPECIAL_MARKER_TOKENS
from spancall.data.schema import UnifiedSample
from spancall.infer.resolver import SpanCallResolver
from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.model.model import SpanCallModel
from spancall.train.config import TrainingConfig

cfg = TrainingConfig.from_yaml(CONFIG_PATH)
tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name, use_fast=True)
encoder = SpanCallTransformerEncoder(
    model_name_or_path=cfg.model.backbone_name,
    hidden_dropout_prob=cfg.model.dropout,
    tokenizer=tokenizer,
)
encoder.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))
model = SpanCallModel(
    encoder=encoder,
    heads=cfg.model.heads,
    head_dim=cfg.model.head_dim,
    max_span_width=cfg.model.max_span_width,
    hidden_dim=encoder.hidden_dim,
)

# Raw mode: pure argmax decoding, no confidence/abstain gating (Phase A1/A2)
resolver = SpanCallResolver(
    model_path=CHECKPOINT_DIR,
    raw_mode=True,
    encoder=encoder,
    model=model,
    tokenizer=tokenizer,
    max_seq_len=cfg.model.max_seq_len,
    max_span_width=cfg.model.max_span_width,
)
if DEVICE in ("cuda", "mps"):
    model.to(DEVICE)
model.eval()
if encoder is not None and hasattr(encoder, "eval"):
    encoder.eval()
print("resolver ready (checkpoint =", CHECKPOINT_DIR, ")")'''

MARKDOWN_EVAL_PART1 = """---
## 5a. Evaluation Part 1 — RAW Parameter Extraction (Oracle Tool Given, No Calibration)

Evaluates parameter extraction directly on ground-truth tools using `resolver.try_resolve(gt_tool, ...)`, completely decoupling parameter extraction from function/tool prediction.

> **Uncalibrated diagnostics mode.** This cell reads `res.parameters[p].value` — the **raw argmax candidate** the resolver keeps even when the abstain head vetoes the parameter ([`resolver.py:445`](../spancall/infer/resolver.py:445) stores it in `best_unfiltered`, and the unresolved branch re-exposes it at [`resolver.py:534`](../spancall/infer/resolver.py:534)) — instead of `res.payload_partial`, which the gate leaves empty before calibration. It isolates **pointer-head quality** from the abstain/gate machinery:
> - `source_type == "abstain"` → raw argmax was correct but the abstain head vetoed it (miscalibrated abstain head, not a bad pointer)
> - `source_type == "unresolved"` → no candidate beat abstain; `value` is the best canon_value anyway
> - `pointer_wrong` / `pointer_empty` → genuine pointer-head failures
>
> The gated (production-like) metrics (`payload_partial`, `hit`, `status`) are still available on the same `res` objects if a with-gate comparison is needed — they are simply not what this cell scores."""

CODE_EVAL_PART1 = '''import json, os
from collections import Counter, defaultdict
from spancall.data.schema import Stratum, UnifiedSample

def _norm(v):
    """Normalize a GT/pred value for comparison (None-safe, case/space tolerant)."""
    if v is None:
        return None
    s = str(v).strip()
    return s.lower() if s.lower() in ("true", "false", "none", "null") else s

def eval_param_extraction_raw(split_name: str, limit: Optional[int] = 500):
    """RAW parameter-extraction accuracy (no calibration, no gate).

    Reads `res.parameters[p].value` — the raw argmax candidate (populated even
    when the abstain head vetoed the param) — instead of `res.payload_partial`,
    which the gate leaves empty before calibration. This isolates pointer-head
    quality and separates "pointer wrong" from "pointer right but the abstain
    head vetoed it".
    """
    src = os.path.join(DATA_DIR, f"{split_name}.jsonl")
    if not os.path.exists(src):
        print(f"File not found: {src}")
        return

    total = 0
    stratum_f_total = 0
    stratum_f_exact = 0
    param_tp = 0
    param_fp = 0
    param_fn = 0
    raw_extracted = 0
    confs = []
    p_abs_all = []
    buckets = Counter()          # source_type -> count
    vetoed = 0                   # params the abstain head killed despite a candidate
    pointer_correct_vetoed = 0   # raw argmax correct but abstain head vetoed
    pointer_wrong = 0            # raw argmax present and != GT
    pointer_empty = 0            # no candidate found at all
    per_param = defaultdict(lambda: {"gt": 0, "tp": 0, "fp": 0, "fn": 0, "n_absent": 0})

    with open(src) as fh:
        for line in fh:
            if not line.strip():
                continue
            s = UnifiedSample.from_dict(json.loads(line))
            if not s.ground_truth_calls or not s.tools:
                continue
            gt_call = s.ground_truth_calls[0]
            gt_tool = next((t for t in s.tools if t.name == gt_call.tool_name), None)
            if not gt_tool:
                continue

            total += 1
            res = resolver.try_resolve(gt_tool, s.utterance, s.recent_context, s.current_time_iso)
            gt_args = gt_call.arguments or {}

            # RAW payload: every parameter's argmax value (cast value for chosen
            # params, canon_value for vetoed/unresolved params).
            raw_pred = {}
            for p in gt_tool.parameters:
                pr = res.parameters.get(p.name)
                if pr is None:
                    continue
                buckets[pr.source_type] += 1
                confs.append(pr.confidence)
                p_abs_all.append(pr.p_abstain)
                if pr.source_type == "abstain":
                    vetoed += 1
                if pr.value is not None:
                    raw_pred[p.name] = pr.value

            norm_pred = {k: _norm(v) for k, v in raw_pred.items()}
            norm_gt = {k: _norm(v) for k, v in gt_args.items()}
            if s.stratum == Stratum.F:
                stratum_f_total += 1
                if norm_pred == norm_gt:
                    stratum_f_exact += 1
            if raw_pred:
                raw_extracted += 1

            # parameter-level precision/recall over RAW argmax values
            for k, v in norm_pred.items():
                if k in norm_gt and norm_gt[k] == v:
                    param_tp += 1
                    pr = res.parameters.get(k)
                    if pr is not None and pr.source_type == "abstain":
                        pointer_correct_vetoed += 1
                else:
                    param_fp += 1
                    if k in norm_gt:
                        pointer_wrong += 1
            for k in norm_gt:
                if k not in norm_pred:
                    param_fn += 1
                    pr = res.parameters.get(k)
                    if pr is None or pr.value is None:
                        pointer_empty += 1

            # per-parameter accumulation (Stratum-F rows only — the extractable signal)
            if s.stratum == Stratum.F:
                for p in gt_tool.parameters:
                    pn = p.name
                    st = per_param[pn]
                    if pn in norm_gt:
                        st["gt"] += 1
                    if pn in norm_pred:
                        if pn in norm_gt and norm_pred[pn] == norm_gt[pn]:
                            st["tp"] += 1
                        else:
                            st["fp"] += 1
                    elif pn in norm_gt:
                        st["fn"] += 1
                    pr = res.parameters.get(pn)
                    if pr is not None and pr.source_type == "abstain":
                        st["n_absent"] += 1

            if limit is not None and total >= limit:
                break

    p = param_tp / max(1, param_tp + param_fp)
    r = param_tp / max(1, param_tp + param_fn)
    f1 = 2 * p * r / max(1e-8, p + r)
    f_exact = stratum_f_exact / max(1, stratum_f_total)
    mean_conf = sum(confs) / max(1, len(confs))
    mean_pabs = sum(p_abs_all) / max(1, len(p_abs_all))

    print(f"=== RAW Parameter Extraction (Oracle Tool) on {split_name} (N={total}) — no calibration ===")
    print(f"Raw Param Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    print(f"Raw Exact Payload Match (Stratum F): {f_exact:.4f} ({stratum_f_exact}/{stratum_f_total})")
    print(f"Raw Extraction Rate (>=1 param): {raw_extracted/max(1,total):.4f} ({raw_extracted}/{total})")
    print(f"Mean raw argmax confidence: {mean_conf:.4f} | Mean p_abstain: {mean_pabs:.4f}")
    print(f"Abstain-head vetoes: {vetoed} | ... of which raw argmax CORRECT (veto wrong): {pointer_correct_vetoed}")
    print(f"Pointer wrong (argmax present, != GT): {pointer_wrong} | Pointer empty (no candidate): {pointer_empty}")
    print(f"Per-param source_type distribution: {dict(buckets)}")

    if per_param:
        print("\\n--- Per-parameter RAW extraction (Stratum F only; sorted by most false-negatives) ---")
        print(f"{'param':<26}{'gt':>5}{'tp':>5}{'fp':>5}{'fn':>5}{'prec':>7}{'rec':>7}{'f1':>7}{'vetoed':>8}")
        for pn, st in sorted(per_param.items(), key=lambda kv: (-kv[1]["fn"], -kv[1]["gt"])):
            prec = st["tp"] / max(1, st["tp"] + st["fp"])
            rec = st["tp"] / max(1, st["gt"])
            f1p = 2 * prec * rec / max(1e-8, prec + rec)
            print(f"{pn:<26}{st['gt']:>5}{st['tp']:>5}{st['fp']:>5}{st['fn']:>5}"
                  f"{prec:>7.3f}{rec:>7.3f}{f1p:>7.3f}{st['n_absent']:>8}")
    print("================================================================================")

eval_param_extraction_raw(EVAL_SPLIT, limit=500)'''

MARKDOWN_EVAL_PART2 = """---
## 5b. Evaluation Part 2 — RAW Function (Tool) Prediction (No Calibration)

Evaluates H1 tool selection across candidate tool schemas independently, isolating tool choice accuracy from parameter extraction.

> **Uncalibrated diagnostics mode.** This cell **always feeds ≥ 2 candidate schemas** so the single-candidate shortcut in [`predict_tool`](../spancall/infer/resolver.py:261) (which returns `tool_prob/margin = 1.0` without a model forward pass) never fires. The GT tool is guaranteed present in the candidate set (patching the ~24% of `hermes_eval` rows whose `s.tools` lacks the GT tool name), and single-tool rows get a same-corpus distractor appended. `gated_hit` is ignored (it depends on `tau_tool`)."""

CODE_EVAL_PART2 = '''import json, os
from collections import Counter
from spancall.data.schema import UnifiedSample

def eval_tool_prediction_raw(split_name: str, limit: Optional[int] = 500,
                             distractor_pool: Optional[dict] = None):
    """RAW H1 tool-selection accuracy (no calibration, no gate).

    Always feeds >= 2 candidate schemas per row so the single-candidate
    shortcut in predict_tool (which returns prob/margin = 1.0 without a model
    forward pass) never fires. The true tool is guaranteed present in the
    candidate set; single-tool rows get a same-corpus distractor appended.
    """
    src = os.path.join(DATA_DIR, f"{split_name}.jsonl")
    if not os.path.exists(src):
        print(f"File not found: {src}")
        return

    # collect rows + a tool-name pool for distractor sampling
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
        # guarantee the true tool is present (hermes_eval has ~24% rows where
        # the GT tool name is not in s.tools; removing that data confound)
        if gt_name not in names and gt_name in distractor_pool:
            candidates.append(distractor_pool[gt_name])
            names.add(gt_name)
            grew = True
        # guarantee >= 2 distinct tools so the H1 shortcut never fires
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

    print(f"=== RAW Tool Prediction (H1) on {split_name} (N={total}) — >=2 candidates, no gate ===")
    print(f"Raw Top-1 Tool Accuracy (Overall): {acc_all:.4f} ({(single_correct + multi_correct)}/{total})")
    print(f"  Multi-tool subset: {acc_multi:.4f} ({multi_correct}/{multi_total})")
    print(f"  Single+distractor subset: {acc_single:.4f} ({single_correct}/{single_total})")
    print(f"Mean Winner Tool Probability: {mean_p:.4f} | Mean Margin: {mean_m:.4f}")
    print(f"Rows with empty tool_scores (no winner emitted): {empty_scores}")
    print(f"Rows where candidates were augmented beyond s.tools: {augmented}")
    print(f"Most common wrong picks: {wrong_as.most_common(8)}")

    if per_tool:
        print("\\n--- Per-tool H1 accuracy (sorted by most wrong picks) ---")
        print(f"{'tool':<30}{'n':>6}{'acc':>8}{'wrong_as':<40}")
        for tname, st in sorted(per_tool.items(), key=lambda kv: (-kv[1]["n"], kv[1]["correct"])):
            acc_t = st["correct"] / max(1, st["n"])
            wrong_str = ", ".join(f"{w}x{v}" for w, v in st["wrong_as"].most_common(3))
            print(f"{tname:<30}{st['n']:>6}{acc_t:>8.3f}  {wrong_str}")
    print("==================================================================")

eval_tool_prediction_raw(EVAL_SPLIT, limit=500)'''

CODE_PREDICT_SPLIT = '''def _param_detail(pr) -> Dict[str, Any]:
    """Serialize one ParameterResolution for the detailed prediction log."""
    return {
        "value": getattr(pr, "value", None),
        "confidence": float(getattr(pr, "confidence", 0.0) or 0.0),
        "margin": float(getattr(pr, "margin", 0.0) or 0.0),
        "source_type": getattr(pr, "source_type", None),
        "char_span": list(pr.char_span) if getattr(pr, "char_span", None) else None,
        "p_abstain": float(getattr(pr, "p_abstain", 0.0) or 0.0),
    }


def _features_detail(features) -> Dict[str, Any]:
    """A compact, JSON-safe view of resolver.features (latency + H1/H4 stats)."""
    feats = dict(features or {})
    if "tool_scores" in feats and isinstance(feats["tool_scores"], dict):
        # top-3 tool probabilities only (keep the log small)
        top = sorted(feats["tool_scores"].items(), key=lambda kv: -float(kv[1]))[:3]
        feats["tool_scores"] = {k: float(v) for k, v in top}
    return {k: v for k, v in feats.items()
            if k not in ("recent_selected", "tool_scores_all")}


def predict_split(split_name: str, out_path: str, limit: Optional[int] = None,
                  detailed: bool = True) -> None:
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
                "hit": bool(res.hit),
                "status": getattr(res, "status", "hit" if res.hit else "abstain"),
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

# Predict on EVAL_SPLIT (saving exact subset for matching baseline comparison)
predict_split(EVAL_SPLIT, os.path.join(RESULTS_DIR, "preds_eval.jsonl"), limit=EVAL_LIMIT)'''

MARKDOWN_SAMPLES_INTRO = """---
## 5c. Sample-Prediction Forensics

Dives into the per-sample prediction log `preds_eval.jsonl` written by the cell above (each row now carries `stratum`, `gt_tool`/`gt_args`, `status`, per-parameter `parameters` detail with `source_type`/`confidence`/`p_abstain`, and `features` latency/H1 breakdown). This is the manual, qualitative pass that turns aggregate numbers into hypotheses:
- **Hits** — what a correct payload looks like (sanity check that "hit" is real).
- **Misses by bucket** — wrong tool / wrong value / abstain: read the actual utterance vs predicted vs GT side by side.
- **Confidence calibration** — do high-confidence misses exist (dangerous) vs low-confidence abstains (safe)?"""

CODE_SAMPLES_FORENSICS = '''import json, os
import textwrap

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

    print(f"Loaded {len(preds)} prediction records from {PREDS_PATH}\\n")

    # ---- helper: readable utterance ----
    def _utt(sid):
        s = subs.get(sid)
        if s is None:
            return "(subset sample not available)"
        return textwrap.shorten(s.utterance, width=160, placeholder=" …")

    def _verdict(p):
        """Short human verdict for one prediction row."""
        if p.get("hit"):
            return "HIT"
        if p.get("status") == "partial":
            return "PARTIAL"
        if p.get("tool_name") and p.get("gt_tool") and p["tool_name"] != p["gt_tool"]:
            return "WRONG_TOOL"
        return "ABSTAIN"

    # ---- per-bucket counts ----
    from collections import Counter
    verdicts = Counter(_verdict(p) for p in preds)
    print("Verdict distribution:")
    for v, c in verdicts.most_common():
        print(f"  {v:<12}{c:>5}  ({c / max(1, len(preds)):.1%})")
    print()

    # ---- 1. sample hits (up to 5) ----
    hits = [p for p in preds if p.get("hit")]
    print("=" * 72)
    print(f"SAMPLE HITS ({len(hits)} total; showing up to 5)")
    print("=" * 72)
    for p in hits[:5]:
        print(f"\\n--- {p.get('sample_id')} [{p.get('tool_name')}] conf={p.get('confidence'):.3f} ---")
        print(f"  utt : {_utt(p.get('sample_id'))}")
        print(f"  gt  : {json.dumps(p.get('gt_args'), default=str)}")
        print(f"  pred: {json.dumps(p.get('payload'), default=str)}")

    # ---- 2. wrong-tool misses (up to 5) ----
    wrong = [p for p in preds if _verdict(p) == "WRONG_TOOL"]
    if wrong:
        print("\\n" + "=" * 72)
        print(f"WRONG-TOOL MISSES ({len(wrong)} total; showing up to 5)")
        print("=" * 72)
        for p in wrong[:5]:
            print(f"\\n--- {p.get('sample_id')} → predicted {p.get('tool_name')} (gt={p.get('gt_tool')}) conf={p.get('confidence'):.3f} ---")
            print(f"  utt : {_utt(p.get('sample_id'))}")
            if "features" in p and "tool_scores" in (p.get("features") or {}):
                print(f"  H1 scores: {p['features']['tool_scores']}")

    # ---- 3. wrong-value / partial misses (up to 5) ----
    part = [p for p in preds if _verdict(p) == "PARTIAL"]
    if part:
        print("\\n" + "=" * 72)
        print(f"PARTIAL / WRONG-VALUE MISSES ({len(part)} total; showing up to 5)")
        print("=" * 72)
        for p in part[:5]:
            print(f"\\n--- {p.get('sample_id')} [{p.get('tool_name')}] status={p.get('status')} ---")
            print(f"  utt : {_utt(p.get('sample_id'))}")
            print(f"  gt  : {json.dumps(p.get('gt_args'), default=str)}")
            print(f"  pred: {json.dumps(p.get('payload_partial'), default=str)}")

    # ---- 4. abstains (up to 5) + top abstain reasons ----
    abst = [p for p in preds if _verdict(p) == "ABSTAIN"]
    print("\\n" + "=" * 72)
    print(f"ABSTAINS ({len(abst)} total; showing up to 5)")
    print("=" * 72)
    from collections import Counter as _C
    reasons = _C((p.get("abstain_reason") or "no_reason") for p in abst)
    print("Top abstain reasons:")
    for r, c in reasons.most_common(6):
        print(f"  {r:<50}{c:>5}")
    for p in abst[:5]:
        print(f"\\n--- {p.get('sample_id')} [{p.get('tool_name')}] reason={p.get('abstain_reason')} ---")
        print(f"  utt : {_utt(p.get('sample_id'))}")
        print(f"  gt  : {json.dumps(p.get('gt_args'), default=str)}")

    # ---- 5. calibration: high-confidence misses ----
    errs = [p for p in preds if _verdict(p) in ("WRONG_TOOL", "PARTIAL") and p.get("confidence", 0) >= 0.9]
    if errs:
        print("\\n" + "=" * 72)
        print(f"HIGH-CONFIDENCE ERRORS (conf>=0.9; {len(errs)} total — dangerous if gated on confidence)")
        print("=" * 72)
        for p in sorted(errs, key=lambda x: -x.get("confidence", 0))[:5]:
            print(f"\\n--- {p.get('sample_id')} {_verdict(p)} conf={p.get('confidence'):.3f} ---")
            print(f"  utt : {_utt(p.get('sample_id'))}")
            print(f"  gt  : {json.dumps(p.get('gt_args'), default=str)}  pred: {json.dumps(p.get('payload_partial'), default=str)}")
    else:
        print("\\nNo high-confidence errors found — good calibration signal.")'''

MARKDOWN_CALIB_INTRO = """---
## 6. Calibration (Skipped in Raw Mode / Ablation Only)

Per Phase A1 (plans/accuracy_improvement_deep_dive.md), calibration is removed from the critical evaluation path. All model evaluation measures **raw accuracy** (`raw_mode=True`).
This section is preserved for ablation/reference only and is disabled by default."""

CODE_CALIB_ROWS = '''# Skipped in raw-accuracy pipeline (A1)
RUN_CALIBRATION_ABLATION = False

import json, os

CALIB_DATA = os.path.join(RESULTS_DIR, "calib_rows.jsonl")
CALIB_GT   = os.path.join(RESULTS_DIR, "calib_gt.jsonl")

if RUN_CALIBRATION_ABLATION:
    src = os.path.join(DATA_DIR, "hermes_calib.jsonl")
    preds, gts = [], []
    if os.path.exists(src):
        with open(src) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s = UnifiedSample.from_dict(json.loads(line))
                res = resolver.try_resolve_candidates(s.tools, s.utterance, s.recent_context, s.current_time_iso)

                # per-param gate probabilities (0.0 for missing required params)
                missing_req = set(getattr(res, "missing_required", []) or [])
                param_probs = {
                    p.name: (0.0 if p.name in missing_req else float(p.confidence))
                    for p in res.parameters.values()
                }
                preds.append({
                    "sample_id": s.sample_id,
                    "tool_name": res.tool_name,
                    "payload": res.payload,
                    "payload_partial": getattr(res, "payload_partial", {}),
                    "param_probabilities": param_probs,
                    "joint_margin": float(res.joint_margin),
                    "accept_probability": float(res.accept_score),
                    "schema_valid": bool(res.payload is not None),
                    "risk_tier": (s.tools[0].risk_tier if s.tools else "read_only"),
                })
                gt_args = s.ground_truth_calls[0].arguments if s.ground_truth_calls else {}
                gts.append({"sample_id": s.sample_id, "arguments": gt_args})

    with open(CALIB_DATA, "w") as fh:
        for p in preds:
            fh.write(json.dumps(p, default=str) + "\\n")
    with open(CALIB_GT, "w") as fh:
        for g in gts:
            fh.write(json.dumps(g, default=str) + "\\n")
    print(f"wrote {len(preds)} calib rows → {CALIB_DATA}")
else:
    print("§6 Calibration skipped (raw-accuracy mode active per Phase A1)")'''

CODE_CALIB_RUN = '''if RUN_CALIBRATION_ABLATION:
    CALIB_REPORT = os.path.join(CHECKPOINT_DIR or RESULTS_DIR, "calibration_report.json")
    import subprocess, sys
    subprocess.run([
        sys.executable, "scripts/calibrate.py",
        "--config", CONFIG_PATH,
        "--checkpoint", CHECKPOINT_DIR,
        "--calib_data", CALIB_DATA,
        "--ground_truth", CALIB_GT,
        "--output", CALIB_REPORT,
    ], check=False)

    if os.path.exists(CALIB_REPORT):
        import json
        rep = json.load(open(CALIB_REPORT))
        print("report keys:", sorted(rep.keys()))
        print("temperature:", rep.get("temperature"), "| tau_p:", rep.get("tau_p"),
              "| tau_a:", rep.get("tau_a"), "| tau_delta:", rep.get("tau_delta"))
else:
    print("Calibration fitting skipped")'''

MARKDOWN_EVAL_INTRO = """---
## 7. General Function Calling Evaluation (Eval Part 3)

Two runs of [`scripts/eval_toolcall.py`](../scripts/eval_toolcall.py) on identical subsets:

1. **B1a baseline** — deterministic regex extraction (`--baseline b1a`).
2. **SpanCall model** — `--predictions preds_eval.jsonl` generated with candidate resolution (`try_resolve_candidates`).

Both systems run against `subset_{EVAL_SPLIT}.jsonl` so metric comparisons are 100% apples-to-apples."""

CODE_EVAL_B1A = '''import subprocess, sys
# Evaluate B1a against the exact subset predicted by the model
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
    "--output", f"{RESULTS_DIR}/toolcall_metrics_{EVAL_SPLIT}_model.json",
], check=True)'''

MARKDOWN_CMP = """### Compare baselines

Load both metric files and print the headline table."""

CODE_CMP = '''import json, os

def _load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)

rows = []
for label, path in [
    ("B1a (regex)", os.path.join(RESULTS_DIR, f"toolcall_metrics_{EVAL_SPLIT}_b1a.json")),
    ("SpanCall (raw)", os.path.join(RESULTS_DIR, f"toolcall_metrics_{EVAL_SPLIT}_model.json")),
]:
    m = _load_metrics(path)
    if m is None:
        continue
    m_f = m.get("metrics_by_stratum", {}).get("F", {})
    rows.append({
        "system": label,
        "n_F": int(m_f.get("n", 0)),
        "exact_F": round(m_f.get("exact_payload_match", 0.0), 3),
        "hit_F": round(m_f.get("hit_rate", 0.0), 3),
        "param_F1_F": round(m_f.get("param_f1", 0.0), 3),
        "false_hit_on_A": round(m.get("false_hit_rate_on_a", 0.0), 4),
        "exact_all": round(m.get("exact_payload_match", 0.0), 3),
        "hit_all": round(m.get("hit_rate", 0.0), 3),
    })

if rows:
    headers = ["system", "n_F", "exact_F", "hit_F", "param_F1_F", "false_hit_on_A", "exact_all", "hit_all"]
    print(" | ".join(headers))
    print("-|" * len(headers))
    for r in rows:
        print(" | ".join(str(r[h]) for h in headers))
else:
    print("no metric files found yet — run the two eval_toolcall cells above")'''

MARKDOWN_EVAL_DETAIL_INTRO = """### Detailed eval forensics (errors, abstains, confidence)

Beyond the headline table: which errors dominate, which tools fail most, which abstain reasons dominate, and whether confidence separates hits from misses (ECE + calibration)."""
 
CODE_EVAL_DETAIL = '''import json, os
from collections import Counter, defaultdict

def _err_buckets(m):
    return {k: m.get(k, 0) for k in [
        "error_wrong_tool", "error_wrong_value", "error_entity_swap",
        "error_duplicate_span", "error_date_misparse", "error_extra_call", "error_missing_call"]}

def _conf_bins(preds, key="confidence", bins=(0.0, 0.5, 0.8, 0.9, 0.95, 1.01)):
    """Confidence histogram: is confidence separating hits from misses?"""
    out = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        sel = [p for p in preds if lo <= float(p.get(key, 0.0)) < hi]
        if not sel:
            continue
        n_hit = sum(1 for p in sel if p.get("hit"))
        n_ok = sum(1 for p in sel if p.get("payload_equal") or (p.get("hit") and p.get("tool_name") == p.get("gt_tool")))
        out.append((f"{lo:.2f}-{hi:.2f}", len(sel), n_hit, n_ok))
    return out

# ---- model metrics file ----
m_path = os.path.join(RESULTS_DIR, f"toolcall_metrics_{EVAL_SPLIT}_model.json")
if not os.path.exists(m_path):
    print("model metrics file not found — run the eval_toolcall cell first")
else:
    with open(m_path) as fh:
        m = json.load(fh)

    print("=" * 72)
    print(f"ERROR TAXONOMY — {EVAL_SPLIT} (SpanCall model)")
    print("=" * 72)
    errs = _err_buckets(m)
    tot_err = sum(errs.values())
    for k, v in sorted(errs.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"  {k:<22}{v:>5}  ({v / max(1, tot_err):.1%} of errors)")

    print("\\n" + "=" * 72)
    print("ABSTAIN REASONS (model)")
    print("=" * 72)
    for r, c in sorted(m.get("abstain_reasons", {}).items(), key=lambda kv: -kv[1]):
        print(f"  {r:<50}{c:>5}")

    print("\\n" + "=" * 72)
    print("CONFIDENCE CALIBRATION (model predictions vs hit/tool-ok)")
    print("=" * 72)
    p_path = os.path.join(RESULTS_DIR, f"preds_{EVAL_SPLIT}.jsonl")
    preds = []
    if os.path.exists(p_path):
        with open(p_path) as fh:
            preds = [json.loads(l) for l in fh if l.strip()]
    # join with ground truth from the subset, build tool-ok / payload-equal columns
    sub_path = os.path.join(RESULTS_DIR, f"subset_{EVAL_SPLIT}.jsonl")
    gt_map = {}
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
    if preds:
        for label, n, n_hit, n_ok in _conf_bins(preds):
            print(f"  conf {label:<10}{n:>5} rows | hits={n_hit:>4} ({n_hit / max(1, n):.2f}) | tool-ok={n_ok:>4} ({n_ok / max(1, n):.2f})")
        high_e = [p for p in preds if p.get("confidence", 0) >= 0.9 and not p.get("payload_equal")
                  and p.get("hit")]
        print(f"  high-confidence errors (conf>=0.9, hit, payload != GT): {len(high_e)}")
        for p in sorted(high_e, key=lambda x: -x.get("confidence", 0))[:5]:
            print(f"    {p.get('sample_id')} conf={p.get('confidence'):.3f} tool={p.get('tool_name')} "
                  f"pred={json.dumps(p.get('payload'), default=str)}")
    else:
        print("  no prediction rows available to score")

    # ---- per-tool e2e accuracy from metrics' errors_by_sample + preds ----
    print("\\n" + "=" * 72)
    print("PER-TOOL E2E (from preds joined with GT tool)")
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
              f"{st['exact']:>7}{st['exact'] / st['n']:>8.2f}")'''

MARKDOWN_TEST_INTRO = """---
## 8. Test — OOD / adversarial splits

Evaluate the trained checkpoint (or the B1a baseline) on the three held-out test splits:

- `hermes_test_ood_schema` — schema-disjoint OOD tools
- `hermes_test_ood_para` — paraphrase OOD
- `hermes_test_adversarial` — adversarial/ambiguity injection

> These are **unseen** at training time (partitioned by schema-disjoint split in `prepare_data.py`), so they measure generalization, not memorization."""

CODE_TEST_PREDS = '''# generate predictions for each test split (limit keeps runtime sane)
for split in TEST_SPLITS:
    predict_split(split, os.path.join(RESULTS_DIR, f"preds_{split}.jsonl"), limit=TEST_LIMIT)'''

CODE_TEST_EVAL = '''import subprocess, sys

def _run_eval(split: str, label: str, extra: list) -> None:
    out = f"{RESULTS_DIR}/toolcall_metrics_{split}_{label}.json"
    # Match test subset if it exists
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
    _run_eval(split, "model", ["--predictions", f"{RESULTS_DIR}/preds_{split}.jsonl"])

print("test eval done")'''

CODE_TEST_TABLE = '''import json, os

print(f"{'split':<26}{'system':<14}{'n_F':>6}{'exact_F':>9}{'hit_F':>8}{'false_A':>9}{'exact_all':>10}")
for split in TEST_SPLITS + [EVAL_SPLIT]:
    for label, suffix in [("b1a", "b1a"), ("model (raw)", "model")]:
        path = os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_{suffix}.json")
        m = _load_metrics(path)
        if not m:
            continue
        m_f = m.get("metrics_by_stratum", {}).get("F", {})
        n_f = int(m_f.get("n", 0))
        exact_f = m_f.get("exact_payload_match", 0.0)
        hit_f = m_f.get("hit_rate", 0.0)
        false_a = m.get("false_hit_rate_on_a", 0.0)
        exact_all = m.get("exact_payload_match", 0.0)
        print(f"{split:<26}{label:<14}{n_f:>6}"
              f"{exact_f:>9.3f}{hit_f:>8.3f}"
              f"{false_a:>9.4f}{exact_all:>10.3f}")'''

MARKDOWN_TEST_DETAIL_INTRO = """### Test-split forensics

For each OOD/adversarial split: the error-taxonomy breakdown, dominant abstain reasons, and sample prediction forensics (top misses with the actual utterance) — the qualitative signal on *how* generalization breaks, not just that it dropped."""

CODE_TEST_DETAIL = '''import json, os
from collections import Counter
import textwrap

def _err_buckets(m):
    return {k: m.get(k, 0) for k in [
        "error_wrong_tool", "error_wrong_value", "error_entity_swap",
        "error_duplicate_span", "error_date_misparse", "error_extra_call", "error_missing_call"]}

def _show_split(split: str, k: int = 4):
    m_path = os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_model.json")
    if not os.path.exists(m_path):
        print(f"\\n{split}: no model metrics file — run the test eval cell first")
        return
    with open(m_path) as fh:
        m = json.load(fh)
    print("\\n" + "=" * 72)
    print(f"{split} — model (raw)")
    print("=" * 72)
    m_f = m.get("metrics_by_stratum", {}).get("F", {})
    print(f"Stratum F: n={int(m_f.get('n', 0))} exact={m_f.get('exact_payload_match', 0):.3f} "
          f"hit={m_f.get('hit_rate', 0):.3f} | false_hit_on_A={m.get('false_hit_rate_on_a', 0):.4f}")

    errs = _err_buckets(m)
    tot_err = sum(errs.values())
    if tot_err:
        top = sorted(errs.items(), key=lambda kv: -kv[1])[:4]
        print("  top errors: " + ", ".join(f"{k}={v}" for k, v in top if v))
    else:
        print("  no errors (all abstains or perfect)")

    reas = m.get("abstain_reasons", {})
    if reas:
        print("  abstain reasons: " + ", ".join(f"{r}x{c}" for r, c in sorted(reas.items(), key=lambda kv: -kv[1])[:4]))

    # sample preds forensics
    p_path = os.path.join(RESULTS_DIR, f"preds_{split}.jsonl")
    sub_path = os.path.join(RESULTS_DIR, f"subset_{split}.jsonl")
    preds, subs = [], {}
    if os.path.exists(p_path):
        with open(p_path) as fh:
            preds = [json.loads(l) for l in fh if l.strip()]
    if os.path.exists(sub_path):
        from spancall.data.schema import UnifiedSample
        with open(sub_path) as fh:
            for l in fh:
                if l.strip():
                    d = json.loads(l)
                    subs[d.get("sample_id")] = UnifiedSample.from_dict(d)
    if preds:
        def _utt(sid):
            s = subs.get(sid)
            return textwrap.shorten(s.utterance, width=140, placeholder=" …") if s else "(n/a)"
        # misses: hit=False or payload mismatch — with the utterance for qualitative reading
        bad = []
        for p in preds:
            s = subs.get(p.get("sample_id"))
            gt = s.ground_truth_calls[0] if s and s.ground_truth_calls else None
            gt_args = gt.arguments if gt else {}
            if p.get("hit") and p.get("payload") == gt_args:
                continue
            bad.append(p)
        print(f"\\n  Sample misses ({len(bad)} of {len(preds)} rows; showing up to {k}):")
        for p in bad[:k]:
            tag = "WRONG_TOOL" if (p.get("tool_name") and p.get("gt_tool") and p["tool_name"] != p["gt_tool"]) else ("PARTIAL" if p.get("status") == "partial" else "ABSTAIN")
            print(f"    • {p.get('sample_id')} [{tag}] pred={p.get('tool_name')} gt={p.get('gt_tool')}")
            print(f"      utt: {_utt(p.get('sample_id'))}")
            if p.get("payload_partial"):
                print(f"      pred_payload: {json.dumps(p.get('payload_partial'), default=str)}")

for split in TEST_SPLITS:
    _show_split(split)
'''

MARKDOWN_PERSIST = """---
## 9. Persist to Drive (optional)

Mirror the artifacts (tensorized data, checkpoints, metrics) to the mounted Drive folder so nothing is lost when a Colab VM terminates."""

CODE_PERSIST = '''if DRIVE_MOUNT:
    import shutil, os
    os.makedirs(DRIVE_OUT, exist_ok=True)
    for name, src in [
        ("data_tensorized", TENSOR_DIR),
        ("results", RESULTS_DIR),
        ("data_processed", DATA_DIR),
    ]:
        if os.path.isdir(src):
            dst = os.path.join(DRIVE_OUT, name)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"mirrored {src} → {dst}")
    print("\\nartifacts persisted to", DRIVE_OUT)'''

MARKDOWN_LONG_TRAJ = """---
## 9b. Long-trajectory agent check (context plumbing)

Runs `scripts/eval_agent.py --benchmark long_trajectory` at 8k/12k/16k context budgets with the checkpoint-free scripted resolver (validates trajectory → `recent_context` feeding, per-turn stats, and LLM-path trimming). With a trained `CHECKPOINT_DIR`, drop `--scripted_resolver` and pass `--spancall_checkpoint` to measure real hit-rate (needs calibration first)."""

CODE_LONG_TRAJ = '''import subprocess, sys, json, os
for ctx, budget in [(8192, 4096), (12288, 6144), (16384, 8192)]:
    out = f"{RESULTS_DIR}/long_{ctx // 1024}k_nb.json"
    subprocess.run([
        sys.executable, "scripts/eval_agent.py", "--benchmark", "long_trajectory",
        "--scripted_resolver", "--max_context_tokens", str(ctx),
        "--context_budget_tokens", str(budget), "--output", out,
    ], check=False)
    if os.path.exists(out):
        r = json.load(open(out))
        print(ctx, {k: v.get("spancall_hits") for k, v in r.get("per_turn_stats", {}).items()})'''

MARKDOWN_TESTS = """---
## 10. Test suite (repo unit tests)

Runs the repo's own tests to confirm the checked-in code is healthy in this environment (template round-trip, masks, heads/loss shapes, gate, calibration)."""

CODE_TESTS = """!python -m pytest tests -q --disable-warnings --no-header 2>&1 | tail -25"""

CODE_SUMMARY = '''print(f"""
== SpanCall pipeline complete ==

What ran:
  0. git clone/pull + editable pip install
  1. env setup (.env loaded) + direct imports check
  2. HF dataset ({HF_DATASET_ID}) → {DATA_DIR}
  3. tensorizer → {TENSOR_DIR} (closes P7 gap)
  4. staged training T0/T1/T2/C → {RESULTS_DIR}
     └─ telemetry: epoch eval curve + loss components (train_log.jsonl)
  5. resolver predictions → {RESULTS_DIR}/preds_*.jsonl
     └─ forensics: per-sample hits/misses/abstains side-by-side
  6. (optional) calibration → calibration_report.json
  7. offline eval: b1a + model on {EVAL_SPLIT}
     └─ deep dive: error taxonomy, abstain reasons, confidence bins,
        per-tool e2e + RAW per-param / H1 breakdowns
  8. test splits: {TEST_SPLITS}
     └─ per-split forensics: top errors + sampled misses
  9. optional Drive mirror @ {DRIVE_OUT}
  9b. long-trajectory agent check
 10. repo unit tests
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
    new_code_cell(CODE_TENSOR_HELPERS),
    new_code_cell(CODE_TENSOR_RUN),
    new_markdown_cell(MARKDOWN_TENSOR_SANITY),
    new_code_cell(CODE_TENSOR_SANITY),
    new_markdown_cell(MARKDOWN_TRAIN),
    new_code_cell(CODE_COLAB_CFG),
    new_code_cell(CODE_TRAIN_RUN),
    new_markdown_cell(MARKDOWN_TRAIN_TELEMETRY),
    new_code_cell(CODE_TRAIN_TELEMETRY),
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
    new_markdown_cell(MARKDOWN_CALIB_INTRO),
    new_code_cell(CODE_CALIB_ROWS),
    new_code_cell(CODE_CALIB_RUN),
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
    new_markdown_cell(MARKDOWN_TEST_DETAIL_INTRO),
    new_code_cell(CODE_TEST_DETAIL),
    new_markdown_cell(MARKDOWN_PERSIST),
    new_code_cell(CODE_PERSIST),
    new_markdown_cell(MARKDOWN_LONG_TRAJ),
    new_code_cell(CODE_LONG_TRAJ),
    new_markdown_cell(MARKDOWN_TESTS),
    new_code_cell(CODE_TESTS),
    new_code_cell(CODE_SUMMARY),
]

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
