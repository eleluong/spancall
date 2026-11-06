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

print(f"PROJECT_ROOT  = {PROJECT_ROOT}")
print(f"HF_DATASET_ID = {HF_DATASET_ID} (token={'set' if HF_TOKEN else 'unset'})")
print(f"DATA_DIR      = {DATA_DIR}")
print(f"TENSOR_DIR    = {TENSOR_DIR}")
print(f"DEVICE        = {DEVICE}")'''

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

FORCE_HF_DOWNLOAD = False  # set True to always re-pull from Hugging Face Hub

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

BACKBONE = "Qwen/Qwen3-Embedding-0.6B"
MAX_SEQ_LEN, MAX_SPAN_WIDTH = 4096, 16

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


MAX_POS = 3   # positive slots per param (pointer P in the v1 loss)


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


def _sample_seq_len(s: UnifiedSample, tokenizer) -> int:
    """Compute tokenized sequence length of a sample (with fast char lower-bound)."""
    raw_chars = len(s.utterance or "") + sum(
        len(str(r.get("key", ""))) + len(str(r.get("value", ""))) + 16
        for r in (s.recent_context or [])
    )
    if raw_chars < (MAX_SEQ_LEN // 2):
        return raw_chars
    tool = s.tools[0] if s.tools else None
    if s.ground_truth_calls and s.tools:
        gt_name = s.ground_truth_calls[0].tool_name
        tool = next((t for t in s.tools if t.name == gt_name), tool)
    if tool is not None:
        try:
            out = build_input(
                tooldef_to_template_tool(tool),
                s.utterance,
                recent_context=_recent_rows(s),
                now=_now_iso(s),
            )
            return len(tokenizer(out.text, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    return len(tokenizer(s.utterance or "", add_special_tokens=False)["input_ids"])


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

        pos_cols = [ci + 1 for ci in matched_cis[:MAX_POS]]
        if s.stratum in STRATUM_ABSTAIN or not pos_cols:
            pos_cols = [0]

        pos_rows.append((pos_cols + [-1] * MAX_POS)[:MAX_POS])
        reasons.append(_h4(s, param.name))
        cue.append(1.0)

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

CODE_TENSOR_RUN = '''# ---- filter out samples > 4K tokens & tensorize all splits --------------
import json, os

# 1. Remove any samples exceeding MAX_SEQ_LEN (4K) from canonical splits in DATA_DIR
for split_file in CANONICAL_SPLITS:
    split_path = os.path.join(DATA_DIR, split_file)
    if not os.path.exists(split_path):
        continue
    kept_lines, removed_long = [], 0
    with open(split_path, "r") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            s = UnifiedSample.from_dict(json.loads(raw))
            if _sample_seq_len(s, tokenizer) > MAX_SEQ_LEN:
                removed_long += 1
                continue
            kept_lines.append(raw)
    if removed_long > 0:
        with open(split_path, "w") as fh:
            for l in kept_lines:
                fh.write(l + "\\n")
        print(f"filtered {split_file}: kept {len(kept_lines)}, removed {removed_long} (> {MAX_SEQ_LEN} tokens)")

SPLIT_INPUTS = {
    "t0":       "hermes_train.jsonl",
    "t1":       "hermes_train.jsonl",
    "t2":       "hermes_train.jsonl",
    "stage_c":  "hermes_calib.jsonl",
    "dev":      "hermes_dev.jsonl",
    "calib":    "hermes_calib.jsonl",
}

# Full dataset training without subsampling limits (None)
SAMPLE_LIMITS = {"t0": None, "t1": None, "t2": None, "stage_c": None, "dev": None, "calib": None}

os.makedirs(TENSOR_DIR, exist_ok=True)


def _iter_samples(path: str):
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield UnifiedSample.from_dict(json.loads(line))


def _serialize(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}


stats = {}
for out_name, in_name in SPLIT_INPUTS.items():
    src = os.path.join(DATA_DIR, in_name)
    if not os.path.exists(src):
        print(f"SKIP {out_name}: {src} missing")
        continue
    limit = SAMPLE_LIMITS.get(out_name)
    rows, dropped = [], 0
    # Enable pack_multi_tools for t2 and dev so H1 tool head trains and evaluates on candidate selection
    pack_multi = out_name in ("t2", "dev")
    for s in _iter_samples(src):
        row = tensorize_sample(s, tokenizer, mask_builder, pack_multi_tools=pack_multi)
        if row is None:
            dropped += 1
            continue
        rows.append(_serialize(row))
        if limit is not None and len(rows) >= limit:
            break
    out_path = os.path.join(TENSOR_DIR, f"{out_name}.jsonl")
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\\n")
    stats[out_name] = {"rows": len(rows), "dropped": dropped}
    print(f"wrote {out_name}.jsonl: {len(rows)} rows ({dropped} dropped)")

print("\\ntensorize summary:", json.dumps(stats, indent=2))'''

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

EXPERIMENT_NAME = "spancall_colab"
SEED = 42

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
cfg["training"]["gradient_accumulation_steps"] = 4
cfg["training"]["gradient_checkpointing"] = True
cfg["model"]["freeze_encoder_stages"] = []
cfg["model"]["max_seq_len"] = MAX_SEQ_LEN

out = "configs/colab.yaml"
with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh)
print(f"wrote {out} (epochs={NUM_EPOCHS}, bs=8, accum=4, bf16=True, freeze=[])")
CONFIG_PATH = out'''

CODE_TRAIN_RUN = '''# Direct in-notebook training execution (provides live logging and full native tracebacks)
from scripts.train import main as train_main

train_main([
    "--config", CONFIG_PATH,
    "--experiment_name", EXPERIMENT_NAME,
    "--output_dir", RESULTS_DIR,
    "--data_dir", TENSOR_DIR,
    "--stages", "T0,T1,T2,C",
    "--device", DEVICE,
    "--seed", str(SEED),
])'''

MARKDOWN_PICK_CP = """### Pick the best checkpoint

The trainer writes checkpoints per stage under `results/{exp}_seed{seed}/{stage}/checkpoints/`. We pick:

1. The **stage-C checkpoint** if it exists (full curriculum, frozen-encoder H8), else
2. the **last stage with a checkpoint** (T2 → T1 → T0), else
3. the `model_init` dir saved by `scripts/train.py`.

`CHECKPOINT_DIR` is then used by prediction/calibration/eval."""

CODE_PICK_CP = '''import glob, os

exp_root = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_seed{SEED}")

def _latest_stage_cp():
    for stage in ["C", "T2", "T1", "T0"]:
        cps = sorted(glob.glob(os.path.join(exp_root, stage, "checkpoints", "**", "model.pt"), recursive=True))
        if cps:
            return os.path.dirname(cps[-1]), stage
    return None, None

CHECKPOINT_DIR = None
cp_dir, cp_stage = _latest_stage_cp()
model_init = os.path.join(RESULTS_DIR, "model_init")

if cp_dir is not None:
    CHECKPOINT_DIR = cp_dir
    print(f"picked stage-{cp_stage} checkpoint @ {CHECKPOINT_DIR}")
elif os.path.isdir(model_init):
    CHECKPOINT_DIR = model_init
    print("no staged checkpoint; using model_init @", CHECKPOINT_DIR)
else:
    all_m = sorted(glob.glob(os.path.join(RESULTS_DIR, "**", "model.pt"), recursive=True))
    if all_m:
        CHECKPOINT_DIR = os.path.dirname(all_m[-1])
        print("fallback checkpoint @", CHECKPOINT_DIR)

print("CHECKPOINT_DIR =", CHECKPOINT_DIR)
if CHECKPOINT_DIR:
    print("  contains:", sorted(os.listdir(CHECKPOINT_DIR)))'''

MARKDOWN_PREDICT_INTRO = """---
## 5. Inference & Evaluation

To evaluate correctly without conflating failure modes, the evaluation is broken into three distinct, separate parts:
1. **Eval Part 1 — Parameter Extraction Evaluation (Oracle / Given GT Tool)**: Tests parameter extraction (`resolver.try_resolve(gt_tool, ...)`) independently of tool selection, reporting ungated/partial extraction (`param_precision`, `param_recall`, `param_f1`, `partial_extraction_rate`, `exact_payload_match` on `Stratum.F`, mean `confidence` on `Stratum.F` vs `Stratum.A`).
2. **Eval Part 2 — Function (Tool) Prediction Evaluation**: Tests H1 tool selection (`resolver.predict_tool(s.tools, ...)`) across candidate tool schemas independently, reporting top-1 tool accuracy overall and on multi-tool samples (`len(s.tools) > 1`), mean winning tool probability, tool margin (`p_top1 - p_top2`), and gated tool coverage/accuracy.
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

# calib_thresholds_path → calibration_report.json from §6 (optional)
calib_path = None
if CHECKPOINT_DIR and os.path.exists(os.path.join(CHECKPOINT_DIR, "calibration_report.json")):
    calib_path = os.path.join(CHECKPOINT_DIR, "calibration_report.json")

resolver = SpanCallResolver(
    model_path=CHECKPOINT_DIR,
    calib_thresholds_path=calib_path,
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
## 5a. Evaluation Part 1 — Parameter Extraction (Oracle Tool Given)

Evaluates parameter extraction directly on ground-truth tools using `resolver.try_resolve(gt_tool, ...)`, completely decoupling parameter extraction from function/tool prediction."""

CODE_EVAL_PART1 = '''import json, os
from spancall.data.schema import Stratum, UnifiedSample

def eval_param_extraction(split_name: str, limit: Optional[int] = 500):
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
    partial_extracted = 0
    conf_f = []
    conf_a = []
    
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
            pred_payload = res.payload_partial or {}
            gt_args = gt_call.arguments or {}
            
            if s.stratum == Stratum.F:
                stratum_f_total += 1
                conf_f.append(res.confidence)
                if pred_payload == gt_args:
                    stratum_f_exact += 1
            else:
                conf_a.append(res.confidence)
                
            if pred_payload:
                partial_extracted += 1
                
            # Parameter-level precision/recall
            for k, v in pred_payload.items():
                if k in gt_args and str(gt_args[k]).strip() == str(v).strip():
                    param_tp += 1
                else:
                    param_fp += 1
            for k, v in gt_args.items():
                if k not in pred_payload:
                    param_fn += 1
                    
            if limit is not None and total >= limit:
                break
                
    p = param_tp / max(1, param_tp + param_fp)
    r = param_tp / max(1, param_tp + param_fn)
    f1 = 2 * p * r / max(1e-8, p + r)
    f_exact = stratum_f_exact / max(1, stratum_f_total)
    mean_cf = sum(conf_f) / max(1, len(conf_f))
    mean_ca = sum(conf_a) / max(1, len(conf_a))
    partial_rate = partial_extracted / max(1, total)
    
    print(f"=== Parameter Extraction Evaluation (Oracle Tool) on {split_name} (N={total}) ===")
    print(f"Param Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    print(f"Exact Payload Match (Stratum F): {f_exact:.4f} ({stratum_f_exact}/{stratum_f_total})")
    print(f"Partial Extraction Rate: {partial_rate:.4f} ({partial_extracted}/{total})")
    print(f"Mean Confidence on Stratum F (Solvable): {mean_cf:.4f}")
    print(f"Mean Confidence on Stratum A (Unsolvable): {mean_ca:.4f}")
    print("================================================================================")

eval_param_extraction(EVAL_SPLIT, limit=500)'''

MARKDOWN_EVAL_PART2 = """---
## 5b. Evaluation Part 2 — Function (Tool) Prediction Evaluation

Evaluates H1 tool selection across candidate tool schemas independently, isolating tool choice accuracy from parameter extraction."""

CODE_EVAL_PART2 = '''import json, os
from spancall.data.schema import UnifiedSample

def eval_tool_prediction(split_name: str, limit: Optional[int] = 500):
    src = os.path.join(DATA_DIR, f"{split_name}.jsonl")
    if not os.path.exists(src):
        print(f"File not found: {src}")
        return
        
    total = 0
    multi_total = 0
    top1_correct = 0
    multi_top1_correct = 0
    probs = []
    margins = []
    gated_correct = 0
    gated_hits = 0
    
    with open(src) as fh:
        for line in fh:
            if not line.strip():
                continue
            s = UnifiedSample.from_dict(json.loads(line))
            if not s.ground_truth_calls or not s.tools:
                continue
            gt_tool_name = s.ground_truth_calls[0].tool_name
            total += 1
            is_multi = len(s.tools) > 1
            if is_multi:
                multi_total += 1
                
            tool_pred = resolver.predict_tool(s.tools, s.utterance, s.recent_context, s.current_time_iso)
            pred_name = tool_pred["tool_name"]
            is_correct = (pred_name == gt_tool_name)
            
            if is_correct:
                top1_correct += 1
                if is_multi:
                    multi_top1_correct += 1
                    
            probs.append(tool_pred["tool_prob"])
            margins.append(tool_pred["tool_margin"])
            
            if tool_pred["gated_hit"]:
                gated_hits += 1
                if is_correct:
                    gated_correct += 1
                    
            if limit is not None and total >= limit:
                break
                
    acc_all = top1_correct / max(1, total)
    acc_multi = multi_top1_correct / max(1, multi_total)
    mean_p = sum(probs) / max(1, len(probs))
    mean_m = sum(margins) / max(1, len(margins))
    gated_cov = gated_hits / max(1, total)
    gated_prec = gated_correct / max(1, gated_hits)
    
    print(f"=== Function / Tool Prediction (H1) on {split_name} (N={total}) ===")
    print(f"Top-1 Tool Accuracy (Overall): {acc_all:.4f} ({top1_correct}/{total})")
    print(f"Top-1 Tool Accuracy (Multi-tool subset, N={multi_total}): {acc_multi:.4f} ({multi_top1_correct}/{multi_total})")
    print(f"Mean Winner Tool Probability: {mean_p:.4f} | Mean Margin: {mean_m:.4f}")
    print(f"Gated Coverage: {gated_cov:.4f} | Precision@Gated: {gated_prec:.4f}")
    print("==================================================================")

eval_tool_prediction(EVAL_SPLIT, limit=500)'''

CODE_PREDICT_SPLIT = '''def predict_split(split_name: str, out_path: str, limit: Optional[int] = None) -> None:
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
            preds.append({
                "sample_id": s.sample_id,
                "hit": bool(res.hit),
                "status": getattr(res, "status", "hit" if res.hit else "abstain"),
                "payload": res.payload,
                "payload_partial": getattr(res, "payload_partial", {}),
                "missing_required": getattr(res, "missing_required", []),
                "missing_optional": getattr(res, "missing_optional", []),
                "tool_name": res.tool_name,
                "confidence": float(getattr(res, "confidence", res.accept_score)),
                "accept_probability": float(res.accept_score),
                "abstain_reason": res.abstain_reason,
            })
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
    print(f"predicted {n} rows → {out_path} and saved subset → {sub_src}")

# Predict on EVAL_SPLIT (saving exact subset for matching baseline comparison)
predict_split(EVAL_SPLIT, os.path.join(RESULTS_DIR, "preds_eval.jsonl"), limit=500)'''

MARKDOWN_CALIB_INTRO = """---
## 6. Calibration (optional)

Produces `calibration_report.json` (temperature + per-tier thresholds) which the resolver can then consume via `calib_thresholds_path`.

The calibrator expects **resolver prediction rows** with this contract (see [`spancall/train/calibrate.py`](../spancall/train/calibrate.py)):
```
{sample_id, tool_name, payload, param_probabilities, joint_margin, accept_probability, schema_valid, risk_tier}
+ parallel ground truth {sample_id, arguments}
```

> If you prefer to skip temperature fitting, pass `--temperature 1.0` and the report still sweeps tau_p/tau_a/tau_delta."""

CODE_CALIB_ROWS = '''import json, os

# Build calibration rows from the calib split via the resolver
CALIB_DATA = os.path.join(RESULTS_DIR, "calib_rows.jsonl")
CALIB_GT   = os.path.join(RESULTS_DIR, "calib_gt.jsonl")

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
print(f"wrote {len(preds)} calib rows → {CALIB_DATA}")'''

CODE_CALIB_RUN = '''CALIB_REPORT = os.path.join(CHECKPOINT_DIR or RESULTS_DIR, "calibration_report.json")

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
          "| tau_a:", rep.get("tau_a"), "| tau_delta:", rep.get("tau_delta"))'''

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
    ("SpanCall",    os.path.join(RESULTS_DIR, f"toolcall_metrics_{EVAL_SPLIT}_model.json")),
]:
    m = _load_metrics(path)
    if m is None:
        continue
    rows.append({
        "system": label,
        "n": m.get("total_samples"),
        "exact_payload_match": round(m.get("exact_payload_match", 0.0), 3),
        "hit_rate": round(m.get("hit_rate", 0.0), 3),
        "precision_at_hit": round(m.get("precision_at_hit", 0.0), 3),
        "param_P": round(m.get("param_precision", 0.0), 3),
        "param_R": round(m.get("param_recall", 0.0), 3),
        "param_F1": round(m.get("param_f1", 0.0), 3),
        "false_hit_on_A": round(m.get("false_hit_rate_on_a", 0.0), 4),
        "ECE": round(m.get("ece", 0.0), 4),
    })

if rows:
    headers = ["system", "n", "exact_payload_match", "hit_rate", "precision_at_hit", "param_P", "param_R", "param_F1", "false_hit_on_A", "ECE"]
    print(" | ".join(headers))
    print("-|" * len(headers))
    for r in rows:
        print(" | ".join(str(r[h]) for h in headers))
else:
    print("no metric files found yet — run the two eval_toolcall cells above")'''

MARKDOWN_TEST_INTRO = """---
## 8. Test — OOD / adversarial splits

Evaluate the trained checkpoint (or the B1a baseline) on the three held-out test splits:

- `hermes_test_ood_schema` — schema-disjoint OOD tools
- `hermes_test_ood_para` — paraphrase OOD
- `hermes_test_adversarial` — adversarial/ambiguity injection

> These are **unseen** at training time (partitioned by schema-disjoint split in `prepare_data.py`), so they measure generalization, not memorization."""

CODE_TEST_PREDS = '''# generate predictions for each test split (limit keeps runtime sane)
for split in TEST_SPLITS:
    predict_split(split, os.path.join(RESULTS_DIR, f"preds_{split}.jsonl"), limit=200)'''

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

print(f"{'split':<26}{'system':<8}{'n':>6}{'exact':>8}{'hit':>7}{'P@hit':>8}{'falseA':>9}{'ECE':>8}")
for split in TEST_SPLITS + [EVAL_SPLIT]:
    for label, suffix in [("b1a", "b1a"), ("model", "model")]:
        path = os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_{suffix}.json")
        m = _load_metrics(path)
        if not m:
            continue
        n = m.get("total_samples", "-")
        print(f"{split:<26}{label:<8}{n:<6}"
              f"{m.get('exact_payload_match', 0.0):<8.3f}{m.get('hit_rate', 0.0):<7.3f}"
              f"{m.get('precision_at_hit', 0.0):<8.3f}{m.get('false_hit_rate_on_a', 0.0):<9.4f}"
              f"{m.get('ece', 0.0):<8.4f}")'''

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
  5. resolver predictions → {RESULTS_DIR}/preds_*.jsonl
  6. (optional) calibration → calibration_report.json
  7. offline eval: b1a + model on {EVAL_SPLIT}
  8. test splits: {TEST_SPLITS}
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
    new_markdown_cell(MARKDOWN_PICK_CP),
    new_code_cell(CODE_PICK_CP),
    new_markdown_cell(MARKDOWN_PREDICT_INTRO),
    new_code_cell(CODE_RESOLVER),
    new_markdown_cell(MARKDOWN_EVAL_PART1),
    new_code_cell(CODE_EVAL_PART1),
    new_markdown_cell(MARKDOWN_EVAL_PART2),
    new_code_cell(CODE_EVAL_PART2),
    new_code_cell(CODE_PREDICT_SPLIT),
    new_markdown_cell(MARKDOWN_CALIB_INTRO),
    new_code_cell(CODE_CALIB_ROWS),
    new_code_cell(CODE_CALIB_RUN),
    new_markdown_cell(MARKDOWN_EVAL_INTRO),
    new_code_cell(CODE_EVAL_B1A),
    new_code_cell(CODE_EVAL_MODEL),
    new_markdown_cell(MARKDOWN_CMP),
    new_code_cell(CODE_CMP),
    new_markdown_cell(MARKDOWN_TEST_INTRO),
    new_code_cell(CODE_TEST_PREDS),
    new_code_cell(CODE_TEST_EVAL),
    new_code_cell(CODE_TEST_TABLE),
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
