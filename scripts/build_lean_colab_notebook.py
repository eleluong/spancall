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
try:
    _cfg_yaml = yaml.safe_load(open("configs/default.yaml")) or {}
    MAX_POS   = int((_cfg_yaml.get("optimization") or {}).get("max_pos", 3))
except Exception:
    MAX_POS   = 3

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
from spancall.lean.tensorize import tensorize_sample_lean, has_positive_targets

tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_MARKER_TOKENS)})
mask_builder = CandidateMaskBuilder(max_seq_len=MAX_SEQ_LEN, max_span_width=MAX_SPAN_WIDTH)
print("tokenizer ready; vocab =", len(tokenizer), "| MAX_SEQ_LEN =", MAX_SEQ_LEN)'''

CODE_TENSOR_RUN = '''# ---- tensorize splits into Lean 0-indexed format (parallel fork pool) ----

def _iter_raw_lines(path: str):
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
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
    """Tensorize one sample; returns (row_or_None, param_drop_reasons, errors).

    The drop-reason audit travels back through the return value because the
    fork pool cannot mutate a parent dict. It is reported even for rows that
    are filtered out (all params dropped => no positives), which is exactly
    the case that used to vanish silently.

    Exceptions are NO LONGER swallowed. The exception class name is counted
    and returned so a tokenizer/schema mismatch shows up as a number instead
    of silently shrinking the training set.
    """
    d, pack_multi = args
    drops = {}
    errors = {}
    try:
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
        if row is None or not has_positive_targets(row):
            return None, drops, errors
        return _serialize(row), drops, errors
    except Exception as exc:  # noqa: BLE001 - counted, not swallowed
        cls = type(exc).__name__
        errors[cls] = errors.get(cls, 0) + 1
        return None, drops, errors

_CHUNK = 256

def _tensorize_many(raw, pack_multi: bool, need, workers: int):
    if need is None:
        need = len(raw)
    rows = []
    drops_total = Counter()
    errors_total = Counter()

    def _take(res):
        row, drops, errors = res
        for k, v in (drops or {}).items():
            drops_total[k] += v
        for k, v in (errors or {}).items():
            errors_total[k] += v
        if row is not None:
            rows.append(row)
        return len(rows) >= need

    ctx = None
    if workers > 1:
        try:
            ctx = mp.get_context("fork")
        except Exception:
            ctx = None
    if ctx is not None:
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                done = False
                for start in range(0, len(raw), _CHUNK):
                    batch = raw[start:start + _CHUNK]
                    for res in ex.map(_tensorize_lean_task, [(d, pack_multi) for d in batch]):
                        if _take(res):
                            done = True
                            break
                    if done:
                        break
            return rows, drops_total, errors_total
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"  [!] process pool failed ({type(exc).__name__}: {exc}); "
                  f"falling back to single-process")
            rows = []
            drops_total = Counter()
            errors_total = Counter()
    for d in raw:
        if _take(_tensorize_lean_task((d, pack_multi))):
            break
    return rows, drops_total, errors_total

groups = defaultdict(list)
for out_name, in_name in SPLIT_INPUTS.items():
    pack_multi = True
    groups[(in_name, pack_multi)].append(out_name)

os.makedirs(TENSOR_DIR, exist_ok=True)
stats = {}
drop_summary = {}
error_summary = {}

for (in_name, pack_multi), out_names in groups.items():
    src = os.path.join(DATA_DIR, in_name)
    if not os.path.exists(src):
        for on in out_names:
            print(f"SKIP {on}: {src} missing")
        continue
    raw = _fraction_subset(_iter_raw_lines(src), DATA_FRACTION, SEED)
    need = None
    for on in out_names:
        lim = SAMPLE_LIMITS.get(on)
        if lim is not None:
            need = lim if need is None else max(need, lim)
    rows, drops, errors = _tensorize_many(raw, pack_multi, need, TENSORIZE_WORKERS)
    missed = int(drops.get("match_missed", 0))
    unresolved = int(drops.get("unresolved_target", 0))
    drop_summary[in_name] = dict(drops)
    if errors:
        error_summary[in_name] = dict(errors)
    for on in out_names:
        limit = SAMPLE_LIMITS.get(on)
        out_rows = rows if limit is None else rows[:limit]
        out_path = os.path.join(TENSOR_DIR, f"{on}.jsonl")
        with open(out_path, "w") as fh:
            for r in out_rows:
                fh.write(json.dumps(r) + "\\n")
        stats[on] = {
            "rows": len(out_rows),
            "filtered_or_dropped": len(raw) - len(rows),
            "raw_after_fraction": len(raw),
            "fraction": DATA_FRACTION,
            "pack_multi": pack_multi,
            "limit": limit,
            "param_drop_match_missed": missed,
            "param_drop_unresolved": unresolved,
            "param_drop_other": {k: v for k, v in drops.items()
                                 if k not in ("match_missed", "unresolved_target")},
            "tensorize_errors": dict(errors),
        }
        print(f"wrote {on}.jsonl: {len(out_rows)} groundable rows "
              f"({len(raw) - len(rows)} ungroundable/Stratum-A/over-length filtered)")
    if missed:
        print(f"  [!] MATCH_MISSED={missed} on {in_name}: extractable targets the matcher "
              f"failed on -> these params get NO pointer supervision (real bug).")
    other = {k: v for k, v in drops.items()
             if k not in ("match_missed", "unresolved_target")}
    for k in sorted(other):
        label = "BUG" if k in ("unaligned_target", "null_slot_missing") else "note"
        print(f"  [{label}] {k}={other[k]} on {in_name}")
    if unresolved:
        print(f"  [i] unresolved_target={unresolved} on {in_name}: needs generation/transform; "
              f"out of scope for the retrieval-first fast path (correct exclusion).")
    for cls_name in sorted(errors):
        print(f"  [!!] tensorize raised {cls_name} x{errors[cls_name]} on {in_name} "
              f"-> samples silently lost from training. Investigate.")

print("\\nLean tensorize summary:", json.dumps(stats, indent=2))
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
- Uses the same optimizer hyperparameters (`num_epochs=5`, `batch_size=8`, `gradient_accumulation_steps=4`, `bf16`, `CosineWarmupSchedule`) as the legacy notebook for a clean apples-to-apples comparison."""

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
cfg["optimization"]["batch_size"] = 8
cfg["optimization"]["mixed_precision"] = "bf16"
cfg["training"]["gradient_accumulation_steps"] = 4
cfg["training"]["gradient_checkpointing"] = True
cfg["training"]["eval_splits"] = ["dev"]
cfg["training"]["eval_limit"] = 500
cfg["model"]["freeze_encoder_stages"] = []
cfg["model"]["max_seq_len"] = MAX_SEQ_LEN

cfg.setdefault("data", {})["fraction"] = TRAIN_FRACTION

out = "configs/colab_lean.yaml"
with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh)
print(f"wrote {out} (epochs={NUM_EPOCHS}, bs=8, accum=4, bf16=True, freeze=[], data_fraction={TRAIN_FRACTION})")
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
- Per-epoch held-out eval curve (`span_acc`, `exact_payload_match`)
- Per-epoch mean train loss
- LR + loss-component trajectory (`pointer`, `tool`)"""

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
        print(f"  {'stage':<8}{'epoch':>6}{'n':>6}{'span_acc':>10}{'exact_payload':>15}")
        for r in evals:
            print(f"  {str(r.get('stage','?')):<8}{r.get('epoch','?'):>6}{r.get('n','?'):>6}"
                  f"{r.get('span_acc', 0):>10.3f}{r.get('exact_payload_match', 0):>15.3f}")
    else:
        print("  no epoch_eval events")

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

resolver = LeanResolver(
    model_path=CHECKPOINT_DIR,
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
print(f"fast_path_param_coverage = {_m('fast_path_param_coverage')}")
print(f"fast_path_call_coverage  = {_m('fast_path_call_coverage')}")
print(f"correct refusals         = {_m('refused_unextractable')} "
      f"| real fast-path misses = {_m('missed_extractable')} "
      f"| pointer_wrong = {_m('pointer_wrong')}")
print(f"E3 aligner drift         = {_m('e3_aligner_drift')} sample(s) "
      f"(>0 is a DATA bug, not a model miss)")'''

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

    print(f"=== Lean Tool Prediction (H1) on {split_name} (N={total}) — >=2 candidates ===")
    print(f"Top-1 Tool Accuracy (Overall): {acc_all:.4f} ({(single_correct + multi_correct)}/{total})")
    print(f"  Multi-tool subset: {acc_multi:.4f} ({multi_correct}/{multi_total})")
    print(f"  Single+distractor subset: {acc_single:.4f} ({single_correct}/{single_total})")
    print(f"Mean Winner Tool Probability: {mean_p:.4f} | Mean Margin: {mean_m:.4f}")
    print(f"Rows with empty tool_scores: {empty_scores}")
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
    print("no metric files found yet — run the eval_toolcall cells above")'''

MARKDOWN_EVAL_DETAIL_INTRO = """### Detailed Eval Forensics (Error Taxonomy & Per-Tool E2E)"""

CODE_EVAL_DETAIL = '''import json, os
from collections import defaultdict

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
              f"{st['exact']:>7}{st['exact'] / st['n']:>8.2f}")'''

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

