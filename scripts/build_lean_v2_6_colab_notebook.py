#!/usr/bin/env python3
"""Generate notebooks/spancall_lean_v2_6_train_eval.ipynb via nbformat (valid JSON guaranteed).

SpanCall Lean Flow **v2.6** — Hermes + xLAM-60k (1-Call) ModernBERT-large Pipeline:
- Adds configurable `Salesforce/xlam-function-calling-60k` (single-call subset) training/dev mixing (`XLAM_TRAIN_ROWS`, `XLAM_DEV_ROWS`) alongside Hermes V1.
- Evaluates on `hermes_eval`, `hermes_test_ood_schema`, `xlam_eval`, and `xlam_test_ood_schema`.
"""
from pathlib import Path
import json

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parent / ".." / "notebooks" / "spancall_lean_v2_6_train_eval.ipynb"

# --------------------------------------------------------------------------
# Cell sources
# --------------------------------------------------------------------------

MARKDOWN_TITLE = """# SpanCall Lean Flow **v2.6** — Hermes + xLAM-60k (1-Call) ModernBERT-large Pipeline

End-to-end training and step-by-step evaluation notebook extending **v2.5** with **Salesforce/xlam-function-calling-60k (1-call subset)** for multi-tool H1 and schema-generalized H3 supervision:
- **Backbone**: `answerdotai/ModernBERT-large` (native bidirectional attention, `MAX_SEQ_LEN = 4096`).
- **Dataset Mix**: `hermes_train.jsonl` + configurable `XLAM_TRAIN_ROWS` (default `15000`; set `0` for Hermes-only baseline, `5000`/`15000` for ablations, `None` for full 69k xLAM train).
- **Evaluation**: `hermes_eval` step-by-step waterfall + OOD/cross-corpus splits (`hermes_test_ood_schema`, `xlam_eval`, `xlam_test_ood_schema`).
"""

MARKDOWN_SETUP = """---
## 0. Setup & Configuration

Clones or updates the SpanCall repository, installs dependencies in editable mode, and configures v2.6 paths (`data/tensorized_lean_v2_6` and `results_lean_v2_6`)."""

CODE_SETUP = '''# 0a. Pull latest code from public GitHub repository and install dependencies
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
%pip install -q -e ".[model,data,agent,dev]" huggingface_hub'''

CODE_CONFIG = '''# 0b. Configuration & Environment (Full Data)
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

from spancall.agent.backends.litellm_adapter import _load_dotenv_values

ENV = {**_load_dotenv_values(PROJECT_ROOT / ".env"), **os.environ}
HF_DATASET_ID = ENV.get("DATASET_NAME", "convoicon/spancall")
HF_TOKEN      = ENV.get("HUGGING_FACE_TOKEN") or ENV.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

DRIVE_MOUNT       = False
DRIVE_OUT         = "/content/drive/MyDrive/spancall_lean_artifacts"

DATA_DIR          = str(PROJECT_ROOT / "data" / "processed_v2")
V1_DATA_DIR       = str(PROJECT_ROOT / "data" / "processed")
TENSOR_DIR        = str(PROJECT_ROOT / "data" / "tensorized_lean_v2_6")
RESULTS_DIR       = str(PROJECT_ROOT / "results_lean_v2_6")
EVAL_SPLIT        = "hermes_eval"
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
DATA_FRACTION     = 1.0
TENSORIZE_WORKERS = max(1, min(os.cpu_count() or 1, 16))
EVAL_LIMIT        = 500
TEST_LIMIT        = 200

print("=" * 72)
print("LEAN FLOW v2.6 CONFIG (HERMES + xLAM-60K 1-CALL)")
print("=" * 72)
print(f"EXPERIMENT_NAME   = {EXPERIMENT_NAME}  SEED = {SEED}")
print(f"BACKBONE          = {BACKBONE}")
print(f"MAX_SEQ_LEN       = {MAX_SEQ_LEN}  MAX_SPAN_WIDTH = {MAX_SPAN_WIDTH}  MAX_POS = {MAX_POS}")
print(f"DATA_FRACTION     = {DATA_FRACTION}  TENSORIZE_WORKERS = {TENSORIZE_WORKERS}")
print(f"XLAM_TRAIN_ROWS   = {XLAM_TRAIN_ROWS}  XLAM_DEV_ROWS = {XLAM_DEV_ROWS}  SINGLE_CALL_ONLY = {SINGLE_CALL_ONLY}")
print(f"EVAL_LIMIT        = {EVAL_LIMIT}  TEST_LIMIT = {TEST_LIMIT}")
print(f"DATA_DIR          = {DATA_DIR}")
print(f"TENSOR_DIR        = {TENSOR_DIR}")
print(f"RESULTS_DIR       = {RESULTS_DIR}")
print(f"DEVICE            = {DEVICE}")'''

CODE_RUNTIME_CHECK = '''# 0c. Runtime, GPU & Package Check (+ Optional Colab Drive Mount)
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
    print("Drive mounted:", DRIVE_OUT)'''

MARKDOWN_DATA = """---
## 1. Dataset (`data/processed_v2/`) & Pre-Flight Gate

Downloads the raw splits from Hugging Face if missing, runs `scripts/realign_v2.py` to produce render-invariant `utt_rel_start`/`utt_rel_end` offsets, and verifies the template stamp (`Gate A`) and `utterance_span` offset validity (`Gate B`)."""

CODE_DATA_AND_PREFLIGHT = '''from huggingface_hub import snapshot_download
from span_template.stamp import check_stamp
from spancall.lean.diagnostics import audit_gt_offsets
import json, os

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
if FORCE_HF_DOWNLOAD or _missing_v2:
    print(f"Downloading v2 splits (Hermes + xLAM) from Hugging Face dataset: {HF_DATASET_ID} ...")
    snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        local_dir=str(PROJECT_ROOT),
        allow_patterns=["data/processed_v2/*.jsonl", "data/processed_v2/*.json"],
        token=HF_TOKEN,
    )

_already = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in CANONICAL_SPLITS)
_stamp_ok = check_stamp(DATA_DIR).ok if _already else False
if FORCE_REALIGN or not _already or not _stamp_ok:
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
        raise SystemExit(f"realign_v2 exited {_rc}. Inspect data/processed_v2/realign_report.json.")
else:
    print(f"v2 splits ready in {DATA_DIR}")

# Fallback: prepare xLAM-60k (1-call) splits in DATA_DIR if still missing
_missing_xlam = [f for f in XLAM_SPLITS if not os.path.exists(os.path.join(DATA_DIR, f))]
if _missing_xlam and (XLAM_TRAIN_ROWS is None or XLAM_TRAIN_ROWS > 0):
    print(f"Preparing xLAM-60k (1-call) splits into {DATA_DIR} via scripts/prepare_data.py ...")
    _prep_rc = subprocess.run([
        sys.executable, os.path.join(str(PROJECT_ROOT), "scripts", "prepare_data.py"),
        "--sources", "xlam",
        "--output_dir", DATA_DIR,
    ], cwd=str(PROJECT_ROOT)).returncode
    if _prep_rc != 0:
        raise SystemExit(f"prepare_data.py --sources xlam exited {_prep_rc}")

# Pre-Flight Gate A: template stamp
_verdict = check_stamp(DATA_DIR)
print(f"[Gate A / Stamp] {_verdict.status}: {_verdict.message()}")
if not _verdict.ok and not ALLOW_STALE_LABELS:
    raise SystemExit("PRE-FLIGHT FAILED (template stamp): " + _verdict.message())

# Pre-Flight Gate B: offset audit on eval & dev (Hermes + xLAM)
_audit_splits = [EVAL_SPLIT, "hermes_dev"]
for _xsp in ("xlam_eval", "xlam_dev"):
    if os.path.exists(os.path.join(DATA_DIR, f"{_xsp}.jsonl")):
        _audit_splits.append(_xsp)
for _split in _audit_splits:
    _m = audit_gt_offsets(_split, DATA_DIR)
    _stale = _m.get("utterance_span_stale_rate")
    _unres = _m.get("unresolvable_rate")
    _ok = (
        isinstance(_stale, float) and _stale == _stale and _stale < MAX_UTT_STALE_RATE
        and ((_unres is None) or (isinstance(_unres, float) and _unres == _unres and _unres < MAX_UNRESOLVABLE))
    )
    print(f"[Gate B / Audit] {_split:<16} utterance_span_stale_rate={_stale} "
          f"({_m.get('utterance_span_targets')} targets), unresolvable_rate={_unres} -> {'PASS' if _ok else 'FAIL'}")
    if not _ok and not ALLOW_STALE_LABELS:
        raise SystemExit(f"PRE-FLIGHT FAILED on {_split}")'''

MARKDOWN_TENSORIZE = """---
## 2. Lean Tensorizer — Hermes + xLAM-60k (1-Call) Mix

Tensorizes `hermes_train.jsonl` + `xlam_train.jsonl` (up to `XLAM_TRAIN_ROWS`) into `t2.jsonl` and `hermes_dev.jsonl` + `xlam_dev.jsonl` (up to `XLAM_DEV_ROWS`) into `dev.jsonl` using a shared multiprocessing fork pool."""

CODE_TENSORIZE = '''from typing import Any, Dict, List, Optional
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

PACK_TOOLS = 3
try:
    with open(os.path.join(str(PROJECT_ROOT), "configs", "default.yaml")) as _fh:
        _default_cfg = yaml.safe_load(_fh) or {}
    PACK_TOOLS = int((_default_cfg.get("data") or {}).get("pack_tools", 3) or 3)
except Exception as _exc:
    print(f"[!] configs/default.yaml unreadable ({_exc}); using PACK_TOOLS={PACK_TOOLS}")

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
            fh.write("\\n".join(rows) + ("\\n" if rows else ""))
        render_hist = {k.split("=")[1]: v for k, v in drops.items() if k.startswith("render_T=")}
        print(f"[tensorize] {out_name:<4} <- {per_src_counts}: {len(rows):>6} rows in {dt:.1f}s "
              f"({len(rows)/max(dt, 1e-6):.1f} rows/s) | render_T={render_hist} "
              f"| match_missed={drops.get('match_missed', 0)} | errors={dict(errors)}")
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
print("wrote config fingerprint ->", fp_path)'''

MARKDOWN_TRAIN = """---
## 3. Training — Lean Flow (`scripts/train_lean.py`)

Trains `LeanSpanCallModel` (`ModernBERT-large` + `ToolSelectionHead` H1 + `LeanPointerHead` H3) on stage `T2` with full data (`TRAIN_FRACTION = 1.0`), effective batch size 128, `bf16`, and best-checkpoint restoration."""

CODE_TRAIN_CFG_AND_RUN = '''import os, yaml
from scripts.train_lean import main as train_lean_main

TRAIN_FRACTION = 1.0
NUM_EPOCHS = 5
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
cfg["training"]["gradient_checkpointing"] = True
cfg["training"]["eval_splits"] = ["dev"]
cfg["training"]["eval_limit"] = 2000
cfg["training"]["freeze_encoder_stages"] = []

cfg["model"]["backbone_name"] = BACKBONE
cfg["model"]["max_seq_len"] = MAX_SEQ_LEN

_EFFECTIVE_PACK_TOOLS = int(globals().get("PACK_TOOLS_EPH", PACK_TOOLS) or 1)
cfg.setdefault("data", {})["fraction"] = TRAIN_FRACTION
cfg["data"]["pack_tools"] = _EFFECTIVE_PACK_TOOLS

CONFIG_PATH = "configs/colab_lean_v2_6.yaml"
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
])'''

CODE_TELEMETRY_AND_RESOLVER = '''# 3b. Training Telemetry, Best Checkpoint Selection & LeanResolver Init
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

best_cp = os.path.join(exp_root, "best_checkpoint")
if os.path.exists(os.path.join(best_cp, "model.pt")):
    CHECKPOINT_DIR = best_cp
else:
    cps = sorted(glob.glob(os.path.join(exp_root, "**", "model.pt"), recursive=True))
    CHECKPOINT_DIR = os.path.dirname(cps[-1]) if cps else None
print("\\nCHECKPOINT_DIR =", CHECKPOINT_DIR)

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
)
if DEVICE in ("cuda", "mps"):
    model.to(DEVICE)
model.eval()
if hasattr(encoder, "eval"):
    encoder.eval()
print("LeanResolver ready.")'''

MARKDOWN_EVAL_SLICE = """---
## 4. Step-by-Step Evaluation (`hermes_eval`)

### 4.0 Stratified Benchmark Slice & Step 0 (Pre-Encode Sanitize Gate)

Builds the shared deterministic stratified slice (`subset_hermes_eval.jsonl` across `F_func`, `F_json`, `A_clean`, `A_adv`) and runs the Step 0 pre-encode marker-injection check."""

CODE_EVAL_SLICE = '''import json, os
from span_template.template import RecentRow, build_input_multi
from span_template.sanitize import SanitizeError
from spancall.data.schema import UnifiedSample
from spancall.data.align import tooldef_to_template_tool
from spancall.lean.eval_p1 import build_stratified_eval_subset, classify_eval_cohort

SUBSET_EVAL_PATH = os.path.join(RESULTS_DIR, f"subset_{EVAL_SPLIT}.jsonl")
CLEAN_EVAL_PATH  = os.path.join(RESULTS_DIR, f"subset_{EVAL_SPLIT}_clean.jsonl")
_slice_info = build_stratified_eval_subset(
    EVAL_SPLIT,
    data_dir=DATA_DIR,
    out_path=SUBSET_EVAL_PATH,
    limit=EVAL_LIMIT,
    seed=SEED,
)
if SINGLE_CALL_ONLY:
    with open(SUBSET_EVAL_PATH, encoding="utf-8") as _fh:
        _one_call_lines = [
            _l for _l in _fh
            if _l.strip() and len((json.loads(_l).get("ground_truth_calls") or [])) == 1
        ]
    with open(SUBSET_EVAL_PATH, "w", encoding="utf-8") as _fh:
        _fh.writelines(_one_call_lines)
    print(f"[SINGLE_CALL_ONLY] filtered {EVAL_SPLIT} slice -> {len(_one_call_lines)} 1-call rows")
print("=== 4.0 Stratified Benchmark Slice ===")
print("Available by cohort:", _slice_info["available_by_cohort"])
print("Sampled   by cohort:", _slice_info["sampled_by_cohort"], f"(total={_slice_info['sampled_total']})")

_step0_stats = {c: {"n": 0, "vetoed": 0} for c in ("F_func", "F_json", "A_clean", "A_adv")}
_clean_lines = []
with open(SUBSET_EVAL_PATH, encoding="utf-8") as _fh:
    for _line in _fh:
        _raw = _line.strip()
        if not _raw:
            continue
        _d = json.loads(_raw)
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

with open(CLEAN_EVAL_PATH, "w", encoding="utf-8") as _fh:
    for _l in _clean_lines:
        _fh.write(_l + "\\n")

print("\\n=== Step 0: Pre-Encode Sanitize Gate ===")
for _c in ("F_func", "F_json", "A_clean", "A_adv"):
    _st = _step0_stats.get(_c, {"n": 0, "vetoed": 0})
    _rate = (_st["vetoed"] / _st["n"]) if _st["n"] else 0.0
    print(f"  {_c:<8}: vetoed={_st['vetoed']:>3} / {_st['n']:<3} ({_rate:.1%})")
print(f"Non-vetoed clean eval rows -> {CLEAN_EVAL_PATH} ({len(_clean_lines)} rows)")'''

MARKDOWN_EVAL_STEP1 = """---
### 4a. Step 1 in Isolation — Tool Selection (`H1`)

Scores `resolver.predict_tool` on native multi-tool rows (`len(s.tools) >= 2`), single-tool rows augmented with random pool distractors (`T=3`), and measures `F_func` vs `A_clean` score/margin separation."""

CODE_EVAL_STEP1 = '''import json, os, random, zlib
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

        pred = resolver.predict_tool(candidates, s.utterance, s.recent_context, s.current_time_iso)
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

    print("\\n=== Step 1b: Single-Tool + Random Pool Distractors (T=3) ===")
    if aug_stats[1]:
        print(f"  Single+random distractors: {aug_stats[0] / aug_stats[1]:.4f} ({aug_stats[0]}/{aug_stats[1]})")

    print("\\n=== Step 1c: Score & Margin Separation (F_func vs A_clean) ===")
    for bk in ("F_func", "A_clean"):
        ps = probs_by_cohort.get(bk, [])
        ms = margins_by_cohort.get(bk, [])
        if ps:
            print(f"  {bk:<8} (n={len(ps):>3}) | tool_prob p10={_pct(ps,0.1):.3f} p50={_pct(ps,0.5):.3f} p90={_pct(ps,0.9):.3f} "
                  f"| margin p10={_pct(ms,0.1):.3f} p50={_pct(ms,0.5):.3f} p90={_pct(ms,0.9):.3f}")
    if wrong_pairs:
        print("  Top confusions (gt -> pred):", [(f"{g} -> {p}", cnt) for (g, p), cnt in wrong_pairs.most_common(6)])

eval_tool_prediction_step1(CLEAN_EVAL_PATH, os.path.join(DATA_DIR, f"{EVAL_SPLIT}.jsonl"))'''

MARKDOWN_EVAL_STEP2 = """---
### 4b. Step 2 in Isolation — Parameter Extraction (`H3`, Oracle Tool Given)

Breaks Step 2 (`resolver.try_resolve(gt_tool, ...)`) into:
- **Step 2a**: Candidate Mask Parity (`T=all` vs `T=1`).
- **Step 2b**: Oracle Candidates (`protect_parity_spans=True`, gates OFF) vs Served Candidates (`protect_parity_spans=False`).
- **Step 2c**: Gated & Schema-Validated Extraction (`collapse_guard=True`, `min_joint_margin`)."""

CODE_EVAL_STEP2 = '''import math
from spancall.lean.eval_p1 import eval_param_extraction_lean
from spancall.lean.resolver import LeanResolver

def _fmt_p1(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)

_p1_2a_all = eval_param_extraction_lean(
    CLEAN_EVAL_PATH, resolver=resolver, data_dir=DATA_DIR, limit=None,
    measure_candidate_parity=True, protect_parity_spans=False,
    context_mode="all", stratum_filter="F", exclude_tools=["json_output"], quiet=True,
)
_p1_2a_t1 = eval_param_extraction_lean(
    CLEAN_EVAL_PATH, resolver=resolver, data_dir=DATA_DIR, limit=None,
    measure_candidate_parity=True, protect_parity_spans=False,
    context_mode="single", stratum_filter="F", exclude_tools=["json_output"], quiet=True,
)
print("=== Step 2a: Candidate Mask Parity (F_func, protection OFF) ===")
print(f"  T=all (multi-tool render): parity={_fmt_p1(_p1_2a_all['candidate_parity'])} "
      f"({_p1_2a_all['candidate_parity_hit']}/{_p1_2a_all['candidate_parity_checked']}) | "
      f"miss_reasons={_p1_2a_all['parity_miss_reasons']}")
print(f"  T=1   (two-step render)  : parity={_fmt_p1(_p1_2a_t1['candidate_parity'])} "
      f"({_p1_2a_t1['candidate_parity_hit']}/{_p1_2a_t1['candidate_parity_checked']}) | "
      f"miss_reasons={_p1_2a_t1['parity_miss_reasons']}")

_step2_arms = [
    ("2b. Oracle Cands (protect=T, gates=OFF)",    True,  False, 0.0,              "single"),
    ("2a+2b. Served Cands (protect=F, gates=OFF)", False, False, 0.0,              "single"),
    (f"2c. + Guards & Margin ({MIN_JOINT_MARGIN}) [T=1]",  False, True, MIN_JOINT_MARGIN, "single"),
    (f"2c-all. + Guards & Margin ({MIN_JOINT_MARGIN}) [T=all]", False, True, MIN_JOINT_MARGIN, "all"),
]

print("\\n=== Step 2b & 2c: Sub-Stage Waterfall (F_func vs A_clean) ===")
print(f"{'Stage / Config':<44}{'Parity':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Coll_F':>8}{'Exact_F':>9}{'A_Hit':>8}")
print("-" * 101)
for _label, _prot, _cg, _mjm, _cmode in _step2_arms:
    _r_arm = LeanResolver(
        model=model, encoder=encoder, tokenizer=tokenizer,
        max_seq_len=cfg.model.max_seq_len, max_span_width=cfg.model.max_span_width,
        max_calls=MAX_CALLS, call_min_prob=CALL_MIN_PROB, call_min_prob_ratio=CALL_MIN_PROB_RATIO,
        min_joint_margin=_mjm, collapse_guard=_cg, prevent_param_collapse=_cg,
    )
    _mf = eval_param_extraction_lean(
        CLEAN_EVAL_PATH, resolver=_r_arm, data_dir=DATA_DIR, limit=None,
        measure_candidate_parity=True, protect_parity_spans=_prot,
        context_mode=_cmode, stratum_filter="F", exclude_tools=["json_output"], quiet=True,
    )
    _ma = eval_param_extraction_lean(
        CLEAN_EVAL_PATH, resolver=_r_arm, data_dir=DATA_DIR, limit=None,
        measure_candidate_parity=False, protect_parity_spans=False,
        context_mode=_cmode, stratum_filter="A", exclude_tools=["json_output"], quiet=True,
    )
    _a_hit = (_ma["extracted"] / _ma["n"]) if _ma["n"] else float("nan")
    print(f"{_label:<44}{_fmt_p1(_mf['candidate_parity']):>8}"
          f"{_fmt_p1(_mf['extraction_precision']):>8}{_fmt_p1(_mf['extraction_recall']):>8}"
          f"{_fmt_p1(_mf['extraction_f1']):>8}{_fmt_p1(_mf['pointer_collapse_rate_F']):>8}"
          f"{_fmt_p1(_mf['exact_payload_match_F']):>9}{_fmt_p1(_a_hit):>8}")'''

MARKDOWN_EVAL_STEP3 = """---
### 4c. Step 3 — Composed E2E (`single_pass` vs `two_step`), Stage Waterfall & Headline Table

Runs E2E inference on `SUBSET_EVAL_PATH` in both `single_pass` and `two_step` modes, prints the stage-by-stage waterfall funnel, and compares `B1a (regex)` vs `SpanCall Lean (single_pass)` vs `SpanCall Lean (two_step)` via `scripts/eval_toolcall.py`."""

CODE_EVAL_STEP3 = '''import json, os, subprocess, sys
from typing import Any, Dict, List, Optional
from spancall.data.schema import UnifiedSample
from spancall.eval.toolcall.evaluator import payload_equal
from spancall.lean.eval_p1 import _parity_spans_for_call, classify_eval_cohort, predicted_call_count as _n_predicted_calls

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
    src = src_path or (sub_src if (split_name == EVAL_SPLIT and os.path.exists(sub_src)) else os.path.join(DATA_DIR, f"{split_name}.jsonl"))
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
            if _supports_parity:
                res = resolver.try_resolve_candidates(
                    s.tools, s.utterance, s.recent_context, s.current_time_iso,
                    parity_spans=_p_spans, protect_parity_spans=protect_parity_spans,
                    resolve_mode=resolve_mode,
                )
            else:
                res = resolver.try_resolve_candidates(s.tools, s.utterance, s.recent_context, s.current_time_iso)
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
            if limit is not None and n >= limit:
                break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for p in preds:
            fh.write(json.dumps(p, default=str) + "\\n")
    if os.path.abspath(src) != os.path.abspath(sub_src):
        with open(sub_src, "w") as fh:
            for s_dict in subset_samples:
                fh.write(json.dumps(s_dict, default=str) + "\\n")
    n_hit = sum(1 for p in preds if p["hit"])
    _par_str = f"{parity_hit}/{parity_checked} ({parity_hit / parity_checked:.3f})" if parity_checked else "n/a"
    print(f"predicted {n} rows ({resolve_mode or 'default'}) -> {out_path} (hit={n_hit}, collapse_hits={collapse_hits}, e2e_parity={_par_str})")

_p_single = os.path.join(RESULTS_DIR, "preds_eval.jsonl")
_p_two    = os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")
predict_split(EVAL_SPLIT, _p_single, limit=None, resolve_mode="single_pass", src_path=SUBSET_EVAL_PATH)
predict_split(EVAL_SPLIT, _p_two,    limit=None, resolve_mode="two_step",    src_path=SUBSET_EVAL_PATH)

_raw_sub = {}
with open(SUBSET_EVAL_PATH) as _fh:
    for _l in _fh:
        if _l.strip():
            _d = json.loads(_l)
            _raw_sub[_d["sample_id"]] = (_d, UnifiedSample.from_dict(_d))

def _waterfall(preds_path: str, label: str) -> None:
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
        if payload_equal(p.get("gt_args") or {}, p.get("payload_partial") or {}, _gt_t):
            s2b_exact.append(p)
    s2c_hit = [p for p in s2b_exact if p.get("hit")]

    print(f"\\n--- Waterfall Funnel [{label}] on F_func (N={n_tot}) ---")
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

_waterfall(_p_single, "single_pass")
_waterfall(_p_two, "two_step")

# Evaluate B1a, single_pass, and two_step via eval_toolcall.py
for _args, _out in [
    (["--baseline", "b1a"], f"toolcall_metrics_{EVAL_SPLIT}_b1a.json"),
    (["--predictions", _p_single], f"toolcall_metrics_{EVAL_SPLIT}_lean.json"),
    (["--predictions", _p_two], f"toolcall_metrics_{EVAL_SPLIT}_lean_twostep.json"),
]:
    subprocess.run([
        sys.executable, "scripts/eval_toolcall.py",
        "--test_data", SUBSET_EVAL_PATH,
        *_args,
        "--output", os.path.join(RESULTS_DIR, _out),
    ], check=True)

def _load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)

print("\\n=== Headline Comparison Table (hermes_eval) ===")
_headers = ["system", "n_F", "exact_F", "set_exact_F", "count_acc", "hit_F", "collapse_F", "param_F1_F", "false_hit_on_A", "exact_all"]
print(" | ".join(_headers))
print("-|" * len(_headers))
for _label, _fn in [
    ("B1a (regex)", f"toolcall_metrics_{EVAL_SPLIT}_b1a.json"),
    ("SpanCall Lean (single_pass)", f"toolcall_metrics_{EVAL_SPLIT}_lean.json"),
    ("SpanCall Lean (two_step)", f"toolcall_metrics_{EVAL_SPLIT}_lean_twostep.json"),
]:
    _m = _load_metrics(os.path.join(RESULTS_DIR, _fn))
    if not _m:
        continue
    _mf = _m.get("metrics_by_stratum", {}).get("F", {})
    _row = {
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
    print(" | ".join(str(_row[h]) for h in _headers))'''

MARKDOWN_EVAL_STEP4 = """---
### 4d. Stage-Attributed Error Forensics

Groups `F_func` misses by the exact stage where they failed (`Step 1 WRONG_TOOL`, `Step 2a PARITY_MISS`, `Step 2b POINTER_RANK_MISS`, `Step 2c GATE_OR_SCHEMA_REJECT`)."""

CODE_EVAL_STEP4 = '''import json, os, textwrap
from collections import defaultdict
from spancall.data.schema import UnifiedSample
from spancall.eval.toolcall.evaluator import payload_equal
from spancall.lean.eval_p1 import classify_eval_cohort

PREDS_PATH = os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")
if not os.path.exists(PREDS_PATH):
    PREDS_PATH = os.path.join(RESULTS_DIR, "preds_eval.jsonl")

if os.path.exists(PREDS_PATH) and os.path.exists(SUBSET_EVAL_PATH):
    with open(PREDS_PATH) as fh:
        preds = [json.loads(l) for l in fh if l.strip()]
    subs, raw_map = {}, {}
    with open(SUBSET_EVAL_PATH) as fh:
        for l in fh:
            if l.strip():
                d = json.loads(l)
                raw_map[d["sample_id"]] = d
                subs[d["sample_id"]] = UnifiedSample.from_dict(d)

    buckets = defaultdict(list)
    for p in preds:
        sid = p.get("sample_id")
        if sid not in raw_map or classify_eval_cohort(raw_map[sid]) != "F_func":
            continue
        feats = p.get("features") or {}
        if str(p.get("abstain_reason") or "").startswith("sanitize"):
            buckets["STEP0_SANITIZE"].append(p)
        elif p.get("tool_name") != p.get("gt_tool"):
            buckets["STEP1_WRONG_TOOL"].append(p)
        elif int(feats.get("parity_hit") or 0) < int(feats.get("parity_checked") or 0):
            buckets["STEP2A_PARITY_MISS"].append(p)
        else:
            s = subs[sid]
            gt_t = next((t for t in s.tools if t.name == p["gt_tool"]), None)
            partial_ok = payload_equal(p.get("gt_args") or {}, p.get("payload_partial") or {}, gt_t)
            if p.get("hit") and payload_equal(p.get("gt_args") or {}, p.get("payload") or {}, gt_t):
                buckets["EXACT_HIT"].append(p)
            elif not partial_ok:
                buckets["STEP2B_POINTER_RANK_MISS"].append(p)
            else:
                buckets["STEP2C_GATE_OR_SCHEMA_REJECT"].append(p)

    tot_f = sum(len(v) for v in buckets.values())
    print(f"F_func Stage Attribution ({os.path.basename(PREDS_PATH)}, N={tot_f}):")
    for k in ("EXACT_HIT", "STEP0_SANITIZE", "STEP1_WRONG_TOOL", "STEP2A_PARITY_MISS", "STEP2B_POINTER_RANK_MISS", "STEP2C_GATE_OR_SCHEMA_REJECT"):
        c = len(buckets[k])
        print(f"  {k:<30}{c:>4}  ({c / max(1, tot_f):>6.1%})")'''

MARKDOWN_TEST = """---
## 5. OOD / Adversarial Test Splits & Drive Persistence

Evaluates `LeanResolver` and `B1a` on `hermes_test_ood_schema`, `hermes_test_ood_para`, and `hermes_test_adversarial`, and optionally mirrors artifacts to Google Drive."""

CODE_TEST_AND_PERSIST = '''import json, os, shutil, subprocess, sys
from spancall.lean.eval_p1 import build_stratified_eval_subset

for split in TEST_SPLITS:
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
        resolve_mode="two_step",
        src_path=sub_test_path,
    )
    data_file = sub_test_path
    for label, extra in [("b1a", ["--baseline", "b1a"]), ("lean", ["--predictions", os.path.join(RESULTS_DIR, f"preds_{split}.jsonl")])]:
        subprocess.run([
            sys.executable, "scripts/eval_toolcall.py",
            "--test_data", data_file,
            *extra,
            "--output", os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_{label}.json"),
        ], check=False)

print(f"\\n{'split':<26}{'system':<12}{'n_F':>6}{'exact_F':>9}{'hit_F':>8}{'param_F1':>9}{'exact_all':>10}")
print("-" * 80)
for split in TEST_SPLITS + [EVAL_SPLIT]:
    for label in ("b1a", "lean"):
        m = _load_metrics(os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_{label}.json"))
        if not m:
            continue
        m_f = m.get("metrics_by_stratum", {}).get("F", {})
        print(f"{split:<26}{label:<12}{int(m_f.get('n', 0)):>6}"
              f"{m_f.get('exact_payload_match', 0.0):>9.3f}{m_f.get('hit_rate', 0.0):>8.3f}"
              f"{m_f.get('param_f1', 0.0):>9.3f}{m.get('exact_payload_match', 0.0):>10.3f}")

if DRIVE_MOUNT:
    os.makedirs(DRIVE_OUT, exist_ok=True)
    for name, src in [("data_tensorized_lean_v2_6", TENSOR_DIR), ("results_lean_v2_6", RESULTS_DIR)]:
        if os.path.isdir(src):
            dst = os.path.join(DRIVE_OUT, name)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"mirrored {src} -> {dst}")'''

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

cells = [
    new_markdown_cell(MARKDOWN_TITLE),
    new_markdown_cell(MARKDOWN_SETUP),
    new_code_cell(CODE_SETUP),
    new_code_cell(CODE_CONFIG),
    new_code_cell(CODE_RUNTIME_CHECK),
    new_markdown_cell(MARKDOWN_DATA),
    new_code_cell(CODE_DATA_AND_PREFLIGHT),
    new_markdown_cell(MARKDOWN_TENSORIZE),
    new_code_cell(CODE_TENSORIZE),
    new_markdown_cell(MARKDOWN_TRAIN),
    new_code_cell(CODE_TRAIN_CFG_AND_RUN),
    new_code_cell(CODE_TELEMETRY_AND_RESOLVER),
    new_markdown_cell(MARKDOWN_EVAL_SLICE),
    new_code_cell(CODE_EVAL_SLICE),
    new_markdown_cell(MARKDOWN_EVAL_STEP1),
    new_code_cell(CODE_EVAL_STEP1),
    new_markdown_cell(MARKDOWN_EVAL_STEP2),
    new_code_cell(CODE_EVAL_STEP2),
    new_markdown_cell(MARKDOWN_EVAL_STEP3),
    new_code_cell(CODE_EVAL_STEP3),
    new_markdown_cell(MARKDOWN_EVAL_STEP4),
    new_code_cell(CODE_EVAL_STEP4),
    new_markdown_cell(MARKDOWN_TEST),
    new_code_cell(CODE_TEST_AND_PERSIST),
]

# --------------------------------------------------------------------------
# Build-time guard: compile every code cell and resolve names
# --------------------------------------------------------------------------

def _cell_source(cell) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _is_ipython_cell(src: str) -> bool:
    for line in src.splitlines():
        head = line.lstrip()
        if head.startswith(("!", "%", "%%")):
            return True
    return False


def _scope_walk(tbl, enclosing, available, unresolved, is_module: bool) -> None:
    local = set()
    for sym in tbl.get_symbols():
        if sym.is_assigned() or sym.is_imported() or sym.is_parameter() or sym.is_namespace():
            local.add(sym.get_name())
    visible = local | enclosing
    for sym in tbl.get_symbols():
        name = sym.get_name()
        if name in visible:
            continue
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

with open(OUT, "r") as fh:
    json.load(fh)
nb2 = nbformat.read(OUT, as_version=4)
nbformat.validate(nb2)
print(f"wrote {OUT.resolve()} — {len(cells)} cells, valid nbformat 4 JSON")

