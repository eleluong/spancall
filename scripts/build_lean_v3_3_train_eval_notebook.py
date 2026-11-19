#!/usr/bin/env python3
"""Build ``notebooks/spancall_lean_v3_3_train_eval.ipynb``.

Source of truth for the Lean Flow v3.3 training & evaluation notebook:
- Frozen fair eval dataset (``data/eval_v3_3/`` with SHA256 ``manifest.json``)
- Max-data 1:1 balanced Hermes + xLAM training (all xLAM + upsampled Hermes
  with distinct per-copy ``jitter_seed`` for fresh distractor packing)
- Full separated ``dev_hermes`` and ``dev_xlam`` splits with
  ``early_stopping_agg="macro_mean"``
- Balanced calib-sourced mid-training probe with hard zero-overlap assertion
- Two-step-only evaluation on Tier 1 fair eval + all Tier 2 full/OOD/adv splits
- Unified export via ``scripts/export_v3_3_results.py``

Usage::

    python scripts/build_lean_v3_3_train_eval_notebook.py           # write notebook
    python scripts/build_lean_v3_3_train_eval_notebook.py --check   # fail if stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / ".." / "notebooks" / "spancall_lean_v3_3_train_eval.ipynb"

CODE = "code"
MD = "markdown"

CELLS: List[Tuple[str, str]] = [
    (MD, r'''# SpanCall Lean Flow **v3.3** — Balanced Max-Data Training & Frozen Fair Eval

End-to-end training and evaluation notebook for **v3.3**:

1. **Frozen Fair Eval (`data/eval_v3_3/`):** Built via `scripts/build_fair_eval_v3_3.py` with SHA256 `manifest.json`. Tier 1 (`fair_hermes_eval`, `fair_xlam_eval`) enforces exact 1:1 `(cohort × complexity)` parity after 1-call + scoreable + Step-0 filtering; Tier 2 covers full ID, OOD-schema, OOD-para, and matched adversarial splits for both corpora.
2. **Balanced Max-Data Training (~138.8k rows, 1:1 Hermes:xLAM):** Uses 100% of `xlam_train` (`69,412` rows) and upsamples 1-call scoreable `hermes_train` (`19,096` → `69,412`) with **distinct per-copy `jitter_seed`** so every Hermes copy gets fresh distractor tools from the combined `ToolPool`.
3. **Macro-Averaged Dev Early Stopping:** Scores full `dev_hermes` and `dev_xlam` separately and selects the best checkpoint via `early_stopping_agg="macro_mean"`. Mid-training probe (`300` Hermes + `300` xLAM) is drawn from `calib` with a hard zero-overlap assertion against `dev` and `eval`.
4. **Unified Audited Export:** Runs `scripts/export_v3_3_results.py` directly at the end and packages `results_lean_v3_3/` so local multi-model LLM evaluation (`scripts/eval_llm_local.py`) and final paper export share byte-identical data.

**Paths:** `data/processed_v3_3`, `data/eval_v3_3`, `data/tensorized_lean_v3_3`, `results_lean_v3_3`.'''),

    (MD, r'''---
## 0. Setup & Configuration'''),

    (CODE, r'''# 0a. Pull latest code from GitHub and install dependencies
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
    print(f"Cloning repository from {GIT_REPO_URL}...")
    !git clone --branch {GIT_BRANCH} {GIT_REPO_URL}
    PROJECT_ROOT = cwd / "spancall"

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if (PROJECT_ROOT / ".git").is_dir():
    !git pull origin {GIT_BRANCH} || echo "git pull note: continuing with local files"

print(f"SpanCall project root: {PROJECT_ROOT}")
%pip install -q -e ".[model,data,agent,dev]" huggingface_hub openai'''),

    (CODE, r'''# 0b. Configuration & Environment (Lean v3.3 — Balanced Max-Data + Fair Eval)
import os, sys
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

HF_DATASET_ID     = "convoicon/spancall"
DRIVE_MOUNT       = False
DRIVE_OUT         = "/content/drive/MyDrive/spancall_lean_v3_3_artifacts"
DOWNLOAD_RESULTS  = False

DATA_DIR          = str(PROJECT_ROOT / "data" / "processed_v3_3")
V1_DATA_DIR       = str(PROJECT_ROOT / "data" / "processed")
V2_DATA_DIR       = str(PROJECT_ROOT / "data" / "processed_v2")
EVAL_DIR          = str(PROJECT_ROOT / "data" / "eval_v3_3")
TENSOR_DIR        = str(PROJECT_ROOT / "data" / "tensorized_lean_v3_3")
RESULTS_DIR       = str(PROJECT_ROOT / "results_lean_v3_3")

# Max-data 1:1 balanced training knobs
XLAM_TRAIN_ROWS       = None   # None => use 100% of xlam_train (~69.4k)
BALANCE_TRAIN_CORPORA = True   # upsample smaller corpus (Hermes) to 1:1 with fresh jitter_seed
SINGLE_CALL_ONLY      = True
ADV_LIMIT             = 5000   # matched per-corpus cap for adversarial splits (0 = all)

MAX_UTT_STALE_RATE = 0.02
MAX_UNRESOLVABLE   = 0.02
FORCE_REALIGN      = False
FORCE_REBUILD_EVAL = False
MIN_JOINT_MARGIN   = 0.02

DEVICE          = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
EXPERIMENT_NAME = "spancall_lean_v3_3_modernbert"
SEED            = 42

BACKBONE       = "answerdotai/ModernBERT-large"
MAX_SEQ_LEN    = 4096
MAX_SPAN_WIDTH = 16
MAX_CANDS      = 1024

_cfg_yaml = {}
with open("configs/default.yaml") as _fh:
    _cfg_yaml = yaml.safe_load(_fh) or {}

MAX_POS = int(((_cfg_yaml.get("optimization") or {}).get("max_pos", 1)) or 1)
_data_cfg = _cfg_yaml.get("data") or {}
PACK_TOOLS = int(_data_cfg.get("pack_tools", 6) or 6)
MAX_TOOLS = _data_cfg.get("max_tools")
MAX_TOOLS = None if MAX_TOOLS in (None, "", 0) else int(MAX_TOOLS)
JITTER_PACK_TOOLS = bool(_data_cfg.get("jitter_pack_tools", True))

TENSORIZE_WORKERS = max(1, min(os.cpu_count() or 1, 16))
EVAL_BATCH_SIZE   = int(os.environ.get("EVAL_BATCH_SIZE", "8"))
EVAL_WORKERS      = 4

# Mid-training probe (balanced 300 Hermes + 300 xLAM from calib; asserted disjoint from dev & eval)
PROBE_SPLIT       = "probe"
PROBE_PER_CORPUS  = 300
PROBE_EVERY_STEPS = 1000
PROBE_PERCENT     = 0.10
PROBE_LIMIT       = PROBE_PER_CORPUS * 2


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else " [optional — Enter to skip]"
    try:
        val = input(f"{label}{suffix}: ").strip()
    except (EOFError, OSError):
        val = ""
    return val or default


HF_TOKEN = os.environ.get("HF_TOKEN") or _prompt("Hugging Face access token")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

HF_REPO_ID      = _prompt("HF model repo ID", "convoicon/spancall-lean-v3-3")
HF_PUSH_PRIVATE = False

print("=" * 72)
print("LEAN FLOW v3.3 CONFIG (BALANCED MAX-DATA + FROZEN FAIR EVAL)")
print("=" * 72)
print(f"EXPERIMENT_NAME       = {EXPERIMENT_NAME}  SEED = {SEED}")
print(f"BACKBONE              = {BACKBONE}  DEVICE = {DEVICE}")
print(f"XLAM_TRAIN_ROWS       = {XLAM_TRAIN_ROWS}  BALANCE_TRAIN_CORPORA = {BALANCE_TRAIN_CORPORA}")
print(f"PACK_TOOLS            = {PACK_TOOLS}  JITTER = {JITTER_PACK_TOOLS}  MAX_POS = {MAX_POS}")
print(f"EVAL_BATCH_SIZE       = {EVAL_BATCH_SIZE}  MAX_CANDS = {MAX_CANDS}")
print(f"DATA_DIR              = {DATA_DIR}")
print(f"EVAL_DIR              = {EVAL_DIR}")
print(f"TENSOR_DIR            = {TENSOR_DIR}")
print(f"RESULTS_DIR           = {RESULTS_DIR}")
print(f"HF_REPO_ID            = {HF_REPO_ID}")'''),

    (CODE, r'''# 0c. Runtime & GPU Check (+ Optional Drive Mount)
!nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo 'No NVIDIA GPU detected (using MPS / CPU)'

import spancall, span_template, transformers, datasets
print("spancall     :", spancall.__file__)
print("span_template:", span_template.__file__)
print("transformers :", transformers.__version__, "| datasets:", datasets.__version__)

if DRIVE_MOUNT:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_OUT, exist_ok=True)
    print("Drive mounted:", DRIVE_OUT)'''),

    (MD, r'''---
## 1. Re-Aligned Data (`data/processed_v3_3/`) & Frozen Fair Eval (`data/eval_v3_3/`)

Re-aligns Hermes + xLAM with v2.7 object decomposition and template stamp verification, then builds (or verifies) the frozen `data/eval_v3_3/` benchmark and its SHA256 `manifest.json`.'''),

    (CODE, r'''import json, os, subprocess, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from span_template.stamp import check_stamp
from spancall.eval.llm_parity import verify_eval_manifest

CANONICAL_SPLITS = [
    "hermes_train.jsonl", "hermes_dev.jsonl", "hermes_calib.jsonl", "hermes_eval.jsonl",
    "hermes_test_ood_schema.jsonl", "hermes_test_ood_para.jsonl", "hermes_test_adversarial.jsonl",
]
XLAM_SPLITS = [
    "xlam_train.jsonl", "xlam_dev.jsonl", "xlam_calib.jsonl", "xlam_eval.jsonl",
    "xlam_test_ood_schema.jsonl", "xlam_test_ood_para.jsonl", "xlam_test_adversarial.jsonl",
]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

_have_hermes = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in CANONICAL_SPLITS)
_stamp_ok = check_stamp(DATA_DIR).ok if _have_hermes else False
_decomp_ok = False
if _have_hermes:
    with open(os.path.join(DATA_DIR, "hermes_eval.jsonl"), encoding="utf-8") as _fh_d:
        for _i_d, _l_d in enumerate(_fh_d):
            if _i_d >= 300:
                break
            if '"object_paths":' in _l_d and '"object_paths": null' not in _l_d:
                _decomp_ok = True
                break
if FORCE_REALIGN or not _have_hermes or not _stamp_ok or not _decomp_ok:
    os.makedirs(V1_DATA_DIR, exist_ok=True)
    if any(not os.path.exists(os.path.join(V1_DATA_DIR, f)) for f in CANONICAL_SPLITS):
        snapshot_download(
            repo_id=HF_DATASET_ID, repo_type="dataset", local_dir=str(PROJECT_ROOT),
            allow_patterns=["data/processed/*.jsonl", "data/processed/*.json"], token=HF_TOKEN,
        )
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "realign_v2.py"),
        "--src_dir", V1_DATA_DIR, "--out_dir", DATA_DIR,
        "--gate_split", "hermes_eval",
        "--max_utt_stale_rate", str(MAX_UTT_STALE_RATE),
        "--max_unresolvable", str(MAX_UNRESOLVABLE),
    ], check=True)

_have_xlam = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in XLAM_SPLITS)
if FORCE_REALIGN or not _have_xlam:
    os.makedirs(V2_DATA_DIR, exist_ok=True)
    if any(not os.path.exists(os.path.join(V2_DATA_DIR, f)) for f in XLAM_SPLITS):
        snapshot_download(
            repo_id=HF_DATASET_ID, repo_type="dataset", local_dir=str(PROJECT_ROOT),
            allow_patterns=["data/processed_v2/xlam_*.jsonl"], token=HF_TOKEN,
        )
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "realign_v2.py"),
        "--src_dir", V2_DATA_DIR, "--out_dir", DATA_DIR,
        "--splits", *[s.replace(".jsonl", "") for s in XLAM_SPLITS],
        "--gate_split", "xlam_eval",
        "--max_utt_stale_rate", str(MAX_UTT_STALE_RATE),
        "--max_unresolvable", str(MAX_UNRESOLVABLE),
    ], check=True)

# Build or verify frozen Tier 1 + Tier 2 fair eval benchmark
_manifest_path = Path(EVAL_DIR) / "manifest.json"
if FORCE_REBUILD_EVAL or not _manifest_path.is_file():
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "build_fair_eval_v3_3.py"),
        "--data-dir", DATA_DIR,
        "--out-dir", EVAL_DIR,
        "--seed", str(SEED),
        "--adv-limit", str(ADV_LIMIT),
    ], check=True)

EVAL_MANIFEST = verify_eval_manifest(EVAL_DIR)
print("[v3.3] Verified frozen eval manifest:", len(EVAL_MANIFEST["splits"]), "splits")
for _sname, _smeta in EVAL_MANIFEST["splits"].items():
    print(f"  {_sname:<32} tier={_smeta['tier']:<11} n={_smeta['n_rows']:>5} sha={_smeta['sha256'][:12]}")'''),

    (MD, r'''---
## 2. Balanced Max-Data Tensorization (`t2`, `dev_hermes`, `dev_xlam`, `probe`)

- Uses **100% of `xlam_train`** + **1:1 upsampled `hermes_train`** where each upsampled copy receives a distinct deterministic `jitter_seed` (`SEED + copy_idx * 1009 + row_idx`) for fresh distractor tool packing.
- Keeps `dev_hermes` and `dev_xlam` separate so `LeanTrainer` optimizes their unweighted macro-average (`early_stopping_agg="macro_mean"`).
- Draws `probe` (`300` Hermes + `300` xLAM) from `calib` and **asserts** zero `sample_id` overlap with `dev` and `eval`.'''),

    (CODE, r'''import json, multiprocessing as mp, os, random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple
from transformers import AutoTokenizer

from span_template import SPECIAL_MARKER_TOKENS
from spancall.data.schema import UnifiedSample
from spancall.lean.eval_p1 import classify_eval_cohort
from spancall.lean.tensorize import ToolPool, has_positive_targets, tensorize_sample_lean
from spancall.model.masks import CandidateMaskBuilder

os.makedirs(TENSOR_DIR, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_MARKER_TOKENS)})
mask_builder = CandidateMaskBuilder(tokenizer=tokenizer, max_span_width=MAX_SPAN_WIDTH)


def _load_1call_scoreable(path: str) -> Tuple[List[str], Dict[str, int]]:
    kept: List[str] = []
    st = {"raw": 0, "multi_drop": 0, "missing_schema": 0}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            st["raw"] += 1
            d = json.loads(raw)
            if SINGLE_CALL_ONLY and len(d.get("ground_truth_calls") or []) != 1:
                st["multi_drop"] += 1
                continue
            if classify_eval_cohort(d) == "A_missing_gt_schema":
                st["missing_schema"] += 1
                continue
            kept.append(raw)
    return kept, st


def _upsample_with_copy_idx(lines: List[str], target_n: int, seed: int) -> List[Tuple[str, int]]:
    """Return ``target_n`` ``(raw_line, copy_idx)`` pairs for fresh per-copy distractor jitter."""
    if not lines or len(lines) >= target_n:
        return [(l, 0) for l in lines[:target_n]]
    out: List[Tuple[str, int]] = []
    full_copies, rem = divmod(target_n, len(lines))
    for c_idx in range(full_copies):
        for l in lines:
            out.append((l, c_idx))
    if rem > 0:
        rng = random.Random(int(seed) + full_copies * 31)
        idxs = list(range(len(lines)))
        rng.shuffle(idxs)
        for i in idxs[:rem]:
            out.append((lines[i], full_copies))
    return out


# Build combined ToolPool from both training corpora
_h_train_lines, _h_tr_st = _load_1call_scoreable(os.path.join(DATA_DIR, "hermes_train.jsonl"))
_x_train_lines, _x_tr_st = _load_1call_scoreable(os.path.join(DATA_DIR, "xlam_train.jsonl"))
if XLAM_TRAIN_ROWS is not None and XLAM_TRAIN_ROWS > 0:
    _rng_cap = random.Random(SEED)
    _x_train_lines = list(_x_train_lines)
    _rng_cap.shuffle(_x_train_lines)
    _x_train_lines = _x_train_lines[: int(XLAM_TRAIN_ROWS)]

_pool_dicts = [json.loads(l) for l in _h_train_lines] + [json.loads(l) for l in _x_train_lines]
TOOL_POOL = ToolPool.from_samples(_pool_dicts, seed=SEED)
print(f"[v3.3] ToolPool built: {len(TOOL_POOL)} unique schemas")

if BALANCE_TRAIN_CORPORA:
    _target = max(len(_h_train_lines), len(_x_train_lines))
    _h_pairs = _upsample_with_copy_idx(_h_train_lines, _target, SEED)
    _x_pairs = _upsample_with_copy_idx(_x_train_lines, _target, SEED + 7)
else:
    _h_pairs = [(l, 0) for l in _h_train_lines]
    _x_pairs = [(l, 0) for l in _x_train_lines]

_t2_pairs = _h_pairs + _x_pairs
random.Random(SEED).shuffle(_t2_pairs)
print(
    f"[v3.3] Train balance: hermes={len(_h_train_lines)} -> {len(_h_pairs)}, "
    f"xlam={len(_x_train_lines)} -> {len(_x_pairs)}, total={len(_t2_pairs)}"
)

# Dev splits (full 1-call scoreable, kept separated by corpus)
_h_dev_lines, _ = _load_1call_scoreable(os.path.join(DATA_DIR, "hermes_dev.jsonl"))
_x_dev_lines, _ = _load_1call_scoreable(os.path.join(DATA_DIR, "xlam_dev.jsonl"))

# Balanced probe from calib (hard zero-overlap assert against dev & eval)
_h_calib_lines, _ = _load_1call_scoreable(os.path.join(DATA_DIR, "hermes_calib.jsonl"))
_x_calib_lines, _ = _load_1call_scoreable(os.path.join(DATA_DIR, "xlam_calib.jsonl"))
_dev_ids = {json.loads(l)["sample_id"] for l in _h_dev_lines + _x_dev_lines}
_eval_ids = set()
for _sp_eval in ("fair_hermes_eval", "fair_xlam_eval"):
    _ep = os.path.join(EVAL_DIR, EVAL_MANIFEST["splits"][_sp_eval]["filename"])
    with open(_ep, encoding="utf-8") as _efh:
        for _el in _efh:
            if _el.strip():
                _eval_ids.add(json.loads(_el)["sample_id"])

_h_probe_pool = [l for l in _h_calib_lines if json.loads(l)["sample_id"] not in _dev_ids and json.loads(l)["sample_id"] not in _eval_ids]
_x_probe_pool = [l for l in _x_calib_lines if json.loads(l)["sample_id"] not in _dev_ids and json.loads(l)["sample_id"] not in _eval_ids]
random.Random(SEED).shuffle(_h_probe_pool)
random.Random(SEED + 1).shuffle(_x_probe_pool)
_probe_lines = _h_probe_pool[:PROBE_PER_CORPUS] + _x_probe_pool[:PROBE_PER_CORPUS]
_probe_ids = {json.loads(l)["sample_id"] for l in _probe_lines}
assert len(_probe_ids & _dev_ids) == 0, "Probe leaked into dev!"
assert len(_probe_ids & _eval_ids) == 0, "Probe leaked into eval!"
print(f"[v3.3] Dev: hermes={len(_h_dev_lines)}, xlam={len(_x_dev_lines)} | Probe: {len(_probe_lines)} (overlap=0 asserted)")


def _serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}


def _tensorize_one(task: Tuple[str, int, int, bool]) -> Tuple[Any, Dict[str, int]]:
    raw_line, copy_idx, row_idx, is_train = task
    try:
        s = UnifiedSample.from_dict(json.loads(raw_line))
        ts_stats: Dict[str, Any] = {}
        row = tensorize_sample_lean(
            s,
            tokenizer=tokenizer,
            mask_builder=mask_builder,
            max_seq_len=MAX_SEQ_LEN,
            max_cands=MAX_CANDS,
            max_pos=int(MAX_POS),
            pack_multi_tools=True,
            stats=ts_stats,
            tool_pool=TOOL_POOL if is_train else None,
            pack_tools=PACK_TOOLS if is_train else 1,
            max_tools=MAX_TOOLS,
            jitter=JITTER_PACK_TOOLS if is_train else False,
            jitter_seed=int(SEED) + int(copy_idx) * 1009 + int(row_idx),
            include_abstain=True,
        )
        if row is None or (not has_positive_targets(row) and not bool(row.get("is_abstain", 0))):
            return None, dict(ts_stats.get("param_drop_reasons") or {})
        return json.dumps(_serialize_row(row)), dict(ts_stats.get("param_drop_reasons") or {})
    except Exception as exc:
        return None, {f"err:{type(exc).__name__}": 1}


def _write_tensorized(split_name: str, pairs: List[Tuple[str, int]], is_train: bool) -> int:
    out_p = os.path.join(TENSOR_DIR, f"{split_name}.jsonl")
    tasks = [(raw, c_idx, i, is_train) for i, (raw, c_idx) in enumerate(pairs)]
    kept = 0
    drops = Counter()
    _use_fork = "fork" in mp.get_all_start_methods() and TENSORIZE_WORKERS > 1
    with open(out_p, "w", encoding="utf-8") as out_fh:
        if _use_fork:
            _ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=TENSORIZE_WORKERS, mp_context=_ctx) as ex:
                for res_line, d_map in ex.map(_tensorize_one, tasks, chunksize=64):
                    for k, v in d_map.items():
                        drops[k] += v
                    if res_line is not None:
                        out_fh.write(res_line + "\n")
                        kept += 1
        else:
            for t in tasks:
                res_line, d_map = _tensorize_one(t)
                for k, v in d_map.items():
                    drops[k] += v
                if res_line is not None:
                    out_fh.write(res_line + "\n")
                    kept += 1
    print(f"[tensorize] {split_name:<12}: kept={kept}/{len(tasks)}  top_drops={drops.most_common(4)}")
    return kept


N_T2 = _write_tensorized("t2", _t2_pairs, is_train=True)
N_DEV_HERMES = _write_tensorized("dev_hermes", [(l, 0) for l in _h_dev_lines], is_train=False)
N_DEV_XLAM = _write_tensorized("dev_xlam", [(l, 0) for l in _x_dev_lines], is_train=False)
N_PROBE = _write_tensorized("probe", [(l, 0) for l in _probe_lines], is_train=False)

RUN_METADATA = {
    "version": "v3.3",
    "experiment_name": EXPERIMENT_NAME,
    "seed": SEED,
    "backbone": BACKBONE,
    "max_seq_len": MAX_SEQ_LEN,
    "max_cands": MAX_CANDS,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "balance_train_corpora": BALANCE_TRAIN_CORPORA,
    "train_raw_counts": {"hermes": len(_h_train_lines), "xlam": len(_x_train_lines)},
    "train_balanced_counts": {"hermes": len(_h_pairs), "xlam": len(_x_pairs), "tensorized_t2": N_T2},
    "dev_counts": {"dev_hermes": N_DEV_HERMES, "dev_xlam": N_DEV_XLAM, "probe": N_PROBE},
    "eval_manifest_template_stamp": EVAL_MANIFEST.get("template_stamp"),
}
with open(os.path.join(RESULTS_DIR, "run_metadata.json"), "w", encoding="utf-8") as _mfh:
    json.dump(RUN_METADATA, _mfh, indent=2)
    _mfh.write("\n")'''),

    (MD, r'''---
## 3. Training (`LeanTrainer` with Macro-Averaged Dev Early Stopping)'''),

    (CODE, r'''# 3a. Configure and run LeanTrainer (v3.3)
import os, yaml
from scripts.train_lean import main as train_lean_main

NUM_EPOCHS           = 3
MICRO_BATCH_SIZE     = 32
EFFECTIVE_BATCH_SIZE = 128

with open("configs/default.yaml") as fh:
    cfg = yaml.safe_load(fh)

cfg["experiment"]["name"]       = EXPERIMENT_NAME
cfg["experiment"]["seed"]       = SEED
cfg["experiment"]["output_dir"] = RESULTS_DIR
cfg["experiment"]["data_dir"]   = TENSOR_DIR

cfg["optimization"]["num_epochs"]              = NUM_EPOCHS
cfg["optimization"]["batch_size"]              = MICRO_BATCH_SIZE
cfg["optimization"]["effective_batch_size"]    = EFFECTIVE_BATCH_SIZE
cfg["optimization"]["max_tokens_per_batch"]    = 4096
cfg["optimization"]["optimizer"]               = "adamw"
cfg["optimization"]["scheduler"]               = "cosine_warmup"
cfg["optimization"]["min_lr_ratio"]            = 0.10
cfg["optimization"]["early_stopping_patience"] = 1
cfg["optimization"]["early_stopping_agg"]      = "macro_mean"
cfg["optimization"]["restore_best_on_exit"]    = True
cfg["optimization"]["mixed_precision"]         = "bf16"

cfg["training"]["gradient_accumulation_steps"] = max(int(round(EFFECTIVE_BATCH_SIZE / MICRO_BATCH_SIZE)), 1)
cfg["training"]["gradient_checkpointing"]      = False
cfg["training"]["eval_splits"]                 = ["dev_hermes", "dev_xlam"]
cfg["training"]["eval_limit"]                  = 10000
cfg["training"]["probe_split"]                 = PROBE_SPLIT
cfg["training"]["probe_every_steps"]           = PROBE_EVERY_STEPS
cfg["training"]["probe_percent"]               = PROBE_PERCENT
cfg["training"]["probe_limit"]                 = PROBE_LIMIT
cfg["training"]["freeze_encoder_stages"]       = []

cfg["model"]["backbone_name"] = BACKBONE
cfg["model"]["max_seq_len"]   = MAX_SEQ_LEN
cfg.setdefault("data", {})["pack_tools"] = PACK_TOOLS

CONFIG_PATH = "configs/colab_lean_v3_3.yaml"
with open(CONFIG_PATH, "w") as fh:
    yaml.safe_dump(cfg, fh)

train_lean_main([
    "--config", CONFIG_PATH,
    "--experiment_name", EXPERIMENT_NAME,
    "--output_dir", RESULTS_DIR,
    "--data_dir", TENSOR_DIR,
    "--stages", "T2",
    "--device", DEVICE,
    "--seed", str(SEED),
])'''),

    (CODE, r'''# 3b. Training Curve Summary & Best-Checkpoint LeanResolver Init
import glob, json, os
import torch
from transformers import AutoTokenizer

from span_template import SPECIAL_MARKER_TOKENS
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver
from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.train.config import TrainingConfig

exp_root = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_seed{SEED}")
log_path = os.path.join(exp_root, "train_log.jsonl")
if os.path.exists(log_path):
    with open(log_path) as fh:
        _tlog = [json.loads(l) for l in fh if l.strip()]
    for r in _tlog:
        if r.get("event") in ("epoch_eval", "epoch_eval_macro"):
            print(r)

best_cp = os.path.join(exp_root, "best_checkpoint")
if os.path.exists(os.path.join(best_cp, "model.pt")):
    CHECKPOINT_DIR = best_cp
else:
    cps = sorted(glob.glob(os.path.join(exp_root, "**", "model.pt"), recursive=True))
    CHECKPOINT_DIR = os.path.dirname(cps[-1]) if cps else None
if CHECKPOINT_DIR is None:
    raise SystemExit(f"No model.pt found under {exp_root}")
print("CHECKPOINT_DIR =", CHECKPOINT_DIR)

cfg_obj = TrainingConfig.from_yaml(CONFIG_PATH)
tok_eval = AutoTokenizer.from_pretrained(cfg_obj.model.backbone_name, use_fast=True)
encoder = SpanCallTransformerEncoder(
    model_name_or_path=cfg_obj.model.backbone_name,
    hidden_dropout_prob=cfg_obj.model.dropout,
    tokenizer=tok_eval,
)
encoder.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))
model = LeanSpanCallModel(
    encoder=encoder,
    head_dim=cfg_obj.model.head_dim,
    max_span_width=cfg_obj.model.max_span_width,
    hidden_dim=encoder.hidden_dim,
)

resolver = LeanResolver(
    model_path=CHECKPOINT_DIR,
    encoder=encoder,
    model=model,
    tokenizer=tok_eval,
    max_seq_len=cfg_obj.model.max_seq_len,
    max_span_width=cfg_obj.model.max_span_width,
    max_cands=MAX_CANDS,
    max_calls=1 if SINGLE_CALL_ONLY else int(getattr(cfg_obj.data, "max_calls", 1)),
    call_min_prob=float(getattr(cfg_obj.data, "call_min_prob", 0.25)),
    call_min_prob_ratio=float(getattr(cfg_obj.data, "call_min_prob_ratio", 0.5)),
    min_joint_margin=MIN_JOINT_MARGIN,
    collapse_guard=True,
    prevent_param_collapse=True,
    resolve_mode="two_step",
)
if DEVICE in ("cuda", "mps"):
    model.to(DEVICE)
model.eval()
print("LeanResolver ready (two_step, max_cands =", MAX_CANDS, ")")'''),

    (MD, r'''---
## 4. Two-Step Inference on Frozen Tier 1 & Tier 2 Eval Splits (`data/eval_v3_3/`)'''),

    (CODE, r'''import json, os, time
from pathlib import Path
from spancall.data.schema import UnifiedSample
from spancall.eval.llm_parity import verify_eval_manifest

_manifest = verify_eval_manifest(EVAL_DIR)
TARGET_SPLITS = [
    "fair_hermes_eval",
    "fair_xlam_eval",
    "full_hermes_eval",
    "full_xlam_eval",
    "full_hermes_test_ood_schema",
    "full_xlam_test_ood_schema",
    "full_hermes_test_ood_para",
    "full_xlam_test_ood_para",
    "full_hermes_test_adversarial",
    "full_xlam_test_adversarial",
]


def _predict_frozen_split(split_name: str) -> None:
    if split_name not in _manifest["splits"]:
        print(f"[eval] {split_name} not in manifest — skipped")
        return
    src = Path(EVAL_DIR) / _manifest["splits"][split_name]["filename"]
    out_p = Path(RESULTS_DIR) / f"preds_{split_name}_twostep.jsonl"
    samples = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                samples.append(UnifiedSample.from_dict(json.loads(line)))
    batch_in = [
        {
            "candidate_schemas": s.tools,
            "utterance": s.utterance,
            "recent_context": s.recent_context,
            "now": s.current_time_iso,
        }
        for s in samples
    ]
    t0 = time.perf_counter()
    results = resolver.resolve_candidates_batch(batch_in, batch_size=EVAL_BATCH_SIZE, num_workers=EVAL_WORKERS)
    wall_s = time.perf_counter() - t0
    with open(out_p, "w", encoding="utf-8") as out_fh:
        for s, res in zip(samples, results):
            rec = {
                "sample_id": s.sample_id,
                "stratum": s.stratum.value,
                "hit": bool(res.hit),
                "status": getattr(res, "status", "hit" if res.hit else "abstain"),
                "tool_name": res.tool_name,
                "payload": res.payload,
                "payload_partial": getattr(res, "payload_partial", {}),
                "calls": [c.as_dict() for c in (getattr(res, "calls", None) or [])],
                "confidence": float(getattr(res, "confidence", res.accept_score) or 0.0),
                "joint_margin": float(getattr(res, "joint_margin", 0.0) or 0.0),
                "abstain_reason": res.abstain_reason,
                "latency_ms": float(getattr(res, "latency_ms", 0.0) or 0.0),
            }
            out_fh.write(json.dumps(rec, default=str) + "\n")
    print(f"[eval] {split_name:<30} n={len(samples):>5} wall={wall_s:.1f}s -> {out_p}")


for _sp in TARGET_SPLITS:
    _predict_frozen_split(_sp)'''),

    (MD, r'''---
## 5. Unified Audited Export (`scripts/export_v3_3_results.py`)'''),

    (CODE, r'''import subprocess, sys
from pathlib import Path

EXPORT_DIR = Path(RESULTS_DIR) / "export"
subprocess.run([
    sys.executable, str(PROJECT_ROOT / "scripts" / "export_v3_3_results.py"),
    "--eval-dir", EVAL_DIR,
    "--results-dir", RESULTS_DIR,
    "--out-dir", str(EXPORT_DIR),
    "--bootstrap-n", "0",
], check=True)

print((EXPORT_DIR / "audit_report.md").read_text(encoding="utf-8"))'''),

    (MD, r'''---
## 6. Guarded Hugging Face Push & Results Archive'''),

    (CODE, r'''import json, os, shutil, tarfile
from pathlib import Path

# Persist resolve_config.json with head_dim, resolve_mode, and max_cands
_rcfg = {
    "version": "v3.3",
    "backbone_name": BACKBONE,
    "head_dim": int(cfg_obj.model.head_dim),
    "max_seq_len": int(cfg_obj.model.max_seq_len),
    "max_span_width": int(cfg_obj.model.max_span_width),
    "max_cands": int(MAX_CANDS),
    "resolve_mode": "two_step",
    "min_joint_margin": float(MIN_JOINT_MARGIN),
    "collapse_guard": True,
    "prevent_param_collapse": True,
}
with open(os.path.join(CHECKPOINT_DIR, "resolve_config.json"), "w", encoding="utf-8") as _fh:
    json.dump(_rcfg, _fh, indent=2)
    _fh.write("\n")

if HF_TOKEN and HF_REPO_ID:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", private=HF_PUSH_PRIVATE, exist_ok=True)
    api.upload_folder(folder_path=CHECKPOINT_DIR, repo_id=HF_REPO_ID, repo_type="model")
    print(f"[hf] Pushed v3.3 checkpoint -> https://huggingface.co/{HF_REPO_ID}")
else:
    print("[hf] Skipped push (HF_TOKEN not set).")

# Snapshot eval manifest alongside results so local export_v3_3_results.py
# can mechanically verify byte-identical data/eval_v3_3 parity with Colab.
_src_manifest = Path(EVAL_DIR) / "manifest.json"
if _src_manifest.is_file():
    shutil.copy2(_src_manifest, Path(RESULTS_DIR) / "eval_manifest.json")

_archive = Path(RESULTS_DIR).with_suffix(".tar.gz")
with tarfile.open(_archive, "w:gz") as _tar:
    _tar.add(RESULTS_DIR, arcname=Path(RESULTS_DIR).name)
print(f"[archive] Wrote {_archive} (copy locally for scripts/eval_llm_local.py + scripts/export_v3_3_results.py)")

if DRIVE_MOUNT:
    shutil.copy2(_archive, os.path.join(DRIVE_OUT, _archive.name))
    print("[drive] Copied archive to", DRIVE_OUT)
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(_archive))'''),
]


def _guard() -> Any:
    """Load ``_guard_code_cells`` from the v2.6 builder (the shared AST scope guard)."""
    v26 = HERE / "build_lean_v2_6_colab_notebook.py"
    src = v26.read_text(encoding="utf-8")
    start = src.index("def _cell_source")
    end = src.rindex("_guard_code_cells(cells)")
    ns: Dict[str, Any] = {"__name__": "build_lean_v3_3_guard"}
    exec(compile(src[start:end], str(v26), "exec"), ns)
    return ns["_guard_code_cells"]


def _executed_records(out_path: Path) -> Dict[str, Dict[str, Any]]:
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
            cell = nbformat.from_dict({**record, "source": body})
        else:
            cell = new_code_cell(body) if kind == CODE else new_markdown_cell(body)
        if idx < len(existing) and existing[idx].get("id"):
            if "".join(existing[idx].get("source", [])).rstrip("\n") == src.rstrip("\n"):
                cell["id"] = existing[idx]["id"]
        out.append(cell)
    return out


def build(out_path: Path = OUT, check: bool = False) -> Path:
    out_path = Path(out_path)
    executed = _executed_records(out_path)
    previous = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    cells = _cell_objects(executed, previous.get("cells") or [])
    _guard()(cells)

    nb = new_notebook(cells=cells)
    for key, value in (previous.get("metadata") or {}).items():
        nb.metadata[key] = value
    nb.metadata.setdefault("kernelspec", {
        "display_name": "Python 3", "language": "python", "name": "python3",
    })
    nb.metadata.setdefault("language_info", {"name": "python", "version": "3.11"})
    nb.metadata.setdefault("colab", {"gpuType": "A100", "toc_visible": True, "provenance": []})

    if check:
        current = json.loads(out_path.read_text(encoding="utf-8"))
        on_disk = [(c["cell_type"], "".join(c["source"]).rstrip("\n")) for c in current["cells"]]
        expected = [(c["cell_type"], c["source"].rstrip("\n")) for c in cells]
        if on_disk != expected:
            raise SystemExit(
                f"{out_path} is stale. Run: python {Path(__file__).name}"
            )
        print(f"{out_path} matches the builder ({len(cells)} cells)")
        return out_path

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.reads(json.dumps(written), as_version=4))
    print(f"wrote {out_path} - {len(cells)} cells, valid nbformat 4 JSON")
    return out_path


if __name__ == "__main__":
    build(check="--check" in sys.argv[1:])

