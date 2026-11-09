#!/usr/bin/env python3
"""Generate notebooks/spancall_experiment_verify.ipynb via nbformat.

Small-data verification experiments for the param-recall diagnosis
(docs/param_recall_diagnosis.md). Each cell is independently skippable via
flags, so the notebook runs partially on CPU or fully on Colab GPU.

Experiments:
  A. Label audit (H-D2)   — are unmatched GT positives silently labeled abstain?
  B. Oracle-candidate ceiling (H-cand) — is the candidate set the ceiling?
  C. Pointer-prob histogram (H-T1, pre-train) — abstain collapse on existing ckpt
  D. Mini-train (H-T2)     — F-heavy subset, abstain_weight=0.5, few epochs
  E. Post-train eval (H-T2/H-E2) — emission rate + stratum-F-only recall, pre vs post
"""
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parent / ".." / "notebooks" / "spancall_experiment_verify.ipynb"

# --------------------------------------------------------------------------
# Cell sources
# --------------------------------------------------------------------------

MARKDOWN_TITLE = """# SpanCall — Small-Data Experiment Notebook (verify the param-recall diagnosis)

Targets [`docs/param_recall_diagnosis.md`](../docs/param_recall_diagnosis.md): **why is param_recall ≈ 0.0022?**

Four hypotheses are tested cheaply on a small slice of the canonical splits:

| ID | Hypothesis | Experiment |
|---|---|---|
| H-D2 | Unmatched GT positives are silently labeled abstain (Killer #7) | A. Label audit |
| H-cand | GT values are missing from the regex candidate set (structural ceiling) | B. Oracle-candidate ceiling |
| H-T1 | H3 (pointer), trained T2-only on abstain-majority data, converged to abstain | C. Pointer-prob histogram |
| H-T2 | A positive-heavy mini-train (abstain_weight=0.5, F-heavy subset) lifts emission/recall | D. Mini-train |
| H-E2 | Stratum A rows inflate the recall denominator | E. Stratum-F-only metrics |

Run cells in order; every experiment is independently skippable via `RUN_*` flags."""

MARKDOWN_SETUP = """## 0. Setup — Pull Code from Git & Install"""
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
    !git pull origin {GIT_BRANCH} || echo "git pull note: continuing with local files"

print(f"SpanCall project root: {PROJECT_ROOT}")
%pip install -q -e ".[model,data,agent,dev]" huggingface_hub'''

MARKDOWN_CONFIG = """### 0b. Configuration & Environment

All experiment knobs are defined here, before any download or processing."""

CODE_CONFIG = '''import os, sys, json
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "spancall" / "__init__.py").is_file():
    if (PROJECT_ROOT.parent / "spancall" / "__init__.py").is_file():
        PROJECT_ROOT = PROJECT_ROOT.parent
    elif (PROJECT_ROOT / "spancall" / "spancall" / "__init__.py").is_file():
        PROJECT_ROOT = PROJECT_ROOT / "spancall"
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- experiment knobs (single source of truth) -----------------------------
EXPERIMENT_NAME  = "spancall_verify"     # artifacts namespace under RESULTS_DIR
SEED             = 13                    # seeds all deterministic subsampling
BACKBONE         = "Qwen/Qwen3-Embedding-0.6B"
MAX_SEQ_LEN      = 4096
MAX_SPAN_WIDTH   = 16
MAX_POS          = 3                     # positive slots per param (config fingerprint)

# --- data slice ---
SPLIT_INPUTS = {
    "exp_train": "hermes_train.jsonl",   # F-heavy mix -> t2 rows
    "exp_dev":   "hermes_dev.jsonl",
}
# F-heavy mix: keep ALL F rows up to TRAIN_LIMIT, then sample A rows at ~3:1 F:A
TRAIN_LIMIT = 800     # max rows read from hermes_train
DEV_LIMIT   = 200     # max rows read from hermes_dev
F_A_RATIO   = 3.0     # F : A mixing ratio for the training subset
MIN_F_ROWS  = 60      # stop early once this many F rows are in the train subset

# --- which experiments to run (independent flags) ---
RUN_SETUP         = True
RUN_AUDIT         = True   # H-D2
RUN_ORACLE        = True   # H-cand
RUN_PRE_HIST      = True   # H-T1 (needs EXISTING_CKPT_DIR)
RUN_MINITRAIN     = True   # H-T2
RUN_POST_EVAL     = True   # H-T2 + H-E2

# --- training knobs (H-T2) ---
NUM_EPOCHS        = 2
BATCH_SIZE        = 8
GRAD_ACCUM        = 4
ABSTAIN_WEIGHT    = 0.5    # HALF of the default 1.0 — the H-T2 knob
MIXED_PRECISION   = "bf16"
EVAL_LIMIT        = 200

# --- dataset pull (HF) ---
HF_DATASET_ID  = os.environ.get("DATASET_NAME", "convoicon/spancall")
HF_TOKEN       = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
FORCE_HF_DOWNLOAD = False            # True -> always re-pull canonical splits from HF

# --- paths ---
DATA_DIR          = str(PROJECT_ROOT / "data" / "processed")     # canonical splits
TENSOR_DIR        = str(PROJECT_ROOT / "data" / "tensorized_exp")  # experiment rows
RESULTS_DIR       = str(PROJECT_ROOT / "results" / EXPERIMENT_NAME)
EXISTING_CKPT_DIR = str(PROJECT_ROOT / "results" / "spancall_colab_seed42" / "T2" / "checkpoints")
MAKE_NEW_CKPT     = True   # save the mini-trained checkpoint under RESULTS_DIR

os.makedirs(TENSOR_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

print("=" * 72)
print("EXPERIMENT CONFIG")
print("=" * 72)
print(f"EXPERIMENT_NAME = {EXPERIMENT_NAME}  SEED = {SEED}")
print(f"RUN  audit={RUN_AUDIT}  oracle={RUN_ORACLE}  pre_hist={RUN_PRE_HIST}  minitrain={RUN_MINITRAIN}  post_eval={RUN_POST_EVAL}")
print(f"DATA slice: TRAIN_LIMIT={TRAIN_LIMIT}  DEV_LIMIT={DEV_LIMIT}  F:A={F_A_RATIO}")
print(f"TRAIN knobs: epochs={NUM_EPOCHS}  bs={BATCH_SIZE}  accum={GRAD_ACCUM}  abstain_weight={ABSTAIN_WEIGHT}")
print(f"DEVICE = {DEVICE}")
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"HF_DATASET_ID = {HF_DATASET_ID} (token={'set' if HF_TOKEN else 'unset'})")
print(f"DATA_DIR     = {DATA_DIR}")
print(f"TENSOR_DIR   = {TENSOR_DIR}")
print(f"RESULTS_DIR  = {RESULTS_DIR}")'''

MARKDOWN_DATA_PULL = """## 1. Dataset — Load the canonical splits (Hugging Face)

Downloads the canonical SpanCall splits (`hermes_train`, `hermes_dev`, …) from
`HF_DATASET_ID` (`convoicon/spancall`) via `snapshot_download` into
`data/processed/`. If the splits already exist locally they are used directly
(set `FORCE_HF_DOWNLOAD = True` to re-pull)."""

CODE_DATA_PULL = '''from huggingface_hub import snapshot_download

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
missing_after = [f for f in ("hermes_train.jsonl", "hermes_dev.jsonl")
                 if not os.path.exists(os.path.join(DATA_DIR, f))]
if missing_after:
    raise SystemExit(f"Dataset still missing in {DATA_DIR}: {missing_after}")'''

MARKDOWN_DATA = """## 2. Build the F-heavy experiment subset

Filters each canonical split to a **positive-rich mix** (H-T2's core idea):
keep all stratum-F rows (up to limits) and sample stratum-A rows so the
training slice is ~F:A=3:1 instead of the 84%-A eval composition that the
full pipeline inherits. The subset is written to `tensorized_exp/` as raw
`.jsonl` that the tensorizer (§2) consumes, then text rows are tensorized."""

CODE_DATA = '''# ---- build F-heavy subset from canonical splits ----
import json, random
from spancall.data.schema import UnifiedSample, Stratum

def build_f_heavy_subset(src: str, out_path: str, limit: int, f_a_ratio: float,
                         min_f_rows: int, seed: int) -> dict:
    """Read canonical split, keep F rows up to `limit`, then sample A rows
    so F:A ~ f_a_ratio (also keeping any other strata as-is but capped)."""
    rng = random.Random(seed)
    f_rows, a_rows, other_rows = [], [], []
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if len(f_rows) >= min_f_rows:
                break
            s = UnifiedSample.from_dict(json.loads(line))
            if s.stratum == Stratum.F:
                f_rows.append(line)
            elif s.stratum == Stratum.A:
                a_rows.append(line)
            else:
                other_rows.append(line)

    n_f = len(f_rows)
    n_a_target = int(n_f / f_a_ratio) if f_a_ratio > 0 else 0
    a_rows = a_rows[:n_a_target]
    all_rows = f_rows + a_rows + other_rows[: max(0, limit - n_f - len(a_rows))]

    rng.shuffle(all_rows)
    with open(out_path, "w") as fh:
        fh.write("\\n".join(all_rows) + "\\n")
    return {"out": out_path, "n_f": n_f, "n_a": len(a_rows), "n_other": len(other_rows),
            "n_total": len(all_rows), "f_pct": (n_f / max(1, len(all_rows))) * 100}

stats_train = build_f_heavy_subset(
    src=os.path.join(DATA_DIR, "hermes_train.jsonl"),
    out_path=os.path.join(TENSOR_DIR, "exp_train.jsonl"),
    limit=TRAIN_LIMIT, f_a_ratio=F_A_RATIO, min_f_rows=MIN_F_ROWS, seed=SEED)
stats_dev = build_f_heavy_subset(
    src=os.path.join(DATA_DIR, "hermes_dev.jsonl"),
    out_path=os.path.join(TENSOR_DIR, "exp_dev.jsonl"),
    limit=DEV_LIMIT, f_a_ratio=F_A_RATIO, min_f_rows=min(30, DEV_LIMIT // 2), seed=SEED)

print("train subset:", json.dumps(stats_train, indent=2))
print("dev subset:", json.dumps(stats_dev, indent=2))'''

MARKDOWN_TENSOR = """## 3. Tensorize the experiment subset (multi-tool packed)

Reuses the exact production tensorizer semantics from `scripts/build_colab_notebook.py`
(candidate building, matched-positive column resolution, MAX_C=512 capping, padding) —
so the same label-corruption paths that H-D2 suspects are exercised here."""

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

STRATUM_ABSTAIN = {Stratum.A}

tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_MARKER_TOKENS)})
mask_builder = CandidateMaskBuilder(max_seq_len=MAX_SEQ_LEN, max_span_width=MAX_SPAN_WIDTH)
print("tokenizer ready; vocab =", len(tokenizer), "| MAX_SEQ_LEN =", MAX_SEQ_LEN)'''

CODE_TENSOR_HELPERS = '''def _now_iso(s: UnifiedSample) -> date:
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
    H4_REASONS = ("OK", "ABSENT", "AMBIGUOUS", "NEEDS_REASONING", "OFF_WINDOW")
    return H4_REASONS.index(val) if val in H4_REASONS else 0


def _snap(cs: int, ce: int, offsets: List[tuple]) -> Optional[tuple]:
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
    anchors: List[int] = []
    for e in entries:
        if getattr(e, "kind", None) != "param_anchor" or getattr(e, "value", None) == "<|none|>":
            continue
        r = _snap(e.char_start, e.char_end, offsets)
        anchors.append(r[0] if r is not None else 0)
    return (anchors + [0] * n_params)[:n_params]


def tensorize_sample(s: UnifiedSample, tokenizer, mask_builder,
                     pack_multi_tools: bool = True) -> Optional[Dict[str, Any]]:
    """Production tensorizer: candidate gather, matched-positive columns, capping."""
    from span_template import build_input_multi
    if not s.tools or not s.ground_truth_calls:
        return None
    gt_tool_name = s.ground_truth_calls[0].tool_name
    gt_idx = next((i for i, t in enumerate(s.tools) if t.name == gt_tool_name), None)
    if gt_idx is None:
        return None
    gt_tool = s.tools[gt_idx]
    if not gt_tool.parameters:
        return None
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
            out = build_input(tooldef_to_template_tool(gt_tool), s.utterance,
                              recent_context=_recent_rows(s), now=_now_iso(s))
    except Exception:
        return None

    enc = tokenizer(out.text, return_offsets_mapping=True, add_special_tokens=False)
    ids: List[int] = enc["input_ids"]
    offsets: List[tuple] = enc["offset_mapping"]
    if len(ids) > MAX_SEQ_LEN:
        return None

    utt_entries = [e for e in out.entries if getattr(e, "segment", None) == "utterance"]
    if not utt_entries:
        return None
    u_cs = min(e.char_start for e in utt_entries)
    u_ce = max(e.char_end for e in utt_entries)
    usnap = _snap(u_cs, u_ce, offsets)
    if usnap is None:
        return None
    utt_starts, utt_ends = usnap

    tool_positions: List[int] = []
    if use_multi:
        t_entries = [e for e in out.entries if getattr(e, "kind", None) == "bridge_text"
                     and getattr(e, "tool", None) is not None]
        for e in t_entries:
            tsnap = _snap(e.char_start, e.char_end, offsets)
            tool_positions.append(tsnap[0] if tsnap else 0)

    K = len(gt_tool.parameters)
    if use_multi:
        t_entries = [e for e in out.entries if getattr(e, "kind", None) == "bridge_text"
                     and getattr(e, "tool", None) is not None]
        s_cs = t_entries[gt_idx].char_start
        s_ce = (t_entries[gt_idx + 1].char_start if gt_idx + 1 < len(t_entries)
                else min(e.char_start for e in utt_entries))
        p_entries = [e for e in out.entries
                     if getattr(e, "kind", None) == "param_anchor"
                     and getattr(e, "value", None) != "<|none|>"
                     and s_cs <= e.char_start < s_ce]
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

        # Fallback: direct GT-arg match when target_spans missed
        if not matched_cis and s.stratum == Stratum.F:
            gt_call = s.ground_truth_calls[0] if s.ground_truth_calls else None
            gt_args = dict(gt_call.arguments or {}) if gt_call else {}
            gt_val = gt_args.get(param.name)
            if gt_val is not None:
                g_val = str(gt_val).strip().lower()
                for ci, c in enumerate(cands):
                    if c.canon_value is not None:
                        c_val = str(c.canon_value).strip().lower()
                        if c.canon_value == gt_val or c_val == g_val:
                            if ci not in matched_cis:
                                matched_cis.append(ci)

        # Candidate capping (hard negatives near positives + uniform stride)
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
            pos_cols = [0]   # <-- the H-D2 silent-abstain path
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
    return row


def serialize_row(row):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}'''

CODE_TENSOR_RUN = '''# ---- tensorize exp_train / exp_dev ----
def tensorize_split(split: str, workers: int = None) -> dict:
    src = os.path.join(TENSOR_DIR, f"{split}.jsonl")
    out = os.path.join(TENSOR_DIR, f"t2_{split}.jsonl") if split == "exp_train" else os.path.join(TENSOR_DIR, f"{split}.jsonl")
    if split == "exp_train":
        out = os.path.join(TENSOR_DIR, "t2.jsonl")
    if split == "exp_dev":
        out = os.path.join(TENSOR_DIR, "dev.jsonl")
    rows = []
    dropped = 0
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context("fork") if workers else None
    raw_lines = []
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if line:
                raw_lines.append(line)
    n_workers = workers if workers is not None else max(1, min(os.cpu_count() or 1, 16))

    def _task(line):
        try:
            s = UnifiedSample.from_dict(json.loads(line))
            row = tensorize_sample(s, tokenizer, mask_builder, pack_multi_tools=True)
            return serialize_row(row) if row is not None else None
        except Exception:
            return None

    if ctx is not None and n_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                for res in ex.map(_task, raw_lines, chunksize=16):
                    if res is not None:
                        rows.append(res)
                    else:
                        dropped += 1
            with open(out, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\\n")
            return {"split": split, "rows": len(rows), "dropped": dropped, "file": out}
        except Exception:
            rows, dropped = [], 0
    for line in raw_lines:
        res = _task(line)
        if res is not None:
            rows.append(res)
        else:
            dropped += 1
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\\n")
    return {"split": split, "rows": len(rows), "dropped": dropped, "file": out}

print(json.dumps(tensorize_split("exp_train"), indent=2))
print(json.dumps(tensorize_split("exp_dev"), indent=2))'''

MARKDOWN_AUDIT = """## 4. Experiment A — Label audit (H-D2)

Re-scan the **tensorized** `t2.jsonl` rows and count, per F param, how many
matched ≥1 candidate column (`pos_indices` nonzero) vs. silently defaulted to
abstain (`pos_cols == [0]`). This directly measures the Killer #7 label-corruption rate."""

CODE_AUDIT = '''# H-D2: silent-abstain rate in the tensorized training rows
from collections import Counter

def audit_labels(t2_path: str, limit: int = None) -> dict:
    stats = Counter()
    by_type = Counter()
    examples = []
    with open(t2_path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            pos = r["pos_indices"]          # (K, MAX_POS), -1 padding
            stats["rows"] += 1
            for k, rowp in enumerate(pos):
                nz = [x for x in rowp if x > 0]
                if not nz:
                    stats["silent_abstain_params"] += 1
                    if len(examples) < 8:
                        examples.append({"row": i, "k": k, "pos": rowp})
                else:
                    stats["matched_params"] += 1
    total = stats["silent_abstain_params"] + stats["matched_params"]
    stats["total_params"] = total
    stats["silent_rate_pct"] = (stats["silent_abstain_params"] / max(1, total)) * 100
    stats["examples"] = examples
    return dict(stats)

audit = audit_labels(os.path.join(TENSOR_DIR, "t2.jsonl"))
print("Audit summary:")
print(f"  silent-abstain params : {audit['silent_abstain_params']}")
print(f"  matched params        : {audit['matched_params']}")
print(f"  silent rate           : {audit['silent_rate_pct']:.2f}%  (of {audit['total_params']})")
for ex in audit["examples"][:4]:
    print("  example:", ex)
with open(os.path.join(RESULTS_DIR, "audit_labels.json"), "w") as fh:
    json.dump(audit, fh, indent=2)
VERDICT_D2 = "CONFIRMED (>10% silent)" if audit["silent_rate_pct"] > 10 else \
             ("PARTIAL (5-10%)" if audit["silent_rate_pct"] > 5 else "REFUTED (<=5%)")
print(f"→ H-D2 verdict: {VERDICT_D2}  (silent_rate={audit['silent_rate_pct']:.2f}%)")'''

MARKDOWN_ORACLE = """## 5. Experiment B — Oracle-candidate ceiling (H-cand)

With the **pointer head bypassed** (oracle picks the GT-matching candidate per
param), how high can exact-payload match go on the F dev rows? If the ceiling
is far below 100%, the candidate set / cast path is the structural ceiling —
the model/H3 is not the only problem."""

CODE_ORACLE = '''# H-cand: oracle-pointer ceiling (pointer head bypassed, oracle candidate pick)
from collections import Counter
from span_template import build_input
from spancall.infer.cast import StrictTypeCaster, SchemaValidator
from spancall.eval.toolcall.evaluator import payload_equal, _deep_equal

def oracle_ceiling(dev_path: str, limit: int = None) -> dict:
    caster = StrictTypeCaster()
    validator = SchemaValidator()
    total = 0
    oracle_matches = 0
    unmatched: Counter = Counter()
    reasons: Counter = Counter()
    with open(dev_path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            s = UnifiedSample.from_dict(json.loads(line))
            if s.stratum != Stratum.F:
                continue
            if not s.tools or not s.ground_truth_calls:
                continue
            gt_call = s.ground_truth_calls[0]
            gt_tool = next((t for t in s.tools if t.name == gt_call.tool_name), None)
            if gt_tool is None:
                continue
            gt_args = dict(gt_call.arguments or {})
            try:
                out = build_input(tooldef_to_template_tool(gt_tool), s.utterance,
                                  recent_context=_recent_rows(s), now=_now_iso(s))
            except Exception:
                reasons["render_err"] += 1
                continue
            enc = tokenizer(out.text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = enc["offset_mapping"]
            total += 1
            oracle_payload: Dict[str, Any] = {}
            row_fail = False
            for param in gt_tool.parameters:
                if param.name not in gt_args:
                    continue
                gt_val = gt_args[param.name]
                masks = mask_builder.build_param_masks(param, out.entries, offsets, out.text)
                cands = list(masks["candidates"])
                picked = None
                for c in cands:
                    if c.canon_value is not None and _deep_equal(c.canon_value, gt_val):
                        picked = c
                        break
                if picked is not None:
                    ok, cast_val, reason = caster.cast_choice(picked, param, output=out)
                    oracle_payload[param.name] = cast_val if ok else picked.canon_value
                elif gt_val is None:
                    oracle_payload[param.name] = None
                else:
                    unmatched[f"unmatched:{param.name}"] += 1
                    row_fail = True
            if row_fail:
                reasons["unmatched_candidate"] += 1
            if payload_equal(gt_args, oracle_payload, schema=gt_tool):
                oracle_matches += 1
            else:
                reasons["payload_unequal"] += 1
    pct = (oracle_matches / max(1, total)) * 100
    return {"n_f": total, "oracle_matches": oracle_matches, "ceiling_pct": pct,
            "unmatched": dict(unmatched), "reasons": dict(reasons)}

ceiling = oracle_ceiling(os.path.join(TENSOR_DIR, "exp_dev.jsonl"), limit=DEV_LIMIT)
print("Oracle ceiling:", json.dumps(ceiling, indent=2))
with open(os.path.join(RESULTS_DIR, "oracle_ceiling.json"), "w") as fh:
    json.dump(ceiling, fh, indent=2)
VERDICT_CAND = ("CONFIRMED (ceiling < 90%)" if ceiling["ceiling_pct"] < 90
                else "REFUTED (ceiling >= 90% — H3/cast is the problem)")
print(f"→ H-cand verdict: {VERDICT_CAND}")'''

MARKDOWN_RESOLVER = """## 6. Shared resolver (pre/post training)

Builds `SpanCallResolver(raw_mode=True)` — bypasses the τ gate so any remaining
empty-payload rows must come from upstream (candidate generation / pointer /
cast), not gating. Two resolvers are constructed here:
- `resolver_pre`  → existing checkpoint (`EXISTING_CKPT_DIR`), for Experiment C
- `resolver_post` → mini-trained checkpoint (after Experiment D), for Experiment E

Each resolver is built lazily by `make_resolver(ckpt_dir)`."""

CODE_RESOLVER = '''import torch
from transformers import AutoTokenizer
from span_template import SPECIAL_MARKER_TOKENS
from spancall.data.schema import UnifiedSample
from spancall.infer.resolver import SpanCallResolver
from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.model.model import SpanCallModel
from spancall.train.config import TrainingConfig

def make_resolver(ckpt_dir: str):
    """Build a raw-mode SpanCallResolver loading state from ckpt_dir."""
    cfg = TrainingConfig.from_yaml("configs/default.yaml")
    tok = AutoTokenizer.from_pretrained(cfg.model.backbone_name, use_fast=True)
    enc = SpanCallTransformerEncoder(
        model_name_or_path=cfg.model.backbone_name,
        hidden_dropout_prob=cfg.model.dropout,
        tokenizer=tok,
    )
    enc.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))
    model = SpanCallModel(
        encoder=enc,
        heads=cfg.model.heads,
        head_dim=cfg.model.head_dim,
        max_span_width=cfg.model.max_span_width,
        hidden_dim=enc.hidden_dim,
    )
    res = SpanCallResolver(
        model_path=ckpt_dir, raw_mode=True,
        encoder=enc, model=model, tokenizer=tok,
        max_seq_len=cfg.model.max_seq_len,
        max_span_width=cfg.model.max_span_width,
    )
    if DEVICE in ("cuda", "mps"):
        model.to(DEVICE)
    model.eval()
    if enc is not None and hasattr(enc, "eval"):
        enc.eval()
    return res, tok

resolver_pre = None
resolver_post = None
if RUN_PRE_HIST and os.path.isdir(EXISTING_CKPT_DIR):
    resolver_pre, _tok = make_resolver(EXISTING_CKPT_DIR)
    print("resolver_pre ready (existing checkpoint =", EXISTING_CKPT_DIR, ")")
else:
    print("resolver_pre SKIPPED — EXISTING_CKPT_DIR not found:", EXISTING_CKPT_DIR)'''

MARKDOWN_PRE_HIST = """## 7. Experiment C — Pointer-prob histogram (H-T1, pre-train)

Run the **existing** checkpoint in raw mode over the dev subset, collecting:
- per-param `abstain_p` (H3 column-0 softmax mass)
- top-1 span probability
- **emission rate** (share of rows with non-empty `payload_partial`) — the raw-mode
  diagnostic that matters most (197/200 in the full run).

Also audits the anchor-0 fallback (`param_positions == 0` for params whose anchor
wasn't resolved at render time)."""

CODE_PRE_HIST = '''# H-T1: pointer-prob histogram + emission rate on the existing checkpoint
from collections import Counter, defaultdict

def probe_pointer(res, split_path: str, limit: int = None, label: str = "pre") -> dict:
    abstains, spans, emissions = [], [], {"rows": 0, "emitted": 0, "empty": 0}
    anchor_zero = 0
    anchor_rows = 0
    reason_counter = Counter()
    with open(split_path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            s = UnifiedSample.from_dict(json.loads(line))
            if s.stratum != Stratum.F:
                continue
            if not s.tools or not s.ground_truth_calls:
                continue
            gt_tool = next((t for t in s.tools if t.name == s.ground_truth_calls[0].tool_name), None)
            if gt_tool is None:
                continue
            try:
                rres = res.try_resolve(gt_tool, s.utterance, s.recent_context, s.current_time_iso)
            except Exception:
                continue
            emissions["rows"] += 1
            pp = getattr(rres, "payload_partial", {}) or {}
            if pp:
                emissions["emitted"] += 1
            else:
                emissions["empty"] += 1
            feats = getattr(rres, "features", {}) or {}
            reason_counter[feats.get("h4_reason")] += 1
            for pr in (getattr(rres, "parameters", {}) or {}).values():
                abstains.append(float(getattr(pr, "p_abstain", 0.0) or 0.0))
                spans.append(float(getattr(pr, "confidence", 0.0) or 0.0))
                if getattr(pr, "char_span", None) is not None:
                    cs = pr.char_span
                    if cs[0] == 0 and cs[1] == 0:
                        anchor_zero += 1
                anchor_rows += 1
    import statistics
    out = {
        "label": label,
        "rows": emissions["rows"],
        "emitted": emissions["emitted"],
        "empty": emissions["empty"],
        "emission_rate": emissions["emitted"] / max(1, emissions["rows"]),
        "n_param_resolutions": anchor_rows,
        "anchor_zero": anchor_zero,
        "mean_abstain_p": statistics.mean(abstains) if abstains else float("nan"),
        "mean_top1_span_conf": statistics.mean(spans) if spans else float("nan"),
        "h4_reasons": dict(reason_counter),
    }
    return out

if resolver_pre is not None:
    pre_hist = probe_pointer(resolver_pre, os.path.join(TENSOR_DIR, "exp_dev.jsonl"),
                             limit=DEV_LIMIT, label="pre")
    print(json.dumps(pre_hist, indent=2))
    with open(os.path.join(RESULTS_DIR, "pointer_pre.json"), "w") as fh:
        json.dump(pre_hist, fh, indent=2)
    VERDICT_T1_PRE = ("CONFIRMED (abstain collapse)" if pre_hist["mean_abstain_p"] > 0.7 else
                      ("PARTIAL" if pre_hist["mean_abstain_p"] > 0.5 else "REFUTED"))
    print(f"→ H-T1 pre verdict: {VERDICT_T1_PRE}  (mean_abstain_p={pre_hist['mean_abstain_p']:.3f}, emission_rate={pre_hist['emission_rate']:.3f})")
else:
    print("skipped — no existing checkpoint")'''

MARKDOWN_MINITRAIN = """## 8. Experiment D — Mini-train (H-T2)

Train the model on the **F-heavy** tensorized subset (`t2.jsonl`) with
**abstain_weight=0.5** (half the default) and 2 epochs — the exact knobs the
diagnosis predicts should de-collapse the pointer. Uses the same
`SpanCallTrainer` + T2 stage as the production recipe (scripts/train.py)."""

CODE_MINITRAIN = '''# H-T2: mini-train on F-heavy subset with reduced abstain weight
from transformers import AutoTokenizer
from span_template import SPECIAL_MARKER_TOKENS
from spancall.train.config import TrainingConfig
from spancall.train.trainer import SpanCallTrainer
from spancall.model.model import SpanCallModel
from spancall.model.encoder import SpanCallTransformerEncoder

if RUN_MINITRAIN:
    cfg = TrainingConfig.from_yaml("configs/default.yaml")
    cfg.experiment_name = EXPERIMENT_NAME
    cfg.seed = SEED
    cfg.output_dir = RESULTS_DIR
    cfg.data_dir = TENSOR_DIR
    cfg.data.fraction = 1.0            # tensorized files ARE the subset
    cfg.optimization.num_epochs = NUM_EPOCHS
    cfg.optimization.batch_size = BATCH_SIZE
    cfg.optimization.mixed_precision = MIXED_PRECISION
    cfg.optimization.abstain_weight = ABSTAIN_WEIGHT
    cfg.gradient_accumulation_steps = GRAD_ACCUM
    cfg.gradient_checkpointing = True
    cfg.model.max_seq_len = MAX_SEQ_LEN

    tok = AutoTokenizer.from_pretrained(cfg.model.backbone_name, use_fast=True)
    enc = SpanCallTransformerEncoder(
        model_name_or_path=cfg.model.backbone_name,
        hidden_dropout_prob=cfg.model.dropout,
        tokenizer=tok,
    )
    enc.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))
    model = SpanCallModel(
        encoder=enc,
        heads=cfg.model.heads,
        head_dim=cfg.model.head_dim,
        max_span_width=cfg.model.max_span_width,
        hidden_dim=enc.hidden_dim,
    )
    trainer = SpanCallTrainer(config=cfg, model=model, encoder=enc, tokenizer=tok,
                              device=DEVICE)
    summary = trainer.train(stages=("T2",))
    print("mini-train summary:", json.dumps(summary, indent=2, default=str))
    with open(os.path.join(RESULTS_DIR, "minitrain_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    NEW_CKPT = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(NEW_CKPT, exist_ok=True)
    _ckpt_saved = False
    try:
        trainer.save_checkpoint(NEW_CKPT)
        print("saved mini-trained checkpoint ->", NEW_CKPT)
        _ckpt_saved = True
    except Exception as exc:
        print("WARNING: checkpoint save failed:", exc)
    if _ckpt_saved and RUN_POST_EVAL:
        resolver_post, _tok = make_resolver(NEW_CKPT)
        print("resolver_post ready ->", NEW_CKPT)
else:
    print("skipped (RUN_MINITRAIN=False)")'''

MARKDOWN_POST_EVAL = """## 9. Experiment E — Post-train eval (H-T2 + H-E2)

Compare pre vs post on the SAME dev slice:

- pre  = `pointer_pre.json` (Experiment C)
- post = mini-trained checkpoint (Experiment D)

Metrics: **emission rate**, stratum-F-only param recall/precision/F1,
hit_rate, and error taxonomy — reporting both the headline (all strata) and
the honest F-only number to quantify H-E2's denominator inflation."""

CODE_POST_EVAL = '''# H-T2 + H-E2: post-train evaluation on the dev slice
import json

def evaluate_dev(res, split_path: str, limit: int = None, label: str = "post") -> dict:
    from spancall.eval.toolcall.evaluator import ToolCallEvaluator, _deep_equal
    samples, preds = [], []
    with open(split_path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            s = UnifiedSample.from_dict(json.loads(line))
            samples.append(s)
            try:
                rres = res.try_resolve_candidates(
                    s.tools, s.utterance, s.recent_context, s.current_time_iso)
            except Exception:
                rres = None
            preds.append({
                "sample_id": s.sample_id,
                "stratum": s.stratum.value,
                "hit": bool(rres.hit) if rres is not None else False,
                "status": getattr(rres, "status", "abstain"),
                "payload": getattr(rres, "payload", None),
                "payload_partial": getattr(rres, "payload_partial", {}) or {},
                "tool_name": getattr(rres, "tool_name", None),
                "confidence": float(getattr(rres, "confidence", 0.0) or 0.0),
                "accept_probability": float(getattr(rres, "accept_score", 0.0) or 0.0),
                "joint_margin": float(getattr(rres, "joint_margin", 0.0) or 0.0),
                "abstain_reason": getattr(rres, "abstain_reason", None),
            })
    ev = ToolCallEvaluator()
    m = ev.evaluate(samples, preds)
    fb = m.metrics_by_stratum["F"]
    out = {
        "label": label,
        "total": m.total_samples,
        "hit_rate": m.hit_rate,
        "exact_payload_match": m.exact_payload_match,
        "param_precision": m.param_precision,
        "param_recall": m.param_recall,
        "param_f1": m.param_f1,
        "stratum_F": {k: fb[k] for k in ("n", "hit_rate", "param_precision", "param_recall", "param_f1")},
        "abstain_reasons": m.abstain_reasons,
        "errors": m.errors_by_sample,
    }
    return out

POST = {}
if resolver_post is not None and RUN_POST_EVAL:
    POST = evaluate_dev(resolver_post, os.path.join(TENSOR_DIR, "exp_dev.jsonl"),
                        limit=DEV_LIMIT, label="post")
    print(json.dumps(POST, indent=2))
    with open(os.path.join(RESULTS_DIR, "eval_post.json"), "w") as fh:
        json.dump(POST, fh, indent=2)
elif not RUN_POST_EVAL:
    print("skipped (RUN_POST_EVAL=False)")
else:
    print("skipped — resolver_post not built (RUN_MINITRAIN or checkpoint missing)")'''

MARKDOWN_VERDICT = """## 10. Verdict — did the experiments confirm the diagnosis?

Maps measured numbers back to hypotheses and prints a single decisive table."""

CODE_VERDICT = '''# Verdict synthesis
import json, os

def _load(name):
    p = os.path.join(RESULTS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else {}

audit = _load("audit_labels.json")
ceiling = _load("oracle_ceiling.json")
pre = _load("pointer_pre.json")
post = _load("eval_post.json")
mini = _load("minitrain_summary.json")

rows = []
rows.append(("H-D2 silent-abstain rate", f"{audit.get('silent_rate_pct', float('nan')):.1f}%",
             "CONFIRMED" if audit.get('silent_rate_pct', 0) > 10 else "not confirmed"))
rows.append(("H-cand oracle ceiling", f"{ceiling.get('ceiling_pct', float('nan')):.1f}%",
             "CONFIRMED (ceil < 90%)" if ceiling.get('ceiling_pct', 100) < 90 else "REFUTED"))
if pre:
    rows.append(("H-T1 pre abstain_p", f"{pre.get('mean_abstain_p', float('nan')):.3f}",
                 "CONFIRMED" if pre.get('mean_abstain_p', 0) > 0.7 else "not confirmed"))
if pre and post:
    em_pre = pre.get("emission_rate", 0)
    em_post = post.get("hit_rate", 0)   # raw-mode emission proxy
    rows.append(("H-T2 emission pre->post", f"{em_pre:.3f} -> {em_post:.3f}",
                 "CONFIRMED" if em_post > em_pre * 2 else "not confirmed"))
if post:
    f = post.get("stratum_F", {})
    rows.append(("H-E2 F-only param_recall", f"{f.get('param_recall', float('nan')):.4f}",
                 "report F-only alongside headline"))

print("=" * 72)
print("VERDICT TABLE")
print("=" * 72)
for hypo, val, verdict in rows:
    print(f"  {hypo:<32} {val:<18} {verdict}")

print()
if pre:
    print(f"PRE-TRAIN : emission_rate={pre.get('emission_rate', float('nan')):.3f}  mean_abstain_p={pre.get('mean_abstain_p', float('nan')):.3f}")
if post:
    f = post.get("stratum_F", {})
    print(f"POST-TRAIN: hit_rate={post.get('hit_rate', float('nan')):.3f}  F.param_recall={f.get('param_recall', float('nan')):.4f}  F.n={f.get('n', 0):.0f}")
print()
print("Artifacts under", RESULTS_DIR)'''

cells = [
    new_markdown_cell(MARKDOWN_TITLE),
    new_markdown_cell(MARKDOWN_SETUP),
    new_code_cell(CODE_SETUP),
    new_markdown_cell(MARKDOWN_CONFIG),
    new_code_cell(CODE_CONFIG),
    new_markdown_cell(MARKDOWN_DATA_PULL),
    new_code_cell(CODE_DATA_PULL),
    new_markdown_cell(MARKDOWN_DATA),
    new_code_cell(CODE_DATA),
    new_markdown_cell(MARKDOWN_TENSOR),
    new_code_cell(CODE_TENSOR_SETUP),
    new_code_cell(CODE_TENSOR_HELPERS),
    new_code_cell(CODE_TENSOR_RUN),
    new_markdown_cell(MARKDOWN_AUDIT),
    new_code_cell(CODE_AUDIT),
    new_markdown_cell(MARKDOWN_ORACLE),
    new_code_cell(CODE_ORACLE),
    new_markdown_cell(MARKDOWN_RESOLVER),
    new_code_cell(CODE_RESOLVER),
    new_markdown_cell(MARKDOWN_PRE_HIST),
    new_code_cell(CODE_PRE_HIST),
    new_markdown_cell(MARKDOWN_MINITRAIN),
    new_code_cell(CODE_MINITRAIN),
    new_markdown_cell(MARKDOWN_POST_EVAL),
    new_code_cell(CODE_POST_EVAL),
    new_markdown_cell(MARKDOWN_VERDICT),
    new_code_cell(CODE_VERDICT),
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