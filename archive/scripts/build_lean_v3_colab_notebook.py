#!/usr/bin/env python3
"""Generate notebooks/spancall_lean_v3_train_eval.ipynb via nbformat (valid JSON guaranteed).

SpanCall Lean Flow **v3** — Two-Step-Only, Larger Stratified Eval, Probe Eval, HF Push
=========================================================================================

The architecture is unchanged from v2.7 (Hermes V1 + xLAM-60k 1-call subset,
``answerdotai/ModernBERT-large``, ``MAX_SEQ_LEN = 4096``, object-param
decomposition into dotted leaves). What v3 changes is *what the notebook
measures and how it reports it*:

1. **One extraction mode.** ``two_step`` is the only mode: it is set on
   ``LeanResolver`` at construction, and the retired ``single_pass`` prediction
   run, its waterfall and every table row derived from it are deleted. Keeping
   two modes in the notebook invited reading a retired number as a candidate.
2. **Larger, stratified evaluation.** ``EVAL_LIMIT`` 500 -> 2000 and
   ``TEST_LIMIT`` 200 -> 600. §4.0b prints the slice composition as
   corpus x tool-count bucket, and the new §5c pivots exact-match per
   ``(split, bucket)`` so behaviour against distractor-pool size is visible
   instead of averaged away.
3. **Corpus separation at the source.** ``hermes_eval`` and ``xlam_eval`` are
   scored as their own slices (§5, §5b, §5c); no blended number backs a
   Hermes-vs-xLAM claim.
4. **Mid-training probe eval.** A small fixed probe split is scored every
   ``PROBE_EVERY_STEPS`` optimiser steps. The probe rows are drawn from the
   *dev* corpora and filtered to those that did **not** land in ``dev.jsonl``,
   so the probe is disjoint from dev and never touches the ``hermes_eval``
   slice the run is judged on.
5. **Detailed training logs.** ``train/*`` step losses (total, pointer, tool,
   LR, elapsed time) and ``probe_eval`` metrics are printed live during
   training and summarized in §3b from ``train_log.jsonl``.
6. **Guarded HF export/push (§7).** Writes a self-describing export directory
   and uploads it only when an HF token exists. ``LeanSpanCallModel`` is a plain
   ``nn.Module`` with **no** ``save_pretrained()``, so the artifact is a
   ``state_dict`` + tokenizer + ``resolve_config.json`` (which is the honest
   shape — a transformers-style ``config.json`` would misdescribe it).

Derivation strategy
-------------------
``scripts/build_lean_v2_7_colab_notebook.py`` already patches the v2.6 sources.
Duplicating its ~300-line ``_V27_FIXES`` table here would guarantee drift, so
this builder **execs the v2.7 builder's declaration section** (everything up to
its ``_CELLS`` assembly) to reuse ``_patch`` and ``_V27_FIXES`` verbatim, then
applies one more layer: v3 path tokens, v3 prose, and :data:`_V3_FIXES`.

Patch order per cell is therefore
``v2.7 patch -> v3 path tokens -> v3 prose -> _V3_FIXES``, which means every
``_V3_FIXES`` NEEDLE is written against *post-v2.7* text and must not itself
contain a token a later step would rewrite.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
V26_SCRIPT = HERE / "build_lean_v2_6_colab_notebook.py"
V27_SCRIPT = HERE / "build_lean_v2_7_colab_notebook.py"
OUT = HERE / ".." / "notebooks" / "spancall_lean_v3_train_eval.ipynb"

# --------------------------------------------------------------------------
# Reuse the v2.7 builder's declaration section (v2.6 sources + v2.7 fixes)
# --------------------------------------------------------------------------
# Running the v2.7 module wholesale would rewrite its own notebook as a side
# effect and then leave us nothing to assemble. Slicing at ``_CELLS`` stops
# right before its build tail while keeping ``_NS`` (the v2.6 cell constants),
# ``_patch`` and ``_V27_FIXES`` in scope.
_V27_SRC = V27_SCRIPT.read_text(encoding="utf-8")
# Start at the v2.7 builder's import block, not at ``_SRC = ...``: the slice has
# to carry ``HERE``/``V26_SCRIPT``/``nbformat`` into the exec namespace, otherwise
# the v2.7 module dies on NameError before it ever builds ``_NS``.
_V27_START = _V27_SRC.index("import json")
_V27_END = _V27_SRC.index("_CELLS: Dict[str, Any] = {}")
_V27: Dict[str, Any] = {
    "__name__": "build_lean_v2_7_colab_notebook_sources",
    "__file__": str(V27_SCRIPT),
}
exec(
    compile(
        "from __future__ import annotations\n" + _V27_SRC[_V27_START:_V27_END],
        str(V27_SCRIPT),
        "exec",
    ),
    _V27,
)

_V26_NS: Dict[str, Any] = _V27["_NS"]
_V27_PATCH = _V27["_patch"]
_V27_PATCH_PATHS = _V27["_patch_paths"]

# --------------------------------------------------------------------------
# v3 path tokens
# --------------------------------------------------------------------------
# ``processed_v2_7`` is deliberately NOT bumped: v3 does not change the grammar
# or the decomposition, so it trains on the splits v2.7 already produced.
# Rewriting them would force a full re-align for no experimental reason.
_V3_VERSION_RULES = [
    (re.compile(r"tensorized_lean_v2_7"), "tensorized_lean_v3"),
    (re.compile(r"results_lean_v2_7"), "results_lean_v3"),
    (re.compile(r"colab_lean_v2_7\.yaml"), "colab_lean_v3.yaml"),
]

#: Prose-only rewrites. Everything else that still says "v2.7" is a comment
#: about a v2.7 *behaviour still in force* (object decomposition, Gate C, the
#: flat/nested eval comparison), so it must stay verbatim.
_V3_REPLACEMENTS = [
    ("LEAN FLOW v2.7 CONFIG", "LEAN FLOW v3 CONFIG"),
    ("# SpanCall Lean v2.7 Run Summary", "# SpanCall Lean v3 Run Summary"),
    (
        "# 0b. Configuration & Environment (Full Data, v2.7 object decomposition)",
        "# 0b. Configuration & Environment (Full Data, v3 two-step only)",
    ),
    ('[v2.7] {len(_missing_v2)} of', '[v3] {len(_missing_v2)} of'),
]

# --------------------------------------------------------------------------
# v3 prose cells
# --------------------------------------------------------------------------
MARKDOWN_TITLE = """# SpanCall Lean Flow **v3** — Two-Step-Only, Larger Stratified Eval, Probe Eval, HF Push

End-to-end training and evaluation notebook for the **v3** change set. The architecture is v2.7 (Hermes V1 + `Salesforce/xlam-function-calling-60k` 1-call subset, `answerdotai/ModernBERT-large`, `MAX_SEQ_LEN = 4096`, `object` params decomposed into dotted scalar leaves and reassembled by the resolver); what changes is **what this notebook measures and how it reports it**:

- **One extraction mode.** `two_step` is now the *only* mode — it is set on `LeanResolver` at construction, and the retired `single_pass` prediction run, its waterfall, and every table row derived from it are gone. A two-mode notebook invited reading a retired number as a live candidate.
- **Larger, stratified evaluation.** `EVAL_LIMIT = 2000` (was 500) and `TEST_LIMIT = 600` (was 200). §4.0b prints the slice composition as **corpus × tool-count bucket**, and the new §5c pivots exact-match per `(split, bucket)` so behaviour against distractor-pool size is visible instead of being averaged across it.
- **Corpus separation at the source.** `hermes_eval` and `xlam_eval` are scored as their own slices (§5, §5b, §5c). No blended number is used to support a Hermes-vs-xLAM claim.
- **Mid-training probe eval.** A small fixed probe split (`PROBE_N = 300` rows, built from *dev* source rows that did **not** land in `dev.jsonl`, so it is disjoint from both `dev` and the `hermes_eval` slice) is scored every `PROBE_EVERY_STEPS = 1000` optimiser steps. The per-epoch dev eval is kept for train/val loss; the probe curve is what makes convergence and overfit visible *mid-run*.
- **Detailed training logs.** Step-level training losses (total, pointer, tool/tool_binary), learning rate, and elapsed time are logged to `train_log.jsonl`, printed live during training, and summarized in §3b.
- **Guarded HF export/push (§7).** Writes a self-describing export directory (`model.pt` + tokenizer + `resolve_config.json` + provenance + auto-generated model card) and uploads it only when an HF token is available.

**Path changes vs v2.7:** `data/tensorized_lean_v3` and `results_lean_v3`. The processed splits stay `data/processed_v2_7`: v3 changes evaluation protocol, not the grammar, so forcing a re-align would churn labels for no experimental reason."""

MARKDOWN_SETUP = """---
## 0. Setup & Configuration

Clones or updates the SpanCall repository, installs dependencies in editable mode, and configures the v3 paths (`data/tensorized_lean_v3`, `results_lean_v3`), the mid-training probe knobs, the tool-count bucket edges, and the HF export target (prompted via `input()`). Nothing here requires a key — the HF push is guarded later."""

MARKDOWN_TRAIN = """---
## 3. Training — Lean Flow (`scripts/train_lean.py`)

Trains `LeanSpanCallModel` (`ModernBERT-large` + `ToolSelectionHead` H1 + `LeanPointerHead` H3) on stage `T2` with full data (`TRAIN_FRACTION = 1.0`), effective batch size 128, `bf16`, and best-checkpoint restoration.

v3 adds two things to the training loop itself, both configured here and implemented in `LeanTrainer.run_stage`:

- **Mid-training probe eval.** A small probe split is scored every `PROBE_EVERY_STEPS` optimiser steps (`probe_percent` is the fallback cadence when no step count is set) and written to `train_log.jsonl` as `event="probe_eval"`. It fires only on an accumulation-window boundary, so the weights being scored are exactly the ones the next micro-batch trains from — scoring mid-window would report a mix of stale and updated weights.
- **Detailed step logging.** `train` events (total loss, pointer loss, tool/tool_binary loss, LR, elapsed time) are written to `train_log.jsonl` and printed live every `total_steps // 100` steps, then summarized in §3b."""

MARKDOWN_EVAL_STEP2 = """---
### 4b. Step 2 in Isolation — Parameter Extraction (`H3`, Oracle Tool Given)

Scores `H3` (`resolver.try_resolve(gt_tool, ...)`) under the two-step render (`T=1`) with **Oracle Candidates** (`protect_parity_spans=True`, gates OFF) on `F_func` to measure the pointer head's isolated ranking ceiling."""

CODE_EVAL_STEP2 = '''import math
from spancall.lean.eval_p1 import eval_param_extraction_lean
from spancall.lean.resolver import LeanResolver

def _fmt_p1(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)

# Lean v3: keep only 2b (Two-Step Render, Oracle Cands, protect=T, gates=OFF) on F_func
# to cut Cell 4b from 10 passes down to 1 pass.
_r_2b = LeanResolver(
    model=model, encoder=encoder, tokenizer=tokenizer,
    max_seq_len=cfg.model.max_seq_len, max_span_width=cfg.model.max_span_width,
    max_calls=MAX_CALLS, call_min_prob=CALL_MIN_PROB, call_min_prob_ratio=CALL_MIN_PROB_RATIO,
    min_joint_margin=0.0, collapse_guard=False, prevent_param_collapse=False,
    resolve_mode="two_step",
)
_mf = eval_param_extraction_lean(
    CLEAN_EVAL_PATH, resolver=_r_2b, data_dir=DATA_DIR, limit=None,
    measure_candidate_parity=True, protect_parity_spans=True,
    context_mode="single", stratum_filter="F", exclude_tools=["json_output"], quiet=True,
)

print("=== Step 2b: Oracle Cands (T=1, protect=T, gates=OFF, F_func) ===")
print(f"{'Stage / Config':<44}{'Parity':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Coll_F':>8}{'Exact_F':>9}")
print("-" * 93)
print(f"{'2b. Oracle Cands (protect=T, gates=OFF)':<44}{_fmt_p1(_mf['candidate_parity']):>8}"
      f"{_fmt_p1(_mf['extraction_precision']):>8}{_fmt_p1(_mf['extraction_recall']):>8}"
      f"{_fmt_p1(_mf['extraction_f1']):>8}{_fmt_p1(_mf['pointer_collapse_rate_F']):>8}"
      f"{_fmt_p1(_mf['exact_payload_match_F']):>9}")'''

MARKDOWN_EVAL_STEP3 = """---
### 4c. Step 3 — Composed E2E (`two_step`), Stage Waterfall & Headline Table

Runs E2E inference on `SUBSET_EVAL_PATH` in `two_step` mode — the only extraction mode in v3 — prints the stage-by-stage waterfall funnel, and compares `B1a (regex)` vs `SpanCall Lean (two_step)` via `scripts/eval_toolcall.py`."""

MARKDOWN_TOOL_PIVOT = """---
### 5c. Tool-Count Bucket Pivot (Hermes vs xLAM)

Re-runs `scripts/eval_toolcall.py` per **tool-count bucket** (`1-2`, `3-5`, `6-10`, `11+`) on `hermes_eval` and `xlam_eval`, then prints one pivot.

The bucket is keyed on `len(sample.tools)` — the distractor-pool size H1 must actually choose from — **not** on the number of ground-truth calls. That distinction matters: the call count moves when `SINGLE_CALL_ONLY` filtering is applied, which would make the pivot's row definitions depend on a filter rather than on the row. Bucketing on `len(tools)` keeps each bucket well defined no matter what ran before it."""

MARKDOWN_HF_EXPORT = """---
## 7. Hugging Face Export & (Guarded) Push

Writes a self-describing export directory and uploads it **only when an HF token is available**.

- `model.pt` — `state_dict` of the trained `LeanSpanCallModel`. Note it is a plain `nn.Module` with **no `save_pretrained()`**, so a `state_dict` plus the rebuild recipe is the honest artifact; emitting a transformers-shaped `config.json` would misdescribe the architecture.
- tokenizer files, `resolve_config.json` (backbone, `max_seq_len`, `max_span_width`, `max_calls`, gate thresholds, `resolve_mode="two_step"`, template stamp) and `provenance.json`.
- `README.md` — an auto-generated model card carrying this run's headline metrics.

Without a token the export is still written locally and the push is skipped, so this cell is a no-op on CI and on local smoke runs."""

# --------------------------------------------------------------------------
# v3-added code cells
# --------------------------------------------------------------------------
CODE_TOOL_BUCKET_PIVOT = '''# 5c. Tool-count bucket pivot (per split, per bucket, both systems)
import json, os, subprocess, sys
from collections import defaultdict

# Default scope is the two slices that each carry ONE corpus, so the pivot's
# rows are genuinely (corpus x bucket) instead of a blended slice re-filtered
# after the fact. Flip PIVOT_ALL_SPLITS to cover every test split at roughly
# four times the subprocess count.
PIVOT_ALL_SPLITS = False
PIVOT_SPLITS = (TEST_SPLITS + [EVAL_SPLIT]) if PIVOT_ALL_SPLITS else [EVAL_SPLIT, "xlam_eval"]
BUCKETS = [b[2] for b in TOOL_BUCKET_EDGES]

_pivot = []
for _split in PIVOT_SPLITS:
    _sub = SUBSET_EVAL_PATH if (_split == EVAL_SPLIT and os.path.exists(SUBSET_EVAL_PATH)) \\
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
                _fh.write(json.dumps(_r, default=str) + "\\n")
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
                        _fh.write(json.dumps(_p, default=str) + "\\n")
            _out = os.path.join(RESULTS_DIR, f"toolcall_metrics_{_split}_{_bucket}_{_label}.json")
            _cmd = [sys.executable, "scripts/eval_toolcall.py", "--test_data", _bsub]
            _cmd += ["--baseline", "b1a"] if _label == "b1a" else ["--predictions", _bp]
            _cmd += ["--output", _out]
            subprocess.run(_cmd, check=False)
            _pivot.append((_split, _corpus_of_row(_brows[0]), _bucket, _label, _out))

print("\\n=== 5c. exact_F by tool-count bucket (corpus x bucket) ===")
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
    print("  no bucket rows - no subset was available.")'''

CODE_HF_EXPORT = '''# 7. HF export + guarded push (model card generated from this run)
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
    "version": "v3",
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
    f"# SpanCall Lean v3 — `{EXPERIMENT_NAME}`",
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
_card += ["", "_Auto-generated by `notebooks/spancall_lean_v3_train_eval.ipynb`._", ""]
with open(os.path.join(HF_EXPORT_DIR, "README.md"), "w", encoding="utf-8") as _fh:
    _fh.write("\\n".join(_card))

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
        commit_message=f"SpanCall Lean v3 ({EXPERIMENT_NAME}, seed={SEED})",
    )
    print(f"[hf] pushed -> https://huggingface.co/{HF_REPO_ID}")'''

# --------------------------------------------------------------------------
# v3 surgical cell fixes
# --------------------------------------------------------------------------
# NEEDLEs are written against **post-v2.7** text (v2.7 already bumped the path
# tokens and applied its own fixes) *after* the v3 path/prose rewrites, and must
# not contain a token a later step would rewrite.
_V3_FIXES: Dict[str, Any] = {
    # -- §0b config: probe/HF/bucket knobs + a bigger eval slice -------------
    "CODE_CONFIG": [
        (
            "from spancall.agent.backends.litellm_adapter import _load_dotenv_values\n"
            "\n"
            'ENV = {**_load_dotenv_values(PROJECT_ROOT / ".env"), **os.environ}\n'
            'HF_DATASET_ID = ENV.get("DATASET_NAME", "convoicon/spancall")\n'
            'HF_TOKEN      = ENV.get("HUGGING_FACE_TOKEN") or ENV.get("HF_TOKEN")\n'
            "if HF_TOKEN:\n"
            '    os.environ["HF_TOKEN"] = HF_TOKEN\n'
            '    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN',
            'HF_DATASET_ID = "convoicon/spancall"',
        ),
        # EVAL_LIMIT 500 -> 2000 answers "the evaluation set should be larger".
        # Both limits are top-of-notebook constants so the runtime cost stays
        # adjustable without touching a cell body.
        (
            "EVAL_LIMIT        = 500\n"
            "TEST_LIMIT        = 200",
            "EVAL_LIMIT        = 2000\n"
            "TEST_LIMIT        = 600\n"
            "\n"
            "# ---- Lean v3: mid-training probe eval (T13) -------------------------------\n"
            "# A SMALL, fixed set scored every PROBE_EVERY_STEPS optimiser steps so\n"
            "# convergence / overfit is visible mid-run instead of once per epoch.\n"
            "# Sources are the *dev* corpora, filtered to rows that did NOT land in\n"
            "# `dev.jsonl` (see the tensorize cell) and never the `hermes_eval` slice --\n"
            "# the probe must not leak into the number the run is judged on.\n"
            'PROBE_SPLIT       = "probe"\n'
            "PROBE_N           = 300\n"
            "PROBE_EVERY_STEPS = 1000\n"
            "PROBE_PERCENT     = 0.10   # fallback cadence: every 10% of optimiser steps\n"
            "PROBE_LIMIT       = 300\n"
            'PROBE_SOURCES     = ["xlam_dev.jsonl", "hermes_dev.jsonl"]\n'
            "\n"
            "# ---- Lean v3: Hugging Face Hub (interactive input) -----------------------\n"
            "# Press Enter at any prompt to keep the default (or skip optional token).\n"
            "# Non-interactive runs (e.g. nbconvert / CI) catch EOFError and use defaults.\n"
            "def _prompt(label: str, default: str = \"\") -> str:\n"
            '    suffix = f" [{default}]" if default else " [optional — Enter to skip]"\n'
            "    try:\n"
            '        val = input(f"{label}{suffix}: ").strip()\n'
            "    except (EOFError, OSError):\n"
            '        val = ""\n'
            "    return val or default\n"
            "\n"
            'HF_TOKEN = _prompt("Hugging Face access token")\n'
            "if HF_TOKEN:\n"
            '    os.environ["HF_TOKEN"] = HF_TOKEN\n'
            '    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN\n'
            "\n"
            'HF_REPO_ID      = _prompt("HF model repo ID", "convoicon/spancall-lean-v3")\n'
            "HF_PUSH_PRIVATE = False\n"
            "\n"
            "# ---- Lean v3: tool-count buckets for the eval pivot ----------------------\n"
            "# Buckets are keyed on ``len(sample.tools)`` -- the distractor-pool size H1\n"
            "# must actually choose from -- NOT on the number of GT calls, which moves\n"
            "# when SINGLE_CALL_ONLY filtering is applied and would make the pivot's row\n"
            "# definitions depend on which filter ran rather than on the row itself.\n"
            'TOOL_BUCKET_EDGES = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, None, "11+")]\n'
            "\n"
            "\n"
            "def _tool_bucket(n_tools) -> str:\n"
            '    """Map a row\'s tool count onto its bucket label."""\n'
            "    n = int(n_tools or 0)\n"
            "    for _lo, _hi, _label in TOOL_BUCKET_EDGES:\n"
            "        if n >= _lo and (_hi is None or n <= _hi):\n"
            "            return _label\n"
            '    return "<1"\n'
            "\n"
            "\n"
            "def _corpus_of_row(row) -> str:\n"
            '    """Corpus label for a raw row (source_dataset, then sample_id prefix)."""\n'
            '    _sd = str((row or {}).get("source_dataset") or "")\n'
            '    if _sd.startswith("xlam"):\n'
            '        return "xlam"\n'
            '    if _sd.startswith("hermes"):\n'
            '        return "hermes"\n'
            '    return "xlam" if str((row or {}).get("sample_id") or "").startswith("xlam") else "hermes"',
        ),
        (
            'print(f"EVAL_LIMIT        = {EVAL_LIMIT}  TEST_LIMIT = {TEST_LIMIT}")',
            'print(f"EVAL_LIMIT        = {EVAL_LIMIT}  TEST_LIMIT = {TEST_LIMIT}")\n'
            'print(f"PROBE             = {PROBE_SPLIT} n={PROBE_N} every={PROBE_EVERY_STEPS} "\n'
            '      f"opt-steps (limit={PROBE_LIMIT})")\n'
            'print(f"HF_REPO_ID        = {HF_REPO_ID}  (push guarded on an HF token)")',
        ),
    ],
    # -- §2 tensorizer: PACK_TOOLS=6 (~4 tools/row mean) + disjoint probe ----
    "CODE_TENSORIZE": [
        (
            "PACK_TOOLS = 3\n"
            "try:\n"
            '    with open(os.path.join(str(PROJECT_ROOT), "configs", "default.yaml")) as _fh:\n'
            "        _default_cfg = yaml.safe_load(_fh) or {}\n"
            '    PACK_TOOLS = int((_default_cfg.get("data") or {}).get("pack_tools", 3) or 3)\n'
            "except Exception as _exc:\n"
            '    print(f"[!] configs/default.yaml unreadable ({_exc}); using PACK_TOOLS={PACK_TOOLS}")',
            "# Lean v3: PACK_TOOLS = 6 with JITTER_PACK_TOOLS = True draws the per-row\n"
            "# floor from [1, 6] (uniform mean 3.5; ~4.0 tools/row after native multi-tool\n"
            "# rows), giving H1 harder distractor sets while keeping T=1 in-distribution.\n"
            "PACK_TOOLS = 6",
        ),
        (
            "finally:\n"
            "    if _shared_ex is not None:\n"
            "        _shared_ex.shutdown(wait=True)",
            "    # --- Lean v3: build the probe split (small, fixed, DISJOINT from dev) ---\n"
            "    # The probe is what LeanTrainer scores every PROBE_EVERY_STEPS optimiser\n"
            "    # steps. It must be (a) small, so that cadence is affordable, and (b) NOT\n"
            "    # a subset of dev.jsonl -- otherwise the probe curve would just re-measure\n"
            "    # the rows the per-epoch dev eval already reports, and a plateau would be\n"
            "    # indistinguishable from a re-scored flat dev set.\n"
            "    _dev_ids = set()\n"
            '    _dev_tensor_path = os.path.join(TENSOR_DIR, "dev.jsonl")\n'
            "    if os.path.exists(_dev_tensor_path):\n"
            "        with open(_dev_tensor_path, \"r\") as _fh_dev:\n"
            "            for _l in _fh_dev:\n"
            "                if _l.strip():\n"
            '                    _dev_ids.add(json.loads(_l).get("sample_id"))\n'
            "    _probe_raw: List[str] = []\n"
            "    _probe_cap = max(PROBE_N * 3, PROBE_N)\n"
            "    for _pf in PROBE_SOURCES:\n"
            "        if len(_probe_raw) >= _probe_cap:\n"
            "            break\n"
            "        _pf_path = os.path.join(DATA_DIR, _pf)\n"
            "        if not os.path.exists(_pf_path):\n"
            '            print(f"  [probe] source {_pf} missing - skipped")\n'
            "            continue\n"
            "        for _l in _iter_raw_lines(_pf_path):\n"
            '            if json.loads(_l).get("sample_id") in _dev_ids:\n'
            "                continue\n"
            "            _probe_raw.append(_l)\n"
            "            if len(_probe_raw) >= _probe_cap:\n"
            "                break\n"
            "    random.Random(int(SEED) + 977).shuffle(_probe_raw)\n"
            "    _prows, _pdrops, _perrors = _tensorize_many(\n"
            "        _probe_raw, True, PROBE_N, TENSORIZE_WORKERS, executor=_shared_ex\n"
            "    )\n"
            "    _probe_out = os.path.join(TENSOR_DIR, f\"{PROBE_SPLIT}.jsonl\")\n"
            "    with open(_probe_out, \"w\") as _fh_p:\n"
            '        _fh_p.write("\\n".join(_prows) + ("\\n" if _prows else ""))\n'
            "    _probe_ids = {\n"
            '        json.loads(_l).get("sample_id") for _l in _prows if _l.strip()\n'
            "    } - {None}\n"
            "    print(f\"[probe] {len(_prows)} rows -> {_probe_out} \"\n"
            "          f\"(sources={PROBE_SOURCES}, overlap_with_dev={len(_probe_ids & _dev_ids)}, \"\n"
            "          f\"render_drops={dict(_pdrops)}, errors={dict(_perrors)})\")\n"
            "    if not _prows:\n"
            '        print("[probe] EMPTY - mid-training probe eval will not run. Check that "\n'
            '              "PROBE_SOURCES rows survive tensorization.")\n'
            "finally:\n"
            "    if _shared_ex is not None:\n"
            "        _shared_ex.shutdown(wait=True)",
        ),
    ],
    # -- §3 trainer config: probe knobs into cfg["training"] ---------------
    "CODE_TRAIN_CFG_AND_RUN": [
        (
            'cfg["training"]["eval_splits"] = ["dev"]\n'
            'cfg["training"]["eval_limit"] = 2000',
            'cfg["training"]["eval_splits"] = ["dev"]\n'
            'cfg["training"]["eval_limit"] = 2000\n'
            "\n"
            "# Lean v3: mid-training probe eval. Counted in OPTIMISER steps, so it\n"
            "# fires many times per epoch rather than once -- that is the whole point\n"
            '# of the amendment ("run eval each 1k ... with a small test set"). The\n'
            "# per-epoch dev eval above stays, but it now only has to track train/val\n"
            "# loss, which is cheap.\n"
            'cfg["training"]["probe_split"] = PROBE_SPLIT\n'
            'cfg["training"]["probe_every_steps"] = PROBE_EVERY_STEPS\n'
            'cfg["training"]["probe_percent"] = PROBE_PERCENT\n'
            'cfg["training"]["probe_limit"] = PROBE_LIMIT',
        ),
    ],
    # -- §3b resolver: two_step at construction + probe curve ----------------
    "CODE_TELEMETRY_AND_RESOLVER": [
        (
            "    min_joint_margin=MIN_JOINT_MARGIN,\n"
            "    collapse_guard=True,\n"
            "    prevent_param_collapse=True,\n"
            ")",
            "    min_joint_margin=MIN_JOINT_MARGIN,\n"
            "    collapse_guard=True,\n"
            "    prevent_param_collapse=True,\n"
            "    # Lean v3: THE extraction mode, pinned at construction even though\n"
            "    # `LeanResolver` now defaults to `two_step`. Pinning it means the\n"
            "    # mode this notebook scores is visible on the line that creates the\n"
            "    # resolver, not inferred from a library default.\n"
            '    resolve_mode="two_step",\n'
            ")",
        ),
        (
            'best_cp = os.path.join(exp_root, "best_checkpoint")',
            "# Lean v3: step-level training log + mid-training probe curve.\n"
            "# Read straight from train_log.jsonl so this cell stays self-contained.\n"
            '_probe_log = os.path.join(exp_root, "train_log.jsonl")\n'
            "_train_step_recs = []\n"
            "_probe_recs = []\n"
            "if os.path.exists(_probe_log):\n"
            "    with open(_probe_log) as _fh_pb:\n"
            "        for _l in _fh_pb:\n"
            "            if not _l.strip():\n"
            "                continue\n"
            "            _rec = json.loads(_l)\n"
            '            _ev = _rec.get("event")\n'
            '            if _ev == "train":\n'
            "                _train_step_recs.append(_rec)\n"
            '            elif _ev == "probe_eval":\n'
            "                _probe_recs.append(_rec)\n"
            "if _train_step_recs:\n"
            '    print("\\n" + "=" * 84)\n'
            '    print(f"TRAINING STEP LOG ({len(_train_step_recs)} logged checkpoints)")\n'
            '    print("=" * 84)\n'
            '    print(f"  {\'ep\':>3}{\'opt_step\':>10}{\'loss\':>10}{\'ptr\':>10}"\n'
            '          f"{\'tool\':>10}{\'tool_bin\':>10}{\'lr\':>11}{\'elapsed\':>10}")\n'
            "    _stride = max(len(_train_step_recs) // 30, 1)\n"
            "    _shown = _train_step_recs[::_stride]\n"
            "    if _train_step_recs[-1] is not _shown[-1]:\n"
            "        _shown.append(_train_step_recs[-1])\n"
            "    for _tr in _shown:\n"
            '        _el = _num(_tr, "elapsed_s")\n'
            '        _el_str = f"{_el:>9.0f}s" if _el is not None else f"{\'—\':>10}"\n'
            '        print(f"  {_tr.get(\'epoch\', \'?\'):>3}"\n'
            '              f"{int(_tr.get(\'step\') or 0):>10}"\n'
            '              f"{_fmt_cell(_num(_tr, \'loss\'), 10, \'.4f\')}"\n'
            '              f"{_fmt_cell(_num(_tr, \'pointer\'), 10, \'.4f\')}"\n'
            '              f"{_fmt_cell(_num(_tr, \'tool\'), 10, \'.4f\')}"\n'
            '              f"{_fmt_cell(_num(_tr, \'tool_binary\'), 10, \'.4f\')}"\n'
            '              f"{_fmt_cell(_num(_tr, \'lr\'), 11, \'.2e\')}"\n'
            '              f"{_el_str}")\n'
            "if _probe_recs:\n"
            '    print("\\n" + "=" * 84)\n'
            '    print(f"MID-TRAINING PROBE CURVE (\'{PROBE_SPLIT}\', n<={PROBE_LIMIT}, "\n'
            '          f"every {PROBE_EVERY_STEPS} opt-steps)")\n'
            '    print("=" * 84)\n'
            '    print(f"  {\'opt_step\':>9}{\'ep\':>4}{\'n\':>6}{\'exact_call\':>12}"\n'
            '          f"{\'span_acc\':>10}{\'tool_acc\':>10}{\'loss\':>10}")\n'
            "    for _pr in _probe_recs:\n"
            "        print(f\"  {int(_pr.get('global_step') or 0):>9}{_pr.get('epoch', '?'):>4}\"\n"
            "              f\"{int(_pr.get('n') or 0):>6}\"\n"
            "              f\"{_fmt_cell(_num(_pr, 'exact_call'), 12)}\"\n"
            "              f\"{_fmt_cell(_num(_pr, 'span_acc'), 10)}\"\n"
            "              f\"{_fmt_cell(_num(_pr, 'tool_acc'), 10)}\"\n"
            "              f\"{_fmt_cell(_num(_pr, 'eval_loss'), 10, '.4f')}\")\n"
            "    _pb_best = max(_probe_recs, key=lambda r: float(r.get(\"exact_call\") or -1.0))\n"
            "    print(f\"  [peak] opt_step={_pb_best.get('global_step')} \"\n"
            "          f\"exact_call={float(_pb_best.get('exact_call') or 0.0):.4f}\")\n"
            "else:\n"
            '    print(f"\\n[probe] no probe_eval records in {_probe_log} - "\n'
            '          "probe eval was disabled for this run.")\n'
            "\n"
            'best_cp = os.path.join(exp_root, "best_checkpoint")',
        ),
    ],
    # -- §4.0 eval slice: composition table (corpus x tool bucket) -----------
    "CODE_EVAL_SLICE": [
        (
            "import json, os\n"
            "from span_template.template import RecentRow, build_input_multi",
            "import json, os\n"
            "from collections import Counter\n"
            "from span_template.template import RecentRow, build_input_multi",
        ),
        (
            'print("Sampled   by cohort:", _slice_info["sampled_by_cohort"], '
            'f"(total={_slice_info[\'sampled_total\']})")',
            'print("Sampled   by cohort:", _slice_info["sampled_by_cohort"], '
            'f"(total={_slice_info[\'sampled_total\']})")\n'
            "\n"
            "# --- Lean v3 §4.0b: evaluation-set composition (corpus x tool bucket) -----\n"
            "# Printed here so every metric table below can be read against what the\n"
            "# slice actually contains. An exact_F that moved because the slice gained\n"
            "# rows with 20 distractor tools must not be read as a model improvement,\n"
            "# and this table is the only place that fact is visible.\n"
            "_comp_rows = 0\n"
            "_comp_bucket = Counter()\n"
            "_comp_corpus = Counter()\n"
            "_comp_grid = Counter()\n"
            'with open(SUBSET_EVAL_PATH, encoding="utf-8") as _fh_comp:\n'
            "    for _l in _fh_comp:\n"
            "        if not _l.strip():\n"
            "            continue\n"
            "        _d = json.loads(_l)\n"
            '        _bk = _tool_bucket(len(_d.get("tools") or []))\n'
            "        _cp = _corpus_of_row(_d)\n"
            "        _comp_rows += 1\n"
            "        _comp_bucket[_bk] += 1\n"
            "        _comp_corpus[_cp] += 1\n"
            "        _comp_grid[(_cp, _bk)] += 1\n"
            'print("\\n=== 4.0b Evaluation-Set Composition (corpus x tool-count bucket) ===")\n'
            "_comp_buckets = [b[2] for b in TOOL_BUCKET_EDGES]\n"
            'print(f"{\'corpus\':<10}" + "".join(f"{_b:>9}" for _b in _comp_buckets) '
            '+ f"{\'total\':>9}")\n'
            'print("-" * (10 + 9 * (len(_comp_buckets) + 1)))\n'
            'for _cp in ("hermes", "xlam"):\n'
            "    _vals = [_comp_grid.get((_cp, _b), 0) for _b in _comp_buckets]\n"
            '    print(f"{_cp:<10}" + "".join(f"{_v:>9}" for _v in _vals) '
            '+ f"{_comp_corpus.get(_cp, 0):>9}")\n'
            'print(f"{\'total\':<10}" + "".join(f"{_comp_bucket.get(_b, 0):>9}" '
            'for _b in _comp_buckets) + f"{_comp_rows:>9}")\n'
            "_empty_buckets = [b for b in _comp_buckets if _comp_bucket.get(b, 0) == 0]\n"
            "if _empty_buckets:\n"
            '    print(f"  [i] empty bucket(s): {_empty_buckets} - the per-bucket "\n'
            '          "metrics in 5c skip them.")',
        ),
    ],
    # -- §4c: drop the retired single_pass run, waterfall and rows -----------
    "CODE_EVAL_STEP3": [
        (
            '_p_single = os.path.join(RESULTS_DIR, "preds_eval.jsonl")\n'
            '_p_two    = os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")\n'
            'predict_split(EVAL_SPLIT, _p_single, limit=None, resolve_mode="single_pass", '
            "src_path=SUBSET_EVAL_PATH)\n"
            'predict_split(EVAL_SPLIT, _p_two,    limit=None, resolve_mode="two_step",    '
            "src_path=SUBSET_EVAL_PATH)",
            "# Lean v3: `two_step` is the ONLY extraction mode. The retired\n"
            "# `single_pass` run -- and every waterfall line and table row derived from\n"
            "# it -- is gone. The mode is fixed on the resolver in 3b, so there is no\n"
            "# `resolve_mode=` argument here: leaving it optional is what allowed a\n"
            "# second mode to creep back in.\n"
            '_p_two = os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")\n'
            "predict_split(EVAL_SPLIT, _p_two, limit=None, src_path=SUBSET_EVAL_PATH)",
        ),
        (
            '_waterfall(_p_single, "single_pass")\n'
            '_waterfall(_p_two, "two_step")\n'
            "\n"
            "# Evaluate B1a, single_pass, and two_step via eval_toolcall.py\n"
            "for _args, _out in [\n"
            '    (["--baseline", "b1a"], f"toolcall_metrics_{EVAL_SPLIT}_b1a.json"),\n'
            '    (["--predictions", _p_single], f"toolcall_metrics_{EVAL_SPLIT}_lean.json"),\n'
            '    (["--predictions", _p_two], f"toolcall_metrics_{EVAL_SPLIT}_lean_twostep.json"),\n'
            "]:",
            '_waterfall(_p_two, "two_step")\n'
            "\n"
            "# Evaluate B1a and the one Lean mode via eval_toolcall.py\n"
            "for _args, _out in [\n"
            '    (["--baseline", "b1a"], f"toolcall_metrics_{EVAL_SPLIT}_b1a.json"),\n'
            '    (["--predictions", _p_two], f"toolcall_metrics_{EVAL_SPLIT}_lean_twostep.json"),\n'
            "]:",
        ),
        (
            "for _label, _fn in [\n"
            '    ("B1a (regex)", f"toolcall_metrics_{EVAL_SPLIT}_b1a.json"),\n'
            '    ("SpanCall Lean (single_pass)", f"toolcall_metrics_{EVAL_SPLIT}_lean.json"),\n'
            '    ("SpanCall Lean (two_step)", f"toolcall_metrics_{EVAL_SPLIT}_lean_twostep.json"),\n'
            "]:\n"
            "    _m = _load_metrics(os.path.join(RESULTS_DIR, _fn))\n"
            "    if not _m:\n"
            "        continue\n"
            '    _mf = _m.get("metrics_by_stratum", {}).get("F", {})\n'
            "    _row = {",
            "for _label, _fn in [\n"
            '    ("B1a (regex)", f"toolcall_metrics_{EVAL_SPLIT}_b1a.json"),\n'
            '    ("SpanCall Lean (two_step)", f"toolcall_metrics_{EVAL_SPLIT}_lean_twostep.json"),\n'
            "]:\n"
            "    _m = _load_metrics(os.path.join(RESULTS_DIR, _fn))\n"
            "    if not _m:\n"
            "        continue\n"
            '    _mf = _m.get("metrics_by_stratum", {}).get("F", {})\n'
            "    _row = {",
        ),
    ],
    # -- §4e stage attribution: no fallback to the retired prediction file ---
    "CODE_EVAL_STEP4": [
        (
            'PREDS_PATH = os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")\n'
            "if not os.path.exists(PREDS_PATH):\n"
            '    PREDS_PATH = os.path.join(RESULTS_DIR, "preds_eval.jsonl")',
            'PREDS_PATH = os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")\n'
            "# Lean v3: this cell used to fall back to `preds_eval.jsonl`, which\n"
            "# §4c no longer writes. The fallback could only ever resolve to a\n"
            "# stale pre-v3 file, so attribution would silently describe a run that\n"
            "# is not the one on screen.",
        ),
    ],
    # -- §4f FP deep dive: one prediction file, no single_pass fallback ------
    "CODE_EVAL_STEP5": [
        (
            "if not os.path.exists(PREDS_PATH):\n"
            '    PREDS_PATH = os.path.join(RESULTS_DIR, "preds_eval.jsonl")\n'
            "if not os.path.exists(PREDS_PATH):\n"
            '    raise SystemExit(f"no predictions at {PREDS_PATH} - run 4c first.")',
            "if not os.path.exists(PREDS_PATH):\n"
            '    raise SystemExit(f"no predictions at {PREDS_PATH} - run 4c first.")',
        ),
    ],
    # -- §4d object audit: score the mode the notebook actually runs ---------
    "CODE_OBJECT_AUDIT": [
        (
            '_PRED_PATH = os.path.join(RESULTS_DIR, "preds_eval.jsonl")',
            '_PRED_PATH = os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")',
        ),
    ],
    # -- §5 test persist: mode is fixed on the resolver ----------------------
    "CODE_TEST_AND_PERSIST": [
        (
            "    predict_split(\n"
            "        split,\n"
            '        os.path.join(RESULTS_DIR, f"preds_{split}.jsonl"),\n'
            "        limit=None,\n"
            '        resolve_mode="two_step",\n'
            "        src_path=sub_test_path,\n"
            "    )",
            "# Lean v3: no `resolve_mode=` argument -- the resolver is already\n"
            "# constructed with two_step, and passing it per call is what let the\n"
            "# notebook keep a second mode alive.\n"
            "    predict_split(\n"
            "        split,\n"
            '        os.path.join(RESULTS_DIR, f"preds_{split}.jsonl"),\n'
            "        limit=None,\n"
            "        src_path=sub_test_path,\n"
            "    )",
        ),
    ],
    # -- §5b per-corpus: one Lean prediction file ----------------------------
    "CODE_CORPUS_EVAL": [
        (
            "    if split == EVAL_SPLIT:\n"
            "        _pred_specs = [\n"
            '            ("b1a", None),\n'
            '            ("lean", os.path.join(RESULTS_DIR, "preds_eval.jsonl")),\n'
            '            ("lean_twostep", os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")),\n'
            "        ]",
            "    if split == EVAL_SPLIT:\n"
            "        # Lean v3: exactly one Lean arm, labelled `lean_twostep` so a\n"
            "        # stale `_lean.json` from a pre-v3 run can never be picked up.\n"
            "        _pred_specs = [\n"
            '            ("b1a", None),\n'
            '            ("lean_twostep", os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")),\n'
            "        ]",
        ),
    ],
    # -- §6 summary: headline rows + eval-split label -----------------------
    "CODE_SUMMARY_REPORT": [
        (
            "for _label, _fn in [\n"
            '    ("B1a (regex)", f"toolcall_metrics_{EVAL_SPLIT}_b1a.json"),\n'
            '    ("SpanCall Lean (single_pass)", f"toolcall_metrics_{EVAL_SPLIT}_lean.json"),\n'
            '    ("SpanCall Lean (two_step)", f"toolcall_metrics_{EVAL_SPLIT}_lean_twostep.json"),\n'
            "]:\n"
            "    _m = _load_metrics(os.path.join(RESULTS_DIR, _fn))\n"
            "    if not _m:\n"
            "        continue\n"
            '    _mf = _m.get("metrics_by_stratum", {}).get("F", {})\n'
            "    _headline[_label] = {",
            "for _label, _fn in [\n"
            '    ("B1a (regex)", f"toolcall_metrics_{EVAL_SPLIT}_b1a.json"),\n'
            '    ("SpanCall Lean (two_step)", f"toolcall_metrics_{EVAL_SPLIT}_lean_twostep.json"),\n'
            "]:\n"
            "    _m = _load_metrics(os.path.join(RESULTS_DIR, _fn))\n"
            "    if not _m:\n"
            "        continue\n"
            '    _mf = _m.get("metrics_by_stratum", {}).get("F", {})\n'
            "    _headline[_label] = {",
        ),
        (
            "_split_results = {}\n"
            "for _sp in TEST_SPLITS + [EVAL_SPLIT]:\n"
            "    _split_results[_sp] = {}\n"
            '    for _lbl in ("b1a", "lean"):\n'
            '        _m = _load_metrics(os.path.join(RESULTS_DIR, f"toolcall_metrics_{_sp}_{_lbl}.json"))',
            "_split_results = {}\n"
            "for _sp in TEST_SPLITS + [EVAL_SPLIT]:\n"
            "    _split_results[_sp] = {}\n"
            "    # Lean v3: the eval split's only Lean artifact is `_lean_twostep`, so\n"
            "    # reading `_lean` there would silently drop the eval split from the\n"
            "    # summary. Test splits still use the `_lean` label written by section 5.\n"
            '    _labels = ("b1a", "lean_twostep") if _sp == EVAL_SPLIT else ("b1a", "lean")\n'
            "    for _lbl in _labels:\n"
            '        _m = _load_metrics(os.path.join(RESULTS_DIR, f"toolcall_metrics_{_sp}_{_lbl}.json"))',
        ),
    ],
}


# --------------------------------------------------------------------------
# Patch pipeline
# --------------------------------------------------------------------------
def _patch(name: str, src: str) -> str:
    """v2.7 patch, then v3 path tokens, v3 prose, then ``_V3_FIXES``.

    Order matters for the same reason as in the v2.7 builder: the semantic
    fixes run **last** so their needles match the text the notebook will
    actually contain (post-v2.7, post-v3-bump, post-prose-rewrite), and their
    replacements are not re-mangled by an earlier regex.
    """
    out = _V27_PATCH(name, src)
    for pattern, new in _V3_VERSION_RULES:
        out = pattern.sub(new, out)
    for old, new in _V3_REPLACEMENTS:
        if old in out:
            out = out.replace(old, new)
    for old, new in _V3_FIXES.get(name, ()):
        if old not in out:
            raise SystemExit(
                f"[_patch] v3 fix for {name} did not apply: needle not found\n"
                f"--- needle ---\n{old[:500]}"
            )
        out = out.replace(old, new)
    return out


def _patch_paths(src: str) -> str:
    """v2.7 path/renumber pass, then the v3 path tokens.

    Markdown cells bypass :func:`_patch` because they legitimately contain
    *comparative* prose about v2.6/v2.7 that must survive verbatim, but their
    paths still have to move.
    """
    out = _V27_PATCH_PATHS(src)
    for pattern, new in _V3_VERSION_RULES:
        out = pattern.sub(new, out)
    return out


_MARKDOWN: Dict[str, str] = {
    _name: _patch_paths(_V26_NS[_name])
    for _name in (
        "MARKDOWN_DATA",
        "MARKDOWN_TENSORIZE",
        "MARKDOWN_EVAL_SLICE",
        "MARKDOWN_EVAL_STEP1",
        "MARKDOWN_EVAL_STEP4",
        "MARKDOWN_EVAL_STEP5",
        "MARKDOWN_TEST",
        "MARKDOWN_CORPUS_EVAL",
        "MARKDOWN_SUMMARY_REPORT",
    )
}

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
cells = [
    new_markdown_cell(MARKDOWN_TITLE),
    new_markdown_cell(MARKDOWN_SETUP),
    new_code_cell(_patch("CODE_SETUP", _V26_NS["CODE_SETUP"])),
    new_code_cell(_patch("CODE_CONFIG", _V26_NS["CODE_CONFIG"])),
    new_code_cell(_patch("CODE_RUNTIME_CHECK", _V26_NS["CODE_RUNTIME_CHECK"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_DATA"]),
    new_code_cell(_patch("CODE_DATA_AND_PREFLIGHT", _V26_NS["CODE_DATA_AND_PREFLIGHT"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_TENSORIZE"]),
    # Tensorizer now also emits the probe split.
    new_code_cell(_patch("CODE_TENSORIZE", _V26_NS["CODE_TENSORIZE"])),
    new_markdown_cell(MARKDOWN_TRAIN),
    new_code_cell(_patch("CODE_TRAIN_CFG_AND_RUN", _V26_NS["CODE_TRAIN_CFG_AND_RUN"])),
    new_code_cell(_patch("CODE_TELEMETRY_AND_RESOLVER", _V26_NS["CODE_TELEMETRY_AND_RESOLVER"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_EVAL_SLICE"]),
    new_code_cell(_patch("CODE_EVAL_SLICE", _V26_NS["CODE_EVAL_SLICE"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_EVAL_STEP1"]),
    new_code_cell(_patch("CODE_EVAL_STEP1", _V26_NS["CODE_EVAL_STEP1"])),
    new_markdown_cell(MARKDOWN_EVAL_STEP2),
    new_code_cell(CODE_EVAL_STEP2),
    new_markdown_cell(MARKDOWN_EVAL_STEP3),
    new_code_cell(_patch("CODE_EVAL_STEP3", _V26_NS["CODE_EVAL_STEP3"])),
    # v2.7 object-decomposition audit, kept ahead of the stage attribution so the
    # per-stage buckets can be read against the per-param-type table.
    new_markdown_cell(_patch_paths(_V27["MARKDOWN_OBJECT_AUDIT"])),
    new_code_cell(_patch("CODE_OBJECT_AUDIT", _V27["CODE_OBJECT_AUDIT"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_EVAL_STEP4"]),
    new_code_cell(_patch("CODE_EVAL_STEP4", _V26_NS["CODE_EVAL_STEP4"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_EVAL_STEP5"]),
    new_code_cell(_patch("CODE_EVAL_STEP5", _V26_NS["CODE_EVAL_STEP5"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_TEST"]),
    new_code_cell(_patch("CODE_TEST_AND_PERSIST", _V26_NS["CODE_TEST_AND_PERSIST"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_CORPUS_EVAL"]),
    new_code_cell(_patch("CODE_CORPUS_EVAL", _V26_NS["CODE_CORPUS_EVAL"])),
    # --- v3 addition: tool-count bucket pivot (corpus x bucket) ---
    new_markdown_cell(MARKDOWN_TOOL_PIVOT),
    new_code_cell(CODE_TOOL_BUCKET_PIVOT),
    new_markdown_cell(_MARKDOWN["MARKDOWN_SUMMARY_REPORT"]),
    new_code_cell(_patch("CODE_SUMMARY_REPORT", _V26_NS["CODE_SUMMARY_REPORT"])),
    # --- v3 addition: HF export + guarded push ---
    new_markdown_cell(MARKDOWN_HF_EXPORT),
    new_code_cell(CODE_HF_EXPORT),
]

# --------------------------------------------------------------------------
# Build-time guard: compile every code cell and resolve names
# --------------------------------------------------------------------------
_V26_SRC = V26_SCRIPT.read_text(encoding="utf-8")
_GUARD_START = _V26_SRC.index("def _cell_source")
_GUARD_END = _V26_SRC.rindex("_guard_code_cells(cells)")
_GUARD_NS: Dict[str, Any] = {"__name__": "build_lean_v3_guard"}
exec(compile(_V26_SRC[_GUARD_START:_GUARD_END], str(V26_SCRIPT), "exec"), _GUARD_NS)
_guard_code_cells = _GUARD_NS["_guard_code_cells"]

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