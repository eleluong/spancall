# Plan: SpanCall Lean Flow **v3.3** — Fair Eval, Balanced Max-Data Training, Local Multi-Model LLM Eval & Unified Paper Export

## 0. Motivation (What v3.2 Revealed)

The Nov 19 full-scale v3.2 runs ([`plans/v3_2_paper_update_plan.md`](v3_2_paper_update_plan.md)) exposed four structural issues that make Hermes vs. xLAM comparisons unfair and paper auditing fragile:

1. **Unfair Eval Dataset (`hermes_eval` vs. `xlam_eval`)**
   - `SINGLE_CALL_ONLY` was applied *after* `build_stratified_eval_subset(limit=2000)`, dropping ~341 multi-call Hermes rows and shrinking `hermes_eval` to `n=1,659` (`72% F / 28% A`) while `xlam_eval` stayed at `n=2,000` (`60% F / 40% A`).
   - `hermes_eval` includes Hermes-only `F_json` (structured JSON mode, 65 rows) and `A_missing_gt_schema` (644 rows), whereas `xlam_eval` has 0 of either.
   - Complexity and tool-count skew: raw 1-call scoreable `hermes_eval` is `26%` complex (`86%` in `1-2` tools, `14%` in `3-5`, `0%` in `6-10`), while `xlam_eval` is `15%` complex (`41%` in `1-2`, `50%` in `3-5`, `9%` in `6-10`).
   - OOD/test evaluation was capped at `TEST_LIMIT=600` and omitted `xlam_test_ood_para` and `xlam_test_adversarial` entirely.
2. **Imbalanced & Under-Utilized Training/Dev Data**
   - `data/processed_v2/` holds **`69,412`** 1-call `xlam_train` rows vs. **`19,096`** 1-call scoreable `hermes_train` rows (`20,105` before `A_missing_gt_schema`).
   - v3.2 capped `XLAM_TRAIN_ROWS=15,000` (discarding **78%** of xLAM!) and `XLAM_DEV_ROWS=1,000` (discarding **80%** of `xlam_dev`), leaving training Hermes-heavy (`20.1k` vs. `15.0k`), which contributed to SpanCall scoring `.733` on Hermes vs. `.666` on xLAM while `gpt-oss-120b` scored `.423` on Hermes vs. `.818` on xLAM.
3. **Notebook-Bound Single-Model LLM Eval**
   - `notebooks/spancall_lean_v3_2_llm_baseline.ipynb` runs one LLM at a time in Colab, has no row-level resume/retry cache, and is gitignored — making multi-model sweeps slow, fragile, and hard to reproduce locally.
4. **Manual Paper Transcription & Audit Drift**
   - v3.2 paper tables had to be transcribed manually across 15+ cells of two separate notebooks (`train_eval` and `llm_baseline`). There was no single script to verify dataset hashes, re-score all raw prediction JSONLs through one evaluator, and emit paper-ready JSON/CSV/Markdown/LaTeX artifacts altogether.

---

## 1. Empirical Baseline of Available 1-Call Scoreable Data (`data/processed_v2/`)

Audited directly from `data/processed_v2/*.jsonl` (1-call rows, excluding `A_missing_gt_schema`):

| Split | 1-Call Scoreable | `F_func` (N / C) | `F_json` (N / C) | `A_clean` (N / C) | `A_adv` (N / C) |
| :--- | ---: | :---: | :---: | :---: | :---: |
| `hermes_train` | 19,096 | 13,859 / 2,502 | 793 / 103 | 166 / 440 | 806 / 427 |
| `xlam_train` | 69,412 | 46,756 / 12,244 | 0 / 0 | 3,690 / 397 | 5,120 / 1,205 |
| `hermes_dev` | 1,095 | 643 / 74 | 0 / 0 | 45 / 10 | 281 / 42 |
| `xlam_dev` | 4,906 | 2,439 / 0 | 0 / 0 | 929 / 2 | 1,535 / 1 |
| `hermes_eval` | 2,305 | 1,108 / 154 | 59 / 6 | 119 / 208 | 412 / 239 |
| `xlam_eval` | 9,101 | 3,802 / 777 | 0 / 0 | 1,669 / 167 | 2,244 / 442 |
| `hermes_test_ood_schema` | 2,069 | 1,015 / 140 | 0 / 0 | 135 / 164 | 453 / 162 |
| `xlam_test_ood_schema` | 9,814 | 5,010 / 0 | 0 / 0 | 1,815 / 0 | 2,989 / 0 |
| `hermes_test_ood_para` | 1,342 | 1,139 / 149 | 49 / 5 | 0 / 0 | 0 / 0 |
| `xlam_test_ood_para` | 6,876 | 6,456 / 420 | 0 / 0 | 0 / 0 | 0 / 0 |
| `hermes_test_adversarial` | 21,223 | 0 / 0 | 0 / 0 | 0 / 0 | 15,460 / 5,763 |
| `xlam_test_adversarial` | 88,327 | 0 / 0 | 0 / 0 | 0 / 0 | 74,825 / 13,502 |

---

## 2. Pillar 1 — Frozen Fair Eval Dataset (`scripts/build_fair_eval_v3_3.py` → `data/eval_v3_3/`)

### Design
Create a deterministic builder script `scripts/build_fair_eval_v3_3.py` that reads re-aligned `data/processed_v3_3/` (with v2.7 object decomposition + template stamp verified), applies **1-call + scoreable + Step-0 sanitize filtering FIRST**, and writes a frozen, content-addressed benchmark directory `data/eval_v3_3/` with `manifest.json`.

### Two-Tier Benchmark Structure
1. **Tier 1 — Matched Fair Eval (Primary SpanCall vs. Multi-LLM Benchmark)**
   - Files: `fair_hermes_eval.jsonl` and `fair_xlam_eval.jsonl` (plus combined `fair_eval_combined.jsonl`).
   - Excludes Hermes-only `F_json` (written separately to `hermes_eval_f_json.jsonl` for transparency, never mixed into the cross-corpus comparison).
   - **Exact 1:1 `(cohort, complexity)` cell matching** using `min(hermes_cell, xlam_cell)` (deterministic `seed=42`):
     - `F_func` normal: **`1,108`** each
     - `F_func` complex: **`154`** each (`F` total = **`1,262`** each, `12.2%` complex)
     - `A_clean` normal: **`119`** each, `A_clean` complex: **`167`** each (`A_clean` total = **`286`** each)
     - `A_adv` normal: **`412`** each, `A_adv` complex: **`239`** each (`A_adv` total = **`651`** each)
     - **Total**: **`2,199` rows per corpus** (`4,398` combined) — larger than v3.2 (`1,659` / `2,000`), with **exact 1:1 parity** on total size, `F / A_clean / A_adv` ratios, and `normal / complex` composition!
   - **Tool-count fairness note**: Hermes natively has almost zero `6-10` tool rows (`1` in `hermes_eval`), so tool-count buckets (`1-2`, `3-5`, `6-10`, `11+`) are preserved naturally and reported in the corpus × bucket pivot, plus the export script reports a **shared-bucket (`1-2` & `3-5`)** sub-slice so reviewers can verify tool-count-controlled parity.
   - Optional `--fair-limit` flag for smaller matched slices if ever needed, defaulting to `None` (full matched intersection `2,199` each).

2. **Tier 2 — Full 1-Call Scoreable Splits (Exhaustive SpanCall ID + OOD + Adversarial Eval)**
   - No arbitrary `EVAL_LIMIT=2000` or `TEST_LIMIT=600` caps:
     - `full_hermes_eval.jsonl` (`~2,305` rows)
     - `full_xlam_eval.jsonl` (`~9,101` rows)
     - `full_hermes_test_ood_schema.jsonl` (`~2,069` rows)
     - `full_xlam_test_ood_schema.jsonl` (`~9,814` rows)
     - `full_hermes_test_ood_para.jsonl` (`~1,342` rows)
     - `full_xlam_test_ood_para.jsonl` (`~6,876` rows)
     - `full_hermes_test_adversarial.jsonl` and `full_xlam_test_adversarial.jsonl` (stratified deterministic sample of `5,000` each by default via `--adv-limit 5000`, or `--adv-limit 0` for all `21k`/`88k` rows, matched in size between Hermes and xLAM).

3. **Integrity & Provenance (`data/eval_v3_3/manifest.json`)**
   - Records `version: "v3.3"`, `seed`, `template_stamp`, and for every file: `filename`, `tier`, `corpus`, `n_rows`, `sha256`, `cohort_counts`, `complexity_counts`, `tool_bucket_counts`, and `sample_id_hash`.
   - Step-0 marker-injection sanitization runs during build so every downstream consumer (SpanCall notebook, local LLM CLI, export script) reads the exact same clean JSONL files — eliminating v3.2's `subset_*` vs. `subset_*_clean` drift.
   - Optional `--push-hf convoicon/spancall` (or `hf_hub_download` in Colab) so Colab and local machines share byte-identical files verified against `manifest.json`.

---

## 3. Pillar 2 — Balanced Max-Data Training & Dev (`scripts/build_lean_v3_3_train_eval_notebook.py` + `spancall/lean/trainer.py`)

### Training Data Balance (Max Data, 1:1 Corpus Ratio)
- **Use 100% of `xlam_train`** (`69,412` 1-call rows, `XLAM_TRAIN_ROWS = None`).
- **Upsample `hermes_train`** (`19,096` 1-call scoreable rows → `69,412` rows: `3` full copies + `12,124` stratified remainder rows with `SEED`) to achieve an **exact 1:1 Hermes:xLAM ratio** (`~138,824` total training rows, vs. `28,991` in v3.2 — **4.8× more training data** and fully balanced!).
- **Fresh Distractor Jitter per Upsampled Copy**: Upsampling happens *before* tensorization, passing a distinct deterministic seed (`SEED + copy_idx * 1009`) to `ToolPool` (`pack_tools=6`, `jitter_pack_tools=True`) over the combined Hermes+xLAM schema pool. Every upsampled Hermes copy therefore sees **different distractor tools**, acting as distractor-packing data augmentation rather than duplicate memorization.
- Log exact pre- and post-upsampling counts, corpus ratios, cohort ratios, and `render_T` histograms into `results_lean_v3_3/run_metadata.json`.

### Dev, Probe & Macro-Averaged Early Stopping
- **Full Dev Splits (Separated by Corpus)**:
  - Tensorize `dev_hermes.jsonl` (100% of 1-call scoreable `hermes_dev`, `~1,095` rows) and `dev_xlam.jsonl` (100% of 1-call scoreable `xlam_dev`, `~4,906` rows) separately.
  - Pass `eval_splits = ["dev_hermes", "dev_xlam"]` to `LeanTrainer`.
- **Small Targeted Trainer Enhancement (`spancall/lean/trainer.py`)**:
  - Add `early_stopping_agg: str = "first"` to `OptimizationConfig` / `LeanTrainer` (default `"first"` preserves 100% backward compatibility with all existing tests and v3.2).
  - When `early_stopping_agg == "macro_mean"`, `LeanTrainer` evaluates all `eval_splits` at epoch end, computes `macro_exact_call = mean(exact_call(es) for es in eval_splits)`, logs `dev/macro_exact_call` + `dev/macro_eval_loss` to `train_log.jsonl` and W&B, and uses `macro_exact_call` for best-checkpoint saving, early stopping, and `restore_best_on_exit`.
- **Balanced Mid-Training Probe (`probe.jsonl`)**:
  - Draw `300` Hermes + `300` xLAM (`600` total) from `hermes_calib` / `xlam_calib` (since we now use 100% of `hermes_dev` and `xlam_dev`!), with a **hard `assert overlap_with_dev == 0 and overlap_with_eval == 0`**. (Using `calib` for the mid-training cadence probe keeps `dev` 100% intact and `eval` 100% untouched.)

### v3.3 Notebook (`scripts/build_lean_v3_3_train_eval_notebook.py` → `notebooks/spancall_lean_v3_3_train_eval.ipynb`)
- Built with the same AST cell guard, `--check` mode, and lossless output preservation as v3.2.
- Paths: `data/processed_v3_3`, `data/eval_v3_3`, `data/tensorized_lean_v3_3`, `results_lean_v3_3`, HF repo `convoicon/spancall-lean-v3-3`.
- Evaluates the trained `two_step` checkpoint (`EVAL_BATCH_SIZE=8`, `MAX_CANDS=1024`) on:
  - Tier 1: `fair_hermes_eval`, `fair_xlam_eval`
  - Tier 2: all `full_*` ID, OOD-schema, OOD-para, and adversarial splits
- Writes standardized prediction JSONLs (`preds_<split>_twostep.jsonl`), `run_metadata.json`, `train_log.jsonl`, and `stage_forensics_<split>.json` into `results_lean_v3_3/` (with optional zip/Drive/HF artifact backup so local `eval_llm_local.py` and `export_v3_3_results.py` can ingest them cleanly).

---

## 4. Pillar 3 — Local Multi-Model LLM Eval CLI (`scripts/eval_llm_local.py`)

Replaces `notebooks/spancall_lean_v3_2_llm_baseline.ipynb` for local execution across any number of LLMs.

### Architecture
- **Shared Parity Module (`spancall/eval/llm_parity.py`)**: Extract/reuse the core fairness helpers (`_openai_tools`, `_messages_for`, `_llm_pred_row`, `_lean_pred_row`, `score_system`, `merge_fallback`, `TOOL_BUCKET_EDGES`, `_tool_bucket`, `_corpus_of_row`) so `scripts/compare_llm_spancall.py`, `scripts/eval_llm_local.py`, and `scripts/export_v3_3_results.py` all share **one** implementation with zero drift.
- **Model Config (`configs/llm_eval_v3_3.yaml`)**: Declarative list of models to evaluate, each specifying:
  ```yaml
  models:
    - slug: gpt-oss-120b
      model: openai/gpt-oss-120b
      base_url_env: LLM_BASE_URL
      api_key_env: LLM_API_KEY
      temperature: 0.0
      max_tokens: 1024
      concurrency: 8
    - slug: qwen3-32b
      model: Qwen/Qwen3-32B
      base_url_env: QWEN_BASE_URL
      api_key_env: QWEN_API_KEY
      temperature: 0.0
      concurrency: 8
  ```
  Also supports quick single-model CLI overrides (`--model`, `--base-url`, `--api-key-env`, `--slug`) without editing YAML.
- **OpenAI SDK Direct Caller**: Uses `openai.OpenAI(base_url=..., api_key=...)` directly (works with OpenAI, OpenRouter, Together, Fireworks, Gemini OpenAI-compat, local vLLM, Ollama, LM Studio). Reads `.env` automatically if present; never prints or logs secret values.
- **Resumable, Fault-Tolerant Execution**:
  - Before calling any API, verifies `data/eval_v3_3/manifest.json` SHA256 for every requested split (`--splits fair_hermes_eval fair_xlam_eval` by default).
  - Writes append-only JSONL to `results_lean_v3_3/llm/<slug>/<split>_preds.jsonl` (one row per completed `sample_id` with `tool_calls`, `raw_text`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `error`, `model`, `timestamp`).
  - On restart, loads existing `sample_id`s and skips completed rows; `--retry-errors` re-runs only rows with `error != null`.
  - Bounded thread-pool concurrency (`--concurrency`) with exponential backoff + jitter on 429/5xx/timeouts.
  - Supports `--limit N` (for quick smoke tests, tagged as `"partial": true` in metadata) and `--fake-responses <path>` for offline deterministic CI tests.
- **Per-Model Summary**: After each model finishes, writes `results_lean_v3_3/llm/<slug>/summary.json` (including `llm_only` and, if `--spancall-dir` is provided, `spancall_plus_llm_fallback` scores).

---

## 5. Pillar 4 — Unified Paper & Audit Export Script (`scripts/export_v3_3_results.py`)

Single command to verify, re-score, and export every number for the paper and human audit:

```bash
.venv/bin/python scripts/export_v3_3_results.py \
    --eval-dir data/eval_v3_3 \
    --results-dir results_lean_v3_3 \
    --out-dir results_lean_v3_3/export
```

### Audit & Verification Gates (Fail-Loud by Default)
1. **Manifest SHA256 check**: Verifies every `data/eval_v3_3/*.jsonl` file matches `manifest.json`.
2. **100% `sample_id` coverage check**: Every SpanCall and LLM prediction JSONL must match the exact `sample_id` set of its split (no missing rows, no extra rows, no Hermes-on-xLAM mix-ups). Fails loudly unless `--allow-partial` is passed (which stamps `"audit_status": "PARTIAL"` in the output).
3. **Zero-drift re-scoring**: Re-computes all metrics directly from raw prediction JSONLs using `ToolCallEvaluator` (never copying pre-baked numbers from notebooks).
4. **Error-rate audit**: Flags any LLM run with `api_errors > 0` prominently in the audit report.

### Generated Artifacts (`results_lean_v3_3/export/`)
- **`final_report.json`**: Complete machine-readable canonical JSON containing:
  - Provenance (`git_commit`, `timestamp`, `eval_manifest_sha256`, `spancall_checkpoint`, `models_evaluated`, `audit_checks`).
  - Training / Dev / Probe curves parsed from `train_log.jsonl` + `run_metadata.json`.
  - Tier 1 Fair Eval metrics (`exact_F`, `hit_F`, `param_F1`, `false_hit_A`, `exact_all`, `call_set`, `lat_ms_mean`, `lat_ms_p50`, `lat_ms_p90`, `tok_in_mean`, `tok_out_mean`, `fallback_rate`) for SpanCall, every LLM, and every `SpanCall + LLM fallback` pair, per corpus + **Hermes/xLAM Macro-Average**, with **10k paired bootstrap 95% CIs** (`ComparativeEvaluator`).
  - Normal vs. Complex paired breakdown (SpanCall vs. all LLMs).
  - Corpus × Tool-Count Bucket pivot (`1-2`, `3-5`, `6-10`, `11+`, plus shared `1-2`/`3-5` parity).
  - Tier 2 Full ID & OOD / Adversarial generalization results (all 8 splits).
  - Optional Agentic 3-mode summary if `results_lean_v3_3/agentic/` is present.
- **`audit_report.md`**: Clean, human-readable Markdown report containing all tables + an explicit Audit Checklist (hash verification, row counts, coverage, API errors, batch size).
- **`tables/*.csv`**: One CSV per table (`fair_headline.csv`, `fair_complexity.csv`, `fair_tool_buckets.csv`, `full_ood_generalization.csv`, `train_dev_curve.csv`) for spreadsheet inspection / plotting.
- **`latex/*.tex`**: Auto-generated LaTeX `booktabs` tables (`tab_v3_3_headline.tex`, `tab_v3_3_complexity.tex`, `tab_v3_3_buckets.tex`, `tab_v3_3_ood.tex`) ready to `\input{}` into `research_paper/`.

---

## 6. Verification & Testing Plan

New dedicated test suite `tests/test_v3_3_pipeline.py` (offline, zero network, fast):
1. **Fair eval builder tests**: verifies 1-call/scoreable filtering, `F_json` exclusion from Tier 1, exact 1:1 `(cohort, complexity)` parity between `fair_hermes_eval` and `fair_xlam_eval`, deterministic reproducibility across runs, and `manifest.json` SHA256 integrity.
2. **Balanced training & trainer tests**: verifies 1:1 Hermes upsampling with distinct per-copy distractor jitter seeds, `probe` zero-overlap assertion, and `LeanTrainer` `early_stopping_agg="macro_mean"` selecting the epoch with highest macro-average across two dev splits (while preserving default `"first"` behavior).
3. **Local LLM eval CLI tests**: tests `scripts/eval_llm_local.py` with `--fake-responses`, verifying manifest SHA256 enforcement, resume/skip of completed `sample_id`s, `--retry-errors`, multi-model YAML loading, and `spancall_plus_llm_fallback` merge scoring.
4. **Unified export & audit tests**: end-to-end synthetic run producing `final_report.json`, `audit_report.md`, `tables/*.csv`, and `latex/*.tex`; verifies fail-loud on missing `sample_id` or corrupted manifest hash; verifies LaTeX output compiles cleanly as valid `booktabs` syntax.
5. **Notebook builder tests**: `python scripts/build_lean_v3_3_train_eval_notebook.py --check` + AST scope guard + `nbformat.validate`, plus running existing `pytest tests/test_notebook_v3_fixes.py` to confirm zero regressions.

---

## 7. Implementation Order

1. `spancall/eval/llm_parity.py` — extract shared scoring/shaping/bucketing helpers from `scripts/compare_llm_spancall.py` (keeping `compare_llm_spancall.py` importing from it for 100% compatibility).
2. `scripts/build_fair_eval_v3_3.py` — build & freeze `data/eval_v3_3/` + `manifest.json`.
3. `spancall/lean/trainer.py` + `spancall/train/config.py` — add `early_stopping_agg="macro_mean"` option.
4. `scripts/build_lean_v3_3_train_eval_notebook.py` — generate `notebooks/spancall_lean_v3_3_train_eval.ipynb` with balanced max-data training, macro-dev early stopping, and Tier 1 + Tier 2 eval.
5. `scripts/eval_llm_local.py` + `configs/llm_eval_v3_3.yaml` — local resumable multi-model OpenAI-SDK LLM evaluator.
6. `scripts/export_v3_3_results.py` — unified audited JSON/Markdown/CSV/LaTeX exporter.
7. `tests/test_v3_3_pipeline.py` — full offline regression suite + run `pytest`.

