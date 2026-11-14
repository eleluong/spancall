# xLAM Function-Calling 60k: Data Ingestion & Integration Plan

Status: **Phases 0–3 implemented, verified, and uploaded to `convoicon/spancall` (`data/processed_v2/`); ready for Phase 4 Colab training ablation.**

Scope: bring [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) into the SpanCall `UnifiedSample` pipeline (filtered to **single-call rows** at this stage), partition with zero API-name leakage across splits, and run a controlled ablation alongside `NousResearch/hermes-function-calling-v1` in Lean Flow v2.6.

---

## 1. Goal & Motivation

In [`docs/v2_run_13-11-26_analysis.md`](../docs/v2_run_13-11-26_analysis.md), two core bottlenecks were identified on Hermes alone:
1. **`tool_acc` = 0.667** (H1 tool selection): Hermes rows mostly ship 1 native tool, so H1 only sees corpus-sampled distractors. xLAM ships **real multi-tool candidate sets** (71.8% of 1-call rows have 2–8 tools) where the model must pick the right API from plausible alternatives.
2. **Pointer collapse on multi-parameter slots**: xLAM provides natural, verbatim, multi-parameter extraction queries across **3,426 unique tools**, giving H3 diverse schema-generalization supervision.

---

## 2. Phase 0 Inspection Results (Verified on Hub Dataset)

Inspected all 60,000 rows of `Salesforce/xlam-function-calling-60k` (`train` split, columns `['id', 'query', 'answers', 'tools']`):

| Metric | Observed Value | Notes |
| --- | --- | --- |
| Total rows | 60,000 | 0 JSON parse errors, 0 empty answers |
| Unique tool names (full) | 3,605 | — |
| **1-call rows (`len(answers) == 1`)** | **28,461 (47.4%)** | **Selected scope for this stage** (avoids `multi_call_ambiguous_anchors` from 12,567 parallel same-tool rows) |
| Unique tools (1-call subset) | 3,426 | — |
| Tools per row (1-call) | 1: 8,023 (28.2%) · 2–4: 16,529 (58.1%) · 5–8: 3,909 (13.7%) | 71.8% have real multi-tool candidate lists |
| Alignment yield (first 2,000 1-call rows) | **Stratum F: 1,257 (62.85%)** · Stratum A: 743 (37.15%) | Parameter H4: `OK`: 3,684 · `OFF_WINDOW`: 1,147 · `ABSENT`: 7 |

Spot-checked `OFF_WINDOW` parameters are genuinely non-extractive conversions (e.g., `"sorted by release date in descending order"` → `'release_date_desc'`, `"September 9, 2015"` → `'20150909'`, `"French (France) with UTC+1 offset"` → `'fr-FR'` / `3600`, `"2%"` → `0.02`, `"Cape Town"` → `latitude=-34, longitude=18`), which correctly route to `Stratum.A` (abstain to LLM). No changes to `ReverseAligner` are required for Phase 1.

---

## 3. Data Format & Mapping Contract (`XLAMLoader`)

Raw row shape:
- `id`: integer
- `query`: user utterance string
- `tools`: JSON string of `[{name, description, parameters: {param_name: {type, description, default?}}}]`
- `answers`: JSON string of `[{name, arguments: {...}}]`

Differences from Hermes OpenAI JSON Schema:
1. **`parameters` is a flat dict**, not `{properties, required}` — cannot reuse `_tool_from_json_schema` directly.
2. **Python-style type strings** with `, optional` / `Optional[...]` wrappers:
   - `str`, `int`, `float`, `bool`, `list`, `dict`, `set`, `List[int]`, `List[float]`, `List[str]`, `Tuple[...]`, `Callable[...]`, etc.
   - Mapping rules:
     - Strip `, optional` suffix or `Optional[...]` wrapper → `is_optional = True`.
     - `str` / `string` → `"string"`
     - `int` / `integer` → `"integer"`
     - `float` / `number` / `double` → `"number"`
     - `bool` / `boolean` → `"boolean"`
     - `list` / `List[...]` / `Tuple[...]` / `set` → `"array"` with `items_type` inferred (`integer` / `number` / `boolean` / `string`)
     - `dict` / `Dict[...]` → `"object"`
     - Fallback for unknown types (e.g. `Callable[...]`) → `"string"`
3. **Optionality**: `required = not (is_optional or ("default" in pspec))`.
4. **1-call filter**: keep only rows where `len(answers) == 1` (configurable via `single_call_only: bool = True` so multi-call can be enabled later without rewriting the loader).
5. **Identifiers & telemetry**:
   - `sample_id = f"xlam::{row['id']}"`
   - `source_dataset = "xlam-60k"`
   - `metadata = {"config": "xlam", "n_calls": len(answers), "n_tools": len(tools)}`
   - Track `stats`: `rows_seen`, `rows_yielded`, `dropped_multi_call`, `dropped_parse_error`, `dropped_empty_gt`, `parsed_gt_calls`.

---

## 4. Schema-Disjoint Partitioning & Mixing

### 4.1 Preventing API Leakage (`group_by="gt_tool"`)
- Currently `schema_key(sample)` in [`spancall/data/partition.py`](../spancall/data/partition.py) uses `tuple(sorted(tool.name for tool in sample.tools))`.
- Because xLAM combines overlapping subsets of its 3,426 APIs across rows, connected components over *all* candidate tools merge 25,846 / 28,461 rows (90.8%) into a single giant component (`settag`).
- **Fix (`gt_tool_key`, `group_by="gt_tool"`)**: group xLAM rows by their ground-truth tool name (`ground_truth_calls[0].tool_name`, 2,639 distinct groups, largest group `search` = 509 rows / 1.8%) before greedy largest-first split assignment. This guarantees **zero ground-truth tool overlap** between `train`, `dev`, `calib`, and `test_ood_schema` while preserving exact 80/5/5/10 split ratios.

### 4.2 Split Emission & Tool-Name Collisions
- Keep `hermes_*.jsonl` untouched so baseline results remain directly comparable.
- Emit `xlam_{train,dev,calib,eval,test_ood_schema,test_ood_para,test_adversarial}.jsonl` alongside Hermes in `data/processed_v2/`.
- Check `ToolPool.name_collisions` when combining Hermes and xLAM in tensorization; namespace or log collisions if generic tool names (e.g. `get_weather`) differ in parameter schemas across datasets.

### 4.3 Balancing Hermes vs. xLAM
- 28,461 1-call xLAM rows vs. ~8,555 Hermes train rows would skew training ~3:1 toward xLAM if uncapped.
- Add `max_train_rows` / `xlam_train_cap` and `single_call_only` options in [`configs/default.yaml`](../configs/default.yaml) to support stratified subsampling (by GT tool / category) for controlled ablations: `0`, `5k`, `15k`, `full`.

---

## 5. Implementation Steps

1. **Phase 1 — `XLAMLoader` & Tests**
   - Implement `XLAMLoader` in [`spancall/data/sources/xlam.py`](../spancall/data/sources/xlam.py) (supporting HF Hub `datasets` + `local_path` JSON/JSONL, `single_call_only=True`, `limit`).
   - Register in [`spancall/data/sources/__init__.py`](../spancall/data/sources/__init__.py).
   - Add unit tests in `tests/data/test_xlam.py` covering flat parameter parsing, `, optional` / `Optional[...]` / `default`, type mapping, 1-call filtering, and drop counters.

2. **Phase 2 — Partitioning & `prepare_data.py`**
   - Add `partition_schema_components()` in [`spancall/data/partition.py`](../spancall/data/partition.py) + unit tests verifying zero tool-name overlap across splits.
   - Generalize [`scripts/prepare_data.py`](../scripts/prepare_data.py) to iterate over configured sources (`hermes`, `xlam`) and emit `<source>_<split>.jsonl` + combined `prepare_report.json`.
   - Update [`configs/default.yaml`](../configs/default.yaml) with `xlam` source config (`hub_id: "Salesforce/xlam-function-calling-60k"`, `single_call_only: true`).

3. **Phase 3 — Notebook v2.6 (`build_lean_v2_6_colab_notebook.py`)**
   - Copy v2.5 builder to `scripts/build_lean_v2_6_colab_notebook.py` → `notebooks/spancall_lean_v2_6_train_eval.ipynb`.
   - Add `XLAM_TRAIN_ROWS` knob (e.g. `0`, `5000`, `15000`, `None`) and include `xlam_train.jsonl` / `xlam_dev.jsonl` in tensorization + pre-flight offset audit.
   - Upload pre-processed `xlam_*.jsonl` splits to `convoicon/spancall` (or build on-the-fly with `HF_TOKEN`).

4. **Phase 4 — Ablation & Acceptance Gate**
   - Run 4 controlled experiments with `answerdotai/ModernBERT-large`:
     - **A (Baseline)**: Hermes only
     - **B**: Hermes + xLAM (5k)
     - **C**: Hermes + xLAM (15k)
     - **D (Optional)**: xLAM only
   - Evaluate every run on: `hermes_eval` (stratified F/A), `hermes_test_ood_schema`, `xlam_eval`, and `xlam_test_ood_schema`.
   - Track: `tool_acc`, `exact_call`, per-param exact match, false-hit / pointer-collapse rate, `abstain_acc`, and paired bootstrap CI via `ComparativeEvaluator`.
   - **Success criterion**: Run B or C improves OOD-schema `tool_acc` and `exact_call` without regressing `hermes_eval` Stratum-F accuracy.

---

## 6. Verification & Upload Summary (2026-11-14)

- **Pipeline run (`scripts/prepare_data.py --sources xlam --output_dir data/processed_v2`)**:
  - Ingested: **28,461** 1-call rows (31,539 multi-call rows filtered).
  - Aligned: **Stratum F = 18,193 (63.9%)**, **Stratum A = 10,268 (36.1%)** (passes E0 gate `>= 0.50`).
  - Final splits (`group_by="gt_tool"`):
    - `xlam_train.jsonl`: 69,412 (`F=59,000`, `A=10,412`)
    - `xlam_dev.jsonl`: 4,906 (`F=2,439`, `A=2,467`)
    - `xlam_calib.jsonl`: 4,907 (`F=2,452`, `A=2,455`)
    - `xlam_eval.jsonl` (`test_id`): 9,101 (`F=4,579`, `A=4,522`)
    - `xlam_test_ood_schema.jsonl`: 9,814 (`F=5,010`, `A=4,804`)
    - `xlam_test_ood_para.jsonl`: 6,876 (`F=6,876`, `A=0`)
    - `xlam_test_adversarial.jsonl`: 88,327 (`F=0`, `A=88,327`)
- **Offset & Stamp Gates**:
  - `check_stamp("data/processed_v2")` → `ok`.
  - `audit_gt_offsets` on `hermes_eval`, `hermes_dev`, `xlam_eval`, `xlam_dev`, `xlam_test_ood_schema` → `utterance_span_stale_rate = 0.0`, `unresolvable_rate = 0.0`.
- **Hugging Face Upload**:
  - Uploaded all 18 files in `data/processed_v2/` (Hermes + xLAM splits, reports, `template_stamp.json`) to `convoicon/spancall` ([commit 9d6410c](https://huggingface.co/datasets/convoicon/spancall/commit/9d6410c35823aee45303d461a48c1dd22c3d9c74)).
- **Notebook v2.6**:
  - Built [`notebooks/spancall_lean_v2_6_train_eval.ipynb`](../notebooks/spancall_lean_v2_6_train_eval.ipynb) via [`scripts/build_lean_v2_6_colab_notebook.py`](../scripts/build_lean_v2_6_colab_notebook.py) with stratified test subset slicing (`build_stratified_eval_subset`) across all `TEST_SPLITS` so `TEST_LIMIT = 200` samples uniformly across tools and cohorts instead of reading only the leading largest schema groups.


