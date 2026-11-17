# SpanCall — Project State

*Snapshot: 2026-11-16 · Lean v3 (two-step-only resolution, corpus × tool-count stratified eval, W&B + mid-training probe eval, three-mode agentic comparison) · 717 local tests passing*

---

## 1. What the project is for

**SpanCall** is a research system for a *speculative fast path* for tool-calling agents, as an alternative to letting an autoregressive LLM generate `tool_call` JSON token-by-token (25–35 sequential steps per call).

### 1.1 Problem

Mainstream tool-calling relies on autoregressive LLMs generating JSON token-by-token: flexible, but slow, compute-heavy, and prone to inventing enum values or altering entities. Real production traffic (search filters, CRM, booking, IoT, ERP) is heavily skewed: a large majority of calls reuse a small set of schemas and a smaller set of phrase patterns — verbatim entities, "direct/non-stop", "next week", "business class", "today/tomorrow", literal numbers. These are cases where extractive, span-based prediction is structurally safer and faster than generation.

### 1.2 Core mechanism (Lean Flow)

1. For the **easy majority** of tool calls — where every argument is already present, extractable, and context-ready (verbatim entities, literals, booleans, schema enums, literal dates) — a compact bidirectional encoder (`SpanCallTransformerEncoder`) + **Lean 2-head stack** (`ToolHead` / `BinaryToolHead` H1 + `LeanPointerHead` H3) emits the **complete payload in one forward pass**.
2. Parameter values are **0-indexed span-selection operations** over valid candidates from the utterance + a deterministic **Virtual Context Pool** (anchored dates, booleans, schema enums, recent turns), scored by `LeanPointerHead` with **no in-head abstain shortcut** (preventing the abstain-collapse seen in v1).
3. **Whole-call gating** is enforced at serve time by `LeanResolver` via H1 tool confidence (`call_min_prob`, `call_min_prob_ratio`, or `BinaryToolHead` `tool_call_threshold`), H3 posterior margin (`min_joint_margin`), and deterministic `StrictTypeCaster` + `SchemaValidator`. If any required parameter fails or confidence falls below gate thresholds, the resolver **abstains entirely** and the agent runs its **normal, unchanged LLM tool call**.
4. **Pluggable Agent Compatibility:** `ToolCallRouter` and `ReActAgentLoop` program against a version-agnostic `BaseResolver` / `BaseResolveResult` boilerplate contract (`spancall/infer/base.py`), allowing `LeanResolver` (v2) and any future v2.x / v3 resolver to drop in seamlessly.

### 1.3 Input contract — the unified marker grammar template (`span_template/`)

Both alignment/training and online serving render through the unified multi-tool grammar (`build_input_multi`, `span_template/template_multi.jinja`), guarded by a cryptographic template stamp (`span_template/stamp.py`):

```text
<|cls|> <|tool|> ... <|param|> ... <|none|> <|sep|> <|ctx_meta|> <|ctx_time|> <|ctx_enum|> <|sep|> utterance <|sep|> <|ctx_recent|> <|eos|>
```

Contract rules R1–R9:
- **R1/R2** quoting — values are quoted forms; no free text token can be confused with a value
- **R3** targetable region Ω — only the query + context pool are span-selection targets
- **R4** equal-value probability aggregation & all-or-nothing whole-call gating
- **R5** time window — `<|ctx_time|>` holds anchored, pre-resolved dates
- **R6** enum pool — `<|ctx_enum|>` enumerates the schemas' allowed values
- **R7** sanitization — enforced in `span_template/sanitize.py` before rendering
- **R8** entry-snapped spans — span boundaries snap to token entries
- **R9** validity masks — only valid candidate positions score (`CandidateMaskBuilder`)

---

## 2. Implemented Stack & Key Milestones

### 2.1 Hermes Function-Calling V1 Integration & v2 Re-Alignment (`spancall/data/`)

- `spancall/data/sources/hermes.py` (`HermesLoader`) ingests all 5 configs of `NousResearch/hermes-function-calling-v1` (10,693 rows, 92.4% ingestion, zero silent drops) into canonical `UnifiedSample` records.
- `spancall/data/partition.py` (`partition_schema_disjoint`) splits schemas disjointly across `train / dev / calib / eval / test_ood_schema / test_ood_para / test_adversarial`.
- **Lean v2 Re-alignment (`scripts/realign_v2.py` + `spancall/data/align.py`):**
  - Renders with `build_input_multi` (matching tensorize and resolver grammar) and emits render-invariant `utt_rel_start` / `utt_rel_end` offsets alongside absolute offsets into `data/processed_v2/`.
  - Stamps datasets with `template_stamp.json` (`span_template/stamp.py`) and enforces `utterance_span_stale_rate < 0.02` before training.

### 2.2 Lean Model, Loss, Tensorizer, and Trainer (`spancall/lean/`)

- `spancall/lean/model.py` — `LeanPointerHead` (0-indexed `(B, K, C)` span scorer) + `LeanSpanCallModel` (`encoder` + `ToolHead` or `BinaryToolHead` + `LeanPointerHead`).
- `spancall/lean/loss.py` — `LeanLossContainer`: multi-positive pointer cross-entropy (`PAD_INDEX = -1` ignored) + single/multi-call softmax CE or independent binary BCE (`use_binary_tool_head=True`).
- `spancall/lean/tensorize.py` — `tensorize_sample_lean`, `ToolPool` (`jitter_pack_tools=True` distractor tool packing), `LeanBatchCollate` with explicit `PAD_INDEX_KEYS`.
- `spancall/lean/trainer.py` + `scripts/train_lean.py` — `LeanTrainer`: length-bucketed micro-batching, slot-weighted `_MicroLossAccumulator` (batch-composition-invariant train/eval loss), differential LR + layerwise decay, optional LoRA (`use_lora`), early stopping (`early_stopping_patience`, `restore_best_on_exit`).

### 2.3 Lean Decoder, Resolver, Part-1 Eval & Diagnostics (`spancall/lean/`)

- `spancall/lean/decode.py` — `LeanJointDecoder`: equal-value aggregation (R4), top-K collapse, DFS exclusivity-constrained assignment, and `joint_margin`.
- `spancall/lean/resolver.py` — `LeanResolver`: single-pass multi-tool encoding reuse (`encoded_bundle`), multi-call emission (`max_calls`, `call_min_prob`, `call_min_prob_ratio`, or sigmoid `tool_call_threshold`), calibrated `min_joint_margin` gate (W2), `StrictTypeCaster` + `SchemaValidator`, and render-invariant candidate parity tracking.
- `spancall/lean/eval_p1.py` & `spancall/lean/diagnostics.py` — Part-1 retrieval-first evaluation (`eval_param_extraction_lean`), `run_diagnostics`, `analyze_predictions`, and `audit_gt_offsets`.

### 2.4 Shareable Boilerplate & Pluggable Agent Harness (`spancall/infer/`, `spancall/agent/`, `spancall/eval/`)

- **Shareable ABCs:** `BaseDatasetLoader`, `BaseAligner`, `BaseAugmenter` (`spancall/data/base.py` + loader shells in `spancall/data/sources/`), `BaseEncoder` (`spancall/model/encoder.py`), `BaseHead` (`spancall/model/heads.py`), `BaseResolver`, `BaseResolveResult`, `BaseJointDecoder`, `BaseTypeCaster`, `BaseExporter` (`spancall/infer/`), `BaseTrainer`, `BaseCalibrator`, `CosineWarmupSchedule` (`spancall/train/`).
- **Agent Router & Loop (`spancall/agent/`):** `ToolCallRouter` (`LLM_ONLY`, `SPANCALL_ONLY`, `ROUTER`, `HYBRID_FILL`, `SHADOW`; `s_skip`, `s_inject`), `ReActAgentLoop`, `UniversalAPIBackend` / `LocalVLLMBackend`, `MockBookingEnvironment`.
- **Offline & Agentic Evaluators (`spancall/eval/`):** `ToolCallEvaluator` (Part-3 7-bucket taxonomy, risk-coverage, ECE, `B1aBaseline`), `ComparativeEvaluator` (10k paired bootstrap CI, non-inferiority test), `MultiSeedAgentEvaluator`, `CostModel`.

### 2.5 Lean v3 Protocol (`notebooks/*_v3_*.ipynb`, `scripts/`)

v3 changes the **reporting contract**, not just the numbers, so its results are not diffable against v2.x columns. Seven invariants:

1. **Two-step only.** `single_pass` is retired, not deprioritised: `LeanResolver` resolves `tool → params` in two decodes. `LeanResolver.__init__` now **defaults `resolve_mode` to `two_step`**, and `single_pass` remains constructible only when named, so a run cannot inherit the retired arm by omission. Consumers still pin the mode explicitly — the train/eval notebook in the resolver construction, the agentic notebook in the rebuild path, and `resolve_config.json` in the export — so the mode a run used is readable at the call site rather than inferred from a library default.
2. **`head_dim` is architecture.** The HF export writes `head_dim` into `resolve_config.json`. A consumer that rebuilds `LeanSpanCallModel` without it silently gets the 256 default and loads a state dict with `strict=False` while leaving `pointer_head` mis-dimensioned; the agentic notebook therefore reads the exported value and **aborts** if any `pointer_head` / `tool_head` key fails to load.
3. **Hub weights are loaded with `weights_only=True`.** `model.pt` crosses a trust boundary (it is fetched over the network), so the agentic reload unpickles it with arbitrary code execution disabled and **aborts** rather than falling back to `weights_only=False` if the file is not a plain tensor dict. A pickle from the Hub runs with the notebook's credentials in scope; "my own repo" is not an enforcement policy.
4. **Mid-training probe eval.** `LeanTrainer` evaluates a small probe set every `probe_every_steps` (1000) **optimiser** steps rather than once per epoch, appending `event="probe_eval"` records to `train_log.jsonl` and logging `probe/*` to W&B (`probe/eval_loss`, `probe/exact_call`, `probe/span_acc`, `probe/tool_acc`). The probe set (`PROBE_SPLIT="probe"`, `PROBE_N=300`) is drawn from `PROBE_SOURCES` and filtered against the dev `sample_id` set; `overlap_with_dev` is **printed, not asserted**, so it must be read as 0 before the curve is trusted. It is a cadence signal, never a substitute for the large eval.
5. **Stratified eval.** Corpus separation is at the **split** level, not a nested key: `hermes_eval` and `xlam_eval` are scored as their own slices, each carrying one corpus. Tool-count buckets come from `TOOL_BUCKET_EDGES` (`1-2`, `3-5`, `6-10`, `11+`) and land in the **filename** (`toolcall_metrics_<split>_<bucket>_<system>.json`). §4.0b prints the eval-set corpus × bucket **row counts** before any metric table, and §5c prints the corpus × bucket metric pivot — so a flat aggregate cannot hide a bucket regression, and a slice that gained 20-distractor-tool rows cannot be mistaken for a model improvement.
6. **LLM parity.** `scripts/compare_llm_spancall.py` scores an LLM and the lean path on the *same* samples through the same `ToolCallEvaluator`, adds a SpanCall→LLM fallback merge arm, and prints the corpus × bucket pivot. Credentials come from the environment only.
7. **Three agentic modes (on the `v3.2` checkpoint).** `spancall_lean_v3_2_agentic.ipynb` loads `convoicon/spancall-lean-v3-2` and compares exactly three integrations: `LLM_ONLY`, `ROUTER` (SpanCall with full LLM fallback on abstain), and `HYBRID_FILL` (SpanCall predicts the call skeleton, the LLM fills only `unresolved_param_names`, `merge_filled_arguments` never overwriting a resolved value, schema validation falling back to the full LLM call on failure). `spancall/agent/fill.py` holds the helpers. The resolver is built from `resolve_config.json`, with `max_cands=1024` pinned explicitly (it is a resolver search width, absent from the export, and the library default of 2048 is not the `v3.2` value), and both collapse guards defaulting to `True`. Credentials follow the same env → `.env` → prompt order as the other `v3.2` notebooks.

Notebooks that require API keys (LLM calls, HF push, W&B) are **never committed** — `.gitignore` excludes `notebooks/*_llm_*.ipynb`, `notebooks/*_hf_push*.ipynb`, `notebooks/*_agentic*.ipynb`, and `wandb/`.

---

## 3. Component Status Overview

| Component | Status | Notes |
| --- | --- | --- |
| `span_template/` — template render, parser, sanitize, stamp | ✅ | `build_input_multi`, `resolve_utterance_span`, `stamp.py` template hash guard |
| `spancall/data/` — schema, align, augment, mix, partition, sources | ✅ | `HermesLoader` (5 configs) + v2 `ReverseAligner` (`utt_rel_*`) + shareable loader stubs (`xlam`, `glaive`, `toolbench`, `bfcl`, `tau_bench`) |
| `spancall/model/` — `encoder.py`, `masks.py`, `heads.py` | ✅ | `SpanCallTransformerEncoder`, `CandidateMaskBuilder`, `BaseHead`, `ToolHead`, `BinaryToolHead` |
| `spancall/lean/` — `model`, `loss`, `tensorize`, `trainer`, `decode`, `resolver`, `eval_p1`, `diagnostics` | ✅ | Primary active training, inference, and Part-1 evaluation stack |
| `spancall/infer/` — `base`, `cast`, `clock`, `export` | ✅ | `BaseResolver` / `BaseResolveResult` boilerplate, `StrictTypeCaster` (Q4/Q5), `SchemaValidator` (Q6), clock utils |
| `spancall/train/` — `base`, `config` | ✅ | `TrainingConfig` YAML loader + `BaseTrainer`, `BaseCalibrator`, `CosineWarmupSchedule` |
| `spancall/agent/` & `spancall/eval/` | ✅ | Pluggable `ToolCallRouter`, `ReActAgentLoop`, `ToolCallEvaluator`, `ComparativeEvaluator`, `CostModel` |
| `notebooks/spancall_lean_v2_train_eval.ipynb` | ✅ | Built and AST-scope-guarded by `scripts/build_lean_v2_colab_notebook.py` |
| `notebooks/spancall_lean_v3_train_eval.ipynb` | ✅ | Built by `scripts/build_lean_v3_colab_notebook.py`: two-step-only, W&B logging, probe eval every 1k optimiser steps, corpus × tool-count stratification, guarded HF push. Cell guard + 38 v3 regression tests |
| `notebooks/spancall_lean_v3_2_agentic.ipynb` | ✅ (gitignored) | Built by `scripts/build_lean_v3_agentic_notebook.py`. Targets the **`v3.2`** export (`convoicon/spancall-lean-v3-2`, `results_lean_v3_2/`), pins `max_cands=1024`, and has exactly three modes (`LLM_ONLY`, `ROUTER`, `HYBRID_FILL`). Credentials resolve env → `.env` → prompt (`getpass` for secrets, reported only as `<set>`), so it runs with or without a key. `.gitignore` excludes it |
| `scripts/compare_llm_spancall.py` | ✅ | LLM vs SpanCall parity on identical samples through `ToolCallEvaluator`; 4 systems incl. fallback merge, corpus × bucket pivot, env-only keys |
| `tests/test_notebook_v3_fixes.py` | ✅ | 39 tests: two-step enforcement, `head_dim` round-trip, `weights_only=True` on hub weights, stratification totality, probe/W&B key parity with `TrainingConfig`, three-mode contract, secret scan |

---

## 4. What's Next (Active Roadmap)

1. **Run v3 end-to-end and fill the ledger.** Execute [`notebooks/spancall_lean_v3_train_eval.ipynb`](notebooks/spancall_lean_v3_train_eval.ipynb) on Colab, then record the stratified results in [`docs/experiments.md`](docs/experiments.md) §II.3. The probe curve is the first artifact to inspect: a probe that stops improving while train loss keeps falling is the early-stopping signal that per-epoch eval used to hide.
2. **LLM-parity sweep ([`scripts/compare_llm_spancall.py`](scripts/compare_llm_spancall.py)).** Run the same-sample comparison across every corpus × tool-count bucket, not just the aggregate. The pivot decides whether the fast path's advantage is concentrated in `1-2` tools or survives into `6-10` / `11+`, which is what the cost model assumes.
3. **Calibrate the gate per bucket.** If `3-5` / `6-10` show a different precision-coverage curve than `1-2`, `call_min_prob` / `min_joint_margin` need per-bucket values rather than one global conjunction — the v3 eval now produces exactly the cells needed to fit them.
4. **Agentic three-mode run on a real backend.** The notebook runs live when `API_KEY` is set **and** a one-call probe succeeds; otherwise it falls back to `DeterministicBackend`, which validates plumbing only (mode wiring, fallback, hybrid-fill merge). It prints that caveat beside its numbers and records `backend` / `resolver.kind` in the JSON, so a keyless table can never be mistaken for an LLM-quality result. Re-run all three modes against a hosted LLM on a task suite with non-trivial trajectories before quoting any success-rate or token-cost delta.
