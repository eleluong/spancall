# SpanCall — Project State

*Snapshot: 2026-11-13 · Lean v2 architecture (`spancall/lean/`) + shareable boilerplate & pluggable agent harness · 630 local tests passing*

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
- **Agent Router & Loop (`spancall/agent/`):** `ToolCallRouter` (`LLM_ONLY`, `SPANCALL_ONLY`, `ROUTER`, `SHADOW`; `s_skip`, `s_inject`), `ReActAgentLoop`, `UniversalAPIBackend` / `LocalVLLMBackend`, `MockBookingEnvironment`.
- **Offline & Agentic Evaluators (`spancall/eval/`):** `ToolCallEvaluator` (Part-3 7-bucket taxonomy, risk-coverage, ECE, `B1aBaseline`), `ComparativeEvaluator` (10k paired bootstrap CI, non-inferiority test), `MultiSeedAgentEvaluator`, `CostModel`.

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

---

## 4. What's Next (Active Roadmap)

1. **Execute Lean Cleanup ([`plans/remove_v1_keep_lean_plan.md`](plans/remove_v1_keep_lean_plan.md)):**
   - Extract `spancall/infer/base.py` and `spancall/train/base.py` boilerplate, wire `spancall/agent/` to `BaseResolver`, and delete the retired v1 5-head files and obsolete docs/plans.
2. **Candidate Parity Fix ([`plans/lean_v2_candidate_parity_fix.md`](plans/lean_v2_candidate_parity_fix.md) & [`plans/v2_candidate_parity_and_w5_plan.md`](plans/v2_candidate_parity_and_w5_plan.md)):**
   - Run protected parity comparison (`protect_parity_spans=True`), decompose missed GT spans (`width`, `typefilter`, `entrysnap`, `truncated`, `fallback_address`), and raise production serve-time `candidate_parity >= 0.95`.
3. **Binary Tool Selection Head v2.1 Ablation ([`plans/binary_tool_selection_plan.md`](plans/binary_tool_selection_plan.md)):**
   - Run the softmax (`ToolHead`) vs. independent sigmoid (`BinaryToolHead`, `use_binary_tool_head=True`) ablation on `data/processed_v2/` and re-evaluate the W5 decision gate.
