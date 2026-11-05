  # SpanCall — Project State

*Snapshot: 2026-11-03 · 364 tests passing · working tree: research PoC with all four roadmap phases implemented — data pipeline (E0 gate run), model/training stack (smoke-verified), inference/calibration/offline eval (E2–E7 harness), and agent loop + comparative eval (E9 protocol run, review issues fixed) — see §2.5–§2.9*

---

## 1. What the project is for

**SpanCall** is a research system for a *speculative fast path* for tool-calling agents, as an alternative to letting an autoregressive LLM generate `tool_call` JSON token-by-token (25–35 sequential steps per call).

### 1.1 Problem

Mainstream tool-calling relies on autoregressive LLMs generating JSON token-by-token: flexible, but slow, compute-heavy, and prone to inventing enum values or altering entities. Real production traffic (search filters, CRM, booking, IoT, ERP) is heavily skewed: a large majority of calls reuse a small set of schemas and a smaller set of phrase patterns — verbatim entities, "direct/non-stop", "next week", "business class", "today/tomorrow", literal numbers. These are cases where extractive, span-based prediction is structurally safer and faster than generation.

### 1.2 Core mechanism

1. For the **easy majority** of tool calls — where every argument is already present, extractable, and context-ready (verbatim entities, literals, booleans, schema enums, literal dates) — a small (**~150M-param**) bidirectional encoder emits the **complete payload in one forward pass** (1 decoding step).
2. Parameter values are **span-selection operations** against the query + a deterministic **Virtual Context Pool** (anchored dates, booleans, schema enums), scored by a boundary-factorized pointer head with **linear span cost**.
3. A **whole-call confidence gate** emits the call **only if every required parameter resolves at ≥ τ** — otherwise it **abstains entirely** and the agent runs its **normal, unchanged LLM tool call**. No partial payload, no per-parameter fill, no hybrid, no forced hit. Abstention is the safe default.
4. **Agentic compatibility is by construction:** the router sits *before* LLM tool-call generation; easy, context-ready calls short-circuit the LLM, everything else flows through the agent's untouched pipeline.
5. **Targets (claim ledger):** ≥ 95% whole-call precision on held-out paraphrases (a *calibrated target*, not an assumption); expected order-of-magnitude lower FLOPs than a 7B autoregressive decode (a hypothesis; measured FLOPs/latency is a Phase-4 deliverable); p50 latency in tens of ms.

### 1.3 Explicit non-claims

- No zero hallucination, no ~95% of *all* calls hitting the fast path. Wrong-value errors (pointing at the wrong mentioned entity) can and will occur — measured and calibrated against via abstention. Coverage depends on traffic.
- No general solution to the canonicalization problem. Paraphrase→canonical mapping is learned semantic behavior bounded to the class enumerable in `[CTX_ENUM]` / `[CTX_TIME]`; unlisted paraphrases abstain and the LLM handles them.
- No strict O(1) total cost. Encoder attention is O(L²); the pointer head is a boundary-factorized O(L) scorer. "O(1)" refers only to decoding steps (= 1) vs. autoregressive M steps.

### 1.4 Input contract — the marker grammar template

The template (`span_template/`) is the single source of truth for the input format, with a legend **parsed from the rendered text**:

```
[TOOL] [PARAM] [CTX_META] [CTX_TIME] [CTX_ENUM] [CTX_RECENT] [SEP] [EOS]
```

Contract rules R1–R9:
- **R1/R2** quoting — values are quoted forms; no free text token can be confused with a value
- **R3** targetable region Ω — only the query + context pool are span-selection targets
- **R4** gating (≥τ per required param, all-or-nothing)
- **R5** time window — `[CTX_TIME]` holds anchored, pre-resolved dates
- **R6** enum pool — `[CTX_ENUM]` enumerates the schemas' allowed values
- **R7** sanitization — enforced in the reference implementation (never produce a partial/fabricated payload)
- **R8** entry-snapped spans — span boundaries snap to token entries
- **R9** validity masks — only valid span positions score

Mechanism stages (worked example and score walkthrough in the mechanism doc): input construction + tokenization → bidirectional encoding (ModernBERT-base backbone) → param query vectors, boundary scores (O(L) per param), abstain logit, per-parameter distribution, call-level accept head → joint assignment (swap errors can't slip through: assignment is jointly solved, not greedy per-parameter) → deterministic typed caster + schema validation + whole-call gate → calibration (how the 95% target is actually met). Candidate-schema selection has a v1 contract (D6): modes + selection contract.

---

## 2. Contributions (implemented so far)

### 2.1 Hermes function-calling V1 integration (all 5 configs) — the headline contribution

`spancall/data/sources/hermes.py` is a **real, verified loader** for `NousResearch/hermes-function-calling-v1`, covering all 5 dataset configs ("setups"), pulled from the HuggingFace hub (with an optional `datasets` dependency, `data` extra):

| Config | Mode | Rows seen | Ingested | GT calls | Drops | Drop reason |
| --- | --- | --- | --- | --- | --- | --- |
| `func_calling_singleturn` | `<tool_call>` tags, single turn | 1893 | 1883 | 2981 | 10 | assistant clarification turns with no tool call |
| `func_calling` | `<tool_call>` tags, multi-turn | 1893 | 1883 | 2981 | 10 | same as above |
| `glaive_func_calling` | Glaive `{"function":...}` format | 5209 | 4344 | 4344 | 865 | natural-language QA rows (inherent ~16.6% QA mix in this config) |
| `json_mode_agentic` | structured JSON (JSON-Schema), multi-turn | 1342 | 1342 | 1342 | 0 | — |
| `json_mode_singleturn` | structured JSON (JSON-Schema), single turn | 1241 | 1241 | 1241 | 0 | — |
| **Total** | | **11578** | **10693** | **12889** | **885** | **92.4% ingestion** |

Every remaining drop is a row with **genuinely no tool call** (assistant clarification turns, plain-Q&A rows). There are **no silent drops** — all reasons are counted in the loader's stats dict (`rows_seen`, `rows_yielded`, `dropped_no_conversation`, `dropped_no_human_turn`, `dropped_no_gt`, `parsed_gt_calls`, `multi_call_rows`, `json_mode_rows`) and covered by tests.

Loader pipeline per row: parse the shared row shape `{id, conversations[{from,value}], tools(JSON-string), category, subcategory, task}`; for json-mode configs use `schema` (JSON-Schema string); extract tool definitions from `[TOOLS{...}]` blocks / JSON-Schema blocks; parse the ground-truth call — `<tool_call>{...}</tool_call>` tags, Glaive `{"function":..., "arguments":{...}}` dicts, or bare JSON — and emit the canonical `UnifiedSample` record (shared by all future loaders).

### 2.2 Robust JSON/tool-call extraction (against real hub quirks)

Real hub data forced four parsing fixes, all locked in by regression tests with real-hub-shape fixtures:

- **Brace-matching extraction** (`_extract_brace_object` + `_iter_tool_call_payloads`) replaces the naive `\{.*?\}` regex, which truncates payloads containing nested braces and literal `\n` escapes inside string values (the regex `[TOOLS{...}]`-style non-greedy match dies on nested objects).
- **`ast.literal_eval` fallback** in `_safe_json_loads` handles single-quoted (Python-literal) JSON payloads emitted by some configs (e.g. `{'queries': ['How do you define "alternative methods" ...']}`).
- **All-assistant-turns scan**: multi-turn rows end with a natural-language summary, not `<tool_call>` — so the GT is searched across *every* assistant turn (first tool-carrying turn wins), not just the last; json-mode scans assistant texts in reverse for the last dict.
- **JSON-array GT wrapping**: json-mode outputs can be arrays (`[{...}, ...]` flight/booking lists); these are wrapped under `arguments["items"]` (`tool_name="json_output"`) to fit the `GroundTruthCall.arguments` dict contract.

### 2.3 Schema-disjoint partitioning (train / dev / eval)

`spancall/data/partition.py` implements `partition_schema_disjoint`: whole tool-name groups are kept together so dev/eval are **zero-shot on unseen schemas**. Key details:

- Canonical split enum (`Split` in `spancall/data/schema.py`): `TRAIN / DEV / CALIB / TEST_ID / TEST_OOD_SCHEMA / TEST_OOD_PARA / TEST_ADVERSARIAL`. There is **no `EVAL` member**; the name "eval" maps to `TEST_ID` (ratios keyed `train`/`dev`/`eval`, filenames `hermes_{train,dev,eval}.jsonl`).
- Default ratios `{"train": 0.80, "dev": 0.10, "eval": 0.10}`, seed 42 (overridable via `schema_split` in YAML).
- **Largest-first greedy assignment**: shuffle schema-group keys, then stable-sort by group size descending; each group goes to the split with the largest current deficit. This fixes the naive greedy that produced a lopsided 8088/603/2002 on this heavy-tailed dataset.
- Final verified result: **train 8555 (80.0%) / dev 1069 (10.0%) / eval 1069 (10.0%)**, schema-disjoint with **zero overlap** over all **4120 distinct schemas** (checked `train ∩ dev ∩ eval == ∅`).
- **Documented property:** the two mega-schemas (`json_output` 1488 rows, `ExpertQAExtractor` 1464, each ~14% of the corpus) necessarily land in **train** — a schema exceeding one 10% share mathematically cannot be held out as OOD without a lopsided split.
- Fallback: when fewer than 3 distinct schemas exist, falls back to a proportional weighted split with `schema_disjoint: False` + a warning in the report.
- Output: `partition_report.json` with `seed`, `ratios`, `total_samples` (10693), `distinct_schemas` (4120), `schema_disjoint` (true), `per_config` and `per_split` counts.

### 2.4 Pipeline wiring

- `scripts/prepare_data.py`: reads the `hermes` source entry (hub_id / local_path / limit / configs) + `schema_split` ratios from `configs/default.yaml`, ingests all 5 configs, splits schema-disjoint, writes `hermes_{train,dev,eval}.jsonl` + `partition_report.json`.
- `spancall/train/config.py`: full training-config dataclasses with YAML loading — `TrainingConfig`, `ModelConfig` (backbone `answerdotai/ModernBERT-base`), `DataConfig` (incl. `pack_tools`), `HeadsConfig`, `OptimizationConfig`, `CalibrationConfig` (target precision 0.95), `GateConfig` (tau thresholds).
- `pyproject.toml`: package `spancall` 0.1.0; optional extras `dev`, `model`, `agent`, `data` (`datasets>=2.0.0`).

### 2.5 Reverse alignment + E0 gate on the ingested Hermes corpus (Phase 1e)

`ReverseAligner` (`spancall/data/align.py`) runs over all 10,693 ingested records: renders the marker template, matches GT argument values against the utterance + context-pool legend, emits multi-positive `SpanTarget`s, and classifies each row **F** (fully groundable) or **A** (absent/unresolvable) with per-parameter H4 reasons.

**Sanitizer cap fix:** the pre-tokenization utterance cap (`U_MAX_DEFAULT`) was originally lifted to 2,048 chars and is now completely un-capped (`None`) in long-context mode (the authoritative overflow guard is the backbone-driven token budget `max_seq_len` = 32,768 post-tokenization).

**E0 gate result** (per-config F-share at align time, floor `>= 0.50` on function-calling configs; logged by `prepare_data.py` and persisted in `prepare_report.json["align"]["per_config"]` per plan line 366):

| Config | F | A | F_share | Gate |
| --- | --- | --- | --- | --- |
| `func_calling_singleturn` | 536 | 1347 | 0.285 | below floor |
| `func_calling` | 536 | 1347 | 0.285 | below floor |
| `glaive_func_calling` | 3293 | 1051 | 0.758 | **PASS** |
| `json_mode_agentic` | 53 | 1289 | 0.039 | structural gap |
| `json_mode_singleturn` | 41 | 1200 | 0.033 | structural gap |
| **Total** | **4459** | **6234** | **0.417** | — |

- **glaive passes the E0 floor** — GT arguments are short, verbatim-groundable values with recent-context grounding.
- **`func_calling*` sits below the floor.** Verified root cause (sampled A rows): a dominant share are `ExpertQAExtractor` / `json_output` "extract queries/passages" tasks whose GT `queries` arrays are *synthesized questions* — the values are not verbatim anywhere in utterance, recent context, or enum/time pools (`OFF_WINDOW` / `NEEDS_REASONING`). Genuine structural ungroundability, not a masking bug.
- **`json_mode_*` is structurally ungroundable (~0.04)** — GT is a synthetic JSON object whose keys/values are not verbatim in the utterance; `align_json_mode_keys=True` recovers only a thin slice.

**7 canonical splits (cap=2048) + augmentation:** load → align → augment (23,679 variants: `shuffle_f` 7,346 · `adversarial_r7` 10,693 · `paraphrase_f` 3,186 · `swap_f` 2,454) → schema-disjoint partition → finalize:

`train 24,071 (F 20,460) · dev 1,720 · calib 1,718 · test_id 3,426 · test_ood_schema 3,437 · test_ood_para 1,558 (all F) · test_adversarial 30,934 (all A)`

### 2.6 Model, heads, loss, trainer (Phase 2)

- `spancall/model/masks.py` — `CandidateMaskBuilder`: P4 type pre-filtering, R8 entry-snapped spans, R9 per-param enum masks (`[NULL]` for optional), regex numeric/date pre-filtering (date window uses ctx time + iso patterns), W_max width cap; `register_marker_tokens` idempotent.
- `spancall/model/encoder.py` — `SpanCallTransformerEncoder`: AutoModel (ModernBERT default) wrapper, marker-token embedding resize, `forward -> H` with attention mask.
- `spancall/model/heads.py` — `ToolHead` (H1), `PointerHead` (H3: start/end boundary scores + width bias + abstain logit), `AbstainHead` (H4: 5-class with consistency logit), `CallAcceptHead` (accept BCE), `CorrectnessHead` (H8, stop-grad guarding).
- `spancall/model/loss.py` — `SpanCallLossContainer`: multi-positive softmax CE via logsumexp, H4 CE, H1 CE, accept BCE, H3↔H4 consistency, H8 BCE; verified no gradient flows into the correctness head via stop-grad.
- `spancall/train/trainer.py` + `scripts/train.py` — bucket collator, differential LR + layer decay 0.90, bf16 AMP, staged curriculum (T0/T1/A-B/C), checkpointing with provenance.
- Tests cover mask shapes R4/R9, head shapes + gradient flow, loss numerics, H8 stop-grad, equal-value invariance. Phase 2f full suite: **176 passed**.

### 2.7 Inference, calibration, offline eval (Phase 3)

- `spancall/infer/decode.py` — `ExclusivityJointDecoder`: equal-value aggregation (R4), left-to-right top-K collapse, DFS joint assignment with span exclusivity, `Delta_joint`.
- `spancall/infer/cast.py` — `StrictTypeCaster` (Q4 provenance norm2orig mapping; Q5 strict casts: bool/enum/int/float/date/array/object) + `SchemaValidator` (Q6: type/enum checks, default materialization, optional jsonschema backend).
- `spancall/infer/resolver.py` — `SpanCallResolver`: end-to-end `try_resolve` / `try_resolve_candidates` with latency telemetry; abstains by default; `calib_thresholds` from a calibration report; pure-Python `FakeTokenizer` for smoke tests.
- `spancall/train/calibrate.py` + `scripts/calibrate.py` — `SpanCallCalibrator`: L-BFGS temperature scaling (scipy + pure-Python fallback, cross-verified), Clopper-Pearson lower bounds, tau sweep, per-risk-tier thresholds, calibration-report JSON → resolver gate cfg round-trip.
- `spancall/eval/toolcall/evaluator.py` + `scripts/eval_toolcall.py` — `ToolCallEvaluator`: payload equality (numeric normalization, optional default materialization), stratified metrics, 7-bucket error taxonomy (incl. entity-swap, wrong-value, duplicate-span, date-misparse classification), risk-coverage curve, ECE, B1a regex baseline; wrote `results/toolcall_metrics_b1a_eval.json`.
- Tests cover every Q4/Q5/Q6 row, swap-margin edges, LBD math, calibrator sweeps, evaluator taxonomy. Phase 3f full suite: **304 passed** (fixed one real bug: stratum-A false hit now classified `error_extra_call`).

### 2.8 Agent loop, router, backends (Phase 4a)

- `spancall/agent/router.py` — `ToolCallRouter` with modes `LLM_ONLY` / `SPANCALL_ONLY` / `ROUTER` / `SHADOW` and variants `s_skip` / `s_inject`; result dict contract (`hit` / `tool_name` / `arguments` / `payload` / `spancall` fast-path flag); `synthesize_injected_call` (history injection) + `synthesize_tool_response`; latency + token telemetry via `_trace`.
- `spancall/agent/loop.py` — `ReActAgentLoop`: multi-turn loop, environment reset per run, `ExecutionTrace` output; `_bare_trace` fallback.
- `spancall/agent/backends/litellm_adapter.py` — `UniversalAPIBackend` (LiteLLM / OpenAI-compatible, `.env` wiring via `_load_dotenv_values`, tool-definitions→OpenAI shape); `local_adapter.py` — `LocalVLLMBackend` (vLLM/Transformers, typed tool-call parse).
- `spancall/agent/tools/mock.py` — `MockBookingEnvironment` (flights F001–F003, search/book/cancel, idempotent bookings, task-success evaluation) + `make_booking_tools()`.
- Phase 4a full suite: **351 passed** (incl. byte-exact chat-template round-trip and `.env` parsing tests).

### 2.9 Comparative agent evaluation (Phase 4b — E9/E10/E11 harness)

- `spancall/eval/compare.py` — `ComparativeEvaluator` + `ComparativeStudyResult`: **paired bootstrap CI** (default 10,000 resamples, 2.5/97.5 percentiles), one-sided **non-inferiority p-value** (fraction of resample means < −margin), `non_inferiority_passed = ci_lower >= −margin`, per-task speedup/token-savings, and cost validation vs `CostModel.theoretical_expected_cost(hit_rate, precision_at_hit)`. `compare()` enforces **identical `task_id` ordering** across arms (unpaired lists are rejected with a `ValueError`); `to_dict()` now carries the full `per_task_comparisons` rows and a `theoretical_cost_note` explaining the hit-rate-0 gap between the theoretical formula and the empirical ledger.
- `spancall/eval/agentic/` — `AgentMetrics` (+`aggregate_metrics`): 9 fields — task success / turns / tool-call accuracy per step / **same-trace recovery rate** (a failure in one task is never "recovered" by a later success in a different task) / spancall hit rate / wrong-hit rate / side-effecting errors / LLM calls saved. The former self-referential `percent_*` reduction fields were **removed** (baseline vs same-trace metrics, always 0); cross-mode latency/token/cost savings live only in `ComparativeStudyResult`.
- `scripts/eval_agent.py` — CLI: `--benchmark custom_mock`, `--llm_backend`, `--backend auto|universal|deterministic`, `--spancall_checkpoint`, `--calib_thresholds`, `--num_seeds`, `--variant s_skip|s_inject`, `--mode router|spancall_only|shadow`, `--margin`, `--num_bootstrap`; rule-based `DeterministicBackend` for offline reproducible runs; `_EmptyResolver` all-abstain fallback when no checkpoint is provided.
- **E9 smoke run** (deterministic backend, 9 tasks × 3 seeds, s_inject, 10k bootstrap) → `results/e9_s_inject.json`:

  `B0 success 1.000 · S success 1.000 · delta +0.000 · 95%CI [0.000, 0.000] · p < 0.05 · non_inferiority PASSED · latency speedup 1.14× · tokens saved 0.0% · empirical cost saved 0.0% · theoretical cost saved −7.0% · theoretical_cost_note` (hit-rate-0 caveat, §5.10)

  With no checkpoint the resolver abstains on every turn, so all cost savings trace to the s_inject history short-circuit in `DeterministicBackend`; the negative theoretical cost reflects the all-abstain hit-rate-0 case (`c_s + c_L` per turn) and is **explicitly annotated** in the artifact via `theoretical_cost_note` — wiring a trained checkpoint + real thresholds will populate the fast-path metrics. Phase 4b + review fixes full suite: **364 passed**.

---

## 3. Main components & status

Legend: ✅ **real implementation** · ⛔ **stub** (`NotImplementedError` / boilerplate)

| Component | Status | Notes |
| --- | --- | --- |
| `span_template/` — template render, parser, sanitize | ✅ | Golden render/parse tests pass; R1–R9 rules; legend parsed from text; `build_input` / `build_input_multi` / `gate_tool_call` / `GateResult`; Jinja templates `template.jinja` / `template_multi.jinja` |
| `spancall/data/schema.py` — `UnifiedSample`, `Split`, `Stratum`, `ToolDefinition`, `ParameterSchema`, `GroundTruthCall`, `SpanTarget` | ✅ | Canonical record all loaders emit; enum-aware JSON serialization |
| `spancall/data/partition.py` — `partition_schema_disjoint`, `write_splits_jsonl` | ✅ | Verified 80/10/10 split on 10693 samples over 4120 schemas; reusable by future loaders |
| `spancall/data/sources/hermes.py` — `HermesLoader` | ✅ | 5 configs, 92.4% ingestion, per-reason stats, real-hub regression tests |
| `spancall/data/sources/{xlam,glaive,toolbench,bfcl,tau_bench}.py` | ⛔ | Boilerplate `BaseDatasetLoader` shells; will reuse `partition_schema_disjoint` + hardened JSON parsing |
| `spancall/data/align.py` — `ReverseAligner` | ✅ | Real F/A span tagging + alignment engine: synthetic clock anchoring, template render ↔ legend matching, multi-positive `SpanTarget`s, H4 reason classification, json-mode key inference |
| `spancall/data/augment.py` — `DataAugmenter` | ✅ | Order shuffling, paraphrase passthrough, entity swaps + H7 negatives, distractor injection, R7 adversarial rows |
| `spancall/data/mix.py` — `DatasetMixer` | ✅ | 55/20/10/15 stratified mix (F / paraphrase / swap-distractor / abstention), `partition_by_schema`, 7-way `finalize_splits` (incl. `test_ood_para`, `test_adversarial`) |
| `spancall/model/` — encoder, heads, loss, masks | ✅ | `CandidateMaskBuilder`, `SpanCallTransformerEncoder`, `ToolHead`/`PointerHead`/`AbstainHead`/`CallAcceptHead`/`CorrectnessHead`, `SpanCallLossContainer` — smoke-verified, tests cover shapes, gradients, H8 stop-grad, R4 invariance |
| `spancall/train/trainer.py` + `scripts/train.py` | ✅ | Bucket collator, differential LR + layer decay, bf16 AMP, staged curriculum, checkpoint provenance |
| `spancall/train/calibrate.py` + `scripts/calibrate.py` | ✅ | L-BFGS temperature scaling (scipy + pure-Python verified), Clopper-Pearson sweep, per-tier thresholds, report→resolver round-trip |
| `spancall/infer/` — resolver, decode, cast, gate, export | ✅ | `ExclusivityJointDecoder` (R4 + exclusivity + Delta_joint), `StrictTypeCaster` (Q4/Q5), `SchemaValidator` (Q6), `SpanCallResolver` (end-to-end, latency telemetry, calibration-aware gates); `export.py` shell remains |
| `spancall/agent/` — loop, router, backends (litellm/local), tools | ✅ | `ToolCallRouter` (4 modes × 2 variants), `ReActAgentLoop`, `UniversalAPIBackend` (litellm + `.env`) / `LocalVLLMBackend`, `MockBookingEnvironment` |
| `spancall/eval/` — toolcall + agentic evaluators | ✅ | `ToolCallEvaluator` (7-bucket taxonomy, risk-coverage, ECE, B1a), `ComparativeEvaluator` (paired bootstrap 10k, non-inferiority margin), `MultiSeedAgentEvaluator`, `CostModel` |
| `scripts/prepare_data.py` | ✅ | End-to-end ingest → schema-disjoint split → JSONL + report |
| `scripts/train.py` | ✅ | CLI for staged training (T0/T1/A-B/C curriculum) |
| `scripts/calibrate.py` | ✅ | CLI for temperature fit + threshold sweep + report write |
| `scripts/eval_toolcall.py` | ✅ | CLI for offline tool-call eval + B1a baseline; wrote `toolcall_metrics_b1a_eval.json` |
| `scripts/eval_agent.py` | ✅ | CLI for E9-style comparative agent eval (deterministic backend offline; universal litellm; checkpoint resolver); wrote `e9_s_inject.json` |
| `scripts/probe.py` | ✅ | E0 traffic probe |

**Size:** 40+ test files across `tests/` (`model`, `train`, `infer`, `eval/toolcall`, `eval/agentic`, `agent`, `data`), ~6,500+ lines of implementation.

**Test suite (364 passing after Phase 4b + review fixes):** Phase 1 — `tests/data/test_align.py` (15), `tests/data/test_augment_mix.py` (12), `tests/data/test_hermes.py`, `span_template/test_template.py`, `tests/test_gate.py`, `tests/test_clock.py`, `tests/test_probe.py`, `tests/test_boilerplate.py`; Phase 2 — `tests/model/test_masks.py`, `tests/model/test_heads_loss.py`; Phase 3 — `tests/infer/test_cast_decode.py`, `tests/train/test_calibrate.py`, `tests/eval/toolcall/test_evaluator.py`; Phase 4 — `tests/agent/test_router_loop.py`, `tests/agent/test_backends_env.py`, `tests/agent/test_chat_template_roundtrip.py`, `tests/eval/agentic/test_compare.py` (13 — bootstrap CI, non-inferiority, paired-list + task-id-ordering guards, recovery same-trace test, to_dict serialization incl. per_task_comparisons + theoretical_cost_note, cost-model bounds, multiseed eval, script smoke test). Last run: `.venv/bin/python -m pytest -q` → **364 passed in ~7 s**.

---

## 4. Progress vs. roadmap

Roadmap phases (original 6-week plan):

| Phase | Deliverable | Est. |
| --- | --- | --- |
| P0 | Boilerplate, configs, record schema (+ E0 traffic probe, template fixes D1–D3, config defaults D4–D5) | — |
| P1 | Loaders + aligner + F/A stats | 1 wk |
| P2 | Train T0–T2; E1 (pointer ceiling); B0/B1 offline harness | 1–1.5 wk |
| P3 | Decode/cast/validate/gate + calibration; E2–E5 | 1 wk |
| P4 | Agent loop, router, shadow mode, mock tool envs | 1 wk |
| P5 | Agentic eval E7, E9–E11; cost E6 | 1 wk |
| P6 | Ablations, quantization, write-up | 1 wk |

Current standing:

| Phase | Status | Detail |
| --- | --- | --- |
| **P0** | ✅ done | Boilerplate, configs, record schema, E0 traffic probe, template fixes |
| **P1** | ✅ **done** | Hermes 5-config loader, schema-disjoint partition, `ReverseAligner`, `DataAugmenter`, `DatasetMixer` implemented & tested; E0 gate run end-to-end (see §2.5) |
| **P2** | ✅ **done** | `CandidateMaskBuilder`, encoder wrapper, 5 heads (H1/H3/H4/H8 + accept), `SpanCallLossContainer`, `SpanCallTrainer` + `scripts/train.py` — 176 tests green (see §2.6) |
| **P3** | ✅ **done** | `ExclusivityJointDecoder`, `StrictTypeCaster` + `SchemaValidator`, `SpanCallResolver`, `SpanCallCalibrator`, `ToolCallEvaluator` — 304 tests green; B1a eval report written (see §2.7) |
| **P4** | ✅ **done** | `ToolCallRouter` (4 modes × 2 variants), `ReActAgentLoop`, litellm/vLLM backends, `MockBookingEnvironment` — 351 tests green (see §2.8) |
| **P5** | ✅ **done** | `ComparativeEvaluator` (paired bootstrap, non-inferiority, task-id pairing guard, cost note), `MultiSeedAgentEvaluator`, `scripts/eval_agent.py` — 364 tests green; E9 smoke run in `results/e9_s_inject.json` (see §2.9) |
| **P6** | ⬜ pending | Ablations, quantization, write-up |

Experiment gates (kill floors deliberately below success gates so results can't overlap): E1 pointer-ceiling on F — success ≥ 90% exact payload, kill < 80%; E2 precision gating — success ≥ 95% precision-at-hit at ≥ 60% coverage of F, kill = cannot reach ≥ 95% precision at any coverage ≥ 40% of F. Between gate and floor = stalled experiment (investigate); at/below floor = stop and write up the negative result.

---

## 5. Known gaps & open questions (tracked, not glossed)

1. **Base LLM(s) for the agent comparison** — resolved to a degree in Phase 4a: `.env`-driven `UniversalAPIBackend` (litellm, any OpenAI-compatible endpoint) + `LocalVLLMBackend`; `--backend deterministic` gives an offline fallback for CI/smoke.
2. Agentic benchmark choice: τ-bench, BFCL multi-turn, ToolSandbox, or custom mock — the E9/E10/E11 harness runs on `custom_mock` (deterministic `MockBookingEnvironment`); a real τ-bench/TAU-Bench task suite can drop into `MultiSeedAgentEvaluator` via the existing `{"task_id", "instruction", "environment"}` contract.
3. Compute budget (GPUs; vLLM serving) — still open; the E9 smoke run needs no GPU (deterministic backend).
4. Whether to explore an `s_hint` variant or stay strictly all-or-nothing — still open; current variants are `s_skip` / `s_inject` only.
5. ~~Data-side: paraphrase-disjoint `TEST_OOD_PARA` / adversarial `TEST_ADVERSARIAL` / `calib`~~ — resolved in Phase 1e: produced by `prepare_data.py` (`test_ood_para` 1,558 all-F; `test_adversarial` 30,934 all-A; `calib` 1,718).
6. The glaive config contains an inherent ~16.6% plain-Q&A mix — decide at mix time whether to keep it as natural-language filler (F stratum purity vs. coverage).
7. **E0 gate (Phase 1e):** glaive passes the 0.50 F-share floor (0.758); `func_calling*` sits at 0.285 — dominated by `ExpertQAExtractor` synthesised-query rows with genuinely ungroundable GT (`OFF_WINDOW`/`NEEDS_REASONING`), not a masking bug; `json_mode_*` is structurally ungroundable (~0.04). Decision: proceed with glaive F + R7/adversarial A for training; func_calling/json_mode feed Stratum.A + the E7 abstention check as documented coverage gaps.
8. **E9 smoke state (Phase 4b + review fixes):** with no trained checkpoint the resolver abstains everywhere, so `results/e9_s_inject.json` shows a **latency-only** 1.14× speedup from the s_inject history short-circuit and zero token/cost savings — the definitive E9–E11 numbers require a trained checkpoint + calibrated thresholds wired through `--spancall_checkpoint` / `--calib_thresholds` (the harness, statistical machinery, and CLI are all in place).
9. **Latent metric caveat (tracked, non-blocking):** `aggregate_metrics.side_effecting_error_count` reads `step.spancall_features.get("risk_tier")`, but `router._trace` only stores resolver features — the router currently never populates `risk_tier`, so the metric reads 0. No test asserts it; wiring the risk tier through `ResolveResult` telemetry is a small follow-up.
10. **Theoretical vs empirical cost gap (annotated, not hidden):** at hit_rate = 0 the theoretical formula charges the per-call attempt cost `c_s` on every turn (`E[cost] = 1.07` vs baseline 1.0 → −7.0%), while the empirical ledger only bills executed fast-path/LLM calls (ROUTER-mode abstains bill nothing → 0.0%). `ComparativeStudyResult.to_dict()` now emits `theoretical_cost_note` explaining this, so the artifact can't be misread as a contradiction.

Research hygiene (frozen test sets; calibration set never trains; schemas disjoint across splits; every number traces to `results/<run_id>/` with config + seed + checkpoint hash; ≥ 3 seeds for reported results) — noted here to keep the standing rules in the single doc.

---

## 6. What's next (in priority order)

1. **Run the real E1/T1 pointer-ceiling gate** with the implemented training stack: train on the Phase-1e 7-split output (train 24,071 / dev 1,720 / calib 1,718 / test_id 3,426 / test_ood_schema 3,437; F slice) via `scripts/train.py`; E1 success ≥ 90% exact payload on dev-F, kill < 80%.
2. **Run E2–E7 offline gates end-to-end:** calibrate on `calib` via `scripts/calibrate.py`, then eval via `scripts/eval_toolcall.py` against `test_id` / `test_ood_schema` / `test_ood_para` / `test_adversarial`; E2 success ≥ 95% precision-at-hit at ≥ 60% coverage of F.
3. **Train a real checkpoint + threshold set, then re-run the E9/E10/E11 comparative protocol** with `scripts/eval_agent.py --spancall_checkpoint <dir> --calib_thresholds <file>` (universal litellm backend or `--backend deterministic` for CI) to populate the fast-path cost/token metrics that the current all-abstain `results/e9_s_inject.json` does not exercise.
4. **Port a real multi-turn benchmark** (τ-bench / BFCL multi-turn / ToolSandbox) into the `{"task_id", "instruction", "environment"}` task-suite contract for `MultiSeedAgentEvaluator`, replacing the custom mock domain.
5. **Small follow-ups:** wire `risk_tier` from resolver telemetry into `StepTrace.spancall_features` (see §5.9); extend the remaining loader stubs (xlam/glaive/toolbench/bfcl/tau_bench) reusing `partition_schema_disjoint`; decide on an `s_hint` variant vs. strictly all-or-nothing (§5.4).
6. **P6** — ablations (s_skip vs s_inject, margin sensitivity), quantization, and the write-up.