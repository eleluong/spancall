# Plan: Remove Legacy v1 Flow & Dead Docs — Keep Only Lean Flows

**Status:** Draft for execution  
**Goal:** Remove the legacy 5-head non-lean architecture (`SpanCallModel`, H2/H4/H8/CallAccept abstain machinery, temperature calibrator, legacy curriculum trainer) and obsolete v1 notebooks/scripts/docs/plans, leaving a clean, self-contained **Lean** codebase (`spancall/lean/` + shared template, data, encoder, mask, casting, and evaluation primitives).

---

## 1. Target Repository Architecture (What Stays)

```text
multi_span_tool_calling/
├── PROJECT_STATE.md
├── pyproject.toml
├── configs/
│   └── default.yaml                          # Cleaned of legacy H4/H8/curriculum keys
├── span_template/                            # Grammar, parser, sanitizer, template stamp
│   ├── __init__.py
│   ├── parser.py
│   ├── sanitize.py
│   ├── stamp.py
│   ├── template.py
│   ├── template.jinja
│   ├── template_multi.jinja
│   └── test_template.py
├── spancall/
│   ├── __init__.py
│   ├── data/                                 # Ingestion, v2 alignment, augmentation, splits
│   │   ├── __init__.py
│   │   ├── align.py
│   │   ├── augment.py
│   │   ├── base.py
│   │   ├── mix.py
│   │   ├── partition.py
│   │   ├── schema.py
│   │   └── sources/
│   │       ├── __init__.py
│   │       └── hermes.py
│   ├── model/                                # Shared backbone, candidate masks, H1 tool heads
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── heads.py                          # BaseHead, ToolHead, BinaryToolHead only
│   │   └── masks.py                          # CandidateMaskBuilder, register_marker_tokens
│   ├── infer/                                # Shared deterministic caster, validator, clock
│   │   ├── __init__.py
│   │   ├── cast.py                           # StrictTypeCaster (Q4/Q5), SchemaValidator (Q6)
│   │   └── clock.py                          # local_today, tz_offset_minutes
│   ├── train/                                # Shared config dataclasses
│   │   ├── __init__.py
│   │   └── config.py
│   ├── lean/                                 # Primary Lean model, trainer, resolver, eval
│   │   ├── __init__.py
│   │   ├── decode.py                         # LeanJointDecoder, LeanValueChoice
│   │   ├── diagnostics.py                    # run_diagnostics, analyze_predictions, audit_gt_offsets
│   │   ├── eval_p1.py                        # eval_param_extraction_lean, headline metrics
│   │   ├── loss.py                           # LeanLossContainer (pointer + softmax/binary tool)
│   │   ├── model.py                          # LeanPointerHead, LeanSpanCallModel
│   │   ├── resolver.py                       # LeanResolver, FakeTokenizer
│   │   ├── tensorize.py                      # tensorize_sample_lean, ToolPool, LeanBatchCollate
│   │   └── trainer.py                        # LeanTrainer, CosineWarmupSchedule
│   ├── eval/                                 # Part-3 offline evaluator & comparative harness
│   │   └── ...
│   └── agent/                                # Context selection & router (wired to LeanResolver)
│       └── ...
├── scripts/
│   ├── prepare_data.py
│   ├── realign_v2.py
│   ├── train_lean.py
│   ├── eval_toolcall.py
│   └── build_lean_v2_colab_notebook.py
├── notebooks/
│   └── spancall_lean_v2_train_eval.ipynb
├── docs/                                     # Core design & mechanism docs (updated for Lean)
├── plans/                                    # Active Lean v2 / v2.1 plans only
└── tests/                                    # Lean + shared infrastructure test suite
```

---

## 2. Phase 1 — Untangle Lean from Legacy v1 Modules

Before deleting any legacy files, sever the 4 remaining imports from `spancall/lean/` and `spancall/agent/` into legacy v1 modules:

1. **Relocate `FakeTokenizer`, `_backbone_max_seq_len`, and `_groupkey` into `spancall/lean/resolver.py`:**
   - Currently imported at [`spancall/lean/resolver.py:26`](../spancall/lean/resolver.py#L26) from `spancall.infer.resolver`.
   - Move their definitions directly into `spancall/lean/resolver.py` and re-export `FakeTokenizer` in `spancall/lean/__init__.py`.
   - Update [`tests/test_lean_flow.py:18`](../tests/test_lean_flow.py#L18) to import `FakeTokenizer` from `spancall.lean.resolver`.

2. **Relocate `CosineWarmupSchedule` into `spancall/lean/trainer.py`:**
   - Currently imported at [`spancall/lean/trainer.py:58`](../spancall/lean/trainer.py#L58) from `spancall.train.trainer`.
   - Move `CosineWarmupSchedule` into `spancall/lean/trainer.py` and re-export from `spancall/train/__init__.py` (or `spancall/lean/__init__.py`) if needed.

3. **Remove legacy 1-indexed batch conversion shims from `spancall/lean/tensorize.py` and `spancall/lean/trainer.py`:**
   - Delete `is_legacy_batch`, `convert_legacy_batch_to_lean`, and `convert_legacy_row_to_lean` in [`spancall/lean/tensorize.py:61-118`](../spancall/lean/tensorize.py#L61-L118).
   - Remove their imports and call sites in [`spancall/lean/trainer.py:51-57`](../spancall/lean/trainer.py#L51-L57) and [`spancall/lean/__init__.py`](../spancall/lean/__init__.py).

4. **Re-wire the agent/eval-agentic layer to `LeanResolver` (larger than a one-line change):**
   - `LeanResolveResult` already exposes every field the router reads (`hit, tool_name, payload, joint_margin, accept_score, abstain_reason, status, payload_partial, missing_required, confidence, features`), and `LeanResolver.try_resolve_candidates` exists — so the router needs only the type-hint change at [`spancall/agent/router.py:22`](../spancall/agent/router.py#L22).
   - But these files **import or subclass the legacy `SpanCallResolver` / `ResolveResult`** and must be migrated too (or deleted if the agent harness is dropped — see Open Decision 2):
     - [`scripts/eval_agent.py`](../scripts/eval_agent.py#L41) (`_ScriptedResolver`, `_EmptyResolver`, and the checkpoint-loading `SpanCallResolver(...)` call at ~L246 — the latter needs real rewrite to `LeanResolver` construction)
     - [`tests/agent/test_router_loop.py`](../tests/agent/test_router_loop.py#L26), [`tests/agent/test_chat_template_roundtrip.py`](../tests/agent/test_chat_template_roundtrip.py#L169), [`tests/agent/test_long_context_agent.py`](../tests/agent/test_long_context_agent.py#L33) (also an end-to-end `SpanCallResolver` test at L484–505 that must be rewritten for `LeanResolver` or dropped)
     - [`tests/eval/agentic/test_compare.py`](../tests/eval/agentic/test_compare.py#L31) (`_StaticResolver(SpanCallResolver)`)
   - Keep `docs/chat_template_gemma4.jinja` (required by `test_chat_template_roundtrip.py` and `local_adapter.py`).

---

## 3. Phase 2 — Delete Legacy v1 Code & Scripts

### 3.1 `spancall/model/`
- **Delete** [`spancall/model/model.py`](../spancall/model/model.py) (`SpanCallModel` 5-head container).
- **Delete** [`spancall/model/loss.py`](../spancall/model/loss.py) (`SpanCallLossContainer` with H3 abstain, H4, H8, CallAccept losses).
- **Edit** [`spancall/model/heads.py`](../spancall/model/heads.py):
  - Keep `BaseHead`, `ToolHead`, and `BinaryToolHead`.
  - Delete `CallCountHead` (H2), `PointerHead` (legacy H3 with abstain logit), `AbstainHead` (H4), `CallAcceptHead`, and `CorrectnessHead` (H8).
- **Edit** [`spancall/model/__init__.py`](../spancall/model/__init__.py): export only `BaseEncoder`, `SpanCallTransformerEncoder`, `BaseHead`, `ToolHead`, `BinaryToolHead`, `CandidateMaskBuilder`.

### 3.2 `spancall/infer/`
- **Delete** [`spancall/infer/decode.py`](../spancall/infer/decode.py) (superseded by [`spancall/lean/decode.py`](../spancall/lean/decode.py)).
- **Delete** [`spancall/infer/resolver.py`](../spancall/infer/resolver.py) (superseded by [`spancall/lean/resolver.py`](../spancall/lean/resolver.py)).
- **Delete** [`spancall/infer/gate.py`](../spancall/infer/gate.py) and [`spancall/infer/export.py`](../spancall/infer/export.py).
- **Edit** [`spancall/infer/__init__.py`](../spancall/infer/__init__.py): export only `BaseTypeCaster`, `StrictTypeCaster`, `SchemaValidator`, `local_today`, `tz_offset_minutes`.

### 3.3 `spancall/train/` & `configs/`
- **Delete** [`spancall/train/trainer.py`](../spancall/train/trainer.py) (superseded by [`spancall/lean/trainer.py`](../spancall/lean/trainer.py)).
- **Delete** [`spancall/train/calibrate.py`](../spancall/train/calibrate.py) (legacy temperature + Clopper-Pearson calibrator for H8/Accept heads).
- **Edit** [`spancall/train/__init__.py`](../spancall/train/__init__.py): export config classes only.
- **Edit** [`spancall/train/config.py`](../spancall/train/config.py) & [`configs/default.yaml`](../configs/default.yaml) — **do this LAST and only after a field-by-field audit** (the lean flow reads `model.*`, `optimization.*`, `data.*`, `output_dir`, `experiment_name`, `seed`, `data_dir`, `use_lora*`, `use_binary_tool_head`, early-stopping fields). Verified legacy-only candidates:
  - `HeadsConfig`: `count`, `abstain_reason`, `source`, `list_size`, `swap`, `correctness` (H2/H4/H5–H8)
  - `OptimizationConfig.abstain_weight`
  - `CalibrationConfig`, `GateConfig` (used only by legacy calibrator/gate; confirm `tests/test_lean_margin_gate.py` and `LeanResolver` don't read them)
  - `TrainingConfig`: `stop_on_gate_kill`, `resolver_factory`, `resolver_eval_split`, `resolver_eval_limit`, `freeze_encoder_stages` (curriculum-only; confirm unused by `LeanTrainer`)
  - Stale docstring at `config.py:210` referencing `spancall.train.trainer.SpanCallTrainer.load_rows`.
  - Delete [`configs/verify20.yaml`](../configs/verify20.yaml) only after confirming nothing loads it.
  - Note: `TrainingConfig.data_dir` default and `prepare_data.py --output_dir` default both point at `data/processed` (see §4.2 caveat).

### 3.4 `spancall/data/sources/`
- **Delete** unimplemented `NotImplementedError` loader stubs:
  - [`spancall/data/sources/bfcl.py`](../spancall/data/sources/bfcl.py)
  - [`spancall/data/sources/glaive.py`](../spancall/data/sources/glaive.py)
  - [`spancall/data/sources/tau_bench.py`](../spancall/data/sources/tau_bench.py)
  - [`spancall/data/sources/toolbench.py`](../spancall/data/sources/toolbench.py)
  - [`spancall/data/sources/xlam.py`](../spancall/data/sources/xlam.py)
- **Edit** [`spancall/data/sources/__init__.py`](../spancall/data/sources/__init__.py) to export `HermesLoader` and `HERMES_CONFIGS`.

### 3.5 `scripts/` & `scratch/`
- **Delete** legacy v1 & Lean v1 scripts:
  - [`scripts/train.py`](../scripts/train.py)
  - [`scripts/calibrate.py`](../scripts/calibrate.py)
  - [`scripts/smoke_trainer.py`](../scripts/smoke_trainer.py)
  - [`scripts/smoke_resolver.py`](../scripts/smoke_resolver.py)
  - [`scripts/smoke_vram_flags.py`](../scripts/smoke_vram_flags.py)
  - [`scripts/diagnose_ceiling_and_audit.py`](../scripts/diagnose_ceiling_and_audit.py)
  - [`scripts/build_colab_notebook.py`](../scripts/build_colab_notebook.py)
  - [`scripts/build_experiment_notebook.py`](../scripts/build_experiment_notebook.py)
  - [`scripts/build_lean_colab_notebook.py`](../scripts/build_lean_colab_notebook.py) *(superseded by `scripts/build_lean_v2_colab_notebook.py`)*
  - [`scratch/verify_t1_t9.py`](../scratch/verify_t1_t9.py)

---

## 4. Phase 3 — Remove Obsolete Notebooks, Stale Data & Legacy Results

### 4.1 `notebooks/`
- **Delete** all non-lean and Lean v1 notebooks (safe to delete only after the v2 builder no longer reads the v1 notebook/results — see §4.2 warning):
  - [`notebooks/spancall_train_eval_test.ipynb`](../notebooks/spancall_train_eval_test.ipynb)
  - [`notebooks/spancall_train_eval_test_old.ipynb`](../notebooks/spancall_train_eval_test_old.ipynb)
  - [`notebooks/spancall_train_eval_test_v1.ipynb`](../notebooks/spancall_train_eval_test_v1.ipynb)
  - [`notebooks/spancall_train_eval_test_v2.ipynb`](../notebooks/spancall_train_eval_test_v2.ipynb)
  - [`notebooks/spancall_experiment_verify.ipynb`](../notebooks/spancall_experiment_verify.ipynb)
  - [`notebooks/spancall_lean_train_eval_test.ipynb`](../notebooks/spancall_lean_train_eval_test.ipynb)
  - [`notebooks/notebook_old/spancall_lean_train_eval_test_prev.ipynb`](../notebooks/notebook_old/spancall_lean_train_eval_test_prev.ipynb) (and remove `notebooks/notebook_old/`)
- **Keep:** [`notebooks/spancall_lean_v2_train_eval.ipynb`](../notebooks/spancall_lean_v2_train_eval.ipynb).
- **Edit** [`scripts/build_lean_v2_colab_notebook.py`](../scripts/build_lean_v2_colab_notebook.py): make the v1 comparison cell optional/self-contained or remove references to `results/lean/12-11-26/preds_eval.jsonl` and `data/processed/`.

### 4.2 `data/` & `results/`

> [!WARNING]
> **Do not delete `data/processed/` or `results/lean/12-11-26/` as originally drafted.** Verified live dependencies:
> - [`scripts/realign_v2.py`](../scripts/realign_v2.py#L77) default `--src_dir data/processed` — `data/processed_v2` is *derived* from it; deleting the source removes the regeneration path (re-running `prepare_data.py` needs HF network).
> - [`tests/test_lean_margin_gate.py:25`](../tests/test_lean_margin_gate.py#L25) reads `results/lean/12-11-26/preds_eval.jsonl` (W2 margin-gate derivation).
> - [`scripts/build_lean_v2_colab_notebook.py`](../scripts/build_lean_v2_colab_notebook.py#L150) (`V1_RESULTS_DIR`, v1-vs-v2 comparison cells, `data/processed` Colab download at L324) and [`scratch/smoke_v2_cells.py`](../scratch/smoke_v2_cells.py#L48).
> - [`spancall/lean/resolver.py:382`](../spancall/lean/resolver.py#L382) comment cites that file as the evidence for the gate default.
>
> Recommended: **keep both** (they are small evidence/source data). If removed, first (a) rename/relocate the margin-gate fixture into `tests/data/`, (b) strip the V1 comparison cells + `V1_RESULTS_DIR` from the v2 builder and `scratch/smoke_v2_cells.py`, (c) decide that `data/processed_v2` is now the canonical source and repoint `prepare_data`/`realign_v2` defaults.

- **Delete** legacy result artifacts (safe — no code references; re-verify with grep before deleting):
  - `results/e9_s_inject.json`
  - `results/long_8k.json`, `results/long_20k.json`, `results/long_32k.json`, `results/long_scripted.json`
  - `results/probe_smoke.json`, `results/test_legacy.json`
  - `results/toolcall_metrics_b1a_eval.json`, `results/toolcall_metrics_hermes_eval.json`
  - `results/lean/12-11-26/` (stale-offset Lean v1 predictions)
- **Keep:** `data/processed_v2/`, `data/tensorized_lean/`, `results/lean/12-11-26-v2/`.

---

## 5. Phase 4 — Prune & Update Test Suite

1. **Delete legacy test files:**
   - [`tests/model/test_heads_loss.py`](../tests/model/test_heads_loss.py) *(port `ToolHead` shape/grad unit test into `tests/test_lean_binary_tool.py` before deleting)*
   - [`tests/train/test_calibrate.py`](../tests/train/test_calibrate.py)
   - [`tests/train/test_trainer_fraction.py`](../tests/train/test_trainer_fraction.py) *(delete `tests/train/` directory)*
   - [`tests/infer/test_partial_extraction.py`](../tests/infer/test_partial_extraction.py)
   - [`tests/test_gate.py`](../tests/test_gate.py)
   - [`tests/test_boilerplate.py`](../tests/test_boilerplate.py)
2. **Update shared test files:**
  - [`tests/infer/test_cast_decode.py`](../tests/infer/test_cast_decode.py): keep `StrictTypeCaster` (Q4/Q5) and `SchemaValidator` (Q6) tests; remove `ExclusivityJointDecoder` and `SpanCallResolver` tests (line 30 imports `ExclusivityJointDecoder, ValueChoice, _Cand` from the deleted `spancall.infer.decode`; Lean equivalents live in `tests/test_lean_flow.py`).
  - [`tests/test_lean_margin_gate.py`](../tests/test_lean_margin_gate.py#L25): reads `results/lean/12-11-26/preds_eval.jsonl` — keep that file or relocate it (see §4.2).
   - [`tests/test_lean_flow.py`](../tests/test_lean_flow.py): remove tests targeting `convert_legacy_batch_to_lean` / `convert_legacy_row_to_lean` / `is_legacy_batch`; import `FakeTokenizer` from `spancall.lean.resolver`.
   - [`tests/test_notebook_cell_guard.py`](../tests/test_notebook_cell_guard.py): point `BUILD_SCRIPT` at `scripts/build_lean_v2_colab_notebook.py` and `nb_path` at `notebooks/spancall_lean_v2_train_eval.ipynb`.
   - [`tests/agent/test_router_loop.py`](../tests/agent/test_router_loop.py) & [`tests/eval/agentic/test_compare.py`](../tests/eval/agentic/test_compare.py): update mock/resolver imports to `LeanResolver` / `LeanResolveResult`.

---

## 6. Phase 5 — Clean Up Documentation & Plans

### 6.1 Delete Dead Docs (`docs/`)
| File to Delete | Reason |
| :--- | :--- |
| [`docs/full_implementation_plan.md`](../docs/full_implementation_plan.md) | Original 5-head implementation blueprint (superseded by Lean) |
| [`docs/notebook_results_analysis.md`](../docs/notebook_results_analysis.md) | Post-mortem of legacy 5-head 5% Colab run |
| [`docs/notebook_v2_review.md`](../docs/notebook_v2_review.md) | Technical recap of legacy 5-head architecture & curriculum |
| [`docs/param_recall_diagnosis.md`](../docs/param_recall_diagnosis.md) | Root-cause diagnosis of legacy H3 abstain dominance (`0.0022` recall) |
| [`docs/remediation_plan.md`](../docs/remediation_plan.md) | Duplicate of legacy `plans/spancall_remediation_plan.md` |
| [`docs/lean_eval_diagnosis_walkthrough.md`](../docs/lean_eval_diagnosis_walkthrough.md) | Diagnosis of Lean v1 stale-offset run (`results/lean/12-11-26`), superseded by v2 |

### 6.2 Delete Completed / Superseded Plans (`plans/`)
| File to Delete | Reason |
| :--- | :--- |
| [`plans/accuracy_improvement_deep_dive.md`](../plans/accuracy_improvement_deep_dive.md) | Legacy 5-head accuracy plan |
| [`plans/close_but_no_hit_diagnosis.md`](../plans/close_but_no_hit_diagnosis.md) | Legacy 5-head gate/abstain diagnosis |
| [`plans/experiment_notebook_plan.md`](../plans/experiment_notebook_plan.md) | Plan for deleted `spancall_experiment_verify.ipynb` |
| [`plans/long_context_fix.md`](../plans/long_context_fix.md) | Historical v1 long-context migration plan (already integrated) |
| [`plans/notebook_staleness_fix.md`](../plans/notebook_staleness_fix.md) | Plan for deleted legacy `spancall_train_eval_test.ipynb` |
| [`plans/spancall_remediation_plan.md`](../plans/spancall_remediation_plan.md) | Historical v1 remediation plan |
| [`plans/lean_0.2_param_extraction_and_pack_tools_investigation.md`](../plans/lean_0.2_param_extraction_and_pack_tools_investigation.md) | Completed T1–T9 investigation on old Lean v1 notebook |
| [`plans/lean_remaining_t10_t13_plan.md`](../plans/lean_remaining_t10_t13_plan.md) | Completed T10–T13 plan |
| [`plans/lean_workflow_mismatch_audit.md`](../plans/lean_workflow_mismatch_audit.md) | Completed audit (all items resolved) |
| [`plans/lean_workflow_remaining_issues.md`](../plans/lean_workflow_remaining_issues.md) | Completed R0–R10 issue list |
| [`plans/lean_workflow_upgrade_plan.md`](../plans/lean_workflow_upgrade_plan.md) | Completed R0–R10 upgrade plan |

### 6.3 Keep Active Plans (`plans/`)
- [`plans/hermes_function_calling_v1.md`](hermes_function_calling_v1.md) — Data ingestion & schema-disjoint split reference.
- [`plans/lean_v2_flow_plan.md`](lean_v2_flow_plan.md) — Lean v2 alignment, template stamp, and margin gate architecture.
- [`plans/binary_tool_selection_plan.md`](binary_tool_selection_plan.md) — `BinaryToolHead` design & ablation spec.
- [`plans/v2_candidate_parity_and_w5_plan.md`](v2_candidate_parity_and_w5_plan.md) — Active v2.1 parity & binary-head roadmap.

### 6.4 Fix dangling doc references in kept code (new — found in re-check)
Many kept source files cite docs/plans in docstrings. After deletion these point nowhere:
- `docs/full_implementation_plan.md` §refs in: `spancall/data/{align,augment,mix}.py`, `spancall/eval/toolcall/evaluator.py`, `spancall/model/masks.py`, `spancall/infer/cast.py`-adjacent tests (`tests/infer/test_cast_decode.py`, `tests/eval/toolcall/test_evaluator.py`).
- `docs/param_recall_diagnosis.md` in [`spancall/lean/diagnostics.py:3`](../spancall/lean/diagnostics.py#L3).
- `docs/remediation_plan.md` in `scripts/probe.py:58` (probe is kept).
- `docs/heads.md` / `docs/mechanism.md` / `docs/pipeline_io.md` / `docs/proposal.md` sections cited across `span_template/*`, `spancall/model/{encoder,masks}.py`, `spancall/agent/router.py`, etc. These docs are **kept**, but heads.md §H2/H4/H8 content is being removed (§6.5), so section refs like `heads.md §H4` in `spancall/data/align.py:26,55` and `scripts/probe.py:58` become stale.
- Action: one grep-driven sweep replacing deleted-file references with the kept equivalent (`docs/param_extraction_eval_coverage.md`, `docs/mechanism.md`) or removing the `Ref:` line. Do **not** delete `docs/param_extraction_eval_coverage.md`, `docs/chat_template_gemma4.jinja`, `docs/proposal.md` (it is `pyproject.toml`'s `readme`), `docs/mechanism.md`, `docs/pipeline_io.md`, `docs/experiments.md`.

### 6.5 Update Living Docs
- [`PROJECT_STATE.md`](../PROJECT_STATE.md): Rewrite §§2.6–2.7 and §3 component table to reflect the Lean architecture (`LeanSpanCallModel`, `LeanPointerHead`, `ToolHead`/`BinaryToolHead`, `LeanLossContainer`, `LeanResolver`, `data/processed_v2`).
- [`docs/architecture.md`](../docs/architecture.md) & [`docs/heads.md`](../docs/heads.md): Remove H2/H4/H8/CallAccept head sections and document the two-head Lean architecture (H1 `ToolHead`/`BinaryToolHead` + H3 `LeanPointerHead`).

---

## 7. Open Decisions (need your call before execution)

1. **Keep v1 evidence?** `data/processed/` and `results/lean/12-11-26/` are still wired into `realign_v2.py`, `test_lean_margin_gate.py`, the v2 notebook builder and `scratch/smoke_v2_cells.py` (§4.2). Default in this plan: **keep**.
2. **Agent / agentic-eval harness** (`spancall/agent/`, `spancall/eval/{agentic,compare,cost}`, `scripts/eval_agent.py`, `tests/agent/*`, `tests/eval/agentic/*`): all coupled to legacy `SpanCallResolver`. Either **migrate to `LeanResolver`** (Phase 1.4) or **delete entirely** (also drops `PROJECT_STATE.md` §2.8–2.9 claims and the E9 results). Not "lean" either way until chosen.
3. **Scripts not mentioned in the first draft** — `scripts/probe.py` (E0 probe, with `tests/test_probe.py` + `tests/data/probe_samples.jsonl`) is independent of the legacy stack: default **keep**.

## 8. Verification Gates

1. **Zero dangling imports:**
   - `grep -rn "spancall\.model\.\(loss\|model\)" spancall/ scripts/ tests/` returns 0 matches.
   - `grep -rn "spancall\.infer\.\(decode\|gate\|export\|resolver\)" spancall/ scripts/ tests/` returns 0 matches.
   - `grep -rn "spancall\.train\.\(trainer\|calibrate\)" spancall/ scripts/ tests/` returns 0 matches.
2. **Notebook builder & cell guard:**
   - `python scripts/build_lean_v2_colab_notebook.py` exits 0 with cell-scope guard passing.
3. **Full test suite:**
   - `.venv/bin/python -m pytest -q` passes 100% with zero collection errors.

