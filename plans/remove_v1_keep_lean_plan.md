# Plan: Remove Legacy v1 Flow & Dead Docs — Keep Only Lean Flows + Shareable Boilerplate

**Status:** ✅ **Completed** (Phases 1–5 executed, dead code/docs removed, Lean + boilerplate tests green)  
**Goal:** Remove the concrete legacy 5-head non-lean implementation (`SpanCallModel`, H2/H4/H8/CallAccept abstain machinery, legacy `PointerHead` with abstain logit, legacy `SpanCallTrainer`, legacy `SpanCallCalibrator`, legacy `SpanCallResolver`) and obsolete v1 notebooks/scripts/docs/plans, while **keeping**:
1. **All data & Lean evidence** (`data/processed/`, `data/processed_v2/`, `data/tensorized_lean/`, `results/lean/12-11-26*`, and all `spancall/data/` including loader stubs).
2. **All shareable building blocks & ABCs** (`span_template/`, `BaseEncoder`/`SpanCallTransformerEncoder`, `BaseHead`/`ToolHead`/`BinaryToolHead`, `CandidateMaskBuilder`, `BaseTypeCaster`/`StrictTypeCaster`/`SchemaValidator`, `clock.py`, `BaseExporter`, `BaseJointDecoder`, `BaseTrainer`, `BaseCalibrator`, `CosineWarmupSchedule`, `FakeTokenizer`, `spancall/eval/`).
3. **A version-agnostic Agent & Resolver Boilerplate (`BaseResolver` / `ResolveResultProtocol`)** so `spancall/agent/` and `scripts/eval_agent.py` work with `LeanResolver` today and can plug in any future v2.x / v3 resolver without rewriting the agent loop or router.

---

## 1. Target Repository Architecture (What Stays)

```text
multi_span_tool_calling/
├── PROJECT_STATE.md                          # Updated for Lean v2 + pluggable agent contract
├── pyproject.toml
├── configs/
│   └── default.yaml                          # Cleaned of legacy H4/H8/curriculum-only keys
├── data/                                     # KEPT (all splits & tensorized caches)
│   ├── processed/                            #   Source splits (read by realign_v2.py)
│   ├── processed_v2/                         #   Re-aligned v2 splits + template_stamp.json
│   └── tensorized_lean/                      #   Tensorized Lean caches
├── results/
│   └── lean/                                 # KEPT (12-11-26 & 12-11-26-v2 evidence)
├── span_template/                            # KEPT (shared grammar, parser, sanitizer, stamp)
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
│   ├── data/                                 # KEPT (all ingestion, alignment, mix, loader stubs)
│   │   ├── __init__.py
│   │   ├── align.py
│   │   ├── augment.py
│   │   ├── base.py                           #   BaseDatasetLoader, BaseAligner, BaseAugmenter
│   │   ├── mix.py
│   │   ├── partition.py
│   │   ├── schema.py
│   │   └── sources/                          #   HermesLoader + shareable loader boilerplate stubs
│   │       ├── __init__.py
│   │       ├── hermes.py
│   │       └── {bfcl,glaive,tau_bench,toolbench,xlam}.py
│   ├── model/                                # Shared backbone, candidate masks, H1 tool heads
│   │   ├── __init__.py
│   │   ├── encoder.py                        #   BaseEncoder, SpanCallTransformerEncoder
│   │   ├── heads.py                          #   BaseHead, ToolHead, BinaryToolHead only
│   │   └── masks.py                          #   CandidateMaskBuilder, register_marker_tokens
│   ├── infer/                                # Shared deterministic caster, clock & ABC boilerplate
│   │   ├── __init__.py
│   │   ├── base.py                           #   NEW: BaseResolver, BaseResolveResult, BaseJointDecoder, FakeTokenizer
│   │   ├── cast.py                           #   BaseTypeCaster, StrictTypeCaster (Q4/Q5), SchemaValidator (Q6)
│   │   ├── clock.py                          #   local_today, tz_offset_minutes
│   │   └── export.py                         #   BaseExporter boilerplate
│   ├── train/                                # Shared config & trainer/calibrator ABCs
│   │   ├── __init__.py
│   │   ├── base.py                           #   NEW: BaseTrainer, BaseCalibrator, CosineWarmupSchedule
│   │   └── config.py                         #   TrainingConfig, ModelConfig, DataConfig, OptimizationConfig
│   ├── lean/                                 # Active Lean implementation (plugs into shared ABCs)
│   │   ├── __init__.py
│   │   ├── decode.py                         #   LeanJointDecoder(BaseJointDecoder), LeanValueChoice
│   │   ├── diagnostics.py                    #   run_diagnostics, analyze_predictions, audit_gt_offsets
│   │   ├── eval_p1.py                        #   eval_param_extraction_lean, headline metrics
│   │   ├── loss.py                           #   LeanLossContainer (pointer + softmax/binary tool)
│   │   ├── model.py                          #   LeanPointerHead(BaseHead), LeanSpanCallModel
│   │   ├── resolver.py                       #   LeanResolver(BaseResolver), LeanResolveResult
│   │   ├── tensorize.py                      #   tensorize_sample_lean, ToolPool, LeanBatchCollate
│   │   └── trainer.py                        #   LeanTrainer(BaseTrainer)
│   ├── eval/                                 # KEPT (Part-3 offline evaluator & comparative harness)
│   │   ├── __init__.py
│   │   ├── compare.py
│   │   ├── cost.py
│   │   ├── agentic/
│   │   └── toolcall/
│   └── agent/                                # KEPT (pluggable via BaseResolver boilerplate)
│       ├── __init__.py
│       ├── context.py
│       ├── loop.py
│       ├── router.py                         #   Accepts any BaseResolver (LeanResolver or future v3)
│       ├── trace.py
│       ├── backends/
│       └── tools/
├── scripts/
│   ├── prepare_data.py
│   ├── realign_v2.py
│   ├── probe.py                              # KEPT (E0 template/traffic probe)
│   ├── train_lean.py
│   ├── eval_toolcall.py
│   ├── eval_agent.py                         # Re-wired to BaseResolver / LeanResolver factory
│   └── build_lean_v2_colab_notebook.py
├── notebooks/
│   └── spancall_lean_v2_train_eval.ipynb
├── docs/                                     # Updated core design & mechanism docs
├── plans/                                    # Active Lean v2 / v2.1 plans only
└── tests/                                    # Lean + shared infrastructure test suite
```

---

## 2. Phase 1 — Create Shareable Boilerplate (`spancall/infer/base.py` & `spancall/train/base.py`) and Pluggable Agent Contract

To make sure new model/resolver versions can slot in without touching `spancall/agent/`, `spancall/eval/`, or shared utilities:

### 2.1 `spancall/infer/base.py` (New Shareable Inference Boilerplate)
Extract the version-agnostic contracts and shared tokenizer utilities out of the legacy files into `spancall/infer/base.py`:
- **`BaseResolveResult` / `ResolveResult`** (dataclass):
  - Standard fields read by `ToolCallRouter`, `ReActAgentLoop`, and `ComparativeEvaluator`:
    `hit: bool`, `tool_name: Optional[str]`, `payload: Optional[Dict[str, Any]]`, `joint_margin: float`, `accept_score: float`, `abstain_reason: Optional[str]`, `status: str`, `payload_partial: Dict[str, Any]`, `missing_required: List[str]`, `confidence: float`, `latency_ms: float`, `features: Dict[str, Any]`.
  - `LeanResolveResult` subclasses `BaseResolveResult` (so `isinstance(res, BaseResolveResult)` holds everywhere).
- **`BaseResolver` (ABC)**:
  - Defines the minimal pluggable interface any version (Lean v2, v2.1 binary-head, or future v3) implements:
    ```python
    class BaseResolver(ABC):
        @abstractmethod
        def try_resolve(
            self,
            schema: ToolDefinition,
            utterance: str,
            recent_context: Optional[List[Dict[str, Any]]] = None,
            now: Optional[date] = None,
        ) -> BaseResolveResult:
            raise NotImplementedError

        @abstractmethod
        def try_resolve_candidates(
            self,
            candidate_schemas: List[ToolDefinition],
            utterance: str,
            recent_context: Optional[List[Dict[str, Any]]] = None,
            now: Optional[date] = None,
        ) -> BaseResolveResult:
            raise NotImplementedError

        @classmethod
        def from_checkpoint(cls, checkpoint_dir: str, **kwargs: Any) -> "BaseResolver":
            """Optional factory hook for CLI/agent scripts."""
            raise NotImplementedError
    ```
  - Also provide `NoOpResolver(BaseResolver)` (always abstains with `abstain_reason="no_checkpoint"`) and `ScriptedResolver(BaseResolver)` right in `spancall/infer/base.py` so `scripts/eval_agent.py` and agent tests never need to subclass a concrete model resolver just to mock outcomes.
- **`BaseJointDecoder` (ABC)**: relocated from `spancall/infer/decode.py`; `LeanJointDecoder` subclasses it.
- **`FakeTokenizer`, `_backbone_max_seq_len`, `_groupkey`**: relocated from `spancall/infer/resolver.py` into `spancall/infer/base.py` (and re-exported in `spancall/infer/__init__.py` and `spancall/lean/resolver.py`).

### 2.2 `spancall/train/base.py` (New Shareable Training Boilerplate)
Extract the reusable training abstractions into `spancall/train/base.py`:
- **`BaseTrainer` (ABC)**: `train()`, `evaluate()`, `save_checkpoint()`, `load_checkpoint()` — subclassed by `LeanTrainer` and any future trainer.
- **`BaseCalibrator` (ABC)**: `fit_temperature()`, `sweep_thresholds()` — kept as an interface for future Lean/v2.1 gate calibration.
- **`CosineWarmupSchedule`**: shared LR schedule used by `LeanTrainer`.

### 2.3 Remove Legacy 1-Indexed Batch Conversion Shims
- In [`spancall/lean/tensorize.py:61-118`](../spancall/lean/tensorize.py#L61-L118), remove `is_legacy_batch`, `convert_legacy_batch_to_lean`, and `convert_legacy_row_to_lean`.
- Remove their imports/calls in [`spancall/lean/trainer.py:51-57`](../spancall/lean/trainer.py#L51-L57) and [`spancall/lean/__init__.py`](../spancall/lean/__init__.py).

### 2.4 Wire `spancall/agent/` and `scripts/eval_agent.py` to `BaseResolver`
- [`spancall/agent/router.py`](../spancall/agent/router.py): type-hint `resolver: BaseResolver` and `BaseResolveResult` from `spancall.infer.base`.
- [`scripts/eval_agent.py`](../scripts/eval_agent.py): use `BaseResolver`, `NoOpResolver`, and `LeanResolver` (when `--spancall_checkpoint` is provided).
- [`tests/agent/test_router_loop.py`](../tests/agent/test_router_loop.py), [`tests/agent/test_chat_template_roundtrip.py`](../tests/agent/test_chat_template_roundtrip.py), [`tests/agent/test_long_context_agent.py`](../tests/agent/test_long_context_agent.py), and [`tests/eval/agentic/test_compare.py`](../tests/eval/agentic/test_compare.py): import `BaseResolver` / `BaseResolveResult` (or `ResolveResult` alias) from `spancall.infer.base`, and run the end-to-end long-context resolver test in `test_long_context_agent.py` against `LeanResolver`.

---

## 3. Phase 2 — Delete Only Concrete Legacy v1 Code & Scripts

### 3.1 `spancall/model/`
- **Delete** [`spancall/model/model.py`](../spancall/model/model.py) (legacy 5-head `SpanCallModel`).
- **Delete** [`spancall/model/loss.py`](../spancall/model/loss.py) (legacy `SpanCallLossContainer` with H3 abstain, H4, H8, CallAccept losses).
- **Edit** [`spancall/model/heads.py`](../spancall/model/heads.py):
  - **Keep:** `BaseHead`, `ToolHead`, `BinaryToolHead`.
  - **Delete:** `CallCountHead` (H2), `PointerHead` (legacy H3 with abstain logit), `AbstainHead` (H4), `CallAcceptHead`, `CorrectnessHead` (H8).
- **Edit** [`spancall/model/__init__.py`](../spancall/model/__init__.py): export `BaseEncoder`, `SpanCallTransformerEncoder`, `BaseHead`, `ToolHead`, `BinaryToolHead`, `CandidateMaskBuilder`.

### 3.2 `spancall/infer/`
- **Delete** [`spancall/infer/decode.py`](../spancall/infer/decode.py) (concrete 1-indexed `ExclusivityJointDecoder`; `BaseJointDecoder` lives in `spancall/infer/base.py` and `LeanJointDecoder` in `spancall/lean/decode.py`).
- **Delete** [`spancall/infer/resolver.py`](../spancall/infer/resolver.py) (concrete legacy `SpanCallResolver`; `BaseResolver` / `BaseResolveResult` / `FakeTokenizer` live in `spancall/infer/base.py` and `LeanResolver` in `spancall/lean/resolver.py`).
- **Delete** [`spancall/infer/gate.py`](../spancall/infer/gate.py) (legacy `SingleTauGate` / `ConjunctionGate`).
- **Keep** [`spancall/infer/cast.py`](../spancall/infer/cast.py), [`spancall/infer/clock.py`](../spancall/infer/clock.py), [`spancall/infer/export.py`](../spancall/infer/export.py), and [`spancall/infer/base.py`](../spancall/infer/base.py).

### 3.3 `spancall/train/` & `configs/`
- **Delete** [`spancall/train/trainer.py`](../spancall/train/trainer.py) (concrete legacy `SpanCallTrainer`; `BaseTrainer` and `CosineWarmupSchedule` live in `spancall/train/base.py`).
- **Delete** [`spancall/train/calibrate.py`](../spancall/train/calibrate.py) (concrete legacy `SpanCallCalibrator`; `BaseCalibrator` lives in `spancall/train/base.py`).
- **Delete** [`configs/verify20.yaml`](../configs/verify20.yaml).
- **Edit** [`spancall/train/config.py`](../spancall/train/config.py) & [`configs/default.yaml`](../configs/default.yaml) (last step, after tests pass): prune legacy-only flags (`HeadsConfig` H2/H4/H5–H8 booleans, `OptimizationConfig.abstain_weight`, `TrainingConfig.stop_on_gate_kill` / `resolver_factory` / `freeze_encoder_stages`).

### 3.4 `spancall/data/`
- **Keep everything in `spancall/data/`**, including `base.py` (`BaseDatasetLoader`, `BaseAligner`, `BaseAugmenter`) and the dataset loader boilerplate shells in `spancall/data/sources/{bfcl,glaive,tau_bench,toolbench,xlam}.py` so additional benchmarks can be plugged in easily.

### 3.5 `scripts/` & `scratch/`
- **Delete** legacy v1 & obsolete Lean v1 scripts:
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
- **Keep:** `prepare_data.py`, `realign_v2.py`, `probe.py`, `train_lean.py`, `eval_toolcall.py`, `eval_agent.py`, `build_lean_v2_colab_notebook.py`, and `scratch/smoke_v2_cells.py`.

---

## 4. Phase 3 — Clean Up Notebooks & Non-Lean Result Files (Keep Data & Lean Evidence)

### 4.1 `notebooks/`
- **Delete** obsolete non-lean and Lean v1 notebooks:
  - [`notebooks/spancall_train_eval_test.ipynb`](../notebooks/spancall_train_eval_test.ipynb)
  - [`notebooks/spancall_train_eval_test_old.ipynb`](../notebooks/spancall_train_eval_test_old.ipynb)
  - [`notebooks/spancall_train_eval_test_v1.ipynb`](../notebooks/spancall_train_eval_test_v1.ipynb)
  - [`notebooks/spancall_train_eval_test_v2.ipynb`](../notebooks/spancall_train_eval_test_v2.ipynb)
  - [`notebooks/spancall_experiment_verify.ipynb`](../notebooks/spancall_experiment_verify.ipynb)
  - [`notebooks/spancall_lean_train_eval_test.ipynb`](../notebooks/spancall_lean_train_eval_test.ipynb)
  - [`notebooks/notebook_old/spancall_lean_train_eval_test_prev.ipynb`](../notebooks/notebook_old/spancall_lean_train_eval_test_prev.ipynb) (and directory `notebooks/notebook_old/`)
- **Keep:** [`notebooks/spancall_lean_v2_train_eval.ipynb`](../notebooks/spancall_lean_v2_train_eval.ipynb).

### 4.2 `data/` & `results/`
- **Keep all of `data/`:** `data/processed/`, `data/processed_v2/`, `data/tensorized_lean/`.
- **Keep `results/lean/`:** `results/lean/12-11-26/` and `results/lean/12-11-26-v2/`.
- **Delete** top-level legacy v1 smoke JSONs in `results/`:
  - `results/long_8k.json`, `results/long_20k.json`, `results/long_32k.json`, `results/long_scripted.json`
  - `results/test_legacy.json`, `results/toolcall_metrics_hermes_eval.json`

---

## 5. Phase 4 — Prune & Update Test Suite

1. **Delete tests for removed concrete v1 classes:**
   - [`tests/model/test_heads_loss.py`](../tests/model/test_heads_loss.py) *(port `ToolHead` unit tests into `tests/test_lean_binary_tool.py` first)*
   - [`tests/train/test_calibrate.py`](../tests/train/test_calibrate.py)
   - [`tests/train/test_trainer_fraction.py`](../tests/train/test_trainer_fraction.py) *(remove `tests/train/` dir)*
   - [`tests/infer/test_partial_extraction.py`](../tests/infer/test_partial_extraction.py)
   - [`tests/test_gate.py`](../tests/test_gate.py)
2. **Update shared & boilerplate tests:**
   - [`tests/test_boilerplate.py`](../tests/test_boilerplate.py): **keep and update** so it tests the shareable ABCs (`BaseDatasetLoader`, `BaseEncoder`, `BaseHead`, `BaseResolver`, `BaseResolveResult`, `BaseJointDecoder`, `BaseTypeCaster`, `BaseExporter`, `BaseTrainer`, `BaseCalibrator`) and `LeanResolver` / `LeanSpanCallModel` / `LeanTrainer` contracts!
   - [`tests/infer/test_cast_decode.py`](../tests/infer/test_cast_decode.py): keep all `StrictTypeCaster` (Q4/Q5) and `SchemaValidator` (Q6) tests; remove the legacy 1-indexed `ExclusivityJointDecoder` and `SpanCallResolver` tests.
   - [`tests/test_lean_flow.py`](../tests/test_lean_flow.py): remove tests for deleted `convert_legacy_batch_to_lean` / `convert_legacy_row_to_lean` / `is_legacy_batch`.
   - [`tests/test_notebook_cell_guard.py`](../tests/test_notebook_cell_guard.py): point `BUILD_SCRIPT` at `scripts/build_lean_v2_colab_notebook.py` and `nb_path` at `notebooks/spancall_lean_v2_train_eval.ipynb`.
   - [`tests/agent/*`](../tests/agent/) & [`tests/eval/agentic/test_compare.py`](../tests/eval/agentic/test_compare.py): use `BaseResolver` / `BaseResolveResult` / `LeanResolver`.

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
| [`docs/lean_eval_diagnosis_walkthrough.md`](../docs/lean_eval_diagnosis_walkthrough.md) | Diagnosis of Lean v1 stale-offset run, superseded by v2 plans |

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
- [`plans/lean_v2_candidate_parity_fix.md`](lean_v2_candidate_parity_fix.md) — Active v2 candidate-parity investigation & fix plan.
- [`plans/binary_tool_selection_plan.md`](binary_tool_selection_plan.md) — `BinaryToolHead` design & ablation spec.
- [`plans/v2_candidate_parity_and_w5_plan.md`](v2_candidate_parity_and_w5_plan.md) — Active v2.1 parity & binary-head roadmap.

### 6.4 Fix Dangling Doc References & Update Living Docs
- Sweep docstrings in kept `.py` files to remove references to deleted docs (`full_implementation_plan.md`, `param_recall_diagnosis.md`, `remediation_plan.md`).
- Update [`PROJECT_STATE.md`](../PROJECT_STATE.md), [`docs/architecture.md`](../docs/architecture.md), and [`docs/heads.md`](../docs/heads.md) to document the Lean architecture, shared ABC boilerplate, and pluggable `BaseResolver` agent contract (already updated in this pass).

---

## 7. Verification Gates

1. **Zero dangling legacy imports:**
   - `grep -rn "spancall\.model\.\(loss\|model\)" spancall/ scripts/ tests/` returns 0 matches.
   - `grep -rn "spancall\.infer\.\(decode\|gate\|resolver\)" spancall/ scripts/ tests/` returns 0 matches.
   - `grep -rn "spancall\.train\.\(trainer\|calibrate\)" spancall/ scripts/ tests/` returns 0 matches.
2. **Notebook builder & cell guard:**
   - `python scripts/build_lean_v2_colab_notebook.py` exits 0 with cell-scope guard passing.
3. **Full test suite:**
   - `.venv/bin/python -m pytest -q` passes 100% with zero collection errors.
