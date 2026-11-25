# SpanCall Research Experiment Ledger

This document tracks hypotheses, configurations, verification gates, and decisions across all experimental phases.

**Reading rule:** every section is a *reporting contract*, and a number is only
comparable to another number produced under the same contract. §I covers the
Lean Phases A–J upgrade; §II covers the Lean **v3** upgrade, which retires an
arm rather than adding one. A v2.x `single_pass` number and a v3 `two_step`
number are different experiments, not two readings of the same quantity.

## Experiment Tracker

| Exp ID | Question | Hypothesis | Metric / Gate | Baseline | Status |
| --- | --- | --- | --- | --- | --- |
| **E0** | Traffic probe (D7) | >=80% of live calls are stratum F (fully groundable by the current template) | Per-workload F-share >= cost-model floor (default 0.5); abstain-reason mix; scope-cut deltas; retriever R@T | No fast path (LLM-only) | Planned |
| **E1** | Ceiling on Stratum F | Pointer head achieves >=90% exact payload on groundable calls | Dev Exact Payload >= 90% | Zero-shot LLM | Planned |
| **E2** | Precision Gating | Calibrated conjunction gate reaches >=95% precision | Precision-at-Hit >= 95% at Coverage >= 60% of F | Ungated Argmax | Planned |
| **E3** | Swap Resistance | Entity swap hard negatives reduce swap errors by >= 2x | Swap Error Rate <= 2% | Model w/o swaps | Planned |
| **E4** | Delta_joint Utility | Joint assignment margin detects ambiguous entity bindings | Delta-AUROC on swap errors | Per-param margin | Planned |
| **E5** | Generalization | Head generalizes to unseen schemas with <=5 pt drop | Held-out Schema Exact Payload | In-distribution | Planned |
| **E6** | Compute Savings | Speculative fast path yields >=5x latency win over 0.5B generator | p50 latency, TFLOPs | 0.5B LM, 7B LM | Planned |
| **E7** | Non-harm on Stratum A | Stratum A calls false-hit <= 1-2% | False-hit rate on A <= 2% | Ungated Model | Planned |
| **E8** | Multi-schema Selection | Batched encoder scoring selects correct tool schema | Wrong-tool hit <= 1% | LLM Tool Choice | Planned |
| **E9** | Agent Non-inferiority | Agent + SpanCall matches Normal Agent success within 2% margin | Task Success Delta >= -2% (p < 0.05) | Normal Agent (B0) | Planned |
| **E10** | History Consistency | S-inject maintains context coherence on later multi-turn steps | Turn N>1 call accuracy | S-skip variant | Planned |
| **E11** | Context Scaling | Recent context rows beyond 5 rows yield diminishing returns | Precision vs latency curve | Full history | Planned |
| **E12** | Backbone Comparison | ModernBERT outperforms DeBERTa-v3 at lower inference latency | Latency vs exact match | DeBERTa-v3 | Planned |
| **E13** | Quantization Shift | INT8 post-training quantization preserves calibrated precision | Post-quant Precision >= 94.5% | BF16 baseline | Planned |
| **E14** | Shadow Mode Transfer | Calibration on logged user traffic transfers across domains | Precision on Target Domain | Static calibration | Planned |
| **E15** | Head Ablations | Dedicated [PARAM] query vectors outperform pooled CLS query | Per-param Exact Match | CLS-only pointer | Planned |

---

## Metrics Delta — Lean Workflow Upgrade (Phases A–J)

Phases A–J changed **what the model sees** (prompt layout, tool coverage, recent
context, call axis) and **how a prediction is scored** (every GT call instead of
call 0). Numbers produced before the upgrade are therefore **not comparable** to
numbers produced after it. This section fixes the reporting contract: the same
keys, read off the same artifacts, so two runs can be diffed without guessing
which definition moved.

### I.1 Where each number now lives

**There is no `tensorize_stats.json`.** The tensorize pass accumulates a `stats`
dict and the notebook *prints* it (`"Lean tensorize summary:"`); nothing writes
it to disk. So the "Artifact" column below says **printed report** wherever that
is the truth rather than inventing a filename. Verified against
`spancall/lean/tensorize.py`, `spancall/lean/trainer.py`,
`spancall/lean/eval_p1.py`, and the v2.2 builder that holds the notebook-side
aggregation.

| Quantity | Artifact | Key(s) |
| --- | --- | --- |
| Train prompt layout | **printed report** — `stats` dict built in `spancall/lean/tensorize.py` | `render_T_hist`, `render_T_sum`, `render_T_rows`, `jittered_rows`, `jitter_draw_hist`. Note `render_T_t1_share` is **not** a tensorize key — it is computed in `eval_p1.py`; the notebook derives its own share from `render_T_hist` |
| Train H1 job size | **split two ways** | `mean_n_candidates` comes from `LeanTrainer.evaluate` (`trainer.py`); `n_candidates_hist` and `rows_with_h1_supervision` / `rows_without_h1_supervision` are **notebook-computed** in the v2.2/v2.4 builders and are **absent from the v3 notebook**, which no longer builds that histogram |
| Train supervision losses | **printed report** | `dropped_over_length`, `tool_snap_misses`, `param_snap_misses`, `multi_call_ambiguous_anchors`, `multi_call_anchor_missing_calls`, `tool_truncated_rows`, `abstain_rows`. The drop-reason split lives under the nested `param_drop_reasons` dict whose **actual values** are `unresolved_target` / `unaligned_target` / `null_slot_missing` / `absent_from_gt` / `match_missed`; the flat `param_drop_match_missed` / `param_drop_unresolved` / `param_drop_other` names are a **notebook-side regrouping** of those five, not keys any module emits |
| Train recent-context budget | **printed report** | `n_recent_rows`, `recent_truncated_rows` |
| Part 1 param extraction | `p1_metrics` (in-notebook dict) | `extraction_recall` / `extraction_precision` / `extraction_f1`, `fast_path_param_coverage`, `fast_path_call_coverage`, `mean_render_T`, `render_T_hist`, `render_T_t1_share`, `recent_budget_effective` |
| Part 1 call axis (J6) | `p1_metrics` | `call_exact_set_match`, `call_exact_set_match_F`, `call_count_acc`, `n_calls_hist`, `pred_n_calls_hist`, `n_multi_call_samples` |
| Part 2 H1 | printed report | Top-1 tool accuracy, `mean_n_candidates` + histogram, `mean_n_candidates`-vs-1.5 vacuity warning |
| Part 3 end-to-end | **v3:** `results_lean_v3/toolcall_metrics_{split}_{bucket}_{system}.json` — **v2.x:** `results_lean/toolcall_metrics_{split}_lean.json`. The v3 lean arm is named `lean_twostep`, since `single_pass` is retired | `exact_payload_match`, `hit_rate`, `param_f1`, `metrics_by_stratum.F.*`, `errors_by_sample` |
| Part 3 call axis (J6) | same file | `call_exact_set_match`, `call_count_acc`, `n_calls_hist`, `pred_n_calls_hist`, `metrics_by_stratum.F.call_exact_set_match`, `metrics_by_stratum.F.call_count_acc`, plus per-row `n_gt_calls` / `n_pred_calls` / `call_set_equal` / `call_count_equal` in the scored row set |
| Part 3 latency | per-row `latency_ms` in `preds_eval.jsonl` | now **additive** across stages (R5) |

### I.2 What each change does to the number

- **`render_T` (R1c, G).** Training now jitters `pack_tools` down to 1, so the
  train histogram develops mass at `T=1` instead of being `T>=3` only. The
  eval side renders the oracle tool (Part 1) or the candidate packet (Part 3).
  Read the **T=1 share on both sides**: if train is 0% at `T=1`, the eval layout
  is out of distribution and Part 1 numbers are optimistic; if eval is 100% at
  `T=1`, H1 is a single-class softmax and its accuracy is vacuous. Both cases
  are now printed as explicit warnings.
- **Tool coverage (R0.1, B).** `pack_tools` is a **floor**, never a cap, so a
  sample with 5 native tools renders 5 tool blocks. `mean_render_T` and
  `mean_n_candidates` both rise; a `tool_truncated_rows > 0` means an explicit
  `max_tools` still cut a native tool and H1 can never select it.
- **Context (R0.2/R1b, D).** Part 1 passes `context_schemas=s.tools`, so the
  prompt it scores is the same packet inference will see. Part 1 recall is
  therefore **lower and honest** where it used to be measured on a `T=1`
  oracle-only prompt.
- **Recent budget (R8/R2, E).** Training now selects recent rows with the same
  `select_recent_context` the resolver uses. `n_recent_rows` and
  `recent_truncated_rows` quantify how much of the block the model actually saw;
  Part 1 reports `recent_budget_effective` so a clamped budget is visible.
- **Call axis (R0.4, J1–J6).** The headline changes in two ways at once:
  - Part 1 scores **every** GT call's parameters (per-param rows tagged
    `call_index::param`) instead of silently dropping calls 1..n.
  - `exact_payload_match` / `exact_payload_match_F` remain **call-0 only**, by
    design, so their historical series stays continuous. The new
    `call_exact_set_match` requires every GT call exact **and** matching call
    counts, so `set < call-0` is the expected signature of dropped or invented
    calls. Report both; never replace one with the other.
- **Tie to Phase I denominator.** `n_multi_call_samples` is the base rate that
  decides how much any of this can move: with 0 multi-call rows the two
  exactness numbers must be identical, and a gap there is a bug.

### I.3 Expected direction (gate, not a result)

These are predictions to falsify, not measurements. Record actual values in the
rows below once a post-upgrade run exists.

| Quantity | Pre-upgrade expectation | Post-upgrade expectation | Why |
| --- | --- | --- | --- |
| Part 1 `extraction_recall` | measured on `T=1` oracle prompt | **lower** | scored on the real multi-tool packet, all calls |
| Part 1 `fast_path_param_coverage` | under-counted | **higher** | every call's params now counted |
| Part 2 tool accuracy | optimistic (T=1 rows included) | **closer to reality** | `mean_n_candidates` reported beside it |
| Part 3 `exact_payload_match` | call-0 only | unchanged definition | series stays comparable |
| Part 3 `call_exact_set_match` | n/a (did not exist) | **<= call-0 number** | it also requires call count agreement |
| `multi_call_ambiguous_anchors` | invisible | **non-zero allowed** | honest failure count on shared anchors |

### I.4 Post-upgrade run log

| Run ID | Date | Split(s) | Part 1 recall / coverage | Part 1 set-exact / count-acc | Part 2 tool acc (cands/row) | Part 3 exact / set-exact / count-acc | train T=1 share | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _(pending)_ | | | | | | | | run the notebook end-to-end and fill from the printed report |

---

## Metrics Delta — Lean v3 Upgrade (Two-Step-Only, Stratified Eval, W&B, Three-Mode Agentic)

v3 differs from the A–J upgrade in kind. A–J changed **what was measured** while
keeping one resolution arm. v3 **deletes an arm** (`single_pass`), **changes the
eval design** (per-corpus, per-tool-count-bucket instead of one aggregate),
**changes eval cadence** (every 1000 optimiser steps on a small disjoint probe,
not once per epoch), and **adds a serving comparison** (three agentic modes).
Any of the four alone breaks comparability; together they make v2.x and v3
different experiments.

### II.1 Where each number now lives

| Quantity | Artifact | Key(s) |
| --- | --- | --- |
| Resolution arm | `resolve_config.json` (HF export) | `resolve_mode` must be `two_step`; `single_pass` is retired. `LeanResolver.__init__` still defaults to `single_pass`, so a run that did not pin the arm is a `single_pass` run regardless of what it is called |
| Pointer-head geometry | `resolve_config.json` | `head_dim`, `max_span_width`, `max_calls` — a consumer rebuilding without `head_dim` silently gets the 256 default |
| Mid-training probe curve (trainer) | `train_log.jsonl` in the run's `exp_root` | records with `event == "probe_eval"`; also `LeanTrainer.history["probe_eval"]` |
| Mid-training probe curve (W&B) | W&B run (`WANDB_PROJECT`) | `probe/eval_loss`, `probe/exact_call`, `probe/span_acc`, `probe/tool_acc`, `probe/epoch` — logged every `probe_every_steps` (1000) **optimiser** steps. Note the W&B names are `exact_call` / `span_acc`, not `exact_payload` / `hit_rate` |
| Probe-set integrity | printed by the probe-build cell | `overlap_with_dev` (printed, not asserted — must be 0 to trust the curve), `PROBE_SOURCES`, `PROBE_N`, `PROBE_PERCENT` (fallback cadence when no step count is set) |
| Train/val loss curves | W&B, distinct from `probe/*` | `train/*` at log cadence, `dev/<split>/*` per **epoch** — `probe/` and `dev/` must not share an axis |
| Per-corpus metrics | **separate eval runs per split** | corpus separation is at the *file* level, not a nested key: `hermes_eval` and `xlam_eval` are scored independently (`results_lean_v3/toolcall_metrics_<split>_*.json`). There is no `by_corpus` dict to read |
| Per-tool-count metrics | `results_lean_v3/toolcall_metrics_<split>_<bucket>_<system>.json` | bucket is in the **filename** (`1-2`, `3-5`, `6-10`, `11+`); metrics under `metrics_by_stratum.F` (`n`, `exact_payload_match`, `hit_rate`, `param_f1`) |
| Eval-set composition | notebook §4.0b printed table | corpus × bucket **row counts**. Read before any metric table: an `exact_F` that moved because the slice gained 20-distractor-tool rows is not a model improvement. Empty buckets are named explicitly |
| Corpus × bucket pivot | notebook §5c printed table | `split × corpus × bucket × system`; a bucket absent from a split is **skipped**, not reported as 0 |
| LLM vs SpanCall parity | `scripts/compare_llm_spancall.py` stdout + `--out` JSON (default `results/llm_parity.json`) | four arms: `llm_only`, `lean_fast_path`, `lean_plus_fallback` (SpanCall with LLM backfill), `b1a_regex`. Per arm `exact_payload_match`, `hit_rate`, `param_f1`, `call_exact_set_match`, plus cost (`fallback_rate`, `usd`) and a `pivot` keyed `"<corpus>|<bucket>"` → `exact_payload_match` |
| Agentic mode comparison | `results_lean_v3_2/agentic/agentic_mode_table.json` → `rows` | per (`suite`, `mode`): `success`, `turns_per_task`, `spancall_hit_rate`, `wrong_hit_rate`, `llm_calls`, `delta_llm_calls`, `tok_in`, `tok_out`, `delta_tokens`, `wall_ms`, `usd`, `decisions` histogram. Envelope also records `resolver` (incl. `max_cands`, `collapse_guard`, `prevent_param_collapse`), `backend`, `llm_model_id`, `num_seeds`, `max_turns` |
| Agentic honesty flag | same file → `resolver.kind`, `backend` | `resolver.kind != "lean"` means modes 2–3 abstained every turn and are indistinguishable from mode 1 by construction; `backend == "deterministic"` means the LLM arm tests plumbing, not LLM quality. Both cases print a caveat beside the table |
| Agentic hybrid-fill behavior | same → `decisions` histogram | `HYBRID_FILL` count, params filled vs resolved; a fill that fails schema validation must appear as a full `LLM_FALLBACK`, never as a partial payload |

### II.2 What each change does to the number

- **`single_pass` → `two_step` (retired arm).** This is not a tuning
  preference. The two arms produce *different predictions on the same input*,
  so pre-v3 numbers cannot be reused as a baseline column and must be
  re-measured. `LeanResolver.__init__` still defaults to `single_pass`, so a
  number produced by a consumer that forgot to pin `two_step` is a
  **`single_pass` number mislabelled as v3** — the train/eval notebook, the
  agentic rebuild, and `resolve_config.json` all pin it, and
  [`tests/test_notebook_v3_fixes.py`](tests/test_notebook_v3_fixes.py) fails if
  a fourth site appears.
- **Eval cadence: per-epoch → per-1000-optimiser-steps.** The probe curve is
  more informative *and* less final: it is measured on 300 rows drawn from
  `PROBE_SOURCES` with dev `sample_id`s removed, so it can move on a
  comparatively small sample. Read it as a **trend**, and check the printed
  `overlap_with_dev` first — it is **reported, not asserted**, so a non-zero
  value will not fail the run; it means the curve is tracking the dev set and
  the early-stopping signal is contaminated. Probe eval fires only on an
  accumulation-window boundary, so the weights scored are exactly the ones the
  next micro-batch trains from.
- **Stratification: aggregate → corpus × bucket.** Aggregate exact-payload can
  be unchanged while `11+` degrades and `1-2` improves, because the aggregate
  weights by sample count. A bucket with fewer than a handful of samples
  reports `n` beside every metric; do not quote a bucket whose `n` is in the
  noise floor. The pivot is the point: it is the only view in which a
  bucket-specific regression is visible at all.
- **LLM parity: separate evaluations → same-sample comparison.** The parity
  script scores both systems with the same `ToolCallEvaluator` on the same
  `sample_id` set, so a difference is a difference in prediction quality, not a
  difference in scoring. The `lean_plus_fallback` arm answers the serving
  question the pure arms cannot: what the *deployed* system's payload accuracy
  is once abstains are backfilled.
- **Three agentic modes: `LLM_ONLY` / `ROUTER` / `HYBRID_FILL`.** `ROUTER` moves
  the whole call between SpanCall and the LLM; `HYBRID_FILL` splits a single
  call — SpanCall commits the tool and the resolvable arguments, the LLM fills
  only `unresolved_param_names`, and `merge_filled_arguments` never overwrites a
  resolved value. So `HYBRID_FILL` cuts LLM tokens **and** is the mode where a
  wrong SpanCall argument is guaranteed to survive into the payload; its
  success rate is a direct measure of SpanCall argument precision, not a
  measure of the LLM's ability to recover. A scripted/`DeterministicBackend`
  run measures **mode wiring only** and prints that caveat next to its numbers.
- **`selected_tool` is a hypothesis, not a fact.** `ToolCallRouter` records the
  tool it would use even on `LLM_FALLBACK`, so per-turn accuracy must be read
  off `tool_output is not None` (the call actually executed) — the agentic
  table keys on that, and a mode that reports high accuracy with low executions
  is reporting hypotheses.

### II.3 Post-v3 run log

| Run ID | Date | Arm pinned | Probe every | Probe overlap w/ dev | Eval slice (corpus rows / empty buckets) | Part 3 exact / set-exact / count-acc | LLM parity (Δ exact on `lean_plus_fallback`) | Agentic: mode1 / mode2 / mode3 success | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `v3.3` | 2026-11-20 | `two_step` | 2000 optimiser steps | 0 (313-row disjoint probe) | Tier 1 Fair Eval (`n=1,735` per corpus) + Tier 2 Full Splits (`34,969` rows) | Macro `exact_F`: `0.755` (`0.743` H / `0.766` X) | `Router+Schema`: `0.857` (Gemma 4) / `0.855` (`gpt-oss-120b`); +11.1–11.2 pp over LLMs on xLAM | Offline Tier 1 `Router` / `Router+Schema` / `HybridFill` / `HybridFill+Schema` complete; multi-turn agentic suites (BFCL / $\tau$-bench) are future work | Full audited report in [`results_lean_v3_3/export/final_report.md`](../results_lean_v3_3/export/final_report.md) and §III below |

---

## Section III — Lean v3.3 Paper Benchmark Results (Completed)

Aligned with [`research_paper/sections/05_results.tex`](../research_paper/sections/05_results.tex) and [`07_appendix.tex`](../research_paper/sections/07_appendix.tex).

- **Model & Training:** `answerdotai/ModernBERT-large` (395M, $d=1024$, $L_{\max}=4096$, $W_{\max}=16$, $C \le 1024$, $d_h=256$), 1:1 balanced Hermes V1 (19,096 → 69,412) + xLAM-60k (69,412) = 138,824 records → 117,760 tensorized instances, 3 epochs `bf16`, effective batch 128, cosine decay with 10% floor, seed 42. Macro dev `exact_call`: `0.857 → 0.868 → 0.874` (Epoch 3 restored).
- **Tier 1 Matched Fair Eval (`data/eval_v3_3/`, $n=1{,}735$ per corpus, $n_F=1{,}277$ [1,108 Normal + 169 Complex], $n_A=458$, 10,000-resample paired bootstrap CIs):**

| System | Hermes `exact_F` | xLAM `exact_F` | **Macro `exact_F`** | Macro Complex `exact_F` | `false_hit_A` ↓ | Mean Lat. (ms) | Mean API Tok / Turn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SpanCall v3.3** | `.743` | `.766` | **`.755`** | `.331` | **`.664`** | **77.0** | **0** |
| `gemma-4-26b-a4b-it` | `.390` | `.778` | `.584` | **`.754`** | `.895` | 7025.6 | 443 |
| Router (Gemma 4) | `.801` | `.815` | `.808` | `.379` | `.975` | **1364.0** | **88** |
| **Router+Schema (Gemma 4)** | **`.826`** | **`.889`** | **`.857`** | **`.754`** | `.956` | 2565.6 | 179 |
| HybridFill (Gemma 4) | `.805` | `.796` | `.800` | `.364` | `.927` | 2958.7 | 225 |
| HybridFill+Schema (Gemma 4) | `.822` | `.866` | `.844` | `.698` | **`.853`** | 4721.4 | 345 |
| `gpt-oss-120b` | `.420` | `.777` | `.598` | **`.746`** | `.879` | 2986.7 | 500 |
| Router (gpt-oss-120b) | `.800` | `.814` | `.807` | `.376` | `.965` | **657.2** | 102 |
| **Router+Schema (gpt-oss-120b)** | `.821` | **`.889`** | **`.855`** | **`.746`** | `.936` | 1151.1 | 200 |
| HybridFill (gpt-oss-120b) | `.805` | `.797` | `.801` | `.367` | `.931` | 809.0 | **100** |
| HybridFill+Schema (gpt-oss-120b) | **`.823`** | `.871` | `.847` | `.710` | **`.862`** | 1446.5 | 177 |

- **Normal vs. Complex (`exact_F`):** SpanCall leads decisively on all-scalar `Normal` calls (`0.798` Hermes vs. `0.360`/`0.397`; `0.840` xLAM vs. `0.755`/`0.755`), while LLMs dominate `Complex` (`array`/`object`) calls (`0.586`/`0.568` Hermes, `0.923` xLAM vs. SpanCall `0.385`/`0.278`).
- **Tier 2 Full Splits (34,969 rows, standalone SpanCall):** ID eval (`0.708` H / `0.757` X), OOD-Schema (`0.711` H / `0.799` X), OOD-Para (`0.735` H / `0.793` X), Adversarial `false_hit_A` (`0.816` H / `0.788` X).

---

## Experiment Execution Protocol

1. **Config-First**: Every experiment must be runnable via `python scripts/... --config configs/experiments/<exp_id>.yaml`.
2. **Deterministic Seeds**: Run experiments with at least 3 random seeds (e.g. 42, 43, 44).
3. **Artifact Logging**: Results, metrics, predictions, and calibration curves stored in `results/<run_id>/`.
4. **Kill Criteria (floors)**: Kill floors are deliberately set **below** the success gates in the tracker above, so a "passed gate" and a "killed" result can never overlap:
   - **E0 floor**: per-workload F-share < cost-model floor `F_floor = h_min / 0.6` (default 0.5). F-share is a coverage *upper bound* on hit rate, so F=0.5 under E2's 60%-of-F coverage target implies achievable `h ≈ 0.3`. Below the floor => stop and write up the negative result.
   - **E1 floor**: exact payload < 80% on F dev (success gate is >= 90%).
   - **E2 floor**: cannot reach >= 95% precision-at-hit at **any** coverage >= 40% of F (success gate is >= 95% at coverage >= 60% of F).
   - Between gate and floor = *stalled experiment*: investigate before continuing. At/below the floor, stop and write up the negative result.
