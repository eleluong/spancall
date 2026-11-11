# SpanCall Research Experiment Ledger

This document tracks hypotheses, configurations, verification gates, and decisions across all experimental phases.

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

| Quantity | Artifact | Key(s) |
| --- | --- | --- |
| Train prompt layout | `data_tensorized_lean/tensorize_stats.json` | `render_T_hist`, `render_T_t1_share`, `jittered_rows`, `jitter_pack_tools` |
| Train H1 job size | same | `mean_n_candidates`, `n_candidates_hist`, `rows_without_h1_supervision` |
| Train supervision losses | same | `dropped_over_length`, `tool_snap_misses`, `param_snap_misses`, `multi_call_ambiguous_anchors`; plus the drop-reason split — `param_drop_match_missed` / `param_drop_unresolved` / `param_drop_other` are **notebook-flattened** from the nested `param_drop_reasons` dict (the JSON artifact keeps the nested form) |
| Train recent-context budget | same | `n_recent_rows`, `recent_truncated_rows` |
| Part 1 param extraction | `p1_metrics` (in-notebook) | `extraction_recall/precision/f1`, `fast_path_param_coverage`, `fast_path_call_coverage`, `mean_render_T`, `render_T_hist`, `render_T_t1_share`, `recent_budget_effective` |
| Part 1 call axis (J6) | `p1_metrics` | `call_exact_set_match`, `call_exact_set_match_F`, `call_count_acc`, `n_calls_hist`, `pred_n_calls_hist`, `n_multi_call_samples` |
| Part 2 H1 | printed report | Top-1 tool accuracy, `mean_n_candidates` + histogram, `mean_n_candidates`-vs-1.5 vacuity warning |
| Part 3 end-to-end | `results_lean/toolcall_metrics_{split}_lean.json` | `exact_payload_match`, `hit_rate`, `param_f1`, `metrics_by_stratum.F.*` |
| Part 3 call axis (J6) | same file | `call_exact_set_match`, `call_count_acc`, `n_calls_hist`, `pred_n_calls_hist`, `metrics_by_stratum.F.call_exact_set_match`, `metrics_by_stratum.F.call_count_acc`, per-row `n_gt_calls` / `n_pred_calls` / `call_set_equal` / `call_count_equal` in `errors_by_sample`-adjacent rows |
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

## Experiment Execution Protocol

1. **Config-First**: Every experiment must be runnable via `python scripts/... --config configs/experiments/<exp_id>.yaml`.
2. **Deterministic Seeds**: Run experiments with at least 3 random seeds (e.g. 42, 43, 44).
3. **Artifact Logging**: Results, metrics, predictions, and calibration curves stored in `results/<run_id>/`.
4. **Kill Criteria (floors)**: Kill floors are deliberately set **below** the success gates in the tracker above, so a "passed gate" and a "killed" result can never overlap:
   - **E0 floor**: per-workload F-share < cost-model floor `F_floor = h_min / 0.6` (default 0.5). F-share is a coverage *upper bound* on hit rate, so F=0.5 under E2's 60%-of-F coverage target implies achievable `h ≈ 0.3`. Below the floor => stop and write up the negative result.
   - **E1 floor**: exact payload < 80% on F dev (success gate is >= 90%).
   - **E2 floor**: cannot reach >= 95% precision-at-hit at **any** coverage >= 40% of F (success gate is >= 95% at coverage >= 60% of F).
   - Between gate and floor = *stalled experiment*: investigate before continuing. At/below the floor, stop and write up the negative result.
