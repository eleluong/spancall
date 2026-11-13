# Param-Extraction: Retrieval-First Design Contract & Honest Metrics

**Status:** ✅ **Implemented in the Lean flow** (L1, L2, E1, E2, E3) — tests green in [`tests/test_lean_flow.py`](../tests/test_lean_flow.py)
**Scope:** Lean flow ([`spancall/lean/`](../spancall/lean/))
**Notebook:** [`notebooks/spancall_lean_v2_train_eval.ipynb`](../notebooks/spancall_lean_v2_train_eval.ipynb)

---

## 1. The design philosophy: extraction is a fast path, not a universal solver

SpanCall's param extraction is **retrieval by design**. The contract is:

> Extract any parameter value that is *present* in the context — literally or through light deterministic transformation (date arithmetic from the clock, enum/boolean resolution, recent-context lookup, null). Params that require **generation** (composing new text, multi-step reasoning) are **out of scope by intent**.

This is not a weakness to apologize for. It is a **routing architecture**:

- **Easy case (majority of real traffic):** the value is in the context. A 0.6B encoder with span pointers extracts it in a single forward pass — orders of magnitude cheaper and faster than an autoregressive LLM generating the full call. This is the "fast path."
- **Hard case (needs generation/derivation beyond the deterministic transforms):** extraction correctly *refuses* — emits nothing / `<|null|>` — and the caller routes to a heavier path (LLM generation, clarification turn, or default value). Refusing is a *correct behavior*, not a failure.

The current loss and eval code does not reflect this contract: the loss silently drops ungroundable params (fine) but doesn't distinguish *why*, and the eval scores ungroundable params as misses (wrong framing — it penalizes the fast path for not being the slow path).

## 2. Background: what is extractable?

`spancall/data/align.py` classifies every param into target types:

| `target_type` | Meaning | Extractable? (pointer + deterministic transform) |
|---|---|---|
| `utterance_span` | Value literally in the user utterance | ✅ |
| `recent_ctx` | Value from a `recent_context` row | ✅ |
| `time_ctx` | Date derivable from `current_time_iso` + clock arithmetic | ✅ (light transform, rendered as text) |
| `enum_ctx` | One of schema enum values | ✅ (rendered) |
| `boolean_ctx` | Yes/no phrasing in utterance | ✅ |
| `null_ctx` | Param legitimately null/omitted | ✅ (`<|null|>`) |
| `unresolved` | Not aligned — needs generation or alignment failed | ❌ by design (or a bug — see §3) |

Candidate spans are built from exactly the ✅ regions ([`spancall/model/masks.py:177`](../spancall/model/masks.py:177)).

**Key subtlety:** "retrieval" already includes *light transformation* — `time_ctx` means the aligner rendered a computed date ("tomorrow" + clock → concrete date text) into the input, and the model points at it. The boundary is: transform that can be **deterministically materialized as text before the model runs** is in-scope; transform requiring open-ended reasoning/generation is out-of-scope.

---

## 3. The two real problems

With the retrieval-first contract in mind, what's actually wrong:

### 3.1 Loss: unmatched params are silently dropped without reason classification

In [`tensorize_sample_lean`](../spancall/lean/tensorize.py:246), a param with no matched candidate gets all-`-1` `pos_indices`, and [`compute_lean_pointer_loss`](../spancall/lean/loss.py:64) zeroes it. Fine for genuinely unextractable params. But three situations are indistinguishable:

- (a) `unresolved` target — correctly out of scope (route to slow path at inference)
- (b) extractable target where the **matcher failed** (canonical-value normalization mismatch, char-offset drift, BPE snapping edge case) — training data silently lost
- (c) `null_ctx` where the `<|null|>` region match failed — same as (b)

No counter anywhere distinguishes (a) from (b)/(c). Since (b)/(c) are pure data-pipeline bugs, they must not hide behind (a)'s legitimacy. (Historical precedent: the legacy D2 silent-abstain bug corrupted ~15% of F-params silently; Lean fixed the corruption but kept the silence.)

### 3.2 Eval: unextractable params scored as misses — wrong framing for a fast path

Eval Part 1 in both notebooks scores recall against **all** GT arguments. Under the retrieval-first contract this is a category error: it measures the fast path against the slow path's job. The metrics a retrieval-first system actually needs:

1. **Extraction quality** (precision/recall over *extractable* params) — is the fast path accurate when it fires?
2. **Fast-path coverage** (fraction of calls/params where extraction suffices) — how often can we skip the expensive path?
3. **Refusal correctness** — when extraction emits nothing, is it because the param is genuinely unextractable (correct refusal) or because the model failed on an extractable param (missed fast-path opportunity)?

Current eval reports none of these cleanly: it blends (1) and (3) into one recall number, doesn't report (2) at all, and `pointer_empty` conflates correct refusals with real failures.

---

## 4. Fixes (implemented in Lean, proposed for legacy)

| Fix | Where it now lives |
|---|---|
| L1 drop-reason classification (5-way) | [`tensorize_sample_lean(stats=...)`](../spancall/lean/tensorize.py:129) → `stats["param_drop_reasons"]`, row key `_param_drop_reasons` |
| L2 deterministic `<|null|>` | per-param `_find_null_idx(cands)` in [`tensorize.py`](../spancall/lean/tensorize.py:263) |
| E1 + E2 | [`eval_p1.py`](../spancall/lean/eval_p1.py) (`extraction_*`, `fast_path_*`, `refused_unextractable` / `missed_extractable`) |
| E3 aligner tripwire | [`e3_aligner_violation()`](../spancall/lean/eval_p1.py) — surfaced as `e3_aligner_drift` / `e3_examples` in the eval metrics and printed by §5a |
| Shared GT-param predicate (nulls included) | [`_gt_params()`](../spancall/lean/eval_p1.py) used by both coverage classification and the E3 tripwire |
| Notebook surfacing | §3 tensorize summary + dev sanity audit + §5a cell in [`scripts/build_lean_v2_colab_notebook.py`](../scripts/build_lean_v2_colab_notebook.py) |

### Fix L1 (loss): classify drop reasons, audit (b)/(c)

In `tensorize_sample_lean`, when `matched_cis == []` for a Stratum-F param, classify and count per split. Five reasons, three of which are defects:

| drop reason | meaning | expected frequency |
|---|---|---|
| `unresolved_target` | target type is `unresolved` — needs generation/transform | **expected** (out of scope) |
| `absent_from_gt` | no target row **and** no GT argument — the param is not part of this call | **benign** |
| `match_missed` | extractable target existed but no candidate matched | **must be ~0 (bug)** |
| `unaligned_target` | target row has neither `canonical_value` nor `char_start` | **must be 0 (data bug)** |
| `null_slot_missing` | answer is null but the mask builder emitted no `<|null|>` candidate | **must be 0 (bug)** |

The tensorize summary prints each reason per split and ends with three tripwires: `match_missed total`, `aligner/null-slot defects total`, and `tensorize exceptions total`. Exceptions raised inside the fork worker are counted by class instead of being silently swallowed.

### Fix L2 (loss): deterministic `<|null|>` supervision

Null supervision is exact: the `<|null|>` slot is located by `region == "null"` per param (candidate sets differ per param), so it can never silently vanish. An **explicitly null GT argument** (`{"unit": null}`) now takes this path because `tensorize.py` distinguishes "GT arg present but null" from "GT arg absent" — the two used to be conflated by `dict.get()`.

### Fix E1 (eval): two primary metrics instead of one blended recall

- `extraction_recall` / `extraction_f1` — restricted to params with extractable target types (utterance_span, recent_ctx, time_ctx, enum_ctx, boolean_ctx, null_ctx). **This is the headline number.**
- `fast_path_coverage` — fraction of GT params (and of calls) that are extractable, i.e. how often the fast path *could* handle everything. This is an architecture-value metric: even if coverage is 60%, a 0.6B encoder handling those 60% at near-zero latency is the win.
- Keep `recall_all` as a secondary diagnostic (the blended number), clearly labeled as mixing in out-of-scope params.

### Fix E2 (eval): split `pointer_empty` into refusal vs failure

- `refused_unextractable` — no extractable target exists (correct refusal → routing signal)
- `missed_extractable` — extractable target existed, model emitted nothing (real fast-path failure)

### Fix E3 (eval): Stratum-F invariant tripwire

Assert every F row's GT args all have extractable-or-null targets; report violations as aligner drift rather than folding them into misses.

**Implemented** as [`e3_aligner_violation()`](../spancall/lean/eval_p1.py), called per Stratum-F sample inside `eval_param_extraction_lean`. Metrics: `e3_aligner_drift` (count), `e3_aligner_drift_rate`, `e3_examples` (up to 10 `sample_id:param:reason`). Violations are reported *alongside*, never folded into, `extraction_recall`.

### Null handling in the eval GT universe (C1/C2)

`_gt_params()` defines the GT universe as *every schema param with a GT entry, explicit nulls included*. A GT `{"param": null}` is a real supervision target — the fast path must select `<|null|>` — so it counts in `fast_path_param_coverage`, in the per-param `gt` column, and in every denominator. Prediction state is null-aware too: "predicted null" (`source_type == "null"`) is distinct from "no prediction", and a wrong value on a null GT param is booked in `pointer_wrong`.

### Metric honesty (S1/S5/S6)

- Zero-denominator ratios are `NaN` (printed `n/a`), never a fabricated `1.0`.
- `resolver_hit_rate_F` (formerly `hit_rate_F`) is the resolver's own schema-validity flag, **not** GT accuracy; it is named to stop it being read as correctness.
- `truncation = {evaluated, total_available, truncated}` makes a `limit`-capped run explicit; a missing split raises `FileNotFoundError` instead of returning `{}`.

### Architectural direction (explicit non-goal of the metric fixes)

The slow path for out-of-scope params already exists conceptually (agent loop / LLM backend). Two optional enhancements that *raise coverage* without breaking the retrieval philosophy:

- **Expand deterministic transforms:** more derived-value candidates materialized pre-inference (arithmetic results, unit conversions, formatted dates) — each addition converts `unresolved` params into `time_ctx`-style extractable ones. Cheap, auditable wins.
- **Confidence-routed hybrid:** when extraction refuses a *required* param, the resolver emits a structured "needs generation for {param}" signal the caller can act on — the refusal becomes an API, not a dead end.

---

## 5. Verification plan

**Done (unit level — 12 tests in [`tests/test_lean_flow.py`](../tests/test_lean_flow.py)):**

1. ✅ Synthetic sample with one extractable + one `unresolved` param: tensorize reports exactly `{"unresolved_target": 1}` and the row carries `_param_drop_reasons` (`test_tensorize_classifies_unresolved_target_drop`).
2. ✅ Matcher failure is not disguised as legitimate exclusion: an extractable target absent from the context yields `{"match_missed": 1}` (`test_tensorize_flags_match_missed_for_extractable_target`).
3. ✅ A targetless param with no GT arg is `absent_from_gt`, **not** a false `match_missed` alarm (`test_tensorize_required_param_without_gt_is_not_match_missed`).
4. ✅ `null_ctx` supervision maps to exactly one candidate slot (`test_tensorize_null_target_uses_single_deterministic_slot`).
5. ✅ An explicit-null GT param is fully supervised (no drop reason, both params get a positive) (`test_tensorize_unresolved_only_call_has_no_supervision_bug`).
6. ✅ Eval scores the extractable subset only: synthetic split gives `extraction_f1 == 1.0` while `recall_all == 0.5`, and the unresolved param is booked as `refused_unextractable` (not a miss) (`test_eval_param_extraction_scores_groundable_params_only`).
7. ✅ A null-only call is 100% fast-path coverable: `fast_path_param_coverage == fast_path_call_coverage == 1.0`, `exact_payload_match_F == 1.0` (`test_eval_counts_explicit_null_gt_param_as_groundable`).
8. ✅ Predicting a value for a null GT param is counted in `pointer_wrong` and halves precision (`test_eval_counts_null_param_false_positive_as_pointer_wrong`).
9. ✅ Zero predictions report `NaN` precision (`n/a`), never a fake `1.0` (`test_eval_reports_nan_precision_when_nothing_predicted`).
10. ✅ E3 flags Stratum-F aligner drift with an actionable example (`test_eval_e3_tripwire_flags_stratum_f_aligner_drift`, `test_e3_aligner_violation_accepts_clean_and_unresolved_cases`).
11. ✅ A missing split raises `FileNotFoundError` instead of returning `{}` (`test_eval_raises_for_missing_split`).
12. ✅ Renamed/added metric keys are asserted (`resolver_hit_rate_F` present, `hit_rate_F` absent, `truncation` shape).

**Still to do (data level, needs a real run):**

13. Re-run the Lean tensorize cell on `hermes_train`/`hermes_dev`; confirm all three tripwires print `0`: `match_missed total = 0`, `aligner/null-slot defects total = 0`, `tensorize exceptions total = 0`. Anything larger is a pipeline bug to fix **before** trusting extraction metrics.
14. Re-run Eval Part 1 on the same checkpoint; report `extraction_recall` vs `recall_all` and the implied `fast_path_param_coverage` / `fast_path_call_coverage` — those two numbers are the architecture-value argument for the fast path.
15. Confirm `e3_aligner_drift == 0` on `hermes_eval`; a non-zero value means the aligner, not the model, needs attention.
16. Confirm `exact_payload_match (Stratum F)` is comparable to the previous notebook build. Note the E-fixes are reporting-only, **except** that explicitly-null GT params are now included in the GT universe, so `exact_payload_match_F` can move in either direction for calls containing null args — that movement is the bug fix, not a regression.
