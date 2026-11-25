# Latency Fix Plan — honest per-sample time for model comparison

## Goal

One number that answers: **"how long does each model take to process one sample?"**
Comparable across SpanCall and the LLM baselines. No queue-wait contamination.

## What's wrong today

In [`spancall/lean/resolver.py`](../spancall/lean/resolver.py), `t0` is captured per
row at prep start ([line 2128](../spancall/lean/resolver.py:2128)), but all rows are
prepped before any batch forward runs. So `latency_ms = perf_counter() - t0` includes
the entire batch queue wait. Confirmed linear-in-N: 4.9x → 18x → 66.8x inflation at
n = 8/32/128 with constant per-row work. Real v2.3 files: 34.2% of reported latency
is queue wait, not compute. Every v3.2 latency number is inflated and the LLM-vs-SpanCall
latency comparison is invalid.

## The metric (definition)

**`latency_ms` = amortised per-sample cost** = own-row prep + mask + amortised share of
the batch forward (`batch_fwd_ms / batch_row_count`) + head scoring + decode + post.

The honest batched-GPU per-sample cost is the amortised forward share; individual rows
inside one padded forward are not separable, and this is exactly what a fair
model-vs-model comparison needs. Additionally the notebooks report **throughput**
(wall-clock rows/sec over the whole split) so both views exist.

## Implementation Steps

### Step 1 — `_run_encoded_batches` measures and distributes forward time
File: [`spancall/lean/resolver.py:1946-2008`](../spancall/lean/resolver.py:1946)

- Time each batch forward: `t_fwd_0` before `H_batch = backbone(...)`, `fwd_ms`
  after (inside the `torch.inference_mode()` block).
- Callback signature becomes `on_encoded(orig_idx, H_k, enc_ms)` where
  `enc_ms = fwd_ms / len(grp)`.
- Batching logic (length-sort, batch_size, token cap) unchanged.

### Step 2 — Thread `enc_ms` through the four call sites
File: [`spancall/lean/resolver.py`](../spancall/lean/resolver.py)

| Call site | Callback | Change |
|---|---|---|
| [`predict_tools_batch`](../spancall/lean/resolver.py:2114) | `_on_enc` (line 2089) | accept `enc_ms`, record it on the result |
| [`resolve_oracle_batch`](../spancall/lean/resolver.py:2249) | `_on_enc` (line 2232) | pass `enc_ms` to `_finalize_param_resolve` |
| step 1 of `resolve_candidates_batch` ([line 2395](../spancall/lean/resolver.py:2395)) | `_on_s1_enc` (line 2340) | use the passed `enc_ms`; the head-scoring duration currently mislabeled `enc_ms` becomes `heads_ms` |
| step 2 of `resolve_candidates_batch` ([line 2454](../spancall/lean/resolver.py:2454)) | `_on_s2_enc` (line 2437) | pass `enc_ms` to `_finalize_param_resolve` |

### Step 3 — `_finalize_param_resolve` recomputes latency as a stage sum
File: [`spancall/lean/resolver.py:1187-1600`](../spancall/lean/resolver.py:1187)

- New keyword param `enc_ms: float = 0.0` (the non-batched `_resolve` path keeps
  measuring its own encode and passes it).
- Features (lines 1535-1542): `enc_ms` = passed-in amortised value, all other stages
  unchanged.
- `latency_ms` = `pre_ms + mask_ms + enc_ms + heads_ms + decode_ms + post_ms`
  (additive by construction — this *strengthens* the R5 test at
  [`tests/test_lean_flow.py:2596`](../tests/test_lean_flow.py:2596)).

### Step 4 — Fix the two-step merge
File: [`resolver.py:2476`](../spancall/lean/resolver.py:2476)

Step-1 contributions become `pre_ms + enc_ms + heads_ms` (true amortised forward +
scoring), replacing the current merge that omits the forward and double-counts
mislabelled scoring.

### Step 5 — Regression tests (`tests/test_lean_flow.py`)

1. `test_latency_is_not_linear_in_n` — n=8 vs n=64, fixed-cost stub backbone,
   constant per-row prep; assert per-row `latency_ms` stays flat within tolerance.
   **Fails on current code.**
2. `test_latency_equals_stage_sum` — `abs(latency_ms - stage_sum) < 1e-6` per row.
3. `test_amortised_enc_ms_sums_to_batch_forward` — Σ`enc_ms` over a batch equals the
   batch forward time.
4. `test_step1_forward_is_attributed` — two-step `latency_ms` includes a positive
   step-1 `enc_ms`.
5. Existing `test_resolver_latency_is_additive_across_stages` keeps passing.

### Step 6 — v3.2 train_eval notebook (cell 19, `predict_split`)

- `latency_ms` now carries the corrected per-sample value (same key, no schema break).
- Add one throughput line to the eval report: `rows_per_sec = n_rows / wall_clock`
  over the whole split.
- Update the prose to define `latency_ms` as amortised per-sample cost.

### Step 7 — LLM baseline notebook
File: [`scripts/build_llm_baseline_notebook.py`](../scripts/build_llm_baseline_notebook.py)

- The comparison table (lines 1973-1993, `lat_ms` / `p90_ms`) automatically carries the
  corrected numbers once prediction files are regenerated; keys unchanged.
- Rewrite the false prose at lines 2157-2174 ("local decode time on a GPU") to state:
  amortised per-sample cost; throughput reported separately.
- Optionally add a `rows/sec` column next to `lat_ms` for the throughput view.

### Step 8 — Re-run inference, regenerate predictions

- Re-run v3.2 train_eval §4c on the existing 2,000-row eval subset → corrected
  `preds_*.jsonl`.
- Re-run the LLM baseline notebook's SpanCall scoring against the new files (the
  EVAL_LIMIT population contract guarantees matching populations).
- Old frozen files stay as historical record; `CANDIDATE_PRED_PATHS` points at the new
  files.

### Step 9 — Verification

- Full test suite passes.
- Re-run the row-count sweep (n = 8/32/128): per-row latency flat, not linear.
- On regenerated predictions: `latency_ms == stage_sum` for ~100% of rows.
- The LLM baseline table now shows a valid per-sample comparison; update the paper's
  latency numbers and prose.

## Execution Order

resolver fix → tests (fail-then-pass) → notebooks → re-run inference →
LLM baseline scoring → paper numbers.
