# Plan: Lean v2 Train Flow + Notebook v2 (separate from v1)

## Decision context

- Chosen option: **E+B** — S1 label re-alignment first, plus a calibrated
  softmax margin gate as a zero-training stopgap; revisit head design only
  after clean-label eval.
- User direction: build a **Lean v2 train flow** and a **separate notebook
  v2**. V1 (`notebooks/spancall_lean_train_eval_test.ipynb`) stays frozen as
  the baseline/evidence artifact; v2 is the fix artifact.

## What v1 vs v2 mean

| | Lean v1 (frozen) | Lean v2 |
|---|---|---|
| Labels | Nov-2 processed data, 100% stale offsets | Re-aligned against current template |
| Aligner grammar | `build_input` (no `<|none|>`, -9 bias) | `build_input_multi` parity (S1b root fix) |
| Tool gate | `call_min_prob` uncalibrated | Calibrated margin/prob gate (B stopgap) |
| Template safety | none | Template-hash stamp (S6) |
| Head design | softmax H1 | unchanged in v2.0; revisit after clean-label eval |

## Workstreams

### W1. Label re-alignment (S1) — the prerequisite
1. Fix `ReverseAligner` grammar parity: `_render`
   ([align.py:544-559](spancall/data/align.py:544)) renders with
   `build_input_multi` (or `build_input` gains a `<|none|>`-parity flag) so
   aligner/tensorize/resolver share ONE grammar — kills the +9 permanently.
2. Make `align()` idempotent: `_emit` ([align.py:775-790](spancall/data/align.py:775))
   replaces instead of `setdefault(...).extend(...)`.
3. Re-run alignment over train/dev/eval processed splits with
   `target_spans` cleared; write to a NEW data dir (e.g.
   `data/processed_v2/`) — v1 data untouched for comparison.
4. Verify with `audit_gt_offsets`: **acceptance = span_stale_rate < 0.02**
   (was 1.0). Non-negotiable gate before any v2 training.

### W2. Zero-training margin gate (B stopgap)
1. Dev-split sweep over the CURRENT v1 checkpoint: winner softmax prob and
   top-2 margin vs tool correctness → PR curve, F1-optimal operating point.
   (The calibration finding — correct 0.489 vs wrong 0.039 — predicts this
   works.)
2. Wire chosen (prob, margin) into `LeanResolver.call_min_prob` /
   `call_min_prob_ratio` ([resolver.py:274-275](spancall/lean/resolver.py:274))
   as v2 defaults, persisted in the v2 checkpoint config.
3. Report v1-gate vs v2-gate abstain/hit/precision deltas in the v2 notebook
   on BOTH label sets (stale + re-aligned) — this isolates the gate's
   contribution from the label fix's.

### W3. Template-hash stamp (S6)
1. Hash `span_template/*.jinja` + `template.py`; stamp each processed dataset
   with it.
2. Train/eval entry points refuse (or loudly banner) on hash mismatch — the
   Nov-2-data vs Nov-10-template skew that caused the +30 drift can never
   recur silently.

### W4. Lean v2 train flow + notebook v2
1. **New script `scripts/build_lean_v2_colab_notebook.py`** — copy-then-evolve
   from [`scripts/build_lean_colab_notebook.py`](scripts/build_lean_colab_notebook.py):
   - DATA_DIR → `data/processed_v2/`; RESULTS_DIR → `results_lean_v2/`.
   - Pre-flight cell: template-hash check + `audit_gt_offsets` acceptance
     gate (training proceeds only if stale rate < 0.02).
   - Calibrated-gate cell (W2) with the sweep + chosen operating point.
   - Unchanged heads (softmax H1) — v2.0 measures the label fix + gate, and
     ONLY that; head redesign is deferred by explicit decision.
   - Post-eval: v1-vs-v2 comparison table (same checkpoint re-evaled on both
     label sets; then v2-trained checkpoint) using the existing
     `analyze_predictions` decomposition.
2. **Output `notebooks/spancall_lean_v2_train_eval.ipynb`** — v1 notebook
   untouched. Same build-time cell guard, same nbformat validation.
3. v1 artifacts (`results/lean/12-11-26/`, v1 notebook) referenced read-only
   for baseline comparison.

### W5. Decision gate after v2 eval
After clean-label eval of the re-aligned data + calibrated gate:
- If tool errors are now the dominant failure mode → implement the binary
  head plan ([plans/binary_tool_selection_plan.md](plans/binary_tool_selection_plan.md))
  as v2.1.
- If param precision is still dominated by pointer/offset issues → extend
  the offset audit to the v2 training loop's `pos_indices` (tensorize-side),
  since that was masked by the `canonical_value` fallback in v1.

## Execution order

1. W1.1–W1.2 (aligner parity + idempotency) + tests — code fix.
2. W1.3–W1.4 (re-align all splits, acceptance gate) — data regeneration.
3. W3 (hash stamp) — small, do before regenerating so v2 data is stamped.
4. W2 (margin gate calibration on current checkpoint) — zero training.
5. W4 (v2 notebook script + build) — the deliverable.
6. W5 — decision, not code.

## Files

- `spancall/data/align.py` (grammar parity + idempotency)
- `span_template/template.py` (optional parity flag)
- `scripts/prepare_data.py` or a new `scripts/realign_v2.py` (W1.3)
- `spancall/lean/resolver.py` (gate defaults), `scripts/calibrate.py` (sweep)
- NEW `scripts/build_lean_v2_colab_notebook.py`,
  NEW `notebooks/spancall_lean_v2_train_eval.ipynb`
- `tests/data/test_align.py` (idempotency + parity), hash-stamp tests
- v1 artifacts untouched: `notebooks/spancall_lean_train_eval_test.ipynb`,
  `results/lean/12-11-26/`
