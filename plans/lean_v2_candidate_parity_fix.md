# Plan: Lean v2 Candidate-Parity Fix (T2) + Protected Re-run

## Evidence (from the latest v2 eval run)

- `notebooks/spancall_lean_v2_train_eval.ipynb` Part 1 (hermes_eval N=500):
  `candidate_parity = 0.2903 (216 / 744)`, `candidate_miss_rate = 0.7097`.
- `extraction_recall = 0.6522` vs trainer-side `exact_call = 0.569` /
  `span_acc = 0.876` — the two-evaluators gap is dominated by parity.
- Addressing split: `rel=649` (render-invariant) | `fallback=95` (absolute,
  skew-exposed) | `unresolvable=0`.
- `cands_per_param` hist: many params sit at the full 512 cap, **yet parity is
  still 0.29** → a large share of missed GT spans are not evicted by
  truncation; they are **never generated** as candidates (width cap
  `max_span_width`, R8 entry-snapping, P4 region/type pre-filter, or
  tokenization snapping).
- Existing hooks (already implemented, unused in the v2 run):
  - `protect_parity_spans` in
    [resolver.py:797-832](../spancall/lean/resolver.py:797) forces GT spans
    past `max_cands` truncation only — NOT past generation.
  - `pack_protected` / `_pack` priority tier 2 in
    [masks.py:575-642](../spancall/model/masks.py:575).
  - `eval_p1.eval_param_extraction_lean(..., protect_parity_spans=True)`.

## Goal

Raise served-candidate parity toward ~1.0 on extractable GT spans so that
`exact_payload_match_F` / `call_exact_set_match` measure **model** quality,
not serve-side candidate-set holes. Success gate: after the fix,
`candidate_parity >= 0.95` with `protect_parity_spans=False` (production
behavior), and `extraction_recall` / `exact_payload_match_F` improve by the
parity-limited headroom.

## Workstreams

### W1. Protected re-run first (zero code change) — measure the headroom
1. Re-run Part 1 with `protect_parity_spans=True` (diagnosis arm already
   exists). Two possible outcomes:
   - parity goes to ~1.0 → truncation IS the cause → go W3.
   - parity stays < 0.99 → the residual is generation-side (region/type
     filter, width cap, entry-snapping, or fallback addressing) → go W2.
2. Record both arms (protection on/off) in the notebook comparison table:
   `candidate_parity`, `extraction_recall`, `extraction_precision`,
   `exact_payload_match_F`, `pointer_wrong` — this isolates how much of the
   703 `pointer_wrong` / low precision is actually un-offerable GT.

### W2. Missed-span decomposition (new instrumentation, small)
1. In `_pack` / `build_param_masks` add a debug counter per param:
   `miss_reason ∈ {never_generated_width, never_generated_typefilter,
   never_generated_entrysnap, truncated, fallback_address_unresolved}` for
   each parity span NOT in the final candidate set.
2. Expose it via `LeanResolveResult.features` (e.g.
   `parity_miss_reasons`) and print the histogram in `eval_p1` output — same
   style as the existing refusal split.
3. Unit test: build synthetic cases where the GT span exceeds
   `max_span_width`, fails the P4 type filter, or tokenizes to a
   non-entry-snapped boundary; assert the miss reason is attributed.

### W3. Fix the dominant miss class (code change, scoped to evidence)
Ordered by expected dominance; implement only what W2 shows:
1. **Fallback addressing (95 spans)**: the eval split lacks `utt_rel_*`
   pairs so parity is a LOWER BOUND. Emit `utt_rel_*` anchors in the v2
   re-alignment output (extend
   [scripts/realign_v2.py](../scripts/realign_v2.py) / aligner emit) so
   every GT span addresses render-invariantly.
2. **Width cap misses**: if GT spans exceed `max_span_width`, either raise
   the cap for eval or add a "protected-width" exception — decide from the
   W2 histogram (a data-derived width distribution, not a guess).
3. **P4 type/region filter misses**: relax per-param candidate admission for
   the filtered class OR surface the filter as an explicit abstain reason —
   never silently drop a GT-shaped span.
4. **Truncation (if W1 shows it)**: change the serve-time default
   `candidate_pack_fill` handling so supervised-positive-shaped spans (from
   the legend, e.g. quoted values, enum slots) are always in the structured
   priority tier — mirroring the trainer's retention rule.

### W4. Production default + regression tests
1. Whatever fix lands, the *production* resolver must reach parity >= 0.95
   WITHOUT `protect_parity_spans` (protection stays a diagnostic/eval-only
   crutch; it cannot ship because serve time has no GT).
2. Regression tests:
   - masks test: GT-shaped span from the legend survives packing under
     `stratified` fill at a small cap.
   - resolver test: end-to-end resolve on a fixture whose GT value sits
     late in the utterance (the legacy `head` failure) and past the width
     boundary.
3. Re-run the v2 notebook Part 1 → update the v1-vs-v2 comparison table;
   re-read the W5 decision gate in
   [lean_v2_flow_plan.md](lean_v2_flow_plan.md): if after parity ≈ 1.0 the
   residual failure is dominated by `pointer_wrong`, tool selection
   (binary head, v2.1) is next.

## Execution order

1. W1 (protected re-run — notebook-only, no library change).
2. W2 (miss decomposition — small library + tests).
3. W3 items strictly in the order the W2 histogram ranks them.
4. W4 (production default + regression tests + re-run + comparison table).

## Files touched (expected)

- `spancall/lean/eval_p1.py` — protected-arm table, miss-reason printout.
- `spancall/lean/resolver.py` — `parity_miss_reasons` telemetry passthrough.
- `spancall/model/masks.py` — miss-reason attribution, packing fix (W3.4).
- `scripts/realign_v2.py` / `spancall/data/align.py` — `utt_rel_*` anchors (W3.1).
- `tests/test_tensorize_pool.py` / new `tests/lean/test_parity_fix.py`.

## Non-goals

- No head redesign, no calibration/gate work (separate workstream, W2 of the
  lean v2 plan), no training changes — this plan is serve-side only.
