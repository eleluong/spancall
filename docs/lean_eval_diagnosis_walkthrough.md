# Lean Flow — Eval Diagnosis Walkthrough: Problems & Suggestions

> Source of evidence: the validation diagnostics added in `spancall/lean/diagnostics.py`, run
> against `results/lean/12-11-26/preds_eval.jsonl` and the eval split. The numbers below come
> from `run_diagnostics()` (also embedded as the **"5d. Why the Number Is Low"** cell in
> `notebooks/spancall_lean_train_eval_test.ipynb`).
>
> Per your instruction, **no fix has been applied yet** — this document is the
> "validation logs first" deliverable; every suggestion at the end is a decision point.

---

## 1. The two questions you asked

1. **Why do tool calls get a `partial` status?** (You expected only hit/abstain.)
2. **Why is parameter-extraction precision still low?**

Short answers:

1. `partial` is **not** a model head — it is retained **resolver post-processing**. After the
   Lean flow dropped the AbstainHead/CallAcceptHead/CorrectnessHead, the resolver
   (`spancall/lean/resolver.py`) still runs the P9 cast + P10 schema-validation pipeline.
   A call gets `status = "hit"` when the payload is schema-valid, `partial` when the payload
   is non-empty but fails schema validation (typically `missing_required: …`), and `abstain`
   when nothing was emitted. In the eval run: 257 abstain, 215 hit, **28 partial** (5.6%).
   It's a normal, benign category — but see §4.5 for the "empty-payload hit" inflation issue.

2. **The ground-truth char offsets are 100% stale.** The pointer head (H3,
   `LeanPointerHead`) was supervised on the wrong characters for every single
   `utterance_span` target. The model never had a chance. Full chain in §3.

---

## 2. What the new diagnostics measured

### 2.1 Offset audit (`audit_gt_offsets`)

Re-rendering every eval sample with the *resolver's* grammar
(`render_for_sample` → `build_input_multi`) and checking whether
`output.text[cs:ce] == canonical_value` for each stored GT target:

| Metric | Value |
|---|---|
| Samples read / rendered | 500 / 500 |
| Samples with span targets | 47 (of those with GT+tools) |
| `span_valid` | **0** |
| `stale_recoverable` | 285 |
| `stale_lost` | 5 |
| `span_stale_rate` | **1.0 (100%)** |
| `utterance_span_valid_rate` | **0.0** |

Drift histogram (stored offset − true offset, constant within a sample):

```
{39: 156, 31: 41, 35: 34, 33: 32, 29: 12, 43: 7}
```

Every drift is **positive and in a narrow band (+29 … +43)**, i.e. the stored offsets point
*past* the true value by a fixed per-sample amount. A constant, additive, per-sample drift
is the signature of a **prefix-length change** — the utterance text itself is intact.

### 2.2 Prediction decomposition (`analyze_predictions`)

500 prediction rows joined to the eval split:

| Bucket | n | Precision |
|---|---|---|
| **hit** | 215 | |
| **partial** | 28 | |
| **abstain** | 257 | |
| hit rows with **empty payload** | 25 | (inflates hit count) |

GT parameter decomposition (865 GT params):

- **Extractable** (value present in context: utterance/null/bool/enum): 369
- **Unextractable** (cast_fail/missing from args): 496 → correctly absent
- **Scored**: 332 → **correct 13, wrong 319, absent 37**
- **Extractable precision = 0.0392**

By predicted `source_type`:

| source_type | n | precision |
|---|---|---|
| utterance | 230 | **0.026** |
| null | 95 | **0.021** |
| bool | 5 | 0.800 |
| enum | 2 | 0.500 |
| cast_fail (absent) | 30 | — |
| missing (absent) | 7 | — |

By GT `target_type`: `utterance_span` 359 scored / 0.017; `boolean_ctx` 6 / 0.667;
`enum_ctx` 2 / 0.500; `null_ctx` 2 / 1.000.

**Where predicted spans land (region split):**

| Region | correct | wrong | share of wrong |
|---|---|---|---|
| in_utterance | 6 | 224 | 69.3% |
| before_utterance | 7 | 95 | 30.7% |

**30.7% of wrong spans point *before* the utterance** — into the legend/context region of
the render. This is the model pointing into the wrong *region* of the input, consistent
with offsets trained against a misaligned coordinate system.

**Confidence separation:**

- correct: mean 0.4885, median 0.5270
- wrong:   mean 0.0389, median 0.0210

The head's confidence cleanly separates correct from wrong — **the model knows when it is
right**. It is weak (0.49 mean on correct) but calibrated. This is actually good news and
matters for the suggestions (§5.4).

**Collapse signature:**

- 49 of 107 multi-value GT rows: model fills **all params with the same value**.
- 27 predictions equal the **first utterance token**.

Combined with the `null` rows all carrying the identical span `[237, 245]` (a schema-anchor
position, not a real value), this is the "pointer defaulted to a fixed anchor" failure
mode — what you'd expect when supervision targets are noise.

---

## 3. Root cause: why 100% of the offsets are stale

Three observations, then the chain.

**Observation A — drift is constant per sample, always positive, narrow band.**
Rules out fuzzy-matching / value-shape problems; points at prefix-length change.

**Observation B — timeline mismatch.** The processed data was built **Nov 2**;
`span_template/template.py` / `template_multi.jinja` were last changed **Nov 10–11**
(verified via file mtimes and `git log`). The stored offsets address a render
**that no longer exists**.

**Observation C — clean re-align is uniformly −9.** Re-running
`ReverseAligner.align()` with `target_spans` cleared (it is **not idempotent** — see
§4.6) produces offsets that are uniformly **9 characters before** the
`build_input_multi` render. And 9 is exactly the length of the `<|none|>` anchor line.

**The chain:**

1. `ReverseAligner._render` (`spancall/data/align.py:544`) renders with
   **`build_input`** — the single-tool template, which **omits the `<|none|>` anchor
   line by design** (documented in `span_template/template.py:298`'s docstring:
   *"no `<|none|>` anchor: that belongs only to the packed multi-tool form"*).
   → offsets are 9 chars *short* of the multi render.
2. `tensorize.py` and `LeanResolver` render with **`build_input_multi`**, which
   **includes** the `<|none|>` line. → +9 chars relative to the aligner's render.
3. Template drift after the data was built added another ~+30 chars
   (the bulk of the +29…+43 histogram: +30 template change, +9 `<|none|>`, ±small
   per-sample variation from legend content).

**Net: +39 measured = +30 (template version skew) + 9 (`<|none|>` grammar mismatch).**

So the pointer head's supervision targets pointed at the wrong characters for *every*
utterance-span param. Training "survived" only because `tensorize.py`'s span-match has a
`canonical_value` fallback (`rescued_by_fallback=2267`, `match_missed` tripwire fired 641
times) — i.e. the model was effectively trained on value-text matches wherever the raw
offset match failed, and on **wrong offsets** wherever it didn't fail loudly.

Corollary: `candidate_parity = 2/743` is not measuring candidate-set coverage — it is
measuring **label drift**. And `match_missed`/`rescued_by_fallback` were the tripwires
that correctly fired and were treated as noise.

---

## 4. Secondary findings (all measured, all validated)

### 4.1 The model points before the utterance 30.7% of the time
Even among "wrong" predictions the head is not random — it lands in the wrong *region*
(systematically before the utterance). Supervision under a shifted coordinate system
produces exactly this.

### 4.2 `null` slots are the worst extractable class (0.021)
All 95 `null` predictions carry the same span `[237, 245]` — a fixed anchor. The null
detection (an entry-legends classification) is degenerate rather than learned.

### 4.3 Collapse: 49/107 multi-value rows are all-identical
When the model must fill two params, it fills both with the same value (27 of them equal
the first utterance token). Under noisy supervision, the cheapest strategy is to point at
the single most salient token — and the loss didn't punish this because the labels
themselves were shifted.

### 4.4 Confidence is well-calibrated (0.49 vs 0.04)
The head discriminates its own correctness. Any thresholding/gating you build later can
lean on this signal today.

### 4.5 "Hits" are inflated by 25 empty-payload rows
25 of 215 hits have an empty payload. Depending on how tool-level metrics count these,
the hit rate overstates usable call generation.

### 4.6 `ReverseAligner.align()` is not idempotent
`_emit` (`spancall/data/align.py:775`) does
`sample.target_spans.setdefault(key, []).extend(targets)` — re-aligning an aligned
sample duplicates targets (12 → 24 → 36). This contaminated an early
"the aligner reproduces the stored offsets" conclusion and is a **regression trap** for
any re-alignment repair: you must clear `target_spans` first.

---

## 5. Suggestions (ranked; none applied yet — your call)

### S1. Regenerate the labels against the *current* template — the highest-leverage fix
Re-run data preparation (or just the alignment stage) so offsets address the
render the trainer and resolver actually use. Two sub-options:

- **S1a (preferred, minimal):** re-run `ReverseAligner` over the processed data with
  `sample.target_spans = {}` cleared first (mandatory, see §4.6), then re-tensorize and
  verify `span_stale_rate == 0` with `audit_gt_offsets` **before** any training. This
  fixes the −30 template-drift component.
- **S1b (root fix):** make the aligner render with **`build_input_multi`** (or give
  `build_input` a `<|none|>`-parity flag) so the aligner, `tensorize.py`, and
  `LeanResolver` share **one grammar**. This kills the +9 permanently instead of
  re-anchoring every time the templates drift apart again.

After S1: re-evaluate the *existing* checkpoint against the re-aligned labels before
retraining — some of the 0.039 may recover for free because eval was scoring against
stale labels too.

### S2. Re-evaluate before retraining
The current 0.0392 was measured against stale GT offsets. Once labels are regenerated,
re-run `analyze_predictions` first. If precision stays low, then retrain; if it jumps,
the model was better than the eval said (the `canonical_value` fallback in tensorize
partially taught it real value positions).

### S3. Fix `partial`/hit bookkeeping
- Decide whether empty-payload "hits" count as hits (25 rows).
- `partial` is fine to keep, but consider emitting `missing_required` details into the
  preds file so the 28 partials are diagnosable without re-running.

### S4. Exploit the calibration with a confidence gate
Correct mean 0.4885 vs wrong mean 0.0389 — a simple threshold (~0.2) on parameter
confidence would trade a little recall for a large precision gain on extractable
params, as an interim mitigation until retraining lands.

### S5. Make the tripwires loud
- `candidate_parity` should be compared against *re-derived* spans, not stored ones, or
  renamed to `label_drift_rate`.
- `match_missed` / `rescued_by_fallback` firing at 641/2267 should fail the data build
  (or at least print a `[!]` banner) — that's the earliest signal that would have caught
  this before training.

### S6. Guard against template/label skew structurally
The `data mtime (Nov 2) < template mtime (Nov 10)` skew is the systemic failure. Stamp
each processed dataset with a template hash and refuse to train/eval when the live
template hash differs from the recorded one. Cheap, permanent.

---

## 6. Suggested decision order

1. **S1a** (re-align with cleared `target_spans`, verify `span_stale_rate == 0`) —
   low risk, unblocks everything.
2. **S2** (re-evaluate current checkpoint against fresh labels) — tells you whether
   retraining is even needed.
3. **S1b + S6** (single grammar + template-hash stamp) — the durable fix, do before the
   next training run.
4. **S4** (confidence gate) — optional interim mitigation.
5. **S3 + S5** — bookkeeping hygiene.

---

## Appendix: where the numbers live

- Diagnostics module: `spancall/lean/diagnostics.py`
  - `audit_gt_offsets` — §2.1 numbers
  - `analyze_predictions` — §2.2 numbers
  - `run_diagnostics` — one-call entry point (used by the notebook cell)
- Tests: `tests/test_lean_diagnostics.py` (38 tests)
- Notebook cell: "5d. Why the Number Is Low — Validation Diagnostics" in
  `notebooks/spancall_lean_train_eval_test.ipynb`
- Eval artifacts: `results/lean/12-11-26/preds_eval.jsonl`
- Related docs: `docs/param_extraction_eval_coverage.md`,
  `docs/param_recall_diagnosis.md`, `docs/remediation_plan.md`
