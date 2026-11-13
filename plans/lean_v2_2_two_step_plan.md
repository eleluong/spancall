# Lean v2.2 — BinaryToolHead Step-1 + Two-Step Inference Plan

Decision context: [`docs/v2_run_13-11-26_analysis.md`](v2_run_13-11-26_analysis.md), [`plans/binary_tool_selection_plan.md`](binary_tool_selection_plan.md), [`plans/lean_v2_two_step_probe.md`](lean_v2_two_step_probe.md), [`plans/lean_v2_candidate_parity_fix.md`](lean_v2_candidate_parity_fix.md).

## 0. What v2.2 is

v2.1 shipped the BinaryToolHead wiring (head, BCE loss, trainer flag, resolver sigmoid mode) but never ran the ablation or calibration. v2.2 **completes** v2.1 (run the two-arm ablation + τ calibration) and turns the user's decomposition into the serving architecture:

- **Step-1 — tool selection**: `BinaryToolHead` sigmoid scores per candidate anchor, calibrated threshold τ decides {one winner, abstain}.
- **Step-2 — param extraction**: re-render the context with **only the winner tool** (T=1), fresh encode, pointer decode of that tool's params.

Decomposition into the two optimization problems the user defined:
1. Raise tool-name accuracy (step-1): 0.667 → target ≥ 0.95.
2. Extract the right params of the chosen tool (step-2): gated by candidate parity (0.29) and pointer collapse — those are generation-side defects, owned by [`lean_v2_candidate_parity_fix.md`](lean_v2_candidate_parity_fix.md), executed as a prerequisite here.

Latency is an accepted non-goal (user directive).

## 1. What v2.2 does NOT claim (falsifiability)

- It does **not** claim the T=1 re-render improves param accuracy by itself. Run 13-11-26 shows collapse + parity failures are generation-side; the re-render is a layout change with train/serve shift risk. The re-render arm must **beat the single-pass decode** on the same checkpoint, or it is dropped.
- It does **not** fix the eval-split composition problem (A-dominant synthetics, dev without A rows). That is W0 instrumentation + bench stratification, done first so v2.2 numbers are measurable at all.
- BinaryHead is not assumed better than softmax CE: the v2.1 kill criterion applies (binary dies if `tool_binary_acc <= tool_acc` AND τ calibration does not improve the decision margin).

## 2. Work items

### W0 — Instrumentation + bench hygiene (prerequisite, zero-training)
1. Thread tool-qualified parity spans (`f"{gt_call.tool_name}::{param_name}"`) through `predict_split` → [`try_resolve_candidates`](../spancall/lean/resolver.py:673) so e2e runs report `parity_checked`/`parity_hit` without colliding on shared parameter names when H1 picks the wrong tool (wrong-tool rows record under `parity_wrong_tool` / `parity_undetermined`).
2. **False-hit audit**: flag rows where ≥ 2 distinct decoded parameters with `source_type == "utterance"` and non-null `char_span` collapse to one identical `(char_start, char_end)` span (the "USA"/"2023" signature; ignoring legitimate shared `<|null|>`, boolean, or enum slots) — report `collapse_hit_rate` separately from true hits.
3. Measure `abstain_acc` / `false_hit_rate_on_a` via `LeanResolver` + `ToolCallEvaluator` on a stratified F+A dev/eval split (and mask zero-positive rows from `exact_call` in [`trainer.py`](../spancall/lean/trainer.py) so unsupervised Stratum-A rows cannot artificially inflate `exact_call`).
4. Report e2e metrics stratified F-only vs A.
**Exit**: re-run 13-11-26 config; scoreboard now separates true hits / collapse hits / correct abstains / parity misses.

### W1 — Step-1: run the v2.1 ablation + calibrate τ
1. Flip `RUN_BINARY_TOOL_ABLATION = True` in the v2 notebook (two-arm `ce` vs `binary` retrain, kill criterion already printed by the cell).
2. **τ calibration**: sweep thresholds on dev binary scores; pick τ by target step-1 abstain budget (e.g. ≤ 2% false-abstain on F rows). Store τ with the checkpoint config.
3. Resolver binary mode: `try_resolve_candidates` uses sigmoid `P(called)` — winner = argmax above τ, else abstain with the existing `"tool_below_threshold"` reason. Note: `len(candidate_schemas) == 1` (`T=1`) remains short-circuited (`tool_prob=1.0`), so Stratum-A `T=1` rows rely on Step-2 abstention.
4. Diagnose the tool_acc 0.667 regression vs v1's 1.0: per-tool confusion on dev (pooled layout, `mean_n_candidates` 2.36).
**Success**: step-1 `tool_acc ≥ 0.95`, step-1 abstain rate ≤ 2% on F rows. **Kill**: v2.1 criterion; if killed, keep softmax CE for step-1 and proceed with W2 anyway.

### W2 — Step-2 prerequisite: parity fix (per [`lean_v2_candidate_parity_fix.md`](lean_v2_candidate_parity_fix.md))
1. Protected re-run (`protect_parity_spans` in e2e path, not just Part 1).
2. Miss-reason histogram (`parity_miss_reasons`: utt_rel addressing / width cap / P4 filter / packing priority) → fix the dominant class.
3. Pointer-collapse guard: distinct-span constraint (or demote/abstain) scoped strictly to `source_type == "utterance"` parameters with non-null `char_span` where ≥ 2 distinct params resolve to the identical `(char_start, char_end)` span.
**Success**: e2e parity ≥ 0.95 without protection on F rows; `collapse_hit_rate < 1%`.

### W3 — Step-2 training: T=1 curriculum
Current training pools T≥2 on native multi-tool rows because `ToolPool.candidates_for` treats `pack_tools` as a floor and retains all native tools (`T = max(row_pack_tools, len(native_tools))`), so `jitter_pack_tools` does not produce T=1 for native multi-tool rows; step-2 serves a T=1 render. Close the shift:
- **3a (preferred)**: keep pooled training (single checkpoint serves both paths) and validate zero-shot transfer to T=1 on native multi-tool rows.
- **3b (fallback)**: if 3a validation shows a gap, add a fine-tune stage on T=1-only renders (e.g. `force_single_tool=True` in `tensorize_sample_lean`).
**Success**: single-pass decode (multi-tool render) and 2-step decode (T=1 re-render) from the same checkpoint within 2 pts of each other on F-row exact payload match.

### W4 — Resolver two-step path
1. New mode in [`LeanResolver`](../spancall/lean/resolver.py:267): `resolve_mode = "single_pass" | "two_step"`.
   - `two_step`: H1/binary winner (+ τ gate) → re-render bridge with `context_schemas=[winner]` → fresh `_encode` → pointer decode of winner params only.
   - Preserve Step-1 telemetry (`step1_pre_ms`, `step1_enc_ms`, `step1_render_T`) and `step2_render_T` (`== 1`), summing `pre_ms` and `enc_ms` across both passes instead of overwriting Step-2's clocks in `res.features.update(features)`.
2. Keep `single_pass` as the ablation arm (same checkpoint, same eval rows).
3. W2 margin gate and W2.3 collapse guard apply identically in both arms.
**Success**: `two_step` beats `single_pass` on F-row exact-payload-match by a margin the W0 audit can actually measure (i.e. after collapse hits are excluded).

### W5 — Notebook scoreboard + decision gate
One cell comparing on the same eval split, stratified F/A:

| Metric | single_pass CE | single_pass binary | two_step binary |
|---|---|---|---|
| step1 tool_acc / abstain rate | — | — | — |
| exact payload match (F, collapse-excluded) | — | — | — |
| parity_hit (e2e) | — | — | — |
| collapse_hit_rate | — | — | — |
| correct abstain (A rows) | — | — | — |

Decision rules:
- Binary head survives only if it wins ≥ one of {tool_acc, step-1 abstain calibration} without losing the other.
- Two-step path survives only if it beats single_pass on F exact payload match.
- If both die: keep v2.1 softmax single-pass, funnel effort into W2 parity/candidate generation.

## 3. Execution order

W0 → W1 → W2 → W3 (validate 3a first) → W4 → W5. W1 and W2 are independent and can interleave; W4 depends on both.

## 4. Files touched (expected)

- [`spancall/lean/resolver.py`](../spancall/lean/resolver.py) — parity in e2e, `resolve_mode`, collapse guard, τ gate.
- [`spancall/lean/eval_p1.py`](../spancall/lean/eval_p1.py) — parity threading, `tool_source="predicted"`.
- [`scripts/build_lean_v2_2_colab_notebook.py`](../scripts/build_lean_v2_2_colab_notebook.py) / [`notebooks/spancall_lean_v2_2_train_eval.ipynb`](../notebooks/spancall_lean_v2_2_train_eval.ipynb) — separate v2.2 notebook (`v2` kept untouched): ablation flag ON, dev τ calibration cell, 3-arm runner + scoreboard cell.
- [`spancall/lean/trainer.py`](../spancall/lean/trainer.py) — A-row dev slice hook (if needed).
- `tests/` — collapse guard, τ gate, two-step resolver mode unit tests.
