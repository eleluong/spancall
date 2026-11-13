# Lean v2 Run 13-11-26 — Analysis, Comparisons & Verdicts

Sources: [`results/lean/13-11-26-v2/preds_eval.jsonl`](../results/lean/13-11-26-v2/preds_eval.jsonl), [`notebooks/spancall_lean_v2_train_eval.ipynb`](../notebooks/spancall_lean_v2_train_eval.ipynb), cross-referenced with [`results/lean/12-11-26-v2/preds_eval.jsonl`](../results/lean/12-11-26-v2/preds_eval.jsonl).

## 1. Run-over-run comparison

| Dimension | 12-11-26-v2 (Part 1 oracle) | 13-11-26-v2 (Part 3 e2e) |
|---|---|---|
| Tool source | oracle | predicted (H1 winner via [`try_resolve_candidates`](../spancall/lean/resolver.py:673)) |
| Parity instrumentation | active, measured 0.29 | **inactive** — `parity_checked: 0` in every row |
| W2 margin gate (`min_joint_margin=0.02`) | off | **active** — `low_joint_margin:0.0000<0.0200`, `0.0061<0.0200` demotions present |
| r7 adversarial rows | — | cleanly vetoed pre-encode: `sanitize: marker_injection` at ~0.07ms |
| Dominant miss mode | unofferable GT spans (parity) | mass `schema_invalid: missing_required` abstains + false hits |

Working as designed: sanitize veto, W2 margin floor, best-epoch restore (epoch 3, `exact_call` 0.569 restored).

## 2. Verdicts

**V1 — The headline hit rate is not trustworthy.** Stratum-F `swap_*` rows are marked "hit" with schema-valid but garbage payloads where **every param decodes to the identical span** (all = `"USA"`, all = `"Wisconsin', the postal code is '53004"`, all = `"2k"`, all = `"2023"`). This is **pointer collapse**: the pointer head locks onto one high-attention span and reuses it for all slots. Payload validation only checks schema validity, so these count as hits. W2 catches the zero-margin cases but only partially overlaps with the collapse set. Net: e2e hit rate is inflated; true param quality is lower than the scoreboard says.

**V2 — The eval split is a structural ceiling.** The run is dominated by stratum-A `json_output`/`ExpertQAExtractor` synthetics where GT is ungroundable from utterance text, so mass abstain is *correct behavior on a synthetic-dominant set*. Trainer dev has zero A rows → `abstain_acc: 0.0` is meaningless, and the 0.569 `exact_call` is measured on a different distribution than the e2e run. The two evaluators are not measuring the same problem. This is a third sub-problem beyond the 2-step decomposition.

**V3 — The parity diagnosis does not cover this run.** `parity_checked: 0` everywhere: `_parity_spans_for_call` runs only in the Part-1 oracle arm ([`eval_p1.py`](../spancall/lean/eval_p1.py)). The 0.29 parity number remains the best generation-side diagnosis, but there is no e2e parity number. Cheap fix: thread parity spans through `predict_split`.

**V4 — Step-1 (tool-name accuracy) is currently the weaker sub-problem.** `tool_acc` 0.667 vs v1's 1.0 is a regression; H1's job is harder with the pooled candidate layout (`mean_n_candidates: 2.36`). BinaryToolHead v2.1 is implemented but untested — its ablation cell sits disabled in the notebook. Cheapest path to the step-1 target (≥ 0.95).

**V5 — Step-2's bottleneck is generation-side, not layout.** Pointer collapse (V1) and the Part-1 parity 0.29 both point at candidate generation/scoring, not at multi-tool ambiguity. The single-tool re-render changes the layout the pointer sees, but there is no evidence the layout is the constraint — and it introduces a train/serve distribution shift (jitter trains T∈[1,3] pooled).

## 3. The 2-step idea vs this run

| Idea | Run evidence | Verdict |
|---|---|---|
| Split into tool-name + param extraction | tool_acc 0.667 isolated from collapse hits | ✅ Correct decomposition |
| Step-1 needs its own optimization | 0.667 regression; BinaryToolHead v2.1 built, ablation cell disabled | ✅ Do first — cheapest |
| Step-2 = single-tool re-render | Collapse + parity are generation-side | ⚠️ Re-render is not the fix; parity/candidate fix is |
| Latency fine, so 2 passes OK | Nothing in the run contradicts | ✅ Non-goal stands |
| (not covered) | False hits + A-dominant eval split | ➕ New: fix measurement first |

Note: the current serving path *already* decodes only the H1 winner's params — so 13-11-26-v2 is functionally a live 2-step measurement, minus the fresh re-render. The re-render's real delta is the T=1 layout shift, which is unproven and carries train/serve risk.

## 4. Recommended order

1. **Instrumentation / false-hit audit** (zero-training): flag repeated-span payloads in eval; thread parity through `predict_split`. Without this, every subsequent number is suspect.
2. **Step-1**: run the disabled binary-tool ablation cell + τ calibration in the v2 notebook → target tool_acc ≥ 0.95.
3. **Step-2**: execute [`lean_v2_candidate_parity_fix.md`](../plans/lean_v2_candidate_parity_fix.md) (protected re-run + miss-reason histogram) — attacks both parity 0.29 and collapse-into-garbage.
4. **Fix the bench**: add an A-row slice to trainer dev (makes `abstain_acc` meaningful); report e2e metrics stratified F-only.
5. **Only then** evaluate the re-render path from [`lean_v2_two_step_probe.md`](../plans/lean_v2_two_step_probe.md) as an *experiment*, not a prerequisite — it must earn its place by beating the single-pass decode after parity is fixed.

## 5. Bottom line

The pipeline machinery (gates, sanitize, restore) is solid, but this run's scorecard overstates quality: pointer collapse manufactures hits, the synthetic-dominant composition manufactures abstains, and parity is unmeasured on the e2e path. Fix measurement before optimizing either step of the 2-step flow.
