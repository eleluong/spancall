# Plan: `notebooks/spancall_experiment_verify.ipynb` — small-data verification experiments

**Goal:** verify the four-axis diagnosis in [`docs/param_recall_diagnosis.md`](../docs/param_recall_diagnosis.md) using a small slice of data, so each assumption gets a cheap, decisive yes/no.

## Hypotheses to verify

| ID | Hypothesis (from diagnosis) | Experiment | Expected signal if TRUE |
|---|---|---|---|
| H-D2 | Unmatched GT positives are silently labeled abstain (Killer #7) | Label audit on tensorized batches | >10% of GT params land on abstain col 0 despite existing candidates |
| H-cand | GT values are not in the regex candidate set | Oracle-candidate ceiling (C3) | payload_equal stays low even with oracle picking |
| H-T1 | H3 trained T2-only on abstain-majority data converges to abstain | Pointer-prob histogram on dev | abstain_p ≈ 1, top-1 span_p flat/low |
| H-T2 | Positive-heavy mini-train fixes emission | Train on F-only subset, abstain_weight=0.5, few epochs | Raw-mode emission rate and F recall rise materially |
| H-E2 | Stratum A inflates the recall denominator | Stratum-F-only recall + emission-rate metrics | F-only recall ≠ headline recall (≈3× gap) |

## Notebook structure (cells)

1. **Setup** — same pattern as v2 notebook: clone/pull repo, editable install, device detect. Add `EXPERIMENT_*` flags at top: `TRAIN_LIMIT=800`, `DEV_LIMIT=200`, `EPOCHS=2`, `ABSTAIN_WEIGHT=0.5`, `STAGES="T2"`, `SUBSET="F_heavy"` (F rows + minimal A rows, e.g. 3:1 F:A), `EVAL_LIMIT=200`, `SEED=13`.
2. **Data prep** — pull canonical splits (HF hub or local `data/processed`), tensorize; then **build the experiment subset**:
   - filter `train.jsonl` to F-he stratum-heavy mix (keep all F rows up to TRAIN_LIMIT, sample A rows at ~3:1 ratio),
   - write to `data/processed_exp/train.jsonl` + `dev.jsonl`,
   - print stratum composition + `% rows with ≥1 positive param` (tests D1/D4 directly).
3. **Experiment A — Label audit (H-D2):** iterate the subset through the tensorizer path; for each GT param, record whether it matched ≥1 candidate column vs. defaulted to abstain col 0. Report: `unmatched_rate`, `positive_rows_pct`, breakdown by param type (string/number/date/array). Reuse logic from [`scripts/diagnose_ceiling_and_audit.py`](../scripts/diagnose_ceiling_and_audit.py) C1.
4. **Experiment B — Oracle-candidate ceiling (H-cand):** run C3 on the dev subset: force per-param selection to the GT-matching candidate (skip H3), cast, and check `payload_equal`. Report ceiling % and failure reasons (`unmatched_candidate:<param>`, `payload_unequal`). Reuse C3 from the same script (import or inline).
5. **Experiment C — Pointer-prob histogram (H-T1, pre-train):** load the *existing* v2 checkpoint (or skip if absent), resolve dev subset in raw mode, collect `per_param_abstain_p` and top-1 span_p per param, plot histograms + report mean abstain_p, emission rate (non-empty `payload_partial`), anchor-audit (any row using token-0 fallback). This re-quantifies the 197/200 abstain signature on the small slice.
6. **Experiment D — Mini-train (H-T2):** construct `TrainingConfig` with experiment overrides (small batch/accum, `abstain_weight=0.5`, 2 epochs, T2 only), train `SpanCallTrainer` on the F-heavy subset; log per-epoch pointer loss + abstain-vs-positive loss split (telemetry pattern from v2 notebook §train_log).
7. **Experiment E — Post-train eval:** rerun Experiment C with the new checkpoint; also run the offline `ToolCallEvaluator` over the dev subset in raw mode; report **emission rate**, **stratum-F-only param recall/precision/F1**, hit_rate, error taxonomy — side-by-side table pre vs post.
8. **Verdict cell:** automatically maps results to hypotheses: for each H-ID print CONFIRMED / REFUTED / INCONCLUSIVE with the measured number, and a one-line implication.

## Key implementation notes

- Train path: mirror [`scripts/train.py`](../scripts/train.py) `build_encoder` + `SpanCallModel` + `SpanCallTrainer` (marker tokens must be added to the encoder).
- Eval path: `SpanCallResolver(raw_mode=True, model_path=...)` + `ToolCallEvaluator`; stratum-F-only metrics come from `metrics_by_stratum["F"]` (already in [`evaluator.py`](../spancall/eval/toolcall/evaluator.py)).
- Keep every experiment independently skippable via flags so the notebook can run partially on CPU/Colab GPU.
- Output artifacts to `results/experiment_verify/` (audit JSON, histograms PNG, pre/post metrics JSON).

## Todos for Code mode

- [ ] Write `scripts/build_experiment_notebook.py` (generator, nbformat, cells 1–8 above)
- [ ] Generate `notebooks/spancall_experiment_verify.ipynb`
- [ ] Validate JSON + cell count, spot-check cell sources compile (py_compile on concatenated code)
- [ ] Update `docs/param_recall_diagnosis.md` §7 to point at the experiment notebook
