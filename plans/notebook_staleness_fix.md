# Plan: Bring notebooks/spancall_train_eval_test.ipynb up to date

## Verdict (answered)

The checked-in notebook is a **generated artifact** of `scripts/build_colab_notebook.py` (OUT at line 9). It is **partially stale** relative to the shipped five-fold upgrade:

### Stale in the checked-in .ipynb
1. **L201 (§0b)** — `MAX_POS = 3` hardcoded; the builder now reads `optimization.max_pos` from `configs/default.yaml` (fallback 3) and default.yaml ships `max_pos: 1`.
2. **L1012-1016 (§3 tensorize)** — old pre-D2 pattern `if s.stratum in STRATUM_ABSTAIN or not pos_cols: pos_cols = [0]` (the exact label-corruption fix removed from the builder); no `MASK_F_UNMATCHED`. The builder now emits the fixed block at `scripts/build_colab_notebook.py:556-567`.
3. **L1024 (§3)** — `cue.append(1.0)` unconditional; the builder now refines cue per unmatched-F (colab builder L567).
4. **L1508 (§4 config)** — `cfg["optimization"]["abstain_weight"] = 1.0` clobbers the shipped H-T2 default of 0.5.
5. **resolver_eval hook disabled** — `resolver_eval_split` left at `""` from base yaml (wiring exists in `SpanCallTrainer.evaluate_resolver`).

### Already up to date (flow through from default.yaml base, §4 loads then overrides)
- `data.pack_tools: 3` (H1 distractor training) — multi-tool packing for t0/t1/t2/dev present in §2.
- `data.train_f_ratio: 0.7` (F-heavy sampling) — applied in `load_rows`.
- Config fingerprint includes `max_pos`.

### Root cause in the builder (one remaining defect)
`scripts/build_colab_notebook.py:817` still forces `abstain_weight = 1.0` in the §4 cell — so regenerating alone would NOT pick up the shipped 0.5. This line must be removed first.

## Execution steps (code mode)

1. **Edit `scripts/build_colab_notebook.py` §4 cell (lines ~816-822)**
   - Delete the `cfg["optimization"]["abstain_weight"] = 1.0` line; replace with a comment that the H-T2 recipe (0.5) flows through from `configs/default.yaml`.
   - Add `cfg["training"]["resolver_eval_split"] = "dev"` and `cfg["training"]["resolver_eval_limit"] = 40` to enable the e2e resolver eval hook per epoch.
   - Update the final print to mention `abstain_weight` from yaml if convenient.
2. **Regenerate the notebook**: `.venv/bin/python scripts/build_colab_notebook.py` (uses nbformat — valid JSON guaranteed).
3. **Verify the regenerated .ipynb** contains:
   - MAX_POS read from `optimization.max_pos` (§0b print includes MAX_POS from yaml).
   - The fixed pos_cols block: `if s.stratum in STRATUM_ABSTAIN: pos_cols=[0]` + `elif not pos_cols and MASK_F_UNMATCHED: pos_cols=[]` + opt-out fallback.
   - Refined cue append (not unconditional 1.0).
   - No `abstain_weight` override in §4 (search the .ipynb source for `abstain_weight`).
   - `resolver_eval_split: dev` present in §4.
4. **Re-run validation**: `pytest -q` (expect 396 passed) + py_compile of the builder script.

## Notes
- `scripts/build_experiment_notebook.py` (generates `notebooks/spancall_experiment_verify.ipynb`) is already fully up to date (D2 fix at L510-516, MASK_F_UNMATCHED, H-T2 knobs) — no changes needed there.
- The old/v1/v2 notebook copies are historical artifacts; leave untouched.
