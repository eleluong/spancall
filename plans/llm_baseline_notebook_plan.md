# Plan: LLM Baseline Comparison Notebook (replaces B1a)

## Goal
Create `notebooks/spancall_lean_v3_2_llm_baseline.ipynb` that compares SpanCall Lean v3.2 against a
hosted LLM under the **exact same v3.2 evaluation setup** — same splits, same rows, same
order, same tool descriptions, same scorer. The B1a regex baseline in the paper will be
replaced by this LLM once the run is executed.

## User decisions (locked)
- **One LLM arm**, called through the **OpenAI SDK** (`openai` python package). The model is
  hosted by the user elsewhere; the notebook takes a configurable `base_url` +
  `model` name. **No local open-weights LLM, no litellm.**
- **Credentials are typed in at run time**, the same UX as the HF token in
  `spancall_lean_v3_2_train_eval.ipynb`: cell 0 prompts for HF token, `LLM_MODEL`,
  `LLM_BASE_URL` and `LLM_API_KEY` through plain `input()`. Resolution order is
  **env → repo-root `.env` → prompt**, so CI/nbconvert pre-sets them and never blocks on
  stdin; `input()` honours EOF, whereas `getpass` reads `/dev/tty` and cannot be driven
  from a pipe. Secrets print only as `<set>` and are never written to the file.
- SpanCall side: pull **code and trained checkpoint from HuggingFace**
  (`convoicon/spancall-lean-v3-2`), dataset from `convoicon/spancall` — the same bootstrap
  pattern the v3.2 notebook uses.

## What the notebook reuses (no re-implementation)
- [`scripts/compare_llm_spancall.py`](../scripts/compare_llm_spancall.py) already contains the
  entire fairness contract: same samples/order, same
  [`ToolCallEvaluator`](../spancall/eval/toolcall/evaluator.py) scoring, tool-schema
  conversion via `_tool_definitions_to_openai`, LLM response → prediction-row shaping
  (`_llm_pred_row`), token/latency capture, corpus × tool-bucket pivot, JSON + JSONL outputs.
- It uses litellm; the notebook either (a) calls the script with an OpenAI-compatible
  `api_base` (litellm speaks the OpenAI SDK protocol to any `api_base`), or (b) inlines a thin
  OpenAI-SDK caller that reuses the script's shaping/scoring functions. **Chosen: (b)** — a
  small OpenAI-SDK call cell that imports the shaping/scoring helpers from the script, since
  the user explicitly wants "openai sdk".
- SpanCall predictions: **not re-run.** The v3.2 train/eval notebook already wrote
  `results_lean_v3_2/preds_eval_twostep.jsonl` (hermes_eval) and
  `preds_xlam_eval_twostep.jsonl` (xlam_eval), and the paper's numbers came from exactly
  those files. The notebook **loads** them and validates that every subset `sample_id` is
  covered. Re-inferring would risk scoring a slightly different prediction file and moving
  the published SpanCall column without anyone noticing.
  - Checkpoint rebuild (`model.pt` + `resolve_config.json` + tokenizer from
    `convoicon/spancall-lean-v3-2`, `two_step`) is kept as a **lazy fallback**, invoked only
    when the v3.2 predictions are absent; it prints a warning that the numbers are new.
  - Therefore the notebook additionally runs only the LLM arm, which is the whole point.

## Eval protocol (identical to the paper's v3.2 numbers)
- Splits: `subset_hermes_eval_clean.jsonl` (1,659 rows) and `subset_xlam_eval_clean.jsonl`
  (2,000 rows) — the exact files behind the paper's headline table.
- A configurable `LLM_LIMIT` (default: full rows; a lower value like 300 for a smoke run)
  trims **all arms to the same rows** so nothing is compared on different populations.
- Stratification F / A_clean / A_adv, `exact_F`, `hit_F`, `param_F1`, `false_hit_on_A` are
  reported per split for both systems.

## Notebook structure (cells)
1. **Title / markdown** — purpose: LLM vs SpanCall v3.2, same-rows-same-scorer contract.
2. **Setup** — clone `github.com/eleluong/spancall`, `pip install -e .[model,data,agent,dev]`
   + `openai` + `huggingface_hub`; define `RESULTS_DIR = results_lean_v3_2_llm`,
   `DATA_DIR`.
3. **Data build** — **`data/processed_v3_2` cannot be downloaded.** The Hub repo
   `convoicon/spancall` hosts `data/processed/` (v1) and `data/processed_v2/` only; there is
   no `processed_v3_2`. Because v2.7 object decomposition changes the rendered param list, a
   prebuilt directory would reintroduce v2.6-shaped labels. So the cell reproduces the v3.2
   notebook's pipeline: pull v1 (Hermes) / v2 (xLAM) canonicals, then
   `scripts/realign_v2.py` into `data/processed_v3_2` (per-split, Hermes and xLAM
   independently), then `build_stratified_eval_subset(..., limit=None, seed=42)` →
   `subset_hermes_eval_clean.jsonl` / `subset_xlam_eval_clean.jsonl`. `limit=None` is
   mandatory: the signature default is 500 rows and would silently shrink the comparison.
4. **SpanCall arm — reuse first.** `hf_hub_download` checkpoint + `resolve_config.json` +
   tokenizer from `convoicon/spancall-lean-v3-2`, but **only lazily** (inside
   `load_resolver()`), because the normal path never calls it. The cell looks for the frozen
   v3.2 predictions and validates full `sample_id` coverage; inference is a warned fallback.

   **Known constraint:** `results_lean_v3_2/` is not in git and not on the Hub (the training
   notebook backs it up to Google Drive only). A fresh clone therefore needs
   `SPANCALL_V3_2_RESULTS_DIR` (or `V3_2_RESULTS_DIR`) pointed at the directory holding
   `preds_eval_twostep.jsonl` (hermes) and `preds_xlam_eval_twostep.jsonl` (xlam). When it is
   unset the cell says so and re-infers, and `llm_parity.json` records `spancall_source` so
   the distinction survives. `preds_eval_twostep.jsonl` is deliberately **not** accepted for
   xLAM — that file is the Hermes run, and scoring it against the xLAM subset would yield a
   plausible but meaningless number.
5. **LLM arm (OpenAI SDK)** — client with `base_url` + `api_key` from env
   (`LLM_API_KEY_ENV`, `LLM_BASE_URL_ENV`); per row: `_openai_tools(sample)` tool defs +
   `_messages_for(sample)` prompt; parse `tool_calls`, malformed JSON args kept as raw so the
   row scores as a wrong value; record latency + prompt/completion tokens.
6. **Score & compare** — import `score`, `_llm_pred_row`, `_lean_pred_row`, `_slices`,
   `_print_table`, `_print_pivot` from `scripts/compare_llm_spancall.py`; produce the
   per-split table `system × {exact_F, hit_F, param_F1, call_set, false_hit_on_A, lat_ms,
   tok_in, tok_out}` and the corpus × bucket pivot.
7. **Normal vs Complex partition for the LLM** — reuse
    `classify_sample_complexity` so the LLM table can be read against the paper's Normal/Complex
    table (the LLM's complex-argument performance is the paper's key comparison point).
8. **Persist** — `results_lean_v3_2_llm/llm_parity.json`, per-split LLM preds JSONL
   (re-scorable), and a summary block ready to drop into `05_results.tex`.

## Key fairness contract (stated in the notebook markdown)
- Both systems see the same utterance, same `recent_context`, same clock, same tool list.
- Same rows, same order, same scorer; abstain/hit semantics identical.
- No price table hardcoded; tokens reported, USD optional.

## Follow-up (after notebook runs)
- Replace the B1a column in `05_results.tex` headline + OOD tables with the LLM numbers,
  update `04_experimental_setup.tex` (baseline description), abstract/intro mentions of B1a.

## Defects found while building, and how they were fixed
1. **`data/processed_v3_2` is not on the Hub.** `convoicon/spancall` holds only
   `data/processed/` (v1) and `data/processed_v2/`. The first draft's
   `snapshot_download(local_dir=DATA_DIR)` would have produced an empty directory.
   Cell 1 now replicates the real pipeline: v1 → `scripts/realign_v2.py` → `processed_v3_2`
   (Hermes, `--gate_split hermes_eval`), then v2 → same for xLAM (`--gate_split xlam_eval`),
   with a completeness gate before the subsets are built.
2. **`results_lean_v3_2/` is in neither git nor the Hub** (Drive-only backup), so the
   reuse path could never fire in a fresh clone and would silently fall through to
   inference. Added the `SPANCALL_V3_2_RESULTS_DIR` knob; a miss now warns loudly and
   records `spancall_source` in `llm_parity.json`.
3. **`preds_eval_twostep.jsonl` must not be accepted for `xlam_eval`.** That file is the
   *Hermes* run; accepting it would score Hermes predictions against the xLAM subset. The
   reuse-path smoke test caught this and the candidate list now documents the exclusion.
4. **Secrets were env-only.** Changed to interactive-first with env/`.env` pre-set support,
   matching the v3.2 notebook's `_prompt()` pattern.

## TODO list
1. Create `notebooks/spancall_lean_v3_2_llm_baseline.ipynb` with the cell structure above.
2. Wire the OpenAI-SDK caller (`LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY`, prompted at
   cell 0, no litellm).
3. Wire SpanCall arm: reuse the frozen v3.2 predictions; lazy HF rebuild only as fallback.
4. Wire shared scoring via `scripts/compare_llm_spancall.py` helpers + Normal/Complex split.
5. Rebuild `data/processed_v3_2` via `realign_v2.py` (no such dir on the Hub).
6. Add the `SPANCALL_V3_2_RESULTS_DIR` knob (preds are not in git/Hub).
7. Smoke-test the notebook logic offline (credentials, reuse path, e2e scoring; no network).
8. (Later, after execution) swap B1a → LLM numbers in the paper sections and recompile.
