# Lean v3 — Two-Step-Only, Larger Stratified Eval, HF Push, LLM Parity, Agentic Notebooks, W&B Logging & Mid-Training Probe Eval

**Status:** Approved (with W&B + periodic probe-eval amendments)
**Base:** v2.7 pipeline (Hermes V1 + xLAM-60k 1-call subset, ModernBERT-large, `MAX_SEQ_LEN = 4096`), builder-derived from `scripts/build_lean_v2_7_colab_notebook.py`.

---

## 1. Scope (from user request)

1. **v3 train/eval notebook** — token extraction uses **only the best extraction mode: `two_step`** (`LeanResolver.resolve_mode="two_step"`); drop the `single_pass` runs from the notebook flow.
2. **Push model to Hugging Face** — a notebook cell (or appendix section) exporting the trained `LeanSpanCallModel` + resolver config and uploading via `huggingface_hub` (`HF_TOKEN` from env/Colab secrets; never hardcoded).
2b. **W&B experiment logging** — training cell initializes `wandb` (project/run name from config; `WANDB_API_KEY` from env/Colab secrets, offline `mode="disabled"` fallback so the notebook runs without a key); logs per-step train loss + per-epoch eval metrics, the probe-eval curves below, and a final metrics summary table.
2c. **Mid-training probe eval (replaces per-epoch full eval)** — instead of only testing each epoch:
   - a **small probe set** (e.g. `PROBE_N = 300` stratified rows, fixed seed, drawn from dev — **disjoint from the large evaluation set** and never reused as final-eval rows);
   - probe cadence **every `probe_every_steps = 1000` optimizer steps** (fallback `every 10% of total steps` if step-based is awkward in `LeanTrainer`'s loop);
   - probe metrics (exact_F / hit_F / param_F1 on the probe set) logged to W&B as training curves so convergence/overfit is visible mid-run;
   - per-epoch eval kept only for train/val loss (cheap), full eval-set scoring still happens once at the end.
3. **E2E LLM-parity comparison script** — `scripts/compare_llm_spancall.py`: same eval subset, same metrics, run (a) the lean v3 model and (b) a chat LLM (via `litellm` / HF Inference API) prompted for the same tool calls; emit a paired comparison table (exact_F, hit_F, param_F1, latency, cost) reusing `ComparativeEvaluator`-style output. Consistent prompts/renders for both systems.
4. **Larger evaluation set with per-corpus and tool-count-range breakdown**:
   - Increase `EVAL_LIMIT` / stratified slice size (e.g. 3–5× current).
   - Report metrics **separately for Hermes and xLAM** (per-corpus split cell — v2.6 already has a by-corpus rerun; promote it to a first-class section).
   - **Tool-count ranges**: bucket rows by `len(tools)` (e.g. `1–2`, `3–5`, `6–10`, `11+` or finer bins) and report per-bucket metrics so scale-of-distractor-pool behavior is visible.
5. **Agentic usage notebook** — `notebooks/spancall_lean_v3_agentic.ipynb`: loads the pushed HF model, wires `LeanResolver` into `ToolCallRouter` + `ReActAgentLoop` (`LLM_ONLY` vs `ROUTER` vs `SHADOW`), demonstrates the mock booking environment, and compares agent runs with/without SpanCall fast path.
6. **Git hygiene** — notebooks that involve an LLM API or pushing to HF must be **gitignored** (they require API keys).

---

## 2. Design decisions

### 2.1 Two-step-only

- `LeanResolver(resolve_mode="two_step")` at construction; all `predict_split(...)` calls drop the `resolve_mode` argument (default comes from the resolver).
- Remove `_p_single` / `preds_eval.jsonl` (single_pass) prediction runs and their metric rows; keep the resolver's internal `two_step` code path unchanged (it already exists and is validated by tests).
- Comparison table keeps only: `B1a (regex)`, `SpanCall Lean (two_step)`, and (new) `LLM baseline` from the parity script where available.

### 2.2 HF push (notebook cell)

- New cell after final metrics:
  1. `model.save_pretrained(RESULTS_DIR / "hf_export")` (+ tokenizer files, `resolve_config.json` with `resolve_mode`, gate thresholds, template stamp).
  2. `from huggingface_hub import HfApi; api.create_repo / upload_folder` with repo id from a config constant `HF_REPO_ID` and token from `os.environ["HF_TOKEN"]` or Colab `userdata.get("HF_TOKEN")`.
  3. A model card (README) auto-generated with headline metrics from the run.
- The cell is guarded: skipped with a clear message if `HF_TOKEN` is absent (so the notebook still runs CI smoke).

### 2.3 LLM-parity script (`scripts/compare_llm_spancall.py`)

- CLI: `--preds <lean preds.jsonl> --model <llm name> --api_key_env VAR --split <subset path> --limit N --out results/llm_parity.json`.
- For each eval row: render tools + utterance into a standard tool-calling chat prompt (reuse `docs/chat_template_gemma4.jinja` / the agent backends' prompt format), query the LLM, parse `tool_calls` JSON, score with the **same** `ToolCallEvaluator` metrics used for lean predictions → apples-to-apples exact_F / hit_F / param_F1.
- Records per-row latency, tokens in/out, and estimated cost (`CostModel`).
- Output: JSON + printed markdown table (`lean vs LLM` per corpus × tool-count bucket).
- No keys in code: only env var names; script refuses to run if the env var is unset.

### 2.4 Larger, stratified-by-corpus-and-tool-count eval

- `build_stratified_eval_subset` gains (or the notebook wraps it with) a secondary stratification: after cohort sampling, also record `n_tools` per row.
- Notebook §4 gains:
  - **§4.0b Eval-set composition table**: rows by corpus (Hermes / xLAM) × tool-count bucket.
  - Metrics cells loop over `(corpus, tool_bucket)` slices, writing `toolcall_metrics_{split}_{corpus}_{bucket}.json` and one consolidated pivot table (System × corpus × bucket → exact_F).
- Separate Hermes / xLAM cleanly at the source split level too (evaluate `hermes_eval` and `xlam_eval` slices independently, not only by re-filtering a blended slice).

### 2.5 Agentic notebook (`spancall_lean_v3_agentic.ipynb`) — three integration modes

The notebook compares exactly three modes:

1. **LLM-only** — the plain agent: every tool call comes from the LLM (existing `LLM_ONLY` router mode). Baseline.
2. **SpanCall + LLM fallback** — SpanCall tries first; when the resolver **abstains** or fails a whole-call gate (the "complex cases"), the call falls back to the LLM (existing `ROUTER` router mode). The speedup-with-safety option.
3. **Combined (hybrid fill)** — *new mode*: SpanCall predicts the call skeleton (tool + extractable params) **first**, and the LLM is invoked to **fill only the params SpanCall could not extract** (missing / low-margin / cast-failed leaves), in a single constrained completion — not a full fallback. This needs a new router variant (`IntegrationVariant.HYBRID_FILL` or similar) in `spancall/agent/router.py` that:
   - runs `LeanResolver`, keeps admitted tool calls and their high-confidence params;
   - for each predicted tool call, collects unresolved params (resolver abstain reasons / per-param failures exposed by `res.features`);
   - renders a **masked prompt** to the LLM listing only those params (with schemas) and asks for the missing values as JSON;
   - merges LLM-filled values into the SpanCall payload, re-runs `StrictTypeCaster`/`SchemaValidator`, and emits the merged call.

- Sections: install/env → load pushed HF model (`from_pretrained`, rebuild resolver) → mock booking tools → **mode 1/2/3 demos with side-by-side traces of which system produced each tool call and which params were filled by whom** → run the canonical + long-trajectory task suites through all three modes → per-task success / latency / token / cost table (mode × task) → optional live LLM backend section (guarded by env key).
- Reuses `scripts/eval_agent.py`'s task suite definitions (import or copy) so agent results stay comparable with offline numbers.
- Router changes land in `spancall/agent/router.py` + `spancall/agent/backends/` (pure code, unit-testable without keys), notebook only wires them.

### 2.6 Git hygiene

`.gitignore` additions (keys/API-related notebooks — pattern-based, not per-file whack-a-mole):

```gitignore
# Notebooks that require API keys (LLM calls, HF push, W&B) — not committed
notebooks/*_llm_*.ipynb
notebooks/*_hf_push*.ipynb
notebooks/*_agentic*.ipynb
```

- v3 main train/eval notebook stays **committed** (it runs fine without keys; HF push is a guarded optional cell).
- The LLM-parity and agentic notebooks are generated by builders (committed builders, ignored artifacts) so the *content* stays in git while the runnable, key-requiring notebooks don't.

---

## 3. Execution steps

1. **`.gitignore`** — add the notebook patterns above (plus `results/llm_parity/` if desired).
2. **`scripts/build_lean_v3_colab_notebook.py`** — derive from the v2.7 builder (same exec-and-patch pattern):
   - W&B init cell (key-guarded) + `wandb.log` hooks in the training loop;
   - probe-eval set construction (fixed-seed stratified dev slice, disjoint from final eval) + probe-every-1k-steps scoring during training;
   - remove single_pass prediction runs & table rows; resolver constructed with `two_step`;
   - larger `EVAL_LIMIT`; add corpus × tool-count bucket metrics section (§4.0b + pivot table);
   - add HF export + guarded push cell with model-card generation;
   - version-specific paths: `data/tensorized_lean_v3`, `results_lean_v3`.
3. **`scripts/compare_llm_spancall.py`** — new LLM-parity E2E script (env-key only, no hardcoded secrets), reusing `ToolCallEvaluator` and the same eval subsets produced by the notebook.
4. **`scripts/build_lean_v3_agentic_notebook.py`** — new builder generating `notebooks/spancall_lean_v3_agentic.ipynb` (HF model load, router modes, agent runs, guarded live-LLM section).
5. **Tests** — extend `tests/test_notebook_cell_guard.py` (or add `tests/test_notebook_v3_fixes.py`):
   - v3 notebook contains no `single_pass` prediction calls;
   - no hardcoded `HF_TOKEN` / `sk-` strings in any generated notebook or script;
   - tool-count bucket + per-corpus metric cells present;
   - agentic notebook loads model from HF id string, not local-only path.
6. **Run builders** — regenerate all three notebooks; confirm git status shows only the train/eval notebook as newly tracked.
7. **Docs** — update `PROJECT_STATE.md` + `docs/experiments.md` with v3 protocol entries (E-LLM parity, tool-count stratification).

## 4. Risks / open notes

- HF push cell must be a no-op without `HF_TOKEN` (CI + local smoke).
- LLM-parity script depends on an external API at run time — keep it out of unit tests; only its parsing/scoring functions get tested with fixtures.
- Larger eval slice increases notebook runtime; make slice size a top-of-notebook constant.
- Tool-count buckets must be defined off `len(sample.tools)` *before* any `SINGLE_CALL_ONLY` filtering so corpus × bucket tables stay well-defined.
