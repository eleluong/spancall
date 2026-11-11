# Lean Workflow — Remaining Issues & Proposed Changes

Status of the issues open after the audit in
[`lean_workflow_mismatch_audit.md`](lean_workflow_mismatch_audit.md), updated
with the second-pass code verification and the **full-tool-list contract** audit.

Each entry states:
1. **What is actually wrong today** (with verified file/line references and measurements),
2. **Audit notes / corrections** from the second-pass code inspection, and
3. **Suggested code changes** ready to apply.

Severity key:
- **P0** = affects headline numbers today or violates the function-calling prompt contract.
- **P1** = latent defect, performance waste, or train/serve skew under non-default settings.
- **P2** = cosmetic / telemetry attribution / dead-code hygiene.

## Resolution status

`✅ resolved` means the change is in `main` **and** pinned by a test. `⏳ open`
means the issue still reproduces.

| Issue | What it was | Status | Guarding test |
|---|---|---|---|
| **R0.1** | `ToolPool` truncated native tools to `pack_tools` | ✅ resolved — `pack_tools` is a floor; `max_tools` is the only brake and it is counted loudly | `test_tool_pool_never_truncates_native_tools`, `test_tensorize_counts_truncation_loudly`, `test_truncation_and_jitter_are_counted_not_swallowed` |
| **R0.2** | Eval Part 1 rendered `[gt_tool]` only (`T=1`) | ✅ resolved — `try_resolve(context_schemas=s.tools)` | `test_resolver_renders_all_context_schemas_but_scores_one`, `test_resolver_always_renders_the_scored_tool` |
| **R0.3** | `n_candidates` was ambiguous between tool count and mask width | ✅ resolved — `render_T` / `n_tool_candidates` split | `test_resolver_reports_effective_recent_budget` |
| **R0.4** | Only `ground_truth_calls[0]` is supervised / decoded / scored | ✅ resolved — Phase J1–J6: call-aware `target_spans`, call axis in tensorize, multi-positive tool loss, `decode_calls`, `LeanResolveResult.calls`, set-level eval | `test_two_calls_keep_separate_param_targets_on_a_shared_anchor`, `test_two_calls_to_different_tools_get_one_anchor_each`, `test_call_axis_pads_with_pad_index_across_a_batch`, `test_compute_lean_tool_loss_accepts_a_set_of_tools`, `test_decode_calls_allows_two_calls_to_share_one_span`, `test_call_schemas_decode_one_call_per_oracle_tool`, `test_multi_call_set_match_scores_every_gt_call`, `test_eval_part1_scores_every_gt_call` |
| **R1** | Train renders `T=3`, eval renders `T=1` | ✅ resolved for coverage+warm-start; `T=1` is in-distribution via jitter | `test_jitter_pack_tools_*`, `test_tensorize_jitter_reaches_t1_and_reports_the_draw` |
| **R2** | Recent budget silently halved and unreported | ✅ resolved — `recent_budget_effective`/`_tokens`/`_clamped` in `features` | `test_resolver_reports_effective_recent_budget` |
| **R3** | Masks collided on shared parameter names | ✅ resolved — `f"{tool}::{param}"` keys | `test_candidate_masks_key_shared_param_names_by_tool` |
| **R4** | Per-parameter re-tokenization and `_RowCtx` rebuild | ✅ resolved — one `_RowCtx.for_row` + `tokenize_legend` per row | `test_candidate_masks_key_shared_param_names_by_tool` |
| **R5** | Inverted `t_mask`/`t_enc`; `latency_ms` omitted H1 | ✅ resolved — additive stage timings | `test_resolver_latency_is_additive_across_stages` |
| **R6** | `pack_multi_tools=False` trained on an unserved grammar | ✅ resolved — `build_input_multi` is the only path | `test_pack_multi_tools_flag_does_not_change_the_grammar` |
| **R7** | `max_pos=1` left multi-positive loss untrained | ✅ resolved — tested at `max_pos=3`; production default stays `1` | `test_tensorize_max_pos_populates_multiple_positive_slots`, `test_compute_lean_pointer_loss_weights_every_positive` |
| **R8** | Training skipped recent-context sanitization/budgeting | ✅ resolved — both paths call `select_recent_context` | `test_tensorize_sanitizes_recent_rows_like_inference` |
| **R9** | `T` missing from every reported metric | ✅ resolved — tensorize stats, Part 1 report, Part 3 forensics | `test_tensorizer_telemetry_counts_every_T`, `test_eval_part1_reports_render_T_telemetry`, `test_part3_forensics_surfaces_render_T_from_persisted_features` |
| **R10** | `build_input` dead in the Lean path | ✅ resolved — imports removed, documented on `build_input` | `test_pack_multi_tools_flag_does_not_change_the_grammar` |

All R0–R10 issues are now resolved and pinned by tests. Invariant 7
(multi-call completeness) is satisfied by Phase J1–J6. Invariant 3 (train/serve
`T` parity) is satisfied at the *distributional* level: `pack_tools_jitter`
draws `T` from `[pack_tools, max_tools]` during training so `T=1` is
in-distribution, while eval may still legitimately render more tools than the
floor. `render_T` is reported on train, Part 1 and Part 3 so any residual skew
is visible rather than assumed away.

**Current state:** `pytest tests/` — **493 passed**. The notebook regenerates to
49 cells and passes the cell-guard. The metrics-reporting contract lives in
[`docs/experiments.md`](experiments.md:28); the post-upgrade run log there is
intentionally empty until a real end-to-end run is recorded.

---

## P0 — Affects the numbers read today & prompt contract

### R0. Every function-calling sample must mention all of its candidate tools in the prompt

> **Status.** R0.1 / R0.2 / R0.3 are fixed (see the table above). **R0.4
> (multi-call) is now fixed too**, implemented as Phase J1–J6 in
> [`lean_workflow_upgrade_plan.md`](lean_workflow_upgrade_plan.md:341): every GT
> call is supervised via call-keyed `target_spans` and the tensorizer call axis,
> the tool loss accepts a set of GT tools, `decode_calls` allocates one call at a
> time, `LeanResolveResult.calls` exposes them all, and Part 1 / Part 3 score the
> predicted call **set** (`call_exact_set_match`) rather than only call 0.

**Requirement.** Every function-calling sample should include all of its candidate tools in the rendered prompt (`<|tool|>` blocks) so the model sees the complete tool inventory when selecting the tool (`H1`) and extracting parameters (`H3`).

**What the canonical splits actually contain (`data/processed/hermes_*.jsonl`).**
Across `hermes_train.jsonl` (24,071 rows) and `hermes_eval.jsonl` (3,426 rows), `len(sample.tools)` varies from `1` to `6`:

| Subset (`metadata.config`) | `len(sample.tools)` in `hermes_train` | `len(sample.tools)` in `hermes_eval` |
|---|---|---|
| `func_calling` | `1`: 1,181 · `2`: 939 · `3`: 1,429 · `4`: 550 · `5`: 123 · `6`: 1 | `1`: 247 · `2`: 113 · `3`: 158 · `4`: 65 · `5`: 14 · `6`: 1 |
| `func_calling_singleturn` | `1`: 1,183 · `2`: 990 · `3`: 1,441 · `4`: 504 · `5`: 95 | `1`: 250 · `2`: 99 · `3`: 169 · `4`: 73 · `5`: 13 |
| `glaive_func_calling` | `2`: 13,768 | `2`: 1,522 |
| `json_mode_agentic` / `singleturn` | `1`: 1,867 (`json_output`) | `1`: 702 (`json_output`) |

In addition, **477 / 3,426 rows (13.9%)** in `hermes_eval.jsonl` carry **multiple ground-truth calls** (`len(sample.ground_truth_calls) ∈ [2, 10]`).

**Four places where the current Lean workflow violates this requirement:**

1. **Training truncates native tools when `len(sample.tools) > pack_tools` (`T > 3`).**
   [`ToolPool.candidates_for`](../spancall/lean/tensorize.py#L235-L247) sets `budget = max(int(pack_tools), 1)` (`3` in [`configs/default.yaml`](../configs/default.yaml#L18)) and breaks out of the native-tool loop as soon as `len(candidates) >= budget`:
   ```python
   for t in native:
       if len(candidates) >= budget:
           break
   ```
   As a result, **1,273 training samples** (`1,054` with 4 tools, `218` with 5, `1` with 6) have their own native tools **dropped from the prompt** during training.
2. **Eval Part 1 (`eval_param_extraction_lean`) renders only `[gt_tool]` (`T = 1`), omitting all other tools in `sample.tools`.**
   [`eval_param_extraction_lean`](../spancall/lean/eval_p1.py#L264-L269) looks up `gt_tool` and calls `resolver.try_resolve(gt_tool, ...)`, which renders `[gt_tool]` alone via [`build_input_multi([tooldef_to_template_tool(schema)], ...)`](../spancall/lean/resolver.py#L404-L409). None of the sibling tools in `s.tools` appear in the prompt.
3. **Eval Part 3 (`predict_split`) leaves `len(s.tools) == 1` rows un-augmented (`T = 1`) and passes `T = 4..6` on multi-tool rows.**
   [`predict_split`](../scripts/build_lean_colab_notebook.py#L1459) calls `resolver.try_resolve_candidates(s.tools, ...)`.
   - For the **1,199 / 3,426 rows (35%)** in `hermes_eval.jsonl` with `len(s.tools) == 1` (including 497 `func_calling*` rows), [`try_resolve_candidates`](../spancall/lean/resolver.py#L287-L288) hits `if len(candidate_schemas) == 1: return self.try_resolve(candidate_schemas[0], ...)`, bypassing `H1` completely and rendering only 1 tool.
   - For the **166 rows** in `hermes_eval.jsonl` with 4–6 tools, `try_resolve_candidates` renders `T = 4..6` tools — a layout the model never saw in training because of gap (1).
4. **Multi-call rows (`len(sample.ground_truth_calls) > 1`) only supervise and extract `ground_truth_calls[0]`.**
   Both [`tensorize_sample_lean`](../spancall/lean/tensorize.py#L309) and [`LeanResolver.try_resolve_candidates`](../spancall/lean/resolver.py#L350-L354) operate on a single winning tool call per sample.

**Suggested changes:**

- **Change R0.1 — Never drop native tools in [`ToolPool.candidates_for`](../spancall/lean/tensorize.py#L235-L247):**
  Treat `pack_tools` as a *minimum* floor for distractor augmentation, never an upper cap on `sample.tools`:
  ```python
  # spancall/lean/tensorize.py (inside ToolPool.candidates_for)
  native = [t for t in (sample.tools or []) if t.name != gt_tool.name]
  candidates: List[ToolDefinition] = [gt_tool]
  seen = {gt_tool.name}

  # Keep ALL tools the row genuinely shipped alongside the GT tool.
  for t in native:
      if t.name not in seen:
          candidates.append(t)
          seen.add(t.name)

  # Pad with corpus distractors only when native tools are fewer than pack_tools.
  budget = max(int(pack_tools), len(candidates))
  ```
- **Change R0.2 — Render all `sample.tools` in Eval Part 1 while scoring `gt_tool`'s parameters ([`spancall/lean/resolver.py`](../spancall/lean/resolver.py#L359-L414), [`spancall/lean/eval_p1.py`](../spancall/lean/eval_p1.py#L269)):**
  Add an optional `context_schemas: Optional[Sequence[ToolDefinition]] = None` parameter to `LeanResolver.try_resolve(schema, utterance, recent_context, now, context_schemas=...)` (and `_resolve`). When `context_schemas` is supplied (e.g., `context_schemas=s.tools` in `eval_param_extraction_lean`), `build_input_multi` renders all tools in `context_schemas` (guaranteeing every tool in the sample is mentioned in the prompt), while `_candidate_masks` and `_score_parameters` still extract and decode the oracle `schema` (`gt_tool`).

---

### R1. Training renders `T=3`, oracle-tool eval renders `T=1`

> **Status — resolved.** (b) `context_schemas` and (c) jittered `T` floor both
> landed, and (d) `render_T` is now reported by the tensorize cell, the Part 1
> report and the Part 3 forensics table. `jitter_pack_tools` is on by default, so
> `T=1` is inside the training distribution rather than outside it.

**What differs.** Both paths now use the same template (`build_input_multi`),
which closed the finding-1 grammar mismatch — but not the *block count*.

| | Training | Eval Part 1 / `try_resolve` |
|---|---|---|
| Renderer | [`build_input_multi`](../spancall/lean/tensorize.py#L338) | [`build_input_multi`](../spancall/lean/resolver.py#L404) |
| Tools passed | `pack_tools=3` (`ToolPool`) | `[gt_tool]` only |
| `<|tool|>` blocks | 3 | 1 |
| `<|param|>` anchors | 5 (2 + 1 + 2) | 3 |

Verified with the same sample (gt `get_weather` with `city`+`unit`, two
distractors) at 2026-01-01:

```
T=1      chars= 452  <|tool|>=1  <|param|>=3  <|none|>=1  enum_line=yes
T=3      chars= 559  <|tool|>=3  <|param|>=5  <|none|>=1  enum_line=yes
```

**Why it matters (Audit-corrected).**

1. **Prompt `<|ctx_enum|>` block length & attention context.**
   [`build_input_multi`](../span_template/template.py#L421) builds `<|ctx_enum|>` in the prompt text as `sorted({v for t in tools for p in t.params for v in p.enum})`. Training renders the union of enums across all packed tools; `T=1` eval renders only `gt_tool`'s enums.
   *(Audit correction: the earlier draft claimed this changed the candidate mask width via R9. In fact, [`CandidateMaskBuilder.build_param_masks`](../spancall/model/masks.py#L434-L444) filters `<|ctx_enum|>` entries against `enum_vals = set(param_schema.enum)` for the individual parameter, so the candidate mask `A_k` has the exact same width at `T=1` and `T=3`. The real skew is in the rendered prompt text, token offsets, and bidirectional self-attention context.)*
2. **Positional drift.** Every anchor after the tool header shifts when `T` changes. The pointer and H1 read token representations whose positions and surrounding `<|tool|>` neighborhood differ whenever Eval Part 1 renders `T=1` instead of the sample's full tool set.

**Options.**

- **(a) Train `T=1`.** Delete `pack_tools` from the tensorize path. Breaks H1 supervision (`tool_acc` reverts to vacuous `1.0`).
- **(b) Pass `context_schemas=s.tools` in Eval Part 1 (`try_resolve`) + keep all native tools in `ToolPool.candidates_for` (Change R0.1 & R0.2).** Every sample in Eval Part 1 renders all of its native `s.tools` in the prompt while still evaluating `gt_tool` as the oracle tool (so H1 errors do not contaminate Part 1).
- **(c) Randomize minimum `T` at train time** (`pack_tools` drawn deterministically per sample from `{1, 2, 3}` when `len(s.tools) == 1`, while keeping all native tools when `len(s.tools) >= 2`). This exposes the encoder to `T=1..6` layouts during training so single-tool and multi-tool prompts are both in-distribution.
- **(d) Record `render_T` in reported features** (see R9) so every eval table states its `T` distribution.

**Recommendation: Combine (b) + (c) + (d).**
- In [`spancall/lean/tensorize.py`](../spancall/lean/tensorize.py#L235-L264), preserve all native tools (R0.1) and optionally vary the distractor pad floor across `{1, ..., pack_tools}` so `T=1` is in-distribution while H1 still trains on multi-tool rows.
- In [`spancall/lean/eval_p1.py`](../spancall/lean/eval_p1.py#L269), pass `context_schemas=s.tools` to `resolver.try_resolve` so Eval Part 1 mentions all tools in `s.tools`.

---

### R2. The recent-context budget is silently halved (and absent in training)

> **Status — resolved.** `select_recent_context` reports `recent_budget_tokens`
> and the resolver adds `recent_budget_effective` / `recent_budget_clamped`, so
> the clamp is visible instead of silent. Training now runs the same selector.

**What is wrong.**
1. **Silent clamp at inference:** [`configs/default.yaml`](../configs/default.yaml#L26) declares `context_budget_tokens: 4096`, but [`predict_tool`](../spancall/lean/resolver.py#L214), [`try_resolve_candidates`](../spancall/lean/resolver.py#L299), and [`_resolve`](../spancall/lean/resolver.py#L394) all clamp it:
   ```python
   eff_budget = min(self.context_budget_tokens, max(64, self.max_seq_len // 2))
   ```
   At the shipped `max_seq_len: 4096`, the operative budget is **2048** (half of what the config advertises), and `eff_budget` is absent from the returned `features` dict (`['abstain_p', 'decode_ms', 'enc_ms', 'heads_ms', 'mask_ms', 'n_params', 'n_recent_rows', 'n_resolved', 'n_valid', 'post_ms', 'pre_ms', 'recent_tokens', 'temperature', 'truncated']`).
2. **Train/serve mismatch (Audit addition):** [`tensorize_sample_lean`](../spancall/lean/tensorize.py#L334) never calls [`select_recent_context`](../spancall/agent/context.py#L232) at all! It converts all `sample.recent_context` entries unconditionally and then drops the entire training sample at [`tensorize.py:357`](../spancall/lean/tensorize.py#L357) if `len(ids) > max_seq_len`.

**Suggested changes:**
- In [`spancall/agent/context.py`](../spancall/agent/context.py#L274-L300), include `"recent_budget_tokens": int(context_budget_tokens)` in the `features` dict returned by `select_recent_context` (both on the early-return empty branch at line 274 and the normal return at line 297).
- In [`configs/default.yaml`](../configs/default.yaml#L26), set `context_budget_tokens: 2048` (matching `max_seq_len // 2`) and document the `max_seq_len // 2` ceiling in the comment.
- In [`spancall/lean/tensorize.py`](../spancall/lean/tensorize.py#L334), route `sample.recent_context` through `select_recent_context` with `eff_budget = min(context_budget_tokens, max(64, max_seq_len // 2))` (see also R8) so long recent-context histories are budgeted identically in train and eval instead of dropping the training row.

---

### R3. Candidate masks collide across schemas by parameter name

> **Status — resolved.** Keys are `f"{tool}::{param}"`; the bare name is kept as
> a fallback only when a single schema is supplied.

**What is wrong.** [`_candidate_masks`](../spancall/lean/resolver.py#L625-L629) keys the mask dict by the bare parameter name `p.name`:

```python
for schema in schemas:
    for p in schema.parameters:
        masks[p.name] = builder.build_param_masks(...)
```

Parameter names are only unique *within* a schema, not across schemas. Two tools in `schemas` that share a parameter name (e.g., `query` or `id`) overwrite each other, and [`_score_parameters`](../spancall/lean/resolver.py#L665) looks up `cand_masks.get(name)` without a tool qualifier.

**Audit clarification on when this triggers:**
- The earlier draft stated that Eval Part 2 (`predict_tool`) and Eval Part 3 (`try_resolve_candidates`) hit this today. In fact:
  - [`predict_tool`](../spancall/lean/resolver.py#L174-L268) never calls `_candidate_masks`.
  - [`try_resolve_candidates`](../spancall/lean/resolver.py#L350-L354) picks `winner` via H1 first and calls `self._resolve(winner, ..., encoded_bundle=encoded_bundle)`, which calls [`self._candidate_masks(output, output.text, [schema])`](../spancall/lean/resolver.py#L391) with a **single-element list `[schema]`** (`[winner]`).
- However, `_candidate_masks` accepts `schemas: Sequence[ToolDefinition]` and `_score_parameters` accepts `schema: Optional[ToolDefinition] = None`. Whenever multiple schemas are passed to `_candidate_masks` (or `_score_parameters` is called with `schema=None`), same-named parameters collide silently.

**Suggested changes in [`spancall/lean/resolver.py`](../spancall/lean/resolver.py#L610-L700):**
- Key `masks` by `f"{schema.name}::{p.name}"` (and keep `p.name` as a fallback lookup key only when `len(schemas) == 1` for backward compatibility with any external caller):
  ```python
  # Inside LeanResolver._candidate_masks:
  for schema in schemas:
      for p in schema.parameters:
          built = builder.build_param_masks(
              p, entries, token_offsets, text, ctx=row_ctx, max_cands=self.max_cands
          )
          masks[f"{schema.name}::{p.name}"] = built
          if len(schemas) == 1:
              masks[p.name] = built
  ```
- In [`_score_parameters`](../spancall/lean/resolver.py#L640-L699), carry `(tool_name, param_name, tok_idx)` on each anchor and look up `cand_masks.get(f"{tool_name}::{name}") or cand_masks.get(name)`.

---

### R4. The resolver re-tokenizes the prompt and rebuilds `_RowCtx` once per parameter

> **Status — resolved.** `_candidate_masks` builds one `_RowCtx.for_row` and
> reuses the offsets from `tokenize_legend`, so `output.text` is tokenized once.

**What is wrong.** Two redundant `O(n)` / `O(K · E · n)` operations happen inside [`LeanResolver._candidate_masks`](../spancall/lean/resolver.py#L610-L630):

1. **Missing `_RowCtx` sharing:** [`build_param_masks`](../spancall/model/masks.py#L389-L390) accepts a prebuilt `_RowCtx` and rebuilds `TokenIndex` + `EntryIndex` from scratch when `ctx is None`. [`tensorize.py:486`](../spancall/lean/tensorize.py#L486) builds `_RowCtx.for_row(out.entries, offsets, max_seq_len)` once per row and passes `ctx=row_ctx` across all `K` parameters. [`_candidate_masks`](../spancall/lean/resolver.py#L627) does **not** pass `ctx`, so every parameter re-scans and re-indexes every legend entry.
2. **Redundant full tokenization (Audit addition):** [`_candidate_masks`](../spancall/lean/resolver.py#L618) runs:
   ```python
   enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
   ```
   even though [`tokenize_offset_mapping`](../span_template/template.py#L463) was *just* called on the exact same `output.text` at [`resolver.py:315`](../spancall/lean/resolver.py#L315) or [`resolver.py:411`](../spancall/lean/resolver.py#L411)!

**Suggested changes in [`spancall/lean/resolver.py`](../spancall/lean/resolver.py#L610-L630):**
- Build `row_ctx = _RowCtx.for_row(entries, token_offsets, self.max_seq_len)` once at the top of `_candidate_masks` and pass `ctx=row_ctx` to `builder.build_param_masks(...)`:
  ```python
  from spancall.model.masks import CandidateMaskBuilder, _RowCtx

  if token_offsets is None:
      enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
      token_offsets = list(enc["offset_mapping"])
  builder = CandidateMaskBuilder(
      max_seq_len=self.max_seq_len, max_span_width=self.max_span_width
  )
  entries = output.entries
  row_ctx = _RowCtx.for_row(entries, token_offsets, self.max_seq_len)
  ```
- Optionally accept `token_offsets: Optional[Sequence[Tuple[int, int]]] = None` on `_candidate_masks` so callers that already tokenized `output.text` do not invoke the HF tokenizer a second time.

---

## P1 — Latent defects, timing bugs & train/serve skew

### R5. Eval Part 3's per-stage timings and total `latency_ms` are corrupted

> **Status — resolved.** Stage timestamps are ordered correctly, H1's own
> `h1_*` timings are merged rather than clobbered, and `latency_ms` is measured
> from `try_resolve_candidates`'s `t0`, so
> `pre + mask + enc + heads + decode + post <= latency_ms` holds.

**What is wrong (Audit-expanded to 3 root causes).**
When `try_resolve_candidates` runs the multi-tool bundle path ([`spancall/lean/resolver.py:298-357`](../spancall/lean/resolver.py#L298-L357) → [`_resolve:387-392`](../spancall/lean/resolver.py#L387-L392)):

1. **Inverted `t_mask` / `t_enc` timestamps in `_resolve`'s bundle branch ([`resolver.py:389-392`](../spancall/lean/resolver.py#L389-L392)):**
   ```python
   if encoded_bundle is not None:
       output, H, per_token, ctx_features = encoded_bundle
       t_pre = t0
       t_mask = time.perf_counter()
       cand_masks = self._candidate_masks(output, output.text, [schema])
       t_enc = time.perf_counter()
   ```
   Because `t_mask` is taken *before* `_candidate_masks` and `t_enc` *after* it, `mask_ms = (t_mask - t_pre) * 1000` measures nothing (~`0.0004 ms`), while `_candidate_masks` is billed to `enc_ms = (t_enc - t_mask) * 1000`!
2. **Overwritten `pre_ms` / `enc_ms` ([`resolver.py:333-334, 355`](../spancall/lean/resolver.py#L333-L355)):**
   `try_resolve_candidates` records `pre_ms = (t_enc - t0) * 1000` (which actually covers *both* render and `_encode`!) and `enc_ms = (now - t_enc) * 1000` (which actually covers `_tool_scores`!), then overwrites `_resolve`'s `res.features` via `res.features.update(features)`.
3. **`res.latency_ms` omits the H1 pass ([`resolver.py:298, 378, 586`](../spancall/lean/resolver.py#L586)):**
   `res.latency_ms` is computed at line 586 from `_resolve`'s local `t0` (line 378), *not* `try_resolve_candidates`'s `t0` (line 298). That is why `sum_of_stages` (`23.7 ms`) exceeds `res.latency_ms` (`5.6 ms`): `res.latency_ms` left out the entire H1 render + backbone forward pass!

**Suggested changes in [`spancall/lean/resolver.py`](../spancall/lean/resolver.py#L298-L393):**
1. In `try_resolve_candidates` (lines 298–357), time each stage accurately:
   - `t_pre`: after `select_recent_context` + `build_input_multi` + `tokenize_offset_mapping`.
   - `t_enc`: after `self._encode(ids)`.
   - `t_h1`: after `self._tool_scores(output, H, per_token)`.
2. In `_resolve`'s `encoded_bundle` branch (lines 387–392), fix the timestamp order:
   ```python
   if encoded_bundle is not None:
       output, H, per_token, ctx_features = encoded_bundle
       t_pre = t0
       cand_masks = self._candidate_masks(output, output.text, [schema])
       t_mask = time.perf_counter()
       t_enc = t_mask
   ```
3. In `try_resolve_candidates` (lines 354–357), merge timings instead of blindly clobbering `mask_ms` / `heads_ms`, and set `res.latency_ms = (time.perf_counter() - t0) * 1000`:
   ```python
   res.features["h1_pre_ms"] = h1_pre_ms
   res.features["h1_enc_ms"] = h1_enc_ms
   res.features["h1_heads_ms"] = h1_heads_ms
   res.features["pre_ms"] = h1_pre_ms
   res.features["enc_ms"] = h1_enc_ms
   res.features["heads_ms"] = h1_heads_ms + float(res.features.get("heads_ms", 0.0))
   res.latency_ms = (time.perf_counter() - t0) * 1000
   ```
   This guarantees `pre_ms + mask_ms + enc_ms + heads_ms + decode_ms + post_ms <= res.latency_ms`.

---

### R6. `pack_multi_tools=False` trains on a grammar inference never sees

> **Status — resolved.** Both branches render through `build_input_multi`; the
> flag survives only as a no-op compatibility shim.

**What is wrong.** [`tensorize_sample_lean`](../spancall/lean/tensorize.py#L281) defaults to `pack_multi_tools: bool = False`, and its `else` branch ([`tensorize.py:345-350`](../spancall/lean/tensorize.py#L345-L350)) calls `build_input` ([`template.jinja`](../span_template/template.jinja)), which emits **no `<|none|>` anchor** and a single-tool header layout. Meanwhile [`LeanResolver._resolve`](../spancall/lean/resolver.py#L404) always calls `build_input_multi` ([`template_multi.jinja`](../span_template/template_multi.jinja)), even for `T=1`:

```
T=1 (build_input_multi, 1 tool)   <|tool|>=1  <|param|>=3  <|none|>=1
build_input       (template.jinja) <|tool|>=1  <|param|>=3  <|none|>=0
```

**Suggested changes in [`spancall/lean/tensorize.py`](../spancall/lean/tensorize.py#L281-L352):**
- Change the default parameter on `tensorize_sample_lean` to `pack_multi_tools: bool = True`.
- When `pack_multi_tools=False`, still render via `build_input_multi([tooldef_to_template_tool(gt_tool)], ...)` so the Lean workflow has **one** canonical template grammar (`template_multi.jinja`) across all `T >= 1`.

---

### R7. `max_pos: 1` leaves the multi-positive loss untrained

> **Status — resolved as specified.** `max_pos` stays `1` in
> [`configs/default.yaml`](../configs/default.yaml), and the multi-positive path
> is now continuously exercised by a `max_pos=3` test instead of being dead code.

**What is wrong.** [`configs/default.yaml`](../configs/default.yaml#L88) sets `max_pos: 1` (which [`build_lean_colab_notebook.py`](../scripts/build_lean_colab_notebook.py#L126) reads into `MAX_POS`). [`compute_lean_pointer_loss`](../spancall/lean/loss.py#L22-L69) implements multi-positive marginal log-likelihood (`logsumexp(all) − logsumexp(positives)`), but with `max_pos: 1` only one positive slot is populated, reducing the loss to single-positive cross-entropy in every production run.

**Suggested changes:**
- Keep `max_pos: 1` in `configs/default.yaml` (since the comment notes `max_pos: 3` can diffuse top-1 probability unless paired with equal-value aggregation), **and** add an explicit unit test in [`tests/test_lean_flow.py`](../tests/test_lean_flow.py) exercising `tensorize_sample_lean(..., max_pos=3)` + `compute_lean_pointer_loss` on an utterance with repeated entity mentions so the multi-positive path is continuously tested.

---

### R8. Training skips recent-context sanitization and budgeting that inference applies

> **Status — resolved.** `_recent_rows` routes through `select_recent_context`
> with the same clamp the resolver uses, so a row with key `"user city"` survives
> in sanitized form instead of being dropped.

**What is wrong (Audit-expanded).** Training and inference handle `sample.recent_context` via different code paths:

- **Train ([`_recent_rows`](../spancall/lean/tensorize.py#L134-L151)):** Wraps `RecentRow(key=..., value=..., source=...)` in `except ValueError: continue` without calling `sanitize_key` or `_normalize_recent_value`, and without calling `select_recent_context`. Thus:
  1. A row with a non-identifier key (e.g. `"user city"`) is **dropped** at train time, whereas inference renames it to `"user_city"` and **keeps** it.
  2. Duplicate `(key, value)` pairs, >`k_recent` rows, and rows exceeding `eff_budget` are **not** evicted at train time (causing the whole sample to be dropped if `len(ids) > max_seq_len`).
- **Serve ([`select_recent_context`](../spancall/agent/context.py#L232-L300)):** Runs R7 marker checks, `sanitize_key`, `_normalize_recent_value`, deduplication, keyword-relevance + `k_recent` filtering, and `eff_budget` token eviction.

**Suggested changes in [`spancall/lean/tensorize.py`](../spancall/lean/tensorize.py#L134-L151):**
- Update `_recent_rows` in `spancall/lean/tensorize.py` to normalize keys/values via `select_recent_context` (or `sanitize_key` + `_normalize_recent_value` when `tools`/`tokenizer` are not passed):
  ```python
  def _recent_rows(
      sample: UnifiedSample,
      tools: Optional[Sequence[ToolDefinition]] = None,
      tokenizer: Optional[Any] = None,
      k_recent: int = 32,
      context_budget_tokens: int = 2048,
  ) -> List[RecentRow]:
      """Convert sample.recent_context to template RecentRow list via R6/R7 rules."""
      selected, _ = select_recent_context(
          sample.recent_context or [],
          tools or sample.tools or [],
          k_recent=k_recent,
          context_budget_tokens=context_budget_tokens,
          tokenizer=tokenizer,
      )
      rows_out: List[RecentRow] = []
      for r in selected:
          try:
              rows_out.append(
                  RecentRow(
                      key=str(r.get("key", "")),
                      value=str(r.get("value", "")),
                      source=str(r.get("source", "recent")),
                  )
              )
          except ValueError:
              continue
      return rows_out
  ```

---

## P2 — Cosmetic / attribution / dead-code hygiene

### R9. `T` is not recorded on any reported metric

> **Status — resolved.** `render_T` / `n_tool_candidates` are on the result
> features, the tensorize cell prints a `render_T` histogram beside
> `mean_n_candidates`, Eval Part 1 reports `mean_render_T` / `render_T_hist` /
> `render_T_t1_share`, and Part 3's forensics cell histograms `render_T` off the
> persisted `features` block.

**What is wrong.**
| Implementation | Feature key carrying tool count |
|---|---|
| [`_resolve`](../spancall/lean/resolver.py#L558-L571) (Part 1) | **absent** — neither `render_T` nor `n_candidates` |
| [`try_resolve_candidates`](../spancall/lean/resolver.py#L332) (Part 3) | `n_candidates` = number of *tools* passed to H1 (ambiguous with per-param candidate count `C`) |

**Suggested changes in [`spancall/lean/resolver.py`](../spancall/lean/resolver.py#L330-L337) & [`L558-L571`](../spancall/lean/resolver.py#L558-L571):**
- Add `"render_T": len(output.tools)` and `"n_tool_candidates": len(output.tools)` to `features` in `_resolve` (line 558) and in `try_resolve_candidates` (line 332, alongside `n_candidates` for backward compatibility).

---

### R10. `build_input` is dead code in the Lean path

> **Status — resolved.** The imports are gone from `spancall/lean/`, and
> `build_input`'s docstring states that the Lean workflow uses
> `build_input_multi` exclusively.

**What is wrong.**
- [`spancall/lean/resolver.py:34`](../spancall/lean/resolver.py#L34) imports `build_input` from `span_template.template` but **never calls it**.
- Once R6 switches `tensorize_sample_lean`'s single-tool branch to `build_input_multi([tooldef_to_template_tool(gt_tool)], ...)`, [`spancall/lean/tensorize.py:22`](../spancall/lean/tensorize.py#L22) no longer calls `build_input` either.

**Suggested changes:**
- Remove the unused `build_input` import from [`spancall/lean/resolver.py`](../spancall/lean/resolver.py#L34) and [`spancall/lean/tensorize.py`](../spancall/lean/tensorize.py#L22), and add a docstring note on `span_template.template.build_input` stating that the Lean workflow exclusively uses `build_input_multi`.

---

## Suggested Implementation Order

1. **R0 + R1 + R6 + R10 (Full tool coverage & unified multi-tool grammar):**
   - Never drop native tools in `ToolPool.candidates_for` (`budget = max(int(pack_tools), len(candidates))`).
   - Support `context_schemas=s.tools` in `LeanResolver.try_resolve` / `eval_param_extraction_lean` so Eval Part 1 renders all tools of the sample while scoring `gt_tool`.
   - Always use `build_input_multi` in `tensorize_sample_lean` and remove dead `build_input` imports from `spancall/lean/`.
2. **R3 + R4 (Resolver correctness & speed):**
   - Key `_candidate_masks` by `f"{schema.name}::{p.name}"` and thread `(tool, param)` into `_score_parameters`.
   - Share `_RowCtx.for_row` across all parameters in `_candidate_masks` and avoid re-tokenizing `output.text`.
3. **R8 + R2 (Recent-context parity & budget visibility):**
   - Route `_recent_rows` in `tensorize.py` through `select_recent_context` with the same `eff_budget` clamp as `LeanResolver`.
   - Emit `recent_budget_tokens` in `select_recent_context` features.
4. **R5 + R9 (Resolver timing & `render_T` telemetry):**
   - Fix `t_mask` / `t_enc` ordering in `_resolve`, preserve stage timings in `try_resolve_candidates`, measure `res.latency_ms` from `try_resolve_candidates`'s `t0`, and record `render_T`.
5. **R7 (Multi-positive loss regression test):**
   - Add a unit test in `tests/test_lean_flow.py` covering `max_pos=3` end-to-end.

## Regression Tests to Add (`tests/test_lean_flow.py`)

| Issue | Test |
|---|---|
| **R0** | `ToolPool.candidates_for` on a sample with 5 native tools and `pack_tools=3` returns all 5 native tools (`len(cands) == 5`), never truncating to 3 |
| **R0 / R1** | `resolver.try_resolve(gt_tool, ..., context_schemas=sample.tools)` renders `len(sample.tools)` `<|tool|>` blocks (`features["render_T"] == len(sample.tools)`) while decoding only `gt_tool` parameters |
| **R2** | `res.features["recent_budget_tokens"] == eff_budget`, and a render under budget is not truncated |
| **R3** | `_candidate_masks(out, text, [tool_a, tool_b])` preserves distinct masks for both `tool_a::query` and `tool_b::query` when both schemas define `query` |
| **R4** | `_RowCtx.for_row` is called once per `_candidate_masks` invocation (not once per parameter) |
| **R5** | On `try_resolve_candidates` with `>= 2` tools, `mask_ms > 0` and `pre_ms + mask_ms + enc_ms + heads_ms + decode_ms + post_ms <= res.latency_ms` |
| **R6** | `tensorize_sample_lean` with `pack_multi_tools=False` and `pack_multi_tools=True` (`pack_tools=1`) produce identical `input_ids` (including `<|none|>`) |
| **R7** | `tensorize_sample_lean(..., max_pos=3)` on a multi-mention utterance populates multiple positive slots and `compute_lean_pointer_loss` produces finite, non-zero gradients |
| **R8** | A recent row with key `"user city"` survives `tensorize_sample_lean` in sanitized form (`"user_city"`) rather than being silently dropped |
| **R9** | Both `try_resolve` and `try_resolve_candidates` populate `res.features["render_T"]` and `res.features["n_tool_candidates"]` |
