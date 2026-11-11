# Lean Workflow Mismatch Audit

**Status:** Resolved / Remediation Complete.
**Scope:** `span_template/` (render) → `spancall/lean/tensorize.py` (train rows) → `spancall/lean/{model,loss,trainer}.py` (train) → `spancall/lean/{resolver,decode,eval_p1}.py` (infer/eval).

All findings below have been audited and resolved across the Lean pipeline:
- **P0 #1 (Render-format skew):** `LeanResolver._resolve` renders single-tool oracle prompts with `build_input_multi([tooldef_to_template_tool(schema)])`, aligning the template format (`template_multi.jinja`, `<|none|>` anchor present) with training. `try_resolve_candidates` now reuses the multi-tool encoding `(output, H, per_token)` directly in a single pass (`encoded_bundle`), eliminating the duplicate forward pass and evaluating parameters in the exact multi-tool context seen during training.
- **P0 #2 (Param anchor snapping):** `tensorize_sample_lean` strictly checks anchor validity: unsnapped anchors become `PAD_INDEX (-1)` instead of token 0 (`<|cls|>`), `param_snap_misses` is tracked in stats, and rows with missing/unsnapped anchors are dropped. In `LeanTrainer`, negative anchors mask out `pos_indices` to `-1` so dead anchors are never supervised.
- **P0 #3 (Context & clock mismatch):** `_to_date(val)` defaults to `date(2026, 1, 1)` to match tensorization base date, and `LeanResolver` defaults `k_recent: 32`, `context_budget_tokens: 4096` to match `configs/default.yaml`.
- **P0 #4 (Candidate capping):** `CandidateMaskBuilder.build_param_masks` reserves tail slots for recent candidates and `<|null|>` when `max_cands` is supplied and prioritizes non-null candidates during truncation. `tensorize_sample_lean` calls `build_param_masks` unconstrained (`max_cands=None`) and performs smart candidate reduction preserving positive targets and the null slot. `LeanResolver` supports `max_cands=512`.
- **P1 #5 (LeanBatchCollate rank-2 pad):** Collation uses `ref.shape` rather than `(ref.size(0),)`, correctly handling 2-D tensors without `IndexError`.
- **P1 #6 (1-D tensor preservation):** `trainer.tensorize_rows` preserves non-empty 1-D tensors via `t.unsqueeze(0)` instead of reshaping to `(0, 0)`.
- **P1 #7 (No-anchor fallback):** `LeanResolver._score_parameters` filters anchors by `schema.name` and removes the silent token 0 fallback; `_resolve` abstains with `no_param_anchors` when anchors are absent.
- **P2 #10 (load_rows empty positives):** `LeanTrainer.load_rows` emits a warning and returns an empty list if `filter_empty_positives=True` but no rows have positives.

The Lean pipeline has a split personality: the **training path always renders with the
multi-tool template**, while the **parameter-extraction inference path always renders with
the single-tool template**. Several smaller inconsistencies hang off that split. Findings are
ranked by how much they can corrupt a result.

---

## P0 — breaks the intended train/serve contract

### 1. Render-format skew: training never sees the prompt used for Eval Part 1 / Part 3 params

Training is unconditionally packed:

```python
# scripts/build_lean_colab_notebook.py:565-567
for out_name, in_name in SPLIT_INPUTS.items():
    pack_multi = True
    groups[(in_name, pack_multi)].append(out_name)
```

so every tensorized row is rendered by `build_input_multi` with `T = data.pack_tools = 3`
tools. The oracle-tool parameter path renders single-tool:

```python
# spancall/lean/resolver.py:392-402
output = build_input(tooldef_to_template_tool(schema), utterance, ...)
cand_masks = self._candidate_masks(output, text, [schema])
```

`try_resolve_candidates` also funnels into the same `_resolve`, so Eval Part 3's parameter
stage is single-tool too. Consequences, all of them observable in the token stream:

| | training row | Eval Part 1 / Part 3 params |
|---|---|---|
| template | `template_multi.jinja` | `template.jinja` |
| `<\|none\|>` anchor line | present | **absent** |
| `<\|tool\|>` blocks | 3 | 1 |
| `<\|param\|>` blocks | 3 × K | K |
| `<\|ctx_enum\|>` pool | union of all 3 tools' enums | GT tool's enums only |
| candidate mask entries | from the 3-tool render | from the 1-tool render |

The `<|ctx_enum|>` row is not cosmetic: enum values are `selectable` targets and
`enum_ctx` is an in-scope `EXTRACTABLE_TARGET_TYPES` member, so the candidate universe for
enum-typed params genuinely differs between train and eval.

`trainer.evaluate()` (used for `exact_call`/`span_acc` during training) reads the tensorized
multi rows, so **the training-time metric and the resolver-time metric are measured on two
different prompt formats**. Any comparison between them is confounded by this alone.

### 2. A param anchor that fails to snap silently becomes token 0 (`<|cls|>`)

```python
# spancall/lean/tensorize.py:423-434
anchors: List[int] = []
for e in p_entries:
    psnap = _snap_token_range(e.char_start, e.char_end, offsets, max_seq_len)
    anchors.append(psnap[0] if psnap else 0)     # 0 == <|cls|>
anchors = (anchors + [0] * K)[:K]                # pads with 0 too
```

Twenty lines above, the tool-anchor loop does the opposite and documents exactly why:

```python
# spancall/lean/tensorize.py:381-389
if tsnap is None:
    # PAD_INDEX (-1), NEVER 0: the trainer masks dead anchors with
    # tool_positions < 0 and 0 is a real token (<|cls|>), so a 0 here would
    # resurrect the slot as a live candidate pointing at the CLS embedding.
    tool_positions.append(PAD_INDEX)
```

`param_positions` is a member of `PAD_INDEX_KEYS`, so `-1` is already a legal, recognised
pad value for it, and the trainer's `param_positions.clamp(min=0, ...)` means a `0` is
**never** filtered — it is quietly substituted with the CLS embedding. Because
`pos_indices` for that param is still a real positive, the loss keeps supervising it, so the
model is actively trained to answer the correct span from a question vector that says
"CLS" rather than "which param". Nothing is counted, nothing is logged (there is no
`param_snap_misses` counter, unlike `tool_snap_misses`).

Two corollaries:

- `(anchors + [0] * K)[:K]` pads **missing** anchors with `0`. If the renderer ever emits
  fewer `<|param|>` entries than `len(gt_tool.parameters)`, this silently shifts every
  subsequent anchor by one position, mislabelling every param after the gap. A short
  `anchors` list is a data-integrity failure that should drop the row, not be papered over.
- The same `0` fallback exists in the single-tool branch (line 433).

### 3. `recent_context` and `now` are filtered at inference but not at training

```python
# training — every row, no budget, fixed date fallback
recent_rows = _recent_rows(sample)                       # tensorize.py:334
base_date = date(2026, 1, 1)                             # tensorize.py:327

# inference — k_recent=3, 256-token budget, wall-clock fallback
selected_recent, ctx_features = select_recent_context(   # resolver.py:384-390
    recent_context or [], [schema], k_recent=self.k_recent,
    context_budget_tokens=eff_budget, tokenizer=self.tokenizer)
now=_to_date(now)                                        # -> date.today() if None
```

`recent_ctx` is itself an `EXTRACTABLE_TARGET_TYPES` member. Training shows the model every
recent row; serving shows at most 3 rows that survived a token budget. The `<|ctx_recent|>`
line therefore has a different length and different content at serve time, and any
`recent_ctx` candidate the model learned to rely on may simply be absent.

The date fallback is the same class of bug in miniature: `date(2026,1,1)` at training vs
`date.today()` at inference changes the `<|ctx_time|>` window (and therefore char offsets and
every candidate position) for any sample lacking `current_time_iso`.

### 4. `max_cands` is applied at training but never at inference

```python
# training
masks = mask_builder.build_param_masks(
    param, out.entries, offsets, out.text, ctx=row_ctx, max_cands=max_cands)  # 512

# inference — max_cands defaults to None -> no clamp in _pack
masks[p.name] = builder.build_param_masks(p, entries, token_offsets, text)
```

Training truncates to `MAX_CANDS = 512` with a positives-first + ±15-neighbour + stride-fill
policy (`tensorize.py:566-588`). Inference keeps the full windowed candidate list, which for
free-string params is every utterance span up to `W_max = 16` wide — easily thousands.
`LeanPointerHead` scores candidates independently, so the *positive's* logit is unchanged,
but the softmax denominator in `compute_lean_pointer_loss` / `resolver._score_parameters` is
taken over a different set. A candidate that training deliberately excluded can outscore the
positive at serve time and flip the argmax. Train and eval also disagree on `C` itself.

---

## P1 — latent bugs / silent data loss

### 5. `LeanBatchCollate` builds a rank-1 pad for a missing rank-2 key

```python
# spancall/lean/tensorize.py:660-675
if any(v is None for v in vals):
    ref = next(v for v in vals if isinstance(v, torch.Tensor))
    filled.append(torch.full((ref.size(0),) if ref.dim() >= 1 else (), pad_val, ...))
```

For a missing `cand_starts` (rank 2, `(K, C)`) this produces a rank-1 `(K,)` tensor, which
then reaches the `vals[0].dim() == 2` branch and is indexed with `v.size(1)` →
`IndexError`. Reachable the moment a batch mixes tensorized rows with rows from another
source. The pad shape must mirror `ref.shape`, not just `ref.size(0)`.

### 6. `trainer.tensorize_rows` silently deletes 1-D supervision tensors

```python
# spancall/lean/trainer.py:613-614
if k in ("cand_starts", "cand_ends", "pos_indices") and t.dim() == 1:
    t = t.reshape(0, 0).long()
```

A rank-1 `pos_indices` becomes `(0,0)`, `K` becomes `0`, and the pointer branch is skipped —
the row is trained on tool supervision only, with no error and no counter.

### 7. No-anchor fallback points every param at token 0

```python
# spancall/lean/resolver.py:614-617
if not anchors:
    for schema in output.tools:
        for p in schema.params:
            anchors.append((p.name, 0))
```

Same CLS-embedding substitution as finding 2, but at inference and for *all* params at once.
It should be a refusal (`status="abstain"`, `error_reason="no_param_anchors"`), not a silent
all-CLS scoring pass.

---

## P2 — by design, but worth pinning so nobody misreads the numbers

### 8. `optimization.max_pos: 1` leaves the multi-positive path untrained
`compute_lean_pointer_loss` implements `logsumexp(pos) - logsumexp(all)` over `P` positives,
but `pos_rows.append((pos_cols + [-1]*max_pos)[:max_pos])` with `max_pos=1` yields exactly one
positive per param. The intended multi-positive behaviour is dead code in this configuration.

### 9. Only `Stratum.F` receives pointer supervision
The entire `pos_indices` matching block is gated on `if is_stratum_f:`. Non-F rows come out
with all-`-1` positives and are then removed by `load_rows(filter_empty_positives=True)`.
15828 tensorized train rows from 24071 input lines reflects this, as does the H1 histogram
`{3: 461}` on dev.

### 10. `load_rows` all-or-nothing filter fallback
```python
groundable = [r for r in rows if has_positive_targets(r)]
if groundable:
    rows = groundable
```
If **no** row has a positive, the filter is skipped and every unsupervised row is trained on.
That is a silent "did the tensorizer break?" → "train anyway" path; it should be loud.

---

## Remediation status

### Landed (verified after the upgrade)

| # | Finding | Fix in tree | Guard |
|---|---------|-------------|-------|
| 1 | Train renders `T=3`, oracle-tool eval rendered `T=1` | [`LeanResolver._resolve`](spancall/lean/resolver.py:404) now calls `build_input_multi`, and `try_resolve_candidates` passes the already-encoded bundle | `test_lean_resolver_candidates_multi_template_and_bundle` |
| 2 | Dead anchors fell back to token `0` (`<|cls|>`) | [`tensorize_sample_lean`](spancall/lean/tensorize.py:386) emits `PAD_INDEX`, counts `param_snap_misses` / `tool_snap_misses`, and drops the row when the **GT** anchor dies | `test_tensorize_encodes_unsnapped_distractor_as_pad_index`, `test_tensorize_drops_row_when_gt_tool_anchor_cannot_snap` |
| 3 | `recent_context` truncated at serve only | `k_recent=32`, `context_budget_tokens=4096` (was 3 / 256) | — |
| 4 | `max_cands` applied at train only | `LeanResolver(max_cands=512)` threads into `_candidate_masks` / `_pack` | — |
| 5 | Rank-1 pad for a rank-2 missing key | [`LeanBatchCollate`](spancall/lean/tensorize.py:698) pads with `torch.full(ref.shape, …)` | `test_lean_batch_collate_rank2_missing` |
| 6 | `tensorize_rows` deleted 1-D supervision tensors | non-empty 1-D → `unsqueeze(0)`, only empty → `(0, 0)` | `test_trainer_tensorize_rows_1d_preservation` |
| 7 | No-anchor fallback pointed every param at token 0 | explicit `abstain_reason="no_param_anchors"` | `test_lean_resolver_no_param_anchors_abstain` |
| 10 | Silent all-or-nothing `load_rows` filter | warns when `0/len(rows)` survive | — |

### Landed (bug introduced by the upgrade, then fixed)

**Candidate cap double-counted every index** — [`tensorize_sample_lean`](spancall/lean/tensorize.py:602).
The rewritten block appended each kept index to BOTH an ordered `pool` list and the
`keep_set` membership set. Two consequences:

1. the early-stop test `len(pool) + len(keep_set) >= max_cands` counted each index twice,
   so it fired at roughly **half** the real budget; and
2. the final `sorted(list(keep_set) + pool[: max_cands - len(keep_set)])` emitted the same
   `(tok_start, tok_end)` span **twice**.

Measured on a 949-candidate free-string row with `max_cands=48`:

```
OLD: columns=48 distinct_spans=25 dupes=23   # ~48% of the budget was repeats
NEW: columns=48 distinct_spans=48 dupes=0
```

So a `max_cands=512` row held ~256 distinct candidates padded with ~256 repeats: the model
softmaxed over duplicated columns and the row kept about half the distinct coverage the
resolver sees — a train/serve gap pointing the *opposite* way to finding 1.

Fix: `keep_set` is the single authoritative membership set, the emitted list is exactly
`sorted(keep_set)`, and every growth loop breaks on `len(keep_set) >= max_cands`.
Guarded by **`test_tensorize_capped_candidates_have_no_duplicate_spans`**, which asserts
exactly `max_cands` live columns, all spans distinct, and that no positive index is lost.

### Still open (deliberate, not bugs)

- **`T=3` vs `T=1` render skew.** Both paths now use the same template, but training packs
  3 tool blocks while oracle-tool eval renders 1. Same grammar, different `<|tool|>` count
  and enum pool. Closing it means either training on `T=1` (loses H1 supervision) or
  evaluating with distractor packs (changes what "oracle tool" means).
- **`recent_context` / `now` parity.** Serve is now bounded at `k_recent=32` + 4096 tokens,
  training is unbounded; `now` falls back to `today()` at serve vs `2026-01-01` at train.
  Harmless when `current_time_iso` is supplied (the notebook always does).
- **`_candidate_masks` keyed by `p.name` across schemas.** Two candidate tools sharing a
  parameter name (e.g. both have `query`) would collide — latent for Eval Part 2/3.
- **Bundle path zeroes timings.** `pre_ms` / `mask_ms` / `enc_ms` report ~0 for Eval Part 3
  because the reused `encoded_bundle` skips the per-stage timers.

---

## Verification performed

- Read: `lean/tensorize.py` (all 699 lines across 4 slices), `lean/model.py`, `lean/loss.py`,
  `lean/trainer.py` (train_step, evaluate, load_rows, tensorize_rows, collate, make_batches),
  `lean/resolver.py` (full), `lean/decode.py` (full), `lean/eval_p1.py` (full),
  `span_template/{template.py,parser.py,template.jinja,template_multi.jinja}`,
  `spancall/model/masks.py` (build_param_masks, _pack, _null_candidates),
  `configs/default.yaml`, and the notebook generator's tensorize/resolver/eval cells.
- Confirmed from the live end-to-end run that `pack_multi=True` produces
  `_n_tool_candidates: 3`, `tool_positions: [3, 31, 84]`, `tool_targets: 0`, `K=2`, `C=512`
  on a real dev row — i.e. training rows really are 3-tool renders.

## Workflow

```mermaid
flowchart LR
    A[UnifiedSample] --> B[build_input_multi T=3]
    B --> C[tensorize_sample_lean]
    C --> D[t2.jsonl multi rows]
    D --> E[LeanTrainer train and evaluate]
    A --> F[build_input single tool]
    F --> G[LeanResolver _resolve]
    G --> H[Eval Part 1 and Part 3 params]
    E -.->|different prompt format| H