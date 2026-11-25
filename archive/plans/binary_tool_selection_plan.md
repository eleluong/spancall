# Plan: Binary (0-1) Tool Selection — grounded in the actual Lean flow

## 0. Proposed plan vs current Lean flow — stage-by-stage comparison

| Stage | Current flow (verified in code) | Proposed (binary, flag on) | Why it should be better |
|---|---|---|---|
| **H1 objective** | CE / set-marginal over `<|tool|>` anchors ([loss.py:22-75](spancall/lean/loss.py:22)). Softmax forces sum-to-1: raising one tool's logit *lowers* the others'. | BCE per anchor: each tool independently learns P(this is the one). No forced competition. | A confident distractor no longer pushes the correct tool's score down. Softmax "prob" is only relative to the rendered set; sigmoid prob is an absolute P(called), so a threshold is a real operating point. |
| **T=1 renders** | `L_tool = CE over one class = 0`, `tool_acc` structurally 1.0 ([tensorize.py:215-239](spancall/lean/tensorize.py:215)). H1 learns nothing from single-tool rows. | BCE also degenerate at T=1 (one positive anchor, no negatives) — **same weakness, honestly stated**. | The tensorizer's `jitter_pack_tools` distractor injection ([tensorize.py:206-228](spancall/lean/tensorize.py:206)) already guarantees T>=2 with distractors on training rows. Under pack-tools rows the binary signal is DENSE (1 positive, T−1 negatives per row) where CE yields one softmax event. |
| **Dead anchors** | Logits masked to −1e4; CE ignores via ignore_index=−1 on targets ([trainer.py:524-532](spancall/lean/trainer.py:524)). | Same −1e4 fill; BCE(logit=−1e4, target=0) ~ 0 — silent under BCE too. | Neutral — no regression, verified in tests. |
| **Tool-name extraction at serving** | `_tool_scores` reads names from the rendered anchors (`e.tool`), softmax over logits, argmax winner ([resolver.py:1202-1231](spancall/lean/resolver.py:1202)). | Same anchor name read-back; sigmoid instead of softmax; winner = top-1 binary prob. | The "name comes from context" property is already present and preserved; only scoring semantics change. |
| **Abstain at serving** | Softmax always yields a winner; `call_min_prob` on a softmax prob is a *relative* threshold — abstains only when top-2 are close, not when the model is unsure. | Sigmoid P >= tau gate on the absolute score (predict_tool:389, try_resolve_candidates:467 in resolver.py). | Absolute confidence matches the calibration finding: correct-tool mean 0.489 vs wrong 0.039 separation survives an absolute threshold, not a margin threshold. Current 257/500 abstains come from post-hoc gates, not model uncertainty. |
| **Wrong-tool to param damage** | Serving: argmax winner decides which `cand_masks` get pointer-decoded (`_mask_key`, resolver.py:973). One distractor edge-out = all of the correct tool's params scored 0. | Same top-1 single-call dispatch; gate admits on absolute score. The correct tool can still lose the top-1 race. | **Honest limit:** binary does not make param extraction immune to a wrong top-1; it removes *score suppression* (the distractor's confidence no longer actively pushes the correct tool's prob down) and converts close-race wrong emissions into abstains. The param-favoring effect is: fewer wrong-tool emissions, not more correct-tool decodes. |
| **Pointer head interaction (training)** | Supervised on the GT tool's own `<|param|>` anchors (`param_positions`); no interaction with H1 at train time. | Unchanged. | Neutral by design — training-time head composition untouched. |
| **Joint loss bookkeeping** | `tool` term with `*_sum`/`*_count` micro-averaging ([loss.py:245-251](spancall/lean/loss.py:245)). | New `tool_binary` term with identical bookkeeping; CE term still computed and logged. | Both heads train simultaneously on the same backbone; ablation arms are directly comparable (`tool_acc` vs `tool_binary_acc` share argmax semantics). |
| **Metrics** | `tool_acc` (argmax), `tool_loss`. | Plus `tool_binary_acc` (argmax at tau=0.5), PR/F1 at calibrated tau, param precision conditioned on gate outcome. | The conditioned metric is the falsifiable claim: if param precision among "binary gate admitted the GT tool" rows is not materially higher than among "softmax winner was the GT tool" rows, the hypothesis is rejected and we keep softmax. |
| **Checkpoints / compat** | `tool_head` in state dict. | Plus `tool_head_binary` (strict=False already tolerates it, [resolver.py:299](spancall/lean/resolver.py:299)); flag off by default. | Zero risk to existing artifacts and the current 558-test suite paths. |

### What this plan does NOT claim (falsifiability)

- It does **not** fix the 0.0392 param precision — that is the 100%-stale-offset root cause (S1 in [docs/lean_eval_diagnosis_walkthrough.md](docs/lean_eval_diagnosis_walkthrough.md)). Binary targets are anchor-index-based and immune, so this head can be trained on current data, but its param-favoring effect is only measurable after S1.
- It does **not** increase correct-tool decodes per se; it converts close-race wrong emissions into abstains (absolute confidence) and removes cross-tool score suppression. If tool selection accuracy is already fine at argmax, the gain shows up in abstain precision, not param recall.
- T=1 degeneracy persists under binary **unless** `jitter_pack_tools` is active — it is (distractors injected, targets recomputed), so training-time T>=2 is already the norm.
- **Kill criterion:** on the ablation, if `tool_binary_acc` <= `tool_acc` and gate-conditioned param precision shows no separation, revert to softmax and record the negative result.

## What the deep dive established (corrections to v2)

1. **CE loss is already set-valued** ([`loss.py:22-75`](spancall/lean/loss.py:22)):
   `(B,)` legacy and `(B, P)` R0.4 multi-call marginal-likelihood forms. The binary
   head's genuine advantages over softmax are therefore only:
   - no sum-to-1 competition (a strong distractor doesn't suppress the correct
     tool's absolute score);
   - a per-tool probability that means P(called), so a threshold τ is a real
     operating point (softmax "prob" is only relative);
   - future abstain supervision via all-zero targets (deferred — see 5).
2. **The H1 forward is anchor-aligned** ([`trainer.py:520-532`](spancall/lean/trainer.py:520)):
   `tool_h` gathered at `<|tool|>` anchors, dead anchors filled `-1e4`, targets
   `-1` when unsupervised. A binary head consumes the **same** tensors — and the
   `-1e4` fill is BCE-compatible: `BCEWithLogits(logit=-1e4, target=0)` ≈ 0, so
   dead anchors stay silent under BCE exactly as under softmax.
3. **No tensorize change needed.** Binary targets are derived inside the loss by
   scattering the existing `tool_targets` / `call_tool_targets` to one-hot.
4. **The tool name is already read from the render** (context): `_tool_scores`
   builds choices from `per_token` anchors' `e.tool`
   ([`resolver.py:1202-1206`](spancall/lean/resolver.py:1202)). Binary mode
   keeps this — the model's choice is surfaced as the extracted anchor name.
5. **Pointer supervision is GT-tool-anchored at train time** (`param_positions`
   = primary call's tool's `<|param|>` anchors); the wrong-winner damage is at
   serving only, where the winner determines which `cand_masks` are decoded
   ([`_mask_key`](spancall/lean/resolver.py:973)).
6. **`predict_tool` short-circuits T=1** ([`resolver.py:324-333`](spancall/lean/resolver.py:324))
   to prob 1.0 — binary mode keeps that (nothing to decide with one candidate).

## Design

### 1. Head: `BinaryToolHead` ([`model.py`](spancall/lean/model.py))
- Same call signature as `ToolHead`: `forward(tool_h (B, T, d), cls_h (B, d))`
  → `(B, T)` logits (one per rendered tool anchor). Independent MLP
  (concat anchor hidden + cls hidden → hidden → 1), mirroring `ToolHead`'s
  inputs so both heads see the same representation.
- New attr `self.tool_head_binary` on `LeanSpanCallModel` (line 140); legacy
  `tool_head` untouched. `load_state_dict(strict=False)` already tolerates the
  new submodule on old checkpoints ([`resolver.py:299`](spancall/lean/resolver.py:299)).

### 2. Loss: `compute_lean_tool_loss_binary` ([`loss.py`](spancall/lean/loss.py))
- Input: the **same** `tool_logits`-shaped `(B, T)` binary logits and the
  **existing** targets (`tool_targets` (B,) or `call_tool_targets` (B, P)).
- One-hot scatter inside the function: 1 at each valid target index, 0 elsewhere.
  No new batch keys.
- `F.binary_cross_entropy_with_logits(logits, one_hot, pos_weight=T-1)`:
  counters the 1-positive-of-T imbalance. Dead anchors: logit already -1e4 →
  contribution ≈ 0 (no extra masking needed, verified in tests).
- Returns `None` when no row has supervision, mirroring
  `compute_lean_tool_loss`'s skip-don't-zero contract.
- `LeanLossContainer.forward`: new optional branch keyed on
  `predictions["tool_binary_logits"]` — dispatched only when the trainer emits
  it; the existing CE path is byte-identical otherwise. Weighted by the same
  `lambda_tool` (or new `lambda_tool_binary`, default 1.0).
- Micro-average bookkeeping (`*_sum` / `*_count`) copied from the CE term
  ([`loss.py:245-251`](spancall/lean/loss.py:245)) so the trainer's
  slot-weighted averaging stays consistent.

### 3. Trainer ([`trainer.py:520-539`](spancall/lean/trainer.py:520))
- Next to the existing `tool_head` call (line 523), when
  `self.use_binary_tool_head` (flag, **default False**):
  `bl = self.model.tool_head_binary(tool_h, cls_h).masked_fill(b["tool_positions"] < 0, -1e4)`
  → `preds["tool_binary_logits"]`; targets reuse `targs["tool_targets"]` /
  `targs["call_tool_targets"]` set by the existing code.
- Metrics: `tool_binary_loss` + `tool_binary_acc` (acc at τ=0.5 vs argmax —
  same argmax semantics as the CE `tool_acc` at
  [`trainer.py:760`](spancall/lean/trainer.py:760), so the arms are comparable).
- Eval loop mirror at [`trainer.py:725-728`](spancall/lean/trainer.py:725).

### 4. Resolver ([`resolver.py`](spancall/lean/resolver.py))
- `_tool_scores()` (line 1180): binary mode swaps line 1231's
  `torch.softmax(tl/temp)` for `torch.sigmoid(tl/temp)`. Choices list still
  `(name, prob)` built from the **rendered anchors** (name read from context,
  unchanged). Sorted descending.
- Winner + gate, in the two existing call sites:
  - `predict_tool` ([`resolver.py:389-406`](spancall/lean/resolver.py:389)):
    binary mode admits the top-1 only if `prob >= self.tool_call_threshold`
    (constructor arg, default 0.5); below → `gated_hit=False` with
    `abstain_reason="tool_below_threshold"`. Softmax path untouched.
  - `try_resolve_candidates` (line 467 ff.): the admitted list / `call_probs`
    feature fields keep their shape — single admission, same as today.
- Downstream pointer decode needs **zero changes**: `_resolve` already keys
  `cand_masks` by the winner's name via `_mask_key` — the binary gate only
  changes *who* wins and *whether* we abstain. Per-tool binary prob rides the
  existing `tool_prob` field in `LeanCallResult`.
- Threshold stored on the resolver; calibration (below) sets it.

### 5. Abstain supervision (deferred, explicitly)
True all-zero targets need rows with no GT call. Current tensorize rows
without supervised calls produce no `tool_positions`
([`tensorize.py:1004`](spancall/lean/tensorize.py:1004)) — those rows simply
aren't in H1's supervision. Adding a `tool_call_present` flag is only useful
if the dataset has no-call samples routed through the Lean path; deferred
until we verify that in the data. The threshold still gives serving-time
abstain from positive-side supervision alone.

### 6. Calibration + eval
- Dev-split sweep over τ on `tool_binary_logits` → PR curve, F1-optimal τ;
  persisted in the checkpoint dir alongside metrics (pattern of
  [`scripts/calibrate.py`](scripts/calibrate.py)).
- [`eval_p1.py`](spancall/lean/eval_p1.py): report binary tool P/R/F1 at τ,
  abstain rate, and — the payoff metric — param precision conditioned on
  "binary gate admitted the GT tool" vs not, using the existing
  record decomposition plumbing from diagnostics.

### 7. Notebook ablation
- Softmax vs binary arm in [`scripts/build_lean_colab_notebook.py`](scripts/build_lean_colab_notebook.py)
  (LoRA-ablation cell pattern, [`tests/test_notebook_cell_guard.py:145`](tests/test_notebook_cell_guard.py:145)),
  default off; regenerate.

### 8. Tests (new `tests/test_lean_binary_tool.py`)
- Loss: one-hot scatter from scalar and (B, P) targets; pos_weight; dead
  anchors (-1e4) contribute ≈0; None when unsupervised; CE path untouched
  when `tool_binary_logits` absent.
- Head: output shape, masked anchors.
- Trainer flag off → no binary keys in preds; flag on → keys present.
- Resolver: sigmoid (not softmax) probs; threshold abstain; T=1 short-circuit
  preserved; name still from rendered anchors.

## Execution order

1. Head + loss + tests (no behavior change; flag off).
2. Trainer branch + tests.
3. Resolver binary mode + tests.
4. Calibration + eval metrics.
5. Notebook ablation + regenerate.
6. **Decision gate:** retrain the binary arm after S1 label re-alignment
   (binary targets are anchor-index-based, immune to the offset rot, but the
   param-favoring effect is only measurable on clean labels).

## Implementation status

Steps **1–3 are done and green** (658 tests, 28 of them new in
`tests/test_lean_binary_tool.py`; the 630 pre-existing tests pass unchanged).

| Step | State | Where |
|---|---|---|
| BinaryToolHead | done | [`heads.py:89`](spancall/model/heads.py:89), wired at [`model.py:154`](spancall/lean/model.py:154) |
| `compute_lean_tool_loss_binary` | done | [`loss.py:107`](spancall/lean/loss.py:107) |
| `tool_objective` / `lambda_tool_binary` on the container | done | [`loss.py:248`](spancall/lean/loss.py:248) |
| Trainer flag, lock-step objective, `tool_binary_acc` | done | [`trainer.py:301`](spancall/lean/trainer.py:301) |
| Resolver sigmoid + absolute gate (both entry points) | done | [`resolver.py:216`](spancall/lean/resolver.py:216) |
| Config knobs | done | [`config.py:114`](spancall/train/config.py:114) |
| Provenance records the arm | done | [`trainer.py:1441`](spancall/lean/trainer.py:1441) |
| Calibration sweep + `eval_p1` P/R/F1 at τ | **pending** | — |
| Notebook ablation cell | **pending** | — |

### Deviations from the design above (and why)

1. **Dead anchors are excluded from the mean, not merely zeroed.** `-1e4` makes
   `BCEWithLogits` ≈ 0 for that column, but a plain `mean(-1)` would still
   divide by the PADDED width, so a row whose anchor failed to snap would report
   a systematically smaller loss than the same row fully live. The CE path
   normalises over live anchors only (`exp(-1e4) == 0`), so the binary loss now
   does too: `sum(-1) / live.sum(-1)`. Pinned by
   `test_binary_loss_scale_is_invariant_to_padding_columns`.
2. **A row whose every anchor is dead is unsupervised**, not a 0.0 loss —
   otherwise a dead row is reported as a suspiciously perfect prediction.
3. **The binary branch is gated on `tool_objective == "binary"`, not only on
   the presence of `tool_binary_logits`.** With the flag off the trainer never
   emits that key, so gating on presence alone was equivalent *for the current
   trainer* — but an injected predictions dict would then silently train two
   competing H1 objectives in the CE arm.
4. **`use_binary_tool_head` resolves as explicit arg > optimization config >
   False**, so a YAML-only ablation works and every existing caller is
   unaffected (the arg is `Optional[bool]`, default `None`).
5. **The trainer force-aligns an injected loss container's `tool_objective`.**
   A CE container plus the flag on would train the OLD head while reporting a
   binary arm — the one failure mode a kill-criterion ablation cannot have.
6. **The resolver raises at construction in binary mode if the model has no
   `tool_head_binary`**, and `_tool_scores` records `no_binary_tool_head`
   rather than falling back to the CE head.
7. **The `T == 1` short-circuit is shared by both arms** and the gate is
   deliberately inert there: with one rendered anchor neither objective carries
   information, and abstaining would break the single-candidate contract.
8. **Provenance records `tool_objective`** so a checkpoint cannot be scored
   with the wrong head — the resolver must be constructed with the same value.

## Files touched
`spancall/model/heads.py` (new head); `spancall/lean/model.py`, `loss.py`,
`trainer.py`, `resolver.py`; `spancall/train/config.py`; new
`tests/test_lean_binary_tool.py`. **No tensorize changes** — the one-hot scatter
happens inside the loss, exactly as designed above.
