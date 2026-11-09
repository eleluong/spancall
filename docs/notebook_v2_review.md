# SpanCall — Technical Recap: Architecture, Training, Loss, Accuracy

> Grounded in `spancall/model/{model,heads,loss,encoder}.py`, `spancall/train/trainer.py`, `configs/default.yaml`, and the measured eval outputs in `results/toolcall_metrics_{hermes_eval,b1a}_eval.json`.

## 1. Architecture

### 1.1 Overall shape
```
utterance + rendered tool schema (markers) 
        │
        ▼
Encoder backbone: Qwen/Qwen3-Embedding-0.6B (bidirectional, is_causal=False, 4096 seq len)
        │  hidden states H (B, L, d)
        ▼
Multi-task heads (v1 minimal set, D4):
  H1  ToolHead          — softmax over <|tool|> anchors {t1..tT, NONE}
  H3  PointerHead       — boundary-factorized span scorer, abstain logit at col 0
  H4  AbstainHead       — 5 classes: OK / ABSENT / AMBIGUOUS / NEEDS_REASONING / OFF_WINDOW
  H8  CorrectnessHead   — learned P(top-1 == GT), stop-grad inputs
      CallAcceptHead    — whole-call binary gate from <|cls|>
  (H2/H5/H6/H7 spec-only, constructed when their flag flips on)
```

### 1.2 The span-template input
A jinja template (`span_template/template*.jinja`) renders the candidate tool schemas + utterance + recent-context virtual rows into a single token stream with special marker tokens (`<|tool|>`, `<|param|>`, `<|none|>`, ...). Every parameter anchor `<|param|> p_k` defines a pointer query; candidate answer spans are enumerated by `CandidateMaskBuilder` (max width 16, capped at C=512 with hard negatives near positives + uniform stride).

### 1.3 Head mechanics (heads.py)
- **H1 ToolHead**: shared 2-layer MLP over `[H_tau_t; H_CLS]` → one logit per `<|tool|>` anchor. Logit: `u_t = w·GELU(W[H_tau_t; H_CLS])`. With `pack_tools=1` it is a no-op — but the notebook's uniform multi-tool packing keeps it active for every row.
- **H3 PointerHead**: boundary factorization —
  - `q_k = W_q·LayerNorm(H_pi_k)` (query from the param anchor),
  - `K_s = H·W_s^T`, `K_e = H·W_e^T` (start/end keys),
  - `s_{k,i} = q_k^T·K_{s,i}/√d_h`, `e_{k,j} = q_k^T·K_{e,j}/√d_h` plus a width bias and an abstain logit prepended at column 0. Softmax runs over `{abstain} ∪ A_k`.
- **H4 AbstainHead**: 5-way CE over reason classes per param, with an H3–H4 consistency term (see loss).
- **H8 CorrectnessHead**: BCE on P(pointer top-1 ∈ GT positives), computed on **detached** pointer logits (stop-grad — it learns to predict correctness, not to influence it).
- **CallAcceptHead**: BCE on whole-call accept (stratum F → 1, A → 0).

### 1.4 Inference chain (resolver.py §4.1.3)
One encoder pass per candidate-set: H1 scores tools → winner schema → H3 argmax over `{abstain} ∪ candidates` per param → gate (tau_p/tau_a in production; `raw_mode=True` skips gating in the notebook) → cast/canonicalize → payload. `joint_margin`, `p_abstain` per param, `accept_score` feed the abstain decision + reason attribution.

## 2. Loss design (loss.py)

Total = weighted sum of active components (weights in `__init__`):

| component | weight | form | notes |
|---|---|---|---|
| H3 pointer | λ_ptr = 1.0 | multi-positive softmax CE, `-log Σ_p exp(z_p) / logsumexp(z)` | abstain rows (P={0}) up-weighted by `abstain_weight=1.5` (default; notebook sets 1.0) |
| accept | λ_acc = 0.5 | BCE | whole-call binary |
| H4 reason | λ_rsn = 0.5 | weighted 5-class CE + `0.1·MSE(logit(¬OK), M(k,∅))` | class weights (0.25, 1, 1, 1, 1) — OK down-weighted; **the MSE ties H4's ¬OK logit to H3's abstain logit** |
| H1 tool | λ_tool = 1.0 | CE over anchors | only with pack_tools > 1 (notebook: always) |
| H8 corr | λ_corr = 1.0 | BCE on detached-argmax membership | stage C |
| H2 count | 0.3 | CE | spec-only |

Two design points worth flagging:
1. **Multi-positive softmax** (`compute_pointer_loss`): equal-valued targets share probability mass via logsumexp over positives — correct for entities that appear multiple times.
2. **H3–H4 coupling**: the reason loss's MSE term forces H4's ¬OK probability to track H3's abstain logit. This is a *feature* for consistency but a *risk*: a miscalibrated H3 abstain logit drags H4 with it (and vice versa) — relevant to the abstain-collapse observed below.

## 3. Training (trainer.py)

### 3.1 Staged curriculum (design)
| stage | data | what trains | purpose |
|---|---|---|---|
| T0 | hermes_train (single-tool) | encoder warmup + H3 | representation |
| T1 | hermes_train (single-tool) | H3 + H4 joint | extraction + abstain |
| T2 | hermes_train (multi-tool) | + H1 | tool selection in candidate sets |
| C | hermes_calib | H8 only, encoder frozen | learned confidence |

Optimizer: AdamW with layer-wise LR decay on the backbone (encoder LR 3e-5, heads 1e-4), linear warmup 6% then cosine decay. bf16, gradient checkpointing, batch 16 × accum 4 = effective 64.

**Reality in the notebook**: only `--stages T2` runs. T0/T1 warmup and stage C (H8 calibration) never execute — H8 is untrained while the resolver's confidence reporting still flows through it. The per-epoch held-out eval (`epoch_eval` events: span_acc / exact_payload_match / abstain_acc on dev, capped at 500) drives A6 best-checkpoint selection.

### 3.2 Data
Hermes function-calling v1 → `UnifiedSample` (stratum F=extractable ~55%, A=abstain ~15%, + paraphrase/swap-distractor augmentation). Schema-disjoint partition: train 80% / dev 5% / calib 5% / OOD-test 10%. Tensorization: BPE-aware snapping, positives verified via canonical_value, capped candidate sets, seeded fractioning mirroring `load_rows`.

## 4. Measured accuracy (results/, hermes_eval, 200-row subset)

### 4.1 Model (raw mode, no calibration gate)
| metric | value |
|---|---|
| tool_accuracy (H1) | **1.000** |
| hit_rate | 0.015 |
| exact_payload_match | 0.000 |
| param_precision / recall / F1 | 0.333 / 0.0022 / 0.0044 |
| abstain_reasons | **"abstained": 197 / 200** |
| Stratum F (n=32): hit_rate | 0.0625 |
| Stratum A (n=168): false_hit_rate | 0.006 |

### 4.2 B1a regex baseline (full 3426 rows)
| metric | value |
|---|---|
| tool_accuracy | 0.806 |
| hit_rate | 0.023 |
| exact_payload_match | 0.010 |
| abstained | 3347 / 3426 (98%) |

### 4.3 What the numbers actually say
1. **H1 tool selection is solved.** 100% top-1 on the eval subset (and the baseline's 80.6% shows it's a learnable signal, not trivial). The multi-tool packing + anchor design works.
2. **The pointer/abstain machinery collapses to abstention.** 197/200 abstained, param recall 0.2%. This is not "wrong values" — error_wrong_value = 2 total, everything else abstains. Either:
   - the abstain logit dominates the softmax (pointer never trained enough / abstain_weight tipping the balance — notebook sets 1.0 vs default 1.5, so the notebook is already relaxed and *still* abstains), or
   - positives are not being matched at tensorization (the sanity cell's "rows with ≥1 positive target" ratio is the first place to look — if the pos-matching rate is low, H3 trains on mostly-abstain rows), or
   - the H3–H4 MSE coupling drags both toward ¬OK.
3. **Stratum behavior is right where it fires.** Stratum F hit 6.25% vs stratum A false-hit 0.6% — when the model does emit, it emits on the right rows. The accept/abstain *decision* boundary is fine; the extraction behind it is empty.
4. **Both systems sit at the same ceiling.** B1a also abstains 98% with exact 1%. The current pipeline — regex or trained model — extracts almost nothing end-to-end. The gap between "raw argmax correct" (per §5a diagnostics: vetoed-vs-wrong decomposition exists exactly for this) and "hit" is where the problem lives.

## 5. Diagnosis priorities (by evidence strength)

1. **Check the positive-matching rate at tensorization** (`dev.jsonl` rows with ≥1 positive: the sanity cell prints it). If stratum-F rows mostly have `pos_indices == [0]`, the pointer head is literally being trained to abstain.
2. **Check the §5a veto decomposition** — `pointer_correct_vetoed` vs `pointer_wrong` vs `pointer_empty`. If raw argmax is right and vetoed, it's gate calibration; if pointer is empty, it's candidates/pos-matching; if pointer is wrong, it's training.
3. **Complete the curriculum.** T2-only skips the encoder warmup that H3 needs; and H8/stage-C is what converts raw scores into usable confidence.
4. **Watch the H3–H4 MSE term** when tuning abstain behavior — it couples the two heads, so a fix to one moves the other.

## 6. One-line verdict

The architecture (marker-template + anchor-conditioned boundary-factorized pointer + multi-head loss) is sound and already solves tool selection perfectly; the pipeline's bottleneck is that the pointer/abstain head pair has collapsed to near-total abstention (param recall 0.2%), which points at positive-span supervision at tensorization or the abstain-weighting/coupling in the loss — not at the model class itself.
