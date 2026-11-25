# SpanCall — Project State

*Snapshot: 2026-11-25 · Lean v3.3 (paper-aligned: `ModernBERT-large` 395M, 1:1 balanced Hermes V1 + xLAM-60k training, frozen Tier 1 Fair Eval + Tier 2 34,969-row ID/OOD/Adversarial eval, zero-training `Router+Schema` & `HybridFill+Schema` collaboration) · Paper: ["Not Every Tool Call Needs Generation: Function Calling as Span Extraction"](research_paper/main.tex)*

---

## 1. What the project is for

**SpanCall** is a non-autoregressive extractive function-calling system built on `answerdotai/ModernBERT-large` (395M, $d=1024$, 28 layers, $L_{\max}=4096$) that replaces sequential JSON decoding on groundable tool calls (**Stratum F**) with a single parallel pointer pass—and pairs cleanly with any off-the-shelf LLM via zero-training schema-aware routing (**Router+Schema**, **HybridFill+Schema**) when a call crosses the extractive boundary (`array`/`object` schemas or abstentions).

### 1.1 Core mechanism (Lean v3.3)

1. **Schema-grounded serialization (`span_template/` + `spancall/data/flatten.py`):** Candidate tools (`<|tool|>`), parameters (`<|param|>`), and `<|none|>` anchors are serialized alongside the R7-sanitized utterance and a canonical literal pool $\Omega$ (`<|ctx_meta|>`, `<|ctx_time|>`, `<|ctx_enum|>`, `<|ctx_recent|>`). Nested `object` parameters up to depth 3 are flattened into scalar dotted leaves (`filters.color`) and reconstructed after extraction.
2. **Anchor-conditioned two-head encoder (`spancall/lean/model.py`, `spancall/model/heads.py`):**
   - **$H_1$ Tool Head (`ToolHead`):** 2-layer GELU MLP scoring `MLP([h_tool; h_cls])` under randomized multi-tool packing (up to 6 distractors), enabling zero-shot generalization to unseen schemas (>99.5% pre-gate tool accuracy).
   - **$H_3$ Boundary-Factorized Pointer (`LeanPointerHead`):** Type-filtered candidate scoring ($W_{\max}=16$, $C \le 1024$, $d_h=256$) with learned width bias $r(\min(j-i, W_{\max}))$.
3. **Exclusivity-constrained two-step DFS (`spancall/lean/decode.py`, `spancall/lean/resolver.py`):** Value probability aggregation across repeated mentions → non-overlapping scalar DFS yielding joint log-margin $\Delta_{\text{joint}} \ge 0.02$ → strict type casting, unflattening, collapse guard, and JSON Schema validation. Two-step focus (`resolve_mode="two_step"`) re-encodes a compact $T=1$ sequence for $\hat{t}$ so the full $C=1024$ candidate budget serves $\hat{t}$.
4. **Zero-training LLM collaboration (`spancall/eval/llm_parity.py`, `spancall/eval/hybrid_fill.py`, `spancall/agent/`):**
   - **Router / Router+Schema:** Whole-call speculative routing; `Router+Schema` adds a deterministic post-$H_1$ schema-type gate that delegates declared `array`/`object` tools and abstentions to the LLM (zero ground-truth labels).
   - **HybridFill / HybridFill+Schema:** Keeps 100% of tool selection on $H_1$ and preserves grounded scalars, invoking the LLM once in plain JSON mode (`tools=[]`, `response_format=json_object`) to fill only unresolved or complex parameters.

### 1.2 Input contract — unified marker grammar (`span_template/`)

Both alignment/training and online serving render through `build_input_multi` (`span_template/template_multi.jinja`), guarded by a cryptographic template stamp (`span_template/stamp.py`, v3.3 stamp `1e62a355a351...`):

```text
<|cls|> <|tool|> ... <|param|> ... <|none|> <|sep|> <|ctx_meta|> <|ctx_time|> <|ctx_enum|> <|sep|> utterance <|sep|> <|ctx_recent|> <|eos|>
```

---

## 2. Lean v3.3 Paper Benchmark Summary (Completed)

- **Training:** 1:1 balanced Hermes V1 (19,096 → 69,412 with distinct distractor-packing seeds) + xLAM-60k (69,412) = 138,824 records → **117,760** tensorized training instances; `ModernBERT-large` fine-tuned 3 epochs (`bf16`, effective batch 128, cosine decay with 10% floor, seed 42). Held-out macro dev `exact_call` rises monotonically `0.857 → 0.868 → 0.874` (Epoch 3 restored).
- **Tier 1 Matched Fair Eval (`data/eval_v3_3/`, $n=1{,}735$ per corpus, $n_F=1{,}277$ [1,108 Normal + 169 Complex], $n_A=458$):**
  - **Standalone SpanCall:** **0.755** macro `exact_F` (0.743 Hermes / 0.766 xLAM), **0.801** macro `param_F1`, **77.0 ms** mean latency (Colab A100, batch 8), **0** generation tokens, **0** duplicate-span collapse, 4×–19× fewer entity swaps than hosted LLMs. Leads decisively on **Normal all-scalar calls** (0.798 Hermes, 0.840 xLAM vs. 0.360–0.397 and 0.755 for Gemma 4 / `gpt-oss-120b`).
  - **Router (abstention-only):** **0.807–0.808** macro `exact_F` (~80% token savings, 4.5×–5.2× lower mean latency).
  - **Router+Schema (post-$H_1$ schema gate):** **0.857** (Gemma 4) / **0.855** (`gpt-oss-120b`) macro `exact_F` (**0.889** on xLAM, +11.1–11.2 pp over either LLM), 100% complex recovery, **~60%** API token savings.
  - **HybridFill+Schema (plain-JSON partial fill):** **0.844** (Gemma 4) / **0.847** (`gpt-oss-120b`) macro `exact_F`, 38%–61% lower xLAM prompt tokens, lowest Stratum A false calls (**0.853–0.862**, safer than both standalone LLMs), and 64.5% total token reduction on concise-reasoning `gpt-oss-120b`.
- **Tier 2 Full Splits (34,969 rows):** Zero-shot OOD-Schema (`0.711` Hermes / `0.799` xLAM) and OOD-Para (`0.735` / `0.793`) match or exceed ID eval (`0.708` / `0.757`). Standalone Stratum A abstention remains the main open gap (`false_hit_on_A` 0.638–0.690 on Tier 1; 0.788–0.816 on adversarial splits).

See [`README.md`](README.md), [`MODEL_CARD.md`](MODEL_CARD.md), [`docs/experiments.md`](docs/experiments.md), and [`results/v3_3/export/final_report.md`](results/v3_3/export/final_report.md) for complete tables.

---

## 3. Lean v3.x Engineering Invariants

1. **Two-step only.** `LeanResolver` defaults `resolve_mode="two_step"` (`tool → params` with $T=1$ focus re-encode); `single_pass` is retired.
2. **`head_dim` & `max_cands` are explicit in `resolve_config.json`.** Consumers rebuild `LeanSpanCallModel` and `LeanResolver` from `resolve_config.json` (`head_dim=256`, `max_cands=1024`, `min_joint_margin=0.02`, `collapse_guard=True`, `prevent_param_collapse=True`).
3. **Safe Hub loading.** `spancall.hub.load_resolver` loads `model.pt` with `torch.load(..., weights_only=True)`. (The legacy `LeanResolver.load_checkpoint` still uses `weights_only=False`; use `load_resolver` for untrusted/Hub weights.)
4. **Frozen manifest integrity.** `data/eval_v3_3/manifest.json` locks SHA256 hashes of all 12 eval splits and the template stamp; `scripts/export_v3_3_results.py` enforces byte-identical parity between Colab SpanCall runs and local LLM runs.
5. **Secret hygiene.** API keys live only in environment variables or gitignored `.env` (see `.env.example`); notebooks with API keys are gitignored.

---

## 4. Component Status Overview

| Component | Status | Notes |
| --- | --- | --- |
| `span_template/` — template render, parser, sanitize, stamp | ✅ | `build_input_multi`, `resolve_utterance_span`, R7 sanitization, SHA256 template stamp |
| `spancall/data/` — schema, align, flatten, augment, mix, partition, sources | ✅ | `HermesLoader`, `XLAMLoader`, depth-3 object-leaf flattening (`flatten.py`), render-invariant `utt_rel_*` alignment |
| `spancall/model/` — `encoder.py`, `masks.py`, `heads.py` | ✅ | `SpanCallTransformerEncoder`, `CandidateMaskBuilder`, `BaseHead`, `ToolHead`, `BinaryToolHead` |
| `spancall/lean/` — `model`, `loss`, `tensorize`, `trainer`, `decode`, `resolver`, `eval_p1`, `diagnostics` | ✅ | Primary active v3.3 training, two-step inference, and Part-1 evaluation stack |
| `spancall/infer/` — `base`, `cast`, `clock`, `export` | ✅ | `BaseResolver` / `BaseResolveResult`, `StrictTypeCaster`, `SchemaValidator`, clock utils |
| `spancall/combine.py`, `spancall/hub.py` | ✅ | Public API: `SpanCallLLM` (6 policies), `load_resolver`; lazy top-level exports (`from spancall import SpanCallLLM`) |
| `spancall/agent/` — `router`, `loop`, `fill`, `context`, `trace`, `backends`, `tools` | ✅ | Version-agnostic `ToolCallRouter` (`ROUTER_SCHEMA`, `HYBRID_FILL_SCHEMA`, duplicate & `risk_tier` guards), interactive `ReActAgentLoop`, `OpenAIAPIBackend`, `SimpleTauRetailEnvironment` + `TauBenchAgentAdapter` |
| `spancall/eval/` — `toolcall`, `agentic`, `llm_parity`, `hybrid_fill`, `compare`, `cost` | ✅ | `ToolCallEvaluator`, paired bootstrap CIs, `Router`/`Router+Schema`, `HybridFill`/`HybridFill+Schema` |
| `scripts/` — v3.3 & agentic CLIs | ✅ | `build_fair_eval_v3_3.py`, `build_lean_v3_3_train_eval_notebook.py`, `build_tau_bench_agentic_notebook.py`, `eval_tau_bench.py`, `eval_llm_local.py`, `eval_hybrid_fill.py`, `export_v3_3_results.py` |
| `archive/` — superseded v2.x–v3.2 assets | ✅ | Archived historical notebooks, builders, plans, and legacy notebook tests |
| `research_paper/` — ACL LaTeX manuscript | ✅ | Aligned with v3.3 audited results (`main.tex`, `sections/`, `figures/`, `Makefile`) |
| `README.md` & `MODEL_CARD.md` | ✅ | Aligned with current paper and `convoicon/spancall-lean-v3-3` checkpoint |

---

## 5. Active Roadmap (Post-Paper Next Steps)

1. **Explicit Stratum A negative supervision:** Train dedicated abstention / refusal supervision on ungroundable and adversarial turns to close the standalone `false_hit_on_A` gap (0.638–0.690 on Tier 1; 0.788–0.816 on adversarial).
2. **Multi-turn stateful agentic benchmarks:** Evaluate `Router+Schema` and `HybridFill+Schema` end-to-end on BFCL, $\tau$-bench, and ToolSandbox.
3. **Long-context 32K bidirectional MLM backbone (`dmis-lab/UltraBERT-380M`):** Evaluate `dmis-lab/UltraBERT-380M` (*UltraBERT: Beyond the Limit of BERT Pre-Training*, arXiv:2610.15028 — 380M params, $d=1024$, 24 layers, RoPE/SwiGLU/RMSNorm, alternating local-128 & global attention, native **32,768** context, decoupled 5%/50% MLM heads) as a drop-in 32K replacement for `ModernBERT-large` on long multi-turn agentic trajectories without the token-boundary degradation seen with contrastive embedding backbones.
4. **Multi-seed & CPU/edge benchmarking:** Measure training-run variance across multiple seeds and benchmark single-request CPU / ONNX INT8 latency alongside GPU batch serving.
