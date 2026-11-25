---
language:
- en
base_model: answerdotai/ModernBERT-large
datasets:
- NousResearch/hermes-function-calling-v1
- Salesforce/xlam-function-calling-60k
tags:
- function-calling
- tool-calling
- span-extraction
- pointer-network
- modernbert
- spancall
- non-autoregressive
library_name: spancall
---

# SpanCall Lean v3.3 (`convoicon/spancall-lean-v3-3`)

**SpanCall** is a non-autoregressive extractive function-calling model built on [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) (395M parameters).

Instead of autoregressively generating JSON tokens, SpanCall jointly encodes candidate tool/parameter schema anchors, the sanitized user utterance, and a canonical literal context pool, then emits schema-valid JSON in parallel using a tool selection head, a type-filtered span pointer head, and exclusivity-constrained two-step decoding.

---

## Model Specifications

| Property | Value |
| :--- | :--- |
| **Backbone** | `answerdotai/ModernBERT-large` (395M params, 28 layers, hidden size 1024) |
| **Heads** | Tool scorer (`H1`): 2-layer GELU MLP · Span pointer (`H3`): boundary-factorized pointer (`head_dim = 256`, learned span-width bias) |
| **Sequence & Candidate Budget** | Max length 4096 tokens, max span width 16 tokens, up to 1024 type-filtered candidate spans |
| **Context Serialization** | `<|ctx_meta|>` (booleans/null), `<|ctx_time|>` (resolved calendar offsets), `<|ctx_enum|>` (deduplicated schema enums), `<|ctx_recent|>` (recent tool outputs); nested `object` parameters flattened to dotted paths up to depth 3 |
| **Inference Mode** | `two_step` (selects the top tool first, then re-encodes a single-tool sequence so all 1024 candidate slots serve that tool) |
| **Decoding & Gating** | Value probability aggregation across repeated mentions -> exclusivity-constrained joint search (min margin 0.02) -> strict type casting, unflattening, duplicate-span guard, and JSON Schema validation |
| **Export Format** | `model.pt` (`state_dict`, load with `weights_only=True`), tokenizer files (with atomic `<|...|>` special marker tokens), `resolve_config.json` |

---

## Intended Use & Collaboration Policies

SpanCall is designed for **Stratum F** tool calls—turns where all arguments are directly groundable in the user request, recent dialogue, or schema literals—and excels on **all-scalar (`Normal`)** schemas (`string`, `integer`, `number`, `boolean`, `date`, `enum`).

For production traffic containing complex schemas (`array`, `object`) or ungroundable/compositional turns (**Stratum A**), pair SpanCall with any off-the-shelf LLM using one of the zero-training collaboration policies below:

| Policy | LLM Trigger | What SpanCall Keeps | LLM Mode | Recommended When |
| :--- | :--- | :--- | :--- | :--- |
| **Router** | SpanCall abstains | Committed scalar hits | Full multi-tool FC | Maximum latency/token savings (~80% fewer tokens) on mostly scalar traffic |
| **Router+Schema** *(default)* | Selected tool has `array`/`object` params, or abstain | All-scalar (`Normal`) hits | Full multi-tool FC | Highest accuracy (**0.855–0.857** macro EM) & 100% complex recovery (~60% token savings) |
| **HybridFill+Schema** | Selected tool has `array`/`object` params, or abstain | Selected tool + grounded scalars | Plain JSON (`tools=[]`) | Concise-reasoning LLMs when prompt-token cost or ungroundable-turn safety dominates |

### Out-of-Scope Uses
- Standalone deployment without an LLM fallback on open-ended prose generation, multi-step arithmetic/compositional arguments, or unanswerable/adversarial traffic (see **Limitations** below).
- Multi-call turns in a single step (this checkpoint was trained and evaluated on single-call turns).

---

## How to Load & Run Inference

Install the `spancall` package from the [repository](README.md), then:

```bash
pip install -e ".[model,data,agent]" huggingface_hub openai
```

```python
from spancall import SpanCallLLM

caller = SpanCallLLM.from_pretrained(
    "convoicon/spancall-lean-v3-3",
    policy="router_schema",   # llm_only | spancall_only | router | router_schema | hybrid_fill | hybrid_fill_schema
    base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o-mini",
)
out = caller(
    "Flights from Hanoi to Da Nang tomorrow, direct only",
    tools=[...],              # OpenAI function dicts or spancall ToolDefinition objects
    now="2026-10-23",
)
print(out.hit, out.tool_name, out.arguments, out.source)
```

For SpanCall alone (no LLM), use `from spancall import load_resolver; resolver = load_resolver("convoicon/spancall-lean-v3-3")` and call `resolver.try_resolve_candidates(candidate_schemas=[...], utterance=..., now="2026-10-23")`. `load_resolver` reads `head_dim`, `max_cands=1024`, and `resolve_mode="two_step"` from `resolve_config.json` and loads weights with `weights_only=True`.

---

## Training Details

- **Training Corpora:** Single-call turns from **NousResearch Hermes Function-Calling V1** and **Salesforce xLAM-60k** (instances whose gold tool is absent from candidate schemas are filtered out).
- **1:1 Corpus Balancing:** All 69,412 xLAM training records combined with upsampled Hermes (19,096 → 69,412, each copy using a distinct distractor-packing seed with up to 6 tools) into 138,824 records, yielding **117,760** aligned tensorized training instances.
- **Optimization:** 3 epochs in `bf16`, effective batch size 128, cosine learning-rate decay with 10% floor, seed `42`.
- **Model Selection:** Separate held-out dev splits (Hermes 669 rows / 1,239 targets; xLAM 2,384 rows / 4,145 targets) using unweighted macro-average `exact_call`, plus a disjoint 313-row probe evaluated every 2,000 optimizer steps.

| Epoch | Train Loss | Hermes Dev `exact_call` | xLAM Dev `exact_call` | **Macro Dev `exact_call`** | Tool Acc (H / X) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.503 | `.849` | `.864` | `.857` | `.999` / `.991` |
| 2 | 0.354 | `.865` | `.871` | `.868` | `.999` / `.995` |
| **3 (best)** | **0.287** | **`.876`** | **`.872`** | **`.874`** | **`1.000`** / **`.992`** |

---

## Evaluation Results

### Tier 1 Matched Fair Eval (1,735 rows per corpus: 1,277 groundable, 458 ungroundable)

| System | Hermes `exact_F` | xLAM `exact_F` | **Macro `exact_F`** | Macro `param_F1` | Macro Complex `exact_F` | `false_hit_A` ↓ | Mean Lat. (ms) | API Tokens / Turn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SpanCall v3.3** | **`.743`** | `.766` | **`.755`** | **`.801`** | `.331` | **`.664`** | **77.0** | **0** |
| `gemma-4-26b-a4b-it` | `.390` | `.778` | `.584` | `.704` | `.754` | `.895` | 7025.6 | 443 |
| Router (Gemma 4) | `.801` | `.815` | `.808` | `.851` | `.379` | `.975` | 1364.0 | 88 |
| **Router+Schema (Gemma 4)** | **`.826`** | **`.889`** | **`.857`** | **`.889`** | **`.754`** | `.956` | 2565.6 | 179 |
| HybridFill+Schema (Gemma 4) | `.822` | `.866` | `.844` | — | `.698` | **`.853`** | 4721.4 | 345 |
| `gpt-oss-120b` | `.420` | `.777` | `.598` | `.715` | `.746` | `.879` | 2986.7 | 500 |
| Router (gpt-oss-120b) | `.800` | `.814` | `.807` | `.853` | `.376` | `.965` | 657.2 | 102 |
| **Router+Schema (gpt-oss-120b)** | `.821` | **`.889`** | **`.855`** | **`.889`** | **`.746`** | `.936` | 1151.1 | 200 |
| HybridFill+Schema (gpt-oss-120b) | `.823` | `.871` | `.847` | — | `.710` | `.862` | 1446.5 | 177 |

*Note: SpanCall latency is measured locally on a Google Colab A100 (batch 8); hosted LLM latencies are Novita AI API round trips.*

### Normal (All-Scalar) vs. Complex (`array`/`object`) & Error Funnel

- **Normal all-scalar calls** (1,108 per corpus, 86.8% of groundable turns): SpanCall achieves **0.798** on Hermes (vs. 0.360 / 0.397 for Gemma 4 / `gpt-oss-120b`) and **0.840** on xLAM (vs. 0.755 / 0.755).
- **Complex calls** (169 per corpus): SpanCall drops to **0.385** (Hermes) and **0.278** (xLAM), while LLMs reach 0.568–0.586 (Hermes) and 0.923 (xLAM)—motivating `Router+Schema`.
- **Precision upon commitment (`P@hit`) & swaps:** SpanCall reaches **0.806** (Hermes) and **0.824** (xLAM) `P@hit` with **0** duplicate-span collapse and only **4** (Hermes) / **34** (xLAM) entity swaps (vs. 76–79 and 136–162 for the LLMs).

### Tier 2 Full Splits: Zero-Shot OOD & Adversarial Generalization (34,969 Rows)

| Split | Total | Groundable | `exact_F` | `hit_F` | `param_F1` | `false_hit_A` | Mean Lat. (ms) |
| :--- | ---: | ---: | :---: | :---: | :---: | :---: | :---: |
| Hermes ID eval | 1,829 | 1,367 | `.708` | `.873` | `.776` | `.636` | 127.6 |
| xLAM ID eval | 6,811 | 4,571 | `.757` | `.871` | `.793` | `.661` | 46.9 |
| Hermes OOD-Schema | 1,599 | 1,204 | `.711` | `.835` | `.774` | `.648` | 89.6 |
| xLAM OOD-Schema | 7,227 | 5,010 | **`.799`** | **`.887`** | `.753` | `.714` | 43.2 |
| Hermes OOD-Para | 1,342 | 1,341 | `.735` | `.834` | `.779` | `1.00` | 87.4 |
| xLAM OOD-Para | 6,876 | 6,868 | **`.793`** | `.880` | `.780` | `.875` | 38.7 |
| Hermes Adversarial | 4,285 | 0 | — | — | — | `.816` | 164.6 |
| xLAM Adversarial | 5,000 | 0 | — | — | — | `.788` | 48.3 |

---

## Limitations & Safety

> [!NOTE]
> **Verification scope:** `spancall.hub.load_resolver` was checked against current constructor signatures and `SpanCallLLM` was unit-tested across all 6 policies with fakes, but not executed end-to-end against the Hub repo. It assumes the pushed folder contains `model.pt` and `resolve_config.json` (the v3.3 notebook pushes the best-checkpoint directory and writes `resolve_config.json` there). `Router+Schema` rows are produced by `scripts/eval_hybrid_fill.py` (`merge_router_schema`) and are not yet included in the audited `final_report.md` export.

1. **Abstention on ungroundable turns (Stratum A):** Standalone semantic abstention remains weak (`false_hit_on_A` 0.638–0.690 on Tier 1; 0.788–0.816 on adversarial splits). Explicit Stratum A negative supervision is needed before standalone deployment on high-risk or unanswerable traffic.
2. **Complex argument structures:** Linear span extraction cannot reliably model multi-span list cardinality (`array`) or nested `object` grouping; use `Router+Schema` or `HybridFill+Schema` to delegate complex schemas to an LLM.
3. **Single training seed:** Trained with seed `42` (statistical significance on Tier 1 is assessed via 10,000-resample paired bootstrap CIs, not multi-seed training runs).
4. **Single-call scope:** Trained and evaluated on single-call Hermes V1 and xLAM-60k turns; multi-turn stateful benchmarks (BFCL, tau-bench, ToolSandbox) are future work.
