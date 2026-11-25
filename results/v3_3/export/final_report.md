# SpanCall Lean v3.3 — Benchmark Report

> **Audit:** ⚠️ PASS_WITH_LLM_ERRORS (247 LLM API errors) · **Commit:** `e568daf` · **Eval:** `data/eval_v3_3` (1:1 matched Hermes & xLAM) · **Bootstrap:** `10,000` resamples

---

## 1. Headline Comparison (Tier 1 Fair Eval — Side by Side)

Both `fair_hermes_eval` and `fair_xlam_eval` have **identical** size and `(cohort × complexity)` composition.

### 1a. Accuracy (`exact_F` & `param_F1` on Stratum F)

| System | Hermes `exact_F` | xLAM `exact_F` | **Macro `exact_F`** | Hermes `param_F1` | xLAM `param_F1` | **Macro `param_F1`** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **SpanCall v3.3** (fast path) | 74.3% | 76.6% | **75.5%** | 80.4% | 79.8% | **80.1%** |
| LLM (`google-gemma-4-26b-a4b-it`) | 39.0% | 77.8% | **58.4%** | 59.5% | 81.3% | **70.4%** |
| **Router** (`SC + google-gemma-4-26b-a4b-it`) | 80.1% | 81.5% | **80.8%** | 86.1% | 84.1% | **85.1%** |
| LLM (`openai-gpt-oss-120b`) | 42.0% | 77.7% | **59.8%** | 60.9% | 82.1% | **71.5%** |
| **Router** (`SC + openai-gpt-oss-120b`) | 80.0% | 81.4% | **80.7%** | 86.2% | 84.3% | **85.3%** |

### 1b. Speed, Cost & Safety (Macro Average across Hermes + xLAM)

| System | Mean Lat. (`p50` / `p90`) | Speedup vs. LLM | Tokens / Row (`in + out`) | Token Savings | LLM Calls | `hit_F` | `false_hit_A` | API Errs |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **SpanCall v3.3** (fast path) | 77.0 ms (`48.1` / `122.1`) | **38.8–91.3×** | 0.0 (`0+0`) | **100.0%** | 0.0% | 88.1% | 66.4% | 0 |
| LLM (`google-gemma-4-26b-a4b-it`) | 7025.6 ms (`6414.1` / `11776.4`) | 1.0× | 443.4 (`400+44`) | 0.0% | 100.0% | 83.8% | 89.5% | 175 |
| **Router** (`SC + google-gemma-4-26b-a4b-it`) | 1364.0 ms (`56.9` / `6618.6`) | **5.2×** | 88.1 (`75+13`) | **80.1%** | 17.6% | 99.3% | 97.5% | 32 |
| LLM (`openai-gpt-oss-120b`) | 2986.7 ms (`1112.8` / `8995.1`) | 1.0× | 499.9 (`368+132`) | 0.0% | 100.0% | 87.0% | 87.9% | 72 |
| **Router** (`SC + openai-gpt-oss-120b`) | 657.2 ms (`56.9` / `1384.4`) | **4.5×** | 101.8 (`69+33`) | **79.6%** | 17.6% | 99.3% | 96.5% | 13 |

### 1c. Paired Bootstrap 95% CIs (`exact_F` Delta)

| Corpus | Comparison | $n_F$ | Delta | 95% CI | Stat. Sig.? |
| :--- | :--- | ---: | ---: | :---: | :---: |
| Hermes | `spancall_v3_3 - llm:google-gemma-4-26b-a4b-it` | 1277 | **+35.3 pp** | `[+31.25, +39.31] pp` | ✅ Yes |
| Hermes | `router:google-gemma-4-26b-a4b-it - llm:google-gemma-4-26b-a4b-it` | 1277 | **+41.1 pp** | `[+37.43, +44.71] pp` | ✅ Yes |
| Hermes | `spancall_v3_3 - llm:openai-gpt-oss-120b` | 1277 | **+32.3 pp** | `[+28.35, +36.41] pp` | ✅ Yes |
| Hermes | `router:openai-gpt-oss-120b - llm:openai-gpt-oss-120b` | 1277 | **+38.0 pp** | `[+34.30, +41.58] pp` | ✅ Yes |
| xLAM | `spancall_v3_3 - llm:google-gemma-4-26b-a4b-it` | 1277 | **-1.2 pp** | `[-4.23, +1.88] pp` | — Tie |
| xLAM | `router:google-gemma-4-26b-a4b-it - llm:google-gemma-4-26b-a4b-it` | 1277 | **+3.8 pp** | `[+1.02, +6.58] pp` | ✅ Yes |
| xLAM | `spancall_v3_3 - llm:openai-gpt-oss-120b` | 1277 | **-1.1 pp** | `[-4.07, +1.96] pp` | — Tie |
| xLAM | `router:openai-gpt-oss-120b - llm:openai-gpt-oss-120b` | 1277 | **+3.7 pp** | `[+0.94, +6.42] pp` | ✅ Yes |

---

## 2. Normal vs. Complex Calls (`exact_F` Side by Side)

| Corpus | Type | $n_F$ | SpanCall v3.3 | LLM (google-gemma-4-26b-a4b-it) | Router (google-gemma-4-26b-a4b-it) | LLM (openai-gpt-oss-120b) | Router (openai-gpt-oss-120b) | Δ (`SpanCall − LLM`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hermes | Normal | 1108 | 79.8% | 36.0% | **86.4%** | 39.7% | 86.1% | +43.8 pp |
| Hermes | Complex | 169 | 38.5% | **58.6%** | 39.1% | 56.8% | 39.6% | -20.1 pp |
| xLAM | Normal | 1108 | 84.0% | 75.5% | **88.4%** | 75.5% | **88.4%** | +8.5 pp |
| xLAM | Complex | 169 | 27.8% | **92.3%** | 36.7% | **92.3%** | 35.5% | -64.5 pp |

---

## 3. Tool-Count Bucket Breakdown (`exact_F` Side by Side)

| Corpus | Tools | $n_F$ | SpanCall v3.3 | LLM (google-gemma-4-26b-a4b-it) | Router (google-gemma-4-26b-a4b-it) | LLM (openai-gpt-oss-120b) | Router (openai-gpt-oss-120b) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hermes | `1-2` | 1165 | 78.2% | 36.0% | **84.4%** | 38.6% | 84.1% |
| Hermes | `3-5` | 112 | 33.9% | 70.5% | 35.7% | **76.8%** | 36.6% |
| Hermes | **`shared 1-5`** | 1277 | 74.3% | 39.0% | **80.1%** | 42.0% | 80.0% |
| xLAM | `1-2` | 512 | 77.7% | 77.1% | **83.6%** | 77.0% | 83.0% |
| xLAM | `3-5` | 637 | 75.8% | 79.0% | **81.2%** | 78.3% | **81.2%** |
| xLAM | `6-10` | 128 | 75.8% | 74.2% | 75.0% | **77.3%** | 75.8% |
| xLAM | **`shared 1-5`** | 1149 | 76.7% | 78.2% | **82.2%** | 77.7% | 82.0% |

---

## 4. Full ID, OOD & Adversarial Generalization (SpanCall v3.3)

| Corpus | Split | $n$ | $n_F$ | `exact_F` | `hit_F` | `param_F1` | `false_hit_A` | Mean Lat. (`p50`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hermes | Full ID Eval | 1,829 | 1,367 | **70.8%** | 87.3% | 77.6% | 63.6% | 127.6 ms (`72.6`) |
| xLAM | Full ID Eval | 6,811 | 4,571 | **75.7%** | 87.1% | 79.3% | 66.1% | 46.9 ms (`26.6`) |
| Hermes | OOD Schema | 1,599 | 1,204 | **71.1%** | 83.5% | 77.4% | 64.8% | 89.6 ms (`60.0`) |
| xLAM | OOD Schema | 7,227 | 5,010 | **79.9%** | 88.7% | 75.3% | 71.4% | 43.2 ms (`26.3`) |
| Hermes | OOD Paraphrase | 1,342 | 1,341 | **73.5%** | 83.4% | 77.9% | 100.0% | 87.4 ms (`55.6`) |
| xLAM | OOD Paraphrase | 6,876 | 6,868 | **79.3%** | 88.0% | 78.0% | 87.5% | 38.7 ms (`24.7`) |
| Hermes | Adversarial (A-only) | 4,285 | 0 | — | — | — | 81.6% | 164.6 ms (`81.3`) |
| xLAM | Adversarial (A-only) | 5,000 | 0 | — | — | — | 78.8% | 48.3 ms (`29.9`) |

---

## 5. Training & Held-Out Dev Curve

Backbone: `answerdotai/ModernBERT-large` · Balanced 1:1 train: Hermes `19,096` → `69,412`, xLAM `69,412` (`t2 = 117,760` tensorized rows).

| Epoch | Train Loss | `dev_hermes` `exact_call` | `dev_xlam` `exact_call` | **Macro Dev `exact_call`** | Best |
| :---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 1.5027 | 84.9% | 86.4% | **85.7%** |  |
| 2 | 0.3538 | 86.5% | 87.1% | **86.8%** |  |
| 3 | 0.2874 | 87.6% | 87.2% | **87.4%** | ⭐ |

---

## 6. Audit Trail & Details

<details>
<summary>Mid-training probe curve (21 checkpoints)</summary>

| Opt Step | Epoch | `n` | `exact_call` | `span_acc` | `tool_acc` | `eval_loss` |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 1 | 559 | 36.4% | 49.4% | 87.9% | 3.4664 |
| 4,000 | 1 | 559 | 59.7% | 71.2% | 98.7% | 2.0374 |
| 6,000 | 1 | 559 | 73.8% | 82.1% | 99.0% | 1.4672 |
| 8,000 | 1 | 559 | 79.6% | 86.2% | 99.4% | 1.2550 |
| 10,000 | 1 | 559 | 80.2% | 86.9% | 99.7% | 1.1828 |
| 12,000 | 1 | 559 | 82.7% | 88.9% | 99.7% | 1.1018 |
| 14,000 | 1 | 559 | 83.1% | 89.4% | 99.7% | 1.0903 |
| 16,000 | 2 | 559 | 85.3% | 90.7% | 99.7% | 1.0283 |
| 18,000 | 2 | 559 | 85.0% | 90.5% | 99.7% | 1.0183 |
| 20,000 | 2 | 559 | 84.7% | 90.3% | 100.0% | 0.9849 |
| 22,000 | 2 | 559 | 85.6% | 90.5% | 100.0% | 1.0411 |
| 24,000 | 2 | 559 | 85.9% | 90.7% | 100.0% | 0.9648 |
| 26,000 | 2 | 559 | 85.6% | 91.1% | 100.0% | 0.9797 |
| 28,000 | 2 | 559 | 86.3% | 91.2% | 100.0% | 0.9180 |
| 30,000 | 3 | 559 | 88.2% | 92.8% | 100.0% | 0.9325 |
| 32,000 | 3 | 559 | 86.9% | 91.8% | 100.0% | 0.9651 |
| 34,000 | 3 | 559 | 86.9% | 92.1% | 100.0% | 0.9423 |
| 36,000 | 3 | 559 | 85.6% | 91.2% | 100.0% | 0.9552 |
| 38,000 | 3 | 559 | 85.9% | 91.2% | 100.0% | 0.9541 |
| 40,000 | 3 | 559 | 85.9% | 91.6% | 100.0% | 0.9617 |
| 42,000 | 3 | 559 | 85.9% | 91.8% | 99.7% | 0.9598 |

</details>

<details>
<summary>Frozen dataset manifest & SHA256 hashes (12 splits)</summary>

- **Manifest SHA256:** `3a78c8fcb5fcb6b7f29deee7d68e27612173652483e0e0753a4403846f3bc534`
- **Template Stamp:** `1e62a355a351ae531c64f137a5d66cf873de81c2987829243a1559cda6b25cf2`
- **Generated (UTC):** `2026-11-20T06:35:58.960842+00:00`

| Split | Tier | Corpus | `n` | SHA256 (prefix) |
| :--- | :--- | :--- | ---: | :---: |
| `full_hermes_eval` | `tier2_full` | `hermes` | 1,829 | `df46d8dfcb9c` |
| `full_xlam_eval` | `tier2_full` | `xlam` | 6,811 | `dbf27d300a9c` |
| `full_hermes_test_ood_schema` | `tier2_full` | `hermes` | 1,599 | `bd8fbfc36b34` |
| `full_xlam_test_ood_schema` | `tier2_full` | `xlam` | 7,227 | `4fd3efe0742a` |
| `full_hermes_test_ood_para` | `tier2_full` | `hermes` | 1,342 | `1ad9247f0adb` |
| `full_xlam_test_ood_para` | `tier2_full` | `xlam` | 6,876 | `452e0947f908` |
| `full_hermes_test_adversarial` | `tier2_full` | `hermes` | 4,285 | `3d0ee362c724` |
| `full_xlam_test_adversarial` | `tier2_full` | `xlam` | 5,000 | `8a3e3eae389b` |
| `fair_hermes_eval` | `tier1_fair` | `hermes` | 1,735 | `72f80719c275` |
| `fair_xlam_eval` | `tier1_fair` | `xlam` | 1,735 | `b34ca47b7e99` |
| `fair_eval_combined` | `tier1_fair` | `combined` | 3,470 | `7acbac7caa3b` |
| `hermes_eval_f_json` | `auxiliary` | `hermes` | 90 | `2b091ba29f07` |

</details>
