# Plan: Update `research_paper/` with Latest `v3.2` Train/Eval & LLM Baseline Outputs

## Sources of Truth
1. **Latest `v3.2` Train & Eval**: [`notebooks/spancall_lean_v3_2_train_eval.ipynb`](../notebooks/spancall_lean_v3_2_train_eval.ipynb) (executed Nov 18, 2026; `t2=15,728`, `dev=639`).
2. **LLM Baseline Comparison**: [`notebooks/spancall_lean_v3_2_llm_baseline.ipynb`](../notebooks/spancall_lean_v3_2_llm_baseline.ipynb) (executed Nov 18, 2026; `openai/gpt-oss-120b` via OpenAI SDK on the paired 10% slice: `hermes_eval` $n=166, n_F=120$; `xlam_eval` $n=200, n_F=120$).

---

## 1. Key Deltas: Previous Paper (`v3.2` Nov 17) → Latest Notebooks (`v3.2` Nov 18 + LLM Baseline)

### A. Training Set, Dev Curve & Probe (`spancall_lean_v3_2_train_eval.ipynb`, Cells 3, 8, 10, 11)
- **Data & Packing**:
  - `t2`: `14,076` $\to$ **`15,728` rows** (`hermes_train` 4,021 single-call rows after dropping 3,966 multi-call + `xlam_train` 15,000 rows; `match_missed=1,772`).
  - `dev`: `578` $\to$ **`639` rows** (`1,177` span targets; `hermes_dev` 337 + `xlam_dev` 1,000; 32 multi-call dropped).
  - `probe`: **`300` rows** (`651` span targets, `overlap_with_dev=0`), evaluated every 1,000 optimizer steps.
  - `render_T` distribution (`pack_tools=6`, `jitter_pack_tools=True` over `4,854` schemas, `2,118` collisions): `T=1: 3,043`, `T=2: 1,491`, `T=3: 2,620`, `T=4: 3,665`, `T=5: 3,178`, `T=6: 3,233`, `T=7: 393`, `T=8: 439`.
  - Steps: `2,409` $\to$ **`2,725` steps/epoch** (`8,175` total steps, **`2,044` optimizer steps**), `MAX_POS=1`.
- **Per-Epoch Dev Curve ($n=639$ rows, $1,177$ span targets)**:
  - **Epoch 1**: `span_acc=.777`, `params_ok=.721`, `exact_call=.712`, `exact_T1=.732`, `exact_T2+=.708`, `tool_all=.983`, `tool_T2+=.980`, `train_loss=3.217`, `dev_loss=1.386`
  - **Epoch 2**: `span_acc=.832`, `params_ok=.797`, `exact_call=.790`, `exact_T1=.804`, `exact_T2+=.788`, `tool_all=.989`, `tool_T2+=.987`, `train_loss=1.086`, `dev_loss=1.073`
  - **Epoch 3 (best)**: `span_acc=.844` (`.8437`), `params_ok=.797` (`.7966`), `exact_call=.792` (`.7919`), `exact_T1=.794` (`.7938`), `exact_T2+=.792` (`.7915`), `tool_all=.991` (`.9906`), `tool_T2+=.989` (`.9889`), `train_loss=0.752`, `dev_loss=1.006`
  - **Mid-Training Probe Curve ($n=651$ targets)**: opt-step 1000 `.667`, 2000 `.813`, 3000 `.807`, **4000 `.817` (peak)**, 5000 `.813`.

### B. Full-Split Headline & OOD Results (`spancall_lean_v3_2_train_eval.ipynb`, Cells 19, 27, 29, 33)
| Split | System | $n_F$ | `exact_F` (old $\to$ **new**) | `hit_F` (old $\to$ **new**) | `param_F1` (old $\to$ **new**) | `false_hit_A` (old $\to$ **new**) | `exact_all` (old $\to$ **new**) | E2E Parity | Mean `lat_ms` |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| `hermes_eval` | `SpanCall v3.2` | 1200 | `.636` $\to$ **`.660`** | `.880` $\to$ **`.887`** | `.779` $\to$ **`.797`** | `.482` $\to$ **`.499`** | `.493` $\to$ **`.513`** | `5611/6963 (.806)` | `174.8` |
| `hermes_eval` | `B1a` | 1200 | `.019` | `.028` | `.025` | `.055` | `.015` | — | — |
| `xlam_eval` | `SpanCall v3.2` | 1200 | `.672` $\to$ **`.660`** | `.864` $\to$ **`.856`** | `.712` $\to$ **`.709`** | `.574` $\to$ **`.560`** | `.446` $\to$ **`.440`** | `3669/4983 (.736)` | `59.3` |
| `xlam_eval` | `B1a` | 1200 | `.023` | `.036` | `.008` | `.031` | `.026` | — | — |
| `hermes_test_ood_schema` | `SpanCall v3.2` | 334 | `.626` $\to$ **`.698`** | `.838` $\to$ **`.859`** | `.796` $\to$ **`.818`** | `.542` $\to$ **`.558`** | `.507` $\to$ **`.564`** | `1482/1762 (.841)` | `120.6` |
| `hermes_test_ood_para` | `SpanCall v3.2` | 511 | `.628` $\to$ **`.679`** | `.849` $\to$ **`.857`** | `.776` $\to$ **`.809`** | `1.000` $\to$ **`.000`** | `.627` $\to$ **`.678`** | `2036/2428 (.839)` | `134.6` |
| `hermes_test_adversarial` | `SpanCall v3.2` | 0 | `.000` | `.000` | `.000` | `.437` | `.281` $\to$ **`.285`** | n/a | `182.3` |
| `xlam_test_ood_schema` | `SpanCall v3.2` | 360 | `.786` $\to$ **`.778`** | `.911` $\to$ **`.897`** | `.703` $\to$ **`.701`** | `.604` $\to$ **`.596`** | `.517` $\to$ **`.508`** | `624/662 (.943)` | `52.9` |

*(Note: On `hermes_test_ood_para`, the 1 `A_clean` row is now correctly abstained, moving `false_hit_on_A` from `1.000` to `0.0000`!)*

### C. Waterfall, Stage Attribution & Refusal Forensics (Cells 15, 19, 21, 23, 25)
- **Step 1 (`H1` in isolation, Cell 15)**:
  - `hermes_eval`: Native multi-tool `.9536` (`1213/1272`; `F_func` `.9923`), random distractors $T{=}3$ `.9380` (`363/387`); `F_func` ($n{=}1110$) `tool_prob` p10=`.999`, `A_clean` ($n{=}287$) p10=`.965`.
  - `xlam_eval`: Native multi-tool `.9505` (`1459/1535`; `F_func` `.9947`), random distractors $T{=}3$ `.9419` (`438/465`); `F_func` ($n{=}1200$) `tool_prob` p10=`.999`, `A_clean` ($n{=}600$) p10=`.991`.
- **Step 3 Waterfall & Stage Attribution (Cells 19 & 23)**:
  - `hermes_eval` (`F_func` $N=1,110$): Step 0 `1110 (100%)` $\to$ Step 1 `1102 (99.3%)` $\to$ Step 2a `706 (63.6%)` $\to$ Step 2b `595 (53.6%)` $\to$ Step 2c `564 (50.8%)`. `[A_clean]` abstain `140/287 (48.8%)`, `[A_adv]` sanitize-vetoed `64/172 (37.2%)`.
    - Attribution: `EXACT_HIT` `564 (50.8%)` [normal `530 (54.7%)`, complex `34 (24.1%)`], `STEP2A_PARITY_MISS` `396 (35.7%)` [normal `36.1%`, complex `32.6%`], `STEP2B_POINTER_RANK_MISS` `111 (10.0%)` [normal `5.3%`, complex `42.6%`], `STEP2C_GATE_OR_SCHEMA_REJECT` `31 (2.8%)`, `STEP1_WRONG_TOOL` `8 (0.7%)`.
  - `xlam_eval` (`F_func` $N=1,200$): Step 0 `1200 (100%)` $\to$ Step 1 `1195 (99.6%)` $\to$ Step 2a `964 (80.3%)` $\to$ Step 2b `786 (65.5%)` $\to$ Step 2c `729 (60.8%)`. `[A_clean]` abstain `240/600 (40.0%)`, `[A_adv]` sanitize-vetoed `80/200 (40.0%)`.
    - Attribution: `EXACT_HIT` `729 (60.8%)` [normal `694 (70.0%)`, complex `35 (16.7%)`], `STEP2A_PARITY_MISS` `231 (19.2%)` [normal `9.4%`, complex `66.0%`], `STEP2B_POINTER_RANK_MISS` `178 (14.8%)` [normal `14.5%`, complex `16.3%`], `STEP2C_GATE_OR_SCHEMA_REJECT` `57 (4.8%)`, `STEP1_WRONG_TOOL` `5 (0.4%)`.
- **Normal vs. Complex Call Partition (Full Splits, Cells 19 & 27)**:
  - `hermes_eval`: Normal ($n_F{=}1028$) **`.730`** `exact_F` / `.897` `hit_F` / **`.845`** `param_F1` / `.680` `falA`; Complex ($n_F{=}172$) **`.238`** / `.820` / `.455` / `.329`.
  - `xlam_eval`: Normal ($n_F{=}991$) **`.760`** `exact_F` / `.869` `hit_F` / **`.753`** `param_F1` / `.568` `falA`; Complex ($n_F{=}209$) **`.187`** / `.794` / `.449` / `.495`.
  - `hermes_test_ood_schema`: Normal ($n_F{=}285$) **`.779`** / `.860` / **`.888`**; Complex ($n_F{=}49$) **`.245`** / `.857` / `.380`.
  - `hermes_test_ood_para`: Normal ($n_F{=}461$) **`.722`** / `.859` / **`.834`**; Complex ($n_F{=}50$) **`.300`** / `.840` / `.538`.
  - `xlam_test_ood_schema`: Normal ($n_F{=}360$) **`.778`** / `.897` / **`.701`**.
  - Per-param audit (Cell 21): `hermes_eval` `ALL_NORMAL` `.726` (`2385/3285`) vs `ALL_COMPLEX` `.150` (`188/1254`), **260 rows** fail only on complex params; `xlam_eval` `ALL_NORMAL` `.635` (`2636/4148`) vs `ALL_COMPLEX` `.206` (`78/379`), **213 rows** fail only on complex params.
- **Offline Per-Param Refusal Sweep (`T6`, Cell 25)**:
  - Precision-FP rows: `142/1110` (`joint_margin == 0` on 29 rows); FP reasons: `wrong_value=136`, `param_not_in_gt=28`.
  - Confidence p10/p50/p90: `.162 / .608 / .991`; Margin p10/p50/p90: `.052 / .397 / .986`.
  - Sweep rows: `(0.00, 0.05) -> 27 (8 OK / 19 Bad)`, `(0.00, 0.20) -> 100 (18 OK / 82 Bad, 4.6:1)`, `(0.30, 0.10) -> 89 (18 OK / 71 Bad, 3.9:1)`, `(0.50, 0.10) -> 107 (21 OK / 86 Bad, 4.1:1)`, `(0.50, 0.20) -> 118 (23 OK / 95 Bad, 4.1:1)`, `(0.70, 0.00) -> 163 (41 OK / 122 Bad, 3.0:1)`.
- **Tool-Count Bucket Pivot (Cell 31)**:
  - `hermes_eval`: `1-2` ($n_F{=}1103$) **`.683`** `exact_F` / `.881` `hit_F` / **`.816`** `pF1` (`B1a` `.021`); `3-5` ($n_F{=}97$) **`.402`** / `.948` / `.626` (`B1a` `.000`).
  - `xlam_eval`: `1-2` ($n_F{=}451$) `.634` / `.858` / `.678` (`B1a` `.018`); `3-5` ($n_F{=}624$) **`.683`** / `.846` / `.722` (`B1a` `.030`); `6-10` ($n_F{=}125$) `.640` / **`.896`** / **`.767`** (`B1a` `.008`).

### D. Paired LLM Baseline Comparison (`spancall_lean_v3_2_llm_baseline.ipynb`, Cells 12, 14, 16)
Evaluated on identical rows (`LLM_LIMIT_FRACTION = 0.1`, i.e., `hermes_eval` $n=166, n_F=120$ and `xlam_eval` $n=200, n_F=120$) against `openai/gpt-oss-120b` via OpenAI SDK:

1. **Overall Stratum F, False-Hit & Latency Comparison (Cell 12 & 16)**:
| Split | System | $n$ | $n_F$ | `exact_F` | `hit_F` | `param_F1` | `call_set` | `false_hit_A` | Mean `lat_ms` | `p50_ms` | `p90_ms` | Tokens (in / out) |
| :--- | :--- | ---: | ---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | :---: |
| `hermes_eval` | **`SpanCall v3.2`** | 166 | 120 | **0.625** | **0.917** | **0.754** | **0.506** | **0.5000** | **177.0** | **90.7** | **442.1** | **0 / 0** |
| `hermes_eval` | `gpt-oss-120b` | 166 | 120 | 0.442 | 0.767 | 0.600 | 0.392 | 0.7609 | 1647.9 | 1072.3 | 2863.3 | 67,958 / 32,476 |
| `xlam_eval` | **`SpanCall v3.2`** | 200 | 120 | 0.617 | 0.858 | 0.672 | 0.415 | **0.4375** | **67.4** | **23.3** | **177.2** | **0 / 0** |
| `xlam_eval` | `gpt-oss-120b` | 200 | 120 | **0.767** | **0.950** | **0.814** | **0.740** | 0.9375 | 1366.6 | 855.1 | 1466.1 | 72,325 / 21,559 |

2. **Normal vs. Complex Head-to-Head vs. `gpt-oss-120b` (Cell 14)**:
| Split | Type | $n_{\text{rows}}$ | $n_F$ | `SpanCall` `exact_F` (`hit_F` / `pF1`) | `gpt-oss-120b` `exact_F` (`hit_F` / `pF1`) | Winner |
| :--- | :--- | ---: | ---: | :---: | :---: | :--- |
| `hermes_eval` | **Normal (scalar)** | 132 | 103 | **0.680** (0.951 / **0.790**) | 0.427 (0.748 / 0.604) | **`SpanCall` (+25.3 pp)** |
| `hermes_eval` | Complex (`arr`/`obj`) | 34 | 17 | 0.294 (0.706 / 0.459) | **0.529** (0.882 / **0.575**) | `gpt-oss-120b` (+23.5 pp) |
| `xlam_eval` | **Normal (scalar)** | 163 | 94 | **0.734** (0.872 / 0.735) | 0.713 (0.947 / **0.779**) | **`SpanCall` (+2.1 pp)** |
| `xlam_eval` | Complex (`arr`/`obj`) | 37 | 26 | 0.192 (0.808 / 0.312) | **0.962** (0.962 / **0.987**) | `gpt-oss-120b` (+77.0 pp) |

---

## 2. File-by-File Paper Update Plan

1. **[`research_paper/sections/00_abstract.tex`](../research_paper/sections/00_abstract.tex)**
   - Update full-split `exact_F` (`0.636/0.672` $\to$ **`0.660` on both `hermes_eval` and `xlam_eval`**), `param_F1` (`0.779/0.712` $\to$ **`0.797/0.709`**), OOD `exact_F` range (`0.626–0.786` $\to$ **`0.679–0.778`**), and Normal vs. Complex split (`0.692/0.773` vs `0.302/0.191` $\to$ **`0.730/0.760` vs `0.238/0.187`**).
   - Add the head-to-head `openai/gpt-oss-120b` comparison finding: `SpanCall` matches or exceeds the 120B LLM on scalar calls (`0.680` vs `0.427` on Hermes; `0.734` vs `0.713` on xLAM) at $9.3\times$–$20.3\times$ lower mean latency (`177.0`/`67.4 ms` vs `1,647.9`/`1,366.6 ms`), while the LLM excels on complex array/object calls (`0.529` and `0.962`).

2. **[`research_paper/sections/01_introduction.tex`](../research_paper/sections/01_introduction.tex)**
   - Update E2E span parity (`0.806` on `hermes_eval`, `0.841` on unseen Hermes schemas, `0.943` on unseen xLAM schemas).
   - Update dev `exact_call` (`0.792`, probe peak `0.817`), full-split `exact_F` (`0.660`), OOD range (`0.679–0.778`), and highlight the `gpt-oss-120b` Normal-vs-Complex division of labor and latency speedup.

3. **[`research_paper/sections/03_methodology.tex`](../research_paper/sections/03_methodology.tex)**
   - Fix `pack_tools = 3` $\to$ `pack_tools = 6` in §3.1 (`H1` paragraph) to match §4 and `v3.2` config.

4. **[`research_paper/sections/04_experimental_setup.tex`](../research_paper/sections/04_experimental_setup.tex)**
   - Update training rows (`15,728`), dev rows (`639` rows / `1,177` span targets), disjoint probe (`300` rows / `651` targets every 1,000 opt-steps), `render_T` counts, steps (`2,725`/epoch, `2,044` opt-steps, `MAX_POS=1`), and Normal/Complex slice composition (`hermes_eval` 24.7% complex; `xlam_eval` 15.0% complex).
   - Describe both baselines: `B1a` on all full splits and `openai/gpt-oss-120b` via OpenAI SDK on the paired 10% stratified slice (`n=166` / `n=200`), plus the amortised per-sample latency metric.

5. **[`research_paper/sections/05_results.tex`](../research_paper/sections/05_results.tex)**
   - **§5.1 Table 1 (`tab:dev_curve`)**: Update with the `n=639` (`1,177` slots) 3-epoch curve + mention the 1,000-step probe curve peaking at `0.817`.
   - **§5.2 Table 2 (`tab:headline`) & LLM Baseline Comparison**:
     - Update full-split `SpanCall v3.2` vs. `B1a` numbers (`0.660` on both `hermes_eval` and `xlam_eval`).
     - Add the paired head-to-head table (`tab:llm_parity`) comparing `SpanCall v3.2` vs. `openai/gpt-oss-120b` on `hermes_eval` and `xlam_eval` (`exact_F`, `hit_F`, `param_F1`, `false_hit_A`, mean/p50/p90 latency, and token counts).
   - **§5.3 Error Attribution**: Update Step 1 (`99.3%` / `99.6%`), Step 2a (`63.6%` / `80.3%`), Step 2b (`53.6%` / `65.5%`), Step 2c (`50.8%` / `60.8%`), stage attribution counts, and `A_clean` abstention (`48.8%` / `40.0%`).
   - **§5.4 Normal vs. Complex Partition (`tab:complexity` & `tab:refusal_sweep`)**:
     - Update full-split Normal/Complex rows in `tab:complexity` (`0.730` vs `0.238` on Hermes; `0.760` vs `0.187` on xLAM; `0.779` vs `0.245` on `ood_schema`; `0.722` vs `0.300` on `ood_para`) and integrate the paired `SpanCall` vs `gpt-oss-120b` Normal/Complex comparison (`0.680` vs `0.427` and `0.734` vs `0.713` on Normal; `0.294` vs `0.529` and `0.192` vs `0.962` on Complex).
     - Update `tab:refusal_sweep` and precision-FP forensics (`142` rows, `136 wrong_value`, `28 param_not_in_gt`, `4.1:1`–`4.6:1` bad-to-good ratio).
   - **§5.5 Zero-Shot OOD (`tab:ood`)**: Update all OOD split metrics (`0.698`, `0.679`, `0.285`, `0.778`).
   - **§5.6 Tool-Count Buckets (`tab:buckets`)**: Update bucket numbers (`0.683` / `0.402` on Hermes; `0.634` / `0.683` / `0.640` on xLAM).

6. **[`research_paper/sections/06_conclusion_limitations.tex`](../research_paper/sections/06_conclusion_limitations.tex)**
   - Synchronize all summary metrics, residual complex-only failure counts (`260` on Hermes, `213` on xLAM), refusal sweep ratios (`4.1:1`–`4.6:1`), and the empirical validation from the `gpt-oss-120b` comparison showing why hybrid routing (`SpanCall` for scalars + LLM for complex/abstained calls) is the natural operating point.

7. **[`research_paper/PAPER_WRITING_GUIDE.md`](../research_paper/PAPER_WRITING_GUIDE.md)**
   - Synchronize all reference tables and locked numbers with the Nov 18 notebooks.

8. **Compile & Verify (`Makefile`)**
   - Run `make` in `research_paper/` (`pdflatex` $\times 3$ + `bibtex`) and verify zero undefined references, no overfull tables, and a clean `main.pdf`.
