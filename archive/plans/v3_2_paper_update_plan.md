# Plan: Update `research_paper/` with Nov 19 Full-Scale (`v3.2` + LLM Baseline) Outputs

## Sources of Truth
1. **Latest `v3.2` Train & Eval**: [`notebooks/spancall_lean_v3_2_train_eval.ipynb`](../notebooks/spancall_lean_v3_2_train_eval.ipynb) (executed Nov 19, 2026; `DATA_FRACTION=1`, `t2=28,991`, `dev=1,181`).
2. **LLM Baseline Comparison**: [`notebooks/spancall_lean_v3_2_llm_baseline.ipynb`](../notebooks/spancall_lean_v3_2_llm_baseline.ipynb) (executed Nov 19, 2026; `LLM_LIMIT_FRACTION=1.0`, `openai/gpt-oss-120b` via OpenAI SDK on 100% of `hermes_eval` $n=1,659, n_F=1,200$ and `xlam_eval` $n=2,000, n_F=1,200$, **0 API errors**).

---

## 1. Locked Numbers (Nov 19 Full Runs)

### A. Training Set, Dev Curve & Probe (`spancall_lean_v3_2_train_eval.ipynb`, Cells 3, 8, 10, 11)
- **Data & Packing**:
  - `t2`: **`28,991` rows** (`hermes_train` 20,105 single-call rows after dropping 3,966 multi-call + `xlam_train` 15,000 rows; `match_missed=4,085`).
  - `dev`: **`1,181` rows** (`2,167` span targets; `hermes_dev` 1,688 + `xlam_dev` 1,000; 32 multi-call dropped).
  - `probe`: **`300` rows** (`651` span targets, `overlap_with_dev=0`), evaluated every 1,000 optimizer steps.
  - `render_T` distribution (`pack_tools=6`, `jitter_pack_tools=True` over `4,854` schemas, `2,118` collisions): `T=1: 5,276`, `T=2: 3,623`, `T=3: 5,292`, `T=4: 6,314`, `T=5: 5,488`, `T=6: 5,758`, `T=7: 393`, `T=8: 439`.
  - Steps: **`5,238` steps/epoch** (`15,714` total steps, **`3,929` optimizer steps**), `MAX_POS=1`.
- **Per-Epoch Dev Curve ($n=1,181$ rows, $2,167$ span targets)**:
  - **Epoch 1**: `span_acc=.811`, `params_ok=.754`, `exact_call=.750`, `exact_T1=.562`, `exact_T2+=.783`, `tool_all=.993`, `tool_T2+=.992`, `train_loss=2.746`, `dev_loss=1.327`
  - **Epoch 2**: `span_acc=.851`, `params_ok=.804`, `exact_call=.798`, `exact_T1=.693`, `exact_T2+=.816`, `tool_all=.992`, `tool_T2+=.991`, `train_loss=0.784`, `dev_loss=1.102`
  - **Epoch 3 (best)**: `span_acc=.863` (`.8634`), `params_ok=.818` (`.8180`), `exact_call=.814` (`.8137`), `exact_T1=.739`, `exact_T2+=.827`, `tool_all=.994` (`.9941`), `tool_T2+=.993`, `train_loss=0.551`, `dev_loss=1.058`
  - **Mid-Training Probe Curve ($n=651$ targets)**: step 1k `.313`, 2k `.727`, 3k `.770`, **4k `.797` (peak)**, 5k `.770`, 6k `.793`, 7k `.783`, 8k `.783`, 9k `.783`, 10k `.790`.

### B. Full-Split Headline & OOD Results (`train_eval` Cells 19, 27, 29, 33 + `llm_baseline` Cells 12, 16)
| Split | System | $n_F$ | `exact_F` | `hit_F` | `param_F1` | `false_hit_A` | `exact_all` | E2E Parity | Mean `lat_ms` (`p90`) |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| `hermes_eval` | `B1a` | 1200 | `.019` | `.028` | `.025` | `.055` | `.015` | — | — |
| `hermes_eval` | `gpt-oss-120b` | 1200 | `.423` | `.782` | `.600` | `.830` | `.397` (`call_set`) | — | `1365.2` (`2156.4`) |
| `hermes_eval` | **`SpanCall v3.2`** | 1200 | **`.733`** | **`.891`** | **`.855`** | `.510` (`.508`*) | **`.570`** | `5631/6985 (.806)` | **`170.2`** (`438.5`) |
| `xlam_eval` | `B1a` | 1200 | `.023` | `.036` | `.008` | `.031` | `.026` | — | — |
| `xlam_eval` | `gpt-oss-120b` | 1200 | **`.818`** | **`.989`** | **`.846`** | `.984` | `.769` (`call_set`) | — | `1015.5` (`1321.0`) |
| `xlam_eval` | **`SpanCall v3.2`** | 1200 | `.666` | `.862` | `.711` | **`.568`** | `.445` | `3661/4974 (.736)` | **`54.3`** (`98.7`) |
| `hermes_test_ood_schema` | **`SpanCall v3.2`** | 334 | **`.740`** | `.859` | **`.863`** | `.600` | **`.604`** | `1482/1762 (.841)` | `120.7` |
| `hermes_test_ood_para` | **`SpanCall v3.2`** | 511 | **`.759`** | `.875` | **`.867`** | `1.000` | **`.758`** | `2048/2440 (.839)` | `134.0` |
| `hermes_test_adversarial` | **`SpanCall v3.2`** | 0 | `.000` | `.000` | `.000` | `.463` | **`.327`** | n/a | `179.5` |
| `xlam_test_ood_schema` | **`SpanCall v3.2`** | 360 | **`.750`** | `.900` | `.660` | `.617` | **`.493`** | `612/651 (.940)` | `52.2` |

### C. Waterfall, Stage Attribution, Complexity & Refusal (`train_eval` Cells 15, 19, 21, 23, 25, 31 + `llm_baseline` Cell 14)
- **Step 1 (`H1` in isolation)**: `hermes_eval` `F_func` `99.7%` (`1107/1110`); `xlam_eval` `99.0%` (`1188/1200`).
- **Stage Attribution**:
  - `hermes_eval` ($N=1,110$): `EXACT_HIT 611 (55.0%)` [N `548 (56.6%)`, C `63 (44.7%)`], `STEP2A_PARITY_MISS 398 (35.9%)`, `STEP2B_POINTER_RANK_MISS 68 (6.1%)`, `STEP2C_GATE_OR_SCHEMA_REJECT 30 (2.7%)`, `STEP1_WRONG_TOOL 3 (0.3%)`; `[A_clean]` `137/287 (47.7%)`.
  - `xlam_eval` ($N=1,200$): `EXACT_HIT 732 (61.0%)` [N `699 (70.5%)`, C `33 (15.8%)`], `STEP2A_PARITY_MISS 230 (19.2%)`, `STEP2B_POINTER_RANK_MISS 167 (13.9%)`, `STEP2C_GATE_OR_SCHEMA_REJECT 59 (4.9%)`, `STEP1_WRONG_TOOL 12 (1.0%)`; `[A_clean]` `236/600 (39.3%)`.
- **Normal vs. Complex (100% Paired)**:
  - `hermes_eval`: Normal ($n_F=1028$) **`.786`** SC vs `.407` LLM (+37.9 pp); Complex ($n_F=172$) `.413` SC vs **`.523`** LLM.
  - `xlam_eval`: Normal ($n_F=991$) `.768` SC vs **`.793`** LLM (within 2.5 pp); Complex ($n_F=209$) `.182` SC vs **`.938`** LLM.
  - `hermes_ood_sch`: Normal `.832`, Complex `.265`; `hermes_ood_para`: Normal `.796`, Complex `.420`.
  - Residual complex-only failures: `237` (`hermes_eval`), `208` (`xlam_eval`).
- **Refusal Sweep (`T6`)**: `98/1110` FP rows (`77 wrong_value`, `19 param_not_in_gt`), median conf `.775` / margin `.662`. Sweep: `11 (2/9)`, `36 (5/31)`, `36 (7/29)`, `45 (9/36)`, `50 (9/41)`, `77 (14/63)`.
- **Tool-Count Buckets**: `hermes_eval` `1-2` `.753`, `3-5` `.505`; `xlam_eval` `1-2` `.650`, `3-5` `.683`, `6-10` `.640`.

---

## 2. Verification Summary
- All sections updated and compiled cleanly via `make clean && make` in `research_paper/`.
- Duplicate 10%-slice tables (`tab:llm_parity`, `tab:llm_complexity`) merged into `tab:headline` and `tab:complexity` to satisfy the 8-page ACL main-content limit.
