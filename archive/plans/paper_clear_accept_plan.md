# Paper Upgrade Plan: From 4–5 (Borderline) to Clear 6–7+ (Clear Accept)

> **Purpose:** Living tracker separating (a) **writing/clarity confusions** resolved or resolvable in text from (b) **genuine experimental or methodological gaps** needed for a clear accept.
>
> **Current Status (Synchronized with `research_paper/` & `results_lean_v3_3/`):**
> - `research_paper/main.pdf` compiles cleanly at **12 pages**; **Section 6 (`Conclusion and Future Work`) currently lands on Page 9**, meaning the main body exceeds the standard 8-page ACL/EMNLP limit by ~0.5–1 page and needs table relocation to the Appendix (`C5`).
> - **Score Trajectory:** ~4.5–5.0 $\rightarrow$ **~5.3–5.5** after implementing and integrating plain-JSON `HybridFill` / `HybridFill+Schema` (`G6`), hardware/serving clarity (`C4`), and ReAct-style prompt documentation + Appendix A (`C3`). Reaching **Clear 6.5–7.5** still depends on closing **`C1`/`C2`/`C5`** cleanup plus executing **`G2`** (component ablations), **`G3`** (Stratum A abstention head), **`G4`** (BFCL zero-shot + Stratum census), and **`G5`** (multi-seed, error taxonomy, Pareto/Risk-Coverage figures).

---

## 1. Summary Checklist (Confusion vs. Real Gap)

| ID | Issue | Severity | Type | Status / Decision |
|---|---|---|---|---|
| **C1** | `69,412` xLAM + `69,412` Hermes $\neq$ `117,760` total | Moderate (Credibility) | **Explainable in Text** | **Root Cause Verified (`run_metadata.json`) — 1-line `.tex` Fix Needed** |
| **C2** | Hosted LLM `HTTP 429` errors in frozen JSONLs (`241` LLM + `40` HF cache rows) | Fatal if unretried | **Verify / Retry Cleanup** | **Needs Verification / Cleanup** (stale 429 text removed from `06_conclusion_limitations.tex`, but frozen JSONLs still show 429s) |
| **C3** | Zero-shot LLM `false_hit_on_A` (ReAct prompt) & SpanCall `false_hit_on_A` (`0.638–0.816`) | Major (Core Thesis) | **Done (LLM) + Open (`G3`)** | **Done in Paper (LLM prompt & ReAct rationale in Appendix A / §4 / §6)**; **SpanCall abstention (`G3`) open** |
| **C4** | Local GPU batch-8 latency (`77.0 ms`) vs. hosted API round-trip (`3–7 s`) | Moderate (Framing) | **Done in Paper** | **Done** — Colab A100 vs. Novita AI stated in §4 & §6; framed as deployment wall-clock orders of magnitude + API token savings |
| **C5** | ACL/EMNLP 8-page main-body budget overflow (Conclusion starts on Page 9) | Moderate (Compliance) | **Layout Fix Needed** | **Open (Easy Fix)** — move `tab:dev_curve` (§5.1) and `tab:refusal_sweep` (§5.5) to Appendix B |
| **G1** | No fine-tuned small generative baseline (e.g., Qwen2.5-0.5B/1.5B, xLAM-1B) on same data | Deferred | **Skipped for Now** | **Deferred** per user decision (practitioners mostly use pretrained models; optional zero-shot pretrained FC baseline later) |
| **G2** | No component ablation table (two-step, DFS, val-agg, margin gate, type filter, $\Omega$, 1:1 balance) | Fatal (Standard ACL/ICLR) | **Inference + Training Ablations** | **#1 Next Priority** (zero ablations currently in `05_results.tex`) |
| **G3** | No abstention / groundability head ($H_0$ / `NO_CALL`) for Stratum A | Major (Core Thesis) | **New Experiment Needed** | **High Priority** (needed to cut standalone SpanCall `false_hit_on_A` below 0.64–0.82) |
| **G4** | No external standard benchmark (BFCL) + empirical Stratum F/A census | Major (Generalization) | **New Experiment + Audit** | **High Priority (6 $\rightarrow$ 7)** |
| **G5** | Single seed (`42`) + `wVal` error taxonomy + Pareto/Risk-Coverage figures | Moderate (Rigor/Visuals) | **Seeds + Analysis + Figures** | **Medium Priority** |
| **G6** | `HYBRID_FILL` & `HYBRID_FILL+Schema` plain-JSON partial-fill routing | High-ROI (Systems) | **Done in Paper** | **Done** — implemented, tested (7/7), run on `gemma-4-26b` & `gpt-oss-120b`, added to §3.4, §4, §5.8 (`tab:hybrid_fill`), §6, Appendix A |

---

## 2. Clarified Confusions & Cleanup Items (`C1`–`C5`)

### C1. Training Set Arithmetic (`69,412` vs. `117,760`) — **Root Cause Verified (`run_metadata.json`)**
- **What the draft says (`04_experimental_setup.tex:5`):** *"retain 100% of xLAM training data (`69,412` instances) and upsample Hermes (`19,096` $\rightarrow$ `69,412`) for an exact 1:1 balance (`117,760` total)."*
- **Verified pipeline counts (`results_lean_v3_3/run_metadata.json`):**
  - `train_raw_counts`: `hermes = 19,096`, `xlam = 69,412`.
  - `train_balanced_counts`: upsampled 1:1 balanced pool = `69,412` Hermes + `69,412` xLAM = **`138,824`** raw balanced records.
  - `tensorized_t2`: **`117,760`** post-tensorization trainable instances (after filtering/alignment into the t2 training tensor cache).
- **Action:** 1-line fix in `research_paper/sections/04_experimental_setup.tex` to state explicitly: *"upsample Hermes (`19,096` $\rightarrow$ `69,412`) for an exact 1:1 balanced pool (`138,824` records, yielding `117,760` tensorized training instances)."*

### C2. Hosted LLM `HTTP 429` Rate-Limit Errors — **Needs Verification / Cleanup**
- **Current state:** Stale `HTTP 429` disclosure was removed from `06_conclusion_limitations.tex` after the user noted retries were run. **However**, direct inspection of the frozen files in `results_lean_v3_3/` shows rate-limit errors still present:
  - `results_lean_v3_3/llm/`: **`241`** `RateLimitError: 429` rows (`google-gemma-4-26b-a4b-it`: `98` Hermes + `72` xLAM; `openai-gpt-oss-120b`: `38` Hermes + `33` xLAM; `audit_report.md` also flags `WARN: 247 total API error(s)`).
  - `results_lean_v3_3/hybrid_fill/` caches: **`20`** error rows for `google-gemma-4-26b-a4b-it` (`0` Hermes HF, `1` xLAM HF, `4` Hermes HF+S, `15` xLAM HF+S) and **`20`** error rows for `openai-gpt-oss-120b` (`0` Hermes HF, `7` xLAM HF, `3` Hermes HF+S, `10` xLAM HF+S).
- **Action:** Either (a) run `scripts/eval_llm_local.py --retry-errors` and `scripts/eval_hybrid_fill.py --retry-errors`, then run `scripts/export_v3_3_results.py` and refresh tables to achieve **true 0 API errors**, OR (b) if keeping the frozen files as-is, restore a concise 1-sentence disclosure in `06_conclusion_limitations.tex` (noting `Router+Schema` still leads even if every errored LLM row were scored exact).

### C3. Zero-Shot LLM Prompt & Stratum A `false_hit_on_A` — **Done (LLM Side); SpanCall `G3` Open**
- **What changed in paper:**
  - Added **Appendix A (`sections/07_appendix.tex`)** documenting the exact verbatim prompts for both standard `tool_choice=auto` evaluation and plain-JSON `HybridFill`.
  - Updated `04_experimental_setup.tex` and `06_conclusion_limitations.tex` to explain that both LLMs run with a standard ReAct-style agent prompt (no specialized function-calling-only decline instruction), so their `false_hit_on_A` (`0.879–0.895`) reflects default ReAct behavior.
  - `HybridFill+Schema` already cuts router `false_hit_on_A` by **7.4–10.3 pp** (`0.853–0.862`, beating both standalone LLMs) because the JSON extractor can return `{}` on unanswerable turns.
- **Remaining gap:** Standalone `SpanCall` still has `false_hit_on_A = 0.638–0.690` (Tier 1) and `0.788–0.816` (Adversarial). Addressed in **`G3`**.

### C4. Hardware & Latency Framing (`77.0 ms` vs. `3–7 s`) — **Done in Paper**
- **What changed in paper:** Updated `04_experimental_setup.tex` and `06_conclusion_limitations.tex` to explicitly specify **local Google Colab A100 GPU (batch size 8)** for `SpanCall` vs. **Novita AI hosted API** round-trips for both LLMs, framing the latency columns strictly as deployment wall-clock orders of magnitude alongside API token savings rather than identical-hardware FLOPs parity.

### C5. ACL/EMNLP 8-Page Main-Body Budget — **Open (Easy Fix)**
- **Observation:** After adding §5.8 (`tab:hybrid_fill`), `Section 6 (Conclusion and Future Work)` now begins on **Page 9** (`main.pdf` is 12 pages total with References + Appendix A). Standard ACL/EMNLP long-paper limits allow **8 pages** for main body (excluding Limitations, Ethics, References, Appendix).
- **Action:** Move `tab:dev_curve` (§5.1) and `tab:refusal_sweep` (§5.5) into **Appendix B** (leaving concise 2–3 sentence summaries in §5). This recovers ~0.7–1.0 page in the main body, puts Conclusion inside Page 8, and creates room for the **Component Ablation Table (`G2`)** and **Pareto / Risk-Coverage Figure (`G5`)**.

---

## 3. Core Experimental Upgrades (`G1`–`G6`)

### G1. Fine-Tuned Small Generative Baseline — **Deferred**
- **Decision:** Deferred per user guidance — practitioners predominantly consume pretrained models off-the-shelf rather than fine-tuning 0.5B–1.5B FC models per schema distribution.
- **Optional low-cost revisit:** If reviewers later demand a small generative comparison, run a **zero-shot** pretrained small tool-calling baseline (e.g., `xLAM-1B-fc-r` or `Hammer2.1-1.5B`) on Tier 1 using `scripts/eval_llm_local.py` without new training.

### G2. Component Ablation Table — **#1 Next Priority (Fatal Blocker)**
Currently `05_results.tex` has zero architectural ablations for the 5 claimed components. Split cleanly into two sub-tables/groups:
1. **Zero-retraining inference ablations (fast, run directly on the frozen v3.3 checkpoint on Tier 1):**
   - Full `SpanCall` (two-step + exclusivity DFS + value aggregation + type pre-filter + margin gate)
   - `− Two-step re-encoding` (single-pass multi-tool; isolates accuracy drop on 3–5 and 6–10 tool buckets)
   - `− Exclusivity DFS` (independent per-slot argmax; quantifies exact spike in duplicate-span collapse and entity swaps)
   - `− Value aggregation` (raw unaggregated span argmax)
   - `− Margin gate` ($\Delta_{\text{joint}} = 0$; isolates precision vs. coverage tradeoff)
2. **Training ablations (3–4 focused runs on the same 1:1 setup):**
   - `− Type pre-filtering` (score all spans in $\mathcal{C}$ without $\mathcal{A}_k$)
   - `− Canonical context pool Ω` (utterance-only spans; quantifies gains on booleans/enums/dates)
   - `− 1:1 corpus balancing` (raw unbalanced Hermes+xLAM mix; demonstrates why Hermes collapses without balancing)
   - `− Width prior r(j-i)` or `− Multi-positive loss` (single-span CE)

### G3. Dedicated Stratum A Abstention / Groundability Head ($H_0$ / `NO_CALL`) — **High Priority (Major)**
- **Why still needed:** While `HybridFill+Schema` lowers router `falA` to `0.853–0.862`, standalone `SpanCall` still commits on `63.8%–69.0%` of Tier 1 Stratum A and `78.8%–81.6%` of Adversarial splits because it lacks explicit negative supervision.
- **Action:**
  - Train a joint binary groundability/callability head $H_0$ on $\mathbf{h}_{\text{cls}}$ (or a dedicated `NO_CALL` anchor in $H_1$) using **Stratum F (positive) + Stratum A (negative)** examples, calibrated via temperature scaling (`BaseCalibrator`) on dev/calib.
  - Target: drop standalone `false_hit_on_A` dramatically (e.g., `< 0.25`) while maintaining Stratum F recall, and make the abstention-gated Router strictly safer than standalone LLMs.

### G4. External Zero-Shot Benchmark (BFCL) + Empirical Stratum F/A Census — **High Priority (6 $\rightarrow$ 7)**
- **Stratum F/A Census (Fast Data Audit):** Compute and report the natural breakdown of **Stratum F (Normal)**, **Stratum F (Complex)**, and **Stratum A** across raw Hermes V1, xLAM-60k, and BFCL *before* filtering, empirically grounding Section 1's claim about how much real traffic is extractive.
- **Zero-Shot BFCL Evaluation:** Use the existing `bfcl` loader stub (`spancall/data/sources/`) to evaluate `SpanCall` and `Router+Schema` **zero-shot** (no BFCL training) on BFCL `Simple`, `Multiple`, `Live-Simple`, `Live-Multiple`, and `Irrelevance` (directly testing Stratum A abstention on an external standard benchmark).

### G5. Multi-Seed Stability + `wVal` Error Taxonomy + Figures — **Medium Priority**
- **Multi-seed:** Train 2 additional seeds (`1337`, `2025`) for the main `SpanCall` configuration and report `mean ± std`.
- **Error taxonomy:** Manually audit 100 remaining `wVal` errors (50 Hermes, 50 xLAM) into concrete buckets (boundary off-by-one, formatting/normalization, implicit/unexpressed value, array cardinality, nested object grouping, label noise).
- **High-signal figures (replacing main-body `tab:dev_curve` & `tab:refusal_sweep` once moved to Appendix B):**
  - **Figure 2 (Pareto Frontier):** `exact_F` vs. API tokens (and wall-clock latency) across `SpanCall`, `LLM-only`, `Router`, `Router+Schema`, `HybridFill`, and `HybridFill+Schema` for both LLMs.
  - **Figure 3 (Risk-Coverage Curve):** Stratum F recall vs. Stratum A `false_hit_on_A` across threshold $\tau$, comparing raw confidence, $\Delta_{\text{joint}}$, and $H_0$ (`G3`).

### G6. `HYBRID_FILL` & `HYBRID_FILL+Schema` Evaluation — **Done in Paper**
- **Implementation & Verification:** Built lightweight plain-JSON partial-fill evaluator (`spancall/eval/hybrid_fill.py`, `scripts/eval_hybrid_fill.py`, 7/7 tests passing in `tests/test_hybrid_fill_eval.py`). Offloads 100% of tool selection to $H_1$, preserves grounded scalars, drops the OpenAI `tools=[...]` wrapper, and enforces single-shot conservative merge.
- **Completed Results integrated in §3.4, §4, §5.8 (`tab:hybrid_fill`), §6, & Appendix A:**
  - **Prompt tokens & safety:** Cuts multi-tool xLAM prompt tokens by **56.3%–61.3%** (Gemma~4) and **37.8%–41.5%** (`gpt-oss-120b`), and lowers `false_hit_on_A` by **7.4–10.3 pp** (`0.853–0.862`, beating both standalone LLMs at `0.879–0.895`).
  - **On `gpt-oss-120b`:** `HybridFill+Schema` achieves **`0.847` macro `exact_F`** (`0.823` Hermes vs. `0.821` `Router+Schema`; `0.871` xLAM) while cutting total tokens **11.4% below `Router+Schema`** (`177.3` vs. `200.2`; **−19.9%** on xLAM, `128.9` vs. `160.9`, or **64.5%** macro savings vs. LLM-only).
  - **Reasoning-verbosity contrast on `gemma-4-26b`:** JSON-mode reasoning inflates completion tokens (`215.5–285.7` per turn vs. `13.5–33.5`), raising total tokens (`344.9` vs. `178.5`) and latency (`4,721` vs. `2,566 ms`), establishing **`Router+Schema`** (`0.855–0.857` macro) as the accuracy/latency default and **`HybridFill+Schema`** as the prompt-token/safety option when reasoning is concise.

---

## 4. Updated Execution Roadmap (Ordered by Leverage)

1. **Step 1 — Quick Cleanup (15–30 min, fixes credibility & page limit):**
   - Fix **`C1`** in `04_experimental_setup.tex` (`138,824` balanced pool $\rightarrow$ `117,760` tensorized instances).
   - Resolve **`C2`** (`--retry-errors` on the remaining 241 LLM + 40 HybridFill error rows and re-export, or add 1-sentence note).
   - Fix **`C5`** by moving `tab:dev_curve` and `tab:refusal_sweep` to **Appendix B** so the main body fits $\le 8$ pages cleanly.
2. **Step 2 — Component Ablation Table (`G2`, closes biggest reviewer reject trigger):**
   - Write & run the **zero-retraining inference ablation script** on the v3.3 checkpoint (`−two-step`, `−exclusivity DFS`, `−value agg`, `−margin gate`).
   - Run 3–4 focused training ablations (`−type pre-filter`, `−Ω pool`, `−1:1 balance`, `−width prior`).
3. **Step 3 — Breakthrough Upgrades (`G3`, `G4`, `G5`, lifts 6 $\rightarrow$ 7+):**
   - Train & calibrate **$H_0$ / `NO_CALL` Stratum A abstention** + generate **Risk-Coverage curve** (`G3`).
   - Compute **Stratum F/A census** across Hermes, xLAM, and BFCL + run **zero-shot BFCL eval** (`G4`).
   - Run 2 extra seeds (`1337`, `2025`), audit 100 `wVal` errors, and add the **Pareto Frontier figure** (`G5`).
