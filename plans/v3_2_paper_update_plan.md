# Plan: Update `research_paper/` from v2.7 → v3.2 experiment numbers

Source of truth: executed outputs of `notebooks/spancall_lean_v3_2_train_eval.ipynb`.
Target: replace all v2.7-locked numbers in the ACL LaTeX paper (no v2.7-vs-v3.2 comparison).

## Locked v3.2 numbers (from notebook outputs)

### Training / dev curve (n=1,058 probe-style dev eval; note dev set = 578 rows, dev eval n=1058 per curve)
- Epoch 1: span .773, par .740, exact .721, exT1 .716, exT2+ .722, tool .978, tr_L 3.0612, dv_L 1.3869
- Epoch 2: span .807, par .784, exact .773, exT1 .784, exT2+ .771, tool .986, tr_L 1.0653, dv_L 1.0419
- Epoch 3 (best): span .822, par .806, exact .799, exT1 .795, exT2+ .800, tool .991, tr_L 0.7458, dv_L 1.0373
- Config: t2 = 14,076 rows, dev = 578 rows, 3 epochs, 2,409 steps/epoch, 1,807 opt steps, bs 32 × accum 4 (eff 128), L4, bf16, seed 42
- Deltas vs paper's v2.7: 20,779→14,076 rows, 850→578 dev, 1,564→(2,409 steps/epoch)

### Headline hermes_eval (n_F=1200, F_func n=1200 / A_clean n=600)
- B1a: exact_F .019, hit .028, param_F1 .025, false_hit_A .0545, exact_all .015
- Lean two_step: exact_F .636, hit .880, collapse 0.0, param_F1 .779, false_hit_A .4815, exact_all .493
- v3.2 is two-step-only → headline table becomes B1a vs SpanCall v3.2 (two_step); drop v2.5/v2.6/v2.7 progression rows and the single_pass/two_step split

### xlam_eval (n=1200)
- lean two_step: exact_F .672, hit .864, param_F1 .712, false_hit_A .5737, exact_all .446; B1a exact .023

### Complexity partition (replaces per-param-type Table 4)
- hermes_eval: normal n=1028 exact .692 / hit .888 / param_F1 .823; complex n=172 exact .302 / hit .831 / param_F1 .464
- xlam_eval: normal n=991 exact .773; complex n=209 exact .191 / hit .833 / param_F1 .445
- OOD: ood_schema normal n=285 .698 / complex n=49 .184; ood_para normal n=461 .662 / complex n=50 .360
- Normal = string/integer/number/boolean/date/enum; Complex = array/object/object-leaf

### OOD table (two_step, n per split)
- hermes_test_ood_schema: n=334, exact .626, hit .838, param_F1 .796
- hermes_test_ood_para: n=511, exact .628
- hermes_test_adversarial: false_hit_A .4371, exact_all .281
- xlam_test_ood_schema: n=360, exact .786, hit .911, param_F1 .703
- e2e_parity: hermes_eval .807, xlam_eval .736, ood_schema .842, ood_para .839, xlam_ood .941

### Tool-count buckets
- hermes_eval: 1-2 n=1103 exact .665; 3-5 n=97 exact .299
- xlam_eval: 1-2 n=451 .654; 3-5 n=624 .689; 6-10 n=125 .648

### Not available in v3.2 notebook
- Stage-level latency table (no p50/enc_ms measurements) → keep v2.7 latency table with footnote, or drop section (pending user choice)
- Waterfall/oracle-ceiling step numbers (v2.7 §5.3): v3.2 has Step 2a/2b/2c waterfall for F_func 1110 (63.5/51.8/49.1) and xlam 1200 (80.2/66.2/61.7) → replace
- Refusal sweep table: v3.2 has confidence/margin percentiles (conf p10=.174 p50=.613; margin p10=.056 p50=.420) but no full (τ_conf, τ_margin) sweep grid; FP reasons wrong_value 159 / param_not_in_gt 37 → replace narrative numbers, drop or re-cast Table 5

## Section-by-section edits

1. **00_abstract.tex**: v2.7→v3.2; 0.649 exact_F → 0.636; 0.818 param F1 → 0.779; 0.734–0.775 OOD → 0.626–0.786; drop/soften the 15.1 ms / 90.4 ms latency claim (no v3.2 timing); add Normal-vs-Complex framing (0.692/0.302 hermes; 0.773/0.191 xlam); "4,854 schemas" → recompute or drop.
2. **01_introduction.tex** contribution bullets: 0.780 dev → 0.799; 0.649 → 0.636; OOD range 0.734–0.775 → 0.626–0.786; coverage 21.6%→87.8% claim (v2.7) → replace with v3.2 two-step story or drop; update latency claim.
3. **03_methodology.tex**: `(\texttt{v2.7})` → `(\texttt{v3.2})` (2 places, incl. architecture figure caption); "two-step variant" wording fine; object-leaf decomposition bullet stays.
4. **04_experimental_setup.tex**: training rows 20,779→14,076; dev 850→578; steps 1,564→2,409/epoch (1,807 opt steps); evaluation protocol: 500-row/415 single-call slice → v3.2 full splits (hermes_eval n_F=1200, F_func 1200/A_clean 600; xlam_eval n=1200; ood_schema 334; ood_para 511; adversarial; xlam ood 360); remove v2.5/v2.6 baseline mentions; keep B1a; note two-step-only protocol and Normal/Complex stratification.
5. **05_results.tex**:
   - Table 1 (dev_curve): new 3-epoch numbers above; narrative rewritten (tool .978→.991; exact .721→.799; losses 3.06→0.75 train, 1.387→1.037 dev).
   - Table 2 (headline): B1a row + single **v3.2 (2-step)** row (n_F 1200, exF .636, hit .880, pF1 .779, falA .482, exAll .493); remove version progression; optionally add xlam_eval row .672.
   - §5.3 error attribution: replace with v3.2 waterfall (hermes F_func: 2a 63.5%, 2b 51.8%, 2c 49.1%; xlam: 80.2/66.2/61.7) and H1 separation stats (F_func n=1200 tool_prob p10 .999; A_clean n=600 p10 .986).
   - Table 4 (param_type) → **Normal/Complex complexity table**: hermes normal/complex, xlam normal/complex, OOD normal/complex rows; narrative on 0.692 vs 0.302 etc.
   - Table 5 (refusal sweep): replace with confidence/margin separation percentiles or drop; narrative updated (FP reasons 159/37).
   - Table 6 (ood): new n and numbers; adversarial row falA .437/exAll .281; xlam_ood .786/.911/.703; caption v3.2.
   - §Latency: pending user decision.
6. **06_conclusion_limitations.tex**: 0.780→0.799 dev; 0.649→0.636; 0.818→0.779; OOD 0.743/0.775→0.626/0.786; limitations: replace scalar 72.4–82.4%/array 11.5%/object 0.0% per-type claims with Normal 0.692–0.773 vs Complex 0.191–0.302 partition; false_hit_on_A 0.346–0.353 → 0.48–0.57 (worse; strengthen limitation); latency claim softened; "82 calls fail solely on complex types" → 172/1200 hermes + 209/1200 xlam fail only on complex.
7. **PAPER_WRITING_GUIDE.md**: replace all locked v2.7 numbers with the v3.2 set above; update "two modes" → two-step-only; note unavailable latency table.
8. Recompile `make` (pdflatex ×3 + bibtex) and check for undefined refs.

## Resolved
- Latency section: DROP the latency table (Table 7) and all timing claims (15.1 ms / 90.4 ms) per user decision. Remove from abstract, intro contribution bullet, conclusion, results section, PAPER_WRITING_GUIDE.md.
