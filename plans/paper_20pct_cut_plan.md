# Plan: Cut ~20% of the Paper While Keeping Essentials

Target: reduce `research_paper/` body text by ~20% (tables largely preserved; prose is the cut surface). All numbers, claims, and the logical flow (Stratum F framing → architecture → fair eval → router) stay intact.

## Current state (approx. word counts of prose-heavy sections)

| File | Role | Action |
|---|---|---|
| `sections/00_abstract.tex` | ~340 words, number-dense | Trim ~30% (drop duplicate Router/Router+Schema breakdowns, keep headline: 0.755 macro, +32–35pp Hermes, 38.8–91.3× faster, Router+Schema 0.855–0.857, 60% token savings) |
| `sections/01_introduction.tex` | ~600 words | Trim ~25%: shorten the JSON-token counting sentence, condense "Why not just NER" paragraph, shrink contribution bullets (remove stats already in abstract/results, keep references) |
| `sections/02_related_work.tex` | ~430 words, 4 paragraphs | Trim ~25%: keep one best citation per system family, compress the trailing "SpanCall differs" sentences to a single clause each |
| `sections/03_methodology.tex` | ~700 words + figure | Trim ~15%: tighten Sec 3.3 bullets (Ω pool, object flattening, alignment), merge "Normal vs. complex partitioning" into decoding subsection, shorten router policy prose |
| `sections/04_experimental_setup.tex` | ~450 words | Trim ~15%: compress the two-tier itemize into one paragraph each; drop redundant "amortised vs. hosted latency" caveat (repeated in limitations) |
| `sections/05_results.tex` | ~900 words + 6 tables | Trim ~30% of prose: (a) merge subsection 5.1 (dev curve) into one short paragraph — keep table; (b) condense 5.2 paragraph 3's nested CI lists (keep headline CIs only); (c) shorten funnel/complexity narratives to topic sentences + key numbers; (d) cut refusal-sweep prose but keep the table |
| `sections/06_conclusion_limitations.tex` | ~450 words | Trim ~30%: the conclusion paragraph re-states every results number — reduce to the 4 headline numbers; compress limitations to one sentence each |

## Concrete steps

1. [ ] `00_abstract.tex` — rewrite at ~230 words; keep: problem, approach, 0.755 macro exact_F, beats both LLMs on Hermes +32–35pp, 38.8–91.3× faster, Router+Schema 0.855–0.857 with 60% token savings.
2. [ ] `01_introduction.tex` — shorten intro/example paragraph, NER paragraph, and both contribution bullets; keep all `\S` refs and the contribution list structure.
3. [ ] `02_related_work.tex` — remove secondary citations and the final "SpanCall does X" recap sentences (covered in intro).
4. [ ] `03_methodology.tex` — tighten 3.3 bullets, fold "Normal vs. complex" paragraph into \S\ref{sec:method:decode} text, shorten router prose; keep all equations and Figure 1.
5. [ ] `04_experimental_setup.tex` — flatten the Tier-1/Tier-2 itemize; remove duplicated latency-comparability caveat.
6. [ ] `05_results.tex` — main cut: compress 5.1 prose, 5.2 CI lists, funnel/complexity/bucket narratives; keep all 6 tables and every table caption number.
7. [ ] `06_conclusion_limitations.tex` — cut restated numbers to the top-4 headlines; one-sentence limitations.
8. [ ] Rebuild `main.pdf` (Makefile) and verify it compiles with no undefined refs; confirm roughly one page shorter (≈20% of prose).

## Guardrails

- Every table stays (they carry the evidence, and LaTeX tables cost little page space).
- All quantitative claims that appear in the abstract stay backed by a table or CI in results.
- No structural changes to `main.tex`; no renumbering of sections.
- Cut only prose redundancy (numbers repeated 2–4× across abstract/intro/results/conclusion).
