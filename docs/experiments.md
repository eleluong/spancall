# SpanCall Research Experiment Ledger

This document tracks hypotheses, configurations, verification gates, and decisions across all experimental phases.

## Experiment Tracker

| Exp ID | Question | Hypothesis | Metric / Gate | Baseline | Status |
| --- | --- | --- | --- | --- | --- |
| **E0** | Traffic probe (D7) | >=80% of live calls are stratum F (fully groundable by the current template) | Per-workload F-share >= cost-model floor (default 0.5); abstain-reason mix; scope-cut deltas; retriever R@T | No fast path (LLM-only) | Planned |
| **E1** | Ceiling on Stratum F | Pointer head achieves >=90% exact payload on groundable calls | Dev Exact Payload >= 90% | Zero-shot LLM | Planned |
| **E2** | Precision Gating | Calibrated conjunction gate reaches >=95% precision | Precision-at-Hit >= 95% at Coverage >= 60% of F | Ungated Argmax | Planned |
| **E3** | Swap Resistance | Entity swap hard negatives reduce swap errors by >= 2x | Swap Error Rate <= 2% | Model w/o swaps | Planned |
| **E4** | Delta_joint Utility | Joint assignment margin detects ambiguous entity bindings | Delta-AUROC on swap errors | Per-param margin | Planned |
| **E5** | Generalization | Head generalizes to unseen schemas with <=5 pt drop | Held-out Schema Exact Payload | In-distribution | Planned |
| **E6** | Compute Savings | Speculative fast path yields >=5x latency win over 0.5B generator | p50 latency, TFLOPs | 0.5B LM, 7B LM | Planned |
| **E7** | Non-harm on Stratum A | Stratum A calls false-hit <= 1-2% | False-hit rate on A <= 2% | Ungated Model | Planned |
| **E8** | Multi-schema Selection | Batched encoder scoring selects correct tool schema | Wrong-tool hit <= 1% | LLM Tool Choice | Planned |
| **E9** | Agent Non-inferiority | Agent + SpanCall matches Normal Agent success within 2% margin | Task Success Delta >= -2% (p < 0.05) | Normal Agent (B0) | Planned |
| **E10** | History Consistency | S-inject maintains context coherence on later multi-turn steps | Turn N>1 call accuracy | S-skip variant | Planned |
| **E11** | Context Scaling | Recent context rows beyond 5 rows yield diminishing returns | Precision vs latency curve | Full history | Planned |
| **E12** | Backbone Comparison | ModernBERT outperforms DeBERTa-v3 at lower inference latency | Latency vs exact match | DeBERTa-v3 | Planned |
| **E13** | Quantization Shift | INT8 post-training quantization preserves calibrated precision | Post-quant Precision >= 94.5% | BF16 baseline | Planned |
| **E14** | Shadow Mode Transfer | Calibration on logged user traffic transfers across domains | Precision on Target Domain | Static calibration | Planned |
| **E15** | Head Ablations | Dedicated [PARAM] query vectors outperform pooled CLS query | Per-param Exact Match | CLS-only pointer | Planned |

---

## Experiment Execution Protocol

1. **Config-First**: Every experiment must be runnable via `python scripts/... --config configs/experiments/<exp_id>.yaml`.
2. **Deterministic Seeds**: Run experiments with at least 3 random seeds (e.g. 42, 43, 44).
3. **Artifact Logging**: Results, metrics, predictions, and calibration curves stored in `results/<run_id>/`.
4. **Kill Criteria (floors)**: Kill floors are deliberately set **below** the success gates in the tracker above, so a "passed gate" and a "killed" result can never overlap:
   - **E0 floor**: per-workload F-share < cost-model floor `F_floor = h_min / 0.6` (default 0.5). F-share is a coverage *upper bound* on hit rate, so F=0.5 under E2's 60%-of-F coverage target implies achievable `h ≈ 0.3`. Below the floor => stop and write up the negative result.
   - **E1 floor**: exact payload < 80% on F dev (success gate is >= 90%).
   - **E2 floor**: cannot reach >= 95% precision-at-hit at **any** coverage >= 40% of F (success gate is >= 95% at coverage >= 60% of F).
   - Between gate and floor = *stalled experiment*: investigate before continuing. At/below the floor, stop and write up the negative result.
