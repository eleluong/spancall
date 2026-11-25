# SpanCall Lean v3.3 — Unified Audit & Paper Export Report

- **Generated (UTC):** `2026-11-20T06:35:58.960842+00:00`
- **Git commit:** `e568daf`
- **Audit status:** **`PASS_WITH_LLM_ERRORS`**
- **Eval manifest SHA256:** `3a78c8fcb5fcb6b7f29deee7d68e27612173652483e0e0753a4403846f3bc534`
- **Template stamp:** `1e62a355a351ae531c64f137a5d66cf873de81c2987829243a1559cda6b25cf2`
- **Models evaluated:** `google-gemma-4-26b-a4b-it, openai-gpt-oss-120b`

## 1. Audit Checklist

| Check | Status | Details |
| :--- | :---: | :--- |
| Manifest SHA256 integrity | PASS | 12 splits verified |
| 100% `sample_id` coverage | PASS | 0 failure(s) across 14 files |
| LLM API error count | WARN | 247 total API error(s) |

## 2. Tier 1 Fair Eval — Headline Results

| Split | System | $n_F$ | `exact_F` | `hit_F` | `param_F1` | `false_hit_A` | `exact_all` | `lat_ms` (p90) | `tok_in` | `fallback` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fair_hermes_eval` | `spancall_v3_3` | 1277 | 0.743 | 0.879 | 0.804 | 0.638 | 0.602 | 113.0 (190.9) | 0.0 | — |
| `fair_hermes_eval` | `llm:google-gemma-4-26b-a4b-it` | 1277 | 0.390 | 0.718 | 0.595 | 0.854 | 0.372 | 7016.6 (11853.0) | 413.3 | 1.000 |
| `fair_hermes_eval` | `router:google-gemma-4-26b-a4b-it` | 1277 | 0.801 | 0.991 | 0.861 | 0.969 | 0.666 | 1520.1 (6992.1) | 83.5 | 0.184 |
| `fair_hermes_eval` | `llm:openai-gpt-oss-120b` | 1277 | 0.420 | 0.773 | 0.609 | 0.795 | 0.391 | 3713.6 (11487.0) | 395.5 | 1.000 |
| `fair_hermes_eval` | `router:openai-gpt-oss-120b` | 1277 | 0.800 | 0.991 | 0.862 | 0.945 | 0.665 | 868.1 (1763.0) | 79.6 | 0.184 |
| `fair_xlam_eval` | `spancall_v3_3` | 1277 | 0.766 | 0.883 | 0.798 | 0.690 | 0.609 | 41.0 (53.2) | 0.0 | — |
| `fair_xlam_eval` | `llm:google-gemma-4-26b-a4b-it` | 1277 | 0.778 | 0.958 | 0.813 | 0.937 | 0.739 | 7034.5 (11699.8) | 386.2 | 1.000 |
| `fair_xlam_eval` | `router:google-gemma-4-26b-a4b-it` | 1277 | 0.815 | 0.995 | 0.841 | 0.980 | 0.695 | 1207.9 (6245.0) | 67.2 | 0.168 |
| `fair_xlam_eval` | `llm:openai-gpt-oss-120b` | 1277 | 0.777 | 0.966 | 0.821 | 0.963 | 0.740 | 2259.9 (6503.2) | 340.1 | 1.000 |
| `fair_xlam_eval` | `router:openai-gpt-oss-120b` | 1277 | 0.814 | 0.995 | 0.843 | 0.985 | 0.697 | 446.2 (1005.8) | 57.9 | 0.168 |
| `macro_avg` | `spancall_v3_3` | 2554 | 0.755 | 0.881 | 0.801 | 0.664 | 0.605 | 77.0 (122.1) | 0.0 | — |
| `macro_avg` | `llm:google-gemma-4-26b-a4b-it` | 2554 | 0.584 | 0.838 | 0.704 | 0.895 | 0.555 | 7025.6 (11776.4) | 399.8 | 1.000 |
| `macro_avg` | `router:google-gemma-4-26b-a4b-it` | 2554 | 0.808 | 0.993 | 0.851 | 0.975 | 0.681 | 1364.0 (6618.6) | 75.3 | 0.176 |
| `macro_avg` | `llm:openai-gpt-oss-120b` | 2554 | 0.598 | 0.870 | 0.715 | 0.879 | 0.565 | 2986.7 (8995.1) | 367.8 | 1.000 |
| `macro_avg` | `router:openai-gpt-oss-120b` | 2554 | 0.807 | 0.993 | 0.853 | 0.965 | 0.681 | 657.2 (1384.4) | 68.8 | 0.176 |

### Paired Bootstrap 95% CIs (`exact_F` Delta)

| Split | Comparison | $n_F$ | Delta | 95% CI |
| :--- | :--- | ---: | ---: | :---: |
| `fair_hermes_eval` | `spancall_v3_3 - llm:google-gemma-4-26b-a4b-it` | 1277 | +0.3532 | `[+0.3125, +0.3931]` |
| `fair_hermes_eval` | `router:google-gemma-4-26b-a4b-it - llm:google-gemma-4-26b-a4b-it` | 1277 | +0.4111 | `[+0.3743, +0.4471]` |
| `fair_hermes_eval` | `spancall_v3_3 - llm:openai-gpt-oss-120b` | 1277 | +0.3234 | `[+0.2835, +0.3641]` |
| `fair_hermes_eval` | `router:openai-gpt-oss-120b - llm:openai-gpt-oss-120b` | 1277 | +0.3798 | `[+0.3430, +0.4158]` |
| `fair_xlam_eval` | `spancall_v3_3 - llm:google-gemma-4-26b-a4b-it` | 1277 | -0.0117 | `[-0.0423, +0.0188]` |
| `fair_xlam_eval` | `router:google-gemma-4-26b-a4b-it - llm:google-gemma-4-26b-a4b-it` | 1277 | +0.0376 | `[+0.0102, +0.0658]` |
| `fair_xlam_eval` | `spancall_v3_3 - llm:openai-gpt-oss-120b` | 1277 | -0.0110 | `[-0.0407, +0.0196]` |
| `fair_xlam_eval` | `router:openai-gpt-oss-120b - llm:openai-gpt-oss-120b` | 1277 | +0.0368 | `[+0.0094, +0.0642]` |

## 3. Tier 1 Fair Eval — Normal vs. Complex

| Split | Complexity | System | $n_F$ | `exact_F` | `hit_F` | `param_F1` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `fair_hermes_eval` | `normal` | `spancall_v3_3` | 1108 | 0.798 | 0.873 | 0.848 |
| `fair_hermes_eval` | `complex` | `spancall_v3_3` | 169 | 0.385 | 0.923 | 0.510 |
| `fair_hermes_eval` | `normal` | `llm:google-gemma-4-26b-a4b-it` | 1108 | 0.360 | 0.696 | 0.565 |
| `fair_hermes_eval` | `complex` | `llm:google-gemma-4-26b-a4b-it` | 169 | 0.586 | 0.864 | 0.777 |
| `fair_hermes_eval` | `normal` | `router:google-gemma-4-26b-a4b-it` | 1108 | 0.864 | 0.992 | 0.908 |
| `fair_hermes_eval` | `complex` | `router:google-gemma-4-26b-a4b-it` | 169 | 0.391 | 0.982 | 0.538 |
| `fair_hermes_eval` | `normal` | `llm:openai-gpt-oss-120b` | 1108 | 0.397 | 0.756 | 0.588 |
| `fair_hermes_eval` | `complex` | `llm:openai-gpt-oss-120b` | 169 | 0.568 | 0.882 | 0.742 |
| `fair_hermes_eval` | `normal` | `router:openai-gpt-oss-120b` | 1108 | 0.861 | 0.991 | 0.910 |
| `fair_hermes_eval` | `complex` | `router:openai-gpt-oss-120b` | 169 | 0.396 | 0.988 | 0.539 |
| `fair_xlam_eval` | `normal` | `spancall_v3_3` | 1108 | 0.840 | 0.880 | 0.838 |
| `fair_xlam_eval` | `complex` | `spancall_v3_3` | 169 | 0.278 | 0.905 | 0.492 |
| `fair_xlam_eval` | `normal` | `llm:google-gemma-4-26b-a4b-it` | 1108 | 0.755 | 0.957 | 0.794 |
| `fair_xlam_eval` | `complex` | `llm:google-gemma-4-26b-a4b-it` | 169 | 0.923 | 0.964 | 0.951 |
| `fair_xlam_eval` | `normal` | `router:google-gemma-4-26b-a4b-it` | 1108 | 0.884 | 0.994 | 0.878 |
| `fair_xlam_eval` | `complex` | `router:google-gemma-4-26b-a4b-it` | 169 | 0.367 | 1.000 | 0.567 |
| `fair_xlam_eval` | `normal` | `llm:openai-gpt-oss-120b` | 1108 | 0.755 | 0.963 | 0.802 |
| `fair_xlam_eval` | `complex` | `llm:openai-gpt-oss-120b` | 169 | 0.923 | 0.988 | 0.956 |
| `fair_xlam_eval` | `normal` | `router:openai-gpt-oss-120b` | 1108 | 0.884 | 0.995 | 0.881 |
| `fair_xlam_eval` | `complex` | `router:openai-gpt-oss-120b` | 169 | 0.355 | 0.994 | 0.564 |

## 4. Tier 2 Full ID, OOD & Adversarial Generalization (SpanCall v3.3)

| Split | $n$ | $n_F$ | `exact_F` | `hit_F` | `param_F1` | `false_hit_A` | `lat_ms` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_hermes_eval` | 1829 | 1367 | 0.708 | 0.873 | 0.776 | 0.636 | 127.6 |
| `full_xlam_eval` | 6811 | 4571 | 0.757 | 0.871 | 0.793 | 0.661 | 46.9 |
| `full_hermes_test_ood_schema` | 1599 | 1204 | 0.711 | 0.835 | 0.774 | 0.648 | 89.6 |
| `full_xlam_test_ood_schema` | 7227 | 5010 | 0.799 | 0.887 | 0.753 | 0.714 | 43.2 |
| `full_hermes_test_ood_para` | 1342 | 1341 | 0.735 | 0.834 | 0.779 | 1.000 | 87.4 |
| `full_xlam_test_ood_para` | 6876 | 6868 | 0.793 | 0.880 | 0.780 | 0.875 | 38.7 |
| `full_hermes_test_adversarial` | 4285 | 0 | 0.000 | 0.000 | 0.000 | 0.816 | 164.6 |
| `full_xlam_test_adversarial` | 5000 | 0 | 0.000 | 0.000 | 0.000 | 0.788 | 48.3 |
