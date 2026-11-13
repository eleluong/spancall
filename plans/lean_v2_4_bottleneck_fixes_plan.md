# SpanCall Lean v2.4 — Bottleneck Fixes Plan (Parity, Binary H1, Exclusivity, Stratum-A Negatives)

**Date**: 2026-11-14
**Status**: Planned → Implementing
**Motivation**: End-to-end diagnostics from `notebooks/spancall_lean_v2_2_train_eval.ipynb` (`results/lean/13-11-26-v2.3/preds_eval.jsonl`) on the 500-row stratified benchmark (`250 F_func`, `50 F_json`, `150 A_clean`, `50 A_adv`) showed that while label staleness is fixed (`0.9978 -> 0.0000`) and pointer collapse hits are `0`, `F_func` E2E exact match is `0.4%` (`single_pass`) / `1.6%` (`two_step`) due to four compounding bottlenecks:
1. **Step 2a Candidate Parity (`51.6%` of `F_func` misses)**: `CandidateMaskBuilder` fills `C = 512` with `free_recent` sub-spans (including JSON punctuation fragments) before `<|utt|>` spans, truncating `808 / 1362` GT spans (`T=1` parity = `0.3436`, `T=all` parity = `0.2430`) and causing `MATCH_MISSED = 463` during training tensorization.
2. **Step 1 Softmax H1 (`44.4%` of `F_func` misses)**: `RUN_BINARY_TOOL_ABLATION` was `False`, so `H1` ran in softmax CE mode. Multi-tool accuracy degraded from `0.530` (`T=2`) to `0.077` (`T=4`), `F_func` vs `A_clean` had identical median `tool_margin = 0.031`, `false_hit_on_A` was `27.0%`, and `call_count_acc` was `0.382` (over-predicting 2+ calls).
3. **Cross-Parameter Span Repetition (`195 / 323` multi-param rows = `60.4%`)**: `LeanJointDecoder._conflict` has `free_string_overlap_allowed=True` and skips non-`utterance` regions (`recent`) and distinct `entity_class` pairs, allowing sibling parameters of the same call to collapse onto the exact same span (e.g., `side1=4.0, side2=4.0`).
4. **Small Data / No Stratum-A H1 Supervision**: Training at `DATA_FRACTION = 0.1` (`1,989` rows) overfit by epoch 2 (`dev_tool_loss`: `0.705 -> 1.218`) and filtered out all Stratum-A rows (`has_positive_targets == False`), giving `BinaryToolHead` zero negative-query examples.

---

## Workstreams

### W1. `CandidateMaskBuilder` Utterance-First Quota & Clean `<|recent|>` Spans (`spancall/model/masks.py`)
- **Problem**: In `build_param_masks` (free-string branch) and `_pack`, `region != "utterance"` puts *all* `recent_cands` and `free_recent` sub-spans ahead of `<|utt|>` spans, exhausting `max_cands = 512` before utterance tokens are reached, and `_free_recent_spans` generates JSON syntax fragments (e.g., `jazz\"}}`, `cuisine\": \"`).
- **Changes**:
  1. **Filter JSON syntax noise in `_free_recent_spans`**: Skip sub-spans containing structural JSON punctuation (`"`, `{`, `}`, `[`, `]`, `\\`) or colon-quote boundaries (`":`), and cap/order `free_recent` from **most-recent entry to oldest** (`reversed(recent_vals)`).
  2. **Utterance-first budget in `build_param_masks` & `_pack`**:
     - Reserve slots strictly for true structured context anchors (`null`, `bool`, `enum`, `time`) and whole-leaf `recent_cands` first.
     - Allocate the primary remaining budget to `_free_utterance_spans` (using `pack_fill`, e.g., `"stratified"`), and only fill leftover budget with `free_recent` sub-spans (capped at e.g. `min(len(free_recent), max(0, max_cands // 4))` when `max_cands` is set).
     - In `_pack`, prioritize `region in ("null", "bool", "enum", "time")` → `protected` → `utterance` (under `pack_fill`) → `recent` (leaf entries before `free_recent`), then sort final `ordered` list in document order `(tok_start, tok_end)`.
  3. **Default `max_cands` bump option**: Keep `max_cands=512` compatible with existing tensorized rows/tests, and support `max_cands=768` or `1024` via config/notebook if needed, while ensuring `512` alone already achieves high `<|utt|>` parity.

### W2. Cross-Parameter Exclusivity in `LeanJointDecoder` (`spancall/lean/decode.py`, `spancall/lean/resolver.py`)
- **Problem**: Sibling parameters of the same tool call collapse to identical spans because `LeanJointDecoder._conflict` defaults to `free_string_overlap_allowed=True` and ignores `recent` and cross-`entity` exact-span collisions.
- **Changes**:
  1. Change `LeanJointDecoder.__init__` default `free_string_overlap_allowed: bool = False` (or pass `free_string_overlap_allowed=False` from `LeanResolver` while keeping backward-compatible constructor option).
  2. In `_conflict(v1, v2)`:
     - Keep `null` / `bool` / `enum` / `time` non-conflicting (two parameters may legitimately both be `null` or `true`).
     - For `utterance` and `recent` spans with valid `(s, e)`:
       - Exact-span collision `(s1, e1) == (s2, e2)` is ALWAYS a conflict within the same call, regardless of `entity_class` (fixes `title="2010", year=2010`).
       - Overlapping `utterance` spans `(s1 < e2 and s2 < e1)` conflict when `free_string_overlap_allowed=False` (unless `allow_nested` and strictly nested for distinct entity classes).
  3. Keep `LeanJointDecoder._search` resilient greedy fallback only when *no* non-conflicting assignment exists, and ensure `top_k` is large enough (e.g., `top_k=5` in `LeanResolver`) so the second-best non-overlapping span is available.

### W3. Enable `BinaryToolHead` + Stratum-A Negative H1 Supervision (`spancall/lean/loss.py`, `spancall/lean/tensorize.py`, `spancall/lean/trainer.py`)
- **Problem**: `BinaryToolHead` was disabled in the notebook (`RUN_BINARY_TOOL_ABLATION = False`), and even when enabled, `LeanTrainer` filters out Stratum-A rows (`filter_empty_positives=True`), so `BinaryToolHead` never sees ungroundable queries during training.
- **Changes**:
  1. **Tensorizer (`spancall/lean/tensorize.py`)**: Tag each row with `has_tool_negatives: bool` (or preserve `is_abstain: 1` for Stratum-A rows that have valid rendered `tool_positions >= 0` and `tool_targets == -1`). Note: `has_positive_targets` remains unchanged for pointer-loss filtering.
  2. **Loss (`spancall/lean/loss.py`)**: In `compute_lean_tool_loss_binary` (and `_lean_tool_binary_supervision_count`), accept optional `abstain_mask: Optional[torch.Tensor]` (shape `(B,)`, bool). For rows where `abstain_mask` is True and at least one anchor is live (`live.any(-1)`), set `one_hot` to all-zeros and include those rows in `has_row` so `BCEWithLogits` pushes all tool logits down on Stratum-A queries! Pointer loss already zeroes out rows with `n_pos == 0`.
  3. **Trainer (`spancall/lean/trainer.py`)**: When `self.use_binary_tool_head` is True, allow Stratum-A rows (rows with `is_abstain == 1` and live `tool_positions >= 0`) to pass `load_rows(..., filter_empty_positives=True)` (optionally capped to e.g. balanced share or all present in split) and pass `abstain_mask` into `LeanLossContainer`.

### W4. Separate v2.4 Notebook & `DATA_FRACTION = 0.2` (`scripts/build_lean_v2_4_colab_notebook.py`, `notebooks/spancall_lean_v2_4_train_eval.ipynb`)
- Keep `spancall_lean_v2_2_train_eval.ipynb` untouched and build a separate `notebooks/spancall_lean_v2_4_train_eval.ipynb` with isolated `data/tensorized_lean_v2_4` and `results_lean_v2_4` directories.
- Keep `DATA_FRACTION = 0.2` for testing as requested.
- Set `RUN_BINARY_TOOL_ABLATION = True` so both `ce` and `binary` arms train, `tool_threshold_v2.json` ($\tau$) is calibrated on `hermes_dev`, and the 3-arm scoreboard (`single_pass CE`, `single_pass binary`, `two_step binary`) runs end-to-end.
- Generate `notebooks/spancall_lean_v2_4_train_eval.ipynb` via `scripts/build_lean_v2_4_colab_notebook.py`.

---

## Verification Plan
1. Add targeted unit tests in `tests/test_lean_v2_4_fixes.py` covering:
   - `CandidateMaskBuilder` utterance-first packing when `recent_context` is large + JSON punctuation filtering in `_free_recent_spans`.
   - `LeanJointDecoder` cross-parameter span exclusivity (preventing `title="2010", year=2010` and `side1=4.0, side2=4.0` while allowing multi-call shared spans).
   - `compute_lean_tool_loss_binary` with `abstain_mask` (Stratum-A negative supervision) + `LeanLossContainer` micro-average accounting.
2. Run the full `pytest` suite (`.venv/bin/python -m pytest tests -q`) and confirm all tests pass.

