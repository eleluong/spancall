# SpanCall Lean v2.4.1 — Native Bidirectional Backbone (`answerdotai/ModernBERT-large`) Plan

**Date**: 2026-11-14
**Status**: Proposal / Plan Only (No code changes yet)
**Parent Plan**: [`plans/lean_v2_4_bottleneck_fixes_plan.md`](lean_v2_4_bottleneck_fixes_plan.md)

---

## 1. Motivation: Why Test `answerdotai/ModernBERT-large` Against `Qwen/Qwen3-Embedding-0.6B`?

In `v2.2` / `v2.4`, the default backbone is `Qwen/Qwen3-Embedding-0.6B` (`d = 1024`, 28 layers, ~600M params). While 600M parameters is plenty of raw capacity (`train_loss` falls `22×` from `2.77 -> 0.12`), `Qwen3-Embedding-0.6B` has a structural pre-training mismatch with SpanCall's prompt and anchor layout:

1. **Causal Pre-Training vs. Prefix Anchors**:
   - `Qwen3-Embedding-0.6B` is a decoder-derived model (`Qwen3Model`) pre-trained with causal (`left-to-right`) attention and last-token (`<|eos|>`) pooling.
   - In [`span_template/template_multi.jinja`](../span_template/template_multi.jinja), `<|cls|>` (`index 0`), `<|tool|>`, and `<|param|>` are placed at the **prefix** of the prompt, before their own tool/parameter descriptions and before `{{ utterance }}`.
   - Flipping `is_causal = False` via `_force_bidirectional()` at fine-tune time forces prefix anchors to read the tool descriptions and user utterance entirely via un-pretrained backward (`right-to-left`) RoPE relative positions.

2. **`H[:, 0]` (`<|cls|>`) Weakness on Decoder Backbones**:
   - `ToolHead` and `BinaryToolHead` ([`spancall/model/heads.py`](../spancall/model/heads.py)) score `[H_tool; H_cls]` using `cls_h = H[:, 0]`. In a causal-pretrained model, position `0` never attended to any downstream token during pre-training.

3. **Why `answerdotai/ModernBERT-large` (395M params, `d = 1024`, 28 layers, 8,192 context)**:
   - Pre-trained **natively with full bidirectional attention** over 2 trillion tokens with an 8,192-token window (comfortably covering `MAX_SEQ_LEN = 4096`).
   - Position `0` (`[CLS]` / `<|cls|>`) and prefix `<|tool|>` / `<|param|>` anchors naturally attend forward into `{{ utterance }}` and `<|ctx_recent|>` using pre-trained bidirectional RoPE weights from step 0.
   - Identical hidden dimension (`d = 1024`) and layer count (`28`) to `Qwen3-Embedding-0.6B`, giving an apples-to-apples architectural comparison with ~35% fewer parameters and faster step times.

---

## 2. Planned Workstreams for `v2.4.1`

### W1. Fix Notebook Config Bug: Propagate `BACKBONE` into `configs/colab_lean.yaml`
- **Discovery**: In [`scripts/build_lean_v2_2_colab_notebook.py`](../scripts/build_lean_v2_2_colab_notebook.py) / [`scripts/build_lean_v2_4_colab_notebook.py`](../scripts/build_lean_v2_4_colab_notebook.py) (`CODE_COLAB_CFG`), the cell loads `configs/default.yaml` and sets `cfg["model"]["max_seq_len"] = MAX_SEQ_LEN`, but **never sets `cfg["model"]["backbone_name"] = BACKBONE`**.
- Without this line, changing `BACKBONE` at the top of the notebook tensorizes data with `ModernBERT-large`'s tokenizer while `train_lean.py` still loads `Qwen/Qwen3-Embedding-0.6B` from `configs/default.yaml`!
- **Planned Change**: Add `cfg["model"]["backbone_name"] = BACKBONE` in `CODE_COLAB_CFG`.

### W2. Tokenizer & Tensorization Cache Isolation (`TENSOR_DIR` & `RESULTS_DIR`)
- `ModernBERT-large` uses a 50,368-token BPE vocabulary (vs. Qwen3's 151,669-token vocabulary).
- Token IDs, `param_positions`, `tool_positions`, and `cand_starts`/`cand_ends` are strictly tokenizer-dependent.
- **Planned Change**: Configure `v2.4.1` with isolated paths:
  - `TENSOR_DIR = PROJECT_ROOT / "data" / "tensorized_lean_v2_4_1"`
  - `RESULTS_DIR = PROJECT_ROOT / "results_lean_v2_4_1"`
  - `EXPERIMENT_NAME = "spancall_lean_modernbert"`

### W3. `SpanCallTransformerEncoder` Guards for `ModernBERT` ([`spancall/model/encoder.py`](../spancall/model/encoder.py))
1. **Omit `token_type_ids` when `None`**:
   - `ModernBertModel.forward` does not use segment embeddings (`token_type_ids`). While `SpanCallTransformerEncoder.forward`'s docstring states `token_type_ids` is only forwarded when provided, the implementation currently passes `token_type_ids=token_type_ids` unconditionally. Update `forward()` to pass `token_type_ids` only when `token_type_ids is not None`.
2. **Disable `reference_compile` recompilation stalls**:
   - `ModernBertConfig` enables `reference_compile` (`torch.compile`) by default on CUDA when available, which triggers repeated Dynamo recompilations across variable-length batches and after `resize_token_embeddings` (`add_marker_tokens`).
   - Set `self.backbone.config.reference_compile = False` in `SpanCallTransformerEncoder.__init__` when present.

### W4. LoRA Target Module Auto-Detection ([`spancall/lean/trainer.py`](../spancall/lean/trainer.py))
- `Qwen3` uses `["q_proj", "k_proj", "v_proj", "o_proj"]`, whereas `ModernBERT` fuses QKV into `Wqkv` and uses `Wo` / `Wi`.
- Update `apply_lora` in `spancall/lean/trainer.py` to check `backbone.named_modules()` and automatically fall back to `["Wqkv", "Wo", "Wi"]` when `q_proj` is absent and `Wqkv` is present.

### W5. Separate `v2.4.1` Builder & Notebook
- Create `scripts/build_lean_v2_4_1_colab_notebook.py` → `notebooks/spancall_lean_v2_4_1_train_eval.ipynb` with:
  - `BACKBONE = "answerdotai/ModernBERT-large"`
  - `DATA_FRACTION = 0.2`
  - All `v2.4` fixes active (`include_abstain=True`, `RUN_BINARY_TOOL_ABLATION = True`, `prevent_param_collapse=True`, utterance-first `CandidateMaskBuilder`).

---

## 3. Evaluation Criteria (`v2.4` Qwen3-0.6B vs. `v2.4.1` ModernBERT-large)

Compare both runs on the identical 500-row stratified `subset_hermes_eval.jsonl` across:
1. **Step 1 (`H1` Tool Selection)**:
   - Native multi-tool accuracy (`T=2, 3, 4+`)
   - `F_func` vs `A_clean` `tool_prob` and `tool_margin` separation
   - Calibrated $\tau$ (`tool_threshold_v2.json`) and Step-1 false-abstain / false-hit rates
2. **Step 2 (`H3` Pointer Extraction, Oracle Tool)**:
   - Step 2a candidate parity (`T=1` and `T=all` under ModernBERT's BPE tokenization)
   - Step 2b conditional pointer F1 (`protect=True`)
   - Step 2c gated F1 & `Exact_F` (`collapse_guard=True`, `prevent_param_collapse=True`)
3. **Step 3 (Composed E2E)**:
   - 3-arm scoreboard (`single_pass CE`, `single_pass binary`, `two_step binary`)
   - Training & inference wall-clock speed on L4 GPU
