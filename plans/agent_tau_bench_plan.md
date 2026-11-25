# SpanCall Agentic Development Plan (Build → Simple τ-bench → Diagnose → Train)

*Created: 2026-11-25 · Scope: Single-call multi-turn agentic workflow on `ModernBERT-large` (warm-started from `convoicon/spancall-lean-v3-3`)*

---

## Core Design Constraints

1. **Backbone:** Stay on MLM-pretrained `answerdotai/ModernBERT-large` (native 8,192-token window) for fast warm-start iteration. Contrastive embedding backbones (e.g. `Qwen3-Embedding-0.6B`) degraded token-boundary span extraction in prior experiments. **Future 32K candidate:** `dmis-lab/UltraBERT-380M` (arXiv:2610.15028 — 380M params, $d=1024$, 24 layers, native **32K** context, alternating local-128 + global attention, decoupled 5%/50% MLM pre-training for classification + retrieval) is the primary drop-in candidate if Stage 3 diagnostics show trajectories exceeding 8K tokens.
2. **Execution model:** Single tool call per turn, multi-turn trajectory (ReAct / interactive user–agent–tool loop). Keep `SINGLE_CALL_ONLY=True` and the proven two-step $H_1 \to H_3$ architecture.
3. **Evidence-first iteration:** Build the agent and measure failure modes on $\tau$-bench with the existing `v3.3` checkpoint *before* changing training recipes or data mixes.
4. **Paper-grade data hygiene:** Any Stage 4 training uses only published, citable datasets (zero custom synthesis) and stays strictly disjoint from $\tau$-bench (`retail`, `airline`) and BFCL v3/v4 eval splits.

---

## Stage 1 — Build the Multi-Turn Agent (`spancall/agent/`)

- **Policy parity in `ToolCallRouter`:** Add `ROUTER_SCHEMA` and `HYBRID_FILL_SCHEMA` (post-$H_1$ `array`/`object` schema gate) alongside `LLM_ONLY`, `SPANCALL_ONLY`, `ROUTER`, `HYBRID_FILL`, and `SHADOW`.
- **Zero-training runtime guardrails:**
  - Preserve LLM prose (`text`) and `call_id` through `_llm_result` so multi-turn user–agent dialogues work cleanly.
  - **Duplicate-call guard:** Block SpanCall from repeating an identical `(tool_name, arguments)` already executed in the trajectory (`abstain_reason="duplicate_call"` → LLM fallback).
  - **Optional `risk_tier` gate:** Allow delegating high-risk write/destructive tools (e.g., `cancel_pending_order`, `modify_pending_order_*`) to the LLM so policy/confirmation rules are respected.
- **Interactive multi-turn loop (`ReActAgentLoop`):**
  - Support an optional `user_simulator(agent_text, messages) -> Optional[str]` callback for interactive user–agent turns (`###STOP###` terminates).
  - Always synthesize valid `assistant(tool_calls) -> tool` message pairs for both SpanCall hits (`S_INJECT`) and LLM tool calls.
- **OpenAI SDK backend (`OpenAIAPIBackend`):** Direct `openai.OpenAI` adapter reading `.env` (`BASE_URL`, `API_KEY`, `BASE_LLM_MODEL`) without requiring `litellm`.

## Stage 2 — Test on Simple τ-bench Retail (`spancall/agent/tools/tau_retail.py`, `scripts/eval_tau_bench.py`)

- **Self-contained Simple $\tau$-bench Retail suite (10 multi-turn tasks):** Faithful retail schemas (`find_user_id_by_name_zip`, `find_user_id_by_email`, `get_user_details`, `get_order_details`, `get_product_details`, `cancel_pending_order`, `modify_pending_order_address`, `modify_pending_order_items`, `return_delivered_order_items`, `exchange_delivered_order_items`), in-memory DB state mutation, policy-confirmation turns, and final-DB-state verification.
- **Official `tau_bench` adapter (`TauBenchAgentAdapter`):** Drop-in wrapper implementing `tau_bench.agents.base.Agent.solve(env, task_index)` when official `tau-bench` is installed.
- **Paired evaluation CLI (`scripts/eval_tau_bench.py`):** Compares `llm_only`, `router`, `router_schema`, `hybrid_fill_schema`, and `shadow`, writing per-turn JSONL traces + summary metrics.

## Stage 3 — Diagnose Failure Modes from Traces

Aggregate per-turn JSONL telemetry into concrete counts:
- `false_hit_on_prose`: SpanCall fired on a turn where the LLM answered in prose / asked confirmation.
- `duplicate_blocked`: SpanCall attempted to repeat an already-executed tool call.
- `complex_schema_routed`: Clean post-$H_1$ delegation of `array`/`object` tools.
- `risk_tier_routed`: Delegation of policy-sensitive write tools.
- `context_truncated`: `<|ctx_recent|>` or transcript hit the token budget.
- `tool_exec_error`: Tool execution returned an error (invalid ID/args or policy violation).

## Stage 4 — Targeted Improvement via Training (`v4` @ 8k ModernBERT-large)

Driven by Stage 3 counts:
1. **If `false_hit_on_prose` dominates:** Train explicit `<|none|>` supervision in $H_1$ using published abstention + post-tool stop turns (`nvidia/When2Call`, `glaiveai/glaive-function-calling-v2` irrelevance, unrolled `Hermes V1` final turns).
2. **If `tool_output` chaining misses dominate:** Unroll multi-turn trajectories with flattened tool outputs from `Team-ACE/ToolACE`, `Agent-Ark/Toucan-1.5M`, `glaiveai/glaive-function-calling-v2`, and `Hermes V1` (single-call turns only).
3. **If `context_truncated` dominates:** Unlock ModernBERT-large's native **8,192** window with residual context budgeting and `tool_output` candidate quotas, training on length-stratified 4k–8k trajectories from `Agent-Ark/Toucan-1.5M`.
4. **Re-evaluate** on the exact Stage 2 $\tau$-bench slice + full $\tau$-bench (`retail`, `airline`) and frozen `data/eval_v3_3/` Tier 1/2 to verify gains without single-turn regression.
