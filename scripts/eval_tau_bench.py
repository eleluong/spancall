#!/usr/bin/env python3
"""Stage 2 evaluation runner for Simple τ-bench Retail (and official τ-bench).

Runs paired comparisons across routing modes (``llm_only``, ``router``,
``router_schema``, ``hybrid_fill_schema``, ``shadow``), writes per-turn JSONL
traces for Stage 3 failure-mode diagnosis, and prints a diagnostic summary
table.

Examples
--------
Offline smoke verification (no network / API keys required)::

    python scripts/eval_tau_bench.py --backend deterministic --modes llm_only router_schema shadow

Live run against an OpenAI-compatible LLM (reads ``.env``) + SpanCall v3.3::

    python scripts/eval_tau_bench.py \\
        --backend openai \\
        --spancall-checkpoint convoicon/spancall-lean-v3-3 \\
        --modes llm_only router_schema shadow \\
        --skip-high-risk \\
        --out-dir results/agentic/tau_retail_v3_3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.backends.openai_adapter import OpenAIAPIBackend
from spancall.agent.loop import ReActAgentLoop
from spancall.agent.router import IntegrationVariant, RouterMode, ToolCallRouter
from spancall.agent.tools.tau_retail import (
    TAU_RETAIL_SYSTEM_PROMPT,
    ScriptedUserSimulator,
    SimpleTauRetailEnvironment,
    TauRetailTask,
    build_simple_tau_retail_suite,
)
from spancall.agent.trace import ExecutionTrace, RoutingDecision
from spancall.data.schema import ToolDefinition
from spancall.infer.base import NoOpResolver


# --------------------------------------------------------------------------- #
# Deterministic retail backend (offline verification without API keys)
# --------------------------------------------------------------------------- #
class DeterministicTauRetailBackend(BaseLLMBackend):
    """Rule-based backend that solves the Simple τ-bench Retail suite offline.

    Used for reproducible CI / pipeline verification and as a clean oracle-like
    fallback when testing router wiring, ``S_INJECT`` history, duplicate guards,
    and diagnostic trace aggregation without external API calls.
    """

    def __init__(self, latency_ms: float = 2.0) -> None:
        self.latency_ms = float(latency_ms)

    def generate_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        user_texts = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
        full_user = " ".join(user_texts)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        called_names = [str(m.get("name") or "") for m in tool_msgs]
        confirmed = any(
            any(w in u.lower() for w in ("yes", "confirm", "go ahead", "please submit", "please update", "please process"))
            for u in user_texts[1:]
        )

        # Parse state observed from prior tool outputs
        obs_user_id: Optional[str] = None
        obs_order: Optional[Dict[str, Any]] = None
        obs_product: Optional[Dict[str, Any]] = None
        for tm in tool_msgs:
            try:
                data = json.loads(str(tm.get("content") or "{}"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if tm.get("name") in ("find_user_id_by_name_zip", "find_user_id_by_email") and data.get("user_id"):
                obs_user_id = str(data["user_id"])
            elif tm.get("name") == "get_order_details" and data.get("order_id"):
                obs_order = data
            elif tm.get("name") == "get_product_details" and data.get("product_id"):
                obs_product = data

        def _ret(tool_name: Optional[str], arguments: Optional[Dict[str, Any]], text: str = "") -> Dict[str, Any]:
            return {
                "tool_name": tool_name,
                "arguments": arguments,
                "text": text,
                "usage": {"prompt_tokens": 120, "completion_tokens": 24},
                "latency_ms": max(self.latency_ms, (time.perf_counter() - t0) * 1000.0),
            }

        # 1. Product-only check
        prod_m = re.search(r"(prod_[a-z0-9_]+)", full_user)
        if prod_m and "get_product_details" not in called_names and "order" not in full_user.lower():
            return _ret("get_product_details", {"product_id": prod_m.group(1)})
        if prod_m and "get_product_details" in called_names and "order" not in full_user.lower():
            return _ret(None, None, "I checked the product details for you.")

        # 2. Authenticate user first (email or name+zip)
        if obs_user_id is None:
            em = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", full_user)
            if em and "find_user_id_by_email" not in called_names:
                return _ret("find_user_id_by_email", {"email": em.group(0).rstrip(".,;:?!")})
            nz = re.search(r"(?:I'm|am)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+).*?(\d{5})", full_user)
            if nz and "find_user_id_by_name_zip" not in called_names:
                return _ret(
                    "find_user_id_by_name_zip",
                    {"first_name": nz.group(1), "last_name": nz.group(2), "zip": nz.group(3)},
                )

        # 3. User details if requested
        if obs_user_id and "recent orders" in full_user.lower() and "get_user_details" not in called_names:
            return _ret("get_user_details", {"user_id": obs_user_id})
        if "recent orders" in full_user.lower() and "get_user_details" in called_names:
            return _ret(None, None, "Here are your recent orders.")

        # 4. Order details if an order_id is mentioned
        ord_m = re.search(r"(#W\d+)", full_user)
        order_id = ord_m.group(1) if ord_m else None
        if order_id and obs_order is None and "get_order_details" not in called_names:
            return _ret("get_order_details", {"order_id": order_id})

        # 5. Optional product check before item modification
        if prod_m and obs_product is None and "get_product_details" not in called_names:
            return _ret("get_product_details", {"product_id": prod_m.group(1)})

        # 6. Read-only order status inquiry
        if "status of my order" in full_user.lower() and obs_order is not None:
            return _ret(None, None, f"Your order {order_id} is currently {obs_order.get('status')}.")

        # 7. Policy refusal: cannot cancel delivered order
        if "cancel" in full_user.lower() and obs_order and obs_order.get("status") != "pending":
            return _ret(None, None, f"Order {order_id} is already {obs_order.get('status')} and cannot be cancelled.")

        # 8. Mutations require confirmation turn first
        wants_mutate = any(k in full_user.lower() for k in ("cancel", "shipping address", "change my", "return"))
        already_mutated = any(
            n in (
                "cancel_pending_order",
                "modify_pending_order_address",
                "modify_pending_order_items",
                "return_delivered_order_items",
            )
            for n in called_names
        )
        if already_mutated:
            return _ret(None, None, "Your request has been completed.")

        if wants_mutate and not confirmed:
            return _ret(None, None, f"Please confirm that you would like me to proceed with order {order_id}.")

        # 9. Execute confirmed mutation
        if "cancel" in full_user.lower() and order_id:
            reason = "ordered by mistake" if "mistake" in full_user.lower() else "no longer needed"
            return _ret("cancel_pending_order", {"order_id": order_id, "reason": reason})

        if "shipping address" in full_user.lower() and order_id:
            addr = re.search(
                r"\bto\s+(\d+\s+[^,]+),\s*([^,]+),\s*([A-Z]{2}),\s*([A-Z]+),\s*(\d{5})",
                full_user,
            )
            if addr:
                return _ret(
                    "modify_pending_order_address",
                    {
                        "order_id": order_id,
                        "address1": addr.group(1).strip(),
                        "address2": "",
                        "city": addr.group(2).strip(),
                        "state": addr.group(3).strip(),
                        "country": addr.group(4).strip(),
                        "zip": addr.group(5).strip(),
                    },
                )

        if "change my" in full_user.lower() and order_id and obs_order:
            old_item = obs_order["items"][0]["item_id"]
            new_item = "item_101_blue_l"
            if obs_product:
                for v in obs_product.get("variants") or []:
                    if v.get("size") == "L" and v.get("available"):
                        new_item = v["item_id"]
                        break
            return _ret(
                "modify_pending_order_items",
                {
                    "order_id": order_id,
                    "item_ids": [old_item],
                    "new_item_ids": [new_item],
                    "payment_method_id": obs_order.get("payment_method_id", "credit_card_9513926"),
                },
            )

        if "return" in full_user.lower() and order_id and obs_order:
            item_id = obs_order["items"][0]["item_id"]
            pm = obs_order.get("payment_method_id", "")
            pm_m = re.search(r"(gift_card_\d+|credit_card_\d+|paypal_\d+)", full_user)
            if pm_m:
                pm = pm_m.group(1)
            return _ret(
                "return_delivered_order_items",
                {
                    "order_id": order_id,
                    "item_ids": [item_id],
                    "payment_method_id": pm,
                },
            )

        return _ret(None, None, "Done.")

    def generate_response(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        return "Done."


# --------------------------------------------------------------------------- #
# Diagnostic aggregation (Stage 3 failure buckets)
# --------------------------------------------------------------------------- #
def serialize_trace(trace: ExecutionTrace) -> Dict[str, Any]:
    return {
        "task_id": trace.task_id,
        "environment_name": trace.environment_name,
        "mode": trace.mode,
        "integration_variant": trace.integration_variant,
        "task_completed": trace.task_completed,
        "task_success": trace.task_success,
        "total_latency_ms": round(trace.total_latency_ms, 2),
        "total_tokens_prompt": trace.total_tokens_prompt,
        "total_tokens_completion": trace.total_tokens_completion,
        "total_spancall_calls": trace.total_spancall_calls,
        "total_llm_calls": trace.total_llm_calls,
        "total_context_trimmed": trace.total_context_trimmed,
        "steps": [
            {
                "turn_index": s.turn_index,
                "user_input": s.user_input,
                "decision": s.decision.value,
                "selected_tool": s.selected_tool,
                "tool_arguments": s.tool_arguments,
                "tool_execution_success": s.tool_execution_success,
                "llm_tokens_in": s.llm_tokens_in,
                "llm_tokens_out": s.llm_tokens_out,
                "latency_ms": round(s.latency_ms, 2),
                "context_trimmed": s.context_trimmed,
                "spancall_features": s.spancall_features,
            }
            for s in trace.steps
        ],
    }


def summarize_mode_traces(mode: str, traces: Sequence[ExecutionTrace]) -> Dict[str, Any]:
    n = max(1, len(traces))
    decisions = Counter()
    buckets = Counter()
    total_turns = 0
    for tr in traces:
        for s in tr.steps:
            total_turns += 1
            decisions[s.decision.value] += 1
            feats = s.spancall_features or {}
            gate = feats.get("router_gate_reason")
            if gate == "complex_schema":
                buckets["complex_schema_routed"] += 1
            elif gate == "duplicate_call":
                buckets["duplicate_blocked"] += 1
            elif isinstance(gate, str) and gate.startswith("risk_tier:"):
                buckets["risk_tier_routed"] += 1
            if feats.get("truncated"):
                buckets["recent_context_truncated"] += 1
            if feats.get("context_overflow_seq"):
                buckets["context_overflow_seq"] += 1
            if s.context_trimmed:
                buckets["transcript_trimmed"] += 1
            if s.selected_tool and not s.tool_execution_success:
                buckets["tool_exec_error"] += 1
            if feats.get("shadow_false_hit_on_prose"):
                buckets["shadow_false_hit_on_prose"] += 1
            if feats.get("shadow_matched"):
                buckets["shadow_matched"] += 1

    return {
        "mode": mode,
        "tasks": len(traces),
        "success_rate": round(sum(1 for t in traces if t.task_success) / n, 4),
        "avg_turns": round(total_turns / n, 2),
        "total_spancall_hits": decisions.get(RoutingDecision.SPANCALL_HIT.value, 0),
        "total_llm_calls": sum(t.total_llm_calls for t in traces),
        "total_tokens_in": sum(t.total_tokens_prompt for t in traces),
        "total_tokens_out": sum(t.total_tokens_completion for t in traces),
        "mean_task_latency_ms": round(sum(t.total_latency_ms for t in traces) / n, 2),
        "decisions": dict(decisions),
        "diagnostic_buckets": dict(buckets),
    }


def run_simple_tau_retail(
    *,
    resolver: Any,
    backend: BaseLLMBackend,
    modes: Sequence[str],
    tasks: Optional[Sequence[TauRetailTask]] = None,
    skip_risk_tiers: Tuple[str, ...] = (),
    max_turns: int = 12,
    max_context_tokens: int = 16384,
    context_budget_tokens: int = 2048,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[ExecutionTrace]]]:
    suite = list(tasks or build_simple_tau_retail_suite())
    summaries: List[Dict[str, Any]] = []
    traces_by_mode: Dict[str, List[ExecutionTrace]] = {}

    for m in modes:
        router = ToolCallRouter(
            resolver=resolver,
            llm_backend=backend,
            mode=RouterMode(m),
            variant=IntegrationVariant.S_INJECT,
            skip_risk_tiers=skip_risk_tiers,
        )
        mode_traces: List[ExecutionTrace] = []
        for task in suite:
            env = SimpleTauRetailEnvironment(goal_checker=task.goal_checker)
            loop = ReActAgentLoop(
                router=router,
                environment=env,
                max_turns=max_turns,
                system_prompt=TAU_RETAIL_SYSTEM_PROMPT,
                max_context_tokens=max_context_tokens,
                context_budget_tokens=context_budget_tokens,
                prevent_duplicate_calls=True,
            )
            user_sim = ScriptedUserSimulator(task.user_replies)
            tr = loop.run_task(task.instruction, task.task_id, user_simulator=user_sim)
            tr.metadata["category"] = task.category
            mode_traces.append(tr)
        traces_by_mode[m] = mode_traces
        summaries.append(summarize_mode_traces(m, mode_traces))

    return summaries, traces_by_mode


def _format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Mode | Tasks | Success | Avg Turns | SpanCall Hits | LLM Calls | Prompt Tok | Compl Tok | Mean Lat (ms) | Diagnostics |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in rows:
        diag = ", ".join(f"{k}={v}" for k, v in sorted((r.get("diagnostic_buckets") or {}).items())) or "-"
        lines.append(
            f"| `{r['mode']}` | {r['tasks']} | {r['success_rate']:.3f} | {r['avg_turns']:.2f} | "
            f"{r['total_spancall_hits']} | {r['total_llm_calls']} | {r['total_tokens_in']} | "
            f"{r['total_tokens_out']} | {r['mean_task_latency_ms']:.1f} | {diag} |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate SpanCall agent modes on Simple τ-bench Retail.")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["llm_only", "router_schema", "shadow"],
        choices=[m.value for m in RouterMode],
    )
    ap.add_argument("--backend", choices=["deterministic", "openai"], default="deterministic")
    ap.add_argument("--model", default=None, help="LLM model ID (overrides BASE_LLM_MODEL in .env)")
    ap.add_argument("--spancall-checkpoint", default=None, help="Local dir or HF repo (e.g. convoicon/spancall-lean-v3-3)")
    ap.add_argument("--device", default=None, help="Torch device for SpanCall resolver")
    ap.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention kernel for SpanCall encoder (auto selects flash_attention_2 on Ampere+ CUDA if available, else sdpa)",
    )
    ap.add_argument(
        "--dtype",
        default=None,
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Optional weight dtype for SpanCall encoder",
    )
    ap.add_argument("--skip-high-risk", action="store_true", help="Route risk_tier='high' tools to LLM")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of tasks (0 = all 10)")
    ap.add_argument("--out-dir", default="", help="Optional directory for JSONL traces and summary.json")
    args = ap.parse_args(argv)

    if args.spancall_checkpoint:
        from spancall.hub import load_resolver

        resolver = load_resolver(
            args.spancall_checkpoint,
            device=args.device,
            attn_implementation=args.attn_implementation,
            torch_dtype=args.dtype,
        )
    else:
        resolver = NoOpResolver()

    if args.backend == "openai":
        backend: BaseLLMBackend = OpenAIAPIBackend.from_env(model_id=args.model)
    else:
        backend = DeterministicTauRetailBackend()

    tasks = build_simple_tau_retail_suite()
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    skip_risk = ("high",) if args.skip_high_risk else ()
    summaries, traces_by_mode = run_simple_tau_retail(
        resolver=resolver,
        backend=backend,
        modes=args.modes,
        tasks=tasks,
        skip_risk_tiers=skip_risk,
        max_turns=args.max_turns,
    )

    print(_format_markdown_table(summaries))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        (out_dir / "summary.md").write_text(_format_markdown_table(summaries) + "\n", encoding="utf-8")
        for mode, m_traces in traces_by_mode.items():
            p = out_dir / f"traces_{mode}.jsonl"
            with p.open("w", encoding="utf-8") as fh:
                for tr in m_traces:
                    fh.write(json.dumps(serialize_trace(tr), ensure_ascii=False) + "\n")
        print(f"\nWrote traces and summary to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
