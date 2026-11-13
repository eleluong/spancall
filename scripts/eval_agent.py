#!/usr/bin/env python3
"""CLI entrypoint for comparative agent evaluation (Normal Agent vs Agent + SpanCall).

Runs the E9-E11 protocol (docs/experiments.md):
- Paired 3-seed evaluation of B0 (Normal Agent, ``LLM_ONLY``) vs S
  (Agent + SpanCall Router) over s_skip / s_inject variants.
- Non-inferiority test on task success (margin 2 pts, one-sided bootstrap
  p-value) + empirical latency/token/cost reduction vs the cost formula.

Usage:
    python scripts/eval_agent.py --benchmark custom_mock --variant s_inject \\
        --num_seeds 3 [--spancall_checkpoint <dir>] [--output results/e9.json]

When ``litellm`` is unavailable (or ``--backend deterministic``), a
rule-based backend drives the mock booking environment so the full pipeline
can be smoke-tested offline with reproducible results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.backends.litellm_adapter import UniversalAPIBackend
from spancall.agent.loop import ReActAgentLoop
from spancall.agent.router import IntegrationVariant, RouterMode, ToolCallRouter
from spancall.agent.tools.mock import MockBookingEnvironment
from spancall.agent.trace import RoutingDecision
from spancall.data.schema import ToolDefinition
from spancall.eval.agentic.evaluator import MultiSeedAgentEvaluator
from spancall.eval.compare import ComparativeEvaluator, ComparativeStudyResult
from spancall.eval.cost import CostModel
from spancall.infer.base import BaseResolveResult, BaseResolver, NoOpResolver, ResolveResult
from spancall.lean.resolver import LeanResolver


# --------------------------------------------------------------------------- #
# Task suite (deterministic mock booking domain)
# --------------------------------------------------------------------------- #
def _task(task_id: str, instruction: str, required_flight_id: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "instruction": instruction,
        "environment": lambda rng: MockBookingEnvironment(required_flight_id=required_flight_id),  # noqa: E731
    }


def build_task_suite() -> List[Dict[str, Any]]:
    """Three canonical multi-turn booking tasks (F001/F002/F003)."""
    return [
        _task("book_f001", "Book flight F001.", "F001"),
        _task(
            "book_f002",
            "Find a flight from SFO to LAX on 2026-11-05 and book the cheapest one.",
            "F002",
        ),
        _task(
            "book_f003",
            "Search for a flight from JFK to LHR on 2026-11-06, then book it.",
            "F003",
        ),
    ]


def build_long_trajectory_suite() -> List[Dict[str, Any]]:
    """Long-trajectory benchmark requiring multi-turn grounding and context management."""
    return [
        _task("long_f001", "Please find available flights from SFO to JFK on 2026-11-05, inspect options, and book the best flight.", "F001"),
        _task("long_f002", "Find a flight from SFO to LAX on 2026-11-05 and book the cheapest one.", "F002"),
        _task("long_f003", "Search for a flight from JFK to LHR on 2026-11-06, then book it.", "F003"),
        _task("long_direct", "Review the schedule and book flight F001 directly for the user.", "F001"),
    ]


# --------------------------------------------------------------------------- #
# Deterministic fallback backend (offline smoke runs, no litellm needed)
# --------------------------------------------------------------------------- #
class DeterministicBackend(BaseLLMBackend):
    """Rule-based backend that completes the mock booking tasks offline.

    Turn 0: parse the task instruction for a search or a direct book.
    Later turns: read the last tool output from ``messages``; if it contains
    a flight id, issue ``book_flight``; otherwise search with the route
    parsed from the instruction.
    """

    def __init__(self, latency_ms: float = 5.0) -> None:
        self.latency_ms = float(latency_ms)

    def generate_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        instruction = self._collect_user_text(messages)
        tool_output = self._last_tool_output(messages)

        arguments: Optional[Dict[str, Any]] = None
        tool_name: Optional[str] = None

        # 1) A prior search returned a flight id -> book it.
        if tool_output:
            fid = self._flight_id_in(tool_output)
            if fid:
                tool_name = "book_flight"
                arguments = {"flight_id": fid}
        # 2) The instruction names a flight id directly ("Book flight F001").
        if tool_name is None:
            fid = self._flight_id_in(instruction)
            if fid and "book" in instruction.lower():
                tool_name = "book_flight"
                arguments = {"flight_id": fid}
        # 3) Otherwise parse the route; search first, book comes next turn.
        if tool_name is None:
            parsed = self._parse_route(instruction)
            if parsed:
                tool_name = "search_flights"
                arguments = {
                    "origin": parsed[0],
                    "destination": parsed[1],
                    "date": parsed[2],
                }

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if tool_name is None:
            return {
                "tool_name": None,
                "arguments": None,
                "text": "I can help with that booking. Please provide the route and date.",
                "usage": usage,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "usage": usage,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        return "Task complete."

    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect_user_text(messages: List[Dict[str, Any]]) -> str:
        return " ".join(
            str(m.get("content") or "")
            for m in messages
            if m.get("role") == "user"
        )

    @staticmethod
    def _last_tool_output(messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "tool":
                return str(m.get("content") or "")
        return ""

    @staticmethod
    def _flight_id_in(text: str) -> Optional[str]:
        m = re.search(r"\b(F\d{3})\b", text)
        return m.group(1) if m else None

    @staticmethod
    def _parse_route(text: str) -> Optional[List[str]]:
        """Extract (origin, destination, date) from Free text."""
        m = re.search(
            r"from\s+([A-Za-z]{3})\s+to\s+([A-Za-z]{3})"
            r"(?:\s+on\s+(\d{4}-\d{2}-\d{2}))?",
            text,
            re.IGNORECASE,
        )
        if not m:
            return None
        origin = m.group(1).upper()
        destination = m.group(2).upper()
        date = m.group(3) or "2026-11-05"
        return [origin, destination, date]


# --------------------------------------------------------------------------- #
# Resolver construction
# --------------------------------------------------------------------------- #
class _ScriptedResolver(BaseResolver):
    """Checkpoint-free resolver that exercises the fast path on trajectory context.

    Hits ``search_flights`` from the utterance route on turn 0, and ``book_flight``
    on later turns when a flight id appears in ``recent_context`` (grounded in prior
    tool output). Validates context plumbing and per-turn stats only — NOT a
    measurement of the learned model's hit-rate.
    """

    def try_resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Any] = None,
    ) -> BaseResolveResult:
        return self.try_resolve_candidates([schema], utterance, recent_context=recent_context, now=now)

    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[Any] = None,
    ) -> ResolveResult:
        names = {s.name for s in candidate_schemas}
        feats = {
            "n_recent_rows": len(recent_context or []),
            "recent_tokens": sum(len(str(r.get("value", ""))) // 4 for r in (recent_context or [])),
        }
        fid = None
        for r in recent_context or []:
            m = re.search(r"\b(F\d{3})\b", str(r.get("value", "")))
            if m:
                fid = m.group(1)
        if fid and "book_flight" in names:
            return ResolveResult(hit=True, tool_name="book_flight", payload={"flight_id": fid},
                                 joint_margin=3.0, accept_score=0.99, features=feats)
        route = DeterministicBackend._parse_route(utterance)
        if route and "search_flights" in names and not recent_context:
            return ResolveResult(
                hit=True, tool_name="search_flights",
                payload={"origin": route[0], "destination": route[1], "date": route[2]},
                joint_margin=3.0, accept_score=0.99, features=feats)
        return ResolveResult(hit=False, abstain_reason="scripted_no_match", features=feats)


def build_resolver(
    checkpoint: Optional[str],
    calib_thresholds: Optional[str],
    scripted: bool = False,
) -> BaseResolver:
    if scripted:
        return _ScriptedResolver()
    if checkpoint:
        return LeanResolver.from_checkpoint(
            checkpoint_dir=checkpoint,
            calib_thresholds_path=calib_thresholds,
        )
    # No checkpoint: SpanCall abstains on everything, so the ROUTER behaves
    # exactly like the LLM baseline (a fully honest fast path that never hits).
    return NoOpResolver(abstain_reason="no_checkpoint")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _make_llm_backend(args: argparse.Namespace) -> BaseLLMBackend:
    if args.backend == "deterministic":
        return DeterministicBackend()
    try:
        backend = UniversalAPIBackend(model_id=args.llm_backend)
        # Probe imports so we fail fast with a clear message when litellm is
        # missing but the user asked for a live backend.
        import litellm  # noqa: F401
    except ImportError:
        if args.backend == "universal":
            raise
        print("[SpanCall Eval: Agent] litellm unavailable — using deterministic backend")
        return DeterministicBackend()
    return backend


def _make_router(
    backend: BaseLLMBackend,
    resolver: BaseResolver,
    mode: str,
    variant: str,
) -> ToolCallRouter:
    return ToolCallRouter(
        resolver=resolver,
        llm_backend=backend,
        mode=RouterMode(mode),
        variant=IntegrationVariant(variant),
    )


def _make_loop(
    router: ToolCallRouter,
    max_turns: int,
    max_context_tokens: int = 32768,
    keep_last_tool_turns: int = 8,
    k_recent: int = 32,
    context_budget_tokens: int = 16384,
    legacy_context: bool = False,
) -> ReActAgentLoop:
    env = MockBookingEnvironment(required_flight_id="F000")  # placeholder; evaluator rebinds
    return ReActAgentLoop(
        router=router,
        environment=env,
        max_turns=max_turns,
        max_context_tokens=max_context_tokens,
        keep_last_tool_turns=keep_last_tool_turns,
        k_recent=k_recent,
        context_budget_tokens=context_budget_tokens,
        legacy_context=legacy_context,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpanCall Comparative Agent Evaluation")
    parser.add_argument("--benchmark", type=str, default="custom_mock", choices=["tau_bench", "custom_mock", "long_trajectory"])
    parser.add_argument("--llm_backend", type=str, default="google/gemma-4")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "universal", "deterministic"])
    parser.add_argument("--spancall_checkpoint", type=str, default=None)
    parser.add_argument("--calib_thresholds", type=str, default=None)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--variant", type=str, default="s_skip", choices=["s_skip", "s_inject"])
    parser.add_argument("--mode", type=str, default="router", choices=["router", "spancall_only", "shadow"])
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--max_context_tokens", type=int, default=32768)
    parser.add_argument("--keep_last_tool_turns", type=int, default=8)
    parser.add_argument("--k_recent", type=int, default=32)
    parser.add_argument("--context_budget_tokens", type=int, default=16384)
    parser.add_argument("--legacy_context", action="store_true", help="Force pre-fix legacy context feeding (empty utterance after turn 0, env rows only)")
    parser.add_argument("--scripted_resolver", action="store_true", help="Use a checkpoint-free scripted resolver to exercise fast-path context plumbing (not a model measurement)")
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--num_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.benchmark == "tau_bench":
        print(f"[SpanCall Eval: Agent] benchmark '{args.benchmark}' not wired; use custom_mock or long_trajectory")
        return 2

    if args.benchmark == "long_trajectory":
        task_suite = build_long_trajectory_suite()
    else:
        task_suite = build_task_suite()

    if args.limit is not None:
        task_suite = task_suite[: args.limit]
    if not task_suite:
        print("[SpanCall Eval: Agent] empty task suite")
        return 2

    backend = _make_llm_backend(args)
    resolver = build_resolver(args.spancall_checkpoint, args.calib_thresholds, scripted=args.scripted_resolver)
    evaluator = MultiSeedAgentEvaluator(seed=args.seed)

    # ---- B0: Normal Agent (LLM_ONLY) -------------------------------------- #
    normal_router = _make_router(backend, resolver, "llm_only", args.variant)
    normal_loop = _make_loop(
        normal_router,
        args.max_turns,
        max_context_tokens=args.max_context_tokens,
        keep_last_tool_turns=args.keep_last_tool_turns,
        k_recent=args.k_recent,
        context_budget_tokens=args.context_budget_tokens,
        legacy_context=args.legacy_context,
    )
    normal_metrics, normal_traces = evaluator.evaluate_agent(
        normal_loop, task_suite, num_seeds=args.num_seeds
    )

    # ---- S: Agent + SpanCall (ROUTER) ------------------------------------- #
    spancall_router = _make_router(backend, resolver, args.mode, args.variant)
    spancall_loop = _make_loop(
        spancall_router,
        args.max_turns,
        max_context_tokens=args.max_context_tokens,
        keep_last_tool_turns=args.keep_last_tool_turns,
        k_recent=args.k_recent,
        context_budget_tokens=args.context_budget_tokens,
        legacy_context=args.legacy_context,
    )
    spancall_metrics, spancall_traces = evaluator.evaluate_agent(
        spancall_loop, task_suite, num_seeds=args.num_seeds
    )

    # ---- Paired non-inferiority + bootstrap ------------------------------- #
    comparer = ComparativeEvaluator(
        cost_model=CostModel(),
        non_inferiority_margin=args.margin,
        num_bootstrap_resamples=args.num_bootstrap,
        seed=args.seed,
    )
    study: ComparativeStudyResult = comparer.compare(normal_traces, spancall_traces)

    out_path = args.output or f"results/agent_compare_{args.variant}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = study.to_dict()
    payload["integration_variant"] = args.variant
    payload["mode"] = args.mode
    payload["num_seeds"] = args.num_seeds
    payload["spancall_checkpoint"] = args.spancall_checkpoint
    payload["max_context_tokens"] = args.max_context_tokens
    payload["context_budget_tokens"] = args.context_budget_tokens
    payload["k_recent"] = args.k_recent
    payload["legacy_context"] = args.legacy_context
    payload["scripted_resolver"] = args.scripted_resolver
    payload["calibration_status"] = "pending_model_retraining"

    # Per-turn statistics over all evaluated agent traces
    per_turn_stats: Dict[str, Dict[str, Any]] = {}
    abstain_reasons: Dict[str, int] = {}
    for tr in spancall_traces:
        for st in tr.steps:
            t_key = f"turn_{st.turn_index}"
            if t_key not in per_turn_stats:
                per_turn_stats[t_key] = {
                    "turns_observed": 0,
                    "spancall_hits": 0,
                    "spancall_abstains": 0,
                    "llm_fallbacks": 0,
                    "n_recent_rows_sum": 0,
                    "recent_tokens_sum": 0,
                    "context_trimmed_sum": 0,
                }
            per_turn_stats[t_key]["turns_observed"] += 1
            if st.decision == RoutingDecision.SPANCALL_HIT:
                per_turn_stats[t_key]["spancall_hits"] += 1
            elif st.decision == RoutingDecision.SPANCALL_ABSTAIN:
                per_turn_stats[t_key]["spancall_abstains"] += 1
            elif st.decision in (RoutingDecision.LLM_FALLBACK, RoutingDecision.SHADOW_COMPARISON):
                per_turn_stats[t_key]["llm_fallbacks"] += 1

            rf = st.spancall_features or {}
            per_turn_stats[t_key]["n_recent_rows_sum"] += int(rf.get("n_recent_rows", 0) or 0)
            per_turn_stats[t_key]["recent_tokens_sum"] += int(rf.get("recent_tokens", 0) or 0)
            per_turn_stats[t_key]["context_trimmed_sum"] += int(getattr(st, "context_trimmed", 0) or 0)

            reason = rf.get("abstain_reason") or rf.get("h4_reason")
            if reason:
                s_reason = str(reason)
                abstain_reasons[s_reason] = abstain_reasons.get(s_reason, 0) + 1

    payload["per_turn_stats"] = per_turn_stats
    payload["abstain_reasons"] = abstain_reasons

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")

    print(f"[SpanCall Eval: Agent] wrote {out_path}")
    print(f"[SpanCall Eval: Agent] tasks={study.num_tasks} seeds={args.num_seeds} variant={args.variant}")
    print(f"[SpanCall Eval: Agent] B0  success={study.baseline_metrics.task_success_rate:.3f} "
          f"turns/task={study.baseline_metrics.average_turns_per_task:.2f}")
    print(f"[SpanCall Eval: Agent] S   success={study.spancall_metrics.task_success_rate:.3f} "
          f"turns/task={study.spancall_metrics.average_turns_per_task:.2f}")
    print(f"[SpanCall Eval: Agent] delta={study.task_success_delta:+.3f} "
          f"95%CI=[{study.bootstrap_ci_95[0]:+.3f},{study.bootstrap_ci_95[1]:+.3f}] "
          f"p={study.p_value:.4f} non_inferiority_passed={study.non_inferiority_passed}")
    print(f"[SpanCall Eval: Agent] speedup={study.speedup_ratio:.2f}x "
          f"tokens_saved={study.token_savings_percent:.1f}% "
          f"empirical_cost_saved={study.empirical_cost_saving_percent:.1f}% "
          f"theoretical_cost_saved={study.theoretical_cost_saving_percent:.1f}%")
    print("[SpanCall Eval: Agent] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
