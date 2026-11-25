"""Agentic multi-turn interaction loop."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from spancall.agent.router import (
    IntegrationVariant,
    RouterMode,
    ToolCallRouter,
    _canonical_call_key,
)
from spancall.agent.tools.base import BaseToolEnvironment, ToolExecutionResult
from spancall.agent.trace import ExecutionTrace, RoutingDecision, StepTrace


class BaseAgentLoop(ABC):
    """Abstract multi-turn agent interaction loop.

    Concrete loops execute a task against a :class:`BaseToolEnvironment`
    (routing through SpanCall and/or an LLM backend) and return an
    :class:`ExecutionTrace` telemetry record.
    """

    @abstractmethod
    def run_task(self, task_instruction: str, task_id: str) -> ExecutionTrace:
        """Runs a single task to completion (or the turn cap)."""
        raise NotImplementedError


class ReActAgentLoop(BaseAgentLoop):
    """Multi-turn agent loop with SpanCall routing integration.

    Each turn:
      1. Append the user's new instruction to the OpenAI-style message list.
      2. ``router.route_turn(...)`` decides between SpanCall fast path and the
         LLM backend (per ``RouterMode`` / ``IntegrationVariant``).
      3. When a tool call is produced — from SpanCall (``s_skip``) or from the
         LLM — the tool executes in the environment; the *tool response* is
         appended to ``messages`` so subsequent turns see the result.
      4. ``s_inject`` additionally synthesizes the assistant ``tool_calls``
         message so the history is byte-identical to native LLM tool calling.

    The loop never calls ``router.route_turn`` with a mutated system prompt;
    the fallback path always uses the LLM's unchanged native template
    (docs/architecture.md §7).

    Args:
        router: router managing SpanCall fast path + LLM fallback.
        environment: stateful mock/benchmark tool environment.
        max_turns: safety cap on total assistant+tool steps per task.
        system_prompt: optional system message prepended once to the history.
    """

    def __init__(
        self,
        router: ToolCallRouter,
        environment: BaseToolEnvironment,
        max_turns: int = 15,
        system_prompt: Optional[str] = None,
        max_context_tokens: int = 32768,
        keep_last_tool_turns: int = 8,
        k_recent: int = 32,
        context_budget_tokens: int = 16384,
        legacy_context: bool = False,
        tokenizer: Optional[Any] = None,
        prevent_duplicate_calls: bool = True,
    ) -> None:
        self.router = router
        self.environment = environment
        self.max_turns = int(max_turns)
        self.system_prompt = system_prompt
        self.max_context_tokens = int(max_context_tokens)
        self.keep_last_tool_turns = int(keep_last_tool_turns)
        self.k_recent = int(k_recent)
        self.context_budget_tokens = int(context_budget_tokens)
        self.legacy_context = bool(legacy_context)
        self.prevent_duplicate_calls = bool(prevent_duplicate_calls)
        # Real tokenizer => token-accurate budgets. Falls back to the resolver's
        # tokenizer (never the char-level FakeTokenizer, which overcounts ~4x),
        # else the length/4 estimate.
        if tokenizer is None:
            cand = getattr(getattr(router, "resolver", None), "tokenizer", None)
            if cand is not None and type(cand).__name__ != "FakeTokenizer":
                tokenizer = cand
        self.tokenizer = tokenizer

    def run_task(
        self,
        task_instruction: str,
        task_id: str,
        user_simulator: Optional[Callable[[str, List[Dict[str, Any]]], Optional[str]]] = None,
    ) -> ExecutionTrace:
        """Executes multi-turn task until completion or max_turns reached.

        When ``user_simulator`` is provided (e.g., τ-bench interactive user),
        any assistant prose turn (non-tool-call) is sent to ``user_simulator``;
        returning ``None`` or ``"###STOP###"`` ends the conversation, otherwise
        the simulated user reply is appended as a new ``role: "user"`` turn.
        """
        t0 = time.perf_counter()
        mode: RouterMode = (
            self.router.mode if isinstance(self.router.mode, RouterMode)
            else RouterMode(self.router.mode)
        )
        variant: IntegrationVariant = (
            self.router.variant if isinstance(self.router.variant, IntegrationVariant)
            else IntegrationVariant(self.router.variant)
        )

        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": task_instruction})

        trace = ExecutionTrace(
            task_id=task_id,
            environment_name=self.environment.__class__.__name__,
            mode=mode.value,
            integration_variant=variant.value,
        )

        self.environment.reset()
        executed_calls: Set[Tuple[str, str]] = set()
        for turn in range(self.max_turns):
            if self.legacy_context:
                turn_utterance = task_instruction if turn == 0 else ""
                turn_context = self.environment.session_context()
                trimmed_count = 0
                recent_features = {}
            else:
                from spancall.agent.context import (
                    latest_user_utterance,
                    rows_from_messages,
                    select_recent_context,
                    trim_messages,
                )

                messages, trimmed_count = trim_messages(
                    messages,
                    max_context_tokens=self.max_context_tokens,
                    keep_last_tool_turns=self.keep_last_tool_turns,
                    tokenizer=self.tokenizer,
                )
                turn_utterance = latest_user_utterance(messages, fallback=task_instruction)
                traj_rows = rows_from_messages(messages, exclude_last_user=True)
                env_context = self.environment.session_context() or []
                combined_rows = traj_rows + list(env_context)
                turn_context, recent_features = select_recent_context(
                    combined_rows,
                    tools=self.environment.get_available_tools(),
                    k_recent=self.k_recent,
                    context_budget_tokens=self.context_budget_tokens,
                    tokenizer=self.tokenizer,
                )

            # ---- route this turn ------------------------------------------ #
            route_kwargs: Dict[str, Any] = {
                "utterance": turn_utterance,
                "messages": messages,
                "available_tools": self.environment.get_available_tools(),
                "session_context": turn_context,
            }
            if self.prevent_duplicate_calls:
                route_kwargs["executed_calls"] = executed_calls
            result, step = self.router.route_turn(**route_kwargs)
            step = step or self._bare_trace(turn, turn_utterance, result)
            step.turn_index = turn
            step.context_trimmed = trimmed_count
            if isinstance(step.spancall_features, dict):
                step.spancall_features.update(recent_features)
                step.spancall_features["context_trimmed"] = trimmed_count
                resolver_max_seq = getattr(getattr(self.router, "resolver", None), "max_seq_len", None)
                seq_tok = (
                    step.spancall_features.get("seq_tokens")
                    or step.spancall_features.get("seq_len")
                    or step.spancall_features.get("num_tokens")
                )
                abstain_str = str(step.spancall_features.get("spancall_abstain_reason") or "").lower()
                if (
                    step.spancall_features.get("context_overflow_seq")
                    or "overflow" in abstain_str
                    or (resolver_max_seq and seq_tok and int(seq_tok) >= int(resolver_max_seq))
                ):
                    step.spancall_features["context_overflow_seq"] = True
                if "shadow_false_hit_on_prose" in result:
                    step.spancall_features["shadow_false_hit_on_prose"] = bool(
                        result["shadow_false_hit_on_prose"]
                    )
                if "spancall_matched" in result:
                    step.spancall_features["shadow_matched"] = bool(result["spancall_matched"])
            trace.total_context_trimmed += trimmed_count
            stop_requested = False
            if result.get("hit"):
                tool_name = str(result.get("tool_name") or "")
                arguments = result.get("arguments") or {}
                step.selected_tool = tool_name
                step.tool_arguments = dict(arguments)
                executed_calls.add(_canonical_call_key(tool_name, arguments))

                # ---- synthesize assistant tool_calls message ---------------- #
                if variant == IntegrationVariant.S_INJECT:
                    call_id = str(result.get("call_id") or f"call_spancall_{turn}")
                    inj = self.router.synthesize_injected_call(tool_name, arguments, call_id)
                    if result.get("text"):
                        inj["content"] = str(result["text"])
                    if result.get("spancall"):
                        step.spancall_features["injected_call_id"] = call_id
                    messages.append(inj)
                    tool_messages = [self.router.synthesize_tool_response(tool_name, None, call_id)]
                else:
                    tool_messages = []

                # ---- execute the tool in the environment ------------------- #
                exec_result: ToolExecutionResult = (
                    self.environment.execute(tool_name, arguments)
                )
                step.tool_execution_success = exec_result.success
                # If a mutation changed environment state, allow subsequent
                # re-reads of read-only (non-high-risk) tools so legitimate
                # verification lookups are not blocked by the duplicate guard.
                if exec_result.state_changed and executed_calls:
                    high_risk_names = {
                        t.name
                        for t in self.environment.get_available_tools()
                        if str(getattr(t, "risk_tier", "low") or "low").lower() == "high"
                    }
                    executed_calls = {
                        ck for ck in executed_calls if ck[0] in high_risk_names
                    }
                step.tool_output = (
                    exec_result.output if isinstance(exec_result.output, str)
                    else _stringify(exec_result.output)
                )

                # ---- append the tool response to history ------------------- #
                content = step.tool_output if step.tool_output else ""
                if tool_messages:
                    tool_messages[0]["content"] = content
                    messages.extend(tool_messages)
                else:
                    messages.append(
                        self.router.synthesize_tool_response(tool_name, content)
                    )
                    if result.get("text"):
                        messages.append({"role": "assistant", "content": str(result["text"])})
            else:
                # no tool call this turn: LLM prose, or SpanCall abstain
                text = result.get("text") or (result.get("content") or "")
                if result.get("tool_name") is None and not text:
                    text = result.get("abstain_reason") or "no tool call"
                messages.append({"role": "assistant", "content": str(text)})
                if user_simulator is not None:
                    user_reply = user_simulator(str(text), list(messages))
                    if user_reply is None or "###STOP###" in str(user_reply):
                        stop_requested = True
                    else:
                        messages.append({"role": "user", "content": str(user_reply)})

            trace.steps.append(step)
            trace.total_llm_calls += 1 if step.decision in (
                RoutingDecision.LLM_FALLBACK, RoutingDecision.SHADOW_COMPARISON,
                RoutingDecision.HYBRID_FILL) else 0
            if step.decision in (
                RoutingDecision.SPANCALL_HIT, RoutingDecision.SPANCALL_ABSTAIN,
                RoutingDecision.HYBRID_FILL,
            ):
                trace.total_spancall_calls += 1
            trace.total_latency_ms += step.latency_ms
            trace.total_tokens_prompt += step.llm_tokens_in
            trace.total_tokens_completion += step.llm_tokens_out

            # ---- task-completion check -------------------------------------- #
            if stop_requested:
                trace.task_completed = True
                trace.task_success = bool(self.environment.evaluate_task_success())
                break
            if user_simulator is None and self.environment.evaluate_task_success():
                trace.task_completed = True
                trace.task_success = True
                break

        if not trace.task_completed and user_simulator is not None:
            trace.task_success = bool(self.environment.evaluate_task_success())

        trace.total_latency_ms = (time.perf_counter() - t0) * 1000
        if not trace.steps:
            # zero-turn degenerate: record the abstain for telemetry
            trace.steps.append(self._bare_trace(0, task_instruction, {"hit": False}))
        return trace

    @staticmethod
    def _bare_trace(turn: int, user_input: str, result: Dict[str, Any]) -> StepTrace:
        return StepTrace(
            turn_index=turn,
            user_input=user_input,
            decision=(
                RoutingDecision.SPANCALL_HIT if result.get("hit")
                else RoutingDecision.LLM_FALLBACK
            ),
            selected_tool=result.get("tool_name"),
            tool_arguments=result.get("arguments"),
            latency_ms=float(result.get("latency_ms", 0.0) or 0.0),
        )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        import json
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)
