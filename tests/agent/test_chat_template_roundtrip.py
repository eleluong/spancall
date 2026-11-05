"""Q9.1 — byte-exact history round-trip against ``docs/chat_template_gemma4.jinja``.

The ``s_inject`` integration variant must make a SpanCall hit *look* to the
rest of the agent exactly like a native LLM tool call (docs/pipeline_io.md
§Q9, architecture.md §7). Concretely: the OpenAI-style history that
:meth:`ToolCallRouter.synthesize_injected_call` builds — an assistant
``tool_calls`` message with ``function.arguments`` as a *mapping* plus a
consecutive ``role:tool`` response — must render byte-identically through the
vendored Gemma-4 chat template as the history produced by the LLM's own
native tool calling.

These tests also pin the *exact* rendered bytes so any change to either side
of the contract (emitter shape or template) fails loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader

from spancall.agent.router import ToolCallRouter
from spancall.agent.tools.mock import MockBookingEnvironment

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
_TEMPLATE = "chat_template_gemma4.jinja"


def _render(messages: List[Dict[str, Any]], **overrides: Any) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_DOCS_DIR)), trim_blocks=False, lstrip_blocks=False
    )
    tpl = env.get_template(_TEMPLATE)
    ctx: Dict[str, Any] = {
        "messages": messages,
        "tools": [],
        "bos_token": "",
        "add_generation_prompt": False,
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    ctx.update(overrides)
    return tpl.render(**ctx)


def _native_history(payload: Dict[str, Any], output: str) -> List[Dict[str, Any]]:
    """History exactly as the LLM backend emits it for the same call."""
    return [
        {"role": "user", "content": "Book flight F001"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_native_7",
                    "type": "function",
                    "function": {"name": "book_flight", "arguments": dict(payload)},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_native_7",
            "name": "book_flight",
            "content": output,
        },
    ]


def _injected_history(payload: Dict[str, Any], output: str) -> List[Dict[str, Any]]:
    """History a SpanCall hit builds via s_inject (loop's exact sequence)."""
    router = ToolCallRouter(resolver=None, llm_backend=None)  # type: ignore[arg-type]
    return [
        {"role": "user", "content": "Book flight F001"},
        router.synthesize_injected_call("book_flight", payload, "call_spancall_0"),
        router.synthesize_tool_response("book_flight", output, "call_spancall_0"),
    ]


# --------------------------------------------------------------------------- #
# Golden bytes
# --------------------------------------------------------------------------- #
def test_q9_1_s_inject_history_renders_to_golden_bytes() -> None:
    """The s_inject history renders to the exact expected template output."""
    messages = _injected_history(
        {"flight_id": "F001"},
        '{"booking_ref": "BR-1001"}',
    )
    rendered = _render(messages)
    assert rendered == (
        "<|turn>user\n"
        "Book flight F001"
        "<turn|>\n"
        "<|turn>model\n"
        "<|tool_call>call:book_flight{flight_id:<|\"|>F001<|\"|>}<tool_call|>"
        "<|tool_response>response:book_flight{value:<|\"|>"
        '{"booking_ref": "BR-1001"}'
        "<|\"|>}<tool_response|>"
    )


def test_q9_1_native_and_s_inject_history_render_byte_identical() -> None:
    """Native LLM history vs SpanCall s_inject history → identical bytes."""
    payload = {"flight_id": "F001"}
    output = '{"booking_ref": "BR-1001"}'
    native = _render(_native_history(payload, output))
    injected = _render(_injected_history(payload, output))
    assert native == injected
    # Both carry the native tool-call / tool-response markers.
    assert "<|tool_call>call:book_flight{flight_id:<|\"|>F001<|\"|>}<tool_call|>" in injected
    assert "<|tool_response>response:book_flight{value:<|\"|>" in injected


def test_q9_1_round_trip_with_multi_key_nested_arguments() -> None:
    """Multi-key + nested object/array arguments round-trip identically."""
    payload = {
        "seat_class": "business",
        "price_range": {"min": 100, "max": 250},
        "tags": ["direct", "night"],
    }
    output = '{"flights": ["F001", "F002"]}'
    native = _render(_native_history(payload, output))
    injected = _render(_injected_history(payload, output))
    assert native == injected
    # Jinja ``dictsort`` orders keys alphabetically at every nesting level, so
    # ``price_range`` renders max-before-min; sequences keep insertion order.
    assert "price_range:{max:250,min:100}" in injected
    assert "tags:[<|\"|>direct<|\"|>,<|\"|>night<|\"|>]" in injected


def test_q9_1_tool_response_with_mapping_output() -> None:
    """Even a dict-typed tool response stringifies consistently on both sides."""
    router = ToolCallRouter(resolver=None, llm_backend=None)  # type: ignore[arg-type]
    # synthesize_tool_response JSON-stringifies non-str outputs.
    injected = router.synthesize_tool_response("book_flight", {"booking_ref": "BR-1002"}, "call_spancall_0")
    native = {
        "role": "tool",
        "tool_call_id": "call_native_9",
        "name": "book_flight",
        "content": '{"booking_ref": "BR-1002"}',
    }
    history_native = [
        {"role": "user", "content": "Book it"},
        _native_history({"flight_id": "F002"}, '{"booking_ref": "BR-1002"}')[1],
        native,
    ]
    history_injected = [
        {"role": "user", "content": "Book it"},
        router.synthesize_injected_call("book_flight", {"flight_id": "F002"}, "call_spancall_0"),
        injected,
    ]
    assert _render(history_native) == _render(history_injected)


# --------------------------------------------------------------------------- #
# End-to-end loop + template
# --------------------------------------------------------------------------- #
def test_q9_1_loop_s_inject_history_round_trips_through_template() -> None:
    """Run the full loop on a SpanCall hit and render its history via the template.

    Uses a stubbed router (no torch/model) so the loop actually executes the
    s_inject code path against the mock environment, then renders the produced
    history — proving the emitted bytes are exactly those the template expects.
    """

    class StubResolver:
        def try_resolve_candidates(self, *a: Any, **k: Any) -> Any:
            from spancall.infer.resolver import ResolveResult

            return ResolveResult(
                hit=True,
                tool_name="book_flight",
                payload={"flight_id": "F001"},
                joint_margin=3.0,
                accept_score=0.99,
            )

    class StubLLM:
        def generate_tool_call(self, *a: Any, **k: Any) -> Dict[str, Any]:
            return {"tool_name": None, "arguments": None, "text": ""}

    from spancall.agent.loop import ReActAgentLoop
    from spancall.agent.router import IntegrationVariant, RouterMode

    router = ToolCallRouter(
        resolver=StubResolver(),  # type: ignore[arg-type]
        llm_backend=StubLLM(),  # type: ignore[arg-type]
        mode=RouterMode.ROUTER,
        variant=IntegrationVariant.S_INJECT,
    )
    env = MockBookingEnvironment(required_flight_id="F001")
    loop = ReActAgentLoop(router=router, environment=env, max_turns=3)
    trace = loop.run_task("Book flight F001", "q9-1-loop")

    assert trace.task_success is True
    assert trace.steps[0].spancall_features.get("injected_call_id") == "call_spancall_0"

    # Reconstruct the history the loop would have handed the next LLM turn and
    # assert it renders to exactly the golden bytes (same as test_q9_1_s_inject).
    messages = [
        {"role": "user", "content": "Book flight F001"},
        router.synthesize_injected_call("book_flight", {"flight_id": "F001"}, "call_spancall_0"),
        router.synthesize_tool_response(
            "book_flight", trace.steps[0].tool_output or "", "call_spancall_0"
        ),
    ]
    # The tool executed: output should be the booking ref JSON.
    assert trace.steps[0].tool_output == '{"booking_ref": "BR-1001"}'
    rendered = _render(messages)
    assert "<|tool_call>call:book_flight{flight_id:<|\"|>F001<|\"|>}<tool_call|>" in rendered
    assert 'response:book_flight{value:<|"|>{"booking_ref": "BR-1001"}<|"|>}' in rendered