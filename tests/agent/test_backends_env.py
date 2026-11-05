"""Tests for the deterministic mock tool environment and the LLM backend
adapters (plan §4.3).

Covers:
- ``MockBookingEnvironment``: search/book/cancel semantics, idempotent
  re-booking, session context feed, task-success evaluation, reset.
- ``UniversalAPIBackend`` (litellm adapter): OpenAI schema conversion from
  ``ToolDefinition``; graceful ``ImportError`` when litellm is absent;
  tool-call parsing against a stubbed ``litellm.completion``.
- ``LocalVLLMBackend``: native Gemma-4 tool-call text parsing
  (``<|tool_call>call:NAME{...}<tool_call|>``) into ``(tool_name, arguments)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.backends.litellm_adapter import (
    UniversalAPIBackend,
    _load_dotenv_values,
    _tool_definitions_to_openai,
)
from spancall.agent.backends.local_adapter import LocalVLLMBackend, _parse_tool_call
from spancall.agent.tools.base import ToolExecutionResult
from spancall.agent.tools.mock import MockBookingEnvironment, make_booking_tools
from spancall.data.schema import ParameterSchema, ToolDefinition


# --------------------------------------------------------------------------- #
# Mock booking environment
# --------------------------------------------------------------------------- #
def _env(**kw: Any) -> MockBookingEnvironment:
    return MockBookingEnvironment(**kw)


def test_mock_env_get_available_tools_returns_suite() -> None:
    env = _env()
    tools = env.get_available_tools()
    names = {t.name for t in tools}
    assert names == {"search_flights", "book_flight", "cancel_booking"}


def test_mock_env_search_flights_matches_inventory() -> None:
    env = _env()
    res = env.execute(
        "search_flights",
        {"origin": "SFO", "destination": "JFK", "date": "2026-11-05"},
    )
    assert res.success is True
    assert res.error is None
    assert [f["flight_id"] for f in res.output["flights"]] == ["F001"]


def test_mock_env_search_flights_price_filter() -> None:
    env = _env()
    res = env.execute(
        "search_flights",
        {"origin": "SFO", "destination": "JFK", "date": "2026-11-05", "max_price": 100},
    )
    assert res.success is True
    assert res.output["flights"] == []


def test_mock_env_search_flights_missing_required_fails() -> None:
    env = _env()
    res = env.execute("search_flights", {"origin": "SFO"})
    assert res.success is False
    assert "missing required argument" in (res.error or "")


def test_mock_env_book_flight_then_task_success() -> None:
    env = _env(required_flight_id="F001")
    assert env.evaluate_task_success() is False
    res = env.execute("book_flight", {"flight_id": "F001"})
    assert res.success is True
    assert res.output["booking_ref"] == "BR-1001"
    assert res.state_changed is True
    assert env.evaluate_task_success() is True


def test_mock_env_book_flight_idempotent_returns_existing_ref() -> None:
    env = _env(required_flight_id="F001")
    first = env.execute("book_flight", {"flight_id": "F001"})
    second = env.execute("book_flight", {"flight_id": "F001"})
    assert second.success is True
    assert second.output["booking_ref"] == first.output["booking_ref"]
    assert second.output["already_booked"] is True
    assert len(env.bookings) == 1


def test_mock_env_book_unknown_flight_fails() -> None:
    env = _env()
    res = env.execute("book_flight", {"flight_id": "NOPE"})
    assert res.success is False
    assert "unknown flight id" in (res.error or "")


def test_mock_env_booking_disabled_fails() -> None:
    env = _env(booking_enabled=False)
    res = env.execute("book_flight", {"flight_id": "F001"})
    assert res.success is False
    assert "disabled" in (res.error or "")


def test_mock_env_cancel_booking() -> None:
    env = _env()
    env.execute("book_flight", {"flight_id": "F001"})
    res = env.execute("cancel_booking", {"booking_ref": "BR-1001"})
    assert res.success is True
    assert res.output["remaining"] == 0
    assert env.bookings == []


def test_mock_env_cancel_missing_booking_fails() -> None:
    env = _env()
    res = env.execute("cancel_booking", {"booking_ref": "BR-9999"})
    assert res.success is False
    assert "not found" in (res.error or "")


def test_mock_env_unknown_tool_fails() -> None:
    env = _env()
    res = env.execute("fly_to_the_moon", {})
    assert res.success is False
    assert "unknown tool" in (res.error or "")


def test_mock_env_reset_clears_state() -> None:
    env = _env(required_flight_id="F001")
    env.execute("book_flight", {"flight_id": "F001"})
    assert env.evaluate_task_success() is True
    env.reset()
    assert env.bookings == []
    assert env._last_search == []
    assert env.evaluate_task_success() is False


def test_mock_env_session_context_after_search_and_book() -> None:
    env = _env(required_flight_id="F001")
    assert env.session_context() == []
    env.execute(
        "search_flights",
        {"origin": "SFO", "destination": "JFK", "date": "2026-11-05"},
    )
    rows = env.session_context()
    assert any(r["key"] == "flight_id" and r["value"] == "F001" for r in rows)
    env.execute("book_flight", {"flight_id": "F001"})
    rows = env.session_context()
    assert any(r["key"] == "booking_ref" and r["value"] == "BR-1001" for r in rows)


def test_mock_env_multi_turn_task_with_session_disambiguation() -> None:
    """The classic 2-step task: search SFO->LAX then book the returned flight.

    The second turn's session context carries flight F002, which a resolver
    with recent-context awareness uses to disambiguate ``book_flight``.
    """
    env = _env(required_flight_id="F002")
    search = env.execute(
        "search_flights",
        {"origin": "SFO", "destination": "LAX", "date": "2026-11-05"},
    )
    assert [f["flight_id"] for f in search.output["flights"]] == ["F002"]
    rows = env.session_context()
    fids = [r["value"] for r in rows if r["key"] == "flight_id"]
    assert fids == ["F002"]
    booked = env.execute("book_flight", {"flight_id": fids[0]})
    assert booked.success is True
    assert env.evaluate_task_success() is True


# --------------------------------------------------------------------------- #
# UniversalAPIBackend (litellm)
# --------------------------------------------------------------------------- #
def _booking_tool() -> ToolDefinition:
    return ToolDefinition(
        name="book_flight",
        description="Book a flight by id.",
        parameters=[
            ParameterSchema(name="flight_id", description="The flight id.", param_type="string"),
            ParameterSchema(
                name="seat_class",
                description="Cabin class.",
                param_type="string",
                required=False,
                enum=["economy", "premium", "business"],
            ),
        ],
    )


def test_tool_definitions_to_openai_shape() -> None:
    schemas = _tool_definitions_to_openai([_booking_tool()])
    assert len(schemas) == 1
    fn = schemas[0]["function"]
    assert fn["name"] == "book_flight"
    assert fn["parameters"]["required"] == ["flight_id"]
    assert fn["parameters"]["properties"]["seat_class"]["enum"] == ["economy", "premium", "business"]


def test_universal_backend_import_error_without_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graceful ImportError when litellm is absent.

    Skipped when litellm is actually installed (live LLM testing setups) —
    the stub test below covers the parsing path in that case.
    """
    import sys

    if "litellm" in sys.modules:
        pytest.skip("litellm is installed; ImportError path not applicable")
    monkeypatch.setitem(sys.modules, "litellm", None)
    backend = UniversalAPIBackend(model_id="gemma-4")
    with pytest.raises(ImportError, match="litellm"):
        backend.generate_tool_call([], [])


def test_universal_backend_stubbed_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a response without requiring the litellm SDK."""
    import json as _json
    import sys
    import types

    fake = types.ModuleType("litellm")
    kwargs_capture: Dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Dict[str, Any]:
        kwargs_capture.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "book_flight",
                                    "arguments": _json.dumps({"flight_id": "F001"}),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 55, "completion_tokens": 9},
        }

    fake.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake)

    backend = UniversalAPIBackend(model_id="gemma-4-9b", api_base="http://localhost:8000/v1")
    result = backend.generate_tool_call(
        [{"role": "user", "content": "Book F001"}], [_booking_tool()]
    )
    assert result["tool_name"] == "book_flight"
    assert result["arguments"] == {"flight_id": "F001"}
    assert result["usage"]["prompt_tokens"] == 55
    assert result["usage"]["completion_tokens"] == 9
    assert kwargs_capture["tool_choice"] == "auto"
    assert kwargs_capture["api_base"] == "http://localhost:8000/v1"
    # Messages must be passed through unmodified.
    assert kwargs_capture["messages"] == [{"role": "user", "content": "Book F001"}]


def test_load_dotenv_values_parses_export_quotes_comments(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment line\n"
        "BASE_URL = https://api.example.com/v1\n"
        'export API_KEY="sk-secret-123"\n'
        "BASE_LLM_MODEL='gemma-4-27b'\n"
        "TRAILING=value-with-=-sign\n"
        "NO_EQUALS_HERE\n"
        "\n",
        encoding="utf-8",
    )
    vals = _load_dotenv_values(env_file)
    assert vals["BASE_URL"] == "https://api.example.com/v1"
    assert vals["API_KEY"] == "sk-secret-123"
    assert vals["BASE_LLM_MODEL"] == "gemma-4-27b"
    assert vals["TRAILING"] == "value-with-=-sign"
    assert "NO_EQUALS_HERE" not in vals


def test_load_dotenv_values_missing_file_returns_empty(tmp_path) -> None:
    assert _load_dotenv_values(tmp_path / "nope.env") == {}


def test_universal_backend_constructor_env_fallback(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor args win over .env, which wins over process env."""
    import os

    env_file = tmp_path / ".env"
    env_file.write_text(
        "BASE_URL=https://env.example.com/v1\n"
        "API_KEY=env-key\n"
        "BASE_LLM_MODEL=env-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BASE_URL", "process-url")
    monkeypatch.setenv("API_KEY", "process-key")
    monkeypatch.setenv("BASE_LLM_MODEL", "process-model")

    # No explicit args -> .env supplies everything.
    backend = UniversalAPIBackend(env_path=env_file)
    assert backend.model_id == "env-model"
    assert backend.api_base == "https://env.example.com/v1"
    assert backend.api_key == "env-key"

    # Missing .env keys fall back to process environment.
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")
    backend2 = UniversalAPIBackend(env_path=empty)
    assert backend2.model_id == "process-model"
    assert backend2.api_base == "process-url"
    assert backend2.api_key == "process-key"

    # Explicit constructor args always win.
    backend3 = UniversalAPIBackend(
        model_id="explicit", api_base="https://explicit/v1", api_key="explicitkey", env_path=env_file
    )
    assert backend3.model_id == "explicit"


def test_universal_backend_from_env_classmethod(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BASE_URL=https://fromenv.example.com/v1\n"
        "API_KEY=fromenv-key\n"
        "BASE_LLM_MODEL=fromenv-model\n",
        encoding="utf-8",
    )
    backend = UniversalAPIBackend.from_env(env_path=env_file)
    assert backend.model_id == "fromenv-model"
    assert backend.api_base == "https://fromenv.example.com/v1"
    assert backend.api_key == "fromenv-key"
    assert isinstance(backend, UniversalAPIBackend)


# --------------------------------------------------------------------------- #
# LocalVLLMBackend — native Gemma-4 tool-call parsing
# --------------------------------------------------------------------------- #
def test_parse_tool_call_simple() -> None:
    text = "<|tool_call>call:book_flight{flight_id:<|\"|>F001<|\"|>}<tool_call|>"
    name, args = _parse_tool_call(text)
    assert name == "book_flight"
    assert args == {"flight_id": "F001"}


def test_parse_tool_call_multiple_args_and_types() -> None:
    text = (
        "<|tool_call>call:search_flights{origin:<|\"|>SFO<|\"|>,"
        "destination:<|\"|>JFK<|\"|>,max_price:100.0}<tool_call|>"
    )
    name, args = _parse_tool_call(text)
    assert name == "search_flights"
    assert args == {"origin": "SFO", "destination": "JFK", "max_price": 100.0}


def test_parse_tool_call_nested_object_and_array() -> None:
    text = (
        "<|tool_call>call:complex{obj:{a:true,b:false,c:null},"
        "list:[1,2,<|\"|>three<|\"|>]}<tool_call|>"
    )
    name, args = _parse_tool_call(text)
    assert name == "complex"
    assert args == {"obj": {"a": True, "b": False, "c": None}, "list": [1, 2, "three"]}


def test_parse_tool_call_no_call_returns_none() -> None:
    assert _parse_tool_call("I can do that for you.") is None
    assert _parse_tool_call("") is None


def test_local_backend_import_error_without_vllm_and_without_engine() -> None:
    """Constructing the backend is fine; generating without vllm raises."""
    backend = LocalVLLMBackend(model_path="fake/model")  # no engine loaded
    with pytest.raises(ImportError, match="vllm"):
        backend.generate_response([{"role": "user", "content": "hi"}])


def test_local_backend_generate_tool_call_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub vllm.LLM + SamplingParams to exercise the full parse path."""
    import sys
    import types

    class FakeOutputs:
        def __init__(self, text: str) -> None:
            self.text = text
            self.token_ids = [0, 1, 2]

    class FakeRequestOutput:
        prompt_token_ids = [1, 2, 3, 4]

        def __init__(self, text: str) -> None:
            self.outputs = [FakeOutputs(text)]

    fake_vllm = types.ModuleType("vllm")

    class FakeLLM:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

        def generate(self, prompt: str, params: Any) -> List[FakeRequestOutput]:
            return [FakeRequestOutput('<|tool_call>call:book_flight{flight_id:<|"|>F001<|"|>}<tool_call|>')]

    class FakeSamplingParams:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    backend = LocalVLLMBackend(model_path="fake/model")  # chat_template_path=None -> raw user msg
    result = backend.generate_tool_call(
        [{"role": "user", "content": "Book F001"}], [make_booking_tools()[1]]
    )
    assert result["tool_name"] == "book_flight"
    assert result["arguments"] == {"flight_id": "F001"}
    assert result["usage"]["completion_tokens"] == 3
    assert result["latency_ms"] >= 0.0


def test_local_backend_generate_response_prose(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    class FakeOutputs:
        text = "Sure, here is the answer."
        token_ids = [0]

    class FakeRequestOutput:
        prompt_token_ids = [1]

        def __init__(self) -> None:
            self.outputs = [FakeOutputs()]

    fake_vllm = types.ModuleType("vllm")

    class FakeLLM:
        def __init__(self, **kw: Any) -> None:
            pass

        def generate(self, prompt: str, params: Any) -> List[FakeRequestOutput]:
            return [FakeRequestOutput()]

    class FakeSamplingParams:
        def __init__(self, **kw: Any) -> None:
            pass

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    backend = LocalVLLMBackend(model_path="fake/model")
    text = backend.generate_response([{"role": "user", "content": "hi"}])
    assert text == "Sure, here is the answer."