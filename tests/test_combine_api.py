"""Tests for the public SpanCall+LLM combine API (spancall.combine)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import spancall
from spancall import CombinePolicy, SpanCallLLM
from spancall.combine import coerce_tools

SCALAR = {
    "type": "function",
    "function": {
        "name": "mortgage",
        "description": "Monthly payment",
        "parameters": {
            "type": "object",
            "properties": {
                "rate": {"type": "number", "description": "APR"},
                "term": {"type": "integer", "description": "Years"},
            },
            "required": ["rate", "term"],
        },
    },
}
COMPLEX = {
    "type": "function",
    "function": {
        "name": "filter_items",
        "description": "Filter",
        "parameters": {
            "type": "object",
            "properties": {
                "cat": {"type": "string", "description": "Category"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"},
            },
            "required": ["cat", "tags"],
        },
    },
}


class FakeResolver:
    def __init__(self, **kw):
        self.kw = kw
        self.calls = 0

    def try_resolve_candidates(self, schemas, utterance, recent_context=None, now=None, **_):
        self.calls += 1
        return SimpleNamespace(**{"calls": [], "latency_ms": 5.0, "abstain_reason": None, **self.kw})


class FakeClient:
    """OpenAI-SDK-shaped client; records kwargs."""

    def __init__(self, tool_args=None, text=""):
        self.seen = []
        self._tool_args, self._text = tool_args, text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.seen.append(kw)
        tcs = []
        if self._tool_args is not None:
            name, args = self._tool_args
            tcs = [SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args)))]
        msg = SimpleNamespace(content=self._text, tool_calls=tcs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
        )


HIT = dict(hit=True, status="hit", tool_name="mortgage", payload={"rate": 3.5, "term": 30})
HIT_COMPLEX = dict(hit=True, status="hit", tool_name="filter_items", payload={"cat": "x", "tags": ["a"]})
ABSTAIN = dict(
    hit=False, status="abstain", tool_name="mortgage", payload=None,
    payload_partial={"rate": 3.5}, missing_required=["term"], abstain_reason="low_margin",
)


def _caller(res, client, policy):
    return SpanCallLLM(FakeResolver(**res), client, "m", policy)


def test_lazy_public_exports():
    assert spancall.SpanCallLLM is SpanCallLLM
    assert spancall.CombinePolicy is CombinePolicy
    with pytest.raises(AttributeError):
        spancall.nope  # noqa: B018


def test_coerce_openai_dict_types_and_required():
    t = coerce_tools([SCALAR, COMPLEX])
    assert [p.param_type for p in t[0].parameters] == ["number", "integer"]
    assert t[1].parameters[1].param_type == "array" and t[1].parameters[1].items_type == "string"
    assert all(p.required for p in t[0].parameters)


def test_router_hit_makes_no_llm_call():
    client = FakeClient()
    out = _caller(HIT, client, "router")("q", [SCALAR])
    assert out.hit and out.source == "spancall" and not out.llm_called
    assert out.arguments == {"rate": 3.5, "term": 30} and client.seen == []


def test_router_abstain_goes_to_full_llm_with_tools():
    client = FakeClient(("mortgage", {"rate": 3.5, "term": 30}))
    out = _caller(ABSTAIN, client, "router")("q", [SCALAR, COMPLEX])
    assert out.source == "llm" and out.llm_called and out.arguments["term"] == 30
    assert client.seen[0]["tools"] and out.prompt_tokens == 10
    assert out.latency_ms >= out.spancall_ms == 5.0


def test_router_schema_sends_complex_hit_to_llm_but_plain_router_does_not():
    c1 = FakeClient(("filter_items", {"cat": "x", "tags": ["a", "b"]}))
    out = _caller(HIT_COMPLEX, c1, "router_schema")("q", [COMPLEX])
    assert out.source == "llm" and out.arguments["tags"] == ["a", "b"]
    c2 = FakeClient()
    out2 = _caller(HIT_COMPLEX, c2, "router")("q", [COMPLEX])
    assert out2.source == "spancall" and c2.seen == []


def test_router_schema_keeps_scalar_hit():
    client = FakeClient()
    out = _caller(HIT, client, "router_schema")("q", [SCALAR, COMPLEX])
    assert out.source == "spancall" and client.seen == []


def test_hybrid_fill_uses_plain_json_no_tools_and_keeps_grounded_scalar():
    client = FakeClient(text=json.dumps({"term": 30}))
    out = _caller(ABSTAIN, client, "hybrid_fill")("q", [SCALAR])
    kw = client.seen[0]
    assert "tools" not in kw and kw["response_format"] == {"type": "json_object"}
    assert out.hit and out.source == "spancall+llm_fill"
    assert out.arguments == {"rate": 3.5, "term": 30}


def test_hybrid_fill_never_overwrites_grounded_value():
    client = FakeClient(text=json.dumps({"rate": 99, "term": 30}))
    out = _caller(ABSTAIN, client, "hybrid_fill")("q", [SCALAR])
    assert out.arguments["rate"] == 3.5


def test_spancall_only_and_llm_only_and_per_call_override():
    client = FakeClient(("mortgage", {"rate": 1, "term": 2}))
    c = _caller(ABSTAIN, client, "spancall_only")
    out = c("q", [SCALAR])
    assert not out.hit and out.source == "none" and client.seen == []
    assert c("q", [SCALAR], policy="llm_only").source == "llm"
    assert SpanCallLLM(None, client, "m", "llm_only")("q", [SCALAR]).hit


def test_missing_llm_or_model_fails_loudly():
    with pytest.raises(ValueError, match="LLM"):
        SpanCallLLM(FakeResolver(**ABSTAIN), None, "m", "router")("q", [SCALAR])
    with pytest.raises(ValueError, match="model"):
        SpanCallLLM(FakeResolver(**ABSTAIN), FakeClient(), None, "router")("q", [SCALAR])
