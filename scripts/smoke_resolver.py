#!/usr/bin/env python3
"""Smoke test for the one-pass SpanCallResolver (Phase 3c).

Builds a tiny random encoder + heads (no HF download), then runs
``try_resolve`` on a weather tool call and checks the resolve chain
populates a hit / payload / parameters / margins / latency telemetry.

Run:  .venv/bin/python scripts/smoke_resolver.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from spancall.data.schema import ParameterSchema, ToolDefinition  # noqa: E402
from spancall.infer.resolver import SpanCallResolver  # noqa: E402
from spancall.model.encoder import SpanCallTransformerEncoder  # noqa: E402
from spancall.model.model import SpanCallModel  # noqa: E402

D = 32  # hidden dim


def make_model() -> SpanCallModel:
    """Tiny encoder + all heads (pointer/abstain/accept/tool/correctness)."""
    from transformers import BertConfig, BertModel  # ModernBERT-compatible path

    cfg = BertConfig(
        vocab_size=999,
        hidden_size=D,
        intermediate_size=D * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=4096,
        pad_token_id=0,
    )
    backbone = BertModel(cfg)
    enc = SpanCallTransformerEncoder(
        model_name_or_path="tiny-bert",
        tokenizer=None,
        backbone=backbone,
    )
    # No add_marker_tokens: the FakeTokenizer emits id 0 (in-vocab) and the
    # smoke path only needs hidden states, not embedding alignment.
    model = SpanCallModel(encoder=enc, head_dim=16, max_span_width=16)
    model.eval()
    return model


def weather_tool() -> ToolDefinition:
    return ToolDefinition(
        name="get_weather",
        description="Get current weather for a city",
        parameters=[
            ParameterSchema(name="city", description="City name", param_type="string", required=True),
            ParameterSchema(name="unit", description="Temperature unit", param_type="string",
                            required=True, enum=["celsius", "fahrenheit"]),
        ],
    )


def main() -> int:
    model = make_model()
    resolver = SpanCallResolver(
        model=model,
        encoder=model.backbone,
        tokenizer=None,  # FakeTokenizer
        temperature=1.0,
        # max_seq_len defaults to the backbone window
        # (config.max_position_embeddings → model_max_length → 32768)
        # so the smoke path exercises the same overflow/abstain logic as prod.
    )
    schema = weather_tool()
    res = resolver.try_resolve(schema, "please book a flight from Hanoi to Saigon tomorrow")

    print(f"hit            = {res.hit}")
    print(f"tool           = {res.tool_name}")
    print(f"abstain_reason = {res.abstain_reason}")
    print(f"payload        = {res.payload}")
    print(f"joint_margin   = {res.joint_margin:.4f}")
    print(f"accept_score   = {res.accept_score:.4f}")
    print(f"latency_ms     = {res.latency_ms:.4f}")
    print(f"features       = { {k: round(v, 4) if isinstance(v, float) else v for k, v in res.features.items()} }")

    for name, pr in res.parameters.items():
        print(f"  param {name}: value={pr.value!r} conf={pr.confidence:.4f} "
              f"margin={pr.margin:.4f} source={pr.source_type} span={pr.char_span}")

    assert res.tool_name == "get_weather"
    assert set(res.features) >= {"pre_ms", "mask_ms", "enc_ms", "heads_ms", "decode_ms", "post_ms"}
    # random init => unlikely to clear tau_p=0.90; the important thing is the
    # pipeline runs end-to-end and returns a well-formed result.
    assert isinstance(res.hit, bool)
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())