"""Tests for config-driven attention kernel (attn_implementation) and torch_dtype resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from spancall.model.encoder import (
    resolve_attn_implementation,
    resolve_torch_dtype,
)
from spancall.train.config import ModelConfig, TrainingConfig


def test_resolve_attn_auto_falls_back_to_sdpa_without_cuda_flash_attn() -> None:
    with patch("spancall.model.encoder._cuda_supports_flash_attn", return_value=False):
        assert resolve_attn_implementation("auto") == "sdpa"
        assert resolve_attn_implementation("auto", device="cpu") == "sdpa"
        assert resolve_attn_implementation("auto", device="mps") == "sdpa"


def test_resolve_attn_auto_selects_flash_attention_2_on_ampere_cuda() -> None:
    with patch("spancall.model.encoder._cuda_supports_flash_attn", return_value=True):
        assert resolve_attn_implementation("auto", device="cuda") == "flash_attention_2"


def test_resolve_attn_explicit_and_invalid() -> None:
    assert resolve_attn_implementation("flash_attention_2") == "flash_attention_2"
    assert resolve_attn_implementation("sdpa") == "sdpa"
    assert resolve_attn_implementation("eager") == "eager"
    assert resolve_attn_implementation(None) is None
    with pytest.raises(ValueError, match="Unsupported attn_implementation"):
        resolve_attn_implementation("unknown_kernel")


def test_resolve_torch_dtype_mappings() -> None:
    assert resolve_torch_dtype("bf16") is torch.bfloat16
    assert resolve_torch_dtype("bfloat16") is torch.bfloat16
    assert resolve_torch_dtype("fp16") is torch.float16
    assert resolve_torch_dtype("fp32") is torch.float32
    assert resolve_torch_dtype(torch.bfloat16) is torch.bfloat16
    assert resolve_torch_dtype(None, attn_implementation="sdpa") is None
    # FlashAttention-2 requires fp16/bf16 even when torch_dtype is omitted
    assert resolve_torch_dtype(None, attn_implementation="flash_attention_2") in (
        torch.bfloat16,
        torch.float16,
    )
    with pytest.raises(ValueError, match="Unsupported torch_dtype"):
        resolve_torch_dtype("int8")


def test_model_config_and_default_yaml_load_attn_fields() -> None:
    cfg = TrainingConfig.from_yaml(Path("configs/default.yaml"))
    assert cfg.model.attn_implementation == "auto"
    assert cfg.model.torch_dtype is None

    custom = ModelConfig.from_dict(
        {"attn_implementation": "flash_attention_2", "torch_dtype": "bf16"}
    )
    assert custom.attn_implementation == "flash_attention_2"
    assert custom.torch_dtype == "bf16"
