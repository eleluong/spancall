"""Load a released SpanCall checkpoint (local directory or Hugging Face repo).

The checkpoint is a plain ``state_dict`` (``model.pt``) plus
``resolve_config.json``; weights are read with ``weights_only=True`` because a
Hub file crosses a trust boundary.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional


def _resolve_file(path_or_repo: str, name: str, token: Optional[str]) -> str:
    if os.path.isdir(path_or_repo):
        p = os.path.join(path_or_repo, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{name} not found in {path_or_repo}")
        return p
    from huggingface_hub import hf_hub_download

    return hf_hub_download(path_or_repo, name, token=token)


def load_resolver(
    path_or_repo: str,
    *,
    device: Optional[str] = None,
    token: Optional[str] = None,
    attn_implementation: Optional[str] = "auto",
    torch_dtype: Optional[str] = None,
    **resolver_kwargs: Any,
) -> Any:
    """Return a ready :class:`~spancall.lean.resolver.LeanResolver` (``two_step``).

    ``attn_implementation`` controls the attention kernel (``"auto"`` uses
    ``flash_attention_2`` on Ampere+ CUDA when ``flash-attn`` is installed and
    falls back to PyTorch ``sdpa``; override with ``"flash_attention_2"``,
    ``"sdpa"``, or ``"eager"``). ``torch_dtype`` accepts ``"auto"``, ``"bf16"``,
    ``"fp16"``, ``"fp32"``, or ``None``.
    ``resolver_kwargs`` override values from ``resolve_config.json``.
    """
    import torch
    from transformers import AutoTokenizer

    from span_template.sanitize import SPECIAL_MARKER_TOKENS
    from spancall.lean.model import LeanSpanCallModel
    from spancall.lean.resolver import LeanResolver
    from spancall.model.encoder import (
        SpanCallTransformerEncoder,
        resolve_attn_implementation,
        resolve_torch_dtype,
    )

    with open(_resolve_file(path_or_repo, "resolve_config.json", token), encoding="utf-8") as fh:
        cfg = json.load(fh)
    backbone = cfg.get("backbone_name") or cfg.get("backbone")
    if not backbone:
        raise KeyError("resolve_config.json has neither 'backbone_name' nor 'backbone'")

    eff_attn = resolve_attn_implementation(
        attn_implementation if attn_implementation is not None else cfg.get("attn_implementation", "auto"),
        device=device,
    )
    eff_dtype = resolve_torch_dtype(
        torch_dtype if torch_dtype is not None else cfg.get("torch_dtype"),
        attn_implementation=eff_attn,
        device=device,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(path_or_repo, token=token)
    except Exception:  # noqa: BLE001 - checkpoint shipped without tokenizer files
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    encoder = SpanCallTransformerEncoder(
        model_name_or_path=backbone,
        tokenizer=tokenizer,
        attn_implementation=eff_attn,
        torch_dtype=eff_dtype,
    )
    encoder.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))  # idempotent; syncs embedding size
    model = LeanSpanCallModel(
        encoder=encoder,
        head_dim=int(cfg["head_dim"]),
        max_span_width=int(cfg.get("max_span_width", 16)),
        hidden_dim=encoder.hidden_dim,
    )
    state = torch.load(
        _resolve_file(path_or_repo, "model.pt", token), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    if device:
        model.to(device)
    # Keep prediction heads (tool_head, pointer_head) in fp32 to preserve the
    # tight min_joint_margin=0.02 gate calibration; only cast the encoder
    # backbone when an explicit low-precision dtype is required (e.g. FA2).
    # LeanResolver automatically casts H to head_p.dtype before scoring heads.
    if eff_dtype is not None:
        model.backbone.to(dtype=eff_dtype)

    opts = {
        "max_seq_len": cfg.get("max_seq_len"),
        "max_span_width": int(cfg.get("max_span_width", 16)),
        "max_cands": int(cfg.get("max_cands", 1024)),
        "min_joint_margin": cfg.get("min_joint_margin"),
        "resolve_mode": cfg.get("resolve_mode", "two_step"),
        "collapse_guard": bool(cfg.get("collapse_guard", True)),
        "prevent_param_collapse": bool(cfg.get("prevent_param_collapse", True)),
    }
    for k in ("max_calls", "call_min_prob", "call_min_prob_ratio"):
        if k in cfg:
            opts[k] = cfg[k]
    opts.update(resolver_kwargs)
    return LeanResolver(model=model, encoder=encoder, tokenizer=tokenizer, **opts)
