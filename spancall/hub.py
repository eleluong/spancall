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
    **resolver_kwargs: Any,
) -> Any:
    """Return a ready :class:`~spancall.lean.resolver.LeanResolver` (``two_step``).

    ``resolver_kwargs`` override values from ``resolve_config.json``.
    """
    import torch
    from transformers import AutoTokenizer

    from span_template.sanitize import SPECIAL_MARKER_TOKENS
    from spancall.lean.model import LeanSpanCallModel
    from spancall.lean.resolver import LeanResolver
    from spancall.model.encoder import SpanCallTransformerEncoder

    with open(_resolve_file(path_or_repo, "resolve_config.json", token), encoding="utf-8") as fh:
        cfg = json.load(fh)
    backbone = cfg.get("backbone_name") or cfg.get("backbone")
    if not backbone:
        raise KeyError("resolve_config.json has neither 'backbone_name' nor 'backbone'")

    try:
        tokenizer = AutoTokenizer.from_pretrained(path_or_repo, token=token)
    except Exception:  # noqa: BLE001 - checkpoint shipped without tokenizer files
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    encoder = SpanCallTransformerEncoder(model_name_or_path=backbone, tokenizer=tokenizer)
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
