"""Trainer for the SpanCall multi-task model.

Ref: docs/full_implementation_plan.md §3.4; docs/architecture.md §5.3;
docs/heads.md §4.

``SpanCallTrainer(BaseTrainer)`` implements the staged methodology:

- **Optimizer & schedule (plan §3.4.1):**
  - AdamW with parameter groups: encoder layers get layer-wise LR decay
    ``lr_l = lr_enc * 0.90^(n_layers - l)``; newly initialized marker-token
    embeddings + all prediction heads get ``lr_head``.  Weight decay 0.01,
    excluded from biases and LayerNorm.
  - Linear warmup (``warmup_ratio``) then cosine decay to 0.
  - Mixed precision ``torch.amp.autocast`` (bf16) with
    ``clip_grad_norm_(..., 1.0)`` (falls back to fp32/``autocast`` off when the
    device does not support bf16).
- **Staged curriculum (plan §3.4.2):**
  - ``T0``: sanity — a few hundred Stratum.F rows, verify loss -> ~0.
  - ``T1``: pointer ceiling — H3 pointer on F train, exact-payload match gate
    on F dev (floor 0.80, proceed 0.90).
  - ``T2/Stage A-B``: full multi-task (encoder + H1 + H3 + H4 + CallAccept).
  - ``Stage C``: freeze encoder + H1–H4, train H8 correctness on a held-out
    calib slice.
- **Checkpointing & provenance (plan §3.4.3):**
  ``results/<experiment>_seed<seed>/<stage>/checkpoints/`` storing
  ``model.pt``, ``optimizer.pt``, ``metrics.json``, ``provenance.json``
  (git commit, config snapshot, seed, stage, step).

The trainer is device-agnostic (CPU/MPS/CUDA) and never assumes a real HF
backbone: the encoder may be a tiny injected ``nn.Module`` for tests, or
``SpanCallTransformerEncoder`` over ModernBERT.

Batch format (produced by the data pipeline tensorizer / collator):
    input_ids, attention_mask          (B, L) long
    token_type_ids                     (B, L) long (optional)
    param_positions                    (B, K) long in [0, L)
    tool_positions                     (B, T) long (optional, pack_tools > 1)
    utt_starts, utt_ends               (B, 2) long  (utterance extent)
    cand_starts, cand_ends             (B, K, C) long, -1 padded
    pos_indices                        (B, K, P) long, -1 padded (into z)
    pos_regions                        (B, K, P) long (0=utterance,1=recent,2=time,
                                       3=enum,4=bool,5=null) unused in v1 loss
    abstain_flags                      (B, K) long  (1 → row abstains)
    reason_targets                     (B, K) long  (0..4), -1 masked
    accept_targets                     (B,) float
    tool_targets                       (B,) long (optional)
    cue_mask                           (B, K) float (layers to include)
"""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, is_dataclass
from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:  # tqdm.auto picks the notebook widget in Jupyter/Colab, text bar in a terminal
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kw): self.iterable = iterable
        def __iter__(self): return iter(self.iterable or [])
        def update(self, n=1): pass
        def set_postfix(self, *a, **kw): pass
        def close(self): pass
        @staticmethod
        def write(msg, *a, **kw): print(msg, flush=True)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from spancall.model.encoder import BaseEncoder
from spancall.model.heads import (
    AbstainHead,
    CallAcceptHead,
    CallCountHead,
    CorrectnessHead,
    PointerHead,
    ToolHead,
)
from spancall.model.loss import SpanCallLossContainer
from spancall.train.config import TrainingConfig

_LAYER_DECAY = 0.90


def _git_rev(root: Optional[str] = None) -> str:
    """Best-effort current git commit hash ('' when not a git repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=root, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


class BaseTrainer(ABC):
    """Abstract trainer coordinating forward, backward, validation, and checkpointing."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Execute training loop over configured epochs."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, split: str = "dev") -> Dict[str, float]:
        """Run validation metrics across validation or test splits."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persist model weights, optimizer state, and training metadata."""
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Restore checkpoint state."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Parameter grouping + schedule
# --------------------------------------------------------------------------- #
def build_parameter_groups(
    model: nn.Module,
    encoder: BaseEncoder,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
    layer_decay: float = _LAYER_DECAY,
) -> List[Dict[str, Any]]:
    """AdamW parameter groups with layer-wise decay for the encoder backbone.

    - Encoder backbone layers l (0-based from bottom): ``lr = lr_enc * layer_decay^(n_layers - l)``
      (top layers get the full ``lr_enc``, bottom layers the smallest).
    - Marker-token embedding rows (``num_embeddings > base vocab``) + all heads
      (ToolHead, PointerHead, AbstainHead, CallAcceptHead, CallCountHead,
      CorrectnessHead): ``lr_head``.
    - Weight decay excluded on biases and LayerNorm weights.

    Encoder layers are discovered via ``model.named_parameters()`` where the
    name contains ``"backbone.encoder"``/``"backbone." + '<layer-prefix>'`` —
    the trainer does not hard-code backbone internals; it uses a
    ``layers_of(model)`` probe to map parameter names to layer indices when the
    backbone exposes a ``.layers``/``.encoder.layer`` sequence (HF
    ``PreTrainedModel``), else treats all encoder params as one group.
    """
    # --- discover encoder parameter names ----------------------------------
    enc_param_names: set = set()
    layer_index: Dict[str, int] = {}   # param name -> layer index (deepest = 0)
    n_layers: int = 1

    # find any sequence attribute holding stacked layers
    for attr in (
        "layers",
        "encoder.layer",
        "bert.encoder.layer",
        "backbone.backbone.layers",
        "backbone.backbone.encoder.layer",
    ):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, (list, nn.ModuleList)):
            n_layers = len(obj)
            for l, layer in enumerate(obj):
                for name, _p in layer.named_parameters():
                    full = f"{attr}.{name}"
                    layer_index[full] = l
                    enc_param_names.add(full)
            break

    def is_encoder(name: str) -> bool:
        if name.startswith("backbone."):
            return True
        return enc_param_names and name in enc_param_names

    # --- split parameters --------------------------------------------------
    groups: Dict[str, Dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decay = weight_decay if (param.dim() > 1 and "bias" not in name and "LayerNorm" not in name and "layernorm" not in name) else 0.0
        if is_encoder(name):
            l = layer_index.get(name, n_layers - 1)
            lr = lr_encoder * (layer_decay ** (n_layers - 1 - l))
            key = f"enc_l{l}"
        else:
            lr = lr_head
            key = "head"
        g = groups.setdefault(
            key,
            {"params": [], "lr": lr, "weight_decay": decay, "decay": decay},
        )
        g["params"].append(param)
        g["weight_decay"] = decay

    out: List[Dict[str, Any]] = sorted(
        groups.values(), key=lambda g: -g["lr"]
    )
    for g in out:
        g.pop("decay", None)
    return out


class CosineWarmupSchedule:
    """Linear warmup then cosine decay to 0 (plan §3.4.1)."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> None:
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)

        def lr_lambda(step: int) -> float:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return float(step + 1) / float(self.warmup_steps)
            if step >= self.total_steps:
                return 0.0
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(self) -> None:
        self.scheduler.step()


# --------------------------------------------------------------------------- #
# Exact-payload decode (deterministic argmax path; used by T1 gate + eval)
# --------------------------------------------------------------------------- #
def decode_exact_payload(
    z: torch.Tensor,
    cand_starts: torch.Tensor,
    cand_ends: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-param decode used for the T1 gate and dev metrics.

    Args:
        z:      (B, K, C) candidate logits over {abstain(0)} ∪ A_k, or None?
        cand_starts / cand_ends: (B, K, C) long, -1 padded.
        mask:   (B, K) optional; True rows decoded, False rows -> abstain.

    Returns:
        (pred_starts, pred_ends): (B, K) long — the chosen span (inclusive) or
        (-1, -1) when abstain (column 0) wins or the row is masked.
    """
    B, K, C = z.shape
    if mask is None:
        mask = torch.ones(B, K, dtype=torch.bool, device=z.device)
    am = z.argmax(dim=-1)  # (B, K) in [0, C) where C = 1 + |A_k|
    is_cand = (am > 0) & mask
    # index into the candidate arrays (cand_starts/cand_ends have |A_k| cols,
    # while z prepends the abstain column at 0): candidate v sits at am - 1.
    cand_idx = (am - 1).clamp(min=0)
    s = torch.where(is_cand, cand_starts.gather(-1, cand_idx.unsqueeze(-1)).squeeze(-1),
                    torch.full((B, K), -1, dtype=torch.long, device=z.device))
    e = torch.where(is_cand, cand_ends.gather(-1, cand_idx.unsqueeze(-1)).squeeze(-1),
                    torch.full((B, K), -1, dtype=torch.long, device=z.device))
    return s, e


def _mean_losses(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean (B,K) loss across rows where mask is True."""
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1).clamp(min=0),
        reduction="none",
    ).reshape(logits.shape[0], logits.shape[1])
    ce = torch.where(mask, ce, torch.zeros_like(ce))
    n = max(int(mask.sum().item()), 1)
    return ce.sum() / n


# --------------------------------------------------------------------------- #
# SpanCallTrainer
# --------------------------------------------------------------------------- #
class SpanCallTrainer(BaseTrainer):
    """Staged multi-task trainer for the SpanCall model."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        encoder: BaseEncoder,
        tokenizer: Any,
        loss_container: Optional[SpanCallLossContainer] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self.model = model
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.loss = loss_container or SpanCallLossContainer()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model.to(self.device)

        # ---- VRAM levers (default OFF) -------------------------------------
        if getattr(self.config, "gradient_checkpointing", False):
            self._enable_gradient_checkpointing()
        self.scaler: Optional[Any] = None
        if self._use_fp16():
            # GradientScaler for fp16: bf16 needs no scaling, fp32 none either.
            # torch>=2.0 exposes torch.amp.GradScaler; older builds torch.cuda.amp.
            grad_scaler = getattr(torch.amp, "GradScaler", None)
            if grad_scaler is None:
                grad_scaler = getattr(torch.cuda.amp, "GradScaler", None)
            if grad_scaler is not None:
                self.scaler = grad_scaler(
                    enabled=self.device.startswith("cuda") and torch.cuda.is_available()
                )
                print("[SpanCallTrainer] fp16 mixed precision enabled (GradScaler)")

        self.optimizer = optimizer
        self.scheduler: Optional[CosineWarmupSchedule] = None
        self.global_step = 0
        self._accum_i = 0                      # micro-steps since the last optimizer step
        import random as _random
        self._rng = _random.Random(int(getattr(config, "seed", 0) or 0))   # seeded batch shuffling
        self.stage: str = "T0"
        self.history: Dict[str, Any] = {"train_loss": [], "dev": {}}
        self.stage_gates: Dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # VRAM helpers
    # ------------------------------------------------------------------ #
    def _use_fp16(self) -> bool:
        """True when config requests fp16 mixed precision (needs GradScaler)."""
        return (
            getattr(self.config.optimization, "mixed_precision", "bf16") == "fp16"
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
        )

    def _enable_gradient_checkpointing(self) -> None:
        """Enable HF gradient checkpointing on the backbone (reentrant=False).

        Reentrant checkpointing (``use_reentrant=True``) would break because
        ``train_step`` calls ``self.encoder(...)`` directly with no reentrant
        wrapper, and it must also compose with frozen-backbone stages (no
        backward through the backbone). Tiny injected test backbones without a
        ``gradient_checkpointing_enable`` method are skipped silently.
        """
        backbone = getattr(self.encoder, "backbone", None)
        enable = getattr(backbone, "gradient_checkpointing_enable", None)
        if enable is None:
            print(
                "[SpanCallTrainer] gradient_checkpointing requested but backbone has no "
                "gradient_checkpointing_enable — skipping (tiny/test backbone?)"
            )
            return
        try:
            enable(use_reentrant=False)
        except TypeError:  # very old transformers: no use_reentrant kwarg
            enable()
        cfg = getattr(backbone, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False  # incompatible with checkpointing (decoder UQ warns)

    def _make_optimizer(self, groups: List[Dict[str, Any]], lr: float) -> torch.optim.Optimizer:
        """AdamW, or bitsandbytes PagedAdamW8bit when configured.

        Graceful fallback to AdamW when bitsandbytes is not installed, so the
        package stays optional (no hard dependency added).
        """
        if getattr(self.config, "use_8bit_optimizer", False):
            try:
                from bitsandbytes.optim import PagedAdamW8bit

                self._optimizer_kind = "PagedAdamW8bit (8-bit)"
                print("[SpanCallTrainer] using bitsandbytes PagedAdamW8bit (8-bit optimizer)")
                return PagedAdamW8bit(groups, lr=lr)
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[SpanCallTrainer] bitsandbytes unavailable ({exc}); falling back to AdamW")
        use_fused = (
            self.device.startswith("cuda")
            and torch.cuda.is_available()
            and self.scaler is None
            and all(p.is_cuda for g in groups for p in g.get("params", []))
        )
        if use_fused:
            try:
                self._optimizer_kind = "AdamW (fused)"
                return AdamW(groups, lr=lr, fused=True)
            except Exception:
                pass
        self._optimizer_kind = "AdamW"
        return AdamW(groups, lr=lr)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def exact_payload_match(
        self, z: torch.Tensor, cand_starts: torch.Tensor, cand_ends: torch.Tensor,
        pos_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Proportion of (b,k) rows whose argmax is a positive (R4 equal-value aware).

        Returns (acc, acc_tensor, valid).
        """
        B, K, C = z.shape
        if K == 0 or C == 0:
            empty = torch.zeros((B, K), dtype=torch.float32, device=z.device)
            valid = torch.zeros((B, K), dtype=torch.bool, device=z.device)
            return torch.tensor(0.0, device=z.device), empty, valid
        am = z.argmax(-1).unsqueeze(-1)                    # (B,K,1)
        in_range = (pos_indices >= 0) & (pos_indices < C)
        is_pos = ((pos_indices == am) & in_range).any(-1)  # (B,K)
        n_pos = in_range.sum(-1)
        valid = n_pos > 0
        acc = torch.where(valid, is_pos.float(), torch.zeros_like(is_pos.float()))
        return acc.sum() / max(int(valid.sum().item()), 1), acc, valid

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One optimization step; returns scalar losses for logging."""
        self._ensure_optimizer()
        self.model.train()
        b = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        preds: Dict[str, torch.Tensor] = {}
        targs: Dict[str, torch.Tensor] = {}

        # ---- forward under 16-bit autocast (bf16/fp16 on CUDA) ------------
        with self._amp():
            H = self.encoder(b["input_ids"], b["attention_mask"], b.get("token_type_ids"))

            # ---- CLS / param / tool / utterance anchors -------------------
            cls_h = H[:, 0]
            max_tok_idx = max(H.size(1) - 1, 0)
            safe_param_pos = b["param_positions"].clamp(min=0, max=max_tok_idx)
            param_h = H.gather(1, safe_param_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
            K = param_h.size(1)

            if "tool_positions" in b and b["tool_positions"].numel():
                safe_tool_pos = b["tool_positions"].clamp(min=0, max=max_tok_idx)
                tool_h = H.gather(1, safe_tool_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
            else:
                tool_h = None

            # Vectorized utterance pool for H4 (no .item() CPU-GPU syncs; stays in H.dtype)
            L_sz = H.size(1)
            tok_idx = torch.arange(L_sz, device=H.device).unsqueeze(0)          # (1, L)
            s0 = b["utt_starts"][:, :1]                                         # (B, 1)
            e0 = b["utt_ends"][:, :1]                                           # (B, 1)
            valid_utt = (s0 >= 0) & (s0 <= e0) & (e0 < L_sz)                    # (B, 1)
            in_utt = valid_utt & (tok_idx >= s0) & (tok_idx <= e0)              # (B, L)
            utt_counts = in_utt.sum(dim=1, keepdim=True).clamp(min=1).to(H.dtype)
            utt_pool = (H * in_utt.unsqueeze(-1).to(H.dtype)).sum(dim=1) / utt_counts

            # ---- H3 pointer -----------------------------------------------
            ptr = getattr(self.model, "pointer_head", None)
            if ptr is not None and "cand_starts" in b and K > 0:
                p = ptr(
                    param_h, H,
                    candidate_masks={"starts": b["cand_starts"], "ends": b["cand_ends"]},
                )
                starts = b["cand_starts"].clamp(min=0)
                ends = b["cand_ends"].clamp(min=0)
                z = torch.cat([p["abstain_logits"].unsqueeze(-1),
                               p["span_logits"]], dim=-1)
                preds["pointer_logits"] = z
                preds["abstain_logits"] = p["abstain_logits"]
                preds["q"] = p["q"]

            # ---- H4 reason ------------------------------------------------
            abst = getattr(self.model, "abstain_head", None)
            if abst is not None and K > 0:
                r = abst(param_h, cls_h, utt_pool)
                preds["reason_logits"] = r
                targs["reason_logits"] = b["reason_targets"] if "reason_targets" in b else None
                if targs["reason_logits"] is None:
                    targs["reason_logits"] = torch.zeros_like(b["param_positions"]).fill_(-1).to(self.device)
                if "reason_mask" in b:
                    targs["reason_mask"] = b["reason_mask"].float()

            # ---- CallAccept -----------------------------------------------
            acc_head = getattr(self.model, "call_accept_head", None)
            if acc_head is not None:
                preds["accept_logits"] = acc_head(cls_h)
                targs["accept_targets"] = b["accept_targets"] if "accept_targets" in b else None
                if targs["accept_targets"] is None:
                    targs["accept_targets"] = torch.zeros(H.size(0), device=self.device)

            # ---- ToolHead (H1) — only when pack_tools > 1 -----------------
            if tool_h is not None:
                tool_head = getattr(self.model, "tool_head", None)
                if tool_head is not None:
                    tl = tool_head(tool_h, cls_h)
                    if "tool_positions" in b:
                        tl = tl.masked_fill(b["tool_positions"] < 0, -1e4)
                    preds["tool_logits"] = tl
                    targs["tool_targets"] = b.get("tool_targets", torch.zeros(H.size(0), dtype=torch.long, device=self.device))

            # ---- H8 correctness -------------------------------------------
            corr_head = getattr(self.model, "correctness_head", None)
            if corr_head is not None and "q" in preds and "pointer_logits" in preds and K > 0:
                # chosen span pooled features (vectorized on GPU in H.dtype without .item() syncs)
                z_det = preds["pointer_logits"].detach()
                am = z_det.argmax(-1)                       # (B,K) over {abstain} U A_k
                C_cand = b["cand_starts"].size(-1) if "cand_starts" in b and b["cand_starts"].dim() == 3 else 0
                is_cand = (am > 0) & (C_cand > 0)
                if C_cand > 0:
                    cand_am = (am - 1).clamp(min=0, max=C_cand - 1)  # index into cand arrays
                    chosen_s = b["cand_starts"].gather(-1, cand_am.unsqueeze(-1)).squeeze(-1).clamp(min=0, max=max_tok_idx)
                    chosen_e = b["cand_ends"].gather(-1, cand_am.unsqueeze(-1)).squeeze(-1).clamp(min=0, max=max_tok_idx)
                    tok_idx_bk = tok_idx.unsqueeze(1)                                       # (1, 1, L)
                    s_bk = chosen_s.unsqueeze(-1)                                           # (B, K, 1)
                    e_bk = chosen_e.unsqueeze(-1)                                           # (B, K, 1)
                    valid_span = is_cand.unsqueeze(-1) & (s_bk >= 0) & (s_bk <= e_bk) & (e_bk < L_sz)
                    in_span = valid_span & (tok_idx_bk >= s_bk) & (tok_idx_bk <= e_bk)      # (B, K, L)
                    span_counts = in_span.sum(dim=-1, keepdim=True).clamp(min=1).to(H.dtype)
                    span_pool = torch.matmul(in_span.to(H.dtype), H.detach()) / span_counts # (B, K, D)
                    width_raw = (chosen_e - chosen_s).clamp(min=0)
                    width_norm = torch.where(
                        is_cand,
                        width_raw.to(H.dtype) / max(self.model.max_span_width, 1),
                        torch.zeros_like(width_raw, dtype=H.dtype),
                    )
                else:
                    span_pool = torch.zeros_like(cls_h.unsqueeze(1).expand(-1, K, -1))
                    width_norm = torch.zeros(H.size(0), K, device=self.device, dtype=H.dtype)
                # v_stats: p_top1, p_top2, margin, entropy, p_abstain, p_ok, width_norm, is_ctx(=0 stub)
                probs = torch.softmax(z_det, -1)
                p_top1 = probs.max(-1).values
                p_abstain = probs[..., 0]
                sorted_p, _ = probs.sort(-1, descending=True)
                p_top2 = sorted_p[..., 1] if sorted_p.size(-1) > 1 else torch.zeros_like(p_top1)
                margin = (sorted_p[..., 0] - sorted_p[..., 1]) if sorted_p.size(-1) > 1 else torch.zeros_like(p_top1)
                # p_ok comes from the PREDICTED accept head (detached), never from the label:
                # feeding accept_targets here leaked the answer into H8 during Stage C.
                if "accept_logits" in preds:
                    p_ok = torch.sigmoid(preds["accept_logits"].detach().to(probs.dtype).reshape(-1, 1)).expand(-1, K)
                else:
                    p_ok = torch.zeros(H.size(0), K, device=self.device, dtype=probs.dtype)
                is_ctx = torch.zeros_like(p_top1)
                v_stats = torch.stack([p_top1, p_top2, margin,
                                       -(probs * probs.clamp_min(1e-9).log()).sum(-1),
                                       p_abstain, p_ok, width_norm.to(probs.dtype), is_ctx], dim=-1)
                preds["corr_logits"] = corr_head(span_pool, preds["q"], v_stats)

            # ---- targets for pointer --------------------------------------
            targs["pointer_positives"] = b["pos_indices"] if "pos_indices" in b else None
            if targs["pointer_positives"] is None:
                targs["pointer_positives"] = torch.full(
                    (H.size(0), K, 1), -1, dtype=torch.long, device=self.device)
            if "cue_mask" in b:
                targs["pointer_mask"] = b["cue_mask"].float()

            losses = self.loss.forward(preds, targs)
            total = losses["loss"]

        # ---- backward (scaled for fp16; bf16/fp32 scale by 1 implicitly) ---
        loss = total / self.config.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._accum_i += 1
        if self._accum_i % max(self.config.gradient_accumulation_steps, 1) == 0:
            self._optimizer_step()
        self.global_step += 1

        out = {k: float(v.detach().item()) for k, v in losses.items() if k != "loss"}
        out["loss"] = float(total.detach().item())  # scalar joint loss for run_stage
        return out

    def _optimizer_step(self) -> None:
        """Clip, step, zero grads and advance the LR schedule (one *optimizer* step)."""
        if self.scaler is not None:
            # unscale so clip_norm sees true gradients; skip the step when
            # inf/nan gradients were found (scaler.update() grows the scale
            # back next round).
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self._accum_i = 0

    def _flush_accumulation(self) -> None:
        """Apply gradients left over from an incomplete accumulation window (end of stage)."""
        if self._accum_i > 0 and self.optimizer is not None:
            self._optimizer_step()

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def make_batches(
        self, rows: Sequence[Dict[str, Any]], batch_size: int, shuffle: bool = False,
        rng: Optional[Any] = None, max_tokens_per_batch: Optional[int] = None,
    ) -> Iterable[List[Dict[str, Any]]]:
        """Yield length-bucketed, token-capped batches to minimize padding & prevent OOM spikes.

        Grouping sequences of similar ``input_ids`` length into each batch cuts
        ``O(B * L_max^2)`` padding overhead by 2x-3x while shuffling both tie-breaks
        within megabatches and the batch order across the epoch. Capping total padded
        tokens per micro-batch (default ``batch_size * 1024``) prevents tail buckets
        of 2K-4K token sequences from packing ``batch_size * 4096`` tokens at once.
        """
        n = len(rows)
        if n == 0:
            return
        bs = max(int(batch_size), 1)
        cfg_cap = getattr(self.config, "max_tokens_per_batch", None)
        token_cap = int(max_tokens_per_batch or cfg_cap or (bs * 1024))
        idx = list(range(n))

        def _seq_len(j: int) -> int:
            ids = rows[j].get("input_ids") if isinstance(rows[j], dict) else None
            return int(ids.numel()) if isinstance(ids, torch.Tensor) else (len(ids) if ids is not None else 0)

        def _pack_sorted(sorted_indices: Sequence[int]) -> List[List[int]]:
            packed: List[List[int]] = []
            cur: List[int] = []
            cur_max = 0
            for j in sorted_indices:
                l = _seq_len(j)
                new_max = max(cur_max, l)
                if cur and (len(cur) >= bs or (len(cur) + 1) * new_max > token_cap):
                    packed.append(cur)
                    cur = [j]
                    cur_max = l
                else:
                    cur.append(j)
                    cur_max = new_max
            if cur:
                packed.append(cur)
            return packed

        if shuffle:
            r = rng or random_mod()
            r.shuffle(idx)  # randomize tie-breaking for equal-length rows
            pool_sz = bs * 32
            batches: List[List[int]] = []
            for start in range(0, n, pool_sz):
                megabatch = sorted(idx[start : start + pool_sz], key=_seq_len)
                batches.extend(_pack_sorted(megabatch))
            r.shuffle(batches)
            for b_idx in batches:
                yield [rows[j] for j in b_idx]
        else:
            idx.sort(key=_seq_len)
            for b_idx in _pack_sorted(idx):
                yield [rows[j] for j in b_idx]

    def collate(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad a list of per-sample tensor dicts into a batch dict."""
        keys = [k for k in batch[0].keys()]
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            vals = [b[k] for b in batch]
            if vals[0].dim() == 0:
                out[k] = torch.stack(vals)
                continue
            if vals[0].dim() == 1:
                # input_ids must pad with a valid vocab id (-1 indexes the embedding
                # out of range -> CUDA device-side assert); attention_mask hides it.
                pad_val = 0 if k in ("attention_mask", "cue_mask", "input_ids", "token_type_ids") else -1
                out[k] = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=pad_val)
            elif vals[0].dim() == 2:
                # 2D tensors (e.g. cand_starts/ends (K, C), pos_indices (K, P)):
                # Pad both dimensions (dim 0 = K, dim 1 = C/P) to batch max
                max_0 = max(v.size(0) for v in vals)
                max_1 = max(v.size(1) for v in vals)
                out[k] = torch.stack([
                    torch.nn.functional.pad(
                        v, (0, max_1 - v.size(1), 0, max_0 - v.size(0)), value=-1
                    )
                    for v in vals
                ])
            else:
                out[k] = torch.stack(vals)
        return out

    def load_rows(self, split: str) -> List[Dict[str, Any]]:
        """Load a tensorized split file from ``config.experiment.data_dir``.

        Falls back to a no-op empty list when the file is absent so tests /
        dry runs can inject synthetic rows directly.

        Honors ``config.data.fraction`` (default 1.0): when < 1.0 the rows are
        subsampled **deterministically** with ``random.Random(config.seed)``
        (stable across reruns, no duplicate rows). Because both ``_stage_data``
        and ``evaluate`` load through here, the fraction scales the training
        split *and* the dev/calib eval splits together — the intended "verify
        the whole pipeline on N% of the data" knob (``configs/verify20.yaml``).
        """
        data_dir = getattr(self.config, "data_dir", None) or "data/processed"
        path = Path(data_dir) / f"{split}.jsonl"
        if not path.exists():
            kw = getattr(self.config, "data_dir", None)
            return []
        import json as _json
        rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = _json.loads(line)
                rows.append(d)

        frac = float(getattr(getattr(self.config, "data", None), "fraction", 1.0) or 1.0)
        if frac < 1.0:
            import random

            rng = random.Random(int(getattr(self.config, "seed", 42)))
            n = max(int(len(rows) * frac), 1)
            rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), min(n, len(rows))))]
        return rows

    def tensorize_rows(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert dict rows (text form) into tensor rows (collator input).

        The full production tensorizer (P7, tokenization + candidate gather)
        lives one level deeper and is wired by the data pipeline; this shim
        accepts pre-tensorized rows (all values already torch.Tensor) and is a
        no-op passthrough for them, ready to be extended.
        """
        out: List[Dict[str, Any]] = []
        for r in rows:
            d: Dict[str, torch.Tensor] = {}
            for k, v in r.items():
                t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
                if k in ("cand_starts", "cand_ends", "pos_indices") and t.dim() == 1:
                    t = t.reshape(0, 0).long()
                d[k] = t
            out.append(d)
        return out

    # ------------------------------------------------------------------ #
    # Staged curriculum
    # ------------------------------------------------------------------ #
    def _release_cuda_cache(self) -> None:
        """Free unreferenced objects and return cached CUDA allocator blocks between stages."""
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_stage(self, stage: str, data: List[Dict[str, Any]], num_epochs: int) -> Dict[str, float]:
        """Run one training stage over ``data`` (tensorized rows)."""
        self.stage = stage
        rows = self.tensorize_rows(data)
        if not rows:
            return {"loss": float("nan")}
        steps = 0
        losses: Dict[str, float] = {}
        bs = self.config.optimization.batch_size
        steps_per_epoch = max(len(list(self.make_batches(rows, bs, shuffle=False))), 1)
        total_steps = steps_per_epoch * num_epochs
        log_every = max(int(getattr(self.config, "log_every", 0) or 0), 0) or max(total_steps // 100, 1)
        t_start = time.time()
        self._accum_i = 0
        print(f"[train:{stage}] start: rows={len(rows)} epochs={num_epochs} "
              f"steps/epoch={steps_per_epoch} total_steps={total_steps} bs={bs}", flush=True)
        self._log_event({"event": "stage_start", "stage": stage, "rows": len(rows),
                         "epochs": num_epochs, "total_steps": total_steps})
        run_sum: Dict[str, float] = {}
        run_n = 0
        pbar = tqdm(total=total_steps, desc=f"train[{stage}]", dynamic_ncols=True, leave=True)
        for epoch in range(1, num_epochs + 1):
            ep_sum, ep_n = 0.0, 0
            for batch_list in self.make_batches(rows, bs, shuffle=True, rng=self._rng):
                if len(batch_list) == 0:
                    continue
                batch = self.collate(batch_list)
                losses = self.train_step(batch)
                steps += 1
                self.history["train_loss"].append({**losses, "stage": stage, "step": self.global_step})
                for k, v in losses.items():
                    run_sum[k] = run_sum.get(k, 0.0) + v
                run_n += 1
                if math.isfinite(losses.get("loss", float("nan"))):
                    ep_sum += losses["loss"]
                    ep_n += 1
                pbar.update(1)
                pbar.set_postfix(ep=f"{epoch}/{num_epochs}", loss=f"{losses.get('loss', float('nan')):.4f}",
                                 lr=f"{self.optimizer.param_groups[-1]['lr']:.1e}" if self.optimizer is not None else "n/a")
                if steps % log_every == 0 or steps == total_steps:
                    avg = {k: v / run_n for k, v in run_sum.items()}
                    elapsed = time.time() - t_start
                    lr = self.optimizer.param_groups[-1]["lr"] if self.optimizer is not None else float("nan")
                    self._log_event({"event": "train", "stage": stage, "epoch": epoch, "step": self.global_step,
                                     "stage_step": steps, "lr": lr, "elapsed_s": elapsed, **avg})
                    run_sum, run_n = {}, 0
            tqdm.write(f"[train:{stage}] epoch {epoch}/{num_epochs} done: "
                       f"mean_loss={ep_sum / max(ep_n, 1):.4f}")
            self._log_event({"event": "epoch_end", "stage": stage, "epoch": epoch,
                             "mean_loss": ep_sum / max(ep_n, 1)})
        pbar.close()
        self._flush_accumulation()
        self._release_cuda_cache()
        # "loss" = mean over the FINAL epoch (the last micro-batch alone is far too noisy to report)
        return ({"loss": ep_sum / max(ep_n, 1), "steps": float(steps)}
                if steps else {"loss": float("nan"), "steps": 0.0})

    def _log_event(self, rec: Dict[str, Any]) -> None:
        """Append one JSON line to ``<output_dir>/<exp>_seed<seed>/train_log.jsonl`` (never raises)."""
        try:
            d = Path(self.config.output_dir) / f"{self.config.experiment_name}_seed{self.config.seed}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "train_log.jsonl", "a") as fh:
                fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **rec}, default=str) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Mixed precision + stage helpers
    # ------------------------------------------------------------------ #
    def _amp(self) -> Any:
        """Autocast on CUDA: bf16 (default) or fp16; fp32 no-op fallback."""
        mp = getattr(self.config.optimization, "mixed_precision", "bf16")
        if self.device.startswith("cuda") and torch.cuda.is_available():
            if mp == "fp16":
                return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if mp == "bf16":
                return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _rebuild_optimizer(self, lr_scale: float = 1.0) -> None:
        """(Re)build the optimizer over the *currently unfrozen* parameters."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer = None
            self.scheduler = None
            self._release_cuda_cache()
        groups = build_parameter_groups(
            self.model, self.encoder,
            self.config.optimization.learning_rate_encoder * lr_scale,
            self.config.optimization.learning_rate_head * lr_scale,
            self.config.optimization.weight_decay,
        )
        self.optimizer = self._make_optimizer(
            groups, self.config.optimization.learning_rate_head * lr_scale,
        )
        self.optimizer.zero_grad(set_to_none=True)
        total_steps = max(int(getattr(self, "_sched_total_steps", 0) or 0), 1) if getattr(
            self, "_sched_total_steps", 0) else max(
            int(self.config.optimization.num_epochs * 1_000
                / max(self.config.optimization.batch_size, 1)), 1)
        warmup = max(int(total_steps * self.config.optimization.warmup_ratio), 1)
        self.scheduler = CosineWarmupSchedule(self.optimizer, warmup, total_steps)
        return groups

    def _ensure_optimizer(self) -> None:
        """Build the optimizer on first use (safe for direct run_stage calls)."""
        if self.optimizer is None:
            self._rebuild_optimizer()

    def _freeze_all_but_correctness(self) -> None:
        """Stage C: freeze encoder + H1–H4, keep H8 trainable."""
        for name, p in self.model.named_parameters():
            p.requires_grad = name.startswith("correctness_head")

    def _freeze_encoder(self) -> None:
        """Freeze only the backbone (``backbone.*``); all heads stay trainable.

        Used for VRAM-friendly head-only stages (T0/T1 sanity + pointer when
        ``config.freeze_encoder_stages`` lists them): no backbone grads are
        produced, and AdamW is rebuilt over heads only — dropping the
        encoder's ~3x optimizer-state footprint for those stages.
        """
        for name, p in self.model.named_parameters():
            p.requires_grad = not name.startswith("backbone.")

    def _unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def _stage_data(self, stage: str) -> List[Dict[str, Any]]:
        """Rows for each curriculum stage (empty list ⇒ skip the stage)."""
        if stage == "T0":
            return self.load_rows("t0") or self.load_rows("train")[:500]
        if stage == "T1":
            return self.load_rows("t1") or self.load_rows("train")
        if stage in ("T2", "AB"):
            return self.load_rows("t2") or self.load_rows("train")
        if stage == "C":
            return self.load_rows("stage_c") or self.load_rows("calib")
        return []

    def train(self, stages: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Run the full staged curriculum; returns summary metrics.

        Stages (plan §3.4.2): ``T0`` sanity → ``T1`` pointer ceiling gate →
        ``T2`` full multi-task → ``C`` frozen-encoder H8 training. A stage with
        no data is recorded and skipped, so offline/dry runs never crash.
        """
        self._rebuild_optimizer()
        order = tuple(stages or ("T0", "T1", "T2", "C"))
        summary: Dict[str, Any] = {"history": self.history}
        abort_reason = ""
        for stage in order:
            data = self._stage_data(stage)
            if not data:
                summary[stage] = {"status": "skipped", "reason": "no_data"}
                self.history[stage] = summary[stage]
                continue

            n_cfg = self.config.optimization.num_epochs
            n_epochs = n_cfg if len(data) >= 2_000 else max(2, n_cfg)
            # real stage length -> LR warmup/cosine schedule matches the actual run.
            # The scheduler advances once per OPTIMIZER step, so count those (not micro-batches);
            # otherwise with grad-accum the LR never leaves warmup and never decays.
            bs_ = max(self.config.optimization.batch_size, 1)
            accum_ = max(int(self.config.gradient_accumulation_steps), 1)
            steps_per_ep = max(len(list(self.make_batches(data, bs_, shuffle=False))), 1)
            micro_steps = steps_per_ep * n_epochs
            self._sched_total_steps = max((micro_steps + accum_ - 1) // accum_, 1)
            frozen_stages = getattr(self.config, "freeze_encoder_stages", None) or []
            if stage == "C":
                self._freeze_all_but_correctness()
                self._rebuild_optimizer()
            elif stage in frozen_stages:
                # VRAM lever: heads-only stage — no backbone grads, no AdamW
                # state for the ~0.6B encoder parameters.
                self._freeze_encoder()
                self._rebuild_optimizer()
            else:
                if stage == "T1" and not self.stage_gates.get("T0"):
                    # T0 must have run (loss -> ~0) before the pointer gate
                    pass
                self._unfreeze_all()
                self._rebuild_optimizer()

            res = self.run_stage(stage, data, num_epochs=n_epochs)
            summary[stage] = {**res, "status": "ok", "n": float(len(data))}

            if stage == "T1":
                dev = self.evaluate("dev")
                summary["T1"] = {**summary["T1"], **dev}
                if dev.get("n", 0.0) > 0.0:
                    m = dev["exact_payload_match"]
                    self.stage_gates["T1"] = m >= 0.90
                    summary["T1"]["gate_proceed"] = bool(self.stage_gates["T1"])
                    summary["T1"]["gate_kill"] = bool(m < 0.80)
                    if summary["T1"]["gate_kill"]:
                        print(f"[train:T1] GATE KILL: exact_payload_match={m:.4f} < 0.80 "
                              f"(E1 pointer ceiling not met)", flush=True)
                        if getattr(self.config, "stop_on_gate_kill", False):
                            abort_reason = "T1 gate kill"
            elif stage == "T2":
                summary["T2"] = {**summary["T2"], **self.evaluate("dev")}
            self.history[stage] = summary[stage]
            print(f"[train:{stage}] finished: {summary[stage]}", flush=True)
            self._log_event({"event": "stage_end", "stage": stage, **{k: v for k, v in summary[stage].items()}})

            # periodic checkpointing
            try:
                out = Path(self.config.output_dir)
                cp = out / f"{self.config.experiment_name}_seed{self.config.seed}" / stage / "checkpoints"
                self.save_checkpoint(str(cp))
            except Exception as exc:  # checkpointing must never kill training
                print(f"[train:{stage}] WARNING: checkpoint failed: {exc!r}", flush=True)
            if abort_reason:
                summary["aborted"] = abort_reason
                print(f"[train] ABORTED after {stage}: {abort_reason} (stop_on_gate_kill=True)", flush=True)
                break
        return summary

    def evaluate(self, split: str = "dev") -> Dict[str, float]:
        """Evaluate exact-payload match on a split (T1 gate input)."""
        rows = self.load_rows(split)
        if not rows:
            return {"exact_payload_match": float("nan"), "n": 0.0}
        rows = self.tensorize_rows(rows)
        self.model.eval()
        hits = 0.0
        n = 0
        n_gold_abs = abs_hits = n_span = span_hits = n_pred_abs = 0
        tool_hits = 0
        n_tool = 0
        # Same autocast context as train_step: the encoder emits bf16 hidden
        # states while head params are fp32 -> LayerNorm dtype mismatch otherwise.
        t0 = time.time()
        with torch.no_grad():
            for bi, batch_list in enumerate(tqdm(
                    list(self.make_batches(rows, self.config.optimization.batch_size)),
                    desc=f"eval[{split}]", dynamic_ncols=True, leave=False), 1):
                batch = self.collate(batch_list)
                b = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with self._amp():
                    H = self.encoder(b["input_ids"], b["attention_mask"], b.get("token_type_ids"))

                    # Tool head eval if present
                    if "tool_positions" in b and b["tool_positions"].numel() and "tool_targets" in b:
                        tool_head = getattr(self.model, "tool_head", None)
                        if tool_head is not None:
                            max_tok_idx = max(H.size(1) - 1, 0)
                            safe_tool_pos = b["tool_positions"].clamp(min=0, max=max_tok_idx)
                            tool_h = H.gather(1, safe_tool_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
                            tl = tool_head(tool_h, H[:, 0])
                            tl = tl.masked_fill(b["tool_positions"] < 0, -1e4)
                            pred_tools = tl.argmax(-1)
                            tt = b["tool_targets"]
                            valid_tool = tt >= 0
                            if valid_tool.any():
                                tool_hits += int(((pred_tools == tt) & valid_tool).sum().item())
                                n_tool += int(valid_tool.sum().item())

                    if b["param_positions"].size(1) == 0:
                        continue
                    max_tok_idx = max(H.size(1) - 1, 0)
                    safe_param_pos = b["param_positions"].clamp(min=0, max=max_tok_idx)
                    param_h = H.gather(1, safe_param_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
                    p = self.model.pointer_head(
                        param_h, H,
                        candidate_masks={"starts": b["cand_starts"], "ends": b["cand_ends"]},
                    )
                p = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in p.items()}
                starts = b["cand_starts"].clamp(min=0)
                ends = b["cand_ends"].clamp(min=0)
                z = torch.cat([p["abstain_logits"].unsqueeze(-1), p["span_logits"]], dim=-1)
                acc_i, acc_t, valid = self.exact_payload_match(z, starts, ends, b["pos_indices"])
                hits += float((acc_t * valid).sum().item())
                n += int(valid.sum().item())
                # abstain-vs-span breakdown: a model that always abstains can look fine on a
                # mixed set, so report accuracy on gold-abstain and gold-span params separately.
                gold_abs = (b["pos_indices"][..., 0] == 0) & valid
                gold_span = valid & ~gold_abs
                correct = acc_t > 0
                n_gold_abs += int(gold_abs.sum().item())
                abs_hits += int((correct & gold_abs).sum().item())
                n_span += int(gold_span.sum().item())
                span_hits += int((correct & gold_span).sum().item())
                n_pred_abs += int(((z.argmax(-1) == 0) & valid).sum().item())
        res = {
            "exact_payload_match": hits / max(n, 1), "n": float(n),
            "span_acc": span_hits / max(n_span, 1), "n_span": float(n_span),
            "abstain_acc": abs_hits / max(n_gold_abs, 1), "n_gold_abstain": float(n_gold_abs),
            "pred_abstain_rate": n_pred_abs / max(n, 1),
        }
        if n_tool > 0:
            res["tool_acc"] = tool_hits / n_tool
            res["n_tool"] = float(n_tool)
        self._log_event({"event": "eval", "split": split, "stage": self.stage,
                         "step": self.global_step, **res})
        tool_str = f" tool_acc={res.get('tool_acc', 0.0):.4f} (n={n_tool})" if n_tool > 0 else ""
        print(f"[eval:{split}] exact_payload_match={res['exact_payload_match']:.4f} n={n} | "
              f"span_acc={res['span_acc']:.4f} (n={n_span}) abstain_acc={res['abstain_acc']:.4f} "
              f"(n={n_gold_abs}) pred_abstain_rate={res['pred_abstain_rate']:.3f}{tool_str}", flush=True)
        self._release_cuda_cache()
        return res

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str) -> None:
        """Persist model, optimizer, metrics, and provenance to ``path``."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / "model.pt")
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), p / "optimizer.pt")
        with open(p / "metrics.json", "w") as fh:
            json.dump(self.history, fh, indent=2)
        def _snap(obj: Any) -> Dict[str, Any]:
            """Recursively convert a (possibly nested) dataclass config to a dict."""
            if is_dataclass(obj):
                return _snap(asdict(obj))
            if isinstance(obj, dict):
                return {str(k): _snap(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_snap(v) for v in obj]
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            return str(obj)

        prov = {
            "experiment": getattr(self.config, "experiment_name", "span_call"),
            "seed": self.config.seed,
            "stage": self.stage,
            "global_step": self.global_step,
            "git_commit": _git_rev(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "model": _snap(getattr(self.config, "model", None)),
                "optimization": _snap(getattr(self.config, "optimization", None)),
                "data": _snap(getattr(self.config, "data", None)),
            },
        }
        with open(p / "provenance.json", "w") as fh:
            json.dump(prov, fh, indent=2, default=str)

    def load_checkpoint(self, path: str) -> None:
        """Restore model (and optimizer when present) from ``path``."""
        p = Path(path)
        self.model.load_state_dict(torch.load(p / "model.pt", map_location=self.device))
        if (p / "optimizer.pt").exists() and self.optimizer is not None:
            self.optimizer.load_state_dict(torch.load(p / "optimizer.pt", map_location=self.device))


def random_mod():
    import random
    return random
