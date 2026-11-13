"""Lean SpanCall Trainer without abstain machinery.

Provides:
- build_lean_optimizer_groups: AdamW parameter groups with layer-wise decay.
- LeanTrainer: streamlined trainer coordinating LeanSpanCallModel and LeanLossContainer.
  Supports both direct batch-level training (train_step / evaluate(batches)) and
  full config-driven stage training (train / run_stage / evaluate("dev")).
"""

from __future__ import annotations

import gc
import json
import math
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or ())

        def update(self, n=1):
            pass

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            pass

        @staticmethod
        def write(msg: str):
            print(msg, flush=True)

from spancall.lean.loss import LeanLossContainer
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.tensorize import (
    LeanBatchCollate,
    has_positive_targets,
)
from spancall.train.base import BaseTrainer, CosineWarmupSchedule


def _log_loss(value: Any) -> float:
    """Clamp a loss scalar at 0 for LOGGING only (T12c).

    A bf16/fp16 reduction can emit a tiny negative scalar even though the true
    value is non-negative (the CE terms are clamped internally). Those rows then
    land in ``train_log.jsonl`` and get quoted as if a loss could be negative.
    Clamping at the metric boundary keeps the log honest without touching the
    gradient path: ``scaled_loss``/``backward()`` still see the raw tensor.

    Deliberately NOT applied inside :class:`LeanLossContainer` — that would
    change what the optimizer sees, which is a numerics decision nobody asked
    for here.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN: keep it visible rather than silently reporting "fine"
        return v
    return max(0.0, v)


class _MicroLossAccumulator:
    """Slot-weighted average of a loss, invariant to batch composition.

    Why this exists: :meth:`LeanLossContainer.forward` returns a per-BATCH mean
    for each term (the pointer term divides by the batch's active param slots,
    the tool term by the batch's supervised rows). Averaging those means over
    batches -- ``sum(loss_i) / n_batches`` -- is a *mean of ratios*: a 1-row
    batch produced by the length-bucketed token cap carries exactly as much
    weight as a full 8-row batch. Train and eval use different bucketing
    (shuffled megabatches vs. length-sorted), so their ``loss`` series ended up
    on different scales and could not be compared directly.

    This accumulates ``sum(term_sum_i) / sum(term_count_i)`` instead: every
    supervised param slot (pointer) or supervised row (tool) contributes once,
    regardless of how the data was split into batches. The value is therefore
    identical for the same rows under any batching, which is what makes
    ``train_loss`` and ``eval_loss`` a like-for-like pair.

    A term with no supervision anywhere (e.g. a split with no tool targets)
    reports NaN rather than 0.0, so "never measured" cannot be read as a
    perfect score.
    """

    __slots__ = ("_sums", "_counts")

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, float] = {}

    def add(self, out: Dict[str, Any]) -> None:
        """Accumulate one batch's ``*_sum`` / ``*_count`` pairs from the loss dict."""
        for key in ("pointer", "tool", "tool_binary"):
            s = out.get(f"{key}_sum")
            c = out.get(f"{key}_count")
            if s is None or c is None:
                continue
            try:
                s_f = float(s)
                c_f = float(c)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(s_f) or not math.isfinite(c_f):
                continue
            self._sums[key] = self._sums.get(key, 0.0) + s_f
            self._counts[key] = self._counts.get(key, 0.0) + c_f

    def value(self, key: str) -> float:
        """Micro-averaged loss for ``key``; NaN when nothing was supervised."""
        c = self._counts.get(key, 0.0)
        if c <= 0.0:
            return float("nan")
        return _log_loss(self._sums.get(key, 0.0) / c)

    def combined(
        self,
        lambda_pointer: float = 1.0,
        lambda_tool: float = 1.0,
        lambda_tool_binary: float = 0.0,
    ) -> float:
        """The same weighting the objective uses, but slot-weighted.

        Mirrors ``lambda_ptr * L_ptr + lambda_tool * L_tool`` on micro averages,
        and degrades to just the terms that were actually supervised so a
        pointer-only or tool-only split still reports a real number.

        ``lambda_tool_binary`` defaults to 0 so the CE arm's micro loss is
        numerically unchanged; the binary arm passes its own lambda from the
        container. A binary term only exists when the trainer emitted it, so
        ``value("tool_binary")`` is NaN (and thus skipped) in the CE arm.
        """
        parts = []
        if math.isfinite(self.value("pointer")):
            parts.append(lambda_pointer * self.value("pointer"))
        if math.isfinite(self.value("tool")):
            parts.append(lambda_tool * self.value("tool"))
        if lambda_tool_binary and math.isfinite(self.value("tool_binary")):
            parts.append(lambda_tool_binary * self.value("tool_binary"))
        if not parts:
            return float("nan")
        return _log_loss(sum(parts))


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
) -> int:
    """Wrap the backbone in LoRA adapters in place; return adapter param count (T10).

    Deliberately raises when ``peft`` is unavailable instead of falling back to
    full fine-tuning: silently training a different model than the config asked
    for is exactly the `freeze_encoder_stages`-under-`model` defect this
    investigation already caught once.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "training.use_lora=True requires the `peft` package. Install it with "
            "`pip install peft`, or set training.use_lora=False to full fine-tune."
        ) from exc

    targets = list(target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"])
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("apply_lora: model has no `backbone` to adapt")

    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=targets,
        bias="none",
    )
    # ``get_peft_model`` mutates in place and returns the SAME module object, so
    # the caller's reference stays valid and the optimizer groups below see the
    # adapter parameters under their new names.
    model.backbone = get_peft_model(backbone, lora_cfg)
    return sum(
        p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    )


def build_lean_optimizer_groups(
    model: nn.Module,
    lr_encoder: float = 2e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 0.01,
    layer_decay: float = 0.85,
    lora_lr: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Create optimizer parameter groups with layer-wise decay for the backbone.

    When ``lora_lr`` is given, adapter parameters get their own group at that LR.
    They must NOT be folded into the layer-decayed encoder groups: ``lora_*``
    names sit under ``backbone.`` and would otherwise inherit the deepest-layer
    encoder LR (~2e-5), which is two orders of magnitude below what adapters
    need to move at all.
    """
    enc_param_names: set = set()
    layer_index: Dict[str, int] = {}
    n_layers: int = 1

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
        return bool(enc_param_names and name in enc_param_names)

    groups: Dict[str, Dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.dim() <= 1 or any(nd in name.lower() for nd in ("bias", "layernorm"))
        decay = 0.0 if no_decay else weight_decay

        if lora_lr is not None and "lora_" in name:
            # T10: adapters share one group at the adapter LR.
            g = groups.setdefault(
                "lora", {"params": [], "lr": float(lora_lr), "weight_decay": decay}
            )
            g["params"].append(param)
            continue

        if is_encoder(name):
            l = layer_index.get(name, n_layers - 1)
            lr = lr_encoder * (layer_decay ** (n_layers - 1 - l))
            key = f"enc_l{l}"
        else:
            lr = lr_head
            key = "head"

        g = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": decay})
        g["params"].append(param)

    out: List[Dict[str, Any]] = sorted(groups.values(), key=lambda g: -g["lr"])
    return out


class LeanTrainer(BaseTrainer):
    """Streamlined Trainer for Lean SpanCall (ToolHead + LeanPointerHead)."""

    def __init__(
        self,
        model: LeanSpanCallModel,
        config: Optional[Any] = None,
        loss_container: Optional[LeanLossContainer] = None,
        lr_encoder: Optional[float] = None,
        lr_head: Optional[float] = None,
        weight_decay: Optional[float] = None,
        layer_decay: float = 0.85,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
        use_amp: bool = True,
        mixed_precision: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        use_binary_tool_head: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        # v2.1 ablation switch. Resolved as: explicit arg > optimization config >
        # OFF. OFF is the default so every existing caller, test and checkpoint
        # reproduces the shipped softmax H1 byte-for-byte. When ON the binary head
        # is ALSO evaluated and its logits are emitted, and the loss container
        # trains on the binary objective instead of the CE one (see
        # ``LeanLossContainer.tool_objective``).
        if use_binary_tool_head is None:
            _opt = getattr(config, "optimization", None) if config is not None else None
            use_binary_tool_head = bool(getattr(_opt, "use_binary_tool_head", False))
        self.use_binary_tool_head = bool(use_binary_tool_head)
        self.loss_container = loss_container or LeanLossContainer(
            tool_objective="binary" if self.use_binary_tool_head else "ce"
        )

        opt_cfg = getattr(config, "optimization", None) if config is not None else None
        self.lr_encoder = float(
            lr_encoder if lr_encoder is not None else (getattr(opt_cfg, "learning_rate_encoder", 2e-5) if opt_cfg else 2e-5)
        )
        self.lr_head = float(
            lr_head if lr_head is not None else (getattr(opt_cfg, "learning_rate_head", 1e-4) if opt_cfg else 1e-4)
        )
        self.weight_decay = float(
            weight_decay if weight_decay is not None else (getattr(opt_cfg, "weight_decay", 0.01) if opt_cfg else 0.01)
        )
        self.layer_decay = layer_decay
        self.max_grad_norm = max_grad_norm

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        # T10: LoRA ablation. Applied BEFORE ``.to(device)`` and BEFORE the
        # optimizer groups are derived, because the adapter parameters must exist
        # and be trainable by the time ``build_lean_optimizer_groups`` walks
        # ``named_parameters()``.
        self.lora_enabled = bool(getattr(config, "use_lora", False))
        self.lora_param_count = 0
        if self.lora_enabled:
            self.lora_param_count = apply_lora(
                self.model,
                r=int(getattr(config, "lora_r", 16)),
                alpha=int(getattr(config, "lora_alpha", 32)),
                dropout=float(getattr(config, "lora_dropout", 0.05)),
                target_modules=getattr(config, "lora_target_modules", None),
            )

        self.model.to(self.device)

        if config is not None and getattr(config, "gradient_checkpointing", False):
            self._enable_gradient_checkpointing()

        cfg_mp = getattr(opt_cfg, "mixed_precision", "bf16") if opt_cfg is not None else "bf16"
        self.mixed_precision = mixed_precision or (cfg_mp if use_amp else "fp32")
        self.use_amp = bool(use_amp and self.device.startswith("cuda") and torch.cuda.is_available())
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if (self.use_amp and self.mixed_precision == "fp16")
            else None
        )

        groups = build_lean_optimizer_groups(
            self.model,
            lr_encoder=self.lr_encoder,
            lr_head=self.lr_head,
            weight_decay=self.weight_decay,
            layer_decay=self.layer_decay,
            lora_lr=(
                float(getattr(config, "lora_lr", 1e-3)) if self.lora_enabled else None
            ),
        )
        self.optimizer: torch.optim.Optimizer = self._build_optimizer(groups)
        self.scheduler: Optional[CosineWarmupSchedule] = None
        self.step = 0
        self.global_step = 0
        self._accum_i = 0
        self.stage = "T2"
        # Keep the loss objective in lock-step with the flag even when a caller
        # injected its own container: an injected CE container plus the flag on
        # would silently train the OLD head and report a binary arm that was
        # never trained -- the exact failure a kill-criterion ablation must not
        # have.
        if self.use_binary_tool_head:
            self.loss_container.tool_objective = "binary"
        elif getattr(self.loss_container, "tool_objective", "ce") == "binary":
            self.loss_container.tool_objective = "ce"
        self.history: Dict[str, Any] = {"train_loss": [], "dev": {}, "epoch_eval": []}
        self._rng = random.Random(int(getattr(config, "seed", 42) or 42))
        self.collator = LeanBatchCollate()

    def _build_optimizer(self, groups: List[Dict[str, Any]]) -> torch.optim.Optimizer:
        """Build the optimizer named by ``optimization.optimizer``.

        An unknown name raises instead of silently falling back to AdamW: a
        config that asks for an 8-bit optimizer and gets full-precision AdamW is
        a different experiment (and, on a Colab T4, a different OOM outcome).
        """
        opt_name = "adamw"
        if self.config is not None:
            opt_name = str(
                getattr(getattr(self.config, "optimization", None), "optimizer", "adamw") or "adamw"
            ).lower()
        if opt_name in ("adamw", "adam_w", "adam"):
            return AdamW(groups)
        if opt_name in ("adamw_8bit", "paged_adamw_8bit", "adamw8bit"):
            try:
                import bitsandbytes as bnb  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "optimization.optimizer='adamw_8bit' requires `bitsandbytes`. "
                    "Install it (pip install bitsandbytes) or set optimizer: adamw."
                ) from exc
            return bnb.optim.PagedAdamW8bit(groups)
        raise ValueError(
            f"Unknown optimization.optimizer={opt_name!r}; expected "
            "'adamw' or 'adamw_8bit'."
        )

    def effective_batch_size(self) -> int:
        """Rows per OPTIMIZER step (micro-batch x accumulation).

        ``optimization.effective_batch_size`` is authoritative when set; the
        accumulation factor is then DERIVED, so the two knobs cannot disagree
        (which is how the old bs=8 / accum=4 recipe ended up at an effective
        batch of 32 and ~122 optimizer steps per epoch).
        """
        opt_cfg = getattr(self.config, "optimization", None) if self.config is not None else None
        micro = int(getattr(opt_cfg, "batch_size", 8) or 8)
        eff = getattr(opt_cfg, "effective_batch_size", None) if opt_cfg is not None else None
        if eff:
            return max(int(eff), micro)
        accum = max(int(getattr(self.config, "gradient_accumulation_steps", 1) or 1), 1)
        return micro * accum

    def accumulation_steps(self) -> int:
        """Micro-batches accumulated per optimizer step (derived, not guessed)."""
        opt_cfg = getattr(self.config, "optimization", None) if self.config is not None else None
        micro = max(int(getattr(opt_cfg, "batch_size", 8) or 8), 1)
        eff = getattr(opt_cfg, "effective_batch_size", None) if opt_cfg is not None else None
        if eff:
            return max(int(round(max(int(eff), micro) / micro)), 1)
        return max(int(getattr(self.config, "gradient_accumulation_steps", 1) or 1), 1)

    def _amp(self) -> Any:
        """Autocast context on CUDA (bf16 default or fp16); no-op on CPU/MPS."""
        if self.use_amp and self.device.startswith("cuda") and torch.cuda.is_available():
            if self.mixed_precision == "fp16":
                return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self.mixed_precision == "bf16":
                return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _enable_gradient_checkpointing(self) -> None:
        backbone = getattr(self.model.backbone, "backbone", None)
        enable = getattr(backbone, "gradient_checkpointing_enable", None)
        if enable is None:
            return
        try:
            enable(use_reentrant=False)
        except TypeError:
            enable()
        cfg = getattr(backbone, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False

    @staticmethod
    def _live_candidate_counts(batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Per-row count of REAL ``<|tool|>`` anchors, dropping unsupervised rows.

        ``-1`` is the collator's pad sentinel, so a row with no H1 supervision
        has an all-``-1`` row and a count of 0. Including those rows both
        under-reports breadth and hides the vacuity warning, so they are
        filtered out here: the question is "how many candidates does a row that
        H1 is actually trained on see?", not "what is the batch-wide density?".
        """
        tp = batch.get("tool_positions")
        if tp is None or not isinstance(tp, torch.Tensor) or tp.numel() == 0:
            return None
        counts = (tp >= 0).sum(-1).float()
        live = counts > 0
        return counts[live] if bool(live.any()) else None

    def exact_span_match(
        self,
        span_logits: torch.Tensor,
        pos_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Proportion of (b, k) active params whose argmax in [0, C) is a positive."""
        B, K, C = span_logits.shape
        if K == 0 or C == 0:
            empty = torch.zeros((B, K), dtype=torch.float32, device=span_logits.device)
            valid = torch.zeros((B, K), dtype=torch.bool, device=span_logits.device)
            return torch.tensor(0.0, device=span_logits.device), empty, valid

        am = span_logits.argmax(-1).unsqueeze(-1)  # (B, K, 1)
        in_range = (pos_indices >= 0) & (pos_indices < C)
        is_pos = ((pos_indices == am) & in_range).any(-1)  # (B, K)
        n_pos = in_range.sum(-1)
        valid = n_pos > 0

        acc = torch.where(valid, is_pos.float(), torch.zeros_like(is_pos.float()))
        mean_acc = acc.sum() / max(int(valid.sum().item()), 1)
        return mean_acc, acc, valid

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one micro-batch forward/backward and step optimizer on accumulation boundary."""
        self.model.train()
        b = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Derived from `optimization.effective_batch_size` when set, so the
        # accumulation factor and the micro-batch size can never disagree.
        accum_steps = self.accumulation_steps()
        if self._accum_i == 0:
            self.optimizer.zero_grad(set_to_none=True)

        preds: Dict[str, torch.Tensor] = {}
        targs: Dict[str, torch.Tensor] = {}

        with self._amp():
            H = self.model.backbone(b["input_ids"], b["attention_mask"], b.get("token_type_ids"))
            cls_h = H[:, 0]

            max_tok_idx = max(H.size(1) - 1, 0)
            safe_param_pos = b["param_positions"].clamp(min=0, max=max_tok_idx)
            param_h = H.gather(1, safe_param_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
            K = param_h.size(1)

            if "cand_starts" in b and K > 0:
                ptr_out = self.model.pointer_head(
                    param_h,
                    H,
                    candidate_masks={"starts": b["cand_starts"], "ends": b["cand_ends"]},
                )
                preds["span_logits"] = ptr_out["span_logits"]
                t_pos = b["pos_indices"]
                if "param_positions" in b and (b["param_positions"] < 0).any():
                    t_pos = t_pos.clone()
                    t_pos[b["param_positions"] < 0] = -1
                targs["pointer_positives"] = t_pos

            if "tool_positions" in b and b["tool_positions"].numel():
                safe_tool_pos = b["tool_positions"].clamp(min=0, max=max_tok_idx)
                tool_h = H.gather(1, safe_tool_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
                tl = self.model.tool_head(tool_h, cls_h)
                tl = tl.masked_fill(b["tool_positions"] < 0, -1e4)
                preds["tool_logits"] = tl
                # -1 (not 0): 0 would inject a false "tool 0 is correct" target
                # for rows with no supervised tool anchor. LeanLossContainer
                # consumes this via cross_entropy(ignore_index=-1).
                targs["tool_targets"] = b.get(
                    "tool_targets",
                    torch.full((H.size(0),), -1, dtype=torch.long, device=self.device),
                )
                # v2.1: compute the binary head on the SAME anchor hidden states
                # and mask, so the two arms differ only in objective. Emitted
                # only when the flag is on -- its presence is what makes the loss
                # container train the binary term, so flag-off stays identical.
                if self.use_binary_tool_head:
                    bl = self.model.tool_head_binary(tool_h, cls_h)
                    preds["tool_binary_logits"] = bl.masked_fill(b["tool_positions"] < 0, -1e4)
                # R0.4: the call axis is the set-valued target. It is forwarded
                # only when the collator supplied it, so single-call batches keep
                # the bit-identical (B,) cross-entropy path. The scalar
                # ``tool_targets`` above is still set unconditionally because
                # ``tool_acc`` below reads it.
                if "call_tool_targets" in b:
                    targs["call_tool_targets"] = b["call_tool_targets"]

        # Compute loss in fp32 for numerical stability
        fp32_preds = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in preds.items()}
        loss_out = self.loss_container(fp32_preds, targs)
        total_loss = loss_out["loss"]
        scaled_loss = total_loss / accum_steps

        if scaled_loss.requires_grad:
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            self._accum_i += 1
            if self._accum_i % accum_steps == 0:
                self._optimizer_step()

        self.step += 1
        self.global_step += 1

        out_metrics: Dict[str, float] = {"loss": _log_loss(total_loss.detach().item())}
        # Slot-weighted (micro) loss for this micro-batch. It equals `loss` when
        # the batch carries supervision for every term, and is what
        # `run_stage` accumulates so epoch-mean train loss is comparable with
        # the dev `eval_loss` (which uses the same accumulation).
        _acc = _MicroLossAccumulator()
        _acc.add(loss_out)
        _lam_bin = float(getattr(self.loss_container, "lambda_tool_binary", 0.0))
        _train_binary = getattr(self.loss_container, "tool_objective", "ce") == "binary"
        out_metrics["micro_loss"] = _acc.combined(
            self.loss_container.lambda_pointer,
            # In the binary arm the CE term is reference-only, so it must not be
            # added into the comparable micro loss either.
            0.0 if _train_binary else self.loss_container.lambda_tool,
            _lam_bin if _train_binary else 0.0,
        )
        if "pointer" in loss_out:
            out_metrics["pointer_loss"] = _log_loss(loss_out["pointer"].item())
            out_metrics["pointer"] = out_metrics["pointer_loss"]
            out_metrics["pointer_micro"] = _acc.value("pointer")
        if "tool" in loss_out:
            out_metrics["tool_loss"] = _log_loss(loss_out["tool"].item())
            out_metrics["tool"] = out_metrics["tool_loss"]
            out_metrics["tool_micro"] = _acc.value("tool")
        if "tool_binary" in loss_out:
            # Reference metric in the CE arm, trained objective in the binary arm.
            out_metrics["tool_binary_loss"] = _log_loss(loss_out["tool_binary"].item())
            out_metrics["tool_binary_micro"] = _acc.value("tool_binary")
        # Raw sums/counts are returned so `run_stage` can aggregate them across
        # micro-batches without re-deriving them from a rounded mean.
        out_metrics["_pointer_sum"] = float(loss_out.get("pointer_sum", 0.0))
        out_metrics["_pointer_count"] = float(loss_out.get("pointer_count", 0.0))
        out_metrics["_tool_sum"] = float(loss_out.get("tool_sum", 0.0))
        out_metrics["_tool_count"] = float(loss_out.get("tool_count", 0.0))
        # Only present when the binary arm actually ran. Emitting 0.0
        # unconditionally would put ``_tool_binary_*`` keys in every flag-off
        # run's history -- i.e. the shipped arm would stop being byte-identical
        # for exactly the fields this ablation adds.
        if "tool_binary_sum" in loss_out:
            out_metrics["_tool_binary_sum"] = float(loss_out.get("tool_binary_sum", 0.0))
            out_metrics["_tool_binary_count"] = float(loss_out.get("tool_binary_count", 0.0))

        if "span_logits" in fp32_preds and "pos_indices" in b:
            span_acc, _, _ = self.exact_span_match(fp32_preds["span_logits"], b["pos_indices"])
            out_metrics["span_acc"] = float(span_acc.item())

        if "tool_logits" in fp32_preds and "tool_targets" in targs:
            t_logits = fp32_preds["tool_logits"]
            t_targets = targs["tool_targets"]
            # Mask padded (-1) / out-of-range targets: an unmasked batch mean
            # lets -1e4-masked anchors (whose argmax ties to index 0) pollute
            # the metric, and a batch holding only padded rows is not a result.
            valid_t = (t_targets >= 0) & (t_targets < t_logits.size(-1))
            if bool(valid_t.any()):
                t_correct = (t_logits.argmax(-1)[valid_t] == t_targets[valid_t]).float().mean()
                out_metrics["tool_acc"] = float(t_correct.item())
                out_metrics["n_tool"] = float(int(valid_t.sum().item()))
            # v2.1: the SAME argmax semantics as ``tool_acc`` so the two arms are
            # directly comparable -- this is the kill-criterion metric, so it must
            # not be a different statistic. Sigmoid scores are monotonic, so the
            # argmax of the logits equals the argmax of P(called).
            if "tool_binary_logits" in fp32_preds:
                b_logits = fp32_preds["tool_binary_logits"]
                valid_b = (t_targets >= 0) & (t_targets < b_logits.size(-1))
                if bool(valid_b.any()):
                    b_correct = (b_logits.argmax(-1)[valid_b] == t_targets[valid_b]).float().mean()
                    out_metrics["tool_binary_acc"] = float(b_correct.item())
                    out_metrics["n_tool_binary"] = float(int(valid_b.sum().item()))
            n_cand = self._live_candidate_counts(b)
            if n_cand is not None and n_cand.numel():
                out_metrics["mean_n_candidates"] = float(n_cand.mean().item())

        return out_metrics

    def _optimizer_step(self) -> None:
        if self.scaler is not None:
            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self._accum_i = 0

    def evaluate(
        self,
        split_or_batches: Union[str, Sequence[Dict[str, torch.Tensor]]] = "dev",
        limit: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate either on a split name (e.g. 'dev') or a sequence of batch dicts.

        Multi-call note (R0.4): the batch built here carries only ``tool_targets``
        (the PRIMARY tool), not ``call_tool_targets``. A multi-call row is
        therefore scored on its call-0 tool only, and ``tool_acc`` is a
        single-call upper bound — the full call-set accuracy is measured
        end-to-end by Part 3 (:class:`spancall.eval.toolcall.ToolCallEvaluator`),
        not here. This is deliberate: the eval loop mirrors the inference
        contract of one primary call per prompt, so the number stays comparable
        with pre-R0.4 runs.
        """
        self.model.eval()

        if isinstance(split_or_batches, str):
            rows = self.load_rows(split_or_batches)
            if not rows:
                # NaN, not 0.0: a zero is indistinguishable from "predicted
                # everything wrong" once it reaches telemetry. ``run_stage``
                # skips non-finite values instead of recording a fake score.
                nan = float("nan")
                return {
                    "eval_loss": nan,
                    "eval_loss_batchmean": nan,
                    "eval_pointer_loss": nan,
                    "eval_pointer_loss_micro": nan,
                    "eval_tool_loss_micro": nan,
                    "eval_n_pointer_slots": 0.0,
                    "eval_n_tool_rows": 0.0,
                    "eval_span_acc": nan,
                    "span_acc": nan,
                    "exact_call": nan,
                    "exact_payload_match": nan,
                    "tool_acc": nan,
                    "n_tool": 0.0,
                    "mean_n_candidates": nan,
                    "n": 0.0,
                }
            if limit is not None and len(rows) > limit:
                rows = rows[:limit]
            rows = self.tensorize_rows(rows)
            bs = getattr(getattr(self.config, "optimization", None), "batch_size", 8) if self.config else 8
            batches: Iterable[Dict[str, torch.Tensor]] = (
                self.collate(bl) for bl in self.make_batches(rows, bs, shuffle=False)
            )
            split_label = split_or_batches
        else:
            batches = split_or_batches
            split_label = "batches"

        # Micro-accumulator: the SAME slot-weighted average `run_stage` uses for
        # train loss, so `eval_loss` and `mean_loss` become a like-for-like pair
        # (see `_MicroLossAccumulator`). `total_loss`/`n_batches` are kept as the
        # legacy mean-of-batch-means for backward comparison in the logs.
        micro = _MicroLossAccumulator()
        total_loss = 0.0
        total_ptr_loss = 0.0
        n_batches = 0
        n_span_batches = 0
        span_hits = 0.0
        n_span = 0
        tool_hits = 0
        n_tool = 0
        # v2.1 ablation counters: identical denominator and argmax rule as the
        # CE arm above, so ``tool_binary_acc`` and ``tool_acc`` are comparable.
        tool_bin_hits = 0
        n_tool_bin = 0
        tool_cand_sum = 0.0
        tool_cand_rows = 0
        exact_hits = 0.0
        n_exact = 0

        with torch.no_grad():
            for batch in batches:
                b = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                with self._amp():
                    H = self.model.backbone(b["input_ids"], b["attention_mask"], b.get("token_type_ids"))
                    cls_h = H[:, 0]
                    max_tok_idx = max(H.size(1) - 1, 0)
                    safe_param_pos = b["param_positions"].clamp(min=0, max=max_tok_idx)
                    param_h = H.gather(1, safe_param_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
                    K = param_h.size(1)

                    preds: Dict[str, torch.Tensor] = {}
                    targs: Dict[str, torch.Tensor] = {}

                    if "cand_starts" in b and K > 0:
                        ptr_out = self.model.pointer_head(
                            param_h,
                            H,
                            candidate_masks={"starts": b["cand_starts"], "ends": b["cand_ends"]},
                        )
                        preds["span_logits"] = ptr_out["span_logits"]
                        t_pos = b["pos_indices"]
                        if "param_positions" in b and (b["param_positions"] < 0).any():
                            t_pos = t_pos.clone()
                            t_pos[b["param_positions"] < 0] = -1
                        targs["pointer_positives"] = t_pos

                    if "tool_positions" in b and b["tool_positions"].numel():
                        safe_tool_pos = b["tool_positions"].clamp(min=0, max=max_tok_idx)
                        tool_h = H.gather(1, safe_tool_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
                        tl = self.model.tool_head(tool_h, cls_h)
                        tl = tl.masked_fill(b["tool_positions"] < 0, -1e4)
                        preds["tool_logits"] = tl
                        # -1 (not 0) when absent: 0 would masquerade as a real
                        # "tool 0 is correct" target for every row in the batch.
                        targs["tool_targets"] = b.get(
                            "tool_targets",
                            torch.full((H.size(0),), -1, dtype=torch.long, device=self.device),
                        )
                        # v2.1: mirror ``train_step`` exactly. Evaluating the
                        # binary head on the same rows/batches is what makes the
                        # ``tool_binary_acc`` vs ``tool_acc`` pair a fair ablation.
                        if self.use_binary_tool_head:
                            bl = self.model.tool_head_binary(tool_h, cls_h)
                            preds["tool_binary_logits"] = bl.masked_fill(
                                b["tool_positions"] < 0, -1e4
                            )

                fp32_preds = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in preds.items()}
                loss_out = self.loss_container(fp32_preds, targs)
                total_loss += _log_loss(loss_out["loss"].item())
                micro.add(loss_out)
                if "pointer" in loss_out:
                    total_ptr_loss += _log_loss(loss_out["pointer"].item())

                # Per-row whole-call exactness: every supervised param must land
                # on a positive candidate AND the predicted tool must be the GT
                # one. This is the metric the name ``exact_payload_match``
                # promises; it used to be aliased to span_acc, which silently
                # reported a param-level number as a call-level one.
                row_ok = torch.ones(int(H.size(0)), dtype=torch.bool, device=H.device)
                exact_eligible = torch.ones(int(H.size(0)), dtype=torch.bool, device=H.device)

                if "span_logits" in fp32_preds and "pos_indices" in b:
                    _, acc_t, valid = self.exact_span_match(fp32_preds["span_logits"], b["pos_indices"])
                    span_hits += float((acc_t * valid).sum().item())
                    n_span += int(valid.sum().item())
                    n_span_batches += 1
                    # vacuous / unsupervised params (valid == False) do not
                    # block an exact call match; supervised ones must all hit.
                    # W0.3: rows with ZERO supervised parameters (e.g. Stratum-A)
                    # must not be counted as exact_call hits.
                    exact_eligible = exact_eligible & valid.any(-1)
                    row_ok = row_ok & ((acc_t >= 0.5) | ~valid).all(-1)

                if "tool_logits" in fp32_preds and "tool_targets" in b:
                    pred_tools = fp32_preds["tool_logits"].argmax(-1)
                    tt = b["tool_targets"]
                    # A target is only scoreable when it addresses a real
                    # candidate column; -1 is the "absent/padded" sentinel.
                    valid_tool = (tt >= 0) & (tt < fp32_preds["tool_logits"].size(-1))
                    if valid_tool.any():
                        tool_hits += int(((pred_tools == tt) & valid_tool).sum().item())
                        n_tool += int(valid_tool.sum().item())
                    # v2.1 ablation arm: identical denominator and argmax rule.
                    if "tool_binary_logits" in fp32_preds:
                        b_pred = fp32_preds["tool_binary_logits"].argmax(-1)
                        valid_bin = (tt >= 0) & (tt < fp32_preds["tool_binary_logits"].size(-1))
                        if valid_bin.any():
                            tool_bin_hits += int(((b_pred == tt) & valid_bin).sum().item())
                            n_tool_bin += int(valid_bin.sum().item())
                    # rows without a supervised tool target are not penalised.
                    row_ok = row_ok & ((pred_tools == tt) | ~valid_tool)
                    # Diagnostic: mean number of real candidate tools per row.
                    # 1.0 => H1 is a singleton softmax, so tool_acc is vacuous
                    # Rows without a
                    # live anchor are excluded rather than counted as 0.
                    n_cand_b = self._live_candidate_counts(b)
                    if n_cand_b is not None:
                        tool_cand_sum += float(n_cand_b.sum().item())
                        tool_cand_rows += int(n_cand_b.numel())

                exact_hits += float((row_ok & exact_eligible).sum().item())
                n_exact += int(exact_eligible.sum().item())
                n_batches += 1

        if n_batches == 0:
            # NaN (not 0.0) so "never evaluated" cannot be mistaken for
            # "evaluated and scored zero" downstream.
            nan = float("nan")
            return {
                "eval_loss": nan,
                "eval_loss_batchmean": nan,
                "eval_pointer_loss": nan,
                "eval_pointer_loss_micro": nan,
                "eval_tool_loss_micro": nan,
                "eval_n_pointer_slots": 0.0,
                "eval_n_tool_rows": 0.0,
                "eval_span_acc": nan,
                "span_acc": nan,
                "exact_payload_match": nan,
                "exact_call": nan,
                "tool_acc": nan,
                "n_tool": 0.0,
                "mean_n_candidates": nan,
                "n": 0.0,
            }

        overall_span_acc = span_hits / max(n_span, 1)
        overall_exact = exact_hits / max(n_exact, 1)
        # `eval_loss` is now the SLOT-WEIGHTED average (same definition as the
        # train `mean_loss`). The old mean-of-batch-means is preserved under
        # `eval_loss_batchmean` so historical runs stay comparable, but new
        # analysis should read `eval_loss`: it is invariant to batching, so a
        # train/dev gap is a model gap rather than a bucketing artifact.
        # Mirror ``train_step``'s objective isolation so ``eval_loss`` and the
        # train ``mean_loss`` remain ONE statistic. In the binary arm the CE term
        # is reference-only (it has no ``tool_sum``/``tool_count`` to average
        # anyway) and the binary term is the trained objective, so it is the one
        # that must enter the comparable loss.
        _eval_binary = getattr(self.loss_container, "tool_objective", "ce") == "binary"
        micro_loss = micro.combined(
            self.loss_container.lambda_pointer,
            0.0 if _eval_binary else self.loss_container.lambda_tool,
            (
                float(getattr(self.loss_container, "lambda_tool_binary", 0.0))
                if _eval_binary
                else 0.0
            ),
        )
        res: Dict[str, float] = {
            "eval_loss": micro_loss if math.isfinite(micro_loss) else total_loss / n_batches,
            "eval_loss_batchmean": total_loss / n_batches,
            "eval_pointer_loss_micro": micro.value("pointer"),
            "eval_tool_loss_micro": micro.value("tool"),
            "eval_n_pointer_slots": float(micro._counts.get("pointer", 0.0)),
            "eval_n_tool_rows": float(micro._counts.get("tool", 0.0)),
            "eval_pointer_loss": total_ptr_loss / max(n_span_batches, 1),
            "eval_span_acc": overall_span_acc,
            "span_acc": overall_span_acc,
            # Call-level exactness: EVERY supervised param correct AND the tool
            # correct. Distinct from ``span_acc`` (per-param, so usually larger).
            #
            # ``exact_call`` is the canonical name. ``exact_payload_match`` is
            # kept as a deprecated alias because it USED to be a straight alias
            # of ``span_acc``; its meaning changed here, so any historical log or
            # checkpoint keyed on the old name is NOT comparable. Consumers should
            # read ``exact_call`` and treat old ``exact_payload_match`` series as
            # a different metric.
            "exact_call": overall_exact,
            "exact_payload_match": overall_exact,
            "n": float(n_span),
            "n_span": float(n_span),
        }
        if n_tool > 0:
            res["tool_acc"] = tool_hits / n_tool
            res["n_tool"] = float(n_tool)
        if n_tool_bin > 0:
            # Only present when the binary arm actually ran; absent (rather than
            # NaN) keeps the flag-off result dict byte-identical.
            res["tool_binary_acc"] = tool_bin_hits / n_tool_bin
            res["n_tool_binary"] = float(n_tool_bin)
        if tool_cand_rows > 0:
            res["mean_n_candidates"] = tool_cand_sum / tool_cand_rows

        if isinstance(split_or_batches, str):
            self._log_event({"event": "eval", "split": split_label, "stage": self.stage, "step": self.global_step, **res})
            tool_str = ""
            if n_tool > 0:
                tool_str = f" tool_acc={res['tool_acc']:.4f} (n={n_tool})"
                mean_c = res.get("mean_n_candidates")
                if mean_c is not None:
                    # Surface the degeneracy: with ~1.0 candidates per row the
                    # softmax has a single class, so tool_acc is 1.0 by
                    # construction and means nothing.
                    tool_str += f" candidates/row={mean_c:.2f}"
                    if mean_c < 1.5:
                        tool_str += " [!vacuous: single-candidate H1]"
            if "tool_binary_acc" in res:
                # Printed on its own line: with the flag on this is the number the
                # v2.1 decision reads, and it must be impossible to confuse with
                # the CE ``tool_acc`` printed above.
                tool_str += f" tool_bin_acc={res['tool_binary_acc']:.4f} (n={int(res['n_tool_binary'])})"
            # Print the micro (slot-weighted) loss FIRST because it is the scale
            # that matches the train epoch `mean_loss`; the legacy batch-mean is
            # shown alongside so a reader can see how much of the old gap was
            # bucketing rather than model quality.
            ptr_str = ""
            if math.isfinite(res["eval_pointer_loss_micro"]):
                ptr_str = f" ptr={res['eval_pointer_loss_micro']:.4f}"
            if math.isfinite(res["eval_tool_loss_micro"]):
                ptr_str += f" tool={res['eval_tool_loss_micro']:.4f}"
            print(
                f"[lean-eval:{split_label}] span_acc={overall_span_acc:.4f} (n={n_span}) "
                f"loss={res['eval_loss']:.4f} (batchmean={res['eval_loss_batchmean']:.4f})"
                f"{ptr_str}{tool_str}",
                flush=True,
            )
        return res

    # ------------------------------------------------------------------ #
    # Dataset / Stage Helpers
    # ------------------------------------------------------------------ #
    def load_rows(self, split: str, filter_empty_positives: bool = True) -> List[Dict[str, Any]]:
        data_dir = getattr(self.config, "data_dir", None) or "data/processed"
        path = Path(data_dir) / f"{split}.jsonl"
        if not path.exists():
            alt_path = Path(data_dir) / f"hermes_{split}.jsonl"
            if alt_path.exists():
                path = alt_path
            else:
                return []
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        frac = float(getattr(getattr(self.config, "data", None), "fraction", 1.0) or 1.0) if self.config else 1.0
        if frac < 1.0 and rows:
            rng = random.Random(int(getattr(self.config, "seed", 42) or 42))
            n = max(int(len(rows) * frac), 1)
            rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), min(n, len(rows))))]

        if filter_empty_positives and rows and "pos_indices" in rows[0]:
            groundable = [r for r in rows if has_positive_targets(r)]
            if not groundable and rows:
                print(f"[!] WARNING: filter_empty_positives=True but 0/{len(rows)} rows had positive targets!")
            rows = groundable

        return rows

    def tensorize_rows(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rows:
            d: Dict[str, Any] = {}
            for k, v in r.items():
                if k.startswith("_"):
                    d[k] = v
                    continue
                t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
                if k in ("cand_starts", "cand_ends", "pos_indices") and t.dim() == 1:
                    if t.numel() == 0:
                        t = t.reshape(0, 0).long()
                    else:
                        t = t.unsqueeze(0).long()
                d[k] = t
            out.append(d)
        return out

    def collate(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return self.collator(batch)

    def make_batches(
        self,
        rows: Sequence[Dict[str, Any]],
        batch_size: int,
        shuffle: bool = False,
        rng: Optional[Any] = None,
        max_tokens_per_batch: Optional[int] = None,
    ) -> Iterable[Sequence[Dict[str, Any]]]:
        bs = max(int(batch_size), 1)
        # `optimization.max_tokens_per_batch` is the supported knob: without it
        # the cap was silently ``bs * 1024``, which changes meaning whenever the
        # micro-batch size changes and produced the 1-row batches that skewed the
        # loss accounting (see `_MicroLossAccumulator`).
        opt_cfg = getattr(self.config, "optimization", None) if self.config else None
        cfg_cap = getattr(self.config, "max_tokens_per_batch", None) if self.config else None
        if cfg_cap is None and opt_cfg is not None:
            cfg_cap = getattr(opt_cfg, "max_tokens_per_batch", None)
        token_cap = int(max_tokens_per_batch or cfg_cap or (bs * 1024))

        def _seq_len(i: int) -> int:
            ids = rows[i].get("input_ids")
            return int(ids.numel() if isinstance(ids, torch.Tensor) else len(ids or ()))

        def _pack_sorted(indices: List[int]) -> List[List[int]]:
            packed: List[List[int]] = []
            cur: List[int] = []
            cur_max = 0
            for i in indices:
                L = _seq_len(i)
                new_max = max(cur_max, L)
                if cur and (len(cur) >= bs or (len(cur) + 1) * new_max > token_cap):
                    packed.append(cur)
                    cur = [i]
                    cur_max = L
                else:
                    cur.append(i)
                    cur_max = new_max
            if cur:
                packed.append(cur)
            return packed

        if shuffle:
            r = rng if rng is not None else self._rng
            idx = list(range(len(rows)))
            r.shuffle(idx)
            pool_sz = bs * 16
            batches: List[List[int]] = []
            for start in range(0, len(idx), pool_sz):
                megabatch = sorted(idx[start : start + pool_sz], key=_seq_len)
                batches.extend(_pack_sorted(megabatch))
            r.shuffle(batches)
            for b_idx in batches:
                yield [rows[i] for i in b_idx]
        else:
            for b_idx in _pack_sorted(list(range(len(rows)))):
                yield [rows[i] for i in b_idx]

    def run_stage(self, stage: str, data: List[Dict[str, Any]], num_epochs: int) -> Dict[str, float]:
        self.stage = stage
        rows = self.tensorize_rows(data)
        if not rows:
            return {"loss": float("nan")}

        bs = getattr(getattr(self.config, "optimization", None), "batch_size", 8) if self.config else 8
        # Derived from `optimization.effective_batch_size` when set, so the micro
        # batch and the accumulation factor can never disagree.
        accum = self.accumulation_steps()
        steps_per_epoch = max(len(list(self.make_batches(rows, bs, shuffle=False))), 1)
        total_steps = steps_per_epoch * num_epochs
        opt_total_steps = max((total_steps + accum - 1) // accum, 1)
        opt_cfg = getattr(self.config, "optimization", None) if self.config is not None else None
        warmup_ratio = float(getattr(opt_cfg, "warmup_ratio", 0.06) or 0.06)
        # `scheduler` / `min_lr_ratio` are config-driven. The old cosine-to-zero
        # default spent the last epochs at ~1e-9 LR, which is exactly when dev
        # metrics were decaying; an LR floor keeps those epochs informative.
        _sched_name = str(getattr(opt_cfg, "scheduler", "cosine_warmup") or "cosine_warmup").lower()
        if _sched_name not in ("cosine_warmup", "cosine", "linear", "constant"):
            raise ValueError(
                f"Unknown optimization.scheduler={_sched_name!r}; expected "
                "'cosine_warmup', 'linear' or 'constant'."
            )
        _kind = "cosine" if _sched_name.startswith("cosine") else _sched_name
        self.scheduler = CosineWarmupSchedule(
            self.optimizer,
            max(int(opt_total_steps * warmup_ratio), 1),
            opt_total_steps,
            min_lr_ratio=float(getattr(opt_cfg, "min_lr_ratio", 0.0) or 0.0),
            kind=_kind,
        )

        log_every = max(total_steps // 100, 1)
        t_start = time.time()
        self._accum_i = 0
        steps = 0
        # Early-stopping / best-weight bookkeeping for THIS stage. `_best_epoch`
        # is what makes `restore_best_on_exit` possible: the stage checkpoint
        # written by `train()` used to be the last epoch, which is empirically
        # the worst one once dev metrics start decaying.
        self._stopped_early = False
        self._best_epoch = None
        self._epochs_since_best = 0
        print(
            f"[lean-train:{stage}] start: rows={len(rows)} epochs={num_epochs} "
            f"steps/epoch={steps_per_epoch} total_steps={total_steps} bs={bs} "
            f"accum={accum} effective_bs={bs * accum} "
            f"opt_steps={opt_total_steps} lr_floor_ratio="
            f"{float(getattr(opt_cfg, 'min_lr_ratio', 0.0) or 0.0):.2f}",
            flush=True,
        )
        self._log_event({"event": "stage_start", "stage": stage, "rows": len(rows), "epochs": num_epochs, "total_steps": total_steps})

        pbar = tqdm(total=total_steps, desc=f"lean-train[{stage}]", dynamic_ncols=True, leave=True)
        for epoch in range(1, num_epochs + 1):
            ep_sum, ep_n = 0.0, 0
            # Slot-weighted epoch accumulators, mirroring `evaluate`. The epoch
            # `mean_loss` reported to `train_log.jsonl` is the micro average so it
            # can be plotted against `eval_loss` on the same axis; the legacy
            # mean-of-micro-batch-means is kept as `mean_loss_batchmean`.
            ep_micro_sums: Dict[str, float] = {"pointer": 0.0, "tool": 0.0, "tool_binary": 0.0}
            ep_micro_counts: Dict[str, float] = {"pointer": 0.0, "tool": 0.0, "tool_binary": 0.0}
            for batch_list in self.make_batches(rows, bs, shuffle=True, rng=self._rng):
                if not batch_list:
                    continue
                batch = self.collate(batch_list)
                losses = self.train_step(batch)
                steps += 1
                self.history["train_loss"].append({**losses, "stage": stage, "step": self.global_step})
                if math.isfinite(losses.get("loss", float("nan"))):
                    ep_sum += losses["loss"]
                    ep_n += 1
                for _k in ("pointer", "tool", "tool_binary"):
                    ep_micro_sums[_k] += float(losses.get(f"_{_k}_sum", 0.0) or 0.0)
                    ep_micro_counts[_k] += float(losses.get(f"_{_k}_count", 0.0) or 0.0)
                pbar.update(1)
                pbar.set_postfix(
                    ep=f"{epoch}/{num_epochs}",
                    loss=f"{losses.get('loss', float('nan')):.4f}",
                    lr=f"{self.optimizer.param_groups[-1]['lr']:.1e}",
                )
                if steps % log_every == 0 or steps == total_steps:
                    self._log_event({
                        "event": "train",
                        "stage": stage,
                        "epoch": epoch,
                        "step": self.global_step,
                        "lr": self.optimizer.param_groups[-1]["lr"],
                        "elapsed_s": time.time() - t_start,
                        **losses,
                    })

            batch_mean_loss = ep_sum / max(ep_n, 1)

            def _micro(key: str) -> float:
                c = ep_micro_counts.get(key, 0.0)
                if c <= 0.0:
                    return float("nan")
                return _log_loss(ep_micro_sums.get(key, 0.0) / c)

            ptr_micro, tool_micro = _micro("pointer"), _micro("tool")
            bin_micro = _micro("tool_binary")
            _train_binary = getattr(self.loss_container, "tool_objective", "ce") == "binary"
            parts = []
            if math.isfinite(ptr_micro):
                parts.append(self.loss_container.lambda_pointer * ptr_micro)
            # Only the objective that is actually TRAINED enters the epoch mean;
            # the other head's value is still logged (below) as a reference.
            if math.isfinite(tool_micro) and not _train_binary:
                parts.append(self.loss_container.lambda_tool * tool_micro)
            if math.isfinite(bin_micro) and _train_binary:
                parts.append(
                    float(getattr(self.loss_container, "lambda_tool_binary", 1.0)) * bin_micro
                )
            # `mean_loss` is the slot-weighted value so it shares a scale with
            # `eval_loss`; the legacy mean-of-batch-means stays as
            # `mean_loss_batchmean` for continuity with older logs.
            mean_loss = _log_loss(sum(parts)) if parts else batch_mean_loss
            tqdm.write(
                f"[lean-train:{stage}] epoch {epoch}/{num_epochs} done: "
                f"mean_loss={mean_loss:.4f} (batchmean={batch_mean_loss:.4f})"
                + (f" ptr={ptr_micro:.4f}" if math.isfinite(ptr_micro) else "")
                + (f" tool={tool_micro:.4f}" if math.isfinite(tool_micro) else "")
                + (f" tool_bin={bin_micro:.4f}" if math.isfinite(bin_micro) else "")
            )
            self._log_event({
                "event": "epoch_end",
                "stage": stage,
                "epoch": epoch,
                "mean_loss": mean_loss,
                "mean_loss_batchmean": batch_mean_loss,
                "pointer_loss": ptr_micro,
                "tool_loss": tool_micro,
                "tool_binary_loss": bin_micro,
                "n_pointer_slots": ep_micro_counts.get("pointer", 0.0),
                "n_tool_rows": ep_micro_counts.get("tool", 0.0),
            })

            eval_splits = getattr(self.config, "eval_splits", None) or []
            if isinstance(eval_splits, str):
                eval_splits = [eval_splits]
            eval_limit = getattr(self.config, "eval_limit", 500) if self.config else 500
            for es in eval_splits:
                ev_res = self.evaluate(es, limit=eval_limit)

                def _f(key: str) -> float:
                    # Preserve NaN ("never evaluated") instead of coercing it to a
                    # real-looking 0.0; ``.get(default)`` only covers a MISSING key.
                    v = ev_res.get(key, float("nan"))
                    return float(v) if v is not None else float("nan")

                ep_rec = {
                    "stage": stage,
                    "epoch": epoch,
                    "split": es,
                    # Dev loss on the SAME slot-weighted scale as the train
                    # `mean_loss`, plus the legacy batch-mean for continuity.
                    # Persisting both is what makes a train-eval loss plot
                    # meaningful: they were previously different statistics.
                    "eval_loss": _f("eval_loss"),
                    "eval_loss_batchmean": _f("eval_loss_batchmean"),
                    "eval_pointer_loss": _f("eval_pointer_loss_micro"),
                    "eval_tool_loss": _f("eval_tool_loss_micro"),
                    "span_acc": _f("span_acc"),
                    # canonical name; ``exact_payload_match`` is the deprecated
                    # alias whose MEANING changed (it used to equal span_acc), so
                    # it is persisted under both names to keep old logs readable
                    # while new consumers read exact_call.
                    "exact_call": _f("exact_call"),
                    "exact_payload_match": _f("exact_call"),
                    "abstain_acc": 0.0,
                    "n": _f("n"),
                    # H1 (tool selection) — persisted so a vacuous single-candidate
                    # 1.0 is distinguishable from a real multi-tool result after
                    # the fact. ``mean_n_candidates`` ~1.0 => vacuous.
                    "tool_acc": _f("tool_acc"),
                    "n_tool": _f("n_tool"),
                    # v2.1: persisted alongside ``tool_acc`` so the kill criterion
                    # (binary <= CE => revert) can be read off the epoch history
                    # without re-running either arm.
                    "tool_binary_acc": _f("tool_binary_acc"),
                    "n_tool_binary": _f("n_tool_binary"),
                    "mean_n_candidates": _f("mean_n_candidates"),
                }
                self.history["epoch_eval"].append(ep_rec)
                self._log_event({"event": "epoch_eval", **ep_rec})

                cur_metric = _f("exact_call")
                # A NaN metric (empty split / nothing supervised) must not win the
                # best-checkpoint race, and must not overwrite a real best.
                if math.isfinite(cur_metric) and (
                    not hasattr(self, "_best_curve_metric") or cur_metric > self._best_curve_metric
                ):
                    self._best_curve_metric = cur_metric
                    # Track WHERE in the run the best weights were seen. The stage
                    # checkpoint written at `train()` time used to be the LAST
                    # epoch, which is empirically the WORST one (dev metrics peak
                    # around epoch 2 of 5 and then decay monotonically) -- so any
                    # consumer that resolved "latest stage checkpoint" silently got
                    # overfit weights. See `_restore_best_if_requested`.
                    self._best_epoch = epoch
                    self._epochs_since_best = 0
                    self._restore_best = bool(
                        getattr(getattr(self.config, "optimization", None), "restore_best_on_exit", True)
                    )
                    if self.config is not None:
                        out = Path(self.config.output_dir)
                        best_cp = out / f"{self.config.experiment_name}_seed{self.config.seed}" / "best_checkpoint"
                        self.save_checkpoint(str(best_cp))
                        print(f"[lean-train:{stage}:epoch {epoch}] NEW BEST on {es}: "
                              f"exact_call={cur_metric:.4f} -> saved to {best_cp}", flush=True)
                elif math.isfinite(cur_metric):
                    self._epochs_since_best = int(getattr(self, "_epochs_since_best", 0)) + 1
                    patience = int(
                        getattr(getattr(self.config, "optimization", None), "early_stopping_patience", 0) or 0
                    )
                    if patience > 0 and self._epochs_since_best >= patience:
                        tqdm.write(
                            f"[lean-train:{stage}] early stop at epoch {epoch}/{num_epochs}: "
                            f"no {es} exact_call improvement for {self._epochs_since_best} epoch(s) "
                            f"(best={getattr(self, '_best_curve_metric', float('nan')):.4f} "
                            f"@ epoch {getattr(self, '_best_epoch', '?')}, patience={patience})"
                        )
                        self._log_event({
                            "event": "early_stop", "stage": stage, "epoch": epoch,
                            "best_epoch": getattr(self, "_best_epoch", None),
                            "best_metric": getattr(self, "_best_curve_metric", None),
                            "patience": patience,
                        })
                        self._stopped_early = True
                        break  # leave the eval-splits loop
                if self._stopped_early:
                    break  # leave the epoch loop

        pbar.close()
        if self._accum_i > 0:
            self._optimizer_step()

        # `restore_best_on_exit`: put the BEST epoch's weights back into the live
        # model so the stage checkpoint (and anything derived from it) is the
        # metric-best model, not the last -- and typically most overfit -- one.
        restored = False
        best_metric = getattr(self, "_best_curve_metric", None)
        if (
            bool(getattr(getattr(self.config, "optimization", None), "restore_best_on_exit", True))
            and self.config is not None
            and self._best_epoch is not None
        ):
            best_cp = (
                Path(self.config.output_dir)
                / f"{self.config.experiment_name}_seed{self.config.seed}"
                / "best_checkpoint"
                / "model.pt"
            )
            if best_cp.exists():
                try:
                    state = torch.load(str(best_cp), map_location=self.device)
                    self.model.load_state_dict(state, strict=False)
                    restored = True
                    tqdm.write(
                        f"[lean-train:{stage}] restored BEST weights from epoch "
                        f"{self._best_epoch} ({best_cp}) -- the stage checkpoint will "
                        f"NOT be the overfit final epoch"
                    )
                    self._log_event({
                        "event": "restore_best", "stage": stage,
                        "best_epoch": self._best_epoch, "best_metric": best_metric,
                    })
                except Exception as exc:  # noqa: BLE001 - reported, never silent
                    tqdm.write(f"[lean-train:{stage}] WARNING restore_best failed: {exc}")

        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not steps:
            return {"loss": float("nan"), "steps": 0.0}
        # `loss` = last epoch's slot-weighted mean (same definition as eval_loss).
        return {
            "loss": mean_loss,
            "loss_batchmean": batch_mean_loss,
            "steps": float(steps),
            "epochs_run": float(epoch),
            "stopped_early": bool(self._stopped_early),
            "best_epoch": float(self._best_epoch) if self._best_epoch is not None else float("nan"),
            "best_metric": float(best_metric) if best_metric is not None else float("nan"),
            "restored_best": restored,
        }

    def train(self, stages: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Run stages (default ('T2',)); returns summary dict."""
        order = tuple(stages or ("T2",))
        summary: Dict[str, Any] = {"history": self.history}
        for stage in order:
            if stage == "C":
                summary[stage] = {"status": "skipped", "reason": "lean_flow_no_stage_c"}
                continue
            split_map = {"T0": "t0", "T1": "t1", "T2": "t2"}
            data = self.load_rows(split_map.get(stage, "t2")) or self.load_rows("train")
            if not data:
                summary[stage] = {"status": "skipped", "reason": "no_data"}
                continue
            n_epochs = getattr(getattr(self.config, "optimization", None), "num_epochs", 3) if self.config else 3
            res = self.run_stage(stage, data, num_epochs=n_epochs)
            ev = self.evaluate("dev")
            summary[stage] = {**res, **ev, "status": "ok", "n": float(len(data))}
            self._log_event({"event": "stage_end", "stage": stage, **summary[stage]})
            if self.config is not None:
                out = Path(self.config.output_dir)
                cp = out / f"{self.config.experiment_name}_seed{self.config.seed}" / stage / "checkpoints"
                self.save_checkpoint(str(cp))
        return summary

    def _log_event(self, rec: Dict[str, Any]) -> None:
        if self.config is None:
            return
        try:
            d = Path(self.config.output_dir) / f"{self.config.experiment_name}_seed{self.config.seed}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "train_log.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **rec}, default=str) + "\n")
        except Exception:
            pass

    def save_checkpoint(self, output_dir: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model weights and provenance metadata."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        model_path = out_p / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        meta = {
            "flow": "lean",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "step": self.step,
            "lr_encoder": self.lr_encoder,
            "lr_head": self.lr_head,
            # T10: provenance must say WHICH model was trained. A LoRA checkpoint
            # is only loadable with peft present, and a reader of provenance.json
            # otherwise cannot tell the two arms of the ablation apart.
            "use_lora": bool(getattr(self, "lora_enabled", False)),
            "lora_param_count": int(getattr(self, "lora_param_count", 0)),
            # v2.1: the same rule for the H1 objective. The two arms produce
            # DIFFERENT weights from the same data, so a checkpoint that does not
            # name its objective can only be identified by guessing -- and the
            # resolver must be constructed with the matching value or it will
            # score the binary weights with a softmax.
            "tool_objective": str(
                getattr(self.loss_container, "tool_objective", "ce")
            ),
            "use_binary_tool_head": bool(getattr(self, "use_binary_tool_head", False)),
            **(metadata or {}),
        }
        with open(out_p / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return str(model_path)

    def load_checkpoint(self, path: str) -> None:
        """Restore model weights from a checkpoint directory or ``.pt`` file."""
        p = Path(path)
        state_path = (p / "model.pt") if p.is_dir() else p
        if not state_path.exists():
            raise FileNotFoundError(f"checkpoint weights not found: {state_path}")
        state = torch.load(state_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
