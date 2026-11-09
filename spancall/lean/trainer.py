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
    convert_legacy_batch_to_lean,
    convert_legacy_row_to_lean,
    has_positive_targets,
    is_legacy_batch,
)
from spancall.train.trainer import CosineWarmupSchedule


def build_lean_optimizer_groups(
    model: nn.Module,
    lr_encoder: float = 2e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 0.01,
    layer_decay: float = 0.85,
) -> List[Dict[str, Any]]:
    """Create optimizer parameter groups with layer-wise decay for the backbone."""
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


class LeanTrainer:
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
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.loss_container = loss_container or LeanLossContainer()

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
        )
        self.optimizer: torch.optim.Optimizer = AdamW(groups)
        self.scheduler: Optional[CosineWarmupSchedule] = None
        self.step = 0
        self.global_step = 0
        self._accum_i = 0
        self.stage = "T2"
        self.history: Dict[str, Any] = {"train_loss": [], "dev": {}, "epoch_eval": []}
        self._rng = random.Random(int(getattr(config, "seed", 42) or 42))
        self.collator = LeanBatchCollate()

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
        if is_legacy_batch(batch):
            batch = convert_legacy_batch_to_lean(batch)
        b = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        accum_steps = max(int(getattr(self.config, "gradient_accumulation_steps", 1) or 1), 1)
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
                targs["pointer_positives"] = b["pos_indices"]

            if "tool_positions" in b and b["tool_positions"].numel():
                safe_tool_pos = b["tool_positions"].clamp(min=0, max=max_tok_idx)
                tool_h = H.gather(1, safe_tool_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
                tl = self.model.tool_head(tool_h, cls_h)
                tl = tl.masked_fill(b["tool_positions"] < 0, -1e4)
                preds["tool_logits"] = tl
                targs["tool_targets"] = b.get(
                    "tool_targets",
                    torch.zeros(H.size(0), dtype=torch.long, device=self.device),
                )

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

        out_metrics: Dict[str, float] = {"loss": float(total_loss.detach().item())}
        if "pointer" in loss_out:
            out_metrics["pointer_loss"] = float(loss_out["pointer"].item())
            out_metrics["pointer"] = out_metrics["pointer_loss"]
        if "tool" in loss_out:
            out_metrics["tool_loss"] = float(loss_out["tool"].item())
            out_metrics["tool"] = out_metrics["tool_loss"]

        if "span_logits" in fp32_preds and "pos_indices" in b:
            span_acc, _, _ = self.exact_span_match(fp32_preds["span_logits"], b["pos_indices"])
            out_metrics["span_acc"] = float(span_acc.item())

        if "tool_logits" in fp32_preds and "tool_targets" in targs:
            t_correct = (fp32_preds["tool_logits"].argmax(-1) == targs["tool_targets"]).float().mean()
            out_metrics["tool_acc"] = float(t_correct.item())

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
        """Evaluate either on a split name (e.g. 'dev') or a sequence of batch dicts."""
        self.model.eval()

        if isinstance(split_or_batches, str):
            rows = self.load_rows(split_or_batches)
            if not rows:
                return {
                    "eval_loss": 0.0,
                    "eval_pointer_loss": 0.0,
                    "eval_span_acc": 0.0,
                    "span_acc": 0.0,
                    "exact_payload_match": 0.0,
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

        total_loss = 0.0
        total_ptr_loss = 0.0
        n_batches = 0
        n_span_batches = 0
        span_hits = 0.0
        n_span = 0
        tool_hits = 0
        n_tool = 0

        with torch.no_grad():
            for raw_batch in batches:
                batch = convert_legacy_batch_to_lean(raw_batch) if is_legacy_batch(raw_batch) else raw_batch
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
                        targs["pointer_positives"] = b["pos_indices"]

                    if "tool_positions" in b and b["tool_positions"].numel():
                        safe_tool_pos = b["tool_positions"].clamp(min=0, max=max_tok_idx)
                        tool_h = H.gather(1, safe_tool_pos.unsqueeze(-1).expand(-1, -1, H.size(-1)))
                        tl = self.model.tool_head(tool_h, cls_h)
                        tl = tl.masked_fill(b["tool_positions"] < 0, -1e4)
                        preds["tool_logits"] = tl
                        targs["tool_targets"] = b.get(
                            "tool_targets",
                            torch.zeros(H.size(0), dtype=torch.long, device=self.device),
                        )

                fp32_preds = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in preds.items()}
                loss_out = self.loss_container(fp32_preds, targs)
                total_loss += float(loss_out["loss"].item())
                if "pointer" in loss_out:
                    total_ptr_loss += float(loss_out["pointer"].item())

                if "span_logits" in fp32_preds and "pos_indices" in b:
                    _, acc_t, valid = self.exact_span_match(fp32_preds["span_logits"], b["pos_indices"])
                    span_hits += float((acc_t * valid).sum().item())
                    n_span += int(valid.sum().item())
                    n_span_batches += 1

                if "tool_logits" in fp32_preds and "tool_targets" in b:
                    pred_tools = fp32_preds["tool_logits"].argmax(-1)
                    tt = b["tool_targets"]
                    valid_tool = tt >= 0
                    if valid_tool.any():
                        tool_hits += int(((pred_tools == tt) & valid_tool).sum().item())
                        n_tool += int(valid_tool.sum().item())

                n_batches += 1

        if n_batches == 0:
            return {
                "eval_loss": 0.0,
                "eval_pointer_loss": 0.0,
                "eval_span_acc": 0.0,
                "span_acc": 0.0,
                "exact_payload_match": 0.0,
                "n": 0.0,
            }

        overall_span_acc = span_hits / max(n_span, 1)
        res: Dict[str, float] = {
            "eval_loss": total_loss / n_batches,
            "eval_pointer_loss": total_ptr_loss / max(n_span_batches, 1),
            "eval_span_acc": overall_span_acc,
            "span_acc": overall_span_acc,
            "exact_payload_match": overall_span_acc,
            "n": float(n_span),
            "n_span": float(n_span),
        }
        if n_tool > 0:
            res["tool_acc"] = tool_hits / n_tool
            res["n_tool"] = float(n_tool)

        if isinstance(split_or_batches, str):
            self._log_event({"event": "eval", "split": split_label, "stage": self.stage, "step": self.global_step, **res})
            tool_str = f" tool_acc={res['tool_acc']:.4f} (n={n_tool})" if n_tool > 0 else ""
            print(
                f"[lean-eval:{split_label}] span_acc={overall_span_acc:.4f} (n={n_span}) "
                f"loss={res['eval_loss']:.4f}{tool_str}",
                flush=True,
            )
        return res

    # ------------------------------------------------------------------ #
    # Dataset / Stage Helpers (compatible with SpanCallTrainer workflow)
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

        rows = [convert_legacy_row_to_lean(r) if is_legacy_batch(r) else r for r in rows]

        if filter_empty_positives and rows and "pos_indices" in rows[0]:
            groundable = [r for r in rows if has_positive_targets(r)]
            if groundable:
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
                    t = t.reshape(0, 0).long()
                d[k] = t
            out.append(d)
        return out

    def collate(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        collated = self.collator(batch)
        return convert_legacy_batch_to_lean(collated) if is_legacy_batch(collated) else collated

    def make_batches(
        self,
        rows: Sequence[Dict[str, Any]],
        batch_size: int,
        shuffle: bool = False,
        rng: Optional[Any] = None,
        max_tokens_per_batch: Optional[int] = None,
    ) -> Iterable[Sequence[Dict[str, Any]]]:
        bs = max(int(batch_size), 1)
        cfg_cap = getattr(self.config, "max_tokens_per_batch", None) if self.config else None
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
        accum = max(int(getattr(self.config, "gradient_accumulation_steps", 1) or 1), 1)
        steps_per_epoch = max(len(list(self.make_batches(rows, bs, shuffle=False))), 1)
        total_steps = steps_per_epoch * num_epochs
        opt_total_steps = max((total_steps + accum - 1) // accum, 1)
        warmup_ratio = float(getattr(getattr(self.config, "optimization", None), "warmup_ratio", 0.06) or 0.06)
        self.scheduler = CosineWarmupSchedule(self.optimizer, max(int(opt_total_steps * warmup_ratio), 1), opt_total_steps)

        log_every = max(total_steps // 100, 1)
        t_start = time.time()
        self._accum_i = 0
        steps = 0
        print(
            f"[lean-train:{stage}] start: rows={len(rows)} epochs={num_epochs} "
            f"steps/epoch={steps_per_epoch} total_steps={total_steps} bs={bs}",
            flush=True,
        )
        self._log_event({"event": "stage_start", "stage": stage, "rows": len(rows), "epochs": num_epochs, "total_steps": total_steps})

        pbar = tqdm(total=total_steps, desc=f"lean-train[{stage}]", dynamic_ncols=True, leave=True)
        for epoch in range(1, num_epochs + 1):
            ep_sum, ep_n = 0.0, 0
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

            mean_loss = ep_sum / max(ep_n, 1)
            tqdm.write(f"[lean-train:{stage}] epoch {epoch}/{num_epochs} done: mean_loss={mean_loss:.4f}")
            self._log_event({"event": "epoch_end", "stage": stage, "epoch": epoch, "mean_loss": mean_loss})

            eval_splits = getattr(self.config, "eval_splits", None) or []
            if isinstance(eval_splits, str):
                eval_splits = [eval_splits]
            eval_limit = getattr(self.config, "eval_limit", 500) if self.config else 500
            for es in eval_splits:
                ev_res = self.evaluate(es, limit=eval_limit)
                ep_rec = {
                    "stage": stage,
                    "epoch": epoch,
                    "split": es,
                    "span_acc": float(ev_res.get("span_acc", 0.0)),
                    "exact_payload_match": float(ev_res.get("exact_payload_match", 0.0)),
                    "abstain_acc": 0.0,
                    "n": float(ev_res.get("n", 0.0)),
                }
                self.history["epoch_eval"].append(ep_rec)
                self._log_event({"event": "epoch_eval", **ep_rec})

                cur_metric = float(ev_res.get("exact_payload_match", 0.0))
                if not hasattr(self, "_best_curve_metric") or cur_metric > self._best_curve_metric:
                    self._best_curve_metric = cur_metric
                    if self.config is not None:
                        out = Path(self.config.output_dir)
                        best_cp = out / f"{self.config.experiment_name}_seed{self.config.seed}" / "best_checkpoint"
                        self.save_checkpoint(str(best_cp))
                        print(f"[lean-train:{stage}:epoch {epoch}] NEW BEST on {es}: span_acc={cur_metric:.4f} -> saved to {best_cp}", flush=True)

        pbar.close()
        if self._accum_i > 0:
            self._optimizer_step()
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"loss": ep_sum / max(ep_n, 1), "steps": float(steps)} if steps else {"loss": float("nan"), "steps": 0.0}

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
            **(metadata or {}),
        }
        with open(out_p / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return str(model_path)
