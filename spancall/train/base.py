"""Version-agnostic training & calibration boilerplate (ABCs and LR schedules)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch.optim.lr_scheduler import LambdaLR

from spancall.train.config import CalibrationConfig, TrainingConfig


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


class BaseCalibrator(ABC):
    """Abstract interface for calibrating model logits and gating thresholds."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config

    @abstractmethod
    def fit_temperature(self, val_logits: Any, val_labels: Any) -> float:
        """Learn optimal temperature scaling parameter via NLL minimization."""
        raise NotImplementedError

    @abstractmethod
    def sweep_thresholds(
        self,
        calib_predictions: List[Dict[str, Any]],
        calib_ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Find optimal gating thresholds guaranteeing target precision."""
        raise NotImplementedError

    @abstractmethod
    def clopper_pearson_lower_bound(
        self,
        k_successes: int,
        n_trials: int,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Clopper-Pearson exact binomial lower confidence bound."""
        raise NotImplementedError


class CosineWarmupSchedule:
    """Linear warmup then cosine/linear/constant decay with configurable floor."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        kind: str = "cosine",
    ) -> None:
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)
        self.min_lr_ratio = min(max(float(min_lr_ratio), 0.0), 1.0)
        self.kind = str(kind or "cosine").lower()

        def lr_lambda(step: int) -> float:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return float(step + 1) / float(self.warmup_steps)
            if self.total_steps <= self.warmup_steps:
                return 1.0
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            if self.kind == "constant":
                base = 1.0
            elif self.kind == "linear":
                base = 1.0 - progress
            else:
                base = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * base

        self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(self) -> None:
        self.scheduler.step()

