"""Training config and shareable trainer/calibrator boilerplate."""

from spancall.train.base import BaseCalibrator, BaseTrainer, CosineWarmupSchedule
from spancall.train.config import (
    AgentConfig,
    CalibrationConfig,
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    RecentConfig,
    TrainingConfig,
)

__all__ = [
    "AgentConfig",
    "BaseCalibrator",
    "BaseTrainer",
    "CalibrationConfig",
    "CosineWarmupSchedule",
    "DataConfig",
    "ModelConfig",
    "OptimizationConfig",
    "RecentConfig",
    "TrainingConfig",
]
