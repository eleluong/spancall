"""Training and calibration subpackage."""

from spancall.train.config import (
    CalibrationConfig,
    DataConfig,
    GateConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from spancall.train.trainer import (
    BaseTrainer,
    CosineWarmupSchedule,
    SpanCallTrainer,
    build_parameter_groups,
)
from spancall.train.calibrate import BaseCalibrator

__all__ = [
    "BaseCalibrator",
    "BaseTrainer",
    "CalibrationConfig",
    "CosineWarmupSchedule",
    "DataConfig",
    "GateConfig",
    "ModelConfig",
    "OptimizationConfig",
    "SpanCallTrainer",
    "TrainingConfig",
    "build_parameter_groups",
]
