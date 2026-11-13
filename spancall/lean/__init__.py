"""Lean SpanCall Package — Clean SpanCall without abstain machinery.

Provides:
- LeanPointerHead, LeanSpanCallModel (model.py)
- LeanLossContainer, compute_lean_pointer_loss (loss.py)
- LeanJointDecoder, LeanValueChoice (decode.py)
- LeanResolver, LeanResolveResult, LeanParameterResolution (resolver.py)
- tensorize_sample_lean, LeanBatchCollate (tensorize.py)
- LeanTrainer, build_lean_optimizer_groups (trainer.py)
- eval_param_extraction_lean, param_is_groundable, param_targets, classify_param,
  e3_aligner_violation, EXTRACTABLE_TARGET_TYPES, NON_EXTRACTABLE_TARGET_TYPES (eval_p1.py)
"""

from spancall.lean.decode import LeanJointDecoder, LeanValueChoice
from spancall.lean.eval_p1 import (
    EXTRACTABLE_TARGET_TYPES,
    NON_EXTRACTABLE_TARGET_TYPES,
    classify_param,
    e3_aligner_violation,
    eval_param_extraction_lean,
    param_is_groundable,
    param_targets,
)
from spancall.lean.loss import LeanLossContainer, compute_lean_pointer_loss
from spancall.lean.model import LeanPointerHead, LeanSpanCallModel
from spancall.lean.resolver import (
    LeanParameterResolution,
    LeanResolveResult,
    LeanResolver,
)
from spancall.lean.tensorize import (
    LeanBatchCollate,
    has_positive_targets,
    tensorize_sample_lean,
)
from spancall.lean.trainer import LeanTrainer, build_lean_optimizer_groups

__all__ = [
    "LeanPointerHead",
    "LeanSpanCallModel",
    "compute_lean_pointer_loss",
    "LeanLossContainer",
    "LeanJointDecoder",
    "LeanValueChoice",
    "LeanParameterResolution",
    "LeanResolveResult",
    "LeanResolver",
    "has_positive_targets",
    "tensorize_sample_lean",
    "LeanBatchCollate",
    "LeanTrainer",
    "build_lean_optimizer_groups",
    "eval_param_extraction_lean",
    "param_is_groundable",
    "param_targets",
    "classify_param",
    "e3_aligner_violation",
    "EXTRACTABLE_TARGET_TYPES",
    "NON_EXTRACTABLE_TARGET_TYPES",
]

