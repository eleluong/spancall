"""Data processing subpackage for SpanCall."""

from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    SpanTarget,
    Split,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.data.base import BaseAligner, BaseAugmenter, BaseDatasetLoader
from spancall.data.align import ReverseAligner
from spancall.data.augment import DataAugmenter
from spancall.data.mix import DatasetMixer
from spancall.data.partition import (
    gt_tool_key,
    partition_schema_disjoint,
    partition_schema_groups,
    write_splits_jsonl,
)

__all__ = [
    "BaseAligner",
    "BaseAugmenter",
    "BaseDatasetLoader",
    "DataAugmenter",
    "DatasetMixer",
    "GroundTruthCall",
    "ParameterSchema",
    "ReverseAligner",
    "SpanTarget",
    "Split",
    "Stratum",
    "ToolDefinition",
    "UnifiedSample",
    "gt_tool_key",
    "partition_schema_disjoint",
    "partition_schema_groups",
    "write_splits_jsonl",
]
