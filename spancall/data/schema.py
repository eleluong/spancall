"""Canonical data schemas for SpanCall datasets and pipeline processing.

Defines schemas without hardcoding any domain-specific data or values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Type


class Stratum(str, Enum):
    """Data stratum classification per docs/mechanism.md §0."""
    F = "F"  # Fully specified: all arguments groundable in utterance/context
    A = "A"  # Absent/unresolvable: at least one argument missing, implicit, or requiring reasoning


class Split(str, Enum):
    """Dataset partition types for unbiased research evaluation."""
    TRAIN = "train"
    DEV = "dev"
    CALIB = "calib"                     # Paraphrase-disjoint calibration set
    TEST_ID = "test_id"                 # In-distribution test
    TEST_OOD_SCHEMA = "test_ood_schema" # Zero-shot unseen tool schemas
    TEST_OOD_PARA = "test_ood_para"     # Unseen phrasing / paraphrases
    TEST_ADVERSARIAL = "test_adversarial"


def _enum_of(enum_type: Type[Enum], value: Any) -> Any:
    """Coerce a stored string/Enum into the enum ``enum_type`` (identity ok)."""
    if isinstance(value, enum_type):
        return value
    return enum_type(str(value))


@dataclass
class SpanTarget:
    """Represents a ground truth target for a parameter."""
    param_name: str
    target_type: str  # "utterance_span", "recent_ctx", "time_ctx", "enum_ctx", "boolean_ctx", "unresolved"
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    ctx_key: Optional[str] = None
    canonical_value: Optional[Any] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SpanTarget":
        return cls(
            param_name=str(d.get("param_name", "")),
            target_type=str(d.get("target_type", "unresolved")),
            char_start=d.get("char_start"),
            char_end=d.get("char_end"),
            token_start=d.get("token_start"),
            token_end=d.get("token_end"),
            ctx_key=d.get("ctx_key"),
            canonical_value=d.get("canonical_value"),
        )


@dataclass
class GroundTruthCall:
    """Ground truth tool call representation."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GroundTruthCall":
        return cls(
            tool_name=str(d.get("tool_name", "")),
            arguments=dict(d.get("arguments") or {}),
            call_id=d.get("call_id"),
        )


@dataclass
class ParameterSchema:
    """Specification of a tool parameter."""
    name: str
    description: str
    param_type: str  # "string", "integer", "number", "boolean", "array", "object"
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    items_type: Optional[str] = None
    entity_class: Optional[str] = None  # Group for exclusivity constraints

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ParameterSchema":
        return cls(
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            param_type=str(d.get("param_type", "string")),
            required=bool(d.get("required", True)),
            enum=list(d["enum"]) if d.get("enum") is not None else None,
            default=d.get("default"),
            items_type=d.get("items_type"),
            entity_class=d.get("entity_class"),
        )


@dataclass
class ToolDefinition:
    """Specification of an available tool."""
    name: str
    description: str
    parameters: List[ParameterSchema] = field(default_factory=list)
    risk_tier: str = "read_only"  # "read_only", "idempotent", "side_effecting"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ToolDefinition":
        return cls(
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            parameters=[ParameterSchema.from_dict(p) for p in (d.get("parameters") or [])],
            risk_tier=str(d.get("risk_tier", "read_only")),
        )


@dataclass
class UnifiedSample:
    """Single canonical record across all ingestion sources."""
    sample_id: str
    source_dataset: str
    utterance: str
    tools: List[ToolDefinition]
    ground_truth_calls: List[GroundTruthCall]
    recent_context: List[Dict[str, Any]] = field(default_factory=list)
    current_time_iso: Optional[str] = None
    stratum: Stratum = Stratum.F
    # Keys are ``f"{call_index}::{param_name}"`` for multi-call alignment, plus
    # the legacy bare ``param_name`` for single-call samples (see
    # ``ReverseAligner._emit``). Never assume a bare key in new code -- go
    # through ``targets_for_call``.
    target_spans: Dict[str, List[SpanTarget]] = field(default_factory=dict)
    split: Split = Split.TRAIN
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_calls(self) -> int:
        """Number of ground-truth calls this sample asks the model to emit (R0.4)."""
        return len(self.ground_truth_calls or [])

    def targets_for_call(self, call_index: int) -> Dict[str, List[SpanTarget]]:
        """Ground-truth targets owned by ONE call, keyed by bare param name.

        This is the migration accessor for R0.4: call-aware alignment stores
        spans under ``f"{call_index}::{param}"`` so two calls declaring the same
        parameter name no longer collide. Readers that have not migrated still
        find data for the overwhelmingly common single-call case, because the
        bare key is (a) written by ``ReverseAligner._emit`` for single-call
        samples and (b) used as a fallback below.

        The fallback is deliberately restricted to ``n_calls == 1``: in a legacy
        multi-call file the bare key is a *flattened collision* of every call, so
        handing it to ``call_index=1`` would fabricate supervision out of
        another call's arguments. An empty dict is the honest answer there.
        """
        out: Dict[str, List[SpanTarget]] = {}
        prefix = f"{int(call_index)}::"
        for key, spans in (self.target_spans or {}).items():
            if key.startswith(prefix):
                out[key[len(prefix):]] = list(spans)
        if not out and self.n_calls == 1:
            for key, spans in (self.target_spans or {}).items():
                if "::" not in key:
                    out[key] = list(spans)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "UnifiedSample":
        """Deserialize a canonical split JSON record (JSONL line) into the dataclass."""
        return cls(
            sample_id=str(d.get("sample_id", "")),
            source_dataset=str(d.get("source_dataset", "")),
            utterance=str(d.get("utterance", "")),
            tools=[ToolDefinition.from_dict(t) for t in (d.get("tools") or [])],
            ground_truth_calls=[
                GroundTruthCall.from_dict(c) for c in (d.get("ground_truth_calls") or [])
            ],
            recent_context=[dict(r) for r in (d.get("recent_context") or [])],
            current_time_iso=d.get("current_time_iso"),
            stratum=_enum_of(Stratum, d.get("stratum") or Stratum.F.value),
            target_spans={
                str(k): [SpanTarget.from_dict(t) for t in (v or [])]
                for k, v in (d.get("target_spans") or {}).items()
            },
            split=_enum_of(Split, d.get("split") or Split.TRAIN.value),
            metadata=dict(d.get("metadata") or {}),
        )
