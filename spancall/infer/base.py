"""Version-agnostic inference boilerplate, ABCs, and shared utilities.

Defines the pluggable resolver contract (:class:`BaseResolver`,
:class:`BaseResolveResult`) that ``spancall/agent/`` and ``spancall/eval/``
depend on, so concrete versions (``LeanResolver`` or future v2.x/v3 resolvers)
can be swapped in without modifying the agent router or evaluation harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple

from spancall.data.schema import ToolDefinition


@dataclass
class ParameterResolution:
    """Resolved parameter detail."""

    name: str
    value: Any
    confidence: float
    margin: float
    source_type: str
    char_span: Optional[Tuple[int, int]] = None
    p_abstain: float = 0.0


@dataclass
class BaseResolveResult:
    """Version-agnostic end-to-end outcome of a fast-path resolver."""

    hit: bool
    tool_name: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    joint_margin: float = 0.0
    accept_score: float = 1.0
    abstain_reason: Optional[str] = None
    latency_ms: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    status: str = "abstain"  # "hit" | "partial" | "abstain"
    payload_partial: Dict[str, Any] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    confidence: float = 0.0


# Alias so existing agent/eval code can import ResolveResult from spancall.infer.base
ResolveResult = BaseResolveResult


class BaseResolver(ABC):
    """Abstract fast-path resolver contract consumed by ToolCallRouter."""

    @abstractmethod
    def try_resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[date] = None,
    ) -> BaseResolveResult:
        """Resolve a single candidate tool schema."""
        raise NotImplementedError

    @abstractmethod
    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[date] = None,
    ) -> BaseResolveResult:
        """Select and resolve among multiple candidate tool schemas."""
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, **kwargs: Any) -> "BaseResolver":
        """Optional factory hook for CLI/agent scripts."""
        raise NotImplementedError(f"{cls.__name__} does not implement from_checkpoint")


class NoOpResolver(BaseResolver):
    """Always-abstain resolver used when no checkpoint is loaded (B0/smoke fallback)."""

    def __init__(self, abstain_reason: str = "no_checkpoint") -> None:
        self.abstain_reason = abstain_reason

    def try_resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[date] = None,
    ) -> BaseResolveResult:
        return BaseResolveResult(hit=False, abstain_reason=self.abstain_reason)

    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[date] = None,
    ) -> BaseResolveResult:
        return BaseResolveResult(hit=False, abstain_reason=self.abstain_reason)


class ScriptedResolver(BaseResolver):
    """Deterministic resolver backed by a user-provided callback for tests/evals."""

    def __init__(
        self,
        fn: Callable[[List[ToolDefinition], str, Optional[List[Dict[str, Any]]], Optional[date]], BaseResolveResult],
    ) -> None:
        self._fn = fn

    def try_resolve(
        self,
        schema: ToolDefinition,
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[date] = None,
    ) -> BaseResolveResult:
        return self._fn([schema], utterance, recent_context, now)

    def try_resolve_candidates(
        self,
        candidate_schemas: List[ToolDefinition],
        utterance: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        now: Optional[date] = None,
    ) -> BaseResolveResult:
        return self._fn(candidate_schemas, utterance, recent_context, now)


class BaseJointDecoder(ABC):
    """Abstract joint assignment search across candidate spans."""

    @abstractmethod
    def decode(
        self,
        candidate_scores_per_param: Dict[str, List[Tuple[Any, float]]],
        entity_classes: Optional[Dict[str, str]] = None,
        top_k: int = 3,
    ) -> Tuple[Dict[str, Any], float]:
        """Find best non-colliding assignment and compute margin to runner-up."""
        raise NotImplementedError


def _groupkey(value: Any) -> str:
    """Hashable canonical key for grouping equal-valued candidates."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"num:{float(value)!r}"
    if isinstance(value, (list, tuple)):
        return f"seq:{value!r}"
    return f"any:{value!r}"


class FakeTokenizer:
    """Char-level tokenizer for smoke tests (no HF dependency).

    Maps each character to a single token so every legend entry boundary
    coincides with a token start/end.
    """

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        offsets = [(i, i + 1) for i in range(len(text))]
        out: Dict[str, Any] = {"input_ids": [0] * len(offsets)}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs: Any,
    ) -> List[int]:
        return [0] * len(text)


def _backbone_max_seq_len(
    encoder: Any = None,
    tokenizer: Any = None,
    fallback: int = 32768,
) -> int:
    """Derive the context window from the encoder/tokenizer, never a hardcoded 512."""
    if encoder is not None:
        cfg = getattr(encoder, "config", None)
        if cfg is not None:
            mpe = getattr(cfg, "max_position_embeddings", None)
            if mpe:
                return int(mpe)
        ml = getattr(encoder, "model_max_length", None)
        if ml:
            return int(ml)
    if tokenizer is not None:
        ml = getattr(tokenizer, "model_max_length", None)
        if ml and int(ml) > 0:
            return int(ml)
    return fallback

