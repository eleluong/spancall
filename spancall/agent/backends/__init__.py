"""LLM backends and model adapters."""

from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.backends.litellm_adapter import UniversalAPIBackend
from spancall.agent.backends.local_adapter import LocalVLLMBackend

__all__ = [
    "BaseLLMBackend",
    "LocalVLLMBackend",
    "UniversalAPIBackend",
]
