"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from spancall.data.schema import ToolDefinition


class BaseLLMBackend(ABC):
    """Abstract interface for LLM completion and tool calling."""

    @abstractmethod
    def generate_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[ToolDefinition],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate a tool call using the model's native chat template and decoding.
        
        Returns:
            Dict containing tool_name, arguments, usage stats (tokens), latency_ms.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        """Standard conversational text generation."""
        raise NotImplementedError
