"""Tool execution environments and sandbox definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from spancall.data.schema import ToolDefinition


@dataclass
class ToolExecutionResult:
    """Output of executing a tool in the environment."""
    success: bool
    output: Any
    error: Optional[str] = None
    state_changed: bool = False


class BaseToolEnvironment(ABC):
    """Stateful environment executing tool calls."""

    @abstractmethod
    def get_available_tools(self) -> List[ToolDefinition]:
        """List tools supported by this environment."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool call and update internal environment state."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset environment to initial state."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_task_success(self) -> bool:
        """Check whether the user's task goal was satisfied."""
        raise NotImplementedError

    def session_context(self) -> List[Dict[str, Any]]:
        """Recent agent working state exposed to the SpanCall fast path.

        Stateful environments override this to surface e.g. prior tool
        results as ``recent_context`` rows, letting the resolver disambiguate
        values already mentioned in the session. The default is empty.
        """
        return []
