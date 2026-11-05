"""τ-bench agentic multi-turn task loader."""

from __future__ import annotations

from typing import Iterator, Optional
from spancall.data.base import BaseDatasetLoader
from spancall.data.schema import UnifiedSample


class TauBenchLoader(BaseDatasetLoader):
    """Loads and standardizes τ-bench (airline, retail) multi-turn conversational tasks."""

    def __init__(self, domain: str = "retail") -> None:
        self.domain = domain

    def load(self, split: Optional[str] = None) -> Iterator[UnifiedSample]:
        raise NotImplementedError("Implement tau-bench task parser into UnifiedSample")
