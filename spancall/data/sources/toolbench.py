"""ToolBench dataset loader boilerplate."""

from __future__ import annotations

from typing import Iterator, Optional
from spancall.data.base import BaseDatasetLoader
from spancall.data.schema import UnifiedSample


class ToolBenchLoader(BaseDatasetLoader):
    """Loads and standardizes the ToolBench dataset."""

    def __init__(self, data_path_or_hub_id: str) -> None:
        self.data_path_or_hub_id = data_path_or_hub_id

    def load(self, split: Optional[str] = None) -> Iterator[UnifiedSample]:
        raise NotImplementedError("Implement ToolBench dataset parser into UnifiedSample")
