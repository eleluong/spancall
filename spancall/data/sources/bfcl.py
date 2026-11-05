"""Berkeley Function Calling Leaderboard (BFCL) evaluation dataset loader."""

from __future__ import annotations

from typing import Iterator, Optional
from spancall.data.base import BaseDatasetLoader
from spancall.data.schema import UnifiedSample


class BFCLLoader(BaseDatasetLoader):
    """Loads and standardizes BFCL test categories (simple, multiple, parallel, irrelevance)."""

    def __init__(self, data_path_or_hub_id: str, category: Optional[str] = None) -> None:
        self.data_path_or_hub_id = data_path_or_hub_id
        self.category = category

    def load(self, split: Optional[str] = None) -> Iterator[UnifiedSample]:
        raise NotImplementedError("Implement BFCL dataset parser into UnifiedSample")
