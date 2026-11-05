"""xLAM-60k dataset loader boilerplate."""

from __future__ import annotations

from typing import Iterator, Optional
from spancall.data.base import BaseDatasetLoader
from spancall.data.schema import UnifiedSample


class XLAMLoader(BaseDatasetLoader):
    """Loads and standardizes the Salesforce xLAM-60k dataset."""

    def __init__(self, data_path_or_hub_id: str) -> None:
        self.data_path_or_hub_id = data_path_or_hub_id

    def load(self, split: Optional[str] = None) -> Iterator[UnifiedSample]:
        raise NotImplementedError("Implement xLAM dataset parser into UnifiedSample")
