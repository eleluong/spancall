"""Abstract base classes and protocols for data handling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional
from spancall.data.schema import UnifiedSample


class BaseDatasetLoader(ABC):
    """Abstract interface for ingesting raw tool calling datasets."""

    @abstractmethod
    def load(self, split: Optional[str] = None) -> Iterator[UnifiedSample]:
        """Stream or iterate through raw samples converted to UnifiedSample format."""
        raise NotImplementedError


class BaseAligner(ABC):
    """Abstract interface for reverse-aligning GT arguments to candidate spans/contexts."""

    @abstractmethod
    def align(self, sample: UnifiedSample) -> UnifiedSample:
        """Ground the GT arguments into token/char spans or context references.
        
        Tags the sample as Stratum.F (fully groundable) or Stratum.A (abstention negative).
        """
        raise NotImplementedError


class BaseAugmenter(ABC):
    """Abstract interface for dataset augmentation (perturbations, hard negatives, distractors)."""

    @abstractmethod
    def augment(self, sample: UnifiedSample) -> List[UnifiedSample]:
        """Generate augmented variants of the input sample."""
        raise NotImplementedError
