"""Model export and quantization interface (ONNX, TensorRT, INT8).

Ref: docs/mechanism.md §10.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseExporter(ABC):
    """Abstract model exporter for latency optimization and deployment."""

    @abstractmethod
    def export_onnx(
        self,
        output_path: str,
        opset_version: int = 17,
        quantize_int8: bool = False,
    ) -> str:
        """Export model to ONNX runtime format."""
        raise NotImplementedError
