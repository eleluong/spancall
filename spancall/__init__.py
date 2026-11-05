"""SpanCall: Fast-path speculative tool calling via multi-span extraction.

Core modules:
- data: Data ingestion, reverse alignment, augmentation, and split management.
- model: Encoder backbone, multi-span pointer heads, candidate masks, and losses.
- train: Training loop, multi-task optimization, and confidence calibration.
- infer: One-pass inference, joint decoding, casting, and gating.
- agent: Router, agent loop, tool execution environments, and LLM backends.
- eval: Tool-call evaluation, agentic evaluation, and comparative benchmarking.
"""

__version__ = "0.1.0"
