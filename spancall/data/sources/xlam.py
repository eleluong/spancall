"""Salesforce xLAM-60k function-calling dataset loader.

Imports ``Salesforce/xlam-function-calling-60k`` from HuggingFace (or a local
JSON / JSONL file) and converts rows into :class:`UnifiedSample` records.

At the current stage (``single_call_only=True``, default), only single-call
rows (``len(answers) == 1``) are yielded so H1 and H3 receive unambiguous
single-call supervision without parallel same-tool anchor collisions.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from spancall.data.base import BaseDatasetLoader
from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.data.sources.hermes import _safe_json_loads

XLAM_HUB_ID = "Salesforce/xlam-function-calling-60k"


def map_xlam_type(raw_type: Optional[str]) -> Tuple[str, Optional[str], bool]:
    """Map an xLAM Python-style type string to ``(param_type, items_type, is_optional)``.

    Examples:
        ``"str"`` -> ``("string", None, False)``
        ``"int, optional"`` -> ``("integer", None, True)``
        ``"Optional[float]"`` -> ``("number", None, True)``
        ``"List[int]"`` -> ``("array", "integer", False)``
    """
    t = (raw_type or "string").strip()
    is_optional = False
    if t.endswith(", optional"):
        is_optional = True
        t = t[: -len(", optional")].strip()
    elif t.startswith("Optional[") and t.endswith("]"):
        is_optional = True
        t = t[len("Optional[") : -1].strip()

    tl = t.lower()
    if tl in ("str", "string"):
        return "string", None, is_optional
    if tl in ("int", "integer"):
        return "integer", None, is_optional
    if tl in ("float", "number", "double"):
        return "number", None, is_optional
    if tl in ("bool", "boolean"):
        return "boolean", None, is_optional
    if tl == "list" or tl.startswith("list[") or tl.startswith("tuple[") or tl == "set":
        items_type = "string"
        if "int" in tl:
            items_type = "integer"
        elif "float" in tl or "number" in tl:
            items_type = "number"
        elif "bool" in tl:
            items_type = "boolean"
        return "array", items_type, is_optional
    if tl == "dict" or tl.startswith("dict["):
        return "object", None, is_optional
    return "string", None, is_optional


def _parse_xlam_tools(raw_tools: Any) -> Optional[List[ToolDefinition]]:
    """Parse xLAM ``tools`` (JSON string or list of dicts) into ToolDefinition list."""
    parsed = _safe_json_loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
    if not isinstance(parsed, list):
        return None
    tools: List[ToolDefinition] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        params_raw = item.get("parameters") or {}
        param_schemas: List[ParameterSchema] = []
        if isinstance(params_raw, dict):
            for pname, pspec in params_raw.items():
                if not isinstance(pspec, dict):
                    continue
                ptype, items_type, is_opt = map_xlam_type(pspec.get("type"))
                has_default = "default" in pspec
                raw_enum = pspec.get("enum")
                enum = [str(v) for v in raw_enum] if isinstance(raw_enum, list) else None
                param_schemas.append(
                    ParameterSchema(
                        name=str(pname),
                        description=str(pspec.get("description") or ""),
                        param_type=ptype,
                        required=not (is_opt or has_default),
                        enum=enum,
                        default=pspec.get("default"),
                        items_type=items_type,
                    )
                )
        tools.append(
            ToolDefinition(
                name=str(name),
                description=str(item.get("description") or ""),
                parameters=param_schemas,
                risk_tier="read_only",
            )
        )
    return tools


def _parse_xlam_answers(raw_answers: Any) -> Optional[List[GroundTruthCall]]:
    """Parse xLAM ``answers`` (JSON string or list of dicts) into GroundTruthCall list."""
    parsed = _safe_json_loads(raw_answers) if isinstance(raw_answers, str) else raw_answers
    if not isinstance(parsed, list):
        return None
    calls: List[GroundTruthCall] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        args = item.get("arguments")
        if isinstance(args, str):
            args = _safe_json_loads(args) or {}
        if not isinstance(args, dict):
            args = {}
        calls.append(GroundTruthCall(tool_name=str(name), arguments=args))
    return calls


class XLAMLoader(BaseDatasetLoader):
    """Loads and standardizes ``Salesforce/xlam-function-calling-60k`` into UnifiedSample."""

    def __init__(
        self,
        data_path_or_hub_id: str = XLAM_HUB_ID,
        limit: Optional[int] = None,
        local_path: Optional[str] = None,
        single_call_only: bool = True,
        token: Optional[str] = None,
    ) -> None:
        self.data_path_or_hub_id = data_path_or_hub_id
        self.limit = limit
        self.local_path = local_path
        self.single_call_only = single_call_only
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
        self.stats: Dict[str, Any] = {
            "config": "xlam",
            "rows_seen": 0,
            "rows_yielded": 0,
            "dropped_parse_error": 0,
            "dropped_empty_gt": 0,
            "dropped_multi_call": 0,
            "dropped_empty_query": 0,
            "parsed_gt_calls": 0,
        }

    def _iter_raw_rows(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        if self.local_path:
            expanded = os.path.expanduser(self.local_path)
            if expanded.endswith(".jsonl"):
                with open(expanded, "r") as fh:
                    for i, line in enumerate(fh):
                        if self.limit and i >= self.limit:
                            break
                        line = line.strip()
                        if line:
                            yield json.loads(line)
            else:
                with open(expanded, "r") as fh:
                    data = json.load(fh)
                rows = data if isinstance(data, list) else list(data.values())
                for i, row in enumerate(rows):
                    if self.limit and i >= self.limit:
                        break
                    yield row
            return

        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "XLAMLoader hub mode requires the optional 'datasets' dependency. "
                "Install with `pip install 'datasets>=2.0.0'` or pass local_path=..."
            ) from exc

        ds = load_dataset(self.data_path_or_hub_id, split=split, token=self.token)
        for i, row in enumerate(ds):
            if self.limit and i >= self.limit:
                break
            yield dict(row)

    def load(self, split: Optional[str] = None) -> Iterator[UnifiedSample]:
        """Stream rows of xLAM-60k into UnifiedSample records."""
        for row in self._iter_raw_rows(split or "train"):
            self.stats["rows_seen"] += 1
            if not isinstance(row, dict):
                self.stats["dropped_parse_error"] += 1
                continue

            query = row.get("query")
            if not isinstance(query, str) or not query.strip():
                self.stats["dropped_empty_query"] += 1
                continue

            tools = _parse_xlam_tools(row.get("tools"))
            answers = _parse_xlam_answers(row.get("answers"))
            if tools is None or answers is None:
                self.stats["dropped_parse_error"] += 1
                continue

            if not answers:
                self.stats["dropped_empty_gt"] += 1
                continue

            if self.single_call_only and len(answers) != 1:
                self.stats["dropped_multi_call"] += 1
                continue

            rid = row.get("id")
            if rid is None:
                rid = f"row-{self.stats['rows_seen']}"

            self.stats["rows_yielded"] += 1
            self.stats["parsed_gt_calls"] += len(answers)

            yield UnifiedSample(
                sample_id=f"xlam::{rid}",
                source_dataset="xlam-60k",
                utterance=query,
                tools=tools,
                ground_truth_calls=answers,
                recent_context=[],
                stratum=Stratum.F,
                metadata={
                    "config": "xlam",
                    "n_calls": len(answers),
                    "n_tools": len(tools),
                },
            )
