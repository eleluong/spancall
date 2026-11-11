"""Hermes Function-Calling V1 dataset loader.

Imports ``NousResearch/hermes-function-calling-v1`` from HuggingFace (or a local
JSONL file) and converts every row into a :class:`UnifiedSample`. All **five**
dataset configs ("setups") are supported with one loader — the per-config
differences (tool sources, multi-turn recent context, json-mode rows) are
handled by small branches inside :meth:`HermesLoader.load`.

Strategy per row (verified against the hub dataset 2026-10):

- ``func_calling_singleturn`` (default): ``tools`` is a JSON string of OpenAI
  function schemas; GT calls are ``<tool_call>{...}</tool_call>`` blocks in the
  final ``gpt`` turn.
- ``func_calling`` (multi-turn): same as above; all ``human``/``gpt`` pairs
  *before* the last pair become ``recent_context``.
- ``glaive_func_calling``: Glaive-style rows — a single-tool system block plus a
  ``gpt`` turn carrying ``{"function": ..., "arguments": {...}}``.
- ``json_mode_agentic`` / ``json_mode_singleturn``: no OpenAI tools; a
  ``schema`` field (JSON Schema string) or a synthetic ``json_output`` tool is
  used, and the GT is the bare JSON object in the final ``gpt`` turn. These
  rows are tagged ``metadata["mode"] == "json"`` and are *always processed*,
  never silently dropped.

Design notes (docs/mechanism.md §8.1, plans/hermes_function_calling_v1.md):

- ``sample_id`` = ``f"{config}::{id}"`` (collision-safe across configs).
- Rows that cannot be parsed are **counted** (returned in
  :attr:`HermesLoader.stats`) rather than dropped or raised, so downstream
  telemetry always sees the real F/A and skip share per config.
- JSON-mode rows get a synthetic ``json_output`` :class:`ToolDefinition` and
  :class:`GroundTruthCall` so they flow through the same ``UnifiedSample``
  contract as function-calling rows.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterator, List, Optional

from spancall.data.base import BaseDatasetLoader
from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)

HERMES_HUB_ID = "NousResearch/hermes-function-calling-v1"

# All five "setups" of the Hermes FC V1 dataset.
HERMES_CONFIGS = [
    "func_calling_singleturn",
    "func_calling",
    "glaive_func_calling",
    "json_mode_agentic",
    "json_mode_singleturn",
]

_JSON_MODE_CONFIGS = frozenset({"json_mode_agentic", "json_mode_singleturn"})

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_TOOLS_BLOCK_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL | re.IGNORECASE)
_SCHEMA_BLOCK_RE = re.compile(r"<schema>\s*(.*?)\s*</schema>", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences if present around a JSON payload."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _extract_brace_object(text: str, start: int) -> Optional[str]:
    """Extract the balanced `{...}` JSON object starting at index ``start``.

    Handles **nested braces inside string values** (e.g. ``"error_logs":
    "[2023-...]\\n..."`` — the exact shape that broke the naive
    ``<tool_call>\\s*(\\{.*?\\})`` regex). Uses a char-by-char scan with a
    brace-depth counter that ignores braces inside quoted strings and handles
    backslash escapes.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_str = False
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2  # skip escaped char (including \")
                continue
            if c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None  # unbalanced (no closing brace)


def _iter_tool_call_payloads(text: str) -> Iterator[Dict[str, Any]]:
    """Yield the parsed dict payload of every ``<tool_call>`` block in ``text``.

    Uses brace-matching (see :func:`_extract_brace_object`) so nested braces and
    literal ``\\n`` escapes inside string values—both present in the real hub
    rows—don't truncate the body (the naive ``\\{.*?\\}`` regex does).
    """
    for m in _TOOL_CALL_OPEN_RE.finditer(text):
        start = text.find("{", m.end())
        body = _extract_brace_object(text, start) if start != -1 else None
        if not body:
            continue
        parsed = _safe_json_loads(body)
        if isinstance(parsed, dict):
            yield parsed


def _safe_json_loads(text: str) -> Optional[Any]:
    """Parse JSON leniently: strip fences, tolerate null, and fall back to a
    Python-literal parse for rows that use single-quoted strings.

    Real hub rows (e.g. the deep-research subset) ship payloads like
    ``{'queries': ['How do you define "alternative methods" ...']}``, which is
    valid Python but not valid JSON — ``json.loads`` alone would drop them.
    """
    if not text:
        return None
    cleaned = _strip_code_fences(text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # single-quoted (Python literal) payloads
    try:
        value = ast.literal_eval(cleaned)
        if isinstance(value, (dict, list)):
            return value
    except (ValueError, SyntaxError, TypeError):
        pass
    # outermost JSON object as a last resort (greedy across nested braces)
    match = _JSON_OBJECT_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _parse_tools_json(raw: Optional[str]) -> List[ToolDefinition]:
    """Convert an OpenAI JSON-string of tools (``[{...}]`` or ``{...}``) to ToolDefinition."""
    if not raw:
        return []
    parsed = _safe_json_loads(raw)
    if isinstance(parsed, dict):
        # single tool object
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    tools: List[ToolDefinition] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        fn = item.get("function", item)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        tools.append(_tool_from_json_schema(name, fn.get("description", ""), params))
    return tools


# Fallback name for the synthetic tool a json-mode row is supervised against.
JSON_MODE_TOOL_NAME = "json_output"


def _slugify_tool_name(raw: Optional[str]) -> str:
    """Coerce a schema ``title`` into a single token-safe tool name.

    The render/parse contract limits a tool name to one whitespace-free token
    (``<|tool|>\\s+([^\\s(]+)`` in ``span_template/parser.py``), so a title like
    ``"Flight Booking"`` cannot be used verbatim: the parser would record
    ``Flight`` while the GT recorded something else, and every H1 row would read
    as a miss for a reason that has nothing to do with the model.
    """
    if not raw:
        return JSON_MODE_TOOL_NAME
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in str(raw).strip())
    cleaned = cleaned.strip("_")
    return cleaned or JSON_MODE_TOOL_NAME


def _json_mode_tool_name(parsed_schema: Optional[Dict[str, Any]]) -> str:
    """The ONE name used for both the json-mode tool AND its synthetic GT call.

    T7: these used to disagree. ``load`` built the tool from
    ``parsed_schema["title"]`` while ``_build_sample`` hardcoded the GT to
    ``json_output``, so the GT tool was never among ``sample.tools``: Part-2 then
    "augmented" every row with a distractor and reported all 500 GT names as
    ``json_output`` — an alias defect, not model behaviour.
    """
    if not isinstance(parsed_schema, dict):
        return JSON_MODE_TOOL_NAME
    return _slugify_tool_name(parsed_schema.get("title"))


def _tool_from_json_schema(
    name: str, description: str, params: Dict[str, Any]
) -> ToolDefinition:
    """Build a ToolDefinition from a JSON-Schema ``parameters`` object."""
    properties = params.get("properties") or {}
    required = set(params.get("required") or [])
    param_schemas: List[ParameterSchema] = []
    for pname, pspec in properties.items():
        if not isinstance(pspec, dict):
            continue
        ptype = pspec.get("type", "string")
        # item type for array params
        items_type: Optional[str] = None
        if ptype == "array":
            items = pspec.get("items")
            if isinstance(items, dict):
                items_type = items.get("type")
            elif isinstance(items, str):
                items_type = items
        param_schemas.append(
            ParameterSchema(
                name=pname,
                description=pspec.get("description", ""),
                param_type=ptype,
                required=pname in required,
                enum=pspec.get("enum"),
                default=pspec.get("default"),
                items_type=items_type,
            )
        )
    tool_type = str(params.get("type", "object"))
    return ToolDefinition(name=name, description=description, parameters=param_schemas, risk_tier="read_only")


def _parse_schema_block(system_value: str) -> Optional[str]:
    """Extract a JSON-Schema string from ``<schema> ... </schema>`` (json-mode)."""
    match = _SCHEMA_BLOCK_RE.search(system_value)
    if not match:
        match = _TOOLS_BLOCK_RE.search(system_value)
    return match.group(1).strip() if match else None


class HermesLoader(BaseDatasetLoader):
    """Loads all five ``NousResearch/hermes-function-calling-v1`` configs into UnifiedSample."""

    def __init__(
        self,
        data_path_or_hub_id: str = HERMES_HUB_ID,
        config_name: str = "func_calling_singleturn",
        limit: Optional[int] = None,
        local_path: Optional[str] = None,
    ) -> None:
        self.data_path_or_hub_id = data_path_or_hub_id
        self.config_name = config_name
        self.limit = limit
        self.local_path = local_path
        if config_name not in HERMES_CONFIGS:
            raise ValueError(
                f"Unknown hermes config {config_name!r}; expected one of {HERMES_CONFIGS}"
            )
        self.stats: Dict[str, Any] = {
            "config": config_name,
            "rows_seen": 0,
            "rows_yielded": 0,
            "dropped_no_conversation": 0,
            "dropped_no_gt": 0,
            "dropped_no_human_turn": 0,
            "parsed_gt_calls": 0,
            "multi_call_rows": 0,
            "json_mode_rows": 0,
        }

    # ------------------------------------------------------------------ fetch
    def _iter_raw_rows(self) -> Iterator[Dict[str, Any]]:
        """Yield raw hub rows; prefers a local JSONL/JSON path, else ``datasets``."""
        if self.local_path:
            yield from self._iter_local_rows(self.local_path)
            return
        yield from self._iter_hub_rows()

    def _iter_hub_rows(self) -> Iterator[Dict[str, Any]]:
        """Stream rows from the hub via the optional ``datasets`` library."""
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "HermesLoader hub mode requires the optional 'datasets' dependency. "
                "Install with `pip install 'datasets>=2.0.0'` or pass local_path=..."
            ) from exc
        ds = load_dataset(self.data_path_or_hub_id, self.config_name, split="train")
        for i, row in enumerate(ds):
            if self.limit and i >= self.limit:
                break
            yield dict(row)

    def _iter_local_rows(self, path: str) -> Iterator[Dict[str, Any]]:
        """Read a local JSONL (``.jsonl``), JSON-array (``.json``), or a
        directory of ``<config_name>.jsonl`` files.
        """
        import os

        expanded = os.path.expanduser(path)
        if os.path.isdir(expanded):
            # directory: resolve this loader's config file inside it
            candidate = os.path.join(expanded, f"{self.config_name}.jsonl")
            if not os.path.isfile(candidate):
                # fall back to any single *.jsonl in the dir (single-config usage)
                jsonl_files = sorted(f for f in os.listdir(expanded) if f.endswith(".jsonl"))
                if not jsonl_files:
                    raise FileNotFoundError(
                        f"No .jsonl files found in local_path directory {expanded!r}"
                    )
                candidate = os.path.join(expanded, jsonl_files[0])
            yield from self._iter_local_rows(candidate)
            return

        if expanded.endswith(".jsonl"):
            with open(expanded, "r") as fh:
                for i, line in enumerate(fh):
                    if self.limit and i >= self.limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
        else:
            with open(expanded, "r") as fh:
                data = json.load(fh)
            for i, row in enumerate(data if isinstance(data, list) else data.values()):
                if self.limit and i >= self.limit:
                    break
                yield row

    # ------------------------------------------------------------------ mapping
    @staticmethod
    def _conversation_by_role(
        conversations: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        by_role: Dict[str, List[str]] = {}
        for turn in conversations or []:
            if not isinstance(turn, dict):
                continue
            role = turn.get("from") or turn.get("role")
            value = turn.get("value") or turn.get("content")
            if role and isinstance(value, str):
                by_role.setdefault(role, []).append(value)
        return by_role

    def _build_sample(
        self,
        row: Dict[str, Any],
        conversations: List[Dict[str, str]],
        tools: List[ToolDefinition],
        gt_calls: List[GroundTruthCall],
        metadata_extra: Optional[Dict[str, Any]] = None,
        json_tool_name: Optional[str] = None,
    ) -> Optional[UnifiedSample]:
        """Assemble a UnifiedSample from parsed pieces; None if not processable."""
        self.stats["rows_yielded"] += 1
        mid = row.get("id") or f"row-{self.stats['rows_seen']}"
        sample_id = f"{self.config_name}::{mid}"
        by_role = self._conversation_by_role(conversations)
        humans = by_role.get("human", [])
        if not humans:
            self.stats["dropped_no_human_turn"] += 1
            return None

        # GT calls supplied by the caller are authoritative; else fall back to
        # scanning ALL assistant turns for <tool_call> blocks. Multi-turn rows
        # (func_calling / glaive_func_calling) end with a natural-language
        # summary, so the tags live in earlier assistant turns.
        assistant_roles = [r for r in ("gpt", "assistant", "bot") if r in by_role]
        assistant_texts = by_role[assistant_roles[-1]] if assistant_roles else []
        if not gt_calls:
            for a_text in assistant_texts:
                for parsed in _iter_tool_call_payloads(a_text):
                    # T7: fall back to the same json-mode name the tool list
                    # uses, never a hardcoded literal that may not exist there.
                    name = (
                        parsed.get("name")
                        or parsed.get("function")
                        or json_tool_name
                        or JSON_MODE_TOOL_NAME
                    )
                    args = parsed.get("arguments")
                    if isinstance(args, str):
                        args = _safe_json_loads(args) or {}
                    gt_calls.append(GroundTruthCall(tool_name=str(name), arguments=args or {}))
                if gt_calls:
                    break  # first assistant turn that carries tool calls wins

        if not gt_calls and self.config_name not in _JSON_MODE_CONFIGS:
            # A function-calling row MUST have a GT call; json-mode rows fall
            # back to the synthetic json_output GT below.
            self.stats["dropped_no_gt"] += 1
            return None

        if not gt_calls:
            # json-mode: GT is the final assistant JSON (single-turn) or the
            # last JSON-producing assistant turn (agentic multi-turn). The hub
            # also ships ARRAY-typed outputs (e.g. flight/booking lists) which
            # we wrap under "items" to fit the GroundTruthCall.arguments dict
            # contract rather than dropping the row.
            #
            # T7: the GT tool name must be the SAME name the tool list carries.
            # Hardcoding "json_output" here while ``load`` named the tool from
            # ``schema.title`` made the GT unreachable by construction.
            gt_name = json_tool_name or JSON_MODE_TOOL_NAME
            obj = None
            for a_text in reversed(assistant_texts):
                obj = _safe_json_loads(a_text)
                if isinstance(obj, (dict, list)):
                    break
            if isinstance(obj, dict):
                gt_calls = [GroundTruthCall(tool_name=gt_name, arguments=obj)]
            elif isinstance(obj, list):
                gt_calls = [GroundTruthCall(tool_name=gt_name, arguments={"items": obj})]
            else:
                self.stats["dropped_no_gt"] += 1
                return None

        utterance = humans[-1]
        recent_context: List[Dict[str, Any]] = []
        if len(humans) > 1:
            for i, h in enumerate(humans[:-1]):
                recent_context.append({"key": f"turn_{i}", "value": h, "source": "conversation"})
            replies = by_role[assistant_roles[-1]][:-1] if assistant_roles else []
            for i, a in enumerate(replies):
                recent_context.append({"key": f"reply_{i}", "value": a, "source": "conversation"})

        metadata: Dict[str, Any] = {
            "config": self.config_name,
            "category": row.get("category"),
            "subcategory": row.get("subcategory"),
            "task": row.get("task"),
            "mode": "json" if self.config_name in _JSON_MODE_CONFIGS else "function",
            "turns": len(humans),
            "source_style": "glaive" if self.config_name == "glaive_func_calling" else "hermes",
        }
        if metadata_extra:
            metadata.update(metadata_extra)

        if self.config_name in _JSON_MODE_CONFIGS:
            self.stats["json_mode_rows"] += 1

        if len(gt_calls) > 1:
            self.stats["multi_call_rows"] += 1
        self.stats["parsed_gt_calls"] += len(gt_calls)

        return UnifiedSample(
            sample_id=sample_id,
            source_dataset="hermes-function-calling-v1",
            utterance=utterance,
            tools=tools or [ToolDefinition(
                name=json_tool_name or JSON_MODE_TOOL_NAME,
                description="Structured JSON output",
            )],
            ground_truth_calls=gt_calls,
            recent_context=recent_context,
            stratum=Stratum.F,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ load
    def load(self, split: Optional[str] = None) -> Iterator[UnifiedSample]:
        """Stream rows of the configured config into UnifiedSample records."""
        del split  # Hermes V1 ships a single 'train' split per config
        for row in self._iter_raw_rows():
            self.stats["rows_seen"] += 1
            if not isinstance(row, dict):
                continue
            conversations = row.get("conversations")
            if not conversations:
                self.stats["dropped_no_conversation"] += 1
                continue

            tools: List[ToolDefinition] = []
            gt_calls: List[GroundTruthCall] = []
            metadata_extra: Optional[Dict[str, Any]] = None
            # T7: one name decides BOTH the json-mode tool and its synthetic GT.
            json_tool_name: Optional[str] = None

            if self.config_name in _JSON_MODE_CONFIGS:
                # json-mode: no OpenAI tools; use the `schema` field if present,
                # else the synthetic json_output tool.
                schema_raw = row.get("schema")
                if not schema_raw:
                    # fall back to the <schema> block inside the system turn
                    by_role = self._conversation_by_role(conversations)
                    system = (by_role.get("system") or [""])[0]
                    schema_raw = _parse_schema_block(system)
                if schema_raw:
                    parsed_schema = _safe_json_loads(schema_raw)
                    if isinstance(parsed_schema, dict):
                        json_tool_name = _json_mode_tool_name(parsed_schema)
                        tools = [
                            _tool_from_json_schema(
                                json_tool_name,
                                "Structured JSON output",
                                parsed_schema,
                            )
                        ]
                if json_tool_name is None:
                    json_tool_name = JSON_MODE_TOOL_NAME
                metadata_extra = {"agentic": self.config_name == "json_mode_agentic"}
            else:
                # function-calling configs: prefer the `tools` column.
                tools_raw = row.get("tools")
                if isinstance(tools_raw, str):
                    tools = _parse_tools_json(tools_raw)
                elif isinstance(tools_raw, list):
                    tools = _parse_tools_json(json.dumps(tools_raw))
                if not tools:
                    # fall back to <tools> block inside the system turn
                    by_role = self._conversation_by_role(conversations)
                    system = (by_role.get("system") or [""])[0]
                    block = _parse_schema_block(system)
                    if block:
                        tools = _parse_tools_json(block)

                # Glaive config: assistant answers are
                # "{\"function\":...,\"arguments\":{...}}" — scan ALL assistant
                # turns (multi-turn rows interleave tool calls + summaries).
                if self.config_name == "glaive_func_calling":
                    by_role = self._conversation_by_role(conversations)
                    assistant_roles = [r for r in ("gpt", "assistant") if r in by_role]
                    for a_text in by_role.get(assistant_roles[0], []) if assistant_roles else []:
                        parsed = _safe_json_loads(a_text)
                        if isinstance(parsed, dict) and "function" in parsed:
                            name = parsed.get("function")
                            args = parsed.get("arguments") or {}
                            if name:
                                gt_calls.append(GroundTruthCall(tool_name=str(name), arguments=args))
                                break  # first tool-carrying assistant turn wins

            sample = self._build_sample(
                row,
                conversations,
                tools,
                gt_calls,
                metadata_extra=metadata_extra,
                json_tool_name=json_tool_name,
            )
            if sample is not None:
                yield sample
