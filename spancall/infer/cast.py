"""Strict typed caster and JSON schema validator.

Ref: docs/mechanism.md §7.1, §7.2, docs/pipeline_io.md Q4, Q5, Q6.

``StrictTypeCaster`` (Q4 + Q5):

- **Q4 span → original text.** Never detokenizes. For a rendered-utterance span
  ``(char_start, char_end)`` it maps through ``norm2orig`` back to the user's
  original characters, enforces word-boundary snapping (expanding a partial
  piece of at most one character, else abstaining ``bad_boundary``), and trims
  boundary whitespace/punctuation. Non-utterance regions (``recent``, ``time``,
  ``enum``, ``bool``) use the already-parsed legend ``value`` directly.
- **Q5 strict cast.** Deterministic, never fuzzy: booleans from exact
  ``true``/``false``; enums by case-insensitive exact match returning the
  canonical schema string; integers with an optional ``k/m/b`` multiplier that
  reject spelled-out, approximated, or locale-ambiguous forms; numbers as ints
  plus decimals; dates from strict formats only (ISO ``YYYY-MM-DD``,
  ``DD Mon YYYY``, ``Mon DD, YYYY``); arrays cast element-wise (whole parameter
  fails if any element fails).

``SchemaValidator`` (Q6): applies schema defaults for omitted optional
parameters, verifies presence of required parameters, checks value types and
enum membership, and optionally runs a jsonschema validation built from the
``ToolDefinition``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import date as _date_type
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from spancall.data.schema import ParameterSchema, ToolDefinition

# --------------------------------------------------------------------------- #
# Q5 lexical rules
# --------------------------------------------------------------------------- #

# integer: optional sign, digit groups with thousand separators, optional k/m/b
# multiplier. Deliberately rejects "1.000" (locale-ambiguous), "~500", "five".
_INTEGER_RE = re.compile(r"^[-+]?\d+(?:,\d{3})*[kKmMbB]?$")
# number: integer/float with optional k/m/b multiplier.
_NUMBER_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)[kKmMbB]?$")
_MULTIPLIER = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}

# Q4 boundary punctuation to trim from extracted utterance text.
_START_TRIM_RE = re.compile(r"^[\s,.;:!?)(\"']+")
_END_TRIM_RE = re.compile(r"[\s,.;:!?)(\"']+$")

# Q5 date formats: ISO (incl. <|ctx_time|> verbatim) and two strict literal forms.
_DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%b %d, %Y")


def _collapse_ws(text: str) -> str:
    """Q5 string: whitespace-collapse, preserving case/diacritics."""
    return " ".join(text.split())


def _trim(text: str) -> str:
    """Q4.3: strip leading/trailing whitespace and boundary punctuation."""
    out = text
    for _ in range(4):
        nxt = _START_TRIM_RE.sub("", out)
        nxt = _END_TRIM_RE.sub("", nxt)
        if nxt == out:
            break
        out = nxt
    return out


# --------------------------------------------------------------------------- #
# Abstract contract
# --------------------------------------------------------------------------- #


class BaseTypeCaster(ABC):
    """Abstract type caster converting raw character spans to strictly typed Python values."""

    @abstractmethod
    def cast_parameter(
        self,
        raw_text: str,
        param_schema: ParameterSchema,
    ) -> Tuple[bool, Any]:
        """Strict conversion according to parameter type.

        Returns:
            (success, parsed_value_or_none)
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Concrete caster (Q4 + Q5)
# --------------------------------------------------------------------------- #


class StrictTypeCaster(BaseTypeCaster):
    """Q4 provenance mapping + Q5 strict typed casting.

    Args:
        locale: reserved for number/date locale policy (unused in v1; all
            current rules are locale-independent and deliberately strict).
        use_jsonschema: when True, :class:`SchemaValidator` additionally runs
            a jsonschema validation of the payload (requires ``jsonschema``).
    """

    def __init__(self, locale: str = "en_US", use_jsonschema: bool = False) -> None:
        self.locale = locale
        self.use_jsonschema = use_jsonschema
        self.validator = SchemaValidator(use_jsonschema=use_jsonschema)

    # ------------------------------------------------------------------ #
    # Base contract (Q5 over raw text)
    # ------------------------------------------------------------------ #
    def cast_parameter(self, raw_text: str, param_schema: ParameterSchema) -> Tuple[bool, Any]:
        return self._cast(raw_text, param_schema)

    # ------------------------------------------------------------------ #
    # Q4 — span to original text
    # ------------------------------------------------------------------ #
    @staticmethod
    def _utterance_start(rendered_entries: Sequence[Any]) -> Optional[int]:
        """Absolute offset of the utterance line within the rendered text."""
        starts = [
            e.char_start
            for e in rendered_entries
            if getattr(e, "segment", None) == "utterance"
        ]
        return min(starts) if starts else None

    def extract_original(
        self,
        *,
        char_start: int,
        char_end: int,
        output: Any,
    ) -> Tuple[bool, str, Optional[str]]:
        """Q4: map a rendered-utterance span to the user's original characters.

        ``output`` is a :class:`span_template.template.SpanTemplateOutput`
        (duck-typed: needs ``.entries`` and ``.norm`` with ``norm2orig``).

        Returns:
            ``(ok, original_text, reason)``; reason is None on success or one
            of ``"bad_offsets"`` / ``"bad_boundary"``.
        """
        norm = getattr(output, "norm", None)
        entries = list(getattr(output, "entries", None) or [])
        utt_start = self._utterance_start(entries)
        if norm is None or utt_start is None:
            return False, "", "bad_offsets"
        rel_s = char_start - utt_start
        rel_e = char_end - utt_start
        n2o = norm.norm2orig
        if rel_s < 0 or rel_e > len(norm.norm) or rel_e <= rel_s:
            return False, "", "bad_offsets"
        orig_s = n2o[rel_s]
        orig_e = n2o[rel_e - 1] + 1
        text = norm.orig[orig_s:orig_e]
        text, reason = self._snap_word_boundaries(norm.orig, orig_s, orig_e, text)
        if reason is not None:
            return False, text, reason
        return True, _trim(text), None

    @staticmethod
    def _snap_word_boundaries(
        orig: str, s: int, e: int, text: str
    ) -> Tuple[str, Optional[str]]:
        """Q4.2: expand to word boundary when the partial piece is <= 1 char."""
        if not orig:
            return text, None
        s = max(0, min(s, len(orig)))
        e = max(s, min(e, len(orig)))
        # Trim leading/trailing whitespace from the slice first
        while s < e and orig[s].isspace():
            s += 1
        while e > s and orig[e - 1].isspace():
            e -= 1
        if s >= e:
            return "", None

        def _is_word(ch: str) -> bool:
            return ch.isalnum() or ch == "_"

        # Leading partial piece: only expand if boundary splits inside a word
        lead = 0
        if s > 0 and _is_word(orig[s]) and _is_word(orig[s - 1]):
            i = s - 1
            while i >= 0 and _is_word(orig[i]):
                lead += 1
                i -= 1
            if lead > 1:
                return text, "bad_boundary"

        # Trailing partial piece: only expand if boundary splits inside a word
        trail = 0
        if e < len(orig) and _is_word(orig[e - 1]) and _is_word(orig[e]):
            j = e
            while j < len(orig) and _is_word(orig[j]):
                trail += 1
                j += 1
            if trail > 1:
                return text, "bad_boundary"

        return orig[s - lead : e + trail], None

    # ------------------------------------------------------------------ #
    # Resolver-facing combined Q4+Q5
    # ------------------------------------------------------------------ #
    def cast_choice(
        self,
        choice: Any,
        param_schema: ParameterSchema,
        output: Any = None,
    ) -> Tuple[bool, Any, Optional[str]]:
        """Q4 + Q5 for one decoded :class:`ValueChoice`.

        Non-utterance regions use the legend's canonical value; utterance spans
        are mapped through Q4 to original text, then strictly cast.

        Returns:
            ``(ok, value, reason)``; reason is None on success.
        """
        region = getattr(choice, "region", "value")
        span = getattr(choice, "span", None)
        if region == "utterance" and output is not None and span is not None:
            ok, raw, reason = self.extract_original(
                char_start=span[0], char_end=span[1], output=output
            )
            if not ok:
                return False, None, reason
            ok_c, value = self._cast(raw, param_schema)
            return ok_c, value, None if ok_c else "cast_fail"
        raw_val = getattr(choice, "value", None)
        ok_c, value = self.cast_canonical(raw_val, param_schema)
        if not ok_c and isinstance(raw_val, str):
            ok_c, value = self._cast(raw_val, param_schema)
        return ok_c, value, None if ok_c else "cast_fail"

    # ------------------------------------------------------------------ #
    # Q5 dispatchers
    # ------------------------------------------------------------------ #
    def cast_canonical(self, value: Any, param_schema: ParameterSchema) -> Tuple[bool, Any]:
        """Q5 over a non-utterance canonical value (legend-resolved)."""
        raw_ptype = (param_schema.param_type or "string").lower()
        ptype = raw_ptype
        if ptype.startswith("["):
            for candidate in ("boolean", "integer", "number", "date", "array", "object", "string"):
                if candidate in ptype:
                    ptype = candidate
                    break
        if value is None:
            return True, None  # caller decides null-emit policy (default/absent)
        # Enums: never emit a value outside the declared pool, even when the
        # parser handed us a canonical-looking string.
        if param_schema.enum is not None and isinstance(value, str):
            return self._enum(value, param_schema)
        if ptype == "boolean":
            return (True, value) if isinstance(value, bool) else (False, None)
        if ptype == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return True, value
            return False, None
        if ptype == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True, float(value)
            return False, None
        if ptype == "date":
            if isinstance(value, _date_type):
                return True, value
            return False, None
        if ptype == "array" and isinstance(value, list):
            return True, value
        if ptype == "object" and isinstance(value, dict):
            return True, value
        if ptype == "string":
            return True, _collapse_ws(str(value))
        return False, None

    def _cast(self, raw_text: str, ps: ParameterSchema) -> Tuple[bool, Any]:
        raw_ptype = (ps.param_type or "string").lower()
        ptype = raw_ptype
        if ptype.startswith("["):
            for candidate in ("boolean", "integer", "number", "date", "array", "object", "string"):
                if candidate in ptype:
                    ptype = candidate
                    break
        # Enums are declared JSON-Schema style: param_type "string" + enum list.
        if ptype == "enum" or ps.enum is not None:
            return self._enum(_collapse_ws(raw_text), ps)
        text = _collapse_ws(raw_text)
        if ptype == "boolean":
            return self._bool(text)
        if ptype == "integer":
            return self._integer(text)
        if ptype == "number":
            return self._number(text)
        if ptype == "date":
            return self._date(text)
        if ptype == "array":
            return self._array(raw_text, ps)
        if ptype == "object":
            return self._object(raw_text, ps)
        if ptype == "string":
            if not text:
                return False, None
            return True, text
        return False, None

    @staticmethod
    def _bool(text: str) -> Tuple[bool, Any]:
        low = text.lower()
        if low == "true":
            return True, True
        if low == "false":
            return True, False
        return False, None

    @staticmethod
    def _enum(text: str, ps: ParameterSchema) -> Tuple[bool, Any]:
        if not ps.enum:
            return False, None
        low = text.lower()
        for canonical in ps.enum:
            if str(canonical).lower() == low:
                return True, canonical
        return False, None

    @classmethod
    def _integer(cls, text: str) -> Tuple[bool, Any]:
        t = text.strip()
        if not _INTEGER_RE.match(t):
            return False, None
        suffix = t[-1].lower() if t and t[-1] in "kKmMbB" else None
        digits = t[:-1] if suffix else t
        digits = digits.replace(",", "")
        try:
            value = int(digits)
        except ValueError:
            return False, None
        if suffix:
            value = int(value * _MULTIPLIER[suffix])
        return True, value

    @classmethod
    def _number(cls, text: str) -> Tuple[bool, Any]:
        t = text.strip()
        if not _NUMBER_RE.match(t):
            return False, None
        suffix = t[-1].lower() if t and t[-1] in "kKmMbB" else None
        body = t[:-1] if suffix else t
        try:
            value = float(body)
        except ValueError:
            return False, None
        if suffix:
            value = value * _MULTIPLIER[suffix]
        return True, value

    @classmethod
    def _date(cls, text: str) -> Tuple[bool, Any]:
        t = text.strip()
        for fmt in _DATE_FORMATS:
            try:
                return True, datetime.strptime(t, fmt).date()
            except ValueError:
                continue
        return False, None

    def _array(self, raw_text: str, ps: ParameterSchema) -> Tuple[bool, Any]:
        t = raw_text.strip()
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1].strip()
        # Handle natural conjunctions like ", and ", ", or ", " and ", " or "
        t = re.sub(r",\s*(?:and|or)\s+", ",", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+(?:and|or)\s+", ",", t, flags=re.IGNORECASE)
        parts = [p.strip() for p in t.split(",")]
        parts = [p.strip("'\" \t\r\n") for p in parts if p.strip()]
        if not parts:
            return False, None
        items_type = (ps.items_type or "string").lower()
        item_schema = replace(ps, param_type=items_type)
        out: List[Any] = []
        for part in parts:
            ok, v = self._cast(part, item_schema)
            if not ok:
                return False, None
            out.append(v)
        return True, out

    @staticmethod
    def _object(raw_text: str, ps: ParameterSchema) -> Tuple[bool, Any]:
        # v1: strict JSON object parse only; anything else abstains.
        import json

        t = raw_text.strip()
        try:
            obj = json.loads(t)
        except ValueError:
            return False, None
        if not isinstance(obj, dict):
            return False, None
        return True, obj


# --------------------------------------------------------------------------- #
# Q6 — JSON-schema validator
# --------------------------------------------------------------------------- #


_JSON_TYPE = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "date": "string",
}


class SchemaValidator:
    """Q6: validates a fully-assembled payload against a tool definition."""

    def __init__(self, use_jsonschema: bool = False) -> None:
        self.use_jsonschema = use_jsonschema

    def apply_defaults(
        self, payload: Dict[str, Any], tool_schema: ToolDefinition
    ) -> Dict[str, Any]:
        """Materialize schema defaults for omitted optional parameters (Q6.1)."""
        out = dict(payload)
        for p in tool_schema.parameters:
            if p.name not in out and not p.required and p.default is not None:
                out[p.name] = p.default
        return out

    def validate(
        self,
        payload: Dict[str, Any],
        tool_schema: ToolDefinition,
    ) -> Tuple[bool, Optional[str]]:
        """Checks required parameters, types, and enum constraints.

        Returns:
            ``(valid, reason)``; reason is None when valid.
        """
        payload = self.apply_defaults(payload, tool_schema)
        missing = [
            p.name for p in tool_schema.parameters if p.required and p.name not in payload
        ]
        if missing:
            return False, f"missing_required:{','.join(missing)}"
        for p in tool_schema.parameters:
            if p.name not in payload:
                continue
            value = payload[p.name]
            if not self._type_ok(p, value):
                return False, f"type_mismatch:{p.name}"
            if p.enum is not None and not self._enum_ok(p, value):
                return False, f"enum_mismatch:{p.name}"
        if self.use_jsonschema and not self._jsonschema_ok(payload, tool_schema):
            return False, "jsonschema_violation"
        return True, None

    @staticmethod
    def _enum_ok(p: ParameterSchema, value: Any) -> bool:
        if str(value) in {str(e) for e in p.enum}:
            return True
        return any(isinstance(value, type(e)) and value == e for e in p.enum)

    @staticmethod
    def _type_ok(p: ParameterSchema, value: Any) -> bool:
        raw_ptype = (p.param_type or "string").lower()
        is_nullable = "null" in raw_ptype
        if is_nullable and value is None:
            return True
        ptype = raw_ptype
        if ptype.startswith("["):
            for candidate in ("boolean", "integer", "number", "date", "array", "object", "string"):
                if candidate in ptype:
                    ptype = candidate
                    break
        if ptype == "null":
            return value is None
        if ptype == "boolean":
            return isinstance(value, bool)
        if ptype == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if ptype == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if ptype == "string":
            return isinstance(value, str)
        if ptype == "date":
            return isinstance(value, (str, _date_type))
        if ptype == "array":
            return isinstance(value, list)
        if ptype == "object":
            return isinstance(value, dict)
        return True

    def _jsonschema_ok(self, payload: Dict[str, Any], tool_schema: ToolDefinition) -> bool:
        try:
            import jsonschema  # type: ignore
        except ImportError:
            return False
        properties: Dict[str, Any] = {}
        for p in tool_schema.parameters:
            prop: Dict[str, Any] = {"type": _JSON_TYPE.get((p.param_type or "string").lower(), "string")}
            if p.enum is not None:
                prop["enum"] = list(p.enum)
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
        schema = {
            "type": "object",
            "properties": properties,
            "required": [p.name for p in tool_schema.parameters if p.required],
        }
        try:
            jsonschema.validate(instance=payload, schema=schema)
            return True
        except Exception:  # jsonschema.ValidationError / SchemaError
            return False
