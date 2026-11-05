"""Trajectory ingestion, decomposition, selection, and transcript budgeting.

Implements Phase 3 & Phase 4 of long_context_fix.md:
- Extracts the latest user utterance so fast-path resolution is never starved on turn > 0.
- Flattens structured tool outputs (JSON/dicts/lists) into leaf key/value rows for recent context.
- R6 row selection with relevance scoring against packed tool schemas, R7 untrusted-marker checks,
  deduplication, and whole-row token budget eviction (no arbitrary character truncation).
- Atomic turn-group transcript budgeting for the LLM path.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from span_template.sanitize import check_recent_value


class ContextBudgetError(ValueError):
    """Raised when context/transcript length exceeds safety limits."""
    pass


def latest_user_utterance(messages: Sequence[Dict[str, Any]], fallback: str = "") -> str:
    """Return the content of the latest user-role message in messages, or fallback."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if content and isinstance(content, str) and content.strip():
                return content.strip()
    return fallback


def sanitize_key(key: str) -> str:
    """Ensure a recent_context or time key matches ^[A-Za-z_][A-Za-z0-9_.-]*$."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))
    if not cleaned:
        return "key"
    if not re.match(r"^[A-Za-z_]", cleaned):
        cleaned = "_" + cleaned
    return cleaned


def _normalize_recent_value(val: Any) -> str:
    """Collapse newlines/control chars to single spaces so <|ctx_recent|> stays single-line."""
    s = str(val)
    s = re.sub(r"[\r\n\t\x00-\x1f\x7f]+", " ", s)
    s = re.sub(r" {2,}", " ", s).strip()
    return s


def flatten_tool_output(
    turn_idx: int,
    tool_name: str,
    output: Any,
) -> List[Dict[str, Any]]:
    """Recursively decompose structured tool outputs into leaf key/value rows.

    Scalar values become rows with key ``t{turn_idx}_{tool_name}_{path}`` and
    ``source: tool_output``. Non-JSON / unstructured outputs become a single row.
    No arbitrary V_max char cap is applied (long-context fix).
    """
    if output is None:
        return []

    parsed: Any = None
    if isinstance(output, (dict, list)):
        parsed = output
    elif isinstance(output, str):
        trimmed = output.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (
            trimmed.startswith("[") and trimmed.endswith("]")
        ):
            try:
                parsed = json.loads(trimmed)
            except Exception:
                parsed = output
        else:
            parsed = output
    else:
        parsed = str(output)

    def _recurse(val: Any, prefix: str) -> List[Tuple[str, str]]:
        leaves: List[Tuple[str, str]] = []
        if isinstance(val, dict):
            for k, v in val.items():
                safe_k = sanitize_key(str(k))
                sub = f"{prefix}.{safe_k}" if prefix else safe_k
                leaves.extend(_recurse(v, sub))
        elif isinstance(val, list):
            for i, v in enumerate(val):
                sub = f"{prefix}.{i}" if prefix else str(i)
                leaves.extend(_recurse(v, sub))
        else:
            if val is not None:
                leaves.append((prefix, _normalize_recent_value(val)))
        return leaves

    rows: List[Dict[str, Any]] = []
    base_name = sanitize_key(tool_name)
    if isinstance(parsed, (dict, list)):
        leaves = _recurse(parsed, "")
        for path, val_str in leaves:
            key = sanitize_key(f"t{turn_idx}_{base_name}_{path}" if path else f"t{turn_idx}_{base_name}")
            rows.append({
                "key": key,
                "value": val_str,
                "source": "tool_output",
            })
    else:
        key = sanitize_key(f"t{turn_idx}_{base_name}")
        rows.append({
            "key": key,
            "value": _normalize_recent_value(parsed),
            "source": "tool_output",
        })

    return rows


def rows_from_messages(
    messages: Sequence[Dict[str, Any]],
    exclude_last_user: bool = True,
) -> List[Dict[str, Any]]:
    """Convert trajectory messages (tool outputs and conversation prose) into R6 rows.

    When exclude_last_user is True, the active user utterance being routed is not
    duplicated into recent context rows.
    """
    rows: List[Dict[str, Any]] = []
    last_user_idx = -1
    if exclude_last_user:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_idx = idx
                break

    for turn_idx, msg in enumerate(messages):
        if turn_idx == last_user_idx:
            continue
        role = msg.get("role")
        if role == "tool":
            tool_name = msg.get("name") or "tool"
            content = msg.get("content")
            rows.extend(flatten_tool_output(turn_idx, tool_name, content))
        elif role in ("user", "assistant"):
            content = msg.get("content")
            if content and isinstance(content, str) and content.strip():
                key = sanitize_key(f"turn_{turn_idx}_{role}")
                norm_val = _normalize_recent_value(content)
                if norm_val:
                    rows.append({
                        "key": key,
                        "value": norm_val,
                        "source": "conversation",
                    })
    return rows


def _row_field(r: Any, field: str, default: str = "") -> str:
    if isinstance(r, dict):
        return str(r.get(field, default) or default)
    return str(getattr(r, field, default) or default)


def estimate_tokens(text_or_rows: Any, tokenizer: Optional[Any] = None) -> int:
    """Estimate token count for a text string or recent context rows."""
    if tokenizer is not None:
        try:
            if isinstance(text_or_rows, str):
                return len(tokenizer.encode(text_or_rows))
            if isinstance(text_or_rows, (list, tuple)):
                rendered = " ".join(f"{_row_field(r, 'key')} {_row_field(r, 'value')}" for r in text_or_rows)
                return len(tokenizer.encode(rendered))
        except Exception:
            pass

    if isinstance(text_or_rows, str):
        return max(1, len(text_or_rows) // 4)
    if isinstance(text_or_rows, (list, tuple)):
        total_chars = sum(len(_row_field(r, "key")) + len(_row_field(r, "value")) + 8 for r in text_or_rows)
        return max(1, total_chars // 4)
    return 1


def _extract_tool_keywords(tools: Sequence[Any]) -> Set[str]:
    """Extract lowercase tokens/stems from tool definitions for relevance filtering."""
    keywords: Set[str] = set()
    for t in tools:
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "")
        desc = getattr(t, "description", None) or (t.get("description") if isinstance(t, dict) else "")
        params = (
            getattr(t, "parameters", None)
            or getattr(t, "params", None)
            or (t.get("parameters") if isinstance(t, dict) else None)
        )

        for text in (name, desc):
            if text:
                for word in re.findall(r"[A-Za-z0-9_]+", text.lower()):
                    if len(word) >= 3:
                        keywords.add(word)

        if isinstance(params, list):
            for p in params:
                p_name = getattr(p, "name", None) or (p.get("name") if isinstance(p, dict) else "")
                p_desc = getattr(p, "description", None) or (p.get("description") if isinstance(p, dict) else "")
                for text in (p_name, p_desc):
                    if text:
                        for word in re.findall(r"[A-Za-z0-9_]+", text.lower()):
                            if len(word) >= 3:
                                keywords.add(word)
                enums = getattr(p, "enum", None) or (p.get("enum") if isinstance(p, dict) else None)
                if enums and isinstance(enums, (list, tuple)):
                    for ev in enums:
                        for word in re.findall(r"[A-Za-z0-9_]+", str(ev).lower()):
                            if len(word) >= 3:
                                keywords.add(word)
        elif isinstance(params, dict):
            props = params.get("properties", {})
            for p_name, p_dict in props.items():
                keywords.add(p_name.lower())
                p_desc = p_dict.get("description", "") if isinstance(p_dict, dict) else ""
                if p_desc:
                    for word in re.findall(r"[A-Za-z0-9_]+", p_desc.lower()):
                        if len(word) >= 3:
                            keywords.add(word)
    return keywords


def select_recent_context(
    rows: Sequence[Any],
    tools: Sequence[Any],
    k_recent: int = 32,
    context_budget_tokens: int = 16384,
    tokenizer: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select and budget recent_context rows according to R6 / R7 rules.

    1. R7 check: reject any row whose key or value contains reserved marker tokens.
    2. Normalize keys (identifier-safe) and values (single-line, control-char free).
    3. Deduplicate identical (key, value) pairs preserving order.
    4. Relevance scoring: mark rows that share word overlap with packed tools.
    5. Keep all relevant rows plus the k_recent most recent rows.
    6. Token budget: evict whole oldest rows until the rendered recent block fits context_budget_tokens.
       Never truncate individual row values (a truncated id is an invalid id).
    """
    valid_rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    for r in rows:
        raw_key = _row_field(r, "key", "")
        raw_val = _row_field(r, "value", "")
        raw_src = _row_field(r, "source", "recent")
        if check_recent_value(raw_key) is not None or check_recent_value(raw_val) is not None:
            continue
        key = sanitize_key(raw_key)
        val = _normalize_recent_value(raw_val)
        if not val:
            continue
        if check_recent_value(key) is not None or check_recent_value(val) is not None:
            continue
        pair = (key, val)
        if pair in seen:
            continue
        seen.add(pair)
        row_dict = dict(r) if isinstance(r, dict) else {"source": raw_src}
        row_dict["key"] = key
        row_dict["value"] = val
        valid_rows.append(row_dict)

    if not valid_rows:
        return [], {"n_recent_rows": 0, "recent_tokens": 0, "truncated": False}

    keywords = _extract_tool_keywords(tools)

    def _is_relevant(r: Dict[str, Any]) -> bool:
        row_text = f"{r.get('key', '')} {r.get('value', '')}".lower()
        words = re.findall(r"[A-Za-z0-9_]+", row_text)
        return any(w in keywords for w in words)

    relevant_indices = {i for i, r in enumerate(valid_rows) if _is_relevant(r)}
    recent_indices = set(range(max(0, len(valid_rows) - k_recent), len(valid_rows)))
    keep_indices = sorted(relevant_indices | recent_indices)
    candidate_rows = [valid_rows[i] for i in keep_indices]

    truncated = False
    while candidate_rows:
        n_tok = estimate_tokens(candidate_rows, tokenizer)
        if n_tok <= context_budget_tokens:
            break
        candidate_rows.pop(0)
        truncated = True

    final_tokens = estimate_tokens(candidate_rows, tokenizer) if candidate_rows else 0
    features = {
        "n_recent_rows": len(candidate_rows),
        "recent_tokens": final_tokens,
        "truncated": truncated,
    }
    return candidate_rows, features


def trim_messages(
    messages: Sequence[Dict[str, Any]],
    max_context_tokens: int = 32768,
    keep_last_tool_turns: int = 8,
    tokenizer: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Budget LLM-path transcript tokens by evicting whole atomic turn groups.

    Atomic turn groups ensure an assistant message with ``tool_calls`` is never
    orphaned from its corresponding ``role: tool`` responses.
    Pinned messages:
      - System message (if present at index 0)
      - First user message (initial task instruction)
      - Latest user message
    Remaining message groups are evicted from oldest to newest if context exceeds max_context_tokens.
    """
    if not messages:
        return [], 0

    units: List[List[Dict[str, Any]]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        role = msg.get("role")
        if role == "assistant" and (msg.get("tool_calls") or "tool_calls" in msg):
            group = [msg]
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                group.append(messages[j])
                j += 1
            units.append(group)
            i = j
        else:
            units.append([msg])
            i += 1

    pinned_indices: Set[int] = set()
    found_first_user = False
    for idx, u in enumerate(units):
        role = u[0].get("role")
        if idx == 0 and role == "system":
            pinned_indices.add(idx)
        elif role == "user" and not found_first_user:
            pinned_indices.add(idx)
            found_first_user = True

    for idx in range(len(units) - 1, -1, -1):
        if units[idx][0].get("role") == "user":
            pinned_indices.add(idx)
            break

    def _tok_count(selected_units: List[List[Dict[str, Any]]]) -> int:
        flat_list = [m for grp in selected_units for m in grp]
        raw = json.dumps(flat_list, default=str)
        return estimate_tokens(raw, tokenizer)

    tool_unit_indices = [
        idx for idx, u in enumerate(units)
        if any(m.get("role") == "tool" or m.get("tool_calls") for m in u)
    ]
    protected_tool_indices = set(tool_unit_indices[-keep_last_tool_turns:]) if keep_last_tool_turns > 0 else set()

    current_units = list(units)
    trimmed_count = 0

    while _tok_count(current_units) > max_context_tokens:
        evict_idx = -1
        for idx, u in enumerate(current_units):
            # Map back to original unit identity via object equality
            is_pinned = any(u is units[p] for p in pinned_indices if p < len(units))
            is_protected = any(u is units[p] for p in protected_tool_indices if p < len(units))
            if not is_pinned and not is_protected:
                evict_idx = idx
                break

        if evict_idx == -1:
            for idx, u in enumerate(current_units):
                is_pinned = any(u is units[p] for p in pinned_indices if p < len(units))
                if not is_pinned:
                    evict_idx = idx
                    break

        if evict_idx == -1:
            break

        current_units.pop(evict_idx)
        trimmed_count += 1

    flat_result = [m for grp in current_units for m in grp]
    return flat_result, trimmed_count
