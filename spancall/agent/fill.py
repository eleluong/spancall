"""Hybrid-fill helpers: SpanCall predicts the call, the LLM fills only the gaps.

Motivation (v3 agentic mode 3). Two integration modes already exist:

* **LLM-only** — every call comes from the LLM (``RouterMode.LLM_ONLY``).
* **SpanCall + LLM fallback** — SpanCall tries first and the *whole* call is
  delegated when it abstains (``RouterMode.ROUTER``). Safe, but a single
  unresolved parameter throws away an otherwise perfect skeleton (correct tool,
  five correctly extracted arguments).

This module supports the third mode: keep the SpanCall skeleton and ask the LLM
for **only the parameters SpanCall could not ground**. That is a much smaller
ask than emitting a full call (fewer tokens, less room to corrupt the parts that
were already correct), and the merge is provably conservative:
:func:`merge_filled_arguments` never lets an LLM value overwrite a resolved one.

Nothing here talks to a network — the router supplies the backend.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence

from spancall.data.schema import ParameterSchema, ToolDefinition

__all__ = [
    "param_leaves",
    "restrict_tool_to_params",
    "build_fill_messages",
    "merge_filled_arguments",
    "unresolved_param_names",
]


def param_leaves(tool: ToolDefinition) -> List[str]:
    """Flat list of addressable parameter names for a tool.

    Decomposed object params (Lean v2.7) expose dotted leaves
    (``filters.color``) while the *root* stays a schema param. The resolver
    reports partial payloads with whatever namespace it used, so both the root
    and the leaf names are considered addressable here — matching a name is a
    string comparison, not a schema guess.
    """
    names: List[str] = []
    for p in tool.parameters or ():
        names.append(str(p.name))
    for root, leaves in (tool.object_paths or {}).items():
        if root not in names:
            names.append(str(root))
        for leaf in leaves or ():
            leaf_s = str(leaf)
            if leaf_s not in names:
                names.append(leaf_s)
    return names


def unresolved_param_names(
    tool: ToolDefinition,
    payload_partial: Optional[Dict[str, Any]],
    missing_required: Optional[Iterable[str]] = None,
    missing_optional: Optional[Iterable[str]] = None,
) -> List[str]:
    """Parameters that SpanCall did NOT ground and that the LLM should fill.

    ``missing_required`` / ``missing_optional`` carry the resolver's own verdict
    (it can distinguish "the decoder refused" from "the value failed a cast"),
    so they are preferred when present. When they are empty the gap is derived
    from ``payload_partial`` so a partial result with no explicit lists still
    yields something to ask for.
    """
    have = {str(k) for k in (payload_partial or {})}
    names = param_leaves(tool)
    ordered: List[str] = []
    for src in (missing_required, missing_optional):
        for name in src or ():
            name_s = str(name)
            if name_s and name_s not in have and name_s not in ordered:
                ordered.append(name_s)
    if ordered:
        return ordered
    return [n for n in names if n not in have]


def restrict_tool_to_params(tool: ToolDefinition, names: Sequence[str]) -> ToolDefinition:
    """A copy of ``tool`` whose schema lists ONLY ``names`` (as required).

    The restricted schema is what the LLM sees, so it physically cannot answer
    for a parameter SpanCall already resolved — the constraint is enforced by
    the prompt, not by a discipline requirement on the model. Dotted object
    leaves that are not top-level params are kept in ``object_paths`` so the
    restriction stays meaningful for decomposed tools.
    """
    wanted = {str(n) for n in names}
    kept: List[ParameterSchema] = [p for p in (tool.parameters or ()) if p.name in wanted]
    # Leaves live outside `parameters`; keep only the wanted ones.
    object_paths = None
    if tool.object_paths:
        object_paths = {
            str(root): [str(l) for l in (leaves or ()) if str(l) in wanted]
            for root, leaves in tool.object_paths.items()
            if any(str(l) in wanted for l in (leaves or ()))
        } or None
    # A restricted param is by definition required: we are asking for a value,
    # and an optional "you may omit this" would allow an empty answer.
    kept = [replace(p, required=True) for p in kept]
    return ToolDefinition(
        name=tool.name,
        description=str(tool.description or ""),
        parameters=kept,
        risk_tier=str(tool.risk_tier or "read_only"),
        object_paths=object_paths,
    )


def build_fill_messages(
    messages: Sequence[Dict[str, Any]],
    tool_name: str,
    known_arguments: Dict[str, Any],
    missing: Sequence[str],
) -> List[Dict[str, Any]]:
    """Prompt asking for the missing arguments only, with the call as context.

    The already-resolved arguments are included as **context**, explicitly
    marked immutable. Without them the model has to re-derive the whole call to
    make sense of the gap (and then happily "corrects" values it was never
    asked about). With them, the task is a fill-in-the-blank.
    """
    known_json = json.dumps(known_arguments or {}, ensure_ascii=False, sort_keys=True)
    missing_list = ", ".join(str(m) for m in missing)
    instruction = (
        "A tool call has already been partially resolved.\n"
        f"Tool: {tool_name}\n"
        f"Already resolved (DO NOT change or repeat these values): {known_json}\n"
        f"Missing parameters to fill: {missing_list}\n\n"
        "Call the same tool again and supply ONLY the missing parameters above. "
        "Never repeat a resolved parameter, never invent a value that is not "
        "supported by the conversation, and use the declared types exactly."
    )
    out: List[Dict[str, Any]] = [dict(m) for m in messages]
    out.append({"role": "user", "content": instruction})
    return out


def merge_filled_arguments(
    known: Dict[str, Any],
    filled: Optional[Dict[str, Any]],
    allowed: Sequence[str],
) -> Dict[str, Any]:
    """Merge LLM-filled values into the SpanCall payload, conservatively.

    Rules (each one is a bug class that was observed or is trivially reachable):

    * A resolved value is **never** overwritten by the LLM — the whole point of
      the hybrid mode is that SpanCall's grounded spans win.
    * Only ``allowed`` (the requested missing names) may be added; anything else
      the model volunteers is dropped, because it is un-grounded and would
      silently widen the payload.
    * ``None`` is not a value: an explicit null is dropped so a required param
      still counts as missing and the caller's gate can abstain.
    """
    out = dict(known or {})
    allow = {str(a) for a in allowed}
    for name, value in (filled or {}).items():
        name_s = str(name)
        if name_s in out or name_s not in allow:
            continue
        if value is None:
            continue
        out[name_s] = value
    return out