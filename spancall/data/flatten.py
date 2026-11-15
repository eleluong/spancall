"""Object-param decomposition: flatten nested ``object`` values into scalar leaves.

Lean v2.7 (see ``plans/lean_v2_7_object_decomposition_plan.md``) turns every
*schema-declared* nested object into a set of dotted scalar pseudo-params::

    {"filters": {"color": "red", "size": 10}}  ->  {"filters.color": "red", "filters.size": 10}

Each leaf is then aligned / tensorized / decoded exactly like an ordinary scalar
param (Cases B-E in :mod:`spancall.data.align`), and the resolver reassembles the
tree with :func:`unflatten_args` *before* type casting, schema validation and the
whole-call gates. Arrays are deliberately untouched by this module.

Two modes are supported:

``declared``
    The caller passes ``object_paths`` (root -> list of leaf dotted paths) taken
    from the tool schema (``ToolDefinition.object_paths``). Only those leaves are
    split out; everything else is copied through verbatim. This is the mode the
    aligner and resolver use, and it is immune to payload keys that legitimately
    contain a ``.``.

``inferred``
    No schema information available: any nested ``Mapping`` of depth <=
    ``max_depth`` is flattened automatically. Used by tooling/tests and by
    back-compat callers that predate ``object_paths``.

Depth is capped (``OBJECT_MAX_DEPTH``) because every leaf costs one ``<|param|>``
anchor and one pointer slot; deeper trees stay whole and therefore remain
``unresolved`` (routed to the slow path, per the retrieval-first contract).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Maximum number of dotted segments a leaf may have (``filters.a.b`` == 3).
OBJECT_MAX_DEPTH: int = 3

#: Separator between path segments in a flattened param name.
PATH_SEP: str = "."


class _Missing:
    """Sentinel for "path not present" (``None`` is a legal leaf value)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "<MISSING>"


_MISSING = _Missing()

#: Public alias for the "path not present" sentinel. Callers outside this module
#: (the v2.7 audit cell, diagnostics) must compare against this rather than the
#: private ``_MISSING`` so the identity contract can be refactored safely.
MISSING = _MISSING


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #
def derive_object_paths(
    param_names: Sequence[str],
) -> Dict[str, List[str]]:
    """Infer ``root -> [leaf, ...]`` from a flat list of dotted param names.

    ``["filters.color", "filters.size", "query"]`` ->
    ``{"filters": ["filters.color", "filters.size"]}``.
    """
    out: Dict[str, List[str]] = {}
    for name in param_names:
        n = str(name)
        if PATH_SEP not in n:
            continue
        root = n.split(PATH_SEP, 1)[0]
        if not root:
            continue
        out.setdefault(root, []).append(n)
    for leaves in out.values():
        leaves.sort()
    return out


def object_paths_from_params(params: Sequence[Any]) -> Dict[str, List[str]]:
    """Same as :func:`derive_object_paths` but takes ``ParameterSchema`` objects."""
    return derive_object_paths([str(getattr(p, "name", "")) for p in params])


def validate_object_paths(
    param_names: Sequence[str],
    object_paths: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[str]:
    """Return a list of human-readable problems (empty == clean).

    Catches the two failure modes that would silently corrupt payloads:

    * a dotted leaf whose root is *also* a declared literal param (ambiguous
      namespace: ``filters`` and ``filters.color`` both exist);
    * an ``object_paths`` entry listing a leaf that is not in ``param_names``.
    """
    names = [str(n) for n in param_names]
    name_set = set(names)
    problems: List[str] = []

    derived = derive_object_paths(names)
    for root in sorted(derived):
        if root in name_set:
            problems.append(
                f"ambiguous namespace: param {root!r} exists alongside dotted "
                f"leaves {sorted(derived[root])}"
            )

    if object_paths:
        for root, leaves in object_paths.items():
            for leaf in leaves:
                leaf_s = str(leaf)
                if leaf_s not in name_set:
                    problems.append(
                        f"object_paths[{root!r}] lists {leaf_s!r}, which is not a declared param"
                    )
                if leaf_s.split(PATH_SEP, 1)[0] != str(root):
                    problems.append(
                        f"object_paths[{root!r}] lists {leaf_s!r}, whose root is not {root!r}"
                    )
    return problems


def get_path(value: Any, path: str) -> Any:
    """Read a dotted ``path`` out of nested mappings; ``_MISSING`` when absent."""
    cur = value
    for seg in str(path).split(PATH_SEP):
        if not isinstance(cur, Mapping) or seg not in cur:
            return _MISSING
        cur = cur[seg]
    return cur


def is_decomposable(value: Any, *, max_depth: int = OBJECT_MAX_DEPTH) -> bool:
    """True when ``value`` is a non-empty mapping shallow enough to flatten."""
    if not isinstance(value, Mapping) or not value:
        return False
    return _tree_depth(value, 0) <= int(max_depth)


def _tree_depth(value: Any, consumed: int) -> int:
    """Longest dotted path (in segments) produced by flattening ``value``.

    ``consumed`` is the number of segments already used by the caller's prefix;
    the root call passes ``0`` so ``{"a": 1}`` has depth 1 and
    ``{"a": {"b": {"c": 1}}}`` has depth 3.
    """
    if not isinstance(value, Mapping) or not value:
        return consumed
    deepest = consumed
    for k, v in value.items():
        if not _is_path_segment(k):
            return int(OBJECT_MAX_DEPTH) + 1  # opaque: forces "too deep"
        deepest = max(deepest, _tree_depth(v, consumed + 1))
    return deepest


def _is_path_segment(key: Any) -> bool:
    return isinstance(key, str) and bool(key) and PATH_SEP not in key


# --------------------------------------------------------------------------- #
# flatten
# --------------------------------------------------------------------------- #
def flatten_args(
    args: Optional[Mapping[str, Any]],
    object_paths: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    max_depth: int = OBJECT_MAX_DEPTH,
) -> Dict[str, Any]:
    """Flatten nested object values into dotted scalar keys.

    Args:
        args: GT / predicted argument mapping.
        object_paths: ``root -> [leaf, ...]`` schema declaration. When given,
            *only* declared leaves are split out (payload keys containing a dot
            are left alone unless their root is declared). When ``None``, nested
            mappings are inferred automatically up to ``max_depth``.
        max_depth: inferred-mode depth cap. Ignored in declared mode (the schema
            already decided what is a leaf).

    Returns:
        A new flat dict. Non-object values are copied through unchanged.
    """
    flat: Dict[str, Any] = {}
    roots = {str(r) for r in (object_paths or {})}
    for key, value in (args or {}).items():
        name = str(key)
        if object_paths is not None and (not roots or name not in roots):
            # Not a decomposed root: copy verbatim, dots included.
            flat[name] = value
            continue
        _flatten_into(name, value, flat, 1, int(max_depth))
    return flat


def _flatten_into(
    prefix: str,
    value: Any,
    out: Dict[str, Any],
    depth: int,
    max_depth: int,
) -> None:
    """Recursively flatten one value into dotted keys under ``prefix``.

    Every key is preserved (declared by the schema or not) so that
    :func:`unflatten_args` can rebuild the payload exactly. Keys that cannot be
    expressed as a dotted segment (non-string, empty, or already containing the
    separator) stop the descent: that subtree is kept whole rather than being
    silently renamed, and the schema validator will flag it.
    """
    if not isinstance(value, Mapping) or not value or depth >= max_depth:
        out[prefix] = value
        return
    if not all(_is_path_segment(k) for k in value):
        out[prefix] = value
        return
    for k, v in value.items():
        _flatten_into(f"{prefix}{PATH_SEP}{k}", v, out, depth + 1, max_depth)


# --------------------------------------------------------------------------- #
# unflatten
# --------------------------------------------------------------------------- #
def unflatten_args(
    flat: Optional[Mapping[str, Any]],
    object_paths: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Reassemble dotted scalar leaves back into nested objects.

    Args:
        flat: flattened payload.
        object_paths: ``root -> [leaf, ...]``. When given, only those roots are
            reassembled; every other key (including literal keys containing a
            dot) is copied verbatim. When ``None``, any dotted key is treated as
            nested.
        strict: raise ``ValueError`` when a dotted leaf collides with a literal
            key of the same root. When ``False`` the literal value wins and the
            nested write is skipped.

    Returns:
        A new nested dict.
    """
    roots = {str(r) for r in (object_paths or {})}
    out: Dict[str, Any] = {}
    nested: List[Tuple[str, Any]] = []

    for key, value in (flat or {}).items():
        name = str(key)
        root = name.split(PATH_SEP, 1)[0] if PATH_SEP in name else None
        if root is None or (roots and root not in roots):
            out[name] = value
        else:
            nested.append((name, value))

    for name, value in nested:
        segments = name.split(PATH_SEP)
        cur = out
        blocked = False
        for seg in segments[:-1]:
            if seg not in cur:
                cur[seg] = {}
            nxt = cur[seg]
            if not isinstance(nxt, dict):
                if strict:
                    raise ValueError(
                        f"cannot unflatten {name!r}: {seg!r} already holds a "
                        f"non-object value {nxt!r}"
                    )
                blocked = True
                break
            cur = nxt
        if blocked:
            continue
        cur[segments[-1]] = value

    return out


def split_flat_payload(
    payload: Mapping[str, Any],
    object_paths: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a payload into ``(scalar_leaves, whole_value_roots)``.

    Convenience for reporting / diagnostics: which keys the fast path resolved
    as decomposed leaves and which were resolved as opaque objects.
    """
    roots = {str(r) for r in (object_paths or {})}
    leaves: Dict[str, Any] = {}
    whole: Dict[str, Any] = {}
    for key, value in (payload or {}).items():
        name = str(key)
        root = name.split(PATH_SEP, 1)[0] if PATH_SEP in name else None
        if root is not None and (not roots or root in roots):
            leaves[name] = value
        else:
            whole[name] = value
    return leaves, whole