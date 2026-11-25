"""Shared AST & Jupyter scope guard for generated notebooks.

Verifies every generated Python code cell compiles and that every referenced
name is bound either in builtins, IPython globals, or an earlier cell.
"""

from __future__ import annotations

import builtins
import symtable
from typing import Any, Sequence


def _cell_source(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _is_ipython_cell(src: str) -> bool:
    for line in src.splitlines():
        head = line.lstrip()
        if head.startswith(("!", "%", "%%")):
            return True
    return False


def _scope_walk(tbl: Any, enclosing: frozenset[str], available: set[str], unresolved: list[str]) -> None:
    local = set()
    for sym in tbl.get_symbols():
        if sym.is_assigned() or sym.is_imported() or sym.is_parameter() or sym.is_namespace():
            local.add(sym.get_name())
    visible = local | enclosing
    for sym in tbl.get_symbols():
        name = sym.get_name()
        if name in visible or sym.is_free():
            continue
        if not (sym.is_referenced() or sym.is_declared_global()):
            continue
        if name in available:
            continue
        unresolved.append(name)
    for child in tbl.get_children():
        _scope_walk(child, visible, available, unresolved)


def guard_code_cells(cells: Sequence[dict[str, Any]]) -> int:
    """Compile every code cell and assert all referenced names resolve across cells."""
    available = set(dir(builtins)) | {
        "__name__", "__file__", "__doc__", "__builtins__", "__spec__",
        "display", "get_ipython", "In", "Out", "exit", "quit", "_", "__", "___",
    }
    problems: list[str] = []
    checked = skipped = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = _cell_source(cell)
        if _is_ipython_cell(src):
            skipped += 1
            continue
        first = src.strip().splitlines()[0] if src.strip() else ""
        label = first[:72]
        try:
            compile(src, "<generated cell>", "exec")
        except SyntaxError as exc:
            problems.append(f"SYNTAX  [{label}]\n        {exc}")
            continue
        checked += 1
        table = symtable.symtable(src, "<generated cell>", "exec")
        module_bound = {
            sym.get_name()
            for sym in table.get_symbols()
            if sym.is_assigned() or sym.is_imported() or sym.is_parameter() or sym.is_namespace()
        }
        cell_available = available | module_bound
        unresolved: list[str] = []
        _scope_walk(table, frozenset(), cell_available, unresolved)
        for name in sorted(set(unresolved)):
            problems.append(f"NAME    {name!r} is used but never bound in this cell\n        [{label}]")
        available |= module_bound

    if problems:
        print(f"\n!!! {len(problems)} notebook cell problem(s) found — refusing to write !!!\n")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print(f"cell guard: {checked} code cells compile, all names resolve ({skipped} IPython cell(s) skipped)")
    return checked
