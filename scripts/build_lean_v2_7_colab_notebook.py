#!/usr/bin/env python3
"""Generate notebooks/spancall_lean_v2_7_train_eval.ipynb via nbformat (valid JSON guaranteed).

SpanCall Lean Flow **v2.7** — Object-Param Decomposition:
- Keeps the v2.6 pipeline (Hermes + xLAM-60k 1-call, ModernBERT-large) intact.
- Re-renders through the v2.7 grammar, so ``object`` params declared by a tool
  schema are decomposed into scalar dotted pseudo-params
  (``filters.color``) instead of failing to ``unresolved``.
- Adds an **object-decomposition audit** (§4b): how many object params were
  decomposed vs left unresolved, and the precision/recall split by param type,
  which is what this version exists to measure.
- Arrays are unchanged from v2.6 (the v2.6 FP analysis showed array element
  selection is still the dominant precision source; see
  ``plans/lean_v2_7_object_decomposition_plan.md`` §4 "Out of scope").

Cell sources are imported from ``build_lean_v2_6_colab_notebook.py`` and patched
in place. Slicing the module (rather than duplicating ~1.7k lines) keeps the two
builders from drifting: a cell fix in v2.6 flows into v2.7 automatically, and
only the version-specific overrides below are maintained here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
V26_SCRIPT = HERE / "build_lean_v2_6_colab_notebook.py"
OUT = HERE / ".." / "notebooks" / "spancall_lean_v2_7_train_eval.ipynb"

# --------------------------------------------------------------------------
# Load the v2.6 builder's cell sources (exec the module minus its build tail)
# --------------------------------------------------------------------------
_SRC = V26_SCRIPT.read_text(encoding="utf-8")
_START = _SRC.index("MARKDOWN_TITLE = ")
_END = _SRC.rindex("cells = [")
_NS: Dict[str, Any] = {
    "Path": Path,
    "__name__": "build_lean_v2_6_colab_notebook_sources",
}
exec(compile(_SRC[_START:_END], str(V26_SCRIPT), "exec"), _NS)

# Every string constant in the v2.6 module defines a cell or a piece of one.
_CELL_CONSTANTS = [
    k for k, v in _NS.items()
    if k.isupper() and isinstance(v, str)
]

# --------------------------------------------------------------------------
# Version-specific overrides
# --------------------------------------------------------------------------
MARKDOWN_TITLE = """# SpanCall Lean Flow **v2.7** — Object-Param Decomposition

End-to-end training and step-by-step evaluation notebook for the **v2.7** change set. The pipeline is v2.6 (Hermes V1 + `Salesforce/xlam-function-calling-60k` 1-call subset, `answerdotai/ModernBERT-large`, `MAX_SEQ_LEN = 4096`); what changes is **how `object`-typed params are represented**:

- **Object decomposition.** A declared nested object (`{"filters": {"color": "red", "size": 10}}`) is split into scalar dotted pseudo-params (`filters.color`, `filters.size`) at the schema layer, aligned like any other scalar, and **reassembled into the nested dict by the resolver before casting, schema validation and the whole-call gates**. See `spancall/data/flatten.py` and `plans/lean_v2_7_object_decomposition_plan.md`.
- **Why.** In v2.6 an object value fell through the scalar branch of `ReverseAligner._align_value` and ended at `OFF_WINDOW`, i.e. it was silently *unresolved* — a coverage loss, and the reason object params never showed up in the FP dump. Decomposition converts them into the single-span selection the lean flow is good at.
- **What this notebook must answer.** (a) did coverage on object-bearing tools rise; (b) did exact-payload precision hold or improve; (c) how much of the residual error is still array-shaped.
- **Explicitly unchanged.** Arrays (element selection, cardinality, ordering) and per-param refusal. The v2.6 analysis showed arrays are the dominant precision source; that work is out of scope here and the audit in §4b keeps tracking it so it cannot hide.

**Path changes vs v2.6:** `data/tensorized_lean_v2_7` and `results_lean_v2_7`, and the tensors are rebuilt from `data/processed_v2_7` (realigned through the v2.7 template)."""

MARKDOWN_OBJECT_AUDIT = """---
## 4b. Object-Decomposition Audit

Quantifies the v2.7 change on the evaluation split:

- **Decomposition coverage** — of the `object` params the GT declares, how many exist in the schema as dotted leaves (`decomposed`) versus as a single opaque `object` param (`unresolved`, routed to the slow path). A schema-level number, independent of model quality.
- **Outcome split by param type** — exact-match rate per `string` / `integer` / `number` / `boolean` / `enum` / `array` / `object` / `object-leaf`. This is the table that shows whether decomposition moved objects and how large the remaining array gap is.
- **Obj-shaped residual examples** — up to 5 failing rows whose only mismatches are object leaves or arrays, with the leaf-level diff, so "did reassembly round-trip" is checkable by eye rather than inferred from a scalar."""

CODE_OBJECT_AUDIT = '''# 4b. Object-decomposition audit (schema coverage + per-param-type accuracy + residual examples)
import json, os
from collections import Counter, defaultdict
from typing import Any, Dict, List

import spancall.data.flatten as _flat
from spancall.data.schema import UnifiedSample
from spancall.eval.toolcall.evaluator import payload_equal, _deep_equal


def _is_object_leaf(name: str, object_paths: Dict[str, List[str]]) -> bool:
    root = str(name).split(_flat.PATH_SEP, 1)[0]
    return bool(object_paths) and any(
        str(leaf) == str(name) for leaf in (object_paths.get(root) or ())
    )


def _gather_samples(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        print(f"no source {path} - skipped")
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------- schema view
_schema_stats = Counter()
_per_tool_paths: Dict[str, Dict[str, List[str]]] = {}
for _raw in _gather_samples(os.path.join(DATA_DIR, f"{EVAL_SPLIT}.jsonl")):
    _s = UnifiedSample.from_dict(_raw)
    for _t in _s.tools:
        _per_tool_paths[_t.name] = dict(_t.object_paths or {})
        _opaque = [p.name for p in _t.parameters if (p.param_type or "").lower() == "object"]
        _leaves = [p.name for p in _t.parameters if _is_object_leaf(p.name, _t.object_paths or {})]
        if _opaque:
            _schema_stats["tools_with_opaque_object"] += 1
            _schema_stats["opaque_object_params"] += len(_opaque)
        if _leaves:
            _schema_stats["tools_with_decomposed_object"] += 1
            _schema_stats["decomposed_leaf_params"] += len(_leaves)
        if _t.object_paths:
            _schema_stats["object_roots_decomposed"] += len(_t.object_paths)

print("=== Object Decomposition Coverage (schema level) ===")
print(f"  object roots decomposed          : {_schema_stats['object_roots_decomposed']}")
print(f"  scalar leaf params emitted       : {_schema_stats['decomposed_leaf_params']}")
print(f"  tools with decomposed objects    : {_schema_stats['tools_with_decomposed_object']}")
print(f"  tools still carrying an opaque object param : {_schema_stats['tools_with_opaque_object']}")
print(f"  opaque object params (routed to slow path)  : {_schema_stats['opaque_object_params']}")
if not _schema_stats["object_roots_decomposed"]:
    print("  [i] no decomposable object found in this split - decomposition is a no-op here.")

# ------------------------------------------------------------ per-type outcome
_PRED_PATH = os.path.join(RESULTS_DIR, "preds_eval.jsonl")
_preds_by_id: Dict[str, Dict[str, Any]] = {}
for _p in _gather_samples(_PRED_PATH):
    _preds_by_id[_p["sample_id"]] = _p

_type_tot = Counter()
_type_ok = Counter()
_residual: List[Dict[str, Any]] = []

for _raw in _gather_samples(SUBSET_EVAL_PATH):
    _sid = _raw.get("sample_id")
    _p = _preds_by_id.get(_sid)
    if _p is None or not _p.get("gt_args"):
        continue
    _s = UnifiedSample.from_dict(_raw)
    if not _s.ground_truth_calls:
        continue
    _gt_call = _s.ground_truth_calls[0]
    _gt_tool = next((t for t in _s.tools if t.name == _gt_call.tool_name), None)
    if _gt_tool is None:
        continue
    _paths = dict(_gt_tool.object_paths or {})
    _pred_args = _p.get("payload_partial") or {}
    _bad_names: List[str] = []
    _gt_full = _flat.flatten_args(_gt_call.arguments or {}, _paths)

    for _param in _gt_tool.parameters:
        _name = _param.name
        if _name not in _gt_full:
            # Not present in this call's GT arguments -> nothing to score.
            continue
        if _is_object_leaf(_name, _paths):
            _kind = "object-leaf"
        else:
            _kind = (_param.param_type or "string").lower()
        _type_tot[_kind] += 1
        _ok = _name in _pred_args and _deep_equal(_gt_full[_name], _pred_args[_name])
        if _ok:
            _type_ok[_kind] += 1
        else:
            _bad_names.append(_name)

    # Residual rows whose failures are ONLY object leaves / arrays.
    if _bad_names:
        _kinds = set()
        for _n in _bad_names:
            _pp = next((q for q in _gt_tool.parameters if q.name == _n), None)
            _kinds.add("object-leaf" if _is_object_leaf(_n, _paths)
                       else ((_pp.param_type or "string").lower() if _pp else "unknown"))
        if _kinds <= {"object-leaf", "array", "object"}:
            _residual.append({"sample_id": _sid, "tool": _gt_call.tool_name,
                              "bad": _bad_names, "kinds": sorted(_kinds),
                              "gt": _gt_full, "pred": _pred_args})

print("\\n=== Per-Param-Type Exact Match (extractable params, hermes_eval) ===")
print(f"{'param_type':<14}{'n':>6}{'exact':>9}{'rate':>8}")
print("-" * 40)
for _k in sorted(_type_tot, key=lambda x: -_type_tot[x]):
    _n, _ok = _type_tot[_k], _type_ok[_k]
    print(f"{_k:<14}{_n:>6}{_ok:>9}{_ok / max(1, _n):>8.3f}")

_n_obj = sum(v for k, v in _type_tot.items() if k == "object-leaf")
print(f"\\n  object-leaf params scored: {_n_obj}"
      f"  (exact {_type_ok['object-leaf']}/{max(1, _n_obj)})")
print(f"  array params scored      : {_type_tot['array']}"
      f"  (exact {_type_ok['array']}/{max(1, _type_tot['array'])})")
print("  [i] 'object-leaf' is the decomposed form: a scalar slot that reassembles into")
print("      the parent object. A rate here above the pre-v2.7 'object' rate is the win.")

print(f"\\n=== Residual Rows Failing Only On Complex Params ({len(_residual)} total; showing up to 5) ===")
for _r in _residual[:5]:
    print("-" * 100)
    print(f"{_r['sample_id']}  tool={_r['tool']}  kinds={_r['kinds']}")
    for _n in _r["bad"][:6]:
        print(f"  [WRONG] {_n:<28} gt={_r['gt'].get(_n)!r:<42} pred={_r['pred'].get(_n)!r}")
if not _residual:
    print("  none - no row failed exclusively on object/array params.")

# Reassembly integrity: does the emitted payload equal the unflattened GT exactly?
_roundtrip_ok = _roundtrip_tot = 0
for _raw in _gather_samples(SUBSET_EVAL_PATH):
    _p = _preds_by_id.get(_raw.get("sample_id"))
    if _p is None:
        continue
    _s = UnifiedSample.from_dict(_raw)
    if not _s.ground_truth_calls:
        continue
    _gt_tool = next((t for t in _s.tools if t.name == _s.ground_truth_calls[0].tool_name), None)
    if _gt_tool is None or not _gt_tool.object_paths:
        continue
    _roundtrip_tot += 1
    _rebuilt = _flat.unflatten_args(_p.get("payload_partial") or {}, _gt_tool.object_paths, strict=False)
    if payload_equal(_s.ground_truth_calls[0].arguments or {}, _rebuilt, _gt_tool):
        _roundtrip_ok += 1

print("\\n=== Reassembly Integrity (object-bearing calls) ===")
if _roundtrip_tot:
    print(f"  calls with decomposed objects : {_roundtrip_tot}")
    print(f"  reassembled payload == GT     : {_roundtrip_ok} ({_roundtrip_ok / _roundtrip_tot:.3f})")
else:
    print("  no object-bearing call in this split")'''

# --------------------------------------------------------------------------
# Patch the v2.6 cell sources for v2.7
# --------------------------------------------------------------------------
# Version-bumping rules for the v2.7 paths. These are *idempotent regexes* keyed
# on the ``v2_6`` token rather than on surrounding punctuation, because the same
# path fragment shows up in three different shapes that must all be caught:
#   * Python expressions  -- ``str(PROJECT_ROOT / "data" / "tensorized_lean_v2_6")``
#   * prose / markdown    -- ``(`data/tensorized_lean_v2_6` and `results_lean_v2_6`)``
#   * error messages      -- ``Inspect data/processed_v2/realign_report.json``
# The ``(?!_7)`` guard makes re-running the builder harmless (no ``_v2_7_7``).
_VERSION_RULES = [
    (re.compile(r"processed_v2(?!_7)"), "processed_v2_7"),
    (re.compile(r"tensorized_lean_v2_6"), "tensorized_lean_v2_7"),
    (re.compile(r"results_lean_v2_6"), "results_lean_v2_7"),
    (re.compile(r"colab_lean_v2_6\.yaml"), "colab_lean_v2_7.yaml"),
]

# Prose-only rewrites ("v2.6" banners are handled by the generic regex below).
_REPLACEMENTS = [
    ('LEAN FLOW v2.6 CONFIG', 'LEAN FLOW v2.7 CONFIG'),
    ('# SpanCall Lean v2.6 Run Summary', '# SpanCall Lean v2.7 Run Summary'),
    ('# 0b. Configuration & Environment (Full Data)',
     '# 0b. Configuration & Environment (Full Data, v2.7 object decomposition)'),
]


# The v2.6 data cell downloads a prebuilt ``data/processed_v2`` into ``DATA_DIR``.
# v2.7 must NOT reuse it: object decomposition changes the rendered param list, so a
# stale prebuilt directory would train on v2.6 param lists with no dotted leaves.
# Splits must come from re-aligning v1 (the block immediately below it does that).
_STALE_V2_DOWNLOAD = '''_missing_v2 = [f for f in CANONICAL_SPLITS + XLAM_SPLITS if not os.path.exists(os.path.join(DATA_DIR, f))]
if FORCE_HF_DOWNLOAD or _missing_v2:
    print(f"Downloading v2 splits (Hermes + xLAM) from Hugging Face dataset: {HF_DATASET_ID} ...")
    snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        local_dir=str(PROJECT_ROOT),
        allow_patterns=["data/processed_v2/*.jsonl", "data/processed_v2/*.json"],
        token=HF_TOKEN,
    )'''

_V27_DOWNLOAD = '''_missing_v2 = [f for f in CANONICAL_SPLITS + XLAM_SPLITS if not os.path.exists(os.path.join(DATA_DIR, f))]
if _missing_v2:
    # v2.7: never pull a prebuilt processed_v2 - object decomposition changes the
    # rendered param list, so labels must be rebuilt from v1 (next block).
    print(f"{len(_missing_v2)} split(s) missing in {DATA_DIR} -> produced by re-alignment below.")'''


def _patch(name: str, src: str) -> str:
    """Apply the version bump to one cell source.

    Order matters: the stale-download block is swapped *first*, while its exact
    text is still intact, and only then are the path tokens rewritten.
    """
    out = src
    # 1. Drop the prebuilt-processed_v2 download (must see unmutated text).
    out = out.replace(_STALE_V2_DOWNLOAD, _V27_DOWNLOAD)
    # 2. Bump every v2_6 path token, in code, prose and messages alike.
    for pattern, new in _VERSION_RULES:
        out = pattern.sub(new, out)
    # 3. Prose-only banner rewrites, then any remaining "v2.6" stamp.
    for old, new in _REPLACEMENTS:
        if old in out:
            out = out.replace(old, new)
    out = re.sub(r"v2\.6", "v2.7", out)
    return out


def _patch_paths(src: str) -> str:
    """Bump v2_6 path tokens only.

    Used for markdown cells: they legitimately contain *comparative* prose
    ("In v2.6 an object value fell through the scalar branch ..."), which must
    survive verbatim, while a path such as ``data/processed_v2/`` must not.
    """
    out = src
    for pattern, new in _VERSION_RULES:
        out = pattern.sub(new, out)
    return out


_CELLS: Dict[str, Any] = {}
for _name in _CELL_CONSTANTS:
    _val = _NS[_name]
    _CELLS[_name] = _val

# Cells whose entire text is version-banner prose get the v2.7 prose instead.
_CELLS["MARKDOWN_TITLE"] = MARKDOWN_TITLE
_CELLS["MARKDOWN_SETUP"] = _patch(
    "MARKDOWN_SETUP", _NS["MARKDOWN_SETUP"]
).replace(
    "v2.7 paths (`data/tensorized_lean_v2_7` and `results_lean_v2_7`).",
    "v2.7 paths (`data/processed_v2_7`, `data/tensorized_lean_v2_7`, `results_lean_v2_7`). "
    "The data step below re-realigns through the v2.7 template because object "
    "decomposition changes the rendered param list.",
)

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
cells = [
    new_markdown_cell(MARKDOWN_TITLE),
    new_markdown_cell(_CELLS["MARKDOWN_SETUP"]),
    new_code_cell(_patch("CODE_SETUP", _CELLS["CODE_SETUP"])),
    new_code_cell(_patch("CODE_CONFIG", _CELLS["CODE_CONFIG"])),
    new_code_cell(_patch("CODE_RUNTIME_CHECK", _CELLS["CODE_RUNTIME_CHECK"])),
    new_markdown_cell(_CELLS["MARKDOWN_DATA"]),
    new_code_cell(_patch("CODE_DATA_AND_PREFLIGHT", _CELLS["CODE_DATA_AND_PREFLIGHT"])),
    new_markdown_cell(_CELLS["MARKDOWN_TENSORIZE"]),
    new_code_cell(_patch("CODE_TENSORIZE", _CELLS["CODE_TENSORIZE"])),
    new_markdown_cell(_CELLS["MARKDOWN_TRAIN"]),
    new_code_cell(_patch("CODE_TRAIN_CFG_AND_RUN", _CELLS["CODE_TRAIN_CFG_AND_RUN"])),
    new_code_cell(_patch("CODE_TELEMETRY_AND_RESOLVER", _CELLS["CODE_TELEMETRY_AND_RESOLVER"])),
    new_markdown_cell(_CELLS["MARKDOWN_EVAL_SLICE"]),
    new_code_cell(_patch("CODE_EVAL_SLICE", _CELLS["CODE_EVAL_SLICE"])),
    new_markdown_cell(_CELLS["MARKDOWN_EVAL_STEP1"]),
    new_code_cell(_patch("CODE_EVAL_STEP1", _CELLS["CODE_EVAL_STEP1"])),
    new_markdown_cell(_CELLS["MARKDOWN_EVAL_STEP2"]),
    new_code_cell(_patch("CODE_EVAL_STEP2", _CELLS["CODE_EVAL_STEP2"])),
    new_markdown_cell(_CELLS["MARKDOWN_EVAL_STEP3"]),
    new_code_cell(_patch("CODE_EVAL_STEP3", _CELLS["CODE_EVAL_STEP3"])),
    # --- v2.7 addition: object decomposition audit, before the miss attribution
    # so the per-stage buckets in §5 can be read against the per-type table here.
    new_markdown_cell(MARKDOWN_OBJECT_AUDIT),
    new_code_cell(CODE_OBJECT_AUDIT),
    new_markdown_cell(_CELLS["MARKDOWN_EVAL_STEP4"]),
    new_code_cell(_patch("CODE_EVAL_STEP4", _CELLS["CODE_EVAL_STEP4"])),
    new_markdown_cell(_CELLS["MARKDOWN_EVAL_STEP5"]),
    new_code_cell(_patch("CODE_EVAL_STEP5", _CELLS["CODE_EVAL_STEP5"])),
    new_markdown_cell(_CELLS["MARKDOWN_TEST"]),
    new_code_cell(_patch("CODE_TEST_AND_PERSIST", _CELLS["CODE_TEST_AND_PERSIST"])),
    new_markdown_cell(_CELLS["MARKDOWN_CORPUS_EVAL"]),
    new_code_cell(_patch("CODE_CORPUS_EVAL", _CELLS["CODE_CORPUS_EVAL"])),
    new_markdown_cell(_CELLS["MARKDOWN_SUMMARY_REPORT"]),
    new_code_cell(_patch("CODE_SUMMARY_REPORT", _CELLS["CODE_SUMMARY_REPORT"])),
]

# Markdown cells bypass ``_patch`` (it would rewrite intentional v2.6
# comparisons), but their *paths* still have to move to v2.7.
for _cell in cells:
    if _cell["cell_type"] == "markdown":
        _cell["source"] = _patch_paths("".join(_cell["source"]))

# --------------------------------------------------------------------------
# Build-time guard: compile every code cell and resolve names
# --------------------------------------------------------------------------
_GUARD_START = _SRC.index("def _cell_source")
_GUARD_END = _SRC.rindex("_guard_code_cells(cells)")
_GUARD_NS: Dict[str, Any] = {"__name__": "build_lean_v2_7_guard"}
exec(compile(_SRC[_GUARD_START:_GUARD_END], str(V26_SCRIPT), "exec"), _GUARD_NS)
_guard_code_cells = _GUARD_NS["_guard_code_cells"]

_guard_code_cells(cells)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
nb.metadata["colab"] = {"gpuType": "T4", "toc_visible": True, "provenance": []}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)

with open(OUT, "r") as fh:
    json.load(fh)
nb2 = nbformat.read(OUT, as_version=4)
nbformat.validate(nb2)
print(f"wrote {OUT.resolve()} — {len(cells)} cells, valid nbformat 4 JSON")