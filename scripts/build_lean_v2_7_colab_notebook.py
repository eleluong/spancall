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
### 4d. Object-Decomposition Audit

Quantifies the v2.7 change on the evaluation split:

- **Decomposition coverage** — of the `object` params the GT declares, how many exist in the schema as dotted leaves (`decomposed`) versus as a single opaque `object` param (`unresolved`, routed to the slow path). A schema-level number, independent of model quality.
- **Outcome split by param type** — exact-match rate per `string` / `integer` / `number` / `boolean` / `enum` / `array` / `object` / `object-leaf`. This is the table that shows whether decomposition moved objects and how large the remaining array gap is.
- **Obj-shaped residual examples** — up to 5 failing rows whose only mismatches are object leaves or arrays, with the leaf-level diff, so "did reassembly round-trip" is checkable by eye rather than inferred from a scalar."""

CODE_OBJECT_AUDIT = '''# 4d. Object-decomposition audit (schema coverage + per-param-type accuracy + residual examples)
import json, os
from collections import Counter
from typing import Any, Dict, List

import spancall.data.flatten as _flat
from spancall.data.schema import UnifiedSample
from spancall.eval.toolcall.evaluator import _deep_equal


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
# This is a SCHEMA-level number, so it must be counted once per distinct tool.
# The same tool schema repeats in every sample that offers it (~10-50x), and a
# per-occurrence counter would silently turn "tools with decomposed objects"
# into a sample-weighted count - a number that can exceed the number of distinct
# tools in the split and moves when the eval slice size changes.
_schema_stats = Counter()
_seen_tools: set = set()
_dup_tools = 0
for _raw in _gather_samples(os.path.join(DATA_DIR, f"{EVAL_SPLIT}.jsonl")):
    _s = UnifiedSample.from_dict(_raw)
    for _t in _s.tools:
        if _t.name in _seen_tools:
            _dup_tools += 1
            continue
        _seen_tools.add(_t.name)
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

print("=== Object Decomposition Coverage (schema level, distinct tools) ===")
print(f"  distinct tool schemas scanned    : {len(_seen_tools)}"
      f"  ({_dup_tools} repeated occurrences ignored)")
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

if not _preds_by_id:
    print(f"\\n[!] no predictions at {_PRED_PATH} - run 4c first. Per-type table skipped.")

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
    _gt_full = _flat.flatten_args(_gt_call.arguments or {}, _paths)

    # The predicted payload must be flattened into the SAME key space as the GT,
    # otherwise a nested ``{"filters": {...}}`` never matches a ``filters.color``
    # slot and every object leaf scores as wrong regardless of model quality.
    # ``flatten_args`` is idempotent for already-flat payloads, so this is safe
    # both before and after the resolver's unflatten step lands.
    _pred_raw = _p.get("payload_partial") or {}
    _pred_args = _flat.flatten_args(_pred_raw, _paths)

    _bad_names: List[str] = []

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
    _rate = f"{_ok / _n:>8.3f}" if _n else f"{'n/a':>8}"
    print(f"{_k:<14}{_n:>6}{_ok:>9}{_rate}")

_n_obj = _type_tot["object-leaf"]
print(f"\\n  object-leaf params scored: {_n_obj}"
      f"  (exact {_type_ok['object-leaf']}/{_n_obj})")
print(f"  array params scored      : {_type_tot['array']}"
      f"  (exact {_type_ok['array']}/{_type_tot['array']})")
if not _n_obj:
    print("  [!] zero object-leaf params scored: either no object-bearing tool is in")
    print("      this split, or step 3 (aligner object case) is not applied yet.")
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
# Reassembly integrity is checked on the LEAVES, not on the whole payload.
# ``payload_equal`` requires identical key sets, so using it here would report
# "reassembly broken" whenever the model merely omitted an argument -- mixing a
# model-quality signal into a correctness check. What this block must prove is
# narrower and model-independent: for every object leaf the model DID emit, does
# unflatten sit that leaf back at the same dotted path with the same value?
_rt_tot = _rt_ok = 0
_rt_bad: List[Dict[str, Any]] = []
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
    _rt_tot += 1
    # Normalise to the flat key space first: ``get_path`` walks dotted segments,
    # so on a flat payload it would always report ``_MISSING``. The resolver's
    # payload may be flat (pre-step-5) or nested (post-step-5); flattening both
    # forms idempotently makes this check true in either case.
    _pred_flat = _flat.flatten_args(_p.get("payload_partial") or {},
                                    _gt_tool.object_paths)
    _rebuilt = _flat.unflatten_args(_pred_flat, _gt_tool.object_paths, strict=False)
    # Walk every declared leaf: does it survive the flat -> nested round trip?
    _leaf_bad = []
    for _root, _leaves in _gt_tool.object_paths.items():
        for _leaf in _leaves:
            _before = _pred_flat.get(str(_leaf), _flat.MISSING)
            _after = _flat.get_path(_rebuilt, _leaf)
            if _before is _flat.MISSING and _after is _flat.MISSING:
                continue  # not emitted and not reassembled -> consistent
            if _before is _flat.MISSING or not _deep_equal(_before, _after):
                _leaf_bad.append({"leaf": _leaf,
                                  "flat": None if _before is _flat.MISSING else _before,
                                  "nested": None if _after is _flat.MISSING else _after})
    if _leaf_bad:
        _rt_bad.append({"sample_id": _raw.get("sample_id"), "leaves": _leaf_bad})
    else:
        _rt_ok += 1

print("\\n=== Reassembly Integrity (object-bearing calls) ===")
if _rt_tot:
    print(f"  calls with decomposed objects        : {_rt_tot}")
    print(f"  every emitted leaf reassembles at its path : {_rt_ok} ({_rt_ok / _rt_tot:.3f})")
    for _b in _rt_bad[:3]:
        print(f"  [LEAF MISMATCH] {_b['sample_id']}: {_b['leaves'][:3]}")
    if not _rt_bad:
        print("  no leaf-level reassembly defect - flat/nested round trip is sound.")
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
    # v2.7: never pull a prebuilt processed_v2. Object decomposition changes the
    # rendered param list, so a stale directory would train on v2.6-shaped labels
    # (one opaque `object` param, no dotted leaves) and every object number in
    # section 4d would be 0 for a reason that has nothing to do with the model.
    # Hermes splits are rebuilt by the realign block below; xLAM by prepare_data.
    print(f"[v2.7] {len(_missing_v2)} of {len(CANONICAL_SPLITS) + len(XLAM_SPLITS)} split(s) "
          f"missing in {DATA_DIR}")
    print("       no prebuilt v2 download is used in v2.7 - every split is rebuilt below.")'''


# --------------------------------------------------------------------------
# v2.7 surgical cell fixes
# --------------------------------------------------------------------------
# ``_REPLACEMENTS``/``_VERSION_RULES`` are *token* rewrites shared by every cell.
# The fixes below are per-cell and semantic, and they must land **after** the
# version bump, so each NEEDLE is written against the post-bump text and must
# not itself contain a version token (otherwise the bump would rewrite it).
#
# Two defect classes are addressed:
#   1. flat/nested key-space mismatch - the v2.6 eval cells compare nested
#      ``gt_args`` against dotted ``payload_partial`` keys, so every object leaf
#      scores as wrong regardless of model quality;
#   2. silent-failure gates - a missing checkpoint or an un-audited split that
#      reports a number instead of refusing.
_V27_FIXES: Dict[str, Any] = {
    # -- §1 data & pre-flight ------------------------------------------------
    "CODE_DATA_AND_PREFLIGHT": [
        # Explicit imports: the cell calls subprocess/sys but relied on the setup
        # cell having been run first, which breaks when this cell is run alone.
        (
            "from spancall.lean.diagnostics import audit_gt_offsets\nimport json, os",
            "from spancall.lean.diagnostics import audit_gt_offsets\n"
            "import json, os, subprocess, sys",
        ),
        # Gate B audited only eval+dev while §5 reports on the OOD/adversarial
        # splits, which are aligned by the same pass -> a label-offset defect
        # there showed up as a low score with no gate above it.
        (
            "# Pre-Flight Gate B: offset audit on eval & dev (Hermes + xLAM)\n"
            '_audit_splits = [EVAL_SPLIT, "hermes_dev"]\n'
            'for _xsp in ("xlam_eval", "xlam_dev"):\n'
            '    if os.path.exists(os.path.join(DATA_DIR, f"{_xsp}.jsonl")):\n'
            "        _audit_splits.append(_xsp)",
            "# Pre-Flight Gate B: offset audit on every split this notebook later\n"
            "# reports on, not just eval+dev. The OOD / adversarial splits come from\n"
            "# the same aligner, so an offset defect there has to fail here rather\n"
            "# than surface as a mysteriously low section-5 score.\n"
            '_audit_splits = [EVAL_SPLIT, "hermes_dev"]\n'
            'for _osp in list(TEST_SPLITS) + ["xlam_dev"]:\n'
            "    if _osp in _audit_splits:\n"
            "        continue\n"
            '    if os.path.exists(os.path.join(DATA_DIR, f"{_osp}.jsonl")):\n'
            "        _audit_splits.append(_osp)",
        ),
        # Fallback: prepare xLAM-60k (1-call) splits in DATA_DIR if still missing.
        # Prefer re-aligning the public convoicon/spancall xLAM splits (no HF gate
        # on Salesforce/xlam-function-calling-60k); only fall back to
        # prepare_data.py --sources xlam if the hub snapshot did not provide them.
        (
            "# Fallback: prepare xLAM-60k (1-call) splits in DATA_DIR if still missing\n"
            "_missing_xlam = [f for f in XLAM_SPLITS if not os.path.exists(os.path.join(DATA_DIR, f))]\n"
            "if _missing_xlam and (XLAM_TRAIN_ROWS is None or XLAM_TRAIN_ROWS > 0):\n"
            "    print(f\"Preparing xLAM-60k (1-call) splits into {DATA_DIR} via scripts/prepare_data.py ...\")\n"
            "    _prep_rc = subprocess.run([\n"
            "        sys.executable, os.path.join(str(PROJECT_ROOT), \"scripts\", \"prepare_data.py\"),\n"
            "        \"--sources\", \"xlam\",\n"
            "        \"--output_dir\", DATA_DIR,\n"
            "    ], cwd=str(PROJECT_ROOT)).returncode\n"
            "    if _prep_rc != 0:\n"
            "        raise SystemExit(f\"prepare_data.py --sources xlam exited {_prep_rc}\")",
            "# xLAM-60k (1-call) splits: download public xlam_*.jsonl from convoicon/spancall\n"
            "# and re-align through the v2.7 aligner (avoids the gated Salesforce/xlam-60k\n"
            "# hub download unless the snapshot lacks them).\n"
            "_missing_xlam = [f for f in XLAM_SPLITS if not os.path.exists(os.path.join(DATA_DIR, f))]\n"
            "if _missing_xlam and (XLAM_TRAIN_ROWS is None or XLAM_TRAIN_ROWS > 0):\n"
            "    _xlam_src_dir = os.path.join(str(PROJECT_ROOT), \"data\", \"processed_v2\")\n"
            "    _missing_xlam_src = [f for f in XLAM_SPLITS if not os.path.exists(os.path.join(_xlam_src_dir, f))]\n"
            "    if _missing_xlam_src:\n"
            "        print(f\"Downloading xLAM splits from {HF_DATASET_ID} ...\")\n"
            "        snapshot_download(\n"
            "            repo_id=HF_DATASET_ID,\n"
            "            repo_type=\"dataset\",\n"
            "            local_dir=str(PROJECT_ROOT),\n"
            "            allow_patterns=[\"data/processed_v2\" + \"/xlam_*.jsonl\", \"data/processed_v2\" + \"/*.json\"],\n"
            "            token=HF_TOKEN,\n"
            "        )\n"
            "    if all(os.path.exists(os.path.join(_xlam_src_dir, f)) for f in XLAM_SPLITS):\n"
            "        print(f\"Re-aligning xLAM splits -> {DATA_DIR} ...\")\n"
            "        _xlam_realign = [\n"
            "            sys.executable, os.path.join(str(PROJECT_ROOT), \"scripts\", \"realign_v2.py\"),\n"
            "            \"--src_dir\", _xlam_src_dir,\n"
            "            \"--out_dir\", DATA_DIR,\n"
            "            \"--splits\", *[os.path.splitext(f)[0] for f in XLAM_SPLITS],\n"
            "            \"--gate_split\", \"xlam_eval\",\n"
            "            \"--max_utt_stale_rate\", str(MAX_UTT_STALE_RATE),\n"
            "            \"--max_unresolvable\", str(MAX_UNRESOLVABLE),\n"
            "        ]\n"
            "        if ALLOW_STALE_LABELS:\n"
            "            _xlam_realign.append(\"--force\")\n"
            "        _xrc = subprocess.run(_xlam_realign, cwd=str(PROJECT_ROOT)).returncode\n"
            "        if _xrc != 0:\n"
            "            raise SystemExit(f\"realign_v2 (xlam) exited {_xrc}. Inspect data/processed_v2_7/realign_report.json.\")\n"
            "    else:\n"
            "        print(f\"Preparing xLAM-60k (1-call) splits into {DATA_DIR} via scripts/prepare_data.py ...\")\n"
            "        _prep_rc = subprocess.run([\n"
            "            sys.executable, os.path.join(str(PROJECT_ROOT), \"scripts\", \"prepare_data.py\"),\n"
            "            \"--sources\", \"xlam\",\n"
            "            \"--output_dir\", DATA_DIR,\n"
            "        ], cwd=str(PROJECT_ROOT)).returncode\n"
            "        if _prep_rc != 0:\n"
            "            raise SystemExit(f\"prepare_data.py --sources xlam exited {_prep_rc} (set HF_TOKEN if Salesforce/xlam-function-calling-60k is gated)\")\n"
            "        # v2.7 provenance note: template_stamp.json is directory-level, so the\n"
            "        # stamp now attests the xLAM splits prepare_data just wrote, not the\n"
            "        # Hermes splits realigned above. Gate A below still validates the\n"
            "        # template REVISION (the thing that invalidated v1), nothing more.\n"
            "        print(\"[Gate A / note] prepare_data.py re-stamped \" + DATA_DIR + \" -> Hermes splits are no longer attested by that stamp.\")",
        ),
        # Gate C: the v2.7 win is contingent on object params existing as dotted
        # leaves. Warning-level, not fatal: plan steps 2-3 (nested schema parsing
        # and the aligner object case) may not be applied yet, and a hard gate
        # would make the notebook unrunnable rather than merely honest.
        (
            "    if not _ok and not ALLOW_STALE_LABELS:\n"
            '        raise SystemExit(f"PRE-FLIGHT FAILED on {_split}")',
            "    if not _ok and not ALLOW_STALE_LABELS:\n"
            '        raise SystemExit(f"PRE-FLIGHT FAILED on {_split}")\n'
            "\n"
            "# Pre-Flight Gate C (v2.7, warning-only): is object decomposition present?\n"
            "# If the labels are still v2.6-shaped (one opaque `object` param, no dotted\n"
            "# leaves) then every 'object-leaf' figure in section 4d is structurally 0\n"
            "# and the v2.7 question cannot be answered by this run. Warn loudly instead\n"
            "# of aborting, because the label-side work is plan steps 2-3.\n"
            "from spancall.data.schema import UnifiedSample as _C_Sample\n"
            "from spancall.data.flatten import PATH_SEP as _C_SEP\n"
            "_c_roots = _c_leaves = _c_opaque = 0\n"
            '_c_path = os.path.join(DATA_DIR, f"{EVAL_SPLIT}.jsonl")\n'
            "if os.path.exists(_c_path):\n"
            "    with open(_c_path) as _fh_c:\n"
            "        for _i_c, _line_c in enumerate(_fh_c):\n"
            "            if _i_c >= 300:\n"
            "                break\n"
            "            if not _line_c.strip():\n"
            "                continue\n"
            "            _s_c = _C_Sample.from_dict(json.loads(_line_c))\n"
            "            for _t_c in _s_c.tools:\n"
            "                _c_roots += len(_t_c.object_paths or {})\n"
            "                for _p_c in _t_c.parameters:\n"
            "                    if _C_SEP in _p_c.name:\n"
            "                        _c_leaves += 1\n"
            '                    elif (_p_c.param_type or "").lower() == "object":\n'
            "                        _c_opaque += 1\n"
            "if _c_roots + _c_leaves + _c_opaque == 0:\n"
            '    print("[Gate C / Decompose] no object-bearing tool in "\n'
            "          + EVAL_SPLIT + \" - nothing to check.\")\n"
            "else:\n"
            "    _c_ok = _c_roots > 0 and _c_opaque == 0\n"
            '    print(f"[Gate C / Decompose] roots={_c_roots} leaves={_c_leaves} '
            'opaque={_c_opaque} -> {\'PASS\' if _c_ok else \'WARN\'}")\n'
            "    if not _c_ok:\n"
            "        print(\"  [!] labels look v2.6-shaped for object params: opaque `object`\")\n"
            "        print(\"      params outnumber dotted leaves. Section 4d will report zero\")\n"
            "        print(\"      object-leaf slots until plan steps 2-3 land (nested schema\")\n"
            "        print(\"      parsing + aligner object case). Do not read 4d as model quality.\")",
        ),
    ],
    # -- §3 resolver: silent untrained-model path ----------------------------
    "CODE_TELEMETRY_AND_RESOLVER": [
        (
            "    CHECKPOINT_DIR = os.path.dirname(cps[-1]) if cps else None",
            "    CHECKPOINT_DIR = os.path.dirname(cps[-1]) if cps else None\n"
            "if CHECKPOINT_DIR is None:\n"
            "    # LeanResolver(model_path=None) skips load_checkpoint entirely, so the\n"
            "    # eval below would score randomly initialised weights and still print\n"
            "    # plausible-looking numbers. Refuse instead.\n"
            "    raise SystemExit(\n"
            '        "no model.pt under " + exp_root + ": training persisted no checkpoint. "\n'
            '        "LeanResolver(model_path=None) would run UNTRAINED weights, so every "\n'
            '        "metric in sections 3-5 would be noise."\n'
            "    )",
        ),
    ],
    # -- §4c waterfall: flat vs nested key space ----------------------------
    "CODE_EVAL_STEP3": [
        (
            "from spancall.eval.toolcall.evaluator import payload_equal",
            "import spancall.data.flatten as _flat\n"
            "from spancall.eval.toolcall.evaluator import payload_equal",
        ),
        (
            "        _gt_t = next((t for t in _s.tools if t.name == p[\"gt_tool\"]), None)\n"
            "        if payload_equal(p.get(\"gt_args\") or {}, p.get(\"payload_partial\") or {}, _gt_t):\n"
            "            s2b_exact.append(p)",
            "        _gt_t = next((t for t in _s.tools if t.name == p[\"gt_tool\"]), None)\n"
            "        # v2.7: GT args are NESTED while ``payload_partial`` carries dotted\n"
            "        # leaf keys. Comparing them raw makes every object leaf a mismatch,\n"
            "        # which drops the whole waterfall one stage early. flatten_args is\n"
            "        # idempotent, so this holds before and after the resolver unflattens.\n"
            "        _paths = dict(getattr(_gt_t, \"object_paths\", None) or {})\n"
            "        _gt_flat = _flat.flatten_args(p.get(\"gt_args\") or {}, _paths)\n"
            "        _pred_flat = _flat.flatten_args(p.get(\"payload_partial\") or {}, _paths)\n"
            "        if payload_equal(_gt_flat, _pred_flat, _gt_t):\n"
            "            s2b_exact.append(p)",
        ),
    ],
    # -- §4e stage attribution: same key space, both comparisons ------------
    "CODE_EVAL_STEP4": [
        (
            "from spancall.eval.toolcall.evaluator import payload_equal",
            "import spancall.data.flatten as _flat\n"
            "from spancall.eval.toolcall.evaluator import payload_equal",
        ),
        (
            "            gt_t = next((t for t in s.tools if t.name == p[\"gt_tool\"]), None)\n"
            "            partial_ok = payload_equal(p.get(\"gt_args\") or {}, p.get(\"payload_partial\") or {}, gt_t)\n"
            "            if p.get(\"hit\") and payload_equal(p.get(\"gt_args\") or {}, p.get(\"payload\") or {}, gt_t):",
            "            gt_t = next((t for t in s.tools if t.name == p[\"gt_tool\"]), None)\n"
            "            # v2.7: both sides flattened into the declared object key space.\n"
            "            # A nested GT vs dotted prediction would otherwise attribute a\n"
            "            # reassembly-correct row to STEP2B (pointer rank miss).\n"
            "            _paths = dict(getattr(gt_t, \"object_paths\", None) or {})\n"
            "            _gt_flat = _flat.flatten_args(p.get(\"gt_args\") or {}, _paths)\n"
            "            _part_flat = _flat.flatten_args(p.get(\"payload_partial\") or {}, _paths)\n"
            "            partial_ok = payload_equal(_gt_flat, _part_flat, gt_t)\n"
            "            if p.get(\"hit\") and payload_equal(\n"
            "                _gt_flat, _flat.flatten_args(p.get(\"payload\") or {}, _paths), gt_t\n"
            "            ):",
        ),
    ],
    # -- §4f FP deep dive: self-contained + real schema + flat keys ---------
    "CODE_EVAL_STEP5": [
        (
            "from spancall.eval.toolcall.evaluator import payload_equal",
            "import spancall.data.flatten as _flat\n"
            "from spancall.eval.toolcall.evaluator import payload_equal",
        ),
        (
            "N_DUMP = 20  # number of FP rows to print in full\n\n"
            "with open(PREDS_PATH) as fh:\n"
            "    _preds = [json.loads(l) for l in fh if l.strip()]",
            "N_DUMP = 20  # number of FP rows to print in full\n\n"
            "# v2.7: self-contained. This cell used to close over PREDS_PATH and\n"
            "# raw_map left behind by 4e, so running or re-running it alone raised\n"
            "# NameError; and a stale PREDS_PATH from an earlier run silently scored\n"
            "# the wrong file. Recompute both here, same precedence as 4e.\n"
            'PREDS_PATH = os.path.join(RESULTS_DIR, "preds_eval_twostep.jsonl")\n'
            "if not os.path.exists(PREDS_PATH):\n"
            '    PREDS_PATH = os.path.join(RESULTS_DIR, "preds_eval.jsonl")\n'
            'if not os.path.exists(PREDS_PATH):\n'
            '    raise SystemExit(f"no predictions at {PREDS_PATH} - run 4c first.")\n'
            "raw_map, subs = {}, {}\n"
            "if os.path.exists(SUBSET_EVAL_PATH):\n"
            "    with open(SUBSET_EVAL_PATH) as _fh:\n"
            "        for _l in _fh:\n"
            "            if not _l.strip():\n"
            "                continue\n"
            "            _d = json.loads(_l)\n"
            "            raw_map[_d[\"sample_id\"]] = _d\n"
            "            subs[_d[\"sample_id\"]] = UnifiedSample.from_dict(_d)\n"
            "else:\n"
            '    print(f"[!] no eval subset at {SUBSET_EVAL_PATH} - cohort filter skipped.")\n'
            "\n"
            "with open(PREDS_PATH) as fh:\n"
            "    _preds = [json.loads(l) for l in fh if l.strip()]",
        ),
        (
            "    if p.get(\"hit\") and payload_equal(p.get(\"gt_args\") or {}, p.get(\"payload\") or {}, None):\n"
            "        continue\n"
            "    # This row passed Step 0/1/2a but failed payload equality -> precision FP.",
            "    # v2.7: pass the REAL tool schema (None skips optional-default\n"
            "    # materialisation, so a correct row that simply omitted an optional\n"
            "    # param was counted as a precision FP) and flatten both sides into the\n"
            "    # declared object key space (nested GT vs dotted payload).\n"
            "    _s = subs.get(sid)\n"
            "    _gt_t = None\n"
            "    if _s is not None:\n"
            "        _gt_t = next((t for t in _s.tools if t.name == p.get(\"gt_tool\")), None)\n"
            "    _paths = dict(getattr(_gt_t, \"object_paths\", None) or {})\n"
            "    _gt_flat = _flat.flatten_args(p.get(\"gt_args\") or {}, _paths)\n"
            "    p[\"_gt_flat\"] = _gt_flat  # flat GT, reused by the refusal sweep below\n"
            "    if p.get(\"hit\") and payload_equal(\n"
            "        _gt_flat, _flat.flatten_args(p.get(\"payload\") or {}, _paths), _gt_t\n"
            "    ):\n"
            "        continue\n"
            "    # This row passed Step 0/1/2a but failed payload equality -> precision FP.",
        ),
        (
            "    params = p.get(\"parameters\") or {}\n"
            "    gt_args = p.get(\"gt_args\") or {}",
            "    params = p.get(\"parameters\") or {}\n"
            "    gt_args = p.get(\"_gt_flat\") or {}  # flat, so dotted leaves line up",
        ),
        (
            "    print(f\"{p['sample_id']}  tool={p.get('tool_name')}  hit={p.get('hit')}  joint_margin={p.get('joint_margin'):.4f}\")",
            "    print(f\"{p['sample_id']}  tool={p.get('tool_name')}  hit={p.get('hit')}  \"\n"
            "          f\"joint_margin={float(p.get('joint_margin') or 0.0):.4f}\")",
        ),
        (
            "        gt_v = (p.get(\"gt_args\") or {}).get(pn, \"<NOT_IN_GT>\")\n"
            "        flag = \"OK\" if gt_v == pr.get(\"value\") else (\"NULL?\" if gt_v is None else \"WRONG\")",
            "        gt_v = (p.get(\"_gt_flat\") or {}).get(pn, \"<NOT_IN_GT>\")\n"
            "        flag = \"OK\" if gt_v == pr.get(\"value\") else (\"NULL?\" if gt_v is None else \"WRONG\")",
        ),
        (
            "conf={pr.get('confidence'):.3f} margin={pr.get('margin'):.3f}",
            "conf={float(pr.get('confidence') or 0.0):.3f} margin={float(pr.get('margin') or 0.0):.3f}",
        ),
        (
            "            for p in fp_rows:\n"
            "                gt_args = p.get(\"gt_args\") or {}",
            "            for p in fp_rows:\n"
            "                gt_args = p.get(\"_gt_flat\") or {}",
        ),
    ],
}


def _patch(name: str, src: str) -> str:
    """Apply the version bump and the v2.7 semantic fixes to one cell source.

    Order matters:
    1. the stale-download block is swapped *first*, while its exact text is intact;
    2. path tokens are bumped;
    3. banner prose is rewritten and any remaining "v2.6" stamp normalised;
    4. :data:`_V27_FIXES` runs **last**, so its needles match post-bump text and its
       replacements keep intentional v2.6 comparisons ("v2.6-shaped labels") verbatim
       instead of being rewritten by the step-3 regex.
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
    # 4. Per-cell semantic fixes. A needle that no longer matches means the v2.6
    #    source moved underneath this builder - fail loudly rather than shipping a
    #    notebook with a silently missing fix.
    for old, new in _V27_FIXES.get(name, ()):
        if old not in out:
            raise SystemExit(
                f"[_patch] v2.7 fix for {name} did not apply: needle not found\n"
                f"--- needle ---\n{old[:400]}"
            )
        out = out.replace(old, new)
    return out


#: v2.7 inserts a new evaluation section between 4c and 4d, so those two
#: sections shift down. Anchored on the heading + title text rather than on a
#: bare "4d"/"4e", because the numbers also appear in body prose.
_SECTION_RENUMBER = [
    (re.compile(r"(?m)^(#{2,3}\s*)4d\.(\s*Stage-Attributed Error Forensics)"),
     r"\g<1>4e.\g<2>"),
    (re.compile(r"(?m)^(#{2,3}\s*)4e\.(\s*Param-Precision FP Deep Dive)"),
     r"\g<1>4f.\g<2>"),
]


def _patch_paths(src: str) -> str:
    """Bump v2_6 path tokens and renumber shifted sections in prose.

    Used for markdown cells: they legitimately contain *comparative* prose
    ("In v2.6 an object value fell through the scalar branch ..."), which must
    survive verbatim, while a path such as ``data/processed_v2/`` must not.

    Order matters: renumbering runs before nothing else touches the headings,
    and the patterns are applied 4d -> 4e then 4e -> 4f, so a heading moved to
    4e in step 1 would be re-matched in step 2. Renumber the *last* section
    first to avoid that cascade.
    """
    out = src
    for pattern, new in _VERSION_RULES:
        out = pattern.sub(new, out)
    for pattern, new in reversed(_SECTION_RENUMBER):
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