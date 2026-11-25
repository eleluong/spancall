#!/usr/bin/env python3
"""Generate notebooks/spancall_lean_v3_1_train_eval.ipynb via nbformat (valid JSON guaranteed).

SpanCall Lean Flow **v3.1** — Dual-Mode Array, Object Decomposition + Literal Dicts,
Pointer-Favored Loss, and Batched Parallel Evaluation.

Derived from ``scripts/build_lean_v3_colab_notebook.py`` using the same
slice-and-patch pattern so v3 stays untouched and v3.1 writes to its own
``data/processed_v3_1``, ``data/tensorized_lean_v3_1``, and ``results_lean_v3_1``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
V3_SCRIPT = HERE / "build_lean_v3_colab_notebook.py"
OUT = HERE / ".." / "notebooks" / "spancall_lean_v3_1_train_eval.ipynb"

# --------------------------------------------------------------------------
# Reuse the v3 builder's declaration section (v2.6 + v2.7 + v3)
# --------------------------------------------------------------------------
_V3_SRC = V3_SCRIPT.read_text(encoding="utf-8")
_V3_START = _V3_SRC.index("import json")
_V3_END = _V3_SRC.index("cells = [")
_V3: Dict[str, Any] = {
    "__name__": "build_lean_v3_colab_notebook_sources",
    "__file__": str(V3_SCRIPT),
}
exec(
    compile(
        "from __future__ import annotations\n" + _V3_SRC[_V3_START:_V3_END],
        str(V3_SCRIPT),
        "exec",
    ),
    _V3,
)

_V26_NS: Dict[str, Any] = _V3["_V26_NS"]
_V27: Dict[str, Any] = _V3["_V27"]
_V3_PATCH = _V3["_patch"]
_V3_PATCH_PATHS = _V3["_patch_paths"]

# --------------------------------------------------------------------------
# v3.1 path & banner rules
# --------------------------------------------------------------------------
_V31_VERSION_RULES = [
    (re.compile(r"processed_v2_7"), "processed_v3_1"),
    (re.compile(r"tensorized_lean_v3(?!_1)"), "tensorized_lean_v3_1"),
    (re.compile(r"results_lean_v3(?!_1)"), "results_lean_v3_1"),
    (re.compile(r"colab_lean_v3\.yaml"), "colab_lean_v3_1.yaml"),
    (re.compile(r"spancall-lean-v3(?!-1)"), "spancall-lean-v3-1"),
]

_V31_REPLACEMENTS = [
    ("LEAN FLOW v3 CONFIG", "LEAN FLOW v3.1 CONFIG"),
    ("# SpanCall Lean v3 Run Summary", "# SpanCall Lean v3.1 Run Summary"),
    ("# SpanCall Lean v3 —", "# SpanCall Lean v3.1 —"),
    ('"version": "v3"', '"version": "v3.1"'),
    ("SpanCall Lean v3 (", "SpanCall Lean v3.1 ("),
    (
        "# 0b. Configuration & Environment (Full Data, v3 two-step only)",
        "# 0b. Configuration & Environment (Full Data, v3.1)",
    ),
    ("[v3] {len(_missing_v2)} of", "[v3.1] {len(_missing_v2)} of"),
]

MARKDOWN_TITLE = """# SpanCall Lean Flow **v3.1** — Dual-Mode Array, Object Upgrade, Pointer-Favored Loss & Batched Eval

End-to-end training and evaluation notebook for **v3.1**, building on the v3 two-step-only stratified protocol:

- **Dual-mode array extraction.** `CandidateMaskBuilder` emits balanced `[...]` whole-list spans + `items_type`-dispatched element candidates; `ReverseAligner` aligns nested/empty/scalar arrays; `LeanJointDecoder.select_list_choices` enforces strict overlap exclusion, `max_items=32`, and **document-order sorting**.
- **Object decomposition + literal dicts.** `realign_v2.py` upgrades legacy opaque `object` schemas into dotted leaves, `json_mode` decomposes nested dicts, and `ReverseAligner` / `CandidateMaskBuilder` / `StrictTypeCaster` support verbatim `{...}` dict literals.
- **Pointer-favored loss (`H3` > `H1`).** `OptimizationConfig` defaults to `lambda_pointer=2.0`, `lambda_tool=0.2`, `pointer_focal_gamma=1.0`, and `multi_pos_slot_weight=1.5` (while keeping logged `eval_pointer_loss_micro` on the raw CE scale).
- **Batched & parallel evaluation.** `LeanResolver` runs threaded CPU prep + length-sorted batched GPU encoding (`EVAL_BATCH_SIZE=32`, `EVAL_WORKERS=4`) across §4a, §4b, §4c, and §5, with in-process `eval_toolcall_main` scoring.

**Paths:** `data/processed_v3_1`, `data/tensorized_lean_v3_1`, `results_lean_v3_1`."""

_V31_FIXES: Dict[str, Any] = {
    "CODE_CONFIG": [
        (
            "EVAL_LIMIT        = 2000\n"
            "TEST_LIMIT        = 600",
            "EVAL_LIMIT        = 2000\n"
            "TEST_LIMIT        = 600\n"
            "EVAL_BATCH_SIZE   = 32\n"
            "EVAL_WORKERS      = 4",
        ),
    ],
    "CODE_EVAL_STEP1": [
        (
            "    for d in raw_dicts:\n"
            "        cohort = classify_eval_cohort(d)\n"
            "        s = UnifiedSample.from_dict(d)\n"
            "        if not s.ground_truth_calls or not s.tools:\n"
            "            continue\n"
            "        gt_name = s.ground_truth_calls[0].tool_name\n"
            "        native_multi = len(s.tools) >= 2\n"
            "        candidates = list(s.tools)\n"
            "        if not native_multi and pool_names:\n"
            "            seen = {t.name for t in candidates}\n"
            '            salt = zlib.crc32(f"step1::{SEED}::{s.sample_id}".encode("utf-8"))\n'
            "            rng = random.Random(salt)\n"
            "            for pick in rng.sample(pool_names, min(len(pool_names), pack_k + 3)):\n"
            "                if len(candidates) >= pack_k:\n"
            "                    break\n"
            "                if pick not in seen:\n"
            "                    candidates.append(pool[pick])\n"
            "                    seen.add(pick)\n"
            "            rng.shuffle(candidates)\n"
            "        if len(candidates) < 2:\n"
            "            continue\n\n"
            "        pred = resolver.predict_tool(candidates, s.utterance, s.recent_context, s.current_time_iso)",
            "    prepared = []\n"
            "    for d in raw_dicts:\n"
            "        cohort = classify_eval_cohort(d)\n"
            "        s = UnifiedSample.from_dict(d)\n"
            "        if not s.ground_truth_calls or not s.tools:\n"
            "            continue\n"
            "        gt_name = s.ground_truth_calls[0].tool_name\n"
            "        native_multi = len(s.tools) >= 2\n"
            "        candidates = list(s.tools)\n"
            "        if not native_multi and pool_names:\n"
            "            seen = {t.name for t in candidates}\n"
            '            salt = zlib.crc32(f"step1::{SEED}::{s.sample_id}".encode("utf-8"))\n'
            "            rng = random.Random(salt)\n"
            "            for pick in rng.sample(pool_names, min(len(pool_names), pack_k + 3)):\n"
            "                if len(candidates) >= pack_k:\n"
            "                    break\n"
            "                if pick not in seen:\n"
            "                    candidates.append(pool[pick])\n"
            "                    seen.add(pick)\n"
            "            rng.shuffle(candidates)\n"
            "        if len(candidates) < 2:\n"
            "            continue\n"
            "        prepared.append((cohort, s, gt_name, native_multi, candidates))\n\n"
            '    if hasattr(resolver, "predict_tools_batch"):\n'
            "        preds_step1 = resolver.predict_tools_batch(\n"
            "            [(cands, s.utterance, s.recent_context, s.current_time_iso) for _, s, _, _, cands in prepared],\n"
            "            batch_size=EVAL_BATCH_SIZE, num_workers=EVAL_WORKERS,\n"
            "        )\n"
            "    else:\n"
            "        preds_step1 = [\n"
            "            resolver.predict_tool(cands, s.utterance, s.recent_context, s.current_time_iso)\n"
            "            for _, s, _, _, cands in prepared\n"
            "        ]\n\n"
            "    for (cohort, s, gt_name, native_multi, candidates), pred in zip(prepared, preds_step1):",
        ),
    ],
    "CODE_EVAL_STEP3": [
        (
            "import json, os, subprocess, sys",
            "import json, os, subprocess, sys\n"
            "from scripts.eval_toolcall import main as eval_toolcall_main",
        ),
        (
            '    _supports_parity = bool(getattr(resolver, "supports_candidate_parity", False))\n'
            "    with open(src) as fh:\n"
            "        for line in fh:\n"
            "            line = line.strip()\n"
            "            if not line:\n"
            "                continue\n"
            "            raw_dict = json.loads(line)\n"
            "            s = UnifiedSample.from_dict(raw_dict)\n"
            "            _p_spans = None\n"
            "            if _supports_parity and s.ground_truth_calls:\n"
            "                _gt0 = s.ground_truth_calls[0]\n"
            "                _raw_ps = _parity_spans_for_call(s, 0)\n"
            "                if _raw_ps:\n"
            '                    _p_spans = {f"{_gt0.tool_name}::{k}": v for k, v in _raw_ps.items()}\n'
            "            if _supports_parity:\n"
            "                res = resolver.try_resolve_candidates(\n"
            "                    s.tools, s.utterance, s.recent_context, s.current_time_iso,\n"
            "                    parity_spans=_p_spans, protect_parity_spans=protect_parity_spans,\n"
            "                    resolve_mode=resolve_mode,\n"
            "                )\n"
            "            else:\n"
            "                res = resolver.try_resolve_candidates(s.tools, s.utterance, s.recent_context, s.current_time_iso)",
            '    _supports_parity = bool(getattr(resolver, "supports_candidate_parity", False))\n'
            "    _raw_items = []\n"
            "    with open(src) as fh:\n"
            "        for line in fh:\n"
            "            line = line.strip()\n"
            "            if not line:\n"
            "                continue\n"
            "            raw_dict = json.loads(line)\n"
            "            s = UnifiedSample.from_dict(raw_dict)\n"
            "            _p_spans = None\n"
            "            if _supports_parity and s.ground_truth_calls:\n"
            "                _gt0 = s.ground_truth_calls[0]\n"
            "                _raw_ps = _parity_spans_for_call(s, 0)\n"
            "                if _raw_ps:\n"
            '                    _p_spans = {f"{_gt0.tool_name}::{k}": v for k, v in _raw_ps.items()}\n'
            "            _raw_items.append((raw_dict, s, _p_spans))\n"
            "            if limit is not None and len(_raw_items) >= limit:\n"
            "                break\n\n"
            '    if _raw_items and hasattr(resolver, "resolve_candidates_batch") and resolve_mode in (None, "two_step"):\n'
            "        _batch_in = [\n"
            "            {\n"
            '                "candidate_schemas": s.tools,\n'
            '                "utterance": s.utterance,\n'
            '                "recent_context": s.recent_context,\n'
            '                "now": s.current_time_iso,\n'
            '                "parity_spans": _p_spans if _supports_parity else None,\n'
            "            }\n"
            "            for _, s, _p_spans in _raw_items\n"
            "        ]\n"
            "        _batch_res = resolver.resolve_candidates_batch(\n"
            "            _batch_in,\n"
            "            batch_size=EVAL_BATCH_SIZE,\n"
            "            num_workers=EVAL_WORKERS,\n"
            "            protect_parity_spans=protect_parity_spans,\n"
            "        )\n"
            "    else:\n"
            "        _batch_res = [\n"
            "            (\n"
            "                resolver.try_resolve_candidates(\n"
            "                    s.tools, s.utterance, s.recent_context, s.current_time_iso,\n"
            "                    parity_spans=_p_spans, protect_parity_spans=protect_parity_spans,\n"
            "                    resolve_mode=resolve_mode,\n"
            "                )\n"
            "                if _supports_parity\n"
            "                else resolver.try_resolve_candidates(s.tools, s.utterance, s.recent_context, s.current_time_iso)\n"
            "            )\n"
            "            for _, s, _p_spans in _raw_items\n"
            "        ]\n\n"
            "    for (raw_dict, s, _p_spans), res in zip(_raw_items, _batch_res):",
        ),
        (
            "            preds.append(rec)\n"
            "            subset_samples.append(raw_dict)\n"
            "            n += 1\n"
            "            if limit is not None and n >= limit:\n"
            "                break",
            "            preds.append(rec)\n"
            "            subset_samples.append(raw_dict)\n"
            "            n += 1",
        ),
        (
            "    subprocess.run([\n"
            '        sys.executable, "scripts/eval_toolcall.py",\n'
            '        "--test_data", SUBSET_EVAL_PATH,\n'
            "        *_args,\n"
            '        "--output", os.path.join(RESULTS_DIR, _out),\n'
            "    ], check=True)",
            "    eval_toolcall_main([\n"
            '        "--test_data", SUBSET_EVAL_PATH,\n'
            "        *_args,\n"
            '        "--output", os.path.join(RESULTS_DIR, _out),\n'
            "    ])",
        ),
    ],
    "CODE_TEST_AND_PERSIST": [
        (
            '    for label, extra in [("b1a", ["--baseline", "b1a"]), ("lean", ["--predictions", os.path.join(RESULTS_DIR, f"preds_{split}.jsonl")])]:\n'
            "        subprocess.run([\n"
            '            sys.executable, "scripts/eval_toolcall.py",\n'
            '            "--test_data", data_file,\n'
            "            *extra,\n"
            '            "--output", os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_{label}.json"),\n'
            "        ], check=False)",
            '    for label, extra in [("b1a", ["--baseline", "b1a"]), ("lean", ["--predictions", os.path.join(RESULTS_DIR, f"preds_{split}.jsonl")])]:\n'
            "        eval_toolcall_main([\n"
            '            "--test_data", data_file,\n'
            "            *extra,\n"
            '            "--output", os.path.join(RESULTS_DIR, f"toolcall_metrics_{split}_{label}.json"),\n'
            "        ])",
        ),
    ],
    "CODE_CORPUS_EVAL": [
        (
            '            _cmd = [sys.executable, "scripts/eval_toolcall.py", "--test_data", csub]\n'
            '            if label == "b1a":\n'
            '                _cmd += ["--baseline", "b1a"]\n'
            "            else:\n"
            '                _cmd += ["--predictions", cpred]\n'
            '            _cmd += ["--output", out_json]\n'
            "            subprocess.run(_cmd, check=False)",
            '            _cmd = ["--test_data", csub]\n'
            '            if label == "b1a":\n'
            '                _cmd += ["--baseline", "b1a"]\n'
            "            else:\n"
            '                _cmd += ["--predictions", cpred]\n'
            '            _cmd += ["--output", out_json]\n'
            "            eval_toolcall_main(_cmd)",
        ),
    ],
}


def _patch(name: str, src: str) -> str:
    out = _V3_PATCH(name, src)
    for pattern, new in _V31_VERSION_RULES:
        out = pattern.sub(new, out)
    for old, new in _V31_REPLACEMENTS:
        if old in out:
            out = out.replace(old, new)
    for old, new in _V31_FIXES.get(name, ()):
        if old not in out:
            raise SystemExit(
                f"[_patch] v3.1 fix for {name} did not apply: needle not found\n"
                f"--- needle ---\n{old[:500]}"
            )
        out = out.replace(old, new)
    return out


def _patch_standalone(name: str, src: str) -> str:
    """Patch v3-authored cells (not coming from _V26_NS)."""
    out = src
    for pattern, new in _V31_VERSION_RULES:
        out = pattern.sub(new, out)
    for old, new in _V31_REPLACEMENTS:
        if old in out:
            out = out.replace(old, new)
    return out


def _patch_paths(src: str) -> str:
    out = _V3_PATCH_PATHS(src)
    for pattern, new in _V31_VERSION_RULES:
        out = pattern.sub(new, out)
    return out


_CODE_EVAL_STEP2_V31 = _V3["CODE_EVAL_STEP2"].replace(
    "context_mode=\"single\", stratum_filter=\"F\", exclude_tools=[\"json_output\"], quiet=True,",
    "context_mode=\"single\", stratum_filter=\"F\", exclude_tools=[\"json_output\"], quiet=True,\n    batch_size=EVAL_BATCH_SIZE, num_workers=EVAL_WORKERS,",
)

_CODE_TOOL_BUCKET_PIVOT_V31 = (
    _V3["CODE_TOOL_BUCKET_PIVOT"]
    .replace(
        "from collections import defaultdict",
        "from collections import defaultdict\nfrom scripts.eval_toolcall import main as eval_toolcall_main",
    )
    .replace(
        '_cmd = [sys.executable, "scripts/eval_toolcall.py", "--test_data", _bsub]\n'
        '            _cmd += ["--baseline", "b1a"] if _label == "b1a" else ["--predictions", _bp]\n'
        '            _cmd += ["--output", _out]\n'
        "            subprocess.run(_cmd, check=False)",
        '_argv = ["--test_data", _bsub]\n'
        '            _argv += ["--baseline", "b1a"] if _label == "b1a" else ["--predictions", _bp]\n'
        '            _argv += ["--output", _out]\n'
        "            eval_toolcall_main(_argv)",
    )
)

_MARKDOWN: Dict[str, str] = {
    _name: _patch_paths(_V26_NS[_name])
    for _name in (
        "MARKDOWN_DATA",
        "MARKDOWN_TENSORIZE",
        "MARKDOWN_EVAL_SLICE",
        "MARKDOWN_EVAL_STEP1",
        "MARKDOWN_EVAL_STEP4",
        "MARKDOWN_EVAL_STEP5",
        "MARKDOWN_TEST",
        "MARKDOWN_CORPUS_EVAL",
        "MARKDOWN_SUMMARY_REPORT",
    )
}

cells = [
    new_markdown_cell(MARKDOWN_TITLE),
    new_markdown_cell(_patch_paths(_V3["MARKDOWN_SETUP"])),
    new_code_cell(_patch("CODE_SETUP", _V26_NS["CODE_SETUP"])),
    new_code_cell(_patch("CODE_CONFIG", _V26_NS["CODE_CONFIG"])),
    new_code_cell(_patch("CODE_RUNTIME_CHECK", _V26_NS["CODE_RUNTIME_CHECK"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_DATA"]),
    new_code_cell(_patch("CODE_DATA_AND_PREFLIGHT", _V26_NS["CODE_DATA_AND_PREFLIGHT"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_TENSORIZE"]),
    new_code_cell(_patch("CODE_TENSORIZE", _V26_NS["CODE_TENSORIZE"])),
    new_markdown_cell(_patch_paths(_V3["MARKDOWN_TRAIN"])),
    new_code_cell(_patch("CODE_TRAIN_CFG_AND_RUN", _V26_NS["CODE_TRAIN_CFG_AND_RUN"])),
    new_code_cell(_patch("CODE_TELEMETRY_AND_RESOLVER", _V26_NS["CODE_TELEMETRY_AND_RESOLVER"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_EVAL_SLICE"]),
    new_code_cell(_patch("CODE_EVAL_SLICE", _V26_NS["CODE_EVAL_SLICE"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_EVAL_STEP1"]),
    new_code_cell(_patch("CODE_EVAL_STEP1", _V26_NS["CODE_EVAL_STEP1"])),
    new_markdown_cell(_patch_paths(_V3["MARKDOWN_EVAL_STEP2"])),
    new_code_cell(_patch_standalone("CODE_EVAL_STEP2", _CODE_EVAL_STEP2_V31)),
    new_markdown_cell(_patch_paths(_V3["MARKDOWN_EVAL_STEP3"])),
    new_code_cell(_patch("CODE_EVAL_STEP3", _V26_NS["CODE_EVAL_STEP3"])),
    new_markdown_cell(_patch_paths(_V27["MARKDOWN_OBJECT_AUDIT"])),
    new_code_cell(_patch("CODE_OBJECT_AUDIT", _V27["CODE_OBJECT_AUDIT"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_EVAL_STEP4"]),
    new_code_cell(_patch("CODE_EVAL_STEP4", _V26_NS["CODE_EVAL_STEP4"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_EVAL_STEP5"]),
    new_code_cell(_patch("CODE_EVAL_STEP5", _V26_NS["CODE_EVAL_STEP5"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_TEST"]),
    new_code_cell(_patch("CODE_TEST_AND_PERSIST", _V26_NS["CODE_TEST_AND_PERSIST"])),
    new_markdown_cell(_MARKDOWN["MARKDOWN_CORPUS_EVAL"]),
    new_code_cell(_patch("CODE_CORPUS_EVAL", _V26_NS["CODE_CORPUS_EVAL"])),
    new_markdown_cell(_patch_paths(_V3["MARKDOWN_TOOL_PIVOT"])),
    new_code_cell(_patch_standalone("CODE_TOOL_BUCKET_PIVOT", _CODE_TOOL_BUCKET_PIVOT_V31)),
    new_markdown_cell(_MARKDOWN["MARKDOWN_SUMMARY_REPORT"]),
    new_code_cell(_patch("CODE_SUMMARY_REPORT", _V26_NS["CODE_SUMMARY_REPORT"])),
    new_markdown_cell(_patch_paths(_V3["MARKDOWN_HF_EXPORT"])),
    new_code_cell(_patch_standalone("CODE_HF_EXPORT", _V3["CODE_HF_EXPORT"])),
]

_V26_SCRIPT = HERE / "build_lean_v2_6_colab_notebook.py"
_V26_SRC = _V26_SCRIPT.read_text(encoding="utf-8")
_GUARD_START = _V26_SRC.index("def _cell_source")
_GUARD_END = _V26_SRC.rindex("_guard_code_cells(cells)")
_GUARD_NS: Dict[str, Any] = {"__name__": "build_lean_v3_1_guard"}
exec(compile(_V26_SRC[_GUARD_START:_GUARD_END], str(_V26_SCRIPT), "exec"), _GUARD_NS)
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
