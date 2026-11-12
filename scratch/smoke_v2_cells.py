#!/usr/bin/env python3
"""Execute the W4 notebook's new cells against real data.

The cells are extracted from the BUILT notebook so this exercises exactly the
code that ships, not the builder's string literals.

W5 is run twice -- once with a pre-flight artefact that FAILS its audit and once
with one that passes -- because the whole value of that cell is whether it
suppresses the branch recommendation when the labels have not been validated.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NB = ROOT / "notebooks" / "spancall_lean_v2_train_eval.ipynb"


def code_cells() -> List[str]:
    with open(NB, encoding="utf-8") as fh:
        nb = json.load(fh)
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def find(sources: List[str], needle: str) -> str:
    for s in sources:
        if needle in s:
            return s
    raise SystemExit(f"cell containing {needle!r} not found")


def base_ns(results_dir: str) -> Dict[str, Any]:
    return {
        "PROJECT_ROOT": ROOT,
        "DATA_DIR": str(ROOT / "data" / "processed_v2"),
        "V1_DATA_DIR": str(ROOT / "data" / "processed"),
        "RESULTS_DIR": results_dir,
        "V1_RESULTS_DIR": str(ROOT / "results" / "lean" / "12-11-26"),
        "EVAL_SPLIT": "hermes_eval",
        "MAX_UTT_STALE_RATE": 0.02,
        "MAX_UNRESOLVABLE": 0.02,
        "MIN_JOINT_MARGIN": 0.02,
        "Dict": Dict,
        "Any": Any,
        "List": List,
        "__builtins__": __builtins__,
    }


def run(src: str, ns: Dict[str, Any], label: str) -> Dict[str, Any]:
    print("=" * 78)
    print(f"CELL: {label}")
    print("=" * 78)
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(src, f"<{label}>", "exec"), ns)
    print(buf.getvalue().rstrip())
    print()
    return ns


def main() -> int:
    srcs = code_cells()
    print(f"{len(srcs)} code cells in the built notebook\n")

    tmp = Path(tempfile.mkdtemp(prefix="smoke_v2_"))
    shutil.copy(
        ROOT / "results" / "lean" / "12-11-26" / "preds_eval.jsonl",
        tmp / "preds_eval.jsonl",
    )

    calib_src = find(srcs, 'calibration_v2.json')
    v1v2_src = find(srcs, 'v1_vs_v2.json')
    w5_src = find(srcs, 'tool_accuracy')

    # ---- CALIBRATION: full sweep on the frozen v1 preds -----------------
    run(calib_src, base_ns(str(tmp)), "CALIBRATION")

    # ---- V1V2: label + prediction tables -------------------------------
    run(v1v2_src, base_ns(str(tmp)), "V1V2")

    # ---- W5 (a): pre-flight artefact present and PASSING ---------------
    (tmp / "preflight_v2.json").write_text(json.dumps({
        "stamp": {"status": "ok", "ok": True},
        "audits": {
            "hermes_eval": {"ok": True, "utterance_span_stale_rate": 0.0},
            "hermes_train": {"ok": True, "utterance_span_stale_rate": 0.0},
            "hermes_dev": {"ok": True, "utterance_span_stale_rate": 0.0},
        },
    }))
    run(w5_src, base_ns(str(tmp)), "W5 (pre-flight PASSES)")

    # ---- W5 (b): pre-flight artefact present but FAILING ----------------
    (tmp / "preflight_v2.json").write_text(json.dumps({
        "stamp": {"status": "ok", "ok": True},
        "audits": {
            "hermes_eval": {"ok": False, "utterance_span_stale_rate": 0.6},
        },
    }))
    run(w5_src, base_ns(str(tmp)), "W5 (pre-flight FAILS)")

    # ---- W5 (c): no artefact at all ------------------------------------
    (tmp / "preflight_v2.json").unlink()
    run(w5_src, base_ns(str(tmp)), "W5 (no pre-flight artefact)")

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())