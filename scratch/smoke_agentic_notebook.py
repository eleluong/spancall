#!/usr/bin/env python3
"""Execute the generated agentic notebook's code cells offline.

The cell guard proves the cells *compile*; this proves they *run*. Two cells are
stubbed so the smoke test stays hermetic:

* the ``%pip`` / ``!git`` install cell is skipped (IPython magics, and it would
  reinstall the package mid-test);
* ``huggingface_hub.hf_hub_download`` is made to fail, which drives the model
  cell into its documented "no artifact -> NoOpResolver" branch. That branch is
  worth exercising on its own: it is what a reader sees if they never pushed the
  model, and it must not crash the notebook.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat  # noqa: E402

NB = ROOT / "notebooks" / "spancall_lean_v3_2_agentic.ipynb"


def is_ipython(src: str) -> bool:
    return any(line.lstrip().startswith(("!", "%", "%%")) for line in src.splitlines())


def main() -> int:
    nb = nbformat.read(NB, as_version=4)
    ns: dict = {"__name__": "__main__"}

    import huggingface_hub

    def _no_network(*a, **k):
        raise RuntimeError("smoke test: network disabled")

    huggingface_hub.hf_hub_download = _no_network  # type: ignore[assignment]

    ran = skipped = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source
        if is_ipython(src):
            skipped += 1
            print(f"[skip {i}] IPython cell")
            continue
        print(f"\n{'=' * 78}\n[cell {i}]\n{'=' * 78}")
        exec(compile(src, f"<cell {i}>", "exec"), ns)
        ran += 1

    print(f"\n[smoke] exec'd {ran} cell(s), skipped {skipped}")

    # ---- assertions on what the cells actually produced ------------------ #
    rows = ns.get("TABLE_ROWS") or []
    assert rows, "no TABLE_ROWS produced"
    labels = {r["mode"] for r in rows}
    assert labels == {"mode1_llm_only", "mode2_fallback", "mode3_hybrid_fill"}, labels
    suites = {r["suite"] for r in rows}
    assert suites == {"canonical", "long_trajectory"}, suites
    print(f"[smoke] {len(rows)} table rows over suites={sorted(suites)} modes={sorted(labels)}")

    assert ns["RESOLVER_INFO"]["kind"] == "noop", ns["RESOLVER_INFO"]
    assert ns["BACKEND_KIND"] == "deterministic", ns["BACKEND_KIND"]

    # mode 1 must produce LLM decisions; with a NoOp resolver mode 2 matches it.
    m1 = next(r for r in rows if r["mode"] == "mode1_llm_only" and r["suite"] == "canonical")
    assert m1["decisions"].get("llm_fallback", 0) > 0, m1["decisions"]

    # The scripted-partial demo is the one place HYBRID_FILL must fire.
    s3 = ns["SCRIPTED_TRACES"]["mode3_hybrid_fill"]
    fills = [s for s in s3.steps if s.decision.value == "hybrid_fill"]
    print(f"[smoke] scripted mode-3 HYBRID_FILL steps: {len(fills)}")
    for s in fills:
        f = s.spancall_features
        print(f"        known={f.get('known_params')} requested={f.get('requested_params')} "
              f"filled={f.get('filled_params')} args={s.tool_arguments}")
    out_path = Path(ns["RESULTS_DIR"]) / "agentic_mode_table.json"
    assert out_path.is_file(), out_path
    with open(out_path, encoding="utf-8") as fh:
        json.load(fh)
    print(f"[smoke] wrote/validated {out_path}")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())