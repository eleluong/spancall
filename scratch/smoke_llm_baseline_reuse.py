#!/usr/bin/env python3
"""Offline smoke test for the REUSE path of cell 3 in
`notebooks/spancall_lean_v3_2_llm_baseline.ipynb`.

Verifies, with no network and no GPU:
  1. `_find_existing_preds` picks the v3.2 file when present.
  2. `_load_preds` keys by str(sample_id).
  3. `_validate_coverage` passes when every subset row is covered.
  4. `_validate_coverage` raises SystemExit when a row is missing
     (this is the guard that stops the two arms being scored on
     different populations).
  5. `LLM_LIMIT` trims the coverage check to the same prefix.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

NB = REPO / "notebooks" / "spancall_lean_v3_2_llm_baseline.ipynb"


def _cell3_source() -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return "".join(code[3]["source"])


def _cell3_header() -> str:
    """The real cell-3 preamble: the imports, `RESULTS_V3_2`, `_SEARCH_DIRS`,
    `CANDIDATE_PRED_PATHS` and `_find_existing_preds` - but not `_load_preds`
    or the execution loop. Slicing the shipped cell keeps this test honest."""
    lines = _cell3_source().splitlines()
    cut = next(i for i, l in enumerate(lines) if l.startswith("def _load_preds"))
    return "\n".join(lines[:cut]) + "\n"


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _pred(sid: str, hit: bool, tool: str, stratum: str = "F") -> dict:
    return {
        "sample_id": sid,
        "stratum": stratum,
        "hit": hit,
        "tool_name": tool if hit else None,
        "payload": {"city": "Hanoi"} if hit else None,
        "latency_ms": 12.5,
        "status": "hit" if hit else "abstain",
    }


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_reuse_"))
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    try:
        # ---- synthetic layout -------------------------------------------- #
        results_v32 = tmp / "results_lean_v3_2"
        subset_dir = tmp / "subsets"
        results_out = tmp / "results_lean_v3_2_llm"
        results_out.mkdir(parents=True)

        hermes_ids = ["h1", "h2", "h3"]
        xlam_ids = ["x1", "x2"]

        _write_jsonl(
            subset_dir / "subset_hermes_eval_clean.jsonl",
            [{"sample_id": s} for s in hermes_ids],
        )
        _write_jsonl(
            subset_dir / "subset_xlam_eval_clean.jsonl",
            [{"sample_id": s} for s in xlam_ids],
        )
        # hermes: the primary candidate path (preds_eval_twostep.jsonl)
        _write_jsonl(
            results_v32 / "preds_eval_twostep.jsonl",
            [_pred("h1", True, "get_weather"), _pred("h2", False, ""),
             _pred("h3", True, "book_flight"), _pred("h_extra", True, "x")],
        )
        # xlam: the split-specific candidate path
        _write_jsonl(
            results_v32 / "preds_xlam_eval_twostep.jsonl",
            [_pred("x1", True, "search"), _pred("x2", True, "play")],
        )

        # ---- execute cell 3 ---------------------------------------------- #
        ns = {
            "PROJECT_ROOT": str(tmp),
            "RESULTS_DIR": str(results_out),
            "V3_2_RESULTS_DIR": None,     # exercises the in-repo default
            "EVAL_SPLITS": ["hermes_eval", "xlam_eval"],
            "SUBSET_PATHS": {
                "hermes_eval": str(subset_dir / "subset_hermes_eval_clean.jsonl"),
                "xlam_eval": str(subset_dir / "subset_xlam_eval_clean.jsonl"),
            },
            "LLM_LIMIT": None,
            "EVAL_BATCH_SIZE": 32,
            "EVAL_WORKERS": 4,
        }
        # RESULTS_V3_2 is derived from PROJECT_ROOT inside the cell; make the
        # synthetic dir land exactly where the cell expects it.
        shutil.move(str(results_v32), str(tmp / "results_lean_v3_2"))

        print("[exec] cell 3 (reuse path)")
        exec(compile(_cell3_source(), "<cell3>", "exec"), ns)

        preds = ns["SPANCALL_PREDS"]
        source = ns["SPANCALL_SOURCE"]

        print("\n[assert] outcome")
        check(set(preds) == {"hermes_eval", "xlam_eval"},
              "both splits produced a prediction dict")
        check(len(preds["hermes_eval"]) == 4,
              "hermes _load_preds loaded all 4 rows (extra pred kept)")
        check(set(preds["xlam_eval"]) == {"x1", "x2"},
              "xlam keys are the str(sample_id)s")
        check(source["hermes_eval"].startswith("reused:"),
              f"hermes marked reused -> {source['hermes_eval']}")
        check("preds_eval_twostep.jsonl" in source["hermes_eval"],
              "hermes resolved the primary candidate path, not the glob")
        check(source["xlam_eval"].startswith("reused:"),
              "xlam marked reused")

        # ---- coverage guard fires on a genuinely missing row -------------- #
        print("\n[assert] coverage guard")
        try:
            ns["_validate_coverage"](
                "broken",
                str(subset_dir / "subset_hermes_eval_clean.jsonl"),
                {"h1": {}},
            )
        except SystemExit as e:
            ok = "have NO prediction" in str(e)
            check(ok, f"SystemExit raised for missing rows: {str(e)[:60]}...")
        else:
            check(False, "SystemExit raised for missing rows")

        # ---- LLM_LIMIT trims the checked prefix -------------------------- #
        print("\n[assert] LLM_LIMIT prefix trim")
        ns["LLM_LIMIT"] = 2
        try:
            ns["_validate_coverage"](
                "trimmed_ok",
                str(subset_dir / "subset_hermes_eval_clean.jsonl"),
                {"h1": {}, "h2": {}},
            )
            check(True, "limit=2 passes when only h1,h2 are covered")
        except SystemExit as e:
            check(False, f"limit=2 should not require h3: {e}")
        try:
            ns["_validate_coverage"](
                "trimmed_bad",
                str(subset_dir / "subset_hermes_eval_clean.jsonl"),
                {"h1": {}},
            )
            check(False, "limit=2 should still fail when h2 is missing")
        except SystemExit:
            check(True, "limit=2 still fails when h2 is missing")

        # ---- V3_2_RESULTS_DIR redirects the lookup ------------------------ #
        # Run the REAL cell-3 header (everything up to `_load_preds`) in a
        # fresh namespace, so this tests the shipped code rather than a copy.
        header = _cell3_header()

        def _fresh(root: str) -> dict:
            _ns = {
                "V3_2_RESULTS_DIR": root,
                "PROJECT_ROOT": str(tmp),
                "os": os, "json": json, "glob": __import__("glob"),
            }
            exec(compile(header, "<cell3-header>", "exec"), _ns)
            return _ns

        print("\n[assert] V3_2_RESULTS_DIR redirect")
        extern = tmp / "drive_mount" / "results_lean_v3_2"
        _write_jsonl(extern / "preds_eval_twostep.jsonl",
                     [_pred("h1", True, "get_weather")])
        ns2 = _fresh(str(extern))
        found = ns2["_find_existing_preds"]("hermes_eval")
        check(found is not None and str(extern) in found,
              f"knob redirects lookup to the Drive mount (got {found})")
        check(os.path.normpath(ns2["RESULTS_V3_2"]) == os.path.normpath(str(extern)),
              "RESULTS_V3_2 honours the knob")

        # ---- xLAM must NOT silently borrow the hermes file ---------------- #
        print("\n[assert] no cross-split aliasing")
        lone = tmp / "only_hermes"
        _write_jsonl(lone / "preds_eval_twostep.jsonl", [_pred("h1", True, "x")])
        ns3 = _fresh(str(lone))
        ns3["_SEARCH_DIRS"] = [str(lone)]      # drop the in-repo default
        xfound = ns3["_find_existing_preds"]("xlam_eval")
        check(xfound is None,
              f"xlam does not alias the hermes preds file (got {xfound})")
        hfound = ns3["_find_existing_preds"]("hermes_eval")
        check(hfound is not None, "hermes still resolves in that directory")

        # ---- no preds anywhere -> None, not an exception ------------------ #
        print("\n[assert] clean miss returns None")
        ns4 = _fresh(str(tmp / "empty_root"))
        ns4["_SEARCH_DIRS"] = [str(tmp / "empty_root")]
        check(ns4["_find_existing_preds"]("hermes_eval") is None,
              "absent predictions return None (caller warns + re-infers)")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL REUSE-PATH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())