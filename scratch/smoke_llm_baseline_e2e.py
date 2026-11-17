#!/usr/bin/env python3
"""Offline end-to-end smoke test for cells 5, 6 and 7 of
`notebooks/spancall_lean_v3_2_llm_baseline.ipynb`.

Builds a synthetic 4-row subset (3 stratum-F, 1 stratum-A), fake SpanCall
predictions in the v3.2 preds shape, and fake LLM predictions shaped by the
real `cmp._llm_pred_row`. Then executes the notebook's scoring, complexity and
persist cells unchanged and asserts on the numbers.

No network, no GPU, no API key.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

NB = REPO / "notebooks" / "spancall_lean_v3_2_llm_baseline.ipynb"
CMP_PATH = REPO / "scripts" / "compare_llm_spancall.py"


def _load_cmp():
    spec = importlib.util.spec_from_file_location("compare_llm_spancall", CMP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cells() -> list:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# Canonical `ToolDefinition` shape: `parameters` is a LIST of ParameterSchema
# dicts (not an OpenAI JSON-Schema object). This is what the split files
# store, and what `_tool_definitions_to_openai` converts from.
_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "parameters": [
        {"name": "city", "description": "City name.", "param_type": "string",
         "required": True},
    ],
    "risk_tier": "read_only",
}

_FLIGHT_TOOL = {
    "name": "book_flight",
    "description": "Book a flight.",
    "parameters": [
        {"name": "origin", "description": "Origin airport.", "param_type": "string",
         "required": True},
        {"name": "destination", "description": "Destination airport.",
         "param_type": "string", "required": True},
        {"name": "passengers", "description": "Passenger names.",
         "param_type": "array", "required": False, "items_type": "string"},
    ],
    "risk_tier": "side_effecting",
}


def _sample_row(sid: str, utterance: str, gt_tool: str, gt_args: dict,
                tools, stratum: str) -> dict:
    return {
        "sample_id": sid,
        "source_dataset": "hermes",
        "utterance": utterance,
        "tools": tools,
        "ground_truth_calls": [
            {"tool_name": gt_tool, "arguments": gt_args, "call_index": 0}
        ] if gt_tool else [],
        "recent_context": [],
        "current_time_iso": "2025-06-01T12:00:00Z",
        "stratum": stratum,
        "target_spans": {},
        "split": "test_id",
        "metadata": {},
    }


def main() -> int:
    cmp = _load_cmp()
    code = _cells()
    tmp = Path(tempfile.mkdtemp(prefix="smoke_e2e_"))
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    try:
        subset_dir = tmp / "subsets"
        results_out = tmp / "results_lean_v3_2_llm"
        results_out.mkdir(parents=True)

        # 3 stratum-F rows (2 scalar/normal, 1 array/complex) + 1 stratum-A row.
        rows = [
            _sample_row("h1", "What's the weather in Hanoi?", "get_weather",
                        {"city": "Hanoi"}, [_WEATHER_TOOL], "F"),
            _sample_row("h2", "Weather in Da Nang please.", "get_weather",
                        {"city": "Da Nang"}, [_WEATHER_TOOL], "F"),
            _sample_row("h3", "Book a flight from SGN to HAN for an and binh.",
                        "book_flight",
                        {"origin": "SGN", "destination": "HAN",
                         "passengers": ["an", "binh"]},
                        [_FLIGHT_TOOL], "F"),
            _sample_row("a1", "Do the thing with the stuff.", None, {},
                        [_WEATHER_TOOL, _FLIGHT_TOOL], "A"),
        ]
        subset_path = subset_dir / "subset_hermes_eval_clean.jsonl"
        _write_jsonl(subset_path, rows)

        samples = [cmp.UnifiedSample.from_dict(r) for r in rows]
        ids = [s.sample_id for s in samples]

        # ---- SpanCall preds: h1 + h3 exact, h2 wrong payload, a1 abstains --- #
        sp_preds = {
            "h1": {"sample_id": "h1", "stratum": "F", "hit": True,
                   "tool_name": "get_weather", "payload": {"city": "Hanoi"},
                   "latency_ms": 11.0, "status": "hit"},
            "h2": {"sample_id": "h2", "stratum": "F", "hit": True,
                   "tool_name": "get_weather", "payload": {"city": "Hue"},
                   "latency_ms": 12.0, "status": "hit"},
            "h3": {"sample_id": "h3", "stratum": "F", "hit": True,
                   "tool_name": "book_flight",
                   "payload": {"origin": "SGN", "destination": "HAN",
                               "passengers": ["an", "binh"]},
                   "latency_ms": 13.0, "status": "hit"},
            # A row: correct behaviour is abstain
            "a1": {"sample_id": "a1", "stratum": "A", "hit": False,
                   "tool_name": None, "payload": None,
                   "latency_ms": 9.0, "status": "abstain"},
        }

        # ---- LLM preds: built through the real _llm_pred_row ---------------- #
        # Shape matches what cell 4 constructs (notebook line 82):
        #   {"tool_name": fn.name, "payload": dict(args)}
        # h1 exact, h2 wrong payload, h3 misses the array arg, a1 answers anyway.
        llm_specs = {
            "h1": [{"tool_name": "get_weather", "payload": {"city": "Hanoi"}}],
            "h2": [{"tool_name": "get_weather", "payload": {"city": "Hue"}}],
            "h3": [{"tool_name": "book_flight",
                    "payload": {"origin": "SGN", "destination": "HAN"}}],
            "a1": [{"tool_name": "get_weather", "payload": {"city": "Hanoi"}}],
        }
        llm_preds = {}
        for sid, calls in llm_specs.items():
            resp = cmp.LLMResponse(
                tool_calls=calls,
                text="",
                prompt_tokens=100,
                completion_tokens=20,
                latency_ms=850.0,
                error=None,
            )
            llm_preds[sid] = cmp._llm_pred_row(sid, resp)

        # ---- namespace the cells expect ------------------------------------- #
        ns = {
            "cmp": cmp,
            "os": __import__("os"),
            "json": json,
            "PROJECT_ROOT": str(tmp),
            "RESULTS_DIR": str(results_out),
            "HF_REPO_ID": "convoicon/spancall-lean-v3-2",
            "HF_DATASET_ID": "convoicon/spancall",
            "EVAL_SPLITS": ["hermes_eval"],
            "SUBSET_PATHS": {"hermes_eval": str(subset_path)},
            "SPANCALL_PREDS": {"hermes_eval": sp_preds},
            "SPANCALL_SOURCE": {"hermes_eval": "reused: synthetic"},
            "LLM_PREDS": {"hermes_eval": llm_preds},
            "LLM_MODEL": "fake-hosted-llm",
            "LLM_BASE_URL": "",          # cell 7 records this; blank = SDK default
            "LLM_TEMPERATURE": 0.0,
            "LLM_LIMIT": None,
            "SEED": 42,
            "MIN_JOINT_MARGIN": 0.02,
            "_base_url": None,
        }

        def run(idx: int, label: str) -> None:
            print(f"\n[exec] cell {idx} ({label})")
            src = "".join(code[idx]["source"])
            exec(compile(src, f"<cell{idx}>", "exec"), ns)

        run(5, "scoring")
        run(6, "normal/complex")
        run(7, "persist")

        # ---------------- assertions ---------------------------------------- #
        print("\n[assert] scoring")
        systems = ns["SYSTEMS"]["hermes_eval"]
        check(set(systems) == {"spancall_two_step", "llm:fake-hosted-llm"},
              f"two systems scored: {sorted(systems)}")

        sp = systems["spancall_two_step"]
        llm = systems["llm:fake-hosted-llm"]

        sp_f = sp["metrics_by_stratum"]["F"]
        llm_f = llm["metrics_by_stratum"]["F"]

        check(int(sp_f["n"]) == 3, f"SpanCall n_F == 3 (got {sp_f['n']})")
        check(int(llm_f["n"]) == 3, f"LLM n_F == 3 (got {llm_f['n']})")
        check(abs(sp_f["exact_payload_match"] - 2 / 3) < 1e-9,
              f"SpanCall exact_F == 2/3 (got {sp_f['exact_payload_match']})")
        check(abs(llm_f["exact_payload_match"] - 1 / 3) < 1e-9,
              f"LLM exact_F == 1/3 (got {llm_f['exact_payload_match']})")
        check(int(sp["total_samples"]) == 4 and int(llm["total_samples"]) == 4,
              "both systems scored all 4 rows including stratum A")

        # The A row: SpanCall abstains (0.0), the LLM answers anyway (1.0).
        check(abs(float(sp["false_hit_rate_on_a"]) - 0.0) < 1e-9,
              f"SpanCall false_hit_A == 0.0 (got {sp['false_hit_rate_on_a']})")
        check(abs(float(llm["false_hit_rate_on_a"]) - 1.0) < 1e-9,
              f"LLM false_hit_A == 1.0 (got {llm['false_hit_rate_on_a']})")

        # Latency + token bookkeeping.
        check(abs(sp["timing"]["mean_latency_ms"] - 11.25) < 1e-6,
              f"SpanCall mean latency 11.25ms (got {sp['timing']['mean_latency_ms']})")
        check(llm["cost"]["n_llm_calls"] == 4, "LLM cost counts 4 calls")
        check(llm["cost"]["prompt_tokens"] == 400, "LLM prompt tokens summed (400)")
        check(sp["cost"]["n_llm_calls"] == 0, "SpanCall reports 0 LLM calls")
        check("pivot" in sp and len(sp["pivot"]) >= 1,
              f"pivot cells built: {sorted(sp['pivot'])}")

        print("\n[assert] complexity")
        cx = ns["COMPLEXITY"]["hermes_eval"]
        check(set(cx) == {"normal", "complex"}, f"both cohorts populated: {sorted(cx)}")
        # h1,h2,a1 -> normal (scalar-only); h3 -> complex (array arg)
        check(int(cx["normal"]["spancall_two_step"]["total_samples"]) == 3,
              "normal cohort has 3 rows")
        check(int(cx["complex"]["spancall_two_step"]["total_samples"]) == 1,
              "complex cohort has 1 row")

        print("\n[assert] persist")
        out = results_out / "llm_parity.json"
        check(out.is_file(), f"llm_parity.json written -> {out}")
        payload = json.loads(out.read_text(encoding="utf-8"))
        check(payload["meta"]["n_rows"]["hermes_eval"] == 4,
              f"meta n_rows is post-trim row count (got {payload['meta']['n_rows']})")
        check(payload["meta"]["spancall_source"] == {"hermes_eval": "reused: synthetic"},
              "meta records how the SpanCall side was obtained")
        check(payload["meta"]["llm_model"] == "fake-hosted-llm",
              "meta records the LLM model")
        check("systems" in payload and "complexity" in payload,
              "payload carries systems + complexity")
        check(isinstance(payload["systems"]["hermes_eval"]
                         ["spancall_two_step"]["metrics_by_stratum"]["F"]["n"], (int, float)),
              "JSON is serialisable (default=str handled enums)")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL END-TO-END CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())