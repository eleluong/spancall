#!/usr/bin/env python3
"""Offline smoke test for cell 4's checkpoint / resume path.

Executes the REAL cell 4 from
`notebooks/spancall_lean_v3_2_llm_baseline.ipynb` against a fake OpenAI client
and asserts the properties that decide whether a paid run is safe to keep:

  1. first run fetches every row exactly once and writes the merged artifact
  2. a second run reuses the checkpoint: 0 API calls
  3. a crash mid-run loses nothing already paid for; the re-run fetches only
     the missing rows and the result is byte-identical to an uninterrupted run
  4. a failed call is NOT treated as complete, so it is retried next run
  5. changing the model changes the fingerprint, so a resume cannot splice two
     runs together
  6. `LLM_RESUME=0` backs up instead of destroying the previous artifacts

No network, no API key: `client` is injected.
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


def _code_cells() -> list:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


_TOOL = {
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "parameters": [
        {"name": "city", "description": "City name.", "param_type": "string",
         "required": True},
    ],
    "risk_tier": "read_only",
}


def _row(sid: str) -> dict:
    return {
        "sample_id": sid,
        "source_dataset": "hermes",
        "utterance": f"weather in {sid}",
        "tools": [_TOOL],
        "ground_truth_calls": [
            {"tool_name": "get_weather", "arguments": {"city": sid}, "call_index": 0}
        ],
        "recent_context": [],
        "current_time_iso": "2025-06-01T12:00:00Z",
        "stratum": "F",
        "target_spans": {},
        "split": "test_id",
        "metadata": {},
    }


class _Fn:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, name: str, arguments: str):
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, calls):
        self.tool_calls = calls
        self.content = ""


class _Choice:
    def __init__(self, calls):
        self.message = _Msg(calls)


class _Usage:
    prompt_tokens = 100
    completion_tokens = 20


class _Resp:
    def __init__(self, calls):
        self.choices = [_Choice(calls)]
        self.usage = _Usage()


class _Completions:
    """`client.chat.completions.create`, with real nesting so bookkeeping survives."""

    def __init__(self, owner):
        self._o = owner

    def create(self, *, messages, **_kw):
        o = self._o
        text = messages[-1]["content"]
        o.calls.append(text)
        if any(f in text for f in o.fail_on):
            raise RuntimeError("simulated 500 from provider")
        if o.crash_after is not None and o.completions >= o.crash_after:
            raise KeyboardInterrupt("simulated interruption")
        o.completions += 1
        sid = next((t for t in ("s1", "s2", "s3", "s4", "s5") if f"in {t}" in text), "s1")
        return _Resp([_TC("get_weather", json.dumps({"city": sid}))])


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class FakeClient:
    """Records every call; can fail on demand or crash after N completions."""

    def __init__(self, fail_on=(), crash_after=None):
        self.calls = []          # rendered prompts seen, in order
        self.fail_on = set(fail_on)
        self.crash_after = crash_after
        self.completions = 0     # successful completions served
        self.chat = _Chat(self)


def _ns(tmp: Path, results: Path, subset: Path, client, **over):
    """A namespace holding the REAL cell-0 helpers plus cell-4's dependencies."""
    cells = _code_cells()
    c0 = "".join(cells[0]["source"])
    cut = c0.index("def _limit_rows")
    ns = {
        "os": __import__("os"),
        "json": json,
        # Cell 4 loads `cmp` from PROJECT_ROOT/scripts, so this must stay the
        # real repo; every temp path is injected explicitly below.
        "PROJECT_ROOT": str(REPO),
        "RESULTS_DIR": str(results),
        "EVAL_SPLITS": ["hermes_eval"],
        "SUBSET_PATHS": {"hermes_eval": str(subset)},
        "LLM_MODEL": "fake-model",
        "LLM_BASE_URL": "",
        "LLM_API_KEY": "sk-fake",
        "LLM_TEMPERATURE": 0.0,
        "LLM_TIMEOUT": 5.0,
        "LLM_MAX_RETRIES": 0,
        "LLM_LIMIT_FRACTION": 1.0,
        "LLM_RESUME": True,
        "LLM_CHECKPOINT_EVERY": 2,
        "LLM_SOURCE": {},      # cell 0 declares it; only a slice of cell 0 is exec'd
        "client": client,
    }
    ns.update(over)
    exec(compile(c0[cut:c0.index("# Print shapes")], "<c0>", "exec"), ns)
    return ns


def _read(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _run_cell4(ns) -> None:
    src = "".join(_code_cells()[4]["source"])
    src = src.replace("from openai import OpenAI", "")      # injected instead
    # Keep the client we injected; the cell would otherwise build a real one.
    src = src.replace(
        "client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None,\n"
        "                timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)", "")
    exec(compile(src, "<cell4>", "exec"), ns)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_ckpt_"))
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    try:
        results = tmp / "results_lean_v3_2_llm"
        results.mkdir(parents=True)
        subset = tmp / "subset_hermes_eval_clean.jsonl"
        ids = ["s1", "s2", "s3", "s4", "s5"]
        _write_jsonl(subset, [_row(s) for s in ids])
        out = results / "preds_hermes_eval_llm.jsonl"

        # ---- 1. first run pays for every row ----------------------------- #
        print("\n[run 1] cold start")
        c1 = FakeClient()
        ns1 = _ns(tmp, results, subset, c1)
        _run_cell4(ns1)
        got = _read(out)
        check(c1.completions == 5, f"cold run made 5 API calls (got {c1.completions})")
        check([r["sample_id"] for r in got] == ids,
              "artifact holds every row, in subset order")
        check(all("_fingerprint" not in r for r in got),
              "internal _fingerprint is stripped from the artifact")
        ckpts = list(results.glob("preds_hermes_eval_llm.ckpt-*.jsonl"))
        check(len(ckpts) == 1, f"exactly one checkpoint written ({len(ckpts)})")

        # ---- 2. second run reuses it for free ---------------------------- #
        print("\n[run 2] resume a complete run")
        c2 = FakeClient()
        ns2 = _ns(tmp, results, subset, c2)
        _run_cell4(ns2)
        check(c2.completions == 0,
              f"complete checkpoint reused with 0 API calls (got {c2.completions})")
        check("checkpoint reuse" in ns2["LLM_SOURCE"]["hermes_eval"],
              f"LLM_SOURCE records reuse -> {ns2['LLM_SOURCE']['hermes_eval']}")
        again = _read(out)
        check([r["sample_id"] for r in again] == ids, "reused artifact still complete")

        # ---- 3. crash mid-run, then resume ------------------------------- #
        print("\n[run 3] crash after 3 completions, then resume")
        for p in results.glob("preds_hermes_eval_llm.*"):
            p.unlink()
        crashing = FakeClient(crash_after=3)
        ns3 = _ns(tmp, results, subset, crashing)
        try:
            _run_cell4(ns3)
        except KeyboardInterrupt:
            pass
        else:
            check(False, "the simulated crash propagated (did not)")
        check(crashing.completions == 3,
              f"crash happened after 3 paid calls (got {crashing.completions})")
        ck = next(results.glob("preds_hermes_eval_llm.ckpt-*.jsonl"))
        saved = _read(ck)
        check(len(saved) == 3,
              f"checkpoint kept the 3 paid rows across the crash (got {len(saved)})")
        # The mid-run atomic snapshot must be in final shape even after the
        # crash: no internal bookkeeping key, rows in subset order.
        mid = _read(out)
        check(len(mid) >= 2,
              f"mid-run snapshot was written before the crash (got {len(mid)} rows)")
        check(all("_fingerprint" not in r for r in mid),
              "mid-run artifact is _fingerprint-free (same shape as a clean finish)")
        check([r["sample_id"] for r in mid] == ids[:len(mid)],
              "mid-run artifact rows are in subset order")

        c4 = FakeClient()
        ns4 = _ns(tmp, results, subset, c4)
        _run_cell4(ns4)
        check(c4.completions == 2,
              f"resume fetched ONLY the 2 missing rows (got {c4.completions})")
        check("resumed" in ns4["LLM_SOURCE"]["hermes_eval"],
              f"LLM_SOURCE records the partial resume -> {ns4['LLM_SOURCE']['hermes_eval']}")
        resumed = _read(out)
        check([r["sample_id"] for r in resumed] == ids,
              "resumed artifact is complete and in subset order")

        def _shape(rows):
            return [(r["sample_id"], r["hit"], r.get("tool_name"),
                     json.dumps(r.get("payload"), sort_keys=True)) for r in rows]

        check(_shape(resumed) == _shape(got),
              "resumed predictions are identical to an uninterrupted run")

        # ---- 4. a failed call is retried, not frozen --------------------- #
        print("\n[run 4] provider error is retried on the next run")
        for p in results.glob("preds_hermes_eval_llm.*"):
            p.unlink()
        failing = FakeClient(fail_on=("s3",))
        ns5 = _ns(tmp, results, subset, failing)
        _run_cell4(ns5)
        rows5 = {r["sample_id"]: r for r in _read(out)}
        check(bool((rows5["s3"].get("features") or {}).get("llm_error")),
              "the failing row records its error (stays in the denominator)")
        c6 = FakeClient()
        ns6 = _ns(tmp, results, subset, c6)
        _run_cell4(ns6)
        check(c6.completions == 1,
              f"only the failed row is re-fetched (got {c6.completions})")
        rows6 = {r["sample_id"]: r for r in _read(out)}
        check(not (rows6["s3"].get("features") or {}).get("llm_error"),
              "the retried row is now a real result")

        # ---- 5. a different model must not reuse the checkpoint ---------- #
        print("\n[run 5] model change invalidates the checkpoint")
        c7 = FakeClient()
        ns7 = _ns(tmp, results, subset, c7, LLM_MODEL="another-model")
        _run_cell4(ns7)
        check(c7.completions == 5,
              f"a new model re-fetches every row (got {c7.completions})")
        check(len(list(results.glob("preds_hermes_eval_llm.ckpt-*.jsonl"))) == 2,
              "a separate checkpoint is used per fingerprint")

        # ---- 6. LLM_RESUME=0 backs up rather than destroys --------------- #
        print("\n[run 6] LLM_RESUME=0")
        before = {p.name for p in results.glob("preds_hermes_eval_llm.*")}
        c8 = FakeClient()
        ns8 = _ns(tmp, results, subset, c8, LLM_RESUME=False)
        _run_cell4(ns8)
        check(c8.completions == 5, f"forced run re-fetches everything (got {c8.completions})")
        backups = list(results.glob("*.bak-*"))
        check(len(backups) >= 1, f"previous artifacts kept as .bak-* ({len(backups)})")
        check(bool(before), "there was something to back up")

        # ---- 7. a torn checkpoint line does not corrupt the resume -------- #
        print("\n[run 7] torn final checkpoint line")
        for p in results.glob("preds_hermes_eval_llm.*"):
            p.unlink()
        _run_cell4(_ns(tmp, results, subset, FakeClient()))
        ck = next(results.glob("preds_hermes_eval_llm.ckpt-*.jsonl"))
        text = ck.read_text(encoding="utf-8")
        ck.write_text(text[:-17], encoding="utf-8")      # chop the last row mid-JSON
        c9 = FakeClient()
        ns9 = _ns(tmp, results, subset, c9)
        _run_cell4(ns9)
        check(c9.completions == 1,
              f"a torn line costs exactly one re-fetched row (got {c9.completions})")
        check([r["sample_id"] for r in _read(out)] == ids,
              "artifact is still complete after recovering from a torn line")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKPOINT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())