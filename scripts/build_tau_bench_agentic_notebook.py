#!/usr/bin/env python3
"""Generate notebooks/spancall_tau_bench_agentic_eval.ipynb.

Colab/local notebook for:
1. Benchmarking SpanCall v3.3 ModernBERT-large attention kernels (sdpa vs flash_attention_2).
2. Running Stage 2 Simple τ-bench Retail evaluation (LLM-only, Router+Schema, HybridFill+Schema, Shadow).
3. Running Stage 3 per-turn failure-mode diagnosis to guide Stage 4 training.
4. Optional hook for official `tau-bench` evaluation via TauBenchAgentAdapter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _md(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def _code(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


CELLS: List[Dict[str, Any]] = [
    _md(
        """# SpanCall v3.3 — Optimized Attention & Simple τ-bench Agentic Eval

This notebook implements **Stage 1 → Stage 2 → Stage 3** of `plans/agent_tau_bench_plan.md`:
1. **Environment & Kernel Check:** Verify GPU/MPS, optionally install `flash-attn` on Ampere+ CUDA, and confirm `resolve_attn_implementation("auto")`.
2. **Kernel Micro-Benchmark:** Compare `sdpa` vs `flash_attention_2` latency & peak VRAM on `convoicon/spancall-lean-v3-3` (`ModernBERT-large`).
3. **Stage 2 — Simple τ-bench Retail (10 Multi-Turn Tasks):** Run paired `llm_only`, `router_schema`, `hybrid_fill_schema`, and `shadow` evaluations and export per-turn JSONL traces.
4. **Stage 3 — Failure-Mode Diagnosis:** Inspect concrete failure turns (`shadow_false_hit_on_prose`, `duplicate_blocked`, `context_overflow_seq`, `tool_exec_error`) to guide Stage 4 training.
5. **Optional Official `tau-bench` Hook:** Ready-to-run cell for official `tau_bench` (`retail` / `airline`) via `TauBenchAgentAdapter`.
"""
    ),
    _md("## 1. Environment Setup & Optional FlashAttention-2"),
    _code(
        """import os
import subprocess
import sys
from pathlib import Path

# Set to True on Colab A100/L4/H100 (SM >= 8.0) if you want FlashAttention-2 unpadding.
INSTALL_FLASH_ATTN = False

# If running in Colab without the repo cloned yet, clone or cd into your workspace here.
REPO_DIR = Path.cwd()
if not (REPO_DIR / "spancall").is_dir():
    print("Tip: cd into the multi_span_tool_calling repo root before running.")
else:
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

if INSTALL_FLASH_ATTN:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flash-attn", "--no-build-isolation"])

import torch
import transformers
from spancall.model.encoder import (
    _cuda_supports_flash_attn,
    _flash_attn_available,
    resolve_attn_implementation,
    resolve_torch_dtype,
)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), "SM:", torch.cuda.get_device_capability(0))
print("torch:", torch.__version__, "| transformers:", transformers.__version__)
print("flash_attn installed:", _flash_attn_available(), "| cuda_supports_flash_attn:", _cuda_supports_flash_attn(device=device))
print("auto attn_implementation:", resolve_attn_implementation("auto", device=device))
print("auto torch_dtype:", resolve_torch_dtype("auto", attn_implementation=resolve_attn_implementation("auto", device=device), device=device))
"""
    ),
    _md("## 2. Configuration & API Credentials (No Hardcoded Secrets)"),
    _code(
        """from getpass import getpass
from spancall.agent.backends.litellm_adapter import _load_dotenv_values

# --- SpanCall checkpoint & kernel config ---
SPANCALL_CHECKPOINT = "convoicon/spancall-lean-v3-3"  # Hub repo or local checkpoint dir
ATTN_IMPLEMENTATION = "auto"   # "auto" | "flash_attention_2" | "sdpa" | "eager"
TORCH_DTYPE         = None     # None keeps fp32 heads (v3.3 parity); FA2 auto-uses bf16 on encoder only

# --- Agent & τ-bench evaluation config ---
BACKEND        = "openai"      # "openai" (live LLM) | "deterministic" (offline smoke)
MODES          = ["llm_only", "router_schema", "hybrid_fill_schema", "shadow"]
SKIP_HIGH_RISK = True          # Route risk_tier='high' write tools to LLM for confirmation/policy
MAX_TURNS      = 12
TASK_LIMIT     = 0             # 0 = all 10 Simple τ-bench Retail tasks
OUT_DIR        = Path("results/agentic/tau_retail_v3_3")

# Load .env if present; prompt only if BACKEND == "openai" and values are missing.
env_vals = _load_dotenv_values(".env")
BASE_URL = os.environ.get("BASE_URL") or env_vals.get("BASE_URL") or ""
API_KEY = os.environ.get("API_KEY") or env_vals.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
LLM_MODEL = os.environ.get("BASE_LLM_MODEL") or env_vals.get("BASE_LLM_MODEL") or ""

if BACKEND == "openai":
    if not BASE_URL:
        BASE_URL = input("OpenAI-compatible BASE_URL (leave blank for api.openai.com): ").strip()
    if not LLM_MODEL:
        LLM_MODEL = input("LLM_MODEL (e.g. gpt-4o-mini, google/gemma-4-26b-a4b-it): ").strip()
    if not API_KEY:
        API_KEY = getpass("API_KEY: ").strip()

print(f"Checkpoint={SPANCALL_CHECKPOINT} | attn={ATTN_IMPLEMENTATION} | dtype={TORCH_DTYPE} | backend={BACKEND} | model={LLM_MODEL or 'deterministic'}")
"""
    ),
    _md("## 3. Attention Kernel Micro-Benchmark (`sdpa` vs `flash_attention_2`)"),
    _code(
        """import statistics
import time
from spancall.agent.tools.tau_retail import make_tau_retail_tools
from spancall.hub import load_resolver

# Compare available kernels on the 9-tool τ-bench retail schema.
kernels_to_test = ["sdpa"]
if _cuda_supports_flash_attn(device=device):
    kernels_to_test.append("flash_attention_2")

retail_tools = make_tau_retail_tools()
bench_utterance = "Hi, I'm Yusuf Rossi (zip 10003). Can you check the status of my order #W1001?"
bench_recent = [
    {"key": "t0_find_user_id_by_name_zip_user_id", "value": "yusuf_rossi_9620", "source": "tool_output"},
]

bench_rows = []
for kernel in kernels_to_test:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    res_obj = load_resolver(
        SPANCALL_CHECKPOINT,
        device=device,
        attn_implementation=kernel,
        torch_dtype=TORCH_DTYPE,
    )
    # Warmup
    for _ in range(3):
        _ = res_obj.try_resolve_candidates(retail_tools, bench_utterance, recent_context=bench_recent)
    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    last_out = None
    for _ in range(15):
        t0 = time.perf_counter()
        last_out = res_obj.try_resolve_candidates(retail_tools, bench_utterance, recent_context=bench_recent)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    bench_rows.append({
        "kernel": res_obj.encoder.attn_implementation,
        "dtype": str(res_obj.encoder.resolved_torch_dtype),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "peak_vram_mb": round(peak_mb, 1),
        "hit": last_out.hit,
        "tool": last_out.tool_name,
        "payload": last_out.payload,
    })
    del res_obj

for r in bench_rows:
    print(r)

if len(bench_rows) >= 2:
    assert bench_rows[0]["hit"] == bench_rows[1]["hit"], f"Hit mismatch: {bench_rows}"
    assert bench_rows[0]["tool"] == bench_rows[1]["tool"], f"Tool mismatch: {bench_rows}"
    assert bench_rows[0]["payload"] == bench_rows[1]["payload"], f"Payload mismatch: {bench_rows}"
    print("Parity OK: sdpa and flash_attention_2 produced identical hit/tool/payload.")
"""
    ),
    _md("## 4. Stage 2 — Run Simple τ-bench Retail Suite & Export Traces"),
    _code(
        """import json
from scripts.eval_tau_bench import (
    DeterministicTauRetailBackend,
    _format_markdown_table,
    run_simple_tau_retail,
    serialize_trace,
)
from spancall.agent.backends.openai_adapter import OpenAIAPIBackend
from spancall.agent.tools.tau_retail import build_simple_tau_retail_suite
from spancall.hub import load_resolver

resolver = load_resolver(
    SPANCALL_CHECKPOINT,
    device=device,
    attn_implementation=ATTN_IMPLEMENTATION,
    torch_dtype=TORCH_DTYPE,
)
print(
    f"Loaded resolver: attn={resolver.encoder.attn_implementation}, "
    f"dtype={resolver.encoder.resolved_torch_dtype}, max_seq_len={resolver.max_seq_len}"
)

if BACKEND == "openai":
    backend = OpenAIAPIBackend(
        model_id=LLM_MODEL,
        api_base=BASE_URL or None,
        api_key=API_KEY or None,
    )
else:
    backend = DeterministicTauRetailBackend()

tasks = build_simple_tau_retail_suite()
if TASK_LIMIT and TASK_LIMIT > 0:
    tasks = tasks[:TASK_LIMIT]

summaries, traces_by_mode = run_simple_tau_retail(
    resolver=resolver,
    backend=backend,
    modes=MODES,
    tasks=tasks,
    skip_risk_tiers=("high",) if SKIP_HIGH_RISK else (),
    max_turns=MAX_TURNS,
)

table_md = _format_markdown_table(summaries)
print(table_md)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
(OUT_DIR / "summary.md").write_text(table_md + "\\n", encoding="utf-8")
for mode, m_traces in traces_by_mode.items():
    p = OUT_DIR / f"traces_{mode}.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for tr in m_traces:
            fh.write(json.dumps(serialize_trace(tr), ensure_ascii=False) + "\\n")
print(f"Saved traces & summary to {OUT_DIR}")
"""
    ),
    _md("## 5. Stage 3 — Diagnose Concrete Failure Modes"),
    _code(
        """# Inspect failed tasks and shadow false-hits on prose turns to guide Stage 4 training.
for mode, m_traces in traces_by_mode.items():
    failed = [t for t in m_traces if not t.task_success]
    print(f"\\n=== Mode: {mode} | Failed tasks: {len(failed)}/{len(m_traces)} ===")
    for t in failed:
        print(f"  FAILED {t.task_id} ({t.metadata.get('category')}):")
        for s in t.steps:
            feats = s.spancall_features or {}
            print(
                f"    turn={s.turn_index} decision={s.decision.value} tool={s.selected_tool} "
                f"ok={s.tool_execution_success} gate={feats.get('router_gate_reason')} "
                f"abstain={feats.get('spancall_abstain_reason')} args={s.tool_arguments}"
            )

    if mode == "shadow":
        false_hits = [
            (t.task_id, s.turn_index, s.user_input, s.spancall_features)
            for t in m_traces
            for s in t.steps
            if (s.spancall_features or {}).get("shadow_false_hit_on_prose")
        ]
        print(f"  Shadow false_hit_on_prose turns: {len(false_hits)}")
        for tid, tidx, utt, feats in false_hits[:10]:
            print(f"    [{tid} turn={tidx}] utt={utt[:80]!r}")
"""
    ),
    _md("## 6. (Optional) Official `tau-bench` Runner Hook"),
    _code(
        """# Uncomment to install official tau-bench and run TauBenchAgentAdapter on retail/airline:
# !pip install -q git+https://github.com/sierra-research/tau-bench.git
#
# from tau_bench.envs import get_env
# from spancall.agent.router import IntegrationVariant, RouterMode, ToolCallRouter
# from spancall.agent.tools.tau_retail import TauBenchAgentAdapter
#
# env = get_env(
#     env_name="retail",
#     user_strategy="llm",
#     user_model=LLM_MODEL,
#     user_provider="openai",
#     task_split="test",
# )
# router = ToolCallRouter(
#     resolver=resolver,
#     llm_backend=backend,
#     mode=RouterMode.ROUTER_SCHEMA,
#     variant=IntegrationVariant.S_INJECT,
#     skip_risk_tiers=("high",),
# )
# agent = TauBenchAgentAdapter(router=router, tools_info=env.tools_info, wiki=env.wiki)
# for task_idx in range(10):
#     res = agent.solve(env=env, task_index=task_idx)
#     print(f"task={task_idx} reward={res.reward} steps={len(agent.last_traces)}")
"""
    ),
]


def build_notebook(out_path: Path) -> None:
    # Verify every code cell compiles and all referenced names resolve across cells
    from scripts._notebook_guard import guard_code_cells

    guard_code_cells(CELLS)

    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build_notebook(Path("notebooks/spancall_tau_bench_agentic_eval.ipynb"))
