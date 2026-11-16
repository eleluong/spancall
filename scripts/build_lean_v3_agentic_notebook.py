#!/usr/bin/env python3
"""Generate notebooks/spancall_lean_v3_agentic.ipynb via nbformat.

SpanCall v3 — Agentic Integration Notebook (three modes)
========================================================
This notebook answers one question with numbers: *where should the LLM sit
relative to the SpanCall fast path in a multi-turn agent?* It compares exactly
three configurations against the same mock booking environment and the same
task suites:

1. **LLM-only** (``RouterMode.LLM_ONLY``) — every tool call comes from the LLM.
   The baseline (B0).
2. **SpanCall + LLM fallback** (``RouterMode.ROUTER``) — SpanCall tries first;
   when it abstains or fails a whole-call gate, the *entire* call is delegated
   to the LLM. Safe, but one unresolved parameter throws away an otherwise
   correct skeleton (right tool, five right arguments).
3. **Combined / hybrid fill** (``RouterMode.HYBRID_FILL``) — SpanCall emits the
   call skeleton first and the LLM is asked for **only the parameters SpanCall
   could not ground**, in a shortened completion whose tool schema physically
   lists nothing else (``spancall/agent/fill.py``). The merge is conservative:
   an LLM value never overwrites a resolved one, and anything outside the
   requested set is dropped.

Design notes
------------
* **Provenance is the point.** Every demo cell prints, per turn, *which system
  produced the call* (``step.decision``) and *which parameter came from whom*
  (``known_params`` = SpanCall, ``filled_params`` = LLM). The plan requires
  "side-by-side traces of which system produced each tool call and which params
  were filled by whom"; anything less cannot distinguish mode 2 from mode 3.
* **Task suites are imported, not copied** from ``scripts/eval_agent.py``. A
  copy would drift and the agent numbers would stop being comparable with the
  offline ones. If the import fails the notebook says so and stops rather than
  substituting a suite that looks the same but is not.
* **Runs key-less.** With no ``API_KEY`` in the environment the LLM arm is
  driven by ``DeterministicBackend`` (the same rule-based backend the offline
  comparative eval uses), so all three modes execute end to end. The live-LLM
  section re-runs everything against a real provider when a key is present.
* **This file is a builder, and the artifact is git-ignored**
  (``notebooks/*_agentic*.ipynb``): the notebook requires a provider key. The
  *content* stays in git as this builder.

Build-time guard
----------------
Every code cell is compiled and its free names resolved with the same symtable
guard the train/eval builders use, so a missing import is caught here instead of
half-way through a Colab run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
V26_SCRIPT = HERE / "build_lean_v2_6_colab_notebook.py"
OUT = HERE / ".." / "notebooks" / "spancall_lean_v3_agentic.ipynb"

# --------------------------------------------------------------------------
# Prose
# --------------------------------------------------------------------------
MARKDOWN_TITLE = """# SpanCall v3 — Agentic Integration (three modes)

Compares three ways to place the **SpanCall** fast path in a multi-turn tool-calling agent, against the same mock booking environment and the same task suites. Only the router changes.

| # | Mode | `RouterMode` | Who writes the tool call |
| --- | --- | --- | --- |
| 1 | **LLM-only** (baseline B0) | `llm_only` | The LLM, every turn |
| 2 | **SpanCall + LLM fallback** | `router` | SpanCall when it hits; otherwise the LLM writes the **whole** call |
| 3 | **Combined (hybrid fill)** | `hybrid_fill` | SpanCall writes the tool + grounded params; the LLM fills **only** the unresolved required params |

Mode 2 is all-or-nothing: one unresolvable parameter discards an otherwise perfect skeleton. Mode 3 keeps the skeleton and asks a much smaller question — hence a shorter completion, fewer tokens, and no path for the LLM to overwrite a value SpanCall already grounded (`spancall/agent/fill.py`).

**What this notebook measures:** task success, turns per task, SpanCall hit rate, wrong-hit rate, LLM calls and tokens actually spent, wall-clock latency, and (when a price is configured) USD. Savings are reported **relative to mode 1 on the same suite**, because an absolute cost number with no baseline is unreadable."""

MARKDOWN_SETUP = """---
## 0. Setup

Clones or updates the repository, installs dependencies, and reads the model id and any provider key **from the environment only**. No key is written into this notebook: `API_KEY`, `BASE_URL`, `BASE_LLM_MODEL` and `HF_TOKEN` are looked up in `os.environ` / `.env`, and their absence downgrades the LLM arm to a deterministic backend instead of failing."""

MARKDOWN_MODEL = """---
## 1. Load the pushed model from Hugging Face

The v3 export (`§7` of the train/eval notebook) is **not** a transformers checkpoint. `LeanSpanCallModel` is a plain `nn.Module` with no `save_pretrained()`, so the artifact is:

- `model.pt` — a `state_dict`;
- tokenizer files;
- `resolve_config.json` — backbone, `head_dim`, `max_span_width`, gate thresholds, and the extraction mode (`resolve_mode`).

This cell rebuilds the encoder + model **exactly** from `resolve_config.json`, then loads the weights. Two failure modes are made loud instead of silent:

- **Head dimension mismatch.** `head_dim` is part of the architecture. Loading a mismatched `pointer_head` with `strict=False` would leave it randomly initialised while looking like a successful load — so any `pointer_head`/`tool_head` key that fails to load aborts the cell.
- **Missing extraction mode.** A pre-v3 export has no `resolve_mode`. The resolver's own default is `two_step` as of v3, but this notebook does not lean on that: a missing key is detected, the effective mode is printed, and `two_step` is applied explicitly — so the recorded mode is never an inherited one."""

MARKDOWN_TOOLS = """---
## 2. Environment, backends, and task suites

The mock booking environment (`search_flights` / `book_flight` / `cancel_booking`) is deterministic and stateful: bookings persist, and the last search result is fed back as `recent_context` so later turns can be grounded in earlier tool output.

Task suites are **imported** from [`scripts/eval_agent.py`](../scripts/eval_agent.py) — the canonical three booking tasks and the long-trajectory set — so agent numbers here stay comparable with the offline comparative study. A copy would drift silently, so if the import fails, the notebook stops.

The LLM arm is a live provider when `API_KEY` is set **and** a one-call probe succeeds; otherwise it is `DeterministicBackend`, the same rule-based backend the offline runs use. The probe matters: an invalid key would otherwise surface mid-suite, after several modes had already run."""

MARKDOWN_DEMO = """---
## 3. Side-by-side demos: who produced what

Runs one instruction through all three modes and prints, per turn:

- **decision** — `spancall_hit` (SpanCall wrote the whole call), `hybrid_fill` (SpanCall skeleton + LLM-filled gaps), `llm_fallback` (LLM wrote the whole call), `spancall_abstain` (mode-3 diagnostic only);
- **tool** — the call that actually executed;
- **per-parameter provenance** — `SpanCall`, `LLM`, or `LLM (fill)`.

The parameter provenance column is the difference the notebook exists to expose. In mode 2 a fallback re-writes *every* argument; in mode 3 only the gapped ones are re-asked, and the resolved ones are provably untouched."""

MARKDOWN_SCRIPTED = """---
### 3b. Mechanism demo with a scripted partial

A trained checkpoint does not guarantee a *partial* result on cue — it will usually either hit or abstain, and on these six tasks mode 3 often degenerates to one of its two neighbours. That would leave the hybrid-fill path undemonstrated.

So this section drives the **same router code** with a `ScriptedResolver` that returns a deliberate partial result (tool named, one required parameter withheld). It exercises `unresolved_param_names` → `restrict_tool_to_params` → `merge_filled_arguments` → `SchemaValidator`, which is the whole mechanism. It is a plumbing demo, **not** a measurement: the resolver here is hand-written, so nothing in this section says anything about model quality."""

MARKDOWN_SUITE = """---
## 4. Full suites × three modes

Runs both suites (canonical + long trajectory) through all three modes and aggregates with `spancall.eval.agentic.aggregate_metrics`.

Note how `total_llm_calls` is accounted: a `hybrid_fill` step **did** call the LLM, so it counts toward the LLM-call total, while also counting as a fast-path step (the tool and most arguments came from SpanCall). Counting it in only one bucket would inflate or deflate the savings figure depending on which column a reader looks at.

Multi-call note: a `hybrid_fill` step whose fill was rejected (nothing new, or the merged payload fails schema validation) falls back to the full LLM call and records `hybrid_fill_attempt` in the result — it is reported as `llm_fallback`, not silently merged."""

MARKDOWN_TABLE = """---
## 5. Mode × suite table

Success, turns, fast-path share, LLM calls, tokens, wall-clock and USD per `(mode, suite)`. `Δ calls` and `Δ tokens` are computed **against mode 1 on the same suite** — that is the only comparison that isolates the router's effect, since all three modes see identical tasks, tools and instructions.

USD is `n/a` unless `USD_PER_1K_IN` / `USD_PER_1K_OUT` are set, because a hardcoded `0.0` would read as "free" rather than "not priced"."""

MARKDOWN_LIVE = """---
## 6. Live LLM re-run (key-guarded)

Everything above already runs with a real provider whenever `API_KEY` is present — the probe in §2 decides. This section is for the other direction: it lets you pin a specific model id, or re-run only the suites, without re-executing the notebook.

If no key is present it prints why and does nothing. It never prompts for a key, never reads one from a file path, and never writes one into a cell output."""

# --------------------------------------------------------------------------
# Code cells
# --------------------------------------------------------------------------
CODE_INSTALL = '''# 0. Install (Colab)
import os, subprocess, sys
from pathlib import Path

GIT_REPO_URL = "https://github.com/eleluong/spancall.git"
GIT_BRANCH   = "main"

cwd = Path.cwd().resolve()
if (cwd / "spancall" / "__init__.py").is_file():
    PROJECT_ROOT = cwd
elif (cwd.parent / "spancall" / "__init__.py").is_file():
    PROJECT_ROOT = cwd.parent
elif (cwd / "spancall" / "spancall" / "__init__.py").is_file():
    PROJECT_ROOT = cwd / "spancall"
else:
    print(f"Cloning public repository from {GIT_REPO_URL}...")
    !git clone --branch {GIT_BRANCH} {GIT_REPO_URL}
    PROJECT_ROOT = cwd / "spancall"

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if (PROJECT_ROOT / ".git").is_dir():
    !git pull origin {GIT_BRANCH} || echo "git pull note: continuing with local files"

%pip install -q -e ".[model,data,agent,dev]" huggingface_hub'''

CODE_CONFIG = '''# 0b. Configuration (env-key only - nothing secret appears in this cell)
import json, os, sys
from pathlib import Path
import torch

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "spancall" / "__init__.py").is_file():
    if (PROJECT_ROOT.parent / "spancall" / "__init__.py").is_file():
        PROJECT_ROOT = PROJECT_ROOT.parent
    elif (PROJECT_ROOT / "spancall" / "spancall" / "__init__.py").is_file():
        PROJECT_ROOT = PROJECT_ROOT / "spancall"
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spancall.agent.backends.litellm_adapter import _load_dotenv_values

ENV = {**_load_dotenv_values(PROJECT_ROOT / ".env"), **os.environ}

# ---- model artifact ------------------------------------------------------ #
# The v3 train/eval notebook pushes to this id (HF_REPO_ID there). Point it at
# your own namespace if you trained elsewhere.
HF_REPO_ID = ENV.get("SPANCALL_HF_REPO_ID") or "convoicon/spancall-lean-v3"
# Alternative to HF: a local export directory holding model.pt + resolve_config.json
USE_LOCAL_CHECKPOINT = False
LOCAL_CHECKPOINT_DIR = str(PROJECT_ROOT / "results_lean_v3" / "hf_export")

# Token is read, never required: a private repo needs one, a public repo does not.
HF_TOKEN = ENV.get("HUGGING_FACE_TOKEN") or ENV.get("HF_TOKEN") or None

# ---- runtime ------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
NUM_SEEDS = 1          # >1 repeats each task with a fresh environment RNG
MAX_TURNS = 8          # shared with scripts/eval_agent.py's default

# ---- pricing (optional) -------------------------------------------------- #
# Left unset on purpose: USD 0.0 would read as "free" instead of "not priced".
USD_PER_1K_IN = None
USD_PER_1K_OUT = None

RESULTS_DIR = str(PROJECT_ROOT / "results_lean_v3" / "agentic")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 72)
print("SPANCALL v3 - AGENTIC INTEGRATION")
print("=" * 72)
print(f"HF_REPO_ID        = {HF_REPO_ID}  (local fallback: {USE_LOCAL_CHECKPOINT})")
print(f"HF token present  = {bool(HF_TOKEN)}   API key present = {bool(ENV.get('API_KEY'))}")
print(f"DEVICE            = {DEVICE}   SEED = {SEED}   MAX_TURNS = {MAX_TURNS}")
print(f"RESULTS_DIR       = {RESULTS_DIR}")'''

CODE_LOAD_MODEL = '''# 1. Rebuild encoder + model + resolver from the exported artifact
import json, os
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from span_template.sanitize import SPECIAL_MARKER_TOKENS
from spancall.model.encoder import SpanCallTransformerEncoder
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver
from spancall.infer.base import NoOpResolver

# Fallbacks only apply when resolve_config.json is silent; the exported values win.
BACKBONE_FALLBACK = "answerdotai/ModernBERT-large"
MAX_SEQ_LEN_FALLBACK = 4096
MAX_SPAN_WIDTH_FALLBACK = 16
MAX_CALLS_FALLBACK = 1

RESOLVER_INFO = {"kind": "unknown"}


def _fetch(repo_id, filename, token, local_dir):
    """Download one artifact file, or read it from the local export dir."""
    if local_dir is not None:
        return os.path.join(local_dir, filename)
    return hf_hub_download(repo_id=repo_id, filename=filename, token=token)


def _load_state_dict(path):
    """Unpickle ``model.pt`` with arbitrary code execution disabled.

    ``model.pt`` is fetched from the Hugging Face Hub, so it crosses a trust
    boundary: a pickled file can carry a ``__reduce__`` payload that runs with
    this notebook's credentials in scope. The artifact is a plain dict of
    tensors (``LeanTrainer.save_checkpoint`` writes ``model.state_dict()``), so
    ``weights_only=True`` loads it without executing anything.

    There is deliberately no fallback to ``weights_only=False``: if the file is
    not a pure tensor dict, the correct response is to re-export it, not to
    unpickle it unsafely.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raise SystemExit(
            "[model] this torch build has no `weights_only` argument "
            f"(torch {getattr(torch, '__version__', '?')}). Refusing to unpickle "
            "a Hub-downloaded model.pt with arbitrary code execution enabled.\\n"
            "        Upgrade to torch>=2.0 and re-run this cell."
        )
    except Exception as exc:
        raise SystemExit(
            f"[model] model.pt is not a plain tensor state dict ({type(exc).__name__}: {exc}).\\n"
            "        It was downloaded from the Hub and will not be unpickled "
            "unsafely. Re-export it with `torch.save(model.state_dict(), 'model.pt')`."
        )


def _read_resolve_config():
    local_dir = LOCAL_CHECKPOINT_DIR if USE_LOCAL_CHECKPOINT else None
    path = _fetch(HF_REPO_ID, "resolve_config.json", HF_TOKEN, local_dir)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh), local_dir


def _build_model(cfg, local_dir):
    backbone_name = str(cfg.get("backbone") or BACKBONE_FALLBACK)
    max_span_width = int(cfg.get("max_span_width") or MAX_SPAN_WIDTH_FALLBACK)
    head_dim = int(cfg.get("head_dim") or 256)

    tokenizer = AutoTokenizer.from_pretrained(
        local_dir if local_dir is not None else HF_REPO_ID, use_fast=True,
        token=HF_TOKEN if local_dir is None else None,
    )
    encoder = SpanCallTransformerEncoder(
        model_name_or_path=backbone_name,
        tokenizer=tokenizer,
    )
    # Idempotent: the exported tokenizer already carries the markers, so this
    # adds 0 rows and only guarantees the order matches the checkpoint.
    encoder.add_marker_tokens(list(SPECIAL_MARKER_TOKENS))
    model = LeanSpanCallModel(
        encoder=encoder,
        head_dim=head_dim,
        max_span_width=max_span_width,
        hidden_dim=encoder.hidden_dim,
    )

    weights_path = _fetch(HF_REPO_ID, "model.pt", HF_TOKEN, local_dir)
    state = _load_state_dict(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # head_dim is architecture, not a serving preference. If the pointer or tool
    # head did not load, the weights are those of a differently-shaped model and
    # every "hit" from here on would be nonsense -- so refuse instead of scoring.
    head_gaps = [k for k in missing if k.split(".")[0] in ("pointer_head", "tool_head", "tool_head_binary")]
    if head_gaps:
        raise SystemExit(
            "[model] head weights did not load: " + ", ".join(head_gaps[:6])
            + f"\\n        resolve_config.json head_dim={head_dim}, encoder hidden_dim={encoder.hidden_dim}."
            + "\\n        Re-export with head_dim in resolve_config.json (v3 export includes it)."
        )
    if missing:
        print(f"[model] non-head keys not in checkpoint (ignored): {len(missing)} "
              f"e.g. {sorted(missing)[:3]}")
    if unexpected:
        print(f"[model] checkpoint keys unused (ignored): {len(unexpected)} "
              f"e.g. {sorted(unexpected)[:3]}")
    return model, encoder, tokenizer, missing, unexpected


try:
    _cfg, _local_dir = _read_resolve_config()
except Exception as _exc:
    _cfg, _local_dir = None, None
    print(f"[model] could not read resolve_config.json from {HF_REPO_ID} "
          f"({type(_exc).__name__}: {_exc})")

if _cfg is None:
    print("=" * 72)
    print("[model] NO MODEL LOADED -> NoOpResolver (abstains on every turn).")
    print("[model] Modes 2 and 3 therefore behave like mode 1 in this run.")
    print("[model] Set USE_LOCAL_CHECKPOINT + LOCAL_CHECKPOINT_DIR, or push the")
    print("[model] export to HF_REPO_ID (train/eval notebook section 7).")
    print("=" * 72)
    model = None
    encoder = None
    tokenizer = None
    resolver = NoOpResolver(abstain_reason="no_checkpoint")
    RESOLVER_INFO = {"kind": "noop", "reason": "model artifact unavailable"}
else:
    model, encoder, tokenizer, _missing, _unexpected = _build_model(_cfg, _local_dir)
    model.to(DEVICE)
    model.eval()

    # The extraction mode is part of the protocol, not a default to inherit.
    _mode = _cfg.get("resolve_mode")
    if _mode is None:
        print("[model] resolve_config.json has no resolve_mode (pre-v3 export) -> "
              "forcing 'two_step', which is the mode this notebook evaluates.")
        _mode = "two_step"
    _floor = _cfg.get("min_joint_margin")

    resolver = LeanResolver(
        model=model,
        encoder=encoder,
        tokenizer=tokenizer,
        max_seq_len=int(_cfg.get("max_seq_len") or MAX_SEQ_LEN_FALLBACK),
        max_span_width=int(_cfg.get("max_span_width") or MAX_SPAN_WIDTH_FALLBACK),
        max_calls=int(_cfg.get("max_calls") or MAX_CALLS_FALLBACK),
        call_min_prob=float(_cfg.get("call_min_prob", 0.25)),
        call_min_prob_ratio=float(_cfg.get("call_min_prob_ratio", 0.5)),
        min_joint_margin=(float(_floor) if _floor is not None else None),
        resolve_mode=str(_mode),
        collapse_guard=bool(_cfg.get("collapse_guard", False)),
        prevent_param_collapse=bool(_cfg.get("prevent_param_collapse", False)),
    )
    RESOLVER_INFO = {
        "kind": "lean",
        "backbone": _cfg.get("backbone"),
        "head_dim": int(_cfg.get("head_dim") or 256),
        "max_seq_len": int(_cfg.get("max_seq_len") or MAX_SEQ_LEN_FALLBACK),
        "resolve_mode": str(_mode),
        "min_joint_margin": _floor,
        "version": _cfg.get("version"),
        "experiment_name": _cfg.get("experiment_name"),
        "source": "local export" if _local_dir else HF_REPO_ID,
    }
    print(f"[model] loaded from {RESOLVER_INFO['source']}")
    print(f"[model] backbone={RESOLVER_INFO['backbone']} head_dim={RESOLVER_INFO['head_dim']} "
          f"resolve_mode={RESOLVER_INFO['resolve_mode']} min_joint_margin={RESOLVER_INFO['min_joint_margin']}")

print(f"[model] resolver = {type(resolver).__name__}")'''

CODE_TOOLS = '''# 2. Environment, backends, task suites
import importlib.util, os
from spancall.agent.backends.base import BaseLLMBackend
from spancall.agent.loop import ReActAgentLoop
from spancall.agent.router import IntegrationVariant, RouterMode, ToolCallRouter
from spancall.agent.tools.mock import MockBookingEnvironment, make_booking_tools
from spancall.agent.trace import RoutingDecision
from spancall.eval.agentic.evaluator import MultiSeedAgentEvaluator

# ---- task suites: imported from the offline eval, not copied -------------- #
_EVAL_AGENT_PATH = os.path.join(str(PROJECT_ROOT), "scripts", "eval_agent.py")
_spec = importlib.util.spec_from_file_location("spancall_eval_agent", _EVAL_AGENT_PATH)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"[tasks] cannot load {_EVAL_AGENT_PATH}")
eval_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_agent)

TASK_SUITES = {
    "canonical": eval_agent.build_task_suite(),
    "long_trajectory": eval_agent.build_long_trajectory_suite(),
}
for _name, _suite in TASK_SUITES.items():
    if not _suite:
        raise SystemExit(f"[tasks] suite '{_name}' is empty - refusing to report 0 tasks as a result")
    print(f"[tasks] {_name}: {len(_suite)} task(s) -> {[t['task_id'] for t in _suite]}")

# ---- LLM backend --------------------------------------------------------- #
LLM_MODEL_ID = ENV.get("BASE_LLM_MODEL") or "google/gemma-4"


def build_backend(live):
    """Live provider when a key works; deterministic rule-based backend otherwise.

    The one-call probe is the point: without it an invalid key surfaces inside
    the first suite, after some modes have already been scored, and the partial
    results look like a router difference rather than an auth failure.
    """
    if live:
        try:
            from spancall.agent.backends.litellm_adapter import UniversalAPIBackend
            backend = UniversalAPIBackend(model_id=LLM_MODEL_ID)
            probe = backend.generate_tool_call(
                [{"role": "user", "content": "Book flight F001."}],
                make_booking_tools(),
                temperature=0.0,
            )
            print(f"[llm] live backend OK: {LLM_MODEL_ID} -> tool={probe.get('tool_name')!r}")
            return backend, "live"
        except Exception as exc:
            print(f"[llm] live backend unavailable ({type(exc).__name__}: {exc}) "
                  "-> falling back to the deterministic backend")
    else:
        print("[llm] no API_KEY in the environment -> deterministic backend")
    return eval_agent.DeterministicBackend(), "deterministic"


LLM_BACKEND, BACKEND_KIND = build_backend(bool(ENV.get("API_KEY")))
print(f"[llm] backend = {BACKEND_KIND}")


# ---- builders ------------------------------------------------------------ #
def make_router(mode, backend, resolver_obj):
    return ToolCallRouter(
        resolver=resolver_obj,
        llm_backend=backend,
        mode=mode,
        # s_inject makes SpanCall's call appear in history byte-identically to a
        # native LLM tool call, so later turns render the same way in all modes.
        variant=IntegrationVariant.S_INJECT,
    )


def make_loop(mode, backend, resolver_obj, required_flight_id="F001"):
    env = MockBookingEnvironment(required_flight_id=required_flight_id)
    return ReActAgentLoop(
        router=make_router(mode, backend, resolver_obj),
        environment=env,
        max_turns=MAX_TURNS,
    )'''

CODE_DEMO = '''# 3. One instruction through all three modes, with per-parameter provenance
DEMO_INSTRUCTION = "Find a flight from SFO to JFK on 2026-11-05, then book it."
DEMO_REQUIRED_FLIGHT = "F001"

MODE_ORDER = [
    ("mode1_llm_only", RouterMode.LLM_ONLY),
    ("mode2_fallback", RouterMode.ROUTER),
    ("mode3_hybrid_fill", RouterMode.HYBRID_FILL),
]
_PROVENANCE_SPANCALL = "SpanCall"
_PROVENANCE_LLM = "LLM"
_PROVENANCE_FILL = "LLM (fill)"


def param_provenance(step):
    """[(param, value, who)] - the column that separates mode 2 from mode 3.

    ``known_params`` is what SpanCall grounded; ``filled_params`` is what the
    LLM supplied *for the gaps only*. Outside HYBRID_FILL the whole argument
    dict belongs to a single producer, so provenance is uniform.
    """
    args = dict(step.tool_arguments or {})
    feats = step.spancall_features or {}
    if step.decision == RoutingDecision.HYBRID_FILL:
        filled = {str(k) for k in (feats.get("filled_params") or [])}
        return [(k, args[k], _PROVENANCE_FILL if k in filled else _PROVENANCE_SPANCALL)
                for k in sorted(args)]
    if step.decision == RoutingDecision.SPANCALL_HIT:
        return [(k, args[k], _PROVENANCE_SPANCALL) for k in sorted(args)]
    return [(k, args[k], _PROVENANCE_LLM) for k in sorted(args)]


def render_trace(label, trace):
    print(f"\\n{'-' * 84}")
    print(f"{label}   task={trace.task_id}   success={trace.task_success}   "
          f"turns={len(trace.steps)}   llm_calls={trace.total_llm_calls}   "
          f"spancall_steps={trace.total_spancall_calls}")
    print(f"{'-' * 84}")
    if not trace.steps:
        print("  (no steps)")
        return
    for step in trace.steps:
        feats = step.spancall_features or {}
        # ``step.tool_output`` is set by the loop only when a tool actually
        # executed. ``step.selected_tool`` is NOT that signal: on a fallback,
        # ``ToolCallRouter._trace`` records the SpanCall *hypothesis* there, so a
        # turn where the LLM declined to call anything can still show a tool
        # name. Printing them as one field would misattribute the call.
        executed = step.tool_output is not None
        print(f"  turn {step.turn_index}: {step.decision.value:<16} "
              f"executed={step.selected_tool if executed else '(none)'!r:<18} "
              f"tok={step.llm_tokens_in}+{step.llm_tokens_out} "
              f"lat={step.latency_ms:.1f}ms")
        if not executed and step.selected_tool:
            print(f"      SpanCall hypothesis only (LLM called nothing): tool={step.selected_tool!r}")
        if step.decision == RoutingDecision.HYBRID_FILL:
            print(f"      requested from LLM: {list(feats.get('requested_params') or [])}")
            print(f"      SpanCall had     : {list(feats.get('known_params') or [])}")
        provenance = param_provenance(step) if executed else []
        for name, value, who in provenance:
            print(f"      {name:<14} = {value!r:<28} <- {who}")
        if not provenance:
            print(f"      (no arguments) {str(step.tool_output or '')[:60]}")


DEMO_TRACES = {}
for _label, _mode in MODE_ORDER:
    _loop = make_loop(_mode, LLM_BACKEND, resolver, required_flight_id=DEMO_REQUIRED_FLIGHT)
    DEMO_TRACES[_label] = _loop.run_task(DEMO_INSTRUCTION, task_id=f"demo_{_label}")

print("=" * 84)
print(f"DEMO   resolver={RESOLVER_INFO.get('kind')}   backend={BACKEND_KIND}")
print(f"       instruction={DEMO_INSTRUCTION!r}")
print("=" * 84)
for _label, _mode in MODE_ORDER:
    render_trace(_label, DEMO_TRACES[_label])

_hybrid_steps = [
    s for s in DEMO_TRACES["mode3_hybrid_fill"].steps
    if s.decision == RoutingDecision.HYBRID_FILL
]
if not _hybrid_steps:
    print("\\n[note] no HYBRID_FILL decision occurred on this demo (the resolver either "
          "hit cleanly or abstained outright). Section 3b exercises the fill path "
          "directly with a scripted partial result.")'''

CODE_DEMO_SCRIPTED = '''# 3b. Hybrid-fill mechanism with a scripted partial (plumbing demo, not a measurement)
from spancall.infer.base import ScriptedResolver, ResolveResult


def _scripted_partial(candidate_schemas, utterance, recent_context=None, now=None):
    """Name the tool, ground ONE required param, withhold the other.

    For search_flights that is origin grounded but destination/date missing, so
    the router has a real skeleton to keep and exactly two gaps to fill.
    """
    names = {s.name for s in (candidate_schemas or [])}
    features = {"scripted": True, "n_recent_rows": len(recent_context or [])}
    if "search_flights" not in names:
        return ResolveResult(hit=False, abstain_reason="scripted_tool_unavailable", features=features)
    return ResolveResult(
        hit=False,
        status="partial",
        tool_name="search_flights",
        payload=None,
        payload_partial={"origin": "SFO"},
        missing_required=["destination", "date"],
        missing_optional=["max_price"],
        abstain_reason="scripted_partial",
        confidence=0.6,
        joint_margin=0.05,
        features=features,
    )


# Phrased with the literal "from X to Y on DATE" form because the offline
# DeterministicBackend parses exactly that pattern. The point of this section is
# to exercise the fill path, so the LLM arm must be able to answer -- using an
# instruction the stub cannot parse would make the demo fail for a reason that
# has nothing to do with hybrid fill.
SCRIPTED_INSTRUCTION = "Find me a flight from SFO to JFK on 2026-11-05."
scripted_resolver = ScriptedResolver(_scripted_partial)
SCRIPTED_TRACES = {}
for _label, _mode in MODE_ORDER:
    _loop = make_loop(_mode, LLM_BACKEND, scripted_resolver, required_flight_id=DEMO_REQUIRED_FLIGHT)
    SCRIPTED_TRACES[_label] = _loop.run_task(SCRIPTED_INSTRUCTION, task_id=f"scripted_{_label}")

print("=" * 84)
print("SCRIPTED PARTIAL - same router code, deliberate partial resolver")
print("  mode 1 should call the LLM for everything; mode 2 should fall back for")
print("  the whole call; mode 3 should keep origin=SFO and ask only for the gaps.")
print("=" * 84)
for _label, _mode in MODE_ORDER:
    render_trace(_label, SCRIPTED_TRACES[_label])

_mode3 = SCRIPTED_TRACES["mode3_hybrid_fill"]
_fill = [s for s in _mode3.steps if s.decision == RoutingDecision.HYBRID_FILL]
if _fill:
    _known = list((_fill[0].spancall_features or {}).get("known_params") or [])
    _filled = list((_fill[0].spancall_features or {}).get("filled_params") or [])
    print(f"\\n[3b] mode 3 kept {_known} from SpanCall and asked the LLM only for {_filled}.")
    print("[3b] origin stays 'SFO' and is absent from the fill request: that is the")
    print("     conservative merge in spancall/agent/fill.py, visible in a trace.")
else:
    print("\\n[3b] mode 3 did not produce a HYBRID_FILL step: the fill was rejected and the")
    print("     router fell back to a full LLM call (see hybrid_fill_attempt on the result).")'''

CODE_RUN_SUITE = '''# 4. Both suites x three modes
from spancall.eval.agentic.metrics import aggregate_metrics

SUITE_METRICS = {}
SUITE_TRACES = {}

evaluator = MultiSeedAgentEvaluator(seed=SEED)
for _sname, _suite in TASK_SUITES.items():
    for _label, _mode in MODE_ORDER:
        _loop = make_loop(_mode, LLM_BACKEND, resolver, required_flight_id="F001")
        _metrics, _traces = evaluator.evaluate_agent(_loop, _suite, num_seeds=NUM_SEEDS)
        SUITE_METRICS[(_label, _sname)] = _metrics
        SUITE_TRACES[(_label, _sname)] = _traces
        print(f"[run] {_sname:<16} {_label:<18} tasks={_metrics.total_tasks} "
              f"success={_metrics.task_success_rate:.3f} "
              f"turns/task={_metrics.average_turns_per_task:.2f} "
              f"llm_calls={sum(t.total_llm_calls for t in _traces)}")

print(f"\\n[run] {len(SUITE_METRICS)} (mode, suite) cells evaluated "
      f"with resolver={RESOLVER_INFO.get('kind')}, backend={BACKEND_KIND}")'''

CODE_MODE_TABLE = '''# 5. Mode x suite table, with savings measured against mode 1 on the same suite
import json, os


def totals(traces):
    return {
        "llm_calls": sum(t.total_llm_calls for t in traces),
        "spancall_steps": sum(t.total_spancall_calls for t in traces),
        "tok_in": sum(t.total_tokens_prompt for t in traces),
        "tok_out": sum(t.total_tokens_completion for t in traces),
        "wall_ms": sum(t.total_latency_ms for t in traces),
    }


def usd(tok_in, tok_out):
    """USD or None - never 0.0, which would read as free instead of unpriced."""
    if USD_PER_1K_IN is None or USD_PER_1K_OUT is None:
        return None
    return tok_in / 1000.0 * float(USD_PER_1K_IN) + tok_out / 1000.0 * float(USD_PER_1K_OUT)


def decision_histogram(traces):
    hist = {}
    for t in traces:
        for s in t.steps:
            key = s.decision.value
            hist[key] = hist.get(key, 0) + 1
    return hist


def fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_usd(v):
    return "n/a" if v is None else f"{v:.4f}"


TABLE_ROWS = []
for _sname in TASK_SUITES:
    _base = totals(SUITE_TRACES[("mode1_llm_only", _sname)])
    for _label, _mode in MODE_ORDER:
        _m = SUITE_METRICS[(_label, _sname)]
        _t = totals(SUITE_TRACES[(_label, _sname)])
        _d_calls = _base["llm_calls"] - _t["llm_calls"]
        _d_tok = (_base["tok_in"] + _base["tok_out"]) - (_t["tok_in"] + _t["tok_out"])
        TABLE_ROWS.append({
            "suite": _sname,
            "mode": _label,
            "tasks": _m.total_tasks,
            "success": _m.task_success_rate,
            "turns_per_task": _m.average_turns_per_task,
            "spancall_hit_rate": _m.spancall_hit_rate,
            "wrong_hit_rate": _m.wrong_hit_rate,
            "llm_calls": _t["llm_calls"],
            "delta_llm_calls": _d_calls,
            "tok_in": _t["tok_in"],
            "tok_out": _t["tok_out"],
            "delta_tokens": _d_tok,
            "wall_ms": _t["wall_ms"],
            "usd": usd(_t["tok_in"], _t["tok_out"]),
            "decisions": decision_histogram(SUITE_TRACES[(_label, _sname)]),
        })

print("\\n=== 5. mode x suite ===")
print(f"{'suite':<16}{'mode':<19}{'succ':>6}{'turns':>7}{'hit':>7}{'wrong':>7}"
      f"{'calls':>7}{'dcalls':>8}{'tok_in':>8}{'tok_out':>9}{'wall_s':>8}{'usd':>9}")
print("-" * 116)
for r in TABLE_ROWS:
    print(f"{r['suite']:<16}{r['mode']:<19}{fmt(r['success']):>6}"
          f"{fmt(r['turns_per_task'], 2):>7}{fmt(r['spancall_hit_rate']):>7}"
          f"{fmt(r['wrong_hit_rate']):>7}{r['llm_calls']:>7}{r['delta_llm_calls']:>+8}"
          f"{r['tok_in']:>8}{r['tok_out']:>9}{fmt(r['wall_ms'] / 1000.0, 2):>8}"
          f"{fmt_usd(r['usd']):>9}")

print("\\n=== 5b. decision mix (which system produced each tool call) ===")
for r in TABLE_ROWS:
    print(f"  {r['suite']:<16}{r['mode']:<19} {r['decisions']}")

_has_lean = RESOLVER_INFO.get("kind") == "lean"
if not _has_lean:
    print("\\n[!] No model was loaded, so modes 2 and 3 abstain on every turn and are")
    print("    indistinguishable from mode 1 by construction. Treat these rows as a")
    print("    plumbing check only.")
elif BACKEND_KIND == "deterministic":
    print("\\n[!] LLM arm is the deterministic rule-based backend: it is a fair test of")
    print("    router plumbing and token accounting, but NOT of LLM quality. Re-run with")
    print("    API_KEY set for numbers you would put in a paper.")

with open(os.path.join(RESULTS_DIR, "agentic_mode_table.json"), "w", encoding="utf-8") as _fh:
    json.dump(
        {
            "resolver": RESOLVER_INFO,
            "backend": BACKEND_KIND,
            "llm_model_id": LLM_MODEL_ID,
            "num_seeds": NUM_SEEDS,
            "max_turns": MAX_TURNS,
            "rows": TABLE_ROWS,
        },
        _fh,
        indent=2,
        default=str,
    )
    _fh.write("\\n")
print(f"\\n[5] wrote {os.path.join(RESULTS_DIR, 'agentic_mode_table.json')}")'''

CODE_LIVE = '''# 6. Live LLM re-run (key-guarded). No-op without a key.
LIVE_MODEL_ID = None          # e.g. "anthropic/claude-sonnet-4" or a vLLM model id
LIVE_RERUN = False            # flip to True to replace the table with live numbers

if not ENV.get("API_KEY"):
    print("[live] no API_KEY in the environment -> nothing to do.")
    print("[live] Section 4 already used the deterministic backend; the table above")
    print("[live] is a plumbing/token-accounting result, not an LLM-quality result.")
elif LIVE_MODEL_ID:
    print(f"[live] pinning model id to {LIVE_MODEL_ID!r} for a re-run")
    LLM_MODEL_ID = LIVE_MODEL_ID
elif not LIVE_RERUN:
    print(f"[live] a key is present; section 4 already ran live against {LLM_MODEL_ID!r}.")
    print("[live] set LIVE_RERUN = True to repeat the suites, or LIVE_MODEL_ID to pin a")
    print("[live] different provider.")
else:
    print(f"[live] re-running suites against {LLM_MODEL_ID!r}")
    _live_backend, _live_kind = build_backend(True)
    LIVE_METRICS = {}
    for _sname, _suite in TASK_SUITES.items():
        for _label, _mode in MODE_ORDER:
            _loop = make_loop(_mode, _live_backend, resolver, required_flight_id="F001")
            _m, _tr = MultiSeedAgentEvaluator(seed=SEED).evaluate_agent(
                _loop, _suite, num_seeds=NUM_SEEDS)
            LIVE_METRICS[(_label, _sname)] = _m
            print(f"[live] {_sname:<16} {_label:<18} success={_m.task_success_rate:.3f} "
                  f"llm_calls={sum(t.total_llm_calls for t in _tr)} "
                  f"tokens={sum(t.total_tokens_prompt + t.total_tokens_completion for t in _tr)}")
    print(f"[live] backend used: {_live_kind}")'''

# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------
cells = [
    new_markdown_cell(MARKDOWN_TITLE),
    new_markdown_cell(MARKDOWN_SETUP),
    new_code_cell(CODE_INSTALL),
    new_code_cell(CODE_CONFIG),
    new_markdown_cell(MARKDOWN_MODEL),
    new_code_cell(CODE_LOAD_MODEL),
    new_markdown_cell(MARKDOWN_TOOLS),
    new_code_cell(CODE_TOOLS),
    new_markdown_cell(MARKDOWN_DEMO),
    new_code_cell(CODE_DEMO),
    new_markdown_cell(MARKDOWN_SCRIPTED),
    new_code_cell(CODE_DEMO_SCRIPTED),
    new_markdown_cell(MARKDOWN_SUITE),
    new_code_cell(CODE_RUN_SUITE),
    new_markdown_cell(MARKDOWN_TABLE),
    new_code_cell(CODE_MODE_TABLE),
    new_markdown_cell(MARKDOWN_LIVE),
    new_code_cell(CODE_LIVE),
]

# --------------------------------------------------------------------------
# Build-time guard: compile every code cell and resolve names
# --------------------------------------------------------------------------
_V26_SRC = V26_SCRIPT.read_text(encoding="utf-8")
_GUARD_START = _V26_SRC.index("def _cell_source")
_GUARD_END = _V26_SRC.rindex("_guard_code_cells(cells)")
_GUARD_NS: Dict[str, Any] = {"__name__": "build_lean_v3_agentic_guard"}
exec(compile(_V26_SRC[_GUARD_START:_GUARD_END], str(V26_SCRIPT), "exec"), _GUARD_NS)
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