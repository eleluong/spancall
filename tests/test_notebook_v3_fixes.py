"""Regression tests for the v3 notebook / script contract.

The v3 work changed four things that a build-time cell guard cannot see:

1. **`single_pass` is retired.** It is still constructible so pre-v3 numbers
   stay reproducible, but it must never be *inherited*: `LeanResolver.__init__`
   now defaults to `two_step`, and the generated notebook must (a) never *call*
   `predict_split(..., resolve_mode="single_pass")` and (b) pin `two_step`
   explicitly, both in the export and in the agentic reload.
2. **Search must be stratified.** The eval was one blended number; v3 splits it
   by corpus (hermes / xlam) *and* by tool-count bucket, so a gain in the small-
   tool regime cannot hide a regression in the many-tool one.
3. **The agentic notebook is a three-mode comparison.** Mode 3 (hybrid fill)
   only differs from mode 2 if the scripted partial actually *parses* against
   the offline backend's regex, and the trace renderer must read
   `tool_output` — not `selected_tool` — to decide whether a tool executed.
4. **Credentials never live in an artifact.** Anything committed may be run by
   someone else, so a hardcoded `HF_TOKEN` / `sk-...` is a leak.

These tests read the *generated* notebooks, so they assert what the builder
actually wrote rather than what it intended. Several of them extract and
execute the real notebook definitions, which is the only way to catch a cell
that compiles yet behaves wrongly.
"""
from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_NB = ROOT / "notebooks" / "spancall_lean_v3_train_eval.ipynb"
AGENT_NB = ROOT / "notebooks" / "spancall_lean_v3_agentic.ipynb"


# ---------------------------------------------------------------------------
# Notebook loading helpers
# ---------------------------------------------------------------------------
def _cells(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def _code_cells(path: Path) -> List[str]:
    return ["".join(c["source"]) for c in _cells(path) if c["cell_type"] == "code"]


def _is_python(src: str) -> bool:
    """True for cells the guard compiles (i.e. not ``!pip`` / ``%cd`` cells)."""
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def _python_cells(path: Path) -> List[str]:
    return [s for s in _code_cells(path) if _is_python(s)]


def _strip_comments(src: str) -> str:
    """Remove ``#`` comments so a mention in prose is not read as a live call.

    The notebooks *document* the retirement of ``single_pass`` in comments, and
    those comments are the point. A naive substring search would fail on them,
    so code is separated from commentary before asserting.
    """
    try:
        kept = [
            tok
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type != tokenize.COMMENT
        ]
        return tokenize.untokenize(kept)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src


def _live_code(path: Path) -> str:
    return "\n".join(_strip_comments(s) for s in _python_cells(path))


def _bound_names(node: ast.stmt) -> List[str]:
    """Names a module-level statement binds, for dependency seeding."""
    if isinstance(node, ast.Import):
        return [a.asname or a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [a.asname or a.name for a in node.names]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    return []


# Only definitions are executable. ``Expr`` statements are skipped on purpose:
# they carry the cells' side effects (``os.chdir``, ``os.makedirs``, ``print``,
# ``subprocess.run``), so skipping them lets a test import a notebook function
# without entering the notebook's runtime.
_DEFINITION_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.ClassDef,
)


def _referenced_names(node: ast.AST) -> set:
    """Every ``Name`` read anywhere inside ``node`` (free or bound)."""
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _module_index(path: Path) -> Dict[str, ast.stmt]:
    """Map each module-level bound name to the statement that binds it.

    Later bindings overwrite earlier ones, mirroring execution order, so a cell
    that redefines a constant is what a reader of the notebook would see.
    """
    index: Dict[str, ast.stmt] = {}
    for src in _python_cells(path):
        for node in ast.parse(src).body:
            if not isinstance(node, _DEFINITION_NODES):
                continue
            for bound in _bound_names(node):
                index[bound] = node
    return index


def _resolve(path: Path, name: str, seed: Optional[Dict[str, Any]] = None) -> Any:
    """Execute ``name``'s definition and only the notebook definitions it needs.

    Transitive rather than wholesale: replaying every definition in the notebook
    would execute unrelated cells (loading ``scripts/eval_agent.py``, reading the
    filesystem) and couple this test to code it does not assert on. ``seed``
    both provides fixtures and *shadows* a definition — a seeded name is never
    re-executed, which is how ``hf_hub_download`` gets stubbed.
    """
    namespace: Dict[str, Any] = dict(seed or {})
    index = _module_index(path)
    done: set = set()

    def _need(target: str, stack: frozenset) -> None:
        if target in done or target in namespace or target in stack:
            return
        node = index.get(target)
        if node is None:  # builtin, seeded, or a name we deliberately ignore
            return
        for dep in sorted(_referenced_names(node) - set(_bound_names(node))):
            _need(dep, stack | {target})
        if target in namespace:  # a dependency's execution bound it
            return
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), f"<{path.name}:{target}>", "exec"),
            namespace,
        )
        done.update(_bound_names(node))

    _need(name, frozenset())
    if name not in namespace:
        raise AssertionError(f"{path.name} binds no module-level {name!r}")
    return namespace[name]


@pytest.fixture(scope="module")
def train_nb() -> None:
    if not TRAIN_NB.is_file():
        pytest.skip(f"{TRAIN_NB} not built; run scripts/build_lean_v3_colab_notebook.py")


@pytest.fixture(scope="module")
def agent_nb() -> None:
    if not AGENT_NB.is_file():
        pytest.skip(f"{AGENT_NB} not built; run scripts/build_lean_v3_agentic_notebook.py")


# ---------------------------------------------------------------------------
# 1. single_pass is retired: never called, always overridden
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [TRAIN_NB, AGENT_NB], ids=["train_eval", "agentic"])
def test_no_live_single_pass(path: Path) -> None:
    """No executable line may name ``single_pass`` (comments may)."""
    if not path.is_file():
        pytest.skip(f"{path.name} not built")
    offenders = [
        line.strip()
        for line in _live_code(path).splitlines()
        if "single_pass" in line
    ]
    assert not offenders, f"{path.name} still runs single_pass:\n" + "\n".join(offenders)


@pytest.mark.parametrize("path", [TRAIN_NB, AGENT_NB], ids=["train_eval", "agentic"])
def test_resolve_mode_is_never_passed_single_pass(path: Path) -> None:
    """Belt and braces: the retired value must not appear in any call kwargs."""
    if not path.is_file():
        pytest.skip(f"{path.name} not built")
    assert not re.search(r"resolve_mode\s*=\s*[\"']single_pass[\"']", _live_code(path)), (
        f"{path.name} passes resolve_mode='single_pass' in a live call"
    )


def test_train_eval_exports_two_step(train_nb: None) -> None:
    """The exported ``resolve_config.json`` must carry the protocol explicitly.

    The resolver now defaults to ``two_step``, so this key is no longer the only
    thing standing between a consumer and the retired arm -- but it is what a
    third-party reload actually reads, so the artifact must state its mode
    rather than leave it to be inferred from a library default.
    """
    live = _live_code(TRAIN_NB)
    assert re.search(r"[\"']resolve_mode[\"']\s*:\s*[\"']two_step[\"']", live), (
        "train/eval export does not pin resolve_mode='two_step' in resolve_config"
    )


def test_agentic_overrides_missing_resolve_mode(agent_nb: None) -> None:
    """A pre-v3 export must be forced to two_step rather than inherit the default."""
    live = _live_code(AGENT_NB)
    assert re.search(r"resolve_mode\s*=\s*str\(_mode\)", live), (
        "agentic notebook does not forward the resolved mode to LeanResolver"
    )
    assert re.search(r"_mode\s*=\s*[\"']two_step[\"']", live), (
        "agentic notebook does not force two_step when resolve_config.json is silent"
    )
    # The override must be conditional on the key being absent, not unconditional.
    assert re.search(r"if\s+_mode\s+is\s+None", live), (
        "the two_step override is not guarded on a missing resolve_mode"
    )


# ---------------------------------------------------------------------------
# 2. head_dim round-trips: exported by v3, consumed by the agentic reload
# ---------------------------------------------------------------------------
def test_export_writes_head_dim(train_nb: None) -> None:
    """``head_dim`` is architecture, so it must be in the export or a rebuild drifts."""
    assert re.search(r"[\"']head_dim[\"']\s*:\s*int\(cfg\.model\.head_dim\)", _live_code(TRAIN_NB))


def test_agentic_rebuild_uses_exported_head_dim(agent_nb: None) -> None:
    live = _live_code(AGENT_NB)
    assert re.search(r"head_dim\s*=\s*int\(cfg\.get\([\"']head_dim[\"']\)", live), (
        "agentic rebuild does not read head_dim from resolve_config.json"
    )
    assert re.search(r"head_dim\s*=\s*head_dim", live), (
        "agentic rebuild reads head_dim but does not pass it to LeanSpanCallModel"
    )


def test_agentic_never_unpickles_hub_weights_with_arbitrary_code(agent_nb: None) -> None:
    """``model.pt`` comes off the network, so it must load with ``weights_only=True``.

    A pickle can carry a ``__reduce__`` payload that executes with the
    notebook's credentials in scope. There must be no ``torch.load(...,
    weights_only=False)`` anywhere in the reload path, and no fallback that
    reintroduces one.
    """
    live = _live_code(AGENT_NB)
    # Match an actual ``torch.load`` invocation. The cell's docstring *names*
    # ``weights_only=False`` while explaining why it is refused, so a bare
    # substring search would flag the prose rather than the call.
    unsafe = re.findall(r"torch\.load\((?:[^()]|\([^()]*\))*weights_only\s*=\s*False", live)
    assert not unsafe, (
        f"the agentic reload unpickles a Hub-downloaded model.pt unsafely: {unsafe!r}"
    )
    assert re.search(r"torch\.load\([^)]*weights_only=True", live), (
        "the agentic reload does not pass weights_only=True"
    )
    # The loader must be the single choke point, so a future edit cannot add a
    # second, unguarded torch.load call somewhere else in the cell.
    assert re.search(r"state\s*=\s*_load_state_dict\(", live), (
        "the reload does not route through _load_state_dict"
    )


def test_agentic_aborts_on_head_load_gap(agent_nb: None) -> None:
    """Scoring a randomly-initialised head must be impossible, not merely unlikely."""
    live = _live_code(AGENT_NB)
    assert re.search(r"head_gaps\s*=\s*\[", live), "no head-gap detection in the reload path"
    assert re.search(r"in\s*\([\"']pointer_head[\"']", live), (
        "head-gap check does not name pointer_head"
    )
    # The abort must be guarded by ``if head_gaps:`` and not merely nearby.
    guard = re.search(
        r"if\s+head_gaps\s*:\s*\n\s*raise\s+SystemExit\(", live
    )
    assert guard, "a head-gap is detected but does not abort the notebook"

    # Behavioural proof: shadow the loader so the head fails to load, then
    # require the reload to refuse rather than score a random head.
    class _FakeState(dict):
        pass

    class _FakeModel:
        def load_state_dict(self, state, strict=False):
            return (["pointer_head.span_start"], [])

        def to(self, _device):
            return self

        def eval(self):
            return self

    build = _resolve(
        AGENT_NB,
        "_build_model",
        {
            "LeanSpanCallModel": lambda **kw: _FakeModel(),
            "SpanCallTransformerEncoder": lambda **kw: SimpleNamespace(
                add_marker_tokens=lambda toks: 0, hidden_dim=1024
            ),
            "AutoTokenizer": SimpleNamespace(
                from_pretrained=lambda *a, **kw: SimpleNamespace()
            ),
            "torch": SimpleNamespace(
                load=lambda *a, **kw: _FakeState(),
            ),
            "_fetch": lambda *a, **kw: "/tmp/model.pt",
            "SPECIAL_MARKER_TOKENS": [],
        },
    )
    with pytest.raises(SystemExit, match="head weights did not load"):
        build({"head_dim": 256}, None)


# ---------------------------------------------------------------------------
# 3. Stratification: per-corpus and per-tool-count-bucket eval
# ---------------------------------------------------------------------------
def test_per_corpus_cell_exists(train_nb: None) -> None:
    live = _live_code(TRAIN_NB)
    assert "by_corpus" in live, "no per-corpus split in the eval cells"
    assert re.search(r"[\"']hermes[\"']\s*:\s*\[\]", live) and re.search(
        r"[\"']xlam[\"']\s*:\s*\[\]", live
    ), "the corpus split does not enumerate hermes and xlam"


def test_tool_bucket_cell_exists(train_nb: None) -> None:
    assert "TOOL_BUCKET_EDGES" in _live_code(TRAIN_NB), "no tool-count bucket definition"
    assert "_tool_bucket(" in _live_code(TRAIN_NB), "tool-count bucket defined but never applied"


def test_tool_bucket_labels_cover_the_whole_range(train_nb: None) -> None:
    """Behavioural: every tool count maps to exactly one bucket, with no gaps."""
    edges = _resolve(TRAIN_NB, "TOOL_BUCKET_EDGES")
    bucket = _resolve(TRAIN_NB, "_tool_bucket")

    assert [e[2] for e in edges] == ["1-2", "3-5", "6-10", "11+"], (
        f"bucket labels changed: {[e[2] for e in edges]!r}"
    )
    expected = {
        0: "<1", 1: "1-2", 2: "1-2",
        3: "3-5", 5: "3-5",
        6: "6-10", 10: "6-10",
        11: "11+", 50: "11+",
    }
    got = {n: bucket(n) for n in expected}
    assert got == expected, f"tool-count mapping wrong: {got!r}"

    # Adjacent buckets must not overlap, otherwise a row is double-counted.
    for (_lo, hi, label), (_nlo, _nhi, next_label) in zip(edges, edges[1:]):
        assert hi is not None, f"only the final bucket may be open-ended ({label!r})"
        assert hi + 1 == _nlo, (
            f"gap or overlap between {label!r} (ends {hi}) and {next_label!r} "
            f"(starts {_nlo})"
        )
        assert bucket(hi) == label and bucket(hi + 1) == next_label


def test_corpus_label_is_total(train_nb: None) -> None:
    """Every row must land in exactly one corpus, even with no dataset field."""
    corpus = _resolve(TRAIN_NB, "_corpus_of_row")
    assert corpus({"source_dataset": "xlam_train"}) == "xlam"
    assert corpus({"source_dataset": "hermes_v1"}) == "hermes"
    assert corpus({"sample_id": "xlam::abc"}) == "xlam"
    assert corpus({"sample_id": "hermes::abc"}) == "hermes"
    # A row with neither hint still has to be classifiable.
    assert corpus({}) in ("hermes", "xlam")


# ---------------------------------------------------------------------------
# 4. Probe eval + W&B: notebook writes keys the trainer actually reads
# ---------------------------------------------------------------------------
def _training_fields() -> set:
    from spancall.train.config import TrainingConfig

    return {f.name for f in TrainingConfig.__dataclass_fields__.values()}


def test_probe_config_keys_match_the_trainer(train_nb: None) -> None:
    """A typo here is silent: the dataclass ignores unknown keys and the probe
    simply never fires, which looks exactly like a converged run."""
    written = set(re.findall(r"cfg\[[\"']training[\"']\]\[[\"']([a-z_]+)[\"']\]\s*=", _live_code(TRAIN_NB)))
    probe_keys = {k for k in written if k.startswith("probe")}
    assert probe_keys, "no probe_* key is written into cfg['training']"
    unknown = probe_keys - _training_fields()
    assert not unknown, f"probe keys not read by TrainingConfig: {sorted(unknown)}"


def test_wandb_config_keys_match_the_trainer(train_nb: None) -> None:
    written = set(re.findall(r"cfg\[[\"']training[\"']\]\[[\"']([a-z_]+)[\"']\]\s*=", _live_code(TRAIN_NB)))
    wandb_keys = {k for k in written if k.startswith("wandb")}
    assert wandb_keys == {"wandb_project", "wandb_run_name", "wandb_mode"}, (
        f"W&B keys written are {sorted(wandb_keys)!r}"
    )
    unknown = wandb_keys - _training_fields()
    assert not unknown, f"W&B keys not read by TrainingConfig: {sorted(unknown)}"


def test_probe_set_is_disjoint_from_dev(train_nb: None) -> None:
    """A probe drawn from ``dev`` would re-measure the model on the split that
    drives checkpoint selection, so the mid-training curve would be optimistic."""
    live = _live_code(TRAIN_NB)
    assert re.search(r"_dev_ids\s*=\s*set\(\)", live), "probe cell keeps no dev ids to exclude"
    assert re.search(r"if\s+json\.loads\(_l\)\.get\([\"']sample_id[\"']\)\s+in\s+_dev_ids", live), (
        "probe rows are not filtered against dev ids"
    )
    assert re.search(r"overlap_with_dev", live), "probe overlap with dev is never reported"


def test_probe_cadence_is_in_optimiser_steps(train_nb: None) -> None:
    """The amendment asked for ~every 1k steps, not once per epoch."""
    live = _live_code(TRAIN_NB)
    assert re.search(r"PROBE_EVERY_STEPS\s*=\s*\d+", live)
    assert re.search(r"[\"']probe_every_steps[\"']", live)


# ---------------------------------------------------------------------------
# 5. Agentic notebook: three modes, HF-sourced, honest warnings
# ---------------------------------------------------------------------------
def test_agentic_has_exactly_three_modes(agent_nb: None) -> None:
    from spancall.agent.router import RouterMode

    order = _resolve(AGENT_NB, "MODE_ORDER", {"RouterMode": RouterMode})
    modes = [mode for _label, mode in order]

    assert len(order) == 3, f"expected 3 modes, got {len(order)}: {[l for l, _ in order]!r}"
    assert modes == [RouterMode.LLM_ONLY, RouterMode.ROUTER, RouterMode.HYBRID_FILL], (
        f"mode order is {[m.value for m in modes]!r}; the comparison needs "
        "llm_only -> router -> hybrid_fill so each row adds one capability"
    )
    # The retired / out-of-scope modes must not creep back in.
    assert RouterMode.SPANCALL_ONLY not in modes
    assert RouterMode.SHADOW not in modes


def test_agentic_loads_from_hub_by_default(agent_nb: None) -> None:
    live = _live_code(AGENT_NB)
    assert re.search(r"HF_REPO_ID\s*=\s*ENV\.get\([\"']SPANCALL_HF_REPO_ID[\"']\)", live), (
        "repo id is not overridable from the environment"
    )
    assert re.search(r"USE_LOCAL_CHECKPOINT\s*=\s*False", live), (
        "local checkpoint path must be opt-in, not the default"
    )


def test_agentic_fetch_prefers_the_hub(agent_nb: None) -> None:
    """Behavioural: with ``local_dir=None`` the fetch must hit ``hf_hub_download``."""
    calls: List[Dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "/tmp/fake-resolve-config.json"

    fetch = _resolve(AGENT_NB, "_fetch", {"hf_hub_download": fake_download})
    assert fetch("convoicon/thing", "resolve_config.json", None, None) == "/tmp/fake-resolve-config.json"
    assert calls and calls[0]["repo_id"] == "convoicon/thing"
    assert calls[0]["filename"] == "resolve_config.json"

    # A local dir must bypass the network entirely.
    calls.clear()
    local = fetch("convoicon/thing", "model.pt", None, "/tmp/export")
    assert local == "/tmp/export/model.pt"
    assert not calls, "a local export still contacted the Hub"


def test_agentic_falls_back_to_noop_resolver(agent_nb: None) -> None:
    live = _live_code(AGENT_NB)
    # Bound to the NoOpResolver call itself: the other resolvers in the notebook
    # also pass ``abstain_reason``, so an unanchored search would still pass with
    # the fallback's own reason dropped.
    assert re.search(r"NoOpResolver\(\s*abstain_reason=[\"'][a-z_]+[\"']", live), (
        "the noop fallback does not record why it abstained"
    )


def test_agentic_warns_when_numbers_are_not_meaningful(agent_nb: None) -> None:
    """A noop resolver makes modes 2 and 3 identical to mode 1 by construction.

    Each caveat must actually be *printed* under its own guard. A bare mention
    of the word "plumbing" is satisfied by a section heading, which is not a
    warning, and the term also appears in mode 3b's own title text.
    """
    live = _live_code(AGENT_NB)
    assert re.search(r"RESOLVER_INFO\.get\([\"']kind[\"']\)\s*==\s*[\"']lean[\"']", live), (
        "the table cannot tell a real run from a plumbing run"
    )
    assert re.search(r"if\s+not\s+_has_lean\s*:\s*\n\s*print\(", live), (
        "the no-model caveat is not guarded on the resolver kind and printed"
    )
    assert re.search(r"plumbing check only", live), "the no-model caveat text is gone"
    assert re.search(r"BACKEND_KIND\s*==\s*[\"']deterministic[\"']", live), (
        "the deterministic-backend caveat is missing"
    )
    assert re.search(r"NOT of LLM quality", live), (
        "the deterministic-backend caveat does not say the numbers are not LLM quality"
    )


def test_agentic_usd_is_none_when_unpriced(agent_nb: None) -> None:
    """``0.0`` would read as 'free' rather than 'not priced'."""
    usd = _resolve(AGENT_NB, "usd", {"USD_PER_1K_IN": None, "USD_PER_1K_OUT": None})
    assert usd(1000, 1000) is None

    usd = _resolve(AGENT_NB, "usd", {"USD_PER_1K_IN": 2.0, "USD_PER_1K_OUT": 4.0})
    assert usd(1000, 500) == pytest.approx(2.0 + 2.0)


def test_agentic_table_covers_every_mode_and_suite(agent_nb: None) -> None:
    live = _live_code(AGENT_NB)
    assert re.search(r"for\s+_sname\s+in\s+TASK_SUITES", live), (
        "the table does not iterate the suites"
    )
    assert re.search(r"for\s+_label,\s*_mode\s+in\s+MODE_ORDER", live), (
        "the table does not iterate the modes"
    )
    assert "TABLE_ROWS.append(" in live
    assert re.search(r"agentic_mode_table\.json", live), "the table is not persisted"


def test_agentic_suites_are_imported_not_copied(agent_nb: None) -> None:
    """A copied task suite drifts from the offline eval it is compared against."""
    live = _live_code(AGENT_NB)
    assert re.search(r"spec_from_file_location\(", live), (
        "scripts/eval_agent.py is not imported"
    )
    assert "eval_agent.build_task_suite()" in live
    assert "eval_agent.build_long_trajectory_suite()" in live
    assert re.search(r"if\s+not\s+_suite[\s\S]{0,120}raise SystemExit", live), (
        "an empty suite is reported as a result instead of aborting"
    )


# ---------------------------------------------------------------------------
# 6. The two behavioural bugs found by executing the generated notebook
# ---------------------------------------------------------------------------
def test_scripted_instruction_parses_with_the_offline_backend(agent_nb: None) -> None:
    """The scripted demo only proves anything if the LLM arm can actually answer.

    ``DeterministicBackend._parse_route`` matches ``from X to Y on DATE``
    literally; a rephrased instruction makes the fill request unfillable and the
    demo 'falls back correctly' for a reason unrelated to hybrid fill.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "spancall_eval_agent_contract", ROOT / "scripts" / "eval_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    instruction = _resolve(AGENT_NB, "SCRIPTED_INSTRUCTION")
    parsed = module.DeterministicBackend._parse_route(instruction)
    assert parsed is not None, (
        f"SCRIPTED_INSTRUCTION {instruction!r} is unparseable by the offline backend, "
        "so the mode-3 fill demo would fall back for the wrong reason"
    )
    assert parsed == ["SFO", "JFK", "2026-11-05"], f"parsed route is {parsed!r}"


def test_scripted_partial_withholds_exactly_the_gaps(agent_nb: None) -> None:
    """The demo must give mode 3 a real skeleton: one known param, two gaps."""
    from spancall.infer.base import ResolveResult

    partial = _resolve(AGENT_NB, "_scripted_partial", {"ResolveResult": ResolveResult})
    res = partial([SimpleNamespace(name="search_flights")], "from SFO to JFK on 2026-11-05")

    assert res.hit is False
    assert res.status == "partial"
    assert res.tool_name == "search_flights"
    assert res.payload_partial == {"origin": "SFO"}
    assert set(res.missing_required) == {"destination", "date"}
    # A grounded param must be absent from the request, never re-asked.
    assert "origin" not in (res.missing_required or [])
    assert "origin" not in (res.missing_optional or [])


def test_render_trace_keys_execution_on_tool_output(agent_nb: None) -> None:
    """``selected_tool`` is the SpanCall *hypothesis* even on a fallback.

    ``ToolCallRouter._trace`` records ``res.tool_name`` into ``selected_tool``
    before the LLM has spoken, so keying "did a tool run" on it misattributes
    turns where the LLM declined to call anything.
    """
    live = _live_code(AGENT_NB)
    assert re.search(r"executed\s*=\s*step\.tool_output\s+is\s+not\s+None", live), (
        "render_trace does not use tool_output as the execution signal"
    )
    assert "SpanCall hypothesis only" in live, (
        "a hypothetical tool name on a fallback turn is not distinguished"
    )
    # The old defect: deriving provenance (and therefore "executed") from args.
    assert not re.search(r"provenance\s*=\s*param_provenance\(step\)\s*if\s+provenance", live)


def test_no_dead_code_in_the_demo_cell(agent_nb: None) -> None:
    """Two leftovers survived an earlier revision; neither should come back."""
    live = _live_code(AGENT_NB)
    assert "MODE_LABELS" not in live, "unused MODE_LABELS alias is back"
    assert "and None" not in live, "the 'attempt = (x or {}) and None' leftover is back"


# ---------------------------------------------------------------------------
# 7. Credentials never appear in an artifact
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = [
    r"hf_[A-Za-z0-9]{30,}",  # HF user access token
    r"sk-[A-Za-z0-9]{20,}",  # OpenAI-style key
    r"sk-ant-[A-Za-z0-9\-_]{20,}",  # Anthropic
    r"AIza[A-Za-z0-9\-_]{30,}",  # Google
]

_V3_ARTIFACTS = [
    TRAIN_NB,
    AGENT_NB,
    ROOT / "scripts" / "build_lean_v3_colab_notebook.py",
    ROOT / "scripts" / "build_lean_v3_agentic_notebook.py",
    ROOT / "scripts" / "compare_llm_spancall.py",
]


@pytest.mark.parametrize("path", _V3_ARTIFACTS, ids=[p.name for p in _V3_ARTIFACTS])
def test_no_hardcoded_credentials(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"{path.name} not built")
    text = path.read_text(encoding="utf-8")
    for pat in _SECRET_PATTERNS:
        found = re.findall(pat, text)
        assert not found, f"{path.name} contains what looks like a live key: {found[:2]!r}"


@pytest.mark.parametrize("path", [TRAIN_NB, AGENT_NB], ids=["train_eval", "agentic"])
def test_credentials_are_read_from_the_environment(path: Path) -> None:
    """A token may only ever be ``ENV.get(...)`` / ``os.environ.get(...)``."""
    if not path.is_file():
        pytest.skip(f"{path.name} not built")
    live = _live_code(path)
    literal = re.findall(
        r"(?:HF_TOKEN|HUGGING_FACE_TOKEN|API_KEY|WANDB_API_KEY)\s*=\s*[\"'][^\"']{8,}[\"']",
        live,
    )
    assert not literal, f"{path.name} assigns a credential literal: {literal!r}"
    if "HF_TOKEN" in live:
        assert re.search(r"HF_TOKEN\s*=\s*ENV\.get\(", live), (
            "HF_TOKEN is not read from the environment"
        )


def test_agentic_artifact_is_gitignored() -> None:
    """The notebook is runnable-with-a-key, so it must never be committed."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^notebooks/\*_agentic\*\.ipynb$", ignore, re.M), (
        "notebooks/*_agentic*.ipynb is not ignored"
    )
    assert re.search(r"^wandb/$", ignore, re.M), "wandb/ is not ignored"


# ---------------------------------------------------------------------------
# 8. The CLI can actually drive all three modes
# ---------------------------------------------------------------------------
def test_eval_agent_cli_exposes_every_mode_with_a_counter() -> None:
    """Mode 3 was Python-only, so a CLI comparison could not reproduce it."""
    from scripts.eval_agent import parse_args

    args = parse_args(["--mode", "hybrid_fill"])
    assert args.mode == "hybrid_fill"
    assert parse_args(["--mode", "llm_only"]).mode == "llm_only"

    src = (ROOT / "scripts" / "eval_agent.py").read_text(encoding="utf-8")
    # A hybrid fill calls the LLM, so it needs its own counter: folding it into
    # spancall_hits would report mode 3 as a strictly better fast path.
    assert re.search(r"[\"']hybrid_fills[\"']\s*:\s*0", src), (
        "no dedicated hybrid_fills counter; hybrid fills would be uncounted"
    )
    assert re.search(
        r"elif\s+st\.decision\s*==\s*RoutingDecision\.HYBRID_FILL[\s\S]{0,120}hybrid_fills",
        src,
    )