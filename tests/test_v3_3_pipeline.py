"""Offline end-to-end tests for the SpanCall Lean v3.3 pipeline:

1. Frozen fair eval builder & SHA256 manifest verification
2. LeanTrainer ``early_stopping_agg="macro_mean"`` across two dev splits
3. Local multi-model LLM eval CLI (resume, ``--retry-errors``, YAML config, fallback merge)
4. Unified audited exporter (JSON/MD/CSV/LaTeX + fail-loud coverage & hash gates)
5. v3.3 train/eval notebook builder contract
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
import yaml

from spancall.eval.llm_parity import verify_eval_manifest
from spancall.train.config import OptimizationConfig


def _make_sample(
    sample_id: str,
    corpus: str,
    *,
    stratum: str = "F",
    is_complex: bool = False,
    is_f_json: bool = False,
    n_calls: int = 1,
    missing_schema: bool = False,
    is_adv: bool = False,
    n_tools: int = 2,
) -> Dict[str, Any]:
    tool_name = "json_output" if is_f_json else "book_flight"
    param_type = "array" if is_complex else "string"
    arg_val: Any = ["NYC", "LAX"] if is_complex else "NYC"
    tools = [
        {
            "name": "other_tool" if missing_schema else tool_name,
            "description": "Book a flight",
            "parameters": [
                {"name": "dest", "param_type": param_type, "required": True, "description": "Destination"}
            ],
        }
    ]
    for i in range(1, n_tools):
        tools.append({
            "name": f"distractor_{i}",
            "description": f"Distractor tool {i}",
            "parameters": [{"name": "x", "param_type": "string", "required": False, "description": "x"}],
        })

    gt_calls = [
        {"tool_name": tool_name, "arguments": {"dest": arg_val}}
        for _ in range(n_calls)
    ]
    return {
        "sample_id": sample_id,
        "source_dataset": f"{corpus}_eval",
        "utterance": "Book a flight to NYC" if not is_complex else "Book flights to NYC and LAX",
        "stratum": stratum,
        "tools": tools,
        "ground_truth_calls": gt_calls,
        "ground_truth_tool": tool_name,
        "ground_truth_payload": {"dest": arg_val},
        "recent_context": [],
        "current_time_iso": "2026-11-19",
        "metadata": {"adversarial": True} if is_adv else {},
        "abstain_reason": "adversarial" if is_adv else ("ambiguous" if stratum == "A" else None),
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _build_synthetic_corpus(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "processed"
    eval_dir = tmp_path / "eval_v3_3"

    # Hermes: 4 F_func normal, 2 F_func complex, 2 A_clean normal, 1 A_clean complex,
    #         2 A_adv normal, 1 F_json (excluded from Tier 1), 1 multi-call (dropped), 1 missing_schema (dropped)
    hermes_rows = [
        *[_make_sample(f"h_fn_{i}", "hermes", stratum="F", is_complex=False, n_tools=2) for i in range(4)],
        *[_make_sample(f"h_fc_{i}", "hermes", stratum="F", is_complex=True, n_tools=3) for i in range(2)],
        *[_make_sample(f"h_acn_{i}", "hermes", stratum="A", is_complex=False, n_tools=2) for i in range(2)],
        _make_sample("h_acc_0", "hermes", stratum="A", is_complex=True, n_tools=2),
        *[_make_sample(f"h_adv_{i}", "hermes", stratum="A", is_adv=True, n_tools=2) for i in range(2)],
        _make_sample("h_fjson_0", "hermes", stratum="F", is_f_json=True),
        _make_sample("h_multi_0", "hermes", stratum="F", n_calls=2),
        _make_sample("h_miss_0", "hermes", stratum="F", missing_schema=True),
    ]
    # xLAM: larger counts in every cell
    xlam_rows = [
        *[_make_sample(f"x_fn_{i}", "xlam", stratum="F", is_complex=False, n_tools=4) for i in range(8)],
        *[_make_sample(f"x_fc_{i}", "xlam", stratum="F", is_complex=True, n_tools=6) for i in range(5)],
        *[_make_sample(f"x_acn_{i}", "xlam", stratum="A", is_complex=False, n_tools=2) for i in range(4)],
        *[_make_sample(f"x_acc_{i}", "xlam", stratum="A", is_complex=True, n_tools=3) for i in range(3)],
        *[_make_sample(f"x_adv_{i}", "xlam", stratum="A", is_adv=True, n_tools=4) for i in range(4)],
    ]
    _write_jsonl(data_dir / "hermes_eval.jsonl", hermes_rows)
    _write_jsonl(data_dir / "xlam_eval.jsonl", xlam_rows)
    _write_jsonl(data_dir / "hermes_test_ood_schema.jsonl", hermes_rows[:4])
    _write_jsonl(data_dir / "xlam_test_ood_schema.jsonl", xlam_rows[:4])

    from scripts.build_fair_eval_v3_3 import build_fair_eval_dataset

    build_fair_eval_dataset(data_dir, eval_dir, seed=42, check_step0=False, quiet=True)
    return data_dir, eval_dir


# --------------------------------------------------------------------------- #
# 1. Fair Eval Builder & Manifest Integrity
# --------------------------------------------------------------------------- #
def test_fair_eval_builder_exact_parity_and_tamper_detection(tmp_path: Path) -> None:
    _, eval_dir = _build_synthetic_corpus(tmp_path)
    manifest = verify_eval_manifest(eval_dir)

    fh = manifest["splits"]["fair_hermes_eval"]
    fx = manifest["splits"]["fair_xlam_eval"]
    # Exact 1:1 parity on total rows and every (cohort, complexity) cell
    assert fh["n_rows"] == fx["n_rows"] == 11
    assert fh["cell_counts"] == fx["cell_counts"]
    assert "F_json" not in fh["cohort_counts"]
    assert manifest["splits"]["hermes_eval_f_json"]["n_rows"] == 1

    # Tampering with a frozen file must raise ValueError
    target = eval_dir / fh["filename"]
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_eval_manifest(eval_dir)


# --------------------------------------------------------------------------- #
# 2. LeanTrainer macro_mean early stopping
# --------------------------------------------------------------------------- #
def test_optimization_config_early_stopping_agg_default() -> None:
    assert OptimizationConfig().early_stopping_agg == "first"
    assert OptimizationConfig(early_stopping_agg="macro_mean").early_stopping_agg == "macro_mean"


def test_trainer_macro_mean_selects_macro_best_epoch(tmp_path: Path) -> None:
    import torch
    from spancall.lean.trainer import LeanTrainer
    from spancall.train.config import TrainingConfig

    cfg = TrainingConfig(
        experiment_name="test_macro",
        seed=42,
        output_dir=str(tmp_path),
        optimization=OptimizationConfig(
            num_epochs=3,
            early_stopping_patience=1,
            early_stopping_agg="macro_mean",
            restore_best_on_exit=True,
        ),
    )

    model = torch.nn.Linear(2, 2)
    trainer = object.__new__(LeanTrainer)
    trainer.config = cfg
    trainer.model = model
    trainer.global_step = 10
    trainer.history = {"epoch_eval": []}
    trainer._stopped_early = False
    events: List[Dict[str, Any]] = []
    saved_cps: List[str] = []
    trainer._log_event = lambda rec: events.append(rec)  # type: ignore[method-assign]
    trainer._wandb_log = lambda *a, **kw: None  # type: ignore[method-assign]
    trainer.save_checkpoint = lambda path: saved_cps.append(str(path))  # type: ignore[method-assign]

    # Epoch 1: hermes=0.60, xlam=0.80 -> macro=0.70 (best)
    # Epoch 2: hermes=0.72, xlam=0.76 -> macro=0.74 (NEW BEST! even though xlam dropped)
    # Epoch 3: hermes=0.68, xlam=0.76 -> macro=0.72 (worse -> early stop!)
    epochs_recs = [
        [{"split": "dev_hermes", "exact_call": 0.60, "eval_loss": 1.1},
         {"split": "dev_xlam", "exact_call": 0.80, "eval_loss": 0.9}],
        [{"split": "dev_hermes", "exact_call": 0.72, "eval_loss": 0.95},
         {"split": "dev_xlam", "exact_call": 0.76, "eval_loss": 0.92}],
        [{"split": "dev_hermes", "exact_call": 0.68, "eval_loss": 1.0},
         {"split": "dev_xlam", "exact_call": 0.76, "eval_loss": 0.93}],
    ]
    stopped = False
    for epoch, recs in enumerate(epochs_recs, start=1):
        stopped = trainer._update_early_stopping("T2", epoch, 3, recs)
        if stopped:
            break

    assert stopped is True
    assert trainer._best_epoch == 2
    assert pytest.approx(trainer._best_curve_metric, abs=1e-6) == 0.74
    assert len(saved_cps) == 2
    assert len([e for e in events if e["event"] == "epoch_eval_macro"]) == 3
    assert any(e["event"] == "early_stop" and e["epoch"] == 3 for e in events)


# --------------------------------------------------------------------------- #
# 3. Local Multi-Model LLM Eval CLI & 4. Unified Exporter
# --------------------------------------------------------------------------- #
def test_local_llm_eval_resume_retry_and_unified_export(tmp_path: Path) -> None:
    from scripts.eval_llm_local import main as llm_eval_main
    from scripts.export_v3_3_results import export_all_results

    _, eval_dir = _build_synthetic_corpus(tmp_path)
    results_dir = tmp_path / "results_v3_3"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create SpanCall predictions for Tier 1 + Tier 2
    manifest = verify_eval_manifest(eval_dir)
    canned_llm: List[Dict[str, Any]] = []
    for sp in ("fair_hermes_eval", "fair_xlam_eval", "full_hermes_test_ood_schema", "full_xlam_test_ood_schema"):
        rows = [json.loads(l) for l in (eval_dir / manifest["splits"][sp]["filename"]).read_text().splitlines() if l.strip()]
        sc_preds = []
        for idx, r in enumerate(rows):
            gt = r["ground_truth_calls"][0]
            # SpanCall commits on even F rows, abstains on odd F rows and all A rows
            is_f = r["stratum"] == "F"
            hit = is_f and (idx % 2 == 0)
            sc_preds.append({
                "sample_id": r["sample_id"],
                "hit": hit,
                "status": "hit" if hit else "abstain",
                "tool_name": gt["tool_name"] if hit else None,
                "payload": gt["arguments"] if hit else None,
                "calls": [{"tool_name": gt["tool_name"], "payload": gt["arguments"], "status": "hit"}] if hit else [],
                "latency_ms": 12.5,
            })
            if sp in ("fair_hermes_eval", "fair_xlam_eval"):
                # Canned LLM: answers all F rows correctly, abstains on A rows, with 1 deliberate error on idx==0
                canned_llm.append({
                    "sample_id": r["sample_id"],
                    "tool_calls": [{"tool_name": gt["tool_name"], "arguments": gt["arguments"]}] if is_f else [],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 25},
                    "latency_ms": 450.0,
                    "error": "transient_503" if (sp == "fair_hermes_eval" and idx == 0) else None,
                })
        _write_jsonl(results_dir / f"preds_{sp}_twostep.jsonl", sc_preds)

    canned_path = tmp_path / "canned.jsonl"
    _write_jsonl(canned_path, canned_llm)

    # Write a 2-model YAML config (one enabled, one disabled)
    cfg_path = tmp_path / "llm_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "eval_dir": str(eval_dir),
        "results_dir": str(results_dir),
        "splits": ["fair_hermes_eval", "fair_xlam_eval"],
        "models": [
            {"slug": "model-a", "model": "fake/model-a", "enabled": True},
            {"slug": "model-b", "model": "fake/model-b", "enabled": False},
        ],
    }), encoding="utf-8")

    # Run local LLM eval (first pass: 1 error on fair_hermes_eval idx 0)
    rc = llm_eval_main([
        "--config", str(cfg_path),
        "--fake-responses", str(canned_path),
        "--quiet",
    ])
    assert rc == 0
    assert (results_dir / "llm" / "model-a" / "summary.json").is_file()
    assert not (results_dir / "llm" / "model-b").exists()
    sum1 = json.loads((results_dir / "llm" / "model-a" / "summary.json").read_text())
    assert sum1["splits"]["fair_hermes_eval"]["llm_only"]["tokens"]["n_errors"] == 1
    assert "spancall_plus_llm_fallback" in sum1["splits"]["fair_hermes_eval"]

    # Fix the transient error in canned responses and verify --retry-errors recovers it
    for row in canned_llm:
        row["error"] = None
    _write_jsonl(canned_path, canned_llm)
    rc2 = llm_eval_main([
        "--config", str(cfg_path),
        "--fake-responses", str(canned_path),
        "--retry-errors",
        "--quiet",
    ])
    assert rc2 == 0
    sum2 = json.loads((results_dir / "llm" / "model-a" / "summary.json").read_text())
    assert sum2["splits"]["fair_hermes_eval"]["llm_only"]["tokens"]["n_errors"] == 0

    # Verify a subsequent --limit 2 smoke run NEVER truncates the 11 cached rows on disk
    rc_lim = llm_eval_main([
        "--config", str(cfg_path),
        "--fake-responses", str(canned_path),
        "--limit", "2",
        "--quiet",
    ])
    assert rc_lim == 0
    cached_lines = (results_dir / "llm" / "model-a" / "fair_hermes_eval_preds.jsonl").read_text().splitlines()
    assert len(cached_lines) == 11

    # Write a minimal train_log.jsonl
    _write_jsonl(results_dir / "train_log.jsonl", [
        {"event": "epoch_eval", "epoch": 1, "split": "dev_hermes", "n": 100, "span_acc": 0.85, "exact_params_only": 0.80, "exact_call": 0.79, "tool_acc": 0.99, "eval_loss": 0.9},
        {"event": "epoch_eval_macro", "epoch": 1, "splits": ["dev_hermes", "dev_xlam"], "macro_exact_call": 0.80, "macro_eval_loss": 0.88},
    ])

    # Run unified exporter
    export_dir = results_dir / "export"
    report = export_all_results(
        eval_dir, results_dir, export_dir,
        bootstrap_n=200, seed=42, quiet=True,
    )
    assert report["audit"]["status"] == "PASS"
    assert report["audit"]["coverage_all_ok"] is True
    assert report["audit"]["total_llm_errors"] == 0

    # Verify macro_avg rows, shared_1-5 tool-bucket parity, and bootstrap CIs were produced
    splits_in_headline = {r["split"] for r in report["tier1_headline"]}
    assert {"fair_hermes_eval", "fair_xlam_eval", "macro_avg"}.issubset(splits_in_headline)
    assert any(r["bucket"] == "shared_1-5" for r in report["tier1_tool_buckets"])
    assert len(report["tier1_bootstrap_ci"]) >= 2

    # Verify Colab/local eval_manifest.json mismatch fails loud
    bad_snap = dict(manifest)
    bad_snap["splits"] = {k: dict(v) for k, v in manifest["splits"].items()}
    bad_snap["splits"]["fair_hermes_eval"]["sha256"] = "0" * 64
    (results_dir / "eval_manifest.json").write_text(json.dumps(bad_snap), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with"):
        export_all_results(eval_dir, results_dir, export_dir, bootstrap_n=0, quiet=True)
    (results_dir / "eval_manifest.json").unlink()

    # Verify all exported files exist
    assert (export_dir / "final_report.json").is_file()
    assert (export_dir / "audit_report.md").is_file()
    for csv_name in ("fair_headline.csv", "fair_complexity.csv", "fair_tool_buckets.csv", "full_ood_generalization.csv", "train_dev_curve.csv"):
        assert (export_dir / "tables" / csv_name).is_file()
    for tex_name in ("tab_v3_3_headline.tex", "tab_v3_3_complexity.tex", "tab_v3_3_buckets.tex", "tab_v3_3_ood.tex"):
        tex = (export_dir / "latex" / tex_name).read_text(encoding="utf-8")
        assert r"\begin{tabular}" in tex and r"\bottomrule" in tex

    # Verify fail-loud when a sample_id is missing from predictions
    sc_h = results_dir / "preds_fair_hermes_eval_twostep.jsonl"
    lines = sc_h.read_text(encoding="utf-8").splitlines()
    sc_h.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AUDIT FAILURE"):
        export_all_results(eval_dir, results_dir, export_dir, bootstrap_n=0, quiet=True)

    # With allow_partial=True, succeeds and stamps PARTIAL
    partial_rep = export_all_results(eval_dir, results_dir, export_dir, allow_partial=True, bootstrap_n=0, quiet=True)
    assert partial_rep["audit"]["status"] == "PARTIAL"


# --------------------------------------------------------------------------- #
# 5. v3.3 Notebook Builder Contract
# --------------------------------------------------------------------------- #
def test_v3_3_notebook_matches_builder_and_invariants() -> None:
    from scripts.build_lean_v3_3_train_eval_notebook import OUT, build

    build(check=True)
    nb = json.loads(OUT.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    for token in (
        "BALANCE_TRAIN_CORPORA = True",
        "early_stopping_agg",
        "macro_mean",
        "build_fair_eval_v3_3.py",
        "export_v3_3_results.py",
        'resolve_mode="two_step"',
        '"head_dim"',
        "assert len(_probe_ids & _dev_ids) == 0",
    ):
        assert token in all_src, f"Missing v3.3 invariant token in notebook: {token}"
