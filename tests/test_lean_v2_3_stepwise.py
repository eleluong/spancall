"""Tests for Lean v2.3 training layout parity and step-by-step evaluation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from spancall.data.schema import (
    GroundTruthCall,
    ParameterSchema,
    SpanTarget,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.infer.base import FakeTokenizer
from spancall.lean.eval_p1 import (
    build_stratified_eval_subset,
    classify_eval_cohort,
    eval_param_extraction_lean,
)
from spancall.lean.resolver import LeanResolveResult
from spancall.lean.tensorize import ToolPool, tensorize_sample_lean
from spancall.model.masks import CandidateMaskBuilder


def _tool(name: str, param: str = "city") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Run {name}",
        parameters=[ParameterSchema(name=param, description=f"{param} arg", param_type="string", required=True)],
    )


def _sample(sid: str, tools: list[ToolDefinition], gt_name: str, stratum: Stratum = Stratum.F) -> UnifiedSample:
    return UnifiedSample(
        sample_id=sid,
        source_dataset="hermes_eval",
        utterance="Check weather in Paris tomorrow",
        tools=tools,
        ground_truth_calls=[GroundTruthCall(tool_name=gt_name, arguments={"city": "Paris"})],
        stratum=stratum,
        target_spans={
            "city": [
                SpanTarget(
                    param_name="city",
                    char_start=0,
                    char_end=5,
                    utt_rel_start=17,
                    utt_rel_end=22,
                    canonical_value="Paris",
                    target_type="utterance_span",
                )
            ]
        },
    )


def _sample_dict(sid: str, tools: list[ToolDefinition], gt_name: str, stratum: Stratum = Stratum.F) -> dict:
    d = asdict(_sample(sid, tools, gt_name, stratum))
    d["stratum"] = stratum.value
    d["split"] = d["split"].value if hasattr(d["split"], "value") else d["split"]
    return d


def test_tensorize_jitter_reaches_t1_on_native_multi_tool() -> None:
    t1 = _tool("get_weather")
    t2 = _tool("get_forecast")
    t3 = _tool("get_alerts")
    pool = ToolPool([t1, t2, t3], seed=42)
    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=1024, max_span_width=8)

    seen_t = set()
    for i in range(40):
        s = _sample(f"multi-{i}", [t1, t2], "get_weather")
        row = tensorize_sample_lean(
            s,
            tokenizer=tok,
            mask_builder=mb,
            max_seq_len=1024,
            tool_pool=pool,
            pack_tools=3,
            jitter=True,
            jitter_seed=42,
        )
        assert row is not None
        seen_t.add(len(row["tool_positions"]))

    assert 1 in seen_t, f"Native multi-tool row never rendered T=1 under jitter: {sorted(seen_t)}"


def test_build_stratified_eval_subset_excludes_missing_schema(tmp_path: Path) -> None:
    src = tmp_path / "hermes_eval.jsonl"
    out = tmp_path / "subset.jsonl"

    rows = []
    # 10 F_func rows
    for i in range(10):
        rows.append(_sample_dict(f"func-{i}", [_tool("get_weather"), _tool("get_time")], "get_weather", Stratum.F))
    # 5 F_json rows
    for i in range(5):
        rows.append(_sample_dict(f"json-{i}", [_tool("json_output")], "json_output", Stratum.F))
    # 8 A_clean rows
    for i in range(8):
        rows.append(_sample_dict(f"aclean-{i}", [_tool("get_weather")], "get_weather", Stratum.A))
    # 4 A_adv rows
    for i in range(4):
        rows.append(_sample_dict(f"aadv-{i}::r7", [_tool("get_weather")], "get_weather", Stratum.A))
    # 6 A_missing_gt_schema rows (must be excluded)
    for i in range(6):
        rows.append(_sample_dict(f"miss-{i}", [_tool("ExpertQAExtractor")], "json_output", Stratum.A))

    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    summary = build_stratified_eval_subset("hermes_eval", str(tmp_path), str(out), limit=20, seed=42)
    assert summary["available_by_cohort"]["A_missing_gt_schema"] == 6
    assert summary["sampled_total"] == 20

    sampled = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    cohorts = {classify_eval_cohort(r) for r in sampled}
    assert "A_missing_gt_schema" not in cohorts
    assert {"F_func", "F_json", "A_clean", "A_adv"}.issubset(cohorts)


class _DummyCtxResolver:
    supports_context_schemas = True
    supports_candidate_parity = False

    def try_resolve(self, tool, utterance, recent_context=None, now=None, context_schemas=None):
        t_count = len(context_schemas) if context_schemas is not None else 1
        return LeanResolveResult(
            hit=True,
            tool_name=tool.name,
            payload={"city": "Paris"},
            payload_partial={"city": "Paris"},
            features={"render_T": t_count, "pointer_collapsed": False},
        )


def test_eval_param_extraction_context_mode_single_vs_all(tmp_path: Path) -> None:
    src = tmp_path / "sub.jsonl"
    s1 = _sample_dict("f-1", [_tool("get_weather"), _tool("get_time")], "get_weather", Stratum.F)
    s2 = _sample_dict("j-1", [_tool("json_output")], "json_output", Stratum.F)
    src.write_text(json.dumps(s1) + "\n" + json.dumps(s2) + "\n", encoding="utf-8")

    res = _DummyCtxResolver()
    m_single = eval_param_extraction_lean(
        str(src), resolver=res, data_dir=str(tmp_path),
        context_mode="single", exclude_tools=["json_output"], quiet=True,
    )
    m_all = eval_param_extraction_lean(
        str(src), resolver=res, data_dir=str(tmp_path),
        context_mode="all", exclude_tools=["json_output"], quiet=True,
    )
    assert m_single["n"] == 1
    assert m_single["render_T_hist"] == {"1": 1}
    assert m_all["render_T_hist"] == {"2": 1}
    assert m_single["pointer_collapse_rate_F"] == 0.0
