#!/usr/bin/env python3
"""Diagnostic script for C3 (Oracle-pointer ceiling test) and C1 (Label audit).

See plans/accuracy_improvement_deep_dive.md §Phase C.
"""

from collections import Counter
from datetime import date
import json
import os
import sys
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from span_template.template import (
    Param,
    RecentRow,
    SpanTemplateOutput,
    Tool,
    build_input,
)
from spancall.data.align import tooldef_to_template_tool
from spancall.data.schema import (
    ParameterSchema,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from spancall.eval.toolcall.evaluator import payload_equal, _deep_equal
from spancall.infer.cast import StrictTypeCaster, SchemaValidator
from spancall.model.masks import CandidateMaskBuilder


def run_c3_oracle_ceiling(path: str, limit: Optional[int] = 100) -> Dict[str, Any]:
    """C3: Oracle-pointer ceiling test.
    Extracts GT arguments via oracle candidate matching + cast, checks payload_equal on Stratum F rows.
    """
    print(f"\n=================== C3: Oracle-Pointer Ceiling Test on {path} (limit={limit}) ===================")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", local_files_only=True)
    mask_builder = CandidateMaskBuilder()
    caster = StrictTypeCaster()
    validator = SchemaValidator()

    total_f = 0
    oracle_matches = 0
    failure_details = Counter()

    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            s = UnifiedSample.from_dict(json.loads(line))
            if s.stratum != Stratum.F:
                continue
            if not s.tools or not s.ground_truth_calls:
                continue

            gt_call = s.ground_truth_calls[0]
            gt_tool_name = gt_call.tool_name
            gt_tool = next((t for t in s.tools if t.name == gt_tool_name), None)
            if gt_tool is None:
                continue

            gt_args = dict(gt_call.arguments or {})
            clock = date.fromisoformat(s.current_time_iso[:10]) if s.current_time_iso else date(2026, 11, 8)
            recent = [
                RecentRow(key=str(r.get("key", "")), value=str(r.get("value", "")), source=str(r.get("source", "recent")))
                for r in (s.recent_context or [])
            ]

            try:
                out = build_input(
                    tooldef_to_template_tool(gt_tool),
                    s.utterance,
                    recent_context=recent,
                    now=clock,
                )
            except Exception as e:
                failure_details[f"render_err: {type(e).__name__}"] += 1
                continue

            enc = tokenizer(out.text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = enc["offset_mapping"]

            total_f += 1
            oracle_payload: Dict[str, Any] = {}

            for param in gt_tool.parameters:
                if param.name not in gt_args:
                    continue  # optional omitted by GT
                gt_val = gt_args[param.name]
                masks = mask_builder.build_param_masks(param, out.entries, offsets, out.text)
                cands = list(masks["candidates"])

                picked_cand = None
                for c in cands:
                    if c.canon_value is not None and _deep_equal(c.canon_value, gt_val):
                        picked_cand = c
                        break

                if picked_cand is not None:
                    ok, cast_val, reason = caster.cast_choice(picked_cand, param, output=out)
                    oracle_payload[param.name] = cast_val if ok else picked_cand.canon_value
                elif gt_val is None:
                    oracle_payload[param.name] = None
                else:
                    failure_details[f"unmatched_candidate:{param.name}"] += 1

            if payload_equal(gt_args, oracle_payload, schema=gt_tool):
                oracle_matches += 1
            else:
                failure_details["payload_unequal"] += 1

            if limit is not None and total_f >= limit:
                break

    pct = (oracle_matches / max(1, total_f)) * 100
    print(f"Total Stratum F rows examined: {total_f}")
    print(f"Oracle-Pointer Exact Matches: {oracle_matches}/{total_f} ({pct:.2f}%)")
    if failure_details:
        print("Top Failure Breakdown:", failure_details.most_common(10))
    print("==================================================================")
    return {
        "total_f": total_f,
        "oracle_matches": oracle_matches,
        "ceiling_pct": pct,
        "failure_details": dict(failure_details),
    }


def run_c1_label_audit(dev_path: str, n_rows: int = 200) -> Dict[str, Any]:
    """C1: Label audit on dev rows.
    Checks how many Stratum F parameters are silently converted to abstain (pos_indices == [0]).
    """
    print(f"\n=================== C1: Label Audit on {n_rows} rows of {dev_path} ===================")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", local_files_only=True)
    mask_builder = CandidateMaskBuilder()

    stratum_counts = Counter()
    silent_abstain_params = 0
    total_f_params = 0
    rows_with_silent_abstain = 0
    samples_f = 0
    reasons_silent = Counter()

    with open(dev_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n_rows:
                break
            line = line.strip()
            if not line:
                continue
            s = UnifiedSample.from_dict(json.loads(line))
            stratum_counts[s.stratum.value] += 1
            if s.stratum != Stratum.F:
                continue
            if not s.tools or not s.ground_truth_calls:
                continue

            gt_call = s.ground_truth_calls[0]
            gt_tool = next((t for t in s.tools if t.name == gt_call.tool_name), None)
            if not gt_tool:
                continue

            samples_f += 1
            gt_args = dict(gt_call.arguments or {})
            clock = date.fromisoformat(s.current_time_iso[:10]) if s.current_time_iso else date(2026, 11, 8)
            recent = [
                RecentRow(key=str(r.get("key", "")), value=str(r.get("value", "")), source=str(r.get("source", "recent")))
                for r in (s.recent_context or [])
            ]

            try:
                out = build_input(
                    tooldef_to_template_tool(gt_tool),
                    s.utterance,
                    recent_context=recent,
                    now=clock,
                )
            except Exception:
                continue

            enc = tokenizer(out.text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = enc["offset_mapping"]

            row_had_silent = False
            for param in gt_tool.parameters:
                if param.name not in gt_args:
                    continue  # optional not present in GT
                total_f_params += 1
                masks = mask_builder.build_param_masks(param, out.entries, offsets, out.text)
                cands = list(masks["candidates"])

                targets = (s.target_spans or {}).get(param.name, [])
                matched_cis = []
                for t in targets:
                    if t.target_type == "null_ctx":
                        for ci, c in enumerate(cands):
                            if c.region == "null" and ci not in matched_cis:
                                matched_cis.append(ci)
                                break
                        continue
                    found = False
                    if t.canonical_value is not None:
                        t_str = str(t.canonical_value).strip().lower()
                        for ci, c in enumerate(cands):
                            if c.canon_value is not None:
                                c_str = str(c.canon_value).strip().lower()
                                if c.canon_value == t.canonical_value or c_str == t_str:
                                    if ci not in matched_cis:
                                        matched_cis.append(ci)
                                    found = True
                                    break
                    if not found and t.char_start is not None:
                        for ci, c in enumerate(cands):
                            if c.char_start == t.char_start and c.char_end == t.char_end:
                                if ci not in matched_cis:
                                    matched_cis.append(ci)
                                found = True
                                break

                if not matched_cis:
                    silent_abstain_params += 1
                    row_had_silent = True
                    reasons_silent[f"{param.name} (type={param.param_type}) | GT={gt_args.get(param.name)} | n_cands={len(cands)}"] += 1

            if row_had_silent:
                rows_with_silent_abstain += 1

    print(f"Stratum distribution in first {n_rows} rows: {dict(stratum_counts)}")
    print(f"Stratum F samples analyzed: {samples_f}")
    print(f"Total Stratum F ground-truth parameters: {total_f_params}")
    print(f"Silent-abstain parameters (pos_indices defaulted to [0]): {silent_abstain_params}/{total_f_params} ({(silent_abstain_params/max(1, total_f_params))*100:.2f}%)")
    print(f"Stratum F rows with at least one silent-abstain: {rows_with_silent_abstain}/{samples_f} ({(rows_with_silent_abstain/max(1, samples_f))*100:.2f}%)")
    if reasons_silent:
        print("\nSample Silent-Abstain Instances:")
        for r, cnt in reasons_silent.most_common(12):
            print(f"  [{cnt}x] {r}")
    print("==================================================================")
    return {
        "stratum_counts": dict(stratum_counts),
        "samples_f": samples_f,
        "total_f_params": total_f_params,
        "silent_abstain_params": silent_abstain_params,
        "rows_with_silent_abstain": rows_with_silent_abstain,
    }


if __name__ == "__main__":
    eval_path = "data/processed/hermes_eval.jsonl"
    dev_path = "data/processed/hermes_dev.jsonl"
    run_c3_oracle_ceiling(eval_path, limit=100)
    run_c1_label_audit(dev_path, n_rows=200)
