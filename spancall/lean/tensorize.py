"""Lean Tensorization and Batch Adapters.

Provides:
- convert_legacy_batch_to_lean: translates legacy batch tensors (where 0=abstain,
  v>0 = candidate v-1) into 0-indexed candidate positives in [0, C) with -1 pad.
- convert_legacy_row_to_lean: translates a single serialized dict row.
- tensorize_sample_lean: direct tensorization for UnifiedSample without abstain targets.
- LeanBatchCollate: collates a list of sample tensor dicts into a padded batch.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from spancall.data.align import tooldef_to_template_tool
from spancall.data.schema import ParameterSchema, Stratum, ToolDefinition, UnifiedSample
from span_template.template import RecentRow, build_input, build_input_multi


def is_legacy_batch(batch_or_row: Dict[str, Any]) -> bool:
    """Return True if batch/row carries legacy 1-indexed pos_indices (0=abstain)."""
    if batch_or_row.get("_lean_converted"):
        return False
    return any(
        k in batch_or_row
        for k in ("accept_targets", "reason_targets", "abstain_flags", "cue_mask")
    )


def convert_legacy_batch_to_lean(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a legacy batch dict (with 0=abstain, v>0=cand_v-1) into Lean format.

    - pos_indices: maps v > 0 to v - 1, and 0 (abstain) / -1 (pad) to -1.
    - idempotent: marks converted dict with ``_lean_converted = True``.
    """
    if batch.get("_lean_converted"):
        return batch
    lean_batch = dict(batch)
    if "pos_indices" in batch:
        pos = batch["pos_indices"]
        lean_batch["pos_indices"] = torch.where(pos > 0, pos - 1, -1)
        if "pointer_positives" in lean_batch:
            lean_batch["pointer_positives"] = lean_batch["pos_indices"]
    lean_batch["_lean_converted"] = True  # type: ignore[assignment]
    return lean_batch


def convert_legacy_row_to_lean(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single uncollated dict row from legacy format to Lean format."""
    if row.get("_lean_converted"):
        return row
    out = dict(row)
    if "pos_indices" in row:
        pos = row["pos_indices"]
        if isinstance(pos, torch.Tensor):
            out["pos_indices"] = torch.where(pos > 0, pos - 1, -1)
        elif isinstance(pos, list):
            out["pos_indices"] = [
                [val - 1 if val > 0 else -1 for val in k_row]
                for k_row in pos
            ]
    out["_lean_converted"] = True
    return out


def has_positive_targets(row: Dict[str, Any]) -> bool:
    """Return True if a row has at least one valid positive candidate index."""
    pos = row.get("pos_indices")
    if pos is None:
        return False
    is_legacy = is_legacy_batch(row)
    thr = 0 if is_legacy else -1
    if isinstance(pos, torch.Tensor):
        return bool((pos > thr).any().item())
    if isinstance(pos, list):
        return any(
            (val > thr)
            for k_row in pos
            for val in (k_row if isinstance(k_row, list) else [k_row])
        )
    return False


def _snap_token_range(
    cs: int,
    ce: int,
    offsets: Sequence[Tuple[int, int]],
    max_seq_len: int,
) -> Optional[Tuple[int, int]]:
    """Find first and last token overlapping [cs, ce)."""
    if ce <= cs:
        return None
    first = last = None
    for i, (tcs, tce) in enumerate(offsets):
        if i >= max_seq_len:
            break
        if tce <= tcs:
            continue
        if tce > cs and tcs < ce:
            if first is None:
                first = i
            last = i
    return (first, last) if first is not None else None


def _recent_rows(sample: UnifiedSample) -> List[RecentRow]:
    """Convert sample.recent_context to template RecentRow list."""
    rows_out: List[RecentRow] = []
    for r in (sample.recent_context or []):
        if isinstance(r, RecentRow):
            rows_out.append(r)
        elif isinstance(r, dict):
            try:
                rows_out.append(
                    RecentRow(
                        key=str(r.get("key", "")),
                        value=str(r.get("value", "")),
                        source=str(r.get("source", "recent")),
                    )
                )
            except ValueError:
                continue
    return rows_out


def tensorize_sample_lean(
    sample: UnifiedSample,
    tokenizer: Any,
    mask_builder: Any,
    max_seq_len: int = 512,
    max_cands: int = 512,
    max_pos: int = 8,
    pack_multi_tools: bool = False,
) -> Optional[Dict[str, Any]]:
    """Direct tensorization of UnifiedSample into Lean training format.

    Eliminates column-0 abstain: candidate positions are 0-indexed in [0, C).
    Rows with no positive candidate are padded with -1.
    """
    if not sample.tools or not sample.ground_truth_calls:
        return None

    gt_call = sample.ground_truth_calls[0]
    gt_tool_name = gt_call.tool_name
    gt_idx = next((i for i, t in enumerate(sample.tools) if t.name == gt_tool_name), None)
    if gt_idx is None:
        return None
    gt_tool = sample.tools[gt_idx]
    if not gt_tool.parameters:
        return None

    # Resolve date
    base_date = date(2026, 1, 1)
    if sample.current_time_iso:
        try:
            base_date = date.fromisoformat(str(sample.current_time_iso)[:10])
        except ValueError:
            pass

    recent_rows = _recent_rows(sample)

    try:
        if pack_multi_tools:
            out = build_input_multi(
                [tooldef_to_template_tool(t) for t in sample.tools],
                sample.utterance,
                recent_context=recent_rows,
                now=base_date,
            )
        else:
            out = build_input(
                tooldef_to_template_tool(gt_tool),
                sample.utterance,
                recent_context=recent_rows,
                now=base_date,
            )
    except Exception:
        return None

    enc = tokenizer(out.text, return_offsets_mapping=True, add_special_tokens=False)
    ids: List[int] = enc["input_ids"]
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]
    if len(ids) > max_seq_len:
        return None

    # Utterance extent
    utt_entries = [e for e in out.entries if getattr(e, "segment", None) == "utterance"]
    if not utt_entries:
        return None
    u_cs = min(e.char_start for e in utt_entries)
    u_ce = max(e.char_end for e in utt_entries)
    usnap = _snap_token_range(u_cs, u_ce, offsets, max_seq_len)
    if usnap is None:
        return None
    utt_starts, utt_ends = usnap

    # Tool positions
    tool_positions: List[int] = []
    if pack_multi_tools:
        t_entries = [
            e for e in out.entries
            if getattr(e, "kind", None) == "bridge_text" and getattr(e, "tool", None) is not None
        ]
        for e in t_entries:
            tsnap = _snap_token_range(e.char_start, e.char_end, offsets, max_seq_len)
            tool_positions.append(tsnap[0] if tsnap else 0)

    # Param anchors
    K = len(gt_tool.parameters)
    if pack_multi_tools:
        t_entries = [
            e for e in out.entries
            if getattr(e, "kind", None) == "bridge_text" and getattr(e, "tool", None) is not None
        ]
        s_cs = t_entries[gt_idx].char_start
        s_ce = t_entries[gt_idx + 1].char_start if gt_idx + 1 < len(t_entries) else min(e.char_start for e in utt_entries)
        p_entries = [
            e for e in out.entries
            if getattr(e, "kind", None) == "param_anchor"
            and getattr(e, "value", None) != "<|none|>"
            and s_cs <= e.char_start < s_ce
        ]
        anchors: List[int] = []
        for e in p_entries:
            psnap = _snap_token_range(e.char_start, e.char_end, offsets, max_seq_len)
            anchors.append(psnap[0] if psnap else 0)
        anchors = (anchors + [0] * K)[:K]
    else:
        anchors = []
        for e in out.entries:
            if getattr(e, "kind", None) == "param_anchor" and getattr(e, "value", None) != "<|none|>":
                psnap = _snap_token_range(e.char_start, e.char_end, offsets, max_seq_len)
                anchors.append(psnap[0] if psnap else 0)
        anchors = (anchors + [0] * K)[:K]

    cand_start_rows: List[List[int]] = []
    cand_end_rows: List[List[int]] = []
    pos_rows: List[List[int]] = []

    gt_args = dict(gt_call.arguments or {})
    is_stratum_f = (sample.stratum == Stratum.F) or (str( getattr(sample.stratum, "value", sample.stratum)) == "F")

    for idx, param in enumerate(gt_tool.parameters):
        masks = mask_builder.build_param_masks(param, out.entries, offsets, out.text)
        cands = list(masks["candidates"])

        matched_cis: List[int] = []
        targets = (sample.target_spans or {}).get(param.name, [])

        if is_stratum_f:
            for t in targets:
                if t.target_type == "unresolved":
                    continue
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

            # Direct argument fallback if target_spans missed
            if not matched_cis:
                gt_val = gt_args.get(param.name)
                if gt_val is not None:
                    for ci, c in enumerate(cands):
                        if c.canon_value is not None:
                            c_val = str(c.canon_value).strip().lower()
                            g_val = str(gt_val).strip().lower()
                            if c.canon_value == gt_val or c_val == g_val:
                                if ci not in matched_cis:
                                    matched_cis.append(ci)
                elif not param.required or "null" in (param.param_type or "").lower():
                    # Optional/nullable parameter omitted or None -> point to <|null|>
                    for ci, c in enumerate(cands):
                        if c.region == "null" and ci not in matched_cis:
                            matched_cis.append(ci)
                            break

        # Cap candidates if necessary
        if len(cands) > max_cands:
            keep_pos = matched_cis[:max_pos]
            keep_set = set(keep_pos)
            pool = []
            for pi in keep_pos:
                for delta in range(-15, 16):
                    ni = pi + delta
                    if 0 <= ni < len(cands) and ni not in keep_set:
                        pool.append(ni)
                        keep_set.add(ni)
            rem = max_cands - len(keep_set)
            if rem > 0:
                stride = max(1, len(cands) // rem)
                for ci in range(0, len(cands), stride):
                    if ci not in keep_set:
                        pool.append(ci)
                        keep_set.add(ci)
                    if len(pool) + len(keep_pos) >= max_cands:
                        break
            kept_indices = sorted(keep_pos + pool[: max_cands - len(keep_pos)])
            idx_map = {old_ci: new_ci for new_ci, old_ci in enumerate(kept_indices)}
            cands = [cands[ci] for ci in kept_indices]
            matched_cis = [idx_map[ci] for ci in keep_pos if ci in idx_map]

        cand_start_rows.append([c.tok_start for c in cands])
        cand_end_rows.append([c.tok_end for c in cands])

        # 0-indexed candidate positives!
        pos_cols = [ci for ci in matched_cis[:max_pos]]
        pos_rows.append((pos_cols + [-1] * max_pos)[:max_pos])

    C = max((len(c) for c in cand_start_rows), default=0)
    C = max(min(C, max_cands), 1)
    for i in range(len(cand_start_rows)):
        if len(cand_start_rows[i]) < C:
            cand_start_rows[i] += [-1] * (C - len(cand_start_rows[i]))
            cand_end_rows[i] += [-1] * (C - len(cand_end_rows[i]))
        else:
            cand_start_rows[i] = cand_start_rows[i][:C]
            cand_end_rows[i] = cand_end_rows[i][:C]

    row: Dict[str, Any] = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.ones(len(ids), dtype=torch.long),
        "param_positions": torch.tensor(anchors, dtype=torch.long),
        "utt_starts": torch.tensor([utt_starts], dtype=torch.long),
        "utt_ends": torch.tensor([utt_ends], dtype=torch.long),
        "cand_starts": torch.tensor(cand_start_rows, dtype=torch.long),
        "cand_ends": torch.tensor(cand_end_rows, dtype=torch.long),
        "pos_indices": torch.tensor(pos_rows, dtype=torch.long),
        "_lean_converted": True,
    }

    if pack_multi_tools and tool_positions:
        row["tool_positions"] = torch.tensor(tool_positions, dtype=torch.long)
        row["tool_targets"] = torch.tensor(gt_idx, dtype=torch.long)

    return row


class LeanBatchCollate:
    """Collates and pads Lean training sample dicts into batch tensors."""

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}
        keys = [k for k in batch[0].keys() if not k.startswith("_")]
        out: Dict[str, torch.Tensor] = {}

        for k in keys:
            vals = [b[k] for b in batch]
            if not isinstance(vals[0], torch.Tensor):
                continue
            pad_val = -1 if any(sub in k for sub in ("cand_starts", "cand_ends", "pos_indices", "tool_positions")) else 0
            if vals[0].dim() == 1:
                out[k] = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=pad_val)
            elif vals[0].dim() == 2:
                B = len(vals)
                max_k = max(v.size(0) for v in vals)
                max_c = max(v.size(1) for v in vals)
                padded = torch.full((B, max_k, max_c), pad_val, dtype=vals[0].dtype)
                for i, v in enumerate(vals):
                    padded[i, : v.size(0), : v.size(1)] = v
                out[k] = padded
            else:
                out[k] = torch.stack(vals, dim=0)

        if "pos_indices" in out:
            out["pointer_positives"] = out["pos_indices"]
        if any(b.get("_lean_converted") for b in batch):
            out["_lean_converted"] = True  # type: ignore[assignment]

        return out
