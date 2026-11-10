"""Lean Tensorization and Batch Adapters.

Provides:
- convert_legacy_batch_to_lean: translates legacy batch tensors (where 0=abstain,
  v>0 = candidate v-1) into 0-indexed candidate positives in [0, C) with -1 pad.
- convert_legacy_row_to_lean: translates a single serialized dict row.
- tensorize_sample_lean: direct tensorization for UnifiedSample without abstain targets.
- LeanBatchCollate: collates a list of sample tensor dicts into a padded batch.
"""

from __future__ import annotations

import random
import zlib
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from spancall.data.align import tooldef_to_template_tool
from spancall.data.schema import ParameterSchema, Stratum, ToolDefinition, UnifiedSample
from span_template.template import RecentRow, build_input, build_input_multi

# The one and only "absent / padded" sentinel for integer index tensors.
# ``0`` is a *real* index (the first candidate span, the ``<|cls|>`` token,
# candidate 0 of the tool packet), so it must never double as padding —
# otherwise a pad silently becomes a supervision target. Exported so
# ``spancall/lean/loss.py`` can use the same value as ``ignore_index``.
PAD_INDEX: int = -1

# Keys whose padding value is ``PAD_INDEX``. This is an EXPLICIT set, not a
# substring test on the key name: a substring test (``"cand_" in k``) silently
# mis-pads any future key that merely contains one of those words, which is
# exactly how a padding bug becomes invisible instead of loud.
PAD_INDEX_KEYS = frozenset(
    {
        "cand_starts",
        "cand_ends",
        "pos_indices",
        "pointer_positives",
        "param_positions",
        "tool_positions",
        "tool_targets",
    }
)


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


class ToolPool:
    """Deterministic corpus-level pool of ``ToolDefinition`` distractors for H1.

    Why this exists: Hermes rows carry exactly ONE tool in their ``tools`` column
    (json-mode rows carry a single synthetic ``json_output``). Packing just
    ``sample.tools`` therefore renders **one** ``<|tool|>`` anchor, so
    ``tool_logits`` has shape ``(B, 1)`` and ``argmax(-1)`` is identically 0 —
    while ``tool_targets`` is also identically 0. H1 accuracy is then 1.0 for
    *any* weights and ``L_tool = CE over one class = 0``, i.e. H1 receives no
    gradient at all. ``docs/param_recall_diagnosis.md`` calls this the
    "vacuous 1.0, single-tool" baseline.

    This pool supplies the missing distractors: for each row we keep the GT tool
    and append ``pack_tools - 1`` other real tool schemas drawn from the corpus,
    shuffle the candidate order, and recompute ``tool_targets``. That makes H1
    a genuine discrimination task over ``T`` candidates.

    Selection is deterministic per ``(sample_id, seed)`` so tensorization stays
    reproducible under the fork pool (no shared RNG state).

    Args:
        tools: every tool schema observed in the corpus (deduplicated by name).
        seed: deterministic salt; same seed => same distractors per row.
    """

    def __init__(self, tools: Sequence[ToolDefinition], seed: int = 42) -> None:
        by_name: Dict[str, ToolDefinition] = {}
        # Two different schemas can share a name across datasets. The pool keys
        # on the name, so only the FIRST definition survives and the candidate
        # packet renders that one. Counted here (rather than silently dropped)
        # so callers can print it: a big collision rate means the distractor
        # packet shows a *different* schema than some rows were trained on.
        self.n_duplicates: int = 0
        self.name_collisions: List[str] = []
        _collided: set = set()
        for t in tools:
            if not t.name:
                continue
            prev = by_name.get(t.name)
            if prev is not None:
                self.n_duplicates += 1
                if prev != t and t.name not in _collided:
                    _collided.add(t.name)
                    self.name_collisions.append(t.name)
                continue
            by_name[t.name] = t
        self.by_name: Dict[str, ToolDefinition] = by_name
        # Sorted for a stable, process-independent draw order.
        self.names: List[str] = sorted(by_name)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.names)

    @classmethod
    def from_row_dicts(cls, rows: Sequence[Dict[str, Any]], seed: int = 42) -> "ToolPool":
        """Build a pool from raw canonical split records (``tools`` JSON lists)."""
        tools: List[ToolDefinition] = []
        for row in rows:
            for t in row.get("tools") or []:
                # ``tools`` may be dicts (raw JSON) or already-parsed definitions.
                tools.append(t if isinstance(t, ToolDefinition) else ToolDefinition.from_dict(t))
        return cls(tools, seed=seed)

    def _rng(self, sample_id: str) -> random.Random:
        # zlib.crc32 (not hash()) because per-process string hashing is salted.
        salt = zlib.crc32(f"{self.seed}::{sample_id}".encode("utf-8"))
        return random.Random(salt)

    def candidates_for(
        self,
        sample: UnifiedSample,
        gt_tool: ToolDefinition,
        pack_tools: int,
    ) -> Tuple[List[ToolDefinition], int]:
        """Return ``(candidates, gt_index)`` with the GT tool plus distractors.

        The GT tool is always present (at a shuffled index). ``pack_tools <= 1``
        or a pool with no alternatives returns ``([gt_tool], 0)`` — the honest
        single-candidate case, which callers report rather than score.
        """
        budget = max(int(pack_tools), 1)
        native = [t for t in (sample.tools or []) if t.name != gt_tool.name]
        candidates: List[ToolDefinition] = [gt_tool]
        seen = {gt_tool.name}

        # Keep any tools the row genuinely shipped alongside the GT tool.
        for t in native:
            if len(candidates) >= budget:
                break
            if t.name not in seen:
                candidates.append(t)
                seen.add(t.name)

        # Then fill with corpus distractors (deterministic per row).
        if len(candidates) < budget and self.names:
            rng = self._rng(sample.sample_id or gt_tool.name)
            # ``rng.sample`` with a small ``k`` uses CPython's set-based path,
            # i.e. O(k). Building ``[n for n in self.names if n not in seen]``
            # and shuffling it was O(N) *per row* (~5 ms/row at N=20k, i.e.
            # >4 min extra for a 50k-row split).
            want = budget - len(candidates)
            k = min(len(self.names), want + len(seen))
            for name in rng.sample(self.names, k):
                if len(candidates) >= budget:
                    break
                if name in seen:
                    continue
                candidates.append(self.by_name[name])
                seen.add(name)

        # Shuffle so the GT tool is not always at index 0 (positional cheating).
        rng = self._rng(f"shuffle::{sample.sample_id or gt_tool.name}")
        order = list(range(len(candidates)))
        rng.shuffle(order)
        shuffled = [candidates[i] for i in order]
        gt_index = next(i for i, t in enumerate(shuffled) if t.name == gt_tool.name)
        return shuffled, gt_index


def tensorize_sample_lean(
    sample: UnifiedSample,
    tokenizer: Any,
    mask_builder: Any,
    max_seq_len: int = 512,
    max_cands: int = 512,
    max_pos: int = 8,
    pack_multi_tools: bool = False,
    stats: Optional[Dict[str, Any]] = None,
    tool_pool: Optional[ToolPool] = None,
    pack_tools: int = 1,
) -> Optional[Dict[str, Any]]:
    """Direct tensorization of UnifiedSample into Lean training format.

    Eliminates column-0 abstain: candidate positions are 0-indexed in [0, C).
    Rows with no positive candidate are padded with -1.

    Under the retrieval-first contract (docs/param_extraction_eval_coverage.md),
    unmatched Stratum-F params are classified instead of silently dropped:
    ``unresolved_target`` (genuinely unextractable → in-scope exclusion) vs
    ``match_missed`` (groundable target but matcher failed → bug indicator).
    Counts are accumulated into ``stats["param_drop_reasons"]`` when supplied,
    and the per-row classification is attached as ``_param_drop_reasons`` for
    the tensorize/sanity cells to report.

    H1 candidate sets: when ``tool_pool`` is supplied and ``pack_tools > 1``,
    the GT tool is packed together with ``pack_tools - 1`` real distractor
    ``ToolDefinition``s drawn deterministically from the corpus pool, and the
    candidate list is shuffled (``tool_targets`` recomputed accordingly). This
    is what makes ``tool_acc`` a real metric instead of the structurally forced
    1.0 produced by a single candidate (see :class:`ToolPool`).
    """
    if not sample.tools or not sample.ground_truth_calls:
        return None

    gt_call = sample.ground_truth_calls[0]
    gt_tool_name = gt_call.tool_name
    src_gt_idx = next((i for i, t in enumerate(sample.tools) if t.name == gt_tool_name), None)
    if src_gt_idx is None:
        return None
    gt_tool = sample.tools[src_gt_idx]
    if not gt_tool.parameters:
        return None

    # Build the H1 candidate set (GT tool + distractors), shuffled. Without a
    # pool this degrades to sample.tools (honest single-candidate case).
    if tool_pool is not None and int(pack_tools) > 1:
        tool_candidates, gt_idx = tool_pool.candidates_for(sample, gt_tool, int(pack_tools))
    else:
        tool_candidates = list(sample.tools)
        gt_idx = src_gt_idx

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
                [tooldef_to_template_tool(t) for t in tool_candidates],
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
        tool_snap_misses = 0
        for e in t_entries:
            tsnap = _snap_token_range(e.char_start, e.char_end, offsets, max_seq_len)
            if tsnap is None:
                # ``PAD_INDEX`` (-1), NEVER 0: the trainer masks dead anchors with
                # ``tool_positions < 0`` and 0 is a real token (``<|cls|>``), so a
                # 0 here would resurrect the slot as a live candidate pointing at
                # the CLS embedding.
                tool_positions.append(PAD_INDEX)
                tool_snap_misses += 1
            else:
                tool_positions.append(tsnap[0])
        # Record the misses BEFORE the drop guard below: a row dropped because
        # its GT anchor failed to snap is precisely the case an operator must
        # see counted, not the case that silently reports zero.
        if tool_snap_misses and stats is not None:
            stats["tool_snap_misses"] = (
                int(stats.get("tool_snap_misses", 0)) + tool_snap_misses
            )
        # The GT anchor must itself be live. If it is not, H1 has nothing to
        # teach on this row and cross-entropy against a -1e4-masked logit would
        # explode rather than train: drop the row (same policy as the
        # ``gt_idx >= len(t_entries)`` guard below).
        if gt_idx >= len(tool_positions) or tool_positions[gt_idx] < 0:
            return None

    # Param anchors
    K = len(gt_tool.parameters)
    if pack_multi_tools:
        t_entries = [
            e for e in out.entries
            if getattr(e, "kind", None) == "bridge_text" and getattr(e, "tool", None) is not None
        ]
        # The renderer can drop a tool whose name/desc sanitizes badly; fall back
        # to the single-tool slice rather than raising an IndexError here.
        if gt_idx >= len(t_entries):
            return None
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
    # L1: per-param drop-reason audit (only populated for Stratum-F rows when
    # a param ends up with NO matched positive). ``stats`` is an optional
    # mutable dict the caller (tensorize stats cell / sanity cell) can read.
    drop_reasons: List[str] = []

    gt_args = dict(gt_call.arguments or {})
    is_stratum_f = (sample.stratum == Stratum.F) or (str( getattr(sample.stratum, "value", sample.stratum)) == "F")

    # L2: deterministic <|null|> supervision — the <|null|> slot is resolved
    # per param (candidate sets differ per param), always at the SAME position
    # the mask builder emitted it, with no best-effort region search that can
    # silently vanish when the mask builder changes.
    def _find_null_idx(cands_list: List[Any]) -> Optional[int]:
        for ci, c in enumerate(cands_list):
            if getattr(c, "region", "") == "null":
                return ci
        return None

    for idx, param in enumerate(gt_tool.parameters):
        masks = mask_builder.build_param_masks(param, out.entries, offsets, out.text)
        cands = list(masks["candidates"])

        # Per-param deterministic <|null|> slot (appended last by
        # _null_candidates; absent for required non-nullable params).
        null_ci: Optional[int] = _find_null_idx(cands)
        matched_cis: List[int] = []
        targets = (sample.target_spans or {}).get(param.name, [])

        if is_stratum_f:
            for t in targets:
                if t.target_type == "unresolved":
                    continue
                if t.target_type == "null_ctx":
                    # L2: deterministic null slot instead of worst-effort search
                    if null_ci is not None and null_ci not in matched_cis:
                        matched_cis.append(null_ci)
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
                # C1: distinguish "GT arg is explicitly null" from "GT arg absent".
                # ``.get()`` returns None for both; only the former is a real
                # supervision signal that must map to the <|null|> slot.
                gt_present = param.name in gt_args
                gt_val = gt_args.get(param.name)
                gt_is_null = gt_present and gt_val is None
                if gt_val is not None:
                    for ci, c in enumerate(cands):
                        if c.canon_value is not None:
                            c_val = str(c.canon_value).strip().lower()
                            g_val = str(gt_val).strip().lower()
                            if c.canon_value == gt_val or c_val == g_val:
                                if ci not in matched_cis:
                                    matched_cis.append(ci)
                elif gt_is_null or not param.required or "null" in (param.param_type or "").lower():
                    # Explicit null GT, or an optional/nullable param omitted by
                    # the utterance -> deterministic <|null|> slot.
                    if null_ci is not None and null_ci not in matched_cis:
                        matched_cis.append(null_ci)

            # L1: classify WHY a Stratum-F param ended up with no positive.
            # Only ONE of these is expected under the retrieval-first contract
            # ('unresolved_target'); everything else is a data-pipeline or
            # alignment defect that must not hide behind the legitimate
            # exclusion. See docs/param_extraction_eval_coverage.md §4.
            if not matched_cis:
                has_unresolved = any(t.target_type == "unresolved" for t in targets)
                has_null_target = any(t.target_type == "null_ctx" for t in targets)
                all_unaligned = bool(targets) and all(
                    getattr(t, "canonical_value", None) is None
                    and getattr(t, "char_start", None) is None
                    for t in targets
                )
                gt_present = param.name in gt_args
                gt_is_null = gt_present and gt_args.get(param.name) is None
                if has_unresolved:
                    # Expected: needs generation -> fast path must refuse.
                    why = "unresolved_target"
                elif all_unaligned and not has_null_target:
                    # Target row exists but carries neither a canonical value
                    # nor a span -> the aligner produced an unusable target.
                    why = "unaligned_target"
                elif (has_null_target or gt_is_null) and null_ci is None:
                    # We know the answer is "null" but the mask builder emitted
                    # no <|null|> candidate for this param -> cannot supervise.
                    why = "null_slot_missing"
                elif not targets and not gt_present:
                    # No target and no GT arg: the param simply is not part of
                    # this call's supervision set (e.g. optional, omitted).
                    why = "absent_from_gt"
                else:
                    # Extractable target existed but the matcher failed: BUG.
                    why = "match_missed"
                drop_reasons.append(why)
                if stats is not None:
                    stats.setdefault("param_drop_reasons", {})
                    stats["param_drop_reasons"][why] = stats["param_drop_reasons"].get(why, 0) + 1

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

    # L1: expose the per-row drop-reason classification (names, not just counts)
    # so callers can surface ``match_missed`` rates without re-deriving them.
    if drop_reasons:
        row["_param_drop_reasons"] = drop_reasons

    if pack_multi_tools and tool_positions:
        row["tool_positions"] = torch.tensor(tool_positions, dtype=torch.long)
        row["tool_targets"] = torch.tensor(gt_idx, dtype=torch.long)
        # Diagnostic: H1 is only a real task when this is > 1. Reported by the
        # trainer as ``mean_n_candidates`` so a single-candidate (vacuous
        # tool_acc=1.0) run is visible on its face instead of looking perfect.
        # Counts LIVE anchors: dead (unsnapped) slots are -1 and must not inflate
        # the number of choices H1 actually has.
        row["_n_tool_candidates"] = int(sum(1 for p in tool_positions if p >= 0))

    return row


class LeanBatchCollate:
    """Collates and pads Lean training sample dicts into batch tensors."""

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}
        # Union of keys (not batch[0]'s keys): tool_positions/tool_targets can be
        # present on only some rows (e.g. rows whose GT tool failed to render),
        # and taking batch[0] would silently drop H1 supervision for the batch.
        keys: List[str] = []
        for b in batch:
            for k in b:
                if not k.startswith("_") and k not in keys:
                    keys.append(k)
        out: Dict[str, torch.Tensor] = {}
        # PAD_INDEX (-1) is the "absent/padded" sentinel for every index tensor,
        # INCLUDING tool_targets: padding it with 0 would inject a fake "tool 0
        # is correct" supervision signal, and the trainer's validity mask is
        # ``tt >= 0``. Membership is tested against an explicit key set, not a
        # substring of the key name.
        for k in keys:
            vals = [b.get(k) for b in batch]
            pad_val = PAD_INDEX if k in PAD_INDEX_KEYS else 0
            # A row missing this tensor is pad-filled, not dropped.
            if any(v is None for v in vals):
                ref = next(v for v in vals if isinstance(v, torch.Tensor))
                filled: List[torch.Tensor] = []
                for v in vals:
                    if isinstance(v, torch.Tensor):
                        filled.append(v)
                    else:
                        filled.append(
                            torch.full(
                                (ref.size(0),) if ref.dim() >= 1 else (),
                                pad_val,
                                dtype=ref.dtype,
                            )
                        )
                vals = filled
            if not isinstance(vals[0], torch.Tensor):
                continue
            if vals[0].dim() == 0:
                out[k] = torch.stack([v.long() for v in vals], dim=0)
            elif vals[0].dim() == 1:
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
