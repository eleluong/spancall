"""Lean Tensorization and Batch Collator.

Provides:
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
from spancall.data.schema import (
    GroundTruthCall,
    SpanTarget,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)
from span_template.template import RecentRow, build_input_multi, resolve_utterance_span

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
        # J2: one tool index per supervised call, and one owning-call index per
        # param anchor. Both are index tensors, so a pad must be PAD_INDEX (-1)
        # -- padding with 0 would claim "call 0 / tool 0 owns the anchor" on a
        # row that has neither.
        "call_param_index",
        "call_tool_targets",
        "pos_indices",
        "pointer_positives",
        "param_positions",
        "tool_positions",
        "tool_targets",
    }
)


def has_positive_targets(row: Dict[str, Any]) -> bool:
    """Return True if a row has at least one valid positive candidate index."""
    pos = row.get("pos_indices")
    if pos is None:
        return False
    if isinstance(pos, torch.Tensor):
        return bool((pos > -1).any().item())
    if isinstance(pos, list):
        return any(
            (val > -1)
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


def _recent_rows(
    sample: UnifiedSample,
    tools: Optional[Sequence[Any]] = None,
    k_recent: int = 32,
    context_budget_tokens: int = 4096,
    tokenizer: Any = None,
    stats: Optional[Dict[str, Any]] = None,
) -> List[RecentRow]:
    """Convert ``sample.recent_context`` to template rows BEFORE rendering.

    R8: training used to stringify every row verbatim while inference ran it
    through :func:`select_recent_context` (marker rejection, key sanitization,
    value normalization, dedupe, relevance weighting, and token budgeting). The
    model then saw rows at train time that the serving path would have deleted,
    and — worse — saw an *unbounded* recent block that could silently blow the
    sequence budget. Training and inference must select identically, so this now
    calls the same function, with the same arguments, that the resolver calls.
    """
    raw = [(r if isinstance(r, dict) else {"key": getattr(r, "key", ""),
                                          "value": getattr(r, "value", ""),
                                          "source": getattr(r, "source", "recent")})
           for r in (sample.recent_context or [])]
    if not raw:
        return []

    from spancall.agent.context import select_recent_context

    # ``tools`` must be the RENDERED schema set (what the keywords are drawn
    # from) so relevance weighting matches the prompt being built.
    selected, feats = select_recent_context(
        raw,
        list(tools or sample.tools or []),
        k_recent=k_recent,
        context_budget_tokens=context_budget_tokens,
        tokenizer=tokenizer,
    )
    if stats is not None:
        stats["n_recent_rows"] = int(stats.get("n_recent_rows", 0)) + len(selected)
        if feats.get("truncated"):
            stats["recent_truncated_rows"] = int(stats.get("recent_truncated_rows", 0)) + 1

    rows_out: List[RecentRow] = []
    for r in selected:
        try:
            rows_out.append(
                RecentRow(
                    key=str(r.get("key", "")),
                    value=str(r.get("value", "")),
                    source=str(r.get("source", "recent")),
                )
            )
        except ValueError:
            # Still fine to skip: select_recent_context already normalized, so a
            # rejection here means the row cannot be rendered under the grammar
            # at all (e.g. an empty key), not that a value was silently cut.
            continue
    return rows_out


def jitter_pack_tools(
    pack_tools: int,
    sample_id: str,
    seed: int = 42,
    min_tools: int = 1,
) -> int:
    """Draw this row's ``pack_tools`` floor from ``[min_tools, pack_tools]``.

    R1c: with a fixed floor every training row renders exactly ``pack_tools``
    anchors, so the model never sees ``T=1`` — yet the oracle-tool evaluation
    path renders exactly one. ``tool_logits`` then has an untrained shape and H1
    collapses. Randomizing the floor per row puts ``T=1`` (and everything up to
    the ceiling) inside the training distribution without ever exceeding it.

    Deterministic in ``(seed, sample_id)`` so the fork pool stays reproducible
    and a rerun of the same split produces the same anchor counts.
    """
    hi = max(int(pack_tools), 1)
    lo = max(int(min_tools), 1)
    if hi <= lo:
        return hi
    salt = zlib.crc32(f"jitter::{int(seed)}::{sample_id}".encode("utf-8"))
    return lo + (salt % (hi - lo + 1))


class ToolPool:
    """Deterministic corpus-level pool of ``ToolDefinition`` distractors for H1.

    Why this exists: Hermes rows carry exactly ONE tool in their ``tools`` column
    (json-mode rows carry a single synthetic ``json_output``). Packing just
    ``sample.tools`` therefore renders **one** ``<|tool|>`` anchor, so
    ``tool_logits`` has shape ``(B, 1)`` and ``argmax(-1)`` is identically 0 —
    while ``tool_targets`` is also identically 0. H1 accuracy is then 1.0 for
    *any* weights and ``L_tool = CE over one class = 0``, i.e. H1 receives no
    gradient at all. This is the
    "vacuous 1.0, single-tool" baseline.

    This pool supplies the missing distractors: for each row we keep the GT tool
    and every native tool the row shipped, then top the candidate list **up to**
    ``pack_tools`` with real tool schemas drawn from the corpus, shuffle the
    candidate order, and recompute ``tool_targets``. That makes H1 a genuine
    discrimination task over ``T`` candidates.

    ``pack_tools`` is a FLOOR, never a cap. A row that genuinely offers five
    tools must render all five — dropping them would hide from H1 exactly the
    tools the user could have picked, and would make the eval-time prompt
    (which always renders ``sample.tools``) structurally different from the
    training prompt.

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
        max_tools: Optional[int] = None,
    ) -> Tuple[List[ToolDefinition], int]:
        """Return ``(candidates, gt_index)`` with the GT tool plus distractors.

        ``T`` is ``max(pack_tools, len(native_tools) + 1)``: every tool the row
        actually shipped is kept, and ``pack_tools`` only ratchets the count
        *up*. The GT tool is always present (at a shuffled index).

        ``max_tools`` is an optional hard ceiling for pathological rows (a
        BFCL-style schema dump with dozens of tools) so a single row cannot
        consume the whole sequence budget. It is applied before the shuffle
        while the GT tool still sits at index 0, so the GT tool always survives
        truncation. Callers that set it must report the truncation — this
        method never silently drops a native tool on its own.
        """
        budget = max(int(pack_tools), 1)
        if max_tools is not None:
            # The ceiling also caps the distractor fill, otherwise truncation
            # would be immediately undone by the top-up below.
            budget = min(budget, max(1, int(max_tools)))
        candidates: List[ToolDefinition] = [gt_tool]
        seen = {gt_tool.name}

        # Keep EVERY tool the row genuinely shipped alongside the GT tool. This
        # is the R0.1 fix: the old code stopped at ``budget`` and therefore
        # trained H1 on a strict subset of the tools eval renders.
        for t in (sample.tools or []):
            if t.name and t.name not in seen:
                candidates.append(t)
                seen.add(t.name)

        # Optional hard ceiling: GT tool is at index 0 here, so truncation can
        # never remove the answer. The caller counts what was cut.
        if max_tools is not None and len(candidates) > max(1, int(max_tools)):
            candidates = candidates[: max(1, int(max_tools))]

        # Then fill up to the floor with corpus distractors (deterministic/row).
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
    pack_multi_tools: bool = True,
    stats: Optional[Dict[str, Any]] = None,
    tool_pool: Optional[ToolPool] = None,
    pack_tools: int = 1,
    max_tools: Optional[int] = None,
    k_recent: int = 32,
    context_budget_tokens: int = 4096,
    jitter: bool = True,
    jitter_seed: int = 42,
    jitter_min_tools: int = 1,
    force_single_tool: bool = False,
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
    the GT tool is packed together with real distractor ``ToolDefinition``s
    drawn deterministically from the corpus pool, and the candidate list is
    shuffled (``tool_targets`` recomputed accordingly). This is what makes
    ``tool_acc`` a real metric instead of the structurally forced 1.0 produced
    by a single candidate (see :class:`ToolPool`).

    Grammar: every row is rendered with ``template_multi.jinja`` via
    :func:`build_input_multi`, at *every* ``T >= 1``. ``pack_multi_tools`` is
    accepted-but-ignored so pre-existing call sites keep working; training and
    inference must never disagree about which template produced the anchors.
    """
    if not sample.tools or not sample.ground_truth_calls:
        return None

    # R0.4/J2: the FIRST call still drives the row's grammar -- it supplies the
    # tool packet the renderer packs, the recent-context row selection and the
    # anchor layout -- while every other call adds SUPERVISION on that same
    # render. Nothing about the single-call path changes: ``multi_call`` stays
    # False and every structure added below reduces to its previous shape.
    gt_calls: List[GroundTruthCall] = list(sample.ground_truth_calls)
    primary_call_index = 0
    gt_call = gt_calls[primary_call_index]
    gt_tool_name = gt_call.tool_name
    src_gt_idx = next((i for i, t in enumerate(sample.tools) if t.name == gt_tool_name), None)
    if src_gt_idx is None:
        return None
    gt_tool = sample.tools[src_gt_idx]
    if not gt_tool.parameters:
        return None

    # Build the H1 candidate set (GT tool + EVERY native tool + distractors up
    # to the ``pack_tools`` floor), shuffled. ``pack_tools`` is a floor, not a
    # cap: dropping native tools here would train H1 on a subset of what eval
    # renders. Without a pool this degrades to sample.tools (honest case).
    row_pack_tools = int(pack_tools)
    if jitter and tool_pool is not None:
        # Per-row floor so T=1 is in-distribution (R1c). The ceiling is unchanged.
        row_pack_tools = jitter_pack_tools(
            int(pack_tools),
            sample.sample_id or gt_tool_name,
            seed=jitter_seed,
            min_tools=jitter_min_tools,
        )
        if stats is not None:
            # Counted at the point of the DRAW, so this includes rows that are
            # later dropped (over-length, dead anchor). ``jitter_draw_hist`` is
            # recorded at the kept-row return instead — see the T12a note there.
            stats["jittered_rows"] = int(stats.get("jittered_rows", 0)) + 1
    if force_single_tool:
        tool_candidates = [gt_tool]
        gt_idx = 0
    elif tool_pool is not None:
        tool_candidates, gt_idx = tool_pool.candidates_for(
            sample, gt_tool, row_pack_tools, max_tools=max_tools,
        )
    else:
        tool_candidates = list(sample.tools)
        gt_idx = src_gt_idx

    # Map every GT call onto the H1 candidate index of ITS tool. A call whose
    # tool was truncated away (only possible under an explicit ``max_tools``)
    # cannot be supervised: it is counted, never silently dropped, and never
    # pointed at a foreign tool's slot.
    cand_tools: Dict[str, ToolDefinition] = {t.name: t for t in tool_candidates}
    supervised_calls: List[Tuple[int, GroundTruthCall, int]] = []
    for ci, c in enumerate(gt_calls):
        if ci == primary_call_index:
            ti: Optional[int] = int(gt_idx)
        else:
            ti = next(
                (i for i, t in enumerate(tool_candidates) if t.name == c.tool_name),
                None,
            )
        if ti is None or ti < 0:
            continue
        supervised_calls.append((ci, c, int(ti)))
    if not supervised_calls:
        return None
    if len(supervised_calls) < len(gt_calls) and stats is not None:
        stats["multi_call_unrendered_calls"] = int(
            stats.get("multi_call_unrendered_calls", 0)
        ) + (len(gt_calls) - len(supervised_calls))
    multi_call = len(supervised_calls) > 1

    if max_tools is not None and len(sample.tools or []) + 1 > max(1, int(max_tools)):
        # Loud, counted: a truncated row renders fewer tools than the user saw.
        if stats is not None:
            stats["tool_truncated_rows"] = int(stats.get("tool_truncated_rows", 0)) + 1

    # Resolve date
    base_date = date(2026, 1, 1)
    if sample.current_time_iso:
        try:
            base_date = date.fromisoformat(str(sample.current_time_iso)[:10])
        except ValueError:
            pass

    recent_rows = _recent_rows(
        sample,
        tools=tool_candidates,
        k_recent=k_recent,
        context_budget_tokens=context_budget_tokens,
        tokenizer=tokenizer,
        stats=stats,
    )

    # ONE grammar at every T >= 1: ``template_multi.jinja`` via build_input_multi.
    # The single-tool template (``template.jinja``/``build_input``) emits no
    # ``<|none|>`` anchor, so training on it while the resolver always rendered
    # the multi template produced a different anchor layout for the same row.
    # ``pack_multi_tools`` is kept as an accepted-but-ignored keyword so existing
    # call sites (the notebook's per-split groups) keep working unchanged.
    try:
        out = build_input_multi(
            [tooldef_to_template_tool(t) for t in tool_candidates],
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
        # Counted, never silent: once prompts carry every native tool this is the
        # only way to see how much of the split is being discarded for length.
        if stats is not None:
            stats["dropped_over_length"] = int(stats.get("dropped_over_length", 0)) + 1
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

    # Tool positions — the ``<|tool|>`` bridge anchors emitted by the single
    # multi-tool grammar, one per rendered candidate tool.
    t_entries = [
        e for e in out.entries
        if getattr(e, "kind", None) == "bridge_text" and getattr(e, "tool", None) is not None
    ]
    tool_positions: List[int] = []
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
    # R9: record T on EVERY row, including T=1. Counting only T>1 would hide the
    # exact distribution mismatch this metric exists to expose (training packs
    # ``pack_tools`` distractors while oracle-tool eval renders the GT alone).
    if stats is not None:
        T = len(tool_positions)
        stats["render_T_sum"] = int(stats.get("render_T_sum", 0)) + T
        stats["render_T_rows"] = int(stats.get("render_T_rows", 0)) + 1
        hist = stats.get("render_T_hist")
        if not isinstance(hist, dict):
            hist = {}
            stats["render_T_hist"] = hist
        hist[T] = int(hist.get(T, 0)) + 1
    # The GT anchor must itself be live. If it is not, H1 has nothing to
    # teach on this row and cross-entropy against a -1e4-masked logit would
    # explode rather than train: drop the row (same policy as the
    # ``gt_idx >= len(t_entries)`` guard below).
    if gt_idx >= len(tool_positions) or tool_positions[gt_idx] < 0:
        return None

    # The renderer can drop a tool whose name/desc sanitizes badly; bail rather
    # than raising an IndexError here.
    if gt_idx >= len(t_entries):
        return None

    # ---- Param anchors: the UNION across every supervised call (R0.4/J2) ----
    #
    # Each anchor is scoped to ITS OWN tool's render block, so a distractor that
    # happens to share a parameter name can never contribute an anchor. Anchors
    # are deduplicated by ``(tool, param)``: two calls to the same tool share
    # one anchor (and therefore one pointer distribution), while two calls to
    # different tools get one anchor each.
    #
    # Single-call rows are unaffected: one block, params in declaration order,
    # ``K == len(gt_tool.parameters)`` exactly as before.
    anchors: List[int] = []
    anchor_specs: List[Dict[str, Any]] = []
    anchor_index_by_key: Dict[Tuple[str, str], int] = {}
    param_snap_misses = 0
    primary_anchors_found = False
    for call_index, call, ti in supervised_calls:
        call_tool = cand_tools.get(call.tool_name)
        if call_tool is None or ti >= len(t_entries):
            continue
        b_cs = t_entries[ti].char_start
        b_ce = (
            t_entries[ti + 1].char_start
            if ti + 1 < len(t_entries)
            else min(e.char_start for e in utt_entries)
        )
        p_entries = [
            e for e in out.entries
            if getattr(e, "kind", None) == "param_anchor"
            and getattr(e, "value", None) != "<|none|>"
            and b_cs <= e.char_start < b_ce
        ]
        if len(p_entries) != len(call_tool.parameters):
            # A renderer that dropped an anchor for this tool: the primary call
            # cannot be supervised (drop the row, as before); a secondary call
            # is dropped alone and counted, so it cannot destroy the primary
            # row's supervision.
            if call_index == primary_call_index:
                return None
            if stats is not None:
                stats["multi_call_anchor_missing_calls"] = int(
                    stats.get("multi_call_anchor_missing_calls", 0)
                ) + 1
            continue
        for param, e in zip(call_tool.parameters, p_entries):
            key = (call_tool.name, param.name)
            ai = anchor_index_by_key.get(key)
            if ai is None:
                psnap = _snap_token_range(e.char_start, e.char_end, offsets, max_seq_len)
                if psnap is None:
                    param_snap_misses += 1
                    anchors.append(PAD_INDEX)
                else:
                    anchors.append(psnap[0])
                ai = len(anchors) - 1
                anchor_index_by_key[key] = ai
                anchor_specs.append({"param": param, "calls": []})
            anchor_specs[ai]["calls"].append(call_index)
            if call_index == primary_call_index:
                primary_anchors_found = True
    if not primary_anchors_found:
        return None
    if param_snap_misses and stats is not None:
        stats["param_snap_misses"] = (
            int(stats.get("param_snap_misses", 0)) + param_snap_misses
        )
    # Drop row if any anchor failed to snap. A dead anchor has no live
    # distribution to supervise, and the trainer would otherwise receive a
    # column that can never be predicted.
    K = len(anchors)
    if K == 0 or any(a < 0 for a in anchors):
        return None

    cand_start_rows: List[List[int]] = []
    cand_end_rows: List[List[int]] = []
    pos_rows: List[List[int]] = []
    # L1: per-param drop-reason audit (only populated for Stratum-F rows when
    # a param ends up with NO matched positive). ``stats`` is an optional
    # mutable dict the caller (tensorize stats cell / sanity cell) can read.
    drop_reasons: List[str] = []

    call_by_index: Dict[int, GroundTruthCall] = {ci: c for ci, c, _ti in supervised_calls}
    is_stratum_f = (sample.stratum == Stratum.F) or (str( getattr(sample.stratum, "value", sample.stratum)) == "F")
    # Counted, never silent: two calls demanding different answers on one shared
    # anchor cannot be separated by the Lean fast path (R0.4/J2 item 4).
    ambiguous_anchors = 0

    # L2: deterministic <|null|> supervision — the <|null|> slot is resolved
    # per param (candidate sets differ per param), always at the SAME position
    # the mask builder emitted it, with no best-effort region search that can
    # silently vanish when the mask builder changes.
    def _find_null_idx(cands_list: List[Any]) -> Optional[int]:
        for ci, c in enumerate(cands_list):
            if getattr(c, "region", "") == "null":
                return ci
        return None

    # One snapshot of the row's indexes, shared by all K parameters. Building it
    # here (rather than letting build_param_masks rebuild it per param) is what
    # turns O(K·E·n) entry snapping into O(E·n) plus K O(1) lookups.
    try:
        from spancall.model.masks import _RowCtx  # local import: avoid import cycle
        row_ctx = _RowCtx.for_row(out.entries, offsets, max_seq_len)
    except Exception:  # pragma: no cover - never let an optimisation break tensorization
        row_ctx = None

    for spec in anchor_specs:
        param = spec["param"]
        owner_calls: List[int] = spec["calls"]
        # Build unconstrained candidate set first so positive matching and null slot resolution
        # see the complete set of candidates (fixing pre-match candidate truncation).
        masks = mask_builder.build_param_masks(
            param, out.entries, offsets, out.text, ctx=row_ctx, max_cands=None)
        cands = list(masks["candidates"])

        # Per-param deterministic <|null|> slot (appended last by
        # _null_candidates; absent for required non-nullable params).
        null_ci: Optional[int] = _find_null_idx(cands)
        matched_cis: List[int] = []
        # Union of the per-call targets that landed on this anchor. For a
        # single-call sample ``targets_for_call`` returns the legacy bare keys,
        # so this is byte-identical to the old ``target_spans[param.name]``.
        targets: List[SpanTarget] = []
        for ci in owner_calls:
            targets.extend(sample.targets_for_call(ci).get(param.name, []))

        # What do the owning calls actually DEMAND for this anchor? Distinct
        # values only: repeating the same value in two calls is not a conflict.
        demanded: List[Any] = []
        any_null = False
        gt_present = False
        for ci in owner_calls:
            args = dict(call_by_index[ci].arguments or {})
            if param.name not in args:
                continue
            gt_present = True
            v = args[param.name]
            if v is None:
                any_null = True
            elif not any(str(v).strip().lower() == str(d).strip().lower() for d in demanded):
                demanded.append(v)

        # Conflict detection also covers the ALIGNED targets, which are the
        # values the pointer loss is actually trained on.
        t_keys: List[str] = []
        for t in targets:
            if t.target_type == "unresolved":
                continue
            k = (
                f"v:{t.canonical_value}"
                if t.canonical_value is not None
                else f"s:{t.char_start}:{t.char_end}"
            )
            if k not in t_keys:
                t_keys.append(k)
        if len(demanded) > 1 or (demanded and any_null) or len(t_keys) > 1:
            ambiguous_anchors += 1

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
                # Span match FIRST: the same value can legitimately appear at
                # several distinct positions ("move Tokyo to Tokyo"), and those
                # are DISTINCT candidates that must each become a positive.
                # Matching by value first used to ``break`` on the first equal
                # candidate, so every later mention of the same value collapsed
                # into that one slot and ``max_pos`` could never exceed 1 --
                # which silently disabled the multi-positive marginal likelihood
                # in compute_lean_pointer_loss (R7).
                span_hits: List[int] = []
                # Resolve the target against THIS render. A pre-v2 target only
                # carries absolute ``char_*`` offsets, which address whichever
                # render the aligner produced (single-tool, or an older
                # template revision). When the render-invariant
                # ``utt_rel_*`` pair is present it is authoritative: the
                # utterance text is sanitize-stable, so it maps onto this
                # render's absolute offsets no matter how many ``<|tool|>``
                # blocks training packed. Falls back to the absolute pair for
                # context targets and for legacy rows that lack the rel pair.
                t_start, t_end = t.char_start, t.char_end
                if t.utt_rel_start is not None and t.utt_rel_end is not None:
                    resolved = resolve_utterance_span(
                        out,
                        int(t.utt_rel_start),
                        int(t.utt_rel_end),
                        expect=str(t.canonical_value) if t.canonical_value is not None else None,
                    )
                    if resolved is not None:
                        t_start, t_end = resolved
                    elif stats is not None:
                        # Counted, never silent: a rel pair that does not map
                        # onto this render means the utterance text drifted
                        # (sanitization change) and the label is unusable.
                        stats["utt_rel_unresolved"] = int(
                            stats.get("utt_rel_unresolved", 0)
                        ) + 1
                if t_start is not None and t_end is not None:
                    for ci, c in enumerate(cands):
                        if c.char_start == t_start and c.char_end == t_end:
                            span_hits.append(ci)
                if span_hits:
                    for ci in span_hits:
                        if ci not in matched_cis:
                            matched_cis.append(ci)
                    found = True
                elif t.canonical_value is not None:
                    # No usable span (context-derived target): every candidate
                    # carrying the same canonical value is a valid surface form,
                    # mirroring the aligner's own "every equal-valued entry is a
                    # positive" rule for dates. Do NOT break at the first hit.
                    t_str = str(t.canonical_value).strip().lower()
                    for ci, c in enumerate(cands):
                        if c.canon_value is None:
                            continue
                        c_str = str(c.canon_value).strip().lower()
                        if c.canon_value == t.canonical_value or c_str == t_str:
                            if ci not in matched_cis:
                                matched_cis.append(ci)
                            found = True

            # L1/T5: classify WHY a Stratum-F param ended up with no positive,
            # BEFORE any fallback runs.
            #
            # Ordering matters and used to be wrong: the ``<|null|>`` fallback
            # below fills ``matched_cis`` whenever the param is optional or
            # nullable, which made the classifier's ``if not matched_cis`` guard
            # false, so a genuine matcher failure on an extractable target was
            # silently converted into "the answer is null" and never counted. The
            # model was then trained to predict null for a param whose value was
            # sitting in the utterance, and the ``match_missed`` tripwire could
            # never fire for exactly the params most likely to hit it.
            #
            # Only ONE of these reasons is expected under the retrieval-first
            # contract ('unresolved_target'); everything else is a data-pipeline
            # or alignment defect that must not hide behind the legitimate
            # exclusion. See docs/param_extraction_eval_coverage.md §4.
            drop_why: Optional[str] = None
            if not matched_cis:
                has_unresolved = any(t.target_type == "unresolved" for t in targets)
                has_null_target = any(t.target_type == "null_ctx" for t in targets)
                all_unaligned = bool(targets) and all(
                    getattr(t, "canonical_value", None) is None
                    and getattr(t, "char_start", None) is None
                    for t in targets
                )
                gt_is_null = any_null
                if has_unresolved:
                    # Expected: needs generation -> fast path must refuse.
                    drop_why = "unresolved_target"
                elif all_unaligned and not has_null_target:
                    # Target row exists but carries neither a canonical value
                    # nor a span -> the aligner produced an unusable target.
                    drop_why = "unaligned_target"
                elif (has_null_target or gt_is_null) and null_ci is None:
                    # We know the answer is "null" but the mask builder emitted
                    # no <|null|> candidate for this param -> cannot supervise.
                    drop_why = "null_slot_missing"
                elif not targets and not gt_present:
                    # No target and no GT arg: the param simply is not part of
                    # this call's supervision set (e.g. optional, omitted).
                    drop_why = "absent_from_gt"
                else:
                    # Extractable target existed but the matcher failed: BUG.
                    drop_why = "match_missed"

            # Direct argument fallback if target_spans missed.
            #
            # C1: distinguish "GT arg is explicitly null" from "GT arg absent".
            # ``.get()`` returns None for both; only the former is a real
            # supervision signal that must map to the <|null|> slot. Under
            # multi-call the fallback covers EVERY owning call's demanded value,
            # so a second call cannot lose its supervision to the first one.
            if not matched_cis:
                for gt_val in demanded:
                    for ci, c in enumerate(cands):
                        if c.canon_value is not None:
                            c_val = str(c.canon_value).strip().lower()
                            g_val = str(gt_val).strip().lower()
                            if c.canon_value == gt_val or c_val == g_val:
                                if ci not in matched_cis:
                                    matched_cis.append(ci)
                # T5: never manufacture a null positive for a ``match_missed``
                # param. The GT value was extractable and the matcher lost it;
                # supervising null would teach the model that a wrong answer is
                # the right one. Leaving the anchor unsupervised keeps the defect
                # visible in ``param_drop_reasons`` instead of burying it.
                if not matched_cis and drop_why != "match_missed" and (
                    any_null
                    or not param.required
                    or "null" in (param.param_type or "").lower()
                ):
                    # Explicit null GT, or an optional/nullable param omitted by
                    # the utterance -> deterministic <|null|> slot.
                    if null_ci is not None and null_ci not in matched_cis:
                        matched_cis.append(null_ci)

            # Record the reason ONLY if the param is still empty after the
            # fallbacks above. ``drop_why`` is computed before them on purpose
            # (it must be able to suppress the null fallback), but a fallback
            # that RESCUED the anchor means the param is supervised, so nothing
            # was dropped. Recording it anyway double-counts: the audit cell
            # cross-checks ``params with zero positives`` against the recorded
            # reasons, and a spurious ``match_missed`` would inflate the
            # "supervision defects (must be 0)" tripwire with params that were
            # never defects. The invariant this branch enforces is:
            #     len(_param_drop_reasons) == #params with no positive slot.
            if drop_why is not None and not matched_cis:
                drop_reasons.append(drop_why)
                if stats is not None:
                    stats.setdefault("param_drop_reasons", {})
                    stats["param_drop_reasons"][drop_why] = (
                        stats["param_drop_reasons"].get(drop_why, 0) + 1
                    )
            elif drop_why is not None:
                # Rescued by a fallback. This is NOT a drop, but it is still
                # evidence that the primary path (target_spans / matcher)
                # failed, so it must not be silently erased either: a large
                # ``match_missed`` rescue count means the aligner/matcher is
                # carrying a defect that the GT-argument fallback is papering
                # over. Counted on its own channel so it never triggers the
                # "supervision defects must be 0" tripwire.
                if stats is not None:
                    rescued = stats.setdefault("param_rescued_by_fallback", {})
                    rescued[drop_why] = rescued.get(drop_why, 0) + 1

        # Cap candidates if necessary.
        #
        # ``keep_set`` is the ONE authoritative membership set: every index we
        # decide to keep goes into it, and the emitted candidate list is exactly
        # ``sorted(keep_set)``. Do NOT track kept indices in a second list as
        # well -- an earlier version appended to ``pool`` *and* ``keep_set``, so
        # the break test ``len(pool) + len(keep_set)`` counted each index twice
        # and ``keep_set + pool`` emitted duplicate (tok_start, tok_end) spans.
        # That silently filled the 512 columns with ~2x repeats, leaving the
        # model a softmax over duplicated candidates and roughly half the
        # distinct coverage the resolver gets.
        if len(cands) > max_cands:
            keep_pos = matched_cis[:max_pos]
            # Positives first: a capped row must never lose the span it is
            # supervised on (matched_cis indices stay valid through idx_map).
            keep_set = set(keep_pos)
            # The deterministic <|null|> slot is a supervision target too, so it
            # is reserved rather than left to the stride fill.
            if null_ci is not None and null_ci < len(cands):
                keep_set.add(null_ci)
            # Cheap neighbourhood context: +-15 candidates around each positive,
            # so the model can still discriminate near-miss spans.
            for pi in keep_pos:
                if len(keep_set) >= max_cands:
                    break
                for delta in range(-15, 16):
                    ni = pi + delta
                    if 0 <= ni < len(cands) and ni not in keep_set:
                        keep_set.add(ni)
                        if len(keep_set) >= max_cands:
                            break
            # Stratified fill over the remaining budget, so the retained set
            # spans the whole candidate range instead of only its head.
            rem = max_cands - len(keep_set)
            if rem > 0:
                stride = max(1, len(cands) // rem)
                for ci in range(0, len(cands), stride):
                    if len(keep_set) >= max_cands:
                        break
                    keep_set.add(ci)
            kept_indices = sorted(keep_set)
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

    # ``call_param_index``: which supervised call owns each param anchor, using
    # the call's position in ``sample.ground_truth_calls``. Single-call rows are
    # all zeros, which is exactly the old behaviour (one implicit call). A
    # shared anchor names its FIRST owner; the full ownership list stays in
    # ``_anchor_calls`` so a reader can see the fan-out.
    call_param_index = [spec["calls"][0] for spec in anchor_specs]

    row: Dict[str, Any] = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.ones(len(ids), dtype=torch.long),
        "param_positions": torch.tensor(anchors, dtype=torch.long),
        "call_param_index": torch.tensor(call_param_index, dtype=torch.long),
        "n_calls": torch.tensor(len(supervised_calls), dtype=torch.long),
        "utt_starts": torch.tensor([utt_starts], dtype=torch.long),
        "utt_ends": torch.tensor([utt_ends], dtype=torch.long),
        "cand_starts": torch.tensor(cand_start_rows, dtype=torch.long),
        "cand_ends": torch.tensor(cand_end_rows, dtype=torch.long),
        "pos_indices": torch.tensor(pos_rows, dtype=torch.long),
        "_lean_converted": True,
    }

    if multi_call:
        # Which calls share each anchor. Purely diagnostic, but it is the only
        # way to see WHY an anchor got several positives: a genuine multi-call
        # fan-out, or an ambiguous shared parameter.
        row["_anchor_calls"] = [list(spec["calls"]) for spec in anchor_specs]

    # L1: expose the per-row drop-reason classification (names, not just counts)
    # so callers can surface ``match_missed`` rates without re-deriving them.
    if drop_reasons:
        row["_param_drop_reasons"] = drop_reasons

    if ambiguous_anchors:
        # Honest, not silent: the fast path cannot tell two calls' demands apart
        # on a shared anchor, so the marginal likelihood treats BOTH as valid.
        row["_ambiguous_anchor"] = int(ambiguous_anchors)
        if stats is not None:
            stats["multi_call_ambiguous_anchors"] = int(
                stats.get("multi_call_ambiguous_anchors", 0)
            ) + int(ambiguous_anchors)

    if tool_positions:
        row["tool_positions"] = torch.tensor(tool_positions, dtype=torch.long)
        # ``tool_targets`` stays the SCALAR primary-call index: the H1 loss is
        # ``F.cross_entropy(tool_logits (B, T), tool_targets (B,))`` and widening
        # it here would silently break that shape contract. The per-call vector
        # rides a separate key that J3's multi-positive loss consumes.
        row["tool_targets"] = torch.tensor(gt_idx, dtype=torch.long)
        row["call_tool_targets"] = torch.tensor(
            [ti for _ci, _c, ti in supervised_calls], dtype=torch.long
        )
        # Diagnostic: H1 is only a real task when this is > 1. Reported by the
        # trainer as ``mean_n_candidates`` so a single-candidate (vacuous
        # tool_acc=1.0) run is visible on its face instead of looking perfect.
        # Counts LIVE anchors: dead (unsnapped) slots are -1 and must not inflate
        # the number of choices H1 actually has.
        row["_n_tool_candidates"] = int(sum(1 for p in tool_positions if p >= 0))

    # T12a: the histogram of the per-row jitter DRAW, keyed on the realised floor
    # the row was actually built with. Recorded HERE, on the single kept-row exit,
    # so rows dropped downstream (over-length, unmappable GT anchor) cannot skew
    # it — ``jittered_rows`` counts draws, this counts surviving rows. Together
    # they say whether the jitter actually reached the trainer's data.
    if stats is not None and jitter and tool_pool is not None:
        hist = stats.get("jitter_draw_hist")
        if not isinstance(hist, dict):
            hist = {}
            stats["jitter_draw_hist"] = hist
        hist[row_pack_tools] = int(hist.get(row_pack_tools, 0)) + 1

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
                                ref.shape,
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
