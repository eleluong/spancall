"""Data augmentation strategies for SpanCall training robustness.

Ref: docs/mechanism.md §8.3, §8.4; docs/full_implementation_plan.md §2.2.
Generates controlled `UnifiedSample` variants from an *aligned* sample:

1. **Structural order shuffling** (`shuffle_f`): permute ``tool.parameters``,
   ``param.enum`` order and ``recent_context`` row order, then re-align so the
   char offsets reflect the new rendered layout (position invariance).
2. **Paraphrase / surface perturbation** (`paraphrase_f`): either deterministic
   surface perturbations outside target spans or pre-generated LLM paraphrases
   keyed by ``sample_id`` — verified to stay ``Stratum.F`` by re-alignment.
3. **Entity-swap hard negatives / role swaps** (`swap_f` + H7 ``swap_negatives``
   metadata): swap two same-typed entity mentions in the utterance AND their GT
   values, forcing the encoder to rely on syntax (``to X from Y``), not word
   order.
4. **Distractor injection** (`distractor_f`): 1–3 plausible-but-irrelevant
   recent rows keep the sample in Stratum F; **ambiguous anaphora negatives**
   (`ambiguous_a`) inject a key-colliding row, flip to Stratum A and label the
   paras parameter ``AMBIGUOUS`` for H4.
5. **Adversarial marker injection** (``adversarial_r7``): inject reserved
   markers (``<|sep|>``, legacy ``[SEP]``, fullwidth ``＜｜ｓｅｐ｜＞``/``［ＳＥＰ］``,
   zero-width chars, ...) into the utterance or recent values — these rows MUST
   abstain at inference (R7).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from span_template.template import RecentRow, build_input
from spancall.data.align import (
    ReverseAligner,
    _find_utterance_entry,
    _word_bounded_matches,
    tooldef_to_template_tool,
)
from spancall.data.base import BaseAugmenter
from spancall.data.schema import (
    ParameterSchema,
    SpanTarget,
    Stratum,
    ToolDefinition,
    UnifiedSample,
)

# Reserved markers / variants to inject for the R7 suite. Both the Qwen-native
# <|...|> forms (current template grammar) and the legacy bracketed [X] forms
# are reserved by sanitize._MARKER_RE, so both belong in the adversarial pool.
_ADVERSARIAL_MARKERS: Tuple[str, ...] = (
    "<|sep|>",
    "<|param|>",
    "<|ctx_enum|>",
    "<|ctx_meta|>",
    "<|tool|>",
    "[SEP]",
    "[PARAM]",
    "[CTX_ENUM]",
    "[CTX_META]",
    "[TOOL]",
    "＜｜ｓｅｐ｜＞",   # fullwidth NFKC-lookalike of <|sep|> (native form)
    "［ＳＥＰ］",     # fullwidth bracket NFKC-lookalike of [SEP] (legacy form)
    "\u200b",         # zero-width space
    "\ufffd",         # replacement char is NOT a marker; keep list tight
)
_ADVERSARIAL_TARGETS = ("utterance", "recent")


class DataAugmenter(BaseAugmenter):
    """Pipeline of generic, deterministic data augmenters."""

    def __init__(
        self,
        aligner: Optional[ReverseAligner] = None,
        enable_shuffling: bool = True,
        enable_paraphrase: bool = True,
        enable_swaps: bool = True,
        enable_distractors: bool = True,
        enable_adversarial_markers: bool = True,
        seed: int = 42,
        distractor_pool: Sequence[Dict[str, str]] = (),
        paraphrases: Optional[Dict[str, str]] = None,
        max_distractors: int = 3,
    ) -> None:
        self.aligner = aligner or ReverseAligner()
        self.enable_shuffling = enable_shuffling
        self.enable_paraphrase = enable_paraphrase
        self.enable_swaps = enable_swaps
        self.enable_distractors = enable_distractors
        self.enable_adversarial_markers = enable_adversarial_markers
        self.seed = seed
        self.distractor_pool = list(distractor_pool)
        self.paraphrases = dict(paraphrases or {})
        self.max_distractors = max_distractors

    # ---------------------------------------------------------------- main
    def augment(self, sample: UnifiedSample) -> List[UnifiedSample]:
        """Produce augmented variants (excluding the original sample)."""
        out: List[UnifiedSample] = []
        rng = random.Random(self._seed_for(sample))
        if self.enable_shuffling:
            v = self._shuffle_variant(sample, rng)
            if v is not None:
                out.append(v)
        if self.enable_paraphrase:
            out.extend(self._paraphrase_variants(sample, rng))
        if self.enable_swaps:
            out.extend(self._swap_variants(sample, rng))
        if self.enable_distractors:
            out.extend(self._distractor_variants(sample, rng))
        if self.enable_adversarial_markers:
            out.extend(self._adversarial_variants(sample, rng))
        return out

    # -------------------------------------------------------- 1. shuffling
    def _shuffle_variant(self, sample: UnifiedSample, rng: random.Random) -> Optional[UnifiedSample]:
        """Permute parameter/enum/recent-row order and re-align (position invariance)."""
        tools = [
            ToolDefinition(
                name=t.name,
                description=t.description,
                parameters=_shuffled_params(list(t.parameters), rng),
                risk_tier=t.risk_tier,
            )
            for t in sample.tools
        ]
        recent = list(sample.recent_context)
        rng.shuffle(recent)
        if not _order_changed(sample.tools, tools) and recent == list(sample.recent_context):
            return None  # nothing to shuffle -> skip (keep natural_f budget intact)
        variant = _copy_for_aug(sample, aug_type="shuffle_f")
        variant.tools = tools
        variant.recent_context = recent
        # clear stale groundings; offsets change with the new render
        variant.target_spans = {}
        variant.metadata.pop("h4_reasons", None)
        return self.aligner.align(variant)

    # ------------------------------------------------------ 2. paraphrase
    def _paraphrase_variants(self, sample: UnifiedSample, rng: random.Random) -> List[UnifiedSample]:
        """Surface perturbations outside target spans (or LLM paraphrases)."""
        out: List[UnifiedSample] = []
        # (a) pre-generated paraphrases keyed by sample_id
        para = self.paraphrases.get(sample.sample_id)
        if para:
            v = _copy_for_aug(sample, aug_type="paraphrase_f")
            v.sample_id = f"{sample.sample_id}::para"
            v.utterance = para
            v.target_spans = {}
            v.metadata.pop("h4_reasons", None)
            v = self.aligner.align(v)
            if v.stratum == Stratum.F:
                v.metadata["paraphrase_source"] = "llm"
                out.append(v)
        # (b) deterministic surface perturbation on casing (R4: case-insensitive
        #     params keep their span; re-align verifies it stays F)
        if sample.stratum == Stratum.F and sample.utterance:
            swapped = _case_swapped_utterance(sample)
            if swapped and swapped != sample.utterance:
                v = _copy_for_aug(sample, aug_type="paraphrase_f")
                v.sample_id = f"{sample.sample_id}::case"
                v.utterance = swapped
                v.target_spans = {}
                v.metadata.pop("h4_reasons", None)
                v = self.aligner.align(v)
                if v.stratum == Stratum.F:
                    v.metadata["paraphrase_source"] = "surface"
                    out.append(v)
        return out

    # --------------------------------------------------------- 3. swaps
    def _swap_variants(self, sample: UnifiedSample, rng: random.Random) -> List[UnifiedSample]:
        """Utterance+GT entity-role swaps for same-typed string/numeric params.

        ``SpanTarget.char_*`` are ABSOLUTE offsets into the rendered template,
        not the raw utterance; we therefore recover the raw-utterance ranges via
        the canonical value's word-bounded occurrence before swapping.
        """
        if sample.stratum != Stratum.F or not sample.ground_truth_calls:
            return []
        gt = sample.ground_truth_calls[0]
        tool = _find_tool(sample.tools, gt.tool_name)
        if tool is None or len(tool.parameters) < 2:
            return []
        pairs = _same_type_pairs(tool.parameters, sample.target_spans)
        out: List[UnifiedSample] = []
        for (p_a, span_a, p_b, span_b) in pairs:
            if p_a.name not in gt.arguments or p_b.name not in gt.arguments:
                continue
            # raw-utterance occurrences of the two canonical values
            occ_a = _word_bounded_matches(sample.utterance, str(span_a.canonical_value))
            occ_b = _word_bounded_matches(sample.utterance, str(span_b.canonical_value))
            if not occ_a or not occ_b:
                continue  # value not present verbatim -> cannot swap
            ra, rb = occ_a[0], occ_b[0]
            lo, hi = (ra, rb) if ra[0] <= rb[0] else (rb, ra)
            if lo[1] > hi[0]:
                continue  # genuinely overlapping in the raw utterance
            v = _copy_for_aug(sample, aug_type="swap_f")
            v.sample_id = f"{sample.sample_id}::swap_{p_a.name}_{p_b.name}"
            v.utterance = _swap_ranges(sample.utterance, lo, hi)
            args = dict(gt.arguments)
            args[p_a.name], args[p_b.name] = args[p_b.name], args[p_a.name]
            v.ground_truth_calls = [
                replace(g, arguments=args) if g is gt else g
                for g in v.ground_truth_calls
            ]
            v.target_spans = {}
            v.metadata.pop("h4_reasons", None)
            # H7: pre-swap layout — for each param the wrong span is where ITS
            # value originally sat, the correct span is the other value's home.
            v.metadata["swap_negatives"] = {
                p_a.name: {
                    "wrong_span_char": list(ra),
                    "correct_span_char": list(rb),
                },
                p_b.name: {
                    "wrong_span_char": list(rb),
                    "correct_span_char": list(ra),
                },
            }
            v = self.aligner.align(v)
            if v.stratum == Stratum.F:
                out.append(v)
        return out

    # ----------------------------------------------------- 4. distractors
    def _distractor_variants(self, sample: UnifiedSample, rng: random.Random) -> List[UnifiedSample]:
        """Context distractors (Stratum F) + ambiguous anaphora negatives (Stratum A)."""
        out: List[UnifiedSample] = []
        pool = list(self.distractor_pool)
        if pool:
            # (a) 1..3 irrelevant rows; re-align must stay Stratum.F
            k = rng.randint(1, self.max_distractors)
            distractors = rng.sample(pool, min(k, len(pool)))
            v = _copy_for_aug(sample, aug_type="distractor_f")
            v.sample_id = f"{sample.sample_id}::dist"
            v.recent_context = list(sample.recent_context) + list(distractors)
            v.target_spans = {}
            v.metadata.pop("h4_reasons", None)
            v = self.aligner.align(v)
            if v.stratum == Stratum.F:
                v.metadata["distractor_keys"] = [d.get("key") for d in distractors]
                out.append(v)
            # (b) ambiguous anaphora negatives — one per recent-grounded param
            for pname, targets in sample.target_spans.items():
                recent_t = next((t for t in targets if t.target_type == "recent_ctx"), None)
                if recent_t is None or not recent_t.ctx_key or not pool:
                    continue
                d = rng.choice(pool)
                alt_key = f"{recent_t.ctx_key}_alt"
                while alt_key in {r["key"] for r in sample.recent_context}:
                    alt_key += "_x"
                v = _copy_for_aug(sample, aug_type="ambiguous_a")
                v.sample_id = f"{sample.sample_id}::ambig_{pname}"
                v.recent_context = list(sample.recent_context) + [
                    {"key": alt_key, "value": str(d.get("value", "")), "source": "augmented"}
                ]
                v.target_spans = {}
                v.metadata.pop("h4_reasons", None)
                v = self.aligner.align(v)
                # force A + AMBIGUOUS on the relevant param (spec §2.2 step 4)
                v.stratum = Stratum.A
                v.metadata.setdefault("h4_reasons", {})[pname] = "AMBIGUOUS"
                v.metadata["ambiguous_param"] = pname
                v.metadata["expected_abstain"] = "ambiguous"
                out.append(v)
        return out

    # -------------------------------------------------- 5. adversarial R7
    def _adversarial_variants(self, sample: UnifiedSample, rng: random.Random) -> List[UnifiedSample]:
        """Inject reserved markers -> these rows MUST abstain (R7)."""
        out: List[UnifiedSample] = []
        target = rng.choice(_ADVERSARIAL_TARGETS)
        marker = rng.choice(_ADVERSARIAL_MARKERS)
        v = _copy_for_aug(sample, aug_type="adversarial_r7")
        v.sample_id = f"{sample.sample_id}::r7"
        if target == "utterance":
            v.utterance = _inject_marker(sample.utterance, marker, rng)
        else:
            v.recent_context = list(sample.recent_context)
            if v.recent_context:
                idx = rng.randrange(len(v.recent_context))
                row = dict(v.recent_context[idx])
                row["value"] = _inject_marker(str(row.get("value", "")), marker, rng)
                v.recent_context[idx] = row
            else:
                v.recent_context = [{"key": "injected", "value": _inject_marker("x", marker, rng)}]
        v.target_spans = {}
        v.metadata.pop("h4_reasons", None)
        v.metadata["expected_abstain"] = "marker_injection"
        v.metadata["injected_marker"] = repr(marker)
        v.metadata["injection_target"] = target
        v.stratum = Stratum.A
        out.append(v)
        return out

    # -------------------------------------------------------------- rng
    def _seed_for(self, sample: UnifiedSample) -> int:
        digest = hashlib.sha256(sample.sample_id.encode("utf-8")).digest()
        return (self.seed + int.from_bytes(digest[:4], "big")) % (2**31)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _copy_for_aug(sample: UnifiedSample, aug_type: str) -> UnifiedSample:
    """Copy with clean per-variant metadata and provenance tags."""
    out = replace(
        sample,
        sample_id=sample.sample_id,
        tools=list(sample.tools),
        ground_truth_calls=[replace(g) for g in sample.ground_truth_calls],
        recent_context=list(sample.recent_context),
        target_spans={k: list(v) for k, v in sample.target_spans.items()},
        metadata={k: v for k, v in sample.metadata.items() if k not in ("h4_reasons",)},
    )
    augs = out.metadata.get("aug_chain", [])
    if not isinstance(augs, list):
        augs = []
    out.metadata["aug_chain"] = augs + [aug_type]
    out.metadata["aug_type"] = aug_type
    return out


def _shuffled_params(params: List[ParameterSchema], rng: random.Random) -> List[ParameterSchema]:
    out: List[ParameterSchema] = []
    for p in params:
        enum = p.enum
        if enum:
            shuffled = list(enum)
            rng.shuffle(shuffled)
            out.append(replace(p, enum=shuffled))
        else:
            out.append(p)
    rng.shuffle(out)  # parameter order too
    return out


def _order_changed(a: List[ToolDefinition], b: List[ToolDefinition]) -> bool:
    """True if any parameter/enum/recent ordering differs between a and b."""
    for ta, tb in zip(a, b):
        if [p.name for p in ta.parameters] != [p.name for p in tb.parameters]:
            return True
        for pa, pb in zip(ta.parameters, tb.parameters):
            if (pa.enum or []) != (pb.enum or []):
                return True
    return False


def _same_type_pairs(
    params: List[ParameterSchema],
    target_spans: Dict[str, List[SpanTarget]],
) -> List[Tuple[ParameterSchema, SpanTarget, ParameterSchema, SpanTarget]]:
    """(p_a, span_a, p_b, span_b) pairs of same string/numeric type with
    distinct utterance spans (entity roles like origin/destination)."""
    candidates: List[Tuple[ParameterSchema, SpanTarget]] = []
    for p in params:
        if p.param_type not in ("string", "integer", "number"):
            continue
        span = next(
            (t for t in target_spans.get(p.name, []) if t.target_type == "utterance_span"),
            None,
        )
        if span is not None:
            candidates.append((p, span))
    pairs: List[Tuple[ParameterSchema, SpanTarget, ParameterSchema, SpanTarget]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            pa, sa = candidates[i]
            pb, sb = candidates[j]
            # sanity: different params, same class (both string / both numeric)
            same_class = (pa.param_type == pb.param_type)
            if same_class:
                pairs.append((pa, sa, pb, sb))
    return pairs


def _swap_ranges(text: str, r1: Tuple[int, int], r2: Tuple[int, int]) -> str:
    """swap text[a:b] <-> text[c:d] where a<b<=c<d."""
    a, b = r1
    c, d = r2
    return text[:a] + text[c:d] + text[b:c] + text[a:b] + text[d:]


def _case_swapped_utterance(sample: UnifiedSample) -> Optional[str]:
    """Toggle the case of a few *uncharacteristic* words (defensive: safe to
    re-render; the aligner decides if it still grounds)."""
    words = sample.utterance.split()
    if not words:
        return None
    rng = random.Random(hash(sample.sample_id) % (2**31))
    idx = rng.randrange(len(words))
    new_words = list(words)
    w = new_words[idx]
    if w == w.lower() and w != w.upper():
        new_words[idx] = w.upper()
    elif w == w.upper() and len(w) > 1:
        new_words[idx] = w.lower()
    else:
        return None
    return " ".join(new_words)


def _find_tool(tools: Sequence[ToolDefinition], name: str) -> Optional[ToolDefinition]:
    for t in tools:
        if t.name == name:
            return t
    return None


def _inject_marker(text: str, marker: str, rng: random.Random) -> str:
    """Append a reserved marker at a random position (R7 injection)."""
    if not text:
        return text + marker
    pos = rng.randrange(len(text) + 1)
    if marker in ("\u200b", "\ufffd"):
        return text[:pos] + marker + text[pos:]
    # bracket markers are returned verbatim so sanitize raises marker_injection
    return text[:pos] + marker + text[pos:]
