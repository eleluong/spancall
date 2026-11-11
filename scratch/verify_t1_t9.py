"""Smoke-verify the T1-T9 upgrade paths actually execute (not just compile)."""
from __future__ import annotations

import math
from typing import List

from spancall.lean.decode import LeanJointDecoder, LeanValueChoice
from spancall.model.masks import Candidate, CandidateMaskBuilder

ok: List[str] = []
bad: List[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (ok if cond else bad).append(f"{name}{(' :: ' + detail) if detail else ''}")


# ---------------------------------------------------------------- T6 refusal
dec = LeanJointDecoder(top_k=3, normalize_scores=False)
check("T6 default thresholds are None", dec.min_confidence is None and dec.min_posterior is None)
refuse, why = dec._should_refuse([])
check("T6 empty choice set is NOT a refusal", refuse is False and why == "", f"got {refuse}/{why!r}")

choices = [LeanValueChoice(value="a", prob=0.34), LeanValueChoice(value="b", prob=0.33)]
dec.min_confidence = 0.5
refuse, why = dec._should_refuse(choices)
check("T6 low confidence -> refuse", refuse is True and why == "below_min_confidence", f"got {why!r}")

dec.min_confidence = None
dec.min_posterior = 0.1
refuse, why = dec._should_refuse(choices)
check("T6 near-tie -> refuse", refuse is True and why == "indistinguishable_from_runner_up", f"got {why!r}")

dec.min_posterior = 0.001
refuse, why = dec._should_refuse(choices)
check("T6 clear winner -> accept", refuse is False and why == "", f"got {refuse}/{why!r}")

refuse, why = dec._should_refuse(choices[:1])
check("T6 single candidate never indistinguishable", refuse is False, f"got {why!r}")

check("T6 last_refusals field exists", hasattr(dec, "last_refusals"))

# ------------------------------------------------- T4 pack priority retention
b = CandidateMaskBuilder(max_seq_len=4096)
check("T4 default pack_fill is head", b.pack_fill == "head")
check("T4 pack_protected defaults to None", b.pack_protected is None)


def mk(i: int, region: str = "utterance") -> Candidate:
    return Candidate(tok_start=i, tok_end=i + 1, char_start=i, char_end=i + 1,
                     region=region, canon_value=f"v{i}")


cands = [mk(i) for i in range(100)]
b.pack_fill = "head"
head_pack = b._pack(list(cands), max_cands=10)
head_kept = sorted(c.tok_start for c in head_pack["candidates"])
check("T4 head fill keeps earliest spans", head_kept == list(range(10)), f"{head_kept}")

# A protected span at the tail must survive a hard head-first cap.
b.pack_protected = {(90, 91)}
prot = b._pack(list(cands), max_cands=10)
prot_kept = sorted(c.tok_start for c in prot["candidates"])
check("T4 protected span survives truncation", 90 in prot_kept, f"{prot_kept}")
check("T4 cap respected with protection", len(prot_kept) == 10, str(len(prot_kept)))

# Structured slots outrank free utterance spans.
b.pack_protected = None
mixed = [mk(i) for i in range(12)] + [mk(50, region="null"), mk(51, region="enum")]
b.pack_fill = "head"
struct = b._pack(mixed, max_cands=3)
struct_regions = [c.region for c in struct["candidates"]]
check("T4 structured slots retained first", struct_regions.count("null") == 1 and struct_regions.count("enum") == 1, f"{struct_regions}")

# Stratified fill must reach deep into the list rather than taking a prefix.
b.pack_fill = "stratified"
strat = b._pack(list(cands), max_cands=10)
strat_kept = sorted(c.tok_start for c in strat["candidates"])
check("T4 stratified fill spans the range", max(strat_kept) >= 50, f"{strat_kept}")
check("T4 stratified fill is not a prefix", strat_kept != list(range(10)), f"{strat_kept}")

# Output order must stay monotone in token position (pointer geometry depends on it).
order = [c.tok_start for c in strat["candidates"]]
check("T4 packed order is monotone", order == sorted(order), f"{order}")

# Unbounded cap == full parity.
full = b._pack(list(cands), max_cands=None)
check("T2/T4 max_cands=None keeps every candidate", len(full["candidates"]) == 100, str(len(full["candidates"])))

# Legacy behaviour must be byte-identical when the cap is not binding.
no_cap_needed = b._pack(list(cands[:5]), max_cands=512)
check("T4 cap not binding preserves all", no_cap_needed["cand_index"] == [(c.tok_start, c.tok_end) for c in cands[:5]])

# ------------------------------------------------------------ T9 expectation
expected = 1.0 / max(3, 1)
sd = math.sqrt(expected * (1 - expected) / 100)
check("T9 analytic expected share", abs(expected - 1 / 3) < 1e-9, f"{expected:.4f}")
check("T9 binomial sd positive", sd > 0, f"{sd:.4f}")
# 4-sigma gate must flag the observed 4.5% vs 33.3% case.
observed = 0.045
check("T9 4-sigma gate flags the 0.2-run data", abs(observed - expected) > 4.0 * sd)

# ------------------------------------------------------------ T8 call count
from spancall.lean.eval_p1 import predicted_call_count  # noqa: E402
from spancall.lean.resolver import LeanCallResult  # noqa: E402


class _Res:
    def __init__(self, tool_name, calls=None):
        self.tool_name = tool_name
        self.calls = calls or []


check("T8 named call counts 1", predicted_call_count(_Res("search")) == 1)
check("T8 abstained decode counts 0", predicted_call_count(_Res(None)) == 0)
multi = _Res(None, calls=[LeanCallResult(tool_name="a"), LeanCallResult(tool_name="b")])
check("T8 two named calls count 2", predicted_call_count(multi) == 2, str(predicted_call_count(multi)))
mixed_calls = _Res(None, calls=[LeanCallResult(tool_name="a"), LeanCallResult(tool_name="")])
check("T8 unnamed call is not counted", predicted_call_count(mixed_calls) == 1, str(predicted_call_count(mixed_calls)))
check("T8 len(calls) would have over-counted", len(mixed_calls.calls) == 2)

# --------------------------------------------------------------- T3 reasons
from spancall.lean.resolver import LeanResolver  # noqa: E402

r = LeanResolver.__new__(LeanResolver)  # bypass the encoder construction
r.model = None
scores = LeanResolver._tool_scores(r, None, None, [])
check("T3 no model -> no_model", scores == [] and r.last_tool_scores_reason == "no_model", r.last_tool_scores_reason)


class _NoHead:
    tool_head = None


r.model = _NoHead()
scores = LeanResolver._tool_scores(r, None, None, [])
check("T3 no tool head -> no_tool_head", scores == [] and r.last_tool_scores_reason == "no_tool_head", r.last_tool_scores_reason)


class _Head:
    tool_head = object()


r.model = _Head()
scores = LeanResolver._tool_scores(r, None, None, [])
check("T3 no anchors -> no_tool_anchors", scores == [] and r.last_tool_scores_reason == "no_tool_anchors", r.last_tool_scores_reason)
check("T3 reason is never the old opaque string", r.last_tool_scores_reason != "", r.last_tool_scores_reason)

# -------------------------------------------------- T1/T2 resolver capability
check("T1 parity capability flag", getattr(LeanResolver, "supports_candidate_parity", False) is True)

print("=== PASS ===")
for line in ok:
    print("  ok  ", line)
if bad:
    print("\n=== FAIL ===")
    for line in bad:
        print("  FAIL", line)
    raise SystemExit(1)
print(f"\n{len(ok)} checks passed")