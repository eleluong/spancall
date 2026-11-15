#!/usr/bin/env python3
"""End-to-end smoke for Fix 1 (multi-positive array training + list decode).

Trains a tiny Lean model on one array-param sample through the REAL pipeline
(ReverseAligner -> tensorize_sample_lean -> LeanBatchCollate -> LeanTrainer),
then runs LeanResolver.try_resolve on the same utterance and checks that the
full list is emitted — plus a list_decode=False control showing the old
single-element truncation behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spancall.data.align import ReverseAligner
from spancall.data.schema import GroundTruthCall, ParameterSchema, Stratum, ToolDefinition, UnifiedSample
from spancall.infer.base import FakeTokenizer
from spancall.lean.model import LeanSpanCallModel
from spancall.lean.resolver import LeanResolver
from spancall.lean.tensorize import LeanBatchCollate, tensorize_sample_lean
from spancall.lean.trainer import LeanTrainer
from spancall.model.masks import CandidateMaskBuilder


class PosEncoder(nn.Module):
    """Toy encoder with positional info — needed to discriminate spans
    (a pure char-bag embedding cannot sharpen span boundaries)."""

    def __init__(self, hidden_dim: int = 128, maxlen: int = 1100) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(1000, hidden_dim)
        self.pos = nn.Embedding(maxlen, hidden_dim)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        L = input_ids.shape[1]
        return self.emb(input_ids) + self.pos(torch.arange(L).unsqueeze(0))


def make_sample():
    arr = ToolDefinition(
        name="add_tags",
        description="Tag items",
        parameters=[
            ParameterSchema(
                name="tags", description="tags", param_type="array",
                items_type="string", required=True,
            )
        ],
    )
    return arr, ReverseAligner().align(UnifiedSample(
        sample_id="arr-smoke",
        source_dataset="test",
        utterance="tag urgent, legal and finance",
        tools=[arr],
        ground_truth_calls=[
            GroundTruthCall(tool_name="add_tags", arguments={"tags": ["urgent", "legal", "finance"]})
        ],
        current_time_iso="2026-11-09",
        stratum=Stratum.F,
    ))


def main() -> int:
    torch.manual_seed(0)
    schema, sample = make_sample()

    tok = FakeTokenizer()
    mb = CandidateMaskBuilder(max_seq_len=1024, max_span_width=16)
    rows = []
    for _ in range(4):  # replicate the row so a batch exists
        row = tensorize_sample_lean(sample, tok, mb, max_seq_len=1024, max_pos=1, pack_multi_tools=True)
        assert row is not None
        rows.append(row)
    batch = LeanBatchCollate()(rows)

    live = (batch["pos_indices"][0, 0] >= 0).sum().item()
    print(f"[tensorize] live positives for the array param: {live}")
    assert live >= 3, "array param must keep all 3 element positives"

    torch.manual_seed(0)
    model = LeanSpanCallModel(encoder=PosEncoder(hidden_dim=128), head_dim=64, hidden_dim=128)
    trainer = LeanTrainer(model=model, lr_encoder=3e-3, lr_head=3e-3, device="cpu", use_amp=False)
    for step in range(8000):
        m = trainer.train_step(batch)
        if step % 2000 == 0 or step == 7999:
            print(f"[train] step {step:4d}  loss={m['loss']:.4f}  span_acc={m.get('span_acc')}")
    assert not torch.isnan(torch.tensor(m["loss"]))

    def resolve(list_decode: bool):
        resolver = LeanResolver(model=model, max_seq_len=1024, list_decode=list_decode)
        res = resolver.try_resolve(schema, utterance="tag urgent, legal and finance")
        pr = res.parameters.get("tags")
        return res, (pr.value if pr is not None else None)

    res_new, value_new = resolve(list_decode=True)
    print(f"[list_decode=True ] tags value: {value_new!r}")
    res_old, value_old = resolve(list_decode=False)
    print(f"[list_decode=False] tags value: {value_old!r}  (control: old single-winner path)")

    ok = True
    if not (isinstance(value_new, list) and set(value_new) >= {"urgent", "legal", "finance"}):
        print("FAIL: list_decode=True did not emit the full element set")
        ok = False
    if value_new == value_old and not (isinstance(value_new, list) and len(value_new) >= 3):
        print("FAIL: decode path unchanged — fix is not active")
        ok = False
    print("\nSMOKE " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
