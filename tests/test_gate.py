"""D4 gate + heads config tests (docs/remediation_plan.md D4).

Covers:
  - HeadsConfig v1 defaults (H1/H3/H4/H8 on; H2/H5/H6/H7 spec-only) and
    ``enabled()``.
  - GateConfig ``mode`` default ``single_tau`` and YAML roundtrip via
    ``TrainingConfig.from_yaml``.
  - SingleTauGate: hit <=> min_k p_k >= tau_p AND accept >= tau_a AND
    validate(); and each abstain reason.
  - ConjunctionGate: adds the joint-margin check (joint_margin >= tau_delta);
    this is the 8-feature ablation only.
  - build_gate: mode dispatch (single_tau | conjunction | unknown).
"""

import unittest
from pathlib import Path

from spancall.infer import (
    BaseGate,
    ConjunctionGate,
    SingleTauGate,
    build_gate,
)
from spancall.train.config import (
    DataConfig,
    GateConfig,
    HeadsConfig,
    ModelConfig,
    TrainingConfig,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = ROOT / "configs" / "default.yaml"


class HeadsConfigTest(unittest.TestCase):
    def test_v1_defaults_match_d4(self):
        h = HeadsConfig()
        self.assertTrue(h.tool)          # H1
        self.assertFalse(h.count)        # H2 spec-only
        self.assertTrue(h.pointer)       # H3
        self.assertTrue(h.abstain_reason)  # H4
        self.assertFalse(h.source)       # H5 spec-only
        self.assertFalse(h.list_size)    # H6 spec-only
        self.assertFalse(h.swap)         # H7 spec-only
        self.assertTrue(h.correctness)   # H8

    def test_enabled_returns_v1_set(self):
        h = HeadsConfig()
        self.assertEqual(
            h.enabled(),
            ("tool", "pointer", "abstain_reason", "correctness"),
        )

    def test_from_mapping_keeps_unknown_keys_out(self):
        h = HeadsConfig.from_mapping({"tool": False, "bogus": True})
        self.assertFalse(h.tool)
        self.assertTrue(h.pointer)  # default preserved

    def test_model_config_embeds_heads(self):
        m = ModelConfig.from_dict({"heads": {"tool": False, "count": True}})
        self.assertIsInstance(m.heads, HeadsConfig)
        self.assertFalse(m.heads.tool)
        self.assertTrue(m.heads.count)
        self.assertTrue(m.heads.pointer)


class GateConfigTest(unittest.TestCase):
    def test_mode_default_is_single_tau(self):
        self.assertEqual(GateConfig().mode, "single_tau")

    def test_default_yaml_roundtrip(self):
        cfg = TrainingConfig.from_yaml(DEFAULT_YAML)
        # D4 heads defaults land via model.heads
        self.assertEqual(
            cfg.model.heads.enabled(),
            ("tool", "pointer", "abstain_reason", "correctness"),
        )
        # D4 gate mode lands via gate.mode
        self.assertEqual(cfg.gate.mode, "single_tau")
        self.assertEqual(cfg.gate.tau_p, 0.90)
        self.assertEqual(cfg.gate.tau_a, 0.85)
        self.assertEqual(cfg.gate.tau_delta, 2.0)

    def test_conjunction_mode_roundtrip(self):
        cfg = TrainingConfig.from_yaml(DEFAULT_YAML)
        cfg.gate = GateConfig.from_dict({"mode": "conjunction"})
        self.assertEqual(cfg.gate.mode, "conjunction")


class SingleTauGateTest(unittest.TestCase):
    def _mk(self, **kw):
        return SingleTauGate(**kw)

    def test_hit_when_all_pass(self):
        g = self._mk(tau_p=0.90, tau_a=0.85)
        hit, reason = g.evaluate(
            param_probabilities={"a": 0.97, "b": 0.92},
            joint_margin=3.0,           # deliberately unused in v1
            accept_probability=0.91,
            is_schema_valid=True,
        )
        self.assertTrue(hit)
        self.assertIsNone(reason)

    def test_no_params_abstains(self):
        g = self._mk()
        hit, reason = g.evaluate({}, joint_margin=0.0,
                                 accept_probability=1.0, is_schema_valid=True)
        self.assertFalse(hit)
        self.assertEqual(reason, "no_params")

    def test_abstain_when_min_p_below_tau(self):
        g = self._mk(tau_p=0.90, tau_a=0.85)
        hit, reason = g.evaluate(
            {"a": 0.97, "b": 0.84},
            joint_margin=3.0, accept_probability=0.91, is_schema_valid=True,
        )
        self.assertFalse(hit)
        self.assertIn("min_p_below_tau", reason)

    def test_abstain_when_accept_below_tau_a(self):
        g = self._mk(tau_p=0.90, tau_a=0.85)
        hit, reason = g.evaluate(
            {"a": 0.97, "b": 0.92},
            joint_margin=3.0, accept_probability=0.80, is_schema_valid=True,
        )
        self.assertFalse(hit)
        self.assertIn("accept_below_tau_a", reason)

    def test_abstain_when_schema_invalid(self):
        g = self._mk(tau_p=0.90, tau_a=0.85)
        hit, reason = g.evaluate(
            {"a": 0.97, "b": 0.92},
            joint_margin=3.0, accept_probability=0.91, is_schema_valid=False,
        )
        self.assertFalse(hit)
        self.assertEqual(reason, "schema_invalid")

    def test_joint_margin_ignored_in_v1_decision(self):
        # D4: joint margin drops out of the v1 decision (calibration signal only).
        g = self._mk(tau_p=0.90, tau_a=0.85, tau_delta=2.0)
        hit, _ = g.evaluate(
            {"a": 0.97, "b": 0.92},
            joint_margin=0.1,           # below tau_delta — must NOT abstain
            accept_probability=0.91, is_schema_valid=True,
        )
        self.assertTrue(hit)


class ConjunctionGateTest(unittest.TestCase):
    def test_joint_margin_is_decision_input(self):
        g = ConjunctionGate(tau_p=0.90, tau_delta=2.0, tau_a=0.85)
        # Same params that pass single-τ, but the joint margin fails.
        hit, reason = g.evaluate(
            {"a": 0.97, "b": 0.92},
            joint_margin=0.1,
            accept_probability=0.91, is_schema_valid=True,
        )
        self.assertFalse(hit)
        self.assertIn("joint_margin_below_tau_delta", reason)

    def test_conjunction_hit(self):
        g = ConjunctionGate(tau_p=0.90, tau_delta=2.0, tau_a=0.85)
        hit, reason = g.evaluate(
            {"a": 0.97, "b": 0.92},
            joint_margin=2.5,
            accept_probability=0.91, is_schema_valid=True,
        )
        self.assertTrue(hit)
        self.assertIsNone(reason)


class PackToolsTest(unittest.TestCase):
    def test_data_config_default_packs_one_tool(self):
        # The DataConfig dataclass default is 1; the production multi-tool
        # default (3) ships via default.yaml so callers opt in explicitly.
        cfg = DataConfig()
        self.assertEqual(cfg.pack_tools, 1)

    def test_default_yaml_packs_three_tools(self):
        # H1 distractor-training upgrade: default.yaml ships pack_tools: 3 so
        # the H1 tool head is trained against explicit distractor candidate sets.
        cfg = TrainingConfig.from_yaml(DEFAULT_YAML)
        self.assertEqual(cfg.data.pack_tools, 3)

    def test_pack_tools_opt_in(self):
        d = DataConfig.from_dict({"pack_tools": 2})
        self.assertEqual(d.pack_tools, 2)

    def test_from_dict_ignores_unknown_keys(self):
        d = DataConfig.from_dict({"pack_tools": 3, "bogus": "x"})
        self.assertEqual(d.pack_tools, 3)

    def test_data_section_unknown_keys_tolerated(self):
        cfg = TrainingConfig.from_yaml(DEFAULT_YAML)
        # sources/stratum_mix/schema_split are unmodeled YAML — ignored, not errors
        self.assertEqual(cfg.data.pack_tools, 3)
        self.assertFalse(hasattr(cfg.data, "sources"))

    def test_mixer_receives_pack_tools(self):
        from spancall.data import DatasetMixer
        m = DatasetMixer(pack_tools=2)
        self.assertEqual(m.pack_tools, 2)


class BuildGateTest(unittest.TestCase):
    def test_default_is_single_tau(self):
        g = build_gate()
        self.assertIsInstance(g, SingleTauGate)
        self.assertIsInstance(g, BaseGate)

    def test_conjunction_dispatch(self):
        g = build_gate(mode="conjunction")
        self.assertIsInstance(g, ConjunctionGate)

    def test_thresholds_forwarded(self):
        g = build_gate(mode="single_tau", tau_p=0.95, tau_a=0.9, tau_delta=3.0)
        self.assertEqual(g.tau_p, 0.95)
        self.assertEqual(g.tau_a, 0.9)
        self.assertEqual(g.tau_delta, 3.0)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            build_gate(mode="wizard_gate")


if __name__ == "__main__":
    unittest.main()