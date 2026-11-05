"""Test suite verifying that all SpanCall boilerplate modules, classes, and types import and instantiate."""

import unittest


class TestBoilerplateIntegrity(unittest.TestCase):

    def test_imports_data_subpackage(self):
        from spancall.data import (
            BaseAligner,
            BaseAugmenter,
            BaseDatasetLoader,
            DataAugmenter,
            DatasetMixer,
            GroundTruthCall,
            ParameterSchema,
            ReverseAligner,
            SpanTarget,
            Split,
            Stratum,
            ToolDefinition,
            UnifiedSample,
        )
        sample = UnifiedSample(
            sample_id="test_001",
            source_dataset="synthetic",
            utterance="test utterance",
            tools=[],
            ground_truth_calls=[],
            stratum=Stratum.F,
            split=Split.TRAIN,
        )
        self.assertEqual(sample.stratum, Stratum.F)
        self.assertEqual(sample.split, Split.TRAIN)

    def test_imports_model_subpackage(self):
        from spancall.model import (
            AbstainHead,
            BaseEncoder,
            BaseHead,
            CallAcceptHead,
            CallCountHead,
            CandidateMaskBuilder,
            PointerHead,
            SpanCallLossContainer,
            SpanCallTransformerEncoder,
            ToolHead,
        )
        encoder = SpanCallTransformerEncoder()
        self.assertEqual(encoder.model_name_or_path, "Qwen/Qwen3-Embedding-0.6B")
        loss_fn = SpanCallLossContainer()
        self.assertEqual(loss_fn.lambda_tool, 1.0)

    def test_imports_train_subpackage(self):
        from spancall.train import (
            BaseCalibrator,
            BaseTrainer,
            CalibrationConfig,
            GateConfig,
            ModelConfig,
            OptimizationConfig,
            TrainingConfig,
        )
        config = TrainingConfig()
        self.assertEqual(config.optimization.batch_size, 64)
        self.assertEqual(config.optimization.num_epochs, 4)
        calib_config = CalibrationConfig()
        self.assertEqual(calib_config.target_precision, 0.95)
        gate_config = GateConfig()
        self.assertEqual(gate_config.tau_p, 0.90)
        self.assertEqual(gate_config.tau_delta, 2.0)
        self.assertEqual(gate_config.tau_a, 0.85)

    def test_training_config_from_yaml_roundtrip(self):
        # design rule 4: an experiment = YAML + seed — the dataclasses must load
        # straight from configs/default.yaml (unknown keys tolerated).
        from spancall.train import TrainingConfig
        cfg = TrainingConfig.from_yaml("configs/default.yaml")
        self.assertEqual(cfg.model.backbone_name, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(cfg.optimization.learning_rate_encoder, 3e-5)
        self.assertEqual(cfg.optimization.batch_size, 64)
        self.assertEqual(cfg.optimization.num_epochs, 4)
        self.assertEqual(cfg.output_dir, "results")
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.calibration.target_precision, 0.95)
        self.assertEqual(cfg.gate.tau_p, 0.90)

    def test_imports_infer_subpackage(self):
        from spancall.infer import (
            BaseExporter,
            BaseGate,
            BaseJointDecoder,
            BaseTypeCaster,
            ConjunctionGate,
            ParameterResolution,
            ResolveResult,
            SchemaValidator,
            SpanCallResolver,
        )
        res = ResolveResult(hit=False, abstain_reason="test")
        self.assertFalse(res.hit)

    def test_imports_agent_subpackage(self):
        from spancall.agent import (
            BaseAgentLoop,
            ExecutionTrace,
            IntegrationVariant,
            RouterMode,
            RoutingDecision,
            StepTrace,
            ToolCallRouter,
        )
        self.assertEqual(RouterMode.ROUTER, "router")
        self.assertEqual(RoutingDecision.SPANCALL_HIT, "spancall_hit")

    def test_imports_eval_subpackage(self):
        from spancall.eval import (
            AgentMetrics,
            BaseAgentEvaluator,
            BaseToolCallEvaluator,
            ComparativeEvaluator,
            ComparativeStudyResult,
            CostMetrics,
            CostModel,
            ToolCallMetrics,
        )
        cm = CostModel(cost_spancall_call=0.07, cost_llm_call=1.0)
        break_even_h = cm.compute_break_even_hit_rate(precision_at_hit=0.95)
        # Expected break-even around 7-8% per docs/mechanism.md §1
        self.assertAlmostEqual(break_even_h, 0.07 / (1.0 - 0.05 * 2.0), delta=0.01)

    def test_imports_data_sources(self):
        from spancall.data.sources import (
            BFCLLoader,
            GlaiveLoader,
            TauBenchLoader,
            ToolBenchLoader,
            XLAMLoader,
        )
        xlam = XLAMLoader("dummy/path")
        self.assertEqual(xlam.data_path_or_hub_id, "dummy/path")
        tau = TauBenchLoader(domain="airline")
        self.assertEqual(tau.domain, "airline")

    def test_router_defaults(self):
        from spancall.agent.router import ToolCallRouter, RouterMode, IntegrationVariant
        from spancall.infer.resolver import SpanCallResolver
        from spancall.agent.backends.base import BaseLLMBackend
        from unittest.mock import MagicMock

        router = ToolCallRouter(
            resolver=MagicMock(spec=SpanCallResolver),
            llm_backend=MagicMock(spec=BaseLLMBackend),
        )
        self.assertEqual(router.mode, RouterMode.ROUTER)
        self.assertEqual(router.variant, IntegrationVariant.S_SKIP)


if __name__ == "__main__":
    unittest.main()
