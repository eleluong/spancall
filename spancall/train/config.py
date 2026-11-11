"""Training and calibration configuration dataclasses.

Design rule 4: an experiment = YAML + seed. ``TrainingConfig.from_yaml`` loads
the YAML sections (``model``, ``data``, ``optimization``, ``training``,
``experiment``, ``calibration``, ``gate``) into the dataclasses below; unknown
keys anywhere are tolerated so configs stay forward-compatible. See
docs/experiments.md and docs/architecture.md for the section mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar, Union

_T = TypeVar("_T")


def _known_kwargs(cls: Type[_T], d: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only dict keys that name a dataclass field (unknown keys ignored)."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in dict(d).items() if k in names}


@dataclass
class HeadsConfig:
    """Which prediction heads are constructed (docs/remediation_plan.md D4).

    v1 minim head set: H1 (tool) + H3 (pointer) + H4 (abstain_reason) + H8
    (correctness) + the call-accept head. H2/H5/H6/H7 stay **spec-only** and are
    constructed only when the flag is true (and the code path exists).
    """

    tool: bool = True            # H1 — candidate tool scoring (no-op if 1 tool)
    count: bool = False          # H2 — call-count (spec-only; gates multi-call)
    pointer: bool = True         # H3 — boundary-factorized span pointer
    abstain_reason: bool = True  # H4 — abstain-reason classes
    source: bool = False         # H5 — source head (spec-only)
    list_size: bool = False      # H6 — list-size head (spec-only)
    swap: bool = False           # H7 — swap/exclusivity head (spec-only)
    correctness: bool = True     # H8 — learned confidence

    def enabled(self) -> Tuple[str, ...]:
        """Return names of heads enabled by this config (H flags)."""
        return tuple(k for k, v in vars(self).items() if v)

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> "HeadsConfig":
        """Build from the YAML ``model.heads`` section; missing keys keep defaults."""
        return cls(**{k: bool(v) for k, v in _known_kwargs(cls, d).items()})


@dataclass
class ModelConfig:
    # Phase 2 backbone (long_context_fix.md): Qwen3-0.6B embedding model with a
    # 32k window, run bidirectional (encoder is_causal=False). max_seq_len is the
    # resolver/trainer guard derived from the backbone — no hardcoded 512 cap.
    backbone_name: str = "Qwen/Qwen3-Embedding-0.6B"
    head_dim: int = 256
    max_seq_len: int = 32768
    max_span_width: int = 16
    dropout: float = 0.1
    heads: HeadsConfig = field(default_factory=HeadsConfig)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelConfig":
        """Build from the YAML ``model`` section; missing keys keep defaults."""
        known = _known_kwargs(cls, d)
        heads = known.pop("heads", None)
        cfg = cls(**known) if known else cls()
        if isinstance(heads, dict):
            cfg.heads = HeadsConfig.from_mapping(heads)
        return cfg


@dataclass
class OptimizationConfig:
    learning_rate_encoder: float = 3e-5
    learning_rate_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    layerwise_decay: float = 0.90
    max_grad_norm: float = 1.0
    batch_size: int = 64
    num_epochs: int = 4
    mixed_precision: str = "bf16"
    abstain_weight: float = 1.0   # B2/H-T2: abstain column loss weight (0.5 de-collapses)
    max_pos: int = 3              # T3: positive slots per param (1 = top-1 only)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OptimizationConfig":
        """Build from the YAML ``optimization`` section; missing keys keep defaults."""
        return cls(**_known_kwargs(cls, d))


@dataclass
class CalibrationConfig:
    target_precision: float = 0.95
    confidence_level: float = 0.95  # Clopper-Pearson confidence bound
    temperature_scaling: bool = True
    param_margin_min: float = 0.0
    joint_margin_min: float = 0.0
    mode: str = "single_tau"  # single_tau | conjunction — mirrors GateConfig

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibrationConfig":
        """Build from the YAML ``calibration`` section; missing keys keep defaults."""
        return cls(**_known_kwargs(cls, d))


@dataclass
class GateConfig:
    """Whole-call gate thresholds and mode (docs/remediation_plan.md D4).

    ``mode`` selects the gate implementation in infer/gate.py:
      - ``single_tau`` (v1 default): hit <=> min_k p_k >= tau_p AND accept >= tau_a
        AND validate(). The joint-margin (tau_delta) is dropped from the v1
        decision — it becomes a calibration signal, and the 8-feature conjunction
        is kept as an ablation (``conjunction``).
      - ``conjunction``: the calibrated conjunction baseline (8-feature).
    """

    mode: str = "single_tau"     # single_tau | conjunction
    tau_p: float = 0.90
    tau_delta: float = 2.0       # joint margin — used only in conjunction mode
    tau_a: float = 0.85

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GateConfig":
        """Build from the YAML ``gate`` section; missing keys keep defaults."""
        return cls(**_known_kwargs(cls, d))


@dataclass
class RecentConfig:
    k_recent: int = 32
    context_budget_tokens: int = 16384

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RecentConfig":
        return cls(**_known_kwargs(cls, d))


@dataclass
class DataConfig:
    """Data-pipeline knobs (docs/remediation_plan.md D5).

    ``pack_tools`` is the FLOOR on the number of candidate tools rendered into
    one input (``build_input_multi``, T anchor blocks). The Lean pipeline always
    uses the multi-tool grammar (Phase A / R6, R10): there is no single-tool
    ``build_input`` path left to fall back to. Every native tool of the sample is
    always rendered (Phase B / R0.1) and ``ToolPool`` only tops up to the floor
    with distractors, so the realised tool count is
    ``max(row_floor, 1 + #extra_native_tools)`` and can exceed ``pack_tools``.

    ``jitter_pack_tools`` (Phase G / R1c) randomises that per-row floor. The draw
    is uniform over ``[jitter_min_tools, pack_tools]`` **inclusive** — i.e.
    ``pack_tools`` is the *ceiling of the draw*, not its floor, and
    ``jitter_min_tools`` (default 1) is the lower bound. With the defaults this
    makes ``T=1`` (which the oracle-tool eval renders) reachable for every row,
    so that layout stays in-distribution during training. The interval is drawn
    per row and is deterministic in ``(experiment.seed, sample_id)``; only an
    *interval* is supported — there is no set-valued sampler.

    Unknown keys in the YAML ``data`` section are tolerated (forward compat).

    ``fraction`` (default 1.0): the fraction of each tensorized split used by
    the trainer. Applied deterministically (seeded by ``experiment.seed``) in
    :meth:`spancall.train.trainer.SpanCallTrainer.load_rows`, so it scales
    **training and dev/calib evaluation together** — set ``data.fraction: 0.2``
    to smoke-test the whole pipeline end-to-end on 20% of the data (see
    ``configs/verify20.yaml``).

    ``max_calls`` / ``call_min_prob`` / ``call_min_prob_ratio`` (Phase J5 /
    R0.4) are forwarded to :class:`spancall.lean.resolver.LeanResolver`. They
    control how many calls the resolver may emit: a resolver built with the
    default ``max_calls=1`` can NEVER report more than one predicted call, so
    any multi-call metric computed against it (``call_exact_set_match``,
    ``pred_n_calls_hist``) is capped at the single-call share by construction,
    not by model quality. Set these from the YAML so a run log records the
    configuration it actually ran under.
    """

    pack_tools: int = 1
    fraction: float = 1.0
    train_f_ratio: float = 0.0   # H-T2: target F:(A) ratio in trainer subsample
                                 # (0 = old pure-random fraction subsample)
    # R1c: per-row floor drawn uniformly from [jitter_min_tools, pack_tools]
    # inclusive; pack_tools is the CEILING of that draw (see class docstring).
    jitter_pack_tools: bool = True
    max_tools: Optional[int] = None  # R0.1: hard ceiling on rendered tools
    max_calls: int = 4               # R0.4: resolver call budget (1 = single call)
    call_min_prob: float = 0.25      # R0.4: H1 prob floor for an extra call
    call_min_prob_ratio: float = 0.5  # R0.4: extra-call prob / top tool prob
    recent: RecentConfig = field(default_factory=RecentConfig)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DataConfig":
        """Build from the YAML ``data`` section; missing keys keep defaults."""
        known = _known_kwargs(cls, d)
        recent = known.pop("recent", None)
        cfg = cls(**known) if known else cls()
        if isinstance(recent, dict):
            cfg.recent = RecentConfig.from_dict(recent)
        return cfg


@dataclass
class AgentConfig:
    """Agent runtime configuration."""
    mode: str = "router"
    integration: str = "s_skip"
    llm_model: str = "google/gemma-4"
    max_turns: int = 15
    max_context_tokens: int = 32768
    keep_last_tool_turns: int = 8

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AgentConfig":
        return cls(**_known_kwargs(cls, d))


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    gradient_accumulation_steps: int = 1
    seed: int = 42
    output_dir: str = "results"
    checkpoint_every_n_steps: int = 500
    experiment_name: str = "span_call"
    data_dir: str = "data/processed"
    # ---- VRAM levers (notebook-friendly; all default OFF so existing
    # ---- configs/behavior are unchanged unless explicitly enabled) ----------
    gradient_checkpointing: bool = False   # backbone.gradient_checkpointing_enable
    freeze_encoder_stages: List[str] = field(default_factory=list)  # e.g. ["T0", "T1"]
    use_8bit_optimizer: bool = False       # bitsandbytes PagedAdamW8bit (falls back if absent)
    stop_on_gate_kill: bool = False        # abort the curriculum after T1 when the E1 gate says KILL (<0.80)
    eval_splits: List[str] = field(default_factory=list)  # A6: per-epoch held-out evaluation splits
    eval_limit: int = 500                  # row cap per eval split to keep epoch wall-time bounded
    resolver_eval_split: str = ""          # E2E-hook: raw-mode resolver eval split ("" = disabled)
    resolver_eval_limit: int = 40          # row cap for the resolver e2e hook
    resolver_factory: Any = None           # callable(out_ckpt_dir) -> SpanCallResolver (injected)
    # ---- LoRA ablation (T10; default OFF so nothing changes unless asked) -----
    # Motivation (investigation §10.3): at the 0.2 data fraction only the two
    # heads were effectively being trained and the frozen encoder could never
    # re-align its representation to the pointer vocabulary. LoRA adapters give
    # the backbone a small, cheap, low-rank way to move. Requires ``peft``; when
    # it is missing, enabling this must FAIL LOUDLY rather than silently
    # train a full model (that is the `freeze_encoder_stages` bug all over again).
    use_lora: bool = False
    lora_r: int = 16                       # rank of the low-rank update
    lora_alpha: int = 32                   # LoRA scaling numerator (alpha / r)
    lora_dropout: float = 0.05
    # Adapters usually want a much higher LR than a full fine-tune of the same
    # block, so this gets its OWN optimizer group rather than sharing lr_encoder.
    lora_lr: float = 1e-3
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load a full training config from YAML (design rule 4:  experiment = YAML + seed).

        Section mapping:
          - ``model``        -> ModelConfig (heads via ``model.heads``)
          - ``data``         -> DataConfig (pack_tools via ``data.pack_tools``)
          - ``agent``        -> AgentConfig
          - ``optimization`` -> OptimizationConfig
          - ``training``     -> gradient_accumulation_steps / checkpoint_every_n_steps
          - ``experiment``   -> seed / output_dir
          - ``calibration``  -> CalibrationConfig
          - ``gate``         -> GateConfig (mode via ``gate.mode``)

        Unknown keys anywhere are ignored; missing keys keep the dataclass
        defaults. Uses ``dataclasses.replace`` over a default instance so a
        missing section degrades to the documented defaults, not an error.
        """
        import yaml

        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}

        training = data.get("training") or {}
        experiment = data.get("experiment") or {}
        cfg = cls()
        return replace(
            cfg,
            model=ModelConfig.from_dict(data.get("model") or {}),
            data=DataConfig.from_dict(data.get("data") or {}),
            agent=AgentConfig.from_dict(data.get("agent") or {}),
            optimization=OptimizationConfig.from_dict(data.get("optimization") or {}),
            calibration=CalibrationConfig.from_dict(data.get("calibration") or {}),
            gate=GateConfig.from_dict(data.get("gate") or {}),
            gradient_accumulation_steps=training.get(
                "gradient_accumulation_steps", cfg.gradient_accumulation_steps
            ),
            checkpoint_every_n_steps=training.get(
                "checkpoint_every_n_steps", cfg.checkpoint_every_n_steps
            ),
            gradient_checkpointing=bool(
                training.get("gradient_checkpointing", cfg.gradient_checkpointing)
            ),
            freeze_encoder_stages=list(
                training.get("freeze_encoder_stages", cfg.freeze_encoder_stages) or []
            ),
            use_8bit_optimizer=bool(
                training.get("use_8bit_optimizer", cfg.use_8bit_optimizer)
            ),
            stop_on_gate_kill=bool(
                training.get("stop_on_gate_kill", cfg.stop_on_gate_kill)
            ),
            eval_splits=list(
                training.get("eval_splits")
                or data.get("eval_splits")
                or experiment.get("eval_splits")
                or cfg.eval_splits
            ),
            eval_limit=int(training.get("eval_limit", cfg.eval_limit)),
            resolver_eval_split=str(
                training.get("resolver_eval_split", cfg.resolver_eval_split) or ""
            ),
            resolver_eval_limit=int(
                training.get("resolver_eval_limit", cfg.resolver_eval_limit)
            ),
            resolver_factory=cfg.resolver_factory,
            use_lora=bool(training.get("use_lora", cfg.use_lora)),
            lora_r=int(training.get("lora_r", cfg.lora_r)),
            lora_alpha=int(training.get("lora_alpha", cfg.lora_alpha)),
            lora_dropout=float(training.get("lora_dropout", cfg.lora_dropout)),
            lora_lr=float(training.get("lora_lr", cfg.lora_lr)),
            lora_target_modules=list(
                training.get("lora_target_modules", cfg.lora_target_modules) or []
            ),
            seed=experiment.get("seed", cfg.seed),
            output_dir=experiment.get("output_dir", cfg.output_dir),
            experiment_name=experiment.get("name", cfg.experiment_name),
            data_dir=experiment.get("data_dir", cfg.data_dir) or cfg.data_dir,
        )
