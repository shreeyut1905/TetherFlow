# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for TetherFlow (kinematic and statistical tethers for any-step flow maps), both stages, network-agnostic.

TetherFlow is AnyFlow's recipe with one change per stage (see ``tetherflow/methods/distribution_matching/tetherflow.py``):

* ``create_stage1_config``, the kinematic stage. ``MeanFlowModel`` trained with ``apply_kinematic_objective``:
  AnyFlow's flow-map pretraining objective, with improved MeanFlow's self tangent.
* ``create_config``, the statistical stage. ``TetherFlowModel``: AnyFlow's on-policy DMD, co-training the same
  kinematic objective, plus IDM's feature-moment matching (``MomentMatchingConfig``).

An experiment config still has to provide the following:

* The network: a flow-map student with ``r_timestep`` set.
* The time shift, via ``set_time_shift``.
* Guidance handling for the kinematic objective (``guidance_fuse_scale``, ``cond_dropout_prob``).
* For Stage 2: ``moment_matching.feature_block``, the student's EMA settings, and the Stage-1 checkpoint it seeds
  from.
* The data and the schedule.
"""

from typing import Any, Optional

import attrs
from omegaconf import DictConfig

import tetherflow.configs.methods.config_mean_flow as config_mean_flow
from tetherflow.callbacks.grad_clip import GradClipCallback
from tetherflow.configs.callbacks import BASE_CALLBACKS, EMA_CALLBACK, GradClip_CALLBACK
from tetherflow.configs.config import BaseConfig
from tetherflow.configs.instantiate import LazyCall as L
from tetherflow.configs.methods.config_anyflow import ModelConfig as AnyFlowModelConfig
from tetherflow.configs.methods.config_mean_flow import LossConfig, SampleTConfig
from tetherflow.methods import TetherFlowModel


@attrs.define
class MomentMatchingConfig:
    """IDM's feature-moment matching in the Stage-2 student update."""

    # Weight of the moment term; 0 disables it, leaving AnyFlow's on-policy stage with the kinematic objective.
    weight: float = 0.5
    # Read ``weight`` as the ratio of the moment term's gradient norm to the VSD term's at the generations, per rank
    # and step. IDM's weight applies to its own sum-reduced losses. TetherFlow's VSD loss is mean-reduced and normalized,
    # and the moment loss scales with the network's feature magnitudes, so a raw weight transfers to neither.
    grad_balance: bool = True
    # Noise time the generations and real samples are noised to before feature extraction (IDM: a high noise level).
    feature_t: float = 0.8
    # Index of the network block whose output features, mean-pooled over positions, are matched. This is
    # network-specific and required; IDM uses the last block.
    feature_block: Optional[int] = None
    # Detached FIFO queues of earlier features per side and rank that join the moment estimates; 0 uses the current
    # batch only. Not checkpointed: a resumed run refills them.
    queue_size: int = 1024


@attrs.define
class ModelConfig(AnyFlowModelConfig):
    """TetherFlow Stage-2 model config: AnyFlow's on-policy config plus IDM's feature-moment matching."""

    moment_matching: MomentMatchingConfig = attrs.field(factory=MomentMatchingConfig)


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(TetherFlowModel)(config=None))


def apply_kinematic_objective(model: Any, sample_t_cfg: SampleTConfig) -> None:
    """Set TetherFlow's kinematic objective on a MeanFlow or TetherFlow model config.

    This is Stage 1's loss, which Stage 2 co-trains. ``sample_t_cfg`` is the config its (t, r) pairs are drawn from:
    ``model.sample_t_cfg`` in Stage 1, ``model.cotrain_sample_t_cfg`` in Stage 2. The time shift is set separately,
    by ``set_time_shift``.
    """
    loss: LossConfig = model.loss_config
    # AnyFlow's MeanFlow loss: l2 with its fixed beta08 timestep weighting and no adaptive normalization.
    loss.use_cd = False
    loss.loss_type = "l2"
    loss.weight_type = "beta08"
    loss.norm_method = None
    # A central-difference JVP over 5 of 1000 timesteps, as in AnyFlow. FP32 forwards: in bf16 the difference of the
    # two forwards is mostly rounding.
    loss.use_jvp_finite_diff = True
    loss.jvp_finite_diff_eps = 5e-3
    model.precision_amp_jvp = "float32"
    # Rebalance the r < t losses to the global flow-matching (r = t) mean.
    loss.rebalance_to_flow_matching = True
    # Improved MeanFlow: differentiate along the network's own detached instantaneous velocity u(x_t, t, t).
    loss.jvp_tangent = "model"
    # Monitoring: the global residual of each (t, r) bucket below.
    loss.log_bucket_metrics = True

    # (t, r) pairs: shifted uniform, with half of the global batch at r = t and a quarter at r = 0, assigned by a
    # rank-indexed partition, as in AnyFlow.
    sample_t_cfg.time_dist_type = "shifted"
    sample_t_cfg.min_t = 0.0
    sample_t_cfg.max_t = 1.0
    sample_t_cfg.flow_matching_ratio = 0.5
    sample_t_cfg.consistency_ratio = 0.25
    sample_t_cfg.deterministic_buckets = True


def set_time_shift(model: Any, shift: float) -> None:
    """Use one time shift for every time density and grid of either stage.

    AnyFlow does the same with the network's sampling shift. The shift applies to the kinematic (t, r) sampling, to
    Stage 2's DMD and fake-score noising times, and to the student's flow-map grid.
    """
    model.sample_t_cfg.shift = shift
    if hasattr(model, "cotrain_sample_t_cfg"):
        model.cotrain_sample_t_cfg.shift = shift
    model.student_grid_cfg.grid_kwargs["shift"] = shift


def create_stage1_config():
    """TetherFlow Stage 1 (kinematic): ``MeanFlowModel`` with the kinematic objective and AnyFlow's optimization."""
    config = config_mean_flow.create_config()
    apply_kinematic_objective(config.model, config.model.sample_t_cfg)

    # AnyFlow's Stage-1 optimization: AdamW, lr 5e-5, betas (0.9, 0.95), no weight decay, grad clip 1.0, EMA 0.999.
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 5e-5
    config.model.net_optimizer.betas = (0.9, 0.95)
    config.model.net_optimizer.weight_decay = 0.0
    config.trainer.callbacks.grad_clip.grad_norm = 1.0
    config.trainer.callbacks.ema.beta = 0.999

    # The flow map samples with ODE steps on its (shifted) grid.
    config.model.student_sample_type = "ode"
    config.model.student_grid_cfg.steps = 4
    return config


def create_config():
    """TetherFlow Stage 2 (statistical): ``TetherFlowModel``, AnyFlow's on-policy recipe plus IDM's moment matching."""
    config = Config()
    # DMD2's callbacks without the discriminator's grad clip, since no discriminator is trained, plus the EMA.
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            "grad_clip_fake_score": L(GradClipCallback)(grad_norm=None, model_key="fake_score"),
            **EMA_CALLBACK,
            **BASE_CALLBACKS,
        }
    )
    config.model.use_ema = True
    config.model.net.backbone.r_timestep = True
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]

    # AnyFlow's DMD: no adversarial term (the fake score trains by denoising score matching only), and one student
    # and one fake-score update per pair of iterations (AnyFlow updates both every step).
    config.model.gan_loss_weight_gen = 0.0
    config.model.discriminator = None
    config.model.student_update_freq = 2
    # DMD noising time: shifted uniform, kept off the degenerate endpoints. The fake score draws a shifted
    # logit-normal time instead.
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.fake_score_sample_t_overrides = {
        "time_dist_type": "shifted_logitnormal",
        "train_p_mean": 0.0,
        "train_p_std": 1.0,
    }

    # Rollout: an NFE drawn per iteration, each rollout compressed to at most 3 differentiable forwards.
    config.model.student_sample_type = "ode"
    config.model.student_steps_list = [2, 4, 8, 16, 50]
    config.model.max_steps_rollout = 3
    config.model.student_grid_cfg.steps = 4

    # The kinematic objective, co-trained on real data. The weight is 0.5 because TetherFlow's VSD loss carries a 0.5
    # factor that AnyFlow's DMD loss does not, so this keeps AnyFlow's 1:1 gradient ratio.
    config.model.cotrain_pretrain_weight = 0.5
    apply_kinematic_objective(config.model, config.model.cotrain_sample_t_cfg)

    # AnyFlow's Stage-2 optimization for the student and the fake score: AdamW, lr 2e-6, betas (0, 0.999), no
    # weight decay, grad clip 1.0. The EMA decay is 0.99.
    for optimizer in (config.model.net_optimizer, config.model.fake_score_optimizer):
        optimizer.optim_type = "adamw"
        optimizer.lr = 2e-6
        optimizer.betas = (0.0, 0.999)
        optimizer.weight_decay = 0.0
    config.trainer.callbacks.grad_clip.grad_norm = 1.0
    config.trainer.callbacks.grad_clip_fake_score.grad_norm = 1.0
    config.trainer.callbacks.ema.beta = 0.99
    return config
