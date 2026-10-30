# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TetherFlow: its configs (both stages) and ``TetherFlowModel`` (Stage 2).

The model tests run on the tiny EDM backbone with ``r_timestep=True`` and ``schedule_type=rf``, like the AnyFlow
on-policy tests, so they execute on CPU without pretrained weights.
"""

import gc
from types import SimpleNamespace

import attrs
import pytest
import torch

import tetherflow.configs.methods.config_tetherflow as config_tetherflow
import tetherflow.methods.distribution_matching.tetherflow as tetherflow_module
from tetherflow.configs.config_utils import override_config_with_opts
from tetherflow.methods import TetherFlowModel, MeanFlowModel
from tetherflow.methods.distribution_matching.feature_moments import FeatureMomentMatching


def _build_model(weight=0.5, feature_block=0, feature_t=0.8, grad_balance=True, seed=0):
    """TetherFlow Stage 2 on the tiny EDM backbone (img_resolution 8, whose block 0 has 128 channels)."""
    gc.collect()
    torch.manual_seed(seed)
    instance = config_tetherflow.ModelConfig()
    base_opts = [
        "-",
        "backbone.img_resolution=8",
        "backbone.channel_mult=[1]",
        "backbone.channel_mult_noise=1",
        "+schedule_type=rf",
    ]
    instance.teacher = override_config_with_opts(config_tetherflow.ModelConfig().net, list(base_opts))
    instance.net = override_config_with_opts(instance.net, base_opts[:1] + ["backbone.r_timestep=True"] + base_opts[1:])
    instance.discriminator = None
    instance.gan_loss_weight_gen = 0.0
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.precision_model = "float64" if instance.device.type == "cpu" else "bfloat16"
    instance.pretrained_model_path = None
    instance.student_update_freq = 2
    instance.student_grid_cfg.steps = 2
    instance.input_shape = [3, 8, 8]

    # The kinematic co-trained objective, with a coarser finite-difference step for the tiny float64 net.
    config_tetherflow.apply_kinematic_objective(instance, instance.cotrain_sample_t_cfg)
    instance.precision_amp_jvp = None
    instance.loss_config.jvp_finite_diff_eps = 1e-2
    instance.cotrain_sample_t_cfg.min_t = 0.001
    instance.cotrain_sample_t_cfg.max_t = 0.999

    instance.moment_matching.weight = weight
    instance.moment_matching.feature_block = feature_block
    instance.moment_matching.feature_t = feature_t
    instance.moment_matching.grad_balance = grad_balance
    instance.moment_matching.queue_size = 4

    model = TetherFlowModel(instance)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _make_data(model, batch_size=2):
    labels = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10)
    return {
        "idx": list(range(batch_size)),
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": torch.zeros(batch_size, 10).to(model.device, model.precision),
    }


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def test_stage1_config_is_meanflow_with_the_kinematic_objective():
    config = config_tetherflow.create_stage1_config()
    assert config.model_class._target_ is MeanFlowModel
    loss = config.model.loss_config
    assert loss.jvp_tangent == "model"  # improved MeanFlow
    assert (loss.loss_type, loss.weight_type, loss.norm_method) == ("l2", "beta08", None)
    assert loss.use_jvp_finite_diff and loss.jvp_finite_diff_eps == 5e-3 and loss.rebalance_to_flow_matching
    assert config.model.precision_amp_jvp == "float32"
    sample_t = config.model.sample_t_cfg
    assert (sample_t.flow_matching_ratio, sample_t.consistency_ratio, sample_t.deterministic_buckets) == (
        0.5,
        0.25,
        True,
    )


def test_stage2_config_is_tetherflow_with_anyflows_recipe_and_moment_matching():
    config = config_tetherflow.create_config()
    assert config.model_class._target_ is TetherFlowModel
    model = config.model
    assert model.gan_loss_weight_gen == 0.0 and model.discriminator is None
    assert "grad_clip_discriminator" not in config.trainer.callbacks
    assert model.student_update_freq == 2 and model.cotrain_pretrain_weight == 0.5
    assert model.student_steps_list == [2, 4, 8, 16, 50] and model.max_steps_rollout == 3
    # DMD's noising time is not the co-trained loss's (t, r) density.
    assert (model.sample_t_cfg.min_t, model.sample_t_cfg.max_t) == (0.001, 0.999)
    assert model.cotrain_sample_t_cfg.consistency_ratio == 0.25
    moments = model.moment_matching
    assert (moments.weight, moments.grad_balance, moments.feature_t, moments.feature_block) == (0.5, True, 0.8, None)


def test_both_stages_train_the_same_kinematic_objective():
    stage1, stage2 = config_tetherflow.create_stage1_config().model, config_tetherflow.create_config().model
    assert attrs.asdict(stage1.loss_config) == attrs.asdict(stage2.loss_config)
    assert stage1.precision_amp_jvp == stage2.precision_amp_jvp
    assert attrs.asdict(stage1.sample_t_cfg) == attrs.asdict(stage2.cotrain_sample_t_cfg)


def test_set_time_shift_reaches_every_density_and_the_grid():
    stage1, stage2 = config_tetherflow.create_stage1_config().model, config_tetherflow.create_config().model
    for model in (stage1, stage2):
        config_tetherflow.set_time_shift(model, 1.5)
        assert model.sample_t_cfg.shift == 1.5
        assert model.student_grid_cfg.grid_kwargs["shift"] == 1.5
    assert stage2.cotrain_sample_t_cfg.shift == 1.5


# ---------------------------------------------------------------------------
# TetherFlowModel
# ---------------------------------------------------------------------------


def test_student_update_adds_the_balanced_moment_term():
    model = _build_model()
    loss_map, _ = model.single_train_step(_make_data(model), 0)  # iteration 0 -> student update
    assert {"vsd_loss", "cotrain_loss", "moment_loss", "moment_weight", "moment_grad_norm", "vsd_grad_norm"} <= set(
        loss_map
    )
    assert {"moment_mean_term", "moment_std_term", "moment_queue_fill"} <= set(loss_map)
    torch.testing.assert_close(
        loss_map["moment_weight"],
        (0.5 * loss_map["vsd_grad_norm"] / (loss_map["moment_grad_norm"] + 1e-12)).to(loss_map["moment_weight"]),
    )
    assert torch.isfinite(loss_map["total_loss"])
    loss_map["total_loss"].backward()
    assert any(param.grad is not None for param in model.net.parameters())


def test_fake_score_update_has_no_moment_term():
    model = _build_model()
    loss_map, _ = model.single_train_step(_make_data(model), 1)  # odd iteration -> fake-score update
    assert "fake_score_loss" in loss_map and "moment_loss" not in loss_map


def test_moment_term_leaves_the_runs_other_random_draws_unchanged():
    """With and without the moment term, the run draws the same noise, so every other loss is identical."""
    losses = {}
    for weight in (0.0, 0.5):
        model = _build_model(weight=weight)
        torch.manual_seed(1)
        data = _make_data(model)
        loss_map, _ = model.single_train_step(data, 0)
        losses[weight] = loss_map
    assert "moment_loss" not in losses[0.0]
    for key in ("vsd_loss", "cotrain_loss", "student_steps"):
        torch.testing.assert_close(losses[0.0][key], losses[0.5][key])


def test_zero_weight_builds_no_matcher():
    model = _build_model(weight=0.0, feature_block=None)
    assert model._moment_matcher is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"feature_block": None}, "feature_block"),
        ({"feature_t": 0.0}, "feature_t"),
        ({"feature_t": 1.5}, "feature_t"),
        ({"weight": -1.0}, "weight"),
    ],
)
def test_invalid_moment_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _build_model(**kwargs)


def _bare_model(weight=0.5, grad_balance=True):
    model = object.__new__(TetherFlowModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.config = SimpleNamespace(moment_matching=SimpleNamespace(weight=weight, grad_balance=grad_balance))
    model._moment_matcher = FeatureMomentMatching(0)
    model.net = torch.nn.Linear(2, 2)  # handed to the (stubbed) gradient helper only
    model.use_ema = False
    return model


def test_balanced_weight_sets_the_gradient_ratio_to_the_vsd_term(monkeypatch):
    """With ``grad_balance`` the moment gradient at the generations is ``weight`` times the VSD gradient in norm, and
    the term's value is the weighted moment loss."""
    model = _bare_model()
    moment_grad = torch.tensor([[3.0, 4.0]])  # norm 5
    monkeypatch.setattr(tetherflow_module, "moment_loss_input_grad", lambda *args: (torch.tensor(7.0), moment_grad, {}))
    monkeypatch.setattr(model, "_moment_feature_parameters", lambda: {})

    samples = torch.zeros(1, 2, requires_grad=True)
    vsd_loss = (samples * torch.tensor([[0.0, 2.0]])).sum()  # gradient norm 2 at the samples
    metrics, term = model._generator_regularizers(
        SimpleNamespace(data=samples), {"real": torch.zeros(1, 2)}, condition=None, vsd_loss=vsd_loss
    )
    term.backward()
    assert samples.grad.norm().item() == pytest.approx(0.5 * 2.0)
    assert term.item() == pytest.approx(7.0 * 0.5 * 2.0 / 5.0)
    assert metrics["moment_loss"].item() == pytest.approx(7.0)


def test_unbalanced_weight_is_the_raw_weight(monkeypatch):
    model = _bare_model(weight=0.25, grad_balance=False)
    monkeypatch.setattr(
        tetherflow_module, "moment_loss_input_grad", lambda *args: (torch.tensor(2.0), torch.tensor([[1.0, 0.0]]), {})
    )
    monkeypatch.setattr(model, "_moment_feature_parameters", lambda: {})
    samples = torch.zeros(1, 2, requires_grad=True)
    metrics, term = model._generator_regularizers(
        SimpleNamespace(data=samples), {"real": torch.zeros(1, 2)}, condition=None, vsd_loss=None
    )
    term.backward()
    torch.testing.assert_close(samples.grad, torch.tensor([[0.25, 0.0]]))
    assert "vsd_grad_norm" not in metrics and metrics["moment_weight"].item() == pytest.approx(0.25)


def test_moment_term_requires_real_samples():
    model = _bare_model()
    with pytest.raises(ValueError, match="real"):
        model._generator_regularizers(SimpleNamespace(data=torch.zeros(1, 2)), {}, condition=None, vsd_loss=None)


def test_feature_weights_prefer_the_network_ema_then_the_adapter_ema(monkeypatch):
    model = _bare_model()
    live = {name: param for name, param in model.net.named_parameters()}

    # Without an EMA: the live trainable weights, detached, with a single warning.
    warnings = []
    monkeypatch.setattr(tetherflow_module.logger, "warning", warnings.append)
    first, second = model._moment_feature_parameters(), model._moment_feature_parameters()
    assert set(first) == set(live) and not any(t.requires_grad for t in first.values())
    assert torch.equal(first["weight"], live["weight"]) and len(warnings) == 1 and set(second) == set(live)

    # A LoRA student's adapter EMA, registered by AdapterEMACallback.
    adapter_ema = {"weight": torch.ones(2, 2)}
    model.adapter_ema = SimpleNamespace(ema_parameters=lambda: adapter_ema)
    assert model._moment_feature_parameters() is adapter_ema

    # TetherFlow's network EMA wins.
    model.ema = torch.nn.Linear(2, 2)
    model.use_ema = ["ema"]
    ema = model._moment_feature_parameters()
    assert torch.equal(ema["weight"], model.ema.weight) and not ema["weight"].requires_grad
