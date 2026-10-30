# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from tetherflow.callbacks.flow_map_dashboard import FlowMapDashboardCallback
from tetherflow.configs.config_utils import override_config_with_opts
from tetherflow.configs.methods.config_mean_flow import create_config
from tetherflow.methods import MeanFlowModel


def _callback(tmp_path, **kwargs):
    callback = FlowMapDashboardCallback(prompts=["a fox", "a cup"], prompt_tags=["train", "held-out"], **kwargs)
    callback.config = SimpleNamespace(
        log_config=SimpleNamespace(save_path=str(tmp_path), name="run", group="group"),
        trainer=SimpleNamespace(max_iter=6),
    )
    return callback


def _losses(total, buckets):
    loss = {"total_loss": torch.tensor(total)}
    for name, (residual, count) in buckets.items():
        loss[f"bucket_residual/{name}"] = torch.tensor(residual)
        loss[f"bucket_count/{name}"] = torch.tensor(float(count))
    return loss


def test_metrics_combine_accumulation_rounds_by_bucket_count(tmp_path, monkeypatch):
    callback = _callback(tmp_path, eval_every=3)
    evals = []
    monkeypatch.setattr(callback, "_evaluate", lambda model, iteration: evals.append(iteration))
    model = SimpleNamespace(scheduler_dict={})

    callback.on_train_begin(model, iteration=0)
    callback.on_backward_begin(model, {}, {}, _losses(1.0, {"r_eq_t": (0.5, 2), "r_eq_0": (2.0, 1)}), 1, 0)
    callback.on_backward_begin(model, {}, {}, _losses(3.0, {"r_eq_t": (1.5, 6), "r_eq_0": (4.0, 0)}), 1, 1)
    callback.on_training_step_end(model, {}, {}, {"total_loss": torch.tensor(3.0)}, iteration=1)
    for iteration in (2, 3):
        callback.on_backward_begin(model, {}, {}, _losses(1.0, {}), iteration, 0)
        callback.on_training_step_end(model, {}, {}, {"total_loss": torch.tensor(1.0)}, iteration=iteration)

    rows = [json.loads(line) for line in (tmp_path / "dashboard" / "metrics.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [1, 2, 3]
    assert rows[0]["loss/total"] == pytest.approx(2.0)
    assert rows[0]["loss/r_eq_t"] == pytest.approx((0.5 * 2 + 1.5 * 6) / 8)
    assert rows[0]["loss/r_eq_0"] == pytest.approx(2.0)
    assert rows[0]["count/r_eq_t"] == 8
    # Iterations without bucket metrics log only the total.
    assert "loss/r_eq_t" not in rows[1]
    assert evals == [0, 3]

    info = json.loads((tmp_path / "dashboard" / "run.json").read_text())
    assert info["label"] == "run" and info["max_iter"] == 6


def test_extra_metrics_average_over_the_rounds_that_logged_them_and_charts_reach_run_json(tmp_path, monkeypatch):
    charts = [{"key": "loss/vsd", "title": "VSD", "sub": "student"}]
    callback = _callback(
        tmp_path,
        eval_at_start=False,
        bucket_prefix="cotrain_",
        log_keys={"loss/vsd": "vsd_loss", "loss/fake": "fake_score_loss"},
        charts=charts,
    )
    monkeypatch.setattr(callback, "_evaluate", lambda model, iteration: None)
    model = SimpleNamespace(scheduler_dict={})
    callback.on_train_begin(model, iteration=0)
    # A student iteration: VSD and prefixed bucket metrics, no fake-score loss.
    student = _losses(1.0, {})
    student.update({"vsd_loss": torch.tensor(0.2), "cotrain_bucket_residual/r_eq_t": torch.tensor(0.5)})
    student["cotrain_bucket_count/r_eq_t"] = torch.tensor(4.0)
    callback.on_backward_begin(model, {}, {}, student, 1, 0)
    callback.on_training_step_end(model, {}, {}, student, iteration=1)
    # A fake-score iteration: only the fake-score loss.
    fake = {"total_loss": torch.tensor(2.0), "fake_score_loss": torch.tensor(0.7)}
    callback.on_backward_begin(model, {}, {}, fake, 2, 0)
    callback.on_training_step_end(model, {}, {}, fake, iteration=2)

    rows = [json.loads(line) for line in (tmp_path / "dashboard" / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["loss/vsd"] == pytest.approx(0.2) and "loss/fake" not in rows[0]
    assert rows[0]["loss/r_eq_t"] == pytest.approx(0.5)
    assert rows[1]["loss/fake"] == pytest.approx(0.7) and "loss/vsd" not in rows[1]
    assert json.loads((tmp_path / "dashboard" / "run.json").read_text())["charts"] == charts


def test_the_final_iteration_always_evaluates(tmp_path, monkeypatch):
    callback = _callback(tmp_path, eval_every=100, eval_at_start=False)
    evals = []
    monkeypatch.setattr(callback, "_evaluate", lambda model, iteration: evals.append(iteration))
    model = SimpleNamespace(scheduler_dict={})
    callback.on_train_begin(model, iteration=0)
    for iteration in range(1, 7):
        callback.on_training_step_end(model, {}, {}, {"total_loss": torch.tensor(1.0)}, iteration=iteration)
    assert evals == [6]


def test_non_finite_losses_are_logged_as_null(tmp_path, monkeypatch):
    callback = _callback(tmp_path, eval_at_start=False)
    monkeypatch.setattr(callback, "_evaluate", lambda model, iteration: None)
    model = SimpleNamespace(scheduler_dict={})
    callback.on_train_begin(model, iteration=0)
    callback.on_training_step_end(model, {}, {}, {"total_loss": torch.tensor(float("nan"))}, iteration=1)
    row = json.loads((tmp_path / "dashboard" / "metrics.jsonl").read_text())
    assert row["loss/total"] is None


@pytest.mark.parametrize("bad", [{"prompts": []}, {"nfes": [0, 2]}, {"eval_every": 0}, {"prompt_tags": ["x"]}])
def test_invalid_arguments_are_rejected(bad):
    kwargs = {"prompts": ["a", "b"], **bad}
    with pytest.raises(ValueError):
        FlowMapDashboardCallback(**kwargs)


def test_eval_writes_samples_and_any_step_consistency(tmp_path):
    """End to end on a tiny pixel-space MeanFlow student: every NFE is sampled from the same noise and saved,
    and the consistency distances are measured against the largest NFE."""
    config = create_config()
    instance = config.model
    opts = ["-", "backbone.img_resolution=2", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts + ["backbone.r_timestep=True"])
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32"
    instance.pretrained_model_path = None
    instance.input_shape = [3, 2, 2]
    model = MeanFlowModel(instance)
    model.on_train_begin()

    callback = _callback(tmp_path, nfes=(1, 2, 4))

    def prepare(model):
        callback._condition = torch.nn.functional.one_hot(torch.tensor([1, 2]), num_classes=10).to(
            model.device, model.precision
        )
        callback._noise = torch.randn(2, 3, 2, 2, generator=torch.Generator().manual_seed(0)).to(model.device)

    callback._prepare_eval_inputs = prepare
    callback.on_train_begin(model, iteration=0)

    eval_dir = tmp_path / "dashboard" / "eval" / "0000000"
    meta = json.loads((eval_dir / "meta.json").read_text())
    assert meta["nfes"] == [1, 2, 4]
    assert set(meta["consistency_mse"]) == {"1", "2"}
    assert all(value >= 0 for value in meta["consistency_mse"].values())
    assert sorted(path.name for path in eval_dir.glob("*.jpg")) == sorted(
        f"p{j}_nfe{k}.jpg" for j in range(2) for k in (1, 2, 4)
    )
    assert model.net.training, "eval must restore the student's train mode"
    assert meta["weights"] == "live"


def test_eval_samples_with_a_registered_adapter_ema(tmp_path):
    """With an adapter EMA registered, eval swaps it in: perturbing the live weights afterwards must not change the
    samples, and the live weights come back unchanged."""
    from tetherflow.callbacks.adapter_ema import AdapterEMACallback

    config = create_config()
    instance = config.model
    opts = ["-", "backbone.img_resolution=2", "backbone.channel_mult=[1]", "backbone.channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts + ["backbone.r_timestep=True"])
    instance.device = torch.device("cpu")
    instance.precision = "float32"
    instance.precision_model = "float32"
    instance.pretrained_model_path = None
    instance.input_shape = [3, 2, 2]
    model = MeanFlowModel(instance)
    model.on_train_begin()
    ema = AdapterEMACallback(beta=0.5)
    ema.on_train_begin(model, iteration=0)

    callback = _callback(tmp_path, nfes=(1, 2))

    def prepare(model):
        callback._condition = torch.nn.functional.one_hot(torch.tensor([1, 2]), num_classes=10).to(model.precision)
        callback._noise = torch.randn(2, 3, 2, 2, generator=torch.Generator().manual_seed(0))

    callback._prepare_eval_inputs = prepare
    callback.on_train_begin(model, iteration=0)
    first = (tmp_path / "dashboard" / "eval" / "0000000" / "p0_nfe1.jpg").read_bytes()

    live = [p.detach().clone() for p in model.net.parameters() if p.requires_grad]
    with torch.no_grad():
        for p in model.net.parameters():
            if p.requires_grad:
                p.add_(1.0)
    perturbed = [p.detach().clone() for p in model.net.parameters() if p.requires_grad]
    callback._evaluate(model, iteration=1)

    meta = json.loads((tmp_path / "dashboard" / "eval" / "0000001" / "meta.json").read_text())
    assert meta["weights"] == "ema"
    # The EMA (still the initial weights) was sampled, not the perturbed live weights.
    assert (tmp_path / "dashboard" / "eval" / "0000001" / "p0_nfe1.jpg").read_bytes() == first
    for after, before in zip((p for p in model.net.parameters() if p.requires_grad), perturbed):
        torch.testing.assert_close(after.detach(), before)
    assert any(not torch.equal(a, b) for a, b in zip(live, perturbed))
