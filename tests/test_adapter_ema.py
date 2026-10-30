# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tetherflow.callbacks.adapter_ema import AdapterEMACallback


class _LoRALikeNet(torch.nn.Module):
    """A frozen base weight next to two trainable 'adapter' tensors."""

    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.zeros(2, 3))
        self.lora_B = torch.nn.Parameter(torch.zeros(3, 2))


def _model():
    return SimpleNamespace(net=_LoRALikeNet())


def _step(callback, model, iteration):
    callback.on_training_step_end(model, {}, {}, {}, iteration=iteration)


def test_tracks_only_trainable_tensors_and_registers_itself():
    model = _model()
    callback = AdapterEMACallback(beta=0.9, start_iter=0)
    callback.on_train_begin(model, iteration=0)
    assert set(callback.state_dict()["shadow"]) == {"lora_A", "lora_B"}
    assert model.adapter_ema is callback


def test_copies_before_start_iter_and_lerps_after():
    model = _model()
    callback = AdapterEMACallback(beta=0.9, start_iter=3)
    callback.on_train_begin(model, iteration=0)
    with torch.no_grad():
        model.net.lora_A.fill_(1.0)
    _step(callback, model, iteration=2)
    torch.testing.assert_close(callback.state_dict()["shadow"]["lora_A"], torch.ones(2, 3))
    with torch.no_grad():
        model.net.lora_A.fill_(3.0)
    _step(callback, model, iteration=3)
    # 0.9 * 1 + 0.1 * 3
    torch.testing.assert_close(callback.state_dict()["shadow"]["lora_A"], torch.full((2, 3), 1.2))


def test_state_round_trips_through_a_checkpoint_and_wins_over_the_live_weights():
    model = _model()
    callback = AdapterEMACallback(beta=0.5)
    callback.on_train_begin(model, iteration=0)
    with torch.no_grad():
        model.net.lora_B.fill_(4.0)
    _step(callback, model, iteration=1)  # shadow: 2.0
    state = callback.state_dict()

    resumed_model = _model()
    with torch.no_grad():
        resumed_model.net.lora_B.fill_(4.0)
    resumed = AdapterEMACallback(beta=0.5)
    resumed.load_state_dict(state)
    resumed.on_train_begin(resumed_model, iteration=1)
    torch.testing.assert_close(resumed.state_dict()["shadow"]["lora_B"], torch.full((3, 2), 2.0))


def test_resuming_without_ema_state_starts_from_the_resumed_weights():
    model = _model()
    with torch.no_grad():
        model.net.lora_A.fill_(5.0)
    callback = AdapterEMACallback(beta=0.99, start_iter=0)
    callback.load_state_dict({})  # e.g. a checkpoint saved before this callback existed
    callback.on_train_begin(model, iteration=1000)
    torch.testing.assert_close(callback.state_dict()["shadow"]["lora_A"], torch.full((2, 3), 5.0))


def test_mismatched_checkpoint_state_is_rejected():
    callback = AdapterEMACallback()
    callback.load_state_dict({"shadow": {"lora_A": torch.zeros(2, 3)}})
    with pytest.raises(ValueError, match="does not match"):
        callback.on_train_begin(_model(), iteration=0)


def test_swapped_in_exposes_the_ema_and_restores_the_live_weights():
    model = _model()
    callback = AdapterEMACallback(beta=0.5)
    callback.on_train_begin(model, iteration=0)
    with torch.no_grad():
        model.net.lora_A.fill_(2.0)
    _step(callback, model, iteration=1)  # shadow: 1.0
    with callback.swapped_in(model):
        torch.testing.assert_close(model.net.lora_A.detach(), torch.ones(2, 3))
    torch.testing.assert_close(model.net.lora_A.detach(), torch.full((2, 3), 2.0))

    with pytest.raises(RuntimeError, match="boom"):
        with callback.swapped_in(model):
            raise RuntimeError("boom")
    torch.testing.assert_close(model.net.lora_A.detach(), torch.full((2, 3), 2.0))


@pytest.mark.parametrize("beta", [-0.1, 1.0])
def test_invalid_beta_is_rejected(beta):
    with pytest.raises(ValueError):
        AdapterEMACallback(beta=beta)


def _seeding_callback(path, **kwargs):
    callback = AdapterEMACallback(seed_from_pretrained=True, **kwargs)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(checkpointer=SimpleNamespace(pretrained_ckpt_path=path)))
    return callback


def _save_stage1(tmp_path, callbacks):
    path = str(tmp_path / "0001000.pth")
    torch.save({"model": {"net": {}}, "callbacks": callbacks, "iteration": 1000}, path)
    return path


def test_seeds_the_adapters_from_the_previous_stages_ema(tmp_path):
    shadow = {"lora_A": torch.full((2, 3), 3.0), "lora_B": torch.full((3, 2), -1.0)}
    path = _save_stage1(tmp_path, {"adapter_ema": {"beta": 0.997, "shadow": shadow}})
    model = _model()
    callback = _seeding_callback(path, beta=0.99)
    callback.on_load_checkpoint_start(model)
    torch.testing.assert_close(model.net.lora_A.detach(), shadow["lora_A"])
    torch.testing.assert_close(model.net.lora_B.detach(), shadow["lora_B"])
    torch.testing.assert_close(model.net.base.detach(), torch.ones(3))  # outside the adapter: untouched
    # This stage's EMA then starts from the seeded weights.
    callback.on_train_begin(model, iteration=0)
    torch.testing.assert_close(callback.state_dict()["shadow"]["lora_A"], shadow["lora_A"])


def test_seeding_is_off_by_default(tmp_path):
    path = _save_stage1(tmp_path, {"adapter_ema": {"shadow": {"lora_A": torch.ones(2, 3), "lora_B": torch.ones(3, 2)}}})
    model = _model()
    callback = AdapterEMACallback()
    callback.config = SimpleNamespace(trainer=SimpleNamespace(checkpointer=SimpleNamespace(pretrained_ckpt_path=path)))
    callback.on_load_checkpoint_start(model)
    torch.testing.assert_close(model.net.lora_A.detach(), torch.zeros(2, 3))


def test_seeding_reads_the_named_callback_state(tmp_path):
    shadow = {"lora_A": torch.full((2, 3), 2.0), "lora_B": torch.zeros(3, 2)}
    path = _save_stage1(tmp_path, {"stage1_ema": {"shadow": shadow}})
    model = _model()
    _seeding_callback(path, pretrained_name="stage1_ema").on_load_checkpoint_start(model)
    torch.testing.assert_close(model.net.lora_A.detach(), shadow["lora_A"])


@pytest.mark.parametrize(
    "callbacks, message",
    [
        ({"grad_clip": {}}, "no adapter EMA"),  # saved before AdapterEMACallback existed
        ({"adapter_ema": {"shadow": {"lora_A": torch.zeros(2, 3)}}}, "does not match"),
    ],
)
def test_seeding_rejects_a_checkpoint_without_a_matching_ema(tmp_path, callbacks, message):
    path = _save_stage1(tmp_path, callbacks)
    with pytest.raises(ValueError, match=message):
        _seeding_callback(path).on_load_checkpoint_start(_model())


def test_seeding_requires_a_pretrained_checkpoint():
    with pytest.raises(ValueError, match="pretrained_ckpt_path"):
        _seeding_callback("").on_load_checkpoint_start(_model())
