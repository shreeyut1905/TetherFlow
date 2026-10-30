# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tetherflow.methods.distribution_matching.feature_moments import FeatureMomentMatching, moment_loss_input_grad


def test_loss_is_the_diagonal_gaussian_w2_of_the_moments():
    torch.manual_seed(0)
    gen, ref = torch.randn(6, 4) * 2 + 1, torch.randn(5, 4)
    loss, metrics = FeatureMomentMatching(queue_size=0, eps=0.0).loss(gen, ref)
    mean_term = (gen.mean(0) - ref.mean(0)).pow(2).sum()
    std_term = (gen.var(0, unbiased=False).sqrt() - ref.var(0, unbiased=False).sqrt()).pow(2).sum()
    torch.testing.assert_close(loss, mean_term + std_term)
    torch.testing.assert_close(metrics["moment_mean_term"], mean_term)


def test_queues_join_later_moments_detached_and_stay_bounded():
    matcher = FeatureMomentMatching(queue_size=3)
    first = torch.ones(2, 2, requires_grad=True)
    matcher.loss(first, torch.zeros(2, 2))
    assert matcher.gen_queue.shape == (2, 2) and not matcher.gen_queue.requires_grad
    second = torch.full((2, 2), 3.0, requires_grad=True)
    loss, _ = matcher.loss(second, torch.zeros(2, 2))
    # The mean of [3, 3, 1, 1] -> 2 per coordinate; only the current batch carries gradient.
    loss.backward()
    assert first.grad is None and second.grad is not None
    assert matcher.gen_queue.shape == (3, 2)  # bounded FIFO, newest first
    torch.testing.assert_close(matcher.gen_queue[:2], torch.full((2, 2), 3.0))


def test_identical_distributions_give_zero_loss():
    features = torch.randn(8, 5)
    loss, _ = FeatureMomentMatching(queue_size=0).loss(features.clone(), features.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


class _CheckpointedNet(torch.nn.Module):
    """A feature extractor whose forward is recomputed in the backward, as with activation checkpointing."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))

    def forward(self, x):
        return checkpoint(lambda y: torch.tanh(y @ self.weight), x, use_reentrant=False)


def test_input_gradient_uses_the_feature_weights_in_the_recomputed_backward():
    torch.manual_seed(0)
    net = _CheckpointedNet()
    ema_weight = torch.tensor([[0.5, 0.1], [-0.2, 0.7]])
    samples, refs = torch.randn(6, 2), torch.randn(6, 2)

    loss, grad, _ = moment_loss_input_grad(
        samples, refs, lambda x: net(x), FeatureMomentMatching(0), net, {"weight": ema_weight}
    )

    # Reference: a plain copy of the net holding the feature weights.
    reference_net = _CheckpointedNet()
    with torch.no_grad():
        reference_net.weight.copy_(ema_weight)
    inputs = samples.clone().requires_grad_(True)
    ref_loss, _ = FeatureMomentMatching(0).loss(reference_net(inputs), reference_net(refs).detach())
    (ref_grad,) = torch.autograd.grad(ref_loss, inputs)
    torch.testing.assert_close(loss, ref_loss.detach())
    torch.testing.assert_close(grad, ref_grad)
    # The live weights are untouched and received no gradient.
    torch.testing.assert_close(net.weight.detach(), torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    assert net.weight.grad is None


def test_surrogate_carries_the_loss_value_and_its_input_gradient():
    """``loss + (s * g).sum() - (s.detach() * g).sum()``, the form DMD2 adds, has the loss's value and gradient g."""
    params = torch.randn(3, requires_grad=True)
    samples = params * 2.0
    grad = torch.tensor([0.5, -1.0, 2.0])
    loss = torch.tensor(1.25)
    surrogate = loss + (samples * grad).sum() - (samples.detach() * grad).sum()
    assert surrogate.item() == pytest.approx(1.25)
    surrogate.backward()
    torch.testing.assert_close(params.grad, 2.0 * grad)
