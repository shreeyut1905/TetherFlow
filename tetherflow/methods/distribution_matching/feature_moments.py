# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feature-moment matching from Intrinsic Distribution Matching (IDM, Liu et al. 2026, arXiv:2610.39633).

IDM adds to the DMD correction a loss matching the mean and per-coordinate standard deviation of features of
generated and reference samples, in the feature space of the generator's own EMA copy (mean-pooled last-layer
activations at a high noise level). With the moments taken over the current features and detached queues of
earlier ones, the loss is the squared 2-Wasserstein distance between the two diagonal Gaussians (paper Eq. 13):

    L_pop = ||mu_gen - mu_ref||^2 + sum_j (sqrt(var_gen_j) - sqrt(var_ref_j))^2.

Gradients flow through the current generated features into the generated samples, never into the feature
network's weights. ``moment_loss_input_grad`` computes that gradient with the feature weights in place for the
whole backward: networks with activation checkpointing recompute their forward during the backward, so the weights
must still be the feature weights then.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, Iterator, Optional, Tuple

import torch
from torch.nn.utils.stateless import _reparametrize_module


class FeatureMomentMatching:
    """IDM's population moment loss over the current features and detached FIFO queues of earlier ones."""

    def __init__(self, queue_size: int, eps: float = 1e-6):
        if queue_size < 0:
            raise ValueError(f"queue_size must be >= 0, got {queue_size}")
        self.queue_size = queue_size
        self.eps = eps
        self.gen_queue: Optional[torch.Tensor] = None
        self.ref_queue: Optional[torch.Tensor] = None

    def _with_queue(self, features: torch.Tensor, queue: Optional[torch.Tensor]) -> torch.Tensor:
        return features if queue is None else torch.cat([features, queue.to(features.dtype)], dim=0)

    def _push(self, queue: Optional[torch.Tensor], features: torch.Tensor) -> Optional[torch.Tensor]:
        if self.queue_size == 0:
            return None
        features = features.detach()
        merged = features if queue is None else torch.cat([features, queue], dim=0)
        return merged[: self.queue_size]

    def loss(
        self, gen_features: torch.Tensor, ref_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Moment loss of ``[B, d]`` generated features (with gradient) against ``[B', d]`` reference features.

        Pushes both onto their queues afterwards, so the next call also sees them (detached).
        """
        if gen_features.ndim != 2 or ref_features.ndim != 2 or gen_features.shape[1] != ref_features.shape[1]:
            raise ValueError(
                f"Expected [B, d] features of equal width, got {tuple(gen_features.shape)} and {tuple(ref_features.shape)}"
            )
        ref_features = ref_features.detach()
        gen_all = self._with_queue(gen_features, self.gen_queue)
        ref_all = self._with_queue(ref_features, self.ref_queue)
        mean_gen, mean_ref = gen_all.mean(dim=0), ref_all.mean(dim=0)
        std_gen = (gen_all.var(dim=0, unbiased=False) + self.eps).sqrt()
        std_ref = (ref_all.var(dim=0, unbiased=False) + self.eps).sqrt()
        mean_term = (mean_gen - mean_ref).pow(2).sum()
        std_term = (std_gen - std_ref).pow(2).sum()
        self.gen_queue = self._push(self.gen_queue, gen_features)
        self.ref_queue = self._push(self.ref_queue, ref_features)
        metrics = {
            "moment_mean_term": mean_term.detach(),
            "moment_std_term": std_term.detach(),
            "moment_queue_fill": torch.tensor(float(0 if self.gen_queue is None else len(self.gen_queue))),
        }
        return mean_term + std_term, metrics


@contextlib.contextmanager
def feature_weights(module: torch.nn.Module, parameters: Dict[str, torch.Tensor]) -> Iterator[None]:
    """Run ``module`` with ``parameters`` (names as in ``module.named_parameters()``) in place of its own.

    The module's parameter objects are swapped, never modified in place, so autograd graphs that saved the live
    weights stay valid.
    """
    with _reparametrize_module(module, parameters):
        yield


def moment_loss_input_grad(
    samples: torch.Tensor,
    references: torch.Tensor,
    extract: Callable[[torch.Tensor], torch.Tensor],
    matcher: FeatureMomentMatching,
    module: torch.nn.Module,
    parameters: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """The moment loss of ``samples`` against ``references`` and its gradient with respect to ``samples``.

    ``extract`` maps a sample batch to ``[B, d]`` features through ``module``, which runs with ``parameters``
    during the forward and the backward. Returns ``(loss, d loss / d samples, metrics)``, all detached; adding
    ``(samples * grad).sum()`` to a loss back-propagates exactly this gradient into the samples' own graph.
    """
    inputs = samples.detach().requires_grad_(True)
    with feature_weights(module, parameters), torch.enable_grad():
        with torch.no_grad():
            ref_features = extract(references)
        gen_features = extract(inputs)
        loss, metrics = matcher.loss(gen_features, ref_features)
        (grad,) = torch.autograd.grad(loss, inputs)
    return loss.detach(), grad.detach(), metrics
