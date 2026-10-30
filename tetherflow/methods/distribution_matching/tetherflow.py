# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TetherFlow: kinematic and statistical tethers for any-step flow maps.

TetherFlow distills a diffusion model into an any-step flow map ``u_theta(x_t, t, r)``, the average velocity from
``t`` to ``r``. It keeps AnyFlow's two stages and adds one tether to each:

1. **Kinematic tether** (Stage 1, flow-map pretraining). AnyFlow's MeanFlow objective with the improved MeanFlow
   (iMF, Geng et al. 2025, arXiv:2512.02012) tangent, which ties the flow map's average velocity to the network's
   own instantaneous velocity ``u_theta(x_t, t, t)``. The total derivative is taken along that velocity, not along
   the per-sample conditional velocity ``eps - x0``, so the regression input no longer carries each sample's noise
   through the tangent (``loss_config.jvp_tangent="model"``). This stage runs on ``MeanFlowModel``; see
   ``configs/methods/config_tetherflow.create_stage1_config``.
2. **Statistical tether** (Stage 2, on-policy distillation, this module). AnyFlow's DMD, in which the student
   generates by a flow-map rollout and also trains the kinematic objective on real data. Added to it is the
   feature-moment matching of Intrinsic Distribution Matching (IDM, Liu et al. 2026, arXiv:2610.39633), which ties
   the per-coordinate mean and standard deviation of the rollout outputs' features to those of real samples, in the
   feature space of the student's own EMA (see ``feature_moments``).

How the moment term enters the student update:

* **Gradient path.** Its gradient reaches the student only through the generations. It is computed with the EMA
  weights in place for the whole backward (``moment_loss_input_grad``), then added to the student loss as a
  surrogate that carries that gradient.
* **Weighting.** With ``moment_matching.grad_balance``, the weight is a ratio of gradient norms to the VSD term at
  the generations. This makes it independent of the two losses' reductions and of the network's feature scale.
* **Randomness.** Its feature noise is drawn on a forked RNG, so enabling the term leaves every other random draw of
  the run unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from tetherflow.methods.distribution_matching.anyflow import AnyFlowModel
from tetherflow.methods.distribution_matching.dmd2 import Generation
from tetherflow.methods.distribution_matching.feature_moments import FeatureMomentMatching, moment_loss_input_grad
import tetherflow.utils.logging_utils as logger


class TetherFlowModel(AnyFlowModel):
    """TetherFlow's statistical stage: AnyFlow's on-policy DMD plus IDM's feature-moment matching.

    The co-trained objective is the kinematic one (``config_tetherflow.apply_kinematic_objective``). With
    ``moment_matching.weight=0`` this is AnyFlow's on-policy stage with that objective.
    """

    def build_model(self):
        super().build_model()
        self._moment_matcher: Optional[FeatureMomentMatching] = None
        moments = self.config.moment_matching
        if self.config.inference or moments.weight == 0:
            return
        if moments.weight < 0:
            raise ValueError(f"moment_matching.weight must be >= 0, got {moments.weight}")
        if moments.feature_block is None:
            raise ValueError(
                "moment_matching.weight > 0 requires moment_matching.feature_block, the index of the network block "
                "whose features are matched"
            )
        if not 0 < moments.feature_t <= self.net.noise_scheduler.max_t:
            raise ValueError(f"moment_matching.feature_t must be in (0, max_t], got {moments.feature_t}")
        if moments.grad_balance and not self.use_vsd:
            raise ValueError(
                "moment_matching.grad_balance scales the moment term to the VSD term, so it needs vsd_loss_weight > 0"
            )
        self._moment_matcher = FeatureMomentMatching(moments.queue_size)
        logger.info(
            f"TetherFlow moment matching: weight={moments.weight} (grad_balance={moments.grad_balance}), "
            f"feature_block={moments.feature_block}, feature_t={moments.feature_t}, queue_size={moments.queue_size}"
        )

    def _moment_feature_parameters(self) -> Dict[str, torch.Tensor]:
        """The detached weights the moment features are extracted with: the student's EMA when there is one."""
        if self.use_ema:
            ema = getattr(self, self.use_ema[0])
            return {name: param.detach() for name, param in ema.named_parameters()}
        # A LoRA student's EMA, registered by ``AdapterEMACallback``.
        adapter_ema = getattr(self, "adapter_ema", None)
        if adapter_ema is not None:
            return adapter_ema.ema_parameters()
        if not getattr(self, "_warned_live_moment_features", False):
            self._warned_live_moment_features = True
            logger.warning("No EMA of the student: the moment features use its detached live weights.")
        return {name: param.detach() for name, param in self.net.named_parameters() if param.requires_grad}

    def _moment_features(self, x0: torch.Tensor, condition: Any) -> torch.Tensor:
        """Mean-pooled ``[B, C]`` block features of ``x0`` noised to ``feature_t``, in the velocity mode ``r = t``."""
        moments = self.config.moment_matching
        scheduler = self.net.noise_scheduler
        t = torch.full((x0.shape[0],), moments.feature_t, device=x0.device, dtype=scheduler.t_precision)
        x_t = scheduler.forward_process(x0, torch.randn_like(x0), t)
        features = self.net(
            x_t,
            t,
            condition=condition,
            r=t,
            feature_indices={moments.feature_block},
            return_features_early=True,
        )[0]
        return features.float().mean(dim=tuple(range(2, features.ndim)))

    def _generator_regularizers(
        self,
        generation: Generation,
        data: Dict[str, Any],
        condition: Any,
        vsd_loss: Optional[torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """IDM's feature-moment loss of the generation against ``data['real']``, as a surrogate with its gradient.

        With ``grad_balance``, the term is scaled so that its gradient norm at the generation is ``weight`` times
        that of ``vsd_loss``.
        """
        if self._moment_matcher is None:
            return {}, None
        moments = self.config.moment_matching
        real = data.get("real")
        if real is None:
            raise ValueError("TetherFlow's moment matching requires data['real'] as the reference samples")
        samples = generation.data
        if not torch.is_tensor(samples):
            raise NotImplementedError("TetherFlow's moment matching supports single-modality generations only")
        if getattr(self.net, "_is_fsdp", False):
            raise NotImplementedError("TetherFlow's moment matching does not support an FSDP-sharded student yet")

        reference_grad = None
        if moments.grad_balance:
            # Only through the VSD loss itself, not the rollout: cheap, and the graph stays for the backward.
            (reference_grad,) = torch.autograd.grad(vsd_loss, samples, retain_graph=True)
        # The feature noise must not shift the training RNG stream, so runs with and without the term draw alike.
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            loss, grad, metrics = moment_loss_input_grad(
                samples,
                real,
                lambda x: self._moment_features(x, condition),
                self._moment_matcher,
                self.net,
                self._moment_feature_parameters(),
            )
        grad = grad.to(samples.dtype)
        grad_norm = grad.norm()
        metrics["moment_grad_norm"] = grad_norm
        if reference_grad is not None:
            reference_norm = reference_grad.norm()
            metrics["vsd_grad_norm"] = reference_norm
            weight = moments.weight * reference_norm / (grad_norm + 1e-12)
        else:
            weight = torch.tensor(moments.weight, device=samples.device)
        metrics["moment_weight"] = weight.detach()
        # The moment loss's value, with its gradient d loss / d samples flowing into the generation's graph.
        surrogate = loss + (samples * grad).sum() - (samples.detach() * grad).sum()
        return {"moment_loss": loss, **metrics}, weight.detach() * surrogate
