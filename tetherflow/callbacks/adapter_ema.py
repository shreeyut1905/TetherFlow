# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EMA of a LoRA student's trainable tensors.

``EMACallback`` averages a full EMA network built from ``config.net``, which cannot hold an adapter, so
``model.use_ema`` is rejected for LoRA students. This callback keeps an FP32 shadow of only the trainable
tensors of ``model.net`` -- the LoRA adapters and any extra trainable modules, which is what the AnyFlow reference
averages. Every rank keeps its own shadow: DDP replicas are identical, so no communication is needed. The shadow is
saved with the callback state in checkpoints (keyed like ``model.net.state_dict()``), and ``swapped_in`` puts it
into the network temporarily, e.g. for evaluation (``FlowMapDashboardCallback`` does so when one is registered).

Before ``start_iter`` the EMA copies the live weights (decay 0), as the reference does during its warmup. A run
that resumes from a checkpoint without EMA state starts its EMA from the weights it resumes with.

With ``seed_from_pretrained``, a stage starts from the previous stage's EMA. The trainer first loads
``trainer.checkpointer.pretrained_ckpt_path`` through ``pretrained_ckpt_key_map`` as usual: that sets
``model.resume_iter`` and the weights outside the adapter. This callback then overwrites ``model.net``'s trainable
tensors with the adapter EMA stored in that checkpoint, before the run's own checkpoint, if any, is resumed. AnyFlow
seeds its on-policy stage this way (``pretrained_weight: ema``); the native EMA's ``{"net": "ema"}`` key map cannot,
because a LoRA run has no EMA network.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import torch

from tetherflow.callbacks.callback import Callback
from tetherflow.utils.distributed.collectives import raise_across_ranks
from tetherflow.utils.io_utils import load_training_checkpoint
import tetherflow.utils.logging_utils as logger

if TYPE_CHECKING:
    from tetherflow.methods import BaseModel


class AdapterEMACallback(Callback):
    def __init__(
        self,
        beta: float = 0.999,
        start_iter: int = 0,
        seed_from_pretrained: bool = False,
        pretrained_name: str = "adapter_ema",
    ):
        """
        Args:
            beta: EMA decay once ``start_iter`` is reached.
            start_iter: Iteration from which the EMA decays; before it, the EMA tracks the live weights.
            seed_from_pretrained: Start ``model.net``'s trainable tensors from the adapter EMA saved in
                ``trainer.checkpointer.pretrained_ckpt_path`` (see the module docstring).
            pretrained_name: Name of the ``AdapterEMACallback`` whose state that checkpoint holds.
        """
        if not 0 <= beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta = beta
        self.start_iter = start_iter
        self.seed_from_pretrained = seed_from_pretrained
        self.pretrained_name = pretrained_name
        self._shadow: Optional[Dict[str, torch.Tensor]] = None
        self._pairs: List[Tuple[torch.Tensor, torch.nn.Parameter]] = []
        self._loaded: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _trainable(model: BaseModel) -> Dict[str, torch.nn.Parameter]:
        return {name: param for name, param in model.net.named_parameters() if param.requires_grad}

    @staticmethod
    def _check_names(stored: Dict[str, torch.Tensor], params: Dict[str, Any], source: str) -> None:
        if set(stored) != set(params):
            missing = sorted(set(params) - set(stored))[:5]
            unexpected = sorted(set(stored) - set(params))[:5]
            raise ValueError(
                f"{source} does not match the trainable tensors of model.net "
                f"(missing e.g. {missing}, unexpected e.g. {unexpected})"
            )

    def on_load_checkpoint_start(self, model: BaseModel) -> None:
        # Runs after the trainer loads pretrained_ckpt_path and before it resumes the run's own checkpoint.
        if not self.seed_from_pretrained:
            return
        path = self.config.trainer.checkpointer.pretrained_ckpt_path
        with raise_across_ranks("adapter EMA seed"):
            if not path:
                raise ValueError("seed_from_pretrained requires trainer.checkpointer.pretrained_ckpt_path")
            state = load_training_checkpoint(path, mmap=True)
            ema_state = (state.get("callbacks") or {}).get(self.pretrained_name) or {}
            if "shadow" not in ema_state:
                raise ValueError(
                    f"{path} holds no adapter EMA under callback {self.pretrained_name!r} (it was saved without "
                    "AdapterEMACallback); set seed_from_pretrained=False to start from its live weights instead"
                )
            params = self._trainable(model)
            self._check_names(ema_state["shadow"], params, f"The adapter EMA in {path}")
            with torch.no_grad():
                for name, param in params.items():
                    param.copy_(ema_state["shadow"][name])
        logger.info(
            f"Seeded {len(params)} trainable tensors of model.net with the adapter EMA of {path} "
            f"(iteration {state.get('iteration')})"
        )

    def on_train_begin(self, model: BaseModel, iteration: int = 0) -> None:
        params = self._trainable(model)
        if not params:
            raise ValueError("AdapterEMACallback found no trainable parameters in model.net")
        self._shadow = {name: param.detach().float().clone() for name, param in params.items()}
        if self._loaded is not None:
            self._check_names(self._loaded, self._shadow, "Adapter EMA checkpoint")
            for name, tensor in self._loaded.items():
                self._shadow[name].copy_(tensor)
            logger.info(f"Restored the adapter EMA of {len(self._shadow)} tensors from the checkpoint")
            self._loaded = None
        else:
            logger.info(
                f"Started the adapter EMA of {len(self._shadow)} tensors from the weights at iteration {iteration}"
            )
        self._pairs = [(self._shadow[name], param) for name, param in params.items()]
        # Lets other callbacks evaluate with the EMA weights (see ``swapped_in``).
        model.adapter_ema = self

    @torch.no_grad()
    def on_training_step_end(
        self,
        model: BaseModel,
        data_batch: Dict[str, Any],
        output_batch: Dict[str, Any],
        loss_dict: Dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        shadows = [shadow for shadow, _ in self._pairs]
        live = [param.detach().float() for _, param in self._pairs]
        if iteration < self.start_iter:
            torch._foreach_copy_(shadows, live)
        else:
            torch._foreach_lerp_(shadows, live, 1.0 - self.beta)

    def ema_parameters(self) -> Dict[str, torch.Tensor]:
        """The EMA tensors keyed by their ``model.net`` parameter names (for a functional forward)."""
        if self._shadow is None:
            raise RuntimeError("The adapter EMA is not initialized before on_train_begin")
        return dict(self._shadow)

    @contextlib.contextmanager
    def swapped_in(self, model: BaseModel) -> Iterator[None]:
        """Temporarily load the EMA into ``model.net``'s trainable tensors, restoring the live weights after."""
        params = [param for _, param in self._pairs]
        backup = [param.detach().clone() for param in params]
        with torch.no_grad():
            for shadow, param in self._pairs:
                param.copy_(shadow)
        try:
            yield
        finally:
            with torch.no_grad():
                torch._foreach_copy_(params, backup)

    def state_dict(self) -> Dict[str, Any]:
        # Called on rank 0 only, at checkpoint time.
        if self._shadow is None:
            return {}
        return {"beta": self.beta, "shadow": {name: tensor.detach().cpu() for name, tensor in self._shadow.items()}}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Runs before ``on_train_begin`` when resuming; the shadow is rebuilt there.
        if state_dict and "shadow" in state_dict:
            self._loaded = dict(state_dict["shadow"])
