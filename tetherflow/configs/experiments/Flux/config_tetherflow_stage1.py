# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TetherFlow Stage 1 (kinematic) on FLUX.1-dev at 512px: flow-map pretraining with the improved MeanFlow tangent.

* **Objective.** TetherFlow's kinematic objective (``config_tetherflow.apply_kinematic_objective``) is AnyFlow's
  MeanFlow objective with the flow map differentiated along the network's own instantaneous velocity
  ``u(x_t, t, t)``, instead of the conditional velocity ``eps - x0``.
* **Shared setup.** The FLUX.1-dev setup (architecture, LoRA, precision, data, schedule, adapter EMA and dashboard)
  lives in ``flow_map_common.py``.
* **Ablation.** ``model.loss_config.jvp_tangent=data`` gives AnyFlow's Stage 1, which differs only in the tangent.
* **What doesn't carry over.** iMF's guidance conditioning does not apply to the guidance-distilled FLUX.1-dev.
* **Hand-off.** The final checkpoint, with its adapter EMA, seeds Stage 2, ``config_tetherflow_stage2.py``.

Launch on 4 GPUs::

    torchrun --nproc_per_node=4 train.py --config=tetherflow/configs/experiments/Flux/config_tetherflow_stage1.py
"""

import tetherflow.configs.methods.config_tetherflow as config_tetherflow
from tetherflow.configs.experiments.Flux.flow_map_common import TETHERFLOW_GROUP, SHIFT, configure_stage1


def create_config():
    config = config_tetherflow.create_stage1_config()
    config_tetherflow.set_time_shift(config.model, SHIFT)
    configure_stage1(
        config,
        label="TetherFlow Stage 1 · kinematic",
        description="AnyFlow pretraining with the iMF tangent: JVP along the model's own velocity u(x_t, t, t)",
    )
    config.log_config.group = TETHERFLOW_GROUP
    return config
