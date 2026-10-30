# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TetherFlow Stage 2 (statistical) on FLUX.1-dev at 512px: AnyFlow's on-policy DMD with IDM's moment matching.

``TetherFlowModel`` with the recipe of ``config_tetherflow.create_config``, which has two parts:

* **AnyFlow's on-policy DMD.** A flow-map rollout student, no adversarial loss, and 1:1 student / fake-score updates.
  It co-trains the kinematic objective at weight 0.5.
* **IDM's feature-moment matching.** The rollout outputs and the real batch are both noised to t = 0.8. Their
  student-EMA features are taken from the last block and mean-pooled. The mean and standard deviation of these
  features are matched, with a gradient at the rollout outputs half the size of the DMD gradient.

The student starts from the adapter EMA of the final checkpoint of ``config_tetherflow_stage1.py``. The FLUX.1-dev setup
(teacher, LoRA fake score, seeding, schedule, adapter EMA and dashboard) lives in ``flow_map_common.py``.

Launch on 4 GPUs once Stage 1 has finished::

    torchrun --nproc_per_node=4 train.py --config=tetherflow/configs/experiments/Flux/config_tetherflow_stage2.py

Useful overrides:

* ``model.moment_matching.weight=0`` drops the moment term; adding ``model.loss_config.jvp_tangent=data`` and a
  Stage-1 checkpoint trained with that tangent gives AnyFlow's Stage 2.
* To start from a checkpoint saved without an adapter EMA, override both
  ``trainer.checkpointer.pretrained_ckpt_path=<checkpoint>`` and
  ``trainer.callbacks.adapter_ema.seed_from_pretrained=False``. The student then starts from the checkpoint's live
  adapters.
"""

import tetherflow.configs.methods.config_tetherflow as config_tetherflow
from tetherflow.configs.experiments.Flux.flow_map_common import (
    TETHERFLOW_GROUP,
    LAST_BLOCK,
    SHIFT,
    configure_stage2,
    stage1_checkpoint,
)


def create_config():
    config = config_tetherflow.create_config()
    config_tetherflow.set_time_shift(config.model, SHIFT)
    config.model.moment_matching.feature_block = LAST_BLOCK
    configure_stage2(
        config,
        stage1_checkpoint=stage1_checkpoint(TETHERFLOW_GROUP, "config_tetherflow_stage1"),
        label="TetherFlow Stage 2 · statistical",
        description="AnyFlow on-policy DMD + IDM feature-moment matching, from the TetherFlow Stage-1 EMA",
    )
    config.log_config.group = TETHERFLOW_GROUP
    return config
