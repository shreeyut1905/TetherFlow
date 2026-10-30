# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tetherflow.methods.model import BaseModel as BaseModel

# MeanFlow builds on the consistency-model trainer.
from tetherflow.methods.trajectory_matching.CM import CMModel as CMModel
from tetherflow.methods.trajectory_matching.mean_flow import MeanFlowModel as MeanFlowModel

from tetherflow.methods.distribution_matching.dmd2 import DMD2Model as DMD2Model

# AnyFlow pulls in mean_flow, so it must come after the trajectory-matching import above
from tetherflow.methods.distribution_matching.anyflow import AnyFlowModel as AnyFlowModel
from tetherflow.methods.distribution_matching.tetherflow import TetherFlowModel as TetherFlowModel
