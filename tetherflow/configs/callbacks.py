# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tetherflow.configs.instantiate import LazyCall as L

from tetherflow.callbacks.ema import EMACallback
from tetherflow.callbacks.gpu_stats import GPUStatsCallback
from tetherflow.callbacks.grad_clip import GradClipCallback
from tetherflow.callbacks.param_count import ParamCountCallback
from tetherflow.callbacks.train_profiler import TrainProfilerCallback
from tetherflow.callbacks.wandb import WandbCallback

EMA_CALLBACK = dict(
    ema=L(EMACallback)(
        type="constant", beta=0.9999, gamma=16.97, ema_halflife_kimg=500, ema_rampup_ratio=0.05, start_iter=0
    ),
)

GradClip_CALLBACK = dict(
    grad_clip=L(GradClipCallback)(grad_norm=10.0, model_key="net"),
)

GPUStats_CALLBACK = dict(
    gpu_stats=L(GPUStatsCallback)(),
)

ParamCount_CALLBACK = dict(
    param_count=L(ParamCountCallback)(),
)

TrainProfiler_CALLBACK = dict(
    train_profiler=L(TrainProfilerCallback)(),
)

WANDB_CALLBACK = dict(
    wandb=L(WandbCallback)(),
)

BASE_CALLBACKS = {
    **WANDB_CALLBACK,
    **GPUStats_CALLBACK,
    **TrainProfiler_CALLBACK,
    **ParamCount_CALLBACK,
}
