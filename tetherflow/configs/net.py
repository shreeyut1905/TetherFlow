# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network presets: FLUX.1-dev, and the small EDM U-Net that serves as the default network and in the unit tests."""

import os

from omegaconf import DictConfig

from tetherflow.configs.instantiate import LazyCall as L
from tetherflow.networks.EDM.network import EDMPrecond, SongUNet
from tetherflow.networks.Flux.network import Flux
from tetherflow.utils.env import output_root

CKPT_ROOT_DIR = os.getenv("CKPT_ROOT_DIR", f"{output_root()}/MODEL")

EDM_CIFAR10_Config: DictConfig = L(EDMPrecond)(
    sigma_shift=0.0,
    sigma_data=0.5,
    backbone=L(SongUNet)(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=10,
        augment_dim=9,
        model_channels=128,
        channel_mult=[2, 2, 2],
        channel_mult_noise=1,
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        resample_filter=[1, 1],
        dropout=0.0,
        label_dropout=0,
        r_timestep=False,
    ),
    drop_precond=None,
)

# FLUX.1-dev
FluxConfig: DictConfig = L(Flux)(
    guidance_scale=3.5,
)
