# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataloader configs: the image WebDataset loader the experiments use, and the CIFAR-10 loader that is the default."""

import os

from tetherflow.configs.instantiate import LazyCall as L
from tetherflow.datasets.class_cond_dataloader import ImageLoader
from tetherflow.datasets.wds_dataloaders import ImageWDSLoader
from tetherflow.utils.env import output_root

DATA_ROOT_DIR = os.getenv("DATA_ROOT_DIR", f"{output_root()}/DATA")
REMOTE_DATA_ROOT_DIR = os.getenv("REMOTE_DATA_ROOT_DIR")


def _remote_dataset_path(relative_path: str) -> str | None:
    """The mirrored copy of a dataset under ``REMOTE_DATA_ROOT_DIR``, or None without a mirror."""
    return f"{REMOTE_DATA_ROOT_DIR}/{relative_path}" if REMOTE_DATA_ROOT_DIR else None


ImageLoaderConfig = L(ImageWDSLoader)(
    datatags=["WDS:/path/to/images"],
    batch_size=32,
    key_map={"real_raw": "jpg", "condition_raw": "txt"},
    presets_map={"neg_condition_raw": "empty_string"},
    input_res=512,
    sampler_start_idx=None,
)

CIFAR10_Loader_Config = L(ImageLoader)(
    dataset_path=f"{DATA_ROOT_DIR}/cifar10/cifar10-32x32.zip",
    remote_path=_remote_dataset_path("cifar10/cifar10-32x32.zip"),
    use_labels=True,
    cache=True,
    batch_size=128,
    shuffle=True,
    sampler_start_idx=None,
)
