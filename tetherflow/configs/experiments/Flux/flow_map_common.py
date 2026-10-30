# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev setup shared by TetherFlow's two stages at 512px.

``config_tetherflow_stage1.py`` and ``config_tetherflow_stage2.py`` set their objective, then apply
``configure_stage1`` or ``configure_stage2``. This module adapts AnyFlow's Wan2.1 recipe, which TetherFlow builds on,
to FLUX.1-dev in the following ways.

**Architecture and adapters**

* **Gated dual-timestep embedding.** FLUX.1-dev gets AnyFlow's gated embedding: absolute ``r``, a convex gate of
  0.25, and an r-pathway initialized from the t-pathway, so ``r = t`` reproduces FLUX.1-dev exactly at init.
* **LoRA.** Rank-256 LoRA (alpha 256, PEFT's kaiming init) under DDP, as in the AnyFlow reference. The targets are
  the MM-DiT analogue of the reference's:
  self-attention, FFN and the t/r time MLPs, i.e. the image stream of the joint blocks, the single blocks and both
  time embedders. The text stream, the AdaLN modulations, the other embedders and the output head stay frozen.
* **Adapter EMA.** The network EMA cannot hold an adapter, so ``AdapterEMACallback`` keeps the EMA of the
  adapters only, which is what the reference averages. Evaluation samples from the EMA, and Stage 2 starts from
  Stage 1's EMA, as the reference does.
* **Teacher and fake score.** In Stage 2 the teacher is FLUX.1-dev itself (the student with its adapters disabled).
  The fake score is FLUX.1-dev with its own LoRA, because a full-rank 12B fake score would not fit under DDP.

**Guidance, time shift and precision**

* **No CFG.** FLUX.1-dev is guidance-distilled: every network runs at its embedded guidance of 3.5, the scale the
  training data was sampled at. There is no guidance fusion, text dropout or teacher CFG.
* **Time shift.** ``SHIFT`` is FLUX's native dynamic shift at 512px, exp(mu) = 1.878 for 1024 image tokens: the
  analogue of AnyFlow's shift 5, which is Wan2.1's default T2V sampling shift. The experiment configs apply it to
  every time density and grid, as AnyFlow does.
* **Finite-difference precision.** The finite-difference JVP runs FP32 forwards with TF32 GEMMs
  (``trainer.tf32_matmul``). This is within a few percent of IEEE FP32, at 2.5x the speed.

**Data, schedule and monitoring**

* **Data and run length.** The data is the 30-image FLUX.1-dev set that ``scripts/data/make_flux_dataset.py`` writes
  to ``$DATA_ROOT_DIR/flux_tetherflow``. Run lengths are scaled to it: 1000 Stage-1 iterations and 400 Stage-2 student updates. Read the results as a fitting and
  self-consistency comparison.
* **Dashboard.** ``FlowMapDashboardCallback`` writes the per-bucket losses and NFE 1/2/4/8/16 samples, which
  ``scripts/flowmap_dashboard/serve.py`` serves.
"""

import copy
import math

from tetherflow.callbacks.adapter_ema import AdapterEMACallback
from tetherflow.callbacks.flow_map_dashboard import FlowMapDashboardCallback
from tetherflow.configs.config_utils import model_pixel_shape
from tetherflow.configs.data import DATA_ROOT_DIR, ImageLoaderConfig
from tetherflow.configs.instantiate import LazyCall as L
from tetherflow.configs.net import FluxConfig
from tetherflow.utils.env import output_root

# Two training prompts and two held-out ones, sampled from the same noise at every eval.
EVAL_PROMPTS = [
    "A red fox sitting in fresh snow at the edge of a pine forest, its breath visible in the cold morning air, "
    "wildlife photography.",
    'A neon sign on a brick wall that reads "OPEN ALL NIGHT" in pink cursive letters.',
    "A corgi wearing a tiny knitted sweater, sitting on a wooden porch covered in autumn leaves, warm afternoon light.",
    "A glass teapot with blooming flower tea on a sunny windowsill, soft focus background, product photograph.",
]
EVAL_PROMPT_TAGS = ["train", "train", "held-out", "held-out"]

# FLUX's native time shift at 512px: exp(mu), with mu interpolated from 0.5 at 256 to 1.15 at 4096 packed image
# tokens (``Flux._calculate_shift``) at 1024 tokens.
SHIFT = math.exp(0.5 + (1.15 - 0.5) * (1024 - 256) / (4096 - 256))

# AnyFlow's Wan LoRA targets (self-attention, FFN, and the t and r time MLPs) on FLUX's MM-DiT. Suffixes match both
# block types' ``attn.to_q/k/v``; ``proj_out`` is named per single block so the output head stays frozen.
LORA_TARGET_MODULES = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "proj_mlp",
    *[f"single_transformer_blocks.{i}.proj_out" for i in range(38)],
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "r_embedder.linear_1",
    "r_embedder.linear_2",
]

# The last of FLUX.1-dev's 19 joint and 38 single transformer blocks.
LAST_BLOCK = 56

# The run group: $TETHERFLOW_OUTPUT_ROOT/tetherflow/<group>/<config name>.
TETHERFLOW_GROUP = "flux_tetherflow"

STAGE1_ITERATIONS = 1000
# Learning-rate warmup, then a constant rate (AnyFlow warms up for the first 1000 of its 6000 iterations). The
# adapter EMA tracks the live weights during warmup, then decays with a horizon of one warmup length (the
# reference uses 0.999 after its 1000-step warmup).
STAGE1_WARMUP = 333

# 400 student updates, alternating with as many fake-score updates; the EMA warms up for the first sixth of them,
# as in the reference.
STAGE2_ITERATIONS = 800
STAGE2_EMA_START = 133

STAGE2_CHARTS = [
    {"key": "loss/vsd", "title": "DMD (VSD) loss", "sub": "student update: the distribution-matching term"},
    {"key": "loss/fake_score", "title": "Fake-score loss", "sub": "denoising score matching on student samples"},
    {"key": "loss/cotrain", "title": "Co-trained flow-map loss", "sub": "Stage-1 objective on real data"},
    {"key": "loss/moment", "title": "Feature-moment loss", "sub": "statistical tether: EMA-feature mean/std gap"},
]
STAGE2_LOG_KEYS = {
    "loss/vsd": "vsd_loss",
    "loss/fake_score": "fake_score_loss",
    "loss/cotrain": "cotrain_loss",
    "loss/moment": "moment_loss",
    "moment/weight": "moment_weight",
    "moment/grad_norm": "moment_grad_norm",
    "moment/vsd_grad_norm": "vsd_grad_norm",
}


def stage1_checkpoint(group: str, name: str, iteration: int = STAGE1_ITERATIONS) -> str:
    """A Stage-1 run's checkpoint, under the default run directory ``$TETHERFLOW_OUTPUT_ROOT/tetherflow/<group>/<name>``."""
    return f"{output_root()}/tetherflow/{group}/{name}/checkpoints/{iteration:07d}.pth"


def _configure_student(config) -> None:
    """FLUX.1-dev flow-map student with LoRA under DDP, its precision, the data and local logging."""
    config.model.net = copy.deepcopy(FluxConfig)
    config.model.net.r_timestep = True
    config.model.net.r_embedder_init = "pretrained"
    config.model.net.time_cond_type = "abs"
    config.model.net.r_embedder_fusion = "gated"
    config.model.net.r_embedder_gate_value = 0.25
    config.model.net.min_t = 0.0
    config.model.net.max_t = 1.0
    config.model.net_adapter.type = "lora"
    config.model.net_adapter.rank = 256
    config.model.net_adapter.alpha = 256
    config.model.net_adapter.init_weights = True  # PEFT's kaiming-uniform A, as in the reference
    config.model.net_adapter.target_modules = list(LORA_TARGET_MODULES)
    config.model.load_student_weights = False  # the base loads its own pretrained weights

    # DDP with LoRA: no TetherFlow network EMA (see AdapterEMACallback), no meta init.
    config.trainer.ddp = True
    config.model.use_ema = False
    config.trainer.callbacks.pop("ema", None)
    config.model.precision_model = "bfloat16"
    config.model.precision_opt = "float32"
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_infer = "bfloat16"
    config.trainer.tf32_matmul = True
    config.model.input_shape = [16, 64, 64]  # 512x512

    # Guidance-distilled base: no prediction-side guidance fusion and no text dropout in the flow-map loss.
    config.model.guidance_fuse_scale = None
    config.model.cond_dropout_prob = None

    # Raw jpg + caption pairs, encoded on the fly; Stage 2 uses the prompts for the rollout and the images for the
    # co-trained loss.
    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    config.dataloader_train.datatags = [f"WDS:{DATA_ROOT_DIR}/flux_tetherflow/images"]
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)
    config.dataloader_train.batch_size = 8
    config.trainer.batch_size_global = 32
    config.trainer.logging_iter = 10
    config.trainer.checkpointer.save_adapter_only = True
    config.trainer.callbacks.pop("wandb", None)
    config.log_config.wandb_mode = "disabled"


def _dashboard(label: str, description: str, eval_every: int, **kwargs):
    return L(FlowMapDashboardCallback)(
        prompts=EVAL_PROMPTS,
        prompt_tags=EVAL_PROMPT_TAGS,
        nfes=[1, 2, 4, 8, 16],
        eval_every=eval_every,
        seed=0,
        label=label,
        description=description,
        **kwargs,
    )


def configure_stage1(config, label: str, description: str) -> None:
    """A flow-map pretraining run on FLUX.1-dev, apart from its objective and time shift."""
    _configure_student(config)
    config.model.net_scheduler.warm_up_steps = [STAGE1_WARMUP]
    config.trainer.max_iter = STAGE1_ITERATIONS
    config.trainer.save_ckpt_iter = 500
    # The adapter EMA goes before the dashboard, so evals see this step's EMA.
    config.trainer.callbacks["adapter_ema"] = L(AdapterEMACallback)(
        beta=1.0 - 1.0 / STAGE1_WARMUP, start_iter=STAGE1_WARMUP
    )
    config.trainer.callbacks["flow_map_dashboard"] = _dashboard(label, description, eval_every=100)


def configure_stage2(config, stage1_checkpoint: str, label: str, description: str) -> None:
    """An on-policy distillation run on FLUX.1-dev, apart from its objective and time shift."""
    _configure_student(config)
    # The teacher is FLUX.1-dev itself, the student with adapters disabled, queried without r. The fake score is
    # FLUX.1-dev with its own adapter, on the student's targets minus the r-pathway, which it never uses.
    config.model.teacher = None
    config.model.fake_score_adapter.type = "lora"
    config.model.fake_score_adapter.rank = 256
    config.model.fake_score_adapter.alpha = 256
    config.model.fake_score_adapter.init_weights = True
    config.model.fake_score_adapter.target_modules = [m for m in LORA_TARGET_MODULES if not m.startswith("r_embedder")]
    config.trainer.callbacks.pop("grad_clip_discriminator", None)

    # Seed the student from the Stage-1 checkpoint. The key map sets the start iteration and the weights outside
    # the adapters; the adapters then come from the Stage-1 EMA (``seed_from_pretrained``).
    config.trainer.checkpointer.pretrained_ckpt_path = stage1_checkpoint
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"net": "net"}

    config.trainer.max_iter = STAGE2_ITERATIONS
    config.trainer.save_ckpt_iter = 400
    config.trainer.callbacks["adapter_ema"] = L(AdapterEMACallback)(
        beta=0.99, start_iter=STAGE2_EMA_START, seed_from_pretrained=True
    )
    config.trainer.callbacks["flow_map_dashboard"] = _dashboard(
        label,
        description,
        eval_every=50,
        bucket_prefix="cotrain_",
        log_keys=dict(STAGE2_LOG_KEYS),
        charts=copy.deepcopy(STAGE2_CHARTS),
    )
