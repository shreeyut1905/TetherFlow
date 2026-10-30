# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local dashboard artifacts for flow-map runs, served by ``scripts/flowmap_dashboard/serve.py``.

Rank 0 writes below ``<log_config.save_path>/<subdir>/``:

* ``run.json``: the run's label, description and ``max_iter``.
* ``metrics.jsonl``: one line per iteration with the global ``total_loss`` (mean over ranks and accumulation
  rounds), the per-bucket residuals ``FlowMapLossMixin`` logs under ``loss_config.log_bucket_metrics`` (combined
  over accumulation rounds by their global counts; keys prefixed with ``bucket_prefix``), any ``log_keys``
  (means over the ranks and rounds that logged them, e.g. on alternating DMD2 iterations), the learning rate and
  the step time. ``run.json`` also carries ``charts``, the loss charts the dashboard draws.
* ``eval/<iteration:07d>/``: student samples for fixed prompts and fixed noise at every requested NFE, then
  ``meta.json`` (written last, atomically) with each NFE's latent distance to the largest NFE's sample.

Every rank runs the same eval forwards and collectives, so the callback is DDP- and FSDP-safe; only rank 0 writes,
inside ``raise_across_ranks`` so a failed write raises on every rank. Eval runs under ``no_grad`` rather than
``inference_mode``: an FSDP root's first forward (the iteration-0 eval) would otherwise create its reusable unsharded
buffers as inference tensors. Samples come from the student ``model.net``, with the EMA of an
``AdapterEMACallback`` swapped in when one is registered (``meta.json`` records which weights were used).
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import torch

from tetherflow.callbacks.callback import Callback
from tetherflow.callbacks.wandb import vae_decode_context
from tetherflow.utils import basic_utils
from tetherflow.utils.distributed import is_rank0, world_size
from tetherflow.utils.distributed.collectives import raise_across_ranks
import tetherflow.utils.logging_utils as logger

if TYPE_CHECKING:
    from tetherflow.methods import BaseModel

BUCKETS = ("r_eq_t", "r_eq_0", "r_lt_t")


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class FlowMapDashboardCallback(Callback):
    def __init__(
        self,
        prompts: Sequence[str],
        prompt_tags: Optional[Sequence[str]] = None,
        nfes: Sequence[int] = (1, 2, 4, 8, 16),
        eval_every: int = 100,
        eval_at_start: bool = True,
        seed: int = 0,
        label: Optional[str] = None,
        description: Optional[str] = None,
        subdir: str = "dashboard",
        jpeg_quality: int = 92,
        bucket_prefix: str = "",
        log_keys: Optional[Dict[str, str]] = None,
        charts: Optional[Sequence[Dict[str, str]]] = None,
    ):
        """
        Args:
            prompts: Fixed eval prompts; every eval samples each of them from the same noise.
            prompt_tags: Optional per-prompt note shown next to it (e.g. ``"train"`` or ``"held-out"``).
            nfes: Student step counts to sample at; the largest is the any-step consistency reference.
            eval_every: Eval cadence in iterations; the final iteration always evaluates.
            eval_at_start: Also evaluate before the first training step (the initial network on a fresh run).
            seed: Seed of the fixed eval noise, shared by every run that uses the same seed and prompts.
            label: Run name on the dashboard (defaults to ``log_config.name``).
            description: One-line run description on the dashboard.
            subdir: Artifact directory below ``log_config.save_path``.
            jpeg_quality: JPEG quality of the saved samples.
            bucket_prefix: Prefix of the bucket metrics in the loss map, e.g. ``"cotrain_"`` for AnyFlow Stage 2.
            log_keys: Extra metrics to log, as ``{metric name: loss-map key}``.
            charts: Loss charts for the dashboard, ``[{"key", "title", "sub"}]`` over the logged metric names;
                ``None`` keeps the dashboard's per-bucket default.
        """
        self.prompts = [str(p) for p in prompts]
        if not self.prompts:
            raise ValueError("FlowMapDashboardCallback needs at least one eval prompt")
        self.prompt_tags = [str(t) for t in prompt_tags] if prompt_tags is not None else [""] * len(self.prompts)
        if len(self.prompt_tags) != len(self.prompts):
            raise ValueError(f"Got {len(self.prompt_tags)} prompt tags for {len(self.prompts)} prompts")
        self.nfes = sorted({int(k) for k in nfes})
        if not self.nfes or self.nfes[0] < 1:
            raise ValueError(f"nfes must be positive step counts, got {list(nfes)}")
        if eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {eval_every}")
        self.eval_every = eval_every
        self.eval_at_start = eval_at_start
        self.seed = seed
        self.label = label
        self.description = description
        self.subdir = subdir
        self.jpeg_quality = jpeg_quality
        self.bucket_prefix = bucket_prefix
        self.log_keys = dict(log_keys or {})
        self.charts = [dict(chart) for chart in charts] if charts is not None else None
        self._extra_acc: Dict[str, List[float]] = {}

        self._dir: Optional[str] = None
        self._condition: Any = None
        self._noise: Optional[torch.Tensor] = None
        self._round_totals: List[torch.Tensor] = []
        self._bucket_acc: Dict[str, List[torch.Tensor]] = {}
        self._last_step_time: Optional[float] = None
        self._lr: Optional[float] = None

    # ----- lifecycle -----------------------------------------------------------------------------------------

    def on_train_begin(self, model: BaseModel, iteration: int = 0) -> None:
        log_config = self.config.log_config
        self._dir = os.path.join(log_config.save_path, self.subdir)
        with raise_across_ranks("flow-map dashboard setup"):
            if is_rank0():
                self._write_run_info(iteration)
        if self.eval_at_start:
            self._evaluate(model, iteration)
        self._last_step_time = time.time()

    def _write_run_info(self, iteration: int) -> None:
        log_config = self.config.log_config
        os.makedirs(os.path.join(self._dir, "eval"), exist_ok=True)
        info = {
            "label": self.label or log_config.name,
            "description": self.description,
            "max_iter": int(self.config.trainer.max_iter),
            "name": log_config.name,
            "group": log_config.group,
            "start_iteration": int(iteration),
            "nfes": self.nfes,
        }
        if self.charts is not None:
            info["charts"] = self.charts
        _write_atomic(os.path.join(self._dir, "run.json"), json.dumps(info, indent=2).encode())

    def on_optimizer_step_begin(self, model: BaseModel, iteration: int = 0) -> None:
        # Read before the scheduler steps, so the logged LR is the one this iteration's update used.
        scheduler = model.scheduler_dict.get("net") if hasattr(model, "scheduler_dict") else None
        if scheduler is not None:
            self._lr = float(scheduler.get_last_lr()[0])

    def on_backward_begin(
        self,
        model: BaseModel,
        data_batch: Dict[str, Any],
        output_batch: Dict[str, Any],
        loss_dict: Dict[str, torch.Tensor],
        iteration: int = 0,
        accum_iter: int = 0,
    ) -> None:
        # Kept on device: one host sync per iteration, in ``on_training_step_end``.
        self._round_totals.append(loss_dict["total_loss"].detach().float())
        for name in BUCKETS:
            residual = loss_dict.get(f"{self.bucket_prefix}bucket_residual/{name}")
            count = loss_dict.get(f"{self.bucket_prefix}bucket_count/{name}")
            if residual is None or count is None:
                continue
            # The loss step already reduced these over ranks; weight each round by its global count.
            weighted = residual.detach().float() * count.detach().float()
            acc = self._bucket_acc.setdefault(name, [torch.zeros_like(weighted), torch.zeros_like(weighted)])
            acc[0] += weighted
            acc[1] += count.detach().float()
        for name, key in self.log_keys.items():
            value = loss_dict.get(key)
            if value is None:
                continue
            acc = self._extra_acc.setdefault(name, [0.0, 0.0])
            acc[0] += float(value.detach().float().mean()) if torch.is_tensor(value) else float(value)
            acc[1] += 1.0

    def on_training_step_end(
        self,
        model: BaseModel,
        data_batch: Dict[str, Any],
        output_batch: Dict[str, Any],
        loss_dict: Dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        now = time.time()
        step_seconds = None if self._last_step_time is None else now - self._last_step_time

        if self._round_totals:
            total = torch.stack(self._round_totals).mean()
        else:
            total = loss_dict["total_loss"].detach().float()
        # One collective for everything rank-dependent: ranks see different buckets (deterministic partitions),
        # so their total losses differ, and each extra metric is averaged over the ranks and rounds that logged it.
        names = list(self.log_keys)
        sums_counts = [self._extra_acc.get(name, [0.0, 0.0]) for name in names]
        stats = torch.tensor(
            [0.0] + [sc[0] for sc in sums_counts] + [sc[1] for sc in sums_counts],
            dtype=torch.float32,
            device=total.device,
        )
        stats[0] = total
        if world_size() > 1:
            torch.distributed.all_reduce(stats)
            stats[0] = stats[0] / world_size()
        stats = stats.tolist()

        row: Dict[str, Any] = {"iteration": int(iteration), "wall_time": now, "loss/total": stats[0]}
        for index, name in enumerate(names):
            count = stats[1 + len(names) + index]
            if count > 0:
                row[name] = stats[1 + index] / count
        self._extra_acc = {}
        if step_seconds is not None:
            row["sec_per_iter"] = step_seconds
        for name, (weighted, count) in self._bucket_acc.items():
            count_value = float(count)
            row[f"count/{name}"] = count_value
            row[f"loss/{name}"] = float(weighted) / count_value if count_value > 0 else None
        if self._lr is not None:
            row["lr"] = self._lr
        self._round_totals = []
        self._bucket_acc = {}

        with raise_across_ranks("flow-map dashboard metrics"):
            if is_rank0() and self._dir is not None:
                # NaN/inf (a diverged loss) is not JSON: write null.
                row = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}
                with open(os.path.join(self._dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

        if iteration % self.eval_every == 0 or iteration == self.config.trainer.max_iter:
            self._evaluate(model, iteration)
        # Exclude eval time from the next step time.
        self._last_step_time = time.time()

    # ----- eval ---------------------------------------------------------------------------------------------

    def _prepare_eval_inputs(self, model: BaseModel) -> None:
        """Encode the eval prompts once and draw the fixed eval noise (identical on every rank and run)."""
        n = len(self.prompts)
        batch = model.prepare_batch({"condition_raw": list(self.prompts)}, materialize_real=False)
        self._condition = batch["condition"]
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        noise = torch.randn([n, *model.input_shape], generator=generator, dtype=torch.float32)
        self._noise = noise.to(device=model.device, dtype=model.precision)

    @torch.no_grad()
    def _evaluate(self, model: BaseModel, iteration: int) -> None:
        start = time.time()
        if self._condition is None:
            self._prepare_eval_inputs(model)
        net = model.net
        device_type = model.device.type

        amp = model.precision_amp_infer
        ema = getattr(model, "adapter_ema", None)
        weights = "live" if ema is None else "ema"
        latents: Dict[int, torch.Tensor] = {}
        with contextlib.nullcontext() if ema is None else ema.swapped_in(model), basic_utils.train_mode(
            net, mode=False
        ), torch.autocast(device_type=device_type, dtype=amp, enabled=amp is not None):
            for nfe in self.nfes:
                grid = net.noise_scheduler.student_grid(nfe, device=model.device)
                state = net.noise_scheduler.latents(noise=self._noise, t_init=grid[0])
                state = model._student_sample_loop(net, state, student_grid=grid, condition=self._condition)
                latents[nfe] = state.float()
        # A forward without backward leaves FSDP roots unsharded; release them before the decode.
        model.reshard_fsdp_modules()

        reference = latents[self.nfes[-1]]
        consistency = {
            str(nfe): float((latents[nfe] - reference).pow(2).mean()) for nfe in self.nfes if nfe != self.nfes[-1]
        }

        images: Dict[int, torch.Tensor] = {}
        with vae_decode_context(model) as vae:
            for nfe in self.nfes:
                decoded = latents[nfe] if vae is None else vae.decode(latents[nfe].to(model.precision))
                images[nfe] = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()
        model.reshard_fsdp_modules()

        with raise_across_ranks("flow-map dashboard eval write"):
            if is_rank0() and self._dir is not None:
                self._write_eval(iteration, images, consistency, time.time() - start, weights)
        logger.info(
            f"Flow-map dashboard eval at iteration {iteration} ({weights} weights): {time.time() - start:.1f}s, "
            f"latent MSE to NFE {self.nfes[-1]}: {consistency}"
        )

    def _write_eval(
        self,
        iteration: int,
        images: Dict[int, torch.Tensor],
        consistency: Dict[str, float],
        seconds: float,
        weights: str = "live",
    ) -> None:
        from PIL import Image

        eval_dir = os.path.join(self._dir, "eval", f"{iteration:07d}")
        os.makedirs(eval_dir, exist_ok=True)
        for nfe, batch in images.items():
            for index, image in enumerate(batch):
                buffer = io.BytesIO()
                Image.fromarray(image.permute(1, 2, 0).numpy()).save(buffer, format="JPEG", quality=self.jpeg_quality)
                _write_atomic(os.path.join(eval_dir, f"p{index}_nfe{nfe}.jpg"), buffer.getvalue())
        meta = {
            "iteration": int(iteration),
            "wall_time": time.time(),
            "eval_seconds": seconds,
            "nfes": self.nfes,
            "prompts": self.prompts,
            "prompt_tags": self.prompt_tags,
            "consistency_mse": consistency,
            "seed": self.seed,
            "weights": weights,
        }
        _write_atomic(os.path.join(eval_dir, "meta.json"), json.dumps(meta, indent=2).encode())
