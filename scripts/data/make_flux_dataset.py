# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate TetherFlow's FLUX.1-dev training set: one teacher sample per prompt, as WebDataset shards.

Writes to ``--out_dir`` (default ``$DATA_ROOT_DIR/flux_tetherflow``):
  images/00000.tar        {key}.jpg + {key}.txt + {key}.json  -> the training data (real_raw: jpg, condition_raw: txt)
  prompts_only/00000.tar  {key}.txt                           -> prompts only, for data-free runs
  contact_sheet.jpg       grid of all samples for a visual check

FLUX.1-dev is gated on Hugging Face: accept its license and export ``HF_TOKEN``. One process runs per GPU:
  python scripts/data/make_flux_dataset.py
"""

import argparse
import io
import json
import multiprocessing as mp
import os
from pathlib import Path

import torch
import webdataset as wds
from diffusers import FluxPipeline
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.1-dev"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_DIR = os.getenv(
    "DATA_ROOT_DIR", f"{os.getenv('TETHERFLOW_OUTPUT_ROOT', str(REPO_ROOT / 'TETHERFLOW_OUTPUT'))}/DATA"
)


def write_shard(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tar.tmp")
    with wds.TarWriter(str(tmp)) as sink:
        for sample in samples:
            sink.write(sample)
    os.replace(tmp, path)


def generate_on_gpu(job: tuple[int, list[tuple[int, str]], argparse.Namespace]) -> list[tuple[int, bytes]]:
    """Generate one GPU's share of the prompts and return ``(prompt index, JPEG bytes)`` pairs."""
    rank, items, args = job
    device = f"cuda:{rank}"
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir).to(device)
    pipe.set_progress_bar_config(disable=True)

    out = []
    for idx, prompt in items:
        seed = args.seed + idx
        image = pipe(
            prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            max_sequence_length=args.max_sequence_length,
            generator=torch.Generator(device).manual_seed(seed),
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95, subsampling=0)
        out.append((idx, buf.getvalue()))
        print(f"[gpu {rank}] flux_{idx:06d} seed={seed}: {prompt[:70]}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "prompts.txt"))
    parser.add_argument("--out_dir", default=f"{DATA_ROOT_DIR}/flux_tetherflow")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    # TetherFlow's FluxConfig pins the embedded distillation guidance to 3.5.
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--num_gpus", type=int, default=None, help="Default: all visible GPUs")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    num_gpus = min(args.num_gpus or torch.cuda.device_count(), len(prompts))
    print(
        f"{len(prompts)} prompts on {num_gpus} GPUs, {args.width}x{args.height}, {args.steps} steps, "
        f"guidance {args.guidance}"
    )

    # Round-robin the prompts over one process per GPU; seeds follow the prompt index, not the GPU.
    jobs = [(rank, [(i, p) for i, p in enumerate(prompts) if i % num_gpus == rank], args) for rank in range(num_gpus)]
    with mp.get_context("spawn").Pool(num_gpus) as pool:
        results = dict(item for worker_out in pool.map(generate_on_gpu, jobs) for item in worker_out)

    image_samples, prompt_samples, images = [], [], []
    for idx, prompt in enumerate(prompts):
        seed = args.seed + idx
        jpg = results[idx]
        images.append(Image.open(io.BytesIO(jpg)).convert("RGB"))
        key = f"flux_{idx:06d}"
        meta = {
            "prompt": prompt,
            "seed": seed,
            "model": MODEL_ID,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
            "max_sequence_length": args.max_sequence_length,
        }
        image_samples.append(
            {"__key__": key, "jpg": jpg, "txt": prompt.encode("utf-8"), "json": json.dumps(meta).encode()}
        )
        prompt_samples.append({"__key__": key, "txt": prompt.encode("utf-8")})

    write_shard(out_dir / "images" / "00000.tar", image_samples)
    write_shard(out_dir / "prompts_only" / "00000.tar", prompt_samples)

    cols, thumb = 6, 256
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, image in enumerate(images):
        sheet.paste(image.resize((thumb, thumb), Image.BICUBIC), ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_dir / "contact_sheet.jpg", quality=90)
    print(f"Wrote {len(image_samples)} samples to {out_dir}")


if __name__ == "__main__":
    main()
