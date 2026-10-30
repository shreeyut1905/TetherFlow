# TetherFlow: Kinematic and Statistical Tethers for Any-Step Flow Maps

TetherFlow distills a text-to-image diffusion model into an **any-step flow map**. This is a single network
`u(x_t, t, r)` that jumps from noise level `t` to any `r < t`:

```
x_r = x_t - (t - r) * u(x_t, t, r)
```

So one model samples with 1, 2, 4, 8 or more steps. TetherFlow follows AnyFlow's two-stage recipe and adds a
**tether** in each stage, which ties what the model learns to a reference it can trust.

| Stage | Tether | What is tied to what |
|-------|--------|----------------------|
| 1. Flow-map pretraining | **Kinematic** | The flow map's average velocity, to the network's own instantaneous velocity |
| 2. On-policy distillation | **Statistical** | The student's samples, to the data's distribution and its feature statistics |

## Method

**Stage 1: kinematic tether.** Training uses the MeanFlow identity `u = v - (t - r) du/dt`. The total derivative
`du/dt = ∂u/∂x · v' + ∂u/∂t` needs a tangent `v'`:

- **Standard MeanFlow** uses the per-sample conditional velocity `eps - x0`, which carries each sample's noise into
  the regression input.
- **TetherFlow** follows improved MeanFlow (iMF) and uses the network's own detached instantaneous velocity,
  `v' = sg(u(x_t, t, t))`. This ties the average velocity to the field the model itself defines.

The rest of the recipe is AnyFlow's:

- `(t, r)` buckets: half of each batch at `r = t` (flow matching), a quarter at `r = 0` (consistency), and the rest
  at `0 < r < t` (flow map).
- A fixed `beta08` timestep weighting.
- A finite-difference JVP, computed in FP32.
- The `r < t` losses rebalanced to the flow-matching mean.

**Stage 2: statistical tether.** This stage is AnyFlow's on-policy distribution matching:

- The student generates by a flow-map rollout, with the number of steps drawn from {2, 4, 8, 16, 50} each iteration
  and at most 3 differentiable forwards.
- DMD's gradient compares the teacher's score with a fake score that tracks the student.
- The Stage-1 objective keeps training on real data, at weight 0.5.

On top of this, TetherFlow adds the feature-moment matching of IDM (Intrinsic Distribution Matching):

```
L_moment = ||mu_gen - mu_real||^2 + sum_j (sigma_gen_j - sigma_real_j)^2
```

- **Features.** The moments are over mean-pooled last-block features from the student's own EMA, with generated and
  real images both noised to `t = 0.8`. Detached queues of earlier features join the estimates.
- **Gradient path.** The gradient reaches the student only through its samples.
- **Weighting.** The term is gradient-balanced: its gradient is half the size of the DMD gradient.
- **Randomness.** Its noise comes from a separate random stream, so `weight=0` gives an exact ablation.

## Setup

```bash
pip install -e .
export HF_TOKEN=...  # FLUX.1-dev is gated: accept its license on Hugging Face first
```

**Data.** The training set is one FLUX.1-dev sample per prompt in `scripts/data/prompts.txt`: 30 prompts at
1024px, stored as WebDataset shards. Generate it with:

```bash
python scripts/data/make_flux_dataset.py  # writes $DATA_ROOT_DIR/flux_tetherflow (default: TETHERFLOW_OUTPUT/DATA)
```

Copy `.env.example` to `.env` to change the output, data or cache roots.

## Training

Both stages train rank-256 LoRA adapters on FLUX.1-dev at 512px with DDP. Each stage was run on 4 GPUs.

```bash
# Stage 1 (kinematic): 1000 iterations
torchrun --nproc_per_node=4 train.py --config=tetherflow/configs/experiments/Flux/config_tetherflow_stage1.py

# Stage 2 (statistical): 400 student updates, starting from Stage 1's adapter EMA
torchrun --nproc_per_node=4 train.py --config=tetherflow/configs/experiments/Flux/config_tetherflow_stage2.py
```

Runs write to `$TETHERFLOW_OUTPUT_ROOT/tetherflow/flux_tetherflow/<config name>/`. Overrides go after a lone `-`,
for example `... config_tetherflow_stage2.py - model.moment_matching.weight=0.1`.

**Ablations.**

| Run | Overrides |
|-----|-----------|
| No moment matching (Stage 2) | `model.moment_matching.weight=0` |
| AnyFlow baseline, Stage 1 | `model.loss_config.jvp_tangent=data` |
| AnyFlow baseline, Stage 2 | `model.moment_matching.weight=0 model.loss_config.jvp_tangent=data`, from a baseline Stage-1 checkpoint |
| Start Stage 2 from a checkpoint without an adapter EMA | `trainer.checkpointer.pretrained_ckpt_path=<path> trainer.callbacks.adapter_ema.seed_from_pretrained=False` |

## Dashboard

During training, every run logs the per-bucket losses (`r = t`, `r = 0`, `0 < r < t`) and the Stage-2 loss terms.
Every 100 iterations in Stage 1, and every 50 in Stage 2, it also saves samples at 1, 2, 4, 8 and 16 steps from the
EMA. A small local server compares runs side by side:

```bash
python scripts/flowmap_dashboard/serve.py \
    --run stage1=TETHERFLOW_OUTPUT/tetherflow/flux_tetherflow/config_tetherflow_stage1 \
    --run stage2=TETHERFLOW_OUTPUT/tetherflow/flux_tetherflow/config_tetherflow_stage2 --port 8765
```

From another machine, forward the port with `ssh -L 8765:localhost:8765 <host>`.

## Layout

```
tetherflow/
  methods/distribution_matching/tetherflow.py   TetherFlowModel (Stage 2: DMD + moment matching)
  methods/distribution_matching/feature_moments.py   the moment loss and its EMA-weight gradient
  methods/trajectory_matching/mean_flow.py      the flow-map loss, including the kinematic (iMF) tangent
  configs/methods/config_tetherflow.py          both stages' recipe, for any network
  configs/experiments/Flux/                     the FLUX.1-dev runs
  callbacks/adapter_ema.py                      EMA of the LoRA adapters; seeds Stage 2 from Stage 1
  callbacks/flow_map_dashboard.py               writes the dashboard's metrics and samples
scripts/data/make_flux_dataset.py               builds the training set
scripts/flowmap_dashboard/                      the dashboard server
tests/                                          unit tests (make test)
```

## Status

This is a personal research project, in early experiments on a 30-image FLUX.1-dev set. There are no benchmark
results yet.

## License

Apache License 2.0 (see `LICENSE`). TetherFlow's training framework is derived from NVIDIA FastGen. Files carrying
NVIDIA's copyright header come from it and were modified for this project, and `licenses/` lists the third-party
code that is included.

TetherFlow builds on these papers:

- **AnyFlow**: Gu et al., 2026, [arXiv:2605.13724](https://arxiv.org/abs/2605.13724).
- **MeanFlow**: Geng et al., 2025, [arXiv:2505.13447](https://arxiv.org/abs/2505.13447).
- **Improved MeanFlow**: Geng et al., 2025, [arXiv:2512.02012](https://arxiv.org/abs/2512.02012).
- **DMD2**: Yin et al., 2024, [arXiv:2405.14867](https://arxiv.org/abs/2405.14867).
- **IDM**: Liu et al., 2026, [arXiv:2610.39633](https://arxiv.org/abs/2610.39633).
