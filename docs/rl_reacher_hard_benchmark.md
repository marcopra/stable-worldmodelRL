# Reacher hard representation benchmark

From the repository root, activate `swm-rl` and start all 15 runs with:

```bash
conda activate swm-rl
scripts/rl/benchmark_reacher_hard.sh
```

The launcher discovers the first two visible GPUs and assigns eight runs to
one queue and seven to the other. Each GPU runs one experiment at a time. The
five seeds (`1` through `5`) are crossed with `none`, `infonce`, and `lewm`.
Run logs are written under `logs/rl/reacher_hard_taco/`; unique checkpoints
are written under `runs/reacher_hard_taco/`. W&B runs share the
`taco-reacher-hard-3methods-5seeds-egl-v2` group in `stable-worldmodel-rl`.
Before starting the queue, the launcher preflights Reacher rendering on both
GPUs with MuJoCo's headless EGL backend. Each run binds EGL rendering to its
assigned physical GPU.

## TACO settings used

The local TACO `reacher_hard` config inherits `medium.yaml`, not `hard.yaml`.
It specifies 3.1 million physics frames with action repeat 2, or 1.55 million
agent steps. This config matches that budget and uses the TACO values for
discount (0.99), initial/final exploration standard deviation (1.0/0.1),
exploration warmup (2,000 agent steps), target tau (0.01), feature/hidden
dimensions (50/1024), learning rate (1e-4), temporal horizon (3), replay
batch (1,024), three-frame 84px RGB input, evaluation every 10,000 physics
frames, and 10 evaluation episodes. The Reacher wrapper repeats actions twice
and the trainer caps episodes at 500 agent steps (the 20-second task limit).

The local replay buffer stores full three-frame image stacks in RAM. A 1M
transition capacity would exceed this machine's available memory with two
concurrent jobs, so the config uses 100k entries. All other shared training
settings above are retained. TACO's `curl=true` and `reward=true` auxiliary
heads are objective-specific and are not included in the pure InfoNCE, LeWM,
and baseline comparison.

`lewm` selects a LeWM objective adapter over the shared DrQ pixel encoder: it
uses this repository's LeWM action embedder/predictor and SIGReg with the
LeWM default weight. It is not the standalone LeWM ViT encoder/model. The
InfoNCE variant uses the temporal action-conditioned objective already in
the DrQ agent. The `none` run disables the auxiliary representation loss.

All methods receive the same seeds, replay settings, batch size, optimizer
schedule, action repeat, pixel observations, evaluation schedule, and training
budget. See `scripts/train/config/drqv2_reacher_hard_taco.yaml` for the exact
resolved values.
