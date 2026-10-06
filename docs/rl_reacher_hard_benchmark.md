# Reacher hard representation benchmark

From the repository root, activate `swm-rl` and start all 15 full-budget runs with:

```bash
conda activate swm-rl
scripts/rl/benchmark_reach_hard.sh
```

The launcher discovers the first two visible GPUs and runs the experiments in
seed waves (`0` through `4`). Within each wave, `none` and `infonce` run
concurrently; after both finish, `lewm` runs. The next seed does not start
until all three methods for the current seed have finished.
Run logs are written under `logs/rl/reacher_hard_taco/`; unique checkpoints
are written under `runs/reacher_hard_taco/`. W&B runs share the
`taco-reacher-hard-3methods-5seeds-ordered-v3` group in `stable-worldmodel-rl`.
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
batch (512 on the local 1080 Ti cards; TACO specifies 1,024), three-frame
84px RGB input, evaluation every 10,000 physics
frames, and 10 evaluation episodes. The Reacher wrapper repeats actions twice
and the trainer caps episodes at 500 agent steps (the 20-second task limit).

The local replay buffer stores full three-frame image stacks in RAM. A 1M
transition capacity would exceed this machine's available memory with two
concurrent jobs, so the config uses 100k entries. All other shared training
settings above are retained. TACO's `curl=true` and `reward=true` auxiliary
heads are objective-specific and are not included in the pure InfoNCE, LeWM,
and baseline comparison.

The default batch size is 512 because a local InfoNCE run at 1,024 exhausted a
1080 Ti's 11 GiB VRAM. The launcher accepts `BATCH_SIZE=...` to override this
on a larger GPU; the chosen batch is shared across all methods.

`lewm` selects a LeWM objective adapter over the shared DrQ pixel encoder: it
uses this repository's LeWM action embedder/predictor and SIGReg with the
LeWM default weight. It is not the standalone LeWM ViT encoder/model. The
InfoNCE variant uses the temporal action-conditioned objective already in
the DrQ agent. The `none` run disables the auxiliary representation loss.

All methods receive the same seeds, replay settings, batch size, optimizer
schedule, action repeat, pixel observations, evaluation schedule, and training
budget. See `scripts/train/config/drqv2_reacher_hard_taco.yaml` for the exact
resolved values.

## Reduced preliminary comparison

Reacher Hard is the first task to screen because the TACO paper reports its
largest difficult-task gain there: 51% over DrQ-v2 in its 1M-step comparison ([paper,
Figure 3](https://proceedings.neurips.cc/paper_files/paper/2023/file/96d00450ed65531ffe2996daed487536-Paper-Conference.pdf)).
Before running the full benchmark, launch the faster paired screen with:

```bash
scripts/rl/prelim_reacher_hard.sh
```

It starts two tmux sessions (`swm-rh-prelim-none` and
`swm-rh-prelim-infonce`) with three matched seeds. Both use 20,000 agent steps,
36px RGB frames, batch size 32, a 2,000-step random-action warmup, and evaluation
every 5,000 steps. The sessions write per-seed logs under
`logs/rl/prelim_reacher_hard_36px/` and checkpoints under
`runs/prelim_reacher_hard_36px/`. This is a low-resolution integration and
early-trend check; its returns are not comparable to the paper's full visual
benchmark or sufficient to claim a performance improvement. The online trainer
currently requires pixels, so this preliminary path does not provide a
state-only run. By default the launcher uses physical GPUs 0 and 5, which passed
EGL rendering preflight on this host; set `GPU_NONE` and `GPU_INFONCE` to choose
different EGL-capable cards.

### Pilot result (2026-10-06)

The three-seed screen completed all six runs without runtime errors, CUDA
out-of-memory errors, or non-finite InfoNCE metrics. Final evaluation returns
at 20,000 steps (two episodes per evaluation) were:

| Seed | Baseline | InfoNCE |
|---:|---:|---:|
| 0 | 0.0 | 0.0 |
| 1 | 0.0 | 1.5 |
| 2 | 38.5 | 36.5 |
| Mean | 12.8 | 12.7 |

The InfoNCE loss fell from about 3.5 after the first update to 0.29–0.68 at the
end of the runs, with finite positive/negative similarity diagnostics. The
similar final returns and large seed variation provide no evidence of a
performance gap at this short, low-resolution budget. Use the full benchmark
for a performance conclusion.
