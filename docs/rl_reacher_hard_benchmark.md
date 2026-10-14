# Reacher hard representation benchmark

Reviewed 2026-10-14. The 2026-10-06 pilot below is historical evidence, not a
result for the current implementation. All six endpoints were checked against
the original logs; those artifacts are retained. State experiment evidence is
preserved in the [audit](rl_repository_audit.md#preserved-state-experiment-evidence).

## Corrected visual task

The Reacher wrapper defaults to a hidden target (`render_target=0`). The archived
pilot config had no visibility override. This changes the observation problem
from standard visual Reacher: the policy cannot see the randomized goal.
The visual preset now passes `env_kwargs.render_target=true`, which persists
across training/evaluation resets. Real MuJoCo probes verified target material
alpha 0/1 for hidden/visible settings and successful 36px RGB rendering.
The general environment default is preserved for other uses.

New full-budget outputs use `reacher_hard_visible_v4`; new pilot outputs use
`prelim_reacher_hard_visible_36px_v2`. Do not pool them with hidden-target data.
Current DrQ-v2 also restores the reference straight-through action clamp and
explicitly uses the configured auxiliary replay span.

## Full-budget recipe (not launched by this review)

From the repository root:

```bash
conda activate swm-rl
scripts/rl/benchmark_reach_hard.sh
```

The launcher discovers the first two GPUs and queues seeds 0–4. For each seed,
`none` and `infonce` run concurrently, then `lewm` runs before the next seed.
It preflights MuJoCo EGL rendering on both cards and binds each run's renderer
to its assigned physical GPU. The 15 runs write logs to
`logs/rl/reacher_hard_visible_v4/`, checkpoints to
`runs/reacher_hard_visible_v4/`, and W&B group
`taco-reacher-hard-visible-3methods-5seeds-v4` in `stable-worldmodel-rl`.

The local upstream `TACO/cfgs/task/reacher_hard.yaml` inherits `medium`, not
`hard`: 3.1M control-suite frames, action repeat 2, hence 1.55M agent interactions.
These are control steps, not MuJoCo integration substeps.
The local preset `scripts/train/config/drqv2_reacher_hard_taco.yaml` uses:

| Setting | Local full benchmark |
|---|---|
| Budget, repeat, episode cap | 1,550,000 real interactions; repeat 2; 500 agent steps |
| Observations | Three 84px RGB frames; target visible |
| Discount, tau, learning rate | 0.99; 0.01; 1e-4 |
| Feature/hidden dimensions | 50/1024 |
| Exploration | Std 1.0 to 0.1 over 500k agent steps; 2000 random-action steps |
| Critic / update schedule | n-step 3; one update every two real steps |
| WM horizon / replay span | H=3; U=3; one complete temporal window |
| Batch / replay capacity | 512 launcher default; 100k entries |
| Evaluation | Every 5000 agent steps (10k control-suite frames), 10 episodes |
| Seeds / methods | 0–4; none, InfoNCE, LeWM; no MPC |

The YAML preset has batch 1024 and reward/CURL enabled. The launcher overrides
batch to 512, disables both add-ons, and selects only one primary objective,
keeping the primary-loss comparison isolated. Its corrected Hydra path is
`auxiliary.wm.horizon=3`. `BATCH_SIZE=...` changes the batch for every method.
A historical 1024-batch InfoNCE run exhausted a 1080 Ti's 11 GiB VRAM; 512 was
chosen for that host. Replay stores complete frame stacks in RAM: 1M entries
would exceed the local concurrent-run memory budget, so capacity is 100k.

`none` disables auxiliaries; InfoNCE uses normalized bilinear contrast;
`lewm` uses the shared DrQ encoder with LeWM action/prediction modules,
projection 192, hidden 2048, and SIGReg weight 0.09. It does not use the standalone
LeWM ViT. All arms share seeds, batch, replay, action repeat, evaluation and
training schedules. Save resolved configs and source revisions. The local
rendering, raw-action critic, contrastive normalization, replay, and adapters
still differ from upstream TACO; matching settings is not exact reproduction.

## Reduced visible-target screen

```bash
scripts/rl/prelim_reacher_hard.sh
```

The launcher starts tmux sessions `swm-rh-prelim-none` and
`swm-rh-prelim-infonce`, each with seeds 0–2. It uses 20k agent steps, three
36px RGB frames, batch 32, feature/hidden dimensions 32/256, n-step 3, update
frequency 2, replay 30k, commit 500, horizon/span 3, random warmup 2000,
evaluation every 5k with two episodes, and no reward/CURL/MPC.
It writes new logs under `logs/rl/prelim_reacher_hard_visible_36px_v2/` and
checkpoints under `runs/prelim_reacher_hard_visible_36px_v2/`.
Defaults are GPUs 0 and 5; override with `GPU_NONE` and `GPU_INFONCE`.
This low-resolution integration screen cannot establish paper-scale improvement.
The [TACO paper, Figure 3](https://proceedings.neurips.cc/paper_files/paper/2023/file/96d00450ed65531ffe2996daed487536-Paper-Conference.pdf)
reports a 51% Reacher Hard improvement for the ablation without action
representation, relative to DrQ-v2. That published result motivated screening;
it is not the full TACO result or a claim about this adapter.

## Preserved hidden-target pilot (2026-10-06)

Original logs remain in `logs/rl/prelim_reacher_hard_36px/`, named
`reacher-hard-36px-{none,infonce}-seed-{0,1,2}.log`; checkpoints remain under
`runs/prelim_reacher_hard_36px/`. A resolved InfoNCE seed-0 config is
`outputs/2026-10-06/15-14-19/.hydra/config.yaml`.
It records the screen settings above, temperature 0.1, projection 192,
discount 0.99, target tau 0.01, lr 1e-4, augmentation pad 4, episode cap 500,
and exploration std 1.0/0.1 over 500k agent steps. It predates the current
configuration schema and has no target-visibility override. The raw logs use
`representation/*` metrics rather than today's `world_model/*` names.

All six runs reached 20k steps without recorded runtime/OOM/non-finite InfoNCE
errors. Final evaluation means, over two episodes each, were:

| Seed | Baseline | InfoNCE |
|---:|---:|---:|
| 0 | 0.0 | 0.0 |
| 1 | 0.0 | 1.5 |
| 2 | 38.5 | 36.5 |
| Mean | 12.8 | 12.7 |

InfoNCE decreased from about 3.5 to 0.29–0.68 with finite similarity metrics.
The similar endpoint means, three seeds, two evaluation episodes, short budget,
and hidden goal give no reliable evidence of a performance improvement.
A smaller contrastive loss alone does not establish control-relevant dynamics.

The old invocation was the then-current `scripts/rl/prelim_reacher_hard.sh`,
using `scripts/train/drqv2_online.py --config-name=drqv2_reacher_hard_taco`.
The current launcher runs a corrected task. To approximate the old task with
current code, use the recorded settings and `env_kwargs.render_target=false`,
with a fresh checkpoint directory; label it a reimplementation, not an exact
historical reproduction. Exact reproduction requires the historical source
and resolved config. Existing logs/checkpoints/configs were not overwritten.
No corrected full-budget return or MPC advantage is established.
