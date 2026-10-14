# Online RL: SAC, DrQ-v2, TACO, and TACO-MPC

SAC and DrQ-v2 share `scripts/train/rl_online.py` for real collection, replay,
evaluation, and checkpoints. Each agent retains its actor/critic objective.
Here **TACO** means this repository's auxiliary-learning wrapper; its losses
are adaptations of the cited methods. **TACO-MPC** adds planning with those
same trained predictors. It does not allocate another world model.

Use the [audit](rl_repository_audit.md) for loss definitions and limitations,
the [benchmark record](rl_reacher_hard_benchmark.md) for evidence, and the
[SWE handoff](rl_swe_handoff.md) for implementation work.

## Choose the learner and observation

| Config | Learner | Input |
|---|---|---|
| `sac_state` | SAC | Numeric state; Finger Turn Hard preset |
| `sac_pixels` | SAC | Rendered pixels |
| `drqv2_state` | DrQ-v2 | Numeric state |
| `drqv2` | DrQ-v2 | Rendered pixels |

Alternatively compose `rl_online` with `agent=sac` or `agent=drqv2` and
`observation.mode=state` or `pixels`. State mode accepts a numeric Box or a
Box field from Dict observations, selected by `observation.state_key`.
Pixel mode renders RGB, resizes to `observation.image_size` (at least 36), and
stacks `observation.frame_stack` frames. Actions must be a vector Box bounded
in `[-1,1]`. Environment dependencies and a working renderer are required.

Run from the repository root with the `swm-rl` environment active:

```bash
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=sac_state wm=none auxiliary.enabled=false
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=drqv2 wm=none auxiliary.enabled=false
```

These are pure model-free baselines: no auxiliary modules or optimizers are
allocated. `wm=none` alone removes the primary representation model, but
separately enabled clip-reward or pixel CURL losses still change representation
training. Disable the global auxiliary path for the unmodified baseline.

## Choose the auxiliary losses

Set `auxiliary.enabled=true` and select at most one `wm` group. Hydra packages
the group under `auxiliary.wm`; use `auxiliary.wm.horizon`, not `wm.horizon`,
for overrides. Reward prediction and CURL are independent optional add-ons.

| `wm` choice | Objective | Eligible for MPC? |
|---|---|---|
| `none` | No primary auxiliary | No |
| `infonce` | Normalized temporal contrast, action embeddings, bilinear compatibility | Yes, with model horizon 1; composition accuracy is unestablished |
| `lewm` | Action-conditioned causal prediction MSE plus SIGReg | Yes, with context horizon 1 |
| `consistency` | Recurrent one-step latent prediction with detached targets | Yes; model horizon must be 1 |
| `jepa` | Causal prediction MSE, detached future target, no actions | No |
| `pldm` | Action-free alignment plus four variance/covariance penalties | No |

All primary choices accept state and pixel features. These are feature-level
adapters, not the standalone LeWM, DINO-WM, or PLDM models.
`taco`, `prejepa`, and `lewn` are Python selector aliases for InfoNCE, JEPA,
and LeWM; Hydra group names are the table entries.

For state SAC with a temporal-only auxiliary:

```bash
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=sac_state wm=infonce auxiliary.enabled=true auxiliary.sequence_length=3 auxiliary.wm.horizon=3
```

The state preset `sac_taco_state` selects InfoNCE (horizon 3, projection 50),
adds one-step clip-reward prediction, and uses batch 1024. Task variants are
`sac_taco_state_reacher_hard`, `sac_taco_state_acrobot`,
`sac_taco_state_pendulum`, and `sac_taco_metaworld_stick_pull`.
For pixel DrQ-v2 with both add-ons:

```bash
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=drqv2 wm=infonce auxiliary.enabled=true auxiliary.sequence_length=3 auxiliary.wm.horizon=3 auxiliary.reward_prediction.enabled=true auxiliary.reward_prediction.horizon=3 auxiliary.curl.enabled=true
```

Clip reward prediction consumes the start encoder feature and raw action clip;
its target is the discounted replay return. Per-step prediction consumes the
selected WM latent and one action. CURL contrasts independent random shifts
of the same current pixel observation and requires pixels. The canonical path
has no EMA key encoder. Weights live at `auxiliary.wm.weight`,
`auxiliary.reward_prediction.weight`, and `auxiliary.curl.weight`.

The original [TACO paper](https://arxiv.org/abs/2306.13229) and
[authors' code](https://github.com/FrankZheng2022/TACO/blob/main/agents/taco.py)
motivate the temporal objective and add-ons. Local normalization, action
embedding, raw-action critics, and separate add-on projectors are adaptations.

## Enable planning

```bash
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=sac_state wm=infonce auxiliary.enabled=true auxiliary.wm.horizon=1 auxiliary.sequence_length=5 auxiliary.reward_prediction.enabled=true auxiliary.reward_prediction.mode=per_step auxiliary.reward_prediction.horizon=1 mpc.enabled=true mpc.start_steps=10000 mpc.horizon=5
```

Replace `wm=infonce` with `lewm` or `consistency` to use those predictors.
The same settings work with `agent=drqv2` on `rl_online`; task configs may
set an agent override in their defaults.

Three lengths have separate meanings:

- `auxiliary.wm.horizon` is training context/prediction horizon `H_WM`; MPC
  requires 1. This does not force planning to one step.
- `auxiliary.sequence_length` is replay span `U` (default 5). The auxiliary
  clip spans `max(U,H_WM,H_reward)` transitions. Temporal objectives average
  complete windows: `U=H_WM=3` gives one InfoNCE window; `H_WM=1` gives `U`
  one-step windows. Consistency unrolls `U` predictions; PLDM uses the full clip.
- `mpc.horizon` is imagined rollout horizon `K` (default 5), independent of
  replay span and the DrQ-v2 critic's `agent.nstep` horizon.

MPC requires the global auxiliary path, an enabled eligible WM with positive
weight, and an enabled per-step reward head with positive weight and horizon 1.
`none`, JEPA, PLDM, disabled auxiliaries, multi-step endpoint/context models,
clip-return rewards, and zero-weight required predictors fail during setup.
State+CURL, non-finite/negative weights, invalid elite/sample counts, and invalid
MPC numeric settings also fail. Enabled auxiliary learning needs a selected loss.

`mpc.start_steps` delays planning; DrQ-v2 also observes its exploration warmup.
After that threshold, planning raises an error if no auxiliary update completed.
The count proves training occurred, not that the model is accurate. Choose a
delay allowing committed replay and learner updates, especially with large
batches or long commit intervals.

When MPC is enabled, the actor, base critic, target critic, reward model, and
planner share the selected WM latent `z=g(x)` as their state feature. The
original SAC/DrQ-v2 actor and critic losses remain in place, but train on `z` so
the base critic can score planner-predicted latents.

`mpc.value_coef` (default 0.1, denoted \(\gamma_v\) in the method notes) adds
a separate twin two-hot reward-return Q head on `z`. SAC's auxiliary head has
no entropy term, matching reward-only rollout scoring; its base critic retains
the usual soft target. DrQ-v2's auxiliary head uses one-step replay TD targets.
The planner samples a terminal action from the actor. With `mpc.value_coef=0`,
the auxiliary head and loss are omitted and the planner bootstraps from the
original base critic: SAC scores entropy along the imagined actions and at the
terminal bootstrap, while DrQ-v2 uses its reward-only Q value. Zero disables
the extra value loss, not terminal bootstrapping. The empirical benefit of the
separate head and its default coefficient have not been established.

Planning uses shifted previous means, clipped Gaussian candidates, weighted
elite updates, and policy proposals sampled repeatedly at the current latent.
These are open-loop proposals, not TD-MPC2 latent-policy rollouts through
imagined states. Training samples an elite and adds exploration noise;
evaluation executes the best elite's first action. The
[audit comparison](rl_repository_audit.md#planner-comparison) details adaptations
to [TD-MPC2](https://arxiv.org/abs/2310.16828).

## Real interactions and checkpoints

Each training iteration executes one real `env.step`, inserts its transition,
and increments `train/environment_step` once. Imagined transitions only score
actions inside `agent.act`; they never enter replay. Action repeat 2 advances
two control-suite steps per training interaction; record both budgets.
Evaluation uses a separate real environment and does not advance the training
counter or insert replay samples.

Clips stay inside committed episode segments. True termination zeros bootstrap;
time-limit truncation retains it. Planner warm starts reset at episode boundaries
and around evaluation. Replay includes one final-frame sentinel per segment,
so stored rows are not identical to interaction count.

Checkpoints store learner/optimizer state, replay, resolved config, counters,
and RNG state. MPC checkpoints add model-update count and
`mpc_value_objective=reward_td_v1` plus
`mpc_control_features=wm_latent_v1`; earlier MPC checkpoints with encoder-space
critic inputs are rejected because their actor/critic dimensions and meaning do
not match the shared WM-latent planner. Earlier SAC MPC heads learned soft
targets and require retraining for reward-only scoring. Non-MPC checkpoint APIs
remain supported.
Resume restarts the environment episode rather than restoring simulator state.
The legacy trainer/import paths remain; `SACTACOAgent` retains its legacy
TACO-on default.

`auxiliary/loss` logs weighted WM/reward/CURL terms; `mpc/value_loss` and
`mpc/value_weight` separately identify the added value term. Checkpoint/W&B
metadata distinguish algorithm, modality, auxiliaries, and MPC.

## Reproduction and limits

The [benchmark record](rl_reacher_hard_benchmark.md) preserves pilot settings,
seeds, raw-log locations, and corrected visible-target commands. State/Slurm
retests are described in `scripts/rl/slurm/sac_state_retest/README.md`.
Use fresh output directories and matched budgets, batches, update frequency,
replay, evaluation seeds, and add-ons across comparison arms.

Run focused checks with `PYTHONPATH="$PWD" python -m pytest -q tests/rl`.
They establish execution and invariants, not accurate long latent rollouts,
MPC return improvements, or reproduction of source-paper results.
