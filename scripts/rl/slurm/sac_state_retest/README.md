# SAC state TACO and MPC retests (Slurm)

Run the launchers from the repository root:

```bash
bash scripts/rl/slurm/sac_state_retest/launcher_taco.sh
bash scripts/rl/slurm/sac_state_retest/launcher_mpc.sh
```

The TACO launcher submits 15 non-MPC InfoNCE runs: five seeds (11–15) on
Finger Turn Hard, Acrobot Swingup, and Reacher Hard. It uses each existing
`sac_taco_state` task preset with a 1M-step budget and batch size 1024,
three-step InfoNCE objective and one-step clip-return reward prediction. It
sets `auxiliary.sequence_length=3` by default, so InfoNCE uses one
three-transition window, matching the previous temporal clip sampling. The
updated reward head trains on each of the three one-step windows. Set
`TACO_SEQUENCE_LENGTH=5` to average InfoNCE over three overlapping windows and
the reward head over five one-step windows.
The already-run model-free SAC baseline is not resubmitted.

The MPC launcher submits 45 runs across five seeds, the same three tasks, and
the supported InfoNCE, LeWM, and consistency models. It uses a one-step model,
per-step reward prediction, replay sequence length `U=5`, and planner horizon
`K=5`. Its common training settings are 1M environment steps and batch size
256. The TACO and MPC families use separate versioned W&B groups and checkpoint
directories so they do not overwrite prior runs.

The corrected MPC family is `sac-state-mpc-loss-retest-reward-td-v3`: its
separate terminal head learns reward-only TD targets while SAC retains its
soft critic. Old MPC checkpoints lack the new model-update/objective metadata
and are rejected. The recipe uses open-loop base-policy proposals, not a
TD-MPC2 latent policy. Match the TACO/MPC batch size, replay, update frequency,
evaluation schedule, and predictors before interpreting a difference as a
planning benefit; the default batches 1024/256 are different.

Useful overrides can be exported before launching:

```bash
MPC_HORIZON=8 MPC_SEQUENCE_LENGTH=5 \
  bash scripts/rl/slurm/sac_state_retest/launcher_mpc.sh

TACO_SEQUENCE_LENGTH=5 \
  bash scripts/rl/slurm/sac_state_retest/launcher_taco.sh
```

The MPC launcher also accepts `MPC_START_STEPS`, `MPC_VALUE_COEF`,
`MPC_NUM_SAMPLES`, `MPC_NUM_ELITES`, `MPC_ITERATIONS`, and
`MPC_NUM_PI_TRAJS`. LeWM additionally accepts `MPC_LEWM_HIDDEN_DIM` and
`MPC_LEWM_SIGREG_NUM_PROJ`, which default to 2048 and 1024. Add
`MPC_RUN_SUFFIX` when changing settings so the run group and checkpoints remain
separate from the full-size results.

For a faster screening run, for example:

```bash
MPC_RUN_SUFFIX=screen \
MPC_NUM_SAMPLES=128 MPC_ITERATIONS=3 MPC_HORIZON=3 MPC_NUM_PI_TRAJS=8 \
MPC_LEWM_HIDDEN_DIM=512 MPC_LEWM_SIGREG_NUM_PROJ=256 \
  bash scripts/rl/slurm/sac_state_retest/launcher_mpc.sh
```

This cuts planned latent transitions per action from 15,360 (`512 x 6 x 5`)
to 1,152 (`128 x 3 x 3`), before accounting for fixed training and evaluation
costs. The smaller LeWM and SIGReg settings also reduce learner-update cost,
but change model capacity and regularization, so use the full-size defaults for
final comparable results. `WANDB_ENABLE`, `WANDB_ENTITY`, and `WANDB_PROJECT`
control W&B logging; logging is enabled by default to match the existing Slurm
runs.
