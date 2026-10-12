# SAC state TACO and MPC retests (Slurm)

Run the launchers from the repository root:

```bash
bash scripts/rl/slurm/sac_state_retest/launcher_taco.sh
bash scripts/rl/slurm/sac_state_retest/launcher_mpc.sh
```

The TACO launcher submits 15 non-MPC InfoNCE runs: five seeds (11–15) on
Finger Turn Hard, Acrobot Swingup, and Reacher Hard. It uses each existing
`sac_taco_state` task preset, including its 500K-step, batch-1024 settings,
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

Useful overrides can be exported before launching:

```bash
MPC_HORIZON=8 MPC_SEQUENCE_LENGTH=5 \
  bash scripts/rl/slurm/sac_state_retest/launcher_mpc.sh

TACO_SEQUENCE_LENGTH=5 \
  bash scripts/rl/slurm/sac_state_retest/launcher_taco.sh
```

The MPC launcher also accepts `MPC_START_STEPS`, `MPC_VALUE_COEF`,
`MPC_NUM_SAMPLES`, `MPC_NUM_ELITES`, `MPC_ITERATIONS`, and
`MPC_NUM_PI_TRAJS`. `WANDB_ENABLE`, `WANDB_ENTITY`, and `WANDB_PROJECT` control
W&B logging; logging is enabled by default to match the existing Slurm runs.
