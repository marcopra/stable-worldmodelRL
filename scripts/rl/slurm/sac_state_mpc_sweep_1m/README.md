# SAC state MPC sweep (Slurm)

Submit from the repository root with:

```bash
bash scripts/rl/slurm/sac_state_mpc_sweep_1m/launcher.sh
```

The launcher submits one job for each of five seeds (11–15), InfoNCE, LeWM,
and consistency world-model losses, and Finger Turn Hard, Acrobot Swingup, and
Reacher Hard (45 jobs total). A `none` WM is intentionally excluded because it
is the purely model-free baseline and cannot provide latent rollouts for MPC. Every
run enables MPC and trains for 1M real environment steps. MPC reuses the
selected one-step WM and a per-step reward head. `auxiliary.sequence_length`
controls replay-sequence training independently of `mpc.horizon` (defaulting
to the planner horizon); set `MPC_SEQUENCE_LENGTH` to choose another `U`.
Imagined transitions do not call `env.step()` or advance the environment-step counter.
The separate two-hot reward-return value loss is weighted by 0.1 (set
`mpc.value_coef=0` to disable its loss and use the base critic for terminal
bootstrapping). CURL remains
disabled.

Each job writes Slurm output to `%j.out` and `%j.err`, W&B runs under
`sac-state-mpc-sweep-1m-slurm-reward-td-v2`, and checkpoints under
`runs/rl/sac-state-mpc-sweep-1m-slurm-reward-td-v2/`.
The SAC critic keeps its soft target. Earlier MPC heads require retraining;
checkpoints without model-update count and reward-target metadata are rejected.
This recipe is not an exact TD-MPC2 reproduction and has no established
performance result for the corrected implementation.
