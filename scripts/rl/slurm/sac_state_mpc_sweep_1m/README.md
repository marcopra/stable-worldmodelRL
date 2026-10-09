# SAC state MPC sweep (Slurm)

Submit from the repository root with:

```bash
bash scripts/rl/slurm/sac_state_mpc_sweep_1m/launcher.sh
```

The launcher submits one job for each of five seeds (11–15), InfoNCE and LeWM
world-model losses, and Finger Turn Hard, Acrobot Swingup, and Reacher Hard
(30 jobs total). A `none` WM is intentionally excluded because it is the
purely model-free baseline and cannot provide latent rollouts for MPC. Every
run enables MPC and trains for 1M real environment steps. MPC reuses the
selected WM predictor and the existing reward-prediction head; imagined
transitions do not call `env.step()` or advance the environment-step counter.
The TD-MPC2-style value-prediction loss is weighted by 0.1 (set
`mpc.value_coef=0` to disable its loss and terminal bootstrap). CURL remains
disabled.

Each job writes Slurm output to `%j.out` and `%j.err`, W&B runs under
`sac-state-mpc-sweep-1m-slurm-v1`, and checkpoints under
`runs/rl/sac-state-mpc-sweep-1m-slurm-v1/`.
