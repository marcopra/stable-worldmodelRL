# SAC state WM sweep (Slurm)

From the repository root, submit with:

```bash
bash scripts/rl/slurm/sac_state_wm_sweep_1m/launcher.sh
```

The launcher follows the pattern in `scripts/rl/slurm/example`: it loops over
seeds, WM loss settings, and environments, then submits one `base.sh` job per
combination. Defaults here are seeds 11–15, `none`/InfoNCE/LeWM, and Finger
Turn Hard, Acrobot Swingup, Reacher Hard, and Pendulum Control (100 jobs).

The four environments use state observations. WM jobs also enable reward
prediction (weight 1, horizon 3); CURL is disabled. Each job trains for 1M
steps, writes Slurm output to `%j.out` and `%j.err` in the submission
 directory, and saves checkpoints under `runs/rl/sac-state-wm-sweep-1m-slurm-v1/`.
