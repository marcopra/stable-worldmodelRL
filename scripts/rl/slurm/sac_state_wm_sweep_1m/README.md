# SAC state WM sweep (Slurm)

From the repository root, submit with:

```bash
bash scripts/rl/slurm/sac_state_wm_sweep_1m/launcher.sh
```

The launcher loops over seeds, WM loss settings, and environments, then
submits one `base.sh` job per combination. Defaults here are seeds 11–15,
`none`/InfoNCE/LeWM, and Finger Turn Hard, Acrobot Swingup, Reacher Hard, and
Pendulum Control (100 jobs).

The four environments use state observations. WM jobs also enable reward
prediction (weight 1, horizon 3); CURL is disabled. Each job trains for 1M
steps, writes Slurm output to `%j.out` and `%j.err` in the submission
 directory, and saves checkpoints under `runs/rl/sac-state-wm-sweep-1m-slurm-v1/`.

The same cluster job supports optional model-predictive control. Set
`MPC_ENABLED=true` before launching to submit the WM-backed MPC variants:

```bash
MPC_ENABLED=true bash scripts/rl/slurm/sac_state_wm_sweep_1m/launcher.sh
```

In MPC mode the launcher excludes `none` because it has no learned dynamics
for planning. It uses a five-step planner horizon, 0.1 value-prediction loss
weight, 512 candidates, 64 elites, and six planner iterations. The WM and
reward-prediction horizons are set to the planner horizon. Override these
defaults with `MPC_HORIZON`, `MPC_VALUE_COEF`, `MPC_NUM_SAMPLES`,
`MPC_NUM_ELITES`, `MPC_ITERATIONS`, `MPC_NUM_PI_TRAJS`, or
`MPC_START_STEPS`. MPC runs get a separate run and checkpoint directory with
the `-mpc` suffix.
