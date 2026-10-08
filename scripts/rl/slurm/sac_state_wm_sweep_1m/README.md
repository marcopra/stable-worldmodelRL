# SAC state WM sweep (1M, Slurm)

Submit the complete matrix from the repository root with:

```bash
bash scripts/rl/slurm/sac_state_wm_sweep_1m/launcher.sh
```

The outer launcher submits one Slurm job for each seed, selected world-model
loss, and environment. `base.sh` is the inner job script that runs one SAC
experiment. The default matrix is 5 seeds (1–5), five WM settings
(`none`, `infonce`, `lewm`, `jepa`, `pldm`), and four environments, for 100
jobs total.

All non-baseline WM jobs also enable reward prediction with weight 1 and
horizon 3, matching the prior InfoNCE setup. CURL is disabled because this
sweep uses state observations. Each run uses 1M steps and logs to the W&B
project `marcopra/swm-test`.

Environments:

- Finger Turn Hard (`swm/FingerDMControl-v0`)
- Acrobot Swingup (`swm/AcrobotDMControl-v0`)
- Reacher Hard (`swm/ReacherDMControl-v0`)
- Pendulum Control (`swm/PendulumControl-v1`), the additional test task

Slurm output is written under `logs/rl/slurm/sac_state_wm_sweep_1m/`; model
checkpoints are written under `runs/rl/sac-state-wm-sweep-1m-slurm-v1/`.
The conda environment defaults to `swm-rl` and can be changed with
`CONDA_ENV` before submission.
