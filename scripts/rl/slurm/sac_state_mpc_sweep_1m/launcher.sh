#!/usr/bin/env bash

set -euo pipefail

seeds=(11 12 13 14 15)
wm_losses=(infonce lewm)
env_keys=(finger_turn_hard acrobot_swingup reacher_hard)
env_configs=(
  sac_taco_state
  sac_taco_state_acrobot
  sac_taco_state_reacher_hard
)

for seed in "${seeds[@]}"; do
  for wm_loss in "${wm_losses[@]}"; do
    for index in "${!env_keys[@]}"; do
      env_key="${env_keys[index]}"
      env_config="${env_configs[index]}"
      sbatch --export="ALL,SEED=${seed},WM_LOSS=${wm_loss},ENV_KEY=${env_key},ENV_CONFIG=${env_config}" \
        scripts/rl/slurm/sac_state_mpc_sweep_1m/base.sh
    done
  done
done
