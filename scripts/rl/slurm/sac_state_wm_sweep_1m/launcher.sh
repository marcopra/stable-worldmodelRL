#!/bin/bash

set -euo pipefail

seeds="11 12 13 14 15"
mpc_enabled="${MPC_ENABLED:-false}"
case "${mpc_enabled,,}" in
  1|true|yes|on)
    wm_losses=(infonce lewm)
    mpc_enabled=true
    ;;
  0|false|no|off)
    wm_losses=(none infonce lewm)
    mpc_enabled=false
    ;;
  *)
    echo "MPC_ENABLED must be true or false, got: ${mpc_enabled}" >&2
    exit 2
    ;;
esac
env_keys=(finger_turn_hard acrobot_swingup reacher_hard pendulum_control)
env_configs=(
  sac_taco_state
  sac_taco_state_acrobot
  sac_taco_state_reacher_hard
  sac_taco_state_pendulum
)

for seed in $seeds; do
  for wm_loss in "${wm_losses[@]}"; do
    for index in "${!env_keys[@]}"; do
      env_key="${env_keys[index]}"
      env_config="${env_configs[index]}"
      sbatch --export="ALL,SEED=${seed},WM_LOSS=${wm_loss},ENV_KEY=${env_key},ENV_CONFIG=${env_config},MPC_ENABLED=${mpc_enabled}" \
        scripts/rl/slurm/sac_state_wm_sweep_1m/base.sh
    done
  done
done
