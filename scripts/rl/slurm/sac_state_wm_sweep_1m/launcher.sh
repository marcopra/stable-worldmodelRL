#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${repo_root}"

seeds=(11 12 13 14 15)
wm_losses=(none infonce lewm) # jepa pldm)
env_keys=(finger_turn_hard acrobot_swingup reacher_hard pendulum_control)
env_configs=(
  sac_taco_state
  sac_taco_state_acrobot
  sac_taco_state_reacher_hard
  sac_taco_state_pendulum
)
base_script="scripts/rl/slurm/sac_state_wm_sweep_1m/base.sh"
log_dir="logs/rl/slurm/sac_state_wm_sweep_1m"
mkdir -p "${log_dir}"

for seed in "${seeds[@]}"; do
  for wm_loss in "${wm_losses[@]}"; do
    for index in "${!env_keys[@]}"; do
      env_key="${env_keys[index]}"
      env_config="${env_configs[index]}"
      job_name="sac-${wm_loss}-${env_key}-s${seed}"
      echo "Submitting ${job_name}"
      sbatch \
        --job-name="${job_name}" \
        --output="${log_dir}/%x-%j.out" \
        --error="${log_dir}/%x-%j.err" \
        --export="ALL,SEED=${seed},WM_LOSS=${wm_loss},ENV_KEY=${env_key},ENV_CONFIG=${env_config}" \
        "${base_script}"
    done
  done
done

echo "Submitted $(( ${#seeds[@]} * ${#wm_losses[@]} * ${#env_keys[@]} )) jobs."
echo "Experiment: sac-state-wm-sweep-1m-slurm-v1"
echo "Slurm output: ${log_dir}/"
