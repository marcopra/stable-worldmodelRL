#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpuv

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${repo_root}"

if [[ -z "${SEED:-}" || -z "${WM_LOSS:-}" || -z "${ENV_KEY:-}" || -z "${ENV_CONFIG:-}" ]]; then
  echo 'SEED, WM_LOSS, ENV_KEY, and ENV_CONFIG must be provided by launcher.sh.' >&2
  exit 2
fi

source ~/.bashrc
conda activate "${CONDA_ENV:-swm-rl}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE="online"

run_id="sac-state-wm-sweep-1m-slurm-v1"
run_name="${run_id}-${ENV_KEY}-${WM_LOSS}-seed-${SEED}"
run_group="${run_id}-${ENV_KEY}"
checkpoint_dir="${repo_root}/runs/rl/${run_id}/${ENV_KEY}/${WM_LOSS}/seed_${SEED}"
mkdir -p "${checkpoint_dir}"

auxiliary_overrides=(
  auxiliary.enabled=true
  auxiliary.wm.enabled=true
  auxiliary.wm.weight=1.0
  auxiliary.wm.horizon=3
  auxiliary.reward_prediction.enabled=true
  auxiliary.reward_prediction.weight=1.0
  auxiliary.reward_prediction.horizon=3
  auxiliary.curl.enabled=false
)
case "${WM_LOSS}" in
  none)
    auxiliary_overrides=(
      auxiliary.enabled=false
      auxiliary.wm.enabled=false
      auxiliary.reward_prediction.enabled=false
      auxiliary.curl.enabled=false
    )
    ;;
  infonce)
    auxiliary_overrides+=(
      auxiliary.wm.projection_dim=128
      auxiliary.wm.hidden_dim=256
      auxiliary.wm.temperature=0.1
    )
    ;;
  lewm)
    auxiliary_overrides+=(
      auxiliary.wm.projection_dim=192
      auxiliary.wm.lewm_hidden_dim=2048
      auxiliary.wm.sigreg_weight=0.09
    )
    ;;
  jepa)
    auxiliary_overrides+=(auxiliary.wm.projection_dim=128)
    ;;
  pldm)
    auxiliary_overrides+=(auxiliary.wm.projection_dim=128)
    ;;
  *)
    echo "Unsupported WM loss: ${WM_LOSS}" >&2
    exit 2
    ;;
esac

python scripts/train/rl_online.py \
  --config-name="${ENV_CONFIG}" \
  "seed=${SEED}" \
  device=cuda:0 \
  num_steps=1000000 \
  learning_starts=5000 \
  batch_size=256 \
  updates_per_step=1 \
  replay.capacity=100000 \
  replay.commit_interval=1000 \
  evaluation.frequency=50000 \
  evaluation.episodes=3 \
  logging.frequency=1000 \
  checkpoint_frequency=100000 \
  "checkpoint_dir=${checkpoint_dir}" \
  "wm=${WM_LOSS}" \
  "wandb.enable=true" \
  "wandb.entity=marcopra" \
  "wandb.project=swm-test" \
  "wandb.group=${run_group}" \
  "wandb.name=${run_name}" \
  "wandb.tags=[state,sac,${ENV_KEY},${WM_LOSS},1m,slurm]" \
  "${auxiliary_overrides[@]}"
