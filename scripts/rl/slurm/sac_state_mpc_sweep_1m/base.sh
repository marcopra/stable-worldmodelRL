#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate swm-rl
set -u

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

run_id="sac-state-mpc-sweep-1m-slurm-v1"
run_name="${run_id}-${ENV_KEY}-${WM_LOSS}-seed-${SEED}"
run_group="${run_id}-${ENV_KEY}"
checkpoint_dir="./runs/rl/${run_id}/${ENV_KEY}/${WM_LOSS}/seed_${SEED}"

auxiliary_overrides=(
  auxiliary.enabled=true
  auxiliary.wm.enabled=true
  auxiliary.wm.weight=1.0
  auxiliary.wm.horizon=5
  auxiliary.reward_prediction.enabled=true
  auxiliary.reward_prediction.weight=1.0
  auxiliary.reward_prediction.horizon=5
  auxiliary.curl.enabled=false
)
case "${WM_LOSS}" in
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
  *)
    echo "Unsupported WM_LOSS: ${WM_LOSS}" >&2
    exit 2
    ;;
esac

python scripts/train/rl_online.py \
  --config-name="${ENV_CONFIG}" \
  seed="${SEED}" \
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
  checkpoint_dir="${checkpoint_dir}" \
  wm="${WM_LOSS}" \
  mpc.enabled=true \
  mpc.start_steps=10000 \
  mpc.horizon=5 \
  mpc.value_coef=0.1 \
  mpc.num_samples=512 \
  mpc.num_elites=64 \
  mpc.iterations=6 \
  mpc.num_pi_trajs=32 \
  wandb.enable=true \
  wandb.entity=marcopra \
  wandb.project=swm-test \
  wandb.group="${run_group}" \
  wandb.name="${run_name}" \
  "wandb.tags=[state,sac,mpc,${ENV_KEY},${WM_LOSS},1m,slurm]" \
  "${auxiliary_overrides[@]}"
