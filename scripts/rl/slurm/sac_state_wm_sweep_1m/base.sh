#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpua

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:?}"

MPC_ENABLED="${MPC_ENABLED:-false}"
case "${MPC_ENABLED,,}" in
  1|true|yes|on) MPC_ENABLED=true ;;
  0|false|no|off) MPC_ENABLED=false ;;
  *)
    echo "MPC_ENABLED must be true or false, got: ${MPC_ENABLED}" >&2
    exit 2
    ;;
esac

# Load environment
source ~/.bashrc
conda activate swm-rl
set -u

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

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
case ${WM_LOSS} in
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
esac

run_id="sac-state-wm-sweep-1m-slurm-v1"
if [[ "${MPC_ENABLED}" == true ]]; then
  case "${WM_LOSS}" in
    infonce|lewm) ;;
    *)
      echo "MPC requires an action-conditioned WM (infonce or lewm); got WM_LOSS=${WM_LOSS}" >&2
      exit 2
      ;;
  esac
  run_id="${run_id}-mpc"
  mpc_horizon="${MPC_HORIZON:-5}"
  mpc_overrides=(
    mpc.enabled=true
    mpc.start_steps="${MPC_START_STEPS:-10000}"
    mpc.horizon="${mpc_horizon}"
    mpc.value_coef="${MPC_VALUE_COEF:-0.1}"
    mpc.num_samples="${MPC_NUM_SAMPLES:-512}"
    mpc.num_elites="${MPC_NUM_ELITES:-64}"
    mpc.iterations="${MPC_ITERATIONS:-6}"
    mpc.num_pi_trajs="${MPC_NUM_PI_TRAJS:-32}"
  )
  auxiliary_overrides+=(
    auxiliary.wm.horizon="${mpc_horizon}"
    auxiliary.reward_prediction.horizon="${mpc_horizon}"
  )
  mpc_tag=mpc
else
  mpc_overrides=()
  mpc_tag=no_mpc
fi

run_name="${run_id}-${ENV_KEY}-${WM_LOSS}-seed-${SEED}"
run_group="${run_id}-${ENV_KEY}"
checkpoint_dir="./runs/rl/${run_id}/${ENV_KEY}/${WM_LOSS}/seed_${SEED}"

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
  wandb.enable=true \
  wandb.entity=marcopra \
  wandb.project=swm-test \
  wandb.group="${run_group}" \
  wandb.name="${run_name}" \
  "wandb.tags=[state,sac,${ENV_KEY},${WM_LOSS},1m,slurm,${mpc_tag}]" \
  "${mpc_overrides[@]}" \
  "${auxiliary_overrides[@]}"
