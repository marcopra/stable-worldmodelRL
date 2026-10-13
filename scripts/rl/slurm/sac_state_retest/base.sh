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

cd "${SLURM_SUBMIT_DIR:?Run this job from a Slurm submission directory}"
source ~/.bashrc
conda activate swm-rl
set -u

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mode="${MODE:?MODE must be taco or mpc}"
seed="${SEED:?SEED is required}"
wm_loss="${WM_LOSS:?WM_LOSS is required}"
env_key="${ENV_KEY:?ENV_KEY is required}"
env_config="${ENV_CONFIG:?ENV_CONFIG is required}"
wandb_enable="${WANDB_ENABLE:-true}"
wandb_entity="${WANDB_ENTITY:-marcopra}"
wandb_project="${WANDB_PROJECT:-swm-test}"

common_overrides=(
  seed="${seed}"
  device=cuda:0
  logging.frequency=1000
  wandb.enable="${wandb_enable}"
  wandb.entity="${wandb_entity}"
  wandb.project="${wandb_project}"
)

case "${mode}" in
  taco)
    if [[ "${wm_loss}" != infonce ]]; then
      echo "TACO retest requires WM_LOSS=infonce, got ${wm_loss}" >&2
      exit 2
    fi
    sequence_length="${TACO_SEQUENCE_LENGTH:-3}"
    auxiliary_overrides=(
      auxiliary.enabled=true
      auxiliary.sequence_length="${sequence_length}"
      auxiliary.wm.enabled=true
      auxiliary.wm.horizon="${TACO_WM_HORIZON:-3}"
      auxiliary.reward_prediction.enabled=true
      auxiliary.reward_prediction.mode=clip_return
      auxiliary.curl.enabled=false
      mpc.enabled=false
      num_steps="${TACO_NUM_STEPS:-1000000}"
      batch_size="${TACO_BATCH_SIZE:-1024}"
      learning_starts=5000
      checkpoint_frequency=50000
      evaluation.frequency=25000
      evaluation.episodes=5
    )
    run_tag=taco_retest
    run_id=sac-state-taco-retest-new-implementation-v2
    ;;
  mpc)
    case "${wm_loss}" in
      infonce|lewm|consistency) ;;
      *)
        echo "MPC WM_LOSS must be infonce, lewm, or consistency; got ${wm_loss}" >&2
        exit 2
        ;;
    esac
    mpc_horizon="${MPC_HORIZON:-5}"
    sequence_length="${MPC_SEQUENCE_LENGTH:-5}"
    auxiliary_overrides=(
      replay.capacity=100000
      replay.commit_interval=1000
      auxiliary.enabled=true
      auxiliary.sequence_length="${sequence_length}"
      auxiliary.wm.enabled=true
      auxiliary.wm.weight=1.0
      auxiliary.wm.horizon=1
      auxiliary.reward_prediction.enabled=true
      auxiliary.reward_prediction.mode=per_step
      auxiliary.reward_prediction.horizon=1
      auxiliary.reward_prediction.weight=1.0
      auxiliary.curl.enabled=false
      mpc.enabled=true
      mpc.start_steps="${MPC_START_STEPS:-10000}"
      mpc.horizon="${mpc_horizon}"
      mpc.value_coef="${MPC_VALUE_COEF:-0.1}"
      mpc.num_samples="${MPC_NUM_SAMPLES:-512}"
      mpc.num_elites="${MPC_NUM_ELITES:-64}"
      mpc.iterations="${MPC_ITERATIONS:-6}"
      mpc.num_pi_trajs="${MPC_NUM_PI_TRAJS:-32}"
      num_steps="${MPC_NUM_STEPS:-1000000}"
      batch_size="${MPC_BATCH_SIZE:-256}"
      learning_starts=5000
      checkpoint_frequency=100000
      evaluation.frequency=50000
      evaluation.episodes=3
    )
    case "${wm_loss}" in
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
          auxiliary.wm.lewm_hidden_dim="${MPC_LEWM_HIDDEN_DIM:-2048}"
          auxiliary.wm.sigreg_num_proj="${MPC_LEWM_SIGREG_NUM_PROJ:-1024}"
          auxiliary.wm.sigreg_weight=0.09
        )
        ;;
      consistency)
        auxiliary_overrides+=(
          auxiliary.wm.projection_dim=128
          auxiliary.wm.hidden_dim=256
          auxiliary.wm.rho=0.5
        )
        ;;
    esac
    run_tag=mpc_retest
    run_id=sac-state-mpc-loss-retest-v2
    ;;
  *)
    echo "MODE must be taco or mpc, got ${mode}" >&2
    exit 2
    ;;
esac

if [[ "${mode}" == mpc && -n "${MPC_RUN_SUFFIX:-}" ]]; then
  run_id="${run_id}-${MPC_RUN_SUFFIX}"
fi

common_overrides+=(
  checkpoint_dir="./runs/rl/${run_id}/${env_key}/${wm_loss}/seed_${seed}"
  wandb.group="${run_id}-${env_key}"
  wandb.name="${run_id}-${env_key}-${wm_loss}-seed-${seed}"
  "wandb.tags=[state,sac,${run_tag},${env_key},${wm_loss},slurm]"
)

python scripts/train/rl_online.py \
  --config-name="${env_config}" \
  wm="${wm_loss}" \
  "${common_overrides[@]}" \
  "${auxiliary_overrides[@]}"
