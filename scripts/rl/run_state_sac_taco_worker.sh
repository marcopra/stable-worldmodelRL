#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 <physical-gpu> <shared-jobs.tsv>" >&2
  exit 2
fi

gpu="$1"
job_file="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_id="${RUN_ID:-sac_state_3env_1m_v1}"
num_steps="${NUM_STEPS:-1000000}"
evaluation_frequency="${EVALUATION_FREQUENCY:-50000}"
checkpoint_frequency="${CHECKPOINT_FREQUENCY:-100000}"
log_root="${repo_root}/logs/rl/${run_id}"
run_root="${repo_root}/runs/rl/${run_id}"
cursor_file="${job_file}.cursor"
lock_file="${job_file}.lock"
conda_env="${CONDA_ENV:-swm-rl}"
wandb_entity="marcopra"
wandb_project="swm-test"
wandb_base_url="https://api.wandb.ai"

claim_job() {
  local lock_fd
  exec {lock_fd}>"${lock_file}"
  flock -x "${lock_fd}"

  local cursor line_number
  cursor="$(<"${cursor_file}")"
  line_number=$((cursor + 1))
  claimed_job="$(sed -n "${line_number}p" "${job_file}")"
  if [[ -n "${claimed_job}" ]]; then
    printf '%s\n' "${line_number}" >"${cursor_file}.tmp.$$"
    mv "${cursor_file}.tmp.$$" "${cursor_file}"
  fi

  flock -u "${lock_fd}"
  exec {lock_fd}>&-
}

run_job() {
  local task="$1"
  local method="$2"
  local seed="$3"
  local config group run_name log_path checkpoint_dir
  local auxiliary_overrides=()

  case "${task}" in
    finger-turn-hard)
      config=sac_taco_state
      ;;
    acrobot-swingup)
      config=sac_taco_state_acrobot
      ;;
    reacher-hard)
      config=sac_taco_state_reacher_hard
      ;;
    *)
      echo "Unknown task in experiment queue: ${task}" >&2
      return 2
      ;;
  esac

  case "${method}" in
    baseline)
      auxiliary_overrides=(
        auxiliary.enabled=false
        wm=none
        auxiliary.wm.enabled=false
        auxiliary.reward_prediction.enabled=false
        auxiliary.curl.enabled=false
      )
      ;;
    taco)
      auxiliary_overrides=(
        auxiliary.enabled=true
        wm=infonce
        auxiliary.wm.enabled=true
        auxiliary.wm.weight=1.0
        auxiliary.wm.horizon=3
        auxiliary.reward_prediction.enabled=true
        auxiliary.reward_prediction.weight=1.0
        auxiliary.reward_prediction.horizon=3
        auxiliary.curl.enabled=false
      )
      ;;
    *)
      echo "Unknown method in experiment queue: ${method}" >&2
      return 2
      ;;
  esac

  group="state-sac-taco-${task}-1m-v1"
  run_name="state-sac-${task}-${method}-seed-${seed}-1m-v1"
  log_path="${log_root}/${task}/${method}/seed_${seed}.log"
  checkpoint_dir="${run_root}/${task}/${method}/seed_${seed}"
  mkdir -p "$(dirname "${log_path}")" "${checkpoint_dir}"

  echo "[GPU ${gpu}] starting ${run_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    WANDB_BASE_URL="${wandb_base_url}" WANDB_MODE=online \
    conda run --no-capture-output -n "${conda_env}" \
      python scripts/train/rl_online.py \
      --config-name="${config}" \
      "seed=${seed}" \
      device=cuda:0 \
      "num_steps=${num_steps}" \
      learning_starts=5000 \
      batch_size=256 \
      updates_per_step=1 \
      max_episode_steps=500 \
      replay.capacity=100000 \
      replay.commit_interval=1000 \
      "evaluation.frequency=${evaluation_frequency}" \
      evaluation.episodes=3 \
      logging.frequency=1000 \
      "checkpoint_frequency=${checkpoint_frequency}" \
      agent.feature_dim=256 \
      agent.hidden_dim=256 \
      "checkpoint_dir=${checkpoint_dir}" \
      "wandb.enable=true" \
      "wandb.entity=${wandb_entity}" \
      "wandb.project=${wandb_project}" \
      "wandb.group=${group}" \
      "wandb.name=${run_name}" \
      "wandb.tags=[state,sac,${task},${method},1m]" \
      "${auxiliary_overrides[@]}" \
      2>&1 | tee "${log_path}"

  echo "[GPU ${gpu}] completed ${run_name}"
}

if [[ ! -f "${job_file}" || ! -f "${cursor_file}" ]]; then
  echo "Experiment queue is incomplete: ${job_file}" >&2
  exit 2
fi

while true; do
  claimed_job=''
  claim_job
  [[ -n "${claimed_job}" ]] || break
  IFS=$'\t' read -r task method seed <<<"${claimed_job}"
  run_job "${task}" "${method}" "${seed}"
done

echo "[GPU ${gpu}] experiment queue complete"
