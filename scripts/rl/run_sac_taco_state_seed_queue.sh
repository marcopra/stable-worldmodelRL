#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
  echo "Usage: $0 <config> <sac|taco> <wandb-group> <seed-a,seed-b,...> <repo-root> [Hydra overrides...]" >&2
  exit 2
fi

config="$1"
method="$2"
group="$3"
seed_list="$4"
repo_root="$5"
shift 5

conda_env="${CONDA_ENV:-swm-rl}"
entity="marcopra"
project="stable-worldmodel-rl"
log_root="${repo_root}/logs/rl/sac_taco_state"
run_root="${repo_root}/runs/sac_taco_state/${config}/${method}"

case "${method}" in
  sac) taco_enabled=false ;;
  taco) taco_enabled=true ;;
  *) echo "method must be 'sac' or 'taco', got '${method}'." >&2; exit 2 ;;
esac

IFS=',' read -r -a seeds <<< "${seed_list}"
if (( ${#seeds[@]} == 0 )); then
  echo 'Provide at least one comma-separated seed.' >&2
  exit 2
fi

for seed in "${seeds[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed '${seed}'; seeds must be non-negative integers." >&2
    exit 2
  fi

  run_name="${group}-${method}-seed-${seed}"
  run_dir="${run_root}/seed_${seed}"
  log_path="${log_root}/${run_name}.log"
  mkdir -p "${run_dir}" "${log_root}"

  echo "Starting ${run_name} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  command=(
    conda run --no-capture-output -n "${conda_env}"
    python scripts/train/sac_taco_online.py
    --config-name "${config}"
    "seed=${seed}"
    device=cuda:0
    "taco.enabled=${taco_enabled}"
    "checkpoint_dir=${run_dir}"
    "wandb.enable=true"
    "wandb.entity=${entity}"
    "wandb.project=${project}"
    "wandb.group=${group}"
    "wandb.name=${run_name}"
    "$@"
  )
  "${command[@]}" 2>&1 | tee "${log_path}"
  echo "Completed ${run_name}"
done

echo "Queue complete: method=${method} seeds=${seed_list} group=${group}"
