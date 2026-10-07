#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
  cat >&2 <<'USAGE'
Usage:
  scripts/rl/launch_sac_taco_state_tmux.sh \
    <tmux-session> <physical-gpu> <config> <sac|taco> <seed> [Hydra overrides...]

Examples:
  scripts/rl/launch_sac_taco_state_tmux.sh finger-sac-g0 0 sac_taco_state sac 0
  scripts/rl/launch_sac_taco_state_tmux.sh finger-taco-g1 1 sac_taco_state taco 0
  scripts/rl/launch_sac_taco_state_tmux.sh stick-taco-g0 0 sac_taco_metaworld_stick_pull taco 0
USAGE
  exit 2
fi

session="$1"
gpu="$2"
config="$3"
method="$4"
seed="$5"
conda_env="${CONDA_ENV:-swm-rl}"
shift 5

case "${method}" in
  sac) auxiliary_enabled=false; wm_config=none ;;
  taco) auxiliary_enabled=true; wm_config=infonce ;;
  *) echo "method must be 'sac' or 'taco', got '${method}'." >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log_root="${repo_root}/logs/rl/sac_taco_state"
run_root="${repo_root}/runs/sac_taco_state/${config}/${method}/seed_${seed}"
run_name="${config}-${method}-seed-${seed}"
log_path="${log_root}/${run_name}.log"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session '${session}' already exists; choose another session name." >&2
  exit 1
fi

mkdir -p "${log_root}" "${run_root}"

# Check that PyTorch can address the selected physical CUDA device before the
# detached session starts. SAC itself does not need a display or EGL renderer.
CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${conda_env}" \
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

pythonpath="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
command=(
  env
  "CUDA_VISIBLE_DEVICES=${gpu}"
  "PYTHONPATH=${pythonpath}"
  conda run --no-capture-output -n "${conda_env}"
  python scripts/train/sac_taco_online.py
  --config-name "${config}"
  "seed=${seed}"
  device=cuda:0
  "auxiliary.enabled=${auxiliary_enabled}"
  "wm=${wm_config}"
  "checkpoint_dir=${run_root}"
  "wandb.name=${run_name}"
  "$@"
)
printf -v quoted_command '%q ' "${command[@]}"
printf -v quoted_log '%q' "${log_path}"
shell_command="cd $(printf '%q' "${repo_root}") && ${quoted_command}2>&1 | tee ${quoted_log}"

tmux new-session -d -s "${session}" -c "${repo_root}" "${shell_command}"
tmux set-window-option -t "${session}" remain-on-exit on
echo "Started ${run_name} in tmux session '${session}' on physical GPU ${gpu}."
echo "Conda environment: ${conda_env}"
echo "Log: ${log_path}"
echo "Attach with: tmux attach -t ${session}"
