#!/usr/bin/env bash
set -euo pipefail

if (( $# < 6 )); then
  cat >&2 <<'USAGE'
Usage:
  scripts/rl/launch_sac_taco_state_queue_tmux.sh \
    <tmux-session> <physical-gpu> <config> <sac|taco> <wandb-group> <seed-a,seed-b,...> [Hydra overrides...]

Each queue runs its listed seeds sequentially in one tmux session. Start at most
two queues per GPU to cap concurrency at two runs on that GPU.
Set CONDA_ENV to select the environment and WANDB_BASE_URL to override the
CoreWeave W&B instance URL.
USAGE
  exit 2
fi

session="$1"
gpu="$2"
config="$3"
method="$4"
group="$5"
seeds="$6"
shift 6

case "${method}" in
  sac|taco) ;;
  *) echo "method must be 'sac' or 'taco', got '${method}'." >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_env="${CONDA_ENV:-swm-rl}"
wandb_base_url="${WANDB_BASE_URL:-https://forge.coreweave.com/wandb}"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session '${session}' already exists; choose another session name." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${conda_env}" \
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

pythonpath="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
command=(
  env
  "CUDA_VISIBLE_DEVICES=${gpu}"
  "PYTHONPATH=${pythonpath}"
  "WANDB_BASE_URL=${wandb_base_url}"
  "CONDA_ENV=${conda_env}"
  bash "${repo_root}/scripts/rl/run_sac_taco_state_seed_queue.sh"
  "${config}"
  "${method}"
  "${group}"
  "${seeds}"
  "${repo_root}"
  "$@"
)
printf -v quoted_command '%q ' "${command[@]}"
shell_command="cd $(printf '%q' "${repo_root}") && ${quoted_command}"

tmux new-session -d -s "${session}" -c "${repo_root}" "${shell_command}"
tmux set-window-option -t "${session}" remain-on-exit on
echo "Started sequential queue '${session}' on physical GPU ${gpu}."
echo "Config=${config} method=${method} seeds=${seeds} W&B=${group}"
echo "W&B base URL=${wandb_base_url}"
echo "Attach with: tmux attach -t ${session}"
