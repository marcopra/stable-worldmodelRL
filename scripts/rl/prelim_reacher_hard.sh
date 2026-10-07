#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script_path="${repo_root}/scripts/rl/prelim_reacher_hard.sh"
log_root="${repo_root}/logs/rl/prelim_reacher_hard_36px"
run_root="${repo_root}/runs/prelim_reacher_hard_36px"
gpu_none="${GPU_NONE:-0}"
gpu_infonce="${GPU_INFONCE:-5}"

worker() {
  local gpu="$1"
  local method="$2"
  local loss_overrides=()
  if [[ "${method}" == none ]]; then
    loss_overrides=(
      auxiliary.enabled=false wm=none
      auxiliary.reward_prediction.enabled=false
      auxiliary.curl.enabled=false
    )
  else
    loss_overrides=(
      auxiliary.enabled=true
      wm=infonce
      auxiliary.reward_prediction.enabled=false
      auxiliary.curl.enabled=false
    )
  fi
  for seed in 0 1 2; do
    local run_name="reacher-hard-36px-${method}-seed-${seed}"
    local log_path="${log_root}/${run_name}.log"
    local checkpoint_dir="${run_root}/${method}/seed_${seed}"
    mkdir -p "$(dirname "${log_path}")" "${checkpoint_dir}"
    echo "[GPU ${gpu}] starting ${run_name}; log=${log_path}"
    if CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu}" \
      conda run --no-capture-output -n swm-rl \
      python scripts/train/drqv2_online.py \
      --config-name drqv2_reacher_hard_taco \
      "seed=${seed}" \
      "device=cuda:0" \
      "num_steps=20000" \
      "observation.image_size=36" \
      "batch_size=32" \
      "updates_per_step=1" \
      "agent.feature_dim=32" \
      "agent.hidden_dim=256" \
      "agent.num_expl_steps=2000" \
      "replay.capacity=30000" \
      "replay.commit_interval=500" \
      "evaluation.frequency=5000" \
      "evaluation.episodes=2" \
      "logging.frequency=1000" \
      "checkpoint_frequency=20000" \
      "${loss_overrides[@]}" \
      "wandb.enable=false" \
      "checkpoint_dir=${checkpoint_dir}" \
      2>&1 | tee "${log_path}"; then
      echo "[GPU ${gpu}] completed ${run_name}"
    else
      local status=$?
      echo "[GPU ${gpu}] FAILED ${run_name} (exit ${status})" >&2
      return "${status}"
    fi
  done
}

if [[ "${1:-}" == worker ]]; then
  worker "$2" "$3"
  exit 0
fi

if [[ "${gpu_none}" == "${gpu_infonce}" ]]; then
  echo "GPU_NONE and GPU_INFONCE must identify different GPUs." >&2
  exit 1
fi

for session in swm-rh-prelim-none swm-rh-prelim-infonce; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session '${session}' already exists; refusing to replace it." >&2
    exit 1
  fi
done

mkdir -p "${log_root}" "${run_root}"
for gpu in "${gpu_none}" "${gpu_infonce}"; do
  echo "Checking Reacher EGL rendering on GPU ${gpu}."
  CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu}" \
    conda run --no-capture-output -n swm-rl python -c \
    'import gymnasium as gym, stable_worldmodel; e=gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array", task="hard"); e.reset(seed=0); print("EGL frame:", e.unwrapped.render(width=36, height=36).shape); e.close()' \
    || { echo "EGL preflight failed on GPU ${gpu}; no jobs launched." >&2; exit 1; }
done

tmux new-session -d -s swm-rh-prelim-none \
  "cd '${repo_root}' && bash '${script_path}' worker '${gpu_none}' none"
tmux new-session -d -s swm-rh-prelim-infonce \
  "cd '${repo_root}' && bash '${script_path}' worker '${gpu_infonce}' infonce"
echo "Started paired Reacher Hard preliminary jobs in tmux sessions:"
echo "  swm-rh-prelim-none    (GPU ${gpu_none}, seeds 0-2)"
echo "  swm-rh-prelim-infonce (GPU ${gpu_infonce}, seeds 0-2)"
echo "Logs: ${log_root}"
