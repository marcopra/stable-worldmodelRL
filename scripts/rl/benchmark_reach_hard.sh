#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
pid0=""
pid1=""

stop_benchmark() {
  trap - INT TERM
  echo "Stopping Reacher benchmark workers and active training processes." >&2
  [[ -z "$pid0" ]] || kill -TERM "$pid0" 2>/dev/null || true
  [[ -z "$pid1" ]] || kill -TERM "$pid1" 2>/dev/null || true
  pkill -TERM -f '[s]cripts/train/drqv2_online.py.*drqv2_reacher_hard_taco' \
    2>/dev/null || true
  exit 130
}
trap stop_benchmark INT TERM

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to find the two GPUs." >&2
  exit 1
fi
mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader)
if (( ${#gpu_ids[@]} < 2 )); then
  echo "Expected at least two visible GPUs; found ${#gpu_ids[@]}." >&2
  exit 1
fi
gpu0="${gpu_ids[0]}"
gpu1="${gpu_ids[1]}"
batch_size="${BATCH_SIZE:-512}"
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer; got '${batch_size}'." >&2
  exit 1
fi

check_egl() {
  local gpu="$1"
  echo "Checking headless Reacher rendering on GPU ${gpu}."
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$gpu" \
    conda run --no-capture-output -n swm-rl python -c \
    'import gymnasium as gym, stable_worldmodel; e=gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array", task="hard"); e.reset(seed=0); frame=e.render(); print("EGL render:", frame.shape, frame.dtype); e.close()'
}

check_egl "$gpu0" || {
  echo "EGL preflight failed on GPU ${gpu0}; no training jobs launched." >&2
  exit 1
}
check_egl "$gpu1" || {
  echo "EGL preflight failed on GPU ${gpu1}; no training jobs launched." >&2
  exit 1
}

methods=(infonce none lewm)
seeds=(0 1 2 3 4)

run_one() {
  local gpu="$1"
  local seed="$2"
  local method="$3"
  local run_name="reacher-hard-${method}-seed-${seed}"
  local log_path="logs/rl/reacher_hard_taco/${run_name}.log"
  local checkpoint_dir="runs/reacher_hard_taco/${method}/seed_${seed}"
  local loss_overrides=()
  case "${method}" in
    none)
      loss_overrides=(
        auxiliary.enabled=false
        wm=none
        auxiliary.reward_prediction.enabled=false
        auxiliary.curl.enabled=false
      )
      ;;
    infonce|lewm)
      loss_overrides=(
        auxiliary.enabled=true
        "wm=${method}"
        wm.horizon=3
        auxiliary.reward_prediction.enabled=false
        auxiliary.curl.enabled=false
      )
      ;;
    *)
      echo "Unknown method '${method}'." >&2
      return 2
      ;;
  esac
  mkdir -p "$(dirname "$log_path")" "$checkpoint_dir"
  echo "[GPU ${gpu}] starting ${run_name}; log=${log_path}"
  if CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$gpu" \
    conda run --no-capture-output -n swm-rl \
    python scripts/train/drqv2_online.py \
    --config-name drqv2_reacher_hard_taco \
    "seed=${seed}" \
    "batch_size=${batch_size}" \
    "device=cuda:0" \
    "${loss_overrides[@]}" \
    "wandb.name=${run_name}" \
    "wandb.group=taco-reacher-hard-3methods-5seeds-ordered-v3" \
    "checkpoint_dir=${checkpoint_dir}" \
    2>&1 | tee "$log_path"; then
    echo "[GPU ${gpu}] completed ${run_name}"
  else
    local status=$?
    echo "[GPU ${gpu}] FAILED ${run_name} (exit ${status})" >&2
    return "$status"
  fi
}

echo "Launching 15 runs in seed waves across physical GPUs ${gpu0} and ${gpu1}."
echo "For each seed, none and infonce run together, then lewm runs; the next seed waits for all three."
overall_status=0
for seed in "${seeds[@]}"; do
  echo "=== Seed ${seed}: starting none and infonce in parallel ==="
  run_one "$gpu0" "$seed" none &
  pid0=$!
  run_one "$gpu1" "$seed" infonce &
  pid1=$!
  wait "$pid0" || overall_status=1
  pid0=""
  wait "$pid1" || overall_status=1
  pid1=""

  echo "=== Seed ${seed}: starting lewm ==="
  run_one "$gpu0" "$seed" lewm || overall_status=1
done

if (( overall_status != 0 )); then
  echo "One or more benchmark runs failed; see the per-run logs." >&2
  exit 1
fi
echo "All 15 Reacher-hard runs finished."
