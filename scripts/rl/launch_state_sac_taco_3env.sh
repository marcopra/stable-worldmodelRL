#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_env="${CONDA_ENV:-swm-rl}"
gpu_a="${GPU_A:-0}"
gpu_b="${GPU_B:-5}"
# CUDA omits the device that NVML reports as broken index 6, so the idle
# nvidia-smi GPU 7 is selected through CUDA_VISIBLE_DEVICES=6.
gpu_c="${GPU_C:-6}"
run_id="sac_state_3env_100k_v1"
job_root="${repo_root}/logs/rl/${run_id}"
job_file="${job_root}/jobs.tsv"
worker="${repo_root}/scripts/rl/run_state_sac_taco_worker.sh"
sessions=(
  "swm-sac-state-3env-gpu${gpu_a}"
  "swm-sac-state-3env-gpu${gpu_b}"
  "swm-sac-state-3env-gpu7-cuda${gpu_c}-a"
  "swm-sac-state-3env-gpu7-cuda${gpu_c}-b"
)
worker_gpus=("${gpu_a}" "${gpu_b}" "${gpu_c}" "${gpu_c}")

if [[ "${gpu_a}" == "${gpu_b}" || "${gpu_a}" == "${gpu_c}" || \
  "${gpu_b}" == "${gpu_c}" ]]; then
  echo 'GPU_A, GPU_B, and GPU_C must select three distinct physical GPUs.' >&2
  exit 2
fi

for session in "${sessions[@]}"; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session '${session}' already exists; refusing to replace it." >&2
    exit 1
  fi
done

if [[ -e "${job_file}" || -e "${job_file}.cursor" ]]; then
  echo "Queue already exists at ${job_root}; refusing to reset it." >&2
  exit 1
fi

echo 'Checking W&B credentials and project access.'
WANDB_BASE_URL=https://api.wandb.ai conda run --no-capture-output \
  -n "${conda_env}" python - <<'PY'
import wandb

api = wandb.Api()
entity = api.viewer.username
if entity != 'marcopra':
    raise SystemExit(f'Expected W&B entity marcopra, found {entity!r}')
if api.project(name='swm-test', entity=entity) is None:
    raise SystemExit('W&B project marcopra/swm-test is not accessible')
print(f'W&B project verified: {entity}/swm-test')
PY

for gpu in "${gpu_a}" "${gpu_b}" "${gpu_c}"; do
  minimum_free_mb=1024
  if [[ "${gpu}" == "${gpu_c}" ]]; then
    minimum_free_mb=6144
  fi
  free_mb="$(CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output \
    -n "${conda_env}" python -c \
    'import torch; assert torch.cuda.is_available(); print(torch.cuda.mem_get_info()[0] // (1024 * 1024))')"
  if (( free_mb < minimum_free_mb )); then
    echo "GPU ${gpu} has ${free_mb} MiB free; needs at least ${minimum_free_mb} MiB." >&2
    exit 1
  fi
  echo "GPU ${gpu}: ${free_mb} MiB free."
done

mkdir -p "${job_root}"
: >"${job_file}"
for seed in 0 1 2 3 4; do
  for task in finger-turn-hard acrobot-swingup reacher-hard; do
    printf '%s\t%s\t%s\n' "${task}" baseline "${seed}" >>"${job_file}"
    printf '%s\t%s\t%s\n' "${task}" taco "${seed}" >>"${job_file}"
  done
done
printf '0\n' >"${job_file}.cursor"
cat >"${job_root}/plan.txt" <<EOF
Experiment: SAC baseline vs TACO-style auxiliary learning
W&B: https://api.wandb.ai/marcopra/swm-test
Tasks: Finger Turn Hard, Acrobot Swingup, Reacher Hard (proprioceptive state observations)
Arms: SAC baseline; SAC + temporal InfoNCE + reward prediction (equal weights)
CURL: disabled (pixel-only)
Seeds: 0, 1, 2, 3, 4 per task and arm (30 runs total)
Budget: 100000 agent steps, batch 256, learning starts at 5000
Evaluation: every 10000 steps, 3 episodes
CUDA_VISIBLE_DEVICES workers: ${gpu_a} (one new worker), ${gpu_b} (one new worker), ${gpu_c} (two new workers)
CUDA device ${gpu_c} maps to idle nvidia-smi GPU 7; NVML cannot open index 6.
Existing long runs were observed on GPUs 0-5; this plan adds no more than one
run to GPUs ${gpu_a}/${gpu_b} and two runs to nvidia-smi GPU 7.
EOF

chmod +x "${worker}"
for index in "${!sessions[@]}"; do
  session="${sessions[index]}"
  gpu="${worker_gpus[index]}"
  command=(
    env
    "CONDA_ENV=${conda_env}"
    bash "${worker}"
    "${gpu}"
    "${job_file}"
  )
  printf -v quoted_command '%q ' "${command[@]}"
  tmux new-session -d -s "${session}" -c "${repo_root}" \
    "cd $(printf '%q' "${repo_root}") && ${quoted_command}"
  tmux set-window-option -t "${session}" remain-on-exit on
done

echo 'Started the SAC state benchmark in four tmux worker sessions:'
for session in "${sessions[@]}"; do
  echo "  ${session}"
done
echo "Tasks: 3, arms: 2, seeds per task/arm: 5, total seed runs: 30"
echo "W&B project: marcopra/swm-test"
echo "GPU allocation: nvidia-smi ${gpu_a}, ${gpu_b}, and 7 (CUDA-visible ${gpu_c} for the idle GPU 7)"
echo "Logs and plan: ${job_root}"
echo 'Attach with: tmux attach -t <session-name>'
