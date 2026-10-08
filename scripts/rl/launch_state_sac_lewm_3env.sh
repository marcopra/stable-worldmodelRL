#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_id="${RUN_ID:-sac_state_lewm_3env_1m_v1}"
num_steps="${NUM_STEPS:-1000000}"
seeds="${SEEDS:-1 2 3 4 5}"
gpus=("${GPU_A:-3}" "${GPU_B:-4}")
conda_env="${CONDA_ENV:-swm-rl}"
evaluation_frequency="${EVALUATION_FREQUENCY:-50000}"
checkpoint_frequency="${CHECKPOINT_FREQUENCY:-100000}"
job_root="${repo_root}/logs/rl/${run_id}"
job_file="${job_root}/jobs.tsv"

worker() {
  local gpu="$1" cursor_file="${job_file}.cursor" lock_file="${job_file}.lock"
  while true; do
    local lock_fd cursor line job task seed config group name log_path checkpoint_dir
    exec {lock_fd}>"${lock_file}"
    flock -x "${lock_fd}"
    cursor="$(<"${cursor_file}")"
    line=$((cursor + 1))
    job="$(sed -n "${line}p" "${job_file}")"
    if [[ -n "${job}" ]]; then
      printf '%s\n' "${line}" >"${cursor_file}.tmp.$$"
      mv "${cursor_file}.tmp.$$" "${cursor_file}"
    fi
    flock -u "${lock_fd}"
    exec {lock_fd}>&-
    [[ -n "${job}" ]] || break

    IFS=$'\t' read -r task seed <<<"${job}"
    case "${task}" in
      finger-turn-hard) config=sac_taco_state ;;
      acrobot-swingup) config=sac_taco_state_acrobot ;;
      reacher-hard) config=sac_taco_state_reacher_hard ;;
      *) echo "Unknown task: ${task}" >&2; return 2 ;;
    esac
    group="state-sac-lewm-rp-${task}-1m-v1"
    name="state-sac-${task}-lewm-rp-seed-${seed}-1m-v1"
    log_path="${repo_root}/logs/rl/${run_id}/${task}/lewm_rp/seed_${seed}.log"
    checkpoint_dir="${repo_root}/runs/rl/${run_id}/${task}/lewm_rp/seed_${seed}"
    mkdir -p "$(dirname "${log_path}")" "${checkpoint_dir}"
    echo "[GPU ${gpu}] starting ${name}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
      WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online \
      conda run --no-capture-output -n "${conda_env}" \
        python scripts/train/rl_online.py \
        --config-name="${config}" "seed=${seed}" device=cuda:0 \
        "num_steps=${num_steps}" learning_starts=5000 batch_size=256 \
        updates_per_step=1 max_episode_steps=500 replay.capacity=100000 \
        replay.commit_interval=1000 "evaluation.frequency=${evaluation_frequency}" \
        evaluation.episodes=3 logging.frequency=1000 \
        "checkpoint_frequency=${checkpoint_frequency}" \
        agent.feature_dim=256 agent.hidden_dim=256 \
        "checkpoint_dir=${checkpoint_dir}" \
        auxiliary.enabled=true wm=lewm auxiliary.wm.enabled=true \
        auxiliary.wm.weight=1.0 auxiliary.wm.horizon=3 \
        auxiliary.wm.projection_dim=192 auxiliary.wm.lewm_hidden_dim=2048 \
        auxiliary.wm.sigreg_weight=0.09 \
        auxiliary.reward_prediction.enabled=true \
        auxiliary.reward_prediction.weight=1.0 \
        auxiliary.reward_prediction.horizon=3 auxiliary.curl.enabled=false \
        wandb.enable=true wandb.entity=marcopra wandb.project=swm-test \
        "wandb.group=${group}" "wandb.name=${name}" \
        "wandb.tags=[state,sac,${task},lewm,reward_prediction,1m]" \
        2>&1 | tee "${log_path}"
    echo "[GPU ${gpu}] completed ${name}"
  done
  echo "[GPU ${gpu}] LeWM queue complete"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker "$2"
  exit
fi

sessions=("swm-sac-lewm-state-gpu${gpus[0]}" "swm-sac-lewm-state-gpu${gpus[1]}")
if [[ "${gpus[0]}" == "${gpus[1]}" ]]; then
  echo 'GPU_A and GPU_B must be different.' >&2
  exit 2
fi
for session in "${sessions[@]}"; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session ${session} already exists; refusing to replace it." >&2
    exit 1
  fi
done
if [[ -e "${job_file}" || -e "${job_file}.cursor" ]]; then
  echo "Queue exists at ${job_root}; choose a different RUN_ID." >&2
  exit 1
fi

WANDB_BASE_URL=https://api.wandb.ai conda run --no-capture-output -n "${conda_env}" \
  python - <<'PY'
import wandb
api = wandb.Api()
entity = api.viewer.username
if entity != 'marcopra' or api.project(name='swm-test', entity=entity) is None:
    raise SystemExit('W&B project marcopra/swm-test is not accessible')
print(f'W&B project verified: {entity}/swm-test')
PY

for gpu in "${gpus[@]}"; do
  free_mb="$(CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n "${conda_env}" \
    python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.mem_get_info()[0] // (1024 * 1024))')"
  if (( free_mb < 4096 )); then
    echo "GPU ${gpu} has ${free_mb} MiB free; need at least 4096 MiB." >&2
    exit 1
  fi
  echo "GPU ${gpu}: ${free_mb} MiB free."
done

mkdir -p "${job_root}"
: >"${job_file}"
read -r -a seed_list <<<"${seeds}"
for seed in "${seed_list[@]}"; do
  for task in finger-turn-hard acrobot-swingup reacher-hard; do
    printf '%s\t%s\n' "${task}" "${seed}" >>"${job_file}"
  done
done
printf '0\n' >"${job_file}.cursor"
cat >"${job_root}/plan.txt" <<PLAN
SAC + LeWM + reward prediction; state observations
Tasks: Finger Turn Hard, Acrobot Swingup, Reacher Hard
Seeds: ${seeds}; steps: ${num_steps} each; total runs: $(( ${#seed_list[@]} * 3 ))
LeWM: weight 1, horizon 3, projection 192, hidden 2048, SIGReg weight 0.09
Reward prediction: enabled, weight 1, horizon 3; CURL disabled
Evaluation every ${evaluation_frequency}; checkpoint every ${checkpoint_frequency}
W&B: marcopra/swm-test; GPUs: ${gpus[*]}
PLAN

chmod +x "$0"
for i in 0 1; do
  gpu="${gpus[$i]}"
  command=(env "CONDA_ENV=${conda_env}" "RUN_ID=${run_id}" "NUM_STEPS=${num_steps}" \
    "EVALUATION_FREQUENCY=${evaluation_frequency}" "CHECKPOINT_FREQUENCY=${checkpoint_frequency}" \
    bash "$0" --worker "$gpu")
  printf -v quoted '%q ' "${command[@]}"
  tmux new-session -d -s "${sessions[$i]}" -c "${repo_root}" \
    "cd $(printf '%q' "${repo_root}") && ${quoted}"
  tmux set-window-option -t "${sessions[$i]}" remain-on-exit on
done

echo "Started LeWM runs on GPUs ${gpus[*]} (${#seed_list[@]} seeds, 3 tasks)."
echo "Logs/plan: ${job_root}; W&B: marcopra/swm-test"
echo "Attach: tmux attach -t ${sessions[0]} (or ${sessions[1]})"
