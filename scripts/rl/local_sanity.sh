set -euo pipefail
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

common=(--config-name drqv2 num_steps=100000 wandb.enable=true
  wandb.project=stable-worldmodel-rl wandb.group=drqv2-100k)

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n swm-rl \
  python scripts/train/drqv2_online.py "${common[@]}" \
  wandb.name=drqv2-baseline-100k 'wandb.tags=[drqv2,baseline,100k]' \
  device=cuda:0 auxiliary.enabled=false \
  checkpoint_dir=./runs/drqv2_baseline_100k &
baseline_pid=$!

CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n swm-rl \
  python scripts/train/drqv2_online.py "${common[@]}" \
  wandb.name=drqv2-infonce-100k 'wandb.tags=[drqv2,infonce,100k]' \
  device=cuda:0 auxiliary.enabled=true wm=infonce \
  auxiliary.wm.horizon=2 \
  checkpoint_dir=./runs/drqv2_infonce_100k &
infonce_pid=$!

wait "$baseline_pid"
wait "$infonce_pid"
