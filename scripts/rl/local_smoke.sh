#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

# Run from the repository root after installing environment dependencies.
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n swm-rl python scripts/train/drqv2_online.py \
  --config-name drqv2_smoke device=cuda:0 representation.loss=none \
  checkpoint_dir=./runs/drqv2_baseline_smoke

CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n swm-rl python scripts/train/drqv2_online.py \
  --config-name drqv2_smoke device=cuda:0 representation.loss=infonce \
  representation.horizon=2 checkpoint_dir=./runs/drqv2_infonce_smoke
