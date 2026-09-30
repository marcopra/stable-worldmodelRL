# SWE handoff: DrQ-v2 representation learning

## Goal

Add a small online DrQ-v2 path to Stable World Model so the same model-free
agent can be compared with no auxiliary objective, temporal InfoNCE, and
compatible existing representation losses.

## Current repository state

The original repository had online TD-MPC2 and offline representation/world
model training, but no native online DrQ-v2 or DrQ-specific actor/critic loop.
Existing episode replay, Gymnasium wrappers, Hydra, W&B, and checkpoint
patterns were reusable. See [`rl_repository_audit.md`](rl_repository_audit.md)
for the audit and local TACO reference findings.

## Architecture discovered

- `ReplayBuffer` is episode/segment-aware and returns clips that never cross
  episode boundaries.
- `PreJEPA`, `LeWM`, and `PLDM` are full model/training abstractions with
  distinct predictors and targets; they are not generic plug-in callables.
- TD-MPC2 supplies useful online collection/evaluation/checkpoint patterns,
  but remains a separate model-based algorithm.
- TACO learns action-conditioned current-to-future contrast with in-batch
  negatives. Its reference future encoding is detached; it does not use a
  separate EMA target encoder.

## Important files and changes

- `stable_worldmodel/rl/drqv2.py`: encoder, random shifts, DrQ-v2 actor/twin
  critics, target updates, and objective integration.
- `stable_worldmodel/rl/representation.py`: InfoNCE, JEPA-style predictor
  adapter, PLDM loss adapter, and objective factory.
- `scripts/train/drqv2_online.py`: online interaction, pixel rendering,
  replay segments, updates, evaluation, logging, and checkpointing.
- `scripts/train/config/drqv2.yaml`: primary configurable run.
- `scripts/train/config/drqv2_smoke.yaml` and `drqv2_sanity.yaml`: short and
  longer validation settings.
- `scripts/rl/local_smoke.sh` and `local_sanity.sh`: separate GPU 0 baseline
  and GPU 1 InfoNCE commands.
- `tests/rl/test_representation.py` and `tests/rl/test_drqv2.py`: objective
  and CPU update coverage.
- `docs/rl_representation_learning.md`: researcher-facing usage and behavior.

## Configuration options

Important options include `env`, `seed`, `device`, `image_size`, `num_steps`,
`batch_size`, `updates_per_step`, `replay.capacity`, `replay.commit_interval`,
`evaluation.frequency`, `evaluation.episodes`, `checkpoint_dir`, `resume`,
`agent.*`, `representation.loss`, `representation.weight`,
`representation.horizon`, `representation.temperature`, and `wandb.*`.

Supported choices are `none`, `infonce`/`taco`, `lewm`, `jepa`/`prejepa`, and
`pldm`. The LeWM choice is the LeWM prediction + SIGReg objective adapter on
DrQ features, not the complete standalone LeWM ViT model.
The JEPA adapter reuses the existing causal predictor, but is not full
pretrained PreJEPA.

## Tests and commands that work

Automated tests:

```bash
PYTHONPATH="$PWD" conda run -n swm-rl pytest -q tests/rl
```

Smoke launch commands:

```bash
PYTHONPATH="$PWD" conda run -n swm-rl python scripts/train/drqv2_online.py \
  --config-name drqv2_smoke device=cpu representation.loss=none

PYTHONPATH="$PWD" conda run -n swm-rl python scripts/train/drqv2_online.py \
  --config-name drqv2_smoke device=cpu representation.loss=infonce \
  representation.horizon=2
```

Periodic evaluation happens during training. Evaluation-only checkpoint load:

```bash
PYTHONPATH="$PWD" python scripts/train/drqv2_online.py --config-name drqv2 \
  resume=./runs/drqv2/drqv2_step_100000.pt evaluation_only=true
```

Resume training with `resume=<checkpoint>` and leave `evaluation_only=false`.
The saved checkpoint includes the replay object and all agent optimizers, but
does not serialize Gymnasium's live simulator state; an in-flight episode is
restarted.

## Experiments run and results observed

- `tests/rl`: 9 passed on CPU.
- Baseline and InfoNCE 120-step smoke runs completed collection, replay
  sampling, optimization, evaluation, and checkpoint save.
- One evaluation episode yielded -0.270 baseline and -0.236 InfoNCE. These
  numbers only confirm execution and do not indicate performance.
- InfoNCE checkpoint restore and evaluation-only mode worked.
- W&B-disabled operation worked without credentials.
- `compileall` and `git diff --check` passed.
- After replacing CUDA 13 wheels with PyTorch 2.14.0+cu126 and torchvision
  0.29.0+cu126, CUDA matrix/backward tests and a DrQ-v2 + InfoNCE update passed
  on both GTX 1080 Ti cards. 120-step baseline and InfoNCE jobs also completed
  end-to-end on GPU 0 and GPU 1 respectively.

The CUDA 13 wheel had omitted sm_61 kernels. The environment now uses the
CUDA 12.6 wheels, the final prebuilt PyTorch line supporting Pascal GPUs.

## Known issues and remaining tasks

- The default 84px, 100,000-step configuration and the 10,000-step sanity
  configuration have not been run.
- Replay writes completed trajectory segments, so clips cannot cross commit
  boundaries. This is intentional for the current replay contract; evaluate
  effects if longer temporal horizons or continuous replay insertion are
  needed.
- All time-limit truncations currently set the bootstrap discount to zero.
  Consider exposing a configurable bootstrap-on-truncation policy.
- The Gymnasium task is intentionally a small rendered control environment;
  no DMControl pixel benchmark was run.
- The full standalone LeWM and exact full-stack PreJEPA models are not loaded
  as DrQ encoders; the RL choices are feature-level objective adapters.
- Run longer sanity and benchmark experiments on the now-working GPU setup.
- Run broader repository tests after installing the project's `env` and `train`
  extras in the target environment.

## Files worth reading first

1. `docs/rl_repository_audit.md`
2. `docs/rl_representation_learning.md`
3. `stable_worldmodel/data/buffer.py`
4. `stable_worldmodel/wm/prejepa/module.py` (`CausalPredictor`)
5. `stable_worldmodel/wm/loss.py` (`PLDMLoss`)
6. `stable_worldmodel/rl/representation.py`
7. `stable_worldmodel/rl/drqv2.py`
8. `scripts/train/drqv2_online.py`
9. `scripts/train/config/drqv2.yaml`
10. `tests/rl/test_representation.py` and `tests/rl/test_drqv2.py`
