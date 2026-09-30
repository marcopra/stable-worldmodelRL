# DrQ-v2 with Stable World Model representation objectives

## What was present and what was missing

Stable World Model already had episode-aware replay, Gymnasium environment
wrappers, temporal models (PreJEPA, LeWM, PLDM, TD-MPC2), model-free offline
policy training, Hydra configurations, optional W&B logging, evaluation, and
checkpoint helpers. It did not have a native online DrQ-v2 agent or an online
model-free pixel-RL loop. The audit and TACO notes are in
[`rl_repository_audit.md`](rl_repository_audit.md).

The new RL package reuses `stable_worldmodel.data.ReplayBuffer` and the
existing Gymnasium environment layer. The online entry point uses Hydra,
optional W&B, and Gymnasium's rendered RGB frames. It does not depend on or
import from the local `TACO/` clone.

## Added files

- `stable_worldmodel/rl/drqv2.py`: random-shift augmentation, DrQ-v2 pixel
  encoder, actor, twin critics, target critic, action selection, and updates.
- `stable_worldmodel/rl/representation.py`: objective factory and temporal
  InfoNCE, JEPA-style, and PLDM adapters.
- `scripts/train/drqv2_online.py`: collection, replay, updates, evaluation,
  optional W&B, checkpoint/resume.
- `scripts/train/config/drqv2*.yaml`: default, smoke, and sanity configurations.
- `scripts/rl/local_{smoke,sanity}.sh`: paired GPU commands.
- `tests/rl/`: unit coverage for objective shapes, gradients, selection, and
  CPU agent updates.

## Architecture and gradient flow

`DrQV2Agent` owns one pixel encoder shared by the actor, critic, and optional
representation learner. Observations are augmented and encoded for the critic;
critic loss updates the encoder. Actor updates use detached features, matching
the DrQ-v2 convention. Twin target critics are initialized from the online
critics and Polyak-updated after each optimization step. There is no target
encoder. The auxiliary optimizer updates the shared encoder and objective
modules; the future target path is computed under `no_grad`.

`representation.loss=none` constructs no auxiliary module or optimizer and
preserves the DrQ baseline update path. The auxiliary coefficient is
`representation.weight`. This keeps algorithm variants in configuration,
not separate agent classes.

## Objectives

### InfoNCE / TACO-style temporal dynamics

For batch element `i`, encode `o_t` as `z_t`, embed the action sequence
`a_t ... a_{t+k-1}`, and predict the projection of `z_{t+k}`. The matching
future observation is the positive; the other batch entries are negatives.
The objective forms bilinear similarity logits, divides them by
`representation.temperature`, and applies cross entropy with labels
`0 ... B-1`. Embeddings are normalized before scoring. The target encoder path
uses the shared encoder with a separate random-shift augmentation and stops
gradients; it is not an EMA target.

Logged diagnostics include InfoNCE loss, mean positive and negative logits,
and temperature. Batch size one is rejected because it has no in-batch
negatives. `representation.horizon` selects the number of transitions in the
temporal pair.

### JEPA-style adapter

`representation.loss=jepa` reuses Stable World Model's `CausalPredictor` over
projected DrQ features and predicts the final detached future feature with an
MSE loss. This is a small online adapter for the DrQ feature sequence. It does
not instantiate the full PreJEPA pipeline, which uses pretrained vision
backbones, token layouts, and its own training setup.

### PLDM adapter

`representation.loss=pldm` projects DrQ features and applies the existing
`PLDMLoss` temporal alignment and variance/covariance regularizers. Its terms
are summed into one scalar and multiplied by `representation.weight`.

### LeWM objective adapter

`representation.loss=lewm` uses the existing LeWM action embedder and
predictor with DrQ pixel features, and combines latent prediction MSE with
LeWM's SIGReg regularizer (weight `representation.sigreg_weight`, default
0.09). Gradients update the shared DrQ encoder and the LeWM adapter. This is
the LeWM objective on DrQ features, not the standalone LeWM ViT encoder and
full training model.

## Configuration and commands

The default task is `MountainCarContinuous-v0`, which has a normalized
continuous action space and RGB rendering through Gymnasium. Defaults are
stored in [`scripts/train/config/drqv2.yaml`](../scripts/train/config/drqv2.yaml).
The smoke config uses a smaller 36px input and short replay segments; sanity
uses 10,000 steps. Every update samples temporal clips from `ReplayBuffer`.
Completed segments are committed every `replay.commit_interval` transitions,
so training can begin before a long episode ends. Clips never cross segment
boundaries.

Activate `swm-rl` and run from the repository root. Set `PYTHONPATH` so the
checkout is used if a different Stable World Model version is installed in the
environment:

```bash
# DrQ-v2 baseline
PYTHONPATH="$PWD" python scripts/train/drqv2_online.py representation.loss=none

# DrQ-v2 + action-conditioned InfoNCE
PYTHONPATH="$PWD" python scripts/train/drqv2_online.py representation.loss=infonce representation.horizon=2

# Existing Stable World Model objectives/adapters
PYTHONPATH="$PWD" python scripts/train/drqv2_online.py representation.loss=jepa
PYTHONPATH="$PWD" python scripts/train/drqv2_online.py representation.loss=pldm
```

Quick validation and two-GPU local launch commands:

```bash
bash scripts/rl/local_smoke.sh
bash scripts/rl/local_sanity.sh
```

Evaluation is run periodically during training. To evaluate a saved checkpoint
without training:

```bash
PYTHONPATH="$PWD" python scripts/train/drqv2_online.py \
  --config-name drqv2 \
  resume=./runs/drqv2/drqv2_step_100000.pt evaluation_only=true
```

To resume training, set `resume=...` and omit `evaluation_only=true`. Checkpoints
contain agent/optimizer state, replay contents, step/episode counts, and
random-number-generator state. An in-progress environment episode is reset on
resume.

W&B is off by default. Enable it with
`wandb.enable=true wandb.project=<project>`; configure `wandb.name`, `group`,
`tags`, and `run_id` in the same config. Logged metrics include environment
steps, episodes/rewards, critic and actor losses/Q values, representation
diagnostics, evaluation reward/length, and FPS.

## Validation and limits

Verified in `swm-rl`:

- **Automated tests:** `PYTHONPATH="$PWD" conda run -n swm-rl pytest -q tests/rl` —
  9 passed. Coverage includes InfoNCE shapes, finite loss, gradients and
  stop-gradient behavior, target-pair labels, JEPA/PLDM adapters, and CPU
  updates for baseline and auxiliary agents.
- **Smoke runs:** baseline and InfoNCE each completed a 120-step CPU run,
  sampled replay clips, ran optimization, evaluated, and saved checkpoints.
  Observed one-episode evaluation returns were -0.270 (baseline) and -0.236
  (InfoNCE); these are smoke outputs, not evidence of a performance
  improvement.
- **GPU checks:** after installing the Pascal-compatible wheels, matrix and
  backward operations passed on both GTX 1080 Ti cards. A DrQ-v2 + InfoNCE
  optimizer update passed on each card. End-to-end 120-step smoke runs also
  completed on GPU 0 (baseline, evaluation return -0.001) and GPU 1 (InfoNCE,
  evaluation return -4.909). These remain correctness checks, not performance
  comparisons.
- **Checkpoint evaluation:** an InfoNCE checkpoint loaded and ran the
  evaluation-only path.
- **W&B-disabled mode:** verified in both smoke runs without credentials.
- **Compilation:** Python compileall and `git diff --check` passed.

The environment originally had PyTorch 2.14.0+cu130, whose kernels omit the
cards' sm_61 architecture. It now has PyTorch 2.14.0+cu126 and torchvision
0.29.0+cu126. The local environment was also missing pytest, OpenCV, imageio,
Hydra, and pygame; these were installed into `swm-rl` for the checks, without
changing repository dependency files.

The Reacher-hard comparison config and launcher are described in
[`rl_reacher_hard_benchmark.md`](rl_reacher_hard_benchmark.md). The full
standalone LeWorldModel and PreJEPA stacks are not substituted for their
DrQ-feature objective adapters.
