# Stable World Model RL repository audit

Audit date: 2026-09-30. This describes the checked-out implementation, not an
assumption based on module names.

## Repository map

| Path | Purpose and important interfaces | Relationship |
|---|---|---|
| `stable_worldmodel/data/` | `Dataset` and `ReplayBuffer`; episode aware storage, sampling, format adapters, transforms and normalization. | Offline model training and online TD-MPC2 replay can share these data types. |
| `stable_worldmodel/world/` | `World` rollout/evaluation and vectorized `EnvPool`. | Drives `Policy` implementations through Gymnasium envs. |
| `stable_worldmodel/wrapper/` | `MegaWrapper` preprocessing; pixels and state are standardized into `info`. | Used by `World` and dataset-compatible interaction. |
| `stable_worldmodel/wm/` | World model architectures/objectives: GCRL, PreJEPA, LeWM, PLDM, TD-MPC2; `loss.py` has VCReg, PLDM, and temporal straightening utilities. | Models expose encoders/predictors and inference rollouts; train scripts wire their own losses. |
| `stable_worldmodel/planning/` | Objectives, shooting evaluator, and CEM/MPPI/gradient solvers. | Model-based planning over model rollouts; not a model-free actor/critic. |
| `stable_worldmodel/policy.py` | Policy interfaces/adapters for actions, world models, and planners. | Plugs learned components into `World`. |
| `scripts/train/` | Hydra + StablePretraining training for offline models and policies, including PLDM, PreJEPA, LeWM, TD-MPC2, IQL/BC variants. | Existing configuration, optimizer, checkpoint and W&B conventions. |
| `scripts/expert/` | SB3 SAC expert collection and online TD-MPC2 training. | The SAC script uses SB3 and is an expert-policy utility, not a custom visual DrQ-v2 implementation. The TD-MPC2 script is the nearest online-RL loop to extend conceptually. |
| `scripts/data/`, `scripts/visualization/` | Data collection/conversion and visualization tools. | Support dataset and experiment workflows. |
| `tests/` | Unit/integration tests for data, envs, planning, world models, wrappers, and CLI. | No RL agent tests currently exist. |
| `TACO/` | Local reference clone with paper PDF and original DrQ-v2/TACO agents. | Temporary study material only; no runtime imports or dependency should be added. |

## Existing functionality

| Capability | Status | Evidence and reuse |
|---|---|---|
| Image encoders | Already implemented | PreJEPA/PLDM/LeWM accept pretrained vision encoders and projectors; TD-MPC2 has a compact pixel CNN. `RandomShiftsAug` exists in the TACO reference only. |
| Latent dynamics and action conditioning | Already implemented | TD-MPC2 uses one-step `(z, action)` dynamics; PLDM and LeWM use action encoders and temporal predictors; PreJEPA supports temporal prediction and action-conditioned variants. |
| Temporal prediction | Already implemented | PreJEPA, LeWM, PLDM, TD-MPC2 and planning rollouts. |
| JEPA-style learning | Already implemented | PreJEPA model/training entry point. The training code handles frozen pretrained backbones and predicted/target representations. |
| Contrastive learning | Partially implemented | GCRL includes contrastive goal/reward-conditioned learning machinery; no batch InfoNCE temporal dynamics objective for online model-free RL was found. |
| Target encoders / EMA | Partially implemented | Some objectives detach target encodings or use model-specific target behavior; there is no shared online target-encoder API suitable for a DrQ agent. |
| Augmentations | Partially implemented | Dataset transforms and model-specific image transforms exist. DrQ-v2 random-shift augmentation is absent from Stable World Model. |
| Replay / trajectory sampling | Already implemented | `ReplayBuffer` stores whole episodes and samples episode-safe clips. The online TD-MPC2 loop uses it; temporal clip length can supply multi-step actions/observations. |
| Action conditioning | Already implemented | Temporal models accept encoded action streams; replay episodes retain actions. |
| Logging | Already implemented | Hydra training scripts use optional Lightning W&B logging; online TD-MPC2 supports optional W&B. |
| Training scripts | Already implemented | Offline models and online TD-MPC2 have entry points; no DrQ-v2 trainer. |
| Evaluation | Partially implemented | `World.evaluate` runs attached policies, while TD-MPC2 online has task evaluation. No DrQ-specific online evaluation loop exists. |
| Checkpointing | Already implemented | Offline model `save_pretrained` and online TD-MPC2 checkpoint functions. Agent optimizer/replay resume semantics need an RL-specific checkpoint. |
| Configuration/dependencies | Already implemented | Hydra configs; `pyproject.toml` has Torch, Gymnasium, and optional `env`, `train`, and W&B extras. |
| Multi-GPU launch | Partially implemented | Hydra submitit launcher exists for training; local two-GPU commands can use separate processes and `CUDA_VISIBLE_DEVICES`. No RL launcher exists. |

## Gaps for model-free RL evaluation

- **Missing:** a native DrQ-v2 agent, actor/twin critics/target critics, online RL replay interaction, and an RL-specific trainer/config.
- **Missing:** a DrQ random-shift augmentation and temporal InfoNCE dynamics objective in the Stable World Model package.
- **Partially implemented:** episode-aware replay, environment wrappers/collections, online evaluation, logging, and checkpoint patterns exist, but their APIs need adapting for a continuous-control DrQ loop and exact time-limit handling.
- **Could be reused:** `ReplayBuffer`, Gymnasium environment registration/wrappers, TD-MPC2's online environment setup/evaluation/logging/checkpoint structure, Hydra configuration, and `World`/policy conventions where their abstractions fit.
- **Needs modification or an adapter:** sampling consecutive `(obs_t, action_t...action_{t+k-1}, obs_{t+k})` clips from replay, feeding temporal batches to objectives, and sharing an encoder without changing baseline DrQ gradient flow.
- **Existing representation losses are not interchangeable callables:** PLDM and VCReg in `wm/loss.py` are losses, but PreJEPA and LeWM are complete training models with distinct predictors/targets, input schemas and optimizers. They cannot safely be selected by a string and called against DrQ without adapters. This is a design boundary to document and test rather than silently pretending every existing model is plug-compatible.
- **W&B is optional:** `wandb` is included in the `train` extra and existing scripts guard logger creation with config flags.

## TACO technical note

The NeurIPS 2023 TACO paper describes a temporal action-driven contrastive
objective that learns state and action representations by maximizing the
mutual information between a current state paired with a sequence of actions
and the matching future state. It presents the objective as an auxiliary
module for visual RL, with an InfoNCE classification over batch negatives.
The [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/96d00450ed65531ffe2996daed487536-Abstract-Conference.html)
states that this learns control-relevant state and action representations.

The local reference implementation uses a DrQ-style four-layer convolutional
encoder and random-shift augmentation. It embeds actions (including action
sequences) through a learned action tokenizer, predicts a future projection
from the current state and encoded action sequence, forms a batch-by-batch
similarity matrix with a learned bilinear matrix, and uses diagonal labels for
cross entropy. Future/positive encoding is under `no_grad`; it is a detached
view of the shared encoder, not a separately updated EMA target encoder. Its
replay sampler emits temporal action sequences. The code also has optional
reward prediction and CURL-style same-state contrastive losses. The paper and
implementation are conceptually consistent on action-conditioned temporal
contrast, while details such as the exact predictor and action tokenization
are implementation choices.

Generic concepts to retain: temporal positive alignment, in-batch negatives,
action conditioning over the interval, explicit stop-gradient target, and
joint training with RL. Do not copy the reference's agent class layout,
encoder ownership, unused/optional helper modules, or hardwired batch/update
flow. Stable World Model already has model-agnostic replay episodes, temporal
predictors, vision encoders and Hydra training conventions. Reimplement the
objective as a small package module with an explicit input contract and let a
DrQ agent own the shared encoder and optimizer decisions.

This contribution is useful because Stable World Model has strong temporal
world-model losses and a replay abstraction but lacks the model-free DrQ
evaluation path and a batch InfoNCE dynamics objective. The novel engineering
value is the clean, measurable bridge between those representation choices
and online actor/critic learning, not a new claim about TACO itself.

## Phase 1–4 architecture and implementation plan

The initial implementation will add a separate `stable_worldmodel.rl` package
for a compact DrQ-v2 agent and online trainer. It will reuse `ReplayBuffer`
for completed episodes, the existing Gymnasium envs and configs, and existing
W&B conventions. Baseline DrQ will retain the usual DrQ-v2 path: critic loss
updates the shared encoder; actor updates use detached encoded observations;
target critics are updated by Polyak averaging. In the auxiliary run, a
configurable objective gets temporally sampled replay clips and contributes
`representation.weight * loss` to encoder/dynamics optimization. The target
representation is detached; the first implementation will not add EMA unless
the objective needs it. Random shifts will be applied to both endpoints
independently.

The first compatible objective is action-conditioned temporal InfoNCE: encode
`o_t`, predict a projected `o_{t+k}` from `z_t` and the intervening actions,
compare against detached target encodings in the same batch, use the diagonal
as positives, and all other rows as negatives. Use normalized embeddings,
logits divided by configurable temperature, and cross entropy with labels
`arange(batch)`. The objective should expose loss and inexpensive similarity
diagnostics. For `representation.loss=none`, the baseline gets no auxiliary
modules or extra updates. The existing PLDM temporal-alignment/VCReg loss is
compatible after a small projection adapter. A JEPA-style adapter can reuse
Stable World Model's `CausalPredictor` with the DrQ latent sequence and a
detached future target; it is explicitly not the full pretrained/frozen
backbone PreJEPA training pipeline. LeWM remains incompatible until an adapter
can preserve its predictor/target and optimizer contracts.

We will implement in increments: objective plus its requested unit tests,
DrQ agent, replay/training integration, optional W&B/checkpointing, local
smoke configs/commands, then final summary and SWE handoff. Initial validation
will distinguish unit and smoke checks from benchmark evaluation.
