# Online RL architecture audit

Audit date: 2026-10-07. This describes the current online RL implementation
and the intended extension path for algorithms and world-model losses.

## Repository map

| Area | Location | Responsibility |
|---|---|---|
| Algorithm agents | `stable_worldmodel/rl/algorithms/` | SAC and DrQ-v2 update logic, optimizer ownership, action selection, and agent checkpoints. Both satisfy the `OnlineAgent` protocol. |
| Observation encoders | `stable_worldmodel/rl/encoders.py` | Shared numeric-state MLP and rendered-pixel convolutional encoders. |
| Objective adapters | `stable_worldmodel/rl/representation.py` | Registry and implementations for one selected primary world-model objective plus optional add-ons. |
| Pixel augmentation | `stable_worldmodel/rl/augmentation.py` | Random-shift augmentation for DrQ-v2 and CURL views. |
| Online experiment runner | `scripts/train/rl_online.py` | Collection, replay, evaluation, logging, checkpointing, and algorithm dispatch. |
| Hydra base config | `scripts/train/config/rl_online.yaml` | Shared online-run infrastructure and defaults. |
| Agent config group | `scripts/train/config/agent/` | Common agent defaults plus algorithm-specific settings. |
| World-model config group | `scripts/train/config/wm/` | A base config and one config per selected objective, including `none`. |
| Task configs | `scripts/train/config/*.yaml` | Select agent and objective groups, then override environment and task-specific values. |
| Compatibility launchers/imports | `scripts/train/{drqv2_online,sac_taco_online}.py`, `stable_worldmodel/rl/{drqv2,sac_taco}.py` | Preserve established command and import paths while delegating to the shared implementation. |
| Replay | `stable_worldmodel/data/buffer.py` | Episode-aware sampling of aligned temporal `observation`, `action`, `reward`, and `discount` sequences. |

Each algorithm agent owns its gradient flow and optimizer steps. The runner
depends only on the common `act`, `update`, `state_dict`, and `load_state_dict`
interface and the shared replay batch, so new algorithms do not need a second
collection loop.

## Current RL status

The repository now has a shared online runner and two custom continuous
control agents: SAC and DrQ-v2. Both accept state or rendered pixel inputs,
use the same episode-aware replay schema, and keep their algorithm-specific
gradient and optimizer logic in the agent. Baseline training does not allocate
auxiliary modules. An enabled auxiliary path selects at most one primary
world-model loss, with reward prediction and CURL available as independent
add-ons.

The previous audit's findings that no native DrQ-v2 trainer, DrQ augmentation,
or temporal InfoNCE adapter existed are stale. LeWM is now available through a
feature-level adapter as well; its earlier “not compatible” status applied to
using the standalone LeWM training model directly. Existing offline world-model
training scripts remain separate from these online RL adapters.

## Supported modality and loss matrix

| Capability | Numeric state | Rendered pixels |
|---|---:|---:|
| SAC baseline | Yes | Yes |
| DrQ-v2 baseline | Yes | Yes |
| Temporal InfoNCE (TACO) | Yes | Yes |
| LeWM adapter | Yes | Yes |
| JEPA adapter | Yes | Yes |
| PLDM adapter | Yes | Yes |
| Reward prediction add-on | Yes | Yes; independently selectable |
| CURL add-on | Rejected during setup | Yes; independently selectable |

Each run selects at most one primary loss from InfoNCE, LeWM, JEPA, or PLDM.
Reward prediction and CURL are explicit optional add-ons; either or both may be
enabled with the selected primary loss. CURL requires pixels.

State mode accepts a numeric Box observation, or a numeric Box field selected
with `observation.state_key` from a Dict observation. Pixel mode renders RGB
frames, resizes them, and optionally stacks them. The action space must be a
continuous Box normalized to `[-1, 1]`.

LeWM, JEPA, and PLDM are feature-level adapters over the selected RL encoder.
They reuse compatible predictors or losses from this library; they are not
the standalone LeWM, PreJEPA/JEPA, or PLDM training stacks. SPR, ATC, and DRIML
do not yet have online RL adapters.

## Configuration model

The top-level `rl_online.yaml` composes an agent and a world-model config. The
Hydra group stays in `config/wm/`, while its package directive places the
selected config at `auxiliary.wm` in the resolved run config:

```yaml
defaults:
  - agent: sac
  - wm: none
  - _self_
```

`auxiliary.enabled` is the global on/off switch and defaults to false. A run
selects at most one primary `wm` config. For example, `wm=infonce` selects
temporal InfoNCE, while `wm=lewm` selects LeWM. That config is packaged under
`auxiliary.wm`, where `target` names the objective and `enabled`, its
parameters, and scalar weight control it. Override those settings using paths
such as `auxiliary.wm.horizon=3`. `auxiliary.reward_prediction` and
`auxiliary.curl` are the two explicit add-on switches, with their own
parameters and weights. Do not configure multiple primary world-model losses
or add a generic keyed objective collection. Compare primary losses in
separate experiments; toggle either or both supported add-ons per experiment.

Task configs inherit `rl_online`, choose an `agent` and `wm`, then override
task-specific settings. For example:

```yaml
defaults:
  - rl_online
  - override /agent: sac
  - override /wm: infonce
  - _self_

auxiliary:
  enabled: true
  wm:
    target: infonce
    enabled: true
    horizon: 3
  reward_prediction:
    enabled: true
  curl:
    enabled: true
```

The resolved `algorithm.name` comes from `agent.name` and appears in structured
logs, W&B config, and checkpoints. Selected objectives are constructed only
when `auxiliary.enabled=true`; disabled runs do not allocate or update
auxiliary modules. CURL is valid only with pixel observations and fails clearly
at setup for state runs, even if the global auxiliary switch is off. Replay
history covers the maximum selected primary/reward-prediction horizon and the
DrQ-v2 n-step target. True terminations disable bootstrap; time-limit
truncations retain it.

## Adding an algorithm

1. Add `stable_worldmodel/rl/algorithms/<name>.py`. Implement `OnlineAgent`
   methods: `act(observation, step, eval_mode)`, `update(batch, step)`,
   `state_dict()`, and `load_state_dict(state)`.
2. Keep the algorithm's critic/actor updates, target networks, gradient
   ownership, and optimizers inside its agent. Use `make_observation_encoder`
   for the shared state/pixel encoders where appropriate.
3. Add `scripts/train/config/agent/<name>.yaml` with `name` and that
   algorithm's settings. Add task configs that inherit `rl_online`, select the
   agent group, and override environment-specific values.
4. Add dispatch and replay-history requirements to `rl_online.py` only where
   algorithm semantics require them. Keep collection, evaluation, logging, and
   checkpoint logic shared.
5. Validate state and pixel baselines, enabled and disabled auxiliary paths,
   replay requirements, truncation bootstrapping, and checkpoint restore.

## Adding a world-model objective

1. Implement an `nn.Module` adapter in `stable_worldmodel/rl/representation.py`
   that accepts the shared keyword inputs (`features`, `actions`, optional
   target features, rewards, discounts, and pixel views) and returns a scalar
   differentiable `loss` plus scalar diagnostics.
2. Register primary world-model losses in `build_world_model`. Reward
   prediction and CURL remain the only explicit add-ons. Validate modality and
   horizon requirements so invalid configs fail before training.
3. Add `scripts/train/config/wm/<name>.yaml`, inheriting `wm/base.yaml`, with
   that objective's own parameters and default weight.
4. Add tests for the objective, target-gradient behavior, weighting,
   checkpoint restore, and rejected modality combinations. Do not add generic
   composition of primary losses; a run selects at most one `wm` config.
5. Update the capability matrix and document how an adapter differs from its
   standalone model implementation.

## Validation guide

The online RL checks should cover:

- CPU agent updates for SAC and DrQ-v2 in state and pixel modes, with the
  auxiliary switch off, one primary loss on, and representative add-ons on.
- Box and selected Dict state extraction; rendered frame resizing and stacking;
  and replay clips that remain inside their episode.
- True termination versus time-limit truncation bootstrap values.
- Loss weight application, objective/module checkpoint restore, algorithm name
  in logs/checkpoints, and a clear state-plus-CURL setup error.
- Hydra composition for every task config and short collection/update smokes for
  all four algorithm/modality pairs.

These checks validate execution and interfaces; they do not establish relative
algorithm performance.

Validation run on 2026-10-07 after optional add-ons were integrated: all 36
tests in `tests/rl` passed; Hydra composition succeeded for every SAC and
DrQ-v2 task config, with each selected objective resolving under
`auxiliary.wm.target`. Six CPU collection/update smokes covered all four
algorithm/modality pairs, state SAC with InfoNCE plus reward prediction, and
pixel DrQ-v2 with InfoNCE, reward prediction, and CURL. After moving the
configuration under `auxiliary.wm`, two more short CPU smokes exercised the
nested overrides for state SAC with InfoNCE/reward prediction and pixel
DrQ-v2 with InfoNCE/reward prediction/CURL. The unit suite also covers Box and
selected Dict state inputs, pixel resize/stack behavior, episode-bounded
temporal replay, truncation bootstrap, objective weighting and restore,
algorithm-name checkpoint metadata, and state-plus-CURL rejection. Changed RL
files passed Ruff, `git diff --check` passed, and shell scripts passed
`bash -n`.
