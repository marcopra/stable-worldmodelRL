# Online RL implementation handoff

Online SAC and DrQ-v2 share the same collection runner and replay contract.
Both agents accept state or rendered pixel observations and can run with the
auxiliary learning path disabled or with one primary world-model loss. Reward
prediction and CURL are separately configurable add-ons.
The current architecture and extension guide live in
[`rl_repository_audit.md`](rl_repository_audit.md).

## Main components

- `stable_worldmodel/rl/algorithms/base.py`: the `OnlineAgent` protocol.
- `stable_worldmodel/rl/algorithms/sac.py` and `drqv2.py`: algorithm-owned
  actor, critic, optimizer, update, and checkpoint logic.
- `stable_worldmodel/rl/encoders.py`: shared state and pixel observation
  encoders.
- `stable_worldmodel/rl/representation.py`: InfoNCE/TACO, LeWM, JEPA, PLDM,
  reward-prediction, and pixel-only CURL adapters. Each run selects at most one
  primary loss and may enable reward prediction and/or CURL.
- `scripts/train/rl_online.py`: environment collection, replay, evaluation,
  logging, checkpointing, and algorithm dispatch.
- `scripts/train/config/rl_online.yaml`: shared run defaults composed with the
  `agent/` and `wm/` config groups; task YAML files inherit and override them.

LeWM, JEPA, and PLDM are feature-level objective adapters over the selected RL
encoder. They do not load their respective standalone training stacks. State
mode selects a numeric Box observation or an explicit numeric Box field from a
Dict observation. Pixel mode renders RGB, resizes frames, and can stack them.
The replay stores aligned `observation`, `action`, `reward`, and `discount`
sequences. Time-limit truncations retain a bootstrap discount; true
terminations do not.

## Running and configuring

The primary entry point accepts any composed RL config:

```bash
PYTHONPATH="$PWD" conda run -n swm-rl python scripts/train/rl_online.py \
  --config-name=sac_state auxiliary.enabled=false

PYTHONPATH="$PWD" conda run -n swm-rl python scripts/train/rl_online.py \
  --config-name=drqv2_reacher_hard_taco
```

Every run config sets `algorithm.name` and `observation.mode`. Auxiliary
learning defaults off. Select one primary `wm` config and turn on
`auxiliary.enabled`. The primary config carries its weight and loss-specific
settings. Toggle optional reward prediction and CURL with
`auxiliary.reward_prediction.enabled` and `auxiliary.curl.enabled`. For
example:

```bash
PYTHONPATH="$PWD" conda run -n swm-rl python scripts/train/rl_online.py \
  --config-name=sac_state \
  wm=infonce \
  auxiliary.enabled=true \
  auxiliary.wm.horizon=3 \
  auxiliary.reward_prediction.enabled=true \
  auxiliary.curl.enabled=true
```

CURL requires pixel observations and setup rejects it for state runs. The
legacy `drqv2_online.py` and `sac_taco_online.py` entry points remain thin
compatibility launchers. Package imports such as
`stable_worldmodel.rl.drqv2.DrQV2Agent` and
`stable_worldmodel.rl.sac_taco.SACTACOAgent` remain available.

## Validation

`tests/rl` exercises SAC and DrQ-v2 updates for both modalities, baseline and
primary-plus-add-on updates, Dict-state extraction, objective weighting, modality
validation, bootstrap discounts, and agent checkpoint restoration. The
repository audit records the CPU collection smokes, current capability matrix,
and recommended validation for future algorithm and loss contributions.
