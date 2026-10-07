# Online SAC and DrQ-v2 with representation objectives

The online RL package lets SAC and DrQ-v2 run with state or rendered pixel
observations. A shared Hydra runner handles collection, replay, evaluation,
logging, and checkpoints; each agent keeps its algorithm-specific update.
See the [RL architecture audit](rl_repository_audit.md) for extension guidance.

## Observation modes and configs

Every composed run config includes `algorithm.name` and `observation.mode`.
State mode accepts a numeric Box observation or a selected numeric Box field
from a Dict observation. Set `observation.state_key` for Dict observations.
Pixel mode renders RGB observations, resizes them to `observation.image_size`,
and concatenates `observation.frame_stack` frames along the channel axis.

Example configs:

| Config | Algorithm | Observation mode |
|---|---|---|
| `sac_state` | SAC | State |
| `sac_pixels` | SAC | Pixels |
| `drqv2_state` | DrQ-v2 | State |
| `drqv2` | DrQ-v2 | Pixels |

Run from the repository root after installing the environment dependencies:

```bash
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=sac_state
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=sac_pixels
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=drqv2_state
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=drqv2
```

The previous `drqv2_online.py` and `sac_taco_online.py` entry points remain as
thin launchers. Agents are also exported from `stable_worldmodel.rl`; the old
`SACTACOAgent` name remains as a compatibility alias with the legacy TACO-on
default.

## Selecting a primary loss and optional add-ons

Auxiliary learning is off by default. Select at most one primary objective
through the Hydra `wm` config group. Reward prediction and CURL are separate
optional add-ons, each with its own switch, weight, and parameters. Turn the
auxiliary path on with `auxiliary.enabled=true`. For example, a pixel run can
select InfoNCE and enable both add-ons:

```bash
PYTHONPATH="$PWD" python scripts/train/rl_online.py \
  --config-name=drqv2 \
  wm=infonce \
  auxiliary.enabled=true \
  auxiliary.wm.horizon=3 \
  auxiliary.reward_prediction.enabled=true \
  auxiliary.reward_prediction.horizon=3 \
  auxiliary.curl.enabled=true
```

The registered choices are:

- **`infonce` / TACO:** predict a future encoded observation from the current
  feature and intervening actions, using the other batch entries as negatives.
  The learned action encoder is shared with the RL critic. InfoNCE needs at
  least two samples in a batch.
- **`lewm`, `jepa`, and `pldm`:** feature-level adapters that reuse compatible
  components from the existing Stable World Model objectives. They are not the
  standalone LeWM or PreJEPA image-model training pipelines.
- **Reward prediction add-on:** predict the discounted reward accumulated over
  its configured action horizon. It can be enabled alongside the selected
  primary objective.
- **CURL add-on:** contrast two independently random-shifted views of the same
  pixel observation. It can be enabled alongside the selected primary
  objective; state+CURL configs fail validation.

Choose only one of `infonce`, `lewm`, `jepa`, or `pldm` as the primary `wm`
loss. Reward prediction and CURL may each be enabled or disabled independently;
both can be enabled together when the observation mode is pixels. The TACO
paper describes InfoNCE as its central objective with reward prediction and
CURL as additional terms. [TACO
paper](https://papers.neurips.cc/paper_files/paper/2023/file/96d00450ed65531ffe2996daed487536-Paper-Conference.pdf)

For a baseline, select `wm=none` and leave `auxiliary.enabled=false`. For a
temporal-only run, select InfoNCE:

```bash
PYTHONPATH="$PWD" python scripts/train/rl_online.py \
  --config-name=sac_state \
  wm=infonce \
  auxiliary.enabled=true \
  auxiliary.wm.horizon=3
```

The selected Hydra `wm` config is packaged as `auxiliary.wm`; it sets
`target`, `enabled`, and the primary loss parameters. The primary loss is
scaled by `auxiliary.wm.weight`. Reward prediction and CURL use their own
configured weights; the selected terms contribute to one auxiliary update over
the shared encoder. Replay history covers the longest selected temporal
horizon and clips stay within committed episode segments. The runner sets
bootstrap discount to zero only for true terminations, preserving bootstrap
across time limits.

## Logging, checkpoints, and validation

`algorithm.name` and `observation.mode` are included in the composed Hydra/W&B
config and every checkpoint. Logs include the algorithm, environment, and
observation mode. Checkpoints contain agent and optimizer state, replay, step
and episode counts, the resolved config, and RNG state. Resume resets any
in-progress environment episode.

Run unit coverage with:

```bash
PYTHONPATH="$PWD" conda run -n swm-rl pytest -q tests/rl
```

The suite covers both algorithms in state and pixel modes, baseline updates,
primary objectives plus reward/CURL add-ons, state+CURL rejection, selected
Dict state inputs, loss weights, action-token gradients, and truncation
bootstrapping. Short CPU smoke runs cover all four algorithm/modality
combinations. These checks verify execution and do not measure performance.
