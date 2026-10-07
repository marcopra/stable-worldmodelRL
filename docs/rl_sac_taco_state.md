# State SAC with TACO objectives

State SAC and pixel DrQ-v2 now use the same online runner and auxiliary-loss
configuration. `sac_state` is the state-vector SAC baseline; `sac_taco_state`
preserves the existing Finger Turn Hard task preset with temporal InfoNCE
selected and reward prediction enabled. CURL is disabled for this state task,
but can be enabled for pixel observations. Both agents can also use rendered
pixel inputs.

This is not an exact reproduction of the TACO paper's visual DrQ-v2 benchmark.
It applies TACO-style representation learning to a SAC state encoder. The
paper reports InfoNCE as the central temporal objective and CURL and reward
prediction as additional terms. This implementation selects one primary
world-model loss per experiment and allows reward prediction and CURL as
separately configured add-ons. [TACO
paper](https://papers.neurips.cc/paper_files/paper/2023/file/96d00450ed65531ffe2996daed487536-Paper-Conference.pdf)

## Objective and gradient flow

For a replay clip `(s_t, a_t, ..., a_{t+K-1}, s_{t+K})`, temporal InfoNCE
encodes the start and future observations, represents each action, predicts
the future feature, and classifies the matching batch entry against in-batch
negatives. The action encoder is shared with the RL critic. The positive
future feature is stop-gradient. Optional reward prediction regresses the
discounted reward over its configured horizon. Pixel runs can also enable CURL
to contrast independently augmented views.

The critic update trains the observation encoder, critic, and—when InfoNCE is
enabled—the shared action encoder. The actor uses detached observation
features; critic and action-encoder weights are frozen while gradients pass
through the action value to the policy. The entropy objective updates SAC's
temperature. The auxiliary update trains the shared encoder and every selected
loss head. It does not use a separate EMA target encoder.

State observations do not support CURL. Enabling CURL in state mode fails
during setup with a configuration error. The shared runner bootstraps across
time-limit truncations and suppresses bootstrap only for true terminations.

## Running matched SAC baselines and TACO

From the repository root, run the baseline and TACO preset with the shared
runner:

```bash
PYTHONPATH="$PWD" conda run -n swm-rl python scripts/train/rl_online.py \
  --config-name=sac_state auxiliary.enabled=false

PYTHONPATH="$PWD" conda run -n swm-rl python scripts/train/rl_online.py \
  --config-name=sac_taco_state
```

To enable reward prediction alongside temporal InfoNCE:

```bash
PYTHONPATH="$PWD" conda run -n swm-rl python scripts/train/rl_online.py \
  --config-name=sac_taco_state \
  auxiliary.reward_prediction.enabled=true
```

The Reacher Hard DrQ-v2 TACO preset selects InfoNCE and enables both reward
prediction and CURL. Either add-on can be switched off independently. To
enable only CURL on a pixel task:

```bash
PYTHONPATH="$PWD" conda run -n swm-rl python scripts/train/rl_online.py \
  --config-name=drqv2 \
  wm=infonce auxiliary.enabled=true \
  auxiliary.wm.enabled=true \
  auxiliary.curl.enabled=true
```

The older `sac_taco_online.py` command remains as a thin compatibility launcher.
Task variants remain available as `sac_taco_state_reacher_hard`,
`sac_taco_state_acrobot`, and `sac_taco_metaworld_stick_pull`. SAC and DrQ-v2
also have pixel/state modality configs in `scripts/train/config/`.

Keep seed, replay capacity, batch size, update-to-data ratio, evaluation
schedule, and total steps matched across arms. The default SAC state TACO
preset uses batch size 1024 and horizon 3. Use multiple seeds and report
per-seed curves; short smoke-run scores are execution checks, not performance
evidence.

## Implementation references

- Canonical SAC agent: `stable_worldmodel/rl/algorithms/sac.py`
- Shared online runner: `scripts/train/rl_online.py`
- State SAC baseline: `scripts/train/config/sac_state.yaml`
- State SAC TACO preset: `scripts/train/config/sac_taco_state.yaml`
- Pixel DrQ-v2 TACO preset: `scripts/train/config/drqv2_reacher_hard_taco.yaml`
