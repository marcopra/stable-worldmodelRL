# State-vector SAC with TACO

This adds a true SAC baseline and a state-vector SAC agent with the temporal
TACO objective. It does not use pixels, image augmentations, or CURL. The
implementation is independent of the cloned `TACO/` reference repository.

## Relationship to TACO and DrQ-v2

DrQ-v2 is not SAC: it uses a deterministic policy and scheduled exploration
noise, while SAC uses a stochastic squashed-Gaussian policy and an entropy
temperature. This implementation compares SAC against SAC plus TACO so the
RL algorithm stays fixed within the experiment. It is not an exact reproduction
of the paper's DrQ-v2 experiments.

The TACO paper's image-based DMC table reports strong results on both Reacher
Hard (`883 ± 63` for TACO vs `572 ± 51` for DrQ-v2 at 1M steps) and Finger
Turn Hard (`632 ± 75` vs `220 ± 21`), averaged over six seeds. Reacher Hard is
the best first screen because it is the task already in this repository's slow
pixel benchmark, and this trainer can run its native low-dimensional state
observations without rendering. Finger Turn Hard is a second in-repository
check. These visual results motivate the task choices; they do not predict the
result of state-based SAC. See the [TACO paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/96d00450ed65531ffe2996daed487536-Paper-Conference.pdf).

## Implementation and gradient flow

For a replay clip `(s_t, a_t, ..., a_{t+K-1}, s_{t+K})`, the objective embeds
each action, flattens and tokenizes the action sequence, predicts the projected
future state, and applies bilinear in-batch cross entropy. Other batch entries
are negatives. The default is `K=3`; the optional reward head predicts the
discounted reward accumulated over those `K` transitions. The future-state
projection uses the shared online encoder under `no_grad`, matching the
reference's stop-gradient target path. There is no separate EMA encoder.
The state encoder, policy, critics, and TACO predictor use compact MLPs; they
do not copy the paper's image convolutional backbone and 1,024-unit heads.

| Loss/update | Encoder | Policy | Q functions | Action tokenizer / TACO heads | Entropy temperature |
|---|---|---|---|---|---|
| SAC critic TD error | Updated | — | Updated | Tokenizer updated in TACO mode | — |
| SAC actor objective | Detached features | Updated | Frozen weights; action gradient passes through Q | Frozen tokenizer weights; action gradient passes through tokenizer | — |
| SAC entropy objective | — | Detached log probability | — | — | Updated |
| TACO contrast/reward | Anchor path updated; future path detached | — | — | Updated | — |
| Target critic calculation | No gradient | No gradient | Target critic is frozen | Target-action encoding has no gradient | Detached |

The reference also has overlapping Adam optimizers for the shared encoder and
action tokenizer: the critic update and TACO update each use their own Adam
state. This implementation preserves that reference behavior. The reference
actor update computes unused gradients for Q and tokenizer weights; here those
weights are frozen during the actor update while retaining the derivative from
Q through the action to the policy. This avoids stale, unused gradients without
changing the actor objective.

With `taco.enabled=false`, the agent creates no auxiliary module, uses raw
actions as critic inputs, and runs the same SAC update. This is the matched
baseline for the experiments below. The TACO treatment learns a latent action
representation for the critic and includes the TACO temporal objective. The
reward head is enabled by default and CURL is absent.

## Environments and experiment plan

### Primary task: Reacher Hard with state observations

Use `swm/ReacherDMControl-v0` with `task=hard`. The Gymnasium wrapper exposes
the task's low-dimensional state and does not render frames. Run a five-seed
paired comparison at 500k environment steps. This preserves the physical task
from the slow pixel benchmark while removing image encoding and rendering.

### Second task: Finger Turn Hard

This is built into Stable World Model as `swm/FingerDMControl-v0`, has a
low-dimensional observation and two continuous actions, and requires no image
rendering. It is also a paper benchmark with a large TACO/DrQ-v2 difference.
Run the same five-seed comparison at 500k steps:

1. SAC (`taco.enabled=false`)
2. SAC + TACO, reward prediction enabled (default)
3. SAC + TACO, reward prediction disabled (`taco.reward_prediction=false`)

The third arm separates the temporal contrastive contribution from the
optional reward prediction loss. If resources are tight, run arms 1 and 2 for
seeds 0–4 first, then run arm 3 for seeds 0–2 as an ablation. A 100k, three-seed
pilot can catch integration or gross learning problems, but do not use it to
decide that there is no performance gap: the paper comparison is at 1M steps.

### Secondary task: Stick Pull

`Meta-World/MT1` with `env_name=stick-pull-v3` is a useful contact-rich
manipulation check with vector observations. Run the SAC and SAC+TACO arms for
at least three matched seeds at 500k steps. MetaWorld is an optional dependency;
use a compatible environment and set `CONDA_ENV` when launching. The trainer
does not request a render mode.

If MetaWorld setup is inconvenient, use the built-in
`swm/AcrobotDMControl-v0` as another cheap vector-observation task. The TACO
paper also reports better 1M-step visual performance than DrQ-v2 on Acrobot
Swingup (`241 ± 21` vs `128 ± 8`), though again that result is not a state-SAC
prediction.

Keep seed, replay capacity, batch size, update-to-data ratio, evaluation
schedule, and total steps identical across arms. The default batch size is
1024, matching the large-batch setting selected in the TACO paper for stronger
in-batch negatives; lower it only if necessary, and lower it for every arm.
The config uses a 5k random-action warmup, one update per environment step,
`K=3`, and five deterministic evaluation episodes every 25k steps. Compare
paired per-seed final evaluation return, area under the evaluation-return
curve, and success rate on Stick Pull. Report mean and standard deviation (or
paired confidence intervals) across seeds; retain the per-seed curves because
RL results can have broken seeds. Also log wall-clock time/FPS: TACO adds an
extra representation-learning backward pass per update.

## Running the experiments

The launcher starts exactly one detached tmux session per run, pins it to a
physical GPU, maps that device to `cuda:0` inside the process, and writes a
separate log and checkpoint directory. It checks that the selected GPU is
visible first. Run one paired seed at a time across available GPUs to avoid
oversubscribing the CPU physics simulator; use new session names for each run.

```bash
# Finger Turn Hard, seed 0: matched SAC and SAC+TACO on GPUs 0 and 1
scripts/rl/launch_sac_taco_state_tmux.sh finger-sac-s0 0 sac_taco_state sac 0
scripts/rl/launch_sac_taco_state_tmux.sh finger-taco-s0 1 sac_taco_state taco 0

# Reacher Hard state observations: same task as the slow pixel benchmark
scripts/rl/launch_sac_taco_state_tmux.sh reacher-sac-s0 0 \
  sac_taco_state_reacher_hard sac 0
scripts/rl/launch_sac_taco_state_tmux.sh reacher-taco-s0 1 \
  sac_taco_state_reacher_hard taco 0

# Optional temporal-only ablation, same seed and budget
scripts/rl/launch_sac_taco_state_tmux.sh finger-taco-no-reward-s0 1 \
  sac_taco_state taco 0 taco.reward_prediction=false

# Stick Pull, after installing MetaWorld in an environment named swm-mw
CONDA_ENV=swm-mw scripts/rl/launch_sac_taco_state_tmux.sh stick-sac-s0 0 \
  sac_taco_metaworld_stick_pull sac 0
CONDA_ENV=swm-mw scripts/rl/launch_sac_taco_state_tmux.sh stick-taco-s0 1 \
  sac_taco_metaworld_stick_pull taco 0

# Built-in, cheap state-vector alternative to Stick Pull
scripts/rl/launch_sac_taco_state_tmux.sh acrobot-taco-s0 0 \
  sac_taco_state_acrobot taco 0

# Attach to a run and follow its output
tmux attach -t finger-taco-s0
```

Repeat with seed values 1 through 4 and unique tmux names. To run a short
integration pilot, pass Hydra overrides such as `num_steps=100000`,
`evaluation.frequency=10000`, and `evaluation.episodes=3`. Install the
MetaWorld extra in the selected compatible environment with
`pip install -e ".[env,metaworld]"`; the optional dependency is not needed for
Finger Turn Hard or Acrobot.

## Files and validation

- Agent and objective: [`sac_taco.py`](../stable_worldmodel/rl/sac_taco.py)
- Online trainer: [`sac_taco_online.py`](../scripts/train/sac_taco_online.py)
- Finger config: [`sac_taco_state.yaml`](../scripts/train/config/sac_taco_state.yaml)
- Stick Pull config: [`sac_taco_metaworld_stick_pull.yaml`](../scripts/train/config/sac_taco_metaworld_stick_pull.yaml)
- Reacher Hard state config: [`sac_taco_state_reacher_hard.yaml`](../scripts/train/config/sac_taco_state_reacher_hard.yaml)
- Acrobot config: [`sac_taco_state_acrobot.yaml`](../scripts/train/config/sac_taco_state_acrobot.yaml)
- tmux launcher: [`launch_sac_taco_state_tmux.sh`](../scripts/rl/launch_sac_taco_state_tmux.sh)

The TACO loss, gradient ownership, plain SAC path, discounted reward target,
CPU online smoke runs, checkpoint restore, and launcher syntax were validated.
The full 500k runs have not been started. Stick Pull runtime validation also
requires installing MetaWorld in a compatible environment.
