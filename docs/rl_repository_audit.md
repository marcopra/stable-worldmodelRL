# Online RL implementation and methodology audit

Reviewed 2026-10-14 against source, resolved configurations, tests, local logs,
and primary papers/code. Starting revision: `8d3a0b0`; no existing user changes
were present in the root or manuscript checkout. No `AGENTS.md` was found.
This review concerns the native online RL path, not every offline WM trainer.

## Findings: confirmed defects and completed corrections

| Confirmed defect | Evidence and correction |
|---|---|
| DrQ-v2 ignored configured replay span | `_make_agent` omitted `sequence_length` in the DrQ branch. Both branches now reuse one common parameter mapping, including the configured span. |
| Short auxiliary batches could reshape into a wrong feature dimension | DrQ reshaped using configured length rather than actual available frames; SAC silently shortened some losses. Both validate required observations/actions before updating. Fixtures now supply the configured span. |
| SAC MPC terminal-value modes needed consistent objectives | The optional MPC head uses reward-only TD targets and reward-only planning. With `mpc.value_coef=0`, the planner reuses the base SAC critic and includes SAC entropy terms along its imagined actions and terminal bootstrap; the base critic retains its standard soft target. |
| Planning could query an entirely untrained model | A step threshold did not prove replay learning occurred. Planning now requires a completed WM/reward update; checkpoints preserve its count. No actor fallback hides a failed readiness check. |
| Invalid or meaningless settings escaped validation | Non-finite MPC settings, consistency horizon other than 1, per-step reward horizon other than 1, and degenerate PLDM statistics now fail clearly. Non-finite plan scores raise an error. The WM sweep explicitly selects reward horizon 1 under MPC. |
| DrQ actor gradients stopped at bounded-action saturation | Ordinary clamp differed from the reference `TruncatedNormal`. The sampler now uses the reference straight-through action bound. |
| Benchmark command used a stale Hydra path | `wm.horizon=3` was invalid because the group is packaged under `auxiliary.wm`. Corrected to `auxiliary.wm.horizon=3`. |
| Public reward factory ignored configured mode | Its compatibility branch always constructed clip-return mode and rejected a null discount. It now preserves per-step mode and uses the supplied default discount, matching the canonical builder. |
| Pixel Reacher target was hidden | The wrapper's visibility default is 0 and archived configs did not override it. The visual preset now sets explicit `env_kwargs.render_target=true`, with new output families preserving prior evidence. |
| Docs overstated supported commands and validation | The handoff enabled CURL on state SAC; its instructions are merged into the modality-aware guide. Baseline verification actually had 9 failures/36 tests. Current results appear below. |

When MPC is enabled, the base actor and critic consume the selected WM latent,
so the original critic can evaluate planner-predicted states. A positive
`mpc.value_coef` adds a separate reward-return head on that same latent; setting
it to zero omits that loss/head and uses the base critic for terminal
bootstrapping. The selected world model, reward head, and base observation
encoder are reused by MPC; no second dynamics model or extra representation
loss was introduced. `wm=none` cannot enable MPC. Training still counts one
real step per environment call; imagined rollouts never enter replay.

## Architecture and optimizer ownership

| Component | Key source | Role |
|---|---|---|
| SAC / DrQ-v2 | `stable_worldmodel/rl/algorithms/{sac,drqv2}.py` | Base losses, targets, action selection, optimizers, checkpoints |
| Shared encoders | `stable_worldmodel/rl/encoders.py` | Numeric MLP or rendered-pixel convolutional features `x` |
| Auxiliaries | `stable_worldmodel/rl/representation.py` | One primary objective, optional reward/CURL, weighting |
| Planner / latent Q | `stable_worldmodel/rl/mpc.py` | Recurrent scoring and weighted-elite action search |
| Replay and collection | `stable_worldmodel/data/buffer.py`, `scripts/train/rl_online.py` | Episode/segment-bounded batches, real counters, evaluation |
| Configs | `scripts/train/config/{rl_online.yaml,agent/,wm/}` | Learner group, one WM group packaged under `auxiliary.wm`, add-ons |
| Manuscript | `Model-free-to-Model-based/main.tex` | Equations, adaptation statements, research limits |

Critic Adam updates the base critic; encoder Adam updates the shared encoder
from critic loss and, with MPC enabled, also updates the WM projector through
that path. Actor Adam consumes detached control features and a frozen critic
with action gradients. SAC temperature has its own Adam. Auxiliary Adam then
updates the shared encoder, selected auxiliary heads, WM projector, and optional
MPC value head. The encoder/projector optimizers maintain independent moments.
Target critics and target MPC heads have no gradients and are Polyak-updated
with the base algorithm's tau. There is no separate target/EMA encoder.
The displayed sum of losses describes ownership; it is not one joint
simultaneous gradient step.

## Complete online loss inventory

Let `x=e(o)`, `z=g(x)` (including normalization for InfoNCE), `c` be replay
continuation, `H` the model horizon, `U` the auxiliary replay span, and
`L=max(U,H,H_reward)`. Real batches also accommodate DrQ's n-step target.
Temporal adapters average `M=min(U,L-H+1)` complete windows. The table lists
raw losses; outer weights multiply them exactly once in auxiliary Adam.

| Loss and owner | Inputs and target | Stop-gradient | Reduction, weight, horizon, and trained path |
|---|---|---|---|
| SAC critic; base | Replay `(x,a,r,c,x')`; `y=r+gamma*c*(min Q_target(x',a')-alpha*log pi(a'|x'))`; in TACO-MPC, `x` is replaced by WM latent `z` | Entire target | Mean of sum of twin squared errors; encoder, WM projector (when MPC is enabled), and critic optimizers; one-step |
| SAC actor; base | Sampled squashed Gaussian action; `alpha*log pi-min Q` | Encoder features and alpha; critic weights frozen | Batch mean; actor optimizer only |
| SAC temperature; base | `-log(alpha)*(log pi+target_entropy)` | Log probability | Mean; alpha optimizer only; default entropy target `-action_dim` |
| DrQ-v2 critic; base | Augmented pixel features or state; n-step discounted rewards and target twin minimum; in TACO-MPC, use WM latent `z` | Entire target | Mean twin MSE sum; encoder, WM projector (when MPC is enabled), and critic optimizers; `agent.nstep` |
| DrQ-v2 actor; base | Scheduled, noise-clipped action; `-min Q` | Encoder features; critic weights frozen | Batch mean; actor optimizer; straight-through action clamp |
| InfoNCE; TACO | Normalized source projection + H action embeddings; predicted normalized endpoint versus in-batch projected future keys; score `q W k / temperature` | Future feature and key projector branch | Cross entropy averaged over batch and M windows; `wm.weight`; encoder, source projector, action encoder, predictor, W |
| LeWM prediction; TACO | Projected real context and embedded actions; prediction MLP outputs against next projected real features | None on either MSE side | Elementwise MSE over B,H,D and M windows; `wm.weight`; encoder, projector, action embedder, causal predictor, prediction MLP |
| LeWM SIGReg; TACO component | Random unit projections of projected features; empirical characteristic function versus standard Gaussian | None | Gaussian-weighted quadrature, multiplied by B, averaged over time/projections/windows; inner `sigreg_weight=0.09`, outer `wm.weight`; encoder/projector |
| JEPA; TACO | H projected context features, no actions; final causal prediction against future projection | Future feature/projector branch | Elementwise MSE, mean over M windows; `wm.weight`; encoder, projector, causal predictor |
| PLDM alignment; TACO component | Adjacent projected features across full L+1 clip | None | Elementwise adjacent MSE; outer `wm.weight`; encoder/projector; no dynamics or inverse-action head |
| PLDM four VCReg terms; TACO components | Features centered across batch at each time | None | Batch std hinge, temporal std hinge, batch off-diagonal covariance mean, temporal off-diagonal covariance mean, all internal weights 1; outer `wm.weight`; encoder/projector |
| Consistency; TACO | Recurrent predictions from first projection and U real actions versus next real projections | All future feature/projector targets | `(1/U) sum rho_wm^k mean_B ||prediction-target||^2`; sums latent dimensions (differs from TD-MPC2's elementwise MSE); `wm.weight`; encoder, initial projector, transition |
| Clip reward; TACO add-on | Start encoder feature and raw H_reward-action clip; discounted masked real reward sum | Replay reward target | MSE averaged over B and `min(max(1,U-H_reward+1),L-H_reward+1)` complete windows; `reward_prediction.weight`; encoder and reward head |
| Per-step reward; TACO add-on, required by MPC | Selected WM projection and one action; immediate replay reward | Replay reward target | MSE over B,U; horizon 1; `reward_prediction.weight`; encoder/projector and existing reward head |
| CURL; TACO pixel add-on | Two independently shifted current pixel views; bilinear diagonal positive versus batch negatives | Key encoder/projector branch | Batch cross entropy; `curl.weight`; query encoder/projector and CURL W; no temporal horizon |
| Twin two-hot latent Q; TACO-MPC only, optional | Encoded replay `z,a`; `r+gamma*c*min Q_value_target(z',a')` using base-policy action | Next encoding, actor action, target Q, entire TD target | Mean twin CE against clipped symlog two-hot scalar targets; `(1/U) sum rho_value^k`; `mpc.value_coef`; encoder/projector and value head, not transition or base critic; omitted at weight zero |

Default latent-Q support is 101 symlog bins in `[-10,10]`; scalar decoding is
symexp of the probability-weighted bin coordinate. This is scalar regression
with a two-hot target, not a return-distribution Bellman update. Temporal weights
are divided by U, not by their sum. Per-step reward and value losses train on
encoded real states; consistency trains through recurrent predicted states.

For PLDM, variance uses sample correction and epsilon `1e-4`; covariance divides
by sample count minus one and averages squared off-diagonal entries. Temporal
covariance uses the same batch-centered features, not a new temporal centering.
B, time, and projected dimension must exceed one. `SIGReg` uses 17 knots from
0 to 3 and 1024 projections by default. No inverse-dynamics or temporal
straightening loss is selected by the online wrapper.

`legacy_taco.TACOStateObjective` remains a notebook-only implementation:
unnormalized bilinear InfoNCE plus `reward_weight` times clip-reward MSE,
using shared state/action-sequence projections and detached future keys.
The caller owns its optimizer and supplies the reward target. The native runner
never constructs it; the `SACTACOAgent` compatibility name delegates to SACAgent.
Standalone offline WM losses and native TD-MPC2 are separate training stacks.

## Base critic and optional value head

With MPC enabled, the actor and base critic are constructed on the WM latent
`z=g(x)`, and the same projection is used to encode replay states and initialize
imagined rollouts. Thus the original critic is a valid planner terminal critic;
no inverse mapping from `z` to `x` is needed. The base SAC/DrQ-v2 losses remain
the model-free update. A positive `mpc.value_coef` adds a distinct two-hot
reward-return head on `z`; setting it to zero omits that head and uses the base
critic at the planner horizon.

For SAC, the base critic estimates a soft return. In the zero-weight mode, the
planner scores each imagined action with `r-alpha*log_pi(a|z)` and bootstraps
with `min Q(z,a)-alpha*log_pi(a|z)` at a sampled terminal policy action. The
separate head instead estimates ordinary reward return, so its planning score
contains no entropy term. DrQ-v2 uses reward-only scoring in either mode. The
auxiliary head's benefit, coefficient 0.1, and calibration at imagined
endpoints remain research questions; its encoded-replay training is not the
full TD-MPC2 rollout value objective.

## Planner comparison

The reference comparison uses the [TD-MPC2 paper, sections 3.1–3.2](https://arxiv.org/html/2310.16828v2)
and [authors' planner and update code](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py),
inspected 2026-10-14. The paper provides the method; the code determines precise
execution behavior. Current upstream code includes later episodic extensions.

| Component | Local implementation | Reference comparison |
|---|---|---|
| Action search | Clipped diagonal Gaussian, top-E elites, `softmax(temperature*(J-max J))`, weighted mean/std, clamped std | Same weighted-elite MPPI pattern; CEM-like moment updates. Temperature is a multiplier/inverse temperature. |
| Initialization | Zero mean at episode start; shift previous optimized mean afterward; zero final action; reset std to max each call | Same mean shifting and std reset |
| Policy proposals | Fixed proposals drawn repeatedly from base actor at current z | Reference latent policy rolls through predicted z. Local proposals are open-loop, not latent feedback. |
| Scoring | K shared transitions and scalar immediate rewards | Reference has categorical reward regression, and termination masking in current episodic code |
| Terminal value | Sample terminal policy action; use the optional reward-return head when weighted, otherwise the base critic | Reference samples terminal action from latent policy and uses its Q aggregation. |
| Executed training action | Weighted elite sample's first action plus std-scaled noise, clipped | Reference similarly samples an elite and adds noise |
| Executed evaluation action | Best final elite's first action | Reference samples an elite even during evaluation, omitting added exploration noise |
| Learning | Base RL plus selected auxiliary objective; reward/value at real encoded states | Reference learns joint dynamics/reward/value on latent rollouts, plus its own latent policy |
| Model/heads | Selected projection, twin categorical value head, scalar reward MSE; no second dynamics | Not TD-MPC2 SimNorm, Q ensemble, categorical reward model, or training stack |

The plan scores K candidate actions and samples one terminal policy action.
With a value head, it sums predicted rewards and bootstraps from the mean twin
head; with a zero value weight, it uses the base critic (including SAC entropy
terms where applicable). Only the selected first candidate action is executed.
All rollout computation is under no-grad; LeWM dropout is disabled during
planning and module training modes are restored afterward.

## Research questions and unsupported conclusions

- InfoNCE's learned bilinear W can give correct discrimination without
  predicting the actual next projected-state coordinates. A one-step callable
  interface does not establish a calibrated recurrent world model.
- One-step teacher-forced LeWM/InfoNCE training leaves K-step drift untested.
  Measure errors along replay action rollouts and reward/value calibration.
- JEPA and consistency lack an explicit anti-collapse term; base RL gradients
  do not prove collapse cannot occur. PLDM alignment can penalize useful motion.
- The base actor remains trained against its own critic while MPC changes
  behavior. The latent value head evaluates continuation under the base policy,
  not the replanning policy. Encoded-versus-imagined inputs add distribution shift.
- No learned termination model exists. Replay targets use correct real masks,
  but imagined trajectories assume continuation. Episodic tasks need an explicit
  treatment before interpreting the score as their expected return.
- Existing state runs vary completeness and seeds; TACO retests use batch 1024
  while MPC recipes use 256. Those families do not isolate a planning benefit.
- Required matched ablations: actor versus MPC with identical predictors;
  value off/on; K versus U; policy proposals off/on; InfoNCE versus consistency;
  reward/CURL terms separately. Report per-seed curves, uncertainty, evaluation
  seeds, real training/evaluation interactions, and runtime/planning cost.

## Preserved state experiment evidence

Inspected `logs/rl/sac_state_3env_1m_v1/{plan.txt,jobs.tsv,**/*.log}`.
The historical queue used seeds 6,7,8, state SAC, baseline versus InfoNCE plus
clip reward, both auxiliary weights 1 and horizons 3, no CURL; budget 1M real
steps, batch 256, learning start 5000, one update per step, replay 100k,
commit 1000, episode cap 500, evaluation every 50k with three episodes,
checkpoint every 100k. Current presets are not exact historical source snapshots.
The last locally available evaluation is reported below; a shorter endpoint
means incomplete local evidence, not an inferred final result.

| Task | Seed | Baseline last step / return | InfoNCE + reward last step / return |
|---|---:|---:|---:|
| Reacher Hard | 6 | 1M / 977.667 | 1M / 987.333 |
| Reacher Hard | 7 | 1M / 983.333 | 1M / 984.667 |
| Reacher Hard | 8 | 1M / 976.000 | 1M / 982.333 |
| Finger Turn Hard | 6 | 1M / 855.667 | 1M / 659.667 |
| Finger Turn Hard | 7 | 1M / 981.000 | 1M / 983.667 |
| Finger Turn Hard | 8 | 1M / 965.667 | 50k / 325.333 |
| Acrobot Swingup | 6 | 1M / 0 | 1M / 0 |
| Acrobot Swingup | 7 | 1M / 0 | 1M / 0 |
| Acrobot Swingup | 8 | 400k / 0 | 250k / 0 |

`logs/rl/sac_state_lewm_3env_1m_v1/plan.txt` schedules seeds 1–5, with the
same 1M/batch-256/replay/evaluation settings, LeWM projection 192, hidden 2048,
SIGReg weight 0.09, clip-reward horizon 3, and no CURL. Only ten logs are local:

| Task | Locally available seed: last step / return |
|---|---|
| Reacher Hard | 1: 1M / 973.333; 2: 1M / 986.333; 3: 950k / 987.667 |
| Finger Turn Hard | 1: 1M / 965; 2: 1M / 656.333; 3: 1M / 0; 4: 900k / 941 |
| Acrobot Swingup | 1: 1M / 0; 2: 1M / 0; 3: 1M / 0 |

Do not average these into a matched three-method result: seeds and endpoints
differ. Launch templates are `launch_state_sac_taco_3env.sh`,
`run_state_sac_taco_worker.sh`, and `launch_state_sac_lewm_3env.sh`.
For fresh replications use `SEEDS="6 7 8" NUM_STEPS=1000000` with a new
`RUN_ID`, preserve resolved config/revision, and match auxiliary sequence span
explicitly. The pilot and visual launch budgets are in the
[benchmark record](rl_reacher_hard_benchmark.md). No long training was run for
this audit and no logs, checkpoints, or experiment outputs were removed.

## Documentation dispositions

| Reviewed file | Decision | Rationale and completed action |
|---|---|---|
| `docs/rl_reacher_hard_benchmark.md` | UPDATE | Preserved unique seeds, budgets, commands, pilot numbers and limitations; verified endpoints; separated hidden-target history from corrected visible-target launch families. |
| `docs/rl_repository_audit.md` | UPDATE | Replaced stale validation claims and repeated usage text with findings, complete loss/gradient inventory, TD-MPC2 comparison, preserved state evidence, and citations. |
| `docs/rl_representation_learning.md` | UPDATE | Canonical learner/loss/modality/MPC usage guide, explicit invalid configs and horizon semantics, checkpoint behavior, and experiment links. |
| `docs/rl_sac_taco_state.md` | MERGE | Its useful presets and SAC gradient details duplicate the canonical guide/audit. Consolidated them there; retained a short link stub and removed duplicate navigation. |
| `docs/rl_swe_handoff.md` | UPDATE | One concise current handoff with architecture, commands, acceptance criteria, known issues, and open questions. |

## Citations added or corrected

- [SAC](https://arxiv.org/abs/1812.05905): base soft critic, actor, and entropy-temperature objectives; kept separate from reward-only MPC TD.
- [DrQ-v2 paper](https://arxiv.org/abs/2107.09645) and newly cited
  [reference sampler](https://github.com/facebookresearch/drqv2/blob/main/utils.py):
  noisy actor and straight-through bounds. Code citation added to manuscript.
- [TACO](https://arxiv.org/abs/2306.13229) and newly cited
  [authors' implementation](https://github.com/FrankZheng2022/TACO/blob/main/agents/taco.py):
  temporal contrast/add-ons; corrected exact-reproduction language and explained
  local normalization/action/critic differences. Local reference is `84c38e34f4f9`.
  The benchmark now attributes the 51% Reacher claim to the paper's
  no-action-representation ablation rather than its full TACO method.
- [TD-MPC2](https://arxiv.org/abs/2310.16828), newly cited
  [planner/learner](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py),
  and [two-hot utilities](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/math.py):
  weighted search and two-hot regression; corrected claims of latent-policy
  proposals, terminal policy use, evaluation behavior, and full rollout training.
- [CURL](https://proceedings.mlr.press/v119/laskin20a.html): bilinear view contrast;
  clarified that this adapter does not use standalone CURL's momentum encoder.
- [LeWM](https://arxiv.org/abs/2603.19312) and
  [LeJEPA](https://arxiv.org/abs/2511.08544): prediction/SIGReg motivation;
  retained feature-adapter distinction and documented actual gradient paths.
- [PLDM](https://arxiv.org/abs/2502.14819): corrected author initial from U. to
  V. Sobal in manuscript; distinguished this action-free regularizer from the
  paper's latent dynamics/planning method. JEPA/DINO-WM/VICReg references remain
  motivations rather than proofs about the local adapters.

## Verification

The pre-change focused suite had **9 failures, 27 passes**. Fixtures from before
sequence training omitted the configured span; one synthetic Hydra fixture also
omitted `auxiliary.sequence_length`. They were corrected rather than adding
silent truncation or configuration recovery.

Final result: **94 tests passed in 10.98 s**. Checks include state/pixel
updates and plans for both algorithms and all three eligible WMs, value off/on,
restore/readiness, detached reward-only TD targets, two-hot endpoints,
first-action/terminal scoring, and two 12-step CPU runner smokes. The smokes
use U=2, K=3, evaluation and checkpoints, confirming 12 real training
interactions and 12 valid replay transitions despite repeated planning.

Exact focused commands run from the repository root:

```bash
PYTHONPATH="$PWD" /home/mprattico/miniconda3/envs/swm-rl/bin/python -m pytest -q tests/rl --tb=short
/home/mprattico/miniconda3/envs/swm-rl/bin/ruff check stable_worldmodel/rl scripts/train/rl_online.py stable_worldmodel/envs/dmcontrol/reacher.py tests/rl
for script in scripts/rl/*.sh scripts/rl/slurm/*/*.sh; do bash -n "$script" || exit; done
git diff --check
git -C Model-free-to-Model-based diff --check
PYTHONPATH="$PWD" /home/mprattico/miniconda3/envs/swm-rl/bin/python -m mkdocs build --strict --site-dir /tmp/rl-audit-site
```

Ruff, shell syntax, and whitespace checks pass. MkDocs builds strictly;
it reports the compatibility SAC link page as intentionally outside navigation.
From `Model-free-to-Model-based`, the exact manuscript command was:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/rl-audit-latex main.tex
```

Compilation passes with all references resolved and no overfull equations.
The PDF is `/tmp/rl-audit-latex/main.pdf`. Build outputs stay outside source.
The root tracks the manuscript as a gitlink without a `.gitmodules` mapping,
so inspect its source diff separately.

This exact read-only script verified **13 presets and three override scenarios**:

```bash
PYTHONPATH="$PWD" /home/mprattico/miniconda3/envs/swm-rl/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scripts.train.rl_online import _replay_history
config_dir = Path('scripts/train/config').resolve()
names = sorted({p.stem for pattern in ('sac*.yaml', 'drqv2*.yaml', 'rl_online.yaml') for p in config_dir.glob(pattern)})
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    for name in names:
        cfg = compose(config_name=name)
        OmegaConf.to_container(cfg, resolve=True)
        assert cfg.algorithm.name in {'sac', 'drqv2'}
        assert cfg.auxiliary.wm.target in {'none', 'infonce', 'lewm', 'jepa', 'pldm', 'consistency'}
        assert _replay_history(cfg) >= 2
    for name in ('sac_state', 'drqv2_state'):
        cfg = compose(config_name=name, overrides=['wm=infonce', 'auxiliary.enabled=true', 'auxiliary.sequence_length=2', 'auxiliary.wm.horizon=1', 'auxiliary.reward_prediction.enabled=true', 'auxiliary.reward_prediction.mode=per_step', 'mpc.enabled=true', 'mpc.horizon=5'])
        assert cfg.auxiliary.sequence_length == 2 and cfg.mpc.horizon == 5
    cfg = compose(config_name='drqv2_reacher_hard_taco', overrides=['wm=lewm', 'auxiliary.wm.horizon=3', 'auxiliary.sequence_length=3'])
    assert cfg.env_kwargs.render_target is True and cfg.auxiliary.wm.target == 'lewm'
print(f'{len(names)} RL presets and 3 override scenarios composed successfully')
print('presets:', ', '.join(names))
PY
```

The real MuJoCo probe used this command; both settings rendered `(36,36,3)`,
with target alpha 0/1 for hidden/visible on both resets:

```bash
PYTHONPATH="$PWD" MUJOCO_GL=egl /home/mprattico/miniconda3/envs/swm-rl/bin/python - <<'PY'
import gymnasium as gym
import stable_worldmodel
for visible in (False, True):
    env = gym.make('swm/ReacherDMControl-v0', task='hard', render_mode='rgb_array', render_target=visible)
    env.reset(seed=0)
    physics = env.unwrapped.env.physics
    alpha = float(physics.named.model.mat_rgba['target', 3])
    frame = env.unwrapped.render(width=36, height=36)
    print('render_target=', visible, 'alpha=', alpha, 'frame=', frame.shape)
    assert alpha == float(visible)
    env.reset(seed=1)
    assert float(env.unwrapped.env.physics.named.model.mat_rgba['target', 3]) == float(visible)
    env.close()
PY
```

These checks establish execution and consistency, not RL performance. No long
training sweep, Slurm submission, W&B write, or benchmark rerun was launched.
