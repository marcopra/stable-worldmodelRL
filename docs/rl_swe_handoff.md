# Online RL SWE handoff

Updated 2026-10-14. Read the [usage guide](rl_representation_learning.md) for
supported configurations and the [audit](rl_repository_audit.md) for findings,
loss math, research questions, citations, and preserved experiment evidence.

## Architecture and key files

- `rl/algorithms/{sac,drqv2}.py`: original base actor/critic objectives and
  optimizer ownership, plus auxiliary updates and action selection.
- `rl/representation.py`: one primary loss, optional reward/CURL; shared encoder
  gradients. Eligible MPC predictors: one-step InfoNCE, LeWM, consistency.
- `rl/mpc.py`: weighted-elite planner and optional separate latent two-hot Q.
  Reuses the selected WM and per-step reward head; no duplicate dynamics.
- `scripts/train/rl_online.py`: real collection, segment-bounded replay,
  evaluation, checkpoint/RNG state, and SAC/DrQ dispatch.
- `scripts/train/config/{rl_online.yaml,agent/,wm/}`: Hydra composition;
  the `wm` group is packaged as `auxiliary.wm`.
- `tests/rl/test_mpc.py`: model reuse, checkpoint semantics, gradient/TD targets,
  numeric rejection, terminal scoring, and real-interaction regression checks.
- `Model-free-to-Model-based/main.tex`: corrected manuscript in its own checkout.

Paths beginning `rl/` are under `stable_worldmodel/`. Both algorithms support
state and pixels. `wm=none auxiliary.enabled=false` is the pure baseline.
MPC requires positive-weight eligible WM, `auxiliary.wm.horizon=1`, and
positive-weight reward prediction in `per_step` mode with horizon 1. Replay
span `auxiliary.sequence_length=U` and planner `mpc.horizon=K` are independent.
State+CURL, none/JEPA/PLDM with MPC, clip-return MPC rewards, untrained MPC,
and invalid numeric settings fail loudly.

## Commands

Activate `swm-rl`, then run from the repository root:

```bash
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=sac_state wm=none auxiliary.enabled=false
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=sac_taco_state
PYTHONPATH="$PWD" python scripts/train/rl_online.py --config-name=rl_online agent=drqv2 observation.mode=state wm=infonce auxiliary.enabled=true auxiliary.wm.horizon=1 auxiliary.sequence_length=5 auxiliary.reward_prediction.enabled=true auxiliary.reward_prediction.mode=per_step auxiliary.reward_prediction.horizon=1 mpc.enabled=true mpc.start_steps=10000 mpc.horizon=5
```

The [benchmark record](rl_reacher_hard_benchmark.md) has historical pilot
commands/settings and corrected visible-target recipes. Slurm recipes live in
`scripts/rl/slurm/sac_state_retest/README.md`; the default TACO/MPC batches
1024/256 are not a matched ablation. Do not launch sweeps as verification.

## Acceptance criteria

- Disabled auxiliaries allocate no auxiliary modules; MPC rejects missing,
  disabled, zero-weight, or non-composable dynamics/reward predictors.
- DrQ forwards configured U; short batches fail before any learner update.
- With MPC enabled, the SAC/DrQ actor and base critic consume the WM latent.
  A positive `mpc.value_coef` adds a reward-only MPC Q head; at zero, planning
  bootstraps from the base critic. SAC's zero-weight planner includes entropy.
  Value targets and target heads have no gradients.
- Planning reuses the same WM/reward objects, rolls K times, executes only the
  first action, and leaves interaction/replay counters unchanged.
- Model-update count gates planning and survives restore; reset clears warm starts.
- Both algorithms/modalities and all three eligible WMs work with value on/off.
- Preserve resolved configs, source revisions, seeds, budgets, logs and checkpoints.

## Known issues and next research work

MPC's default value coefficient 0.1 is unvalidated. With MPC enabled, the
actor and base critic consume projected latents. A positive value coefficient
adds a separate reward-return head; a zero coefficient bootstraps from the
original SAC/DrQ critic on those latents. Policy proposals remain open-loop at
the current latent, and value training uses real replay states. There is no
termination model. Test compounding model
error, InfoNCE coordinate ambiguity, latent collapse, and value error at imagined
states before claiming MPC improvement. Run matched actor/MPC and value-off/on
ablations, then compare K/U and eligible losses with paired seeds.

Old MPC checkpoints without model-update count/objective metadata are rejected;
earlier SAC soft terminal heads must be retrained. Corrected MPC recipes use
new reward-TD output families; visual Reacher now explicitly shows the target
and writes new directories. Historical hidden-target scores are kept separately.
No corrected MPC advantage or exact source-paper reproduction is established.

The manuscript is a nested Git checkout tracked as a gitlink, but the root has
no `.gitmodules` mapping. Review its diff separately and preserve/publish its
changes explicitly; a root-only diff does not contain `main.tex`. This review
leaves repository wiring unchanged.

## Verification

Baseline: 9 failed / 27 passed. Final: **94 RL tests passed**, including two
12-step CPU collection/evaluation/checkpoint tests with K=3 and U=2. All 13 RL
presets and three override scenarios compose. Real Reacher probes verify
visibility across resets and 36px RGB rendering. Ruff, shell syntax, whitespace,
strict MkDocs build, and LaTeX compilation pass. No long training was launched.
Exact commands and probe scripts are in the [audit](rl_repository_audit.md#verification).
