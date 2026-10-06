"""Online state-vector SAC with optional TACO auxiliary representation loss."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import stable_worldmodel  # noqa: F401 — registers built-in environments
from stable_worldmodel.data import ReplayBuffer
from stable_worldmodel.rl.sac_taco import SACTACOAgent

logger = logging.getLogger(__name__)


def _make_env(cfg) -> gym.Env:
    env_id = str(cfg.env)
    # A null mapping is useful for Hydra child configs that replace a parent
    # task's constructor kwargs (an empty mapping otherwise merges with them).
    env_kwargs = (
        {}
        if cfg.env_kwargs is None
        else OmegaConf.to_container(cfg.env_kwargs, resolve=True)
    )
    # Child Hydra configs use null to clear inherited constructor arguments.
    env_kwargs = {
        key: value for key, value in env_kwargs.items() if value is not None
    }
    if env_id.startswith('Meta-World/'):
        try:
            import metaworld  # noqa: F401 — registers Gymnasium environments
        except ImportError as exc:
            raise RuntimeError(
                'Meta-World tasks need the optional dependency. Install with '
                '`pip install -e ".[env,metaworld]"` in the active environment.'
            ) from exc
        env_kwargs.setdefault('seed', int(cfg.seed))
    return gym.make(env_id, **env_kwargs)


def _validate_env(env: gym.Env) -> tuple[int, int]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError('State SAC requires a flat Box observation space')
    if len(env.observation_space.shape) != 1:
        raise ValueError(
            'State SAC requires a one-dimensional observation; got '
            f'{env.observation_space.shape}'
        )
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError('SAC requires a continuous Box action space')
    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
        env.action_space.high, 1.0
    ):
        raise ValueError(
            'SAC expects action bounds [-1, 1]; normalize the environment '
            'action space before training'
        )
    return int(env.observation_space.shape[0]), int(env.action_space.shape[0])


def _evaluate(cfg, agent: SACTACOAgent, step: int) -> dict[str, float]:
    env = _make_env(cfg)
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    try:
        for episode in range(cfg.evaluation.episodes):
            state, info = env.reset(seed=cfg.seed + 100_000 + episode)
            total = 0.0
            length = 0
            done = False
            success_values = []
            while not done and (
                cfg.max_episode_steps <= 0
                or length < cfg.max_episode_steps
            ):
                action = agent.act(state, eval_mode=True)
                state, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                length += 1
                if isinstance(info, dict) and 'success' in info:
                    value = float(np.asarray(info['success']).mean())
                    if np.isfinite(value):
                        success_values.append(value)
                done = bool(terminated or truncated)
            returns.append(total)
            lengths.append(length)
            if success_values:
                successes.append(float(np.max(success_values)))
    finally:
        env.close()

    metrics = {
        'evaluation/episode_reward': float(np.mean(returns)),
        'evaluation/episode_length': float(np.mean(lengths)),
        'train/environment_step': float(step),
    }
    if successes:
        metrics['evaluation/success_rate'] = float(np.mean(successes))
    return metrics


def _commit_segment(
    replay: ReplayBuffer,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    discounts: list[float],
) -> None:
    if not actions:
        return
    action_dim = int(actions[0].shape[0])
    replay.write_episode(
        {
            'state': np.stack(states).astype(np.float32),
            'action': np.concatenate(
                [
                    np.stack(actions).astype(np.float32),
                    np.zeros((1, action_dim), dtype=np.float32),
                ],
                axis=0,
            ),
            'reward': np.asarray([*rewards, 0.0], dtype=np.float32),
            'discount': np.asarray([*discounts, 0.0], dtype=np.float32),
        }
    )


def _save_checkpoint(
    path: Path,
    agent: SACTACOAgent,
    replay: ReplayBuffer,
    step: int,
    episode: int,
    cfg,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'agent': agent.state_dict(),
            'replay': replay,
            'step': step,
            'episode': episode,
            'config': OmegaConf.to_container(cfg, resolve=True),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        path,
    )


@hydra.main(version_base=None, config_path='./config', config_name='sac_taco_state')
def run(cfg) -> None:
    logging.basicConfig(level=logging.INFO)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(int(cfg.torch_threads))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    env = _make_env(cfg)
    env.action_space.seed(cfg.seed)
    obs_dim, action_dim = _validate_env(env)
    agent = SACTACOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=cfg.agent.hidden_dim,
        feature_dim=cfg.agent.feature_dim,
        lr=cfg.agent.lr,
        taco_lr=cfg.taco.lr,
        alpha_lr=cfg.agent.alpha_lr,
        discount=cfg.agent.discount,
        tau=cfg.agent.tau,
        initial_alpha=cfg.agent.initial_alpha,
        target_entropy=cfg.agent.target_entropy,
        taco_enabled=cfg.taco.enabled,
        taco_horizon=cfg.taco.horizon,
        taco_feature_dim=cfg.taco.feature_dim,
        taco_hidden_dim=cfg.taco.hidden_dim,
        taco_latent_action_dim=cfg.taco.latent_action_dim,
        taco_reward_prediction=cfg.taco.reward_prediction,
        taco_reward_weight=cfg.taco.reward_weight,
    )
    horizon = cfg.taco.horizon if cfg.taco.enabled else 1
    replay = ReplayBuffer(
        max_steps=cfg.replay.capacity,
        history_len=horizon + 1,
    )
    step, episode = 0, 0
    if cfg.resume:
        checkpoint = torch.load(
            cfg.resume, map_location='cpu', weights_only=False
        )
        agent.load_state_dict(checkpoint['agent'])
        replay = checkpoint['replay']
        step, episode = int(checkpoint['step']), int(checkpoint['episode'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if torch.cuda.is_available() and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        logger.info('Resumed at step=%d episode=%d', step, episode)

    if cfg.evaluation_only:
        metrics = _evaluate(cfg, agent, step)
        logger.info('evaluation metrics=%s', metrics)
        env.close()
        return

    logger.info(
        'state SAC env=%s taco=%s obs_dim=%d action_dim=%d device=%s',
        cfg.env,
        cfg.taco.enabled,
        obs_dim,
        action_dim,
        device,
    )
    wandb_run = None
    if cfg.wandb.enable:
        import wandb

        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    state, _ = env.reset(seed=cfg.seed)
    state = np.asarray(state, dtype=np.float32)
    segment_states = [state]
    segment_actions: list[np.ndarray] = []
    segment_rewards: list[float] = []
    segment_discounts: list[float] = []
    episode_reward = 0.0
    episode_length = 0
    last_metrics: dict[str, float] = {}
    start_time = time.monotonic()
    checkpoint_dir = Path(cfg.checkpoint_dir)
    try:
        while step < cfg.num_steps:
            if step < cfg.learning_starts:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = agent.act(state, eval_mode=False)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32)
            episode_length += 1
            hit_step_limit = (
                cfg.max_episode_steps > 0
                and episode_length >= cfg.max_episode_steps
            )
            if hit_step_limit and not terminated and not truncated:
                truncated = True
            done = bool(terminated or truncated)

            # Bootstrap through time-limit truncations, but not true terminal
            # transitions. This keeps finite episode caps from changing SAC's
            # Bellman target.
            bootstrap = 0.0 if terminated else 1.0
            segment_actions.append(np.asarray(action, dtype=np.float32))
            segment_rewards.append(float(reward))
            segment_discounts.append(bootstrap)
            segment_states.append(next_state)
            episode_reward += float(reward)
            step += 1
            state = next_state

            if done or len(segment_actions) >= cfg.replay.commit_interval:
                _commit_segment(
                    replay,
                    segment_states,
                    segment_actions,
                    segment_rewards,
                    segment_discounts,
                )
                segment_states = [next_state]
                segment_actions, segment_rewards, segment_discounts = [], [], []

            if done:
                episode += 1
                log_metrics = {
                    'train/environment_step': step,
                    'train/episode': episode,
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_length,
                }
                if wandb_run:
                    wandb_run.log(log_metrics, step=step)
                logger.info(
                    'step=%d episode=%d reward=%.3f length=%d success=%s',
                    step,
                    episode,
                    episode_reward,
                    episode_length,
                    info.get('success') if isinstance(info, dict) else None,
                )
                state, _ = env.reset()
                state = np.asarray(state, dtype=np.float32)
                segment_states = [state]
                segment_actions, segment_rewards, segment_discounts = [], [], []
                episode_reward, episode_length = 0.0, 0

            if (
                step >= cfg.learning_starts
                and replay.num_valid_ends(horizon + 1) >= cfg.batch_size
                and step % cfg.update_frequency == 0
            ):
                for _ in range(cfg.updates_per_step):
                    sample = replay.sample(
                        cfg.batch_size,
                        history_len=horizon + 1,
                        step=step,
                    )
                    batch = {
                        key: torch.as_tensor(value, device=device)
                        for key, value in sample.items()
                    }
                    last_metrics = agent.update(batch)

            if step % cfg.evaluation.frequency == 0:
                eval_metrics = _evaluate(cfg, agent, step)
                logger.info('evaluation %s', eval_metrics)
                if wandb_run:
                    wandb_run.log(eval_metrics, step=step)

            if step % cfg.logging.frequency == 0:
                metrics = {
                    'train/environment_step': float(step),
                    'system/fps': step / max(time.monotonic() - start_time, 1e-6),
                    **last_metrics,
                }
                logger.info('step=%d metrics=%s', step, metrics)
                if wandb_run:
                    wandb_run.log(metrics, step=step)

            if step % cfg.checkpoint_frequency == 0 or step == cfg.num_steps:
                if step == cfg.num_steps and segment_actions:
                    _commit_segment(
                        replay,
                        segment_states,
                        segment_actions,
                        segment_rewards,
                        segment_discounts,
                    )
                    segment_states = [state]
                    segment_actions, segment_rewards, segment_discounts = (
                        [],
                        [],
                        [],
                    )
                _save_checkpoint(
                    checkpoint_dir / f'sac_taco_state_step_{step}.pt',
                    agent,
                    replay,
                    step,
                    episode,
                    cfg,
                )
    finally:
        # Persist any final partial trajectory so the run can resume without
        # silently dropping its most recent transitions.
        if segment_actions:
            _commit_segment(
                replay,
                segment_states,
                segment_actions,
                segment_rewards,
                segment_discounts,
            )
        env.close()
        if wandb_run:
            wandb_run.finish()


if __name__ == '__main__':
    run()
