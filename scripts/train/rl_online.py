"""Shared online runner for SAC and DrQ-v2.

Examples::

    python scripts/train/rl_online.py --config-name=sac_state
    python scripts/train/rl_online.py --config-name=drqv2_state
    python scripts/train/rl_online.py --config-name=drqv2
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

import stable_worldmodel  # noqa: F401 — registers built-in environments
from stable_worldmodel.data import ReplayBuffer
from stable_worldmodel.rl import DrQV2Agent, SACAgent

logger = logging.getLogger(__name__)


def _env_kwargs(cfg) -> dict:
    if cfg.env_kwargs is None:
        kwargs = {}
    else:
        kwargs = OmegaConf.to_container(cfg.env_kwargs, resolve=True)
    return {key: value for key, value in kwargs.items() if value is not None}


def _make_env(cfg) -> gym.Env:
    kwargs = _env_kwargs(cfg)
    env_id = str(cfg.env)
    if env_id.startswith('Meta-World/'):
        try:
            import metaworld  # noqa: F401 — registers Gymnasium environments
        except ImportError as exc:
            raise RuntimeError(
                'Meta-World tasks need the optional dependency. Install with '
                '`pip install -e ".[env,metaworld]"`.'
            ) from exc
        kwargs.setdefault('seed', int(cfg.seed))
    if cfg.observation.mode == 'pixels':
        kwargs.setdefault('render_mode', 'rgb_array')
    return gym.make(env_id, **kwargs)


def _space_for_state(env: gym.Env, state_key: str | None):
    space = env.observation_space
    if isinstance(space, gym.spaces.Dict):
        if not state_key:
            raise ValueError(
                'Dict state observations require observation.state_key'
            )
        if state_key not in space.spaces:
            raise KeyError(
                f'observation.state_key={state_key!r} is absent from '
                f'{tuple(space.spaces)}'
            )
        space = space.spaces[state_key]
    if not isinstance(space, gym.spaces.Box):
        raise TypeError('State observations must select a numeric Box space')
    if not np.issubdtype(space.dtype, np.number):
        raise TypeError(
            f'State observations need numeric dtype, got {space.dtype}'
        )
    return space


def _validate_env(env: gym.Env, cfg) -> tuple[int, ...]:
    mode = str(cfg.observation.mode)
    if mode == 'state':
        state_key = cfg.observation.state_key
        state_space = _space_for_state(env, state_key)
        shape = tuple(int(dim) for dim in state_space.shape)
    elif mode == 'pixels':
        if not hasattr(env.unwrapped, 'render'):
            raise TypeError('Pixel observations require env.render()')
        stack = int(cfg.observation.frame_stack)
        size = int(cfg.observation.image_size)
        if stack < 1 or size < 36:
            raise ValueError(
                'Pixel mode requires frame_stack >= 1 and image_size >= 36'
            )
        shape = (3 * stack, size, size)
    else:
        raise ValueError("observation.mode must be 'state' or 'pixels'")

    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError('SAC and DrQ-v2 require a continuous Box action space')
    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
        env.action_space.high, 1.0
    ):
        raise ValueError(
            'RL agents expect actions bounded in [-1, 1]; normalize the '
            'environment action space before training'
        )
    return shape


def _state_observation(raw_observation, state_key: str | None) -> np.ndarray:
    if isinstance(raw_observation, dict):
        if not state_key:
            raise ValueError(
                'Dict state observations require observation.state_key'
            )
        if state_key not in raw_observation:
            raise KeyError(f'State observation has no key {state_key!r}')
        raw_observation = raw_observation[state_key]
    observation = np.asarray(raw_observation)
    if not np.issubdtype(observation.dtype, np.number):
        raise TypeError('Selected state observation must be numeric')
    if not np.all(np.isfinite(observation)):
        raise ValueError('State observation contains non-finite values')
    return observation.astype(np.float32, copy=False).reshape(-1)


def _pixels(env: gym.Env, size: int) -> np.ndarray:
    render_env = env.unwrapped
    try:
        frame = render_env.render(width=size, height=size)
    except TypeError:
        frame = env.render()
    if frame is None:
        raise RuntimeError('Environment render() returned no RGB frame')
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert('RGB')
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8).copy()


def _observation(
    env: gym.Env,
    raw_observation,
    cfg,
    frames: deque[np.ndarray],
    *,
    reset: bool = False,
) -> np.ndarray:
    if cfg.observation.mode == 'state':
        return _state_observation(raw_observation, cfg.observation.state_key)
    frame = _pixels(env, int(cfg.observation.image_size))
    frame_stack = int(cfg.observation.frame_stack)
    if reset:
        frames.clear()
        for _ in range(frame_stack):
            frames.append(frame)
    else:
        frames.append(frame)
    return np.concatenate(tuple(frames), axis=-1)


def _world_model_config(cfg) -> dict:
    return OmegaConf.to_container(cfg.auxiliary.wm, resolve=True)


def _optional_loss_config(cfg, name: str) -> dict:
    return OmegaConf.to_container(cfg.auxiliary[name], resolve=True)


def _mpc_config(cfg) -> dict:
    config = cfg.get('mpc', {})
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _selected_auxiliary_losses(cfg) -> list[str]:
    if not cfg.auxiliary.enabled:
        return []
    selected = []
    if cfg.auxiliary.wm.enabled and cfg.auxiliary.wm.target not in {
        'none',
        'null',
        '',
    }:
        selected.append(str(cfg.auxiliary.wm.target))
    for name in ('reward_prediction', 'curl'):
        if cfg.auxiliary[name].enabled:
            selected.append(name)
    return selected


def _bootstrap_discount(terminated: bool) -> float:
    """Keep bootstrapping across time limits; stop only at true terminals."""
    return 0.0 if terminated else 1.0


def _replay_history(cfg) -> int:
    horizons = [1]
    if cfg.auxiliary.enabled:
        if cfg.auxiliary.wm.enabled and cfg.auxiliary.wm.target not in {
            'none',
            'null',
            '',
        }:
            horizons.append(int(cfg.auxiliary.wm.horizon))
        if cfg.auxiliary.reward_prediction.enabled:
            horizons.append(int(cfg.auxiliary.reward_prediction.horizon))
    nstep = int(cfg.agent.nstep) if cfg.algorithm.name == 'drqv2' else 1
    mpc_config = _mpc_config(cfg)
    if mpc_config.get('enabled', False):
        horizons.append(int(mpc_config.get('horizon', 5)))
    return max(2, max(horizons) + 1, nstep + 1)


def _make_agent(cfg, observation_shape: tuple[int, ...], action_dim: int):
    world_model_config = _world_model_config(cfg)
    reward_prediction_config = _optional_loss_config(cfg, 'reward_prediction')
    curl_config = _optional_loss_config(cfg, 'curl')
    mpc_config = _mpc_config(cfg)
    common = {
        'obs_shape': observation_shape,
        'action_dim': action_dim,
        'observation_mode': str(cfg.observation.mode),
        'device': str(cfg.device),
        'hidden_dim': int(cfg.agent.hidden_dim),
        'feature_dim': int(cfg.agent.feature_dim),
        'discount': float(cfg.agent.discount),
        'auxiliary_enabled': bool(cfg.auxiliary.enabled),
        'world_model_config': world_model_config,
        'reward_prediction_config': reward_prediction_config,
        'curl_config': curl_config,
        'mpc_config': mpc_config,
        'augmentation_pad': int(cfg.agent.augmentation_pad),
    }
    if cfg.algorithm.name == 'sac':
        return SACAgent(
            **common,
            lr=float(cfg.agent.lr),
            alpha_lr=float(cfg.agent.alpha_lr),
            tau=float(cfg.agent.tau),
            initial_alpha=float(cfg.agent.initial_alpha),
            target_entropy=(
                None
                if cfg.agent.target_entropy is None
                else float(cfg.agent.target_entropy)
            ),
        )
    if cfg.algorithm.name == 'drqv2':
        return DrQV2Agent(
            obs_shape=observation_shape,
            action_shape=(action_dim,),
            observation_mode=str(cfg.observation.mode),
            device=str(cfg.device),
            lr=float(cfg.agent.lr),
            feature_dim=int(cfg.agent.feature_dim),
            hidden_dim=int(cfg.agent.hidden_dim),
            critic_target_tau=float(cfg.agent.critic_target_tau),
            discount=float(cfg.agent.discount),
            stddev=float(cfg.agent.stddev),
            stddev_final=float(cfg.agent.stddev_final),
            stddev_schedule_steps=int(cfg.agent.stddev_schedule_steps),
            stddev_clip=float(cfg.agent.stddev_clip),
            num_expl_steps=int(cfg.agent.num_expl_steps),
            auxiliary_enabled=bool(cfg.auxiliary.enabled),
            world_model_config=world_model_config,
            reward_prediction_config=reward_prediction_config,
            curl_config=curl_config,
            mpc_config=mpc_config,
            augmentation_pad=int(cfg.agent.augmentation_pad),
            nstep=int(cfg.agent.nstep),
        )
    raise ValueError(f'Unknown algorithm {cfg.algorithm.name!r}')


def _commit_segment(
    replay: ReplayBuffer,
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    discounts: list[float],
) -> None:
    if not actions:
        return
    action_dim = int(actions[0].shape[0])
    replay.write_episode(
        {
            'observation': np.stack(observations),
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


def _checkpoint(
    path: Path,
    agent,
    replay: ReplayBuffer,
    step: int,
    episode: int,
    cfg,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'algorithm': str(cfg.algorithm.name),
            'world_model': str(cfg.auxiliary.wm.target),
            'auxiliary_losses': _selected_auxiliary_losses(cfg),
            'mpc_enabled': bool(_mpc_config(cfg).get('enabled', False)),
            'observation_mode': str(cfg.observation.mode),
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


def _evaluate(cfg, agent, step: int) -> dict[str, float]:
    env = _make_env(cfg)
    env.action_space.seed(int(cfg.seed) + 100_000)
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    try:
        for episode in range(int(cfg.evaluation.episodes)):
            if hasattr(agent, 'reset'):
                agent.reset()
            raw, info = env.reset(seed=int(cfg.seed) + 100_000 + episode)
            frames: deque[np.ndarray] = deque(
                maxlen=int(cfg.observation.frame_stack)
            )
            observation = _observation(env, raw, cfg, frames, reset=True)
            total, length, done = 0.0, 0, False
            success_values = []
            while not done and (
                cfg.max_episode_steps <= 0 or length < cfg.max_episode_steps
            ):
                # Any latent planning stays inside act(); it does not advance
                # the real environment-step counter below.
                action = agent.act(observation, step=step, eval_mode=True)
                raw, reward, terminated, truncated, info = env.step(action)
                observation = _observation(env, raw, cfg, frames)
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
        if hasattr(agent, 'reset'):
            agent.reset()
        env.close()

    metrics = {
        'evaluation/episode_reward': float(np.mean(returns)),
        'evaluation/episode_length': float(np.mean(lengths)),
        'train/environment_step': float(step),
    }
    if successes:
        metrics['evaluation/success_rate'] = float(np.mean(successes))
    return metrics


def train(cfg) -> None:
    logging.basicConfig(level=logging.INFO)
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    torch.set_num_threads(int(cfg.torch_threads))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    device = torch.device(cfg.device)
    env = _make_env(cfg)
    env.action_space.seed(int(cfg.seed))
    observation_shape = _validate_env(env, cfg)
    agent = _make_agent(cfg, observation_shape, int(env.action_space.shape[0]))
    history_len = _replay_history(cfg)
    replay = ReplayBuffer(
        max_steps=int(cfg.replay.capacity),
        history_len=history_len,
    )
    step, episode = 0, 0
    if cfg.resume:
        checkpoint = torch.load(
            cfg.resume, map_location='cpu', weights_only=False
        )
        saved_algorithm = checkpoint.get('algorithm')
        if saved_algorithm and saved_algorithm != cfg.algorithm.name:
            raise ValueError(
                f'Checkpoint algorithm {saved_algorithm!r} does not match '
                f'configured algorithm {cfg.algorithm.name!r}'
            )
        saved_world_model = checkpoint.get('world_model')
        if saved_world_model and saved_world_model != cfg.auxiliary.wm.target:
            raise ValueError(
                f'Checkpoint world model {saved_world_model!r} does not '
                'match configured world model '
                f'{cfg.auxiliary.wm.target!r}'
            )
        saved_mpc = checkpoint.get('mpc_enabled')
        if saved_mpc is not None and bool(saved_mpc) != bool(cfg.mpc.enabled):
            raise ValueError(
                f'Checkpoint MPC setting {saved_mpc!r} does not match '
                f'configured setting {bool(cfg.mpc.enabled)!r}'
            )
        saved_observation_mode = checkpoint.get('observation_mode')
        if (
            saved_observation_mode
            and saved_observation_mode != cfg.observation.mode
        ):
            raise ValueError(
                f'Checkpoint observation mode {saved_observation_mode!r} '
                f'does not match configured mode {cfg.observation.mode!r}'
            )
        saved_auxiliary_losses = checkpoint.get('auxiliary_losses')
        if (
            saved_auxiliary_losses is not None
            and saved_auxiliary_losses != _selected_auxiliary_losses(cfg)
        ):
            raise ValueError(
                f'Checkpoint auxiliary losses {saved_auxiliary_losses!r} '
                f'do not match configured losses '
                f'{_selected_auxiliary_losses(cfg)!r}'
            )
        agent.load_state_dict(checkpoint['agent'])
        replay = checkpoint.get('replay', checkpoint.get('buffer'))
        if replay is None:
            raise ValueError('Checkpoint has no replay buffer')
        step, episode = int(checkpoint['step']), int(checkpoint['episode'])
        if checkpoint.get('numpy_rng') is not None:
            np.random.set_state(checkpoint['numpy_rng'])
        if checkpoint.get('python_rng') is not None:
            random.setstate(checkpoint['python_rng'])
        if checkpoint.get('torch_rng') is not None:
            torch.set_rng_state(checkpoint['torch_rng'])
        if (
            torch.cuda.is_available()
            and checkpoint.get('cuda_rng') is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        logger.info('Resumed at step=%d episode=%d', step, episode)

    if cfg.evaluation_only:
        metrics = _evaluate(cfg, agent, step)
        logger.info('evaluation metrics=%s', metrics)
        env.close()
        return

    algorithm_name = str(cfg.algorithm.name)
    logger.info(
        'algorithm=%s observation=%s env=%s auxiliary=%s wm=%s '
        'reward_prediction=%s curl=%s mpc=%s device=%s',
        algorithm_name,
        cfg.observation.mode,
        cfg.env,
        cfg.auxiliary.enabled,
        cfg.auxiliary.wm.target,
        cfg.auxiliary.reward_prediction.enabled,
        cfg.auxiliary.curl.enabled,
        cfg.mpc.enabled,
        device,
    )
    wandb_run = None
    if cfg.wandb.enable:
        import wandb

        wandb_kwargs = {
            'project': cfg.wandb.project,
            'name': cfg.wandb.name,
            'group': cfg.wandb.group,
            'tags': list(cfg.wandb.tags),
            'config': OmegaConf.to_container(cfg, resolve=True),
        }
        if cfg.wandb.get('entity'):
            wandb_kwargs['entity'] = cfg.wandb.entity
        if cfg.wandb.get('run_id'):
            wandb_kwargs['id'] = cfg.wandb.run_id
            wandb_kwargs['resume'] = 'allow'
        wandb_run = wandb.init(**wandb_kwargs)
        logger.info('W&B run: %s', wandb_run.url)

    raw_observation, _ = env.reset(seed=int(cfg.seed))
    frames: deque[np.ndarray] = deque(maxlen=int(cfg.observation.frame_stack))
    observation = _observation(env, raw_observation, cfg, frames, reset=True)
    segment_observations = [observation]
    segment_actions: list[np.ndarray] = []
    segment_rewards: list[float] = []
    segment_discounts: list[float] = []
    episode_reward, episode_length = 0.0, 0
    last_metrics: dict[str, float] = {}
    start_time = time.monotonic()
    checkpoint_dir = Path(cfg.checkpoint_dir)
    try:
        while step < int(cfg.num_steps):
            learning_starts = int(cfg.learning_starts)
            if algorithm_name == 'drqv2':
                learning_starts = int(cfg.agent.num_expl_steps)
            action = (
                env.action_space.sample().astype(np.float32)
                if step < learning_starts
                else agent.act(observation, step=step, eval_mode=False)
            )
            # Planning rollouts happen inside agent.act(); this one real
            # environment transition is the only step counted below.
            next_raw, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            hit_step_limit = (
                cfg.max_episode_steps > 0
                and episode_length >= cfg.max_episode_steps
            )
            if hit_step_limit and not terminated and not truncated:
                truncated = True
            done = bool(terminated or truncated)
            next_observation = _observation(env, next_raw, cfg, frames)

            segment_actions.append(np.asarray(action, dtype=np.float32))
            segment_rewards.append(float(reward))
            # Gymnasium truncations are time limits, so value bootstrapping
            # remains enabled. Only a true MDP terminal suppresses bootstrap.
            segment_discounts.append(_bootstrap_discount(bool(terminated)))
            segment_observations.append(next_observation)
            episode_reward += float(reward)
            step += 1
            observation = next_observation

            if done or len(segment_actions) >= int(cfg.replay.commit_interval):
                _commit_segment(
                    replay,
                    segment_observations,
                    segment_actions,
                    segment_rewards,
                    segment_discounts,
                )
                segment_observations = [next_observation]
                segment_actions, segment_rewards, segment_discounts = (
                    [],
                    [],
                    [],
                )

            if done:
                episode += 1
                episode_metrics = {
                    'train/environment_step': step,
                    'train/episode': episode,
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_length,
                }
                if wandb_run:
                    wandb_run.log(episode_metrics, step=step)
                logger.info(
                    'algorithm=%s step=%d episode=%d reward=%.3f length=%d success=%s',
                    algorithm_name,
                    step,
                    episode,
                    episode_reward,
                    episode_length,
                    info.get('success') if isinstance(info, dict) else None,
                )
                raw_observation, _ = env.reset()
                if hasattr(agent, 'reset'):
                    agent.reset()
                observation = _observation(
                    env, raw_observation, cfg, frames, reset=True
                )
                segment_observations = [observation]
                segment_actions, segment_rewards, segment_discounts = (
                    [],
                    [],
                    [],
                )
                episode_reward, episode_length = 0.0, 0

            if (
                step >= learning_starts
                and replay.num_valid_ends(history_len) >= int(cfg.batch_size)
                and step % int(cfg.update_frequency) == 0
            ):
                for _ in range(int(cfg.updates_per_step)):
                    sample = replay.sample(
                        int(cfg.batch_size), history_len=history_len, step=step
                    )
                    batch = {
                        key: torch.as_tensor(value, device=device)
                        for key, value in sample.items()
                    }
                    last_metrics = agent.update(batch, step=step)

            if step % int(cfg.evaluation.frequency) == 0:
                eval_metrics = _evaluate(cfg, agent, step)
                logger.info('evaluation %s', eval_metrics)
                if wandb_run:
                    wandb_run.log(eval_metrics | last_metrics, step=step)

            if step % int(cfg.logging.frequency) == 0:
                metrics = {
                    'algorithm/name': algorithm_name,
                    'train/environment_step': float(step),
                    'system/fps': step
                    / max(time.monotonic() - start_time, 1e-6),
                    **last_metrics,
                }
                logger.info('step=%d metrics=%s', step, metrics)
                if wandb_run:
                    wandb_run.log(metrics, step=step)

            if step % int(cfg.checkpoint_frequency) == 0 or step == int(
                cfg.num_steps
            ):
                if step == int(cfg.num_steps) and segment_actions:
                    _commit_segment(
                        replay,
                        segment_observations,
                        segment_actions,
                        segment_rewards,
                        segment_discounts,
                    )
                    segment_observations = [observation]
                    segment_actions, segment_rewards, segment_discounts = (
                        [],
                        [],
                        [],
                    )
                _checkpoint(
                    checkpoint_dir / f'{algorithm_name}_step_{step}.pt',
                    agent,
                    replay,
                    step,
                    episode,
                    cfg,
                )
    finally:
        if segment_actions:
            _commit_segment(
                replay,
                segment_observations,
                segment_actions,
                segment_rewards,
                segment_discounts,
            )
        env.close()
        if wandb_run:
            wandb_run.finish()


@hydra.main(version_base=None, config_path='./config', config_name='rl_online')
def run(cfg) -> None:
    train(cfg)


if __name__ == '__main__':
    run()
