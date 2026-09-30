"""Online DrQ-v2 training with optional temporal representation learning.

Example: ``python scripts/train/drqv2_online.py representation.loss=infonce``.
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

import stable_worldmodel  # noqa: F401 — registers built-in Gymnasium environments
from stable_worldmodel.data import ReplayBuffer
from stable_worldmodel.rl import DrQV2Agent

logger = logging.getLogger(__name__)


def _pixels(env: gym.Env, size: int) -> np.ndarray:
    try:
        frame = env.render(width=size, height=size)
    except TypeError:
        frame = env.render()
    if frame is None:
        raise RuntimeError('Environment render() returned no RGB frame')
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert('RGB')
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8).copy()


def _stacked_pixels(
    env: gym.Env,
    size: int,
    frames: deque[np.ndarray],
    frame_stack: int,
    reset: bool = False,
) -> np.ndarray:
    frame = _pixels(env, size)
    if reset:
        frames.clear()
        for _ in range(frame_stack):
            frames.append(frame)
    else:
        frames.append(frame)
    return np.concatenate(tuple(frames), axis=-1)


def _make_env(cfg) -> gym.Env:
    env_kwargs = OmegaConf.to_container(cfg.env_kwargs, resolve=True)
    return gym.make(cfg.env, render_mode='rgb_array', **env_kwargs)


def _evaluate(cfg, agent: DrQV2Agent, step: int) -> tuple[float, float]:
    env = _make_env(cfg)
    env.action_space.seed(cfg.seed)
    returns, lengths = [], []
    for episode in range(cfg.evaluation.episodes):
        env.reset(seed=cfg.seed + 100_000 + episode)
        frames: deque[np.ndarray] = deque(maxlen=cfg.frame_stack)
        obs = _stacked_pixels(
            env, cfg.image_size, frames, cfg.frame_stack, reset=True
        )
        total, length, done = 0.0, 0, False
        while not done and (
            cfg.max_episode_steps <= 0 or length < cfg.max_episode_steps
        ):
            action = agent.act(obs, step, eval_mode=True)
            _, reward, terminated, truncated, _ = env.step(action)
            obs = _stacked_pixels(env, cfg.image_size, frames, cfg.frame_stack)
            total += float(reward)
            length += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(length)
    env.close()
    return float(np.mean(returns)), float(np.mean(lengths))


def _checkpoint(
    path: Path, agent, buffer, step: int, episode: int, cfg
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'agent': agent.state_dict(),
            'buffer': buffer,
            'step': step,
            'episode': episode,
            'config': OmegaConf.to_container(cfg, resolve=True),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        path,
    )


@hydra.main(version_base=None, config_path='./config', config_name='drqv2')
def run(cfg) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)

    env = _make_env(cfg)
    env.action_space.seed(cfg.seed)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError('DrQ-v2 requires a continuous Box action space')
    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
        env.action_space.high, 1.0
    ):
        raise ValueError(
            'DrQ-v2 expects actions bounded in [-1, 1]; use a normalized '
            'continuous-control environment'
        )

    obs_shape = (3 * cfg.frame_stack, cfg.image_size, cfg.image_size)
    agent = DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=env.action_space.shape,
        device=device,
        lr=cfg.agent.lr,
        feature_dim=cfg.agent.feature_dim,
        hidden_dim=cfg.agent.hidden_dim,
        critic_target_tau=cfg.agent.critic_target_tau,
        discount=cfg.agent.discount,
        stddev=cfg.agent.stddev,
        stddev_final=cfg.agent.stddev_final,
        stddev_schedule_steps=cfg.agent.stddev_schedule_steps,
        stddev_clip=cfg.agent.stddev_clip,
        num_expl_steps=cfg.agent.num_expl_steps,
        representation_loss=cfg.representation.loss,
        representation_weight=cfg.representation.weight,
        representation_horizon=cfg.representation.horizon,
        representation_temperature=cfg.representation.temperature,
        representation_dim=cfg.representation.projection_dim,
        representation_hidden_dim=cfg.representation.hidden_dim,
        representation_lewm_hidden_dim=cfg.representation.lewm_hidden_dim,
        representation_sigreg_weight=cfg.representation.sigreg_weight,
        augmentation_pad=cfg.agent.augmentation_pad,
        nstep=cfg.nstep,
    )
    replay = ReplayBuffer(
        max_steps=cfg.replay.capacity,
        history_len=max(
            2,
            cfg.representation.horizon + 1,
            cfg.nstep + 1,
        ),
    )
    step, episode = 0, 0
    if cfg.resume:
        checkpoint = torch.load(
            cfg.resume, map_location='cpu', weights_only=False
        )
        agent.load_state_dict(checkpoint['agent'])
        replay = checkpoint['buffer']
        step, episode = int(checkpoint['step']), int(checkpoint['episode'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if torch.cuda.is_available() and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        logger.info('Resumed at step=%s episode=%s', step, episode)

    if cfg.evaluation_only:
        score, length = _evaluate(cfg, agent, step)
        logger.info('evaluation reward=%.3f length=%.1f', score, length)
        env.close()
        return

    logger.info(
        'DrQ-v2 env=%s representation=%s device=%s',
        cfg.env,
        cfg.representation.loss,
        device,
    )
    wandb_run = None
    if cfg.wandb.enable:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            id=cfg.wandb.run_id or None,
            resume='allow' if cfg.wandb.run_id else None,
        )
        logger.info('W&B run: %s', wandb_run.url)

    obs, _ = env.reset(seed=cfg.seed)
    frames: deque[np.ndarray] = deque(maxlen=cfg.frame_stack)
    obs = _stacked_pixels(
        env, cfg.image_size, frames, cfg.frame_stack, reset=True
    )
    episode_obs = [obs]
    episode_actions, episode_rewards, episode_discounts = [], [], []
    episode_reward = 0.0
    episode_length = 0
    last_metrics: dict[str, float] = {}
    start_time = time.monotonic()
    run_dir = Path(cfg.checkpoint_dir)
    try:
        while step < cfg.num_steps:
            action = (
                env.action_space.sample().astype(np.float32)
                if step < cfg.agent.num_expl_steps
                else agent.act(obs, step, eval_mode=False)
            )
            _, reward, terminated, truncated, _ = env.step(action)
            next_obs = _stacked_pixels(
                env, cfg.image_size, frames, cfg.frame_stack
            )
            episode_actions.append(np.asarray(action, dtype=np.float32))
            episode_rewards.append(np.float32(reward))
            episode_obs.append(next_obs)
            episode_reward += float(reward)
            episode_length += 1
            done = bool(
                terminated
                or truncated
                or (
                    cfg.max_episode_steps > 0
                    and episode_length >= cfg.max_episode_steps
                )
            )
            episode_discounts.append(np.float32(not done))
            step += 1
            obs = next_obs

            if done or len(episode_actions) >= cfg.replay.commit_interval:
                # Segments preserve temporal clips and make recent transitions
                # available without waiting for a potentially long episode.
                # The final frame's outgoing fields are padding only.
                replay.write_episode(
                    {
                        'pixels': np.stack(episode_obs),
                        'action': np.concatenate(
                            [
                                np.stack(episode_actions),
                                np.zeros_like(episode_actions[:1]),
                            ]
                        ),
                        'reward': np.concatenate(
                            [
                                np.asarray(episode_rewards, dtype=np.float32),
                                np.zeros(1, np.float32),
                            ]
                        ),
                        'discount': np.concatenate(
                            [
                                np.asarray(
                                    episode_discounts, dtype=np.float32
                                ),
                                np.zeros(1, np.float32),
                            ]
                        ),
                    }
                )

            if done:
                episode += 1
                if wandb_run:
                    wandb_run.log(
                        {
                            'train/episode': episode,
                            'train/episode_reward': episode_reward,
                            'train/episode_length': episode_length,
                            'train/environment_step': step,
                        },
                        step=step,
                    )
                logger.info(
                    'step=%d episode=%d reward=%.3f length=%d',
                    step,
                    episode,
                    episode_reward,
                    episode_length,
                )
                obs, _ = env.reset()
                obs = _stacked_pixels(
                    env, cfg.image_size, frames, cfg.frame_stack, reset=True
                )
                episode_obs = [obs]
                episode_actions, episode_rewards, episode_discounts = (
                    [],
                    [],
                    [],
                )
                episode_reward, episode_length = 0.0, 0
            elif len(episode_actions) >= cfg.replay.commit_interval:
                # Continue the same trajectory in a new replay segment. The
                # shared boundary observation appears once in each adjacent
                # segment so temporal samples never cross unrelated episodes.
                episode_obs = [next_obs]
                episode_actions, episode_rewards, episode_discounts = (
                    [],
                    [],
                    [],
                )

            if (
                step >= cfg.agent.num_expl_steps
                and len(replay) >= cfg.batch_size
                and step % cfg.update_frequency == 0
            ):
                for _ in range(cfg.updates_per_step):
                    sample = replay.sample(cfg.batch_size)
                    batch = {
                        key: torch.as_tensor(value, device=device)
                        for key, value in sample.items()
                    }
                    last_metrics = agent.update(batch, step)

            if step % cfg.evaluation.frequency == 0:
                score, length = _evaluate(cfg, agent, step)
                logger.info('evaluation step=%d reward=%.3f', step, score)
                metrics = {
                    'evaluation/episode_reward': score,
                    'evaluation/episode_length': length,
                    'train/environment_step': step,
                    **last_metrics,
                }
                if wandb_run:
                    wandb_run.log(metrics, step=step)

            if step % cfg.logging.frequency == 0:
                elapsed = max(time.monotonic() - start_time, 1e-6)
                train_metrics = {
                    'train/environment_step': step,
                    'train/episode': episode,
                    'system/fps': step / elapsed,
                    **last_metrics,
                }
                if wandb_run:
                    wandb_run.log(train_metrics, step=step)
                logger.info(
                    'step=%d fps=%.1f metrics=%s',
                    step,
                    train_metrics['system/fps'],
                    last_metrics,
                )

            if step % cfg.checkpoint_frequency == 0 or step == cfg.num_steps:
                _checkpoint(
                    run_dir / f'drqv2_step_{step}.pt',
                    agent,
                    replay,
                    step,
                    episode,
                    cfg,
                )
    finally:
        env.close()
        if wandb_run:
            wandb_run.finish()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
