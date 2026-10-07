from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces
from omegaconf import OmegaConf

from scripts.train.rl_online import (
    _bootstrap_discount,
    _checkpoint,
    _make_agent,
    _observation,
    _space_for_state,
    _state_observation,
)
from stable_worldmodel.data import ReplayBuffer
from stable_worldmodel.rl import DrQV2Agent, SACAgent
from stable_worldmodel.rl.representation import build_world_model


def _batch(mode: str, batch_size: int = 4, time: int = 3):
    if mode == 'pixels':
        observation = torch.randint(
            0, 256, (batch_size, time, 36, 36, 3), dtype=torch.uint8
        )
    else:
        observation = torch.randn(batch_size, time, 5)
    return {
        'observation': observation,
        'action': torch.rand(batch_size, time, 2) * 2 - 1,
        'reward': torch.randn(batch_size, time),
        'discount': torch.ones(batch_size, time),
    }


@pytest.mark.parametrize('algorithm', ['sac', 'drqv2'])
@pytest.mark.parametrize('mode', ['state', 'pixels'])
def test_each_agent_updates_for_state_and_pixel_observations(algorithm, mode):
    torch.set_num_threads(1)
    observation_shape = (3, 36, 36) if mode == 'pixels' else (5,)
    if algorithm == 'sac':
        agent = SACAgent(
            obs_shape=observation_shape,
            action_dim=2,
            observation_mode=mode,
            hidden_dim=32,
            feature_dim=16,
            device='cpu',
        )
    else:
        agent = DrQV2Agent(
            obs_shape=observation_shape,
            action_shape=(2,),
            observation_mode=mode,
            hidden_dim=32,
            feature_dim=16,
            device='cpu',
            augmentation_pad=2,
        )
    metrics = agent.update(_batch(mode), step=1)
    assert torch.isfinite(torch.tensor(metrics['critic/loss']))
    assert torch.isfinite(torch.tensor(metrics['actor/loss']))
    assert 'world_model/loss' not in metrics
    restored = (
        SACAgent(
            obs_shape=observation_shape,
            action_dim=2,
            observation_mode=mode,
            hidden_dim=32,
            feature_dim=16,
            device='cpu',
        )
        if algorithm == 'sac'
        else DrQV2Agent(
            obs_shape=observation_shape,
            action_shape=(2,),
            observation_mode=mode,
            hidden_dim=32,
            feature_dim=16,
            device='cpu',
            augmentation_pad=2,
        )
    )
    restored.load_state_dict(agent.state_dict())


@pytest.mark.parametrize('algorithm', ['sac', 'drqv2'])
@pytest.mark.parametrize('mode', ['state', 'pixels'])
def test_single_infonce_objective_updates_state_and_pixel_modes(
    algorithm, mode
):
    torch.set_num_threads(1)
    observation_shape = (3, 36, 36) if mode == 'pixels' else (5,)
    common = {
        'obs_shape': observation_shape,
        'observation_mode': mode,
        'device': 'cpu',
        'feature_dim': 16,
        'hidden_dim': 32,
        'auxiliary_enabled': True,
        'world_model_config': {
            'name': 'infonce',
            'horizon': 2,
            'projection_dim': 8,
            'hidden_dim': 16,
            'weight': 0.5,
        },
    }
    if algorithm == 'sac':
        agent = SACAgent(action_dim=2, **common)
    else:
        agent = DrQV2Agent(action_shape=(2,), **common)
    metrics = agent.update(_batch(mode), step=1)
    assert torch.isfinite(torch.tensor(metrics['world_model/loss']))
    assert torch.isfinite(torch.tensor(metrics['world_model/weighted_loss']))
    assert torch.isfinite(torch.tensor(metrics['world_model/infonce_loss']))
    assert metrics['world_model/weighted_loss'] == pytest.approx(
        0.5 * metrics['world_model/loss']
    )
    restored = (
        SACAgent(action_dim=2, **common)
        if algorithm == 'sac'
        else DrQV2Agent(action_shape=(2,), **common)
    )
    restored.load_state_dict(agent.state_dict())


@pytest.mark.parametrize(
    ('mode', 'objective'),
    [('state', 'reward_prediction'), ('pixels', 'curl')],
)
def test_standalone_objectives_update_in_supported_modalities(mode, objective):
    torch.set_num_threads(1)
    configs = {
        'reward_prediction': {
            'enabled': True,
            'horizon': 1,
            'weight': 1.0,
            'hidden_dim': 16,
        },
        'curl': {
            'enabled': True,
            'weight': 1.0,
            'projection_dim': 8,
        },
    }
    configs['reward_prediction']['enabled'] = objective == 'reward_prediction'
    configs['curl']['enabled'] = objective == 'curl'
    agent = DrQV2Agent(
        obs_shape=(3, 36, 36) if mode == 'pixels' else (5,),
        action_shape=(2,),
        observation_mode=mode,
        device='cpu',
        feature_dim=8,
        hidden_dim=16,
        auxiliary_enabled=True,
        world_model_config={'name': 'none'},
        reward_prediction_config=configs['reward_prediction'],
        curl_config=configs['curl'],
        augmentation_pad=2,
    )

    metrics = agent.update(_batch(mode, time=2), step=1)

    metric = f'{objective}/loss'
    assert torch.isfinite(torch.tensor(metrics[metric]))


def test_selected_world_model_can_include_reward_prediction_and_curl():
    torch.set_num_threads(1)
    agent = DrQV2Agent(
        obs_shape=(3, 36, 36),
        action_shape=(2,),
        observation_mode='pixels',
        device='cpu',
        feature_dim=8,
        hidden_dim=16,
        auxiliary_enabled=True,
        world_model_config={
            'name': 'infonce',
            'horizon': 2,
            'projection_dim': 8,
            'hidden_dim': 16,
            'weight': 0.25,
        },
        reward_prediction_config={
            'enabled': True,
            'horizon': 2,
            'hidden_dim': 16,
            'weight': 0.5,
        },
        curl_config={'enabled': True, 'projection_dim': 8, 'weight': 0.75},
        augmentation_pad=2,
    )

    metrics = agent.update(_batch('pixels'), step=1)
    expected_total = sum(
        metrics[name]
        for name in (
            'world_model/weighted_loss',
            'reward_prediction/weighted_loss',
            'curl/weighted_loss',
        )
    )

    assert metrics['world_model/loss'] > 0
    assert metrics['reward_prediction/loss'] > 0
    assert metrics['curl/loss'] > 0
    assert metrics['auxiliary/loss'] == pytest.approx(expected_total)
    restored = DrQV2Agent(
        obs_shape=(3, 36, 36),
        action_shape=(2,),
        observation_mode='pixels',
        device='cpu',
        feature_dim=8,
        hidden_dim=16,
        auxiliary_enabled=True,
        world_model_config={
            'name': 'infonce',
            'horizon': 2,
            'projection_dim': 8,
            'hidden_dim': 16,
            'weight': 0.25,
        },
        reward_prediction_config={
            'enabled': True,
            'horizon': 2,
            'hidden_dim': 16,
            'weight': 0.5,
        },
        curl_config={'enabled': True, 'projection_dim': 8, 'weight': 0.75},
        augmentation_pad=2,
    )
    restored.load_state_dict(agent.state_dict())


def test_curl_is_rejected_for_state_observations():
    with pytest.raises(ValueError, match='only with pixel observations'):
        build_world_model(
            {'name': 'curl'},
            feature_dim=8,
            action_dim=2,
            observation_mode='state',
        )


@pytest.mark.parametrize('algorithm', ['sac', 'drqv2'])
def test_auxiliary_enabled_requires_a_selected_objective(algorithm):
    common = {
        'obs_shape': (5,),
        'observation_mode': 'state',
        'device': 'cpu',
        'feature_dim': 8,
        'hidden_dim': 16,
        'auxiliary_enabled': True,
        'world_model_config': {'name': 'none'},
    }
    with pytest.raises(ValueError, match='no world-model or optional'):
        if algorithm == 'sac':
            SACAgent(action_dim=2, **common)
        else:
            DrQV2Agent(action_shape=(2,), **common)


def test_state_curl_configuration_is_rejected_during_agent_setup():
    cfg = OmegaConf.create(
        {
            'algorithm': {'name': 'sac'},
            'observation': {'mode': 'state'},
            'device': 'cpu',
            'agent': {
                'hidden_dim': 16,
                'feature_dim': 8,
                'discount': 0.99,
                'lr': 0.0003,
                'alpha_lr': 0.0003,
                'tau': 0.005,
                'initial_alpha': 0.2,
                'target_entropy': None,
                'augmentation_pad': 0,
            },
            'auxiliary': {
                'enabled': True,
                'wm': {
                    'target': 'infonce',
                    'enabled': True,
                    'horizon': 1,
                    'projection_dim': 8,
                },
                'reward_prediction': {'enabled': False},
                'curl': {'enabled': True, 'projection_dim': 8},
            },
        }
    )
    with pytest.raises(ValueError, match='only with pixel observations'):
        _make_agent(cfg, observation_shape=(5,), action_dim=2)
    cfg.auxiliary.enabled = False
    with pytest.raises(ValueError, match='only with pixel observations'):
        _make_agent(cfg, observation_shape=(5,), action_dim=2)


def test_dict_proprio_observation_uses_selected_numeric_key():
    observation = {
        'proprio': np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'pixels': np.zeros((8, 8, 3), dtype=np.uint8),
    }
    result = _state_observation(observation, 'proprio')
    assert result.shape == (4,)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match='state_key'):
        _state_observation(observation, None)

    fake_env = type(
        'FakeEnv',
        (),
        {
            'observation_space': spaces.Dict(
                {
                    'proprio': spaces.Box(-1, 1, shape=(2, 2)),
                    'pixels': spaces.Box(
                        0, 255, shape=(8, 8, 3), dtype=np.uint8
                    ),
                }
            )
        },
    )()
    assert _space_for_state(fake_env, 'proprio').shape == (2, 2)
    with pytest.raises(KeyError, match='absent'):
        _space_for_state(fake_env, 'missing')
    box_env = type(
        'FakeBoxEnv',
        (),
        {'observation_space': spaces.Box(-1, 1, shape=(5,))},
    )()
    assert _space_for_state(box_env, None).shape == (5,)


def test_world_model_factory_selects_one_weighted_objective():
    objective = build_world_model(
        {
            'target': 'infonce',
            'enabled': True,
            'weight': 0.25,
            'horizon': 2,
            'projection_dim': 8,
            'hidden_dim': 16,
        },
        feature_dim=8,
        action_dim=2,
    )
    assert objective is not None
    assert objective.objective_name == 'infonce'
    assert objective.loss_weight == 0.25
    features = torch.randn(4, 3, 8)
    targets = torch.randn_like(features)
    actions = torch.randn(4, 2, 2)
    rewards = torch.randn(4, 2)
    discounts = torch.ones_like(rewards)
    result = objective(
        features=features,
        actions=actions,
        target_features=targets,
        rewards=rewards,
        discounts=discounts,
    )
    assert torch.isfinite(result['loss'])


def test_agent_api_does_not_accept_composable_objective_maps():
    with pytest.raises(TypeError, match='auxiliary_objectives'):
        SACAgent(
            obs_shape=(5,),
            action_dim=2,
            auxiliary_enabled=True,
            auxiliary_objectives={
                'infonce': {'horizon': 1},
                'reward_prediction': {'horizon': 1},
            },
        )


def test_time_limit_truncations_keep_bootstrap_discount():
    assert _bootstrap_discount(terminated=False) == 1.0
    assert _bootstrap_discount(terminated=True) == 0.0


def test_pixel_observations_resize_and_stack_rendered_frames():
    class FakeEnv:
        value = 0

        @property
        def unwrapped(self):
            return self

        def render(self, width, height):
            frame = np.full((8, 10, 3), self.value, dtype=np.uint8)
            self.value += 1
            return frame

    cfg = SimpleNamespace(
        observation=SimpleNamespace(
            mode='pixels', image_size=36, frame_stack=2
        )
    )
    env = FakeEnv()
    frames = deque(maxlen=2)
    first = _observation(env, None, cfg, frames, reset=True)
    assert first.shape == (36, 36, 6)
    assert np.all(first[..., :3] == 0)
    assert np.all(first[..., 3:] == 0)

    second = _observation(env, None, cfg, frames)
    assert np.all(second[..., :3] == 0)
    assert np.all(second[..., 3:] == 1)


def test_replay_temporal_windows_stay_inside_each_episode():
    def edge_sampler(step, buffer, batch_size, history_len):
        assert history_len == 3
        return np.asarray([0, buffer.num_valid_ends(history_len) - 1])

    replay = ReplayBuffer(max_steps=16, history_len=3, sampler=edge_sampler)
    for offset in (0, 100):
        replay.write_episode(
            {
                'observation': np.arange(offset, offset + 4)[:, None],
                'action': np.zeros((4, 1), dtype=np.float32),
                'reward': np.zeros(4, dtype=np.float32),
                'discount': np.ones(4, dtype=np.float32),
            }
        )

    batch = replay.sample(batch_size=2, history_len=3)
    observations = batch['observation'][..., 0]
    np.testing.assert_array_equal(
        np.diff(observations, axis=1), [[1, 1], [1, 1]]
    )
    assert observations[0, -1] < 10
    assert observations[1, 0] >= 100


def test_checkpoint_records_algorithm_and_resolved_run_config(tmp_path):
    agent = SACAgent(
        obs_shape=(5,), action_dim=2, hidden_dim=16, feature_dim=8
    )
    replay = ReplayBuffer(max_steps=16, history_len=2)
    cfg = OmegaConf.create(
        {
            'algorithm': {'name': 'sac'},
            'observation': {'mode': 'state'},
            'auxiliary': {
                'enabled': False,
                'wm': {'target': 'none', 'enabled': False},
                'reward_prediction': {'enabled': False},
                'curl': {'enabled': False},
            },
        }
    )
    path = tmp_path / 'checkpoint.pt'

    _checkpoint(path, agent, replay, step=7, episode=2, cfg=cfg)
    saved = torch.load(path, map_location='cpu', weights_only=False)

    assert saved['algorithm'] == 'sac'
    assert saved['world_model'] == 'none'
    assert saved['auxiliary_losses'] == []
    assert saved['config']['algorithm']['name'] == 'sac'
    assert saved['step'] == 7
