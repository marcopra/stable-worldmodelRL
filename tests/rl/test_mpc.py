"""Regression checks for model reuse, TD semantics, and real-step accounting."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from scripts.train.rl_online import _make_agent, _replay_history, train
from stable_worldmodel.rl import DrQV2Agent, SACAgent
from stable_worldmodel.rl.mpc import LatentMPCPlanner, ValuePredictionHead
from stable_worldmodel.rl.representation import build_world_model


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def make_agent(algorithm, mode='state', wm='infonce', value_coef=0.1, **extra):
    kwargs = {
        'obs_shape': (5,) if mode == 'state' else (3, 36, 36),
        'observation_mode': mode,
        'feature_dim': 8,
        'hidden_dim': 16,
        'sequence_length': 2,
        'auxiliary_enabled': True,
        'world_model_config': {
            'target': wm,
            'horizon': 1,
            'projection_dim': 8,
            'hidden_dim': 16,
            'lewm_hidden_dim': 16,
            'sigreg_num_proj': 8,
        },
        'reward_prediction_config': {
            'enabled': True,
            'mode': 'per_step',
            'horizon': 1,
            'hidden_dim': 16,
        },
        'mpc_config': {
            'enabled': True,
            'start_steps': 0,
            'value_coef': value_coef,
            'horizon': 3,
            'num_samples': 8,
            'num_elites': 2,
            'iterations': 2,
            'num_pi_trajs': 2,
        },
        **extra,
    }
    if algorithm == 'sac':
        return SACAgent(action_dim=2, **kwargs)
    return DrQV2Agent(action_shape=(2,), num_expl_steps=0, **kwargs)


def batch(mode='state', time=3):
    return {
        'observation': (
            torch.randn(4, time, 5)
            if mode == 'state'
            else torch.randint(256, (4, time, 36, 36, 3), dtype=torch.uint8)
        ),
        'action': torch.rand(4, time, 2) * 2 - 1,
        'reward': torch.randn(4, time),
        'discount': torch.ones(4, time),
    }


@pytest.mark.parametrize('algorithm', ['sac', 'drqv2'])
@pytest.mark.parametrize('mode', ['state', 'pixels'])
@pytest.mark.parametrize('wm', ['infonce', 'lewm', 'consistency'])
@pytest.mark.parametrize('value_coef', [0.0, 0.1])
def test_mpc_reuses_trained_models_and_restores(
    algorithm, mode, wm, value_coef
):
    agent = make_agent(algorithm, mode, wm, value_coef)
    assert agent.world_model is agent.auxiliary_losses.world_model
    assert agent.reward_prediction is agent.auxiliary_losses.reward_prediction
    sample = batch(mode)
    observation = sample['observation'][0, 0].numpy()
    with pytest.raises(RuntimeError, match='no auxiliary updates'):
        agent.act(observation, step=10)
    metrics = agent.update(sample, step=10)
    assert all(np.isfinite(value) for value in metrics.values())
    assert agent.mpc_model_updates == 1
    assert all(p.grad is None for p in agent.critic_target.parameters())
    assert (agent.mpc_value is None) == (value_coef == 0)
    if agent.mpc_value is not None:
        assert any(p.grad is not None for p in agent.mpc_value.parameters())
        assert all(p.grad is None for p in agent.mpc_value_target.parameters())
    action = agent.act(observation, step=10, eval_mode=True)
    assert action.shape == (2,) and np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1
    agent.reset()
    assert agent.mpc_planner._previous_mean is None
    restored = make_agent(algorithm, mode, wm, value_coef)
    restored.load_state_dict(copy.deepcopy(agent.state_dict()))
    assert restored.mpc_model_updates == 1
    torch.manual_seed(9)
    expected = agent.act(observation, step=10, eval_mode=True)
    torch.manual_seed(9)
    np.testing.assert_allclose(
        restored.act(observation, step=10, eval_mode=True), expected
    )


@pytest.mark.parametrize('algorithm', ['sac', 'drqv2'])
@pytest.mark.parametrize(
    'extra',
    [
        {'world_model_config': {'target': 'none'}},
        {'world_model_config': {'target': 'jepa'}},
        {'world_model_config': {'target': 'pldm'}},
        {'world_model_config': {'target': 'infonce', 'horizon': 2}},
        {'world_model_config': {'target': 'infonce', 'weight': 0}},
        {'auxiliary_enabled': False},
        {'reward_prediction_config': {'enabled': False}},
        {'reward_prediction_config': {'enabled': True, 'mode': 'clip_return'}},
        {
            'reward_prediction_config': {
                'enabled': True,
                'mode': 'per_step',
                'weight': 0,
            }
        },
    ],
)
def test_invalid_mpc_configs_fail_at_setup(algorithm, extra):
    with pytest.raises(ValueError):
        make_agent(algorithm, **extra)


@pytest.mark.parametrize('algorithm', ['sac', 'drqv2'])
def test_short_batches_and_legacy_mpc_checkpoints_fail_loudly(algorithm):
    agent = make_agent(algorithm)
    with pytest.raises(ValueError, match='3 observations'):
        agent.update(batch(time=2))
    state = copy.deepcopy(agent.state_dict())
    del state['mpc_model_updates']
    with pytest.raises(ValueError, match='model-update count'):
        agent.load_state_dict(state)
    state = copy.deepcopy(agent.state_dict())
    state['mpc_value_objective'] = 'soft_sac'
    with pytest.raises(ValueError, match='reward_td_v1'):
        agent.load_state_dict(state)


def test_sac_mpc_target_has_no_entropy_and_stops_target_gradient(monkeypatch):
    agent = make_agent('sac')
    features = torch.randn(4, 3, 8, requires_grad=True)
    actions = torch.zeros(4, 2, 2)
    rewards = torch.tensor([[1.0, 2.0]]).expand(4, -1)
    discounts = torch.tensor([[1.0, 0.0]]).expand(4, -1)
    monkeypatch.setattr(
        agent.actor,
        'sample',
        lambda feature: (
            feature.new_zeros(4, 2),
            feature.new_full((4, 1), -99),
        ),
    )
    monkeypatch.setattr(
        agent.mpc_value_target,
        'forward',
        lambda latent, action: (
            latent.new_full((4, 1), 3),
            latent.new_full((4, 1), 5),
        ),
    )
    targets = []
    original_loss = agent.mpc_value.loss

    def capture(latent, action, target):
        assert not target.requires_grad
        targets.append(target.clone())
        return original_loss(latent, action, target)

    monkeypatch.setattr(agent.mpc_value, 'loss', capture)
    agent._mpc_value_loss(features, actions, rewards, discounts).backward()
    torch.testing.assert_close(targets[0], torch.full((4, 1), 1 + 0.99 * 3))
    torch.testing.assert_close(targets[1], torch.full((4, 1), 2.0))
    assert features.grad[:, :-1].abs().sum() > 0
    assert features.grad[:, -1].abs().sum() == 0
    assert agent.log_alpha.grad is None
    assert all(p.grad is None for p in agent.actor.parameters())


def test_drq_action_clamp_preserves_actor_gradient():
    mean = torch.tensor([[2.0, -2.0]], requires_grad=True)
    action = DrQV2Agent._sample_action(mean, torch.zeros_like(mean))
    assert torch.all(action.abs() < 1)
    action.sum().backward()
    torch.testing.assert_close(mean.grad, torch.ones_like(mean))


def test_two_hot_endpoints_and_finite_loss():
    head = ValuePredictionHead(3, 1, 8, num_bins=5, vmin=-2, vmax=2)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    targets = torch.tensor([[-1e6], [0.0], [1e6]])
    latent = torch.zeros(3, 3)
    action = torch.zeros(3, 1)
    loss = head.loss(latent, action, targets)
    torch.testing.assert_close(loss, torch.tensor(np.log(5), dtype=loss.dtype))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
    torch.testing.assert_close(
        head.q1[-1].bias.grad,
        0.5 * (torch.full((5,), 0.2) - torch.tensor([1, 0, 1, 0, 1]) / 3),
    )


@pytest.mark.parametrize(
    'kwargs',
    [
        {'temperature': float('nan')},
        {'max_std': float('inf')},
        {'horizon': 1.5},
    ],
)
def test_planner_rejects_invalid_numeric_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LatentMPCPlanner(action_dim=1, **kwargs)


def test_planner_scores_recurrent_rewards_and_executes_first_elite(
    monkeypatch,
):
    class Dynamics(torch.nn.Module):
        def encode_mpc_latent(self, feature):
            return feature

        def predict_mpc_step(self, latent, action):
            return latent + action

    class Reward(torch.nn.Module):
        def predict_step(self, latent, action):
            return latent + action

    # With gamma=.5, sequences [-1,1] and [1,-1] score -1 and +1.
    monkeypatch.setattr(
        torch,
        'randn',
        lambda *args, **kwargs: torch.tensor(
            [[[-1.0], [1.0]], [[1.0], [-1.0]]]
        ),
    )
    planner = LatentMPCPlanner(
        action_dim=1,
        horizon=2,
        num_samples=2,
        num_elites=1,
        iterations=1,
        num_pi_trajs=0,
        max_std=1,
    )
    model, reward = Dynamics(), Reward()
    action = planner.plan(
        torch.zeros(1, 1),
        model,
        reward,
        None,
        None,
        discount=0.5,
        eval_mode=True,
    )
    torch.testing.assert_close(action, torch.ones(1))
    assert model.training and reward.training
    assert not action.requires_grad
    monkeypatch.setattr(
        reward, 'predict_step', lambda latent, action: latent * float('nan')
    )
    with pytest.raises(FloatingPointError, match='non-finite'):
        planner.plan(
            torch.zeros(1, 1),
            model,
            reward,
            None,
            None,
            discount=0.5,
            eval_mode=True,
        )


def test_hydra_drq_sequence_length_is_forwarded():
    config_dir = str(Path('scripts/train/config').resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name='drqv2_state',
            overrides=[
                'device=cpu',
                'auxiliary.sequence_length=2',
                'agent.hidden_dim=16',
                'agent.feature_dim=8',
            ],
        )
    assert _make_agent(cfg, (5,), 2).sequence_length == 2
    assert _replay_history(cfg) == 2  # auxiliary path disabled


@pytest.mark.parametrize('algorithm', ['sac', 'drqv2'])
def test_runner_counts_only_real_steps_with_mpc(tmp_path, algorithm):
    config_dir = str(Path('scripts/train/config').resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name='rl_online',
            overrides=[
                f'agent={algorithm}',
                'env=MountainCarContinuous-v0',
                'device=cpu',
                'num_steps=12',
                'learning_starts=2',
                'batch_size=2',
                'replay.commit_interval=3',
                'replay.capacity=100',
                'max_episode_steps=6',
                'evaluation.frequency=12',
                'evaluation.episodes=1',
                'checkpoint_frequency=12',
                f'checkpoint_dir={tmp_path}',
                'agent.hidden_dim=16',
                'agent.feature_dim=8',
                'wm=infonce',
                'auxiliary.enabled=true',
                'auxiliary.sequence_length=2',
                'auxiliary.wm.projection_dim=8',
                'auxiliary.wm.hidden_dim=16',
                'auxiliary.reward_prediction.enabled=true',
                'auxiliary.reward_prediction.mode=per_step',
                'mpc.enabled=true',
                'mpc.start_steps=6',
                'mpc.horizon=3',
                'mpc.num_samples=8',
                'mpc.num_elites=2',
                'mpc.iterations=2',
                'mpc.num_pi_trajs=2',
                *(['agent.num_expl_steps=2'] if algorithm == 'drqv2' else []),
            ],
        )
    train(cfg)
    saved = torch.load(
        tmp_path / f'{algorithm}_step_12.pt', weights_only=False
    )
    assert saved['step'] == 12 and saved['episode'] == 2
    # Four real three-transition segments, each with one final-frame sentinel.
    assert saved['replay'].num_valid_ends(2) == 12
    assert saved['agent']['mpc_model_updates'] > 0


def test_consistency_horizon_is_not_silently_ignored():
    with pytest.raises(ValueError, match='horizon must be 1'):
        build_world_model({'target': 'consistency', 'horizon': 3}, 8, 2)


def test_public_reward_factory_preserves_mode_and_default_discount():
    model = build_world_model(
        {
            'target': 'reward_prediction',
            'mode': 'per_step',
            'horizon': 1,
            'discount': None,
        },
        8,
        2,
        discount=0.95,
    )
    assert model.mode == 'per_step' and model.discount == 0.95
    result = model.predict_step(torch.zeros(4, 8), torch.zeros(4, 2))
    assert result.shape == (4, 1)


def test_infonce_rollout_horizon_is_independent_of_training_horizon():
    model = build_world_model({'target': 'infonce', 'horizon': 1}, 8, 2)
    features = torch.randn(4, 8)
    actions = torch.randn(4, 3, 2)
    recurrent = model.encode_mpc_latent(features)
    for action in actions.unbind(1):
        recurrent = model.predict_mpc_step(recurrent, action)
    torch.testing.assert_close(
        model.predict_mpc_latent(features, actions), recurrent
    )


def test_terminal_value_can_change_the_selected_first_action(monkeypatch):
    class Dynamics(torch.nn.Module):
        def encode_mpc_latent(self, feature):
            return feature

        def predict_mpc_step(self, latent, action):
            return latent + action

    class Reward(torch.nn.Module):
        def predict_step(self, latent, action):
            return -action

    class Value(torch.nn.Module):
        def forward(self, latent, action):
            q = 10 * latent
            return q, q

    def samples(count, horizon, dim, **kwargs):
        return (
            torch.tensor([-1.0, 1.0])
            .reshape(2, 1, 1)
            .expand(count, horizon, dim)
        )

    monkeypatch.setattr(torch, 'randn', samples)
    planner = LatentMPCPlanner(
        action_dim=1,
        horizon=1,
        num_samples=2,
        num_elites=1,
        iterations=1,
        num_pi_trajs=0,
        max_std=1,
    )
    args = (torch.zeros(1, 1), Dynamics(), Reward(), None)
    unvalued = planner.plan(*args, None, discount=0.5, eval_mode=True)
    planner.reset()
    valued = planner.plan(*args, Value(), discount=0.5, eval_mode=True)
    torch.testing.assert_close(unvalued, -torch.ones(1))
    torch.testing.assert_close(valued, torch.ones(1))
