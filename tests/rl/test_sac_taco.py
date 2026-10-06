import torch

from stable_worldmodel.rl.sac_taco import SACTACOAgent, TACOStateObjective


def test_taco_state_objective_stops_positive_target_gradient():
    torch.manual_seed(1)
    batch, horizon, state_dim, action_dim = 8, 2, 16, 3
    objective = TACOStateObjective(
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=horizon,
        feature_dim=8,
        hidden_dim=16,
    )
    features = torch.randn(
        batch, horizon + 1, state_dim, requires_grad=True
    )
    targets = torch.randn(
        batch, horizon + 1, state_dim, requires_grad=True
    )
    actions = torch.randn(
        batch, horizon, action_dim, requires_grad=True
    )
    horizon_reward = torch.randn(batch, 1)

    metrics = objective(features, actions, targets, horizon_reward)
    metrics['loss'].backward()

    assert torch.isfinite(metrics['loss'])
    assert features.grad is not None
    assert actions.grad is not None
    assert targets.grad is None
    assert objective.state_action_predictor[0].weight.grad is not None
    assert objective.action_tokenizer[0].weight.grad is not None
    assert objective.reward_predictor is not None
    assert objective.reward_predictor[0].weight.grad is not None


def test_sac_taco_update_has_explicit_gradient_ownership():
    torch.manual_seed(2)
    agent = SACTACOAgent(
        obs_dim=5,
        action_dim=2,
        device='cpu',
        hidden_dim=32,
        feature_dim=16,
        taco_horizon=2,
        taco_feature_dim=8,
        taco_hidden_dim=16,
    )
    batch_size, horizon = 8, 2
    batch = {
        'state': torch.randn(batch_size, horizon + 1, 5),
        'action': torch.rand(batch_size, horizon + 1, 2) * 2 - 1,
        'reward': torch.randn(batch_size, horizon + 1),
        'discount': torch.ones(batch_size, horizon + 1),
    }

    metrics = agent.update(batch)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert agent.encoder.net[0].weight.grad is not None
    assert agent.actor.mean.weight.grad is not None
    assert agent.log_alpha.grad is not None
    assert agent.taco is not None
    assert agent.taco.state_action_predictor[0].weight.grad is not None
    assert agent.taco.action_tokenizer[0].weight.grad is not None
    # The actor step freezes both Q and action-encoder weights; the TACO loss
    # does not include Q or policy parameters. The target Q remains detached.
    assert all(parameter.grad is None for parameter in agent.critic.parameters())
    assert all(
        parameter.grad is None
        for parameter in agent.critic_target.parameters()
    )


def test_state_sac_baseline_uses_raw_actions_and_no_taco_module():
    agent = SACTACOAgent(
        obs_dim=4,
        action_dim=1,
        device='cpu',
        hidden_dim=16,
        feature_dim=8,
        taco_enabled=False,
    )
    assert agent.taco is None
    batch = {
        'state': torch.randn(4, 2, 4),
        'action': torch.rand(4, 2, 1) * 2 - 1,
        'reward': torch.randn(4, 2),
        'discount': torch.ones(4, 2),
    }
    metrics = agent.update(batch)
    assert 'taco/loss' not in metrics
    assert torch.isfinite(torch.tensor(metrics['critic/loss']))
    assert torch.isfinite(torch.tensor(metrics['actor/loss']))


def test_horizon_reward_respects_terminal_bootstrap_mask():
    rewards = torch.tensor([[1.0, 2.0, 100.0]])
    discounts = torch.tensor([[1.0, 0.0, 0.0]])
    result = SACTACOAgent._horizon_reward(
        rewards, discounts, horizon=3, discount=0.5
    )
    assert result.shape == (1, 1)
    assert torch.allclose(result, torch.tensor([[2.0]]))
