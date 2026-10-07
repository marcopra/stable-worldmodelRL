import pytest
import torch

from stable_worldmodel.rl import DrQV2Agent


@pytest.mark.parametrize('representation', ['none', 'infonce', 'jepa', 'pldm'])
def test_agent_initializes_and_updates_on_cpu(representation):
    torch.set_num_threads(1)
    horizon = 2 if representation in {'infonce', 'jepa'} else 1
    agent = DrQV2Agent(
        obs_shape=(3, 36, 36),
        action_shape=(2,),
        device='cpu',
        feature_dim=8,
        hidden_dim=16,
        representation_loss=representation,
        representation_horizon=horizon,
        representation_dim=8,
        representation_hidden_dim=16,
        augmentation_pad=2,
    )
    batch = {
        'pixels': torch.randint(
            0, 256, (4, horizon + 1, 36, 36, 3), dtype=torch.uint8
        ),
        'action': torch.rand(4, horizon + 1, 2) * 2 - 1,
        'reward': torch.randn(4, horizon + 1),
        'discount': torch.ones(4, horizon + 1),
    }

    metrics = agent.update(batch, step=1)

    assert torch.isfinite(torch.tensor(metrics['critic/loss']))
    assert torch.isfinite(torch.tensor(metrics['actor/loss']))
    if representation == 'none':
        assert 'world_model/loss' not in metrics
    else:
        assert torch.isfinite(torch.tensor(metrics['world_model/loss']))
        assert any(
            parameter.grad is not None
            for parameter in agent.encoder.parameters()
        )
