import torch

from stable_worldmodel.rl.representation import (
    InfoNCERepresentation,
    JEPATemporalRepresentation,
    LeWMTemporalRepresentation,
    PLDMRepresentation,
    build_representation,
)


def test_infonce_shapes_finite_and_positive_pair_labels():
    batch, horizon, feature_dim, action_dim = 4, 2, 12, 3
    objective = InfoNCERepresentation(
        feature_dim, action_dim, horizon=horizon, projection_dim=8
    )
    features = torch.randn(batch, horizon + 1, feature_dim)
    target = torch.randn_like(features)
    actions = torch.randn(batch, horizon, action_dim)

    result = objective(features, actions, target)

    assert result['logits'].shape == (batch, batch)
    assert result['labels'].tolist() == list(range(batch))
    assert torch.isfinite(result['loss'])
    assert torch.isfinite(result['positive_similarity'])
    assert torch.isfinite(result['negative_similarity'])


def test_infonce_gradients_stop_at_target_features():
    objective = InfoNCERepresentation(12, 3, horizon=1, projection_dim=8)
    features = torch.randn(4, 2, 12, requires_grad=True)
    target = torch.randn(4, 2, 12, requires_grad=True)
    actions = torch.randn(4, 1, 3, requires_grad=True)

    objective(features, actions, target)['loss'].backward()

    assert features.grad is not None
    assert actions.grad is not None
    assert target.grad is None
    assert objective.predictor[0].weight.grad is not None
    assert objective.action_encoder[0].weight.grad is not None
    assert objective.projector[0].weight.grad is not None


def test_infonce_rejects_batch_without_negatives():
    objective = InfoNCERepresentation(12, 3)
    with torch.no_grad():
        try:
            objective(
                torch.randn(1, 2, 12),
                torch.randn(1, 1, 3),
                torch.randn(1, 2, 12),
            )
        except ValueError as exc:
            assert 'at least two' in str(exc)
        else:
            raise AssertionError('batch size one should be rejected')


def test_pldm_is_selectable_and_returns_scalar_loss():
    objective = build_representation('pldm', 12, 3)
    assert isinstance(objective, PLDMRepresentation)
    result = objective(torch.randn(4, 2, 12))
    assert result['loss'].ndim == 0
    assert torch.isfinite(result['loss'])


def test_jepa_adapter_reuses_causal_predictor_and_stops_target_gradient():
    objective = build_representation(
        'jepa', feature_dim=12, action_dim=3, horizon=2, projection_dim=8
    )
    assert isinstance(objective, JEPATemporalRepresentation)
    features = torch.randn(4, 3, 12, requires_grad=True)
    target = torch.randn(4, 3, 12, requires_grad=True)

    result = objective(features, target)
    result['loss'].backward()

    assert torch.isfinite(result['loss'])
    assert features.grad is not None
    assert target.grad is None


def test_lewm_adapter_is_selectable_and_returns_scalar_loss():
    objective = build_representation(
        'lewm',
        feature_dim=12,
        action_dim=3,
        horizon=1,
        projection_dim=16,
        lewm_hidden_dim=32,
    )
    assert isinstance(objective, LeWMTemporalRepresentation)
    result = objective(
        features=torch.randn(4, 2, 12),
        actions=torch.randn(4, 1, 3),
    )
    assert result['loss'].ndim == 0
    assert torch.isfinite(result['loss'])
