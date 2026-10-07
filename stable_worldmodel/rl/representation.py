"""Auxiliary temporal representation objectives for online RL."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor
from stable_worldmodel.wm.loss import PLDMLoss, SIGReg
from stable_worldmodel.wm.prejepa.module import CausalPredictor


class InfoNCERepresentation(nn.Module):
    """Action-conditioned temporal contrastive prediction.

    Predict the projected representation at ``t + horizon`` from the encoded
    observation at ``t`` and all actions in between. The matching target is on
    the diagonal of the batch similarity matrix; other batch entries are
    negatives. Target features are computed with the shared encoder under
    ``no_grad`` (stop gradient, no separate EMA encoder).
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        horizon: int = 1,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError(f'horizon must be >= 1, got {horizon}')
        if temperature <= 0:
            raise ValueError(f'temperature must be > 0, got {temperature}')
        self.horizon = int(horizon)
        self.temperature = float(temperature)
        self.latent_action_dim = int(projection_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Tanh(),
        )
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Tanh(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim + horizon * projection_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.bilinear = nn.Parameter(torch.eye(projection_dim))

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Return the action token also consumed by the TACO RL critic."""
        return self.action_encoder(action)

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        target_features: torch.Tensor,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return scalar loss and diagnostics.

        Args:
            features: Online features, ``(B, horizon + 1, feature_dim)``.
            actions: Actions leaving each frame, ``(B, horizon, action_dim)``.
            target_features: Detached target features, same time shape as
                ``features``. Only the final feature is used.
        """
        if features.ndim != 3 or target_features.ndim != 3:
            raise ValueError(
                'features and target_features must have shape (B,T,D)'
            )
        if actions.ndim != 3:
            raise ValueError('actions must have shape (B,T,A)')
        batch, time, _ = features.shape
        if batch < 2:
            raise ValueError(
                'InfoNCE needs at least two samples for negatives'
            )
        if time < self.horizon + 1 or actions.shape[1] < self.horizon:
            raise ValueError(
                f'Expected at least {self.horizon + 1} frames and '
                f'{self.horizon} actions; '
                f'got {time} frames and {actions.shape[1]} actions'
            )
        if target_features.shape != features.shape:
            raise ValueError(
                'target_features must have the same shape as features'
            )

        encoded_actions = self.encode_action(actions[:, : self.horizon])
        source = self.projector(features[:, 0])
        prediction_input = torch.cat(
            [source, encoded_actions.flatten(start_dim=1)], dim=-1
        )
        predicted = F.normalize(
            self.predictor(prediction_input).float(), dim=-1
        )
        with torch.no_grad():
            target = F.normalize(
                self.projector(
                    target_features[:, self.horizon].detach()
                ).float(),
                dim=-1,
            )
        # Bilinear scores match TACO's learned compatibility function; scaling
        # and cross entropy are evaluated in fp32 for mixed-precision safety.
        logits = (predicted @ self.bilinear.float()) @ target.T
        logits = logits / self.temperature
        labels = torch.arange(batch, device=features.device)
        loss = F.cross_entropy(logits, labels)
        positive = logits.diagonal().mean()
        if batch > 1:
            negative = (logits.sum() - logits.diagonal().sum()) / (
                batch * (batch - 1)
            )
        else:  # guarded above; keeps the expression well-defined for tracing
            negative = logits.new_zeros(())
        return {
            'loss': loss,
            'infonce_loss': loss.detach(),
            'positive_similarity': positive.detach(),
            'negative_similarity': negative.detach(),
            'temperature': logits.new_tensor(self.temperature),
            'logits': logits,
            'labels': labels,
        }


class PLDMRepresentation(nn.Module):
    """Adapter for Stable World Model's temporal alignment/VCReg loss."""

    def __init__(self, feature_dim: int, projection_dim: int = 128) -> None:
        super().__init__()
        self.loss_fn = PLDMLoss()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # PLDMLoss expects (B,T,D) and returns named terms. Normalize each term
        # into an explicit scalar objective for the RL trainer.
        terms = self.loss_fn(self.projector(features))
        loss = sum(terms.values())
        return {'loss': loss, **terms}


class JEPATemporalRepresentation(nn.Module):
    """JEPA-style latent prediction using Stable World Model's causal predictor.

    This adapter reuses the repository's predictor architecture while letting
    the DrQ encoder provide trainable image features. The future latent is a
    stop-gradient target; this does not instantiate the full pretrained
    PreJEPA image-backbone training stack.
    """

    def __init__(
        self, feature_dim: int, horizon: int, projection_dim: int = 128
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError(f'horizon must be >= 1, got {horizon}')
        self.horizon = int(horizon)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.predictor = CausalPredictor(
            num_patches=1,
            num_frames=horizon,
            dim=projection_dim,
            depth=2,
            heads=4,
            mlp_dim=projection_dim * 4,
            dim_head=max(1, projection_dim // 4),
        )

    def forward(
        self,
        features: torch.Tensor,
        target_features: torch.Tensor,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or target_features.ndim != 3:
            raise ValueError('features and targets must have shape (B,T,D)')
        if features.shape[1] < self.horizon + 1:
            raise ValueError(
                f'Expected at least {self.horizon + 1} frames, '
                f'got {features.shape[1]}'
            )
        context = self.projector(features[:, : self.horizon])
        prediction = self.predictor(context)[:, -1]
        with torch.no_grad():
            target = self.projector(target_features[:, self.horizon].detach())
        loss = F.mse_loss(prediction, target)
        return {'loss': loss, 'jepa_loss': loss.detach()}


class LeWMTemporalRepresentation(nn.Module):
    """LeWM prediction + SIGReg objective on shared policy features.

    This reuses Stable World Model's action embedder, predictor, projector, and
    SIGReg loss. This is a feature-level objective adapter, not the standalone
    LeWM model or its ViT training pipeline.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        horizon: int,
        projection_dim: int = 192,
        hidden_dim: int = 2048,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        sigreg_weight: float = 0.09,
        sigreg_num_proj: int = 1024,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError(f'horizon must be >= 1, got {horizon}')
        self.horizon = int(horizon)
        self.sigreg_weight = float(sigreg_weight)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.action_encoder = Embedder(
            input_dim=action_dim,
            smoothed_dim=projection_dim,
            emb_dim=projection_dim,
        )
        self.predictor = Predictor(
            num_frames=horizon,
            input_dim=projection_dim,
            hidden_dim=projection_dim,
            output_dim=projection_dim,
            depth=depth,
            heads=heads,
            mlp_dim=hidden_dim,
            dim_head=dim_head,
            dropout=0.1,
            emb_dropout=0.0,
        )
        self.pred_proj = MLP(
            input_dim=projection_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )
        self.sigreg = SIGReg(knots=17, num_proj=sigreg_num_proj)

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or actions.ndim != 3:
            raise ValueError(
                'features/actions must have shapes (B,T,D)/(B,T,A)'
            )
        if features.shape[1] < self.horizon + 1:
            raise ValueError(
                f'Expected at least {self.horizon + 1} frames, '
                f'got {features.shape[1]}'
            )
        if actions.shape[1] < self.horizon:
            raise ValueError(
                f'Expected at least {self.horizon} actions, got {actions.shape[1]}'
            )

        batch = features.shape[0]
        projected = self.projector(
            features[:, : self.horizon + 1].flatten(0, 1)
        ).reshape(batch, self.horizon + 1, -1)
        action_embeddings = self.action_encoder(actions[:, : self.horizon])
        predictions = self.predictor(
            projected[:, : self.horizon], action_embeddings
        )
        predictions = self.pred_proj(predictions.flatten(0, 1)).reshape_as(
            projected[:, 1:]
        )
        prediction_loss = F.mse_loss(predictions, projected[:, 1:])
        sigreg_loss = self.sigreg(projected.transpose(0, 1))
        loss = prediction_loss + self.sigreg_weight * sigreg_loss
        return {
            'loss': loss,
            'prediction_loss': prediction_loss.detach(),
            'sigreg_loss': sigreg_loss.detach(),
        }


class RewardPredictionRepresentation(nn.Module):
    """Predict the discounted reward accumulated over a temporal action clip."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        discount: float = 0.99,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError('reward prediction horizon must be >= 1')
        self.horizon = int(horizon)
        self.discount = float(discount)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim + horizon * action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if actions.shape[1] < self.horizon or rewards.shape[1] < self.horizon:
            raise ValueError(
                f'reward prediction needs {self.horizon} transitions'
            )
        action_sequence = actions[:, : self.horizon].flatten(start_dim=1)
        prediction = self.predictor(
            torch.cat([features[:, 0], action_sequence], dim=-1)
        )
        reward = rewards[:, : self.horizon]
        mask = discounts[:, : self.horizon]
        if reward.ndim == 3:
            reward = reward.squeeze(-1)
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
        target = torch.zeros_like(reward[:, :1])
        scale = torch.ones_like(target)
        for index in range(self.horizon):
            target = target + scale * reward[:, index : index + 1]
            scale = scale * self.discount * mask[:, index : index + 1]
        loss = F.mse_loss(prediction, target)
        return {'loss': loss, 'reward_prediction_loss': loss.detach()}


class CURLRepresentation(nn.Module):
    """Same-observation contrast between two independently augmented views."""

    def __init__(self, feature_dim: int, projection_dim: int = 128) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Tanh(),
        )
        self.bilinear = nn.Parameter(torch.eye(projection_dim))

    def forward(
        self,
        curl_features: tuple[torch.Tensor, torch.Tensor] | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if curl_features is None:
            raise ValueError(
                'CURL requires two independently augmented pixel views'
            )
        query, key = curl_features
        if query.ndim != 2 or key.shape != query.shape:
            raise ValueError('CURL views must have matching shape (B,D)')
        if query.shape[0] < 2:
            raise ValueError('CURL needs at least two samples for negatives')
        query = self.projector(query)
        with torch.no_grad():
            key = self.projector(key.detach())
        logits = (query @ self.bilinear) @ key.T
        labels = torch.arange(query.shape[0], device=query.device)
        loss = F.cross_entropy(logits, labels)
        return {
            'loss': loss,
            'curl_loss': loss.detach(),
            'curl_positive_similarity': logits.diagonal().mean().detach(),
        }


_OBJECTIVE_ALIASES = {
    'taco': 'infonce',
    'prejepa': 'jepa',
    'lewn': 'lewm',
}
_SUPPORTED_OBJECTIVES = frozenset(
    {'infonce', 'lewm', 'jepa', 'pldm', 'reward_prediction', 'curl'}
)
_WORLD_MODEL_OBJECTIVES = frozenset({'infonce', 'lewm', 'jepa', 'pldm'})


def _validate_objective_config(
    config: Mapping[str, object],
    *,
    observation_mode: str = 'state',
) -> tuple[str, float]:
    """Validate one objective config and return its normalized name/weight."""
    configured_name = str(
        config.get('target', config.get('name', 'none'))
    ).lower()
    if configured_name in {'none', 'null', ''}:
        return 'none', 0.0
    name = _OBJECTIVE_ALIASES.get(configured_name, configured_name)
    if name not in _SUPPORTED_OBJECTIVES:
        choices = ', '.join(['none', *sorted(_SUPPORTED_OBJECTIVES)])
        raise ValueError(
            f'Unknown world-model objective {configured_name!r}; choose {choices}'
        )
    if name == 'curl' and observation_mode != 'pixels':
        raise ValueError('CURL is available only with pixel observations')
    if 'horizon' in config and int(config['horizon']) < 1:
        raise ValueError(
            f'World-model horizon must be >= 1, got {config["horizon"]}'
        )
    weight = float(config.get('weight', 1.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(
            f'World-model weight must be finite and nonnegative, got {weight}'
        )
    return name, weight


def validate_world_model_config(
    config: Mapping[str, object],
    *,
    observation_mode: str = 'state',
) -> tuple[str, float]:
    """Validate the single primary temporal/world-model objective."""
    name, weight = _validate_objective_config(
        config, observation_mode=observation_mode
    )
    if name not in _WORLD_MODEL_OBJECTIVES and name != 'none':
        choices = ', '.join(['none', *sorted(_WORLD_MODEL_OBJECTIVES)])
        raise ValueError(
            f'{name!r} is an auxiliary add-on, not a primary world-model '
            f'loss; select one of {choices}'
        )
    if not bool(config.get('enabled', True)):
        return 'none', 0.0
    return name, weight


def build_world_model(
    config: Mapping[str, object],
    feature_dim: int,
    action_dim: int,
    *,
    discount: float = 0.99,
    observation_mode: str = 'state',
) -> nn.Module | None:
    """Build the one configured world-model objective for an RL agent.

    A Hydra ``wm`` config selects exactly one objective by name. ``none``
    leaves the model-free algorithm unchanged. The selected config's weight is
    attached to the module for the agent's single auxiliary optimizer step.
    """
    name, weight = _validate_objective_config(
        config, observation_mode=observation_mode
    )
    if name == 'none' or not bool(config.get('enabled', True)):
        return None

    horizon = int(config.get('horizon', 1))
    projection_dim = int(config.get('projection_dim', 128))
    hidden_dim = int(config.get('hidden_dim', 256))
    if name == 'infonce':
        module = InfoNCERepresentation(
            feature_dim,
            action_dim,
            horizon=horizon,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            temperature=float(config.get('temperature', 0.1)),
        )
    elif name == 'pldm':
        module = PLDMRepresentation(feature_dim, projection_dim)
    elif name == 'jepa':
        module = JEPATemporalRepresentation(
            feature_dim, horizon, projection_dim
        )
    elif name == 'lewm':
        module = LeWMTemporalRepresentation(
            feature_dim,
            action_dim,
            horizon,
            projection_dim=projection_dim,
            hidden_dim=int(config.get('lewm_hidden_dim', 2048)),
            sigreg_weight=float(config.get('sigreg_weight', 0.09)),
        )
    elif name == 'reward_prediction':
        module = RewardPredictionRepresentation(
            feature_dim,
            action_dim,
            horizon,
            hidden_dim,
            float(config.get('discount', discount)),
        )
    elif name == 'curl':
        module = CURLRepresentation(feature_dim, projection_dim)
    else:
        raise AssertionError('Supported world-model objective was not built')

    module.objective_name = name
    module.loss_weight = weight
    return module


class AuxiliaryLosses(nn.Module):
    """One primary world-model loss plus optional reward/CURL add-ons."""

    def __init__(
        self,
        world_model: nn.Module | None,
        reward_prediction: RewardPredictionRepresentation | None,
        curl: CURLRepresentation | None,
        *,
        world_model_weight: float = 0.0,
        reward_prediction_weight: float = 0.0,
        curl_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.reward_prediction = reward_prediction
        self.curl = curl
        self.world_model_weight = float(world_model_weight)
        self.reward_prediction_weight = float(reward_prediction_weight)
        self.curl_weight = float(curl_weight)
        self.horizon = max(
            1,
            int(getattr(world_model, 'horizon', 1)),
            int(getattr(reward_prediction, 'horizon', 1)),
        )

    @property
    def action_encoder(self) -> nn.Module | None:
        if self.world_model is None:
            return None
        return getattr(self.world_model, 'action_encoder', None)

    @property
    def objective_name(self) -> str:
        if self.world_model is None:
            return 'none'
        return str(self.world_model.objective_name)

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        total = inputs['features'].new_zeros(())
        metrics: dict[str, torch.Tensor] = {}
        objectives = (
            ('world_model', self.world_model, self.world_model_weight),
            (
                'reward_prediction',
                self.reward_prediction,
                self.reward_prediction_weight,
            ),
            ('curl', self.curl, self.curl_weight),
        )
        for prefix, objective, weight in objectives:
            if objective is None:
                continue
            result = objective(**inputs)
            loss = result['loss']
            weighted_loss = loss * weight
            total = total + weighted_loss
            metrics[f'{prefix}/loss'] = loss
            metrics[f'{prefix}/weighted_loss'] = weighted_loss
            for name, value in result.items():
                if (
                    name != 'loss'
                    and torch.is_tensor(value)
                    and value.numel() == 1
                ):
                    metrics[f'{prefix}/{name}'] = value
        metrics['loss'] = total
        return metrics


def build_auxiliary_losses(
    world_model_config: Mapping[str, object],
    reward_prediction_config: Mapping[str, object],
    curl_config: Mapping[str, object],
    feature_dim: int,
    action_dim: int,
    *,
    discount: float = 0.99,
    observation_mode: str = 'state',
    enabled: bool = True,
) -> AuxiliaryLosses | None:
    """Build one primary loss and the explicitly supported optional add-ons.

    Reward prediction and CURL are the only objectives that may accompany the
    selected primary world-model objective. The configs remain separate so
    each loss owns its weight and hyperparameters.
    """
    world_model_name, world_model_weight = validate_world_model_config(
        world_model_config, observation_mode=observation_mode
    )
    reward_enabled = bool(reward_prediction_config.get('enabled', False))
    curl_enabled = bool(curl_config.get('enabled', False))

    if reward_enabled:
        _, reward_weight = _validate_objective_config(
            {**reward_prediction_config, 'name': 'reward_prediction'},
            observation_mode=observation_mode,
        )
    else:
        reward_weight = 0.0
    if curl_enabled:
        _, curl_weight = _validate_objective_config(
            {**curl_config, 'name': 'curl'}, observation_mode=observation_mode
        )
    else:
        curl_weight = 0.0

    if not enabled:
        return None
    if world_model_name == 'none' and not reward_enabled and not curl_enabled:
        raise ValueError(
            'Auxiliary learning is enabled but no world-model or optional '
            'auxiliary loss was selected'
        )

    world_model = (
        build_world_model(
            world_model_config,
            feature_dim,
            action_dim,
            discount=discount,
            observation_mode=observation_mode,
        )
        if world_model_name != 'none'
        else None
    )
    reward_prediction = (
        RewardPredictionRepresentation(
            feature_dim,
            action_dim,
            horizon=int(reward_prediction_config.get('horizon', 1)),
            hidden_dim=int(reward_prediction_config.get('hidden_dim', 256)),
            discount=(
                discount
                if reward_prediction_config.get('discount') is None
                else float(reward_prediction_config['discount'])
            ),
        )
        if reward_enabled
        else None
    )
    curl = (
        CURLRepresentation(
            feature_dim,
            projection_dim=int(curl_config.get('projection_dim', 128)),
        )
        if curl_enabled
        else None
    )
    return AuxiliaryLosses(
        world_model,
        reward_prediction,
        curl,
        world_model_weight=world_model_weight,
        reward_prediction_weight=reward_weight,
        curl_weight=curl_weight,
    )


def build_representation(
    name: str,
    feature_dim: int,
    action_dim: int,
    horizon: int = 1,
    projection_dim: int = 128,
    hidden_dim: int = 256,
    temperature: float = 0.1,
    sigreg_weight: float = 0.09,
    lewm_hidden_dim: int = 2048,
    observation_mode: str = 'state',
) -> nn.Module | None:
    """Compatibility factory for callers that pass an objective name."""
    return build_world_model(
        {
            'name': name,
            'horizon': horizon,
            'projection_dim': projection_dim,
            'hidden_dim': hidden_dim,
            'temperature': temperature,
            'sigreg_weight': sigreg_weight,
            'lewm_hidden_dim': lewm_hidden_dim,
        },
        feature_dim,
        action_dim,
        observation_mode=observation_mode,
    )
