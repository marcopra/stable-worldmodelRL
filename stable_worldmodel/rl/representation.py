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
        if not math.isfinite(temperature) or temperature <= 0:
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

    @property
    def mpc_latent_dim(self) -> int:
        return self.latent_action_dim

    def encode_mpc_latent(self, features: torch.Tensor) -> torch.Tensor:
        """Project encoder features into the WM's normalized prediction space."""
        return F.normalize(self.projector(features).float(), dim=-1)

    def predict_mpc_latent(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Roll out the one-step predictor over an action sequence."""
        if actions.ndim != 3 or actions.shape[1] < 1:
            raise ValueError(
                'InfoNCE MPC needs a nonempty (B,K,A) action sequence; '
                f'got shape {tuple(actions.shape)}'
            )
        if self.horizon != 1:
            raise ValueError(
                'InfoNCE MPC dynamics must be trained with horizon=1'
            )
        latent = self.encode_mpc_latent(features)
        for index in range(actions.shape[1]):
            latent = self.predict_mpc_step(latent, actions[:, index])
        return latent

    def predict_mpc_step(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Apply the composable one-step InfoNCE dynamics model."""
        if self.horizon != 1:
            raise ValueError(
                'InfoNCE MPC dynamics must be trained with horizon=1'
            )
        if latent.ndim != 2 or action.ndim != 2:
            raise ValueError(
                'MPC latent/action inputs must have shape (B,D)/(B,A)'
            )
        action_embedding = self.encode_action(action)
        prediction = self.predictor(
            torch.cat((latent, action_embedding), dim=-1)
        )
        return F.normalize(prediction.float(), dim=-1)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Embed replay actions for the temporal prediction objective."""
        return self.action_encoder(action)

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        target_features: torch.Tensor,
        sequence_length: int = 1,
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

        available = min(
            actions.shape[1] - self.horizon + 1,
            features.shape[1] - self.horizon,
        )
        windows = min(int(sequence_length), available)
        if windows < 1:
            raise ValueError(
                f'Expected at least {self.horizon + 1} frames and '
                f'{self.horizon} actions'
            )
        if windows > 1:
            results = [
                self.forward(
                    features[:, offset : offset + self.horizon + 1],
                    actions[:, offset : offset + self.horizon],
                    target_features[:, offset : offset + self.horizon + 1],
                    sequence_length=1,
                )
                for offset in range(windows)
            ]
            averaged = dict(results[0])
            for name in (
                'loss',
                'infonce_loss',
                'positive_similarity',
                'negative_similarity',
            ):
                averaged[name] = torch.stack(
                    [result[name] for result in results]
                ).mean()
            return averaged

        encoded_actions = self.encode_action(actions[:, : self.horizon])
        source = self.encode_mpc_latent(features[:, 0])
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
        negative = (logits.sum() - logits.diagonal().sum()) / (
            batch * (batch - 1)
        )
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
        if (
            features.ndim != 3
            or features.shape[0] < 2
            or features.shape[1] < 2
        ):
            raise ValueError('PLDM requires B, T, and projected D >= 2')
        if self.projector[0].out_features < 2:
            raise ValueError('PLDM requires projection_dim >= 2')
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
        sequence_length: int = 1,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or target_features.ndim != 3:
            raise ValueError('features and targets must have shape (B,T,D)')
        if features.shape[1] < self.horizon + 1:
            raise ValueError(
                f'Expected at least {self.horizon + 1} frames, '
                f'got {features.shape[1]}'
            )
        available = features.shape[1] - self.horizon
        windows = min(int(sequence_length), available)
        if windows < 1:
            raise ValueError(
                'JEPA needs at least one complete temporal window'
            )
        losses = []
        for offset in range(windows):
            context = self.projector(
                features[:, offset : offset + self.horizon]
            )
            prediction = self.predictor(context)[:, -1]
            with torch.no_grad():
                target = self.projector(
                    target_features[:, offset + self.horizon].detach()
                )
            losses.append(F.mse_loss(prediction, target))
        loss = torch.stack(losses).mean()
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
        self.projection_dim = int(projection_dim)
        self.sigreg_weight = float(sigreg_weight)
        if not math.isfinite(self.sigreg_weight) or self.sigreg_weight < 0:
            raise ValueError('sigreg_weight must be finite and non-negative')
        if sigreg_num_proj < 1:
            raise ValueError('sigreg_num_proj must be positive')
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

    @property
    def mpc_latent_dim(self) -> int:
        return self.projection_dim

    def encode_mpc_latent(self, features: torch.Tensor) -> torch.Tensor:
        """Project encoder features into LeWM's prediction space."""
        return self.projector(features)

    def predict_mpc_latent(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Roll out LeWM's composable one-step predictor."""
        if actions.ndim != 3:
            raise ValueError('planned actions must have shape (B,H,A)')
        if self.horizon != 1:
            raise ValueError(
                'LeWM MPC dynamics must be trained with horizon=1'
            )
        latent = self.encode_mpc_latent(features)
        for index in range(actions.shape[1]):
            latent = self.predict_mpc_step(latent, actions[:, index])
        return latent

    def predict_mpc_step(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict one latent transition in LeWM's projected space."""
        if self.horizon != 1:
            raise ValueError(
                'LeWM MPC dynamics must be trained with horizon=1'
            )
        if latent.ndim != 2 or action.ndim != 2:
            raise ValueError(
                'MPC latent/action inputs must have shape (B,D)/(B,A)'
            )
        state_context = latent.unsqueeze(1)
        action_embedding = self.action_encoder(action.unsqueeze(1))
        prediction = self.predictor(state_context, action_embedding)
        return self.pred_proj(prediction[:, -1])

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        sequence_length: int = 1,
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

        available = min(
            actions.shape[1] - self.horizon + 1,
            features.shape[1] - self.horizon,
        )
        windows = min(int(sequence_length), available)
        if windows < 1:
            raise ValueError(
                'LeWM needs at least one complete training window'
            )
        prediction_losses = []
        sigreg_losses = []
        for offset in range(windows):
            batch = features.shape[0]
            window_features = features[:, offset : offset + self.horizon + 1]
            projected = self.projector(window_features.flatten(0, 1)).reshape(
                batch, self.horizon + 1, -1
            )
            action_embeddings = self.action_encoder(
                actions[:, offset : offset + self.horizon]
            )
            predictions = self.predictor(
                projected[:, : self.horizon], action_embeddings
            )
            predictions = self.pred_proj(predictions.flatten(0, 1)).reshape_as(
                projected[:, 1:]
            )
            prediction_losses.append(F.mse_loss(predictions, projected[:, 1:]))
            sigreg_losses.append(self.sigreg(projected.transpose(0, 1)))
        prediction_loss = torch.stack(prediction_losses).mean()
        sigreg_loss = torch.stack(sigreg_losses).mean()
        loss = prediction_loss + self.sigreg_weight * sigreg_loss
        return {
            'loss': loss,
            'prediction_loss': prediction_loss.detach(),
            'sigreg_loss': sigreg_loss.detach(),
        }


class ConsistencyRepresentation(nn.Module):
    """Action-conditioned one-step dynamics trained by latent consistency."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        rho: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0 <= rho <= 1:
            raise ValueError(f'consistency rho must be in [0, 1], got {rho}')
        self.horizon = 1
        self.projection_dim = int(projection_dim)
        self.rho = float(rho)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.transition = nn.Sequential(
            nn.Linear(projection_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    @property
    def mpc_latent_dim(self) -> int:
        return self.projection_dim

    def encode_mpc_latent(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features)

    def predict_mpc_step(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if latent.ndim != 2 or action.ndim != 2:
            raise ValueError(
                'MPC latent/action inputs must have shape (B,D)/(B,A)'
            )
        return self.transition(torch.cat((latent, action), dim=-1))

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        target_features: torch.Tensor,
        sequence_length: int = 1,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or target_features.shape != features.shape:
            raise ValueError(
                'features and targets must have matching (B,T,D) shapes'
            )
        if actions.ndim != 3:
            raise ValueError('actions must have shape (B,T,A)')
        steps = min(
            int(sequence_length), actions.shape[1], features.shape[1] - 1
        )
        if steps < 1:
            raise ValueError('consistency loss needs at least one transition')

        latent = self.encode_mpc_latent(features[:, 0])
        total = features.new_zeros(())
        for index in range(steps):
            latent = self.predict_mpc_step(latent, actions[:, index])
            with torch.no_grad():
                target = self.projector(target_features[:, index + 1].detach())
            total = (
                total
                + self.rho**index * (latent - target).square().sum(-1).mean()
            )
        loss = total / steps
        return {'loss': loss, 'consistency_loss': loss.detach()}


class RewardPredictionRepresentation(nn.Module):
    """Predict either a clip return or one reward from a latent/action pair."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        discount: float = 0.99,
        mode: str = 'clip_return',
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError('reward prediction horizon must be >= 1')
        self.horizon = int(horizon)
        self.discount = float(discount)
        if mode not in {'clip_return', 'per_step'}:
            raise ValueError(
                "reward prediction mode must be 'clip_return' or 'per_step'"
            )
        self.mode = mode
        if mode == 'per_step' and horizon != 1:
            raise ValueError('per-step reward prediction requires horizon=1')
        action_input_dim = (
            horizon * action_dim if mode == 'clip_return' else action_dim
        )
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim + action_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def predict_return(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Predict the same discounted reward clip optimized by this loss."""
        if self.mode != 'clip_return':
            raise ValueError(
                'predict_return is only available in clip_return mode'
            )
        if actions.ndim != 3 or actions.shape[1] != self.horizon:
            raise ValueError(
                f'Reward prediction MPC needs exactly {self.horizon} planned '
                f'actions; got shape {tuple(actions.shape)}'
            )
        action_sequence = actions.flatten(start_dim=1)
        return self.predictor(torch.cat([features, action_sequence], dim=-1))

    def predict_step(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict immediate reward for a WM latent and its action."""
        if self.mode != 'per_step':
            raise ValueError('predict_step is only available in per_step mode')
        if latent.ndim != 2 or action.ndim != 2:
            raise ValueError(
                'per-step reward inputs must have shape (B,Z)/(B,A)'
            )
        return self.predictor(torch.cat((latent, action), dim=-1))

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        sequence_length: int = 1,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.mode == 'per_step':
            if features.ndim != 3 or actions.ndim != 3:
                raise ValueError(
                    'per-step reward training expects features/actions with '
                    'shapes (B,T,Z)/(B,T,A)'
                )
            if rewards.ndim == 3:
                rewards = rewards.squeeze(-1)
            steps = min(
                int(sequence_length),
                features.shape[1] - 1,
                actions.shape[1],
                rewards.shape[1],
            )
            if steps < 1:
                raise ValueError('per-step reward training needs a transition')
            prediction = self.predict_step(
                features[:, :steps].flatten(0, 1),
                actions[:, :steps].flatten(0, 1),
            ).reshape(features.shape[0], steps)
            loss = F.mse_loss(prediction, rewards[:, :steps])
            return {'loss': loss, 'reward_prediction_loss': loss.detach()}

        if actions.shape[1] < self.horizon or rewards.shape[1] < self.horizon:
            raise ValueError(
                f'reward prediction needs {self.horizon} transitions'
            )
        available = min(
            max(1, int(sequence_length) - self.horizon + 1),
            features.shape[1] - self.horizon,
            actions.shape[1] - self.horizon + 1,
            rewards.shape[1] - self.horizon + 1,
            discounts.shape[1] - self.horizon + 1,
        )
        windows = min(int(sequence_length), available)
        if windows < 1:
            raise ValueError(
                f'reward prediction needs {self.horizon} transitions'
            )
        predictions = []
        targets = []
        if rewards.ndim == 3:
            rewards = rewards.squeeze(-1)
        if discounts.ndim == 3:
            discounts = discounts.squeeze(-1)
        for offset in range(windows):
            prediction = self.predict_return(
                features[:, offset],
                actions[:, offset : offset + self.horizon],
            )
            reward = rewards[:, offset : offset + self.horizon]
            mask = discounts[:, offset : offset + self.horizon]
            target = torch.zeros_like(reward[:, :1])
            scale = torch.ones_like(target)
            for index in range(self.horizon):
                target = target + scale * reward[:, index : index + 1]
                scale = scale * self.discount * mask[:, index : index + 1]
            predictions.append(prediction)
            targets.append(target)
        loss = F.mse_loss(torch.cat(predictions), torch.cat(targets))
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
    {
        'infonce',
        'lewm',
        'jepa',
        'pldm',
        'consistency',
        'reward_prediction',
        'curl',
    }
)
_WORLD_MODEL_OBJECTIVES = frozenset(
    {'infonce', 'lewm', 'jepa', 'pldm', 'consistency'}
)


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
    if name == 'reward_prediction':
        mode = str(config.get('mode', 'clip_return'))
        if mode not in {'clip_return', 'per_step'}:
            raise ValueError(
                "reward prediction mode must be 'clip_return' or 'per_step'"
            )
    if 'horizon' in config and int(config['horizon']) < 1:
        raise ValueError(
            f'World-model horizon must be >= 1, got {config["horizon"]}'
        )
    if name == 'consistency' and int(config.get('horizon', 1)) != 1:
        raise ValueError(
            'consistency uses a one-step model; horizon must be 1'
        )
    if (
        name == 'reward_prediction'
        and config.get('mode', 'clip_return') == 'per_step'
        and int(config.get('horizon', 1)) != 1
    ):
        raise ValueError('per-step reward prediction requires horizon=1')
    if name == 'pldm' and int(config.get('projection_dim', 128)) < 2:
        raise ValueError('PLDM requires projection_dim >= 2')
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
            sigreg_num_proj=int(config.get('sigreg_num_proj', 1024)),
        )
    elif name == 'consistency':
        module = ConsistencyRepresentation(
            feature_dim,
            action_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            rho=float(config.get('rho', 0.5)),
        )
    elif name == 'reward_prediction':
        module = RewardPredictionRepresentation(
            feature_dim,
            action_dim,
            horizon,
            hidden_dim,
            (
                discount
                if config.get('discount') is None
                else float(config['discount'])
            ),
            mode=str(config.get('mode', 'clip_return')),
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
        sequence_length: int = 1,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.reward_prediction = reward_prediction
        self.curl = curl
        self.world_model_weight = float(world_model_weight)
        self.reward_prediction_weight = float(reward_prediction_weight)
        self.curl_weight = float(curl_weight)
        if sequence_length < 1:
            raise ValueError('auxiliary sequence_length must be >= 1')
        self.sequence_length = int(sequence_length)
        self.horizon = max(
            1,
            int(getattr(world_model, 'horizon', 1)),
            int(getattr(reward_prediction, 'horizon', 1)),
            self.sequence_length,
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
            objective_inputs = {
                **inputs,
                'sequence_length': self.sequence_length,
            }
            if prefix == 'reward_prediction' and objective.mode == 'per_step':
                if self.world_model is None or not hasattr(
                    self.world_model, 'encode_mpc_latent'
                ):
                    raise ValueError(
                        'Per-step reward prediction requires an MPC-capable '
                        'world model'
                    )
                objective_inputs['features'] = (
                    self.world_model.encode_mpc_latent(inputs['features'])
                )
            result = objective(**objective_inputs)
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
    sequence_length: int = 1,
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
    reward_mode = str(reward_prediction_config.get('mode', 'clip_return'))
    reward_feature_dim = feature_dim
    if reward_enabled and reward_mode == 'per_step':
        if world_model is None or not hasattr(world_model, 'mpc_latent_dim'):
            raise ValueError(
                'Per-step reward prediction requires an action-conditioned '
                'world model with an MPC latent space'
            )
        reward_feature_dim = int(world_model.mpc_latent_dim)
    reward_prediction = (
        RewardPredictionRepresentation(
            reward_feature_dim,
            action_dim,
            horizon=int(reward_prediction_config.get('horizon', 1)),
            hidden_dim=int(reward_prediction_config.get('hidden_dim', 256)),
            discount=(
                discount
                if reward_prediction_config.get('discount') is None
                else float(reward_prediction_config['discount'])
            ),
            mode=reward_mode,
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
        sequence_length=sequence_length,
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
