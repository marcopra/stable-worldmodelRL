"""Auxiliary temporal representation objectives for online RL."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor
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

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        target_features: torch.Tensor,
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
        if time != self.horizon + 1 or actions.shape[1] != self.horizon:
            raise ValueError(
                f'Expected {self.horizon + 1} frames and {self.horizon} actions; '
                f'got {time} frames and {actions.shape[1]} actions'
            )
        if target_features.shape != features.shape:
            raise ValueError(
                'target_features must have the same shape as features'
            )

        encoded_actions = self.action_encoder(actions)
        source = self.projector(features[:, 0])
        prediction_input = torch.cat(
            [source, encoded_actions.flatten(start_dim=1)], dim=-1
        )
        predicted = F.normalize(
            self.predictor(prediction_input).float(), dim=-1
        )
        with torch.no_grad():
            target = F.normalize(
                self.projector(target_features[:, -1].detach()).float(), dim=-1
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

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
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
        self, features: torch.Tensor, target_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or target_features.ndim != 3:
            raise ValueError('features and targets must have shape (B,T,D)')
        if features.shape[1] != self.horizon + 1:
            raise ValueError(
                f'Expected {self.horizon + 1} frames, got {features.shape[1]}'
            )
        context = self.projector(features[:, : self.horizon])
        prediction = self.predictor(context)[:, -1]
        with torch.no_grad():
            target = self.projector(target_features[:, -1].detach())
        loss = F.mse_loss(prediction, target)
        return {'loss': loss, 'jepa_loss': loss.detach()}


class LeWMTemporalRepresentation(nn.Module):
    """LeWM prediction + SIGReg objective on the shared DrQ pixel features.

    This reuses Stable World Model's action embedder, predictor, projector, and
    SIGReg loss. The DrQ convolutional encoder stays shared with the policy;
    this is the LeWM objective adapter, not the standalone LeWM ViT model.
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
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or actions.ndim != 3:
            raise ValueError('features/actions must have shapes (B,T,D)/(B,T,A)')
        if features.shape[1] != self.horizon + 1:
            raise ValueError(
                f'Expected {self.horizon + 1} frames, got {features.shape[1]}'
            )
        if actions.shape[1] < self.horizon:
            raise ValueError(
                f'Expected at least {self.horizon} actions, got {actions.shape[1]}'
            )

        batch = features.shape[0]
        projected = self.projector(features.flatten(0, 1)).reshape(
            batch, self.horizon + 1, -1
        )
        action_embeddings = self.action_encoder(actions[:, : self.horizon])
        predictions = self.predictor(
            projected[:, : self.horizon], action_embeddings
        )
        predictions = self.pred_proj(
            predictions.flatten(0, 1)
        ).reshape_as(projected[:, 1:])
        prediction_loss = F.mse_loss(predictions, projected[:, 1:])
        sigreg_loss = self.sigreg(projected.transpose(0, 1))
        loss = prediction_loss + self.sigreg_weight * sigreg_loss
        return {
            'loss': loss,
            'prediction_loss': prediction_loss.detach(),
            'sigreg_loss': sigreg_loss.detach(),
        }


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
) -> nn.Module | None:
    """Build a supported objective; ``none`` leaves the DrQ baseline intact."""
    name = name.lower()
    if name in {'none', 'null', ''}:
        return None
    if name in {'infonce', 'taco'}:
        return InfoNCERepresentation(
            feature_dim,
            action_dim,
            horizon,
            projection_dim,
            hidden_dim,
            temperature,
        )
    if name == 'pldm':
        return PLDMRepresentation(feature_dim, projection_dim)
    if name in {'jepa', 'prejepa'}:
        return JEPATemporalRepresentation(feature_dim, horizon, projection_dim)
    if name in {'lewm', 'lewn'}:
        return LeWMTemporalRepresentation(
            feature_dim,
            action_dim,
            horizon,
            projection_dim=projection_dim,
            hidden_dim=lewm_hidden_dim,
            sigreg_weight=sigreg_weight,
        )
    raise ValueError(
        f'Unknown representation loss {name!r}; choose none, infonce, lewm, jepa, or pldm'
    )
