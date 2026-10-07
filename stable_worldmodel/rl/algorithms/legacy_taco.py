"""Compatibility TACO objective retained for older SAC notebooks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _orthogonal_linear_init(module: nn.Module) -> None:
    """Match the TACO reference initialization for its linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TACOStateObjective(nn.Module):
    """TACO temporal InfoNCE and optional discounted reward prediction.

    Positive features use the online state encoder and this module's state
    projector under stop-gradient, matching TACO's detached positive path. The
    online anchor, action tokenizer, dynamics head, bilinear matrix, and reward
    head receive auxiliary gradients. There is no EMA target and no CURL loss.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 3,
        feature_dim: int = 50,
        hidden_dim: int = 256,
        latent_action_dim: int | None = None,
        reward_prediction: bool = True,
        reward_weight: float = 1.0,
    ):
        super().__init__()
        if horizon < 1:
            raise ValueError(f'horizon must be >= 1, got {horizon}')
        if latent_action_dim is None:
            latent_action_dim = int(action_dim * 1.25) + 1
        if latent_action_dim < 1:
            raise ValueError('latent_action_dim must be positive')
        self.horizon = int(horizon)
        self.feature_dim = int(feature_dim)
        self.latent_action_dim = int(latent_action_dim)
        self.reward_prediction = bool(reward_prediction)
        self.reward_weight = float(reward_weight)

        # Same two-stage tokenizer used by TACO: embed each action, then encode
        # the flattened temporal action sequence.
        self.action_tokenizer = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.Tanh(),
            nn.Linear(64, self.latent_action_dim),
        )
        self.action_sequence_tokenizer = nn.Sequential(
            nn.Linear(
                self.latent_action_dim * self.horizon,
                self.latent_action_dim * self.horizon,
            ),
            nn.LayerNorm(self.latent_action_dim * self.horizon),
            nn.Tanh(),
        )
        self.state_projector = nn.Sequential(
            nn.Linear(state_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh(),
        )
        self.state_action_predictor = nn.Sequential(
            nn.Linear(
                self.feature_dim + self.latent_action_dim * self.horizon,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        self.reward_predictor = (
            nn.Sequential(
                nn.Linear(
                    self.feature_dim + self.latent_action_dim * self.horizon,
                    hidden_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            if self.reward_prediction
            else None
        )
        # TACO uses a learned bilinear state compatibility matrix.
        self.bilinear = nn.Parameter(
            torch.rand(self.feature_dim, self.feature_dim)
        )
        # TACO initializes the action tokenizer, projection, predictor, and
        # reward head with orthogonal linear weights and zero biases.
        self.apply(_orthogonal_linear_init)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_tokenizer(action)

    def encode_action_sequence(self, actions: torch.Tensor) -> torch.Tensor:
        batch = actions.shape[0]
        embedded = self.action_tokenizer(actions[:, : self.horizon])
        return self.action_sequence_tokenizer(embedded.reshape(batch, -1))

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        target_features: torch.Tensor,
        horizon_reward: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or target_features.ndim != 3:
            raise ValueError('features and targets must have shape (B,T,D)')
        if features.shape[1] != self.horizon + 1:
            raise ValueError(
                f'Expected {self.horizon + 1} states, got {features.shape[1]}'
            )
        if target_features.shape != features.shape:
            raise ValueError('target_features must match features shape')
        if actions.ndim != 3 or actions.shape[1] < self.horizon:
            raise ValueError(
                f'actions must have at least {self.horizon} temporal entries'
            )
        if features.shape[0] < 2:
            raise ValueError('TACO InfoNCE needs at least two batch elements')

        batch = features.shape[0]
        anchor = self.state_projector(features[:, 0])
        action_sequence = self.encode_action_sequence(actions)
        prediction_input = torch.cat([anchor, action_sequence], dim=-1)
        predicted = self.state_action_predictor(prediction_input)
        with torch.no_grad():
            positive = self.state_projector(target_features[:, -1].detach())

        logits = predicted @ self.bilinear @ positive.T
        # Subtracting each row maximum is softmax invariant and matches the
        # reference's stable logit construction.
        logits = logits - logits.max(dim=1, keepdim=True).values
        labels = torch.arange(batch, device=features.device)
        taco_loss = F.cross_entropy(logits, labels)

        if self.reward_predictor is not None:
            reward_prediction = self.reward_predictor(prediction_input)
            reward_loss = F.mse_loss(reward_prediction, horizon_reward)
        else:
            reward_loss = taco_loss.new_zeros(())

        return {
            'loss': taco_loss + self.reward_weight * reward_loss,
            'taco_loss': taco_loss.detach(),
            'reward_loss': reward_loss.detach(),
            'positive_logit': logits.diagonal().mean().detach(),
            'negative_logit': (
                (logits.sum() - logits.diagonal().sum())
                / (batch * (batch - 1))
            ).detach(),
            'logits': logits.detach(),
        }
