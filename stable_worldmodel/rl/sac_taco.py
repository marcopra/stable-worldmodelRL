"""State-based SAC with TACO-style temporal action representation learning.

The TACO auxiliary objective follows the local reference implementation:
bilinear in-batch temporal contrast, a learned action tokenizer, and optional
multi-step reward prediction. CURL and image augmentations are intentionally
absent for vector observations.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _orthogonal_linear_init(module: nn.Module) -> None:
    """Match the TACO reference initialization for its linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class StateEncoder(nn.Module):
    """Shared MLP encoder for flat continuous observations."""

    def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.apply(_orthogonal_linear_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state.float())


class SquashedGaussianActor(nn.Module):
    """SAC Gaussian policy with tanh squashing and corrected log probability."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, feature: torch.Tensor):
        hidden = self.trunk(feature)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(-20.0, 2.0)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(
        self, feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(feature)
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = distribution.log_prob(pre_tanh) - torch.log(
            1.0 - action.square() + 1e-6
        )
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(feature).mean)


class SACCritic(nn.Module):
    """Twin Q functions with separate state trunks and shared action input."""

    def __init__(
        self,
        feature_dim: int,
        action_input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.q1 = _mlp(feature_dim + action_input_dim, hidden_dim, 1)
        self.q2 = _mlp(feature_dim + action_input_dim, hidden_dim, 1)

    def forward(
        self, feature: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_action = torch.cat([feature, action], dim=-1)
        return self.q1(state_action), self.q2(state_action)


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
                    self.feature_dim
                    + self.latent_action_dim * self.horizon,
                    hidden_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            if self.reward_prediction
            else None
        )
        # TACO uses a learned bilinear state compatibility matrix.
        self.bilinear = nn.Parameter(torch.rand(self.feature_dim, self.feature_dim))
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


class SACTACOAgent:
    """SAC over state vectors, optionally augmented with TACO losses.

    SAC critic loss updates the shared state encoder and Q networks. The actor
    consumes detached encoder features, and its loss updates only the policy
    and entropy temperature. TACO loss updates the shared encoder, TACO heads,
    and action tokenizer; detached positives block gradients into the target
    state path. Q target computations are fully stop-gradient.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str | torch.device = 'cpu',
        hidden_dim: int = 256,
        feature_dim: int = 256,
        lr: float = 3e-4,
        taco_lr: float = 1e-4,
        alpha_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        initial_alpha: float = 0.2,
        target_entropy: float | None = None,
        taco_enabled: bool = True,
        taco_horizon: int = 3,
        taco_feature_dim: int = 50,
        taco_hidden_dim: int = 256,
        taco_latent_action_dim: int | None = None,
        taco_reward_prediction: bool = True,
        taco_reward_weight: float = 1.0,
    ):
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.discount = float(discount)
        self.tau = float(tau)
        self.taco_enabled = bool(taco_enabled)
        self.taco_horizon = int(taco_horizon)
        self.taco_reward_weight = float(taco_reward_weight)

        self.encoder = StateEncoder(obs_dim, feature_dim, hidden_dim).to(
            self.device
        )
        self.taco = (
            TACOStateObjective(
                state_dim=feature_dim,
                action_dim=action_dim,
                horizon=taco_horizon,
                feature_dim=taco_feature_dim,
                hidden_dim=taco_hidden_dim,
                latent_action_dim=taco_latent_action_dim,
                reward_prediction=taco_reward_prediction,
                reward_weight=taco_reward_weight,
            ).to(self.device)
            if self.taco_enabled
            else None
        )
        critic_action_dim = (
            self.taco.latent_action_dim if self.taco is not None else action_dim
        )
        self.actor = SquashedGaussianActor(
            feature_dim, action_dim, hidden_dim
        ).to(self.device)
        self.critic = SACCritic(
            feature_dim, critic_action_dim, hidden_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.log_alpha = torch.tensor(
            float(initial_alpha), device=self.device
        ).log().requires_grad_(True)
        self.target_entropy = (
            -float(action_dim)
            if target_entropy is None
            else float(target_entropy)
        )

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.encoder_opt = torch.optim.Adam(
            list(self.encoder.parameters())
            + (
                list(self.taco.action_tokenizer.parameters())
                if self.taco is not None
                else []
            ),
            lr=lr,
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.taco_opt = (
            torch.optim.Adam(
                list(self.encoder.parameters())
                + list(self.taco.parameters()),
                lr=taco_lr,
            )
            if self.taco is not None
            else None
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _encode_critic_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.taco is None:
            return action
        return self.taco.encode_action(action)

    def act(self, state, eval_mode: bool = False) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(
                state, device=self.device, dtype=torch.float32
            )
            if state.ndim == 1:
                state = state.unsqueeze(0)
            feature = self.encoder(state)
            action = (
                self.actor.deterministic(feature)
                if eval_mode
                else self.actor.sample(feature)[0]
            )
        return action[0].cpu().numpy()

    @staticmethod
    def _horizon_reward(
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        horizon: int,
        discount: float,
    ) -> torch.Tensor:
        if rewards.ndim == 3:
            rewards = rewards.squeeze(-1)
        if discounts.ndim == 3:
            discounts = discounts.squeeze(-1)
        total = torch.zeros_like(rewards[:, :1])
        multiplier = torch.ones_like(total)
        for index in range(horizon):
            total = total + multiplier * rewards[:, index : index + 1]
            multiplier = (
                multiplier * discount * discounts[:, index : index + 1]
            )
        return total

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def update(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        states = batch['state'].to(self.device).float()
        actions = batch['action'].to(self.device).float()
        rewards = batch['reward'].to(self.device).float()
        discounts = batch['discount'].to(self.device).float()
        if states.ndim != 3 or actions.ndim != 3:
            raise ValueError('replay state/action must have shape (B,T,D)')

        state_feature = self.encoder(states[:, 0])
        current_action = self._encode_critic_action(actions[:, 0])
        with torch.no_grad():
            next_feature = self.encoder(states[:, 1])
            next_action, next_log_prob = self.actor.sample(next_feature)
            next_action = self._encode_critic_action(next_action)
            target_q1, target_q2 = self.critic_target(
                next_feature, next_action
            )
            target_value = torch.minimum(target_q1, target_q2) - (
                self.alpha.detach() * next_log_prob
            )
            reward = rewards[:, 0].reshape(-1, 1)
            bootstrap_discount = discounts[:, 0].reshape(-1, 1)
            target_q = (
                reward
                + self.discount * bootstrap_discount * target_value
            )

        q1, q2 = self.critic(state_feature, current_action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        # Freeze Q/action-encoder weights while retaining dQ/da for the policy.
        self.critic_opt.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)
        self._set_requires_grad(self.critic, False)
        if self.taco is not None:
            self._set_requires_grad(self.taco.action_tokenizer, False)
        actor_feature = state_feature.detach()
        policy_action, log_prob = self.actor.sample(actor_feature)
        actor_q1, actor_q2 = self.critic(
            actor_feature, self._encode_critic_action(policy_action)
        )
        actor_loss = (
            self.alpha.detach() * log_prob
            - torch.minimum(actor_q1, actor_q2)
        ).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self._set_requires_grad(self.critic, True)
        if self.taco is not None:
            self._set_requires_grad(self.taco.action_tokenizer, True)

        alpha_loss = -(
            self.log_alpha * (log_prob.detach() + self.target_entropy)
        ).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for source, target in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target.lerp_(source, self.tau)

        metrics = {
            'critic/loss': float(critic_loss.detach()),
            'critic/q1': float(q1.detach().mean()),
            'critic/q2': float(q2.detach().mean()),
            'critic/target_q': float(target_q.detach().mean()),
            'actor/loss': float(actor_loss.detach()),
            'actor/log_prob': float(log_prob.detach().mean()),
            'temperature/alpha': float(self.alpha.detach()),
            'temperature/loss': float(alpha_loss.detach()),
        }

        if self.taco is not None:
            horizon = self.taco_horizon
            if states.shape[1] < horizon + 1 or actions.shape[1] < horizon:
                raise ValueError(
                    f'TACO needs {horizon + 1} states and {horizon} actions'
                )
            temporal_states = states[:, : horizon + 1]
            online_features = self.encoder(
                temporal_states.reshape(-1, temporal_states.shape[-1])
            ).reshape(states.shape[0], horizon + 1, -1)
            with torch.no_grad():
                target_features = self.encoder(
                    temporal_states.reshape(
                        -1, temporal_states.shape[-1]
                    )
                ).reshape(states.shape[0], horizon + 1, -1)
            horizon_reward = self._horizon_reward(
                rewards,
                discounts,
                horizon,
                self.discount,
            )
            if self.taco_opt is None:
                raise RuntimeError('TACO optimizer is missing')
            self.taco_opt.zero_grad(set_to_none=True)
            taco_metrics = self.taco(
                online_features,
                actions[:, :horizon],
                target_features,
                horizon_reward,
            )
            total_taco_loss = taco_metrics['loss']
            # The temporal contrast is the primary objective. Reward prediction
            # is the same optional TACO auxiliary head, controlled by its weight.
            total_taco_loss.backward()
            self.taco_opt.step()
            metrics.update(
                {
                    'taco/loss': float(taco_metrics['loss'].detach()),
                    'taco/contrastive_loss': float(
                        taco_metrics['taco_loss']
                    ),
                    'taco/reward_loss': float(taco_metrics['reward_loss']),
                    'taco/positive_logit': float(
                        taco_metrics['positive_logit']
                    ),
                    'taco/negative_logit': float(
                        taco_metrics['negative_logit']
                    ),
                }
            )

        return metrics

    def state_dict(self) -> dict:
        return {
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'taco': self.taco.state_dict() if self.taco is not None else None,
            'log_alpha': self.log_alpha.detach().cpu(),
            'encoder_opt': self.encoder_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'taco_opt': (
                self.taco_opt.state_dict() if self.taco_opt is not None else None
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state['encoder'])
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        if self.taco is not None:
            if state['taco'] is None:
                raise ValueError('Checkpoint has no TACO objective')
            self.taco.load_state_dict(state['taco'])
        self.log_alpha.data.copy_(state['log_alpha'].to(self.device))
        self.encoder_opt.load_state_dict(state['encoder_opt'])
        self.actor_opt.load_state_dict(state['actor_opt'])
        self.critic_opt.load_state_dict(state['critic_opt'])
        self.alpha_opt.load_state_dict(state['alpha_opt'])
        if self.taco_opt is not None and state['taco_opt'] is not None:
            self.taco_opt.load_state_dict(state['taco_opt'])
