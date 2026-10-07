"""Soft Actor-Critic with configurable observation and auxiliary encoders."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..augmentation import RandomShiftsAug
from ..encoders import make_observation_encoder, pixel_batch_to_nchw
from ..representation import (
    build_auxiliary_losses,
    validate_world_model_config,
)


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
    """Twin Q functions over encoded observations and continuous actions."""

    def __init__(
        self, feature_dim: int, action_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.q1 = _mlp(feature_dim + action_dim, hidden_dim, 1)
        self.q2 = _mlp(feature_dim + action_dim, hidden_dim, 1)

    def forward(
        self, feature: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_action = torch.cat([feature, action], dim=-1)
        return self.q1(state_action), self.q2(state_action)


class SACAgent:
    """SAC agent for vector or pixel observations.

    The critic update owns shared observation-encoder gradients. Auxiliary
    objectives use a separate optimizer over the same encoder and their heads.
    The actor consumes detached features, preserving SAC's usual gradient path.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...] | None = None,
        action_dim: int | None = None,
        *,
        obs_dim: int | None = None,
        observation_mode: str = 'state',
        device: str | torch.device = 'cpu',
        hidden_dim: int = 256,
        feature_dim: int = 256,
        lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        initial_alpha: float = 0.2,
        target_entropy: float | None = None,
        representation_horizon: int = 3,
        representation_dim: int = 128,
        representation_hidden_dim: int = 256,
        auxiliary_enabled: bool | None = None,
        world_model_config: dict | None = None,
        reward_prediction_config: dict | None = None,
        curl_config: dict | None = None,
        augmentation_pad: int = 4,
        # Compatibility parameters for the previous state-only SAC/TACO API.
        taco_enabled: bool | None = None,
        taco_horizon: int | None = None,
        taco_feature_dim: int = 50,
        taco_hidden_dim: int = 256,
    ):
        if action_dim is None:
            raise TypeError('action_dim is required')
        if obs_shape is None:
            if obs_dim is None:
                raise TypeError('obs_shape or obs_dim is required')
            obs_shape = (int(obs_dim),)
        if taco_horizon is not None:
            representation_horizon = int(taco_horizon)
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.discount = float(discount)
        self.tau = float(tau)
        self.observation_mode = str(observation_mode)
        reward_prediction_config = dict(reward_prediction_config or {})
        curl_config = dict(curl_config or {})
        if world_model_config is None and taco_enabled:
            world_model_config = {
                'name': 'infonce',
                'horizon': taco_horizon or representation_horizon,
                'projection_dim': taco_feature_dim,
                'hidden_dim': taco_hidden_dim,
            }
        if auxiliary_enabled is None:
            world_model_config = world_model_config or {}
            configured_name = str(
                world_model_config.get(
                    'target', world_model_config.get('name', 'none')
                )
            ).lower()
            auxiliary_enabled = (
                (
                    configured_name not in {'none', 'null', ''}
                    and bool(world_model_config.get('enabled', True))
                )
                or bool(reward_prediction_config.get('enabled', False))
                or bool(curl_config.get('enabled', False))
            )
        self.world_model_config = dict(world_model_config or {'name': 'none'})
        self.world_model_name, _ = validate_world_model_config(
            self.world_model_config,
            observation_mode=self.observation_mode,
        )
        self.representation_horizon = int(
            self.world_model_config.get('horizon', representation_horizon)
        )
        self.encoder = make_observation_encoder(
            self.observation_mode,
            tuple(obs_shape),
            feature_dim,
            hidden_dim,
            pixel_feature_dim=feature_dim,
        ).to(self.device)

        self.auxiliary_losses = build_auxiliary_losses(
            self.world_model_config,
            reward_prediction_config,
            curl_config,
            feature_dim=self.encoder.repr_dim,
            action_dim=self.action_dim,
            discount=self.discount,
            observation_mode=self.observation_mode,
            enabled=bool(auxiliary_enabled),
        )
        if self.auxiliary_losses is not None:
            self.auxiliary_losses = self.auxiliary_losses.to(self.device)
        self.world_model = (
            self.auxiliary_losses.world_model
            if self.auxiliary_losses is not None
            else None
        )
        self.reward_prediction = (
            self.auxiliary_losses.reward_prediction
            if self.auxiliary_losses is not None
            else None
        )
        self.curl = (
            self.auxiliary_losses.curl
            if self.auxiliary_losses is not None
            else None
        )
        self.auxiliary_horizon = (
            self.auxiliary_losses.horizon
            if self.auxiliary_losses is not None
            else 1
        )
        if self.world_model is not None:
            self.world_model_name = self.world_model.objective_name
        self.world_model_weight = (
            self.auxiliary_losses.world_model_weight
            if self.auxiliary_losses is not None
            else 0.0
        )

        self.action_encoder = (
            self.world_model.action_encoder
            if self.world_model_name in {'infonce', 'taco'}
            and self.world_model is not None
            else None
        )
        critic_action_dim = (
            self.world_model.latent_action_dim
            if self.action_encoder is not None
            else self.action_dim
        )
        self.actor = SquashedGaussianActor(
            self.encoder.repr_dim, self.action_dim, hidden_dim
        ).to(self.device)
        self.critic = SACCritic(
            self.encoder.repr_dim, critic_action_dim, hidden_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.log_alpha = (
            torch.tensor(float(initial_alpha), device=self.device)
            .log()
            .requires_grad_(True)
        )
        self.target_entropy = (
            -float(action_dim)
            if target_entropy is None
            else float(target_entropy)
        )
        encoder_parameters = list(self.encoder.parameters())
        if self.action_encoder is not None:
            encoder_parameters += list(self.action_encoder.parameters())
        self.encoder_opt = torch.optim.Adam(encoder_parameters, lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.world_model_opt = (
            torch.optim.Adam(
                list(self.encoder.parameters())
                + list(self.auxiliary_losses.parameters()),
                lr=lr,
            )
            if self.auxiliary_losses is not None
            else None
        )
        self.aug = RandomShiftsAug(
            augmentation_pad if self.observation_mode == 'pixels' else 0
        ).to(self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def representation(self):
        """Compatibility alias for the single selected world-model loss."""
        return self.world_model

    @property
    def representation_opt(self):
        return self.world_model_opt

    @property
    def taco(self):
        return self.world_model if self.world_model_name == 'infonce' else None

    def _prepare(self, observation: torch.Tensor) -> torch.Tensor:
        if self.observation_mode == 'pixels':
            return pixel_batch_to_nchw(observation)
        return observation.float().flatten(start_dim=1)

    def _encode_sequence(self, observations: torch.Tensor) -> torch.Tensor:
        batch, time = observations.shape[:2]
        encoded = self.encoder(self._prepare(observations.flatten(0, 1)))
        return encoded.reshape(batch, time, -1)

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_encoder is None:
            return action
        return self.action_encoder(action)

    def act(
        self,
        observation,
        step: int = 0,
        eval_mode: bool = False,
    ) -> np.ndarray:
        del step  # SAC has no scheduled exploration noise.
        with torch.no_grad():
            obs = torch.as_tensor(observation, device=self.device)
            if self.observation_mode == 'pixels':
                if obs.ndim == 3:
                    obs = obs.unsqueeze(0)
                obs = self._prepare(obs)
            else:
                obs = obs.float().reshape(1, -1)
            feature = self.encoder(obs)
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
        """Legacy helper retained for state SAC experiment notebooks."""
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

    def update(
        self,
        batch: dict[str, torch.Tensor],
        step: int = 0,
    ) -> dict[str, float]:
        del step
        observations = batch.get(
            'observation', batch.get('state', batch.get('pixels'))
        )
        if observations is None:
            raise KeyError("RL batch needs an 'observation' sequence")
        observations = observations.to(self.device)
        actions = batch['action'].to(self.device).float()
        rewards = batch['reward'].to(self.device).float()
        discounts = batch['discount'].to(self.device).float()
        if observations.ndim < 3 or actions.ndim != 3:
            raise ValueError(
                'replay observations/actions must be temporal batches'
            )

        current_obs = self._prepare(observations[:, 0])
        state_feature = self.encoder(current_obs)
        with torch.no_grad():
            next_feature = self.encoder(self._prepare(observations[:, 1]))
            next_action, next_log_prob = self.actor.sample(next_feature)
            target_q1, target_q2 = self.critic_target(
                next_feature, self._encode_action(next_action)
            )
            target_value = torch.minimum(target_q1, target_q2) - (
                self.alpha.detach() * next_log_prob
            )
            reward = rewards[:, 0].reshape(-1, 1)
            bootstrap_discount = discounts[:, 0].reshape(-1, 1)
            target_q = (
                reward + self.discount * bootstrap_discount * target_value
            )

        q1, q2 = self.critic(state_feature, self._encode_action(actions[:, 0]))
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        self.critic_opt.zero_grad(set_to_none=True)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        if self.action_encoder is not None:
            for parameter in self.action_encoder.parameters():
                parameter.requires_grad_(False)
        actor_feature = state_feature.detach()
        policy_action, log_prob = self.actor.sample(actor_feature)
        actor_q1, actor_q2 = self.critic(
            actor_feature, self._encode_action(policy_action)
        )
        actor_loss = (
            self.alpha.detach() * log_prob - torch.minimum(actor_q1, actor_q2)
        ).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        if self.action_encoder is not None:
            for parameter in self.action_encoder.parameters():
                parameter.requires_grad_(True)

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

        if self.auxiliary_losses is not None:
            horizon = self.auxiliary_horizon
            temporal_observations = observations[:, : horizon + 1]
            features = self._encode_sequence(temporal_observations)
            with torch.no_grad():
                targets = self._encode_sequence(temporal_observations)
            curl_features = None
            if self.curl is not None:
                pixels = self._prepare(observations[:, 0])
                query = self.encoder(self.aug(pixels.float()))
                key = self.encoder(self.aug(pixels.float()))
                curl_features = (query, key)
            if self.world_model_opt is None:
                raise RuntimeError('world-model optimizer missing')
            self.world_model_opt.zero_grad(set_to_none=True)
            auxiliary_metrics = self.auxiliary_losses(
                features=features,
                actions=actions[:, :horizon],
                target_features=targets,
                rewards=rewards,
                discounts=discounts,
                curl_features=curl_features,
            )
            weighted_loss = auxiliary_metrics['loss']
            weighted_loss.backward()
            self.world_model_opt.step()
            for name, value in auxiliary_metrics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    metrics['auxiliary/loss' if name == 'loss' else name] = (
                        float(value.detach())
                    )
        return metrics

    def state_dict(self) -> dict:
        return {
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'world_model': (
                self.world_model.state_dict()
                if self.world_model is not None
                else None
            ),
            'reward_prediction': (
                self.reward_prediction.state_dict()
                if self.reward_prediction is not None
                else None
            ),
            'curl': self.curl.state_dict() if self.curl is not None else None,
            'log_alpha': self.log_alpha.detach().cpu(),
            'encoder_opt': self.encoder_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'world_model_opt': (
                self.world_model_opt.state_dict()
                if self.world_model_opt is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state['encoder'])
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        if self.world_model is not None:
            world_model_state = state.get(
                'world_model', state.get('representation')
            )
            if world_model_state is None:
                raise ValueError('Checkpoint has no world-model loss state')
            if 'objectives' in world_model_state:
                objectives = world_model_state['objectives']
                if len(objectives) != 1:
                    raise ValueError(
                        'Cannot restore a checkpoint with multiple '
                        'world-model objectives into a single-objective agent'
                    )
                world_model_state = next(iter(objectives.values()))
            self.world_model.load_state_dict(world_model_state)
        if self.reward_prediction is not None:
            reward_state = state.get('reward_prediction')
            if reward_state is None:
                raise ValueError(
                    'Checkpoint has no reward-prediction loss state'
                )
            self.reward_prediction.load_state_dict(reward_state)
        if self.curl is not None:
            curl_state = state.get('curl')
            if curl_state is None:
                raise ValueError('Checkpoint has no CURL loss state')
            self.curl.load_state_dict(curl_state)
        self.log_alpha.data.copy_(state['log_alpha'].to(self.device))
        self.encoder_opt.load_state_dict(state['encoder_opt'])
        self.actor_opt.load_state_dict(state['actor_opt'])
        self.critic_opt.load_state_dict(state['critic_opt'])
        self.alpha_opt.load_state_dict(state['alpha_opt'])
        world_model_opt_state = state.get(
            'world_model_opt', state.get('representation_opt')
        )
        if (
            self.world_model_opt is not None
            and world_model_opt_state is not None
        ):
            self.world_model_opt.load_state_dict(world_model_opt_state)


class SACTACOAgent(SACAgent):
    """Backward-compatible name retaining the former TACO-on default."""

    def __init__(self, *args, taco_enabled: bool = True, **kwargs):
        super().__init__(*args, taco_enabled=taco_enabled, **kwargs)
