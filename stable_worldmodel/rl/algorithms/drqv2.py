"""DrQ-v2 agent for vector-state and pixel observations."""

from __future__ import annotations

import copy

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
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(
        self, repr_dim: int, action_dim: int, feature_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.policy = _mlp(feature_dim, hidden_dim, action_dim)

    def forward(
        self, obs: torch.Tensor, stddev: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.policy(self.trunk(obs)).tanh()
        return mean, torch.ones_like(mean) * stddev


class Critic(nn.Module):
    def __init__(
        self, repr_dim: int, action_dim: int, feature_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.q1 = _mlp(feature_dim + action_dim, hidden_dim, 1)
        self.q2 = _mlp(feature_dim + action_dim, hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.trunk(obs)
        obs_action = torch.cat([obs, action], dim=-1)
        return self.q1(obs_action), self.q2(obs_action)


class DrQV2Agent:
    """DrQ-v2 actor/critic with an optional temporal representation objective.

    The critic optimizer owns encoder gradients, matching DrQ-v2. Actor inputs
    are detached. An enabled auxiliary objective has a separate optimizer that
    updates the shared encoder and its own prediction modules.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        observation_mode: str = 'pixels',
        device: str | torch.device = 'cpu',
        lr: float = 1e-4,
        feature_dim: int = 50,
        hidden_dim: int = 1024,
        critic_target_tau: float = 0.01,
        discount: float = 0.99,
        stddev: float = 0.2,
        stddev_final: float = 0.1,
        stddev_schedule_steps: int = 500_000,
        stddev_clip: float = 0.3,
        num_expl_steps: int = 2000,
        representation_loss: str = 'none',
        representation_weight: float = 1.0,
        representation_horizon: int = 1,
        representation_temperature: float = 0.1,
        representation_dim: int = 128,
        representation_hidden_dim: int = 256,
        representation_lewm_hidden_dim: int = 2048,
        representation_sigreg_weight: float = 0.09,
        auxiliary_enabled: bool | None = None,
        world_model_config: dict | None = None,
        reward_prediction_config: dict | None = None,
        curl_config: dict | None = None,
        augmentation_pad: int = 4,
        nstep: int = 1,
    ):
        self.device = torch.device(device)
        self.action_dim = int(action_shape[0])
        self.discount = float(discount)
        self.stddev = float(stddev)
        self.stddev_final = float(stddev_final)
        self.stddev_schedule_steps = max(1, int(stddev_schedule_steps))
        self.stddev_clip = float(stddev_clip)
        self.num_expl_steps = int(num_expl_steps)
        self.tau = float(critic_target_tau)
        self.representation_weight = float(representation_weight)
        self.representation_horizon = int(representation_horizon)
        self.observation_mode = str(observation_mode)
        reward_prediction_config = dict(reward_prediction_config or {})
        curl_config = dict(curl_config or {})
        self.nstep = int(nstep)
        if self.nstep < 1:
            raise ValueError(f'nstep must be >= 1, got {self.nstep}')

        self.encoder = make_observation_encoder(
            self.observation_mode,
            obs_shape,
            feature_dim,
            hidden_dim,
            pixel_feature_dim=None,
        ).to(self.device)
        repr_dim = self.encoder.repr_dim
        if world_model_config is None and representation_loss not in {
            'none',
            '',
            'null',
        }:
            world_model_config = {
                'name': representation_loss,
                'horizon': self.representation_horizon,
                'projection_dim': representation_dim,
                'hidden_dim': representation_hidden_dim,
                'temperature': representation_temperature,
                'lewm_hidden_dim': representation_lewm_hidden_dim,
                'sigreg_weight': representation_sigreg_weight,
                'weight': self.representation_weight,
            }
        elif (
            world_model_config is not None and self.representation_weight != 1
        ):
            world_model_config = dict(world_model_config)
            world_model_config['weight'] = (
                float(world_model_config.get('weight', 1.0))
                * self.representation_weight
            )
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
            self.world_model_config.get('horizon', self.representation_horizon)
        )
        self.auxiliary_losses = build_auxiliary_losses(
            self.world_model_config,
            reward_prediction_config,
            curl_config,
            feature_dim=repr_dim,
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
            if self.world_model is not None
            and self.world_model_name == 'infonce'
            else None
        )
        critic_action_dim = (
            self.world_model.latent_action_dim
            if self.action_encoder is not None
            else self.action_dim
        )
        self.actor = Actor(
            repr_dim, self.action_dim, feature_dim, hidden_dim
        ).to(self.device)
        self.critic = Critic(
            repr_dim, critic_action_dim, feature_dim, hidden_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        encoder_parameters = list(self.encoder.parameters())
        if self.action_encoder is not None:
            encoder_parameters += list(self.action_encoder.parameters())
        self.encoder_opt = torch.optim.Adam(encoder_parameters, lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
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
    def representation(self):
        """Compatibility alias for the single selected world-model loss."""
        return self.world_model

    @property
    def representation_opt(self):
        return self.world_model_opt

    def _stddev(self, step: int) -> float:
        mix = min(max(step / self.stddev_schedule_steps, 0.0), 1.0)
        return self.stddev + mix * (self.stddev_final - self.stddev)

    @staticmethod
    def _sample_action(
        mean: torch.Tensor, std: torch.Tensor, noise_clip: float | None = None
    ) -> torch.Tensor:
        noise = torch.randn_like(mean) * std
        if noise_clip is not None:
            noise = noise.clamp(-noise_clip, noise_clip)
        return (mean + noise).clamp(-1.0, 1.0)

    def _as_nchw(self, obs: torch.Tensor) -> torch.Tensor:
        if self.observation_mode == 'pixels':
            return pixel_batch_to_nchw(obs)
        return obs.float().flatten(start_dim=1)

    def _encode_sequence(self, observations: torch.Tensor) -> torch.Tensor:
        batch, time = observations.shape[:2]
        flat = observations.flatten(0, 1)
        encoded = self.encoder(self._as_nchw(flat))
        return encoded.reshape(batch, time, -1)

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_encoder is None:
            return action
        return self.action_encoder(action)

    def act(self, obs, step: int = 0, eval_mode: bool = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device)
            if self.observation_mode == 'pixels' and obs.ndim == 3:
                obs = obs.unsqueeze(0)
            elif self.observation_mode == 'state':
                obs = obs.float().reshape(1, -1)
            feature = self.encoder(self._as_nchw(obs))
            mean, std = self.actor(feature, self._stddev(step))
            if eval_mode:
                action = mean
            else:
                action = self._sample_action(mean, std)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
        return action[0].cpu().numpy()

    def update(
        self, batch: dict[str, torch.Tensor], step: int = 0
    ) -> dict[str, float]:
        """Run one critic/actor update and optional representation update."""
        observation = batch.get(
            'observation', batch.get('pixels', batch.get('state'))
        )
        if observation is None:
            raise KeyError("RL batch needs an 'observation' sequence")
        obs_seq = observation.to(self.device)
        actions = batch['action'].to(self.device).float()
        rewards = batch['reward'].to(self.device).float()
        discounts = batch['discount'].to(self.device).float()
        # The final frame has no outgoing action/reward. Collector writes a
        # harmless sentinel there to keep every episode column aligned.
        obs = self._as_nchw(obs_seq[:, 0])
        next_obs = self._as_nchw(obs_seq[:, self.nstep])
        if self.observation_mode == 'pixels':
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        obs_feature = self.encoder(obs)
        reward_discount = torch.ones_like(rewards[:, 0:1])
        nstep_reward = torch.zeros_like(rewards[:, 0:1])
        for index in range(self.nstep):
            nstep_reward = (
                nstep_reward + reward_discount * rewards[:, index : index + 1]
            )
            reward_discount = (
                reward_discount
                * self.discount
                * discounts[:, index : index + 1]
            )
        with torch.no_grad():
            next_feature = self.encoder(next_obs)
            mean, std = self.actor(next_feature, self._stddev(step))
            next_action = self._sample_action(mean, std, self.stddev_clip)
            target_q1, target_q2 = self.critic_target(
                next_feature, self._encode_action(next_action)
            )
            target_q = nstep_reward + reward_discount * torch.minimum(
                target_q1, target_q2
            )

        q1, q2 = self.critic(obs_feature, self._encode_action(actions[:, 0]))
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        actor_feature = obs_feature.detach()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        if self.action_encoder is not None:
            for parameter in self.action_encoder.parameters():
                parameter.requires_grad_(False)
        mean, std = self.actor(actor_feature, self._stddev(step))
        actor_action = self._sample_action(mean, std, self.stddev_clip)
        actor_q1, actor_q2 = self.critic(
            actor_feature, self._encode_action(actor_action)
        )
        actor_loss = -torch.minimum(actor_q1, actor_q2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        if self.action_encoder is not None:
            for parameter in self.action_encoder.parameters():
                parameter.requires_grad_(True)

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
        }

        if self.auxiliary_losses is not None:
            h = self.auxiliary_horizon
            temporal_observations = obs_seq[:, : h + 1].flatten(0, 1)
            temporal_input = self._as_nchw(temporal_observations)
            if self.observation_mode == 'pixels':
                temporal_input = self.aug(temporal_input.float())
            temporal = self.encoder(temporal_input).view(
                obs_seq.shape[0], h + 1, -1
            )
            if self.world_model_opt is None:
                raise RuntimeError('world-model optimizer missing')
            self.world_model_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                target_input = self._as_nchw(temporal_observations)
                if self.observation_mode == 'pixels':
                    target_input = self.aug(target_input.float())
                target_temporal = self.encoder(target_input).view(
                    obs_seq.shape[0], h + 1, -1
                )
            curl_features = None
            if self.curl is not None:
                current_pixels = self._as_nchw(obs_seq[:, 0])
                curl_query = self.encoder(self.aug(current_pixels.float()))
                curl_key = self.encoder(self.aug(current_pixels.float()))
                curl_features = (curl_query, curl_key)
            auxiliary_metrics = self.auxiliary_losses(
                features=temporal,
                actions=actions[:, :h],
                target_features=target_temporal,
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
            'encoder_opt': self.encoder_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
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
        self.encoder_opt.load_state_dict(state['encoder_opt'])
        self.actor_opt.load_state_dict(state['actor_opt'])
        self.critic_opt.load_state_dict(state['critic_opt'])
        world_model_opt_state = state.get(
            'world_model_opt', state.get('representation_opt')
        )
        if (
            self.world_model_opt is not None
            and world_model_opt_state is not None
        ):
            self.world_model_opt.load_state_dict(world_model_opt_state)
