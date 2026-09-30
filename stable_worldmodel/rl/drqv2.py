"""A compact DrQ-v2 agent for pixel observations."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .representation import (
    JEPATemporalRepresentation,
    LeWMTemporalRepresentation,
    PLDMRepresentation,
    build_representation,
)


class RandomShiftsAug(nn.Module):
    """Per-image random translation using DrQ's replicate-padded grid sample."""

    def __init__(self, pad: int = 4):
        super().__init__()
        if pad < 0:
            raise ValueError(f'pad must be >= 0, got {pad}')
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad == 0:
            return x
        n, _, height, width = x.shape
        if height != width:
            raise ValueError('RandomShiftsAug expects square images')
        pad = self.pad
        x = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        eps = 1.0 / (height + 2 * pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            height + 2 * pad,
            device=x.device,
            dtype=x.dtype,
        )[:height]
        rows = arange.view(1, height).expand(height, height)
        base = (
            torch.stack([rows, rows.T], dim=-1)
            .unsqueeze(0)
            .expand(n, -1, -1, -1)
        )
        shift = torch.randint(
            0, 2 * pad + 1, (n, 1, 1, 2), device=x.device
        ).to(dtype=x.dtype)
        shift = shift * (2.0 / (height + 2 * pad))
        return F.grid_sample(
            x, base + shift, padding_mode='zeros', align_corners=False
        )


class PixelEncoder(nn.Module):
    """DrQ-v2 convolutional encoder; input pixels are uint8 or 0..255 floats."""

    def __init__(self, obs_shape: tuple[int, int, int]):
        super().__init__()
        channels, height, width = obs_shape
        if height != width or height < 36:
            raise ValueError(
                'DrQ-v2 encoder needs square images of at least 36px'
            )
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.repr_dim = self.convnet(
                torch.zeros(1, channels, height, width)
            ).shape[-1]
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, math.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.convnet(obs.float() / 255.0 - 0.5)


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
        obs_shape: tuple[int, int, int],
        action_shape: tuple[int, ...],
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
        self.nstep = int(nstep)
        if self.nstep < 1:
            raise ValueError(f'nstep must be >= 1, got {self.nstep}')

        self.encoder = PixelEncoder(obs_shape).to(self.device)
        repr_dim = self.encoder.repr_dim
        self.actor = Actor(
            repr_dim, self.action_dim, feature_dim, hidden_dim
        ).to(self.device)
        self.critic = Critic(
            repr_dim, self.action_dim, feature_dim, hidden_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.representation = build_representation(
            representation_loss,
            feature_dim=repr_dim,
            action_dim=self.action_dim,
            horizon=self.representation_horizon,
            projection_dim=representation_dim,
            hidden_dim=representation_hidden_dim,
            temperature=representation_temperature,
            sigreg_weight=representation_sigreg_weight,
            lewm_hidden_dim=representation_lewm_hidden_dim,
        )
        if self.representation is not None:
            self.representation = self.representation.to(self.device)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.representation_opt = (
            torch.optim.Adam(
                list(self.encoder.parameters())
                + list(self.representation.parameters()),
                lr=lr,
            )
            if self.representation is not None
            else None
        )
        self.aug = RandomShiftsAug(augmentation_pad).to(self.device)

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
        if (
            obs.ndim >= 3
            and obs.shape[-1] > 4
            and obs.shape[-1] % 3 == 0
            and obs.shape[-2] == obs.shape[-3]
        ) or obs.shape[-1] in (1, 3, 4):
            return obs.movedim(-1, -3).contiguous()
        return obs

    def act(self, obs, step: int, eval_mode: bool = False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device)
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)
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
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        """Run one critic/actor update and optional representation update."""
        obs_seq = self._as_nchw(batch['pixels'].to(self.device))
        actions = batch['action'].to(self.device).float()
        rewards = batch['reward'].to(self.device).float()
        discounts = batch['discount'].to(self.device).float()
        # The final frame has no outgoing action/reward. Collector writes a
        # harmless sentinel there to keep every episode column aligned.
        obs = self.aug(obs_seq[:, 0].float())
        next_obs = self.aug(obs_seq[:, self.nstep].float())
        obs_feature = self.encoder(obs)
        reward_discount = torch.ones_like(rewards[:, 0:1])
        nstep_reward = torch.zeros_like(rewards[:, 0:1])
        for index in range(self.nstep):
            nstep_reward = nstep_reward + reward_discount * rewards[:, index : index + 1]
            reward_discount = reward_discount * self.discount * discounts[
                :, index : index + 1
            ]
        with torch.no_grad():
            next_feature = self.encoder(next_obs)
            mean, std = self.actor(next_feature, self._stddev(step))
            next_action = self._sample_action(mean, std, self.stddev_clip)
            target_q1, target_q2 = self.critic_target(
                next_feature, next_action
            )
            target_q = nstep_reward + reward_discount * torch.minimum(
                target_q1, target_q2
            )

        q1, q2 = self.critic(obs_feature, actions[:, 0])
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        actor_feature = obs_feature.detach()
        mean, std = self.actor(actor_feature, self._stddev(step))
        actor_action = self._sample_action(mean, std, self.stddev_clip)
        actor_q1, actor_q2 = self.critic(actor_feature, actor_action)
        actor_loss = -torch.minimum(actor_q1, actor_q2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

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

        if self.representation is not None:
            h = self.representation_horizon
            temporal = self.aug(obs_seq[:, : h + 1].float().flatten(0, 1))
            temporal = self.encoder(temporal).view(obs_seq.shape[0], h + 1, -1)
            if self.representation_opt is None:
                raise RuntimeError('representation optimizer missing')
            self.representation_opt.zero_grad(set_to_none=True)
            if isinstance(self.representation, PLDMRepresentation):
                rep_metrics = self.representation(temporal)
            else:
                with torch.no_grad():
                    target_obs = self.aug(
                        obs_seq[:, : h + 1].float().flatten(0, 1)
                    )
                    target_temporal = self.encoder(target_obs).view(
                        obs_seq.shape[0], h + 1, -1
                    )
                if isinstance(self.representation, LeWMTemporalRepresentation):
                    rep_metrics = self.representation(temporal, actions)
                elif isinstance(self.representation, JEPATemporalRepresentation):
                    rep_metrics = self.representation(
                        temporal, target_temporal
                    )
                else:
                    rep_metrics = self.representation(
                        temporal, actions[:, :h], target_temporal
                    )
            rep_loss = rep_metrics['loss'] * self.representation_weight
            rep_loss.backward()
            self.representation_opt.step()
            for name, value in rep_metrics.items():
                if name not in {
                    'loss',
                    'logits',
                    'labels',
                } and torch.is_tensor(value):
                    metrics[f'representation/{name}'] = float(value.detach())
            metrics['representation/loss'] = float(
                rep_metrics['loss'].detach()
            )

        return metrics

    def state_dict(self) -> dict:
        return {
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'representation': (
                self.representation.state_dict()
                if self.representation is not None
                else None
            ),
            'encoder_opt': self.encoder_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'representation_opt': (
                self.representation_opt.state_dict()
                if self.representation_opt is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state['encoder'])
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        if self.representation is not None:
            if state['representation'] is None:
                raise ValueError('Checkpoint has no representation module')
            self.representation.load_state_dict(state['representation'])
        self.encoder_opt.load_state_dict(state['encoder_opt'])
        self.actor_opt.load_state_dict(state['actor_opt'])
        self.critic_opt.load_state_dict(state['critic_opt'])
        if (
            self.representation_opt is not None
            and state['representation_opt'] is not None
        ):
            self.representation_opt.load_state_dict(
                state['representation_opt']
            )
