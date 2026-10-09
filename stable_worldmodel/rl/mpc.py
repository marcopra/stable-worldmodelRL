"""Value prediction and action-sequence planning over the selected WM."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .representation import (
    InfoNCERepresentation,
    LeWMTemporalRepresentation,
    RewardPredictionRepresentation,
)

PolicyAction = Callable[
    [torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]
]


class ValuePredictionHead(nn.Module):
    """TD-MPC2-style twin two-hot Q heads on WM latents and actions."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
    ) -> None:
        super().__init__()
        if num_bins < 2 or vmax <= vmin:
            raise ValueError('MPC value-bin configuration is invalid')
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.register_buffer('bin_values', torch.linspace(vmin, vmax, num_bins))

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_bins),
            )

        self.q1 = make_head()
        self.q2 = make_head()

    def logits(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_action = torch.cat((latent, action), dim=-1)
        return self.q1(state_action), self.q2(state_action)

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both Q estimates after inverse symlog transformation."""
        q1_logits, q2_logits = self.logits(latent, action)
        q1_symlog = (q1_logits.softmax(dim=-1) * self.bin_values).sum(
            dim=-1, keepdim=True
        )
        q2_symlog = (q2_logits.softmax(dim=-1) * self.bin_values).sum(
            dim=-1, keepdim=True
        )
        return (
            torch.sign(q1_symlog) * torch.expm1(q1_symlog.abs()),
            torch.sign(q2_symlog) * torch.expm1(q2_symlog.abs()),
        )

    def loss(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Train both Q heads with soft two-hot TD-target cross entropy."""
        target = target.reshape(-1, 1)
        symlog_target = torch.sign(target) * torch.log1p(target.abs())
        symlog_target = symlog_target.clamp(self.vmin, self.vmax)
        bin_size = (self.vmax - self.vmin) / (self.num_bins - 1)
        position = (symlog_target - self.vmin) / bin_size
        lower = position.floor().long().clamp(0, self.num_bins - 2)
        upper_weight = position - lower
        target_distribution = target.new_zeros(
            target.shape[0], self.num_bins
        )
        target_distribution.scatter_(1, lower, 1.0 - upper_weight)
        target_distribution.scatter_(1, lower + 1, upper_weight)
        q1_logits, q2_logits = self.logits(latent, action)
        q1_loss = -(
            target_distribution * F.log_softmax(q1_logits, dim=-1)
        ).sum(dim=-1).mean()
        q2_loss = -(
            target_distribution * F.log_softmax(q2_logits, dim=-1)
        ).sum(dim=-1).mean()
        return 0.5 * (q1_loss + q2_loss)


class LatentMPCPlanner:
    """Weighted-elite planner that queries the already-trained WM losses."""

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int = 5,
        num_samples: int = 512,
        num_elites: int = 64,
        iterations: int = 6,
        num_pi_trajs: int = 32,
        min_std: float = 0.05,
        max_std: float = 0.5,
        temperature: float = 0.5,
        exploration_noise: float = 1.0,
    ) -> None:
        if horizon < 1:
            raise ValueError(f'MPC horizon must be positive, got {horizon}')
        if num_samples < 2:
            raise ValueError('MPC needs at least two action samples')
        if not 1 <= num_elites <= num_samples:
            raise ValueError('MPC num_elites must be in [1, num_samples]')
        if not 1 <= iterations:
            raise ValueError('MPC iterations must be positive')
        if not 0 <= num_pi_trajs < num_samples:
            raise ValueError('MPC num_pi_trajs must be in [0, num_samples)')
        if min_std < 0 or max_std < min_std:
            raise ValueError('MPC standard deviation bounds are invalid')
        if temperature <= 0:
            raise ValueError('MPC temperature must be positive')
        if exploration_noise < 0:
            raise ValueError('MPC exploration_noise must be non-negative')

        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.iterations = int(iterations)
        self.num_pi_trajs = int(num_pi_trajs)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.temperature = float(temperature)
        self.exploration_noise = float(exploration_noise)
        self._previous_mean: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear the shifted plan at an episode boundary."""
        self._previous_mean = None

    @torch.no_grad()
    def plan(
        self,
        feature: torch.Tensor,
        world_model: InfoNCERepresentation | LeWMTemporalRepresentation,
        reward_model: RewardPredictionRepresentation,
        policy_action: PolicyAction,
        value: ValuePredictionHead | None,
        *,
        discount: float,
        eval_mode: bool,
    ) -> torch.Tensor:
        """Optimize a planned action clip and return its first action."""
        if feature.ndim != 2 or feature.shape[0] != 1:
            raise ValueError('MPC currently plans one environment at a time')

        device, dtype = feature.device, feature.dtype
        action_horizon = self.horizon + int(value is not None)
        mean = torch.zeros(
            action_horizon, self.action_dim, device=device, dtype=dtype
        )
        if self._previous_mean is not None:
            mean[:-1] = self._previous_mean[1:]
        std = torch.full_like(mean, self.max_std)

        policy_candidates = self._policy_trajectories(
            feature, policy_action, action_horizon
        )
        random_count = self.num_samples - self.num_pi_trajs
        initial_feature = feature.expand(self.num_samples, -1)
        elite_actions = None
        elite_weights = None
        elite_values = None

        for _ in range(self.iterations):
            noise = torch.randn(
                random_count,
                action_horizon,
                self.action_dim,
                device=device,
                dtype=dtype,
            )
            sampled = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
                -1.0, 1.0
            )
            candidates = (
                torch.cat((policy_candidates, sampled), dim=0)
                if policy_candidates is not None
                else sampled
            )
            world_model_training = world_model.training
            reward_model_training = reward_model.training
            world_model.eval()
            reward_model.eval()
            try:
                predicted_latent = world_model.predict_mpc_latent(
                    initial_feature, candidates[:, : self.horizon]
                )
                predicted_return = reward_model.predict_return(
                    initial_feature, candidates[:, : self.horizon]
                )
            finally:
                world_model.train(world_model_training)
                reward_model.train(reward_model_training)
            if value is None:
                values = predicted_return.squeeze(-1)
            else:
                q1, q2 = value(predicted_latent, candidates[:, self.horizon])
                terminal_q = 0.5 * (q1 + q2)
                values = (
                    predicted_return + discount**self.horizon * terminal_q
                ).squeeze(-1)

            elite_values, indices = values.topk(self.num_elites)
            elite_actions = candidates[indices]
            elite_weights = torch.softmax(
                self.temperature * (elite_values - elite_values.max()), dim=0
            )
            mean = torch.einsum('n,nha->ha', elite_weights, elite_actions)
            variance = torch.einsum(
                'n,nha->ha',
                elite_weights,
                (elite_actions - mean.unsqueeze(0)).square(),
            )
            std = variance.sqrt().clamp(self.min_std, self.max_std)

        assert elite_actions is not None and elite_weights is not None
        if eval_mode:
            selected = elite_values.argmax()
        else:
            selected = torch.multinomial(elite_weights, num_samples=1).squeeze(0)
        action = elite_actions[selected, 0]
        if not eval_mode and self.exploration_noise:
            action = action + self.exploration_noise * std[0] * torch.randn_like(
                action
            )

        self._previous_mean = mean.detach().clone()
        return action.clamp(-1.0, 1.0)

    def _policy_trajectories(
        self,
        feature: torch.Tensor,
        policy_action: PolicyAction,
        action_horizon: int,
    ) -> torch.Tensor | None:
        if self.num_pi_trajs == 0:
            return None
        batch_feature = feature.expand(self.num_pi_trajs, -1)
        actions = []
        for _ in range(action_horizon):
            action, _ = policy_action(batch_feature)
            actions.append(action.clamp(-1.0, 1.0))
        return torch.stack(actions, dim=1)


__all__ = ['LatentMPCPlanner', 'ValuePredictionHead']
