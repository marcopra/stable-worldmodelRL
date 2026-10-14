"""Value prediction and action-sequence planning over the selected WM."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .representation import (
    AuxiliaryLosses,
    ConsistencyRepresentation,
    InfoNCERepresentation,
    LeWMTemporalRepresentation,
    RewardPredictionRepresentation,
)

PolicyAction = Callable[
    [torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]
]
TerminalValue = Callable[[torch.Tensor], torch.Tensor]
ActionLogProbability = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def validate_mpc_config(config: dict) -> None:
    """Validate value-learning settings shared by both base algorithms."""
    value_coef = float(config.get('value_coef', 0.1))
    if not math.isfinite(value_coef) or value_coef < 0:
        raise ValueError('mpc.value_coef must be finite and non-negative')
    rho = float(config.get('rho', 0.5))
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError('mpc.rho must be finite and in [0, 1]')
    start_steps = config.get('start_steps', 0)
    if not isinstance(start_steps, int) or start_steps < 0:
        raise ValueError('mpc.start_steps must be a non-negative integer')


def validate_mpc_models(
    config: dict,
    world_model: (
        InfoNCERepresentation
        | LeWMTemporalRepresentation
        | ConsistencyRepresentation
        | None
    ),
    reward_prediction: RewardPredictionRepresentation | None,
    auxiliary_losses: AuxiliaryLosses | None,
    world_model_weight: float,
) -> None:
    """Validate the learned predictors required by latent planning."""
    if not config.get('enabled', False):
        return
    if not isinstance(
        world_model,
        (
            InfoNCERepresentation,
            LeWMTemporalRepresentation,
            ConsistencyRepresentation,
        ),
    ):
        raise ValueError(
            'MPC requires an action-conditioned one-step WM '
            '(wm.target=infonce, lewm, or consistency)'
        )
    if not isinstance(reward_prediction, RewardPredictionRepresentation):
        raise ValueError(
            'MPC requires the existing auxiliary reward-prediction '
            'loss to be enabled'
        )
    if (
        auxiliary_losses is None
        or auxiliary_losses.reward_prediction_weight <= 0
    ):
        raise ValueError('MPC requires auxiliary.reward_prediction.weight > 0')
    if world_model_weight <= 0:
        raise ValueError('MPC requires auxiliary.wm.weight > 0')
    if reward_prediction.mode != 'per_step':
        raise ValueError(
            "MPC requires auxiliary.reward_prediction.mode='per_step'"
        )
    if world_model.horizon != 1:
        raise ValueError(
            'MPC world-model prediction horizon must be 1; set '
            'auxiliary.wm.horizon=1 and use auxiliary.sequence_length '
            'for replay rollouts'
        )


class ValuePredictionHead(nn.Module):
    """Twin reward-return Q heads using TD-MPC2's two-hot parameterization."""

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
        if (
            num_bins < 2
            or not math.isfinite(vmin)
            or not math.isfinite(vmax)
            or vmax <= vmin
        ):
            raise ValueError('MPC value-bin configuration is invalid')
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.register_buffer(
            'bin_values', torch.linspace(vmin, vmax, num_bins)
        )

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
        target_distribution = target.new_zeros(target.shape[0], self.num_bins)
        target_distribution.scatter_(1, lower, 1.0 - upper_weight)
        target_distribution.scatter_(1, lower + 1, upper_weight)
        q1_logits, q2_logits = self.logits(latent, action)
        q1_loss = (
            -(target_distribution * F.log_softmax(q1_logits, dim=-1))
            .sum(dim=-1)
            .mean()
        )
        q2_loss = (
            -(target_distribution * F.log_softmax(q2_logits, dim=-1))
            .sum(dim=-1)
            .mean()
        )
        return 0.5 * (q1_loss + q2_loss)


class LatentMPCPlanner:
    """Weighted-elite action search over a trained latent transition model."""

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
        for name, count in (
            ('action_dim', action_dim),
            ('horizon', horizon),
            ('num_samples', num_samples),
            ('num_elites', num_elites),
            ('iterations', iterations),
            ('num_pi_trajs', num_pi_trajs),
        ):
            if not isinstance(count, int):
                raise TypeError(f'MPC {name} must be an integer, got {count}')
        if action_dim < 1:
            raise ValueError('MPC action_dim must be positive')
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
        if (
            not math.isfinite(min_std)
            or not math.isfinite(max_std)
            or min_std < 0
            or max_std <= 0
            or max_std < min_std
        ):
            raise ValueError('MPC standard deviation bounds are invalid')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('MPC temperature must be positive')
        if not math.isfinite(exploration_noise) or exploration_noise < 0:
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
        latent: torch.Tensor,
        world_model: (
            InfoNCERepresentation
            | LeWMTemporalRepresentation
            | ConsistencyRepresentation
        ),
        reward_model: RewardPredictionRepresentation,
        policy_action: PolicyAction,
        terminal_value: TerminalValue,
        *,
        discount: float,
        eval_mode: bool,
        entropy_coefficient: float = 0.0,
        action_log_probability: ActionLogProbability | None = None,
    ) -> torch.Tensor:
        """Optimize a planned action clip and return its first action."""
        if latent.ndim != 2 or latent.shape[0] != 1:
            raise ValueError('MPC currently plans one environment at a time')
        if latent.shape[-1] != world_model.mpc_latent_dim:
            raise ValueError(
                'MPC planner expects the selected WM latent dimension '
                f'{world_model.mpc_latent_dim}, got {latent.shape[-1]}'
            )
        if not math.isfinite(discount) or not 0 <= discount < 1:
            raise ValueError('MPC discount must be finite and in [0, 1)')
        if not math.isfinite(entropy_coefficient) or entropy_coefficient < 0:
            raise ValueError('MPC entropy coefficient must be non-negative')
        if entropy_coefficient and action_log_probability is None:
            raise ValueError(
                'MPC entropy scoring requires an action log-probability'
            )

        device, dtype = latent.device, latent.dtype
        action_horizon = self.horizon
        mean = torch.zeros(
            action_horizon, self.action_dim, device=device, dtype=dtype
        )
        if self._previous_mean is not None:
            mean[:-1] = self._previous_mean[1:]
        std = torch.full_like(mean, self.max_std)

        policy_candidates = self._policy_trajectories(
            latent, policy_action, action_horizon
        )
        random_count = self.num_samples - self.num_pi_trajs
        initial_latent = latent.expand(self.num_samples, -1)
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
                predicted_latent = initial_latent
                predicted_return = latent.new_zeros(self.num_samples, 1)
                for index in range(self.horizon):
                    action = candidates[:, index]
                    predicted_reward = reward_model.predict_step(
                        predicted_latent, action
                    )
                    if predicted_reward.shape != predicted_return.shape:
                        raise ValueError(
                            'MPC reward model must return one scalar per '
                            'candidate, got '
                            f'{tuple(predicted_reward.shape)}'
                        )
                    if action_log_probability is not None:
                        log_probability = action_log_probability(
                            predicted_latent, action
                        )
                        if log_probability.shape != predicted_reward.shape:
                            raise ValueError(
                                'MPC action log-probability must have shape '
                                f'{tuple(predicted_reward.shape)}, got '
                                f'{tuple(log_probability.shape)}'
                            )
                        predicted_reward = predicted_reward - (
                            entropy_coefficient * log_probability
                        )
                    predicted_return = (
                        predicted_return + discount**index * predicted_reward
                    )
                    predicted_latent = world_model.predict_mpc_step(
                        predicted_latent, action
                    )
                terminal = terminal_value(predicted_latent)
                if terminal.shape != predicted_return.shape:
                    raise ValueError(
                        'MPC terminal value must have shape '
                        f'{tuple(predicted_return.shape)}, got '
                        f'{tuple(terminal.shape)}'
                    )
            finally:
                world_model.train(world_model_training)
                reward_model.train(reward_model_training)
            values = (
                predicted_return + discount**self.horizon * terminal
            ).squeeze(-1)

            if not torch.isfinite(values).all():
                raise FloatingPointError('MPC predicted non-finite returns')
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

        assert (
            elite_actions is not None
            and elite_weights is not None
            and elite_values is not None
        )
        if eval_mode:
            selected = elite_values.argmax()
        else:
            selected = torch.multinomial(elite_weights, num_samples=1).squeeze(
                0
            )
        action = elite_actions[selected, 0]
        if not eval_mode and self.exploration_noise:
            action = action + self.exploration_noise * std[
                0
            ] * torch.randn_like(action)

        self._previous_mean = mean.detach().clone()
        return action.clamp(-1.0, 1.0)

    def _policy_trajectories(
        self,
        latent: torch.Tensor,
        policy_action: PolicyAction,
        action_horizon: int,
    ) -> torch.Tensor | None:
        # These are open-loop proposals at the current latent state.
        if self.num_pi_trajs == 0:
            return None
        batch_latent = latent.expand(self.num_pi_trajs, -1)
        actions = []
        for _ in range(action_horizon):
            action, _ = policy_action(batch_latent)
            actions.append(action.clamp(-1.0, 1.0))
        return torch.stack(actions, dim=1)


__all__ = ['LatentMPCPlanner', 'ValuePredictionHead']
