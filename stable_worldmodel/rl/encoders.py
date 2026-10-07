"""Observation encoders shared by model-free RL agents."""

from __future__ import annotations

import math

import torch
from torch import nn


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class StateEncoder(nn.Module):
    """MLP encoder for flat or flattened numeric proprioceptive states."""

    def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.repr_dim = int(feature_dim)
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
        self.apply(_init_linear)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state.float().flatten(start_dim=1))


class PixelEncoder(nn.Module):
    """DrQ-style convolutional encoder for uint8 or 0..255 pixel batches."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        feature_dim: int | None = None,
    ):
        super().__init__()
        channels, height, width = obs_shape
        if height != width or height < 36:
            raise ValueError(
                'Pixel encoder needs square images of at least 36px'
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
            flat_dim = self.convnet(
                torch.zeros(1, channels, height, width)
            ).shape[-1]
        self.apply(self._init_conv)
        if feature_dim is None:
            self.projection = nn.Identity()
            self.repr_dim = int(flat_dim)
        else:
            self.projection = nn.Sequential(
                nn.Linear(flat_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh(),
            )
            self.repr_dim = int(feature_dim)

    @staticmethod
    def _init_conv(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, math.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # Online collection stores RGB observations as uint8 (0..255). Avoid
        # inspecting tensor values here: amax() would synchronize every CUDA
        # forward pass with the host.
        pixels = pixels.float().div(255.0)
        return self.projection(self.convnet(pixels - 0.5))


def pixel_batch_to_nchw(observations: torch.Tensor) -> torch.Tensor:
    """Convert HWC pixel batches (or sequences) to the encoder's NCHW layout."""
    if observations.ndim < 3:
        raise ValueError('Pixel observations must include height and width')
    if observations.shape[-1] in (1, 3, 4) or (
        observations.shape[-1] > 4 and observations.shape[-1] % 3 == 0
    ):
        return observations.movedim(-1, -3).contiguous()
    return observations


def make_observation_encoder(
    mode: str,
    obs_shape: tuple[int, ...],
    feature_dim: int,
    hidden_dim: int,
    *,
    pixel_feature_dim: int | None = None,
) -> nn.Module:
    """Construct a state or pixel encoder from the configured observation mode."""
    if mode == 'state':
        size = math.prod(obs_shape)
        return StateEncoder(size, feature_dim, hidden_dim)
    if mode == 'pixels':
        if len(obs_shape) != 3:
            raise ValueError(f'Pixel obs_shape must be CHW, got {obs_shape!r}')
        return PixelEncoder(obs_shape, feature_dim=pixel_feature_dim)
    raise ValueError(
        f"Unknown observation mode {mode!r}; choose 'state' or 'pixels'"
    )
