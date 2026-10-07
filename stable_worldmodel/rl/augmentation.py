"""Image augmentations used by online pixel RL agents."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RandomShiftsAug(nn.Module):
    """Per-image random translation using replicate-padded grid sampling."""

    def __init__(self, pad: int = 4):
        super().__init__()
        if pad < 0:
            raise ValueError(f'pad must be >= 0, got {pad}')
        self.pad = int(pad)

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
