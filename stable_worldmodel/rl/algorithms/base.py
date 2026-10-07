"""Structural interface shared by online model-free RL agents."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class OnlineAgent(Protocol):
    """Contract consumed by the shared online training runner."""

    def act(
        self, observation, step: int = 0, eval_mode: bool = False
    ) -> np.ndarray: ...

    def update(
        self, batch: dict[str, torch.Tensor], step: int = 0
    ) -> dict[str, float]: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...
