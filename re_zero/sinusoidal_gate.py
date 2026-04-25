from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalGate(nn.Module):
    """
    Bridge from oscillator states to SNN input modulation.

    The gate reads a delayed theta state, projects each oscillator vector into
    R^2, converts that 2D vector into a scalar phase angle, and returns a gate
    in [0, 1].
    """

    def __init__(self, delay: int = 2) -> None:
        super().__init__()
        self.delay = delay
        self.phase_projection = nn.LazyLinear(2)

    def forward(self, theta_delayed: torch.Tensor, gamma: torch.Tensor | None = None) -> torch.Tensor:
        return self.sinusoidal_gating_function(theta_delayed, gamma)

    def sinusoidal_gating(
        self,
        theta_history: List[torch.Tensor],
        theta_current: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the sinusoidal gate from delayed theta history."""
        return self.build_gate_from_history(theta_history, theta_current, gamma)

    def sinusoidal_gating_function(
        self,
        theta_delayed: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert delayed vector states into a gate in the range [0, 1].

        `gamma` is accepted so this module can later use both theta and gamma
        without changing the outer model API.
        """
        theta_phase_plane = self.phase_projection(theta_delayed)
        theta_phase_plane = F.normalize(theta_phase_plane, dim=-1, eps=1e-6)
        theta_phase = torch.atan2(
            theta_phase_plane[..., 1:2],
            theta_phase_plane[..., 0:1],
        )
        return 0.5 * (1.0 + torch.sin(theta_phase))

    def build_gate_from_history(
        self,
        theta_history: List[torch.Tensor],
        theta_current: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the sinusoidal gate from delayed theta history."""
        if theta_history:
            delayed_idx = max(0, len(theta_history) - int(self.delay))
            theta_delayed = theta_history[delayed_idx]
        else:
            theta_delayed = theta_current
        return self.sinusoidal_gating_function(theta_delayed, gamma)
