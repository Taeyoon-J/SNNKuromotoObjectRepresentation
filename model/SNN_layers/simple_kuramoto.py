from __future__ import annotations

# This file holds the simplest possible oscillator update.
# It is useful for debugging because it removes pairwise coupling and only keeps
# the self-dynamics plus external drive.

import torch
import torch.nn as nn


class SimpleKuramoto(nn.Module):
    """
    Minimal Kuramoto-style update:

    theta(t) = theta(t-1) + dt * [omega + kappa * (gamma(t) - theta(t-1))]

    There is no oscillator-to-oscillator coupling term here.
    """

    def __init__(self, dim: int, dt: float = 1.0) -> None:
        super().__init__()
        # Number of oscillator channels per sample.
        self.dim = dim
        # Integration step size.
        self.dt = dt
        # Learnable natural frequency / drift term.
        self.omega = nn.Parameter(torch.zeros(dim))
        # Learnable strength for the external control signal gamma.
        self.kappa = nn.Parameter(torch.ones(dim))

    def forward(self, theta_prev: torch.Tensor, gamma_t: torch.Tensor) -> torch.Tensor:
        """
        Advance one step of the simple dynamics.

        Args:
            theta_prev: Previous oscillator state, shape [B, dim]
            gamma_t: External drive at the current time step, shape [B, dim]
        """
        # First compute the time derivative.
        theta_dot = self.omega.view(1, self.dim) + self.kappa.view(1, self.dim) * (gamma_t - theta_prev)
        # Then take one Euler integration step.
        return theta_prev + self.dt * theta_dot
