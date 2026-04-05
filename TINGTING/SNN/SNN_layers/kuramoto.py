# kuramoto.py
import torch
import torch.nn as nn
import math

class SakaguchiKuramoto(nn.Module):
    """
    Compute Kuramoto phases θ(t) with time delay d
    and phase shift α_ij.
    """

    def __init__(self, N, K=1.0, dt=1.0, device='cuda'):
        super().__init__()
        self.N = N
        self.K = K
        self.dt = dt
        self.device = device

        # Learnable natural frequencies ω_i
        self.omega = nn.Parameter(torch.zeros(N))

        # Learnable phase offsets α_ij
        self.alpha = nn.Parameter(torch.zeros(N, N))

        # Optional learnable stiffness κ_i (control pattern strength)
        self.kappa = nn.Parameter(torch.ones(N))

    def forward(self, theta_prev, gamma):
        """
        theta_prev: [B, N] phases at t-1
        gamma: [B, N] control pattern γ_i(t)
        """
        B = theta_prev.size(0)

        theta_i = theta_prev.unsqueeze(2)        # [B, N, 1]
        theta_j = theta_prev.unsqueeze(1)        # [B, 1, N]

        phase_diff = theta_j - theta_i - self.alpha  # [B, N, N]
        coupling = (self.K / self.N) * torch.sum(torch.sin(phase_diff), dim=-1)  # [B, N]

        theta_dot = self.omega + coupling + 0.1 * self.kappa * (gamma - theta_prev)

        theta_new = (theta_prev + self.dt * theta_dot) % (2 * math.pi)
        return theta_new
