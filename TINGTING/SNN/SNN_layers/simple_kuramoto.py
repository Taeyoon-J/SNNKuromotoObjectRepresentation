import torch
import torch.nn as nn

class SimpleKuramoto(nn.Module):
    """
    theta_new = theta_prev + dt * (omega + kappa*(gamma - theta_prev))
    """
    def __init__(self, dim, dt=1.0, device='cuda'):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.device = device

        self.omega = nn.Parameter(torch.zeros(dim))
        self.kappa = nn.Parameter(torch.ones(dim))

    def forward(self, theta_prev, gamma_t):
        """
        theta_prev: [B, dim]
        gamma_t:    [B, dim]
        """
        theta_dot = self.omega + self.kappa * (gamma_t - theta_prev)
        return theta_prev + self.dt * theta_dot
