# skipDHSRNN/SNN_layers/kuramoto.py
import torch
import torch.nn as nn
import math


class SakaguchiKuramoto(nn.Module):

    def __init__(self, N, K=1.0, dt=1.0, device='cuda'):
        super().__init__()
        self.N = N
        self.K = K
        self.dt = dt
        self.device = device

        #  ω_i
        self.omega = nn.Parameter(torch.zeros(N))

        # α_ij （traveling wave ）
        self.alpha = nn.Parameter(torch.zeros(N, N))

        #  κ_i
        self.kappa = nn.Parameter(torch.ones(N))

    def forward(self, theta_prev, gamma_t):

        B, N = theta_prev.shape
        assert N == self.N

        # θ_i, θ_j 
        theta_i = theta_prev.unsqueeze(2)       # [B, N, 1]
        theta_j = theta_prev.unsqueeze(1)       # [B, 1, N]

        #  (θ_j - θ_i - α_ij)
        phase_diff = theta_j - theta_i - self.alpha   # broadcast 到 [B, N, N]

        # coupling 
        coupling = (self.K / self.N) * torch.sum(torch.sin(phase_diff), dim=-1)  # [B, N]

       
        theta_dot = self.omega + coupling + self.kappa * (gamma_t - theta_prev)  # [B, N]


        theta_new = theta_prev + self.dt * theta_dot
        return theta_new
