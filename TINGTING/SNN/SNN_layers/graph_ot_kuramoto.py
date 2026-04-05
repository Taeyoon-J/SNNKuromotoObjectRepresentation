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
        # α_ij
        self.alpha = nn.Parameter(torch.zeros(N, N))
        #  κ_i
        self.kappa = nn.Parameter(torch.ones(N))

    def forward(self, theta_prev, gamma):
        """
        theta_prev: [B, N]
        gamma:      [B, N]
        """
        B, N = theta_prev.shape
        assert N == self.N

        theta_i = theta_prev.unsqueeze(2)      # [B, N, 1]
        theta_j = theta_prev.unsqueeze(1)      # [B, 1, N]

        phase_diff = theta_j - theta_i - self.alpha  # [B, N, N]
        coupling = (self.K / self.N) * torch.sum(torch.sin(phase_diff), dim=-1)  # [B, N]

        kappa = torch.relu(self.kappa)  
        theta_dot = self.omega + coupling + kappa * (gamma - theta_prev)

        theta_new = theta_prev + self.dt * theta_dot
        return theta_new


class GraphOTKuramoto(nn.Module):
    """
    Graph-aware Kuramoto + entropic OT:
    """

    def __init__(self, N, K=1.0, dt=1.0,
                 eps_ot=0.1, alpha_scale=1.0,
                 sinkhorn_iters=10,
                 device='cuda'):
        super().__init__()
        self.N = N               
        self.K = K
        self.dt = dt
        self.eps_ot = eps_ot
        self.alpha_scale = alpha_scale
        self.sinkhorn_iters = sinkhorn_iters
        self.device = device

        #  ω_i
        self.omega = nn.Parameter(torch.zeros(N))

        # α_ij
        self.alpha = nn.Parameter(torch.zeros(N, N))

        #  κ_i
        self.log_kappa = nn.Parameter(torch.zeros(N))  # softplus→正

    def _reduce_adj(self, A_full):

        B, N_full, _ = A_full.shape
        if N_full == self.N:
            return A_full

        device = A_full.device
        N = self.N


        edges = torch.linspace(0, N_full, steps=N + 1, device=device).long()
        edges[-1] = N_full

        A_latent = torch.zeros(B, N, N, device=device)

        for i in range(N):
            i_start, i_end = edges[i].item(), edges[i + 1].item()
            if i_end <= i_start:
                i_end = min(i_start + 1, N_full)
            idx_i = slice(i_start, i_end)

            for j in range(N):
                j_start, j_end = edges[j].item(), edges[j + 1].item()
                if j_end <= j_start:
                    j_end = min(j_start + 1, N_full)
                idx_j = slice(j_start, j_end)

                block = A_full[:, idx_i, :][:, :, idx_j]   # [B, ni, nj]
                A_latent[:, i, j] = block.mean(dim=(-1, -2))

        return A_latent

    def _build_cost(self, A_latent):

        eps = 1e-8
        W = torch.relu(A_latent)  # [B,N,N]
        row_sum = W.sum(dim=-1, keepdim=True)  # [B,N,1]
        W = torch.where(row_sum > eps, W / (row_sum + eps),
                        torch.full_like(W, 1.0 / self.N))
        C = -torch.log(W + eps)   # 大边权 → 小 cost
        return C

    # ---------- Sinkhorn OT ----------
    def _sinkhorn(self, p, q, C):

        eps = 1e-8
        B, N, _ = C.shape

        # K = exp(-C / ε)
        K = torch.exp(-C / self.eps_ot)  # [B,N,N]

        #  u,v
        u = torch.ones(B, N, device=C.device) / N
        v = torch.ones(B, N, device=C.device) / N

        for _ in range(self.sinkhorn_iters):
            # u <- p / (K v)
            Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)      # [B,N]
            u = p / (Kv + eps)

            # v <- q / (K^T u)
            KT_u = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)  # [B,N]
            v = q / (KT_u + eps)

        # π = diag(u) K diag(v)
        π = u.unsqueeze(-1) * K * v.unsqueeze(-2)               # [B,N,N]
        π = π / (π.sum(dim=(-1, -2), keepdim=True) + eps)
        return π

    def forward(self, theta_prev, gamma, A_full):

        B, N = theta_prev.shape
        assert N == self.N, f"theta dim {N} != N={self.N}"

        if A_full.dim() == 2:
            A_full = A_full.unsqueeze(0)  # [1, N_full, N_full]
        A_full = A_full.to(theta_prev.device)
        A_latent = self._reduce_adj(A_full)              # [B, N, N]

        C = self._build_cost(A_latent)                   # [B, N, N]

        p = torch.softmax(gamma, dim=-1)                 # [B,N]
        q = torch.full_like(p, 1.0 / self.N)             # uniform 目标，简单稳定

        pi = self._sinkhorn(p, q, C)                     # [B,N,N]

        theta_i = theta_prev.unsqueeze(2)                # [B,N,1]
        theta_j = theta_prev.unsqueeze(1)                # [B,1,N]

        phase_diff = theta_j - theta_i - self.alpha_scale * self.alpha  # [B,N,N]

        coupling = self.K * torch.sum(pi * torch.sin(phase_diff), dim=-1)  # [B,N]

        kappa = torch.nn.functional.softplus(self.log_kappa)               # [N] >= 0
        theta_dot = self.omega + coupling + kappa * (gamma - theta_prev)   # [B,N]

        theta_new = theta_prev + self.dt * theta_dot
        return theta_new
