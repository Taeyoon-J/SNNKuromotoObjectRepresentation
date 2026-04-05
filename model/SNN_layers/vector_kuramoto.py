from __future__ import annotations

# This file contains the multi-dimensional Kuramoto dynamics used by the model.
# Unlike the simple version, this one lets oscillators influence each other.

import torch
import torch.nn as nn


class VectorKuramoto(nn.Module):
    """
    Vector-valued Kuramoto block.

    Each spatial location has an oscillator vector of length `osc_dim`.
    Pairwise interactions are weighted by an affinity matrix and shifted by a
    phase-lag matrix `alpha_t`.
    """

    def __init__(self, num_nodes: int, osc_dim: int = 4, coupling: float = 1.0, dt: float = 1.0) -> None:
        super().__init__()
        # Number of flattened spatial locations, e.g. H * W.
        self.num_nodes = num_nodes
        # Number of oscillator channels carried by each location.
        self.osc_dim = osc_dim
        # Global synchronization strength.
        self.coupling = coupling
        # Euler integration step size.
        self.dt = dt

        # Natural frequency for each node and oscillator dimension.
        self.omega = nn.Parameter(torch.zeros(num_nodes, osc_dim))
        # How strongly each node follows the external drive gamma.
        self.kappa = nn.Parameter(torch.ones(num_nodes, osc_dim))
        # Additional learned phase offset between node pairs.
        self.alpha_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def forward(
        self,
        theta_prev: torch.Tensor,
        gamma_t: torch.Tensor,
        affinity: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance the vector Kuramoto system by one step.

        Shapes:
            theta_prev: [B, N, D]
            gamma_t:    [B, N, D]
            affinity:   [B, N, N]
            alpha_t:    [B, N, N]
        """
        _, n, _ = theta_prev.shape

        # `theta_i` and `theta_j` are reshaped so broadcasting can form every
        # pairwise phase difference between nodes i and j.
        theta_i = theta_prev.unsqueeze(2)
        theta_j = theta_prev.unsqueeze(1)

        # Combine the dynamic phase lag coming from feedback with the static,
        # learnable phase bias.
        alpha_total = alpha_t + self.alpha_bias.unsqueeze(0)

        # Sakaguchi-Kuramoto style interaction:
        # sin(theta_j - theta_i - alpha_ij)
        phase_term = torch.sin(theta_j - theta_i - alpha_total.unsqueeze(-1))

        # Sum incoming influences from all other nodes.
        coupling_term = (self.coupling / float(n)) * torch.sum(
            affinity.unsqueeze(-1) * phase_term, dim=2
        )

        # External sensory/control drive pushes the oscillator toward gamma_t.
        drive_term = self.kappa.unsqueeze(0) * (gamma_t - theta_prev)

        # Total time derivative of the oscillator state.
        theta_dot = self.omega.unsqueeze(0) + drive_term + coupling_term

        # Euler update.
        return theta_prev + self.dt * theta_dot


class GraphVectorKuramoto(nn.Module):
    """
    Thin wrapper kept mostly for naming consistency with the TINGTING layout.

    Right now it simply forwards to `VectorKuramoto`, but keeping the wrapper
    makes it easier to later swap in a more graph-specific implementation
    without changing the outer model code.
    """

    def __init__(self, num_nodes: int, osc_dim: int = 4, coupling: float = 1.0, dt: float = 1.0) -> None:
        super().__init__()
        self.core = VectorKuramoto(
            num_nodes=num_nodes,
            osc_dim=osc_dim,
            coupling=coupling,
            dt=dt,
        )

    def forward(
        self,
        theta_prev: torch.Tensor,
        gamma_t: torch.Tensor,
        affinity: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        return self.core(theta_prev, gamma_t, affinity, alpha_t)
