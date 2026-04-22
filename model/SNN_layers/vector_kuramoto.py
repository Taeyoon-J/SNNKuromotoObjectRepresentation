from __future__ import annotations

# This file contains the multi-dimensional Kuramoto dynamics used by the model.
# Unlike the simple version, this one lets oscillators influence each other.

import torch
import torch.nn as nn


class VectorKuramoto(nn.Module):
    """
    Vector-valued Kuramoto block.

    Each spatial location has an oscillator vector of length `osc_dim`.
    Pairwise interactions are weighted by a theta connectivity matrix and
    shifted by a phase-lag matrix `alpha_t`.
    """

    def __init__(
        self,
        num_nodes: int,
        osc_dim: int = 4,
        coupling: float = 1.0,
        dt: float = 1.0,
        attraction_strength: float = 1.0,
        feedback_theta_connectivity_weight_scale: float = 0.25,
        feedback_alpha_scale: float = 0.25,
        alpha_scale: float = 1.0,
        fixed_alpha_during_training: bool = True,
        fixed_alpha_value: float = 0.0,
        coupling_chunk_size: int = 256,
        input_channels: int = 3,
        channel_wise_coupling: bool = True,
    ) -> None:
        super().__init__()
        # Number of flattened spatial locations, e.g. H * W.
        self.num_nodes = num_nodes
        # Number of oscillator channels carried by each location.
        self.osc_dim = osc_dim
        # Global synchronization strength.
        self.coupling = coupling
        # Euler integration step size.
        self.dt = dt
        # Shared attraction strength k_i toward the previous encoder drive.
        self.attraction_strength = attraction_strength
        # Feedback scales used when pairwise terms are generated from spikes.
        self.feedback_theta_connectivity_weight_scale = feedback_theta_connectivity_weight_scale
        self.feedback_alpha_scale = feedback_alpha_scale
        self.alpha_scale = alpha_scale
        self.fixed_alpha_during_training = fixed_alpha_during_training
        self.fixed_alpha_value = fixed_alpha_value
        self.coupling_chunk_size = coupling_chunk_size
        self.input_channels = input_channels
        self.channel_wise_coupling = channel_wise_coupling

    def _matrix_pairwise_coupling(
        self,
        theta_prev: torch.Tensor,
        theta_connectivity_weight: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked coupling for fixed [B, N, N] pairwise matrices."""
        _, n, _ = theta_prev.shape
        chunk_size = max(1, int(self.coupling_chunk_size))
        theta_j = theta_prev.unsqueeze(1)

        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            theta_i = theta_prev[:, start:end].unsqueeze(2)
            theta_connectivity_weight_chunk = theta_connectivity_weight[:, start:end]
            if self.training and self.fixed_alpha_during_training:
                alpha_chunk = torch.full_like(alpha_t[:, start:end], float(self.fixed_alpha_value))
            else:
                alpha_chunk = alpha_t[:, start:end]
            phase_term = torch.sin(theta_j - theta_i - alpha_chunk.unsqueeze(-1))
            chunk = (self.coupling / float(n)) * torch.sum(
                theta_connectivity_weight_chunk.unsqueeze(-1) * phase_term,
                dim=2,
            )
            chunks.append(chunk)

        return torch.cat(chunks, dim=1)

    def forward(
        self,
        theta_prev: torch.Tensor,
        gamma_prev: torch.Tensor,
        theta_connectivity_weight: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance the vector Kuramoto system by one step.

        Shapes:
            theta_prev: [B, N, D]
            gamma_prev: [B, N, D]
            theta_connectivity_weight: [B, N, N]
            alpha_t: [B, N, N]
        """
        if float(self.coupling) == 0.0:
            coupling_term = torch.zeros_like(theta_prev)
        else:
            if theta_connectivity_weight.dim() != 3:
                raise ValueError(
                    "theta_connectivity_weight must have shape [B, N, N], "
                    f"got {tuple(theta_connectivity_weight.shape)}"
                )
            if alpha_t.dim() != 3:
                raise ValueError(f"alpha_t must have shape [B, N, N], got {tuple(alpha_t.shape)}")
            if theta_connectivity_weight.shape != alpha_t.shape:
                raise ValueError("theta_connectivity_weight and alpha_t must have the same shape")
            coupling_term = self._matrix_pairwise_coupling(theta_prev, theta_connectivity_weight, alpha_t)

        # External sensory/control drive pushes the oscillator toward gamma(t-1).
        drive_term = self.attraction_strength * (gamma_prev - theta_prev)

        # Total time derivative of the oscillator state.
        theta_dot = drive_term + coupling_term

        # Euler update followed by vector normalization, keeping oscillator
        # states on the unit sphere as in vector/Kuramoto-style AKOrN dynamics.
        theta_next = theta_prev + self.dt * theta_dot
        return torch.nn.functional.normalize(theta_next, dim=-1, eps=1e-6)


class GraphVectorKuramoto(nn.Module):
    """
    Thin wrapper kept mostly for naming consistency with the TINGTING layout.

    Right now it simply forwards to `VectorKuramoto`, but keeping the wrapper
    makes it easier to later swap in a more graph-specific implementation
    without changing the outer model code.
    """

    def __init__(
        self,
        num_nodes: int,
        osc_dim: int = 4,
        coupling: float = 1.0,
        dt: float = 1.0,
        attraction_strength: float = 1.0,
        feedback_theta_connectivity_weight_scale: float = 0.25,
        feedback_alpha_scale: float = 0.25,
        alpha_scale: float = 1.0,
        fixed_alpha_during_training: bool = True,
        fixed_alpha_value: float = 0.0,
        coupling_chunk_size: int = 256,
        input_channels: int = 3,
        channel_wise_coupling: bool = True,
    ) -> None:
        super().__init__()
        self.core = VectorKuramoto(
            num_nodes=num_nodes,
            osc_dim=osc_dim,
            coupling=coupling,
            dt=dt,
            attraction_strength=attraction_strength,
            feedback_theta_connectivity_weight_scale=feedback_theta_connectivity_weight_scale,
            feedback_alpha_scale=feedback_alpha_scale,
            alpha_scale=alpha_scale,
            fixed_alpha_during_training=fixed_alpha_during_training,
            fixed_alpha_value=fixed_alpha_value,
            coupling_chunk_size=coupling_chunk_size,
            input_channels=input_channels,
            channel_wise_coupling=channel_wise_coupling,
        )

    def forward(
        self,
        theta_prev: torch.Tensor,
        gamma_prev: torch.Tensor,
        theta_connectivity_weight: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        return self.core(theta_prev, gamma_prev, theta_connectivity_weight, alpha_t)
