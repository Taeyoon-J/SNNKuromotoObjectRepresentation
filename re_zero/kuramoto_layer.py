from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KuramotoLayer(nn.Module):
    """
    Graph-aware vector Kuramoto dynamics.

    Each node carries an oscillator vector with dimension `osc_dim`. The state
    is updated by combining an attraction term toward `gamma_prev` with a
    pairwise coupling term derived from explicit pixel-pair theta connectivity
    weights and phase-lag matrices.
    """

    def __init__(
        self,
        num_oscillators: int,
        osc_dim: int = 4,
        global_coupling_strength: float = 1.0,
        step_size: float = 1.0,
        gamma_attraction_strength: float = 1.0,
        fixed_alpha_during_training: bool = True,
        fixed_alpha_value: float = 0.0,
        coupling_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        self.num_oscillators = num_oscillators
        self.osc_dim = osc_dim
        self.global_coupling_strength = global_coupling_strength
        self.step_size = step_size
        self.gamma_attraction_strength = gamma_attraction_strength
        self.fixed_alpha_during_training = fixed_alpha_during_training
        self.fixed_alpha_value = fixed_alpha_value
        self.coupling_chunk_size = coupling_chunk_size

    def forward(
        self,
        theta_prev: torch.Tensor,
        gamma_prev: torch.Tensor,
        theta_connectivity_weight: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        coupling_term = self.compute_coupling_term(theta_prev, theta_connectivity_weight, alpha_t)
        drive_term = self.gamma_attraction_strength * (gamma_prev - theta_prev)
        theta_dot = drive_term + coupling_term
        theta_next = theta_prev + self.step_size * theta_dot
        return F.normalize(theta_next, dim=-1, eps=1e-6)

    def compute_coupling_term(
        self,
        theta_prev: torch.Tensor,
        theta_connectivity_weight: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chunked Kuramoto coupling from fixed [B, N, N] matrices."""
        if float(self.global_coupling_strength) == 0.0:
            return torch.zeros_like(theta_prev)

        _, num_oscillators, _ = theta_prev.shape
        chunks = []
        for theta_i, theta_j, theta_connectivity_weight_chunk, alpha_chunk in self.seperate_chunks(
            theta_prev,
            theta_connectivity_weight,
            alpha_t,
        ):
            projection_term = self.compute_projection_term(theta_i, theta_j, alpha_chunk)
            chunks.append(
                (self.global_coupling_strength / float(num_oscillators)) * torch.sum(
                    theta_connectivity_weight_chunk.unsqueeze(-1) * projection_term,
                    dim=2,
                )
            )

        return torch.cat(chunks, dim=1)

    def compute_projection_term(
        self,
        theta_i: torch.Tensor,
        theta_j: torch.Tensor,
        alpha_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project sender vectors `theta_j` onto receiver directions `theta_i`.

        This replaces elementwise phase-difference sine with a vector similarity
        interaction: neighbors aligned with `theta_i` contribute more strongly,
        while orthogonal neighbors contribute little.
        """
        theta_i_unit = F.normalize(theta_i, dim=-1, eps=1e-6)
        projection_scale = torch.sum(theta_j * theta_i_unit, dim=-1, keepdim=True)
        projected_theta_j = projection_scale * theta_i_unit

        # For the projection-based coupling, keep phase delay fixed for now
        # instead of consuming the dynamic alpha returned by the SNN pathway.
        alpha_weight = torch.full_like(alpha_chunk.unsqueeze(-1), float(self.fixed_alpha_value)).cos()

        return alpha_weight * projected_theta_j

    def seperate_chunks(
        self,
        theta_prev: torch.Tensor,
        theta_connectivity_weight: torch.Tensor,
        alpha_t: torch.Tensor,
    ):
        """Prepare chunked tensors for coupling computation."""
        _, num_oscillators, _ = theta_prev.shape
        chunk_size = max(1, int(self.coupling_chunk_size))
        theta_j = theta_prev.unsqueeze(1)

        for start in range(0, num_oscillators, chunk_size):
            end = min(start + chunk_size, num_oscillators)
            theta_i = theta_prev[:, start:end].unsqueeze(2)
            theta_connectivity_weight_chunk = theta_connectivity_weight[:, start:end]

            if self.training and self.fixed_alpha_during_training:
                alpha_chunk = torch.full_like(alpha_t[:, start:end], float(self.fixed_alpha_value))
            else:
                alpha_chunk = alpha_t[:, start:end]

            yield theta_i, theta_j, theta_connectivity_weight_chunk, alpha_chunk
