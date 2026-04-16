from __future__ import annotations

# This file contains the multi-dimensional Kuramoto dynamics used by the model.
# Unlike the simple version, this one lets oscillators influence each other.

import torch
import torch.nn as nn
from typing import Optional


class VectorKuramoto(nn.Module):
    """
    Vector-valued Kuramoto block.

    Each spatial location has an oscillator vector of length `osc_dim`.
    Pairwise interactions are weighted by an affinity matrix and shifted by a
    phase-lag matrix `alpha_t`.
    """

    def __init__(
        self,
        num_nodes: int,
        osc_dim: int = 4,
        coupling: float = 1.0,
        dt: float = 1.0,
        attraction_strength: float = 1.0,
        feedback_affinity_scale: float = 0.25,
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
        self.feedback_affinity_scale = feedback_affinity_scale
        self.feedback_alpha_scale = feedback_alpha_scale
        self.alpha_scale = alpha_scale
        self.fixed_alpha_during_training = fixed_alpha_during_training
        self.fixed_alpha_value = fixed_alpha_value
        self.coupling_chunk_size = coupling_chunk_size
        self.input_channels = input_channels
        self.channel_wise_coupling = channel_wise_coupling

    def _feedback_pairwise_coupling(self, theta_prev: torch.Tensor, feedback_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise coupling in receiver-node chunks.

        This avoids materializing [B, N, N, D], which is too large for 64x64x3
        inputs. `feedback_spikes` has shape [B, N] and defines the same
        relationship as the previous top-down feedback:
            affinity_ij = scale * (1 - normalized |S_i - S_j|)
            alpha_ij = scale * alpha_scale * normalized |S_i - S_j|
        """
        batch_size, n, _ = theta_prev.shape
        chunk_size = max(1, int(self.coupling_chunk_size))

        spike_range = (feedback_spikes.amax(dim=1, keepdim=True) - feedback_spikes.amin(dim=1, keepdim=True)).clamp_min(1e-6)
        theta_j = theta_prev.unsqueeze(1)
        spike_j = feedback_spikes.unsqueeze(1)

        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            theta_i = theta_prev[:, start:end].unsqueeze(2)
            spike_i = feedback_spikes[:, start:end].unsqueeze(2)
            normalized_delta = torch.abs(spike_i - spike_j) / spike_range.unsqueeze(-1)
            affinity = self.feedback_affinity_scale * (1.0 - normalized_delta)
            if self.training and self.fixed_alpha_during_training:
                alpha = torch.full_like(normalized_delta, float(self.fixed_alpha_value))
            else:
                alpha = self.feedback_alpha_scale * self.alpha_scale * normalized_delta
            phase_term = torch.sin(theta_j - theta_i - alpha.unsqueeze(-1))
            chunk = (self.coupling / float(n)) * torch.sum(affinity.unsqueeze(-1) * phase_term, dim=2)
            chunks.append(chunk)

        return torch.cat(chunks, dim=1)

    def _channel_wise_feedback_pairwise_coupling(
        self,
        theta_prev: torch.Tensor,
        feedback_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feedback coupling separately inside each input channel.

        The model flattens images in HWC order, so channel c occupies nodes
        c, c + C, c + 2C, ... . This method lets red couple only with red,
        green only with green, and blue only with blue.
        """
        if self.input_channels <= 1 or theta_prev.shape[1] % self.input_channels != 0:
            return self._feedback_pairwise_coupling(theta_prev, feedback_spikes)

        coupling_term = torch.zeros_like(theta_prev)
        for channel_idx in range(self.input_channels):
            node_slice = slice(channel_idx, None, self.input_channels)
            channel_theta = theta_prev[:, node_slice, :]
            channel_spikes = feedback_spikes[:, node_slice]
            coupling_term[:, node_slice, :] = self._feedback_pairwise_coupling(channel_theta, channel_spikes)
        return coupling_term

    def _matrix_pairwise_coupling(
        self,
        theta_prev: torch.Tensor,
        affinity: Optional[torch.Tensor],
        alpha_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fallback chunked coupling for callers that still pass full matrices."""
        _, n, _ = theta_prev.shape
        chunk_size = max(1, int(self.coupling_chunk_size))
        theta_j = theta_prev.unsqueeze(1)

        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            theta_i = theta_prev[:, start:end].unsqueeze(2)
            affinity_chunk = affinity[:, start:end]
            if self.training and self.fixed_alpha_during_training:
                alpha_chunk = torch.full_like(alpha_t[:, start:end], float(self.fixed_alpha_value))
            else:
                alpha_chunk = alpha_t[:, start:end]
            phase_term = torch.sin(theta_j - theta_i - alpha_chunk.unsqueeze(-1))
            chunk = (self.coupling / float(n)) * torch.sum(affinity_chunk.unsqueeze(-1) * phase_term, dim=2)
            chunks.append(chunk)

        return torch.cat(chunks, dim=1)

    def forward(
        self,
        theta_prev: torch.Tensor,
        gamma_prev: torch.Tensor,
        affinity: Optional[torch.Tensor],
        alpha_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Advance the vector Kuramoto system by one step.

        Shapes:
            theta_prev: [B, N, D]
            gamma_prev: [B, N, D]
            affinity:   None, [B, N] feedback spikes, or [B, N, N]
            alpha_t:    None or [B, N, N]
        """
        if affinity is None:
            coupling_term = torch.zeros_like(theta_prev)
        elif affinity.dim() == 2:
            if self.channel_wise_coupling:
                coupling_term = self._channel_wise_feedback_pairwise_coupling(theta_prev, affinity)
            else:
                coupling_term = self._feedback_pairwise_coupling(theta_prev, affinity)
        elif affinity.dim() == 3 and alpha_t is not None:
            coupling_term = self._matrix_pairwise_coupling(theta_prev, affinity, alpha_t)
        else:
            raise ValueError("affinity must be None, [B, N] feedback spikes, or [B, N, N] with alpha_t")

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
        feedback_affinity_scale: float = 0.25,
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
            feedback_affinity_scale=feedback_affinity_scale,
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
        affinity: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        return self.core(theta_prev, gamma_prev, affinity, alpha_t)
