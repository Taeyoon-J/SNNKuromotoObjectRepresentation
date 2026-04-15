from __future__ import annotations

# This file groups reusable building blocks that sit between the low-level
# dynamics and the full end-to-end model.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .SNN_layers.vector_kuramoto import GraphVectorKuramoto
except ImportError:
    from SNN_layers.vector_kuramoto import GraphVectorKuramoto


class GraphVectorKuramotoLayer(nn.Module):
    """Small wrapper module around the graph-aware Kuramoto dynamics."""

    def __init__(
        self,
        num_nodes: int,
        osc_dim: int,
        coupling: float,
        dt: float,
        attraction_strength: float,
        feedback_affinity_scale: float,
        feedback_alpha_scale: float,
        alpha_scale: float,
        coupling_chunk_size: int,
        input_channels: int,
        channel_wise_coupling: bool,
    ) -> None:
        super().__init__()
        self.kuramoto = GraphVectorKuramoto(
            num_nodes=num_nodes,
            osc_dim=osc_dim,
            coupling=coupling,
            dt=dt,
            attraction_strength=attraction_strength,
            feedback_affinity_scale=feedback_affinity_scale,
            feedback_alpha_scale=feedback_alpha_scale,
            alpha_scale=alpha_scale,
            coupling_chunk_size=coupling_chunk_size,
            input_channels=input_channels,
            channel_wise_coupling=channel_wise_coupling,
        )

    def forward(
        self,
        theta_prev: torch.Tensor,
        gamma_prev: torch.Tensor,
        affinity: Optional[torch.Tensor],
        alpha_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Delegate the actual oscillator update to the Kuramoto layer.
        return self.kuramoto(theta_prev, gamma_prev, affinity, alpha_t)


class ObjectReadoutSNN(nn.Module):
    """
    Bottom-up spiking readout block.

    This module receives oscillator-driven signals, updates membrane potentials,
    computes soft spikes, and finally classifies a spike trace.
    """

    def __init__(
        self,
        num_nodes: int,
        osc_dim: int,
        num_classes: int,
        membrane_decay: float,
        threshold: float,
        recurrent_scale: float,
        classifier_start_step: int,
        input_channels: int,
    ) -> None:
        super().__init__()
        # Keep these values as attributes so the update rule is easy to inspect.
        self.num_nodes = num_nodes
        self.input_channels = input_channels
        if num_nodes % input_channels != 0:
            raise ValueError(f"num_nodes={num_nodes} must be divisible by input_channels={input_channels}")
        self.num_pixels = num_nodes // input_channels
        self.membrane_decay = membrane_decay
        self.threshold = threshold
        self.recurrent_scale = recurrent_scale
        self.classifier_start_step = classifier_start_step

        # Projects each vector oscillator theta/gamma feature into a scalar current.
        # The raw parameter is passed through softplus in forward_step so the
        # SNN input drive cannot flip an object signal into negative current.
        self.input_weight = nn.Linear(osc_dim, 1, bias=False)
        with torch.no_grad():
            desired_weight = torch.tensor(0.5)
            self.input_weight.weight.fill_(torch.log(torch.expm1(desired_weight)).item())
        # Learned per-pixel RGB mixing weights for sum_z w_xyz * gamma_xyz(t).
        self.channel_weight = nn.Parameter(torch.ones(self.num_pixels, input_channels) / float(input_channels))
        # Recurrent interaction among pixel-level spiking units.
        self.recurrent_weight = nn.Parameter(torch.randn(self.num_pixels, self.num_pixels) * 0.02)
        # Final classifier applied after temporal pooling of spikes.
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_pixels),
            nn.Linear(self.num_pixels, num_classes),
        )

    def forward_step(
        self,
        membrane_prev: torch.Tensor,
        spikes_prev: torch.Tensor,
        modulated_gamma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single SNN update step.

        Args:
            membrane_prev: Previous pixel-level membrane state, [B, H*W]
            spikes_prev: Previous pixel-level spikes, [B, H*W]
            modulated_gamma: Oscillator-driven input current, [B, N, D]
        """
        # Convert each vector oscillator into scalar current per pixel-channel node.
        positive_input_weight = F.softplus(self.input_weight.weight)
        node_drive = F.linear(modulated_gamma, positive_input_weight).squeeze(-1)
        # Compress RGB channel nodes into one pixel-level synaptic drive:
        # I_xy(t) = sum_z w_xyz * gamma_xyz(t).
        channel_drive = node_drive.view(node_drive.shape[0], self.num_pixels, self.input_channels)
        synaptic_drive = (channel_drive * self.channel_weight.unsqueeze(0)).sum(dim=-1)
        # Add recurrent contributions from previous spikes.
        recurrent_drive = self.recurrent_scale * (spikes_prev @ self.recurrent_weight)

        # Standard leaky-integrator style membrane update with threshold-sized reset.
        membrane = (
            self.membrane_decay * membrane_prev
            + synaptic_drive
            + recurrent_drive
            - self.threshold * spikes_prev
        )

        # Use a smooth surrogate spike function so the model stays trainable.
        spikes = torch.sigmoid(8.0 * (membrane - self.threshold))
        return membrane, spikes

    def classify(self, spike_trace: torch.Tensor) -> torch.Tensor:
        """
        Pool spike activity from the configured late time window and map it to class logits.

        `spike_trace` has shape [B, T, H*W].
        """
        # Convert the human-facing time step t=60 into zero-based index 59.
        # If the run is shorter than that, fall back to the final available step.
        start_idx = min(max(self.classifier_start_step - 1, 0), spike_trace.shape[1] - 1)
        pooled = spike_trace[:, start_idx:, :].mean(dim=1)
        return self.classifier(pooled)
