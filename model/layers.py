from __future__ import annotations

# This file groups reusable building blocks that sit between the low-level
# dynamics and the full end-to-end model.

from typing import Tuple

import torch
import torch.nn as nn

try:
    from .SNN_layers.vector_kuramoto import GraphVectorKuramoto
except ImportError:
    from SNN_layers.vector_kuramoto import GraphVectorKuramoto


class GraphVectorKuramotoLayer(nn.Module):
    """Small wrapper module around the graph-aware Kuramoto dynamics."""

    def __init__(self, num_nodes: int, osc_dim: int, coupling: float, dt: float, attraction_strength: float) -> None:
        super().__init__()
        self.kuramoto = GraphVectorKuramoto(
            num_nodes=num_nodes,
            osc_dim=osc_dim,
            coupling=coupling,
            dt=dt,
            attraction_strength=attraction_strength,
        )

    def forward(
        self,
        theta_prev: torch.Tensor,
        gamma_prev: torch.Tensor,
        affinity: torch.Tensor,
        alpha_t: torch.Tensor,
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
    ) -> None:
        super().__init__()
        # Keep these values as attributes so the update rule is easy to inspect.
        self.num_nodes = num_nodes
        self.membrane_decay = membrane_decay
        self.threshold = threshold

        # Projects oscillator features into a scalar current for each node.
        self.input_weight = nn.Linear(osc_dim, 1)
        # Recurrent interaction among spiking units.
        self.recurrent_weight = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.02)
        # Final classifier applied after temporal pooling of spikes.
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_nodes),
            nn.Linear(num_nodes, num_classes),
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
            membrane_prev: Previous membrane state, [B, N]
            spikes_prev: Previous spikes, [B, N]
            modulated_gamma: Oscillator-driven input current, [B, N, D]
        """
        # Convert oscillator features into scalar synaptic current per node.
        synaptic_drive = self.input_weight(modulated_gamma).squeeze(-1)
        # Add recurrent contributions from previous spikes.
        recurrent_drive = spikes_prev @ self.recurrent_weight

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
        Pool spike activity across time and map it to class logits.

        `spike_trace` has shape [B, T, N].
        """
        pooled = spike_trace.mean(dim=1)
        return self.classifier(pooled)
