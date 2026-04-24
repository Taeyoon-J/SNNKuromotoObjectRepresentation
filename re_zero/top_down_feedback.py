from __future__ import annotations

from typing import Tuple

import torch

try:
    from .hyperparameters import ObjectRepresentationConfig
except ImportError:
    from hyperparameters import ObjectRepresentationConfig


class TopDownFeedback:
    """Build pairwise feedback tensors from pixel-level SNN spikes."""

    def __init__(self, config: ObjectRepresentationConfig) -> None:
        self.config = config
        self.num_oscillators = config.num_oscillators

    def top_down_feedback_function(self, spikes: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build pairwise theta connectivity and phase-lag matrices from spikes.

        Returns:
            theta_connectivity_weight: [B, N, N]
            alpha_t: [B, N, N]
        """
        if spikes.shape[-1] != self.num_oscillators:
            raise ValueError(
                f"Expected pixel-level spikes with {self.num_oscillators} nodes, got {spikes.shape[-1]}"
            )

        spike_i = spikes.unsqueeze(2)
        spike_j = spikes.unsqueeze(1)
        spike_range = (spikes.amax(dim=1, keepdim=True) - spikes.amin(dim=1, keepdim=True)).clamp_min(1e-6)
        normalized_delta = torch.abs(spike_i - spike_j) / spike_range.unsqueeze(-1)

        theta_connectivity_weight = (
            self.config.feedback_theta_connectivity_weight_scale * (1.0 - normalized_delta)
        )
        if training and self.config.fixed_alpha_during_training:
            alpha_t = torch.full_like(normalized_delta, float(self.config.fixed_alpha_value))
        else:
            alpha_t = self.config.alpha_scale * (1.0 - theta_connectivity_weight)
        return theta_connectivity_weight, alpha_t
