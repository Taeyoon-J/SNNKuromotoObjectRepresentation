from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNNLayer(nn.Module):
    """
    Bottom-up spiking layer.

    This module receives gated gamma signals, updates membrane potentials,
    generates smooth surrogate spikes, and classifies a spike trace.
    """

    def __init__(
        self,
        num_oscillators: int,
        osc_dim: int,
        num_classes: int,
        membrane_decay: float,
        threshold: float,
        recurrent_scale: float,
        classifier_start_step: int,
        input_channels: int,
    ) -> None:
        super().__init__()
        self.num_oscillators = num_oscillators
        self.input_channels = input_channels
        self.num_pixels = num_oscillators
        self.membrane_decay = membrane_decay
        self.threshold = threshold
        self.recurrent_scale = recurrent_scale
        self.classifier_start_step = classifier_start_step

        self.input_weight = nn.Linear(osc_dim, 1, bias=False)
        with torch.no_grad():
            desired_weight = torch.tensor(0.5)
            self.input_weight.weight.fill_(torch.log(torch.expm1(desired_weight)).item())

        self.recurrent_weight = nn.Parameter(torch.randn(self.num_pixels, self.num_pixels) * 0.02)
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

        Shapes:
            membrane_prev: [B, H*W]
            spikes_prev: [B, H*W]
            modulated_gamma: [B, H*W, D]
        """
        positive_input_weight = F.softplus(self.input_weight.weight)
        synaptic_drive = F.linear(modulated_gamma, positive_input_weight).squeeze(-1)
        recurrent_drive = self.recurrent_scale * (spikes_prev @ self.recurrent_weight)

        membrane = (
            self.membrane_decay * membrane_prev
            + synaptic_drive
            + recurrent_drive
            - self.threshold * spikes_prev
        )
        spikes = torch.sigmoid(8.0 * (membrane - self.threshold))
        return membrane, spikes

    def classify(self, spike_trace: torch.Tensor) -> torch.Tensor:
        """Pool late spike activity and map it to class logits."""
        start_idx = min(max(self.classifier_start_step - 1, 0), spike_trace.shape[1] - 1)
        pooled = spike_trace[:, start_idx:, :].mean(dim=1)
        return self.classifier(pooled)
