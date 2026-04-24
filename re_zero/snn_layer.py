from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

try:
    from .classifier import get_classifier
except ImportError:
    from classifier import get_classifier


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
        classifier_type: str,
        image_height: int,
        image_width: int,
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

        self.membrane_weight = nn.Parameter(torch.full((self.num_pixels,), 0.5))

        self.recurrent_weight = nn.Parameter(torch.randn(self.num_pixels) * 0.02)
        self.classifier = get_classifier(
            name=classifier_type,
            num_pixels=self.num_pixels,
            num_classes=num_classes,
            classifier_start_step=self.classifier_start_step,
            image_height=image_height,
            image_width=image_width,
        )

    def forward_step(
        self,
        membrane_prev: torch.Tensor,
        spikes_prev: torch.Tensor,
        sinusoidal_gate: torch.Tensor,
        gamma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single SNN update step.

        Shapes:
            membrane_prev: [B, H*W]
            spikes_prev: [B, H*W]
            sinusoidal_gate: [B, H*W]
            gamma: [B, H*W]
        """
        sinusoidal_wave = sinusoidal_gate * gamma
        synaptic_drive = sinusoidal_wave * self.membrane_weight.unsqueeze(0)
        recurrent_drive = self.recurrent_scale * (spikes_prev * self.recurrent_weight.unsqueeze(0))

        membrane = (
            self.membrane_decay * membrane_prev
            + synaptic_drive
            + recurrent_drive
            - self.threshold * spikes_prev
        )
        spikes = torch.sigmoid(8.0 * (membrane - self.threshold))
        return membrane, spikes

    def classify(self, spike_trace: torch.Tensor):
        """Map a spike trace using the configured classifier."""
        return self.classifier.classify(spike_trace)
