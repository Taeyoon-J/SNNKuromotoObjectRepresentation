from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeClassifier(nn.Module, ABC):
    """Base class for classifiers built from spike traces."""

    def __init__(
        self,
        num_pixels: int,
        num_classes: int,
        classifier_start_step: int,
        image_height: int,
        image_width: int,
    ) -> None:
        super().__init__()
        self.num_pixels = num_pixels
        self.num_classes = num_classes
        self.classifier_start_step = classifier_start_step
        self.image_height = image_height
        self.image_width = image_width

    @abstractmethod
    def classify(self, spike_trace: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map a spike trace [B, T, H*W] to classifier outputs with B assumed to be 1."""


class PixelPatternEncoder(nn.Module):
    """Encode each pixel's spike-time pattern into a feature vector."""

    def __init__(self, feature_dim: int = 8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.pattern_encoder = nn.Sequential(
            nn.LazyLinear(16),
            nn.SiLU(),
            nn.Linear(16, self.feature_dim),
            nn.SiLU(),
        )

    def forward(self, spike_trace: torch.Tensor, start_idx: int) -> torch.Tensor:
        pixel_pattern_vectors = spike_trace[:, start_idx:, :].transpose(1, 2)
        return self.pattern_encoder(pixel_pattern_vectors)


class MeanSpikeClassifier(SpikeClassifier):
    """Group pixels into object masks using average late spike similarity."""

    def __init__(
        self,
        num_pixels: int,
        num_classes: int,
        classifier_start_step: int,
        image_height: int,
        image_width: int,
        similarity_threshold: float = 0.60,
    ) -> None:
        super().__init__(num_pixels, num_classes, classifier_start_step, image_height, image_width)
        self.similarity_threshold = similarity_threshold

    def classify(self, spike_trace: torch.Tensor) -> Dict[str, torch.Tensor]:
        start_idx = min(max(self.classifier_start_step - 1, 0), spike_trace.shape[1] - 1)
        average_spikes = spike_trace[:, start_idx:, :].mean(dim=1)
        pixel_values = average_spikes[0]
        similarity = 1.0 - torch.abs(pixel_values.unsqueeze(1) - pixel_values.unsqueeze(0))
        object_masks = build_similarity_components_to_masks(
            similarity=similarity,
            similarity_threshold=self.similarity_threshold,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        return {
            "masks": object_masks,
            "mean_spike_grid": pixel_values.view(self.image_height, self.image_width),
        }


class SpikeFeatureClassifier(SpikeClassifier):
    """Group pixels into object masks using feature similarity."""

    def __init__(
        self,
        num_pixels: int,
        num_classes: int,
        classifier_start_step: int,
        image_height: int,
        image_width: int,
        similarity_threshold: float = 0.60,
    ) -> None:
        super().__init__(num_pixels, num_classes, classifier_start_step, image_height, image_width)
        self.pattern_encoder = PixelPatternEncoder(feature_dim=8)
        self.similarity_threshold = similarity_threshold

    def classify(self, spike_trace: torch.Tensor) -> Dict[str, torch.Tensor]:
        start_idx = min(max(self.classifier_start_step - 1, 0), spike_trace.shape[1] - 1)
        pixel_feature_vectors = self.pattern_encoder(spike_trace, start_idx)
        normalized_features = F.normalize(pixel_feature_vectors, dim=-1, eps=1e-6)
        similarity = normalized_features[0] @ normalized_features[0].transpose(0, 1)
        object_masks = build_similarity_components_to_masks(
            similarity=similarity,
            similarity_threshold=self.similarity_threshold,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        return {
            "masks": object_masks,
            "pixel_feature_grid": pixel_feature_vectors[0].view(
                self.image_height,
                self.image_width,
                self.pattern_encoder.feature_dim,
            ),
        }


def build_similarity_components_to_masks(
    similarity: torch.Tensor,
    similarity_threshold: float,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Turn a pixel similarity matrix into connected-component object masks."""
    num_pixels = similarity.shape[0]
    visited = torch.zeros(num_pixels, dtype=torch.bool, device=similarity.device)
    component_masks = []

    for start_pixel in range(num_pixels):
        if visited[start_pixel]:
            continue

        component = []
        stack = [start_pixel]
        visited[start_pixel] = True

        while stack:
            pixel_idx = stack.pop()
            component.append(pixel_idx)
            neighbors = torch.where(similarity[pixel_idx] >= similarity_threshold)[0].tolist()
            for neighbor_idx in neighbors:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    stack.append(neighbor_idx)

        mask = torch.zeros(num_pixels, device=similarity.device, dtype=similarity.dtype)
        mask[component] = 1.0
        component_masks.append(mask.view(image_height, image_width))

    return torch.stack(component_masks, dim=0)
def get_classifier(
    name: str,
    num_pixels: int,
    num_classes: int,
    classifier_start_step: int,
    image_height: int,
    image_width: int,
    similarity_threshold: float = 0.60,
) -> SpikeClassifier:
    """Create a spike classifier from a config keyword."""
    normalized_name = name.lower().strip()
    if normalized_name in {"mean_spike", "average", "default"}:
        return MeanSpikeClassifier(
            num_pixels,
            num_classes,
            classifier_start_step,
            image_height,
            image_width,
            similarity_threshold=similarity_threshold,
        )
    if normalized_name in {"spike_feature", "feature", "objects"}:
        return SpikeFeatureClassifier(
            num_pixels,
            num_classes,
            classifier_start_step,
            image_height,
            image_width,
            similarity_threshold=similarity_threshold,
        )
    raise ValueError(
        f"Unknown classifier '{name}'. "
        "Choose one of: mean_spike, spike_feature."
    )
