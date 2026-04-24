from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

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
    def classify(self, spike_trace: torch.Tensor) -> torch.Tensor:
        """Map a spike trace [B, T, H*W] to object masks [B, O, H, W]."""


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
        similarity_threshold: float = 0.90,
    ) -> None:
        super().__init__(num_pixels, num_classes, classifier_start_step, image_height, image_width)
        self.similarity_threshold = similarity_threshold

    def classify(self, spike_trace: torch.Tensor) -> torch.Tensor:
        start_idx = min(max(self.classifier_start_step - 1, 0), spike_trace.shape[1] - 1)
        average_spikes = spike_trace[:, start_idx:, :].mean(dim=1)
        object_masks: List[torch.Tensor] = []

        for batch_idx in range(average_spikes.shape[0]):
            pixel_values = average_spikes[batch_idx]
            similarity = 1.0 - torch.abs(pixel_values.unsqueeze(1) - pixel_values.unsqueeze(0))
            object_masks.append(
                build_similarity_components_to_masks(
                    similarity=similarity,
                    similarity_threshold=self.similarity_threshold,
                    image_height=self.image_height,
                    image_width=self.image_width,
                )
            )
        return stack_object_masks(object_masks, spike_trace)


class SpikeFeatureClassifier(SpikeClassifier):
    """Group pixels into object masks using feature similarity."""

    def __init__(
        self,
        num_pixels: int,
        num_classes: int,
        classifier_start_step: int,
        image_height: int,
        image_width: int,
        similarity_threshold: float = 0.90,
    ) -> None:
        super().__init__(num_pixels, num_classes, classifier_start_step, image_height, image_width)
        self.pattern_encoder = PixelPatternEncoder(feature_dim=8)
        self.similarity_threshold = similarity_threshold

    def classify(self, spike_trace: torch.Tensor) -> torch.Tensor:
        start_idx = min(max(self.classifier_start_step - 1, 0), spike_trace.shape[1] - 1)
        pixel_feature_vectors = self.pattern_encoder(spike_trace, start_idx)
        normalized_features = F.normalize(pixel_feature_vectors, dim=-1, eps=1e-6)

        object_masks: List[torch.Tensor] = []
        for batch_idx in range(normalized_features.shape[0]):
            similarity = normalized_features[batch_idx] @ normalized_features[batch_idx].transpose(0, 1)
            object_masks.append(
                build_similarity_components_to_masks(
                    similarity=similarity,
                    similarity_threshold=self.similarity_threshold,
                    image_height=self.image_height,
                    image_width=self.image_width,
                )
            )
        return stack_object_masks(object_masks, spike_trace)


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


def stack_object_masks(object_masks: List[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    """Pad per-sample object masks into a dense batch tensor [B, O, H, W]."""
    if not object_masks:
        return reference.new_zeros((0, 0, 0, 0))

    max_objects = max(mask.shape[0] for mask in object_masks)
    batch_size = len(object_masks)
    image_height, image_width = object_masks[0].shape[-2:]
    padded_masks = reference.new_zeros((batch_size, max_objects, image_height, image_width))

    for batch_idx, masks in enumerate(object_masks):
        padded_masks[batch_idx, : masks.shape[0]] = masks.to(device=reference.device, dtype=reference.dtype)
    return padded_masks


def get_classifier(
    name: str,
    num_pixels: int,
    num_classes: int,
    classifier_start_step: int,
    image_height: int,
    image_width: int,
) -> SpikeClassifier:
    """Create a spike classifier from a config keyword."""
    normalized_name = name.lower().strip()
    if normalized_name in {"mean_spike", "average", "default"}:
        return MeanSpikeClassifier(num_pixels, num_classes, classifier_start_step, image_height, image_width)
    if normalized_name in {"spike_feature", "feature", "objects"}:
        return SpikeFeatureClassifier(
            num_pixels,
            num_classes,
            classifier_start_step,
            image_height,
            image_width,
        )
    raise ValueError(
        f"Unknown classifier '{name}'. "
        "Choose one of: mean_spike, spike_feature."
    )
