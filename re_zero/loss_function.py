from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .hyperparameters import ObjectRepresentationConfig
except ImportError:
    from hyperparameters import ObjectRepresentationConfig


class ObjectRepresentationLoss:
    """Standalone loss helper for supervised and object-spike training losses."""

    def __init__(self, config: ObjectRepresentationConfig) -> None:
        self.config = config

    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard supervised classification loss."""
        loss_name = str(self.config.loss_function).lower()
        if loss_name == "cross_entropy":
            return F.cross_entropy(logits, labels)
        raise ValueError(f"Unknown supervised loss function {self.config.loss_function!r}")

    def _late_spike_window(self, spike_trace: torch.Tensor) -> torch.Tensor:
        """Return the spike window used by the unsupervised object losses."""
        start_idx = min(max(self.config.classifier_start_step - 1, 0), spike_trace.shape[1] - 1)
        return spike_trace[:, start_idx:, :]

    def _prepare_object_masks(self, object_masks: torch.Tensor, spike_trace: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel-level object masks into flattened node-level masks.

        Args:
            object_masks: [O, H, W] or [B, O, H, W]
            spike_trace: [B, T, H*W]
        """
        if object_masks.dim() == 3:
            object_masks = object_masks.unsqueeze(0).expand(spike_trace.shape[0], -1, -1, -1)
        if object_masks.dim() != 4:
            raise ValueError(
                f"Expected object masks with shape [O, H, W] or [B, O, H, W], got {tuple(object_masks.shape)}"
            )
        if object_masks.shape[0] != spike_trace.shape[0]:
            raise ValueError(
                f"Mask batch size {object_masks.shape[0]} does not match spike batch size {spike_trace.shape[0]}"
            )

        flat_masks = object_masks.to(device=spike_trace.device, dtype=spike_trace.dtype).view(
            object_masks.shape[0],
            object_masks.shape[1],
            -1,
        )
        if flat_masks.shape[-1] != spike_trace.shape[-1]:
            raise ValueError(
                f"Flattened mask has {flat_masks.shape[-1]} nodes, but spike trace has {spike_trace.shape[-1]}"
            )
        return flat_masks

    def _object_activity(self, spike_trace: torch.Tensor, node_masks: torch.Tensor) -> torch.Tensor:
        """Average spike activity inside each object mask, shape [B, T, O]."""
        mask_sizes = node_masks.sum(dim=-1).clamp_min(1.0)
        return torch.einsum("btn,bon->bto", spike_trace, node_masks) / mask_sizes.unsqueeze(1)

    def object_spike_loss_components(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute reusable unsupervised object-spike loss components."""
        late_spikes = self._late_spike_window(spike_trace)
        node_masks = self._prepare_object_masks(object_masks, late_spikes)
        object_activity = self._object_activity(late_spikes, node_masks)

        _, steps, num_objects = object_activity.shape
        eps = late_spikes.new_tensor(1e-6)
        object_valid = node_masks.sum(dim=-1) > 0.5
        valid_object_count = object_valid.to(late_spikes.dtype).sum().clamp_min(1.0)

        object_mean = object_activity.unsqueeze(-1)
        mask_sizes = node_masks.sum(dim=-1).clamp_min(1.0)
        squared_error = (late_spikes.unsqueeze(2) - object_mean).pow(2) * node_masks.unsqueeze(1)
        within_per_object = squared_error.sum(dim=-1) / mask_sizes.unsqueeze(1)
        within_similarity = (within_per_object * object_valid.unsqueeze(1)).sum() / (
            steps * valid_object_count
        )

        if num_objects < 2:
            zero = late_spikes.new_tensor(0.0)
            between_difference = zero
            between_distance = zero
        else:
            valid_pairs = (
                object_valid.unsqueeze(-1)
                & object_valid.unsqueeze(-2)
                & ~torch.eye(num_objects, device=late_spikes.device, dtype=torch.bool).unsqueeze(0)
            )
            valid_pair_count = valid_pairs.to(late_spikes.dtype).sum().clamp_min(1.0)

            temporal_profile = object_activity / object_activity.sum(dim=1, keepdim=True).clamp_min(eps)
            profile_overlap = (temporal_profile.unsqueeze(-1) * temporal_profile.unsqueeze(-2)).sum(dim=1)
            between_difference = (profile_overlap * valid_pairs).sum() / valid_pair_count

            time_axis = torch.arange(steps, device=late_spikes.device, dtype=late_spikes.dtype)
            activity_mass = object_activity.sum(dim=1).clamp_min(eps)
            centers = (object_activity * time_axis.view(1, steps, 1)).sum(dim=1) / activity_mass
            pairwise_distance = torch.abs(centers.unsqueeze(-1) - centers.unsqueeze(-2))
            distance_scale = max(float(self.config.object_time_distance_scale), 1e-6)
            between_distance = (torch.exp(-pairwise_distance / distance_scale) * valid_pairs).sum() / valid_pair_count

        peak_activity = object_activity.max(dim=1).values
        object_density_per_object = F.relu(float(self.config.object_density_target) - peak_activity)
        object_density = (object_density_per_object * object_valid).sum() / valid_object_count

        object_union = node_masks.amax(dim=1)
        background_mask = 1.0 - object_union
        background_size = background_mask.sum(dim=-1).clamp_min(1.0)
        background_suppression = (late_spikes * background_mask.unsqueeze(1)).sum(dim=-1) / background_size.unsqueeze(1)
        background_suppression = background_suppression.mean()

        return {
            "within_similarity": within_similarity,
            "between_difference": between_difference,
            "object_density": object_density,
            "between_distance": between_distance,
            "background_suppression": background_suppression,
        }

    def unsupervised_object_loss_1234(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        """Loss using components 1, 2, 3, and 4 plus background suppression."""
        components = self.object_spike_loss_components(spike_trace, object_masks)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.object_density_weight * components["object_density"]
            + self.config.between_object_distance_weight * components["between_distance"]
            + self.config.background_suppression_weight * components["background_suppression"]
        )

    def unsupervised_object_loss_124(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        """Loss using components 1, 2, and 4 plus background suppression."""
        components = self.object_spike_loss_components(spike_trace, object_masks)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.between_object_distance_weight * components["between_distance"]
            + self.config.background_suppression_weight * components["background_suppression"]
        )

    def unsupervised_object_loss_123(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        """Loss using components 1, 2, and 3 plus background suppression."""
        components = self.object_spike_loss_components(spike_trace, object_masks)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.object_density_weight * components["object_density"]
            + self.config.background_suppression_weight * components["background_suppression"]
        )

    def unsupervised_object_loss(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        """Select the configured unsupervised object loss variant."""
        loss_name = str(self.config.object_loss_function)
        if loss_name == "1234":
            return self.unsupervised_object_loss_1234(spike_trace, object_masks)
        if loss_name == "124":
            return self.unsupervised_object_loss_124(spike_trace, object_masks)
        if loss_name == "123":
            return self.unsupervised_object_loss_123(spike_trace, object_masks)
        raise ValueError(f"Unknown object loss function {self.config.object_loss_function!r}")
