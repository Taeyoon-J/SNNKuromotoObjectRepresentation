from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .hyperparameters import ObjectRepresentationConfig
except ImportError:
    from hyperparameters import ObjectRepresentationConfig


class ObjectRepresentationLoss:
    """Standalone loss helper for supervised and object-centric unsupervised losses."""

    def __init__(self, config: ObjectRepresentationConfig) -> None:
        self.config = config

    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard supervised classification loss."""
        loss_name = str(self.config.loss_function).lower()
        if loss_name == "cross_entropy":
            return F.cross_entropy(logits, labels)
        raise ValueError(f"Unknown supervised loss function {self.config.loss_function!r}")

    def _extract_masks_and_grid(self, classifier_output: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract object masks [O, H, W] and per-pixel grid features [H, W, D]."""
        if "masks" not in classifier_output:
            raise ValueError("Classifier output must contain a 'masks' tensor.")

        object_masks = classifier_output["masks"]
        if object_masks.dim() != 3:
            raise ValueError(f"Expected 'masks' with shape [O, H, W], got {tuple(object_masks.shape)}")

        if "pixel_feature_grid" in classifier_output:
            grid = classifier_output["pixel_feature_grid"]
            if grid.dim() != 3:
                raise ValueError(
                    f"Expected 'pixel_feature_grid' with shape [H, W, F], got {tuple(grid.shape)}"
                )
        elif "mean_spike_grid" in classifier_output:
            grid = classifier_output["mean_spike_grid"]
            if grid.dim() != 2:
                raise ValueError(
                    f"Expected 'mean_spike_grid' with shape [H, W], got {tuple(grid.shape)}"
                )
            grid = grid.unsqueeze(-1)
        else:
            raise ValueError(
                "Classifier output must contain either 'pixel_feature_grid' or 'mean_spike_grid'."
            )

        if object_masks.shape[-2:] != grid.shape[:2]:
            raise ValueError(
                "Mask spatial shape "
                f"{tuple(object_masks.shape[-2:])} does not match grid shape {tuple(grid.shape[:2])}."
            )
        return object_masks, grid

    def _flatten_masks_and_grid(
        self,
        object_masks: torch.Tensor,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten masks to [O, N] and grid features to [N, D]."""
        flat_masks = object_masks.reshape(object_masks.shape[0], -1)
        flat_grid = grid.reshape(-1, grid.shape[-1]).to(device=object_masks.device, dtype=object_masks.dtype)
        return flat_masks, flat_grid

    def _object_feature_means(self, flat_masks: torch.Tensor, flat_grid: torch.Tensor) -> torch.Tensor:
        """Average grid features inside each object mask, shape [O, D]."""
        mask_sizes = flat_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return flat_masks @ flat_grid / mask_sizes

    def _grid_magnitude(self, flat_grid: torch.Tensor) -> torch.Tensor:
        """Reduce per-pixel grid features to a single activation magnitude."""
        if flat_grid.shape[-1] == 1:
            return flat_grid[:, 0]
        return torch.linalg.norm(flat_grid, dim=-1)

    def _object_centers(self, flat_masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Compute object centroids in image coordinates, shape [O, 2]."""
        y_coords = torch.arange(height, device=flat_masks.device, dtype=flat_masks.dtype)
        x_coords = torch.arange(width, device=flat_masks.device, dtype=flat_masks.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)
        mask_sizes = flat_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return flat_masks @ coords / mask_sizes

    def object_spike_loss_components(self, classifier_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute reusable unsupervised object loss components from classifier outputs."""
        object_masks, grid = self._extract_masks_and_grid(classifier_output)
        flat_masks, flat_grid = self._flatten_masks_and_grid(object_masks, grid)

        num_objects, _ = flat_masks.shape
        valid_objects = flat_masks.sum(dim=-1) > 0.5
        valid_object_count = valid_objects.to(flat_masks.dtype).sum().clamp_min(1.0)
        zero = flat_masks.new_tensor(0.0)

        object_means = self._object_feature_means(flat_masks, flat_grid)
        grid_magnitude = self._grid_magnitude(flat_grid)

        if valid_objects.any():
            flat_grid_expanded = flat_grid.unsqueeze(0)
            object_means_expanded = object_means.unsqueeze(1)
            squared_error = (flat_grid_expanded - object_means_expanded).pow(2).mean(dim=-1)
            within_per_object = (squared_error * flat_masks).sum(dim=-1) / flat_masks.sum(dim=-1).clamp_min(1.0)
            within_similarity = (within_per_object * valid_objects).sum() / valid_object_count
        else:
            within_similarity = zero

        if num_objects < 2 or valid_objects.to(torch.int64).sum() < 2:
            between_difference = zero
            between_distance = zero
        else:
            normalized_means = F.normalize(object_means, dim=-1, eps=1e-6)
            feature_similarity = normalized_means @ normalized_means.transpose(0, 1)

            centers = self._object_centers(flat_masks, object_masks.shape[-2], object_masks.shape[-1])
            center_distances = torch.cdist(centers, centers, p=2)
            distance_scale = max(float(self.config.object_time_distance_scale), 1e-6)
            center_closeness = torch.exp(-center_distances / distance_scale)

            valid_pairs = (
                valid_objects.unsqueeze(0)
                & valid_objects.unsqueeze(1)
                & ~torch.eye(num_objects, device=flat_masks.device, dtype=torch.bool)
            )
            valid_pair_count = valid_pairs.to(flat_masks.dtype).sum().clamp_min(1.0)

            between_difference = (feature_similarity * valid_pairs).sum() / valid_pair_count
            between_distance = (center_closeness * valid_pairs).sum() / valid_pair_count

        object_activation = (flat_masks * grid_magnitude.unsqueeze(0)).sum(dim=-1) / flat_masks.sum(dim=-1).clamp_min(1.0)
        object_density_per_object = F.relu(float(self.config.object_density_target) - object_activation)
        object_density = (object_density_per_object * valid_objects).sum() / valid_object_count

        object_union = flat_masks.amax(dim=0)
        background_mask = (1.0 - object_union).clamp_min(0.0)
        background_size = background_mask.sum().clamp_min(1.0)
        background_suppression = (background_mask * grid_magnitude).sum() / background_size

        return {
            "within_similarity": within_similarity,
            "between_difference": between_difference,
            "object_density": object_density,
            "between_distance": between_distance,
            "background_suppression": background_suppression,
        }

    def unsupervised_object_loss_1234(self, classifier_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Loss using components 1, 2, 3, and 4 plus background suppression."""
        components = self.object_spike_loss_components(classifier_output)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.object_density_weight * components["object_density"]
            + self.config.between_object_distance_weight * components["between_distance"]
            + self.config.background_suppression_weight * components["background_suppression"]
        )

    def unsupervised_object_loss_124(self, classifier_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Loss using components 1, 2, and 4 plus background suppression."""
        components = self.object_spike_loss_components(classifier_output)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.between_object_distance_weight * components["between_distance"]
            + self.config.background_suppression_weight * components["background_suppression"]
        )

    def unsupervised_object_loss_123(self, classifier_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Loss using components 1, 2, and 3 plus background suppression."""
        components = self.object_spike_loss_components(classifier_output)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.object_density_weight * components["object_density"]
            + self.config.background_suppression_weight * components["background_suppression"]
        )

    def unsupervised_object_loss(self, classifier_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select the configured unsupervised object loss variant."""
        loss_name = str(self.config.object_loss_function)
        if loss_name == "1234":
            return self.unsupervised_object_loss_1234(classifier_output)
        if loss_name == "124":
            return self.unsupervised_object_loss_124(classifier_output)
        if loss_name == "123":
            return self.unsupervised_object_loss_123(classifier_output)
        raise ValueError(f"Unknown object loss function {self.config.object_loss_function!r}")
