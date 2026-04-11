from __future__ import annotations

# This file creates a small synthetic object dataset.
# It is intentionally simple so the model can be tested without an external
# dataset pipeline.

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def make_object_pattern(object_id: int, image_size: int) -> np.ndarray:
    """
    Build a binary object mask for one predefined shape class.

    The returned array is 2D and values are 0/1 before color is applied.
    """
    # Create a coordinate grid so shapes can be defined analytically.
    grid_y, grid_x = np.mgrid[0:image_size, 0:image_size]
    cy = (image_size - 1) / 2.0
    cx = (image_size - 1) / 2.0
    radius = image_size / 4.0

    # Each integer label corresponds to one object template.
    if object_id == 0:
        mask = (np.abs(grid_y - cy) <= radius) & (np.abs(grid_x - cx) <= radius)
    elif object_id == 1:
        mask = (np.abs(grid_x - cx) <= 1) | (np.abs(grid_y - cy) <= 1)
    elif object_id == 2:
        mask = np.abs((grid_x - cx) - (grid_y - cy)) <= 1
    elif object_id == 3:
        dist = np.sqrt((grid_y - cy) ** 2 + (grid_x - cx) ** 2)
        mask = (dist >= radius - 1.5) & (dist <= radius + 1.5)
    else:
        mask = ((grid_x // 2 + grid_y // 2) % 2) == 0

    return mask.astype(np.float32)


def colorize_pattern(mask: np.ndarray, object_id: int) -> np.ndarray:
    """Convert a binary shape mask into a simple RGB image."""
    palette = np.asarray(
        [
            [1.0, 0.25, 0.25],
            [0.25, 0.85, 0.35],
            [0.20, 0.45, 1.0],
            [1.0, 0.75, 0.15],
            [0.85, 0.25, 0.85],
        ],
        dtype=np.float32,
    )
    # Pick a stable color for each object class.
    color = palette[object_id % len(palette)]
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    # Slight gray background makes the object easier to see than a pure-black background.
    image += 0.05
    image[mask > 0.0] = color
    return image


def paste_pattern_on_canvas(
    canvas: np.ndarray,
    pattern_image: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray:
    """Paste a smaller RGB pattern image onto a larger canvas."""
    patch_h, patch_w, _ = pattern_image.shape
    bottom = min(top + patch_h, canvas.shape[0])
    right = min(left + patch_w, canvas.shape[1])

    valid_h = max(0, bottom - top)
    valid_w = max(0, right - left)
    if valid_h == 0 or valid_w == 0:
        return canvas

    patch = pattern_image[:valid_h, :valid_w]
    object_mask = np.any(patch > 0.06, axis=-1, keepdims=True)
    canvas[top:bottom, left:right] = np.where(object_mask, patch, canvas[top:bottom, left:right])
    return canvas


def build_composite_scene(
    image_size: int = 16,
    object_specs: Tuple[Tuple[int, int, int], ...] = ((0, 1, 1), (3, 8, 8)),
    patch_size: int = 6,
    background_value: float = 0.05,
) -> torch.Tensor:
    """
    Build one scene that contains multiple predefined objects at chosen locations.

    `object_specs` is a tuple of `(object_id, top, left)` entries.
    """
    canvas = np.full((image_size, image_size, 3), background_value, dtype=np.float32)

    for object_id, top, left in object_specs:
        mask = make_object_pattern(object_id, patch_size)
        patch = colorize_pattern(mask, object_id)
        canvas = paste_pattern_on_canvas(canvas, patch, top=top, left=left)

    return torch.from_numpy(np.clip(canvas, 0.0, 1.0)).float()


def build_composite_masks(
    image_size: int = 16,
    object_specs: Tuple[Tuple[int, int, int], ...] = ((0, 1, 1), (3, 8, 8)),
    patch_size: int = 6,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Reconstruct one binary mask per object in a composite scene.

    Returns:
        masks: [num_objects, H, W]
        names: readable names for legends/titles
    """
    masks = []
    names = []
    object_names = {
        0: "square",
        1: "cross",
        2: "diagonal",
        3: "circle",
        4: "checker",
    }

    for object_id, top, left in object_specs:
        full_mask = np.zeros((image_size, image_size), dtype=np.float32)
        patch_mask = make_object_pattern(object_id, patch_size)
        patch_h, patch_w = patch_mask.shape
        bottom = min(top + patch_h, image_size)
        right = min(left + patch_w, image_size)
        valid_h = max(0, bottom - top)
        valid_w = max(0, right - left)
        if valid_h > 0 and valid_w > 0:
            full_mask[top:bottom, left:right] = patch_mask[:valid_h, :valid_w]
        masks.append(full_mask)
        names.append(object_names.get(object_id, f"object_{object_id}"))

    return torch.from_numpy(np.stack(masks, axis=0)).float(), names


def sample_predefined_objects(
    num_samples: int = 100,
    image_size: int = 16,
    num_classes: int = 5,
    noise_std: float = 0.03,
    seed: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic dataset of colored objects.

    The procedure:
    1. Assign labels.
    2. Create the matching object mask.
    3. Colorize it.
    4. Apply tiny random shifts and noise for variation.
    """
    rng = np.random.default_rng(seed)
    labels = np.arange(num_samples) % num_classes
    rng.shuffle(labels)

    images = []
    for label in labels:
        mask = make_object_pattern(int(label), image_size)
        image = colorize_pattern(mask, int(label))
        # Random one-pixel shifts prevent the task from being too trivial.
        image = np.roll(image, shift=int(rng.integers(-1, 2)), axis=0)
        image = np.roll(image, shift=int(rng.integers(-1, 2)), axis=1)
        # Add small pixel noise so the model learns a more robust representation.
        image += rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
        images.append(np.clip(image, 0.0, 1.0))

    # Convert to PyTorch tensors for use by DataLoader and the model.
    image_tensor = torch.from_numpy(np.stack(images, axis=0)).float()
    label_tensor = torch.from_numpy(labels.astype(np.int64))
    return image_tensor, label_tensor


class PredefinedObjectDataset(Dataset):
    """
    Torch dataset wrapper around the synthetic object generator.

    Returning `(image, label)` makes it directly compatible with a standard
    PyTorch training loop.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 16,
        num_classes: int = 5,
        noise_std: float = 0.03,
        seed: int = 7,
    ) -> None:
        super().__init__()
        # Pre-generate everything once so each epoch sees a stable dataset.
        images, labels = sample_predefined_objects(
            num_samples=num_samples,
            image_size=image_size,
            num_classes=num_classes,
            noise_std=noise_std,
            seed=seed,
        )
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        """Number of samples available in the dataset."""
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch one image and its class label."""
        return self.images[idx], self.labels[idx]
