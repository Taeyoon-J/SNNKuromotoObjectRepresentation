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
    """
    Paste a smaller RGB pattern image onto a larger RGB canvas.

    Later objects overwrite earlier ones where they overlap. For the current
    analysis scenes we place objects apart so the overwrite behavior is simple.
    """
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
    Example:
        ((0, 1, 1), (3, 8, 8))
    means "put object 0 near the top-left and object 3 near the bottom-right".
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


def make_temporal_object_schedule(num_objects: int, steps: int) -> torch.Tensor:
    """
    Build an alternating object schedule over time.

    For two objects and 12 steps this becomes:
    [0, 1, 0, 1, 0, 1, ...]
    """
    return torch.arange(steps, dtype=torch.long) % num_objects


def sample_disjoint_object_specs(
    rng: np.random.Generator,
    image_size: int,
    patch_size: int,
    object_ids: Tuple[int, ...] = (0, 3),
) -> Tuple[Tuple[int, int, int], ...]:
    """
    Sample simple non-overlapping placements for two objects.

    We keep the placements constrained to opposite halves of the image so the
    resulting masks are clearly separated for the temporal binding task.
    """
    if len(object_ids) != 2:
        raise ValueError("Current composite sampler expects exactly two object ids.")

    max_top = max(1, image_size - patch_size)
    left_a = int(rng.integers(0, max(1, image_size // 2 - patch_size + 1)))
    left_b = int(rng.integers(max(1, image_size // 2), max_top + 1))
    top_a = int(rng.integers(0, max_top + 1))
    top_b = int(rng.integers(0, max_top + 1))
    return (
        (object_ids[0], top_a, left_a),
        (object_ids[1], top_b, left_b),
    )


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


class CompositeTemporalBindingDataset(Dataset):
    """
    Dataset for learning time-dependent object selection in composite scenes.

    Each sample contains:
    - one image with two separated objects,
    - a simple class label for the first object,
    - object masks for the two objects,
    - a time schedule describing which object should dominate at each step.
    """

    def __init__(
        self,
        num_samples: int = 120,
        image_size: int = 16,
        patch_size: int = 6,
        steps: int = 12,
        noise_std: float = 0.02,
        seed: int = 7,
        object_ids: Tuple[int, int] = (0, 3),
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)

        images = []
        labels = []
        masks = []
        schedules = []
        object_name_lists = []

        for _ in range(num_samples):
            sampled_ids = object_ids if float(rng.random()) < 0.5 else tuple(reversed(object_ids))
            specs = sample_disjoint_object_specs(
                rng,
                image_size=image_size,
                patch_size=patch_size,
                object_ids=sampled_ids,
            )
            image = build_composite_scene(
                image_size=image_size,
                object_specs=specs,
                patch_size=patch_size,
            ).numpy()
            image += rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
            image = np.clip(image, 0.0, 1.0)
            object_masks, object_names = build_composite_masks(
                image_size=image_size,
                object_specs=specs,
                patch_size=patch_size,
            )

            images.append(image)
            labels.append(0 if specs[0][0] == object_ids[0] else 1)
            masks.append(object_masks.numpy())
            schedules.append(make_temporal_object_schedule(num_objects=len(specs), steps=steps).numpy())
            object_name_lists.append(object_names)

        self.images = torch.from_numpy(np.stack(images, axis=0)).float()
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.object_masks = torch.from_numpy(np.stack(masks, axis=0)).float()
        self.time_targets = torch.from_numpy(np.stack(schedules, axis=0)).long()
        self.object_names = object_name_lists[0] if object_name_lists else ["object_0", "object_1"]

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx], self.object_masks[idx], self.time_targets[idx]


class CompositePhaseBindingDataset(Dataset):
    """
    Simpler dataset for phase-binding experiments.

    Each sample contains a multi-object image and a pixel-wise object-id mask.
    This mirrors the toy proof-of-concept in the TINGTING simulation more
    closely than the SNN-oriented temporal-binding dataset.
    """

    def __init__(
        self,
        num_samples: int = 200,
        image_size: int = 28,
        patch_size: int = 10,
        noise_std: float = 0.05,
        seed: int = 7,
        object_ids: Tuple[int, int] = (0, 3),
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)

        images = []
        masks = []
        for _ in range(num_samples):
            sampled_ids = object_ids if float(rng.random()) < 0.5 else tuple(reversed(object_ids))
            specs = sample_disjoint_object_specs(
                rng,
                image_size=image_size,
                patch_size=patch_size,
                object_ids=sampled_ids,
            )
            image = build_composite_scene(
                image_size=image_size,
                object_specs=specs,
                patch_size=patch_size,
            ).numpy()
            image += rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
            image = np.clip(image, 0.0, 1.0)

            pixel_mask = np.zeros((image_size, image_size), dtype=np.int64)
            for local_idx, (_, top, left) in enumerate(specs, start=1):
                patch_mask = make_object_pattern(sampled_ids[local_idx - 1], patch_size)
                patch_h, patch_w = patch_mask.shape
                bottom = min(top + patch_h, image_size)
                right = min(left + patch_w, image_size)
                valid_h = max(0, bottom - top)
                valid_w = max(0, right - left)
                if valid_h > 0 and valid_w > 0:
                    region = patch_mask[:valid_h, :valid_w] > 0
                    target = pixel_mask[top:bottom, left:right]
                    target[region] = local_idx
                    pixel_mask[top:bottom, left:right] = target

            images.append(image)
            masks.append(pixel_mask)

        self.images = torch.from_numpy(np.stack(images, axis=0)).float()
        self.masks = torch.from_numpy(np.stack(masks, axis=0)).long()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], self.masks[idx]
