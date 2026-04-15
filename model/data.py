from __future__ import annotations

# This file creates a small synthetic object dataset.
# It is intentionally simple so the model can be tested without an external
# dataset pipeline.

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


class CLEVRObjectDataset(Dataset):
    """
    Dataset adapter for the object-centric-library CLEVR HDF5 file.

    The downloaded file stores RGB images and a single integer label map:
        image: [num_samples, 128, 128, 3]
        mask: [num_samples, 128, 128, 1]

    The model wants images as HWC tensors and the unsupervised object loss wants
    one binary mask per object. This adapter performs both conversions:
        image -> [target_size, target_size, 3], float in [0, 1]
        mask -> [max_objects, target_size, target_size], binary float masks

    If `patch_size` is set, each original image is split into fixed-size patches
    before conversion. For CLEVR 128x128 with patch_size=32 and patch_stride=32,
    one original image becomes 16 training examples.

    Mask resize must use nearest-neighbor interpolation so object IDs do not get
    mixed the way they would under bilinear interpolation.
    """

    def __init__(
        self,
        hdf5_path: str,
        target_size: int = 64,
        max_objects: int = 10,
        split: str = "train",
        train_fraction: float = 0.9,
        max_samples: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
        min_object_pixels: int = 0,
    ) -> None:
        super().__init__()
        self.hdf5_path = hdf5_path
        self.target_size = target_size
        self.max_objects = max_objects
        self.split = split
        self.train_fraction = train_fraction
        self.max_samples = max_samples
        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size
        self.min_object_pixels = min_object_pixels
        self._h5 = None

        if split not in {"train", "test", "all"}:
            raise ValueError(f"split must be 'train', 'test', or 'all', got {split!r}")

        # Read only the dataset length here; keep the HDF5 handle lazy so this
        # class remains compatible with DataLoader workers.
        h5py = self._require_h5py()
        with h5py.File(self.hdf5_path, "r") as h5_file:
            total_images = int(h5_file["image"].shape[0])
            image_height = int(h5_file["image"].shape[1])
            image_width = int(h5_file["image"].shape[2])

            train_count = int(total_images * train_fraction)
            if split == "train":
                source_indices = list(range(train_count))
            elif split == "test":
                source_indices = list(range(train_count, total_images))
            else:
                source_indices = list(range(total_images))

            self.examples = self._build_examples(source_indices, image_height, image_width, h5_file)

    def _build_examples(
        self,
        source_indices: List[int],
        image_height: int,
        image_width: int,
        h5_file,
    ) -> List[Tuple[int, int, int]]:
        """Build `(source_idx, top, left)` entries for full images or patches."""
        max_examples = None if self.max_samples is None else int(self.max_samples)
        if self.patch_size is None:
            examples = []
            for source_idx in source_indices:
                if self._patch_has_enough_object_pixels(h5_file, source_idx, 0, 0, image_height, image_width):
                    examples.append((source_idx, 0, 0))
                    if max_examples is not None and len(examples) >= max_examples:
                        break
            return examples

        if self.patch_size <= 0 or self.patch_stride is None or self.patch_stride <= 0:
            raise ValueError("patch_size and patch_stride must be positive when patch mode is enabled")
        if self.patch_size > image_height or self.patch_size > image_width:
            raise ValueError(f"patch_size={self.patch_size} is larger than image shape {(image_height, image_width)}")

        examples = []
        top_positions = range(0, image_height - self.patch_size + 1, self.patch_stride)
        left_positions = range(0, image_width - self.patch_size + 1, self.patch_stride)
        for source_idx in source_indices:
            for top in top_positions:
                for left in left_positions:
                    if self._patch_has_enough_object_pixels(h5_file, source_idx, top, left, self.patch_size, self.patch_size):
                        examples.append((source_idx, top, left))
                        if max_examples is not None and len(examples) >= max_examples:
                            return examples
        return examples

    def _patch_has_enough_object_pixels(
        self,
        h5_file,
        source_idx: int,
        top: int,
        left: int,
        height: int,
        width: int,
    ) -> bool:
        """Return whether a candidate patch has enough non-background mask pixels."""
        if self.min_object_pixels <= 0:
            return True

        mask_patch = h5_file["mask"][source_idx, top : top + height, left : left + width, 0]
        return int(np.count_nonzero(mask_patch)) >= int(self.min_object_pixels)

    @staticmethod
    def _require_h5py():
        """Import h5py lazily so synthetic-data users do not need it installed."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("CLEVRObjectDataset requires h5py. Install it with `pip install h5py`.") from exc
        return h5py

    def _file(self):
        """Open the HDF5 file lazily and reuse the handle within one process."""
        if self._h5 is None:
            h5py = self._require_h5py()
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __getstate__(self):
        """
        Drop the open HDF5 handle when DataLoader workers pickle the dataset.

        Each worker should open its own handle instead of sharing one inherited
        from the parent process.
        """
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __len__(self) -> int:
        """Number of examples in the selected split."""
        return len(self.examples)

    def _resize_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert a uint8 HWC image to a float HWC tensor at target_size."""
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = F.interpolate(
            image_tensor,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )
        return image_tensor.squeeze(0).permute(1, 2, 0).contiguous()

    def _resize_label_map(self, label_map: np.ndarray) -> torch.Tensor:
        """Resize an integer object-id mask with nearest-neighbor interpolation."""
        mask_tensor = torch.from_numpy(label_map.astype(np.int64)).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=(self.target_size, self.target_size), mode="nearest")
        return mask_tensor.squeeze(0).squeeze(0).long()

    def _build_object_masks(self, label_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert one label map into padded binary masks and a validity vector.

        Background label 0 is ignored. If an image has fewer than max_objects,
        the remaining mask slots stay all-zero and are marked invalid.
        """
        object_ids = [int(value) for value in torch.unique(label_map).tolist() if int(value) != 0]
        object_ids = object_ids[: self.max_objects]

        object_masks = torch.zeros(self.max_objects, self.target_size, self.target_size, dtype=torch.float32)
        valid_objects = torch.zeros(self.max_objects, dtype=torch.bool)
        for slot, object_id in enumerate(object_ids):
            object_masks[slot] = (label_map == object_id).float()
            valid_objects[slot] = True

        return object_masks, valid_objects

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetch one CLEVR image plus object masks for unsupervised binding loss."""
        source_idx, top, left = self.examples[idx]
        h5_file = self._file()

        raw_image = h5_file["image"][source_idx]
        raw_label_map = h5_file["mask"][source_idx, :, :, 0]
        if self.patch_size is not None:
            bottom = top + self.patch_size
            right = left + self.patch_size
            raw_image = raw_image[top:bottom, left:right]
            raw_label_map = raw_label_map[top:bottom, left:right]

        image = self._resize_image(raw_image)
        label_map = self._resize_label_map(raw_label_map)
        object_masks, valid_objects = self._build_object_masks(label_map)
        num_objects = torch.tensor(int(valid_objects.sum().item()), dtype=torch.long)

        return {
            "image": image,
            "object_masks": object_masks,
            "valid_objects": valid_objects,
            "label_map": label_map,
            "num_objects": num_objects,
            "source_idx": torch.tensor(source_idx, dtype=torch.long),
            "patch_top": torch.tensor(top, dtype=torch.long),
            "patch_left": torch.tensor(left, dtype=torch.long),
        }
