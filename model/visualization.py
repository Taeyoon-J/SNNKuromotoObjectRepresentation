from __future__ import annotations

# This file collects plotting utilities for understanding the model qualitatively.
# Separating plotting code from model code keeps the main architecture easier to read.

import math
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import torch


def plot_activation_function(activation_fn, value_range: Tuple[float, float] = (-3.0, 3.0), points: int = 200):
    """Plot the activation function g used in the top-down pathway."""
    xs = torch.linspace(value_range[0], value_range[1], points)
    ys = activation_fn(xs)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xs.cpu().numpy(), ys.detach().cpu().numpy(), linewidth=2)
    ax.set_title("Activation Function g")
    ax.set_xlabel("Input")
    ax.set_ylabel("g(x)")
    ax.grid(alpha=0.25)
    return fig


def plot_loss_curve(losses: List[float]):
    """Plot training loss across epochs."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses, linewidth=2)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    return fig


def visualize_objects(images: torch.Tensor, labels: torch.Tensor, max_items: int = 12):
    """Show a grid of sample synthetic objects and their labels."""
    count = min(max_items, images.shape[0])
    cols = 4
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx < count:
            # Convert each tensor image back to NumPy for matplotlib.
            ax.imshow(images[idx].detach().cpu().numpy())
            ax.set_title(f"class={int(labels[idx])}")

    fig.suptitle("Predefined Objects", y=1.02)
    fig.tight_layout()
    return fig


def visualize_dynamics(history: Dict[str, torch.Tensor], sample_idx: int = 0):
    """
    Plot a few averaged internal signals over time for one sample.

    We average across nodes/dimensions here so the figure stays readable.
    """
    # Mean gamma magnitude over time.
    gamma = history["gamma"][sample_idx].mean(dim=(1, 2)).detach().cpu().numpy()
    # Mean gate value over time.
    gate = history["gate"][sample_idx].mean(dim=(1, 2)).detach().cpu().numpy()
    # Mean membrane potential over time.
    membrane = history["membrane"][sample_idx].mean(dim=1).detach().cpu().numpy()
    # Mean spike activity over time.
    spikes = history["spikes"][sample_idx].mean(dim=1).detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    series = [
        (gamma, "Readout Gamma"),
        (gate, "Sinusoidal Gate g(t)"),
        (membrane, "Membrane U(t)"),
        (spikes, "Spike Activity S(t)"),
    ]
    for ax, (values, title) in zip(axes.flatten(), series):
        ax.plot(values, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def visualize_binding_metrics(
    intra_sync: torch.Tensor,
    inter_sync: torch.Tensor,
    object_names: List[str],
):
    """Plot Kuramoto binding metrics over time."""
    steps = intra_sync.shape[1]
    time_axis = np.arange(1, steps + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    for obj_idx in range(intra_sync.shape[0]):
        label = object_names[obj_idx] if obj_idx < len(object_names) else f"object_{obj_idx}"
        axes[0].plot(time_axis, intra_sync[obj_idx].detach().cpu().numpy(), linewidth=2, label=label)
    axes[0].set_title("Intra-Object Synchrony")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("order parameter")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(time_axis, inter_sync.detach().cpu().numpy(), linewidth=2, color="black")
    axes[1].set_title("Inter-Object Synchrony")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("phase similarity")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    return fig


def visualize_scheduled_kuramoto_readout(
    input_image: torch.Tensor,
    gamma0: torch.Tensor,
    history: Dict[str, torch.Tensor],
    image_height: int,
    image_width: int,
    input_channels: int,
    steps_to_show: List[int],
    object_masks: torch.Tensor | None = None,
    label_map: torch.Tensor | None = None,
):
    """
    Show Kuramoto phase maps together with gamma and spike maps at scheduled steps.

    `steps_to_show` uses 1-based model time indices, e.g. [20, 40].

    The model has one vector oscillator per pixel-channel:
        [H * W * C, D]

    A direct 2D image must compress both C and D. To make the compression
    explicit, theta/gamma are visualized as phase-color maps:
        hue        = vector phase angle atan2(dim_1, dim_0)
        brightness = agreement across the RGB-channel oscillators at a pixel

    This preserves the most important qualitative information: whether pixels
    are moving together in oscillator phase, and where RGB-channel oscillators
    disagree.
    """

    def _phase_and_agreement(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress [H*W*C, D] vectors into per-pixel circular phase maps."""
        vectors = values.view(image_height, image_width, input_channels, -1).detach().cpu()
        if vectors.shape[-1] < 2:
            phase = vectors[..., 0]
        else:
            phase = torch.atan2(vectors[..., 1], vectors[..., 0])

        # Circular mean over the three RGB-channel oscillators at each pixel.
        sin_mean = torch.sin(phase).mean(dim=-1)
        cos_mean = torch.cos(phase).mean(dim=-1)
        mean_phase = torch.atan2(sin_mean, cos_mean)
        agreement = torch.sqrt(sin_mean.pow(2) + cos_mean.pow(2)).clamp(0.0, 1.0)
        return mean_phase, agreement

    def vector_phase_to_rgb(values: torch.Tensor) -> np.ndarray:
        """Map vector phase to HSV: hue=phase, value=RGB-channel agreement."""
        phase, agreement = _phase_and_agreement(values)
        hue = ((phase / (2.0 * math.pi)) + 0.5).numpy()
        saturation = np.ones_like(hue) * 0.9
        value = agreement.numpy()
        return hsv_to_rgb(np.stack([hue, saturation, value], axis=-1))

    def vector_agreement_map(values: torch.Tensor) -> np.ndarray:
        """Return how similar the three RGB-channel oscillator phases are."""
        _, agreement = _phase_and_agreement(values)
        return agreement.numpy()

    def vector_component_rgb(values: torch.Tensor) -> np.ndarray:
        """
        Show the first three vector components as RGB after min-max scaling.

        This is not object color. It is a diagnostic view of the learned vector
        representation direction, useful for gamma(0) and gamma(t).
        """
        vectors = values.view(image_height, image_width, input_channels, -1).mean(dim=2).detach().cpu()
        if vectors.shape[-1] < 3:
            pad = torch.zeros(*vectors.shape[:-1], 3 - vectors.shape[-1])
            vectors = torch.cat([vectors, pad], dim=-1)
        rgb = vectors[..., :3]
        vmin = rgb.amin(dim=(0, 1), keepdim=True)
        vmax = rgb.amax(dim=(0, 1), keepdim=True)
        return ((rgb - vmin) / (vmax - vmin).clamp_min(1e-6)).numpy()

    def spike_to_map(spikes: torch.Tensor) -> np.ndarray:
        if spikes.shape[-1] == image_height * image_width:
            image = spikes.view(image_height, image_width).detach().cpu()
        else:
            image = spikes.view(image_height, image_width, input_channels).mean(dim=-1).detach().cpu()
        vmin = float(image.min())
        vmax = float(image.max())
        return ((image - vmin) / max(vmax - vmin, 1e-6)).numpy()

    def add_mask_contours(ax) -> None:
        if object_masks is None:
            return
        masks = object_masks.detach().cpu()
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        valid_masks = masks.reshape(masks.shape[0], -1).sum(dim=1) > 0
        for mask in masks[valid_masks][:10]:
            ax.contour(mask.numpy(), levels=[0.5], linewidths=0.45, colors="white", alpha=0.85)

    cols = len(steps_to_show) + 2
    fig, axes = plt.subplots(5, cols, figsize=(2.8 * cols, 12.0))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(input_image.detach().cpu().numpy())
    axes[0, 0].set_title("Input RGB")
    axes[0, 0].axis("off")
    add_mask_contours(axes[0, 0])

    if label_map is not None:
        axes[1, 0].imshow(label_map.detach().cpu().numpy(), cmap="tab20")
        axes[1, 0].set_title("GT object IDs")
        axes[1, 0].axis("off")
    else:
        axes[1, 0].axis("off")

    axes[2, 0].imshow(vector_component_rgb(gamma0))
    axes[2, 0].set_title("gamma(0)\nRGB = vec dims 0..2")
    axes[2, 0].axis("off")
    add_mask_contours(axes[2, 0])

    axes[3, 0].imshow(vector_phase_to_rgb(gamma0))
    axes[3, 0].set_title("gamma(0) phase\nhue=phase, value=RGB sync")
    axes[3, 0].axis("off")
    add_mask_contours(axes[3, 0])

    axes[4, 0].imshow(vector_agreement_map(gamma0), cmap="viridis", vmin=0.0, vmax=1.0)
    axes[4, 0].set_title("gamma(0)\nRGB-channel sync")
    axes[4, 0].axis("off")
    add_mask_contours(axes[4, 0])

    for row in range(5):
        axes[row, 1].axis("off")
    axes[0, 1].text(
        0.0,
        0.95,
        "Legend\n"
        "theta/gamma phase:\n"
        "  hue = oscillator phase\n"
        "  bright = RGB nodes agree\n"
        "spike:\n"
        "  black = quiet\n"
        "  yellow/white = high spike\n"
        "white lines = GT masks",
        va="top",
        fontsize=9,
    )

    for col_idx, step in enumerate(steps_to_show, start=2):
        hist_idx = step - 1

        axes[0, col_idx].imshow(vector_phase_to_rgb(history["theta"][0, hist_idx]))
        axes[0, col_idx].set_title(f"theta({step}) phase")
        axes[0, col_idx].axis("off")
        add_mask_contours(axes[0, col_idx])

        axes[1, col_idx].imshow(vector_component_rgb(history["gamma"][0, hist_idx]))
        axes[1, col_idx].set_title(f"gamma({step}) vec RGB")
        axes[1, col_idx].axis("off")
        add_mask_contours(axes[1, col_idx])

        axes[2, col_idx].imshow(vector_phase_to_rgb(history["gamma"][0, hist_idx]))
        axes[2, col_idx].set_title(f"gamma({step}) phase")
        axes[2, col_idx].axis("off")
        add_mask_contours(axes[2, col_idx])

        axes[3, col_idx].imshow(vector_agreement_map(history["theta"][0, hist_idx]), cmap="viridis", vmin=0.0, vmax=1.0)
        axes[3, col_idx].set_title(f"theta({step}) RGB sync")
        axes[3, col_idx].axis("off")
        add_mask_contours(axes[3, col_idx])

        axes[4, col_idx].imshow(spike_to_map(history["spikes"][0, hist_idx]), cmap="magma", vmin=0.0, vmax=1.0)
        axes[4, col_idx].set_title(f"spike({step})")
        axes[4, col_idx].axis("off")
        add_mask_contours(axes[4, col_idx])

    fig.suptitle("Scheduled Kuramoto -> Gamma -> Spike (interpretable oscillator maps)", y=1.01)
    fig.tight_layout()
    return fig
