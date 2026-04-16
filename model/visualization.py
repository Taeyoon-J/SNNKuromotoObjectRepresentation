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

    The model has one vector oscillator per pixel:
        [H * W, D]

    A direct 2D image must compress D. To make the compression explicit,
    theta/gamma are visualized as phase-color maps:
        hue        = vector phase angle atan2(dim_1, dim_0)
        brightness = vector magnitude or fixed brightness

    We do not plot theta magnitude because theta is unit-normalized by design,
    so its magnitude map is uninformative. Instead we plot theta-gamma cosine
    alignment to show whether theta is following the current readout drive.
    """

    def _phase_and_magnitude(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress [H*W, D] vectors into per-pixel phase and magnitude maps."""
        vectors = values.view(image_height, image_width, -1).detach().cpu()
        if vectors.shape[-1] < 2:
            phase = vectors[..., 0]
        else:
            phase = torch.atan2(vectors[..., 1], vectors[..., 0])

        magnitude = vectors.norm(dim=-1)
        magnitude = magnitude / magnitude.max().clamp_min(1e-6)
        return phase, magnitude

    def vector_phase_to_rgb(values: torch.Tensor, use_magnitude: bool = True) -> np.ndarray:
        """Map vector phase to HSV: hue=phase, value=normalized magnitude."""
        phase, magnitude = _phase_and_magnitude(values)
        hue = ((phase / (2.0 * math.pi)) + 0.5).numpy()
        saturation = np.ones_like(hue) * 0.9
        value = magnitude.numpy() if use_magnitude else np.ones_like(hue)
        return hsv_to_rgb(np.stack([hue, saturation, value], axis=-1))

    def vector_magnitude_map(values: torch.Tensor) -> np.ndarray:
        """Return normalized per-pixel vector magnitude."""
        _, magnitude = _phase_and_magnitude(values)
        return magnitude.numpy()

    def vector_component_rgb(values: torch.Tensor) -> np.ndarray:
        """
        Show the first three vector components as RGB after min-max scaling.

        This is not object color. It is a diagnostic view of the learned vector
        representation direction, useful for gamma(0) and gamma(t).
        """
        vectors = values.view(image_height, image_width, -1).detach().cpu()
        if vectors.shape[-1] < 3:
            pad = torch.zeros(*vectors.shape[:-1], 3 - vectors.shape[-1])
            vectors = torch.cat([vectors, pad], dim=-1)
        rgb = vectors[..., :3]
        vmin = rgb.amin(dim=(0, 1), keepdim=True)
        vmax = rgb.amax(dim=(0, 1), keepdim=True)
        return ((rgb - vmin) / (vmax - vmin).clamp_min(1e-6)).numpy()

    def alignment_map(theta_values: torch.Tensor, gamma_values: torch.Tensor) -> np.ndarray:
        """Cosine alignment map scaled from [-1, 1] to [0, 1]."""
        theta_vectors = theta_values.view(image_height, image_width, -1).detach().cpu()
        gamma_vectors = gamma_values.view(image_height, image_width, -1).detach().cpu()
        theta_norm = torch.nn.functional.normalize(theta_vectors, dim=-1, eps=1e-6)
        gamma_norm = torch.nn.functional.normalize(gamma_vectors, dim=-1, eps=1e-6)
        cosine = (theta_norm * gamma_norm).sum(dim=-1).clamp(-1.0, 1.0)
        return ((cosine + 1.0) * 0.5).numpy()

    def spike_to_map(spikes: torch.Tensor) -> np.ndarray:
        if spikes.shape[-1] == image_height * image_width:
            image = spikes.view(image_height, image_width).detach().cpu()
        else:
            raise ValueError(f"Expected pixel-level spikes with {image_height * image_width} nodes, got {spikes.shape[-1]}")
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

    theta0 = history.get("theta0")
    gamma0_from_history = history.get("gamma0", gamma0.unsqueeze(0)).squeeze(0)
    if theta0 is None:
        raise ValueError("history must contain theta0 for t=0 visualization")
    theta0 = theta0[0] if theta0.dim() == 3 else theta0
    gamma0_display = gamma0_from_history[0] if gamma0_from_history.dim() == 3 else gamma0_from_history

    display_steps = [0] + sorted(set([step for step in steps_to_show if step > 0]))
    cols = len(display_steps) + 1
    fig, axes = plt.subplots(5, cols, figsize=(2.9 * cols, 12.2))
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

    axes[2, 0].text(
        0.0,
        0.95,
        "Legend\n"
        "theta phase:\n"
        "  hue = oscillator phase\n"
        "  brightness fixed\n"
        "alignment:\n"
        "  black = opposite theta/gamma\n"
        "  yellow = theta follows gamma\n"
        "gamma magnitude:\n"
        "  dark = weak readout\n"
        "  bright = strong readout\n"
        "spike:\n"
        "  dark = quiet\n"
        "  bright = high spike\n"
        "white lines = GT masks",
        va="top",
        fontsize=8.5,
    )
    axes[2, 0].axis("off")

    axes[3, 0].axis("off")
    axes[4, 0].axis("off")

    for col_idx, step in enumerate(display_steps, start=1):
        if step == 0:
            theta_values = theta0
            gamma_values = gamma0_display
            spike_values = None
            title_step = "t=0"
        else:
            hist_idx = step - 1
            theta_values = history["theta"][0, hist_idx]
            gamma_values = history["gamma"][0, hist_idx]
            spike_values = history["spikes"][0, hist_idx]
            title_step = f"t={step}"

        axes[0, col_idx].imshow(vector_phase_to_rgb(theta_values, use_magnitude=False))
        axes[0, col_idx].set_title(f"{title_step}\ntheta phase")
        axes[0, col_idx].axis("off")
        add_mask_contours(axes[0, col_idx])

        axes[1, col_idx].imshow(alignment_map(theta_values, gamma_values), cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, col_idx].set_title(f"{title_step}\ntheta-gamma align")
        axes[1, col_idx].axis("off")
        add_mask_contours(axes[1, col_idx])

        axes[2, col_idx].imshow(vector_component_rgb(gamma_values))
        axes[2, col_idx].set_title(f"{title_step}\ngamma vec RGB")
        axes[2, col_idx].axis("off")
        add_mask_contours(axes[2, col_idx])

        axes[3, col_idx].imshow(vector_magnitude_map(gamma_values), cmap="magma", vmin=0.0, vmax=1.0)
        axes[3, col_idx].set_title(f"{title_step}\ngamma magnitude")
        axes[3, col_idx].axis("off")
        add_mask_contours(axes[3, col_idx])

        if spike_values is None:
            axes[4, col_idx].imshow(np.zeros((image_height, image_width)), cmap="magma", vmin=0.0, vmax=1.0)
            axes[4, col_idx].set_title("t=0\nspike not run")
        else:
            axes[4, col_idx].imshow(spike_to_map(spike_values), cmap="magma", vmin=0.0, vmax=1.0)
            axes[4, col_idx].set_title(f"{title_step}\nspike")
        axes[4, col_idx].axis("off")
        add_mask_contours(axes[4, col_idx])

    fig.suptitle("Theta/Gamma/Spike Timeline: t=0 and scheduled readout steps", y=1.01)
    fig.tight_layout()
    return fig
