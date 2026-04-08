from __future__ import annotations

# This file collects plotting utilities for understanding the model qualitatively.
# Separating plotting code from model code keeps the main architecture easier to read.

import math
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def visualize_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    max_items: int = 12,
):
    """
    Show input images together with ground-truth and predicted labels.

    This is the most direct qualitative check for whether the trained model is
    learning the object categories we expect.
    """
    count = min(max_items, images.shape[0])
    cols = 4
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx < count:
            ax.imshow(images[idx].detach().cpu().numpy())
            truth = int(true_labels[idx])
            pred = int(pred_labels[idx])
            color = "green" if truth == pred else "red"
            ax.set_title(f"true={truth} | pred={pred}", color=color, fontsize=10)

    fig.suptitle("Model Predictions On Input Images", y=1.02)
    fig.tight_layout()
    return fig


def visualize_spike_sequence(
    input_image: torch.Tensor,
    spike_history: torch.Tensor,
    image_height: int,
    image_width: int,
    max_steps: int = 8,
):
    """
    Show one input image next to several time-step spike maps.

    This is meant for qualitative object-representation analysis:
    we want to see whether different object-shaped spike patterns appear over time.
    """
    total_steps = spike_history.shape[0]
    steps_to_show = min(max_steps, total_steps)

    fig, axes = plt.subplots(1, steps_to_show + 1, figsize=((steps_to_show + 1) * 2.6, 3.2))
    axes = np.atleast_1d(axes)

    axes[0].imshow(input_image.detach().cpu().numpy())
    axes[0].set_title("Input")
    axes[0].axis("off")

    spike_maps = spike_history[:steps_to_show].detach().cpu().view(steps_to_show, image_height, image_width)
    vmax = float(spike_maps.max().item()) if spike_maps.numel() > 0 else 1.0
    vmax = max(vmax, 1e-6)

    for idx in range(steps_to_show):
        ax = axes[idx + 1]
        ax.imshow(spike_maps[idx].numpy(), cmap="hot", vmin=0.0, vmax=vmax)
        ax.set_title(f"Spike t={idx + 1}")
        ax.axis("off")

    fig.suptitle("Time-Resolved Spike Patterns", y=1.02)
    fig.tight_layout()
    return fig


def visualize_object_binding_scores(
    region_activations: torch.Tensor,
    overlap_scores: torch.Tensor,
    object_names: List[str],
):
    """
    Plot object-wise activation and overlap over time.

    Args:
        region_activations: [num_objects, T]
        overlap_scores: [num_objects, T]
    """
    num_objects, steps = region_activations.shape
    time_axis = np.arange(1, steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for obj_idx in range(num_objects):
        label = object_names[obj_idx] if obj_idx < len(object_names) else f"object_{obj_idx}"
        axes[0].plot(time_axis, region_activations[obj_idx].detach().cpu().numpy(), linewidth=2, label=label)
        axes[1].plot(time_axis, overlap_scores[obj_idx].detach().cpu().numpy(), linewidth=2, label=label)

    axes[0].set_title("Region Activation Over Time")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Mean Spike In Object Region")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Mask Overlap Score Over Time")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Fraction Of Spike Mass In Region")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    return fig


def visualize_kuramoto_readout(
    input_image: torch.Tensor,
    theta_history: torch.Tensor,
    gamma_history: torch.Tensor,
    gate_history: torch.Tensor,
    image_height: int,
    image_width: int,
    max_steps: int = 4,
):
    """
    Visualize the Kuramoto-side readout path:
    input -> phase-derived map -> gamma map -> gate map.

    This is useful for checking whether the readout layer preserves meaningful
    object structure or washes it out before the SNN sees it.
    """
    total_steps = theta_history.shape[0]
    steps_to_show = min(max_steps, total_steps)

    fig, axes = plt.subplots(3, steps_to_show + 1, figsize=((steps_to_show + 1) * 2.6, 8.0))
    axes = np.atleast_2d(axes)

    phase_maps = torch.sin(theta_history[:steps_to_show]).mean(dim=-1).detach().cpu().view(steps_to_show, image_height, image_width)
    gamma_maps = gamma_history[:steps_to_show].mean(dim=-1).detach().cpu().view(steps_to_show, image_height, image_width)
    gate_maps = gate_history[:steps_to_show].mean(dim=-1).detach().cpu().view(steps_to_show, image_height, image_width)

    row_titles = ["sin(theta)", "gamma readout", "gate"]
    map_sets = [phase_maps, gamma_maps, gate_maps]

    for row_idx in range(3):
        axes[row_idx, 0].imshow(input_image.detach().cpu().numpy())
        axes[row_idx, 0].set_title("Input" if row_idx == 0 else row_titles[row_idx])
        axes[row_idx, 0].axis("off")

        vmax = float(map_sets[row_idx].abs().max().item()) if row_idx == 0 else float(map_sets[row_idx].max().item())
        if vmax < 1e-6:
            vmax = 1.0

        for t in range(steps_to_show):
            ax = axes[row_idx, t + 1]
            if row_idx == 0:
                ax.imshow(map_sets[row_idx][t].numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(map_sets[row_idx][t].numpy(), cmap="viridis", vmin=0.0, vmax=vmax)
            ax.set_title(f"t={t + 1}")
            ax.axis("off")

    fig.suptitle("Kuramoto Readout Diagnostics", y=1.02)
    fig.tight_layout()
    return fig
