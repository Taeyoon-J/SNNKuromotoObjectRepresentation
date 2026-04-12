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
):
    """
    Show Kuramoto phase maps together with gamma and spike maps at scheduled steps.

    `steps_to_show` uses 1-based model time indices, e.g. [20, 40].
    The Kuramoto row visualizes mean sin(theta), which is easier to interpret
    than raw vector-valued theta.
    """

    def node_vector_to_rgb_like(values: torch.Tensor) -> np.ndarray:
        image = values.mean(dim=-1).view(image_height, image_width, input_channels).detach().cpu()
        vmin = float(image.min())
        vmax = float(image.max())
        return ((image - vmin) / max(vmax - vmin, 1e-6)).numpy()

    def theta_to_phase_map(theta: torch.Tensor) -> np.ndarray:
        phase = torch.sin(theta).mean(dim=-1)
        phase = phase.view(image_height, image_width, input_channels).mean(dim=-1).detach().cpu()
        max_abs = float(phase.abs().max())
        return (phase / max(max_abs, 1e-6)).numpy()

    def spike_to_map(spikes: torch.Tensor) -> np.ndarray:
        image = spikes.view(image_height, image_width, input_channels).mean(dim=-1).detach().cpu()
        vmin = float(image.min())
        vmax = float(image.max())
        return ((image - vmin) / max(vmax - vmin, 1e-6)).numpy()

    cols = len(steps_to_show) + 2
    fig, axes = plt.subplots(3, cols, figsize=(2.8 * cols, 8.0))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(input_image.detach().cpu().numpy())
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    axes[0, 1].imshow(node_vector_to_rgb_like(gamma0))
    axes[0, 1].set_title("gamma(0)")
    axes[0, 1].axis("off")
    axes[1, 1].axis("off")
    axes[2, 1].axis("off")

    for col_idx, step in enumerate(steps_to_show, start=2):
        hist_idx = step - 1

        axes[0, col_idx].imshow(theta_to_phase_map(history["theta"][0, hist_idx]), cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[0, col_idx].set_title(f"sin theta({step})")
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(node_vector_to_rgb_like(history["gamma"][0, hist_idx]))
        axes[1, col_idx].set_title(f"gamma({step})")
        axes[1, col_idx].axis("off")

        axes[2, col_idx].imshow(spike_to_map(history["spikes"][0, hist_idx]), cmap="hot", vmin=0.0, vmax=1.0)
        axes[2, col_idx].set_title(f"spike({step})")
        axes[2, col_idx].axis("off")

    fig.suptitle("Scheduled Kuramoto -> Gamma -> Spike", y=1.02)
    fig.tight_layout()
    return fig
