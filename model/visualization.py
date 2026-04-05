from __future__ import annotations

# This file collects plotting utilities for understanding the model qualitatively.
# Separating plotting code from model code keeps the main architecture easier to read.

import math
from typing import Dict, List, Tuple

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
