from __future__ import annotations

# This file is the central place for model and training hyperparameters.
# The goal is to keep "what values we use" separate from "how the model works."
# That makes it easier for new collaborators to tune the system without digging
# through model code.

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ObjectRepresentationConfig:
    """
    Small container that holds all hyperparameters for the object-representation pipeline.

    A dataclass is useful here because:
    1. It keeps related settings in one object.
    2. It gives us readable defaults.
    3. The model can receive one config object instead of many loose arguments.
    """

    # Input image size. The current synthetic dataset uses square RGB images.
    image_height: int = 16
    image_width: int = 16
    input_channels: int = 3

    # Oscillator representation size per spatial location.
    # Each image pixel/location is mapped to an oscillator vector of length `osc_dim`.
    osc_dim: int = 4

    # Classification target size for the synthetic objects.
    num_classes: int = 5

    # Number of recurrent time steps to unroll the oscillator-SNN system.
    steps: int = 12

    # Euler integration step size for the Kuramoto dynamics.
    dt: float = 0.15

    # Global coupling strength between oscillators.
    coupling: float = 1.0

    # Membrane potential carry-over in the SNN update.
    membrane_decay: float = 0.92

    # Spike threshold and reset strength for the spiking layer.
    threshold: float = 0.6
    reset_scale: float = 0.6

    # Scale for phase-lag feedback and delay used in the sinusoidal gate.
    alpha_scale: float = 1.0
    delay: int = 2

    # Noise used when generating synthetic object images.
    noise_std: float = 0.01

    # Standard training hyperparameters.
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 7

    # Number of synthetic samples to generate for toy experiments.
    num_samples: int = 100
    composite_num_samples: int = 120
    composite_patch_size: int = 6

    # Loss weights for the composite temporal-binding training mode.
    classification_loss_weight: float = 0.0
    binding_loss_weight: float = 1.0
    oscillation_loss_weight: float = 1.0
    competition_loss_weight: float = 0.5
    outside_loss_weight: float = 0.25
    sparsity_loss_weight: float = 0.05
    oscillation_period: int = 6
    spike_logit_scale: float = 40.0
    gate_logit_scale: float = 8.0
    phase_logit_scale: float = 8.0

    # Default execution device.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs"
    experiment_name: str = "composite_binding"
    checkpoint_name: str = "composite_binding.pt"

    @property
    def num_nodes(self) -> int:
        """Total number of spatial positions once an image is flattened."""
        return self.image_height * self.image_width

    @property
    def experiment_dir(self) -> Path:
        """Directory where training artifacts for this run should be stored."""
        return Path(self.output_dir) / self.experiment_name


def build_parser() -> argparse.ArgumentParser:
    """
    Build a command-line parser with the same fields as the dataclass above.

    This lets someone run a training or analysis script and override defaults
    from the terminal without editing Python files directly.
    """
    parser = argparse.ArgumentParser(description="Object representation SNN configuration")
    parser.add_argument("--image_height", type=int, default=16)
    parser.add_argument("--image_width", type=int, default=16)
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--osc_dim", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.15)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--membrane_decay", type=float, default=0.92)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--reset_scale", type=float, default=0.6)
    parser.add_argument("--alpha_scale", type=float, default=1.0)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--composite_num_samples", type=int, default=120)
    parser.add_argument("--composite_patch_size", type=int, default=6)
    parser.add_argument("--classification_loss_weight", type=float, default=0.0)
    parser.add_argument("--binding_loss_weight", type=float, default=1.0)
    parser.add_argument("--oscillation_loss_weight", type=float, default=1.0)
    parser.add_argument("--competition_loss_weight", type=float, default=0.5)
    parser.add_argument("--outside_loss_weight", type=float, default=0.25)
    parser.add_argument("--sparsity_loss_weight", type=float, default=0.05)
    parser.add_argument("--oscillation_period", type=int, default=6)
    parser.add_argument("--spike_logit_scale", type=float, default=40.0)
    parser.add_argument("--gate_logit_scale", type=float, default=8.0)
    parser.add_argument("--phase_logit_scale", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--experiment_name", type=str, default="composite_binding")
    parser.add_argument("--checkpoint_name", type=str, default="composite_binding.pt")
    return parser


def get_default_config() -> ObjectRepresentationConfig:
    """Return a config object with the default values defined in the dataclass."""
    return ObjectRepresentationConfig()


def parse_args(argv: Optional[list[str]] = None) -> ObjectRepresentationConfig:
    """
    Parse optional command-line arguments and convert them into our config object.

    `argv=None` means "read from the real command line".
    Passing a list is convenient for tests or notebooks.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return ObjectRepresentationConfig(**vars(args))
