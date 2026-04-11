from __future__ import annotations

# This file contains the main end-to-end model.
# It mirrors the role of `s2net.py` in the TINGTING folder:
# top-down oscillator dynamics + bottom-up spiking readout.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .hyperparameters import ObjectRepresentationConfig
    from .layers import GraphVectorKuramotoLayer, ObjectReadoutSNN
except ImportError:
    from hyperparameters import ObjectRepresentationConfig
    from layers import GraphVectorKuramotoLayer, ObjectReadoutSNN


class TopDownPathway(nn.Module):
    """
    Top-down oscillator pathway.

    Responsibilities:
    1. Initialize gamma(0) from the input image.
    2. Maintain the oscillator state.
    3. Update the Kuramoto state.
    4. Build gamma(t>0) from theta(t) only at scheduled readout steps.
    5. Generate delayed sinusoidal gates when readout/spike updates happen.
    6. Build feedback-derived affinity and phase lag terms.
    """

    def __init__(self, config: ObjectRepresentationConfig) -> None:
        super().__init__()
        self.config = config
        self.num_nodes = config.num_nodes
        self.osc_dim = config.osc_dim

        # Input is used only to seed gamma(0), not to initialize theta(0).
        self.gamma_initializer = nn.Linear(1, config.osc_dim)
        # Lets the current oscillator state influence the next gamma signal.
        self.gamma_readout = nn.Linear(config.osc_dim, config.osc_dim)
        # Per-node scaling for gamma.
        self.gamma_gain = nn.Parameter(torch.ones(1, self.num_nodes, config.osc_dim))
        # Core Kuramoto dynamics block.
        self.kuramoto = GraphVectorKuramotoLayer(
            num_nodes=self.num_nodes,
            osc_dim=config.osc_dim,
            coupling=config.coupling,
            dt=config.dt,
            attraction_strength=config.attraction_strength,
        )

    def validate_input(self, x: torch.Tensor) -> None:
        """Validate the input shape without using it to initialize oscillators."""
        b, h, w, c = x.shape
        if h != self.config.image_height or w != self.config.image_width or c != self.config.input_channels:
            raise ValueError(
                f"Expected input shape [B, {self.config.image_height}, {self.config.image_width}, {self.config.input_channels}], "
                f"got {tuple(x.shape)}"
            )

    def initialize_gamma_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build gamma(0) from the input image only.

        Each pixel-channel entry becomes one oscillator node, but this input
        drive is used only for the initial readout seed.
        """
        self.validate_input(x)
        b, h, w, c = x.shape
        flat_input = x.reshape(b, h * w * c, 1)
        gamma0 = self.gamma_initializer(flat_input)
        return self.activation_function(torch.abs(gamma0) * self.gamma_gain)

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """Nonlinear activation g used in the gamma/readout stage."""
        return torch.tanh(x)

    def readout_gamma_function(self, theta_state: torch.Tensor) -> torch.Tensor:
        """
        Build gamma(t), the control/readout signal that drives oscillator updates.

        Gamma is defined from the oscillator state itself, not from an input
        encoding used to initialize oscillators.
        """
        theta_proj = self.gamma_readout(theta_state)
        return self.activation_function(torch.abs(theta_proj) * self.gamma_gain)

    def sinusoidal_gating_function(self, theta_delayed: torch.Tensor) -> torch.Tensor:
        """
        Convert delayed phases into a gate in the range [0, 1].

        This is the bridge from Kuramoto dynamics to the SNN input modulation.
        """
        return 0.5 * (1.0 + torch.sin(theta_delayed))

    def top_down_feedback_function(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Derive pairwise affinity and phase lag from the current spiking pattern.

        The exact form can still be refined later, but this gives us a runnable
        feedback mechanism now.
        """
        # Build all pairwise spike differences: [B, N, 1] vs [B, 1, N].
        spike_i = spikes.unsqueeze(2)
        spike_j = spikes.unsqueeze(1)
        delta = torch.abs(spike_i - spike_j)

        # Larger spike mismatch produces stronger pairwise signal.
        affinity = torch.log1p(delta)
        # Normalize so values stay in a stable range across batches.
        affinity = affinity / affinity.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)

        # Convert affinity into a phase-lag term.
        alpha = self.config.alpha_scale * (1.0 - affinity)
        return affinity, alpha

    def update_theta(
        self,
        theta_prev: torch.Tensor,
        gamma_prev: torch.Tensor,
        affinity: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Advance theta by one step using the most recent gamma."""
        return self.kuramoto(theta_prev, gamma_prev, affinity, alpha)

    def build_gate_from_history(self, theta_history: List[torch.Tensor], theta_current: torch.Tensor) -> torch.Tensor:
        """Build the sinusoidal gate only when a readout/spike update is scheduled."""
        if theta_history:
            delayed_idx = max(0, len(theta_history) - self.config.delay)
            theta_delayed = theta_history[delayed_idx]
        else:
            theta_delayed = theta_current
        return self.sinusoidal_gating_function(theta_delayed)


class ObjectRepresentationSNN(nn.Module):
    """
    Full object-representation network.

    High-level flow:
    1. Encode image into oscillator features.
    2. Unroll top-down Kuramoto dynamics for `steps` time steps.
    3. Use delayed oscillator gates to modulate SNN input.
    4. Update membrane potentials and spikes over time.
    5. Pool spikes and classify the object.
    """

    def __init__(self, config: Optional[ObjectRepresentationConfig] = None) -> None:
        super().__init__()
        self.config = config or ObjectRepresentationConfig()
        self.num_nodes = self.config.num_nodes

        # Oscillator system.
        self.top_down = TopDownPathway(self.config)
        # Spiking readout system.
        self.bottom_up = ObjectReadoutSNN(
            num_nodes=self.num_nodes,
            osc_dim=self.config.osc_dim,
            num_classes=self.config.num_classes,
            membrane_decay=self.config.membrane_decay,
            threshold=self.config.threshold,
        )

    def loss_function(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard supervised classification loss."""
        return F.cross_entropy(logits, labels)

    def build_adam_optimizer(self, lr: Optional[float] = None, weight_decay: Optional[float] = None):
        """Create an Adam optimizer using either custom values or config defaults."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr if lr is None else lr,
            weight_decay=self.config.weight_decay if weight_decay is None else weight_decay,
        )

    def forward(self, x: torch.Tensor, return_history: bool = False):
        """
        Run the full model on a batch of images.

        Args:
            x: Input images, shape [B, H, W, C]
            return_history: Whether to keep intermediate states for analysis
        """
        # Input is used only to initialize gamma(0); theta(0) remains random.
        self.top_down.validate_input(x)
        batch_size = x.shape[0]
        device = x.device

        # Initialize oscillator states randomly instead of encoding the image.
        theta = (2.0 * torch.pi) * torch.rand(
            batch_size,
            self.num_nodes,
            self.config.osc_dim,
            device=device,
        ) - torch.pi
        gamma = self.top_down.initialize_gamma_from_input(x)
        gate = torch.zeros(batch_size, self.num_nodes, self.config.osc_dim, device=device)
        membrane = torch.zeros(batch_size, self.num_nodes, device=device)
        spikes = torch.zeros(batch_size, self.num_nodes, device=device)

        # Start with identity-like affinity: each node is initially only certain about itself.
        affinity = torch.eye(self.num_nodes, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        alpha = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=device)

        # These lists store the trajectory over time for later visualization/analysis.
        theta_hist: List[torch.Tensor] = []
        gamma_hist: List[torch.Tensor] = []
        gate_hist: List[torch.Tensor] = []
        membrane_hist: List[torch.Tensor] = []
        spike_hist: List[torch.Tensor] = []
        affinity_hist: List[torch.Tensor] = []
        alpha_hist: List[torch.Tensor] = []

        interval = max(1, int(self.config.readout_update_interval))

        for step_idx in range(1, self.config.steps + 1):
            # 1. Theta evolves every step using the most recent gamma.
            theta = self.top_down.update_theta(theta, gamma, affinity, alpha)

            # 2. Gamma/gate/spike only refresh every scheduled interval.
            if step_idx % interval == 0:
                gamma = self.top_down.readout_gamma_function(theta)
                gate = self.top_down.build_gate_from_history(theta_hist, theta)
                modulated_gamma = gate * gamma
                membrane, spikes = self.bottom_up.forward_step(membrane, spikes, modulated_gamma)
                affinity, alpha = self.top_down.top_down_feedback_function(spikes)

            # Save the full state trajectory.
            theta_hist.append(theta)
            gamma_hist.append(gamma)
            gate_hist.append(gate)
            membrane_hist.append(membrane)
            spike_hist.append(spikes)
            affinity_hist.append(affinity)
            alpha_hist.append(alpha)

        # Stack spikes over time to obtain a [B, T, N] spike trace.
        spike_trace = torch.stack(spike_hist, dim=1)
        # Final classification based on pooled spike activity.
        logits = self.bottom_up.classify(spike_trace)

        # History is optional in spirit, but we build it here so analysis code can
        # inspect internal dynamics after a forward pass.
        history: Dict[str, torch.Tensor] = {
            "theta": torch.stack(theta_hist, dim=1),
            "gamma": torch.stack(gamma_hist, dim=1),
            "gate": torch.stack(gate_hist, dim=1),
            "membrane": torch.stack(membrane_hist, dim=1),
            "spikes": spike_trace,
            "affinity": torch.stack(affinity_hist, dim=1),
            "alpha": torch.stack(alpha_hist, dim=1),
        }
        return logits, history if return_history else {}
