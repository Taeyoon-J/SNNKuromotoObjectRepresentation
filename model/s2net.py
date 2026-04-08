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
    1. Encode the image into per-location oscillator features.
    2. Build the readout/control signal gamma(t).
    3. Generate delayed sinusoidal gates.
    4. Update the Kuramoto state.
    5. Build feedback-derived affinity and phase lag terms.
    """

    def __init__(self, config: ObjectRepresentationConfig) -> None:
        super().__init__()
        self.config = config
        self.num_nodes = config.num_nodes
        self.osc_dim = config.osc_dim

        # Converts RGB features at each image location into an oscillator vector.
        self.vector_encoder = nn.Linear(config.input_channels, config.osc_dim)
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
        )

    def vector_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the image and map each spatial position to oscillator features."""
        b, h, w, c = x.shape
        if h != self.config.image_height or w != self.config.image_width or c != self.config.input_channels:
            raise ValueError(
                f"Expected input shape [B, {self.config.image_height}, {self.config.image_width}, {self.config.input_channels}], "
                f"got {tuple(x.shape)}"
            )
        # [B, H, W, C] -> [B, H*W, C] so each node corresponds to one image position.
        return self.vector_encoder(x.view(b, h * w, c))

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """Nonlinear activation g used in the gamma/readout stage."""
        return torch.tanh(x)

    def readout_gamma_function(self, encoded: torch.Tensor, theta_prev: torch.Tensor) -> torch.Tensor:
        """
        Build gamma(t), the control/readout signal that drives oscillator updates.

        We combine encoded input features with a projection of the previous
        oscillator state, then pass them through the activation function.
        """
        theta_proj = self.gamma_readout(theta_prev)
        return self.activation_function(torch.abs(encoded + theta_proj) * self.gamma_gain)

    def sinusoidal_gating_function(self, theta_delayed: torch.Tensor) -> torch.Tensor:
        """
        Convert delayed phases into a gate in the range [0, 1].

        This is the bridge from Kuramoto dynamics to the SNN input modulation.
        """
        return 0.5 * (1.0 + torch.sin(theta_delayed))

    def top_down_feedback_function(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Derive pairwise affinity and phase lag from the current spiking pattern.

        We separate two ideas here:
        1. `affinity` should be large when two nodes behave similarly, because
           similar activity should encourage synchronization.
        2. `alpha` should reflect mismatch/cost, because larger disagreement can
           be interpreted as a larger phase delay.
        """
        # Build all pairwise spike differences: [B, N, 1] vs [B, 1, N].
        spike_i = spikes.unsqueeze(2)
        spike_j = spikes.unsqueeze(1)
        delta = torch.abs(spike_i - spike_j)

        # Normalize pairwise mismatch to [0, 1] so the feedback stays stable.
        norm_delta = delta / delta.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)

        # Similar nodes receive stronger coupling; dissimilar nodes are down-weighted.
        affinity = 1.0 - norm_delta

        # Remove self-coupling from the feedback graph and keep the scale bounded.
        eye = torch.eye(self.num_nodes, device=spikes.device).unsqueeze(0)
        affinity = affinity * (1.0 - eye)

        # Phase lag grows with mismatch rather than similarity.
        alpha = self.config.alpha_scale * norm_delta
        return affinity, alpha

    def forward_step(
        self,
        encoded: torch.Tensor,
        theta_prev: torch.Tensor,
        theta_history: List[torch.Tensor],
        affinity: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advance the top-down pathway by one time step.

        Returns:
            theta_t: updated oscillator state
            gamma_t: current readout/control signal
            gate_t: delayed sinusoidal gate used by the SNN
        """
        gamma_t = self.readout_gamma_function(encoded, theta_prev)

        # Use an exact t-d style delay when enough history exists.
        # Before that point, we fall back to the current state because a true
        # delayed state is not available yet.
        if self.config.delay > 0 and len(theta_history) >= self.config.delay:
            theta_delayed = theta_history[-self.config.delay]
        else:
            theta_delayed = theta_prev
        gate_t = self.sinusoidal_gating_function(theta_delayed)
        theta_t = self.kuramoto(theta_prev, gamma_t, affinity, alpha)
        return theta_t, gamma_t, gate_t


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
            reset_scale=self.config.reset_scale,
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
        # Initial per-node oscillator features extracted from the image.
        encoded = self.top_down.vector_encode(x)
        batch_size = x.shape[0]
        device = x.device

        # Initialize all recurrent states at time t=0.
        theta = torch.zeros(batch_size, self.num_nodes, self.config.osc_dim, device=device)
        membrane = torch.zeros(batch_size, self.num_nodes, device=device)
        spikes = torch.zeros(batch_size, self.num_nodes, device=device)

        # Start with identity-like affinity: each node is initially only certain about itself.
        affinity = torch.eye(self.num_nodes, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        alpha = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=device)

        # `theta_hist` is needed even without history return because delayed gating
        # depends on previous oscillator states.
        theta_hist: List[torch.Tensor] = []

        # Only allocate the remaining history buffers when the caller explicitly
        # asks for analysis outputs.
        gamma_hist: Optional[List[torch.Tensor]] = [] if return_history else None
        gate_hist: Optional[List[torch.Tensor]] = [] if return_history else None
        membrane_hist: Optional[List[torch.Tensor]] = [] if return_history else None
        spike_hist: List[torch.Tensor] = []
        affinity_hist: Optional[List[torch.Tensor]] = [] if return_history else None
        alpha_hist: Optional[List[torch.Tensor]] = [] if return_history else None

        for _ in range(self.config.steps):
            # 1. Evolve the oscillator system and compute the gate.
            theta, gamma_t, gate_t = self.top_down.forward_step(
                encoded,
                theta,
                theta_hist,
                affinity,
                alpha,
            )

            # 2. Use the gate to modulate the oscillator readout before sending it to the SNN.
            modulated_gamma = gate_t * gamma_t

            # 3. Update membrane potentials and spike responses.
            membrane, spikes = self.bottom_up.forward_step(membrane, spikes, modulated_gamma)

            # 4. Feed the current spiking pattern back into the oscillator coupling terms.
            affinity, alpha = self.top_down.top_down_feedback_function(spikes)

            # Keep phase history for delayed gating.
            theta_hist.append(theta)
            spike_hist.append(spikes)

            if return_history:
                gamma_hist.append(gamma_t)
                gate_hist.append(gate_t)
                membrane_hist.append(membrane)
                affinity_hist.append(affinity)
                alpha_hist.append(alpha)

        # Stack spikes over time to obtain a [B, T, N] spike trace.
        spike_trace = torch.stack(spike_hist, dim=1)
        # Final classification based on pooled spike activity.
        logits = self.bottom_up.classify(spike_trace)

        if not return_history:
            return logits, {}

        # Build the analysis dictionary only when requested.
        history: Dict[str, torch.Tensor] = {
            "encoded": encoded,
            "theta": torch.stack(theta_hist, dim=1),
            "gamma": torch.stack(gamma_hist, dim=1),
            "gate": torch.stack(gate_hist, dim=1),
            "membrane": torch.stack(membrane_hist, dim=1),
            "spikes": spike_trace,
            "affinity": torch.stack(affinity_hist, dim=1),
            "alpha": torch.stack(alpha_hist, dim=1),
        }
        return logits, history
