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
        self.reset_gamma_parameters()
        # Core Kuramoto dynamics block.
        self.kuramoto = GraphVectorKuramotoLayer(
            num_nodes=self.num_nodes,
            osc_dim=config.osc_dim,
            coupling=config.coupling,
            dt=config.dt,
            attraction_strength=config.attraction_strength,
            feedback_affinity_scale=config.feedback_affinity_scale,
            feedback_alpha_scale=config.feedback_alpha_scale,
            alpha_scale=config.alpha_scale,
            coupling_chunk_size=config.coupling_chunk_size,
            input_channels=config.input_channels,
            channel_wise_coupling=config.channel_wise_coupling,
        )

    def reset_gamma_parameters(self) -> None:
        """Initialize gamma readout so image/object contrast survives early steps."""
        with torch.no_grad():
            # Monotonic image -> gamma(0): brighter input produces larger gamma.
            self.gamma_initializer.weight.fill_(1.0)
            self.gamma_initializer.bias.zero_()

            # Start theta -> gamma as identity instead of a random mixing.
            self.gamma_readout.weight.zero_()
            self.gamma_readout.bias.zero_()
            identity_dim = min(self.gamma_readout.weight.shape)
            self.gamma_readout.weight[:identity_dim, :identity_dim].copy_(torch.eye(identity_dim))

            self.gamma_gain.fill_(1.0)

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
        return self.activation_function(torch.abs(theta_proj * self.gamma_gain))
        # return self.activation_function(torch.abs(theta_proj) * self.gamma_gain)
    

    def sinusoidal_gating_function(self, theta_delayed: torch.Tensor) -> torch.Tensor:
        """
        Convert delayed phases into a gate in the range [0, 1].

        This is the bridge from Kuramoto dynamics to the SNN input modulation.
        """
        # Compress vector oscillator dimension D to one scalar phase per
        # pixel-channel oscillator: [B, H*W*C, D] -> [B, H*W*C, 1].
        theta_scalar = theta_delayed.mean(dim=-1, keepdim=True)
        return 0.5 * (1.0 + torch.sin(theta_scalar))

    def top_down_feedback_function(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return compact feedback state for pairwise affinity/phase-lag generation.

        The Kuramoto layer expands this spike vector into pairwise terms in
        chunks, avoiding a persistent [B, N, N] matrix for 64x64x3 inputs.
        """
        if spikes.shape[-1] == self.num_nodes:
            feedback_spikes = spikes
        else:
            # SNN spikes are pixel-level [B, H*W]. Expand them back to the
            # pixel-channel oscillator layout [B, H*W*C] for Kuramoto feedback.
            feedback_spikes = spikes.repeat_interleave(self.config.input_channels, dim=-1)
        return feedback_spikes, None

    def update_theta(
        self,
        theta_prev: torch.Tensor,
        gamma_prev: torch.Tensor,
        affinity: Optional[torch.Tensor],
        alpha: Optional[torch.Tensor],
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
            recurrent_scale=self.config.recurrent_scale,
            classifier_start_step=self.config.classifier_start_step,
            input_channels=self.config.input_channels,
        )

    def loss_function(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard supervised classification loss."""
        return F.cross_entropy(logits, labels)

    def _late_spike_window(self, spike_trace: torch.Tensor) -> torch.Tensor:
        """Return the spike window used by the unsupervised object losses."""
        start_idx = min(max(self.config.classifier_start_step - 1, 0), spike_trace.shape[1] - 1)
        return spike_trace[:, start_idx:, :]

    def _prepare_object_masks(self, object_masks: torch.Tensor, spike_trace: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel-level object masks into flattened node-level masks.

        Args:
            object_masks: [O, H, W] or [B, O, H, W]
            spike_trace: [B, T, H*W] for pixel spikes or [B, T, H*W*C] for node spikes

        Returns:
            node_masks: [B, O, spike_trace_nodes]
        """
        if object_masks.dim() == 3:
            object_masks = object_masks.unsqueeze(0).expand(spike_trace.shape[0], -1, -1, -1)
        if object_masks.dim() != 4:
            raise ValueError(f"Expected object masks with shape [O, H, W] or [B, O, H, W], got {tuple(object_masks.shape)}")
        if object_masks.shape[0] != spike_trace.shape[0]:
            raise ValueError(f"Mask batch size {object_masks.shape[0]} does not match spike batch size {spike_trace.shape[0]}")

        flat_masks = object_masks.to(device=spike_trace.device, dtype=spike_trace.dtype).view(object_masks.shape[0], object_masks.shape[1], -1)
        if flat_masks.shape[-1] == spike_trace.shape[-1]:
            node_masks = flat_masks
        else:
            node_masks = flat_masks.repeat_interleave(self.config.input_channels, dim=2)
        if node_masks.shape[-1] != spike_trace.shape[-1]:
            raise ValueError(f"Expanded mask has {node_masks.shape[-1]} nodes, but spike trace has {spike_trace.shape[-1]}")
        return node_masks

    def _object_activity(self, spike_trace: torch.Tensor, node_masks: torch.Tensor) -> torch.Tensor:
        """Average spike activity inside each object mask, shape [B, T, O]."""
        mask_sizes = node_masks.sum(dim=-1).clamp_min(1.0)
        return torch.einsum("btn,bon->bto", spike_trace, node_masks) / mask_sizes.unsqueeze(1)

    def object_spike_loss_components(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute reusable unsupervised object-spike loss components.

        Components:
            within_similarity: same-object pixels should share similar spike traces.
            between_difference: different objects should avoid simultaneous activity.
            object_density: each object should have at least one dense activation time.
            between_distance: object activity centers should be temporally separated.
            background_suppression: background nodes should stay quiet.
        """
        late_spikes = self._late_spike_window(spike_trace)
        node_masks = self._prepare_object_masks(object_masks, late_spikes)
        object_activity = self._object_activity(late_spikes, node_masks)

        batch_size, steps, num_objects = object_activity.shape
        eps = late_spikes.new_tensor(1e-6)
        object_valid = node_masks.sum(dim=-1) > 0.5
        valid_object_count = object_valid.to(late_spikes.dtype).sum().clamp_min(1.0)

        # 1. Within-object spike similarity: penalize variance around each object's mean activity.
        object_mean = object_activity.unsqueeze(-1)
        mask_sizes = node_masks.sum(dim=-1).clamp_min(1.0)
        squared_error = (late_spikes.unsqueeze(2) - object_mean).pow(2) * node_masks.unsqueeze(1)
        within_per_object = squared_error.sum(dim=-1) / mask_sizes.unsqueeze(1)
        within_similarity = (within_per_object * object_valid.unsqueeze(1)).sum() / (steps * valid_object_count)

        if num_objects < 2:
            zero = late_spikes.new_tensor(0.0)
            between_difference = zero
            between_distance = zero
        else:
            valid_pairs = (
                object_valid.unsqueeze(-1)
                & object_valid.unsqueeze(-2)
                & ~torch.eye(num_objects, device=late_spikes.device, dtype=torch.bool).unsqueeze(0)
            )
            valid_pair_count = valid_pairs.to(late_spikes.dtype).sum().clamp_min(1.0)

            # 2. Between-object spike difference: penalize simultaneous object activity.
            pairwise_overlap = object_activity.unsqueeze(-1) * object_activity.unsqueeze(-2)
            between_difference = (pairwise_overlap * valid_pairs.unsqueeze(1)).sum() / (steps * valid_pair_count)

            # 4. Between-object spike distance: push temporal centers of object activity apart.
            time_axis = torch.arange(steps, device=late_spikes.device, dtype=late_spikes.dtype)
            activity_mass = object_activity.sum(dim=1).clamp_min(eps)
            centers = (object_activity * time_axis.view(1, steps, 1)).sum(dim=1) / activity_mass
            pairwise_distance = torch.abs(centers.unsqueeze(-1) - centers.unsqueeze(-2))
            distance_scale = max(float(self.config.object_time_distance_scale), 1e-6)
            between_distance = (torch.exp(-pairwise_distance / distance_scale) * valid_pairs).sum() / valid_pair_count

        # 3. Object pixel density: each object should have a dense spike event at least once.
        peak_activity = object_activity.max(dim=1).values
        object_density_per_object = F.relu(float(self.config.object_density_target) - peak_activity)
        object_density = (object_density_per_object * object_valid).sum() / valid_object_count

        # 5. Background suppression, kept as an extra component for experiments.
        object_union = node_masks.amax(dim=1)
        background_mask = 1.0 - object_union
        background_size = background_mask.sum(dim=-1).clamp_min(1.0)
        background_suppression = (late_spikes * background_mask.unsqueeze(1)).sum(dim=-1) / background_size.unsqueeze(1)
        background_suppression = background_suppression.mean()

        return {
            "within_similarity": within_similarity,
            "between_difference": between_difference,
            "object_density": object_density,
            "between_distance": between_distance,
            "background_suppression": background_suppression,
        }

    def unsupervised_object_loss_1234(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        """Loss using components 1, 2, 3, and 4."""
        components = self.object_spike_loss_components(spike_trace, object_masks)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.object_density_weight * components["object_density"]
            + self.config.between_object_distance_weight * components["between_distance"]
        )

    def unsupervised_object_loss_124(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        """Loss using components 1, 2, and 4."""
        components = self.object_spike_loss_components(spike_trace, object_masks)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.between_object_distance_weight * components["between_distance"]
        )

    def unsupervised_object_loss_123(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        """Loss using components 1, 2, and 3."""
        components = self.object_spike_loss_components(spike_trace, object_masks)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.object_density_weight * components["object_density"]
        )

    def build_adam_optimizer(self, lr: Optional[float] = None, weight_decay: Optional[float] = None):
        """Create an Adam optimizer using either custom values or config defaults."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr if lr is None else lr,
            weight_decay=self.config.weight_decay if weight_decay is None else weight_decay,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_history: bool = False,
        return_spike_trace: bool = False,
        return_pairwise_history: bool = False,
    ):
        """
        Run the full model on a batch of images.

        Args:
            x: Input images, shape [B, H, W, C]
            return_history: Whether to keep intermediate states for analysis
            return_spike_trace: Whether to return spikes without full theta/gamma history
            return_pairwise_history: Whether to store large affinity/alpha matrices
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
        gate = torch.zeros(batch_size, self.num_nodes, 1, device=device)
        membrane = torch.zeros(batch_size, self.bottom_up.num_pixels, device=device)
        spikes = torch.zeros(batch_size, self.bottom_up.num_pixels, device=device)

        # No initial pairwise feedback. The old identity affinity produced zero
        # coupling anyway because sin(theta_i - theta_i) = 0, but it allocated a
        # huge [N, N] matrix for 64x64 inputs.
        affinity = None
        alpha = None

        # These lists store the trajectory over time for later visualization/analysis.
        spike_hist: List[torch.Tensor] = []
        theta_delay_buffer: List[torch.Tensor] = []
        theta_hist: List[torch.Tensor] = []
        gamma_hist: List[torch.Tensor] = []
        gate_hist: List[torch.Tensor] = []
        membrane_hist: List[torch.Tensor] = []
        affinity_hist: List[torch.Tensor] = []
        alpha_hist: List[torch.Tensor] = []

        interval = max(1, int(self.config.readout_update_interval))
        spike_update_offset = int(self.config.spike_update_offset)
        if spike_update_offset not in (0, 1):
            raise ValueError(f"spike_update_offset must be 0 or 1, got {spike_update_offset}")

        for step_idx in range(1, self.config.steps + 1):
            # 1. Theta evolves every step using the most recent gamma.
            theta = self.top_down.update_theta(theta, gamma, affinity, alpha)

            # 2. Gamma refreshes every scheduled interval.
            if step_idx % interval == 0:
                gamma = self.top_down.readout_gamma_function(theta)
                gate = self.top_down.build_gate_from_history(theta_delay_buffer, theta)

            # 3. Spike can update either at the gamma step or one step later.
            should_update_spike = (step_idx - spike_update_offset) % interval == 0
            should_update_spike = should_update_spike and step_idx > spike_update_offset
            if should_update_spike:
                modulated_gamma = gate * gamma
                membrane, spikes = self.bottom_up.forward_step(membrane, spikes, modulated_gamma)
                affinity, alpha = self.top_down.top_down_feedback_function(spikes)

            spike_hist.append(spikes)
            theta_delay_buffer.append(theta)
            max_delay_history = max(1, int(self.config.delay) + 1)
            if len(theta_delay_buffer) > max_delay_history:
                theta_delay_buffer.pop(0)

            if return_history:
                # Save the full state trajectory only for visualization/analysis.
                theta_hist.append(theta)
                gamma_hist.append(gamma)
                gate_hist.append(gate)
                membrane_hist.append(membrane)
                if return_pairwise_history:
                    affinity_hist.append(affinity)
                    alpha_hist.append(alpha)

        # Stack spikes over time to obtain a [B, T, N] spike trace.
        spike_trace = torch.stack(spike_hist, dim=1)
        # Final classification based on pooled spike activity.
        logits = self.bottom_up.classify(spike_trace)

        # History is optional in spirit, but we build it here so analysis code can
        # inspect internal dynamics after a forward pass.
        history: Dict[str, torch.Tensor] = {}
        if return_history:
            history = {
                "theta": torch.stack(theta_hist, dim=1),
                "gamma": torch.stack(gamma_hist, dim=1),
                "gate": torch.stack(gate_hist, dim=1),
                "membrane": torch.stack(membrane_hist, dim=1),
                "spikes": spike_trace,
            }
            if return_pairwise_history:
                history["affinity"] = torch.stack(affinity_hist, dim=1)
                history["alpha"] = torch.stack(alpha_hist, dim=1)
        elif return_spike_trace:
            history = {"spikes": spike_trace}
        return logits, history if (return_history or return_spike_trace) else {}
