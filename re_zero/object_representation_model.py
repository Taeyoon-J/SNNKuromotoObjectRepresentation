from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .hyperparameters import ObjectRepresentationConfig
    from .kuramoto_layer import KuramotoLayer
    from .readout_layer import ReadoutLayer
    from .sinusoidal_gate import SinusoidalGate
    from .snn_layer import SNNLayer
except ImportError:
    from hyperparameters import ObjectRepresentationConfig
    from kuramoto_layer import KuramotoLayer
    from readout_layer import ReadoutLayer
    from sinusoidal_gate import SinusoidalGate
    from snn_layer import SNNLayer


class ObjectRepresentationSNN(nn.Module):
    """
    Full object-representation network assembled from four model parts.

    Parts:
        1. ReadoutLayer initializes and updates gamma.
        2. KuramotoLayer updates theta.
        3. SinusoidalGate sends oscillator state into the SNN pathway.
        4. SNNLayer generates spikes and classification logits.
    """

    def __init__(self, config: Optional[ObjectRepresentationConfig] = None) -> None:
        super().__init__()
        self.config = config or ObjectRepresentationConfig()
        self.num_oscillators = self.config.num_oscillators

        self.readout = ReadoutLayer(self.config)
        self.kuramoto = KuramotoLayer(
            num_oscillators=self.num_oscillators,
            osc_dim=self.config.osc_dim,
            global_coupling_strength=self.config.global_coupling_strength,
            step_size=self.config.step_size,
            gamma_attraction_strength=self.config.gamma_attraction_strength,
            fixed_alpha_during_training=self.config.fixed_alpha_during_training,
            fixed_alpha_value=self.config.fixed_alpha_value,
            coupling_chunk_size=self.config.coupling_chunk_size,
        )
        self.gate = SinusoidalGate(delay=self.config.delay)
        self.snn = SNNLayer(
            num_oscillators=self.num_oscillators,
            osc_dim=self.config.osc_dim,
            num_classes=self.config.num_classes,
            membrane_decay=self.config.membrane_decay,
            threshold=self.config.threshold,
            recurrent_scale=self.config.recurrent_scale,
            classifier_start_step=self.config.classifier_start_step,
            classifier_type=self.config.classifier_type,
            image_height=self.config.image_height,
            image_width=self.config.image_width,
            input_channels=self.config.input_channels,
        )

    def loss_function(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard supervised classification loss."""
        return F.cross_entropy(logits, labels)

    def top_down_feedback_function(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build pairwise theta connectivity and phase-lag matrices from spikes.

        Returns:
            theta_connectivity_weight: [B, N, N]
            alpha_t: [B, N, N]
        """
        if spikes.shape[-1] != self.num_oscillators:
            raise ValueError(
                f"Expected pixel-level spikes with {self.num_oscillators} nodes, got {spikes.shape[-1]}"
            )
        spike_i = spikes.unsqueeze(2)
        spike_j = spikes.unsqueeze(1)
        spike_range = (spikes.amax(dim=1, keepdim=True) - spikes.amin(dim=1, keepdim=True)).clamp_min(1e-6)
        normalized_delta = torch.abs(spike_i - spike_j) / spike_range.unsqueeze(-1)

        theta_connectivity_weight = (
            self.config.feedback_theta_connectivity_weight_scale * (1.0 - normalized_delta)
        )
        if self.training and self.config.fixed_alpha_during_training:
            alpha_t = torch.full_like(normalized_delta, float(self.config.fixed_alpha_value))
        else:
            alpha_t = self.config.feedback_alpha_scale * self.config.alpha_scale * normalized_delta
        return theta_connectivity_weight, alpha_t

    def _late_spike_window(self, spike_trace: torch.Tensor) -> torch.Tensor:
        """Return the spike window used by the unsupervised object losses."""
        start_idx = min(max(self.config.classifier_start_step - 1, 0), spike_trace.shape[1] - 1)
        return spike_trace[:, start_idx:, :]

    def _prepare_object_masks(self, object_masks: torch.Tensor, spike_trace: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel-level object masks into flattened node-level masks.

        Args:
            object_masks: [O, H, W] or [B, O, H, W]
            spike_trace: [B, T, H*W]
        """
        if object_masks.dim() == 3:
            object_masks = object_masks.unsqueeze(0).expand(spike_trace.shape[0], -1, -1, -1)
        if object_masks.dim() != 4:
            raise ValueError(
                f"Expected object masks with shape [O, H, W] or [B, O, H, W], got {tuple(object_masks.shape)}"
            )
        if object_masks.shape[0] != spike_trace.shape[0]:
            raise ValueError(
                f"Mask batch size {object_masks.shape[0]} does not match spike batch size {spike_trace.shape[0]}"
            )

        flat_masks = object_masks.to(device=spike_trace.device, dtype=spike_trace.dtype).view(
            object_masks.shape[0],
            object_masks.shape[1],
            -1,
        )
        if flat_masks.shape[-1] != spike_trace.shape[-1]:
            raise ValueError(
                f"Flattened mask has {flat_masks.shape[-1]} nodes, but spike trace has {spike_trace.shape[-1]}"
            )
        return flat_masks

    def _object_activity(self, spike_trace: torch.Tensor, node_masks: torch.Tensor) -> torch.Tensor:
        """Average spike activity inside each object mask, shape [B, T, O]."""
        mask_sizes = node_masks.sum(dim=-1).clamp_min(1.0)
        return torch.einsum("btn,bon->bto", spike_trace, node_masks) / mask_sizes.unsqueeze(1)

    def object_spike_loss_components(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute reusable unsupervised object-spike loss components.
        """
        late_spikes = self._late_spike_window(spike_trace)
        node_masks = self._prepare_object_masks(object_masks, late_spikes)
        object_activity = self._object_activity(late_spikes, node_masks)

        batch_size, steps, num_objects = object_activity.shape
        eps = late_spikes.new_tensor(1e-6)
        object_valid = node_masks.sum(dim=-1) > 0.5
        valid_object_count = object_valid.to(late_spikes.dtype).sum().clamp_min(1.0)

        object_mean = object_activity.unsqueeze(-1)
        mask_sizes = node_masks.sum(dim=-1).clamp_min(1.0)
        squared_error = (late_spikes.unsqueeze(2) - object_mean).pow(2) * node_masks.unsqueeze(1)
        within_per_object = squared_error.sum(dim=-1) / mask_sizes.unsqueeze(1)
        within_similarity = (within_per_object * object_valid.unsqueeze(1)).sum() / (
            steps * valid_object_count
        )

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

            temporal_profile = object_activity / object_activity.sum(dim=1, keepdim=True).clamp_min(eps)
            profile_overlap = (temporal_profile.unsqueeze(-1) * temporal_profile.unsqueeze(-2)).sum(dim=1)
            between_difference = (profile_overlap * valid_pairs).sum() / valid_pair_count

            time_axis = torch.arange(steps, device=late_spikes.device, dtype=late_spikes.dtype)
            activity_mass = object_activity.sum(dim=1).clamp_min(eps)
            centers = (object_activity * time_axis.view(1, steps, 1)).sum(dim=1) / activity_mass
            pairwise_distance = torch.abs(centers.unsqueeze(-1) - centers.unsqueeze(-2))
            distance_scale = max(float(self.config.object_time_distance_scale), 1e-6)
            between_distance = (torch.exp(-pairwise_distance / distance_scale) * valid_pairs).sum() / valid_pair_count

        peak_activity = object_activity.max(dim=1).values
        object_density_per_object = F.relu(float(self.config.object_density_target) - peak_activity)
        object_density = (object_density_per_object * object_valid).sum() / valid_object_count

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
        """Loss using components 1, 2, 3, and 4 plus background suppression."""
        components = self.object_spike_loss_components(spike_trace, object_masks)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.object_density_weight * components["object_density"]
            + self.config.between_object_distance_weight * components["between_distance"]
            + self.config.background_suppression_weight * components["background_suppression"]
        )

    def unsupervised_object_loss_124(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        """Loss using components 1, 2, and 4 plus background suppression."""
        components = self.object_spike_loss_components(spike_trace, object_masks)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.between_object_distance_weight * components["between_distance"]
            + self.config.background_suppression_weight * components["background_suppression"]
        )

    def unsupervised_object_loss_123(self, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        """Loss using components 1, 2, and 3 plus background suppression."""
        components = self.object_spike_loss_components(spike_trace, object_masks)
        return (
            self.config.within_object_similarity_weight * components["within_similarity"]
            + self.config.between_object_difference_weight * components["between_difference"]
            + self.config.object_density_weight * components["object_density"]
            + self.config.background_suppression_weight * components["background_suppression"]
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
            return_spike_trace: Whether to return spikes without full history
            return_pairwise_history: Whether to store large theta connectivity/alpha matrices
        """
        self.readout.validate_input(x)
        batch_size = x.shape[0]
        device = x.device

        theta = torch.randn(
            batch_size,
            self.num_oscillators,
            device=device,
        )
        gamma = self.readout.initialize_gamma_from_input(x)
        theta_initial = theta
        gamma_initial = gamma

        sinusoidal_gate = torch.zeros(batch_size, self.num_oscillators, device=device)
        membrane = torch.zeros(batch_size, self.snn.num_pixels, device=device)
        spikes = torch.zeros(batch_size, self.snn.num_pixels, device=device)

        theta_connectivity_weight = torch.zeros(
            batch_size,
            self.num_oscillators,
            self.num_oscillators,
            device=device,
        )
        alpha_t = torch.zeros_like(theta_connectivity_weight)

        spike_hist: List[torch.Tensor] = []
        theta_delay_buffer: List[torch.Tensor] = []
        theta_hist: List[torch.Tensor] = []
        gamma_hist: List[torch.Tensor] = []
        gate_hist: List[torch.Tensor] = []
        membrane_hist: List[torch.Tensor] = []
        theta_connectivity_weight_hist: List[torch.Tensor] = []
        alpha_hist: List[torch.Tensor] = []

        interval = max(1, int(self.config.readout_update_interval))
        spike_update_offset = int(self.config.spike_update_offset)
        if spike_update_offset not in (0, 1):
            raise ValueError(f"spike_update_offset must be 0 or 1, got {spike_update_offset}")

        for step_idx in range(1, self.config.steps + 1):
            theta = self.kuramoto(theta, gamma, theta_connectivity_weight, alpha_t)

            if step_idx % interval == 0:
                gamma = self.readout.gamma_update(theta)
                sinusoidal_gate = self.gate.sinusoidal_gating(theta_delay_buffer, theta, gamma)

            should_update_spike = (step_idx - spike_update_offset) % interval == 0
            should_update_spike = should_update_spike and step_idx > spike_update_offset
            if should_update_spike:
                membrane, spikes = self.snn.forward_step(membrane, spikes, sinusoidal_gate, gamma)
                theta_connectivity_weight, alpha_t = self.top_down_feedback_function(spikes)

            spike_hist.append(spikes)
            theta_delay_buffer.append(theta)
            max_delay_history = max(1, int(self.config.delay) + 1)
            if len(theta_delay_buffer) > max_delay_history:
                theta_delay_buffer.pop(0)

            if return_history:
                theta_hist.append(theta)
                gamma_hist.append(gamma)
                gate_hist.append(sinusoidal_gate)
                membrane_hist.append(membrane)
                if return_pairwise_history:
                    theta_connectivity_weight_hist.append(theta_connectivity_weight)
                    alpha_hist.append(alpha_t)

        spike_trace = torch.stack(spike_hist, dim=1)
        logits = self.snn.classify(spike_trace)

        history: Dict[str, torch.Tensor] = {}
        if return_history:
            history = {
                "theta0": theta_initial,
                "gamma0": gamma_initial,
                "theta": torch.stack(theta_hist, dim=1),
                "gamma": torch.stack(gamma_hist, dim=1),
                "gate": torch.stack(gate_hist, dim=1),
                "membrane": torch.stack(membrane_hist, dim=1),
                "spikes": spike_trace,
            }
            if return_pairwise_history:
                if theta_connectivity_weight_hist:
                    history["theta_connectivity_weight"] = torch.stack(theta_connectivity_weight_hist, dim=1)
                if alpha_hist:
                    history["alpha"] = torch.stack(alpha_hist, dim=1)
        elif return_spike_trace:
            history = {"spikes": spike_trace}
        return logits, history if (return_history or return_spike_trace) else {}
