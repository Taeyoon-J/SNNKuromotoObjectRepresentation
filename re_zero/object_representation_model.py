from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:
    from .hyperparameters import ObjectRepresentationConfig
    from .kuramoto_layer import KuramotoLayer
    from .loss_function import ObjectRepresentationLoss
    from .readout_layer import ReadoutLayer
    from .sinusoidal_gate import SinusoidalGate
    from .snn_layer import SNNLayer
    from .top_down_feedback import TopDownFeedback
except ImportError:
    from hyperparameters import ObjectRepresentationConfig
    from kuramoto_layer import KuramotoLayer
    from loss_function import ObjectRepresentationLoss
    from readout_layer import ReadoutLayer
    from sinusoidal_gate import SinusoidalGate
    from snn_layer import SNNLayer
    from top_down_feedback import TopDownFeedback


class ObjectRepresentationSNN(nn.Module):
    """
    Full object-representation network assembled from four model parts.

    Parts:
        1. ReadoutLayer initializes and updates gamma.
        2. KuramotoLayer updates theta.
        3. SinusoidalGate sends oscillator state into the SNN pathway.
        4. SNNLayer generates spikes and classifier outputs.
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
        self.loss_helper = ObjectRepresentationLoss(self.config)
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
        self.top_down_feedback = TopDownFeedback(self.config)

    def build_adam_optimizer(self, lr: Optional[float] = None, weight_decay: Optional[float] = None):
        """Create an Adam optimizer using either custom values or config defaults."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr if lr is None else lr,
            weight_decay=self.config.weight_decay if weight_decay is None else weight_decay,
        )

    def object_spike_loss_components(self, classifier_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute unsupervised object-loss components from classifier outputs."""
        return self.loss_helper.object_spike_loss_components(classifier_output)

    def unsupervised_object_loss_1234(self, classifier_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine components 1, 2, 3, and 4 plus background suppression."""
        return self.loss_helper.unsupervised_object_loss_1234(classifier_output)

    def unsupervised_object_loss_124(self, classifier_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine components 1, 2, and 4 plus background suppression."""
        return self.loss_helper.unsupervised_object_loss_124(classifier_output)

    def unsupervised_object_loss_123(self, classifier_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine components 1, 2, and 3 plus background suppression."""
        return self.loss_helper.unsupervised_object_loss_123(classifier_output)

    def unsupervised_object_loss(self, classifier_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select the configured unsupervised object loss variant."""
        return self.loss_helper.unsupervised_object_loss(classifier_output)

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
                theta_connectivity_weight, alpha_t = self.top_down_feedback.top_down_feedback_function(
                    spikes,
                    self.training,
                )

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
        classifier_output = self.snn.classify(spike_trace)

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
        return classifier_output, history if (return_history or return_spike_trace) else {}
