from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .gamma_initialization import get_gamma_initializer
    from .hyperparameters import ObjectRepresentationConfig
except ImportError:
    from gamma_initialization import get_gamma_initializer
    from hyperparameters import ObjectRepresentationConfig


class ReadoutLayer(nn.Module):
    """
    Gamma readout module.

    Responsibilities:
    1. Initialize gamma(0) from the input image.
    2. Read out gamma(t) from the current theta state.
    """

    def __init__(self, config: ObjectRepresentationConfig) -> None:
        super().__init__()
        self.config = config
        self.num_oscillators = config.num_oscillators
        self.osc_dim = 1

        self.gamma_update_weight = nn.Parameter(torch.eye(config.image_width))
        self.gamma_initializer = get_gamma_initializer(config.gamma_initialization, config)

        self.reset_gamma_parameters()

    # gamma update
    def gamma_update(self, theta_state: torch.Tensor) -> torch.Tensor:
        """
        Build gamma(t) from the current oscillator state theta(t).

        theta_state shape:
            [B, H*W]
        """
        batch_size = theta_state.shape[0]
        theta_grid = theta_state.reshape(batch_size, self.config.image_height, self.config.image_width)
        theta_proj = torch.matmul(theta_grid, self.gamma_update_weight)
        return self.activation_function(torch.abs(theta_proj)).reshape(batch_size, self.num_oscillators)

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """Nonlinear activation used in the gamma readout stage."""
        return torch.tanh(x)

    # gamma initialization
    def initialize_gamma_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build gamma(0) from the input image.

        Input image shape:
            [B, H, W, C]

        Output gamma shape:
            [B, H*W]
        """
        return self.gamma_initializer.initialize(x)

    def validate_input(self, x: torch.Tensor) -> None:
        """Check that the input matches the configured image shape."""
        self.gamma_initializer.validate_input(x)

    def reset_gamma_parameters(self) -> None:
        """Initialize the image encoder and theta-to-gamma readout."""
        with torch.no_grad():
            self.gamma_update_weight.copy_(torch.eye(self.config.image_width))
        self.gamma_initializer.reset_parameters()
