from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .hyperparameters import ObjectRepresentationConfig
except ImportError:
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
        self.osc_dim = config.osc_dim

        self.gamma_encoder = nn.Sequential(
            nn.Conv2d(config.input_channels, config.gamma_encoder_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.gamma_encoder_hidden, config.gamma_encoder_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.gamma_encoder_hidden, config.osc_dim, kernel_size=1),
        )
        self.gamma_encoder_skip = nn.Conv2d(
            config.input_channels,
            config.osc_dim,
            kernel_size=1,
            bias=False,
        )
        self.gamma_readout = nn.Linear(config.osc_dim, config.osc_dim)
        self.gamma_gain = nn.Parameter(torch.ones(1, self.num_oscillators, config.osc_dim))

        self.reset_gamma_parameters()

    def gamma_update(self, theta_state: torch.Tensor) -> torch.Tensor:
        """
        Build gamma(t) from the current oscillator state theta(t).

        theta_state shape:
            [B, H*W, D]
        """
        theta_proj = self.gamma_readout(theta_state)
        return self.activation_function(torch.abs(theta_proj))

    def readout_gamma_function(
        self,
        theta_state: torch.Tensor,
        value_amplitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build gamma(t) as a normalized oscillator-drive vector.

        This keeps the original model behavior available while `gamma_update`
        remains as the simpler activation-based readout experiment.
        """
        theta_proj = self.gamma_readout(theta_state) * self.gamma_gain
        gamma_direction = F.normalize(theta_proj, dim=-1, eps=1e-6)
        if value_amplitude is None:
            return gamma_direction
        return gamma_direction * value_amplitude

    # initialize start
    def initialize_gamma_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build gamma(0) from the input image.

        Input image shape:
            [B, H, W, C]

        Output gamma shape:
            [B, H*W, D]
        """
        encoded_input = self.encode_input_features(x) * self.gamma_gain
        gamma_direction = F.normalize(encoded_input, dim=-1, eps=1e-6)
        return gamma_direction * self.gamma_value_amplitude(x)

    def reset_gamma_parameters(self) -> None:
        """Initialize the image encoder and theta-to-gamma readout."""
        with torch.no_grad():
            for module in self.gamma_encoder:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                    module.weight.mul_(0.05)
                    if module.bias is not None:
                        module.bias.zero_()

            self.gamma_encoder_skip.weight.zero_()
            for channel_idx in range(min(self.config.input_channels, self.osc_dim)):
                self.gamma_encoder_skip.weight[channel_idx, channel_idx, 0, 0] = 1.0

            self.gamma_readout.weight.zero_()
            self.gamma_readout.bias.zero_()
            identity_dim = min(self.gamma_readout.weight.shape)
            self.gamma_readout.weight[:identity_dim, :identity_dim].copy_(torch.eye(identity_dim))

            self.gamma_gain.fill_(1.0)

    def encode_input_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an input image into per-oscillator gamma features."""
        self.validate_input(x)
        batch_size, height, width, _ = x.shape
        x_bchw = x.permute(0, 3, 1, 2)

        blur_kernel = int(self.config.gamma_encoder_blur_kernel)
        if blur_kernel > 1:
            if blur_kernel % 2 == 0:
                raise ValueError("gamma_encoder_blur_kernel must be odd or 1")
            encoder_input = F.avg_pool2d(
                x_bchw,
                kernel_size=blur_kernel,
                stride=1,
                padding=blur_kernel // 2,
            )
        else:
            encoder_input = x_bchw

        feature_map = self.gamma_encoder(encoder_input)
        skip_scale = float(self.config.gamma_encoder_skip_scale)
        if skip_scale != 0.0:
            feature_map = feature_map + skip_scale * self.gamma_encoder_skip(encoder_input)

        return feature_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.osc_dim)

    def gamma_value_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scalar amplitude used to preserve input-value contrast."""
        self.validate_input(x)
        batch_size, height, width, _ = x.shape

        if not self.config.preserve_gamma_value_amplitude:
            return torch.ones(batch_size, height * width, 1, device=x.device, dtype=x.dtype)

        amplitude = x.mean(dim=-1, keepdim=True).reshape(batch_size, height * width, 1)
        value_floor = float(self.config.gamma_value_floor)
        if value_floor > 0.0:
            amplitude = amplitude.clamp_min(value_floor)
        return amplitude

    def validate_input(self, x: torch.Tensor) -> None:
        """Check that the input matches the configured image shape."""
        _, height, width, channels = x.shape
        if (
            height != self.config.image_height
            or width != self.config.image_width
            or channels != self.config.input_channels
        ):
            raise ValueError(
                f"Expected input shape [B, {self.config.image_height}, {self.config.image_width}, "
                f"{self.config.input_channels}], got {tuple(x.shape)}"
            )
    # initialize end

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """Nonlinear activation used in the gamma readout stage."""
        return torch.tanh(x)
