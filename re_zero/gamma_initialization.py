from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class GammaInitialization(nn.Module, ABC):
    """Base class for gamma(0) initialization strategies."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.osc_dim = config.osc_dim

    @abstractmethod
    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        """Build gamma(0) from an input batch."""

    def reset_parameters(self) -> None:
        """Initialize learnable parameters owned by the strategy."""

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


class EncoderGammaInitialization(GammaInitialization):
    """Use a trainable CNN encoder path to initialize gamma(0)."""

    def __init__(self, config) -> None:
        super().__init__(config)
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
        self.gamma_gain = nn.Parameter(torch.ones(1, config.num_oscillators, config.osc_dim))

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        encoded_input = self.encode_input_features(x) * self.gamma_gain
        gamma_direction = F.normalize(encoded_input, dim=-1, eps=1e-6)
        return gamma_direction * self.gamma_value_amplitude(x)

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

    def reset_parameters(self) -> None:
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

            self.gamma_gain.fill_(1.0)


class FlatAutoencoderGammaInitialization(GammaInitialization):
    """Encode the whole flattened image, then decode it into gamma(0)."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.image_feature_dim = config.image_height * config.image_width * config.input_channels
        self.gamma_feature_dim = config.num_oscillators * config.osc_dim
        self.latent_dim = int(config.gamma_autoencoder_latent_dim)

        self.image_encoder = nn.Sequential(
            nn.Linear(self.image_feature_dim, self.latent_dim),
            nn.SiLU(),
        )
        self.gamma_decoder = nn.Linear(self.latent_dim, self.gamma_feature_dim)
        self.gamma_gain = nn.Parameter(torch.ones(1, config.num_oscillators, config.osc_dim))

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        decoded_input = self.decode_image_features(x) * self.gamma_gain
        gamma_direction = F.normalize(decoded_input, dim=-1, eps=1e-6)
        return gamma_direction * self.gamma_value_amplitude(x)

    def decode_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Map a full flattened image through a latent vector into gamma features."""
        self.validate_input(x)
        batch_size, height, width, _ = x.shape
        flat_image = x.reshape(batch_size, self.image_feature_dim)
        latent_features = self.image_encoder(flat_image)
        decoded_features = self.gamma_decoder(latent_features)
        return decoded_features.reshape(batch_size, height * width, self.osc_dim)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for module in self.image_encoder:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                    module.weight.mul_(0.05)
                    if module.bias is not None:
                        module.bias.zero_()

            nn.init.kaiming_normal_(self.gamma_decoder.weight, nonlinearity="linear")
            self.gamma_decoder.weight.mul_(0.05)
            if self.gamma_decoder.bias is not None:
                self.gamma_decoder.bias.zero_()
            self.gamma_gain.fill_(1.0)


def get_gamma_initializer(name: str, config) -> GammaInitialization:
    """Create a gamma initializer from a config keyword."""
    normalized_name = name.lower().strip()
    if normalized_name in {"encoder", "cnn"}:
        return EncoderGammaInitialization(config)
    if normalized_name in {"flat_autoencoder", "autoencoder", "mlp"}:
        return FlatAutoencoderGammaInitialization(config)
    raise ValueError(
        f"Unknown gamma initialization '{name}'. "
        "Choose one of: encoder, flat_autoencoder."
    )
