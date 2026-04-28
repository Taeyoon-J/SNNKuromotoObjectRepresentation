from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class GammaInitialization(nn.Module, ABC):
    """Base class for gamma(0) initialization strategies."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.osc_dim = 1

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
            return torch.ones(batch_size, height * width, device=x.device, dtype=x.dtype)

        amplitude = x.mean(dim=-1).reshape(batch_size, height * width)
        value_floor = float(self.config.gamma_value_floor)
        if value_floor > 0.0:
            amplitude = amplitude.clamp_min(value_floor)
        return amplitude


class ChannelCompressInitialization(GammaInitialization):
    """Use a trainable per-pixel weighted image projection to initialize gamma(0)."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.pixel_weight = nn.Parameter(torch.ones(config.input_channels))
        self.pixel_bias = nn.Parameter(torch.zeros(1))
        self.gamma_gain = nn.Parameter(torch.ones(1, config.num_oscillators))

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        encoded_input = self.encode_input_features(x) * self.gamma_gain
        return torch.tanh(encoded_input) * self.gamma_value_amplitude(x)

    def encode_input_features(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an input image into one scalar gamma value per pixel."""
        self.validate_input(x)
        batch_size, height, width, _ = x.shape
        weighted_pixels = torch.tensordot(x, self.pixel_weight, dims=([-1], [0])) + self.pixel_bias
        return weighted_pixels.reshape(batch_size, height * width)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.pixel_weight.fill_(1.0 / max(1, self.config.input_channels))
            self.pixel_bias.zero_()
            self.gamma_gain.fill_(1.0)


class EncoderInitialization(GammaInitialization):
    """Encode the whole flattened image, then decode it into gamma(0)."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.image_feature_dim = config.image_height * config.image_width * config.input_channels
        self.gamma_feature_dim = config.num_oscillators
        self.latent_dim = int(config.gamma_autoencoder_latent_dim)

        self.image_encoder = nn.Sequential(
            nn.Linear(self.image_feature_dim, self.latent_dim),
            nn.SiLU(),
        )
        self.gamma_decoder = nn.Linear(self.latent_dim, self.gamma_feature_dim)
        self.gamma_gain = nn.Parameter(torch.ones(1, config.num_oscillators))

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        decoded_input = self.decode_image_features(x) * self.gamma_gain
        return torch.tanh(decoded_input) * self.gamma_value_amplitude(x)

    def decode_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Map a full flattened image through a latent vector into gamma features."""
        self.validate_input(x)
        batch_size, height, width, _ = x.shape
        flat_image = x.reshape(batch_size, self.image_feature_dim)
        latent_features = self.image_encoder(flat_image)
        decoded_features = self.gamma_decoder(latent_features)
        return decoded_features.reshape(batch_size, height * width)

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
    if normalized_name in {"channel", "channel_compress", "pixel", "cnn"}:
        return ChannelCompressInitialization(config)
    if normalized_name in {"encoder", "flat_autoencoder", "autoencoder", "mlp"}:
        return EncoderInitialization(config)
    raise ValueError(
        f"Unknown gamma initialization '{name}'. "
        "Choose one of: encoder, channel_compress."
    )
