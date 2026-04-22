"""
Optional shape-validation snippets for `re_zero` modules.

This file is intentionally kept separate so the main model code stays easy to
read. When you want explicit input validation during debugging, copy the
relevant commented block into the target function and uncomment it.

Example:
    In `KuramotoLayer.compute_coupling_term(...)`, paste the validation block
    below right after the early return for `self.coupling == 0.0`.
"""


# Shape checks for `KuramotoLayer.compute_coupling_term(...)`
#
# Use this block when debugging shape mismatches among:
# - `theta_prev` with shape [B, N, D]
# - `theta_connectivity_weight` with shape [B, N, N]
# - `alpha_t` with shape [B, N, N]
#
# if theta_connectivity_weight.dim() != 3:
#     raise ValueError(
#         "theta_connectivity_weight must have shape [B, N, N], "
#         f"got {tuple(theta_connectivity_weight.shape)}"
#     )
# if alpha_t.dim() != 3:
#     raise ValueError(f"alpha_t must have shape [B, N, N], got {tuple(alpha_t.shape)}")
# if (
#     theta_connectivity_weight.shape[:2] != (theta_prev.shape[0], theta_prev.shape[1])
#     or theta_connectivity_weight.shape[2] != theta_prev.shape[1]
# ):
#     raise ValueError(
#         "theta_connectivity_weight must align with theta_prev and have shape [B, N, N]"
#     )
# if alpha_t.shape != theta_connectivity_weight.shape:
#     raise ValueError("alpha_t must have the same shape as theta_connectivity_weight")


# Original CNN-based initialize_gamma_from_input path for `ReadoutLayer`
#
# Keep this block here if you want to restore the richer image-encoder version
# later. The current `readout_layer.py` can stay simpler while this reference
# remains available for comparison.
#
# self.gamma_encoder = nn.Sequential(
#     nn.Conv2d(config.input_channels, config.gamma_encoder_hidden, kernel_size=3, padding=1),
#     nn.SiLU(),
#     nn.Conv2d(config.gamma_encoder_hidden, config.gamma_encoder_hidden, kernel_size=3, padding=1),
#     nn.SiLU(),
#     nn.Conv2d(config.gamma_encoder_hidden, config.osc_dim, kernel_size=1),
# )
# self.gamma_encoder_skip = nn.Conv2d(
#     config.input_channels,
#     config.osc_dim,
#     kernel_size=1,
#     bias=False,
# )
#
# def reset_gamma_parameters(self) -> None:
#     with torch.no_grad():
#         for module in self.gamma_encoder:
#             if isinstance(module, nn.Conv2d):
#                 nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
#                 module.weight.mul_(0.05)
#                 if module.bias is not None:
#                     module.bias.zero_()
#
#         self.gamma_encoder_skip.weight.zero_()
#         for channel_idx in range(min(self.config.input_channels, self.osc_dim)):
#             self.gamma_encoder_skip.weight[channel_idx, channel_idx, 0, 0] = 1.0
#
#         self.gamma_readout.weight.zero_()
#         self.gamma_readout.bias.zero_()
#         identity_dim = min(self.gamma_readout.weight.shape)
#         self.gamma_readout.weight[:identity_dim, :identity_dim].copy_(torch.eye(identity_dim))
#
#         self.gamma_gain.fill_(1.0)
#
# def encode_input_features(self, x: torch.Tensor) -> torch.Tensor:
#     self.validate_input(x)
#     batch_size, height, width, _ = x.shape
#     x_bchw = x.permute(0, 3, 1, 2)
#
#     blur_kernel = int(self.config.gamma_encoder_blur_kernel)
#     if blur_kernel > 1:
#         if blur_kernel % 2 == 0:
#             raise ValueError("gamma_encoder_blur_kernel must be odd or 1")
#         encoder_input = F.avg_pool2d(
#             x_bchw,
#             kernel_size=blur_kernel,
#             stride=1,
#             padding=blur_kernel // 2,
#         )
#     else:
#         encoder_input = x_bchw
#
#     feature_map = self.gamma_encoder(encoder_input)
#     skip_scale = float(self.config.gamma_encoder_skip_scale)
#     if skip_scale != 0.0:
#         feature_map = feature_map + skip_scale * self.gamma_encoder_skip(encoder_input)
#
#     return feature_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.osc_dim)
#
# def gamma_value_amplitude(self, x: torch.Tensor) -> torch.Tensor:
#     self.validate_input(x)
#     batch_size, height, width, _ = x.shape
#
#     if not self.config.preserve_gamma_value_amplitude:
#         return torch.ones(batch_size, height * width, 1, device=x.device, dtype=x.dtype)
#
#     amplitude = x.mean(dim=-1, keepdim=True).reshape(batch_size, height * width, 1)
#     value_floor = float(self.config.gamma_value_floor)
#     if value_floor > 0.0:
#         amplitude = amplitude.clamp_min(value_floor)
#     return amplitude
#
# def validate_input(self, x: torch.Tensor) -> None:
#     _, height, width, channels = x.shape
#     if (
#         height != self.config.image_height
#         or width != self.config.image_width
#         or channels != self.config.input_channels
#     ):
#         raise ValueError(
#             f"Expected input shape [B, {self.config.image_height}, {self.config.image_width}, "
#             f"{self.config.input_channels}], got {tuple(x.shape)}"
#         )
#
# def initialize_gamma_from_input(self, x: torch.Tensor) -> torch.Tensor:
#     encoded_input = self.encode_input_features(x) * self.gamma_gain
#     gamma_direction = F.normalize(encoded_input, dim=-1, eps=1e-6)
#     return gamma_direction * self.gamma_value_amplitude(x)
