from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhaseBindingMetrics:
    intra_sync: float
    inter_sync: float


class GraphPhaseKuramoto(nn.Module):
    """
    Kuramoto proof-of-concept block for object-wise phase binding.

    This intentionally stays close to the toy simulation idea from TINGTING:
    learn a graph from image features, evolve phases, and supervise the final
    phase organization directly.
    """

    def __init__(
        self,
        image_size: int = 28,
        osc_dim: int = 1,
        feature_dim: int = 8,
        coupling_k: float = 18.0,
        dt: float = 0.1,
        lag_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_nodes = image_size * image_size
        self.osc_dim = osc_dim
        self.coupling_k = coupling_k
        self.dt = dt
        self.lag_scale = lag_scale

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.feature_head = nn.Conv2d(16, feature_dim, kernel_size=1)
        self.drive_head = nn.Conv2d(16, osc_dim, kernel_size=1)

        self.omega = nn.Parameter(torch.randn(self.num_nodes, osc_dim) * 0.1)
        self.kappa = nn.Parameter(torch.ones(self.num_nodes, osc_dim))
        self.direction_learner = nn.Parameter(torch.randn(self.num_nodes, self.num_nodes) * 0.01)

    def build_graph(self, shared_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = shared_feat.shape[0]
        feat = self.feature_head(shared_feat).view(b, -1, self.num_nodes).permute(0, 2, 1)
        feat_norm = F.normalize(feat, p=2, dim=-1)
        adjacency = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        eye = torch.eye(self.num_nodes, device=shared_feat.device).unsqueeze(0)
        adjacency = adjacency * (1.0 - eye)

        cost = 1.0 / (adjacency.relu() + 1e-6)
        c_min = cost.amin(dim=(1, 2), keepdim=True)
        c_max = cost.amax(dim=(1, 2), keepdim=True)
        norm_cost = (cost - c_min) / (c_max - c_min + 1e-6)
        direction = torch.tanh(self.direction_learner - self.direction_learner.transpose(0, 1))
        alpha = self.lag_scale * direction.unsqueeze(0) * norm_cost
        return adjacency, alpha.unsqueeze(-1)

    def kuramoto_step(
        self,
        theta_prev: torch.Tensor,
        gamma: torch.Tensor,
        adjacency: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        _, n, _ = theta_prev.shape
        theta_i = theta_prev.unsqueeze(2)
        theta_j = theta_prev.unsqueeze(1)
        phase_diff = theta_j - theta_i - alpha
        interaction = torch.sum(adjacency.unsqueeze(-1) * torch.sin(phase_diff), dim=2)
        coupling = (self.coupling_k / float(n)) * interaction
        drive = self.kappa.unsqueeze(0) * torch.sin(gamma - theta_prev)
        theta_dot = self.omega.unsqueeze(0) + coupling + drive
        return theta_prev + self.dt * theta_dot

    def forward(self, x: torch.Tensor, steps: int = 60, return_history: bool = False):
        # Accept images in either [B, H, W, C] or [B, C, H, W] format.
        if x.dim() != 4:
            raise ValueError(f"Expected a 4D image tensor, got shape {tuple(x.shape)}")
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()

        shared_feat = self.backbone(x)
        gamma = self.drive_head(shared_feat).view(x.shape[0], self.num_nodes, self.osc_dim)
        adjacency, alpha = self.build_graph(shared_feat)

        theta = torch.rand(x.shape[0], self.num_nodes, self.osc_dim, device=x.device) * (2.0 * np.pi)
        theta_hist = []
        for _ in range(steps):
            theta = self.kuramoto_step(theta, gamma, adjacency, alpha)
            if return_history:
                theta_hist.append(theta)

        history = {}
        if return_history:
            history = {
                "theta": torch.stack(theta_hist, dim=1),
                "adjacency": adjacency,
                "alpha": alpha.squeeze(-1),
                "gamma": gamma,
            }
        return theta.squeeze(-1), history


def phase_contrastive_loss(theta: torch.Tensor, mask: torch.Tensor, neg_weight: float = 2.0) -> torch.Tensor:
    """
    Encourage same-object pixels to share phase and different-object pixels to separate.
    """
    batch_size, num_nodes = theta.shape
    mask_flat = mask.view(batch_size, num_nodes)
    loss = theta.new_tensor(0.0)
    valid_count = 0

    for idx in range(batch_size):
        valid = mask_flat[idx] > 0
        if valid.sum() < 2:
            continue
        phases = theta[idx, valid]
        labels = mask_flat[idx, valid]
        cos_sim = torch.cos(phases.unsqueeze(0) - phases.unsqueeze(1))
        same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        diff = 1.0 - same
        pos_term = ((1.0 - cos_sim) * same).sum() / (same.sum() + 1e-8)
        neg_term = ((1.0 + cos_sim) * diff).sum() / (diff.sum() + 1e-8)
        loss = loss + pos_term + neg_weight * neg_term
        valid_count += 1

    if valid_count == 0:
        return theta.new_tensor(0.0, requires_grad=True)
    return loss / valid_count


def phase_binding_metrics(theta: torch.Tensor, mask: torch.Tensor) -> PhaseBindingMetrics:
    """
    Measure intra-object synchrony and inter-object similarity from the final phase state.
    """
    with torch.no_grad():
        phase_vec = theta[0].detach().cpu().numpy()
        label_vec = mask[0].view(-1).cpu().numpy()
        z_values = []
        for object_id in sorted(np.unique(label_vec)):
            if object_id <= 0:
                continue
            current = phase_vec[label_vec == object_id]
            if current.size == 0:
                continue
            z_values.append(np.mean(np.exp(1j * current)))

        if not z_values:
            return PhaseBindingMetrics(intra_sync=0.0, inter_sync=0.0)

        intra = float(np.mean([np.abs(z) for z in z_values]))
        if len(z_values) == 1:
            inter = 1.0
        else:
            phase_diff = np.angle(z_values[0]) - np.angle(z_values[1])
            inter = float(0.5 * (1.0 + np.cos(phase_diff)))
        return PhaseBindingMetrics(intra_sync=intra, inter_sync=inter)
