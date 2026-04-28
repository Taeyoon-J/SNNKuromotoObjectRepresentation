from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.data import CLEVRObjectDataset
from re_zero import ObjectRepresentationConfig, ObjectRepresentationSNN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> ObjectRepresentationConfig:
    """Build a fixed diagnostic config: t=30, gamma update interval=5 by default."""
    return ObjectRepresentationConfig(
        image_height=args.image_size,
        image_width=args.image_size,
        input_channels=3,
        osc_dim=1,
        steps=args.steps,
        readout_update_interval=args.gamma_update_interval,
        spike_update_offset=args.spike_update_offset,
        classifier_start_step=args.classifier_start_step,
        classifier_type=args.classifier_type,
        classifier_similarity_threshold=args.classifier_similarity_threshold,
        step_size=args.step_size,
        global_coupling_strength=args.global_coupling_strength,
        coupling_chunk_size=args.coupling_chunk_size,
        gamma_attraction_strength=args.gamma_attraction_strength,
        membrane_decay=args.membrane_decay,
        recurrent_scale=args.recurrent_scale,
        threshold=args.threshold,
        alpha_scale=args.alpha_scale,
        fixed_alpha_during_training=False,
        fixed_alpha_value=args.fixed_alpha_value,
        feedback_theta_connectivity_weight_scale=args.feedback_theta_connectivity_weight_scale,
        feedback_alpha_scale=args.feedback_alpha_scale,
        delay=args.delay,
        gamma_initialization=args.gamma_initialization,
        gamma_autoencoder_latent_dim=args.gamma_autoencoder_latent_dim,
        gamma_patch_size=args.gamma_patch_size,
        gamma_update_scale=args.gamma_update_scale,
        preserve_gamma_value_amplitude=args.preserve_gamma_value_amplitude,
        gamma_value_floor=args.gamma_value_floor,
        seed=args.seed,
        device=args.device,
    )


def load_sample(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Load one CLEVR sample and optional masks, otherwise use a random image."""
    hdf5_path = Path(args.hdf5_path)
    if args.use_random_input or not hdf5_path.exists():
        return torch.rand(1, args.image_size, args.image_size, 3, device=device), {}

    dataset = CLEVRObjectDataset(
        hdf5_path=str(hdf5_path),
        target_size=args.image_size,
        max_objects=args.max_objects,
        split=args.split,
        train_fraction=args.train_fraction,
        max_samples=max(args.sample_idx + 1, 1),
        patch_size=None if args.no_patch else args.patch_size,
        patch_stride=None if args.no_patch else args.patch_stride,
        min_object_pixels=args.min_object_pixels,
    )
    sample = dataset[args.sample_idx]
    metadata = {
        "object_masks": sample["object_masks"].detach().cpu(),
        "label_map": sample["label_map"].detach().cpu(),
    }
    return sample["image"].unsqueeze(0).to(device), metadata


def build_clevr_dataset(args: argparse.Namespace, split: str, max_samples: int | None):
    """Build the CLEVR dataset using the same crop/resize settings as the diagnostic sample."""
    return CLEVRObjectDataset(
        hdf5_path=args.hdf5_path,
        target_size=args.image_size,
        max_objects=args.max_objects,
        split=split,
        train_fraction=args.train_fraction,
        max_samples=max_samples,
        patch_size=None if args.no_patch else args.patch_size,
        patch_stride=None if args.no_patch else args.patch_stride,
        min_object_pixels=args.min_object_pixels,
    )


def pretrain_gamma_encoder(
    model: ObjectRepresentationSNN,
    args: argparse.Namespace,
    device: torch.device,
) -> List[float]:
    """Pretrain EncoderInitialization to reconstruct scalar image luminance."""
    if not args.pretrain_gamma_encoder:
        return []
    if args.use_random_input:
        print("Skipping gamma encoder pretraining because --use_random_input is set.")
        return []

    dataset = build_clevr_dataset(
        args,
        split="train",
        max_samples=args.pretrain_gamma_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.pretrain_gamma_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    initializer = model.readout.gamma_initializer
    optimizer = torch.optim.Adam(
        initializer.parameters(),
        lr=args.pretrain_gamma_lr,
        weight_decay=args.pretrain_gamma_weight_decay,
    )

    losses: List[float] = []
    initializer.train()
    for epoch in range(args.pretrain_gamma_epochs):
        total_loss = 0.0
        total_seen = 0
        for batch in loader:
            images = batch["image"].to(device)
            target = initializer.gamma_value_amplitude(images)
            prediction = initializer.initialize(images)
            loss = F.mse_loss(prediction, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size

        epoch_loss = total_loss / max(total_seen, 1)
        losses.append(epoch_loss)
        print(f"gamma pretrain epoch {epoch + 1}/{args.pretrain_gamma_epochs}: loss={epoch_loss:.6f}")

    initializer.eval()
    return losses


def scalar_alignment(theta: torch.Tensor, gamma: torch.Tensor) -> Dict[str, float]:
    """Compute scalar theta-gamma relationship metrics for one timestep."""
    theta_flat = theta.reshape(theta.shape[0], -1)
    gamma_flat = gamma.reshape(gamma.shape[0], -1)
    delta = theta_flat - gamma_flat

    theta_centered = theta_flat - theta_flat.mean(dim=1, keepdim=True)
    gamma_centered = gamma_flat - gamma_flat.mean(dim=1, keepdim=True)
    corr_denominator = (
        theta_centered.norm(dim=1) * gamma_centered.norm(dim=1)
    ).clamp_min(1e-8)
    correlation = (theta_centered * gamma_centered).sum(dim=1) / corr_denominator
    cosine = torch.nn.functional.cosine_similarity(theta_flat, gamma_flat, dim=1, eps=1e-8)

    return {
        "theta_mean": float(theta_flat.mean().item()),
        "theta_std": float(theta_flat.std(unbiased=False).item()),
        "gamma_mean": float(gamma_flat.mean().item()),
        "gamma_std": float(gamma_flat.std(unbiased=False).item()),
        "theta_gamma_mse": float(delta.pow(2).mean().item()),
        "theta_gamma_mae": float(delta.abs().mean().item()),
        "theta_gamma_corr": float(correlation.mean().item()),
        "theta_gamma_cosine": float(cosine.mean().item()),
    }


def tensor_stats(prefix: str, value: torch.Tensor) -> Dict[str, float]:
    """Summarize a tensor without keeping the full pairwise matrix."""
    return {
        f"{prefix}_mean": float(value.mean().item()),
        f"{prefix}_std": float(value.std(unbiased=False).item()),
        f"{prefix}_min": float(value.min().item()),
        f"{prefix}_max": float(value.max().item()),
    }


def build_feedback_tensors(
    model: ObjectRepresentationSNN,
    spikes: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Aij and alpha according to the diagnostic feedback condition."""
    if args.feedback_mode == "fixed_affinity":
        batch_size = spikes.shape[0]
        device = spikes.device
        aij = torch.full(
            (batch_size, model.num_oscillators, model.num_oscillators),
            float(args.fixed_affinity_value),
            device=device,
            dtype=spikes.dtype,
        )
        alpha = torch.full_like(aij, float(args.fixed_alpha_value))
        return aij, alpha
    return model.top_down_feedback.top_down_feedback_function(
        spikes,
        training=model.training,
    )


def parse_float_list(value: str) -> List[float]:
    """Parse comma-separated floats such as '2.5,2.0,1.5'."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    """Parse comma-separated ints such as '5,10,15'."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def gamma_attraction_for_step(step_idx: int, args: argparse.Namespace) -> float:
    """Return the diagnostic-only gamma attraction strength for this step."""
    if args.gamma_attraction_schedule == "constant":
        return float(args.gamma_attraction_strength)

    values = parse_float_list(args.gamma_attraction_values)
    boundaries = parse_int_list(args.gamma_attraction_boundaries)
    if len(values) != len(boundaries) + 1:
        raise ValueError(
            "--gamma_attraction_values must contain exactly one more value than "
            "--gamma_attraction_boundaries for step_decay schedule."
        )

    for boundary, value in zip(boundaries, values):
        if step_idx <= boundary:
            return value
    return values[-1]


@torch.no_grad()
def run_diagnostic(model: ObjectRepresentationSNN, x: torch.Tensor, args: argparse.Namespace):
    """Manual rollout so Aij/alpha are computed but not stored as large histories."""
    config = model.config
    batch_size = x.shape[0]
    device = x.device

    theta = torch.randn(batch_size, model.num_oscillators, device=device)
    gamma = model.readout.initialize_gamma_from_input(x)
    gate = torch.zeros(batch_size, model.num_oscillators, device=device)
    membrane = torch.zeros(batch_size, model.snn.num_pixels, device=device)
    spikes = torch.zeros(batch_size, model.snn.num_pixels, device=device)
    feedback_aij = torch.zeros(batch_size, model.num_oscillators, model.num_oscillators, device=device)
    feedback_alpha = torch.zeros_like(feedback_aij)
    kuramoto_aij = torch.zeros_like(feedback_aij)
    kuramoto_alpha = torch.zeros_like(feedback_alpha)

    theta_history: List[torch.Tensor] = []
    gamma_history: List[torch.Tensor] = []
    spike_history: List[torch.Tensor] = []
    theta_delay_buffer: List[torch.Tensor] = []
    metrics: List[Dict[str, float]] = []

    interval = max(1, int(config.readout_update_interval))
    spike_update_offset = int(config.spike_update_offset)

    for step_idx in range(1, config.steps + 1):
        model.kuramoto.gamma_attraction_strength = gamma_attraction_for_step(step_idx, args)
        theta = model.kuramoto(theta, gamma, kuramoto_aij, kuramoto_alpha)
        gamma_updated = False
        feedback_updated = False

        if step_idx % interval == 0:
            gamma = model.readout.gamma_update(theta)
            gate = model.gate.sinusoidal_gating(theta_delay_buffer, theta, gamma)
            gamma_updated = True

        should_update_spike = (step_idx - spike_update_offset) % interval == 0
        should_update_spike = should_update_spike and step_idx > spike_update_offset
        if should_update_spike:
            membrane, spikes = model.snn.forward_step(membrane, spikes, gate, gamma)
            feedback_aij, feedback_alpha = build_feedback_tensors(model, spikes, args)
            if args.feedback_affects_kuramoto:
                kuramoto_aij, kuramoto_alpha = feedback_aij, feedback_alpha
            feedback_updated = True

        theta_history.append(theta.detach().cpu())
        gamma_history.append(gamma.detach().cpu())
        spike_history.append(spikes.detach().cpu())
        theta_delay_buffer.append(theta)
        max_delay_history = max(1, int(config.delay) + 1)
        if len(theta_delay_buffer) > max_delay_history:
            theta_delay_buffer.pop(0)

        row = {
            "step": step_idx,
            "gamma_updated": float(gamma_updated),
            "feedback_updated": float(feedback_updated),
            "feedback_affects_kuramoto": float(args.feedback_affects_kuramoto),
            "kuramoto_coupling_strength": float(model.kuramoto.global_coupling_strength),
            "gamma_attraction_strength": float(model.kuramoto.gamma_attraction_strength),
            **scalar_alignment(theta, gamma),
            "spike_mean": float(spikes.mean().item()),
            "spike_max": float(spikes.max().item()),
            **tensor_stats("aij", feedback_aij),
            **tensor_stats("alpha", feedback_alpha),
        }
        metrics.append(row)

    history = {
        "theta": torch.stack(theta_history, dim=1),
        "gamma": torch.stack(gamma_history, dim=1),
        "spikes": torch.stack(spike_history, dim=1),
    }
    return metrics, history


def selected_step_indices(total_steps: int, visual_steps: List[int]) -> List[int]:
    requested = [step for step in visual_steps if 1 <= step <= total_steps]
    if not requested:
        requested = list(range(1, total_steps + 1))
    requested.append(total_steps)
    return sorted({step - 1 for step in requested})


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().float().cpu().numpy()


def classifier_mask_union(classifier_output: Dict[str, torch.Tensor]) -> np.ndarray | None:
    """Collapse classifier component masks into one display mask."""
    masks = classifier_output.get("masks")
    if masks is None or masks.numel() == 0:
        return None
    return masks.detach().float().cpu().sum(dim=0).clamp(0.0, 1.0).numpy()


def save_diagnostic_plots(
    output_dir: Path,
    x: torch.Tensor,
    sample_metadata: Dict[str, torch.Tensor],
    classifier_output: Dict[str, torch.Tensor],
    history: Dict[str, torch.Tensor],
    metrics: List[Dict[str, float]],
    config: ObjectRepresentationConfig,
    visual_steps: List[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_indices = selected_step_indices(config.steps, visual_steps)
    has_object_masks = "object_masks" in sample_metadata
    predicted_mask = classifier_mask_union(classifier_output)
    has_predicted_mask = predicted_mask is not None
    num_rows = 3 + int(has_predicted_mask) + int(has_object_masks)
    fig, axes = plt.subplots(num_rows, len(step_indices), figsize=(2.5 * len(step_indices), 2.4 * num_rows), constrained_layout=True)
    if len(step_indices) == 1:
        axes = axes.reshape(num_rows, 1)

    start_row = 0
    if has_predicted_mask:
        for col_idx, step_idx in enumerate(step_indices):
            im = axes[start_row, col_idx].imshow(predicted_mask, cmap="gray", vmin=0.0, vmax=1.0)
            axes[start_row, col_idx].set_title(f"classifier mask t={step_idx + 1}")
            axes[start_row, col_idx].set_xticks([])
            axes[start_row, col_idx].set_yticks([])
            axes[start_row, col_idx].figure.colorbar(im, ax=axes[start_row, col_idx], fraction=0.046, pad=0.04)
        start_row += 1

    if has_object_masks:
        object_union = sample_metadata["object_masks"].sum(dim=0).clamp(0.0, 1.0).numpy()
        for col_idx, step_idx in enumerate(step_indices):
            im = axes[start_row, col_idx].imshow(object_union, cmap="gray", vmin=0.0, vmax=1.0)
            axes[start_row, col_idx].set_title(f"GT objects t={step_idx + 1}")
            axes[start_row, col_idx].set_xticks([])
            axes[start_row, col_idx].set_yticks([])
            axes[start_row, col_idx].figure.colorbar(im, ax=axes[start_row, col_idx], fraction=0.046, pad=0.04)
        start_row += 1

    for col_idx, step_idx in enumerate(step_indices):
        for row_idx, (key, cmap) in enumerate((("theta", "coolwarm"), ("gamma", "coolwarm"), ("spikes", "magma"))):
            values = to_numpy(history[key][0, step_idx]).reshape(config.image_height, config.image_width)
            axis = axes[start_row + row_idx, col_idx]
            im = axis.imshow(values, cmap=cmap)
            axis.set_title(f"{key} t={step_idx + 1}")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "theta_gamma_spikes_over_time.png", dpi=160)
    plt.close(fig)

    summary_cols = 3 + int(has_predicted_mask) + int(has_object_masks)
    fig, axes = plt.subplots(1, summary_cols, figsize=(3.7 * summary_cols, 3.5), constrained_layout=True)
    image = to_numpy(x[0]).clip(0.0, 1.0)
    axes[0].imshow(image)
    axes[0].set_title("input")
    axes[0].axis("off")

    metric_axis_offset = 1
    if has_predicted_mask:
        axes[metric_axis_offset].imshow(predicted_mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[metric_axis_offset].set_title("classifier mask")
        axes[metric_axis_offset].axis("off")
        metric_axis_offset += 1

    if has_object_masks:
        label_map = sample_metadata["label_map"].numpy()
        axes[metric_axis_offset].imshow(label_map, cmap="tab20")
        axes[metric_axis_offset].set_title("GT label map")
        axes[metric_axis_offset].axis("off")
        metric_axis_offset += 1

    steps = [row["step"] for row in metrics]
    axes[metric_axis_offset].plot(steps, [row["theta_gamma_mse"] for row in metrics], label="MSE")
    axes[metric_axis_offset].plot(steps, [row["theta_gamma_mae"] for row in metrics], label="MAE")
    axes[metric_axis_offset].set_title("theta-gamma distance")
    axes[metric_axis_offset].set_xlabel("t")
    axes[metric_axis_offset].legend()

    axes[metric_axis_offset + 1].plot(steps, [row["theta_gamma_corr"] for row in metrics], label="corr")
    axes[metric_axis_offset + 1].plot(steps, [row["theta_gamma_cosine"] for row in metrics], label="cosine")
    axes[metric_axis_offset + 1].set_title("theta-gamma alignment")
    axes[metric_axis_offset + 1].set_xlabel("t")
    axes[metric_axis_offset + 1].legend()
    fig.savefig(output_dir / "theta_gamma_metrics.png", dpi=160)
    plt.close(fig)


def write_metrics(
    output_dir: Path,
    metrics: List[Dict[str, float]],
    config: ObjectRepresentationConfig,
    args: argparse.Namespace,
    pretrain_losses: List[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "purpose": "Kuramoto-Gamma relationship diagnostic",
                "steps": config.steps,
                "gamma_update_interval": config.readout_update_interval,
                "kuramoto_coupling_strength": config.global_coupling_strength,
                "gamma_attraction_schedule": args.gamma_attraction_schedule,
                "gamma_attraction_strength": args.gamma_attraction_strength,
                "gamma_attraction_values": args.gamma_attraction_values,
                "gamma_attraction_boundaries": args.gamma_attraction_boundaries,
                "feedback_mode": args.feedback_mode,
                "fixed_affinity_value": args.fixed_affinity_value,
                "fixed_alpha_value": args.fixed_alpha_value,
                "aij_alpha_computed": True,
                "aij_alpha_affects_kuramoto": bool(args.feedback_affects_kuramoto),
                "gamma_encoder_pretrained": bool(pretrain_losses),
                "gamma_encoder_pretrain_losses": pretrain_losses,
                "config": config.__dict__,
                "final_metrics": metrics[-1],
            },
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Kuramoto theta and gamma relationship.")
    parser.add_argument("--hdf5_path", type=str, default="/Data0/tkim1/datasets/object_centric_data/clevr_10-full.hdf5")
    parser.add_argument("--output_dir", type=str, default="outputs/kuramoto_gamma_diagnostic")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--gamma_update_interval", type=int, default=5)
    parser.add_argument("--spike_update_offset", type=int, default=0)
    parser.add_argument("--classifier_start_step", type=int, default=10)
    parser.add_argument("--classifier_type", type=str, default="mean_spike")
    parser.add_argument("--classifier_similarity_threshold", type=float, default=0.60)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--max_objects", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--patch_stride", type=int, default=64)
    parser.add_argument("--no_patch", action="store_true")
    parser.add_argument("--min_object_pixels", type=int, default=1)
    parser.add_argument("--use_random_input", action="store_true")
    parser.add_argument("--step_size", type=float, default=0.15)
    parser.add_argument("--global_coupling_strength", type=float, default=1.0)
    parser.add_argument("--gamma_attraction_strength", type=float, default=3.0)
    parser.add_argument("--gamma_attraction_schedule", type=str, default="constant", choices=["constant", "step_decay"])
    parser.add_argument("--gamma_attraction_values", type=str, default="2.5,2.0,1.5,1.0,0.5")
    parser.add_argument("--gamma_attraction_boundaries", type=str, default="5,10,15,20")
    parser.add_argument("--coupling_chunk_size", type=int, default=128)
    parser.add_argument("--membrane_decay", type=float, default=0.92)
    parser.add_argument("--recurrent_scale", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--alpha_scale", type=float, default=5.0)
    parser.add_argument("--feedback_mode", type=str, default="fixed_affinity", choices=["fixed_affinity", "spike_feedback"])
    parser.add_argument("--feedback_affects_kuramoto", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixed_affinity_value", type=float, default=1.0)
    parser.add_argument("--fixed_alpha_value", type=float, default=0.0)
    parser.add_argument("--feedback_theta_connectivity_weight_scale", type=float, default=0.25)
    parser.add_argument("--feedback_alpha_scale", type=float, default=0.25)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--gamma_initialization", type=str, default="encoder")
    parser.add_argument("--gamma_autoencoder_latent_dim", type=int, default=32)
    parser.add_argument("--gamma_patch_size", type=int, default=2)
    parser.add_argument("--gamma_update_scale", type=float, default=1.0)
    parser.add_argument("--pretrain_gamma_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrain_gamma_epochs", type=int, default=5)
    parser.add_argument("--pretrain_gamma_samples", type=int, default=100)
    parser.add_argument("--pretrain_gamma_batch_size", type=int, default=4)
    parser.add_argument("--pretrain_gamma_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_gamma_weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preserve_gamma_value_amplitude", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma_value_floor", type=float, default=0.0)
    parser.add_argument("--visual_steps", type=str, default="1,5,10,15,20,25,30")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    config = build_config(args)
    device = torch.device(config.device)

    model = ObjectRepresentationSNN(config).to(device)
    pretrain_losses = pretrain_gamma_encoder(model, args, device)
    model.eval()
    x, sample_metadata = load_sample(args, device)

    metrics, history = run_diagnostic(model, x, args)
    classifier_output = model.snn.classify(history["spikes"].to(device))
    visual_steps = [int(item.strip()) for item in args.visual_steps.split(",") if item.strip()]
    write_metrics(output_dir, metrics, config, args, pretrain_losses)
    save_diagnostic_plots(output_dir, x, sample_metadata, classifier_output, history, metrics, config, visual_steps)

    print("=== Kuramoto-Gamma Diagnostic ===")
    print(f"steps: {config.steps}")
    print(f"gamma_update_interval: {config.readout_update_interval}")
    print(f"kuramoto_coupling_strength: {config.global_coupling_strength}")
    print(f"gamma_attraction_schedule: {args.gamma_attraction_schedule}")
    if args.gamma_attraction_schedule != "constant":
        print(f"gamma_attraction_values: {args.gamma_attraction_values}")
        print(f"gamma_attraction_boundaries: {args.gamma_attraction_boundaries}")
    print(f"feedback_mode: {args.feedback_mode}")
    print(f"feedback_affects_kuramoto: {args.feedback_affects_kuramoto}")
    print(f"fixed_affinity_value: {args.fixed_affinity_value}")
    print(f"fixed_alpha_value: {args.fixed_alpha_value}")
    print(f"gamma_initialization: {config.gamma_initialization}")
    print(f"gamma_encoder_pretrained: {bool(pretrain_losses)}")
    print("Aij/alpha computed: yes")
    print(f"Aij/alpha affect Kuramoto: {bool(args.feedback_affects_kuramoto)}")
    print(f"output_dir: {output_dir}")
    print("final_metrics:")
    for key, value in metrics[-1].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
