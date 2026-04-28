from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.data import CLEVRObjectDataset
from re_zero import ObjectRepresentationConfig, ObjectRepresentationSNN
from re_zero.test_re_zero import save_visualizations


def parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated integer list such as '20,40,60'."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> ObjectRepresentationConfig:
    """Create the re_zero config from CLI hyperparameters."""
    return ObjectRepresentationConfig(
        image_height=args.image_size,
        image_width=args.image_size,
        input_channels=3,
        osc_dim=args.osc_dim,
        steps=args.steps,
        readout_update_interval=args.readout_update_interval,
        spike_update_offset=args.spike_update_offset,
        classifier_start_step=args.classifier_start_step,
        classifier_type=args.classifier_type,
        classifier_similarity_threshold=args.classifier_similarity_threshold,
        object_loss_function=args.object_loss_function,
        within_object_similarity_weight=args.within_object_similarity_weight,
        between_object_difference_weight=args.between_object_difference_weight,
        object_density_weight=args.object_density_weight,
        between_object_distance_weight=args.between_object_distance_weight,
        background_suppression_weight=args.background_suppression_weight,
        object_density_target=args.object_density_target,
        object_time_distance_scale=args.object_time_distance_scale,
        step_size=args.step_size,
        global_coupling_strength=args.global_coupling_strength,
        coupling_chunk_size=args.coupling_chunk_size,
        gamma_attraction_strength=args.gamma_attraction_strength,
        membrane_decay=args.membrane_decay,
        recurrent_scale=args.recurrent_scale,
        threshold=args.threshold,
        alpha_scale=args.alpha_scale,
        fixed_alpha_during_training=args.fixed_alpha_during_training,
        fixed_alpha_value=args.fixed_alpha_value,
        feedback_theta_connectivity_weight_scale=args.feedback_theta_connectivity_weight_scale,
        feedback_alpha_scale=args.feedback_alpha_scale,
        delay=args.delay,
        gamma_initialization=args.gamma_initialization,
        gamma_encoder_hidden=args.gamma_encoder_hidden,
        gamma_encoder_blur_kernel=args.gamma_encoder_blur_kernel,
        gamma_encoder_skip_scale=args.gamma_encoder_skip_scale,
        gamma_autoencoder_latent_dim=args.gamma_autoencoder_latent_dim,
        gamma_patch_size=args.gamma_patch_size,
        gamma_update_scale=args.gamma_update_scale,
        preserve_gamma_value_amplitude=args.preserve_gamma_value_amplitude,
        gamma_value_floor=args.gamma_value_floor,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )


def split_binary_mask_components(mask: torch.Tensor) -> List[torch.Tensor]:
    """Split a binary 2D mask into 4-connected components."""
    mask_cpu = mask.detach().cpu().bool()
    height, width = mask_cpu.shape
    visited = torch.zeros_like(mask_cpu, dtype=torch.bool)
    components: List[torch.Tensor] = []

    for row in range(height):
        for col in range(width):
            if not bool(mask_cpu[row, col]) or bool(visited[row, col]):
                continue
            component = torch.zeros_like(mask_cpu, dtype=torch.bool)
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                y, x = stack.pop()
                component[y, x] = True
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                        continue
                    if bool(visited[next_y, next_x]) or not bool(mask_cpu[next_y, next_x]):
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
            components.append(component.to(device=mask.device))
    return components


def pairwise_iou(pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    """Compute IoU between predicted masks and ground-truth masks."""
    if pred_masks.numel() == 0 or target_masks.numel() == 0:
        return target_masks.new_zeros((pred_masks.shape[0], target_masks.shape[0]))
    pred = pred_masks.flatten(start_dim=1).bool()
    target = target_masks.flatten(start_dim=1).bool()
    intersection = (pred.unsqueeze(1) & target.unsqueeze(0)).sum(dim=-1).float()
    union = (pred.unsqueeze(1) | target.unsqueeze(0)).sum(dim=-1).float().clamp_min(1.0)
    return intersection / union


def greedy_one_to_one_match(iou: torch.Tensor) -> List[Tuple[int, int, float]]:
    """Greedily match predicted and target masks by descending IoU."""
    pairs = []
    for pred_idx in range(iou.shape[0]):
        for target_idx in range(iou.shape[1]):
            pairs.append((float(iou[pred_idx, target_idx].item()), pred_idx, target_idx))
    pairs.sort(reverse=True)

    matched_pred = set()
    matched_target = set()
    matches = []
    for score, pred_idx, target_idx in pairs:
        if pred_idx in matched_pred or target_idx in matched_target:
            continue
        matched_pred.add(pred_idx)
        matched_target.add(target_idx)
        matches.append((pred_idx, target_idx, score))
    return matches


def classifier_masks_to_components(
    classifier_output: Dict[str, torch.Tensor],
    min_pixels: int,
) -> torch.Tensor:
    """Turn classifier masks into connected candidate masks."""
    masks = classifier_output["masks"].detach()
    candidates = []
    for mask in masks:
        for component in split_binary_mask_components(mask > 0.5):
            if int(component.sum().item()) >= int(min_pixels):
                candidates.append(component.float())
    if not candidates:
        return masks.new_zeros((0, masks.shape[-2], masks.shape[-1]))
    return torch.stack(candidates, dim=0)


def score_one_sample(
    classifier_output: Dict[str, torch.Tensor],
    object_masks: torch.Tensor,
    iou_threshold: float,
    min_pixels: int,
) -> Dict[str, float]:
    """Score predicted classifier masks against CLEVR ground-truth object masks."""
    valid_targets = object_masks.sum(dim=(1, 2)) > 0
    targets = object_masks[valid_targets].float()
    if targets.shape[0] == 0:
        return {
            "score_90_coverage": 0.0,
            "score_70_coverage": 0.0,
            "score_50_coverage": 0.0,
            "score_iou_threshold_coverage": 0.0,
            "score_mean_one_to_one_iou": 0.0,
            "num_masks": 0.0,
            "num_candidates": 0.0,
        }

    candidates = classifier_masks_to_components(classifier_output, min_pixels=min_pixels)
    iou = pairwise_iou(candidates.cpu(), targets.cpu())
    matches = greedy_one_to_one_match(iou)
    matched_ious = [score for _, _, score in matches]

    return {
        "score_90_coverage": sum(score >= 0.90 for score in matched_ious) / float(targets.shape[0]),
        "score_70_coverage": sum(score >= 0.70 for score in matched_ious) / float(targets.shape[0]),
        "score_50_coverage": sum(score >= 0.50 for score in matched_ious) / float(targets.shape[0]),
        "score_iou_threshold_coverage": sum(score >= iou_threshold for score in matched_ious) / float(targets.shape[0]),
        "score_mean_one_to_one_iou": sum(matched_ious) / float(targets.shape[0]),
        "num_masks": float(targets.shape[0]),
        "num_candidates": float(candidates.shape[0]),
    }


@torch.no_grad()
def evaluate_model(
    model: ObjectRepresentationSNN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate trained model on a held-out CLEVR split."""
    model.eval()
    totals = {
        "test_loss": 0.0,
        "score_90_coverage": 0.0,
        "score_70_coverage": 0.0,
        "score_50_coverage": 0.0,
        "score_iou_threshold_coverage": 0.0,
        "score_mean_one_to_one_iou": 0.0,
        "num_masks": 0.0,
        "num_candidates": 0.0,
    }
    seen = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        object_masks = batch["object_masks"]
        classifier_output, _ = model(images)
        loss = model.unsupervised_object_loss(classifier_output)

        batch_size = images.shape[0]
        totals["test_loss"] += float(loss.item()) * batch_size
        seen += batch_size

        if batch_size != 1:
            raise ValueError("The current re_zero classifier scoring path expects batch_size=1.")
        scores = score_one_sample(
            classifier_output={key: value.detach().cpu() for key, value in classifier_output.items()},
            object_masks=object_masks[0].detach().cpu(),
            iou_threshold=args.iou_threshold,
            min_pixels=args.min_pixels,
        )
        for key, value in scores.items():
            totals[key] += value

        if args.max_eval_batches is not None and batch_idx + 1 >= args.max_eval_batches:
            break

    return {key: value / max(seen, 1) for key, value in totals.items()}


def save_artifacts(
    model: ObjectRepresentationSNN,
    sample: Dict[str, torch.Tensor],
    cfg: ObjectRepresentationConfig,
    run_dir: Path,
    device: torch.device,
) -> None:
    """Save a quick visual summary and raw history tensors for one sample."""
    model.eval()
    image = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        classifier_output, history = model(
            image,
            return_history=True,
            return_pairwise_history=False,
        )
    save_visualizations(
        output_dir=run_dir / "visualizations",
        x=image,
        classifier_output=classifier_output,
        history=history,
        config=cfg,
    )
    torch.save(
        {
            "classifier_output": {key: value.detach().cpu() for key, value in classifier_output.items()},
            "history": {key: value.detach().cpu() for key, value in history.items()},
            "object_masks": sample["object_masks"].detach().cpu(),
            "label_map": sample["label_map"].detach().cpu(),
        },
        run_dir / "artifact_sample.pt",
    )


def train_one_run(args: argparse.Namespace) -> Dict[str, float]:
    """Train one CLEVR run with the provided hyperparameters."""
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    device = torch.device(cfg.device)

    patch_size = None if args.no_patch else args.patch_size
    patch_stride = None if args.no_patch else args.patch_stride
    test_samples = args.test_samples
    if test_samples is None:
        test_samples = int(math.ceil(args.train_samples * (1.0 - args.train_fraction) / args.train_fraction))

    train_dataset = CLEVRObjectDataset(
        hdf5_path=args.hdf5_path,
        target_size=args.image_size,
        max_objects=args.max_objects,
        split="train",
        train_fraction=args.train_fraction,
        max_samples=args.train_samples,
        patch_size=patch_size,
        patch_stride=patch_stride,
        min_object_pixels=args.min_object_pixels if args.object_patches_only else 0,
    )
    test_dataset = CLEVRObjectDataset(
        hdf5_path=args.hdf5_path,
        target_size=args.image_size,
        max_objects=args.max_objects,
        split="test",
        train_fraction=args.train_fraction,
        max_samples=test_samples,
        patch_size=patch_size,
        patch_stride=patch_stride,
        min_object_pixels=args.min_object_pixels if args.object_patches_only else 0,
    )

    model = ObjectRepresentationSNN(cfg).to(device)
    optimizer = model.build_adam_optimizer()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    print(f"Config: image_size={args.image_size}, steps={args.steps}, batch_size={args.batch_size}, device={device}")

    train_losses = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)

            optimizer.zero_grad(set_to_none=True)
            classifier_output, _ = model(images)
            loss = model.unsupervised_object_loss(classifier_output)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size

            if args.max_train_batches is not None and batch_idx + 1 >= args.max_train_batches:
                break

        epoch_loss = total_loss / max(total_seen, 1)
        train_losses.append(epoch_loss)
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={epoch_loss:.6f}")

    metrics = evaluate_model(model, test_loader, device, args)
    metrics["final_train_loss"] = train_losses[-1] if train_losses else math.nan

    summary = {
        "run_name": args.run_name,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "config": cfg.__dict__,
        "train_losses": train_losses,
        **metrics,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[key for key in summary.keys() if key not in {"config", "train_losses"}])
        writer.writeheader()
        writer.writerow({key: value for key, value in summary.items() if key not in {"config", "train_losses"}})

    if args.save_artifacts and len(test_dataset) > 0:
        save_artifacts(model, test_dataset[0], cfg, run_dir, device)
    if args.save_checkpoint:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, run_dir / "checkpoint.pt")

    print("=== Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"Saved run directory: {run_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the current re_zero model on CLEVR object masks.")
    parser.add_argument("--hdf5_path", type=str, default="/Data0/tkim1/datasets/object_centric_data/clevr_10-full.hdf5")
    parser.add_argument("--output_dir", type=str, default="outputs/rezero_clevr_train")
    parser.add_argument("--run_name", type=str, default="single_run")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--patch_stride", type=int, default=64)
    parser.add_argument("--no_patch", action="store_true")
    parser.add_argument("--object_patches_only", action="store_true")
    parser.add_argument("--min_object_pixels", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=10)
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--test_samples", type=int, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--readout_update_interval", type=int, default=10)
    parser.add_argument("--spike_update_offset", type=int, default=0)
    parser.add_argument("--classifier_start_step", type=int, default=60)
    parser.add_argument("--classifier_type", type=str, default="mean_spike", choices=["mean_spike", "spike_feature"])
    parser.add_argument("--classifier_similarity_threshold", type=float, default=0.60)
    parser.add_argument("--object_loss_function", type=str, default="1234", choices=["1234", "123", "124"])
    parser.add_argument("--osc_dim", type=int, default=1)
    parser.add_argument("--step_size", type=float, default=0.15)
    parser.add_argument("--global_coupling_strength", type=float, default=1.0)
    parser.add_argument("--coupling_chunk_size", type=int, default=128)
    parser.add_argument("--gamma_attraction_strength", type=float, default=3.0)
    parser.add_argument("--membrane_decay", type=float, default=0.92)
    parser.add_argument("--recurrent_scale", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--alpha_scale", type=float, default=5.0)
    parser.add_argument("--fixed_alpha_during_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed_alpha_value", type=float, default=0.0)
    parser.add_argument("--feedback_theta_connectivity_weight_scale", type=float, default=0.25)
    parser.add_argument("--feedback_alpha_scale", type=float, default=0.25)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--within_object_similarity_weight", type=float, default=1.0)
    parser.add_argument("--between_object_difference_weight", type=float, default=1.0)
    parser.add_argument("--object_density_weight", type=float, default=1.0)
    parser.add_argument("--between_object_distance_weight", type=float, default=1.0)
    parser.add_argument("--background_suppression_weight", type=float, default=3.0)
    parser.add_argument("--object_density_target", type=float, default=0.6)
    parser.add_argument("--object_time_distance_scale", type=float, default=10.0)
    parser.add_argument("--gamma_initialization", type=str, default="encoder")
    parser.add_argument("--gamma_encoder_hidden", type=int, default=16)
    parser.add_argument("--gamma_encoder_blur_kernel", type=int, default=1)
    parser.add_argument("--gamma_encoder_skip_scale", type=float, default=0.10)
    parser.add_argument("--gamma_autoencoder_latent_dim", type=int, default=32)
    parser.add_argument("--gamma_patch_size", type=int, default=2)
    parser.add_argument("--gamma_update_scale", type=float, default=1.0)
    parser.add_argument("--preserve_gamma_value_amplitude", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma_value_floor", type=float, default=0.0)
    parser.add_argument("--iou_threshold", type=float, default=0.9)
    parser.add_argument("--min_pixels", type=int, default=8)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--save_artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        print("Warning: current re_zero classifier/loss path is designed for batch_size=1.")
    if args.image_size >= 64:
        print("Warning: 64x64 creates 4096 oscillators and pairwise feedback matrices. Start small when debugging.")
    train_one_run(args)


if __name__ == "__main__":
    main()
