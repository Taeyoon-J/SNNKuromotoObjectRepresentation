from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.data import CLEVRObjectDataset
from model.hyperparameters import ObjectRepresentationConfig
from model.s2net import ObjectRepresentationSNN


def parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated integer list such as '10,20'."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated float list such as '0.5,0.25'."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    """Make one run as reproducible as possible."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(
    readout_update_interval: int,
    spike_update_offset: int,
    classifier_start_step: int,
    steps: int,
    feedback_magnitude: float,
    args: argparse.Namespace,
) -> ObjectRepresentationConfig:
    """Create a model config for one sweep run."""
    return ObjectRepresentationConfig(
        image_height=args.image_size,
        image_width=args.image_size,
        input_channels=3,
        osc_dim=args.osc_dim,
        steps=steps,
        readout_update_interval=readout_update_interval,
        spike_update_offset=spike_update_offset,
        classifier_start_step=classifier_start_step,
        feedback_affinity_scale=feedback_magnitude,
        feedback_alpha_scale=feedback_magnitude,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        coupling=args.coupling,
        attraction_strength=args.attraction_strength,
        coupling_chunk_size=args.coupling_chunk_size,
        channel_wise_coupling=args.channel_wise_coupling,
        fixed_alpha_during_training=args.fixed_alpha_during_training,
        fixed_alpha_value=args.fixed_alpha_value,
        background_suppression_weight=args.background_suppression_weight,
        gamma_encoder_hidden=args.gamma_encoder_hidden,
        gamma_encoder_blur_kernel=args.gamma_encoder_blur_kernel,
        gamma_encoder_skip_scale=args.gamma_encoder_skip_scale,
        preserve_gamma_value_amplitude=args.preserve_gamma_value_amplitude,
        gamma_value_floor=args.gamma_value_floor,
        seed=args.seed,
        device=args.device,
    )


def get_loss(model: ObjectRepresentationSNN, loss_name: str, spike_trace: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
    """Select one of the three object-binding losses."""
    if loss_name == "1234":
        return model.unsupervised_object_loss_1234(spike_trace, object_masks)
    if loss_name == "123":
        return model.unsupervised_object_loss_123(spike_trace, object_masks)
    if loss_name == "124":
        return model.unsupervised_object_loss_124(spike_trace, object_masks)
    raise ValueError(f"Unknown loss function {loss_name!r}")


def spike_trace_to_candidate_masks(
    spike_trace: torch.Tensor,
    image_height: int,
    image_width: int,
    input_channels: int,
    start_step: int,
    score_quantile: float,
    min_pixels: int,
) -> torch.Tensor:
    """
    Convert temporal spike maps into candidate object masks.

    Each candidate is "what the model highlighted at one time step." This fits
    our current hypothesis: different bounded objects should spike at different
    times. The candidate mask is made by thresholding the per-pixel spike map at
    a high quantile.
    """
    start_idx = min(max(start_step - 1, 0), spike_trace.shape[0] - 1)
    late_spikes = spike_trace[start_idx:]
    if late_spikes.shape[-1] == image_height * image_width:
        spike_maps = late_spikes.view(-1, image_height, image_width)
    else:
        raise ValueError(f"Expected pixel-level spike trace with {image_height * image_width} nodes, got {late_spikes.shape[-1]}")

    candidates = []
    for spike_map in spike_maps:
        threshold = torch.quantile(spike_map.flatten(), score_quantile)
        candidate = spike_map >= threshold
        for component in split_binary_mask_components(candidate):
            if int(component.sum().item()) >= min_pixels:
                candidates.append(component.float())

    if not candidates:
        return spike_maps.new_zeros((0, image_height, image_width))
    return torch.stack(candidates, dim=0)


def split_binary_mask_components(mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Split a binary 2D mask into 4-connected components.

    Spike maps often highlight two disconnected objects in the same timestep.
    Treating that as one candidate mask destroys IoU, so the scorer evaluates
    each connected region as a separate object hypothesis.
    """
    if mask.dim() != 2:
        raise ValueError(f"Expected a 2D binary mask, got {tuple(mask.shape)}")

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
    """Compute IoU between predicted candidate masks and ground-truth object masks."""
    if pred_masks.numel() == 0 or target_masks.numel() == 0:
        return target_masks.new_zeros((pred_masks.shape[0], target_masks.shape[0]))

    pred = pred_masks.flatten(start_dim=1).bool()
    target = target_masks.flatten(start_dim=1).bool()
    intersection = (pred.unsqueeze(1) & target.unsqueeze(0)).sum(dim=-1).float()
    union = (pred.unsqueeze(1) | target.unsqueeze(0)).sum(dim=-1).float().clamp_min(1.0)
    return intersection / union


def greedy_one_to_one_match(iou: torch.Tensor) -> List[Tuple[int, int, float]]:
    """
    Match predicted objects to masks greedily by IoU.

    This enforces the user's rule: one model object can only be compared with
    one ground-truth mask, and one ground-truth mask can receive at most one
    model object.
    """
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


def score_one_sample(
    spike_trace: torch.Tensor,
    object_masks: torch.Tensor,
    cfg: ObjectRepresentationConfig,
    iou_threshold: float,
    score_quantile: float,
    min_pixels: int,
) -> Dict[str, float]:
    """
    Score how well temporal spike-derived objects match ground-truth masks.

    score_90_coverage:
        fraction of ground-truth masks represented with IoU >= 0.90.

    score_mean_one_to_one_iou:
        average matched IoU over all ground-truth masks, with unmatched masks
        counted as 0.
    """
    valid_targets = object_masks.sum(dim=(1, 2)) > 0
    targets = object_masks[valid_targets].float()
    if targets.shape[0] == 0:
        return {
            "score_90_coverage": 0.0,
            "score_50_coverage": 0.0,
            "score_70_coverage": 0.0,
            "score_iou_threshold_coverage": 0.0,
            "score_mean_one_to_one_iou": 0.0,
            "num_masks": 0.0,
            "num_candidates": 0.0,
        }

    candidates = spike_trace_to_candidate_masks(
        spike_trace=spike_trace,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        input_channels=cfg.input_channels,
        start_step=cfg.classifier_start_step,
        score_quantile=score_quantile,
        min_pixels=min_pixels,
    )
    iou = pairwise_iou(candidates, targets.to(candidates.device))
    matches = greedy_one_to_one_match(iou)

    matched_ious = [score for _, _, score in matches]
    represented = sum(score >= iou_threshold for score in matched_ious)
    represented_50 = sum(score >= 0.50 for score in matched_ious)
    represented_70 = sum(score >= 0.70 for score in matched_ious)
    represented_90 = sum(score >= 0.90 for score in matched_ious)
    mean_iou = sum(matched_ious) / float(targets.shape[0])
    return {
        "score_90_coverage": represented_90 / float(targets.shape[0]),
        "score_50_coverage": represented_50 / float(targets.shape[0]),
        "score_70_coverage": represented_70 / float(targets.shape[0]),
        "score_iou_threshold_coverage": represented / float(targets.shape[0]),
        "score_mean_one_to_one_iou": mean_iou,
        "num_masks": float(targets.shape[0]),
        "num_candidates": float(candidates.shape[0]),
    }


@torch.no_grad()
def evaluate_model(
    model: ObjectRepresentationSNN,
    loader: DataLoader,
    cfg: ObjectRepresentationConfig,
    loss_name: str,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate one trained model on the 20% test split."""
    model.eval()
    totals = {
        "test_loss": 0.0,
        "score_90_coverage": 0.0,
        "score_50_coverage": 0.0,
        "score_70_coverage": 0.0,
        "score_iou_threshold_coverage": 0.0,
        "score_mean_one_to_one_iou": 0.0,
        "num_masks": 0.0,
        "num_candidates": 0.0,
    }
    seen = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        object_masks = batch["object_masks"].to(device)
        _, history = model(images, return_spike_trace=True)
        spike_trace = history["spikes"]
        loss = get_loss(model, loss_name, spike_trace, object_masks)

        batch_size = images.shape[0]
        totals["test_loss"] += float(loss.item()) * batch_size
        for sample_idx in range(batch_size):
            sample_scores = score_one_sample(
                spike_trace[sample_idx].detach().cpu(),
                object_masks[sample_idx].detach().cpu(),
                cfg=cfg,
                iou_threshold=args.iou_threshold,
                score_quantile=args.score_quantile,
                min_pixels=args.min_pixels,
            )
            for key in (
                "score_90_coverage",
                "score_50_coverage",
                "score_70_coverage",
                "score_iou_threshold_coverage",
                "score_mean_one_to_one_iou",
                "num_masks",
                "num_candidates",
            ):
                totals[key] += sample_scores[key]
        seen += batch_size

        if args.max_eval_batches is not None and batch_idx + 1 >= args.max_eval_batches:
            break

    return {key: value / max(seen, 1) for key, value in totals.items()}


def save_history_artifacts(
    model: ObjectRepresentationSNN,
    sample: Dict[str, torch.Tensor],
    cfg: ObjectRepresentationConfig,
    run_dir: Path,
    device: str,
    args: argparse.Namespace,
) -> None:
    """Save theta/gamma/spike maps and a PNG for later visual inspection."""
    from model.visualization import visualize_scheduled_kuramoto_readout

    model.eval()
    image = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        _, history = model(image, return_history=True)
        gamma0 = history["gamma0"][0].detach().cpu()

    steps_to_show = sorted(set([step for step in args.visual_steps if 1 <= step <= cfg.steps]))
    if not steps_to_show:
        steps_to_show = list(range(cfg.readout_update_interval, cfg.steps + 1, cfg.readout_update_interval))

    figure = visualize_scheduled_kuramoto_readout(
        input_image=sample["image"],
        gamma0=gamma0,
        history={key: value.detach().cpu() for key, value in history.items()},
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        input_channels=cfg.input_channels,
        steps_to_show=steps_to_show,
        object_masks=sample["object_masks"],
        label_map=sample["label_map"],
    )
    figure.savefig(run_dir / "theta_gamma_spikes.png", dpi=160, bbox_inches="tight")

    theta_vectors = history["theta"][0].view(cfg.steps, cfg.image_height, cfg.image_width, cfg.osc_dim)
    gamma_vectors = history["gamma"][0].view(cfg.steps, cfg.image_height, cfg.image_width, cfg.osc_dim)
    theta_phase = torch.atan2(theta_vectors[..., 1], theta_vectors[..., 0])
    gamma_phase = torch.atan2(gamma_vectors[..., 1], gamma_vectors[..., 0])
    theta_magnitude = theta_vectors.norm(dim=-1)
    gamma_magnitude = gamma_vectors.norm(dim=-1)
    gamma_components = gamma_vectors[..., : min(3, cfg.osc_dim)]
    if history["spikes"].shape[-1] == cfg.image_height * cfg.image_width:
        spike_map = history["spikes"][0].view(cfg.steps, cfg.image_height, cfg.image_width)
    else:
        raise ValueError(f"Expected pixel-level spikes with {cfg.image_height * cfg.image_width} nodes, got {history['spikes'].shape[-1]}")

    np.savez_compressed(
        run_dir / "history_maps.npz",
        theta0=history["theta0"][0].detach().cpu().numpy(),
        gamma0=history["gamma0"][0].detach().cpu().numpy(),
        theta_phase=theta_phase.detach().cpu().numpy(),
        gamma_phase=gamma_phase.detach().cpu().numpy(),
        theta_magnitude=theta_magnitude.detach().cpu().numpy(),
        gamma_magnitude=gamma_magnitude.detach().cpu().numpy(),
        gamma_components=gamma_components.detach().cpu().numpy(),
        spikes=spike_map.detach().cpu().numpy(),
        object_masks=sample["object_masks"].detach().cpu().numpy(),
        label_map=sample["label_map"].detach().cpu().numpy(),
    )

    if args.save_full_history:
        torch.save(
            {
                "theta": history["theta"][0].detach().cpu(),
                "gamma": history["gamma"][0].detach().cpu(),
                "spikes": history["spikes"][0].detach().cpu(),
            },
            run_dir / "history_full.pt",
        )


def make_run_name(
    readout_update_interval: int,
    spike_update_offset: int,
    classifier_start_step: int,
    steps: int,
    loss_name: str,
    feedback_magnitude: float,
) -> str:
    """Build a stable directory name for one run."""
    spike_tag = "same_t" if spike_update_offset == 0 else "t_plus_1"
    feedback_tag = str(feedback_magnitude).replace(".", "p")
    return (
        f"readout{readout_update_interval}_spike_{spike_tag}_"
        f"classifier{classifier_start_step}_steps{steps}_"
        f"loss{loss_name}_feedback{feedback_tag}"
    )


def run_one_experiment(
    run_id: int,
    params: Tuple[int, int, int, int, str, float],
    train_dataset: CLEVRObjectDataset,
    test_dataset: CLEVRObjectDataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Train and evaluate one sweep configuration."""
    readout_update_interval, spike_update_offset, classifier_start_step, steps, loss_name, feedback_magnitude = params
    run_name = make_run_name(*params)
    run_dir = output_dir / f"{run_id:04d}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(
        readout_update_interval=readout_update_interval,
        spike_update_offset=spike_update_offset,
        classifier_start_step=classifier_start_step,
        steps=steps,
        feedback_magnitude=feedback_magnitude,
        args=args,
    )

    set_seed(args.seed + run_id)
    model = ObjectRepresentationSNN(cfg).to(args.device)
    optimizer = model.build_adam_optimizer()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_losses = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch in train_loader:
            images = batch["image"].to(args.device)
            object_masks = batch["object_masks"].to(args.device)

            optimizer.zero_grad(set_to_none=True)
            _, history = model(images, return_spike_trace=True)
            loss = get_loss(model, loss_name, history["spikes"], object_masks)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size

        train_losses.append(total_loss / max(total_seen, 1))

    metrics = evaluate_model(model, test_loader, cfg, loss_name, args.device, args)
    metrics["final_train_loss"] = train_losses[-1] if train_losses else math.nan

    if args.save_artifacts:
        save_history_artifacts(model, test_dataset[0], cfg, run_dir, args.device, args)

    if args.save_checkpoint:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, run_dir / "checkpoint.pt")

    summary = {
        "run_id": run_id,
        "run_name": run_name,
        "readout_update_interval": readout_update_interval,
        "spike_update_offset": spike_update_offset,
        "spike_interval": "readout_update_interval" if spike_update_offset == 0 else "readout_update_interval+1",
        "classifier_start_step": classifier_start_step,
        "t_limit": steps,
        "loss_function": loss_name,
        "feedback_affinity_scale": feedback_magnitude,
        "feedback_alpha_scale": feedback_magnitude,
        "epochs": args.epochs,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        **metrics,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_sweep(args: argparse.Namespace) -> List[Tuple[int, int, int, int, str, float]]:
    """Create all valid sweep combinations."""
    raw_params = itertools.product(
        args.readout_update_intervals,
        args.spike_update_offsets,
        args.classifier_start_steps,
        args.t_limits,
        args.loss_functions,
        args.feedback_magnitudes,
    )
    valid_params = []
    for params in raw_params:
        _, spike_update_offset, classifier_start_step, steps, _, _ = params
        if classifier_start_step > steps:
            continue
        if spike_update_offset not in (0, 1):
            continue
        valid_params.append(params)
    return valid_params


def write_master_results(output_dir: Path, results: List[Dict[str, object]]) -> None:
    """Save sweep results as JSON and CSV."""
    with (output_dir / "sweep_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if not results:
        return

    fieldnames = []
    for result in results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "sweep_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    """Command-line options for the CLEVR sweep runner."""
    parser = argparse.ArgumentParser(description="Train CLEVR object-binding sweep.")
    parser.add_argument("--hdf5_path", type=str, default="/Data0/tkim1/datasets/object_centric_data/clevr_10-full.hdf5")
    parser.add_argument("--output_dir", type=str, default="outputs/clevr_sweep")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--patch_stride", type=int, default=32)
    parser.add_argument("--no_patch", action="store_true")
    parser.add_argument("--object_patches_only", action="store_true")
    parser.add_argument("--min_object_pixels", type=int, default=1)
    parser.add_argument("--osc_dim", type=int, default=4)
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--attraction_strength", type=float, default=3.0)
    parser.add_argument("--coupling_chunk_size", type=int, default=128)
    parser.add_argument("--channel_wise_coupling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixed_alpha_during_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed_alpha_value", type=float, default=0.0)
    parser.add_argument("--background_suppression_weight", type=float, default=3.0)
    parser.add_argument("--gamma_encoder_hidden", type=int, default=16)
    parser.add_argument("--gamma_encoder_blur_kernel", type=int, default=1)
    parser.add_argument("--gamma_encoder_skip_scale", type=float, default=0.10)
    parser.add_argument("--preserve_gamma_value_amplitude", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma_value_floor", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--readout_update_intervals", type=parse_int_list, default=parse_int_list("10,20"))
    parser.add_argument("--spike_update_offsets", type=parse_int_list, default=parse_int_list("0,1"))
    parser.add_argument("--classifier_start_steps", type=parse_int_list, default=parse_int_list("60,80,100"))
    parser.add_argument("--t_limits", type=parse_int_list, default=parse_int_list("80,100,120"))
    parser.add_argument("--loss_functions", type=lambda value: [item.strip() for item in value.split(",") if item.strip()], default=["1234", "123", "124"])
    parser.add_argument("--feedback_magnitudes", type=parse_float_list, default=parse_float_list("0.5,0.25"))
    parser.add_argument("--iou_threshold", type=float, default=0.9)
    parser.add_argument("--score_quantile", type=float, default=0.9)
    parser.add_argument("--min_pixels", type=int, default=8)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_save_artifacts", dest="save_artifacts", action="store_false")
    parser.add_argument("--save_full_history", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--visual_steps", type=parse_int_list, default=parse_int_list("20,40,60,80,100,120"))
    parser.set_defaults(save_artifacts=True)
    return parser.parse_args()


def main() -> None:
    """Run the requested CLEVR sweep."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = build_sweep(args)
    if args.max_runs is not None:
        sweep = sweep[: args.max_runs]

    if args.image_size >= 64 and args.osc_dim >= 4:
        print(
            f"Warning: {args.image_size}x{args.image_size} uses N={args.image_size * args.image_size} pixel oscillator nodes. Pairwise Kuramoto is chunked now, "
            "but this is still heavy. Start with --max_runs 1 and --max_eval_batches 1."
        )

    if args.dry_run:
        print(f"Valid runs: {len(sweep)}")
        for run_id, params in enumerate(sweep):
            print(run_id, make_run_name(*params))
        return

    patch_size = None if args.no_patch else args.patch_size
    patch_stride = None if args.no_patch else args.patch_stride
    test_samples = int(math.ceil(args.train_samples * (1.0 - args.train_fraction) / args.train_fraction))
    train_dataset = CLEVRObjectDataset(
        hdf5_path=args.hdf5_path,
        target_size=args.image_size,
        max_objects=10,
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
        max_objects=10,
        split="test",
        train_fraction=args.train_fraction,
        max_samples=test_samples,
        patch_size=patch_size,
        patch_stride=patch_stride,
        min_object_pixels=args.min_object_pixels if args.object_patches_only else 0,
    )

    print(f"Valid runs: {len(sweep)}")
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    results = []
    for run_id, params in enumerate(sweep):
        print(f"[{run_id + 1}/{len(sweep)}] {make_run_name(*params)}")
        try:
            summary = run_one_experiment(run_id, params, train_dataset, test_dataset, output_dir, args)
        except RuntimeError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary = {"run_id": run_id, "run_name": make_run_name(*params), "error": str(exc)}
            run_dir = output_dir / f"{run_id:04d}_{make_run_name(*params)}"
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        results.append(summary)
        write_master_results(output_dir, results)


if __name__ == "__main__":
    main()
