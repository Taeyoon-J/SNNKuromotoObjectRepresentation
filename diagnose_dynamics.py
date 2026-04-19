from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from model.hyperparameters import ObjectRepresentationConfig
from train_clevr_sweep import score_one_sample


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def valid_object_masks(object_masks: torch.Tensor) -> torch.Tensor:
    valid = object_masks.reshape(object_masks.shape[0], -1).sum(dim=1) > 0
    return object_masks[valid].float()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() <= 0:
        return 0.0
    return float(values[mask].mean().item())


def circular_coherence(phase: torch.Tensor, mask: torch.Tensor) -> float:
    """Magnitude of mean unit phase vector inside a binary mask."""
    if mask.sum() <= 0:
        return 0.0
    unit = torch.exp(1j * phase[mask])
    return float(torch.abs(unit.mean()).item())


def object_activity_centers(spikes: torch.Tensor, object_masks: torch.Tensor, start_idx: int) -> Dict[str, float]:
    """Return mean pairwise distance between object activity centers in late time."""
    targets = valid_object_masks(object_masks)
    if targets.shape[0] < 2:
        return {"object_temporal_center_distance": 0.0}

    late_spikes = spikes[start_idx:].reshape(spikes.shape[0] - start_idx, -1)
    flat_masks = targets.reshape(targets.shape[0], -1)
    mask_sizes = flat_masks.sum(dim=1).clamp_min(1.0)
    activity = torch.einsum("tn,on->to", late_spikes, flat_masks) / mask_sizes.unsqueeze(0)

    time_axis = torch.arange(activity.shape[0], dtype=activity.dtype)
    mass = activity.sum(dim=0).clamp_min(1e-6)
    centers = (activity * time_axis.unsqueeze(1)).sum(dim=0) / mass

    distances = []
    for i in range(targets.shape[0]):
        for j in range(i + 1, targets.shape[0]):
            distances.append(float(torch.abs(centers[i] - centers[j]).item()))
    return {"object_temporal_center_distance": sum(distances) / max(len(distances), 1)}


def best_iou_over_grid(
    spikes: torch.Tensor,
    object_masks: torch.Tensor,
    cfg: ObjectRepresentationConfig,
    score_quantiles: Iterable[float],
    min_pixels_values: Iterable[int],
    iou_threshold: float,
) -> Dict[str, float]:
    best: Dict[str, float] = {
        "best_grid_mean_iou": 0.0,
        "best_grid_cov50": 0.0,
        "best_grid_cov70": 0.0,
        "best_grid_cov90": 0.0,
        "best_grid_num_candidates": 0.0,
        "best_grid_score_quantile": 0.0,
        "best_grid_min_pixels": 0.0,
    }
    spike_trace = spikes.reshape(spikes.shape[0], -1)
    for score_quantile in score_quantiles:
        for min_pixels in min_pixels_values:
            scores = score_one_sample(
                spike_trace=spike_trace,
                object_masks=object_masks,
                cfg=cfg,
                iou_threshold=iou_threshold,
                score_quantile=score_quantile,
                min_pixels=min_pixels,
            )
            if scores["score_mean_one_to_one_iou"] > best["best_grid_mean_iou"]:
                best = {
                    "best_grid_mean_iou": float(scores["score_mean_one_to_one_iou"]),
                    "best_grid_cov50": float(scores["score_50_coverage"]),
                    "best_grid_cov70": float(scores["score_70_coverage"]),
                    "best_grid_cov90": float(scores["score_90_coverage"]),
                    "best_grid_num_candidates": float(scores["num_candidates"]),
                    "best_grid_score_quantile": float(score_quantile),
                    "best_grid_min_pixels": float(min_pixels),
                }
    return best


def summarize_run(
    run_dir: Path,
    image_height: int,
    image_width: int,
    classifier_start_step: int,
    score_quantiles: List[float],
    min_pixels: List[int],
    iou_threshold: float,
) -> Dict[str, float | str]:
    history_path = run_dir / "history_maps.npz"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing {history_path}")

    data = np.load(history_path)
    theta_phase = torch.from_numpy(data["theta_phase"]).float()
    gamma_phase = torch.from_numpy(data["gamma_phase"]).float()
    theta_magnitude = torch.from_numpy(data["theta_magnitude"]).float()
    gamma_magnitude = torch.from_numpy(data["gamma_magnitude"]).float()
    spikes = torch.from_numpy(data["spikes"]).float()
    object_masks = torch.from_numpy(data["object_masks"]).float()

    targets = valid_object_masks(object_masks)
    union = targets.amax(dim=0).bool() if targets.numel() else torch.zeros(image_height, image_width, dtype=torch.bool)
    background = ~union
    start_idx = min(max(classifier_start_step - 1, 0), spikes.shape[0] - 1)

    alignment = torch.cos(theta_phase - gamma_phase)
    final_idx = spikes.shape[0] - 1

    object_theta_coherences = []
    object_gamma_coherences = []
    object_final_spike_means = []
    for target in targets:
        mask = target.bool()
        object_theta_coherences.append(circular_coherence(theta_phase[final_idx], mask))
        object_gamma_coherences.append(circular_coherence(gamma_phase[final_idx], mask))
        object_final_spike_means.append(masked_mean(spikes[final_idx], mask))

    early_idx = 0
    mid_idx = min(spikes.shape[0] - 1, max(0, spikes.shape[0] // 2))
    metrics: Dict[str, float | str] = {
        "run_dir": str(run_dir),
        "num_valid_masks": float(targets.shape[0]),
        "theta_gamma_align_object_t1": masked_mean(alignment[early_idx], union),
        "theta_gamma_align_background_t1": masked_mean(alignment[early_idx], background),
        "theta_gamma_align_object_mid": masked_mean(alignment[mid_idx], union),
        "theta_gamma_align_background_mid": masked_mean(alignment[mid_idx], background),
        "theta_gamma_align_object_final": masked_mean(alignment[final_idx], union),
        "theta_gamma_align_background_final": masked_mean(alignment[final_idx], background),
        "theta_gamma_align_object_drop": masked_mean(alignment[early_idx], union) - masked_mean(alignment[final_idx], union),
        "theta_gamma_align_background_drop": masked_mean(alignment[early_idx], background) - masked_mean(alignment[final_idx], background),
        "theta_phase_object_coherence_final": sum(object_theta_coherences) / max(len(object_theta_coherences), 1),
        "theta_phase_background_coherence_final": circular_coherence(theta_phase[final_idx], background),
        "gamma_phase_object_coherence_final": sum(object_gamma_coherences) / max(len(object_gamma_coherences), 1),
        "gamma_phase_background_coherence_final": circular_coherence(gamma_phase[final_idx], background),
        "theta_magnitude_object_final": masked_mean(theta_magnitude[final_idx], union),
        "theta_magnitude_background_final": masked_mean(theta_magnitude[final_idx], background),
        "gamma_magnitude_object_final": masked_mean(gamma_magnitude[final_idx], union),
        "gamma_magnitude_background_final": masked_mean(gamma_magnitude[final_idx], background),
        "gamma_magnitude_gap_final": masked_mean(gamma_magnitude[final_idx], union) - masked_mean(gamma_magnitude[final_idx], background),
        "spike_object_final": masked_mean(spikes[final_idx], union),
        "spike_background_final": masked_mean(spikes[final_idx], background),
        "spike_object_background_gap_final": masked_mean(spikes[final_idx], union) - masked_mean(spikes[final_idx], background),
        "spike_object_late_mean": masked_mean(spikes[start_idx:].mean(dim=0), union),
        "spike_background_late_mean": masked_mean(spikes[start_idx:].mean(dim=0), background),
        "spike_object_background_gap_late_mean": masked_mean(spikes[start_idx:].mean(dim=0), union) - masked_mean(spikes[start_idx:].mean(dim=0), background),
        "object_final_spike_mean_per_mask": sum(object_final_spike_means) / max(len(object_final_spike_means), 1),
    }
    metrics.update(object_activity_centers(spikes, object_masks, start_idx))

    cfg = ObjectRepresentationConfig(
        image_height=image_height,
        image_width=image_width,
        input_channels=3,
        classifier_start_step=classifier_start_step,
        steps=int(spikes.shape[0]),
    )
    metrics.update(best_iou_over_grid(spikes, object_masks, cfg, score_quantiles, min_pixels, iou_threshold))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize theta/gamma/spike object diagnostics for one run.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--image_height", type=int, default=64)
    parser.add_argument("--image_width", type=int, default=64)
    parser.add_argument("--classifier_start_step", type=int, default=20)
    parser.add_argument("--score_quantiles", type=parse_float_list, default=parse_float_list("0.35,0.40,0.45,0.50,0.55,0.60"))
    parser.add_argument("--min_pixels", type=parse_int_list, default=parse_int_list("32,64,96,128,192"))
    parser.add_argument("--iou_threshold", type=float, default=0.9)
    args = parser.parse_args()

    metrics = summarize_run(
        run_dir=args.run_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        classifier_start_step=args.classifier_start_step,
        score_quantiles=args.score_quantiles,
        min_pixels=args.min_pixels,
        iou_threshold=args.iou_threshold,
    )

    json_path = args.run_dir / "dynamics_diagnostics.json"
    csv_path = args.run_dir / "dynamics_diagnostics.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print("Wrote", json_path)
    print("Wrote", csv_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
