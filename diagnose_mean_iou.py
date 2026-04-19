from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.hyperparameters import ObjectRepresentationConfig
from train_clevr_sweep import pairwise_iou, score_one_sample, spike_trace_to_candidate_masks


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_history(path: Path) -> Dict[str, torch.Tensor]:
    data = np.load(path)
    required = ["spikes", "object_masks", "label_map"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")
    return {key: torch.from_numpy(data[key]) for key in data.files}


def flatten_spike_trace(spikes: torch.Tensor) -> torch.Tensor:
    """Accept [T, N] or saved image-form [T, H, W] spikes and return [T, N]."""
    if spikes.dim() == 2:
        return spikes.float()
    if spikes.dim() == 3:
        return spikes.reshape(spikes.shape[0], -1).float()
    raise ValueError(f"Expected spikes with shape [T, N] or [T, H, W], got {tuple(spikes.shape)}")


def valid_object_masks(object_masks: torch.Tensor) -> torch.Tensor:
    valid = object_masks.reshape(object_masks.shape[0], -1).sum(dim=1) > 0
    return object_masks[valid].float()


def object_background_stats(spike_map: torch.Tensor, object_masks: torch.Tensor) -> Dict[str, float]:
    targets = valid_object_masks(object_masks)
    if targets.numel() == 0:
        return {
            "object_spike_mean": 0.0,
            "background_spike_mean": float(spike_map.mean().item()),
            "object_background_gap": 0.0,
        }
    union = targets.amax(dim=0).bool()
    background = ~union
    object_mean = float(spike_map[union].mean().item()) if union.any() else 0.0
    background_mean = float(spike_map[background].mean().item()) if background.any() else 0.0
    return {
        "object_spike_mean": object_mean,
        "background_spike_mean": background_mean,
        "object_background_gap": object_mean - background_mean,
    }


def candidate_diagnostics(
    spike_trace: torch.Tensor,
    object_masks: torch.Tensor,
    cfg: ObjectRepresentationConfig,
    score_quantile: float,
    min_pixels: int,
    iou_threshold: float,
) -> Dict[str, object]:
    targets = valid_object_masks(object_masks)
    candidates = spike_trace_to_candidate_masks(
        spike_trace=spike_trace,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        input_channels=cfg.input_channels,
        start_step=cfg.classifier_start_step,
        score_quantile=score_quantile,
        min_pixels=min_pixels,
    )
    iou = pairwise_iou(candidates, targets)
    scores = score_one_sample(
        spike_trace=spike_trace,
        object_masks=object_masks,
        cfg=cfg,
        iou_threshold=iou_threshold,
        score_quantile=score_quantile,
        min_pixels=min_pixels,
    )

    best_iou_per_target = []
    best_candidate_area_per_target = []
    best_candidate_idx_per_target = []
    if targets.numel() > 0 and candidates.numel() > 0:
        best_values, best_indices = iou.max(dim=0)
        best_iou_per_target = [float(value.item()) for value in best_values]
        for pred_idx in best_indices.tolist():
            best_candidate_idx_per_target.append(int(pred_idx))
            best_candidate_area_per_target.append(float(candidates[pred_idx].sum().item()))
    elif targets.numel() > 0:
        best_iou_per_target = [0.0 for _ in range(targets.shape[0])]
        best_candidate_area_per_target = [0.0 for _ in range(targets.shape[0])]
        best_candidate_idx_per_target = [-1 for _ in range(targets.shape[0])]

    candidate_areas = [float(candidate.sum().item()) for candidate in candidates]
    target_areas = [float(target.sum().item()) for target in targets]

    return {
        **scores,
        "score_quantile": score_quantile,
        "min_pixels": min_pixels,
        "iou_threshold": iou_threshold,
        "target_areas": target_areas,
        "candidate_areas": candidate_areas,
        "best_iou_per_target": best_iou_per_target,
        "best_candidate_area_per_target": best_candidate_area_per_target,
        "best_candidate_idx_per_target": best_candidate_idx_per_target,
        "candidates": candidates,
        "targets": targets,
        "iou": iou,
    }


def flatten_row(row: Dict[str, object]) -> Dict[str, object]:
    flat = {}
    for key, value in row.items():
        if isinstance(value, torch.Tensor):
            continue
        if key in {"candidates", "targets", "iou"}:
            continue
        if isinstance(value, list):
            flat[key] = json.dumps(value)
        else:
            flat[key] = value
    return flat


def save_overlay_figure(
    output_path: Path,
    input_image: torch.Tensor | None,
    spike_trace: torch.Tensor,
    object_masks: torch.Tensor,
    diagnostic: Dict[str, object],
    cfg: ObjectRepresentationConfig,
) -> None:
    candidates = diagnostic["candidates"]
    targets = diagnostic["targets"]
    iou = diagnostic["iou"]
    start_idx = min(max(cfg.classifier_start_step - 1, 0), spike_trace.shape[0] - 1)
    late_spikes = spike_trace[start_idx:].view(-1, cfg.image_height, cfg.image_width)
    final_spike = spike_trace[-1].view(cfg.image_height, cfg.image_width)
    mean_late_spike = late_spikes.mean(dim=0)

    show_candidates = min(int(candidates.shape[0]), 8)
    cols = max(4, show_candidates)
    fig, axes = plt.subplots(4, cols, figsize=(2.5 * cols, 9.0))
    axes = np.atleast_2d(axes)

    def contour_targets(ax) -> None:
        for target in targets[:10]:
            ax.contour(target.numpy(), levels=[0.5], colors="white", linewidths=0.7)

    if input_image is not None:
        axes[0, 0].imshow(input_image.numpy())
        axes[0, 0].set_title("input")
    else:
        axes[0, 0].imshow(targets.amax(dim=0).numpy() if targets.numel() else np.zeros((cfg.image_height, cfg.image_width)), cmap="gray")
        axes[0, 0].set_title("GT union")
    contour_targets(axes[0, 0])

    axes[0, 1].imshow(final_spike.numpy(), cmap="magma")
    axes[0, 1].set_title("final spike")
    contour_targets(axes[0, 1])

    axes[0, 2].imshow(mean_late_spike.numpy(), cmap="magma")
    axes[0, 2].set_title("late mean spike")
    contour_targets(axes[0, 2])

    axes[0, 3].imshow(targets.amax(dim=0).numpy() if targets.numel() else np.zeros((cfg.image_height, cfg.image_width)), cmap="tab20")
    axes[0, 3].set_title("GT union")

    for col in range(4, cols):
        axes[0, col].axis("off")

    for col in range(cols):
        for row in range(1, 4):
            axes[row, col].axis("off")

    for idx in range(show_candidates):
        row = 1 + idx // cols
        col = idx % cols
        if row >= 4:
            break
        candidate = candidates[idx]
        axes[row, col].imshow(candidate.numpy(), cmap="gray", vmin=0, vmax=1)
        best_iou = float(iou[idx].max().item()) if iou.numel() else 0.0
        best_target = int(iou[idx].argmax().item()) if iou.numel() else -1
        axes[row, col].set_title(f"cand {idx}\narea={int(candidate.sum())}\nbest IoU={best_iou:.2f} gt={best_target}")
        contour_targets(axes[row, col])
        axes[row, col].axis("off")

    fig.suptitle(
        f"q={diagnostic['score_quantile']} min_pixels={diagnostic['min_pixels']} "
        f"meanIoU={diagnostic['score_mean_one_to_one_iou']:.3f} "
        f"coverage50/70/90={diagnostic['score_50_coverage']:.2f}/"
        f"{diagnostic['score_70_coverage']:.2f}/{diagnostic['score_90_coverage']:.2f}",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why spike-derived mean IoU is low.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory containing history_maps.npz")
    parser.add_argument("--image_path", type=Path, default=None, help="Optional .npy input image file for overlay")
    parser.add_argument("--image_height", type=int, default=32)
    parser.add_argument("--image_width", type=int, default=32)
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--classifier_start_step", type=int, default=20)
    parser.add_argument("--score_quantiles", type=parse_float_list, default=parse_float_list("0.85,0.88,0.9,0.92,0.95"))
    parser.add_argument("--min_pixels", type=parse_int_list, default=parse_int_list("4,8,12,16"))
    parser.add_argument("--iou_threshold", type=float, default=0.9)
    args = parser.parse_args()

    history_path = args.run_dir / "history_maps.npz"
    history = load_history(history_path)
    spike_trace = flatten_spike_trace(history["spikes"])
    object_masks = history["object_masks"].float()
    input_image = None
    if args.image_path is not None:
        input_image = torch.from_numpy(np.load(args.image_path)).float()

    cfg = ObjectRepresentationConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        input_channels=args.input_channels,
        classifier_start_step=args.classifier_start_step,
        steps=int(spike_trace.shape[0]),
    )

    final_spike = spike_trace[-1].view(args.image_height, args.image_width)
    rows = []
    diagnostics = []
    for score_quantile in args.score_quantiles:
        for min_pixels in args.min_pixels:
            diagnostic = candidate_diagnostics(
                spike_trace=spike_trace,
                object_masks=object_masks,
                cfg=cfg,
                score_quantile=score_quantile,
                min_pixels=min_pixels,
                iou_threshold=args.iou_threshold,
            )
            diagnostic.update(object_background_stats(final_spike, object_masks))
            diagnostics.append(diagnostic)
            rows.append(flatten_row(diagnostic))

    rows.sort(key=lambda row: (float(row["score_mean_one_to_one_iou"]), float(row["score_70_coverage"])), reverse=True)
    output_csv = args.run_dir / "iou_diagnostics.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(diagnostics, key=lambda row: (float(row["score_mean_one_to_one_iou"]), float(row["score_70_coverage"])))
    best_json = flatten_row(best)
    with (args.run_dir / "iou_diagnostics_best.json").open("w", encoding="utf-8") as f:
        json.dump(best_json, f, indent=2)

    save_overlay_figure(
        output_path=args.run_dir / "iou_diagnostics_best.png",
        input_image=input_image,
        spike_trace=spike_trace,
        object_masks=object_masks,
        diagnostic=best,
        cfg=cfg,
    )

    print(f"Wrote {output_csv}")
    print(f"Wrote {args.run_dir / 'iou_diagnostics_best.json'}")
    print(f"Wrote {args.run_dir / 'iou_diagnostics_best.png'}")
    print("Top diagnostics:")
    for row in rows[:8]:
        print(
            f"q={row['score_quantile']} min={row['min_pixels']} "
            f"meanIoU={float(row['score_mean_one_to_one_iou']):.3f} "
            f"cov50/70/90={float(row['score_50_coverage']):.2f}/"
            f"{float(row['score_70_coverage']):.2f}/{float(row['score_90_coverage']):.2f} "
            f"candidates={float(row['num_candidates']):.0f} "
            f"gap={float(row['object_background_gap']):.3f}"
        )


if __name__ == "__main__":
    main()
