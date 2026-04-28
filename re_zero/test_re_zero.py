from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from pprint import pprint

import torch

from re_zero import ObjectRepresentationConfig, ObjectRepresentationSNN


# Edit these values directly for quick server-side testing.
TEST_CONFIG = {
    "image_height": 8,
    "image_width": 8,
    "input_channels": 3,
    "osc_dim": 1,
    "steps": 4,
    "readout_update_interval": 2,
    "spike_update_offset": 0,
    "classifier_start_step": 1,
    "classifier_type": "mean_spike",
    "object_loss_function": "1234",
    "gamma_initialization": "encoder",
    "global_coupling_strength": 1.0,
    "gamma_attraction_strength": 3.0,
    "feedback_theta_connectivity_weight_scale": 0.25,
    "alpha_scale": 5.0,
    "fixed_alpha_during_training": False,
    "fixed_alpha_value": 0.0,
    "delay": 2,
    "threshold": 0.6,
    "membrane_decay": 0.92,
    "recurrent_scale": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

BATCH_SIZE = 1
SEED = 7
RETURN_HISTORY = True
RETURN_PAIRWISE_HISTORY = False
SAVE_VISUALIZATIONS = True
OUTPUT_DIR = Path("outputs/test_re_zero")
VISUAL_STEPS = [1, 5, 10, 20, 40, 60, 80]


def main() -> None:
    torch.manual_seed(SEED)

    config = ObjectRepresentationConfig(**TEST_CONFIG)
    device = torch.device(config.device)
    model = ObjectRepresentationSNN(config).to(device)
    model.eval()

    x = torch.rand(
        BATCH_SIZE,
        config.image_height,
        config.image_width,
        config.input_channels,
        device=device,
    )

    with torch.no_grad():
        classifier_output, history = model(
            x,
            return_history=RETURN_HISTORY,
            return_pairwise_history=RETURN_PAIRWISE_HISTORY,
        )
        unsupervised_loss = model.unsupervised_object_loss(classifier_output)
        loss_components = model.object_spike_loss_components(classifier_output)

    print("=== ReZero Smoke Test ===")
    print(f"device: {device}")
    print(f"classifier_type: {config.classifier_type}")
    print(f"object_loss_function: {config.object_loss_function}")
    print(f"batch_size: {BATCH_SIZE}")
    print()

    print("=== Config ===")
    pprint(asdict(config))
    print()

    print("=== Classifier Output ===")
    for key, value in classifier_output.items():
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    print()

    print("=== History ===")
    for key, value in history.items():
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    print()

    print("=== Loss ===")
    print(f"unsupervised_object_loss: {float(unsupervised_loss):.6f}")
    for key, value in loss_components.items():
        print(f"{key}: {float(value):.6f}")

    if SAVE_VISUALIZATIONS:
        saved_paths = save_visualizations(
            output_dir=OUTPUT_DIR,
            x=x,
            classifier_output=classifier_output,
            history=history,
            config=config,
        )
        print()
        print("=== Visualizations ===")
        for path in saved_paths:
            print(path)


def to_numpy(tensor: torch.Tensor):
    """Move a tensor to CPU numpy for plotting."""
    return tensor.detach().float().cpu().numpy()


def save_visualizations(
    output_dir: Path,
    x: torch.Tensor,
    classifier_output: dict[str, torch.Tensor],
    history: dict[str, torch.Tensor],
    config: ObjectRepresentationConfig,
) -> list[Path]:
    """Save quick-look PNGs for server-side debugging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping visualization.")
        return []

    saved_paths: list[Path] = []

    image = to_numpy(x[0]).clip(0.0, 1.0)
    mean_spike_grid = to_numpy(classifier_output["mean_spike_grid"])
    masks = to_numpy(classifier_output["masks"])
    mask_overlay = masks.max(axis=0) if masks.size else mean_spike_grid * 0.0

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    plot_image(axes[0, 0], image, "input")
    plot_map(axes[0, 1], mean_spike_grid, "mean spike", cmap="magma")
    plot_map(axes[0, 2], mask_overlay, "mask overlay", cmap="viridis")
    plot_map(
        axes[1, 0],
        to_numpy(history["spikes"][0, -1]).reshape(config.image_height, config.image_width),
        "last spikes",
        cmap="magma",
    )
    plot_map(
        axes[1, 1],
        to_numpy(history["theta"][0, -1]).reshape(config.image_height, config.image_width),
        "last theta",
        cmap="coolwarm",
    )
    plot_map(
        axes[1, 2],
        to_numpy(history["gamma"][0, -1]).reshape(config.image_height, config.image_width),
        "last gamma",
        cmap="coolwarm",
    )

    overview_path = output_dir / "overview.png"
    fig.savefig(overview_path, dpi=160)
    plt.close(fig)
    saved_paths.append(overview_path)

    temporal_path = output_dir / "theta_gamma_spikes_over_time.png"
    save_temporal_maps(
        temporal_path,
        history=history,
        config=config,
        plt=plt,
    )
    saved_paths.append(temporal_path)

    if masks.shape[0] > 0:
        max_masks = min(masks.shape[0], 8)
        fig, axes = plt.subplots(1, max_masks, figsize=(2.2 * max_masks, 2.4), constrained_layout=True)
        if max_masks == 1:
            axes = [axes]
        for idx in range(max_masks):
            plot_map(axes[idx], masks[idx], f"mask {idx}", cmap="gray", vmin=0.0, vmax=1.0)
        masks_path = output_dir / "predicted_masks.png"
        fig.savefig(masks_path, dpi=160)
        plt.close(fig)
        saved_paths.append(masks_path)

    return saved_paths


def save_temporal_maps(
    path: Path,
    history: dict[str, torch.Tensor],
    config: ObjectRepresentationConfig,
    plt,
) -> None:
    """Save theta, gamma, and spike maps at selected timesteps."""
    total_steps = history["theta"].shape[1]
    step_indices = select_visual_step_indices(total_steps)
    num_cols = len(step_indices)
    fig, axes = plt.subplots(3, num_cols, figsize=(2.5 * num_cols, 7.2), constrained_layout=True)
    if num_cols == 1:
        axes = axes.reshape(3, 1)

    rows = [
        ("theta", "coolwarm"),
        ("gamma", "coolwarm"),
        ("spikes", "magma"),
    ]
    for row_idx, (key, cmap) in enumerate(rows):
        for col_idx, step_idx in enumerate(step_indices):
            values = to_numpy(history[key][0, step_idx]).reshape(config.image_height, config.image_width)
            plot_map(axes[row_idx, col_idx], values, f"{key} t={step_idx + 1}", cmap=cmap)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def select_visual_step_indices(total_steps: int) -> list[int]:
    """Convert 1-based visual step requests into valid 0-based indices."""
    requested = [step for step in VISUAL_STEPS if 1 <= step <= total_steps]
    if not requested:
        requested = list(range(1, total_steps + 1))
    requested.append(total_steps)
    return sorted({step - 1 for step in requested})


def plot_image(axis, image, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_map(axis, values, title: str, cmap: str, vmin=None, vmax=None) -> None:
    im = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)


if __name__ == "__main__":
    main()
