from __future__ import annotations

# This file is a lightweight runner/utility layer around the separated modules.
# Think of it as the easiest place to start if someone wants to:
# 1. train the toy setup,
# 2. test it,
# 3. produce a few analysis plots.

from typing import Dict, List, Optional

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

try:
    from .data import (
        CompositeTemporalBindingDataset,
        PredefinedObjectDataset,
        build_composite_masks,
        build_composite_scene,
    )
    from .hyperparameters import ObjectRepresentationConfig, get_default_config
    from .s2net import ObjectRepresentationSNN
    from .visualization import (
        plot_activation_function,
        plot_loss_curve,
        visualize_object_binding_scores,
        visualize_dynamics,
        visualize_kuramoto_readout,
        visualize_objects,
        visualize_predictions,
        visualize_spike_sequence,
    )
except ImportError:
    from data import (
        CompositeTemporalBindingDataset,
        PredefinedObjectDataset,
        build_composite_masks,
        build_composite_scene,
    )
    from hyperparameters import ObjectRepresentationConfig, get_default_config
    from s2net import ObjectRepresentationSNN
    from visualization import (
        plot_activation_function,
        plot_loss_curve,
        visualize_object_binding_scores,
        visualize_dynamics,
        visualize_kuramoto_readout,
        visualize_objects,
        visualize_predictions,
        visualize_spike_sequence,
    )


def train_on_synthetic_objects(
    config: Optional[ObjectRepresentationConfig] = None,
    dataset: Optional[Dataset] = None,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Train the model on the synthetic predefined-object dataset.

    This is a compact example training loop for quick experimentation.
    """
    # Use defaults unless the caller provides a custom config.
    cfg = config or get_default_config()
    # Fix random seeds so results are reproducible.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build the model and dataset from the same config.
    model = ObjectRepresentationSNN(cfg)
    train_dataset = dataset or PredefinedObjectDataset(
        num_samples=cfg.num_samples,
        image_size=cfg.image_height,
        num_classes=cfg.num_classes,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )
    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Pick execution device and move the model there.
    chosen_device = device or cfg.device
    model = model.to(chosen_device)

    # Adam is created by the model so optimizer defaults stay in one place.
    optimizer = model.build_adam_optimizer()

    losses: List[float] = []
    accuracies: List[float] = []

    for _ in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for images, labels in loader:
            # Move batch data to the same device as the model.
            images = images.to(chosen_device)
            labels = labels.to(chosen_device)

            # Standard training step: zero gradients, forward, loss, backward, update.
            optimizer.zero_grad()
            logits, _ = model(images, return_history=False)
            loss = model.loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            # Keep simple training statistics for plotting and sanity checks.
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.size(0)

        # Save epoch-level averages.
        losses.append(total_loss / max(total_seen, 1))
        accuracies.append(total_correct / max(total_seen, 1))

    return {
        "config": cfg,
        "model": model,
        "dataset": train_dataset,
        "losses": losses,
        "accuracies": accuracies,
        "device": chosen_device,
    }


def compute_temporal_binding_loss(
    history: Dict[str, torch.Tensor],
    object_masks: torch.Tensor,
    time_targets: torch.Tensor,
    config: ObjectRepresentationConfig,
) -> Dict[str, torch.Tensor]:
    """
    Encourage different objects to dominate different time steps.

    Args:
        history: model history dictionary
        object_masks: [B, O, H, W]
        time_targets: [B, T] with target object indices
    """
    spike_trace = history["spikes"]
    batch_size, steps, num_nodes = spike_trace.shape
    num_objects = object_masks.shape[1]
    flat_masks = object_masks.view(batch_size, num_objects, num_nodes)
    mask_mass = flat_masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    # Region means from three signals:
    # 1. spikes, 2. gating values, 3. sin(theta) as a phase-derived oscillator signal.
    gate_trace = history["gate"].mean(dim=-1)
    phase_trace = 0.5 * (1.0 + torch.sin(history["theta"]).mean(dim=-1))

    region_spike = torch.einsum("btn,bon->bto", spike_trace, flat_masks) / mask_mass.transpose(1, 2)
    region_gate = torch.einsum("btn,bon->bto", gate_trace, flat_masks) / mask_mass.transpose(1, 2)
    region_phase = torch.einsum("btn,bon->bto", phase_trace, flat_masks) / mask_mass.transpose(1, 2)

    # Combine the three signals into a single per-object logit.
    combined_logits = (
        config.spike_logit_scale * region_spike
        + config.gate_logit_scale * region_gate
        + config.phase_logit_scale * region_phase
    )

    # Encourage the target object to dominate at each time step.
    binding_loss = torch.nn.functional.cross_entropy(
        combined_logits.reshape(batch_size * steps, num_objects),
        time_targets.reshape(batch_size * steps),
    )

    object_probs = torch.softmax(combined_logits, dim=-1)

    # Encourage explicit anti-phase oscillation templates.
    t = torch.arange(steps, device=spike_trace.device, dtype=torch.float32)
    if num_objects == 2:
        base = 0.5 * (1.0 + torch.cos(2.0 * torch.pi * t / float(config.oscillation_period)))
        target_probs = torch.stack([base, 1.0 - base], dim=-1)
    else:
        phase_offsets = torch.linspace(0.0, 2.0 * torch.pi, steps=num_objects + 1, device=spike_trace.device)[:-1]
        target_probs = []
        for phase in phase_offsets:
            target_probs.append(0.5 * (1.0 + torch.cos(2.0 * torch.pi * t / float(config.oscillation_period) + phase)))
        target_probs = torch.stack(target_probs, dim=-1)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    oscillation_loss = torch.nn.functional.mse_loss(
        object_probs,
        target_probs.unsqueeze(0).expand(batch_size, -1, -1),
    )

    # Discourage multiple objects from being active together at the same time.
    competition_loss = (object_probs[..., 0] * object_probs[..., 1]).mean() if num_objects == 2 else (
        torch.triu(torch.einsum("bto,btp->btop", object_probs, object_probs), diagonal=1).mean()
    )

    # Also discourage spike mass outside all object masks.
    union_mask = flat_masks.amax(dim=1)
    outside_mask = 1.0 - union_mask
    outside_mass = outside_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    outside_scores = (spike_trace * outside_mask.unsqueeze(1)).sum(dim=-1) / outside_mass
    outside_loss = config.outside_loss_weight * outside_scores.mean()

    # Global sparsity keeps the spike field from filling the whole image.
    sparsity_loss = config.sparsity_loss_weight * spike_trace.mean()

    return {
        "binding_loss": binding_loss,
        "oscillation_loss": config.oscillation_loss_weight * oscillation_loss,
        "competition_loss": config.competition_loss_weight * competition_loss,
        "outside_loss": outside_loss,
        "sparsity_loss": sparsity_loss,
        "region_spike": region_spike.detach(),
        "region_gate": region_gate.detach(),
        "region_phase": region_phase.detach(),
        "object_probs": object_probs.detach(),
    }


def train_on_composite_binding(
    config: Optional[ObjectRepresentationConfig] = None,
    dataset: Optional[Dataset] = None,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Train on multi-object composite scenes with an explicit temporal binding loss.
    """
    cfg = config or get_default_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = ObjectRepresentationSNN(cfg)
    train_dataset = dataset or CompositeTemporalBindingDataset(
        num_samples=cfg.composite_num_samples,
        image_size=cfg.image_height,
        patch_size=cfg.composite_patch_size,
        steps=cfg.steps,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )
    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    chosen_device = device or cfg.device
    model = model.to(chosen_device)
    optimizer = model.build_adam_optimizer()

    losses: List[float] = []
    accuracies: List[float] = []
    binding_losses: List[float] = []

    for _ in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_binding = 0.0
        total_correct = 0
        total_seen = 0

        for images, labels, object_masks, time_targets in loader:
            images = images.to(chosen_device)
            labels = labels.to(chosen_device)
            object_masks = object_masks.to(chosen_device)
            time_targets = time_targets.to(chosen_device)

            optimizer.zero_grad()
            logits, history = model(images, return_history=True)

            classification_loss = model.loss_function(logits, labels)
            binding_stats = compute_temporal_binding_loss(
                history,
                object_masks,
                time_targets,
                config=cfg,
            )
            loss = (
                cfg.classification_loss_weight * classification_loss
                + cfg.binding_loss_weight * binding_stats["binding_loss"]
                + binding_stats["oscillation_loss"]
                + binding_stats["competition_loss"]
                + binding_stats["outside_loss"]
                + binding_stats["sparsity_loss"]
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_binding += binding_stats["binding_loss"].item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.size(0)

        losses.append(total_loss / max(total_seen, 1))
        binding_losses.append(total_binding / max(total_seen, 1))
        accuracies.append(total_correct / max(total_seen, 1))

    return {
        "config": cfg,
        "model": model,
        "dataset": train_dataset,
        "losses": losses,
        "binding_losses": binding_losses,
        "accuracies": accuracies,
        "device": chosen_device,
    }


@torch.no_grad()
def run_testing_function(
    model: ObjectRepresentationSNN,
    dataset: Dataset,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run evaluation on a dataset and return predictions plus one captured history.

    The first batch history is stored as an example so we can visualize
    internal dynamics without holding every intermediate tensor in memory.
    """
    chosen_device = device or next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Switch to evaluation mode to disable training-specific behavior.
    model.eval()
    preds = []
    labels_all = []
    example_history = None
    image_batches = []

    for batch in loader:
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1]
        images = images.to(chosen_device)
        labels = labels.to(chosen_device)
        logits, history = model(images, return_history=True)

        # Save predicted class ids and ground-truth labels.
        preds.append(logits.argmax(dim=1).cpu())
        labels_all.append(labels.cpu())
        image_batches.append(images.cpu())

        # Keep one history dictionary for later plots.
        if example_history is None:
            example_history = {key: value.cpu() for key, value in history.items()}

    pred_tensor = torch.cat(preds)
    label_tensor = torch.cat(labels_all)
    # Simple classification accuracy over the full dataset.
    accuracy = (pred_tensor == label_tensor).float().mean().item()

    return {
        "accuracy": accuracy,
        "images": torch.cat(image_batches),
        "predictions": pred_tensor,
        "labels": label_tensor,
        "example_history": example_history,
    }


def run_synthetic_analysis(
    config: Optional[ObjectRepresentationConfig] = None,
    test_ratio: float = 0.2,
    create_plots: bool = True,
) -> Dict[str, object]:
    """
    Convenience function that trains, tests, and optionally creates plots.

    This is helpful for demos and quick analysis runs.
    """
    cfg = config or get_default_config()

    full_dataset = PredefinedObjectDataset(
        num_samples=cfg.num_samples,
        image_size=cfg.image_height,
        num_classes=cfg.num_classes,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )

    test_size = max(1, int(round(len(full_dataset) * test_ratio)))
    train_size = len(full_dataset) - test_size
    if train_size < 1:
        raise ValueError("test_ratio is too large; training split would be empty.")

    split_generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=split_generator,
    )

    training = train_on_synthetic_objects(config=cfg, dataset=train_dataset)
    testing = run_testing_function(
        training["model"],
        test_dataset,
        batch_size=training["config"].batch_size,
        device=training["device"],
    )

    train_indices = train_dataset.indices if isinstance(train_dataset, Subset) else range(len(train_dataset))
    test_indices = test_dataset.indices if isinstance(test_dataset, Subset) else range(len(test_dataset))
    train_images = full_dataset.images[list(train_indices)]
    train_labels = full_dataset.labels[list(train_indices)]

    result = {
        "config": training["config"],
        "model": training["model"],
        "train_losses": training["losses"],
        "train_accuracies": training["accuracies"],
        "test_accuracy": testing["accuracy"],
        "predictions": testing["predictions"],
        "labels": testing["labels"],
        "example_history": testing["example_history"],
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }

    if create_plots:
        # Plot the dataset itself, the activation shape, the training curve,
        # and a compact summary of internal dynamics.
        result["object_figure"] = visualize_objects(train_images, train_labels)
        result["prediction_figure"] = visualize_predictions(
            testing["images"],
            testing["labels"],
            testing["predictions"],
        )
        result["activation_figure"] = plot_activation_function(training["model"].top_down.activation_function)
        result["loss_figure"] = plot_loss_curve(training["losses"])
        result["dynamics_figure"] = visualize_dynamics(testing["example_history"])

    return result


def run_composite_binding_analysis(
    config: Optional[ObjectRepresentationConfig] = None,
    test_ratio: float = 0.2,
    create_plots: bool = True,
) -> Dict[str, object]:
    """
    Train and evaluate the model on composite multi-object scenes.
    """
    cfg = config or get_default_config()

    full_dataset = CompositeTemporalBindingDataset(
        num_samples=cfg.composite_num_samples,
        image_size=cfg.image_height,
        patch_size=cfg.composite_patch_size,
        steps=cfg.steps,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )

    test_size = max(1, int(round(len(full_dataset) * test_ratio)))
    train_size = len(full_dataset) - test_size
    if train_size < 1:
        raise ValueError("test_ratio is too large; training split would be empty.")

    split_generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=split_generator,
    )

    training = train_on_composite_binding(config=cfg, dataset=train_dataset)
    testing = run_testing_function(
        training["model"],
        test_dataset,
        batch_size=training["config"].batch_size,
        device=training["device"],
    )

    result = {
        "config": training["config"],
        "model": training["model"],
        "train_losses": training["losses"],
        "binding_losses": training["binding_losses"],
        "train_accuracies": training["accuracies"],
        "test_accuracy": testing["accuracy"],
        "predictions": testing["predictions"],
        "labels": testing["labels"],
        "example_history": testing["example_history"],
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }

    if create_plots:
        result["prediction_figure"] = visualize_predictions(
            testing["images"],
            testing["labels"],
            testing["predictions"],
        )
        result["activation_figure"] = plot_activation_function(training["model"].top_down.activation_function)
        result["loss_figure"] = plot_loss_curve(training["losses"])
        result["dynamics_figure"] = visualize_dynamics(testing["example_history"])

    return result


@torch.no_grad()
def analyze_composite_scene(
    model: ObjectRepresentationSNN,
    config: ObjectRepresentationConfig,
    object_specs=((0, 1, 1), (3, 8, 8)),
) -> Dict[str, object]:
    """
    Run one multi-object scene through the trained model and collect time-resolved
    spike maps so we can inspect whether object-shaped activity appears.
    """
    scene = build_composite_scene(
        image_size=config.image_height,
        object_specs=object_specs,
        patch_size=6,
    )
    object_masks, object_names = build_composite_masks(
        image_size=config.image_height,
        object_specs=object_specs,
        patch_size=6,
    )

    device = next(model.parameters()).device
    model.eval()
    logits, history = model(scene.unsqueeze(0).to(device), return_history=True)

    pred_label = logits.argmax(dim=1).cpu()
    spike_history = history["spikes"][0].cpu()
    spike_maps = spike_history.view(config.steps, config.image_height, config.image_width)

    region_activations = []
    overlap_scores = []
    flat_masks = object_masks.view(object_masks.shape[0], -1)
    flat_spikes = spike_history.view(config.steps, -1)
    spike_mass = flat_spikes.sum(dim=1).clamp_min(1e-6)

    for mask in flat_masks:
        mask_sum = mask.sum().clamp_min(1e-6)
        in_region = flat_spikes * mask.unsqueeze(0)
        region_activation = in_region.sum(dim=1) / mask_sum
        overlap_score = in_region.sum(dim=1) / spike_mass
        region_activations.append(region_activation)
        overlap_scores.append(overlap_score)

    region_activations = torch.stack(region_activations, dim=0)
    overlap_scores = torch.stack(overlap_scores, dim=0)

    return {
        "scene": scene,
        "object_masks": object_masks,
        "object_names": object_names,
        "prediction": pred_label,
        "spike_history": spike_history,
        "spike_maps": spike_maps,
        "region_activations": region_activations,
        "overlap_scores": overlap_scores,
        "history": {key: value.cpu() for key, value in history.items()},
        "figure": visualize_spike_sequence(
            scene,
            spike_history,
            image_height=config.image_height,
            image_width=config.image_width,
            max_steps=config.steps,
        ),
        "binding_figure": visualize_object_binding_scores(
            region_activations,
            overlap_scores,
            object_names,
        ),
        "readout_figure": visualize_kuramoto_readout(
            scene,
            history["theta"][0].cpu(),
            history["gamma"][0].cpu(),
            history["gate"][0].cpu(),
            image_height=config.image_height,
            image_width=config.image_width,
            max_steps=min(4, config.steps),
        ),
    }


def save_analysis_figures(result: Dict[str, object], output_dir: str = "outputs") -> Dict[str, str]:
    """
    Save generated analysis figures to disk and return their file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, str] = {}
    figure_map = {
        "object_figure": "objects.png",
        "prediction_figure": "predictions.png",
        "activation_figure": "activation.png",
        "loss_figure": "loss.png",
        "dynamics_figure": "dynamics.png",
        "composite_scene_figure": "composite_spike_sequence.png",
        "binding_scores_figure": "binding_scores.png",
        "readout_figure": "kuramoto_readout.png",
    }

    for key, filename in figure_map.items():
        figure = result.get(key)
        if figure is not None:
            path = out_dir / filename
            figure.savefig(path, dpi=200, bbox_inches="tight")
            saved_paths[key] = str(path)

    return saved_paths


if __name__ == "__main__":
    # Running this file directly triggers an end-to-end composite binding run.
    analysis = run_composite_binding_analysis(create_plots=True)
    composite = analyze_composite_scene(
        analysis["model"],
        analysis["config"],
    )
    analysis["composite_scene_figure"] = composite["figure"]
    analysis["binding_scores_figure"] = composite["binding_figure"]
    analysis["readout_figure"] = composite["readout_figure"]
    saved = save_analysis_figures(analysis)
    print(
        {
            "final_train_loss": round(analysis["train_losses"][-1], 4),
            "final_train_accuracy": round(analysis["train_accuracies"][-1], 4),
            "test_accuracy": round(analysis["test_accuracy"], 4),
            "composite_prediction": int(composite["prediction"][0]),
            "object_names": composite["object_names"],
            "saved_figures": saved,
        }
    )
