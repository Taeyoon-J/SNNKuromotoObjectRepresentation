from __future__ import annotations

# This file is a lightweight runner/utility layer around the separated modules.
# Think of it as the easiest place to start if someone wants to:
# 1. train the toy setup,
# 2. test it,
# 3. produce a few analysis plots.

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .data import PredefinedObjectDataset
    from .hyperparameters import ObjectRepresentationConfig, get_default_config
    from .s2net import ObjectRepresentationSNN
    from .visualization import (
        plot_activation_function,
        plot_loss_curve,
        visualize_dynamics,
        visualize_objects,
    )
except ImportError:
    from data import PredefinedObjectDataset
    from hyperparameters import ObjectRepresentationConfig, get_default_config
    from s2net import ObjectRepresentationSNN
    from visualization import (
        plot_activation_function,
        plot_loss_curve,
        visualize_dynamics,
        visualize_objects,
    )


def train_on_synthetic_objects(
    config: Optional[ObjectRepresentationConfig] = None,
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
    dataset = PredefinedObjectDataset(
        num_samples=cfg.num_samples,
        image_size=cfg.image_height,
        num_classes=cfg.num_classes,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

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
        "dataset": dataset,
        "losses": losses,
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

    The first batch history is stored so we can visualize internal dynamics.
    """
    chosen_device = device or next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Switch to evaluation mode to disable training-specific behavior.
    model.eval()
    preds = []
    labels_all = []
    histories = None

    for images, labels in loader:
        images = images.to(chosen_device)
        labels = labels.to(chosen_device)
        logits, history = model(images, return_history=True)

        # Save predicted class ids and ground-truth labels.
        preds.append(logits.argmax(dim=1).cpu())
        labels_all.append(labels.cpu())

        # Keep one history dictionary for later plots.
        if histories is None:
            histories = {key: value.cpu() for key, value in history.items()}

    pred_tensor = torch.cat(preds)
    label_tensor = torch.cat(labels_all)
    # Simple classification accuracy over the full dataset.
    accuracy = (pred_tensor == label_tensor).float().mean().item()

    return {
        "accuracy": accuracy,
        "predictions": pred_tensor,
        "labels": label_tensor,
        "history": histories,
    }


def run_synthetic_analysis(
    config: Optional[ObjectRepresentationConfig] = None,
    create_plots: bool = True,
) -> Dict[str, object]:
    """
    Convenience function that trains, tests, and optionally creates plots.

    This is helpful for demos and quick analysis runs.
    """
    training = train_on_synthetic_objects(config=config)
    testing = run_testing_function(
        training["model"],
        training["dataset"],
        batch_size=training["config"].batch_size,
        device=training["device"],
    )

    result = {
        "train_losses": training["losses"],
        "train_accuracies": training["accuracies"],
        "test_accuracy": testing["accuracy"],
        "predictions": testing["predictions"],
        "labels": testing["labels"],
        "history": testing["history"],
    }

    if create_plots:
        # Plot the dataset itself, the activation shape, the training curve,
        # and a compact summary of internal dynamics.
        result["object_figure"] = visualize_objects(training["dataset"].images, training["dataset"].labels)
        result["activation_figure"] = plot_activation_function(training["model"].top_down.activation_function)
        result["loss_figure"] = plot_loss_curve(training["losses"])
        result["dynamics_figure"] = visualize_dynamics(testing["history"])

    return result


if __name__ == "__main__":
    # Running this file directly triggers a tiny end-to-end smoke test.
    analysis = run_synthetic_analysis(create_plots=False)
    print(
        {
            "final_train_loss": round(analysis["train_losses"][-1], 4),
            "final_train_accuracy": round(analysis["train_accuracies"][-1], 4),
            "test_accuracy": round(analysis["test_accuracy"], 4),
        }
    )
