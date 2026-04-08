from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from model.data import CompositePhaseBindingDataset
from model.hyperparameters import parse_args
from model.phase_binding import GraphPhaseKuramoto, phase_binding_metrics, phase_contrastive_loss


def main() -> None:
    cfg = parse_args()
    exp_dir = cfg.experiment_dir.parent / f"{cfg.experiment_name}_phase"
    exp_dir.mkdir(parents=True, exist_ok=True)

    dataset = CompositePhaseBindingDataset(
        num_samples=cfg.composite_num_samples,
        image_size=cfg.image_height,
        patch_size=cfg.composite_patch_size,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )
    test_size = max(1, int(round(0.2 * len(dataset))))
    train_size = len(dataset) - test_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = GraphPhaseKuramoto(
        image_size=cfg.image_height,
        osc_dim=1,
        coupling_k=cfg.coupling,
        dt=cfg.dt,
        lag_scale=cfg.alpha_scale,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses = []
    last_metrics = None
    for _ in range(cfg.epochs):
        model.train()
        epoch_losses = []
        for images, masks in train_loader:
            images = images.to(cfg.device)
            masks = masks.to(cfg.device)
            theta, _ = model(images, steps=cfg.steps, return_history=False)
            loss = phase_contrastive_loss(theta, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        train_losses.append(sum(epoch_losses) / max(len(epoch_losses), 1))

        # Quick metric snapshot from the last train batch.
        with torch.no_grad():
            last_metrics = phase_binding_metrics(theta, masks)

    model.eval()
    test_losses = []
    test_metric_values = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(cfg.device)
            masks = masks.to(cfg.device)
            theta, _ = model(images, steps=cfg.steps, return_history=False)
            test_losses.append(float(phase_contrastive_loss(theta, masks).item()))
            test_metric_values.append(phase_binding_metrics(theta, masks))

    mean_test_loss = sum(test_losses) / max(len(test_losses), 1)
    mean_intra = sum(metric.intra_sync for metric in test_metric_values) / max(len(test_metric_values), 1)
    mean_inter = sum(metric.inter_sync for metric in test_metric_values) / max(len(test_metric_values), 1)

    checkpoint_path = exp_dir / "phase_binding.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "train_losses": train_losses,
            "test_loss": mean_test_loss,
            "mean_intra_sync": mean_intra,
            "mean_inter_sync": mean_inter,
        },
        checkpoint_path,
    )

    summary = {
        "experiment_dir": str(exp_dir),
        "checkpoint_path": str(checkpoint_path),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "last_train_intra_sync": None if last_metrics is None else last_metrics.intra_sync,
        "last_train_inter_sync": None if last_metrics is None else last_metrics.inter_sync,
        "test_loss": mean_test_loss,
        "mean_test_intra_sync": mean_intra,
        "mean_test_inter_sync": mean_inter,
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
