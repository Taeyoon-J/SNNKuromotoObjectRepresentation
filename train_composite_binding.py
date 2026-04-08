from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from model.hyperparameters import parse_args
from model.snn import analyze_composite_scene, run_composite_binding_analysis, save_analysis_figures


def main() -> None:
    cfg = parse_args()
    exp_dir = cfg.experiment_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    analysis = run_composite_binding_analysis(config=cfg, create_plots=True)
    composite = analyze_composite_scene(analysis["model"], analysis["config"])
    analysis["composite_scene_figure"] = composite["figure"]
    analysis["binding_scores_figure"] = composite["binding_figure"]

    figure_paths = save_analysis_figures(analysis, output_dir=str(exp_dir))

    checkpoint_path = exp_dir / cfg.checkpoint_name
    checkpoint = {
        "model_state_dict": analysis["model"].state_dict(),
        "config": asdict(cfg),
        "train_losses": analysis["train_losses"],
        "binding_losses": analysis["binding_losses"],
        "train_accuracies": analysis["train_accuracies"],
        "test_accuracy": analysis["test_accuracy"],
        "composite_prediction": int(composite["prediction"][0]),
        "object_names": composite["object_names"],
    }
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "experiment_dir": str(exp_dir),
        "checkpoint_path": str(checkpoint_path),
        "test_accuracy": analysis["test_accuracy"],
        "final_train_loss": analysis["train_losses"][-1],
        "final_binding_loss": analysis["binding_losses"][-1],
        "final_train_accuracy": analysis["train_accuracies"][-1],
        "composite_prediction": int(composite["prediction"][0]),
        "object_names": composite["object_names"],
        "figure_paths": figure_paths,
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
