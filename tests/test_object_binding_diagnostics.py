from __future__ import annotations

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.hyperparameters import ObjectRepresentationConfig
from model.s2net import ObjectRepresentationSNN
from train_clevr_sweep import score_one_sample, split_binary_mask_components


def test_split_binary_mask_components_separates_disconnected_objects() -> None:
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True

    components = split_binary_mask_components(mask)

    assert len(components) == 2
    assert sorted(int(component.sum().item()) for component in components) == [4, 4]


def test_scoring_can_match_two_objects_from_one_timestep_after_component_split() -> None:
    object_masks = torch.zeros(2, 8, 8)
    object_masks[0, 1:3, 1:3] = 1.0
    object_masks[1, 5:7, 5:7] = 1.0

    spike_trace = torch.zeros(4, 8 * 8)
    spike_map = torch.zeros(8, 8)
    spike_map[1:3, 1:3] = 1.0
    spike_map[5:7, 5:7] = 1.0
    spike_trace[-1] = spike_map.flatten()

    cfg = ObjectRepresentationConfig(
        image_height=8,
        image_width=8,
        input_channels=3,
        steps=4,
        classifier_start_step=4,
    )

    scores = score_one_sample(
        spike_trace=spike_trace,
        object_masks=object_masks,
        cfg=cfg,
        iou_threshold=0.9,
        score_quantile=0.9,
        min_pixels=1,
    )

    assert scores["score_90_coverage"] == 1.0
    assert scores["score_mean_one_to_one_iou"] == 1.0


def test_background_spikes_increase_unsupervised_object_loss() -> None:
    cfg = ObjectRepresentationConfig(
        image_height=8,
        image_width=8,
        input_channels=3,
        steps=6,
        classifier_start_step=1,
        background_suppression_weight=1.0,
    )
    model = ObjectRepresentationSNN(cfg)

    object_masks = torch.zeros(1, 1, 8, 8)
    object_masks[0, 0, 2:6, 2:6] = 1.0

    clean_spikes = torch.zeros(1, 6, 8 * 8)
    noisy_background_spikes = clean_spikes.clone()
    background = (object_masks[0, 0].flatten() == 0)
    noisy_background_spikes[:, :, background] = 0.5

    clean_loss = model.unsupervised_object_loss_1234(clean_spikes, object_masks)
    noisy_loss = model.unsupervised_object_loss_1234(noisy_background_spikes, object_masks)

    assert noisy_loss > clean_loss


if __name__ == "__main__":
    test_split_binary_mask_components_separates_disconnected_objects()
    test_scoring_can_match_two_objects_from_one_timestep_after_component_split()
    test_background_spikes_increase_unsupervised_object_loss()
    print("object binding diagnostics passed")
