#!/usr/bin/env python
"""Temporal OccWorld baseline with an explicit persistence change gate."""

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from tools.data_converter.kl_occworld_dataset import IGNORE_INDEX
from tools.tutorials.occworld_toy.step01_current_to_future_baseline import (
    masked_world_cross_entropy,
)
from tools.tutorials.occworld_toy.step02_temporal_fusion_baseline import (
    TemporalFusionOccWorld,
)


def persistence_class_from_state(
        current_state: torch.Tensor,
        horizon_count: int) -> torch.Tensor:
    """Map raw current state to repeated free/static/instance class IDs."""
    if current_state.ndim != 4:
        raise ValueError('Expected current state [B,Z,H,W]')
    prediction = torch.zeros_like(current_state, dtype=torch.long)
    known = current_state != 0
    prediction[known] = current_state[known] - 1
    return prediction[:, None].expand(
        -1, int(horizon_count), -1, -1, -1)


class GatedTemporalOccWorld(TemporalFusionOccWorld):
    """Predict whether persistence changes, then classify changed voxels."""

    def __init__(self, history_count=5, state_channels=4,
                 hidden_channels=4, horizon_count=4, class_count=3,
                 change_prior=0.05,
                 include_frame_differences=True):
        super().__init__(
            history_count=history_count,
            state_channels=state_channels,
            hidden_channels=hidden_channels,
            horizon_count=horizon_count,
            class_count=class_count,
            persistence_logit_scale=1.0,
            include_frame_differences=include_frame_differences)
        if not 0.0 < change_prior < 1.0:
            raise ValueError('change_prior must be between 0 and 1')
        # The parent residual head is intentionally replaced by two explicit
        # tasks with separate supervision.
        self.residual_head = None
        self.change_head = nn.Conv3d(
            hidden_channels, self.horizon_count, kernel_size=1)
        self.changed_class_head = nn.Conv3d(
            hidden_channels, self.horizon_count * self.class_count,
            kernel_size=1)
        nn.init.zeros_(self.change_head.weight)
        nn.init.constant_(
            self.change_head.bias,
            math.log(change_prior / (1.0 - change_prior)))
        nn.init.zeros_(self.changed_class_head.weight)
        nn.init.zeros_(self.changed_class_head.bias)

    def forward(self, history_one_hot_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        fusion_input = self._fusion_input(history_one_hot_3d)
        features = self.temporal_encoder(fusion_input)
        batch_size, _, z_size, height, width = features.shape
        change_logits = self.change_head(features)
        class_logits = self.changed_class_head(features).reshape(
            batch_size, self.horizon_count, self.class_count,
            z_size, height, width)
        return {
            'change_logits': change_logits,
            'changed_class_logits': class_logits,
        }

    def predict(self, history_one_hot_3d: torch.Tensor,
                threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('threshold must be between 0 and 1')
        outputs = self(history_one_hot_3d)
        change_probability = outputs['change_logits'].sigmoid()
        changed_class = outputs['changed_class_logits'].argmax(dim=2)
        current_state = history_one_hot_3d[:, -1].argmax(dim=1)
        persistence = persistence_class_from_state(
            current_state, self.horizon_count)
        prediction = torch.where(
            change_probability >= threshold,
            changed_class, persistence)
        return {
            **outputs,
            'change_probability': change_probability,
            'persistence_class': persistence,
            'prediction_class': prediction,
        }


def gated_occworld_loss(
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        current_state: torch.Tensor,
        change_positive_weight: float = 10.0,
        change_focal_gamma: float = 0.0,
        changed_class_weights: torch.Tensor = None,
        changed_class_loss_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    """Supervise change detection and class only where change is real."""
    change_logits = outputs['change_logits']
    class_logits = outputs['changed_class_logits']
    if change_positive_weight <= 0 or change_focal_gamma < 0:
        raise ValueError('Invalid change loss parameters')
    if change_logits.shape != target.shape:
        raise ValueError('Change logits and target shapes must match')
    if (class_logits.shape[0:2] != target.shape[0:2] or
            class_logits.shape[3:] != target.shape[2:]):
        raise ValueError('Changed-class logits and target shapes must match')
    persistence = persistence_class_from_state(
        current_state, target.shape[1])
    known = target != IGNORE_INDEX
    changed = known & (target != persistence)
    change_target = changed.to(change_logits.dtype)
    positive_weight = torch.as_tensor(
        change_positive_weight,
        dtype=change_logits.dtype, device=change_logits.device)
    change_per_voxel = F.binary_cross_entropy_with_logits(
        change_logits, change_target,
        pos_weight=positive_weight, reduction='none')
    if change_focal_gamma > 0:
        probability = change_logits.sigmoid()
        target_probability = torch.where(
            changed, probability, 1.0 - probability)
        change_per_voxel = change_per_voxel * (
            1.0 - target_probability).pow(change_focal_gamma)
    if torch.any(known):
        change_loss = change_per_voxel[known].mean()
    else:
        change_loss = change_per_voxel.sum() * 0.0

    changed_target = target.clone()
    changed_target[~changed] = IGNORE_INDEX
    changed_class_loss = masked_world_cross_entropy(
        class_logits, changed_target,
        class_weights=changed_class_weights)
    total = change_loss + (
        float(changed_class_loss_weight) * changed_class_loss)
    return {
        'loss': total,
        'change_loss': change_loss,
        'changed_class_loss': changed_class_loss,
        'known_voxel_count': known.sum(),
        'changed_voxel_count': changed.sum(),
    }
