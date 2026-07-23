#!/usr/bin/env python
"""Separate future reveal/completion from visible-state transitions."""

import math
from typing import Dict, Optional

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
from tools.tutorials.occworld_toy.step03_change_gate_baseline import (
    persistence_class_from_state,
)


def _prior_bias(prior: float) -> float:
    if not 0.0 < prior < 1.0:
        raise ValueError('Gate prior must be between 0 and 1')
    return math.log(prior / (1.0 - prior))


class DecomposedTemporalOccWorld(TemporalFusionOccWorld):
    """Use separate gates and class heads for unknown and visible voxels."""

    def __init__(self, history_count=5, state_channels=4,
                 hidden_channels=4, horizon_count=4, class_count=3,
                 reveal_prior=0.05, transition_prior=0.01,
                 include_frame_differences=True):
        super().__init__(
            history_count=history_count,
            state_channels=state_channels,
            hidden_channels=hidden_channels,
            horizon_count=horizon_count,
            class_count=class_count,
            persistence_logit_scale=1.0,
            include_frame_differences=include_frame_differences)
        self.residual_head = None
        self.reveal_head = nn.Conv3d(
            hidden_channels, self.horizon_count, kernel_size=1)
        self.reveal_class_head = nn.Conv3d(
            hidden_channels, self.horizon_count * self.class_count,
            kernel_size=1)
        self.transition_head = nn.Conv3d(
            hidden_channels, self.horizon_count, kernel_size=1)
        self.transition_class_head = nn.Conv3d(
            hidden_channels, self.horizon_count * self.class_count,
            kernel_size=1)
        for head in (self.reveal_head, self.reveal_class_head,
                     self.transition_head, self.transition_class_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.reveal_head.bias, _prior_bias(reveal_prior))
        nn.init.constant_(
            self.transition_head.bias, _prior_bias(transition_prior))

    def _class_logits(self, head: nn.Conv3d,
                      features: torch.Tensor) -> torch.Tensor:
        batch_size, _, z_size, height, width = features.shape
        return head(features).reshape(
            batch_size, self.horizon_count, self.class_count,
            z_size, height, width)

    def forward(self, history_one_hot_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.temporal_encoder(
            self._fusion_input(history_one_hot_3d))
        return {
            'reveal_logits': self.reveal_head(features),
            'reveal_class_logits': self._class_logits(
                self.reveal_class_head, features),
            'transition_logits': self.transition_head(features),
            'transition_class_logits': self._class_logits(
                self.transition_class_head, features),
        }

    def predict(self, history_one_hot_3d: torch.Tensor,
                reveal_threshold: float = 0.5,
                transition_threshold: float = 0.5,
                decision_mode: str = 'hard') -> Dict[str, torch.Tensor]:
        for threshold in (reveal_threshold, transition_threshold):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError('Thresholds must be between 0 and 1')
        outputs = self(history_one_hot_3d)
        reveal_probability = outputs['reveal_logits'].sigmoid()
        transition_probability = outputs['transition_logits'].sigmoid()
        reveal_class = outputs['reveal_class_logits'].argmax(dim=2)
        transition_class = outputs['transition_class_logits'].argmax(dim=2)
        current_state = history_one_hot_3d[:, -1].argmax(dim=1)
        persistence = persistence_class_from_state(
            current_state, self.horizon_count)
        current_unknown = (current_state == 0)[:, None].expand_as(persistence)
        reveal_open = reveal_probability >= reveal_threshold
        transition_open = transition_probability >= transition_threshold
        if decision_mode == 'hard':
            prediction = torch.where(
                current_unknown & reveal_open, reveal_class, persistence)
            prediction = torch.where(
                ~current_unknown & transition_open,
                transition_class, prediction)
            prediction_scores = None
        elif decision_mode in ('probabilistic', 'reveal_only'):
            persistence_scores = F.one_hot(
                persistence, num_classes=self.class_count
            ).permute(0, 1, 5, 2, 3, 4).to(reveal_probability.dtype)
            reveal_scores = (
                (1.0 - reveal_probability[:, :, None]) *
                persistence_scores +
                reveal_probability[:, :, None] *
                outputs['reveal_class_logits'].softmax(dim=2))
            transition_scores = (
                (1.0 - transition_probability[:, :, None]) *
                persistence_scores +
                transition_probability[:, :, None] *
                outputs['transition_class_logits'].softmax(dim=2))
            visible_scores = (
                persistence_scores
                if decision_mode == 'reveal_only'
                else transition_scores)
            prediction_scores = torch.where(
                current_unknown[:, :, None], reveal_scores, visible_scores)
            prediction = prediction_scores.argmax(dim=2)
        else:
            raise ValueError(
                "decision_mode must be 'hard', 'probabilistic' or "
                "'reveal_only'")
        return {
            **outputs,
            'reveal_probability': reveal_probability,
            'transition_probability': transition_probability,
            'persistence_class': persistence,
            'prediction_class': prediction,
            'prediction_scores': prediction_scores,
        }


def _masked_binary_loss(logits: torch.Tensor,
                        target: torch.Tensor,
                        mask: torch.Tensor,
                        positive_weight: float) -> torch.Tensor:
    if positive_weight <= 0:
        raise ValueError('Positive weight must be greater than zero')
    per_voxel = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype),
        pos_weight=torch.as_tensor(
            positive_weight, dtype=logits.dtype, device=logits.device),
        reduction='none')
    if not torch.any(mask):
        return per_voxel.sum() * 0.0
    return per_voxel[mask].mean()


def decomposed_occworld_loss(
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        current_state: torch.Tensor,
        reveal_positive_weight: float = 1.0,
        transition_positive_weight: float = 1.0,
        reveal_class_weights: Optional[torch.Tensor] = None,
        transition_class_weights: Optional[torch.Tensor] = None,
        reveal_gate_loss_weight: float = 1.0,
        reveal_class_loss_weight: float = 1.0,
        transition_gate_loss_weight: float = 1.0,
        transition_class_loss_weight: float = 1.0,
        ) -> Dict[str, torch.Tensor]:
    """Apply each loss only on the subset matching its semantics."""
    reveal_logits = outputs['reveal_logits']
    transition_logits = outputs['transition_logits']
    reveal_class_logits = outputs['reveal_class_logits']
    transition_class_logits = outputs['transition_class_logits']
    if reveal_logits.shape != target.shape or transition_logits.shape != target.shape:
        raise ValueError('Gate logits and target shapes must match')
    if current_state.ndim != 4 or current_state.shape[0] != target.shape[0]:
        raise ValueError('Current state must have shape [B,Z,H,W]')
    if current_state.shape[1:] != target.shape[2:]:
        raise ValueError('Current state and target spatial shapes must match')

    known = target != IGNORE_INDEX
    current_unknown = (current_state == 0)[:, None].expand_as(target)
    current_visible = ~current_unknown
    persistence = persistence_class_from_state(
        current_state, target.shape[1])

    reveal_target = known
    reveal_loss = _masked_binary_loss(
        reveal_logits, reveal_target, current_unknown,
        reveal_positive_weight)
    reveal_class_target = target.clone()
    reveal_class_mask = current_unknown & known
    reveal_class_target[~reveal_class_mask] = IGNORE_INDEX
    reveal_class_loss = masked_world_cross_entropy(
        reveal_class_logits, reveal_class_target,
        class_weights=reveal_class_weights)

    transition_mask = current_visible & known
    transition_target = transition_mask & (target != persistence)
    transition_loss = _masked_binary_loss(
        transition_logits, transition_target, transition_mask,
        transition_positive_weight)
    transition_class_target = target.clone()
    transition_class_target[~transition_target] = IGNORE_INDEX
    transition_class_loss = masked_world_cross_entropy(
        transition_class_logits, transition_class_target,
        class_weights=transition_class_weights)

    total = (
        float(reveal_gate_loss_weight) * reveal_loss +
        float(reveal_class_loss_weight) * reveal_class_loss +
        float(transition_gate_loss_weight) * transition_loss +
        float(transition_class_loss_weight) * transition_class_loss)
    return {
        'loss': total,
        'reveal_gate_loss': reveal_loss,
        'reveal_class_loss': reveal_class_loss,
        'transition_gate_loss': transition_loss,
        'transition_class_loss': transition_class_loss,
        'reveal_voxel_count': current_unknown.sum(),
        'revealed_voxel_count': reveal_class_mask.sum(),
        'transition_voxel_count': transition_mask.sum(),
        'changed_visible_voxel_count': transition_target.sum(),
    }
