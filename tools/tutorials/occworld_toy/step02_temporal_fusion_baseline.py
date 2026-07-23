#!/usr/bin/env python
"""Tiny reference-aligned temporal OccWorld baseline."""

import torch
from torch import nn

from tools.tutorials.occworld_toy.step01_current_to_future_baseline import (
    persistence_logits_from_one_hot,
)


class TemporalFusionOccWorld(nn.Module):
    """Fuse aligned history states and first differences with a 3D CNN."""

    def __init__(self, history_count=5, state_channels=4,
                 hidden_channels=4, horizon_count=4, class_count=3,
                 persistence_logit_scale=1.0,
                 include_frame_differences=True):
        super().__init__()
        if history_count < 2:
            raise ValueError('Temporal fusion requires at least two frames')
        if state_channels != 4 or class_count != 3:
            raise ValueError(
                'Temporal persistence requires 4 states and 3 classes')
        self.history_count = int(history_count)
        self.state_channels = int(state_channels)
        self.horizon_count = int(horizon_count)
        self.class_count = int(class_count)
        self.persistence_logit_scale = float(persistence_logit_scale)
        self.include_frame_differences = bool(include_frame_differences)
        fusion_channels = self.history_count * self.state_channels
        if self.include_frame_differences:
            fusion_channels += (
                (self.history_count - 1) * self.state_channels)
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(
                fusion_channels, hidden_channels,
                kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                hidden_channels, hidden_channels,
                kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual_head = nn.Conv3d(
            hidden_channels, self.horizon_count * self.class_count,
            kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _fusion_input(self, history_one_hot_3d: torch.Tensor) -> torch.Tensor:
        if history_one_hot_3d.ndim != 6:
            raise ValueError('Expected history [B,P,C,Z,H,W]')
        batch_size, history_count, state_channels = (
            history_one_hot_3d.shape[:3])
        if (history_count != self.history_count or
                state_channels != self.state_channels):
            raise ValueError(
                f'Expected history P={self.history_count}, '
                f'C={self.state_channels}; got P={history_count}, '
                f'C={state_channels}')
        spatial_shape = history_one_hot_3d.shape[3:]
        states = history_one_hot_3d.reshape(
            batch_size, history_count * state_channels, *spatial_shape)
        if not self.include_frame_differences:
            return states
        differences = (
            history_one_hot_3d[:, 1:] - history_one_hot_3d[:, :-1]
        ).reshape(
            batch_size, (history_count - 1) * state_channels,
            *spatial_shape)
        return torch.cat([states, differences], dim=1)

    def forward(self, history_one_hot_3d: torch.Tensor) -> torch.Tensor:
        fusion_input = self._fusion_input(history_one_hot_3d)
        current = history_one_hot_3d[:, -1]
        persistence = persistence_logits_from_one_hot(
            current, self.persistence_logit_scale)
        batch_size, _, z_size, height, width = persistence.shape
        persistence = persistence[:, None].expand(
            -1, self.horizon_count, -1, -1, -1, -1)
        residual = self.residual_head(
            self.temporal_encoder(fusion_input))
        residual = residual.reshape(
            batch_size, self.horizon_count, self.class_count,
            z_size, height, width)
        return persistence + residual
