#!/usr/bin/env python
"""Tiny current-occupancy to future-occupancy baseline.

This is an interface smoke test, not the proposed final OccWorld architecture.
It verifies tensor layout, horizon-specific outputs and masked 3D CE.
"""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from tools.data_converter.kl_occworld_dataset import (
    IGNORE_INDEX,
    KLOccWorldSequenceDataset,
)


class TinyOccWorld(nn.Module):
    """Small fully-convolutional baseline with one output head per horizon."""

    def __init__(self, in_channels=4, hidden_channels=4,
                 horizon_count=4, class_count=3):
        super().__init__()
        self.horizon_count = int(horizon_count)
        self.class_count = int(class_count)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.future_head = nn.Conv3d(
            hidden_channels, self.horizon_count * self.class_count,
            kernel_size=1)

    def forward(self, input_one_hot_3d: torch.Tensor) -> torch.Tensor:
        if input_one_hot_3d.ndim != 5:
            raise ValueError(
                'Expected [B,C,Z,H,W] input, got '
                f'{tuple(input_one_hot_3d.shape)}')
        features = self.encoder(input_one_hot_3d)
        logits = self.future_head(features)
        batch_size, _, z_size, height, width = logits.shape
        return logits.reshape(
            batch_size, self.horizon_count, self.class_count,
            z_size, height, width)


def persistence_logits_from_one_hot(
        input_one_hot_3d: torch.Tensor,
        logit_scale: float) -> torch.Tensor:
    """Map unknown/free/static/instance one-hot to 3-class persistence."""
    if input_one_hot_3d.ndim != 5 or input_one_hot_3d.shape[1] != 4:
        raise ValueError('Persistence input must have shape [B,4,Z,H,W]')
    return torch.cat([
        input_one_hot_3d[:, 0:1] + input_one_hot_3d[:, 1:2],
        input_one_hot_3d[:, 2:3],
        input_one_hot_3d[:, 3:4],
    ], dim=1) * float(logit_scale)


class PersistenceResidualOccWorld(nn.Module):
    """Initialize from constant-current logits and learn future corrections."""

    def __init__(self, in_channels=4, hidden_channels=4,
                 horizon_count=4, class_count=3,
                 persistence_logit_scale=2.0):
        super().__init__()
        if in_channels != 4 or class_count != 3:
            raise ValueError(
                'Persistence mapping requires 4 input states and 3 classes')
        self.horizon_count = int(horizon_count)
        self.class_count = int(class_count)
        self.persistence_logit_scale = float(persistence_logit_scale)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual_head = nn.Conv3d(
            hidden_channels, self.horizon_count * self.class_count,
            kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, input_one_hot_3d: torch.Tensor) -> torch.Tensor:
        if input_one_hot_3d.ndim != 5:
            raise ValueError(
                'Expected [B,C,Z,H,W] input, got '
                f'{tuple(input_one_hot_3d.shape)}')
        # Unknown current voxels follow the same convention as the explicit
        # constant baseline: predict free until the residual branch learns
        # stronger evidence for an occupied class.
        persistence = persistence_logits_from_one_hot(
            input_one_hot_3d, self.persistence_logit_scale)
        batch_size, _, z_size, height, width = persistence.shape
        persistence = persistence[:, None].expand(
            -1, self.horizon_count, -1, -1, -1, -1)
        residual = self.residual_head(self.encoder(input_one_hot_3d))
        residual = residual.reshape(
            batch_size, self.horizon_count, self.class_count,
            z_size, height, width)
        return persistence + residual


def masked_world_cross_entropy(logits: torch.Tensor,
                               target: torch.Tensor,
                               class_weights: torch.Tensor = None) -> torch.Tensor:
    """Compute CE only over known world voxels (target != 255)."""
    if logits.ndim != 6 or target.ndim != 5:
        raise ValueError('Expected logits [B,T,C,Z,H,W] and target [B,T,Z,H,W]')
    if (logits.shape[0:2] != target.shape[0:2] or
            logits.shape[3:] != target.shape[2:]):
        raise ValueError('Logits and target shapes are incompatible')
    batch_size, horizon_count, class_count = logits.shape[:3]
    flat_logits = logits.reshape(
        batch_size * horizon_count, class_count, *logits.shape[3:])
    flat_target = target.reshape(batch_size * horizon_count, *target.shape[2:])
    per_voxel = F.cross_entropy(
        flat_logits, flat_target, weight=class_weights,
        ignore_index=IGNORE_INDEX, reduction='none')
    known = flat_target != IGNORE_INDEX
    if not torch.any(known):
        return per_voxel.sum() * 0.0
    return per_voxel.masked_select(known).mean()


def change_aware_world_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        current_state: torch.Tensor,
        class_weights: torch.Tensor = None,
        change_voxel_weight: float = 1.0) -> torch.Tensor:
    """Upweight voxels whose future class differs from persistence."""
    if change_voxel_weight < 1.0:
        raise ValueError('change_voxel_weight must be at least 1')
    if logits.ndim != 6 or target.ndim != 5 or current_state.ndim != 4:
        raise ValueError('Invalid change-aware loss tensor dimensions')
    if (logits.shape[0:2] != target.shape[0:2] or
            logits.shape[3:] != target.shape[2:] or
            current_state.shape[0] != target.shape[0] or
            current_state.shape[1:] != target.shape[2:]):
        raise ValueError('Change-aware loss shapes are incompatible')
    batch_size, horizon_count, class_count = logits.shape[:3]
    flat_logits = logits.reshape(
        batch_size * horizon_count, class_count, *logits.shape[3:])
    flat_target = target.reshape(
        batch_size * horizon_count, *target.shape[2:])
    per_voxel = F.cross_entropy(
        flat_logits, flat_target, weight=class_weights,
        ignore_index=IGNORE_INDEX, reduction='none')
    persistence = torch.zeros_like(current_state, dtype=torch.long)
    current_known = current_state != 0
    persistence[current_known] = current_state[current_known] - 1
    persistence = persistence[:, None].expand_as(target).reshape_as(
        flat_target)
    known = flat_target != IGNORE_INDEX
    changed = known & (flat_target != persistence)
    voxel_weights = torch.ones_like(per_voxel)
    voxel_weights[changed] = change_voxel_weight
    if not torch.any(known):
        return per_voxel.sum() * 0.0
    return (
        per_voxel[known] * voxel_weights[known]
    ).sum() / voxel_weights[known].sum()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--hidden-channels', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-file', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.hidden_channels < 1:
        raise ValueError('batch-size and hidden-channels must be positive')
    torch.manual_seed(args.seed)
    dataset = KLOccWorldSequenceDataset(args.sequence_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))
    target = batch['target_class_3d']
    model = TinyOccWorld(
        in_channels=batch['input_one_hot_3d'].shape[1],
        hidden_channels=args.hidden_channels,
        horizon_count=target.shape[1])
    class_weights = torch.tensor([0.384, 1.443, 1.172], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch['input_one_hot_3d'].float())
    loss = masked_world_cross_entropy(logits, target, class_weights)
    loss.backward()
    optimizer.step()
    result = {
        'batch_size': args.batch_size,
        'input_shape': list(batch['input_one_hot_3d'].shape),
        'target_shape': list(target.shape),
        'logits_shape': list(logits.shape),
        'known_target_voxels': int((target != IGNORE_INDEX).sum()),
        'loss': float(loss.detach()),
        'gradient_norm_encoder_first': float(
            model.encoder[0].weight.grad.norm().detach()),
    }
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        with args.out_file.open('w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
