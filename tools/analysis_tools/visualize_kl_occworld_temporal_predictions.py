#!/usr/bin/env python
"""Visualize target, persistence, single and temporal OccWorld predictions."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from tools.analysis_tools.audit_kl_occworld_dual_representation import _tile
from tools.data_converter.generate_kl_occworld_history_batch import (
    _voxel_z_centers,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    OCCUPANCY_PALETTE,
    _project_world_to_bev,
)
from tools.data_converter.kl_occworld_dataset import (
    KLOccWorldSequenceDataset,
)
from tools.tutorials.occworld_toy.step01_current_to_future_baseline import (
    PersistenceResidualOccWorld,
)
from tools.tutorials.occworld_toy.step02_temporal_fusion_baseline import (
    TemporalFusionOccWorld,
)


def _bev_tile(state_3d: np.ndarray, z_centers: np.ndarray,
              collision_z, title: str) -> np.ndarray:
    bev = _project_world_to_bev(state_3d, z_centers, collision_z)
    rgb = OCCUPANCY_PALETTE[bev]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return _tile(bgr, title, (300, 225))


def _prediction_raw_state(logits: torch.Tensor) -> np.ndarray:
    return (logits.argmax(dim=2)[0].cpu().numpy() + 1).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root', required=True)
    parser.add_argument('--history-root', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    summary = checkpoint['summary']
    dataset = KLOccWorldSequenceDataset(
        args.sequence_root,
        expected_shape=(10, 120, 160),
        history_root=args.history_root,
        drop_missing_history=True)
    sample0 = dataset[0]
    history_count = sample0['history_one_hot_3d'].shape[0]
    state_channels = sample0['history_one_hot_3d'].shape[1]
    horizon_count = sample0['target_class_3d'].shape[0]
    single_model = PersistenceResidualOccWorld(
        in_channels=state_channels,
        hidden_channels=summary['hidden_channels'],
        horizon_count=horizon_count,
        persistence_logit_scale=(
            summary['persistence_logit_scale']))
    temporal_model = TemporalFusionOccWorld(
        history_count=history_count,
        state_channels=state_channels,
        hidden_channels=summary['hidden_channels'],
        horizon_count=horizon_count,
        persistence_logit_scale=(
            summary['persistence_logit_scale']),
        include_frame_differences=True)
    single_model.load_state_dict(checkpoint['single_frame_state_dict'])
    temporal_model.load_state_dict(checkpoint['temporal_state_dict'])
    single_model.eval()
    temporal_model.eval()

    reference_to_index = {
        int(reference): index
        for index, reference in enumerate(dataset.reference_indices)}
    test_references = summary['test_reference_indices']
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contact_paths = []
    overview_images = []
    with torch.no_grad():
        for reference_index in test_references:
            sample = dataset[reference_to_index[int(reference_index)]]
            current = sample['input_state_3d']
            current_class = torch.zeros_like(current)
            current_known = current != 0
            current_class[current_known] = current[current_known] - 1
            persistence = current_class[None].expand(
                horizon_count, -1, -1, -1).cpu().numpy() + 1
            single_logits = single_model(
                sample['input_one_hot_3d'][None])
            temporal_logits = temporal_model(
                sample['history_one_hot_3d'][None])
            predictions = {
                'target': sample['target_state_3d'].cpu().numpy(),
                'persistence': persistence.astype(np.uint8),
                'single': _prediction_raw_state(single_logits),
                'temporal': _prediction_raw_state(temporal_logits),
            }
            target_valid = sample['target_valid_3d'].cpu().numpy()
            for method in ('persistence', 'single', 'temporal'):
                predictions[method][~target_valid] = 0
            pc_range = sample['pc_range'].cpu().numpy()
            occ_size = sample['occ_size'].cpu().numpy()
            z_centers = _voxel_z_centers(pc_range, occ_size)
            collision_z = (0.3, 2.5)
            target_times = sample['target_times_s'].cpu().numpy()
            rows = []
            for method, states in predictions.items():
                rows.append(np.concatenate([
                    _bev_tile(
                        state, z_centers, collision_z,
                        f'{method} | t={time_value:.1f}s')
                    for state, time_value in zip(states, target_times)
                ], axis=1))
            separator = np.full(
                (6, rows[0].shape[1], 3), 28, dtype=np.uint8)
            contact = rows[0]
            for row in rows[1:]:
                contact = np.concatenate(
                    [contact, separator, row], axis=0)
            contact_path = args.out_dir / (
                f'{int(reference_index):06d}__temporal_compare.png')
            cv2.imwrite(str(contact_path), contact)
            contact_paths.append(str(contact_path))
            overview_images.append(_tile(
                contact, f'#{int(reference_index)}', (1200, 620)))
    overview_path = args.out_dir / 'temporal_prediction_overview.png'
    cv2.imwrite(str(overview_path), np.concatenate(overview_images, axis=0))
    result = {
        'test_reference_indices': test_references,
        'contact_paths': contact_paths,
        'overview_path': str(overview_path),
    }
    with (args.out_dir / 'summary.json').open('w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
