#!/usr/bin/env python
"""Visualize decomposed OccWorld predictions and both gate probabilities."""

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
from tools.analysis_tools.visualize_kl_occworld_change_gate import _change_tile
from tools.analysis_tools.visualize_kl_occworld_temporal_predictions import (
    _bev_tile,
)
from tools.data_converter.generate_kl_occworld_history_batch import (
    _voxel_z_centers,
)
from tools.data_converter.kl_occworld_dataset import (
    KLOccWorldSequenceDataset,
)
from tools.tutorials.occworld_toy.step04_decomposed_change_baseline import (
    DecomposedTemporalOccWorld,
)


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
    threshold_summary = summary['validation_threshold_sweep']
    reveal_threshold = float(
        threshold_summary['selected_reveal']['threshold'])
    transition_threshold = float(
        threshold_summary['selected_visible_transition']['threshold'])
    dataset = KLOccWorldSequenceDataset(
        args.sequence_root,
        expected_shape=(10, 120, 160),
        history_root=args.history_root,
        drop_missing_history=True)
    sample0 = dataset[0]
    model = DecomposedTemporalOccWorld(
        history_count=sample0['history_one_hot_3d'].shape[0],
        state_channels=sample0['history_one_hot_3d'].shape[1],
        hidden_channels=summary['hidden_channels'],
        horizon_count=sample0['target_class_3d'].shape[0],
        reveal_prior=summary['reveal_prior'],
        transition_prior=summary['transition_prior'],
        include_frame_differences=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    reference_to_index = {
        int(reference): index
        for index, reference in enumerate(dataset.reference_indices)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contact_paths = []
    overview_images = []
    with torch.no_grad():
        for reference_index in summary['test_reference_indices']:
            sample = dataset[reference_to_index[int(reference_index)]]
            outputs = model.predict(
                sample['history_one_hot_3d'][None],
                reveal_threshold, transition_threshold,
                decision_mode=summary.get('decision_mode', 'hard'))
            target = sample['target_state_3d'].cpu().numpy()
            valid = sample['target_valid_3d'].cpu().numpy()
            persistence = (
                outputs['persistence_class'][0].cpu().numpy() + 1
            ).astype(np.uint8)
            prediction = (
                outputs['prediction_class'][0].cpu().numpy() + 1
            ).astype(np.uint8)
            reveal_probability = outputs[
                'reveal_probability'][0].cpu().numpy()
            transition_probability = outputs[
                'transition_probability'][0].cpu().numpy()
            persistence[~valid] = 0
            prediction[~valid] = 0
            current_unknown = (
                sample['input_state_3d'].cpu().numpy() == 0)
            reveal_scope = np.broadcast_to(
                current_unknown, reveal_probability.shape)
            transition_scope = np.broadcast_to(
                ~current_unknown, transition_probability.shape)
            pc_range = sample['pc_range'].cpu().numpy()
            occ_size = sample['occ_size'].cpu().numpy()
            z_centers = _voxel_z_centers(pc_range, occ_size)
            collision_z = (0.3, 2.5)
            target_times = sample['target_times_s'].cpu().numpy()
            rows = []
            for method, states in (
                    ('target', target),
                    ('persistence', persistence),
                    ('decomposed', prediction)):
                rows.append(np.concatenate([
                    _bev_tile(
                        state, z_centers, collision_z,
                        f'{method} | t={time_value:.1f}s')
                    for state, time_value in zip(states, target_times)
                ], axis=1))
            rows.append(np.concatenate([
                _change_tile(
                    probability, scope, z_centers, collision_z,
                    f'reveal p | t={time_value:.1f}s')
                for probability, scope, time_value in zip(
                    reveal_probability, reveal_scope, target_times)
            ], axis=1))
            rows.append(np.concatenate([
                _change_tile(
                    probability, scope, z_centers, collision_z,
                    f'transition p | t={time_value:.1f}s')
                for probability, scope, time_value in zip(
                    transition_probability, transition_scope, target_times)
            ], axis=1))
            separator = np.full(
                (6, rows[0].shape[1], 3), 28, dtype=np.uint8)
            contact = rows[0]
            for row in rows[1:]:
                contact = np.concatenate([contact, separator, row], axis=0)
            contact_path = args.out_dir / (
                f'{int(reference_index):06d}__decomposed_compare.png')
            cv2.imwrite(str(contact_path), contact)
            contact_paths.append(str(contact_path))
            overview_images.append(_tile(
                contact,
                f'#{int(reference_index)} | reveal={reveal_threshold:.2f} '
                f'transition={transition_threshold:.2f}',
                (1200, 760)))
    overview_path = args.out_dir / 'decomposed_prediction_overview.png'
    cv2.imwrite(str(overview_path), np.concatenate(overview_images, axis=0))
    result = {
        'reveal_threshold': reveal_threshold,
        'transition_threshold': transition_threshold,
        'test_reference_indices': summary['test_reference_indices'],
        'contact_paths': contact_paths,
        'overview_path': str(overview_path),
    }
    with (args.out_dir / 'summary.json').open('w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
