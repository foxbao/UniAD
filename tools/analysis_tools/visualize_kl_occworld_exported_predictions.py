#!/usr/bin/env python
"""Visualize exported B0/B1 OccWorld predictions against GT/persistence."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.analysis_tools.audit_kl_occworld_dual_representation import _tile
from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
    _split_references,
)
from tools.analysis_tools.visualize_kl_occworld_change_gate import (
    _change_tile,
)
from tools.analysis_tools.visualize_kl_occworld_temporal_predictions import (
    _bev_tile,
)
from tools.data_converter.generate_kl_occworld_history_batch import (
    _voxel_z_centers,
)


def _semantic_row(name, states, target_times, z_centers, collision_z):
    return np.concatenate([
        _bev_tile(
            state, z_centers, collision_z,
            f'{name} | t={time_value:.1f}s')
        for state, time_value in zip(states, target_times)
    ], axis=1)


def _visibility_row(name, probabilities, target_times,
                    z_centers, collision_z, threshold=None):
    scope = np.ones_like(probabilities, dtype=np.bool_)
    suffix = '' if threshold is None else f' | th={threshold:.2f}'
    return np.concatenate([
        _change_tile(
            probability, valid, z_centers, collision_z,
            f'{name}{suffix} | t={time_value:.1f}s')
        for probability, valid, time_value in zip(
            probabilities, scope, target_times)
    ], axis=1)


def _prediction_arrays(path: Path):
    with np.load(path, allow_pickle=False) as prediction:
        states = np.asarray(
            prediction['world_pred_class_3d'], dtype=np.uint8) + 1
        probabilities = np.asarray(
            prediction['world_valid_probability_3d'], dtype=np.float32)
    return states, probabilities


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_batch20'))
    parser.add_argument(
        '--b0-prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_world_only_v1/test/epoch_005'))
    parser.add_argument(
        '--b1-prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_world_only_balanced_v1/test/epoch_003'))
    parser.add_argument(
        '--b2-prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_world_only_anchor_v1/test/epoch_011'))
    parser.add_argument('--b0-threshold', type=float, default=0.20)
    parser.add_argument('--b1-threshold', type=float, default=0.15)
    parser.add_argument('--b2-threshold', type=float, default=0.50)
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_world_only_test_visuals_v1'))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    references = _split_references(manifest, 'test')
    labels = _sequence_mapping(args.sequence_root)
    b0_predictions = _prediction_mapping(args.b0_prediction_root)
    b1_predictions = _prediction_mapping(args.b1_prediction_root)
    b2_predictions = _prediction_mapping(args.b2_prediction_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contact_paths = []
    overview_tiles = []
    for reference_index in references:
        with np.load(labels[reference_index], allow_pickle=False) as label:
            target = np.asarray(
                label['world_target_state_3d'], dtype=np.uint8)
            target_valid = np.asarray(
                label['world_target_valid_3d'], dtype=np.bool_)
            current = np.asarray(
                label['current_observation_state_3d'], dtype=np.uint8)
            target_times = np.asarray(
                label['target_times_s'], dtype=np.float32)
            pc_range = np.asarray(label['pc_range'], dtype=np.float32)
            occ_size = np.asarray(label['occ_size'], dtype=np.int64)
            collision_z = tuple(
                np.asarray(label['collision_z'], dtype=np.float32))
        known = target_valid & (target != 0)
        target = target.copy()
        target[~known] = 0
        persistence = np.broadcast_to(
            current, target.shape).copy()
        persistence[~known] = 0
        b0_states, b0_probability = _prediction_arrays(
            b0_predictions[reference_index])
        b1_states, b1_probability = _prediction_arrays(
            b1_predictions[reference_index])
        b2_states, b2_probability = _prediction_arrays(
            b2_predictions[reference_index])
        b0_states[~known] = 0
        b1_states[~known] = 0
        b2_states[~known] = 0
        z_centers = _voxel_z_centers(pc_range, occ_size)
        rows = [
            _semantic_row(
                'target', target, target_times, z_centers, collision_z),
            _semantic_row(
                'persistence', persistence, target_times,
                z_centers, collision_z),
            _semantic_row(
                'B0 unweighted', b0_states, target_times,
                z_centers, collision_z),
            _semantic_row(
                'B1 balanced', b1_states, target_times,
                z_centers, collision_z),
            _semantic_row(
                'B2 causal anchor', b2_states, target_times,
                z_centers, collision_z),
            _visibility_row(
                'GT visibility', known.astype(np.float32), target_times,
                z_centers, collision_z),
            _visibility_row(
                'B0 visibility', b0_probability, target_times,
                z_centers, collision_z, args.b0_threshold),
            _visibility_row(
                'B1 visibility', b1_probability, target_times,
                z_centers, collision_z, args.b1_threshold),
            _visibility_row(
                'B2 visibility', b2_probability, target_times,
                z_centers, collision_z, args.b2_threshold),
        ]
        separator = np.full(
            (6, rows[0].shape[1], 3), 28, dtype=np.uint8)
        contact = rows[0]
        for row in rows[1:]:
            contact = np.concatenate([contact, separator, row], axis=0)
        contact_path = args.out_dir / (
            f'{reference_index:06d}__world_only_compare.png')
        cv2.imwrite(str(contact_path), contact)
        contact_paths.append(str(contact_path))
        overview_tiles.append(_tile(
            contact, f'#{reference_index}', (1200, 900)))
    overview_path = args.out_dir / 'world_only_test_overview.png'
    cv2.imwrite(str(overview_path), np.concatenate(overview_tiles, axis=0))
    summary = {
        'reference_indices': list(references),
        'b0_threshold': args.b0_threshold,
        'b1_threshold': args.b1_threshold,
        'b2_threshold': args.b2_threshold,
        'contact_paths': contact_paths,
        'overview_path': str(overview_path),
    }
    with (args.out_dir / 'summary.json').open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
