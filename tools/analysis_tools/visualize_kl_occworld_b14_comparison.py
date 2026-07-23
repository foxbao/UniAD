#!/usr/bin/env python
"""Visualize B13 raw semantics and B14 local flow overlay on validation."""

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

from projects.mmdet3d_plugin.uniad.dense_heads.occworld_head import (
    apply_local_flow_overlay,
    apply_physical_flow_fusion,
)
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
from tools.analysis_tools.visualize_kl_occworld_exported_predictions import (
    _semantic_row,
)
from tools.data_converter.generate_kl_occworld_history_batch import (
    _voxel_z_centers,
)


def _mask_prediction(states: np.ndarray, known: np.ndarray) -> np.ndarray:
    states = states.copy()
    states[~known] = 0
    return states


def _binary_row(name, values, valid, target_times, z_centers, collision_z):
    return np.concatenate([
        _change_tile(
            value.astype(np.float32), scope, z_centers, collision_z,
            f'{name} | t={time_value:.1f}s')
        for value, scope, time_value in zip(values, valid, target_times)
    ], axis=1)


def _fusion_predictions(raw_class, current_state, current_valid,
                        warped_instance_probability, threshold):
    raw = torch.from_numpy(raw_class.astype(np.int64))[None]
    observation_known = torch.from_numpy(
        current_valid.astype(np.bool_))[None]
    observation_state = torch.from_numpy(
        current_state.astype(np.int64))[None]
    observation_class = torch.zeros_like(observation_state)
    observation_class[observation_known] = (
        observation_state[observation_known] - 1)
    warped = torch.from_numpy(
        warped_instance_probability.astype(np.float32))[None]
    physical = apply_physical_flow_fusion(
        raw, observation_class, observation_known, warped, threshold)
    local = apply_local_flow_overlay(
        raw, observation_class, observation_known, warped, threshold)
    return (
        physical[0].numpy().astype(np.uint8),
        local[0].numpy().astype(np.uint8),
    )


def _contact_sheet(rows):
    separator = np.full(
        (6, rows[0].shape[1], 3), 28, dtype=np.uint8)
    contact = rows[0]
    for row in rows[1:]:
        contact = np.concatenate([contact, separator, row], axis=0)
    return contact


def _overview_grid(records, columns=2):
    tiles = [
        _tile(
            record['contact'],
            (f"#{record['reference_index']} | changed="
             f"{record['changed_voxels']:,} | fixed="
             f"{record['b14_only_correct_voxels']:,} | harmed="
             f"{record['b13_only_correct_voxels']:,}"),
            (900, 980))
        for record in records
    ]
    blank = np.full_like(tiles[0], 28)
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_dense_train3_manifest_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument(
        '--b13-prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b13_dense_incremental_local_'
            'continuous10_v1/validation/epoch_008'))
    parser.add_argument(
        '--b14-prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b14_local_overlay09_v1/'
            'validation/epoch_008'))
    parser.add_argument('--flow-threshold', type=float, default=0.90)
    parser.add_argument('--overview-count', type=int, default=6)
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b14_validation_visuals_v1'))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.flow_threshold <= 1.0:
        raise ValueError('--flow-threshold must be in [0, 1]')
    if args.overview_count <= 0:
        raise ValueError('--overview-count must be positive')

    manifest = _load_manifest(args.manifest)
    references = _split_references(manifest, 'validation')
    labels = _sequence_mapping(args.sequence_root)
    b13_predictions = _prediction_mapping(args.b13_prediction_root)
    b14_predictions = _prediction_mapping(args.b14_prediction_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for reference_index in references:
        with np.load(labels[reference_index], allow_pickle=False) as label:
            target = np.asarray(
                label['world_target_state_3d'], dtype=np.uint8)
            target_valid = np.asarray(
                label['world_target_valid_3d'], dtype=np.bool_)
            current = np.asarray(
                label['current_observation_state_3d'], dtype=np.uint8)
            current_valid = np.asarray(
                label['current_observation_valid_3d'], dtype=np.bool_)
            target_times = np.asarray(
                label['target_times_s'], dtype=np.float32)
            pc_range = np.asarray(label['pc_range'], dtype=np.float32)
            occ_size = np.asarray(label['occ_size'], dtype=np.int64)
            collision_z = tuple(
                np.asarray(label['collision_z'], dtype=np.float32))

        with np.load(
                b13_predictions[reference_index],
                allow_pickle=False) as prediction:
            raw_class = np.asarray(
                prediction['world_pred_class_3d'], dtype=np.uint8)
            warped_instance = np.asarray(
                prediction['warped_instance_probability_3d'],
                dtype=np.float32)
        with np.load(
                b14_predictions[reference_index],
                allow_pickle=False) as prediction:
            b14_class = np.asarray(
                prediction['world_pred_class_3d'], dtype=np.uint8)

        physical_class, reconstructed_b14_class = _fusion_predictions(
            raw_class, current, current_valid, warped_instance,
            args.flow_threshold)
        if not np.array_equal(reconstructed_b14_class, b14_class):
            mismatch = int(np.count_nonzero(
                reconstructed_b14_class != b14_class))
            raise AssertionError(
                f'B14 export mismatch for {reference_index}: '
                f'{mismatch} voxels')

        known = target_valid & (target != 0)
        target_display = target.copy()
        target_display[~known] = 0
        current_display = np.broadcast_to(current, target.shape).copy()
        persistence = current_display.copy()
        persistence[~known] = 0
        raw_display = _mask_prediction(raw_class + 1, known)
        physical_display = _mask_prediction(physical_class + 1, known)
        b14_display = _mask_prediction(b14_class + 1, known)

        current_known = current_valid & (current != 0)
        current_known_future = np.broadcast_to(
            current_known, target.shape)
        current_future = np.broadcast_to(current, target.shape)
        transition = (
            known & current_known_future & (target != current_future))
        changed = known & (b14_class != raw_class)

        future_scope = known.copy()
        future_scope[0] = False
        target_class = target.astype(np.int16) - 1
        raw_correct = future_scope & (raw_class == target_class)
        b14_correct = future_scope & (b14_class == target_class)
        b14_only_correct = b14_correct & ~raw_correct
        b13_only_correct = raw_correct & ~b14_correct

        z_centers = _voxel_z_centers(pc_range, occ_size)
        rows = [
            _semantic_row(
                'future GT', target_display, target_times,
                z_centers, collision_z),
            _semantic_row(
                'current observation (repeated)', current_display,
                target_times, z_centers, collision_z),
            _semantic_row(
                'persistence baseline', persistence, target_times,
                z_centers, collision_z),
            _semantic_row(
                'B13 raw prediction', raw_display, target_times,
                z_centers, collision_z),
            _semantic_row(
                'legacy physical fusion', physical_display, target_times,
                z_centers, collision_z),
            _semantic_row(
                'B14 local overlay', b14_display, target_times,
                z_centers, collision_z),
            _binary_row(
                'GT visible transition', transition, known,
                target_times, z_centers, collision_z),
            _binary_row(
                'B14 changed vs B13', changed, known,
                target_times, z_centers, collision_z),
            _binary_row(
                'B14 fixed B13 error', b14_only_correct, known,
                target_times, z_centers, collision_z),
        ]
        contact = _contact_sheet(rows)
        contact_path = args.out_dir / (
            f'{reference_index:06d}__b14_compare.png')
        if not cv2.imwrite(str(contact_path), contact):
            raise OSError(f'Failed to write {contact_path}')
        records.append({
            'reference_index': int(reference_index),
            'changed_voxels': int(np.count_nonzero(changed[1:])),
            'b14_only_correct_voxels': int(np.count_nonzero(
                b14_only_correct)),
            'b13_only_correct_voxels': int(np.count_nonzero(
                b13_only_correct)),
            'visible_transition_voxels': int(np.count_nonzero(
                transition[1:])),
            'contact_path': str(contact_path),
            'contact': contact,
        })

    ranked = sorted(
        records, key=lambda item: item['changed_voxels'], reverse=True)
    overview_records = ranked[:min(args.overview_count, len(ranked))]
    overview = _overview_grid(overview_records)
    overview_path = args.out_dir / 'top_b14_changes_overview.png'
    if not cv2.imwrite(str(overview_path), overview):
        raise OSError(f'Failed to write {overview_path}')

    serializable_records = [
        {key: value for key, value in record.items() if key != 'contact'}
        for record in ranked
    ]
    summary = {
        'split': 'validation',
        'reference_count': len(references),
        'flow_threshold': args.flow_threshold,
        'row_order': [
            'future GT',
            'current observation repeated',
            'persistence baseline',
            'B13 raw prediction',
            'legacy physical fusion',
            'B14 local overlay',
            'GT visible transition',
            'B14 changed vs B13',
            'B14 fixed B13 error',
        ],
        'overview_reference_indices': [
            record['reference_index'] for record in overview_records
        ],
        'overview_path': str(overview_path),
        'records_ranked_by_changed_voxels': serializable_records,
    }
    summary_path = args.out_dir / 'summary.json'
    with summary_path.open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'reference_count': summary['reference_count'],
        'overview_path': summary['overview_path'],
        'summary_path': str(summary_path),
        'overview_reference_indices': summary[
            'overview_reference_indices'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
