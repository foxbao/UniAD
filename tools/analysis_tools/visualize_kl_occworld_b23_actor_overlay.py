#!/usr/bin/env python
"""Visualize raw-free Motion actor arrival corrections on validation."""

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
from tools.analysis_tools.visualize_kl_occworld_b14_comparison import (
    _binary_row,
    _contact_sheet,
)
from tools.analysis_tools.visualize_kl_occworld_exported_predictions import (
    _semantic_row,
)
from tools.data_converter.generate_kl_occworld_history_batch import (
    _voxel_z_centers,
)


def _overview_grid(records, columns=2):
    tiles = [
        _tile(
            record['contact'],
            (f"#{record['reference_index']} | changed="
             f"{record['changed_voxels']:,} | fixed="
             f"{record['improved_voxels']:,} | harmed="
             f"{record['harmed_voxels']:,} | net="
             f"{record['net_correct_voxels']:+,}"),
            (900, 820))
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
            'kl_occworld_full_train3_final30_manifest_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b23_raw_free_actor_validation_v1/'
            'validation/epoch_003'))
    parser.add_argument('--split', default='validation')
    parser.add_argument('--overview-count', type=int, default=10)
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b23_validation_visuals_v1'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.overview_count < 1:
        raise ValueError('Overview count must be positive')
    references = _split_references(
        _load_manifest(args.manifest), args.split)
    labels = _sequence_mapping(args.sequence_root)
    predictions = _prediction_mapping(args.prediction_root)
    missing = sorted(set(references).difference(predictions))
    if missing:
        raise KeyError(f'Missing B23 predictions for {missing}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for reference in references:
        with np.load(labels[reference], allow_pickle=False) as label:
            target = np.asarray(
                label['world_target_state_3d'], dtype=np.uint8)
            target_valid = np.asarray(
                label['world_target_valid_3d'], dtype=np.bool_)
            target_times = np.asarray(
                label['target_times_s'], dtype=np.float32)
            pc_range = np.asarray(label['pc_range'], dtype=np.float32)
            occ_size = np.asarray(label['occ_size'], dtype=np.int64)
            collision_z = tuple(
                np.asarray(label['collision_z'], dtype=np.float32))
        with np.load(predictions[reference], allow_pickle=False) as prediction:
            raw = np.asarray(
                prediction['raw_world_pred_class_3d'], dtype=np.uint8)
            candidate = np.asarray(
                prediction['world_pred_class_3d'], dtype=np.uint8)
            arrival = np.asarray(
                prediction['motion_actor_arrival_mask_3d'], dtype=np.bool_)

        known = target_valid & (target != 0)
        target_display = target.copy()
        target_display[~known] = 0
        raw_display = raw + 1
        candidate_display = candidate + 1
        raw_display[~known] = 0
        candidate_display[~known] = 0
        arrival_full = np.zeros_like(known)
        arrival_full[1:] = arrival
        changed = known & (candidate != raw)
        target_class = target.astype(np.int16) - 1
        raw_correct = known & (raw == target_class)
        candidate_correct = known & (candidate == target_class)
        improved = changed & ~raw_correct & candidate_correct
        harmed = changed & raw_correct & ~candidate_correct
        future_scope = known.copy()
        future_scope[0] = False
        z_centers = _voxel_z_centers(pc_range, occ_size)
        rows = [
            _semantic_row(
                'future GT', target_display, target_times,
                z_centers, collision_z),
            _semantic_row(
                'B17A raw', raw_display, target_times,
                z_centers, collision_z),
            _semantic_row(
                'B23 raw-free actor arrival', candidate_display,
                target_times, z_centers, collision_z),
            _binary_row(
                'B23 actor arrival mask', arrival_full, future_scope,
                target_times, z_centers, collision_z),
            _binary_row(
                'B23 fixed raw error', improved, future_scope,
                target_times, z_centers, collision_z),
            _binary_row(
                'B23 harmed raw', harmed, future_scope,
                target_times, z_centers, collision_z),
        ]
        contact = _contact_sheet(rows)
        contact_path = args.out_dir / (
            f'{reference:06d}__b23_actor_overlay.png')
        if not cv2.imwrite(str(contact_path), contact):
            raise OSError(f'Failed to write {contact_path}')
        improved_count = int(np.count_nonzero(improved[1:]))
        harmed_count = int(np.count_nonzero(harmed[1:]))
        records.append({
            'reference_index': int(reference),
            'arrival_voxels': int(np.count_nonzero(arrival)),
            'changed_voxels': int(np.count_nonzero(changed[1:])),
            'improved_voxels': improved_count,
            'harmed_voxels': harmed_count,
            'net_correct_voxels': improved_count - harmed_count,
            'contact_path': str(contact_path),
            'contact': contact,
        })

    ranked = sorted(
        records, key=lambda item: (
            -abs(item['net_correct_voxels']), item['reference_index']))
    overview_records = ranked[:min(args.overview_count, len(ranked))]
    overview = _overview_grid(overview_records)
    overview_path = args.out_dir / 'b23_actor_overlay_overview.png'
    if not cv2.imwrite(str(overview_path), overview):
        raise OSError(f'Failed to write {overview_path}')
    serializable = [
        {key: value for key, value in record.items() if key != 'contact'}
        for record in records
    ]
    summary = {
        'schema_version': 1,
        'reference_count': len(records),
        'prediction_root': str(args.prediction_root),
        'overview_path': str(overview_path),
        'changed_voxels': sum(
            record['changed_voxels'] for record in records),
        'improved_voxels': sum(
            record['improved_voxels'] for record in records),
        'harmed_voxels': sum(
            record['harmed_voxels'] for record in records),
        'net_correct_voxels': sum(
            record['net_correct_voxels'] for record in records),
        'records': serializable,
    }
    summary_path = args.out_dir / 'summary.json'
    with summary_path.open('w') as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
