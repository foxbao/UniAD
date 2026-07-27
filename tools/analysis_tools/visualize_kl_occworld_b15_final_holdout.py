#!/usr/bin/env python
"""Visualize sealed B15 final-holdout predictions without new inference."""

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
from tools.analysis_tools.visualize_kl_occworld_exported_predictions import (
    _prediction_arrays,
    _semantic_row,
    _visibility_row,
)
from tools.data_converter.generate_kl_occworld_history_batch import (
    _voxel_z_centers,
)


def _even_positions(length: int, count: int) -> list:
    if length < 1 or count < 1:
        raise ValueError('length and count must be positive')
    count = min(length, count)
    return sorted(set(
        np.linspace(0, length - 1, count).round().astype(int).tolist()))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b15_final_holdout30_evaluation_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_sequence_b15_final_holdout30_v1'))
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b15_epoch7_final_holdout30_v1/'
            'final_holdout/epoch_007'))
    parser.add_argument('--split', default='final_holdout')
    parser.add_argument(
        '--required-status',
        default='final_holdout_evaluated_once_no_retuning_allowed')
    parser.add_argument(
        '--model-label', default='B15 epoch 7 + overlay 0.9')
    parser.add_argument('--file-prefix', default='b15_final')
    parser.add_argument(
        '--overview-name', default='b15_final_holdout_even6_overview.png')
    parser.add_argument('--visibility-threshold', type=float, default=0.7)
    parser.add_argument('--overview-count', type=int, default=6)
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b15_final_holdout30_visuals_v1'))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    if manifest.get('status') != args.required_status:
        raise ValueError('Requested holdout is not sealed')
    protocol = manifest['frozen_model_protocol']
    if float(protocol['visibility_threshold']) != args.visibility_threshold:
        raise ValueError('Visualization threshold differs from frozen value')
    references = _split_references(manifest, args.split)
    labels = _sequence_mapping(args.sequence_root)
    predictions = _prediction_mapping(args.prediction_root)
    if set(labels).intersection(references) != set(references):
        raise ValueError('Final visualization labels are incomplete')
    if set(predictions) != set(references):
        raise ValueError('Final visualization predictions are not exact')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contact_paths = []
    overview_by_reference = {}
    for reference in references:
        with np.load(labels[reference], allow_pickle=False) as label:
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
        display_target = target.copy()
        display_target[~known] = 0
        persistence = np.broadcast_to(current, target.shape).copy()
        persistence[~known] = 0
        prediction, valid_probability = _prediction_arrays(
            predictions[reference])
        prediction[~known] = 0
        z_centers = _voxel_z_centers(pc_range, occ_size)
        rows = [
            _semantic_row(
                'GT target', display_target, target_times,
                z_centers, collision_z),
            _semantic_row(
                'constant-current', persistence, target_times,
                z_centers, collision_z),
            _semantic_row(
                args.model_label, prediction, target_times,
                z_centers, collision_z),
            _visibility_row(
                'GT visibility', known.astype(np.float32), target_times,
                z_centers, collision_z),
            _visibility_row(
                f'{args.model_label} visibility', valid_probability,
                target_times,
                z_centers, collision_z, args.visibility_threshold),
        ]
        separator = np.full(
            (6, rows[0].shape[1], 3), 28, dtype=np.uint8)
        contact = rows[0]
        for row in rows[1:]:
            contact = np.concatenate([contact, separator, row], axis=0)
        path = args.out_dir / (
            f'{reference:06d}__{args.file_prefix}_compare.png')
        if not cv2.imwrite(str(path), contact):
            raise RuntimeError(f'Failed to write {path}')
        contact_paths.append(str(path))
        overview_by_reference[reference] = _tile(
            contact, f'{args.split} #{reference}', (1200, 800))

    positions = _even_positions(len(references), args.overview_count)
    overview_references = [references[position] for position in positions]
    tiles = [overview_by_reference[reference]
             for reference in overview_references]
    blank = np.full_like(tiles[0], 28)
    columns = 2
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.concatenate(row, axis=1))
    overview = np.concatenate(rows, axis=0)
    overview_path = args.out_dir / args.overview_name
    if not cv2.imwrite(str(overview_path), overview):
        raise RuntimeError(f'Failed to write {overview_path}')
    summary = {
        'split': args.split,
        'source_status': manifest['status'],
        'new_model_inference_performed': False,
        'selection_rule': (
            'six evenly spaced positions in frozen manifest order'),
        'visibility_threshold': args.visibility_threshold,
        'model_label': args.model_label,
        'all_reference_indices': list(references),
        'overview_reference_indices': overview_references,
        'contact_paths': contact_paths,
        'overview_path': str(overview_path),
    }
    with (args.out_dir / 'summary.json').open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'contact_count': len(contact_paths),
        'overview_reference_indices': overview_references,
        'overview_path': str(overview_path),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
