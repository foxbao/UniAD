#!/usr/bin/env python
"""Verify model-side Motion actor overlay against the frozen Numpy oracle."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    RaycastDrivableBuilder,
)
from tools.analysis_tools.audit_kl_occworld_motion_actor_overlay import (
    apply_motion_actor_arrival_overlay,
)
from tools.analysis_tools.audit_kl_occworld_motion_actor_support import (
    motion_actor_support,
)
from tools.analysis_tools.evaluate_kl_occworld import _prediction_mapping


def verify_prediction(path: Path, builder: RaycastDrivableBuilder,
                      score_threshold: float,
                      raw_class_gate: int) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        required = (
            'reference_index', 'world_pred_class_3d',
            'raw_world_pred_class_3d', 'motion_actor_arrival_mask_3d',
            'motion_actor_future_xy', 'motion_actor_boxes_3d',
            'motion_actor_scores', 'motion_actor_valid',
            'motion_actor_step_times_s', 'motion_actor_box_z_origin')
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f'{path} lacks parity fields: {missing}')
        target_times = np.arange(1, 5, dtype=np.float32) * 0.5
        motion, _ = motion_actor_support(
            payload, builder, target_times,
            score_threshold=score_threshold)
        stationary, _ = motion_actor_support(
            payload, builder, target_times,
            score_threshold=score_threshold,
            position_policy='stationary')
        candidate, arrival, _ = apply_motion_actor_arrival_overlay(
            payload['raw_world_pred_class_3d'], motion, stationary,
            raw_class_gate=[raw_class_gate])
        saved_arrival = np.asarray(
            payload['motion_actor_arrival_mask_3d'], dtype=np.bool_)
        saved_prediction = np.asarray(
            payload['world_pred_class_3d'], dtype=np.uint8)
        return {
            'reference_index': int(payload['reference_index']),
            'arrival_voxels': int(np.count_nonzero(arrival)),
            'mask_difference_voxels': int(np.count_nonzero(
                arrival != saved_arrival)),
            'prediction_difference_voxels': int(np.count_nonzero(
                candidate != saved_prediction)),
            'prediction_path': str(path),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b23_raw_free_actor_validation_v1/'
            'validation/epoch_003'))
    parser.add_argument('--score-threshold', type=float, default=0.1)
    parser.add_argument('--raw-class-gate', type=int, default=0)
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument(
        '--occ-size', type=int, nargs=3, default=[160, 120, 10])
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b23_model_overlay_parity_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = _prediction_mapping(args.prediction_root)
    builder = RaycastDrivableBuilder(
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size)
    rows = [
        verify_prediction(
            path, builder,
            score_threshold=args.score_threshold,
            raw_class_gate=args.raw_class_gate)
        for _, path in sorted(predictions.items())
    ]
    mask_difference = sum(
        row['mask_difference_voxels'] for row in rows)
    prediction_difference = sum(
        row['prediction_difference_voxels'] for row in rows)
    report = {
        'schema_version': 1,
        'purpose': (
            'Exact parity between model-side Torch actor raster/overlay and '
            'the frozen Numpy OccWorld box raster oracle.'),
        'prediction_root': str(args.prediction_root),
        'reference_count': len(rows),
        'score_threshold': float(args.score_threshold),
        'raw_class_gate': int(args.raw_class_gate),
        'mask_difference_voxels': int(mask_difference),
        'prediction_difference_voxels': int(prediction_difference),
        'passed': mask_difference == 0 and prediction_difference == 0,
        'rows': rows,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
