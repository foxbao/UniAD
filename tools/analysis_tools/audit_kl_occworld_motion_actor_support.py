#!/usr/bin/env python
"""Audit 3D future-instance support from aligned MotionHead actor geometry."""

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
from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
    _split_references,
)
from tools.data_converter.generate_kl_occworld_labels import _xyz_to_zhw
from tools.data_converter.kl_occworld_track_adapter import (
    track_boxes_to_occworld_z_convention,
)


INSTANCE_OCCUPIED = 3


def align_motion_steps(step_times: np.ndarray,
                       target_times: np.ndarray,
                       tolerance_s: float = 0.05) -> list:
    """Match each future OccWorld horizon to one nominal MotionHead step."""
    step_times = np.asarray(step_times, dtype=np.float32).reshape(-1)
    target_times = np.asarray(target_times, dtype=np.float32).reshape(-1)
    if step_times.size == 0 or target_times.size == 0:
        raise ValueError('Motion and target times must be non-empty')
    if tolerance_s < 0:
        raise ValueError('Time tolerance must be non-negative')
    indices = []
    for target_time in target_times:
        index = int(np.argmin(np.abs(step_times - target_time)))
        error = abs(float(step_times[index] - target_time))
        if error > tolerance_s:
            raise ValueError(
                f'No motion step matches target {target_time:.3f}s; '
                f'nearest error is {error:.3f}s')
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError('Multiple OccWorld horizons map to one motion step')
    return indices


def rasterize_boxes_3d(builder: RaycastDrivableBuilder,
                       boxes: np.ndarray) -> np.ndarray:
    """Rasterize legacy OccWorld box-floor volumes into [Z,H,W]."""
    occupied_xyz = np.zeros(tuple(builder.occ_size.tolist()), dtype=np.bool_)
    for box in np.asarray(boxes, dtype=np.float32):
        indices = builder.box_voxel_indices(box)
        if indices is None:
            continue
        x_idx, y_idx, z_idx = indices
        occupied_xyz[x_idx[:, None], y_idx[:, None], z_idx[None, :]] = True
    return _xyz_to_zhw(occupied_xyz)


def motion_actor_support(payload, builder: RaycastDrivableBuilder,
                         target_times: np.ndarray,
                         score_threshold: float,
                         time_tolerance_s: float = 0.05,
                         position_policy: str = 'motion') -> tuple:
    """Build per-horizon 3D support while preserving current size/yaw/z."""
    required = (
        'motion_actor_future_xy', 'motion_actor_boxes_3d',
        'motion_actor_scores', 'motion_actor_valid',
        'motion_actor_step_times_s', 'motion_actor_box_z_origin')
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError('Prediction lacks motion actor fields: ' +
                         ', '.join(missing))
    future = np.asarray(payload['motion_actor_future_xy'], dtype=np.float32)
    boxes = np.asarray(payload['motion_actor_boxes_3d'], dtype=np.float32)
    scores = np.asarray(payload['motion_actor_scores'], dtype=np.float32)
    valid = np.asarray(payload['motion_actor_valid'], dtype=np.bool_)
    step_times = np.asarray(
        payload['motion_actor_step_times_s'], dtype=np.float32)
    z_origin = str(np.asarray(
        payload['motion_actor_box_z_origin']).item())
    if future.ndim != 3 or future.shape[-1] != 2:
        raise ValueError(f'Invalid actor future shape {future.shape}')
    actor_count, step_count = future.shape[:2]
    if (boxes.shape != (actor_count, 7) or
            scores.shape != (actor_count,) or
            valid.shape != (actor_count,) or
            step_times.shape != (step_count,)):
        raise ValueError('Motion actor arrays have inconsistent shapes')
    if score_threshold < 0:
        raise ValueError('Score threshold must be non-negative')
    if position_policy not in ('motion', 'stationary'):
        raise ValueError('Position policy must be motion or stationary')
    step_indices = align_motion_steps(
        step_times, target_times, tolerance_s=time_tolerance_s)
    boxes = track_boxes_to_occworld_z_convention(
        boxes, source_z_origin=z_origin)
    keep = valid & np.isfinite(scores) & (scores >= score_threshold)
    keep &= np.isfinite(boxes).all(axis=1)
    keep &= np.all(boxes[:, 3:6] > 0, axis=1)
    keep &= np.isfinite(future).all(axis=(1, 2))

    supports = []
    for step_index in step_indices:
        future_boxes = boxes[keep].copy()
        if position_policy == 'motion':
            future_boxes[:, :2] = future[keep, step_index]
        supports.append(rasterize_boxes_3d(builder, future_boxes))
    return np.stack(supports), {
        'input_actor_count': int(actor_count),
        'kept_actor_count': int(np.count_nonzero(keep)),
        'motion_step_indices': [int(index) for index in step_indices],
        'motion_step_times_s': [
            float(step_times[index]) for index in step_indices],
        'target_times_s': [float(value) for value in target_times],
        'score_threshold': float(score_threshold),
        'box_z_input_origin': z_origin,
        'box_z_raster_origin': 'occworld_legacy_center',
        'yaw_policy': 'hold_current',
        'height_policy': 'hold_current',
        'position_policy': position_policy,
    }


def support_counts(support: np.ndarray,
                   target_state: np.ndarray,
                   target_valid: np.ndarray,
                   current_state: np.ndarray,
                   current_valid: np.ndarray) -> dict:
    """Count support overlap, visible arrivals, and continuations."""
    support = np.asarray(support, dtype=np.bool_)
    target_state = np.asarray(target_state, dtype=np.uint8)
    target_known = np.asarray(target_valid, dtype=np.bool_) & (
        target_state != 0)
    if support.shape != target_state.shape or target_known.shape != support.shape:
        raise ValueError('Support and future target shapes must match')
    current_state = np.asarray(current_state, dtype=np.uint8)
    current_known = np.asarray(current_valid, dtype=np.bool_) & (
        current_state != 0)
    if current_state.shape != support.shape[1:]:
        raise ValueError('Current state must have shape [Z,H,W]')

    current_instance = current_known & (current_state == INSTANCE_OCCUPIED)
    current_instance = np.broadcast_to(current_instance, support.shape)
    current_known = np.broadcast_to(current_known, support.shape)
    target_instance = target_known & (target_state == INSTANCE_OCCUPIED)
    known_support = support & target_known
    intersection = support & target_instance
    union = known_support | target_instance
    arrival = target_instance & current_known & ~current_instance
    continuation = target_instance & current_instance
    return {
        'all_support_voxels': int(np.count_nonzero(support)),
        'known_support_voxels': int(np.count_nonzero(known_support)),
        'unknown_support_voxels': int(np.count_nonzero(
            support & ~target_known)),
        'target_instance_voxels': int(np.count_nonzero(target_instance)),
        'intersection_voxels': int(np.count_nonzero(intersection)),
        'union_voxels': int(np.count_nonzero(union)),
        'arrival_target_voxels': int(np.count_nonzero(arrival)),
        'arrival_hit_voxels': int(np.count_nonzero(support & arrival)),
        'continuation_target_voxels': int(np.count_nonzero(continuation)),
        'continuation_hit_voxels': int(np.count_nonzero(
            support & continuation)),
    }


def summarize_counts(counts: dict) -> dict:
    """Derive metrics only after micro-aggregating integer counts."""
    result = dict(counts)
    result.update(
        instance_precision_known=(
            counts['intersection_voxels'] /
            max(counts['known_support_voxels'], 1)),
        instance_recall=(
            counts['intersection_voxels'] /
            max(counts['target_instance_voxels'], 1)),
        instance_iou_known=(
            counts['intersection_voxels'] /
            max(counts['union_voxels'], 1)),
        arrival_recall=(
            counts['arrival_hit_voxels'] /
            max(counts['arrival_target_voxels'], 1)),
        continuation_recall=(
            counts['continuation_hit_voxels'] /
            max(counts['continuation_target_voxels'], 1)))
    return result


def support_correction_counts(candidate: np.ndarray,
                              baseline: np.ndarray,
                              target_state: np.ndarray,
                              target_valid: np.ndarray) -> dict:
    """Compare motion support with stationary support on known voxels."""
    candidate = np.asarray(candidate, dtype=np.bool_)
    baseline = np.asarray(baseline, dtype=np.bool_)
    target_state = np.asarray(target_state, dtype=np.uint8)
    known = np.asarray(target_valid, dtype=np.bool_) & (target_state != 0)
    if not (candidate.shape == baseline.shape == target_state.shape ==
            known.shape):
        raise ValueError('Paired supports and targets must share a shape')
    target_instance = target_state == INSTANCE_OCCUPIED
    changed = known & (candidate != baseline)
    candidate_correct = candidate == target_instance
    baseline_correct = baseline == target_instance
    improved = changed & candidate_correct & ~baseline_correct
    harmed = changed & ~candidate_correct & baseline_correct
    improved_count = int(np.count_nonzero(improved))
    harmed_count = int(np.count_nonzero(harmed))
    return {
        'changed_known_voxels': int(np.count_nonzero(changed)),
        'improved_voxels': improved_count,
        'harmed_voxels': harmed_count,
        'net_correct_voxels': improved_count - harmed_count,
    }


def _zero_counts() -> dict:
    return support_counts(
        np.zeros((1, 1, 1, 1), dtype=bool),
        np.zeros((1, 1, 1, 1), dtype=np.uint8),
        np.zeros((1, 1, 1, 1), dtype=bool),
        np.zeros((1, 1, 1), dtype=np.uint8),
        np.zeros((1, 1, 1), dtype=bool))


def _add_counts(destination: dict, source: dict):
    for key, value in source.items():
        destination[key] += int(value)


def _zero_correction_counts() -> dict:
    return {
        'changed_known_voxels': 0,
        'improved_voxels': 0,
        'harmed_voxels': 0,
        'net_correct_voxels': 0,
    }


def analyze(manifest: dict, sequences: dict, predictions: dict,
            split: str, score_thresholds, builder,
            time_tolerance_s: float) -> dict:
    references = _split_references(manifest, split)
    missing_sequence = sorted(set(references).difference(sequences))
    missing_prediction = sorted(set(references).difference(predictions))
    if missing_sequence or missing_prediction:
        raise KeyError({
            'missing_sequences': missing_sequence,
            'missing_predictions': missing_prediction,
        })
    aggregate = {
        str(threshold): _zero_counts() for threshold in score_thresholds
    }
    stationary_aggregate = {
        str(threshold): _zero_counts() for threshold in score_thresholds
    }
    motion_vs_stationary = {
        str(threshold): _zero_correction_counts()
        for threshold in score_thresholds
    }
    by_horizon = {
        str(threshold): [_zero_counts() for _ in range(4)]
        for threshold in score_thresholds
    }
    stationary_by_horizon = {
        str(threshold): [_zero_counts() for _ in range(4)]
        for threshold in score_thresholds
    }
    correction_by_horizon = {
        str(threshold): [_zero_correction_counts() for _ in range(4)]
        for threshold in score_thresholds
    }
    rows = []
    for reference in references:
        with np.load(sequences[reference], allow_pickle=False) as label:
            target_state = np.asarray(
                label['world_target_state_3d'][1:], dtype=np.uint8)
            target_valid = np.asarray(
                label['world_target_valid_3d'][1:], dtype=np.bool_)
            current_state = np.asarray(
                label['current_observation_state_3d'], dtype=np.uint8)
            current_valid = np.asarray(
                label['current_observation_valid_3d'], dtype=np.bool_)
            target_times = np.asarray(
                label['nominal_target_times_s'][1:], dtype=np.float32)
        if target_state.shape[0] != 4:
            raise ValueError(
                f'Reference {reference} does not have four future targets')
        row = {'reference_index': int(reference), 'thresholds': {}}
        with np.load(predictions[reference], allow_pickle=False) as prediction:
            for threshold in score_thresholds:
                support, metadata = motion_actor_support(
                    prediction, builder, target_times,
                    score_threshold=threshold,
                    time_tolerance_s=time_tolerance_s)
                stationary, stationary_metadata = motion_actor_support(
                    prediction, builder, target_times,
                    score_threshold=threshold,
                    time_tolerance_s=time_tolerance_s,
                    position_policy='stationary')
                counts = support_counts(
                    support, target_state, target_valid,
                    current_state, current_valid)
                _add_counts(aggregate[str(threshold)], counts)
                stationary_counts = support_counts(
                    stationary, target_state, target_valid,
                    current_state, current_valid)
                _add_counts(
                    stationary_aggregate[str(threshold)], stationary_counts)
                correction = support_correction_counts(
                    support, stationary, target_state, target_valid)
                _add_counts(
                    motion_vs_stationary[str(threshold)], correction)
                horizon_rows = []
                for horizon in range(4):
                    horizon_counts = support_counts(
                        support[horizon:horizon + 1],
                        target_state[horizon:horizon + 1],
                        target_valid[horizon:horizon + 1],
                        current_state, current_valid)
                    _add_counts(
                        by_horizon[str(threshold)][horizon], horizon_counts)
                    stationary_horizon_counts = support_counts(
                        stationary[horizon:horizon + 1],
                        target_state[horizon:horizon + 1],
                        target_valid[horizon:horizon + 1],
                        current_state, current_valid)
                    _add_counts(
                        stationary_by_horizon[str(threshold)][horizon],
                        stationary_horizon_counts)
                    horizon_correction = support_correction_counts(
                        support[horizon:horizon + 1],
                        stationary[horizon:horizon + 1],
                        target_state[horizon:horizon + 1],
                        target_valid[horizon:horizon + 1])
                    _add_counts(
                        correction_by_horizon[str(threshold)][horizon],
                        horizon_correction)
                    horizon_rows.append({
                        'motion': summarize_counts(horizon_counts),
                        'stationary': summarize_counts(
                            stationary_horizon_counts),
                        'motion_vs_stationary': horizon_correction,
                    })
                row['thresholds'][str(threshold)] = {
                    'metadata': metadata,
                    'stationary_metadata': stationary_metadata,
                    'motion': summarize_counts(counts),
                    'stationary': summarize_counts(stationary_counts),
                    'motion_vs_stationary': correction,
                    'by_horizon': horizon_rows,
                }
        rows.append(row)
    return {
        'schema_version': 1,
        'purpose': (
            'Validation-only feasibility audit of future 3D instance support '
            'from MotionHead trajectories and aligned current track boxes.'),
        'split': split,
        'reference_count': len(references),
        'score_thresholds': [float(value) for value in score_thresholds],
        'primary_score_threshold': 0.1,
        'motion_time_alignment': 'nominal 0.5-second steps',
        'aggregate': {
            key: summarize_counts(value) for key, value in aggregate.items()
        },
        'stationary_aggregate': {
            key: summarize_counts(value)
            for key, value in stationary_aggregate.items()
        },
        'motion_vs_stationary': motion_vs_stationary,
        'by_horizon': {
            key: [summarize_counts(row) for row in values]
            for key, values in by_horizon.items()
        },
        'stationary_by_horizon': {
            key: [summarize_counts(row) for row in values]
            for key, values in stationary_by_horizon.items()
        },
        'motion_vs_stationary_by_horizon': correction_by_horizon,
        'rows': rows,
    }


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
            'occworld_predictions_b21_motion_actor_validation_v1/'
            'validation/epoch_003'))
    parser.add_argument('--split', default='validation')
    parser.add_argument(
        '--score-thresholds', type=float, nargs='+',
        default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument('--time-tolerance-s', type=float, default=0.05)
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
            'kl_occworld_b21_motion_actor_support_validation_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if len(set(args.score_thresholds)) != len(args.score_thresholds):
        raise ValueError('Score thresholds must be unique')
    builder = RaycastDrivableBuilder(
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size)
    report = analyze(
        _load_manifest(args.manifest),
        _sequence_mapping(args.sequence_root),
        _prediction_mapping(args.prediction_root),
        split=args.split,
        score_thresholds=args.score_thresholds,
        builder=builder,
        time_tolerance_s=args.time_tolerance_s)
    report['inputs'] = {
        'manifest': str(args.manifest),
        'sequence_root': str(args.sequence_root),
        'prediction_root': str(args.prediction_root),
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
