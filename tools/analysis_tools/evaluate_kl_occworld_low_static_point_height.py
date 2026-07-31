#!/usr/bin/env python
"""Evaluate a point-height-based low-static-to-unknown GT candidate.

This tool is deliberately evidence-only. It rebuilds LiDAR evidence to mark
candidate voxels in memory and never overwrites sequence-label NPZ files.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    grouped_linear_percentile,
)
from tools.analysis_tools.audit_kl_occworld_low_static_stability import (
    _distribution,
    _grid_metadata,
    _sequence_paths,
    _transition_counts,
    _with_rates,
)
from tools.data_converter.generate_kl_occworld_labels import (
    FREE,
    INSTANCE_OCCUPIED,
    STATIC_OCCUPIED,
    UNKNOWN,
    MultiLidarOccLabelBuilder,
    _load_infos,
    _resolve_path,
    _xyz_to_zhw,
)
from tools.data_converter.generate_kl_occworld_temporal_labels import (
    _warp_mask_to_reference_xyz,
)


_WORKER_CONTEXT = {}


def _xy_to_hw(volume: np.ndarray) -> np.ndarray:
    """Convert builder-native XY arrays to image-aligned HW arrays."""
    return volume.T[::-1, :]


def point_height_downgrade_mask(
        direct_static: np.ndarray,
        point_p90_above_ground: np.ndarray,
        higher_static_support: np.ndarray,
        threshold_m: float) -> np.ndarray:
    """Return conservative source-frame static-to-unknown candidates."""
    if not (direct_static.shape == point_p90_above_ground.shape ==
            higher_static_support.shape):
        raise ValueError('Point-height candidate arrays must share one shape')
    if threshold_m <= 0:
        raise ValueError('Point-height threshold must be positive')
    return (
        np.asarray(direct_static, dtype=bool) &
        np.isfinite(point_p90_above_ground) &
        (point_p90_above_ground <= threshold_m) &
        ~np.asarray(higher_static_support, dtype=bool))


def apply_point_height_downgrade(
        world_low: np.ndarray,
        direct_low: np.ndarray,
        downgrade: np.ndarray) -> np.ndarray:
    """Apply the candidate after world composition without creating free GT."""
    if not (world_low.shape == direct_low.shape == downgrade.shape):
        raise ValueError('World, direct, and candidate arrays must match')
    candidate = np.asarray(world_low, dtype=np.uint8).copy()
    valid = (
        (candidate == STATIC_OCCUPIED) &
        (np.asarray(direct_low) == STATIC_OCCUPIED) &
        np.asarray(downgrade, dtype=bool))
    candidate[valid] = UNKNOWN
    return candidate


def _empty_transition_total() -> dict:
    return {
        'source_static_voxels': 0,
        'next_unknown_voxels': 0,
        'next_free_voxels': 0,
        'next_static_voxels': 0,
        'next_instance_voxels': 0,
    }


def _add_transition(destination: dict, source: dict) -> None:
    for key in destination:
        destination[key] += int(source[key])


def _empty_counts(horizon_count: int) -> dict:
    return {
        'all_world_known_voxels': 0,
        'low_world_static_voxels': 0,
        'low_direct_static_voxels': 0,
        'source_static_endpoint_voxels': 0,
        'source_p90_valid_voxels': 0,
        'source_p90_at_or_below_threshold_voxels': 0,
        'source_without_higher_static_support_voxels': 0,
        'source_weak_point_height_voxels': 0,
        'warped_weak_point_height_voxels': 0,
        'downgraded_voxels': 0,
        'downgraded_by_horizon': [0] * horizon_count,
        'downgraded_next_state_histogram': {
            'unknown': 0, 'free': 0, 'static': 0, 'instance': 0,
        },
    }


def _add_counts(destination: dict, source: dict) -> None:
    for key in (
            'all_world_known_voxels', 'low_world_static_voxels',
            'low_direct_static_voxels', 'source_static_endpoint_voxels',
            'source_p90_valid_voxels',
            'source_p90_at_or_below_threshold_voxels',
            'source_without_higher_static_support_voxels',
            'source_weak_point_height_voxels',
            'warped_weak_point_height_voxels', 'downgraded_voxels'):
        destination[key] += int(source[key])
    destination['downgraded_by_horizon'] = [
        left + right for left, right in zip(
            destination['downgraded_by_horizon'],
            source['downgraded_by_horizon'])]
    for key, value in source['downgraded_next_state_histogram'].items():
        destination['downgraded_next_state_histogram'][key] += int(value)


def _source_point_height_evidence(evidence: dict, builder,
                                  low_z_index: int,
                                  threshold_m: float) -> tuple:
    """Build one low-Z source mask from actual calibrated LiDAR points."""
    points = np.asarray(evidence['points'], dtype=np.float32)
    point_voxels = builder.builder.coord_to_index_floor(points[:, :3])
    valid = np.all(
        (point_voxels >= 0) &
        (point_voxels < builder.builder.occ_size[None, :]), axis=1)
    points = points[valid]
    point_voxels = point_voxels[valid]
    ground, _ = builder.builder.estimate_ground_height(
        points[:, :3], point_voxels)
    ground_at_point = ground[point_voxels[:, 0], point_voxels[:, 1]]
    valid_ground = np.isfinite(ground_at_point)
    point_voxels = point_voxels[valid_ground]
    relative_height = (
        points[valid_ground, 2] - ground_at_point[valid_ground])
    low_points = point_voxels[:, 2] == low_z_index
    cell_count = int(builder.builder.occ_size[0] * builder.builder.occ_size[1])
    flat_cell = (
        point_voxels[low_points, 0].astype(np.int64) *
        int(builder.builder.occ_size[1]) +
        point_voxels[low_points, 1].astype(np.int64))
    p90_xy, _ = grouped_linear_percentile(
        flat_cell, relative_height[low_points], 0.90, cell_count)
    p90_hw = _xy_to_hw(p90_xy.reshape(tuple(builder.builder.occ_size[:2])))
    static_3d = np.asarray(evidence['static_obstacle_3d'], dtype=bool)
    source_static = static_3d[low_z_index]
    higher_static = np.any(static_3d[low_z_index + 1:], axis=0)
    weak = point_height_downgrade_mask(
        source_static, p90_hw, higher_static, threshold_m)
    stats = {
        'source_static_endpoint_voxels': int(np.count_nonzero(source_static)),
        'source_p90_valid_voxels': int(np.count_nonzero(
            source_static & np.isfinite(p90_hw))),
        'source_p90_at_or_below_threshold_voxels': int(np.count_nonzero(
            source_static & np.isfinite(p90_hw) & (p90_hw <= threshold_m))),
        'source_without_higher_static_support_voxels': int(np.count_nonzero(
            source_static & ~higher_static)),
        'source_weak_point_height_voxels': int(np.count_nonzero(weak)),
        'source_p90_distribution_m': _distribution(
            p90_hw[source_static & np.isfinite(p90_hw)]),
    }
    source_mask = np.zeros_like(static_3d, dtype=bool)
    source_mask[low_z_index] = weak
    return source_mask, stats


def _init_worker(ann_file: str, pc_range: list, occ_size: list,
                 collision_z: list, threshold_m: float) -> None:
    """Load metadata and one reusable builder per worker process."""
    infos, metainfo = _load_infos(_resolve_path(Path(ann_file)))
    _WORKER_CONTEXT['infos'] = infos
    _WORKER_CONTEXT['builder'] = MultiLidarOccLabelBuilder(
        pc_range, (int(occ_size[1]), int(occ_size[0])), occ_size,
        target_frame=str(metainfo.get('lidar_coord_frame', 'FLU')),
        collision_z=collision_z)
    _WORKER_CONTEXT['threshold_m'] = float(threshold_m)


def _evaluate_reference(task: tuple) -> dict:
    """Evaluate all target horizons from one existing sequence archive."""
    reference, path_string, low_z_index, pc_range, occ_size = task
    infos = _WORKER_CONTEXT['infos']
    builder = _WORKER_CONTEXT['builder']
    threshold_m = _WORKER_CONTEXT['threshold_m']
    path = Path(path_string)
    with np.load(path, allow_pickle=False) as archive:
        world = np.asarray(archive['world_target_state_3d'], dtype=np.uint8)
        direct = np.asarray(
            archive['direct_observation_state_3d'], dtype=np.uint8)
        target_indices = np.asarray(archive['target_indices'], dtype=np.int64)
    if int(reference) != int(path.parent.name):
        raise ValueError(f'Reference/path mismatch for {path}')
    if world.shape != direct.shape:
        raise ValueError(f'World/direct shape mismatch for {path}')
    reference_pose = np.asarray(infos[reference]['ego2global'], dtype=np.float64)
    horizon_count = int(world.shape[0])
    candidate_low = world[:, low_z_index].copy()
    downgrade_by_horizon = []
    evidence_totals = _empty_counts(horizon_count)
    frame_rows = []

    for horizon, target_index in enumerate(target_indices):
        evidence = builder.build(
            infos[int(target_index)], diagnostics=False, return_points=True)
        source_mask, source_stats = _source_point_height_evidence(
            evidence, builder, low_z_index, threshold_m)
        target_pose = np.asarray(
            infos[int(target_index)]['ego2global'], dtype=np.float64)
        warped_xyz = _warp_mask_to_reference_xyz(
            source_mask, target_pose, reference_pose, pc_range, occ_size,
            radius_xy=0, radius_z=0)
        warped_mask = _xyz_to_zhw(warped_xyz)[low_z_index]
        downgrade = (
            (world[horizon, low_z_index] == STATIC_OCCUPIED) &
            (direct[horizon, low_z_index] == STATIC_OCCUPIED) &
            warped_mask)
        candidate_low[horizon] = apply_point_height_downgrade(
            world[horizon, low_z_index], direct[horizon, low_z_index],
            downgrade)
        downgrade_by_horizon.append(downgrade)
        for key in (
                'source_static_endpoint_voxels',
                'source_p90_valid_voxels',
                'source_p90_at_or_below_threshold_voxels',
                'source_without_higher_static_support_voxels',
                'source_weak_point_height_voxels'):
            evidence_totals[key] += int(source_stats[key])
        evidence_totals['warped_weak_point_height_voxels'] += int(
            np.count_nonzero(warped_mask))
        evidence_totals['downgraded_voxels'] += int(
            np.count_nonzero(downgrade))
        evidence_totals['downgraded_by_horizon'][horizon] = int(
            np.count_nonzero(downgrade))
        frame_rows.append({
            'reference_index': int(reference),
            'source_horizon': int(horizon),
            'target_index': int(target_index),
            'source_weak_point_height_voxels': int(
                source_stats['source_weak_point_height_voxels']),
            'warped_weak_point_height_voxels': int(
                np.count_nonzero(warped_mask)),
            'downgraded_voxels': int(np.count_nonzero(downgrade)),
            'source_p90_distribution_m': source_stats[
                'source_p90_distribution_m'],
        })

    evidence_totals['all_world_known_voxels'] = int(np.count_nonzero(world))
    evidence_totals['low_world_static_voxels'] = int(np.count_nonzero(
        world[:, low_z_index] == STATIC_OCCUPIED))
    evidence_totals['low_direct_static_voxels'] = int(np.count_nonzero(
        direct[:, low_z_index] == STATIC_OCCUPIED))
    original_transition = _empty_transition_total()
    candidate_transition = _empty_transition_total()
    for horizon in range(horizon_count - 1):
        source_static = world[horizon, low_z_index] == STATIC_OCCUPIED
        candidate_static = candidate_low[horizon] == STATIC_OCCUPIED
        next_original = world[horizon + 1, low_z_index]
        next_candidate = candidate_low[horizon + 1]
        _add_transition(original_transition, _transition_counts(
            source_static, next_original))
        _add_transition(candidate_transition, _transition_counts(
            candidate_static, next_candidate))
        removed_source = downgrade_by_horizon[horizon]
        for state_value, name in enumerate(
                ('unknown', 'free', 'static', 'instance')):
            evidence_totals['downgraded_next_state_histogram'][name] += int(
                np.count_nonzero(removed_source & (next_original == state_value)))
        frame_rows[horizon]['next_state_of_downgraded'] = {
            name: int(np.count_nonzero(
                removed_source & (next_original == state_value)))
            for state_value, name in enumerate(
                ('unknown', 'free', 'static', 'instance'))}
    frame_rows[-1]['next_state_of_downgraded'] = None
    return {
        'reference_index': int(reference),
        'counts': evidence_totals,
        'original_transition': original_transition,
        'candidate_transition': candidate_transition,
        'frame_rows': frame_rows,
    }


def evaluate_labels(sequence_root: Path, ann_file: Path, workers: int,
                    threshold_m: float, top_count: int) -> dict:
    sequences = _sequence_paths(sequence_root)
    first_path = next(iter(sequences.values()))
    with np.load(first_path, allow_pickle=False) as archive:
        pc_range = np.asarray(archive['pc_range'], dtype=np.float64)
        occ_size = np.asarray(archive['occ_size'], dtype=np.int64)
        collision_z = np.asarray(archive['collision_z'], dtype=np.float64)
        horizon_count = int(archive['world_target_state_3d'].shape[0])
    grid = _grid_metadata(pc_range, occ_size, collision_z)
    low_z_index = int(grid['low_static_z_index'])
    totals = _empty_counts(horizon_count)
    original_transition = _empty_transition_total()
    candidate_transition = _empty_transition_total()
    frame_rows = []
    tasks = [
        (reference, str(path), low_z_index, pc_range.tolist(), occ_size.tolist())
        for reference, path in sequences.items()]
    worker_count = min(int(workers), len(tasks))
    context = multiprocessing.get_context('fork')
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context,
            initializer=_init_worker,
            initargs=(str(ann_file), pc_range.tolist(), occ_size.tolist(),
                      collision_z.tolist(), threshold_m)) as executor:
        for position, result in enumerate(executor.map(
                _evaluate_reference, tasks, chunksize=1), start=1):
            _add_counts(totals, result['counts'])
            _add_transition(original_transition, result['original_transition'])
            _add_transition(candidate_transition, result['candidate_transition'])
            frame_rows.extend(result['frame_rows'])
            if position % 25 == 0 or position == len(tasks):
                print(
                    f'[labels {position}/{len(tasks)}] '
                    f'reference={result["reference_index"]}', flush=True)
    original_free = original_transition['next_free_voxels']
    candidate_free = candidate_transition['next_free_voxels']
    totals['candidate_all_world_known_voxels'] = (
        totals['all_world_known_voxels'] - totals['downgraded_voxels'])
    totals['candidate_low_world_static_voxels'] = (
        totals['low_world_static_voxels'] - totals['downgraded_voxels'])
    totals['downgraded_fraction_of_low_static'] = (
        totals['downgraded_voxels'] / totals['low_world_static_voxels']
        if totals['low_world_static_voxels'] else None)
    totals['downgraded_fraction_of_all_known'] = (
        totals['downgraded_voxels'] / totals['all_world_known_voxels']
        if totals['all_world_known_voxels'] else None)
    totals['downgraded_next_state_evaluated_voxels'] = int(sum(
        totals['downgraded_next_state_histogram'].values()))
    totals['downgraded_next_state_fraction'] = {
        key: (
            value / totals['downgraded_next_state_evaluated_voxels']
            if totals['downgraded_next_state_evaluated_voxels'] else None)
        for key, value in totals['downgraded_next_state_histogram'].items()}
    return {
        'sequence_root': str(sequence_root),
        'annotation_file': str(ann_file),
        'reference_count': len(sequences),
        'target_frame_count': len(frame_rows),
        'workers': worker_count,
        'grid': {
            key: value for key, value in grid.items()
            if key not in ('x_centers', 'row_y_centers', 'quadrant_masks')},
        'rule_counts': totals,
        'original_transition': _with_rates(original_transition),
        'candidate_transition': _with_rates(candidate_transition),
        'hard_static_to_free_reduction_voxels': original_free - candidate_free,
        'hard_static_to_free_reduction_fraction': (
            (original_free - candidate_free) / original_free
            if original_free else None),
        'top_frames_by_downgraded_voxels': sorted(
            frame_rows,
            key=lambda row: (
                row['downgraded_voxels'],
                row['source_weak_point_height_voxels']),
            reverse=True)[:top_count],
        'top_frames_by_downgraded_static_next': sorted(
            [
                row for row in frame_rows
                if row['next_state_of_downgraded'] is not None
            ],
            key=lambda row: (
                row['next_state_of_downgraded']['static'],
                row['downgraded_voxels']),
            reverse=True)[:top_count],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_full_train_v1'))
    parser.add_argument('--ann-file', type=Path,
                        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--point-height-threshold-m', type=float,
                        default=0.55)
    parser.add_argument('--top-count', type=int, default=100)
    parser.add_argument(
        '--out-json', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_low_static_point_height_ablation_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be at least one')
    result = evaluate_labels(
        args.sequence_root, args.ann_file, args.workers,
        args.point_height_threshold_m, args.top_count)
    output = {
        'schema_version': 1,
        'name': 'KL OccWorld low-static actual-point-height ablation',
        'analysis_type': 'offline_candidate_mask_no_label_write',
        'rule': {
            'target_z_index': result['grid']['low_static_z_index'],
            'target_z_center_m': result['grid']['low_static_z_center_m'],
            'source_must_be_direct_static': True,
            'actual_point_height_statistic': 'P90',
            'point_height_threshold_m': args.point_height_threshold_m,
            'requires_no_higher_static_support': True,
            'replacement_state': 'unknown',
            'uses_future_frames_for_rule': False,
            'future_frames_used_only_for_evaluation': True,
            'formal_labels_modified': False,
        },
        'labels': result,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open('w') as destination:
        json.dump(output, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    print(json.dumps({
        'out_json': str(args.out_json),
        'rule': output['rule'],
        'rule_counts': result['rule_counts'],
        'hard_static_to_free_reduction_fraction': result[
            'hard_static_to_free_reduction_fraction'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
