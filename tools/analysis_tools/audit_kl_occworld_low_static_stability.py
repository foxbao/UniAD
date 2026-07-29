#!/usr/bin/env python
"""Audit temporal instability of ground-level static OccWorld voxels."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


STATE_NAMES = ('unknown', 'free', 'static', 'instance')
QUADRANT_NAMES = (
    'front_left', 'front_right', 'rear_left', 'rear_right')


def _empty_transition_counts():
    return {
        'source_static_voxels': 0,
        **{f'next_{name}_voxels': 0 for name in STATE_NAMES},
    }


def _transition_counts(source_static: np.ndarray,
                       next_state: np.ndarray) -> dict:
    if source_static.shape != next_state.shape:
        raise ValueError('Transition arrays must have the same shape')
    result = _empty_transition_counts()
    result['source_static_voxels'] = int(np.count_nonzero(source_static))
    for value, name in enumerate(STATE_NAMES):
        result[f'next_{name}_voxels'] = int(np.count_nonzero(
            source_static & (next_state == value)))
    return result


def _add_counts(destination: dict, source: dict) -> None:
    for key in destination:
        destination[key] += int(source[key])


def _with_rates(counts: dict) -> dict:
    result = dict(counts)
    total = int(counts['source_static_voxels'])
    for name in STATE_NAMES:
        count = int(counts[f'next_{name}_voxels'])
        result[f'next_{name}_fraction'] = (
            count / total if total else None)
    result['next_nonstatic_voxels'] = (
        int(counts['next_unknown_voxels']) +
        int(counts['next_free_voxels']) +
        int(counts['next_instance_voxels']))
    result['next_nonstatic_fraction'] = (
        result['next_nonstatic_voxels'] / total if total else None)
    return result


def _distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            'count': 0, 'mean': None, 'p10': None, 'median': None,
            'p90': None, 'p95': None, 'max': None}
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'p10': float(np.percentile(values, 10)),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
        'p95': float(np.percentile(values, 95)),
        'max': float(values.max()),
    }


def _pose_delta_metrics(first: np.ndarray, second: np.ndarray) -> dict:
    relative = np.linalg.inv(first) @ second
    rotation = relative[:3, :3]
    roll = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    pitch = math.degrees(math.atan2(
        -rotation[2, 0],
        math.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)))
    yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    translation = relative[:3, 3]
    return {
        'roll_deg': roll,
        'pitch_deg': pitch,
        'yaw_deg': yaw,
        'tilt_deg': math.hypot(roll, pitch),
        'translation_xy_m': float(np.linalg.norm(translation[:2])),
        'translation_z_m': float(translation[2]),
    }


def _sequence_paths(root: Path) -> dict:
    mapping = {}
    for path in sorted(root.glob('*/*__occworld_sequence.npz')):
        try:
            reference = int(path.parent.name)
        except ValueError as error:
            raise ValueError(
                f'Sequence parent is not a reference index: {path}') from error
        if reference in mapping:
            raise ValueError(f'Duplicate sequence reference {reference}')
        mapping[reference] = path
    if not mapping:
        raise FileNotFoundError(f'No sequence labels under {root}')
    return mapping


def _prediction_paths(root: Path) -> dict:
    mapping = {}
    for path in sorted(root.glob('*/*__occworld_prediction.npz')):
        try:
            reference = int(path.parent.name)
        except ValueError as error:
            raise ValueError(
                f'Prediction parent is not a reference index: {path}') \
                from error
        if reference in mapping:
            raise ValueError(f'Duplicate prediction reference {reference}')
        mapping[reference] = path
    if not mapping:
        raise FileNotFoundError(f'No predictions under {root}')
    return mapping


def _grid_metadata(pc_range: np.ndarray, occ_size: np.ndarray,
                   collision_z: np.ndarray) -> dict:
    voxel_size = (
        (pc_range[3:] - pc_range[:3]) / occ_size.astype(np.float64))
    z_centers = (
        pc_range[2] +
        (np.arange(occ_size[2], dtype=np.float64) + 0.5) * voxel_size[2])
    below_collision = np.flatnonzero(z_centers < float(collision_z[0]))
    if not below_collision.size:
        raise ValueError('No voxel layer lies below the collision band')
    low_index = int(below_collision[np.argmin(
        np.abs(z_centers[below_collision]))])
    x_centers = (
        pc_range[0] +
        (np.arange(occ_size[0], dtype=np.float64) + 0.5) * voxel_size[0])
    y_centers = (
        pc_range[1] +
        (np.arange(occ_size[1], dtype=np.float64) + 0.5) * voxel_size[1])
    row_y_centers = y_centers[::-1]
    quadrant_masks = {
        'front_left': (
            (row_y_centers[:, None] >= 0) &
            (x_centers[None, :] >= 0)),
        'front_right': (
            (row_y_centers[:, None] < 0) &
            (x_centers[None, :] >= 0)),
        'rear_left': (
            (row_y_centers[:, None] >= 0) &
            (x_centers[None, :] < 0)),
        'rear_right': (
            (row_y_centers[:, None] < 0) &
            (x_centers[None, :] < 0)),
    }
    return {
        'pc_range': pc_range.tolist(),
        'occ_size': occ_size.tolist(),
        'collision_z': collision_z.tolist(),
        'voxel_size': voxel_size.tolist(),
        'z_centers': z_centers.tolist(),
        'low_static_z_index': low_index,
        'low_static_z_center_m': float(z_centers[low_index]),
        'x_centers': x_centers,
        'row_y_centers': row_y_centers,
        'quadrant_masks': quadrant_masks,
    }


def _connected_component_rows(
        source_static: np.ndarray,
        next_state: np.ndarray,
        source_completion: np.ndarray,
        x_centers: np.ndarray,
        row_y_centers: np.ndarray,
        reference: int,
        source_horizon: int,
        min_cells: int) -> list:
    component_count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            source_static.astype(np.uint8), connectivity=8))
    rows = []
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < min_cells:
            continue
        selection = labels == component
        transition = _with_rates(_transition_counts(
            selection, next_state))
        selected_completion = source_completion[selection]
        component_rows, component_columns = np.where(selection)
        rows.append({
            'reference_index': reference,
            'source_horizon': source_horizon,
            'target_horizon': source_horizon + 1,
            'area_cells': area,
            'area_m2': None,
            'bbox_x_m': [
                float(x_centers[component_columns].min()),
                float(x_centers[component_columns].max())],
            'bbox_y_m': [
                float(row_y_centers[component_rows].min()),
                float(row_y_centers[component_rows].max())],
            'centroid_column_row': centroids[component].tolist(),
            'source_direct_cells': int(np.count_nonzero(
                selected_completion == 1)),
            'source_future_static_fill_cells': int(np.count_nonzero(
                selected_completion == 3)),
            **transition,
        })
    return rows


def _pearson(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _pose_buckets(pair_rows: list) -> list:
    edges = (0.0, 0.25, 0.5, 1.0, 2.0, float('inf'))
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = [
            row for row in pair_rows
            if lower <= row['pose']['tilt_deg'] < upper]
        counts = _empty_transition_counts()
        for row in selected:
            _add_counts(counts, row['transition_counts'])
        rows.append({
            'tilt_deg_interval': [
                lower, None if math.isinf(upper) else upper],
            'pair_count': len(selected),
            **_with_rates(counts),
        })
    return rows


def audit_sequences(sequence_root: Path, min_component_cells: int,
                    top_count: int) -> tuple:
    paths = _sequence_paths(sequence_root)
    by_z = None
    low_total = _empty_transition_counts()
    low_direct_next = _empty_transition_counts()
    low_source = {
        'direct_voxels': 0,
        'future_free_fill_voxels': 0,
        'future_static_fill_voxels': 0,
        'other_voxels': 0,
    }
    quadrants = {
        name: _empty_transition_counts() for name in QUADRANT_NAMES}
    pair_rows = []
    component_rows = []
    grid = None
    voxel_area = None

    for position, (reference, path) in enumerate(paths.items(), start=1):
        with np.load(path, allow_pickle=False) as archive:
            exported_reference = int(archive['reference_index'])
            if exported_reference != reference:
                raise ValueError(f'Reference mismatch in {path}')
            state = np.asarray(
                archive['world_target_state_3d'], dtype=np.uint8)
            direct = np.asarray(
                archive['direct_observation_state_3d'], dtype=np.uint8)
            completion = np.asarray(
                archive['completion_source_3d'], dtype=np.uint8)
            transforms = np.asarray(
                archive['target_to_reference'], dtype=np.float64)
            pc_range = np.asarray(archive['pc_range'], dtype=np.float64)
            occ_size = np.asarray(archive['occ_size'], dtype=np.int64)
            collision_z = np.asarray(
                archive['collision_z'], dtype=np.float64)
        if state.shape != direct.shape or state.shape != completion.shape:
            raise ValueError(f'Sequence state shapes differ for {reference}')
        if grid is None:
            grid = _grid_metadata(pc_range, occ_size, collision_z)
            by_z = [
                _empty_transition_counts()
                for _ in range(state.shape[1])]
            voxel_area = float(
                grid['voxel_size'][0] * grid['voxel_size'][1])
        else:
            if (not np.allclose(pc_range, grid['pc_range']) or
                    not np.array_equal(occ_size, grid['occ_size'])):
                raise ValueError('Sequence grids are not identical')
        low_index = grid['low_static_z_index']

        for source_horizon in range(state.shape[0] - 1):
            for z_index in range(state.shape[1]):
                counts = _transition_counts(
                    state[source_horizon, z_index] == 2,
                    state[source_horizon + 1, z_index])
                _add_counts(by_z[z_index], counts)

            source_static = state[source_horizon, low_index] == 2
            next_state = state[source_horizon + 1, low_index]
            transition = _transition_counts(source_static, next_state)
            _add_counts(low_total, transition)
            direct_transition = _transition_counts(
                source_static, direct[source_horizon + 1, low_index])
            _add_counts(low_direct_next, direct_transition)
            selected_completion = completion[
                source_horizon, low_index][source_static]
            low_source['direct_voxels'] += int(np.count_nonzero(
                selected_completion == 1))
            low_source['future_free_fill_voxels'] += int(np.count_nonzero(
                selected_completion == 2))
            low_source['future_static_fill_voxels'] += int(np.count_nonzero(
                selected_completion == 3))
            low_source['other_voxels'] += int(np.count_nonzero(
                selected_completion == 0))
            for name, quadrant in grid['quadrant_masks'].items():
                _add_counts(quadrants[name], _transition_counts(
                    source_static & quadrant, next_state))
            pose = _pose_delta_metrics(
                transforms[source_horizon],
                transforms[source_horizon + 1])
            pair_rows.append({
                'reference_index': reference,
                'source_horizon': source_horizon,
                'target_horizon': source_horizon + 1,
                'pose': pose,
                'transition_counts': transition,
                **_with_rates(transition),
            })
            rows = _connected_component_rows(
                source_static, next_state,
                completion[source_horizon, low_index],
                grid['x_centers'], grid['row_y_centers'],
                reference, source_horizon, min_component_cells)
            for row in rows:
                row['area_m2'] = row['area_cells'] * voxel_area
            component_rows.extend(rows)

        if position % 100 == 0 or position == len(paths):
            print(f'[GT {position}/{len(paths)}] reference={reference}')

    pair_with_static = [
        row for row in pair_rows if row['source_static_voxels']]
    pair_free_fractions = [
        row['next_free_fraction'] for row in pair_with_static]
    pair_nonstatic_fractions = [
        row['next_nonstatic_fraction'] for row in pair_with_static]
    pair_tilts = [row['pose']['tilt_deg'] for row in pair_with_static]
    top_pairs = sorted(
        pair_with_static,
        key=lambda row: (
            row['next_free_voxels'], row['next_free_fraction'],
            row['source_static_voxels']),
        reverse=True)[:top_count]
    top_components = sorted(
        component_rows,
        key=lambda row: (
            row['next_free_voxels'], row['next_free_fraction'],
            row['area_cells']),
        reverse=True)[:top_count]
    by_z_rows = []
    for z_index, counts in enumerate(by_z):
        by_z_rows.append({
            'z_index': z_index,
            'z_center_m': grid['z_centers'][z_index],
            **_with_rates(counts),
        })
    component_free_fractions = [
        row['next_free_fraction'] for row in component_rows]
    component_nonstatic_fractions = [
        row['next_nonstatic_fraction'] for row in component_rows]
    result = {
        'sequence_root': str(sequence_root),
        'reference_count': len(paths),
        'adjacent_pair_count': len(pair_rows),
        'sampling_note': (
            'Counts are sequence-window weighted; overlapping target frames '
            'from different references may appear more than once.'),
        'grid': {
            key: value for key, value in grid.items()
            if key not in ('x_centers', 'row_y_centers', 'quadrant_masks')
        },
        'by_z_layer': by_z_rows,
        'low_static': {
            'world_transition': _with_rates(low_total),
            'next_direct_observation_transition': _with_rates(
                low_direct_next),
            'source_completion': low_source,
            'by_quadrant': {
                name: _with_rates(counts)
                for name, counts in quadrants.items()
            },
            'pair_free_fraction_distribution': _distribution(
                pair_free_fractions),
            'pair_nonstatic_fraction_distribution': _distribution(
                pair_nonstatic_fractions),
            'pose_tilt_deg_distribution': _distribution(pair_tilts),
            'pearson_pose_tilt_vs_pair_free_fraction': _pearson(
                pair_tilts, pair_free_fractions),
            'pearson_pose_tilt_vs_pair_nonstatic_fraction': _pearson(
                pair_tilts, pair_nonstatic_fractions),
            'pose_tilt_buckets': _pose_buckets(pair_with_static),
            'component_min_cells': min_component_cells,
            'component_count': len(component_rows),
            'component_area_cells_distribution': _distribution([
                row['area_cells'] for row in component_rows]),
            'component_free_fraction_distribution': _distribution(
                component_free_fractions),
            'component_nonstatic_fraction_distribution': _distribution(
                component_nonstatic_fractions),
            'top_pairs_by_next_free_voxels': top_pairs,
            'top_components_by_next_free_voxels': top_components,
        },
    }
    return result, paths


def audit_predictions(prediction_root: Path, sequences: dict,
                      low_index: int, expected_epoch: int,
                      top_count: int) -> dict:
    predictions = _prediction_paths(prediction_root)
    missing = sorted(set(predictions).difference(sequences))
    if missing:
        raise ValueError(f'Predictions lack sequence labels: {missing[:5]}')
    totals = {
        'source_gt_static_voxels': 0,
        'source_gt_static_next_known_voxels': 0,
        'source_gt_static_next_free_voxels': 0,
        'source_gt_static_next_static_voxels': 0,
        'raw_static_on_next_free_voxels': 0,
        'final_static_on_next_free_voxels': 0,
        'raw_errors_on_next_known_voxels': 0,
        'final_errors_on_next_known_voxels': 0,
        'event_candidate_on_source_static_voxels': 0,
        'raw_final_changed_on_source_static_voxels': 0,
        'raw_final_changed_all_low_layer_voxels': 0,
    }
    rows = []
    for position, (reference, prediction_path) in enumerate(
            predictions.items(), start=1):
        with np.load(sequences[reference], allow_pickle=False) as archive:
            target = np.asarray(
                archive['world_target_state_3d'], dtype=np.uint8)
        with np.load(prediction_path, allow_pickle=False) as archive:
            if int(archive['reference_index']) != reference:
                raise ValueError(
                    f'Prediction reference mismatch for {reference}')
            if int(archive['checkpoint_epoch']) != expected_epoch:
                raise ValueError(
                    f'Prediction epoch mismatch for {reference}')
            raw = np.asarray(
                archive['raw_world_pred_class_3d'], dtype=np.uint8) + 1
            final = np.asarray(
                archive['world_pred_class_3d'], dtype=np.uint8) + 1
            event = np.asarray(
                archive['event_candidate_mask_3d'], dtype=np.bool_)
        source = target[0, low_index] == 2
        next_target = target[1, low_index]
        next_known = next_target != 0
        next_free = next_target == 1
        next_static = next_target == 2
        raw_next = raw[1, low_index]
        final_next = final[1, low_index]
        changed_source = source & (raw_next != final_next)
        row = {
            'reference_index': reference,
            'source_gt_static_voxels': int(np.count_nonzero(source)),
            'source_gt_static_next_known_voxels': int(np.count_nonzero(
                source & next_known)),
            'source_gt_static_next_free_voxels': int(np.count_nonzero(
                source & next_free)),
            'source_gt_static_next_static_voxels': int(np.count_nonzero(
                source & next_static)),
            'raw_static_on_next_free_voxels': int(np.count_nonzero(
                source & next_free & (raw_next == 2))),
            'final_static_on_next_free_voxels': int(np.count_nonzero(
                source & next_free & (final_next == 2))),
            'raw_errors_on_next_known_voxels': int(np.count_nonzero(
                source & next_known & (raw_next != next_target))),
            'final_errors_on_next_known_voxels': int(np.count_nonzero(
                source & next_known & (final_next != next_target))),
            'event_candidate_on_source_static_voxels': int(np.count_nonzero(
                source & event[0, low_index])),
            'raw_final_changed_on_source_static_voxels': int(
                np.count_nonzero(changed_source)),
            'raw_final_changed_all_low_layer_voxels': int(np.count_nonzero(
                raw_next != final_next)),
        }
        for key in totals:
            totals[key] += row[key]
        rows.append(row)
        if position % 100 == 0 or position == len(predictions):
            print(
                f'[B24 {position}/{len(predictions)}] '
                f'reference={reference}')
    source_next_free = totals['source_gt_static_next_free_voxels']
    source_next_known = totals['source_gt_static_next_known_voxels']
    result = {
        'prediction_root': str(prediction_root),
        'prediction_count': len(predictions),
        'checkpoint_epoch': expected_epoch,
        **totals,
        'raw_static_fraction_on_gt_static_to_free': (
            totals['raw_static_on_next_free_voxels'] / source_next_free
            if source_next_free else None),
        'final_static_fraction_on_gt_static_to_free': (
            totals['final_static_on_next_free_voxels'] / source_next_free
            if source_next_free else None),
        'raw_error_fraction_on_next_known': (
            totals['raw_errors_on_next_known_voxels'] / source_next_known
            if source_next_known else None),
        'final_error_fraction_on_next_known': (
            totals['final_errors_on_next_known_voxels'] / source_next_known
            if source_next_known else None),
        'event_candidate_fraction_on_source_gt_static': (
            totals['event_candidate_on_source_static_voxels'] /
            totals['source_gt_static_voxels']
            if totals['source_gt_static_voxels'] else None),
        'raw_final_changed_fraction_on_source_gt_static': (
            totals['raw_final_changed_on_source_static_voxels'] /
            totals['source_gt_static_voxels']
            if totals['source_gt_static_voxels'] else None),
        'top_references_by_final_static_on_next_free': sorted(
            rows,
            key=lambda row: (
                row['final_static_on_next_free_voxels'],
                row['source_gt_static_next_free_voxels']),
            reverse=True)[:top_count],
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_full_train_v1'))
    parser.add_argument('--prediction-root', type=Path)
    parser.add_argument('--expected-epoch', type=int, default=3)
    parser.add_argument('--min-component-cells', type=int, default=8)
    parser.add_argument('--top-count', type=int, default=100)
    parser.add_argument(
        '--out-json', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_low_static_stability_full_train_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_component_cells < 1 or args.top_count < 1:
        raise ValueError('Component and top-count limits must be positive')
    gt, sequences = audit_sequences(
        args.sequence_root, args.min_component_cells, args.top_count)
    result = {
        'schema_version': 1,
        'name': 'KL OccWorld ground-level static temporal stability audit',
        'gt': gt,
        'b24_internal_dev': None,
    }
    if args.prediction_root is not None:
        result['b24_internal_dev'] = audit_predictions(
            args.prediction_root, sequences,
            gt['grid']['low_static_z_index'],
            args.expected_epoch, args.top_count)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open('w') as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    b24 = result['b24_internal_dev']
    if b24 is not None:
        b24 = {
            key: value for key, value in b24.items()
            if key != 'top_references_by_final_static_on_next_free'
        }
    print(json.dumps({
        'out_json': str(args.out_json),
        'gt_reference_count': gt['reference_count'],
        'gt_adjacent_pair_count': gt['adjacent_pair_count'],
        'low_static_world_transition': gt['low_static'][
            'world_transition'],
        'b24_internal_dev': b24,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
