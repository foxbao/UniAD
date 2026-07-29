#!/usr/bin/env python
"""Evaluate an offline low-static/free conflict-to-unknown GT rule."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools.analysis_tools.audit_kl_occworld_low_static_stability import (
    _distribution,
    _grid_metadata,
    _prediction_paths,
    _sequence_paths,
    _transition_counts,
    _with_rates,
)


def apply_low_static_conflict_downgrade(
        world_state: np.ndarray,
        direct_state: np.ndarray,
        future_free_count: np.ndarray,
        low_z_index: int,
        repeated_free_count: int = 2) -> tuple:
    """Downgrade direct low static with repeated future free to unknown."""
    if not (world_state.shape == direct_state.shape ==
            future_free_count.shape):
        raise ValueError('World, direct and future-free arrays must match')
    if world_state.ndim != 4:
        raise ValueError('Expected [T,Z,H,W] sequence arrays')
    if not 0 <= low_z_index < world_state.shape[1]:
        raise IndexError('Low-static z index is outside the sequence')
    if repeated_free_count < 2:
        raise ValueError('Repeated-free evidence must require at least 2')
    candidate_low = np.asarray(
        world_state[:, low_z_index], dtype=np.uint8).copy()
    downgrade = (
        (direct_state[:, low_z_index] == 2) &
        (world_state[:, low_z_index] == 2) &
        (future_free_count[:, low_z_index] >= repeated_free_count))
    candidate_low[downgrade] = 0
    return candidate_low, downgrade


def _empty_rule_counts(horizon_count: int) -> dict:
    return {
        'all_world_known_voxels': 0,
        'all_world_static_voxels': 0,
        'low_world_known_voxels': 0,
        'low_world_static_voxels': 0,
        'low_direct_static_voxels': 0,
        'downgraded_voxels': 0,
        'downgraded_by_horizon': [0] * horizon_count,
        'downgraded_future_free_count_histogram': {
            str(value): 0 for value in range(6)},
    }


def _add_rule_counts(destination: dict, source: dict) -> None:
    for key in (
            'all_world_known_voxels', 'all_world_static_voxels',
            'low_world_known_voxels', 'low_world_static_voxels',
            'low_direct_static_voxels', 'downgraded_voxels'):
        destination[key] += int(source[key])
    destination['downgraded_by_horizon'] = [
        left + right for left, right in zip(
            destination['downgraded_by_horizon'],
            source['downgraded_by_horizon'])]
    for key, value in source[
            'downgraded_future_free_count_histogram'].items():
        destination['downgraded_future_free_count_histogram'][key] += int(
            value)


def _sequence_rule_counts(world, direct, future_free, low_index,
                          downgrade) -> dict:
    histogram = {
        str(value): int(np.count_nonzero(
            downgrade & (future_free[:, low_index] == value)))
        for value in range(6)}
    return {
        'all_world_known_voxels': int(np.count_nonzero(world)),
        'all_world_static_voxels': int(np.count_nonzero(world == 2)),
        'low_world_known_voxels': int(np.count_nonzero(
            world[:, low_index])),
        'low_world_static_voxels': int(np.count_nonzero(
            world[:, low_index] == 2)),
        'low_direct_static_voxels': int(np.count_nonzero(
            direct[:, low_index] == 2)),
        'downgraded_voxels': int(np.count_nonzero(downgrade)),
        'downgraded_by_horizon': [
            int(np.count_nonzero(mask)) for mask in downgrade],
        'downgraded_future_free_count_histogram': histogram,
    }


def _empty_transition_total():
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


def _component_rows(source_static, original_next, candidate_source,
                    candidate_next, downgrade_source,
                    x_centers, row_y_centers, reference,
                    source_horizon, min_cells, voxel_area):
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        source_static.astype(np.uint8), connectivity=8)
    rows = []
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < min_cells:
            continue
        selection = labels == component
        component_rows, component_columns = np.where(selection)
        original_transition = _with_rates(_transition_counts(
            selection, original_next))
        candidate_transition = _with_rates(_transition_counts(
            selection & candidate_source, candidate_next))
        downgraded = int(np.count_nonzero(selection & downgrade_source))
        rows.append({
            'reference_index': reference,
            'source_horizon': source_horizon,
            'target_horizon': source_horizon + 1,
            'area_cells': area,
            'area_m2': area * voxel_area,
            'bbox_x_m': [
                float(x_centers[component_columns].min()),
                float(x_centers[component_columns].max())],
            'bbox_y_m': [
                float(row_y_centers[component_rows].min()),
                float(row_y_centers[component_rows].max())],
            'downgraded_cells': downgraded,
            'downgraded_fraction': downgraded / area,
            'remaining_static_source_cells': int(np.count_nonzero(
                selection & candidate_source)),
            'original_transition': original_transition,
            'candidate_transition': candidate_transition,
        })
    return rows


def evaluate_labels(sequence_root: Path, repeated_free_count: int,
                    min_component_cells: int, top_count: int) -> tuple:
    sequences = _sequence_paths(sequence_root)
    grid = None
    rule_counts = None
    original_transition = _empty_transition_total()
    candidate_transition = _empty_transition_total()
    reference_rows = []
    component_rows = []

    for position, (reference, path) in enumerate(sequences.items(), start=1):
        with np.load(path, allow_pickle=False) as archive:
            world = np.asarray(
                archive['world_target_state_3d'], dtype=np.uint8)
            direct = np.asarray(
                archive['direct_observation_state_3d'], dtype=np.uint8)
            future_free = np.asarray(
                archive['future_free_count_3d'], dtype=np.uint8)
            pc_range = np.asarray(archive['pc_range'], dtype=np.float64)
            occ_size = np.asarray(archive['occ_size'], dtype=np.int64)
            collision_z = np.asarray(
                archive['collision_z'], dtype=np.float64)
        if grid is None:
            grid = _grid_metadata(pc_range, occ_size, collision_z)
            rule_counts = _empty_rule_counts(world.shape[0])
            voxel_area = float(
                grid['voxel_size'][0] * grid['voxel_size'][1])
        low_index = grid['low_static_z_index']
        candidate_low, downgrade = apply_low_static_conflict_downgrade(
            world, direct, future_free, low_index, repeated_free_count)
        local_counts = _sequence_rule_counts(
            world, direct, future_free, low_index, downgrade)
        _add_rule_counts(rule_counts, local_counts)
        local_original = _empty_transition_total()
        local_candidate = _empty_transition_total()
        for source_horizon in range(world.shape[0] - 1):
            source_static = world[source_horizon, low_index] == 2
            original_counts = _transition_counts(
                source_static, world[source_horizon + 1, low_index])
            candidate_source = candidate_low[source_horizon] == 2
            candidate_counts = _transition_counts(
                candidate_source, candidate_low[source_horizon + 1])
            _add_transition(original_transition, original_counts)
            _add_transition(candidate_transition, candidate_counts)
            _add_transition(local_original, original_counts)
            _add_transition(local_candidate, candidate_counts)
            component_rows.extend(_component_rows(
                source_static,
                world[source_horizon + 1, low_index],
                candidate_source,
                candidate_low[source_horizon + 1],
                downgrade[source_horizon],
                grid['x_centers'], grid['row_y_centers'],
                reference, source_horizon, min_component_cells,
                voxel_area))
        reference_rows.append({
            'reference_index': reference,
            'downgraded_voxels': local_counts['downgraded_voxels'],
            'original_transition': _with_rates(local_original),
            'candidate_transition': _with_rates(local_candidate),
        })
        if position % 100 == 0 or position == len(sequences):
            print(f'[labels {position}/{len(sequences)}] reference={reference}')

    original_with_rates = _with_rates(original_transition)
    candidate_with_rates = _with_rates(candidate_transition)
    original_free = original_transition['next_free_voxels']
    candidate_free = candidate_transition['next_free_voxels']
    rule_counts['candidate_all_world_known_voxels'] = (
        rule_counts['all_world_known_voxels'] -
        rule_counts['downgraded_voxels'])
    rule_counts['candidate_low_world_known_voxels'] = (
        rule_counts['low_world_known_voxels'] -
        rule_counts['downgraded_voxels'])
    rule_counts['candidate_low_world_static_voxels'] = (
        rule_counts['low_world_static_voxels'] -
        rule_counts['downgraded_voxels'])
    rule_counts['downgraded_fraction_of_low_static'] = (
        rule_counts['downgraded_voxels'] /
        rule_counts['low_world_static_voxels'])
    rule_counts['downgraded_fraction_of_all_known'] = (
        rule_counts['downgraded_voxels'] /
        rule_counts['all_world_known_voxels'])
    top_components = sorted(
        component_rows,
        key=lambda row: (
            row['original_transition']['next_free_voxels'],
            row['downgraded_cells'], row['area_cells']),
        reverse=True)[:top_count]
    result = {
        'sequence_root': str(sequence_root),
        'reference_count': len(sequences),
        'grid': {
            key: value for key, value in grid.items()
            if key not in ('x_centers', 'row_y_centers', 'quadrant_masks')
        },
        'rule_counts': rule_counts,
        'original_transition': original_with_rates,
        'candidate_transition': candidate_with_rates,
        'hard_free_conflict_reduction_voxels': (
            original_free - candidate_free),
        'hard_free_conflict_reduction_fraction': (
            (original_free - candidate_free) / original_free
            if original_free else None),
        'component_count': len(component_rows),
        'component_downgraded_fraction_distribution': _distribution([
            row['downgraded_fraction'] for row in component_rows]),
        'top_components_by_original_next_free': top_components,
        'top_references_by_downgraded_voxels': sorted(
            reference_rows,
            key=lambda row: (
                row['downgraded_voxels'],
                row['original_transition']['next_free_voxels']),
            reverse=True)[:top_count],
    }
    return result, sequences


def evaluate_b24(prediction_root: Path, sequences: dict,
                 low_index: int, repeated_free_count: int,
                 expected_epoch: int, top_count: int) -> dict:
    predictions = _prediction_paths(prediction_root)
    totals = {
        'original_known_voxels': 0,
        'candidate_known_voxels': 0,
        'downgraded_voxels': 0,
        'raw_original_errors': 0,
        'final_original_errors': 0,
        'raw_candidate_errors': 0,
        'final_candidate_errors': 0,
        'original_static_to_free_voxels': 0,
        'candidate_static_to_free_voxels': 0,
        'raw_static_on_candidate_static_to_free': 0,
        'final_static_on_candidate_static_to_free': 0,
    }
    rows = []
    for position, (reference, prediction_path) in enumerate(
            predictions.items(), start=1):
        with np.load(sequences[reference], allow_pickle=False) as archive:
            world = np.asarray(
                archive['world_target_state_3d'], dtype=np.uint8)
            direct = np.asarray(
                archive['direct_observation_state_3d'], dtype=np.uint8)
            future_free = np.asarray(
                archive['future_free_count_3d'], dtype=np.uint8)
        candidate, downgrade = apply_low_static_conflict_downgrade(
            world, direct, future_free, low_index, repeated_free_count)
        with np.load(prediction_path, allow_pickle=False) as archive:
            if int(archive['checkpoint_epoch']) != expected_epoch:
                raise ValueError(f'Unexpected B24 epoch for {reference}')
            raw = np.asarray(
                archive['raw_world_pred_class_3d'], dtype=np.uint8) + 1
            final = np.asarray(
                archive['world_pred_class_3d'], dtype=np.uint8) + 1
        original_low = world[:, low_index]
        raw_low = raw[:, low_index]
        final_low = final[:, low_index]
        original_known = original_low != 0
        candidate_known = candidate != 0
        original_source = original_low[0] == 2
        candidate_source = candidate[0] == 2
        original_flip = original_source & (original_low[1] == 1)
        candidate_flip = candidate_source & (candidate[1] == 1)
        row = {
            'reference_index': reference,
            'original_known_voxels': int(np.count_nonzero(original_known)),
            'candidate_known_voxels': int(np.count_nonzero(candidate_known)),
            'downgraded_voxels': int(np.count_nonzero(downgrade)),
            'raw_original_errors': int(np.count_nonzero(
                original_known & (raw_low != original_low))),
            'final_original_errors': int(np.count_nonzero(
                original_known & (final_low != original_low))),
            'raw_candidate_errors': int(np.count_nonzero(
                candidate_known & (raw_low != candidate))),
            'final_candidate_errors': int(np.count_nonzero(
                candidate_known & (final_low != candidate))),
            'original_static_to_free_voxels': int(np.count_nonzero(
                original_flip)),
            'candidate_static_to_free_voxels': int(np.count_nonzero(
                candidate_flip)),
            'raw_static_on_candidate_static_to_free': int(np.count_nonzero(
                candidate_flip & (raw_low[1] == 2))),
            'final_static_on_candidate_static_to_free': int(
                np.count_nonzero(candidate_flip & (final_low[1] == 2))),
        }
        for key in totals:
            totals[key] += row[key]
        rows.append(row)
        if position % 100 == 0 or position == len(predictions):
            print(f'[B24 {position}/{len(predictions)}] reference={reference}')
    original_known = totals['original_known_voxels']
    candidate_known = totals['candidate_known_voxels']
    candidate_flip = totals['candidate_static_to_free_voxels']
    return {
        'prediction_root': str(prediction_root),
        'prediction_count': len(predictions),
        'checkpoint_epoch': expected_epoch,
        **totals,
        'raw_original_error_fraction': (
            totals['raw_original_errors'] / original_known
            if original_known else None),
        'final_original_error_fraction': (
            totals['final_original_errors'] / original_known
            if original_known else None),
        'raw_candidate_error_fraction': (
            totals['raw_candidate_errors'] / candidate_known
            if candidate_known else None),
        'final_candidate_error_fraction': (
            totals['final_candidate_errors'] / candidate_known
            if candidate_known else None),
        'raw_static_fraction_on_candidate_static_to_free': (
            totals['raw_static_on_candidate_static_to_free'] /
            candidate_flip if candidate_flip else None),
        'final_static_fraction_on_candidate_static_to_free': (
            totals['final_static_on_candidate_static_to_free'] /
            candidate_flip if candidate_flip else None),
        'top_references_by_downgraded_voxels': sorted(
            rows,
            key=lambda row: (
                row['downgraded_voxels'],
                row['original_static_to_free_voxels']),
            reverse=True)[:top_count],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_full_train_v1'))
    parser.add_argument('--prediction-root', type=Path)
    parser.add_argument('--expected-epoch', type=int, default=3)
    parser.add_argument('--repeated-free-count', type=int, default=2)
    parser.add_argument('--min-component-cells', type=int, default=8)
    parser.add_argument('--top-count', type=int, default=100)
    parser.add_argument(
        '--out-json', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_low_static_downgrade_ablation_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    labels, sequences = evaluate_labels(
        args.sequence_root, args.repeated_free_count,
        args.min_component_cells, args.top_count)
    result = {
        'schema_version': 1,
        'name': 'KL OccWorld low-static repeated-free downgrade ablation',
        'rule': {
            'target_z_index': labels['grid']['low_static_z_index'],
            'target_z_center_m': labels['grid']['low_static_z_center_m'],
            'source_must_be_direct_static': True,
            'future_free_count_minimum': args.repeated_free_count,
            'replacement_state': 'unknown',
            'changes_model_input': False,
            'offline_training_target_only': True,
        },
        'labels': labels,
        'b24_internal_dev': None,
    }
    if args.prediction_root is not None:
        result['b24_internal_dev'] = evaluate_b24(
            args.prediction_root, sequences,
            labels['grid']['low_static_z_index'],
            args.repeated_free_count, args.expected_epoch,
            args.top_count)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open('w') as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    b24 = result['b24_internal_dev']
    if b24 is not None:
        b24 = {
            key: value for key, value in b24.items()
            if key != 'top_references_by_downgraded_voxels'
        }
    print(json.dumps({
        'out_json': str(args.out_json),
        'rule': result['rule'],
        'rule_counts': labels['rule_counts'],
        'original_transition': labels['original_transition'],
        'candidate_transition': labels['candidate_transition'],
        'hard_free_conflict_reduction_fraction': labels[
            'hard_free_conflict_reduction_fraction'],
        'b24_internal_dev': b24,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
