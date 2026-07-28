#!/usr/bin/env python
"""Audit a raw-preserving Motion actor arrival overlay on validation."""

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
from tools.analysis_tools.audit_kl_occworld_motion_actor_support import (
    motion_actor_support,
)
from tools.analysis_tools.audit_kl_occworld_query_flow_overlay import (
    _accumulate_method,
    _correction_counts,
    _method_accumulator,
    _method_summary,
    event_metrics,
)
from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
    _split_references,
)


INSTANCE_CLASS = 2


def apply_motion_actor_arrival_overlay(
        raw_prediction: np.ndarray,
        motion_support: np.ndarray,
        stationary_support: np.ndarray,
        instance_class: int = INSTANCE_CLASS,
        raw_class_gate=None) -> tuple:
    """Set only geometric arrival voxels to instance; preserve all else."""
    raw_prediction = np.asarray(raw_prediction, dtype=np.uint8)
    motion_support = np.asarray(motion_support, dtype=np.bool_)
    stationary_support = np.asarray(stationary_support, dtype=np.bool_)
    if raw_prediction.ndim != 4:
        raise ValueError('Raw prediction must have shape [T,Z,H,W]')
    expected_support = (raw_prediction.shape[0] - 1,
                        *raw_prediction.shape[1:])
    if (motion_support.shape != expected_support or
            stationary_support.shape != expected_support):
        raise ValueError(
            f'Actor support must have shape {expected_support}')
    arrival = motion_support & ~stationary_support
    if raw_class_gate is not None:
        raw_class_gate = tuple(int(value) for value in raw_class_gate)
        if not raw_class_gate or any(
                value not in (0, 1, 2) for value in raw_class_gate):
            raise ValueError('Raw class gate must contain classes from 0, 1, 2')
        arrival &= np.isin(raw_prediction[1:], raw_class_gate)
    candidate = raw_prediction.copy()
    candidate[1:][arrival] = np.uint8(instance_class)
    outside_difference = int(np.count_nonzero(
        (candidate[1:] != raw_prediction[1:]) & ~arrival))
    current_difference = int(np.count_nonzero(
        candidate[0] != raw_prediction[0]))
    return candidate, arrival, {
        'arrival_voxels': int(np.count_nonzero(arrival)),
        'effective_changed_voxels': int(np.count_nonzero(
            arrival & (raw_prediction[1:] != instance_class))),
        'outside_arrival_difference_voxels': outside_difference,
        'current_difference_voxels': current_difference,
    }


def _load_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        required = (
            'world_pred_class_3d', 'raw_world_pred_class_3d',
            'motion_actor_future_xy', 'motion_actor_boxes_3d',
            'motion_actor_scores', 'motion_actor_valid',
            'motion_actor_step_times_s', 'motion_actor_box_z_origin')
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(
                f'{path} lacks B22 diagnostics: {missing}')
        return {
            key: np.array(payload[key], copy=True)
            for key in required
        }


def _combine_events(rows: list) -> dict:
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in ('predicted_count', 'target_count', 'true_positive',
                    'false_positive', 'false_negative')
    }
    precision = totals['true_positive'] / max(
        totals['true_positive'] + totals['false_positive'], 1)
    recall = totals['true_positive'] / max(
        totals['true_positive'] + totals['false_negative'], 1)
    totals.update(
        precision=float(precision),
        recall=float(recall),
        f1=float(2.0 * precision * recall /
                 max(precision + recall, 1e-12)))
    return totals


def _zero_corrections() -> dict:
    return dict(
        changed_voxels=0, improved_voxels=0,
        harmed_voxels=0, net_correct_voxels=0)


def _add_corrections(destination: dict, source: dict):
    for key, value in source.items():
        destination[key] += int(value)


def _changed_class_matrix(raw: np.ndarray, candidate: np.ndarray,
                          target: np.ndarray,
                          selection: np.ndarray) -> np.ndarray:
    """Count target/raw classes only where the fixed overlay changed output."""
    changed = np.asarray(selection, dtype=np.bool_) & (raw != candidate)
    encoded = target[changed].astype(np.int64) * 3 + raw[changed].astype(
        np.int64)
    return np.bincount(encoded, minlength=9).reshape(3, 3)


def _delta_pp(candidate: float, baseline: float) -> float:
    return float((candidate - baseline) * 100.0)


def qualification_checks(methods: dict, corrections: dict,
                         preservation: dict) -> dict:
    """Evaluate the seven predeclared B22 promotion conditions."""
    raw = methods['raw']
    final = methods['b17_final']
    candidate = methods['actor_arrival']

    def metric(method, subset, name='mean_iou'):
        if name == 'instance_iou':
            return method[subset]['iou']['instance_occupied']
        return method[subset][name]

    values = {
        'future_semantic_delta_vs_raw_pp': _delta_pp(
            metric(candidate, 'future'), metric(raw, 'future')),
        'future_instance_delta_vs_raw_pp': _delta_pp(
            metric(candidate, 'future', 'instance_iou'),
            metric(raw, 'future', 'instance_iou')),
        'current_visible_delta_vs_raw_pp': _delta_pp(
            metric(candidate, 'current_visible'),
            metric(raw, 'current_visible')),
        'visible_transition_delta_vs_raw_pp': _delta_pp(
            metric(candidate, 'visible_transition'),
            metric(raw, 'visible_transition')),
        'instance_transition_delta_vs_raw_pp': _delta_pp(
            metric(candidate, 'instance_transition'),
            metric(raw, 'instance_transition')),
        'future_instance_delta_vs_final_pp': _delta_pp(
            metric(candidate, 'future', 'instance_iou'),
            metric(final, 'future', 'instance_iou')),
        'visible_transition_delta_vs_final_pp': _delta_pp(
            metric(candidate, 'visible_transition'),
            metric(final, 'visible_transition')),
        'instance_transition_delta_vs_final_pp': _delta_pp(
            metric(candidate, 'instance_transition'),
            metric(final, 'instance_transition')),
    }
    checks = {
        'raw_preserved_outside_arrival': (
            preservation['outside_arrival_difference_voxels'] == 0 and
            preservation['current_difference_voxels'] == 0),
        'positive_net_correct': corrections['net_correct_voxels'] > 0,
        'future_instance_gain_at_least_0_10_pp': (
            values['future_instance_delta_vs_raw_pp'] >= 0.10),
        'visible_transition_not_below_raw': (
            values['visible_transition_delta_vs_raw_pp'] >= 0.0),
        'instance_transition_not_below_raw': (
            values['instance_transition_delta_vs_raw_pp'] >= 0.0),
        'semantic_losses_within_0_05_pp': (
            values['future_semantic_delta_vs_raw_pp'] >= -0.05 and
            values['current_visible_delta_vs_raw_pp'] >= -0.05),
        'not_below_b17_final_dynamic_metrics': (
            values['future_instance_delta_vs_final_pp'] >= 0.0 and
            values['visible_transition_delta_vs_final_pp'] >= 0.0 and
            values['instance_transition_delta_vs_final_pp'] >= 0.0),
    }
    return {
        'values': values,
        'checks': checks,
        'qualified': all(checks.values()),
    }


def analyze(manifest: dict, sequences: dict, predictions: dict,
            split: str, builder: RaycastDrivableBuilder,
            score_threshold: float,
            time_tolerance_s: float, raw_class_gate=None) -> dict:
    references = _split_references(manifest, split)
    missing = {
        'sequences': sorted(set(references).difference(sequences)),
        'predictions': sorted(set(references).difference(predictions)),
    }
    if any(missing.values()):
        raise KeyError(missing)
    names = ('raw', 'b17_final', 'actor_arrival')
    accumulators = {name: _method_accumulator() for name in names}
    corrections = _zero_corrections()
    correction_subsets = {
        name: _zero_corrections()
        for name in (
            'future', 'current_visible', 'reveal', 'state_change',
            'visible_transition', 'instance_transition',
            'stable_current_visible')
    }
    correction_by_horizon = [_zero_corrections() for _ in range(4)]
    changed_class_matrices = {
        name: np.zeros((3, 3), dtype=np.int64)
        for name in ('future', 'current_visible', 'stable_current_visible',
                     'visible_transition')
    }
    preservation = dict(
        arrival_voxels=0, effective_changed_voxels=0,
        outside_arrival_difference_voxels=0,
        current_difference_voxels=0)
    arrival_event_rows = []
    visible_arrival_event_rows = []
    rows = []

    for reference in references:
        diagnostic = _load_prediction(predictions[reference])
        with np.load(sequences[reference], allow_pickle=False) as label:
            target_state = np.asarray(
                label['world_target_state_3d'], dtype=np.uint8)
            target_valid = np.asarray(
                label['world_target_valid_3d'], dtype=np.bool_)
            current_state = np.asarray(
                label['current_observation_state_3d'], dtype=np.uint8)
            current_valid = np.asarray(
                label['current_observation_valid_3d'], dtype=np.bool_)
            target_times = np.asarray(
                label['nominal_target_times_s'][1:], dtype=np.float32)
        raw = np.asarray(
            diagnostic['raw_world_pred_class_3d'], dtype=np.uint8)
        final = np.asarray(
            diagnostic['world_pred_class_3d'], dtype=np.uint8)
        if raw.shape != target_state.shape or final.shape != target_state.shape:
            raise ValueError(f'Prediction shape mismatch for {reference}')
        motion, metadata = motion_actor_support(
            diagnostic, builder, target_times,
            score_threshold=score_threshold,
            time_tolerance_s=time_tolerance_s)
        stationary, _ = motion_actor_support(
            diagnostic, builder, target_times,
            score_threshold=score_threshold,
            time_tolerance_s=time_tolerance_s,
            position_policy='stationary')
        candidate, arrival, local_preservation = (
            apply_motion_actor_arrival_overlay(
                raw, motion, stationary, raw_class_gate=raw_class_gate))

        target = np.zeros_like(target_state, dtype=np.uint8)
        target_known = target_valid & (target_state != 0)
        target[target_known] = target_state[target_known] - 1
        current_known_3d = current_valid & (current_state != 0)
        current_known = np.broadcast_to(
            current_known_3d, target[1:].shape)
        persistence_3d = np.zeros_like(current_state, dtype=np.uint8)
        persistence_3d[current_known_3d] = (
            current_state[current_known_3d] - 1)
        persistence = np.broadcast_to(
            persistence_3d, target[1:].shape)
        methods = {
            'raw': raw,
            'b17_final': final,
            'actor_arrival': candidate,
        }
        for name, prediction in methods.items():
            _accumulate_method(
                accumulators[name], prediction[1:], target[1:],
                target_known[1:], current_known, persistence)

        local_correction = _correction_counts(
            raw[1:], candidate[1:], target[1:], target_known[1:])
        _add_corrections(corrections, local_correction)
        current_instance = np.broadcast_to(
            current_valid & (current_state == 3), target[1:].shape)
        state_change = target_known[1:] & (target[1:] != persistence)
        reveal = target_known[1:] & ~current_known
        visible_transition = state_change & current_known
        instance_transition = (
            visible_transition &
            (current_instance | (target[1:] == INSTANCE_CLASS)))
        selections = {
            'future': target_known[1:],
            'current_visible': target_known[1:] & current_known,
            'reveal': reveal,
            'state_change': state_change,
            'visible_transition': visible_transition,
            'instance_transition': instance_transition,
            'stable_current_visible': (
                target_known[1:] & current_known & ~state_change),
        }
        for name, selection in selections.items():
            _add_corrections(
                correction_subsets[name],
                _correction_counts(
                    raw[1:], candidate[1:], target[1:], selection))
        for name in changed_class_matrices:
            changed_class_matrices[name] += _changed_class_matrix(
                raw[1:], candidate[1:], target[1:], selections[name])
        for horizon in range(4):
            _add_corrections(
                correction_by_horizon[horizon],
                _correction_counts(
                    raw[horizon + 1:horizon + 2],
                    candidate[horizon + 1:horizon + 2],
                    target[horizon + 1:horizon + 2],
                    target_known[horizon + 1:horizon + 2]))
        for key, value in local_preservation.items():
            preservation[key] += int(value)

        actual_arrival = (
            target_known[1:] & (target[1:] == INSTANCE_CLASS) &
            ~current_instance)
        visible_arrival = actual_arrival & current_known
        arrival_event_rows.append(event_metrics(
            arrival, actual_arrival, target_known[1:]))
        visible_arrival_event_rows.append(event_metrics(
            arrival, visible_arrival,
            target_known[1:] & current_known))
        rows.append({
            'reference_index': int(reference),
            'kept_actor_count': metadata['kept_actor_count'],
            'arrival_voxels': local_preservation['arrival_voxels'],
            'effective_changed_voxels': local_preservation[
                'effective_changed_voxels'],
            'corrections_vs_raw': local_correction,
        })

    methods = {
        name: _method_summary(accumulator)
        for name, accumulator in accumulators.items()
    }
    qualification = qualification_checks(
        methods, corrections, preservation)
    return {
        'schema_version': 1,
        'purpose': (
            'Validation-only raw-preserving overlay of Motion actor '
            'geometric arrival events.'),
        'split': split,
        'reference_count': len(references),
        'reference_indices': references,
        'score_threshold': float(score_threshold),
        'overlay_contract': (
            'candidate=raw; set motion AND NOT stationary support to '
            'instance; do not modify departure'),
        'raw_class_gate': (
            None if raw_class_gate is None else
            [int(value) for value in raw_class_gate]),
        'methods': methods,
        'corrections_vs_raw': corrections,
        'corrections_vs_raw_by_subset': correction_subsets,
        'corrections_vs_raw_by_horizon': correction_by_horizon,
        'changed_voxel_target_rows_raw_columns': {
            key: value.tolist()
            for key, value in changed_class_matrices.items()
        },
        'preservation': preservation,
        'events': {
            'all_target_arrival': _combine_events(arrival_event_rows),
            'visible_target_arrival': _combine_events(
                visible_arrival_event_rows),
        },
        'qualification': qualification,
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
    parser.add_argument('--score-threshold', type=float, default=0.1)
    parser.add_argument('--time-tolerance-s', type=float, default=0.05)
    parser.add_argument(
        '--raw-class-gate', type=int, nargs='+',
        help='Optional raw semantic classes allowed to receive arrival overlay.')
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
            'kl_occworld_b22_motion_actor_arrival_overlay_validation_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    builder = RaycastDrivableBuilder(
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size)
    report = analyze(
        _load_manifest(args.manifest),
        _sequence_mapping(args.sequence_root),
        _prediction_mapping(args.prediction_root),
        split=args.split,
        builder=builder,
        score_threshold=args.score_threshold,
        time_tolerance_s=args.time_tolerance_s,
        raw_class_gate=args.raw_class_gate)
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
