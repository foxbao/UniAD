#!/usr/bin/env python
"""Audit query-conditioned local flow overlay on validation predictions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmdet3d_plugin.uniad.dense_heads.occworld_head import (
    apply_local_flow_overlay,
    apply_query_conditioned_local_flow_overlay,
    query_conditioned_flow_event_masks,
)
from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
)


CLASS_NAMES = ('free', 'static_occupied', 'instance_occupied')


def _update_confusion(confusion: np.ndarray, prediction: np.ndarray,
                      target: np.ndarray, selection: np.ndarray):
    selected = selection & (target >= 0) & (target < len(CLASS_NAMES))
    encoded = (
        target[selected].astype(np.int64) * len(CLASS_NAMES) +
        prediction[selected].astype(np.int64))
    confusion += np.bincount(
        encoded, minlength=len(CLASS_NAMES) ** 2).reshape(
            len(CLASS_NAMES), len(CLASS_NAMES))


def semantic_metrics(confusion: np.ndarray) -> dict:
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(
        intersection, union,
        out=np.zeros_like(intersection), where=union > 0)
    return {
        'voxel_count': int(confusion.sum()),
        'accuracy': float(intersection.sum() / max(confusion.sum(), 1)),
        'iou': {
            name: float(value) for name, value in zip(CLASS_NAMES, iou)
        },
        'mean_iou': float(iou.mean()),
        'confusion_target_rows_prediction_columns': confusion.tolist(),
    }


def event_metrics(prediction: np.ndarray, target: np.ndarray,
                  evaluable: np.ndarray) -> dict:
    prediction = prediction.astype(bool) & evaluable.astype(bool)
    target = target.astype(bool) & evaluable.astype(bool)
    true_positive = int(np.count_nonzero(prediction & target))
    false_positive = int(np.count_nonzero(prediction & ~target & evaluable))
    false_negative = int(np.count_nonzero(~prediction & target))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        'predicted_count': int(np.count_nonzero(prediction)),
        'target_count': int(np.count_nonzero(target)),
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(
            2.0 * precision * recall /
            max(precision + recall, 1e-12)),
    }


def _method_accumulator():
    return {
        key: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
        for key in ('future', 'current_visible', 'reveal',
                    'state_change', 'visible_transition',
                    'instance_transition')
    }


def _method_summary(accumulator: dict) -> dict:
    return {
        key: semantic_metrics(confusion)
        for key, confusion in accumulator.items()
    }


def qualifying_query_thresholds(result: dict) -> list:
    """Return gates meeting the predeclared correction/transition bar."""
    ungated = result['methods']['recreated_ungated']
    qualifying = []
    for threshold in result['query_thresholds']:
        name = f'query_{threshold:g}'
        method = result['methods'][name]
        corrections = result['corrections_vs_raw'][name]
        if (corrections['net_correct_voxels'] > 0 and
                method['visible_transition']['mean_iou'] >=
                ungated['visible_transition']['mean_iou'] and
                method['instance_transition']['mean_iou'] >=
                ungated['instance_transition']['mean_iou']):
            qualifying.append(float(threshold))
    return qualifying


def _correction_counts(raw: np.ndarray, fused: np.ndarray,
                       target: np.ndarray, valid: np.ndarray) -> dict:
    changed = valid & (raw != fused)
    raw_correct = raw == target
    fused_correct = fused == target
    improved = changed & ~raw_correct & fused_correct
    harmed = changed & raw_correct & ~fused_correct
    return {
        'changed_voxels': int(np.count_nonzero(changed)),
        'improved_voxels': int(np.count_nonzero(improved)),
        'harmed_voxels': int(np.count_nonzero(harmed)),
        'net_correct_voxels': int(
            np.count_nonzero(improved) - np.count_nonzero(harmed)),
    }


def _accumulate_method(accumulator, prediction, target, target_known,
                       current_known, persistence):
    state_change = target_known & (target != persistence)
    reveal = target_known & ~current_known
    visible_transition = state_change & current_known
    current_instance = current_known & (persistence == 2)
    instance_transition = (
        visible_transition & (current_instance | (target == 2)))
    selections = {
        'future': target_known,
        'current_visible': target_known & current_known,
        'reveal': reveal,
        'state_change': state_change,
        'visible_transition': visible_transition,
        'instance_transition': instance_transition,
    }
    for key, selection in selections.items():
        _update_confusion(
            accumulator[key], prediction, target, selection)


def _load_diagnostic(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        required = (
            'world_pred_class_3d', 'raw_world_pred_class_3d',
            'warped_instance_probability_3d',
            'query_dynamic_probability_2d',
            'observation_class_3d', 'observation_known_3d')
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f'{path} lacks query diagnostics: {missing}')
        return {
            'final': np.asarray(
                payload['world_pred_class_3d'], dtype=np.uint8),
            'raw': np.asarray(
                payload['raw_world_pred_class_3d'], dtype=np.uint8),
            'warped': np.asarray(
                payload['warped_instance_probability_3d'],
                dtype=np.float32),
            'query': np.asarray(
                payload['query_dynamic_probability_2d'],
                dtype=np.float32),
            'observation_class': np.asarray(
                payload['observation_class_3d'], dtype=np.uint8),
            'observation_known': np.asarray(
                payload['observation_known_3d'], dtype=bool),
        }


def analyze(manifest: dict, sequences: dict, predictions: dict,
            flow_threshold: float, query_thresholds) -> dict:
    references = [
        int(record['reference_index'])
        for record in manifest['splits']['validation']
    ]
    method_names = ['raw', 'model_final', 'recreated_ungated'] + [
        f'query_{threshold:g}' for threshold in query_thresholds
    ]
    accumulators = {
        name: _method_accumulator() for name in method_names
    }
    correction_totals = {
        name: dict(changed_voxels=0, improved_voxels=0,
                   harmed_voxels=0, net_correct_voxels=0)
        for name in method_names if name != 'raw'
    }
    event_rows = {
        name: {'arrival': [], 'departure': []}
        for name in ['ungated'] + [
            f'query_{threshold:g}' for threshold in query_thresholds]
    }
    parity_difference_count = 0

    for reference in references:
        diagnostic = _load_diagnostic(predictions[reference])
        with np.load(sequences[reference], allow_pickle=False) as label:
            target_state = np.asarray(
                label['world_target_state_3d'], dtype=np.uint8)
            target_valid = np.asarray(
                label['world_target_valid_3d'], dtype=bool)
        target = np.zeros_like(target_state, dtype=np.uint8)
        target[target_valid & (target_state > 0)] = (
            target_state[target_valid & (target_state > 0)] - 1)
        target_known = target_valid & (target_state > 0)

        raw = torch.from_numpy(diagnostic['raw'])[None]
        observation_class = torch.from_numpy(
            diagnostic['observation_class'])[None]
        observation_known = torch.from_numpy(
            diagnostic['observation_known'])[None]
        warped = torch.from_numpy(diagnostic['warped'])[None]
        query_future = torch.from_numpy(diagnostic['query'][1:])[None]
        ungated = apply_local_flow_overlay(
            raw, observation_class, observation_known, warped,
            threshold=flow_threshold)[0].numpy()
        parity_difference_count += int(np.count_nonzero(
            ungated != diagnostic['final']))
        methods = {
            'raw': diagnostic['raw'],
            'model_final': diagnostic['final'],
            'recreated_ungated': ungated,
        }
        for threshold in query_thresholds:
            methods[f'query_{threshold:g}'] = (
                apply_query_conditioned_local_flow_overlay(
                    raw, observation_class, observation_known, warped,
                    query_future, flow_threshold=flow_threshold,
                    query_threshold=float(threshold))[0].numpy())

        current_known = np.broadcast_to(
            diagnostic['observation_known'], target.shape)[1:]
        persistence = np.broadcast_to(
            diagnostic['observation_class'], target.shape)[1:]
        for name, prediction in methods.items():
            _accumulate_method(
                accumulators[name], prediction[1:], target[1:],
                target_known[1:], current_known, persistence)
            if name != 'raw':
                counts = _correction_counts(
                    diagnostic['raw'][1:], prediction[1:], target[1:],
                    target_known[1:])
                for key, value in counts.items():
                    correction_totals[name][key] += value

        current_instance = np.broadcast_to(
            (diagnostic['observation_known'] &
             (diagnostic['observation_class'] == 2)),
            target[1:].shape)
        actual_arrival = (
            target_known[1:] & (target[1:] == 2) & ~current_instance)
        actual_departure = (
            target_known[1:] & current_instance & (target[1:] != 2))
        evaluable = target_known[1:]
        current_tensor = (
            observation_known & (observation_class == 2))[:, None].to(
                warped.dtype)
        ungated_arrival = (
            warped - current_tensor >= flow_threshold)[0].numpy()
        ungated_departure = (
            current_tensor - warped >= flow_threshold)[0].numpy()
        event_rows['ungated']['arrival'].append(event_metrics(
            ungated_arrival, actual_arrival, evaluable))
        event_rows['ungated']['departure'].append(event_metrics(
            ungated_departure, actual_departure, evaluable))
        for threshold in query_thresholds:
            arrival, departure = query_conditioned_flow_event_masks(
                observation_class, observation_known, warped,
                query_future, flow_threshold, float(threshold))
            key = f'query_{threshold:g}'
            event_rows[key]['arrival'].append(event_metrics(
                arrival[0].numpy(), actual_arrival, evaluable))
            event_rows[key]['departure'].append(event_metrics(
                departure[0].numpy(), actual_departure, evaluable))

    def combine_event(rows):
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
            precision=float(precision), recall=float(recall),
            f1=float(2 * precision * recall /
                     max(precision + recall, 1e-12)))
        return totals

    return {
        'reference_count': len(references),
        'reference_indices': references,
        'flow_threshold': float(flow_threshold),
        'query_thresholds': [float(value) for value in query_thresholds],
        'model_recreation_difference_voxels': parity_difference_count,
        'methods': {
            name: _method_summary(accumulator)
            for name, accumulator in accumulators.items()
        },
        'corrections_vs_raw': correction_totals,
        'events': {
            name: {
                event: combine_event(rows)
                for event, rows in by_event.items()
            }
            for name, by_event in event_rows.items()
        },
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
            'occworld_predictions_b19_query_diagnostic_validation_v1/'
            'validation/epoch_003'))
    parser.add_argument('--split', default='validation')
    parser.add_argument('--flow-threshold', type=float, default=0.9)
    parser.add_argument(
        '--query-thresholds', type=float, nargs='+',
        default=(0.1, 0.3, 0.5))
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b19_query_flow_overlay_audit_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split != 'validation':
        raise ValueError('Query-flow development audit is validation-only')
    manifest = _load_manifest(args.manifest)
    sequences = _sequence_mapping(args.sequence_root)
    predictions = _prediction_mapping(args.prediction_root)
    expected = {
        int(record['reference_index'])
        for record in manifest['splits']['validation']
    }
    if set(predictions) != expected:
        raise ValueError('Query-flow predictions are not the exact split')
    if not expected.issubset(sequences):
        raise ValueError('Query-flow labels are incomplete')
    result = analyze(
        manifest, sequences, predictions,
        args.flow_threshold, args.query_thresholds)
    if result['model_recreation_difference_voxels'] != 0:
        raise RuntimeError(
            'The offline overlay does not exactly recreate model output')
    qualifying = qualifying_query_thresholds(result)
    report = {
        'schema_version': 1,
        'name': args.out_file.stem,
        'split': args.split,
        'purpose': 'development_validation',
        'model_inference_performed_by_audit': False,
        'holdout_accessed': False,
        'manifest': str(args.manifest),
        'sequence_root': str(args.sequence_root),
        'prediction_root': str(args.prediction_root),
        'acceptance_criteria': {
            'net_correct_voxels_vs_raw': 'greater_than_zero',
            'visible_transition_miou_vs_ungated': 'not_lower',
            'instance_transition_miou_vs_ungated': 'not_lower',
        },
        'decision': {
            'status': (
                'candidate_for_formal_integration' if qualifying
                else 'completed_not_promoted'),
            'qualifying_query_thresholds': qualifying,
            'holdout_evaluation_authorized': False,
        },
        **result,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'name': report['name'],
        'model_recreation_difference_voxels': report[
            'model_recreation_difference_voxels'],
        'methods': {
            name: {
                key: value['mean_iou'] for key, value in method.items()
            } for name, method in report['methods'].items()
        },
        'corrections_vs_raw': report['corrections_vs_raw'],
        'events': report['events'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
