#!/usr/bin/env python
"""Select an OccWorld checkpoint and visibility threshold on validation."""

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
    evaluate_split,
)


def _epoch_from_directory(path: Path) -> int:
    match = re.fullmatch(r'epoch_(\d+)', path.name)
    if match is None:
        raise ValueError(f'Invalid prediction directory: {path}')
    return int(match.group(1))


def _future_mean(metrics, key: str = 'semantic') -> float:
    values = [
        float(item['mean_iou'])
        for item in metrics[key]['by_horizon'][1:]
    ]
    if not values:
        raise ValueError('Evaluation has no future horizons')
    return float(np.mean(values))


def _epoch_row(epoch: int, metrics) -> dict:
    semantic = metrics['semantic']
    change_gate = metrics['future_change_gate']
    changed_class = metrics['future_changed_class_on_transition']
    return {
        'epoch': epoch,
        'future_mean_iou': _future_mean(metrics),
        'overall_mean_iou': float(semantic['overall']['mean_iou']),
        'future_reveal_mean_iou': float(np.mean([
            item['mean_iou'] for item in
            semantic['reveal_completion_subset']['by_horizon'][1:]
        ])),
        'future_state_change_mean_iou': float(np.mean([
            item['mean_iou'] for item in
            semantic['state_change_subset']['by_horizon'][1:]
        ])),
        'future_visible_transition_mean_iou': float(np.mean([
            item['mean_iou'] for item in
            semantic['visible_transition_subset']['by_horizon'][1:]
        ])),
        'future_instance_related_visible_transition_mean_iou': float(
            np.mean([
                item['mean_iou'] for item in
                semantic[
                    'instance_related_visible_transition_subset'
                ]['by_horizon'][1:]
            ])),
        'visibility_f1_at_0_5': (
            None if metrics['visibility'] is None
            else float(metrics['visibility']['overall']['f1'])),
        'change_gate_precision_at_0_5': (
            None if change_gate is None
            else float(change_gate['overall']['precision'])),
        'change_gate_recall_at_0_5': (
            None if change_gate is None
            else float(change_gate['overall']['recall'])),
        'change_gate_f1_at_0_5': (
            None if change_gate is None
            else float(change_gate['overall']['f1'])),
        'changed_class_transition_miou': (
            None if changed_class is None
            else float(changed_class['overall']['mean_iou'])),
        'changed_class_transition_accuracy': (
            None if changed_class is None
            else float(changed_class['overall']['accuracy'])),
    }


def select_checkpoint(manifest, sequence_mapping, prediction_root: Path,
                      thresholds, change_gate_thresholds=None,
                      flow_fusion_thresholds=None,
                      physical_confidence_thresholds=None) -> dict:
    epoch_directories = sorted(
        [path for path in prediction_root.glob('epoch_*') if path.is_dir()],
        key=_epoch_from_directory)
    if not epoch_directories:
        raise FileNotFoundError(
            f'No epoch prediction directories below {prediction_root}')
    epoch_rows = []
    predictions_by_epoch = {}
    for directory in epoch_directories:
        epoch = _epoch_from_directory(directory)
        prediction_mapping = _prediction_mapping(directory)
        predictions_by_epoch[epoch] = prediction_mapping
        metrics = evaluate_split(
            manifest=manifest,
            split='validation',
            sequence_mapping=sequence_mapping,
            prediction_mapping=prediction_mapping,
            visibility_threshold=0.5)
        epoch_rows.append(_epoch_row(epoch, metrics))
    selected_row = max(
        epoch_rows,
        key=lambda row: (row['future_mean_iou'], -row['epoch']))
    selected_epoch = selected_row['epoch']

    threshold_rows = []
    selected_metrics = None
    for threshold in thresholds:
        metrics = evaluate_split(
            manifest=manifest,
            split='validation',
            sequence_mapping=sequence_mapping,
            prediction_mapping=predictions_by_epoch[selected_epoch],
            visibility_threshold=float(threshold))
        visibility = metrics['visibility']['overall']
        threshold_rows.append({
            'threshold': float(threshold),
            'precision': float(visibility['precision']),
            'recall': float(visibility['recall']),
            'f1': float(visibility['f1']),
            'iou': float(visibility['iou']),
        })
    selected_threshold_row = max(
        threshold_rows,
        key=lambda row: (
            row['f1'], -abs(row['threshold'] - 0.5),
            -row['threshold']))
    selected_threshold = selected_threshold_row['threshold']
    selected_metrics = evaluate_split(
        manifest=manifest,
        split='validation',
        sequence_mapping=sequence_mapping,
        prediction_mapping=predictions_by_epoch[selected_epoch],
        visibility_threshold=selected_threshold)
    gate_threshold_rows = []
    selected_gate_threshold_row = None
    hard_gate_rows = []
    selected_hard_gate_row = None
    flow_fusion_rows = []
    selected_flow_fusion_row = None
    local_overlay_rows = []
    selected_local_overlay_row = None
    selected_local_overlay_dual_gate_row = None
    physical_confidence_rows = []
    selected_physical_confidence_row = None
    if selected_metrics['future_change_gate'] is not None:
        if change_gate_thresholds is None:
            change_gate_thresholds = [0.5]
        for threshold in change_gate_thresholds:
            metrics = evaluate_split(
                manifest=manifest,
                split='validation',
                sequence_mapping=sequence_mapping,
                prediction_mapping=predictions_by_epoch[selected_epoch],
                visibility_threshold=selected_threshold,
                change_gate_threshold=float(threshold))
            gate = metrics['future_change_gate']['overall']
            gate_threshold_rows.append({
                'threshold': float(threshold),
                'precision': float(gate['precision']),
                'recall': float(gate['recall']),
                'f1': float(gate['f1']),
                'iou': float(gate['iou']),
            })
        selected_gate_threshold_row = max(
            gate_threshold_rows,
            key=lambda row: (
                row['f1'], -abs(row['threshold'] - 0.5),
                -row['threshold']))
        for epoch, prediction_mapping in predictions_by_epoch.items():
            for threshold in change_gate_thresholds:
                metrics = evaluate_split(
                    manifest=manifest,
                    split='validation',
                    sequence_mapping=sequence_mapping,
                    prediction_mapping=prediction_mapping,
                    visibility_threshold=selected_threshold,
                    change_gate_threshold=float(threshold),
                    apply_change_gate=True)
                row = _epoch_row(epoch, metrics)
                hard_gate_rows.append(dict(
                    row,
                    change_gate_threshold=float(threshold)))
        selected_hard_gate_row = max(
            hard_gate_rows,
            key=lambda row: (
                row['future_mean_iou'],
                -row['epoch'],
                -abs(row['change_gate_threshold'] - 0.5)))
    if selected_metrics['flow_warp_available']:
        if flow_fusion_thresholds is None:
            flow_fusion_thresholds = [0.5]
        for epoch, prediction_mapping in predictions_by_epoch.items():
            for threshold in flow_fusion_thresholds:
                metrics = evaluate_split(
                    manifest=manifest,
                    split='validation',
                    sequence_mapping=sequence_mapping,
                    prediction_mapping=prediction_mapping,
                    visibility_threshold=selected_threshold,
                    apply_flow_fusion=True,
                    flow_fusion_threshold=float(threshold))
                row = _epoch_row(epoch, metrics)
                flow_fusion_rows.append(dict(
                    row,
                    flow_fusion_threshold=float(threshold)))
        selected_flow_fusion_row = max(
            flow_fusion_rows,
            key=lambda row: (
                row['future_mean_iou'], -row['epoch'],
                -abs(row['flow_fusion_threshold'] - 0.5)))
        for threshold in flow_fusion_thresholds:
            metrics = evaluate_split(
                manifest=manifest,
                split='validation',
                sequence_mapping=sequence_mapping,
                prediction_mapping=predictions_by_epoch[selected_epoch],
                visibility_threshold=selected_threshold,
                apply_local_flow_overlay=True,
                flow_fusion_threshold=float(threshold))
            row = _epoch_row(selected_epoch, metrics)
            local_overlay_rows.append(dict(
                row,
                flow_fusion_threshold=float(threshold)))
        selected_local_overlay_row = max(
            local_overlay_rows,
            key=lambda row: (
                row['future_mean_iou'],
                -abs(row['flow_fusion_threshold'] - 0.5)))
        dual_gate_rows = [
            row for row in local_overlay_rows
            if (row['future_mean_iou'] >= selected_row['future_mean_iou'] and
                row['future_visible_transition_mean_iou'] >
                selected_row['future_visible_transition_mean_iou'])
        ]
        if dual_gate_rows:
            selected_local_overlay_dual_gate_row = max(
                dual_gate_rows,
                key=lambda row: (
                    row['future_mean_iou'],
                    row['future_visible_transition_mean_iou'],
                    -abs(row['flow_fusion_threshold'] - 0.5)))
    if selected_metrics['physical_confidence_available']:
        if physical_confidence_thresholds is None:
            physical_confidence_thresholds = [0.5]
        for epoch, prediction_mapping in predictions_by_epoch.items():
            for threshold in physical_confidence_thresholds:
                metrics = evaluate_split(
                    manifest=manifest,
                    split='validation',
                    sequence_mapping=sequence_mapping,
                    prediction_mapping=prediction_mapping,
                    visibility_threshold=selected_threshold,
                    apply_physical_confidence=True,
                    physical_confidence_threshold=float(threshold),
                    physical_confidence_flow_threshold=0.5)
                row = _epoch_row(epoch, metrics)
                physical_confidence_rows.append(dict(
                    row,
                    physical_confidence_threshold=float(threshold),
                    physical_confidence_flow_threshold=0.5))
        selected_physical_confidence_row = max(
            physical_confidence_rows,
            key=lambda row: (
                row['future_mean_iou'], -row['epoch'],
                -abs(row['physical_confidence_threshold'] - 0.5)))
    persistence_metrics = evaluate_split(
        manifest=manifest,
        split='validation',
        sequence_mapping=sequence_mapping,
        visibility_threshold=selected_threshold)
    return {
        'schema_version': 1,
        'selection_split': 'validation',
        'checkpoint_selection_metric': (
            'mean semantic mIoU over future horizons t=1..4'),
        'threshold_selection_metric': 'overall visibility F1',
        'epoch_rows': epoch_rows,
        'selected_epoch': selected_epoch,
        'selected_epoch_row': selected_row,
        'threshold_rows': threshold_rows,
        'selected_visibility_threshold': selected_threshold,
        'selected_threshold_row': selected_threshold_row,
        'change_gate_threshold_rows': gate_threshold_rows,
        'selected_change_gate_threshold_row': (
            selected_gate_threshold_row),
        'hard_change_gate_rows': hard_gate_rows,
        'selected_hard_change_gate_row': selected_hard_gate_row,
        'physical_flow_fusion_rows': flow_fusion_rows,
        'selected_physical_flow_fusion_row': (
            selected_flow_fusion_row),
        'local_flow_overlay_rows': local_overlay_rows,
        'selected_local_flow_overlay_row': selected_local_overlay_row,
        'selected_local_flow_overlay_dual_gate_row': (
            selected_local_overlay_dual_gate_row),
        'physical_confidence_rows': physical_confidence_rows,
        'selected_physical_confidence_row': (
            selected_physical_confidence_row),
        'selected_validation_metrics': selected_metrics,
        'persistence_validation_metrics': persistence_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_batch20'))
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_world_only_v1/validation'))
    parser.add_argument(
        '--thresholds', type=float, nargs='+',
        default=[value / 20.0 for value in range(1, 20)])
    parser.add_argument(
        '--change-gate-thresholds', type=float, nargs='+',
        default=[
            0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015,
            0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5,
        ])
    parser.add_argument(
        '--flow-fusion-thresholds', type=float, nargs='+',
        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument(
        '--physical-confidence-thresholds', type=float, nargs='+',
        default=[value / 20.0 for value in range(1, 20)])
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_world_only_validation_selection_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    sequence_mapping = _sequence_mapping(args.sequence_root)
    summary = select_checkpoint(
        manifest=manifest,
        sequence_mapping=sequence_mapping,
        prediction_root=args.prediction_root,
        thresholds=args.thresholds,
        change_gate_thresholds=args.change_gate_thresholds,
        flow_fusion_thresholds=args.flow_fusion_thresholds,
        physical_confidence_thresholds=(
            args.physical_confidence_thresholds))
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    concise = {
        'selected_epoch': summary['selected_epoch'],
        'selected_epoch_row': summary['selected_epoch_row'],
        'selected_visibility_threshold': (
            summary['selected_visibility_threshold']),
        'selected_threshold_row': summary['selected_threshold_row'],
        'selected_change_gate_threshold_row': (
            summary['selected_change_gate_threshold_row']),
        'selected_hard_change_gate_row': (
            summary['selected_hard_change_gate_row']),
        'selected_physical_flow_fusion_row': (
            summary['selected_physical_flow_fusion_row']),
        'selected_local_flow_overlay_row': (
            summary['selected_local_flow_overlay_row']),
        'selected_local_flow_overlay_dual_gate_row': (
            summary['selected_local_flow_overlay_dual_gate_row']),
        'selected_physical_confidence_row': (
            summary['selected_physical_confidence_row']),
        'out_file': str(args.out_file),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
