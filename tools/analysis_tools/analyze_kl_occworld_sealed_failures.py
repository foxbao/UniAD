#!/usr/bin/env python
"""Build per-scene diagnostics from existing OccWorld predictions."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
    evaluate_split,
)


SUBSETS = (
    'current_visible_subset',
    'reveal_completion_subset',
    'state_change_subset',
    'visible_transition_subset',
    'instance_related_visible_transition_subset',
)
CLASSES = ('free', 'static_occupied', 'instance_occupied')


def _future_mean(summary: dict) -> float:
    return float(np.mean([
        float(row['mean_iou']) for row in summary['by_horizon'][1:]
    ]))


def _future_class_mean(summary: dict, class_name: str) -> float:
    return float(np.mean([
        float(row['iou'][class_name])
        for row in summary['by_horizon'][1:]
    ]))


def _distribution(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        raise ValueError('Cannot summarize an empty distribution')
    return {
        'count': int(len(values)),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90)),
        'minimum': float(values.min()),
        'maximum': float(values.max()),
        'positive_count': int((values > 0).sum()),
        'negative_count': int((values < 0).sum()),
    }


def _rank(rows: list, key: str, count: int = 5,
          descending: bool = False) -> list:
    ordered = sorted(rows, key=lambda row: float(row[key]),
                     reverse=descending)
    return [
        {'reference_index': int(row['reference_index']),
         'scene_token': row['scene_token'], key: float(row[key])}
        for row in ordered[:count]
    ]


def _single_manifest(manifest: dict, split: str, record: dict) -> dict:
    result = dict(manifest)
    result['splits'] = {split: [record]}
    return result


def _metric_row(report: dict) -> dict:
    semantic = report['semantic']
    result = {
        'future_miou': _future_mean(semantic),
        'overall_miou': float(semantic['overall']['mean_iou']),
    }
    for subset in SUBSETS:
        result[subset.replace('_subset', '') + '_future_miou'] = (
            _future_mean(semantic[subset]))
    for class_name in CLASSES:
        result[f'{class_name}_future_iou'] = _future_class_mean(
            semantic, class_name)
    result['instance_h1_iou'] = float(
        semantic['by_horizon'][1]['iou']['instance_occupied'])
    visibility = report['visibility']['overall']
    for key in ('precision', 'recall', 'f1'):
        result[f'visibility_{key}'] = float(visibility[key])
    counts = report['subset_counts_by_horizon']
    for key, values in counts.items():
        result[f'future_{key}_count'] = int(sum(values[1:]))
    return result


def analyze(manifest: dict, split: str, sequence_mapping: dict,
            prediction_mapping: dict,
            visibility_threshold: float) -> dict:
    records = manifest['splits'][split]
    rows = []
    for record in records:
        reference = int(record['reference_index'])
        single = _single_manifest(manifest, split, record)
        model_report = evaluate_split(
            manifest=single,
            split=split,
            sequence_mapping={reference: sequence_mapping[reference]},
            prediction_mapping={reference: prediction_mapping[reference]},
            visibility_threshold=visibility_threshold)
        persistence_report = evaluate_split(
            manifest=single,
            split=split,
            sequence_mapping={reference: sequence_mapping[reference]},
            prediction_mapping=None,
            visibility_threshold=visibility_threshold)
        model = _metric_row(model_report)
        persistence = _metric_row(persistence_report)
        row = {
            'reference_index': reference,
            'scene_token': str(record['scene_token']),
        }
        for key, value in model.items():
            row[f'model_{key}'] = value
            row[f'persistence_{key}'] = persistence[key]
            if not key.endswith('_count'):
                row[f'delta_{key}'] = value - persistence[key]
        rows.append(row)

    diagnostic_keys = (
        'delta_future_miou',
        'delta_current_visible_future_miou',
        'delta_reveal_completion_future_miou',
        'delta_instance_occupied_future_iou',
        'delta_instance_h1_iou',
        'delta_visibility_precision',
        'delta_visibility_recall',
        'delta_visibility_f1',
    )
    distributions = {
        key: _distribution(row[key] for row in rows)
        for key in diagnostic_keys
    }
    rank_count = min(5, len(rows))
    rankings = {
        'lowest_future_miou_delta': _rank(
            rows, 'delta_future_miou', rank_count),
        'lowest_current_visible_delta': _rank(
            rows, 'delta_current_visible_future_miou', rank_count),
        'lowest_short_instance_delta': _rank(
            rows, 'delta_instance_h1_iou', rank_count),
        'lowest_visibility_precision_delta': _rank(
            rows, 'delta_visibility_precision', rank_count),
        'highest_reveal_gain': _rank(
            rows, 'delta_reveal_completion_future_miou',
            rank_count, descending=True),
    }
    return {
        'reference_count': len(rows),
        'rows': rows,
        'distributions': distributions,
        'rankings': rankings,
    }


def _write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
            'occworld_predictions_b17a_full_history_validation_v1/'
            'validation/epoch_003'))
    parser.add_argument('--split', default='validation')
    parser.add_argument('--visibility-threshold', type=float, default=0.7)
    parser.add_argument(
        '--purpose', choices=('development_validation', 'sealed_reporting'),
        default='development_validation')
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b17a_validation_failure_analysis_v1.json'))
    parser.add_argument(
        '--csv-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b17a_validation_failure_analysis_v1.csv'))
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.purpose == 'development_validation' and
            args.split != 'validation'):
        raise ValueError(
            'Development diagnostics are restricted to validation')
    manifest = _load_manifest(args.manifest)
    sequences = _sequence_mapping(args.sequence_root)
    predictions = _prediction_mapping(args.prediction_root)
    expected = {
        int(row['reference_index'])
        for row in manifest['splits'][args.split]
    }
    if not expected.issubset(sequences):
        raise ValueError('Diagnostic labels are incomplete')
    if set(predictions) != expected:
        raise ValueError('Diagnostic predictions are not the exact split')
    result = analyze(
        manifest, args.split, sequences, predictions,
        args.visibility_threshold)
    report = {
        'schema_version': 1,
        'name': args.out_file.stem,
        'split': args.split,
        'purpose': args.purpose,
        'model_inference_performed': False,
        'threshold_scan_performed': False,
        'visibility_threshold': args.visibility_threshold,
        'manifest': str(args.manifest),
        'sequence_root': str(args.sequence_root),
        'prediction_root': str(args.prediction_root),
        'development_use_allowed': (
            args.purpose == 'development_validation'),
        **result,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    _write_csv(args.csv_file, report['rows'])
    print(json.dumps({
        key: report[key] for key in (
            'name', 'split', 'purpose', 'reference_count',
            'distributions', 'rankings')
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
