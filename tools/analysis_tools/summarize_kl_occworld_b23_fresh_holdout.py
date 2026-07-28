#!/usr/bin/env python
"""Seal the read-only B23 fresh-holdout generalization decision."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _read_json(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict:
    return {'path': str(path), 'sha256': _sha256(path)}


def _future_mean(rows: list, value_getter) -> float:
    return float(np.mean([value_getter(row) for row in rows[1:]]))


def _metrics(report: dict) -> dict:
    semantic = report['semantic']
    mean_iou = lambda row: float(row['mean_iou'])
    instance_iou = lambda row: float(row['iou']['instance_occupied'])
    return {
        'overall_miou': float(semantic['overall']['mean_iou']),
        'future_miou': _future_mean(
            semantic['by_horizon'], mean_iou),
        'future_instance_iou': _future_mean(
            semantic['by_horizon'], instance_iou),
        'future_current_visible_miou': _future_mean(
            semantic['current_visible_subset']['by_horizon'], mean_iou),
        'future_reveal_miou': _future_mean(
            semantic['reveal_completion_subset']['by_horizon'], mean_iou),
        'future_state_change_miou': _future_mean(
            semantic['state_change_subset']['by_horizon'], mean_iou),
        'future_visible_transition_miou': _future_mean(
            semantic['visible_transition_subset']['by_horizon'], mean_iou),
        'future_instance_transition_miou': _future_mean(
            semantic['instance_related_visible_transition_subset'][
                'by_horizon'], mean_iou),
        'visibility_f1': (
            None if report['visibility'] is None
            else float(report['visibility']['overall']['f1'])),
    }


def _delta_pp(candidate: dict, baseline: dict) -> dict:
    return {
        key: (
            None if candidate[key] is None or baseline[key] is None
            else float((candidate[key] - baseline[key]) * 100.0))
        for key in candidate
    }


def build_summary(args) -> dict:
    manifest = _read_json(args.evaluation_manifest)
    model = _read_json(args.model_report)
    raw = _read_json(args.raw_report)
    persistence = _read_json(args.persistence_report)
    overlay = _read_json(args.overlay_audit)
    parity = _read_json(args.parity_report)
    if manifest.get('status') != (
            'fresh_holdout_evaluated_once_no_retuning_allowed'):
        raise ValueError('Fresh holdout is not sealed')
    expected = manifest['prediction_artifact_audit']['reference_indices']
    for name, report in (
            ('model', model), ('raw', raw), ('persistence', persistence)):
        if report.get('reference_indices') != expected:
            raise ValueError(f'{name} report changed reference order')
        if int(report.get('sample_count', -1)) != len(expected):
            raise ValueError(f'{name} report changed sample count')
    if parity.get('passed') is not True:
        raise ValueError('Fresh-holdout model/oracle parity failed')

    metrics = {
        'b23': _metrics(model),
        'raw': _metrics(raw),
        'persistence': _metrics(persistence),
    }
    checks = overlay['qualification']['checks']
    promoted = all(bool(value) for value in checks.values())
    return {
        'schema_version': 1,
        'name': 'kl_occworld_b23_fresh_holdout_decision_v1',
        'status': (
            'fresh_generalization_promoted'
            if promoted else
            'fresh_generalization_rejected_no_retuning'),
        'reference_count': len(expected),
        'protocol': {
            'candidate': 'B23 raw-free motion actor arrival overlay',
            'checkpoint_epoch': 3,
            'visibility_threshold': 0.7,
            'motion_actor_score_threshold': 0.1,
            'raw_class_gate': 0,
            'local_flow_overlay_enabled': False,
            'evaluator_post_overlay': False,
            'threshold_scan_performed': False,
            'additional_model_inference_allowed': False,
        },
        'metrics': metrics,
        'deltas_percentage_points': {
            'b23_vs_raw': _delta_pp(metrics['b23'], metrics['raw']),
            'b23_vs_persistence': _delta_pp(
                metrics['b23'], metrics['persistence']),
        },
        'overlay_corrections_vs_raw': overlay['corrections_vs_raw'],
        'overlay_preservation': overlay['preservation'],
        'predeclared_qualification': overlay['qualification'],
        'model_oracle_parity': {
            key: parity[key] for key in (
                'reference_count', 'mask_difference_voxels',
                'prediction_difference_voxels', 'passed')
        },
        'decision': {
            'promoted': promoted,
            'current_model_after_decision': 'B17A epoch 3 raw prediction',
            'reason': (
                'Dynamic transition subsets improve, but net corrected '
                'voxels and future instance IoU fail the frozen promotion '
                'criteria on unseen scenes.'),
            'retuning_on_this_holdout_allowed': False,
        },
        'evidence': {
            'evaluation_manifest': _artifact(args.evaluation_manifest),
            'model_report': _artifact(args.model_report),
            'raw_report': _artifact(args.raw_report),
            'persistence_report': _artifact(args.persistence_report),
            'overlay_audit': _artifact(args.overlay_audit),
            'parity_report': _artifact(args.parity_report),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('documents/patent_2026_occ')
    parser.add_argument(
        '--evaluation-manifest', type=Path,
        default=root / 'kl_occworld_b23_fresh_holdout_evaluation_v2.json')
    parser.add_argument(
        '--model-report', type=Path,
        default=root / 'kl_occworld_b23_fresh_holdout_model_v1.json')
    parser.add_argument(
        '--raw-report', type=Path,
        default=root / 'kl_occworld_b23_fresh_holdout_raw_v1.json')
    parser.add_argument(
        '--persistence-report', type=Path,
        default=root / 'kl_occworld_b23_fresh_holdout_persistence_v1.json')
    parser.add_argument(
        '--overlay-audit', type=Path,
        default=root / 'kl_occworld_b23_fresh_holdout_overlay_audit_v1.json')
    parser.add_argument(
        '--parity-report', type=Path,
        default=root / 'kl_occworld_b23_fresh_holdout_overlay_parity_v1.json')
    parser.add_argument(
        '--out-file', type=Path,
        default=root / 'kl_occworld_b23_fresh_holdout_decision_v1.json')
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_summary(args)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'status': summary['status'],
        'promoted': summary['decision']['promoted'],
        'out_file': str(args.out_file),
        'sha256': _sha256(args.out_file),
        'deltas_percentage_points': summary['deltas_percentage_points'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
