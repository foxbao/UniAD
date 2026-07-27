#!/usr/bin/env python
"""Select B20 query-residual checkpoints on development validation only."""

import argparse
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
)


def _future_mean(report: dict, subset: str = None,
                 class_name: str = None) -> float:
    semantic = report['semantic']
    rows = semantic['by_horizon'] if subset is None else (
        semantic[subset]['by_horizon'])
    if class_name is None:
        values = [float(row['mean_iou']) for row in rows[1:]]
    else:
        values = [float(row['iou'][class_name]) for row in rows[1:]]
    return float(np.mean(values))


def report_metrics(report: dict) -> dict:
    return {
        'future_semantic_miou': _future_mean(report),
        'future_instance_iou': _future_mean(
            report, class_name='instance_occupied'),
        'current_visible_miou': _future_mean(
            report, subset='current_visible_subset'),
        'visible_transition_miou': _future_mean(
            report, subset='visible_transition_subset'),
        'instance_transition_miou': _future_mean(
            report,
            subset='instance_related_visible_transition_subset'),
    }


def correction_counts(candidate: np.ndarray, baseline: np.ndarray,
                      target: np.ndarray, valid: np.ndarray) -> dict:
    if not (candidate.shape == baseline.shape == target.shape == valid.shape):
        raise ValueError('Correction tensors must have identical shapes')
    changed = valid & (candidate != baseline)
    candidate_correct = candidate == target
    baseline_correct = baseline == target
    improved = changed & candidate_correct & ~baseline_correct
    harmed = changed & ~candidate_correct & baseline_correct
    return {
        'changed_voxels': int(np.count_nonzero(changed)),
        'improved_voxels': int(np.count_nonzero(improved)),
        'harmed_voxels': int(np.count_nonzero(harmed)),
        'net_correct_voxels': int(
            np.count_nonzero(improved) - np.count_nonzero(harmed)),
    }


def acceptance_checks(result: dict) -> dict:
    delta = result['delta_percentage_points']
    checks = {
        'exact_scale_zero_semantic_parity': (
            result['external_b17a_replay_parity'][
                'semantic_difference_voxels_vs_b17a'] == 0),
        'exact_scale_zero_visibility_parity': (
            result['external_b17a_replay_parity'][
                'visibility_max_abs_difference_vs_b17a'] == 0.0),
        'positive_net_correct_voxels': (
            result['corrections_vs_scale_zero'][
                'net_correct_voxels'] > 0),
        'future_instance_iou_gain_at_least_0_10pp': (
            delta['future_instance_iou'] >= 0.10),
        'visible_transition_not_lower': (
            delta['visible_transition_miou'] >= 0.0),
        'instance_transition_not_lower': (
            delta['instance_transition_miou'] >= 0.0),
        'future_semantic_drop_at_most_0_05pp': (
            delta['future_semantic_miou'] >= -0.05),
        'current_visible_drop_at_most_0_05pp': (
            delta['current_visible_miou'] >= -0.05),
    }
    checks['all_passed'] = all(checks.values())
    return checks


def _load_prediction(path: Path,
                     class_key: str = 'world_pred_class_3d') -> tuple:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload[class_key], dtype=np.uint8),
            np.asarray(
                payload['world_valid_probability_3d'], dtype=np.float32),
        )


def analyze_epoch(epoch: int, references: list, sequence_mapping: dict,
                  baseline_mapping: dict, candidate_mapping: dict,
                  ablation_mapping: dict, candidate_report: dict,
                  ablation_report: dict) -> dict:
    semantic_difference = 0
    visibility_max_difference = 0.0
    candidate_ablation_visibility_max_difference = 0.0
    correction_total = {
        'changed_voxels': 0,
        'improved_voxels': 0,
        'harmed_voxels': 0,
        'net_correct_voxels': 0,
    }
    for reference in references:
        baseline, baseline_visibility = _load_prediction(
            baseline_mapping[reference])
        candidate, candidate_visibility = _load_prediction(
            candidate_mapping[reference])
        ablation, ablation_visibility = _load_prediction(
            ablation_mapping[reference],
            class_key='query_adapter_ablation_world_pred_class_3d')
        if not (baseline.shape == candidate.shape == ablation.shape):
            raise ValueError('B20 semantic prediction shapes do not match')
        semantic_difference += int(np.count_nonzero(ablation != baseline))
        visibility_max_difference = max(
            visibility_max_difference,
            float(np.max(np.abs(
                ablation_visibility - baseline_visibility))))
        candidate_ablation_visibility_max_difference = max(
            candidate_ablation_visibility_max_difference,
            float(np.max(np.abs(
                candidate_visibility - ablation_visibility))))
        with np.load(
                sequence_mapping[reference], allow_pickle=False) as label:
            state = np.asarray(label['world_target_state_3d'], dtype=np.uint8)
            target_valid = np.asarray(
                label['world_target_valid_3d'], dtype=bool)
        target = np.zeros_like(state, dtype=np.uint8)
        known = target_valid & (state > 0)
        target[known] = state[known] - 1
        counts = correction_counts(
            candidate[1:], ablation[1:], target[1:], known[1:])
        for key, value in counts.items():
            correction_total[key] += value

    candidate_metrics = report_metrics(candidate_report)
    ablation_metrics = report_metrics(ablation_report)
    result = {
        'epoch': int(epoch),
        'paired_ablation_source': 'same_forward_pass_pre_adapter_logits',
        'external_b17a_replay_parity': {
            'semantic_difference_voxels_vs_b17a': semantic_difference,
            'visibility_max_abs_difference_vs_b17a': (
                visibility_max_difference),
        },
        'candidate_ablation_visibility_max_abs_difference': (
            candidate_ablation_visibility_max_difference),
        'candidate_metrics': candidate_metrics,
        'scale_zero_metrics': ablation_metrics,
        'delta_percentage_points': {
            key: float((candidate_metrics[key] - ablation_metrics[key]) * 100)
            for key in candidate_metrics
        },
        'corrections_vs_scale_zero': correction_total,
    }
    result['acceptance_checks'] = acceptance_checks(result)
    return result


def _load_json(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


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
        '--baseline-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b19_query_diagnostic_validation_v1/'
            'validation/epoch_003'))
    parser.add_argument(
        '--candidate-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b20_query_residual_validation_v1/'
            'validation'))
    parser.add_argument('--epochs', type=int, nargs='+', default=(1, 2))
    parser.add_argument(
        '--report-root', type=Path,
        default=Path('documents/patent_2026_occ'))
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b20_query_residual_validation_selection_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    references = [
        int(record['reference_index'])
        for record in manifest['splits']['validation']
    ]
    expected = set(references)
    sequence_mapping = _sequence_mapping(args.sequence_root)
    baseline_mapping = _prediction_mapping(args.baseline_root)
    if set(baseline_mapping) != expected:
        raise ValueError('B17A baseline is not the exact validation split')

    results = []
    for epoch in args.epochs:
        candidate_mapping = _prediction_mapping(
            args.candidate_root / f'epoch_{epoch:03d}')
        ablation_mapping = candidate_mapping
        if set(candidate_mapping) != expected:
            raise ValueError('B20 predictions are not the exact validation split')
        candidate_report = _load_json(
            args.report_root /
            f'kl_occworld_b20_query_residual_epoch{epoch}_validation_v1.json')
        ablation_report = _load_json(
            args.report_root /
            'kl_occworld_b20_query_residual_paired_zero_'
            f'epoch{epoch}_validation_v1.json')
        results.append(analyze_epoch(
            epoch, references, sequence_mapping, baseline_mapping,
            candidate_mapping, ablation_mapping,
            candidate_report, ablation_report))

    qualifying = [
        result for result in results
        if result['acceptance_checks']['all_passed']
    ]
    qualifying.sort(
        key=lambda result: (
            result['delta_percentage_points']['future_instance_iou'],
            result['corrections_vs_scale_zero']['net_correct_voxels'],
            result['delta_percentage_points']['future_semantic_miou']),
        reverse=True)
    selected_epoch = None if not qualifying else qualifying[0]['epoch']
    report = {
        'schema_version': 1,
        'name': args.out_file.stem,
        'split': 'validation',
        'purpose': 'development_validation',
        'holdout_accessed': False,
        'comparison_mode': 'paired_same_forward_pass',
        'epochs': results,
        'decision': {
            'status': (
                'completed_not_promoted' if selected_epoch is None
                else 'validation_candidate'),
            'selected_epoch': selected_epoch,
            'holdout_evaluation_authorized': False,
        },
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
