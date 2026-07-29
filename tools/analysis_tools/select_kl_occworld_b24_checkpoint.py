#!/usr/bin/env python
"""Apply the frozen B24 internal-dev checkpoint selection protocol."""

import argparse
import json
from pathlib import Path


def _load(path):
    with Path(path).open(encoding='utf-8') as input_file:
        return json.load(input_file)


def _future_mean(summary, key, metric='mean_iou'):
    semantic = summary['semantic']
    section = semantic if key == 'semantic' else semantic[key]
    values = [float(row[metric]) for row in section['by_horizon'][1:]]
    if len(values) != 4:
        raise ValueError(f'Expected four future horizons for {key}')
    return sum(values) / len(values)


def _future_instance(summary):
    values = [
        float(row['iou']['instance_occupied'])
        for row in summary['semantic']['by_horizon'][1:]
    ]
    if len(values) != 4:
        raise ValueError('Expected four future instance horizons')
    return sum(values) / len(values)


def metrics(summary):
    return {
        'future_semantic_miou': _future_mean(summary, 'semantic'),
        'future_instance_iou': _future_instance(summary),
        'future_visible_transition_miou': _future_mean(
            summary, 'visible_transition_subset'),
        'future_instance_transition_miou': _future_mean(
            summary, 'instance_related_visible_transition_subset'),
    }


def select(candidates):
    rows = []
    for epoch, raw_evaluation, final_evaluation, audit in candidates:
        current_difference = int(audit['current_difference_voxels'])
        non_event_difference = int(audit['non_event_difference_voxels'])
        raw_metrics = metrics(raw_evaluation)
        final_metrics = metrics(final_evaluation)
        invariants_pass = (
            current_difference == 0 and non_event_difference == 0)
        non_regression_pass = all((
            final_metrics['future_semantic_miou'] >=
            raw_metrics['future_semantic_miou'],
            final_metrics['future_visible_transition_miou'] >=
            raw_metrics['future_visible_transition_miou'],
            final_metrics['future_instance_transition_miou'] >=
            raw_metrics['future_instance_transition_miou'],
        ))
        deltas = {
            f'{key}_delta': final_metrics[key] - raw_metrics[key]
            for key in final_metrics
        }
        rows.append({
            'epoch': int(epoch),
            'paired_raw': raw_metrics,
            'final': final_metrics,
            **deltas,
            'known_event_net_correct_voxels': int(
                audit['corrections']['net_correct_voxels']),
            'current_difference_voxels': current_difference,
            'non_event_difference_voxels': non_event_difference,
            'invariants_pass': invariants_pass,
            'non_regression_pass': non_regression_pass,
            'eligible': invariants_pass and non_regression_pass,
        })
    rows.sort(key=lambda row: (
        -row['final']['future_semantic_miou'],
        -row['known_event_net_correct_voxels'],
        -row['final']['future_instance_iou'],
        row['epoch']))
    eligible = [row for row in rows if row['eligible']]
    selected = None if not eligible else eligible[0]['epoch']
    return {
        'schema_version': 1,
        'protocol': (
            'invariants_then_future_semantic_miou_then_net_correct_then_'
            'future_instance_iou'),
        'baseline_contract': 'same_forward_paired_raw_per_checkpoint',
        'candidates_ranked': rows,
        'selected_epoch': selected,
        'exploratory_pass': selected is not None,
        'independent_generalization_evidence': False,
        'formal_promotion_requires_new_scene_validation': True,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--candidate', nargs=4, action='append', required=True,
        metavar=(
            'EPOCH', 'RAW_EVALUATION_JSON', 'FINAL_EVALUATION_JSON',
            'AUDIT_JSON'))
    parser.add_argument('--out-file', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = [
        (int(epoch), _load(raw_evaluation), _load(final_evaluation),
         _load(audit))
        for epoch, raw_evaluation, final_evaluation, audit in args.candidate
    ]
    summary = select(candidates)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
