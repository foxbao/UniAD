#!/usr/bin/env python
"""Summarize a fixed-protocol OccWorld current-anchor substitution run."""

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


def _future_mean(report: dict, subset: str = 'semantic') -> float:
    semantic = report['semantic']
    by_horizon = (
        semantic['by_horizon'] if subset == 'semantic'
        else semantic[subset]['by_horizon'])
    if len(by_horizon) < 2:
        raise ValueError(f'{subset} has no future horizons')
    return float(np.mean([
        float(row['mean_iou']) for row in by_horizon[1:]
    ]))


def _summary_row(report: dict) -> dict:
    semantic = report['semantic']
    return {
        'overall_mean_iou': float(semantic['overall']['mean_iou']),
        'future_mean_iou': _future_mean(report),
        'future_reveal_mean_iou': _future_mean(
            report, subset='reveal_completion_subset'),
        'future_state_change_mean_iou': _future_mean(
            report, subset='state_change_subset'),
        'future_visible_transition_mean_iou': _future_mean(
            report, subset='visible_transition_subset'),
        'future_instance_related_visible_transition_mean_iou': (
            _future_mean(report, subset='instance_related_visible_transition_subset')),
        'visibility_f1': float(report['visibility']['overall']['f1']),
    }


def _delta(candidate: dict, baseline: dict) -> dict:
    return {
        key: float(candidate[key] - baseline[key])
        for key in baseline
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--anchor-audit', type=Path, required=True)
    parser.add_argument('--out-file', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_report = _load(args.baseline)
    candidate_report = _load(args.candidate)
    anchor_audit = _load(args.anchor_audit)
    baseline = _summary_row(baseline_report)
    candidate = _summary_row(candidate_report)
    summary = {
        'schema_version': 1,
        'purpose': (
            'Compare fixed validation predictions after replacing only the '
            'current B15 observation anchor.'),
        'baseline': baseline,
        'candidate': candidate,
        'candidate_minus_baseline': _delta(candidate, baseline),
        'anchor_audit': {
            'reference_count': int(anchor_audit['reference_count']),
            **anchor_audit['aggregate'],
        },
        'scope': (
            'Current-frame TrackFormer boxes replace annotation boxes; '
            'history still uses the B15 annotation-derived inputs.'),
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
