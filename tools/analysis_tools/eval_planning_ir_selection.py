#!/usr/bin/env python
"""Evaluate Planning-IR candidate reselection without model inference."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.analysis_tools.planning_ir_schema import (
    PlanningIRValidationError,
    validate_planning_ir,
)
from tools.analysis_tools.planning_ir_audit_utils import read_jsonl


STRATEGIES = ('d2', 'fallback', 'teacher', 'audit_oracle')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare D2, fallback, teacher, and audit oracle.')
    parser.add_argument('--audit-jsonl', required=True)
    parser.add_argument('--teacher-jsonl', required=True)
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def new_accumulator():
    return dict(
        frames=0,
        mean_l2_sum=0.0,
        mean_l2_count=0,
        horizon_l2_sum=np.zeros(3, dtype=np.float64),
        horizon_l2_count=np.zeros(3, dtype=np.int64),
        collision_sum=np.zeros(3, dtype=np.int64),
        collision_count=np.zeros(3, dtype=np.int64),
        map_selected=0,
        useful_map_selected=0,
    )


def update(accumulator, candidate, metrics, fallback_metrics):
    accumulator['frames'] += 1
    mean_l2 = metrics.get('mean_l2_m')
    if mean_l2 is not None:
        accumulator['mean_l2_sum'] += float(mean_l2)
        accumulator['mean_l2_count'] += 1
    for index, value in enumerate(metrics.get('horizon_l2_m', [])):
        if value is not None:
            accumulator['horizon_l2_sum'][index] += float(value)
            accumulator['horizon_l2_count'][index] += 1
    for index, value in enumerate(metrics.get('horizon_collision', [])):
        if value is not None:
            accumulator['collision_sum'][index] += int(bool(value))
            accumulator['collision_count'][index] += 1
    is_map = candidate.get('source') == 'map'
    accumulator['map_selected'] += int(is_map)
    fallback_l2 = fallback_metrics.get('mean_l2_m')
    accumulator['useful_map_selected'] += int(
        is_map and mean_l2 is not None and fallback_l2 is not None
        and float(mean_l2) < float(fallback_l2))


def divide(values, counts):
    return np.divide(
        values, counts, out=np.full(values.shape, np.nan, dtype=np.float64),
        where=counts > 0)


def finalize(accumulator):
    frames = accumulator['frames']
    return dict(
        frames=frames,
        mean_l2_m=(accumulator['mean_l2_sum']
                   / accumulator['mean_l2_count']
                   if accumulator['mean_l2_count'] else None),
        horizon_l2_m=[
            None if np.isnan(value) else float(value)
            for value in divide(
                accumulator['horizon_l2_sum'],
                accumulator['horizon_l2_count'])
        ],
        horizon_collision_rate=[
            None if np.isnan(value) else float(value)
            for value in divide(
                accumulator['collision_sum'].astype(np.float64),
                accumulator['collision_count'])
        ],
        map_selection_rate=(accumulator['map_selected'] / frames
                            if frames else None),
        useful_map_selection_rate=(
            accumulator['useful_map_selected'] / frames if frames else None),
    )


def teacher_choice(record, teacher_record):
    labels = record['audit_labels']
    fallback = int(labels['fallback_candidate_id'])
    if teacher_record is None or not teacher_record.get('valid', False):
        return fallback, True
    teacher_input = record['teacher_input']
    candidates = teacher_input['candidates']
    actor_ids = [actor['actor_id']
                 for actor in teacher_input.get('actors', [])]
    rule_ids = [rule['rule_id']
                for rule in teacher_input.get('semantic_rules', [])]
    try:
        normalized = validate_planning_ir(
            teacher_record.get('planning_ir'), candidates,
            actor_ids=actor_ids, rule_ids=rule_ids)
    except PlanningIRValidationError:
        return fallback, True
    return int(normalized['selected_candidate_id']), False


def main():
    args = parse_args()
    audits = read_jsonl(args.audit_jsonl)
    teacher_records = {
        int(record['result_index']): record
        for record in read_jsonl(args.teacher_jsonl)
    }
    groups = defaultdict(lambda: {
        strategy: new_accumulator() for strategy in STRATEGIES})
    invalid_teacher = 0
    evaluated = 0
    for record in audits:
        result_index = int(record['result_index'])
        labels = record['audit_labels']
        candidates = {
            int(candidate['candidate_id']): candidate
            for candidate in record['teacher_input']['candidates']
        }
        metrics = {
            int(candidate_id): value
            for candidate_id, value in labels['candidate_metrics'].items()
        }
        strategy_ids = dict(
            d2=int(labels['d2_selected_candidate_id']),
            fallback=int(labels['fallback_candidate_id']),
            audit_oracle=int(labels['audit_oracle_candidate_id']),
        )
        strategy_ids['teacher'], invalid = teacher_choice(
            record, teacher_records.get(result_index))
        invalid_teacher += int(invalid)
        fallback_metrics = metrics[strategy_ids['fallback']]
        bucket_keys = (
            'overall',
            f'motion:{labels.get("motion_bucket", "unknown")}',
            f'obstacle:{labels.get("obstacle_bucket", "unknown")}',
        )
        for strategy, candidate_id in strategy_ids.items():
            if candidate_id not in candidates or candidate_id not in metrics:
                raise KeyError(
                    f'Candidate {candidate_id} for {strategy} is absent from '
                    f'audit frame {result_index}')
            for bucket in bucket_keys:
                update(
                    groups[bucket][strategy], candidates[candidate_id],
                    metrics[candidate_id], fallback_metrics)
        evaluated += 1

    summary = {
        bucket: {
            strategy: finalize(accumulator)
            for strategy, accumulator in strategies.items()
        }
        for bucket, strategies in sorted(groups.items())
    }
    output = dict(
        evaluated_frames=evaluated,
        teacher_invalid_frames=invalid_teacher,
        teacher_invalid_rate=(invalid_teacher / evaluated
                              if evaluated else None),
        groups=summary,
    )
    print(
        f'Evaluated {evaluated} frames; teacher invalid/missing: '
        f'{invalid_teacher} ({100 * output["teacher_invalid_rate"]:.2f}%)')
    print('strategy       mean_L2      map_rate   useful_map   collision@3s')
    for strategy in STRATEGIES:
        row = summary['overall'][strategy]
        collision = row['horizon_collision_rate'][-1]
        print(
            f'{strategy:12s} '
            f'{row["mean_l2_m"] if row["mean_l2_m"] is not None else float("nan"):8.4f} '
            f'{100 * row["map_selection_rate"]:10.2f}% '
            f'{100 * row["useful_map_selection_rate"]:11.2f}% '
            f'{100 * collision if collision is not None else float("nan"):12.2f}%')
    if args.output_json:
        output_path = osp.abspath(args.output_json)
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
            handle.write('\n')


if __name__ == '__main__':
    main()
