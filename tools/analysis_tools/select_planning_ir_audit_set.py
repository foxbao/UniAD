#!/usr/bin/env python
"""Select a deterministic motion/obstacle-balanced Planning-IR audit set."""

from __future__ import annotations

import argparse
import os.path as osp
import random
import sys
from collections import defaultdict


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.analysis_tools.planning_ir_audit_utils import read_jsonl, write_jsonl


DEFAULT_MOTION_BUCKETS = (
    'static', 'slow', 'moving_straight', 'turning')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a balanced Planning-IR P0 audit subset.')
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--per-motion-bucket', type=int, default=50)
    parser.add_argument('--motion-buckets', nargs='+',
                        default=list(DEFAULT_MOTION_BUCKETS))
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def balanced_obstacle_sample(records, count, rng):
    by_obstacle = defaultdict(list)
    for record in records:
        bucket = record['audit_labels'].get('obstacle_bucket', 'unknown')
        by_obstacle[bucket].append(record)
    for rows in by_obstacle.values():
        rng.shuffle(rows)
    obstacle_order = sorted(by_obstacle)
    selected = []
    while len(selected) < count:
        added = False
        for obstacle_bucket in obstacle_order:
            rows = by_obstacle[obstacle_bucket]
            if rows:
                selected.append(rows.pop())
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
    return selected


def main():
    args = parse_args()
    if args.per_motion_bucket <= 0:
        raise ValueError('--per-motion-bucket must be positive')
    records = read_jsonl(args.input_jsonl)
    rng = random.Random(args.seed)
    selected = []
    for motion_bucket in args.motion_buckets:
        rows = [
            record for record in records
            if record['audit_labels'].get('motion_bucket') == motion_bucket
        ]
        if len(rows) < args.per_motion_bucket:
            raise ValueError(
                f'{motion_bucket} has only {len(rows)} records, fewer than '
                f'the requested {args.per_motion_bucket}')
        bucket_selection = balanced_obstacle_sample(
            rows, args.per_motion_bucket, rng)
        selected.extend(bucket_selection)
        obstacle_counts = defaultdict(int)
        for record in bucket_selection:
            obstacle_counts[
                record['audit_labels'].get('obstacle_bucket', 'unknown')] += 1
        print(
            f'{motion_bucket}: selected {len(bucket_selection)} '
            f'{dict(sorted(obstacle_counts.items()))}')
    rng.shuffle(selected)
    write_jsonl(args.output_jsonl, selected)
    print(f'Wrote {len(selected)} balanced records to {args.output_jsonl}')


if __name__ == '__main__':
    main()
