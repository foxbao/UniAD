#!/usr/bin/env python
"""Audit KL reference frames for fixed-horizon OccWorld sequences."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _resolve_path,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    _same_scene_indices,
    _validate_fixed_frame_times,
)


def _dual_reference_indices(dual_dir: Path) -> List[int]:
    indices = []
    for path in dual_dir.glob('*__cross_scene.npz'):
        with np.load(path, allow_pickle=False) as label:
            indices.append(int(label['frame_index']))
    return sorted(set(indices))


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--dual-dir')
    parser.add_argument('--reference-indices', type=int, nargs='*')
    parser.add_argument('--target-offsets', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--reveal-offsets', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--expected-step-s', type=float, default=0.5)
    parser.add_argument('--max-time-error-s', type=float, default=0.2)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reference_indices:
        indices = sorted(set(args.reference_indices))
    elif args.dual_dir:
        indices = _dual_reference_indices(Path(args.dual_dir))
    else:
        raise ValueError('Provide --reference-indices or --dual-dir')
    infos, _ = _load_infos(_resolve_path(args.ann_file))
    required_offsets = sorted(set(
        target + reveal
        for target in args.target_offsets
        for reveal in args.reveal_offsets))
    rows = []
    for index in indices:
        reason = ''
        status = 'eligible'
        try:
            _same_scene_indices(infos, index, required_offsets)
            _validate_fixed_frame_times(
                infos, index, args.target_offsets, args.reveal_offsets,
                args.expected_step_s, args.max_time_error_s)
        except ValueError as error:
            reason = str(error)
            status = (
                'scene_boundary' if 'crosses the reference scene' in reason
                else 'irregular_timestamp')
        actual_times = []
        reference_time = float(infos[index]['timestamp'])
        for offset in args.target_offsets:
            target_index = index + offset
            if (target_index < len(infos) and
                    infos[target_index].get('scene_token') ==
                    infos[index].get('scene_token')):
                actual_times.append(
                    float(infos[target_index]['timestamp']) - reference_time)
            else:
                actual_times.append(float('nan'))
        row = {
            'frame_index': index,
            'scene_token': infos[index].get('scene_token', ''),
            'status': status,
            'actual_target_times_s': ';'.join(
                'nan' if not np.isfinite(value) else f'{value:.3f}'
                for value in actual_times),
            'reason': reason,
        }
        rows.append(row)
        print(
            f'[{index}] {status} times={row["actual_target_times_s"]}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / 'sequence_eligibility.csv', rows)
    status_counts = {
        status: sum(row['status'] == status for row in rows)
        for status in ('eligible', 'scene_boundary', 'irregular_timestamp')
    }
    summary = {
        'reference_count': len(rows),
        'status_counts': status_counts,
        'eligible_indices': [
            row['frame_index'] for row in rows
            if row['status'] == 'eligible'],
        'rejected_indices': [
            row['frame_index'] for row in rows
            if row['status'] != 'eligible'],
        'target_offsets': args.target_offsets,
        'reveal_offsets': args.reveal_offsets,
        'expected_step_s': args.expected_step_s,
        'max_time_error_s': args.max_time_error_s,
    }
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
