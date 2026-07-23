#!/usr/bin/env python
"""Audit KL OccWorld references for a fixed-time history queue."""

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
    _validate_frame_offsets,
)


def _reference_indices(summary_path: Path,
                       explicit_indices: List[int]) -> List[int]:
    if explicit_indices:
        return sorted(set(explicit_indices))
    if not summary_path:
        raise ValueError(
            'Provide --reference-indices or --sequence-eligibility-summary')
    with summary_path.open() as f:
        summary = json.load(f)
    return sorted(set(int(index) for index in summary['eligible_indices']))


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--sequence-eligibility-summary', type=Path)
    parser.add_argument('--reference-indices', type=int, nargs='*', default=[])
    parser.add_argument('--history-offsets', type=int, nargs='+',
                        default=[-4, -3, -2, -1, 0])
    parser.add_argument('--expected-step-s', type=float, default=0.5)
    parser.add_argument('--max-time-error-s', type=float, default=0.2)
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    offsets = sorted(set(args.history_offsets))
    if not offsets or offsets[-1] != 0 or any(offset > 0 for offset in offsets):
        raise ValueError(
            'history-offsets must end at 0 and contain no future offsets')
    references = _reference_indices(
        args.sequence_eligibility_summary, args.reference_indices)
    infos, _ = _load_infos(_resolve_path(args.ann_file))
    rows = []
    for index in references:
        status = 'eligible'
        reason = ''
        try:
            _same_scene_indices(infos, index, offsets)
            _validate_frame_offsets(
                infos, index, offsets, args.expected_step_s,
                args.max_time_error_s, window_name='history')
        except ValueError as error:
            reason = str(error)
            status = (
                'irregular_timestamp'
                if 'irregular history timestamp' in reason
                else 'history_boundary')
        reference_time = float(infos[index]['timestamp'])
        actual_times = []
        for offset in offsets:
            history_index = index + offset
            if (0 <= history_index < len(infos) and
                    infos[history_index].get('scene_token') ==
                    infos[index].get('scene_token')):
                actual_times.append(
                    float(infos[history_index]['timestamp']) - reference_time)
            else:
                actual_times.append(float('nan'))
        row = {
            'frame_index': index,
            'scene_token': infos[index].get('scene_token', ''),
            'status': status,
            'actual_history_times_s': ';'.join(
                'nan' if not np.isfinite(value) else f'{value:.3f}'
                for value in actual_times),
            'reason': reason,
        }
        rows.append(row)
        print(
            f'[{index}] {status} times='
            f'{row["actual_history_times_s"]}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / 'history_eligibility.csv', rows)
    status_counts = {
        status: sum(row['status'] == status for row in rows)
        for status in ('eligible', 'history_boundary',
                       'irregular_timestamp')
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
        'history_offsets': offsets,
        'expected_step_s': args.expected_step_s,
        'max_time_error_s': args.max_time_error_s,
    }
    with (args.out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
