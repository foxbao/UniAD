#!/usr/bin/env python
"""Validate and summarize generated KL OccWorld history queues."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.analysis_tools.audit_kl_occworld_dual_representation import _tile
from tools.analysis_tools.audit_kl_occworld_sequence_batch import _contact_page


def _sequence_index(sequence_root: Path) -> Dict[int, Path]:
    mapping = {}
    for path in sequence_root.glob('*/*__occworld_sequence.npz'):
        with np.load(path, allow_pickle=False) as sequence:
            mapping[int(sequence['reference_index'])] = path
    return mapping


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path, required=True)
    parser.add_argument('--sequence-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--contact-columns', type=int, default=3)
    parser.add_argument('--contact-rows-per-page', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.contact_columns < 1 or args.contact_rows_per_page < 1:
        raise ValueError('Contact grid dimensions must be positive')
    history_paths = sorted(
        args.history_root.glob('*/*__occworld_history.npz'))
    if not history_paths:
        raise FileNotFoundError(
            f'No history labels below {args.history_root}')
    sequence_map = _sequence_index(args.sequence_root)
    rows = []
    contact_images = []
    for history_path in history_paths:
        with np.load(history_path, allow_pickle=False) as history:
            reference_index = int(history['reference_index'])
            states = history['history_observation_state_3d']
            valid = history['history_observation_valid_3d']
            if states.ndim != 4 or valid.shape != states.shape:
                raise ValueError(f'Invalid history shape in {history_path}')
            if not np.array_equal(valid, states != 0):
                raise ValueError(
                    f'Invalid history valid mask in {history_path}')
            time_error = float(np.max(np.abs(
                history['history_times_s'] -
                history['nominal_history_times_s'])))
            last_transform_error = float(np.max(np.abs(
                history['history_to_reference'][-1] - np.eye(4))))
        if reference_index not in sequence_map:
            raise FileNotFoundError(
                f'No sequence target for reference {reference_index}')
        with np.load(
                sequence_map[reference_index],
                allow_pickle=False) as sequence:
            last_equals_direct = bool(np.array_equal(
                states[-1], sequence['direct_observation_state_3d'][0]))
        if not last_equals_direct:
            raise ValueError(
                f'History current mismatch for reference {reference_index}')
        row = {
            'reference_index': reference_index,
            'history_shape': 'x'.join(str(value) for value in states.shape),
            'max_time_error_s': time_error,
            'last_transform_error': last_transform_error,
            'last_equals_direct_t0': last_equals_direct,
            'mean_known_voxels': float(np.mean(
                np.count_nonzero(states, axis=(1, 2, 3)))),
            'history_path': str(history_path),
        }
        rows.append(row)
        contact_path = next(
            history_path.parent.glob('*__history_contact_sheet.png'))
        image = cv2.imread(str(contact_path))
        if image is None:
            raise FileNotFoundError(contact_path)
        contact_images.append(_tile(
            image,
            f'#{reference_index} | dt_err={time_error:.3f}s',
            (750, 128)))
        print(
            f'[{reference_index}] shape={row["history_shape"]} '
            f'dt_err={time_error:.3f}s')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / 'history_batch_metrics.csv', rows)
    items_per_page = args.contact_columns * args.contact_rows_per_page
    contact_paths = []
    for page_index, start in enumerate(
            range(0, len(contact_images), items_per_page), start=1):
        page_path = args.out_dir / (
            f'history_batch_contact_sheet_{page_index:02d}.png')
        cv2.imwrite(str(page_path), _contact_page(
            contact_images[start:start + items_per_page],
            args.contact_columns))
        contact_paths.append(str(page_path))
    summary = {
        'history_count': len(rows),
        'reference_indices': [row['reference_index'] for row in rows],
        'history_shape': rows[0]['history_shape'],
        'max_time_error_s': max(row['max_time_error_s'] for row in rows),
        'max_last_transform_error': max(
            row['last_transform_error'] for row in rows),
        'all_last_equal_direct_t0': all(
            row['last_equals_direct_t0'] for row in rows),
        'contact_sheet_paths': contact_paths,
    }
    with (args.out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
