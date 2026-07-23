#!/usr/bin/env python
"""Aggregate and validate generated KL OccWorld sequence labels."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.analysis_tools.audit_kl_occworld_dual_representation import _tile


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _contact_page(images: List[np.ndarray], columns: int) -> np.ndarray:
    """Arrange equal-sized sequence summaries into a compact image grid."""
    if not images or columns < 1:
        raise ValueError('A contact page needs images and positive columns')
    height, width = images[0].shape[:2]
    padded = list(images)
    while len(padded) % columns:
        padded.append(np.full((height, width, 3), 28, dtype=np.uint8))
    return np.concatenate([
        np.concatenate(padded[start:start + columns], axis=1)
        for start in range(0, len(padded), columns)
    ], axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root', required=True)
    parser.add_argument('--eligibility-summary')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--contact-columns', type=int, default=2)
    parser.add_argument('--contact-rows-per-page', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    sequence_root = Path(args.sequence_root)
    discovered = sorted(sequence_root.glob('*/*__occworld_sequence.npz'))
    rejected_artifacts = [
        str(path) for path in discovered
        if path.parent.name.startswith('rejected_')]
    label_paths = [
        path for path in discovered
        if not path.parent.name.startswith('rejected_')]
    if not label_paths:
        raise RuntimeError(f'No sequence labels found in {sequence_root}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    contact_rows = []
    for label_path in label_paths:
        summary_path = label_path.parent / 'summary.json'
        with summary_path.open() as f:
            frame_summary = json.load(f)
        with np.load(label_path, allow_pickle=False) as label:
            direct = label['direct_observation_state_3d']
            world = label['world_target_state_3d']
            source = label['completion_source_3d']
            if direct.shape != world.shape or direct.ndim != 4:
                raise ValueError(f'Invalid sequence shape in {label_path}')
            if not np.all(world[direct != 0] == direct[direct != 0]):
                raise ValueError(f'Direct state changed in {label_path}')
            if not np.all(direct[source >= 2] == 0):
                raise ValueError(f'Completion overwrote direct state: {label_path}')
            actual_times = label['target_times_s']
            nominal_times = label['nominal_target_times_s']
            max_time_error = float(np.max(np.abs(
                actual_times - nominal_times)))
            valid_ratio = float(
                label['world_target_valid_3d'].mean())
            sequence_shape = list(world.shape)
        target_rows = frame_summary['targets']
        row = {
            'reference_index': int(frame_summary['reference_index']),
            'stem': frame_summary['stem'],
            'sequence_shape': 'x'.join(str(value) for value in sequence_shape),
            'max_target_time_error_s': max_time_error,
            'mean_direct_known_voxels': float(np.mean([
                item['direct_known_voxels'] for item in target_rows])),
            'mean_world_known_voxels': float(np.mean([
                item['world_known_voxels'] for item in target_rows])),
            'future_free_filled_voxels': sum(
                item['future_free_filled_voxels'] for item in target_rows),
            'future_static_filled_voxels': sum(
                item['future_static_filled_voxels'] for item in target_rows),
            'world_valid_ratio': valid_ratio,
            'label_bytes': label_path.stat().st_size,
            'label_path': str(label_path),
        }
        rows.append(row)
        contact_path = next(
            label_path.parent.glob('*__sequence_contact_sheet.png'))
        image = cv2.imread(str(contact_path))
        if image is None:
            raise FileNotFoundError(contact_path)
        contact_rows.append(_tile(
            image,
            f"#{row['reference_index']} | valid={valid_ratio:.1%} | "
            f"dt_err={max_time_error:.3f}s",
            (900, 474)))
        print(
            f'[{row["reference_index"]}] shape={row["sequence_shape"]} '
            f'valid={valid_ratio:.1%} dt_err={max_time_error:.3f}s')

    _write_csv(out_dir / 'sequence_batch_metrics.csv', rows)
    eligibility = None
    if args.eligibility_summary:
        with Path(args.eligibility_summary).open() as f:
            eligibility = json.load(f)
    if args.contact_columns < 1 or args.contact_rows_per_page < 1:
        raise ValueError('Contact sheet grid dimensions must be positive')
    items_per_page = args.contact_columns * args.contact_rows_per_page
    contact_sheet_paths = []
    for page_index, start in enumerate(
            range(0, len(contact_rows), items_per_page), start=1):
        page_path = out_dir / (
            f'sequence_batch_contact_sheet_{page_index:02d}.png')
        page = _contact_page(
            contact_rows[start:start + items_per_page],
            args.contact_columns)
        cv2.imwrite(str(page_path), page)
        contact_sheet_paths.append(str(page_path))
        if page_index == 1:
            cv2.imwrite(
                str(out_dir / 'sequence_batch_contact_sheet.png'), page)

    summary = {
        'generated_sequence_count': len(rows),
        'reference_indices': [row['reference_index'] for row in rows],
        'sequence_shape': rows[0]['sequence_shape'],
        'mean_world_valid_ratio': float(np.mean([
            row['world_valid_ratio'] for row in rows])),
        'total_future_free_filled_voxels': sum(
            row['future_free_filled_voxels'] for row in rows),
        'total_future_static_filled_voxels': sum(
            row['future_static_filled_voxels'] for row in rows),
        'total_label_bytes': sum(row['label_bytes'] for row in rows),
        'contact_sheet_paths': contact_sheet_paths,
        'rejected_artifacts': rejected_artifacts,
        'eligibility': eligibility,
    }
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
