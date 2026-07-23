#!/usr/bin/env python
"""Batch per-sensor occlusion audit for temporal KL OccWorld labels."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.analysis_tools.audit_kl_occworld_occlusion import (
    FREE,
    STATIC_OCCUPIED,
    UNKNOWN,
    _ray_blocked_by_occupied,
    _save_removed_highlight,
    _save_state,
    _zhw_to_xyz,
)


def _audit_label(path: Path):
    with np.load(path, allow_pickle=False) as label:
        if 'per_sensor_free_3d' not in label:
            raise KeyError(
                f'{path.name} lacks per_sensor_free_3d; regenerate with '
                '--save-per-sensor-diagnostics')
        per_sensor_free = label['per_sensor_free_3d'] > 0
        sensor_origins = label['sensor_origins']
        temporal_state = label['temporal_state_3d']
        h_state = label['state_temporal_3d_projection']
        pc_range = label['pc_range']
        z_centers = label['z_centers']
        collision_z = label['collision_z']
        collision_keep = ((z_centers >= collision_z[0]) &
                          (z_centers <= collision_z[1]))
        occupied_zhw = (
            (label['temporal_occupancy_target'] > 0) |
            (label['box_occupied_3d'] > 0))
        occupied_xyz = _zhw_to_xyz(occupied_zhw)
        h_green = h_state == FREE

        valid_count = np.zeros_like(temporal_state, dtype=np.uint8)
        blocked_count = np.zeros_like(temporal_state, dtype=np.uint8)
        per_sensor_valid = []
        per_sensor_blocked = []
        for sensor_index, sensor_origin in enumerate(sensor_origins):
            active = (
                per_sensor_free[sensor_index] &
                (temporal_state == FREE) &
                h_green[None, :, :])
            sensor_valid = 0
            sensor_blocked = 0
            for z_idx, row, col in np.argwhere(active):
                xyz_index = np.asarray([
                    col, temporal_state.shape[1] - 1 - row, z_idx,
                ], dtype=np.int64)
                if _ray_blocked_by_occupied(
                        xyz_index, sensor_origin, occupied_xyz, pc_range):
                    blocked_count[z_idx, row, col] += 1
                    sensor_blocked += 1
                else:
                    valid_count[z_idx, row, col] += 1
                    sensor_valid += 1
            per_sensor_valid.append(sensor_valid)
            per_sensor_blocked.append(sensor_blocked)

        valid_free_bev = np.any(
            (valid_count[collision_keep] > 0) &
            (temporal_state[collision_keep] == FREE), axis=0)
        removed = h_green & ~valid_free_bev
        corrected = h_state.copy()
        corrected[removed] = UNKNOWN
        lower_left = np.zeros_like(h_green)
        lower_left[h_green.shape[0] // 2:, :h_green.shape[1] // 2] = True
        offsets = label['temporal_offsets']
        source_indices = label['temporal_source_indices']
        current_positions = np.flatnonzero(offsets == 0)
        if current_positions.size != 1:
            raise ValueError(f'Expected one offset=0 in {path.name}')
        frame_index = int(source_indices[current_positions[0]])
        temporal_suffix = '__temporal'
        stem = path.stem
        if stem.endswith(temporal_suffix):
            stem = stem[:-len(temporal_suffix)]
        summary = {
            'frame_index': frame_index,
            'stem': stem,
            'h_green_cells': int(h_green.sum()),
            'removed_green_cells': int(removed.sum()),
            'removal_ratio': float(removed.sum() / max(h_green.sum(), 1)),
            'remaining_green_cells': int(np.count_nonzero(corrected == FREE)),
            'lower_left_green_before': int(np.count_nonzero(
                h_green & lower_left)),
            'lower_left_removed': int(np.count_nonzero(
                removed & lower_left)),
            'red_to_green_below_before': int(np.count_nonzero(
                (h_state[:-1] == STATIC_OCCUPIED) &
                (h_state[1:] == FREE))),
            'red_to_green_below_after': int(np.count_nonzero(
                (corrected[:-1] == STATIC_OCCUPIED) &
                (corrected[1:] == FREE))),
            'per_sensor_valid_free_voxels': ';'.join(
                str(value) for value in per_sensor_valid),
            'per_sensor_blocked_free_voxels': ';'.join(
                str(value) for value in per_sensor_blocked),
        }
        metadata = {
            'sensor_names': label['sensor_names'],
            'sensor_origins': sensor_origins,
            'pc_range': pc_range,
            'collision_z': collision_z,
            'valid_free_sensor_count_3d': valid_count,
            'blocked_free_sensor_count_3d': blocked_count,
        }
    return summary, h_state, corrected, removed, metadata


def _tile(path: Path, title: str,
          size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    header = np.full((28, size[0], 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def _save_k_contact(out_dir: Path, rows: List[dict], columns: int = 6):
    tiles = [_tile(
        out_dir / f"{row['stem']}__K_occlusion.png",
        f"#{row['frame_index']} K rm={row['removal_ratio']:.1%}")
             for row in rows]
    blank = np.full_like(tiles[0], 28)
    grid = []
    for start in range(0, len(tiles), columns):
        chunk = tiles[start:start + columns]
        chunk.extend([blank] * (columns - len(chunk)))
        grid.append(np.concatenate(chunk, axis=1))
    cv2.imwrite(str(out_dir / 'k_contact_sheet.png'),
                np.concatenate(grid, axis=0))


def _save_worst_contact(out_dir: Path, rows: List[dict], count: int = 8):
    worst = sorted(rows, key=lambda row: row['removal_ratio'],
                   reverse=True)[:count]
    grid = []
    for row in worst:
        stem = row['stem']
        grid.append(np.concatenate([
            _tile(out_dir / f'{stem}__H_original.png',
                  f"#{row['frame_index']} H"),
            _tile(out_dir / f'{stem}__K_occlusion.png',
                  f"#{row['frame_index']} K"),
            _tile(out_dir / f'{stem}__K_removed.png',
                  f"removed={row['removal_ratio']:.1%}"),
        ], axis=1))
    cv2.imwrite(str(out_dir / 'worst8_h_k_removed.png'),
                np.concatenate(grid, axis=0))


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(rows: List[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        'min': float(values.min()),
        'median': float(np.median(values)),
        'mean': float(values.mean()),
        'max': float(values.max()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temporal-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    temporal_dir = Path(args.temporal_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(temporal_dir.glob('*__temporal.npz')):
        summary, h_state, corrected, removed, metadata = _audit_label(path)
        stem = summary['stem']
        _save_state(out_dir / f'{stem}__H_original.png', h_state)
        _save_state(out_dir / f'{stem}__K_occlusion.png', corrected)
        _save_removed_highlight(
            out_dir / f'{stem}__K_removed.png', h_state, removed)
        np.savez_compressed(
            out_dir / f'{stem}__occlusion.npz',
            state_occlusion_corrected=corrected,
            occlusion_removed_bev=removed.astype(np.uint8),
            **metadata,
        )
        rows.append(summary)
        print(
            f'[{summary["frame_index"]}] '
            f'removed={summary["removed_green_cells"]} '
            f'({summary["removal_ratio"]:.1%})')
    if not rows:
        raise RuntimeError(f'No temporal NPZ files found in {temporal_dir}')
    rows.sort(key=lambda row: row['frame_index'])
    worst = sorted(rows, key=lambda row: row['removal_ratio'],
                   reverse=True)[:8]
    summary = {
        'frame_count': len(rows),
        'removal_ratio': _quantiles(rows, 'removal_ratio'),
        'removed_green_cells': _quantiles(rows, 'removed_green_cells'),
        'worst8_frame_indices': [row['frame_index'] for row in worst],
        'worst8': worst,
    }
    _write_csv(out_dir / 'occlusion_metrics.csv', rows)
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _save_k_contact(out_dir, rows)
    _save_worst_contact(out_dir, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
