#!/usr/bin/env python
"""Audit whether projected free voxels are occluded for every LiDAR."""

import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np


UNKNOWN = 0
FREE = 1
STATIC_OCCUPIED = 2
INSTANCE_OCCUPIED = 3


def _zhw_to_xyz(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f'Expected [Z,H,W], got {volume.shape}')
    return np.transpose(volume[:, ::-1, :], (2, 1, 0))


def _ray_blocked_by_occupied(
        target_xyz_index: np.ndarray,
        sensor_origin: np.ndarray,
        occupied_xyz: np.ndarray,
        pc_range: Sequence[float]) -> bool:
    """Check occupied voxels before the target using label ray sampling."""
    target_xyz_index = np.asarray(target_xyz_index, dtype=np.int64)
    sensor_origin = np.asarray(sensor_origin, dtype=np.float32)
    pc_range = np.asarray(pc_range, dtype=np.float32)
    occ_size = np.asarray(occupied_xyz.shape, dtype=np.int64)
    voxel_size = ((pc_range[3:] - pc_range[:3]) /
                  occ_size.astype(np.float32))
    target_center = (pc_range[:3] +
                     (target_xyz_index.astype(np.float32) + 0.5) *
                     voxel_size)
    delta = target_center - sensor_origin
    num_steps = int(np.ceil(np.max(np.abs(delta / voxel_size))))
    if num_steps <= 0:
        return False
    t = np.arange(num_steps, dtype=np.float32) / num_steps
    samples = sensor_origin[None, :] + t[:, None] * delta[None, :]
    voxels = np.floor(
        (samples - pc_range[:3]) / voxel_size).astype(np.int64)
    valid = np.all((voxels >= 0) & (voxels < occ_size), axis=1)
    voxels = voxels[valid]
    if voxels.shape[0] == 0:
        return False
    not_target = np.any(voxels != target_xyz_index[None, :], axis=1)
    voxels = voxels[not_target]
    if voxels.shape[0] == 0:
        return False
    return bool(np.any(occupied_xyz[tuple(voxels.T)]))


def _save_state(path: Path, state: np.ndarray, scale: int = 4):
    palette = np.asarray([
        [40, 40, 40],
        [70, 180, 90],
        [225, 80, 70],
        [65, 120, 240],
    ], dtype=np.uint8)
    rgb = palette[state]
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _save_removed_highlight(path: Path, state: np.ndarray,
                            removed: np.ndarray, scale: int = 4):
    palette = np.asarray([
        [40, 40, 40],
        [70, 180, 90],
        [225, 80, 70],
        [65, 120, 240],
    ], dtype=np.uint8)
    rgb = palette[state]
    rgb[removed > 0] = (220, 65, 220)
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _panel(image: np.ndarray, title: str,
           size: Tuple[int, int]) -> np.ndarray:
    image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    header = np.full((28, size[0], 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def _save_lower_left_comparison(path: Path, h_path: Path, k_path: Path,
                                highlight_path: Path):
    images = [cv2.imread(str(item))
              for item in (h_path, k_path, highlight_path)]
    if any(image is None for image in images):
        raise FileNotFoundError('Missing H/K preview for crop comparison')
    height, width = images[0].shape[:2]
    crops = [image[height // 2:, :width // 2] for image in images]
    titles = ('H original', 'K occlusion corrected', 'removed highlight')
    panels = [_panel(crop, title, (480, 360))
              for crop, title in zip(crops, titles)]
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current-label', required=True,
                        help='Single-frame diagnostic NPZ with per-sensor masks.')
    parser.add_argument('--temporal-label', required=True,
                        help='Temporal NPZ containing H state and promoted masks.')
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    current_path = Path(args.current_label)
    temporal_path = Path(args.temporal_label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(current_path, allow_pickle=False) as current, \
            np.load(temporal_path, allow_pickle=False) as temporal:
        if not np.array_equal(current['sensor_names'],
                              temporal['sensor_names']):
            raise ValueError('Current and temporal sensor ordering differs')
        per_sensor_free = current['per_sensor_free_3d'] > 0
        sensor_origins = current['sensor_origins']
        temporal_state = temporal['temporal_state_3d']
        h_state = temporal['state_temporal_3d_projection']
        pc_range = temporal['pc_range']
        z_centers = temporal['z_centers']
        collision_z = temporal['collision_z']
        collision_keep = ((z_centers >= collision_z[0]) &
                          (z_centers <= collision_z[1]))

        occupied_zhw = (
            (temporal['temporal_occupancy_target'] > 0) |
            (temporal['box_occupied_3d'] > 0))
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
            active_indices = np.argwhere(active)
            sensor_valid = 0
            sensor_blocked = 0
            for z_idx, row, col in active_indices:
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
        corrected_state = h_state.copy()
        corrected_state[removed] = UNKNOWN

        old_red_green_below = int(np.count_nonzero(
            (h_state[:-1] == STATIC_OCCUPIED) &
            (h_state[1:] == FREE)))
        new_red_green_below = int(np.count_nonzero(
            (corrected_state[:-1] == STATIC_OCCUPIED) &
            (corrected_state[1:] == FREE)))
        lower_left = np.zeros_like(h_green)
        lower_left[h_green.shape[0] // 2:, :h_green.shape[1] // 2] = True
        summary = {
            'sensor_names': current['sensor_names'].tolist(),
            'h_green_cells': int(np.count_nonzero(h_green)),
            'occlusion_removed_green_cells': int(np.count_nonzero(removed)),
            'remaining_green_cells': int(np.count_nonzero(
                corrected_state == FREE)),
            'lower_left_green_before': int(np.count_nonzero(
                h_green & lower_left)),
            'lower_left_removed': int(np.count_nonzero(
                removed & lower_left)),
            'lower_left_green_after': int(np.count_nonzero(
                (corrected_state == FREE) & lower_left)),
            'red_to_green_below_before': old_red_green_below,
            'red_to_green_below_after': new_red_green_below,
            'per_sensor_valid_free_voxels': per_sensor_valid,
            'per_sensor_blocked_free_voxels': per_sensor_blocked,
        }

        np.savez_compressed(
            out_dir / 'occlusion_audit.npz',
            state_occlusion_corrected=corrected_state,
            occlusion_removed_bev=removed.astype(np.uint8),
            valid_free_sensor_count_3d=valid_count,
            blocked_free_sensor_count_3d=blocked_count,
            sensor_names=current['sensor_names'],
            sensor_origins=sensor_origins,
            pc_range=pc_range,
            collision_z=collision_z,
        )

    h_path = out_dir / 'H_original.png'
    k_path = out_dir / 'K_occlusion_corrected.png'
    highlight_path = out_dir / 'K_removed_highlight.png'
    _save_state(h_path, h_state)
    _save_state(k_path, corrected_state)
    _save_removed_highlight(highlight_path, h_state, removed)
    _save_lower_left_comparison(
        out_dir / 'lower_left_comparison.png',
        h_path, k_path, highlight_path)
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'comparison={out_dir / "lower_left_comparison.png"}')


if __name__ == '__main__':
    main()
