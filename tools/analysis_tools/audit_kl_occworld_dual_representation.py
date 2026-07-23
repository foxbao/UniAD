#!/usr/bin/env python
"""Audit separate 3D occupancy and 2D traversability labels for KL."""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    build_map_mask,
    load_clean_drivable_geometry,
)
from tools.data_converter.generate_kl_occworld_labels import (
    FREE,
    INSTANCE_OCCUPIED,
    STATIC_OCCUPIED,
    UNKNOWN,
    _load_infos,
    _output_stem,
    _resolve_path,
)


TRAVERSABILITY_UNKNOWN = 0
DRIVABLE = 1
NON_DRIVABLE = 2

NAVIGABILITY_UNKNOWN = 0
NAVIGABLE = 1
BLOCKED = 2

OCCUPANCY_STATE_NAMES = np.asarray([
    'unknown', 'free', 'static_occupied', 'instance_occupied'])
TRAVERSABILITY_STATE_NAMES = np.asarray([
    'unknown', 'drivable', 'non_drivable'])
NAVIGABILITY_STATE_NAMES = np.asarray([
    'unknown', 'navigable', 'blocked'])


def _compose_occupancy_3d(
        temporal_state_3d: np.ndarray,
        box_occupied_3d: np.ndarray,
        valid_free_sensor_count_3d: np.ndarray) -> Tuple[np.ndarray,
                                                             np.ndarray]:
    """Apply per-voxel visibility correction and instance precedence."""
    if not (temporal_state_3d.shape == box_occupied_3d.shape ==
            valid_free_sensor_count_3d.shape):
        raise ValueError('All 3D occupancy inputs must have the same shape')
    state = temporal_state_3d.astype(np.uint8, copy=True)
    invalid_free = (
        (state == FREE) & (valid_free_sensor_count_3d == 0))
    state[invalid_free] = UNKNOWN
    state[box_occupied_3d > 0] = INSTANCE_OCCUPIED
    visibility = state != UNKNOWN
    return state, visibility.astype(np.uint8)


def _compose_traversability(
        map_drivable: np.ndarray,
        ground_surface: np.ndarray,
        observation_bev: np.ndarray,
        non_drivable_evidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse map semantics with LiDAR support without treating absence as GT."""
    if not (map_drivable.shape == ground_surface.shape ==
            observation_bev.shape == non_drivable_evidence.shape):
        raise ValueError('All traversability inputs must have the same shape')
    map_drivable = map_drivable > 0
    ground_surface = ground_surface > 0
    observed = (observation_bev > 0) | ground_surface
    state = np.full(map_drivable.shape, TRAVERSABILITY_UNKNOWN,
                    dtype=np.uint8)
    state[map_drivable & observed] = DRIVABLE
    # Map absence alone is not a negative label. A negative needs separate
    # LiDAR geometry evidence, such as a surface enclosed by curb/barrier
    # returns on opposite sides.
    state[
        ~map_drivable & ground_surface & (non_drivable_evidence > 0)
    ] = NON_DRIVABLE
    return state, (state != TRAVERSABILITY_UNKNOWN).astype(np.uint8)


def _compose_navigability(traversability: np.ndarray,
                          observation_state_bev: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Combine persistent surface semantics with current obstacle state."""
    if traversability.shape != observation_state_bev.shape:
        raise ValueError('Traversability and occupancy BEV shapes differ')
    state = np.full(traversability.shape, NAVIGABILITY_UNKNOWN,
                    dtype=np.uint8)
    state[traversability == NON_DRIVABLE] = BLOCKED
    state[
        (traversability == DRIVABLE) &
        ((observation_state_bev == STATIC_OCCUPIED) |
         (observation_state_bev == INSTANCE_OCCUPIED))
    ] = BLOCKED
    state[
        (traversability == DRIVABLE) &
        (observation_state_bev == FREE)
    ] = NAVIGABLE
    return state, (state != NAVIGABILITY_UNKNOWN).astype(np.uint8)


def _shift_mask(mask: np.ndarray, row_offset: int,
                col_offset: int) -> np.ndarray:
    """Return out[r,c] = mask[r+row_offset,c+col_offset], without wrap."""
    height, width = mask.shape
    out = np.zeros_like(mask)
    src_r0 = max(row_offset, 0)
    src_r1 = height + min(row_offset, 0)
    src_c0 = max(col_offset, 0)
    src_c1 = width + min(col_offset, 0)
    dst_r0 = max(-row_offset, 0)
    dst_r1 = height - max(row_offset, 0)
    dst_c0 = max(-col_offset, 0)
    dst_c1 = width - max(col_offset, 0)
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[
            src_r0:src_r1, src_c0:src_c1]
    return out


def _opposing_obstacle_surface(state_bev: np.ndarray,
                               ground_surface: np.ndarray,
                               max_gap: int = 3):
    """Find observed free surfaces bounded by static returns on two sides."""
    free = state_bev == FREE
    red = state_bev == STATIC_OCCUPIED
    opposed = np.zeros_like(free)
    for row_step, col_step in ((1, 0), (0, 1), (1, 1), (1, -1)):
        positive = np.logical_or.reduce([
            _shift_mask(red, offset * row_step, offset * col_step)
            for offset in range(1, max_gap + 1)
        ])
        negative = np.logical_or.reduce([
            _shift_mask(red, -offset * row_step, -offset * col_step)
            for offset in range(1, max_gap + 1)
        ])
        opposed |= positive & negative
    return free & (ground_surface > 0) & opposed


def _filter_small_components(mask: np.ndarray,
                             min_component_cells: int) -> np.ndarray:
    """Keep 8-connected evidence components at or above a size threshold."""
    if min_component_cells <= 1:
        return mask.astype(bool, copy=True)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    keep_ids = np.flatnonzero(
        stats[1:, cv2.CC_STAT_AREA] >= min_component_cells) + 1
    return np.isin(labels, keep_ids)


def _audited_separator_region(state_bev: np.ndarray, max_gap: int = 3):
    """Reproduce the upper-left region selected during the manual audit."""
    free = state_bev == FREE
    red = state_bev == STATIC_OCCUPIED
    upper_left = np.zeros_like(free)
    upper_left[:state_bev.shape[0] // 2, :state_bev.shape[1] // 2] = True
    red_above = np.logical_or.reduce([
        _shift_mask(red, -offset, 0)
        for offset in range(1, max_gap + 1)
    ])
    red_below = np.logical_or.reduce([
        _shift_mask(red, offset, 0)
        for offset in range(1, max_gap + 1)
    ])
    return free & upper_left & red_above & red_below


def _save_state(path: Path, state: np.ndarray, palette: np.ndarray,
                scale: int = 4):
    rgb = palette[state]
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _tile(image: np.ndarray, title: str,
          size: Tuple[int, int] = (480, 360)) -> np.ndarray:
    image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    header = np.full((30, size[0], 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def _save_contact_sheet(path: Path, occupancy_paths: Sequence[Path],
                        map_path: Path, traversability_path: Path,
                        navigability_path: Path):
    top = np.concatenate([
        _tile(_read_rgb(item), title)
        for item, title in zip(occupancy_paths, (
            '3D occupancy z=0.8m',
            '3D occupancy z=1.6m',
            '3D occupancy z=2.4m',
        ))
    ], axis=1)
    bottom = np.concatenate([
        _tile(_read_rgb(map_path), 'HD-map drivable prior'),
        _tile(_read_rgb(traversability_path),
              '2D traversability: green drive / orange non-drive'),
        _tile(_read_rgb(navigability_path),
              '2D navigability: green pass / red blocked'),
    ], axis=1)
    separator = np.full((8, top.shape[1], 3), 28, dtype=np.uint8)
    cv2.imwrite(str(path), np.concatenate([top, separator, bottom], axis=0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--frame-index', type=int, required=True)
    parser.add_argument('--temporal-label', required=True)
    parser.add_argument(
        '--occlusion-detail', required=True,
        help='Single-frame occlusion NPZ containing 3D valid counts.')
    parser.add_argument(
        '--clean-map-file',
        default='data/kl_8/map/base_map_drivable_clean.pkl')
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    infos, _ = _load_infos(_resolve_path(args.ann_file))
    if args.frame_index < 0 or args.frame_index >= len(infos):
        raise ValueError(f'Invalid frame index {args.frame_index}')
    info = infos[args.frame_index]
    stem = _output_stem(info)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.temporal_label, allow_pickle=False) as temporal, \
            np.load(args.occlusion_detail, allow_pickle=False) as occlusion:
        temporal_state_3d = temporal['temporal_state_3d']
        box_occupied_3d = temporal['box_occupied_3d']
        ground_surface = (
            (temporal['filled_ground'] > 0) |
            np.any(temporal['ground_endpoint_3d'] > 0, axis=0))
        z_centers = temporal['z_centers']
        pc_range = temporal['pc_range']
        k_state = occlusion['state_occlusion_corrected']
        valid_free = occlusion['valid_free_sensor_count_3d']
        occupancy_3d, occupancy_visibility_3d = _compose_occupancy_3d(
            temporal_state_3d, box_occupied_3d, valid_free)

    drivable_global = load_clean_drivable_geometry(
        str(_resolve_path(args.clean_map_file)))
    map_drivable = build_map_mask(
        drivable_global, np.asarray(info['ego2global'], dtype=np.float64),
        pc_range, k_state.shape)
    non_drivable_evidence = _opposing_obstacle_surface(
        k_state, ground_surface) & ~(map_drivable > 0)
    traversability, traversability_valid = _compose_traversability(
        map_drivable, ground_surface, k_state != UNKNOWN,
        non_drivable_evidence)
    navigability, navigability_valid = _compose_navigability(
        traversability, k_state)
    separator = _audited_separator_region(k_state)

    label_path = out_dir / f'{stem}__dual_representation.npz'
    np.savez_compressed(
        label_path,
        occupancy_state_3d=occupancy_3d,
        occupancy_visibility_3d=occupancy_visibility_3d,
        occupancy_state_names=OCCUPANCY_STATE_NAMES,
        traversability_state_bev=traversability,
        traversability_valid_bev=traversability_valid,
        traversability_state_names=TRAVERSABILITY_STATE_NAMES,
        navigability_state_bev=navigability,
        navigability_valid_bev=navigability_valid,
        navigability_state_names=NAVIGABILITY_STATE_NAMES,
        map_drivable_prior_bev=map_drivable.astype(np.uint8),
        lidar_ground_surface_bev=ground_surface.astype(np.uint8),
        non_drivable_evidence_bev=non_drivable_evidence.astype(np.uint8),
        observation_state_bev=k_state,
        separator_candidate_bev=separator.astype(np.uint8),
        z_centers=z_centers,
        pc_range=pc_range,
        frame_index=np.int64(args.frame_index),
        timestamp=np.float64(info['timestamp']),
    )

    occupancy_palette = np.asarray([
        [40, 40, 40], [70, 180, 90],
        [225, 80, 70], [65, 120, 240],
    ], dtype=np.uint8)
    traversability_palette = np.asarray([
        [40, 40, 40], [55, 185, 95], [235, 155, 55],
    ], dtype=np.uint8)
    navigability_palette = np.asarray([
        [40, 40, 40], [55, 195, 100], [230, 75, 65],
    ], dtype=np.uint8)
    map_palette = np.asarray([
        [40, 40, 40], [80, 170, 210],
    ], dtype=np.uint8)

    occupancy_paths = []
    collision_indices = np.flatnonzero(
        (z_centers >= 0.3) & (z_centers <= 2.5))
    for z_index in collision_indices:
        path = out_dir / f'{stem}__occupancy_z{z_centers[z_index]:.1f}.png'
        _save_state(path, occupancy_3d[z_index], occupancy_palette)
        occupancy_paths.append(path)
    if len(occupancy_paths) != 3:
        raise ValueError(
            f'Expected three collision slices, got {len(occupancy_paths)}')

    map_path = out_dir / f'{stem}__map_drivable_prior.png'
    traversability_path = out_dir / f'{stem}__traversability_bev.png'
    navigability_path = out_dir / f'{stem}__navigability_bev.png'
    ground_path = out_dir / f'{stem}__lidar_ground_surface.png'
    _save_state(map_path, map_drivable.astype(np.uint8), map_palette)
    _save_state(
        traversability_path, traversability, traversability_palette)
    _save_state(navigability_path, navigability, navigability_palette)
    _save_state(
        ground_path, ground_surface.astype(np.uint8), map_palette)
    contact_path = out_dir / f'{stem}__dual_representation.png'
    _save_contact_sheet(
        contact_path, occupancy_paths, map_path,
        traversability_path, navigability_path)

    separator_count = max(int(separator.sum()), 1)
    summary = {
        'frame_index': args.frame_index,
        'stem': stem,
        'occupancy_shape': list(occupancy_3d.shape),
        'occupancy_counts': {
            name: int(np.count_nonzero(occupancy_3d == index))
            for index, name in enumerate(OCCUPANCY_STATE_NAMES.tolist())
        },
        'traversability_counts': {
            name: int(np.count_nonzero(traversability == index))
            for index, name in enumerate(
                TRAVERSABILITY_STATE_NAMES.tolist())
        },
        'navigability_counts': {
            name: int(np.count_nonzero(navigability == index))
            for index, name in enumerate(NAVIGABILITY_STATE_NAMES.tolist())
        },
        'separator_candidate_cells': int(separator.sum()),
        'separator_non_drivable_cells': int(np.count_nonzero(
            separator & (traversability == NON_DRIVABLE))),
        'separator_non_drivable_ratio': float(np.count_nonzero(
            separator & (traversability == NON_DRIVABLE)) /
            separator_count),
        'separator_unknown_cells': int(np.count_nonzero(
            separator &
            (traversability == TRAVERSABILITY_UNKNOWN))),
    }
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'label={label_path}')
    print(f'preview={contact_path}')


if __name__ == '__main__':
    main()
