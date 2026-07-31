#!/usr/bin/env python
"""Visualize selected actual-point-height low-static candidate frames."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.analysis_tools.audit_kl_occworld_low_static_spatial_expansion import (
    STATE_PALETTE_BGR,
    _cell_point_statistics,
    _crop_bounds,
    _heatmap_image,
    _relative_pointcloud_image,
    _semantic_image,
    _xy_to_hw,
)
from tools.analysis_tools.audit_kl_occworld_low_static_stability import (
    _sequence_paths,
)
from tools.analysis_tools.evaluate_kl_occworld_low_static_point_height import (
    STATIC_OCCUPIED,
    apply_point_height_downgrade,
    point_height_downgrade_mask,
)
from tools.data_converter.generate_kl_occworld_labels import (
    FREE,
    INSTANCE_OCCUPIED,
    UNKNOWN,
    MultiLidarOccLabelBuilder,
    _load_infos,
    _resolve_path,
    _xyz_to_zhw,
)
from tools.data_converter.generate_kl_occworld_temporal_labels import (
    _warp_mask_to_reference_xyz,
)


def _panel(image: np.ndarray, title: str,
           content_size=(460, 340)) -> np.ndarray:
    target_width, target_height = content_size
    ratio = min(target_width / image.shape[1], target_height / image.shape[0])
    resized = cv2.resize(
        image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    content = np.full((target_height, target_width, 3), (24, 26, 30),
                      dtype=np.uint8)
    top = (target_height - resized.shape[0]) // 2
    left = (target_width - resized.shape[1]) // 2
    content[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    header = np.full((34, target_width, 3), (28, 30, 34), dtype=np.uint8)
    cv2.putText(header, title, (9, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.47, (238, 238, 238), 1, cv2.LINE_AA)
    return np.concatenate([header, content], axis=0)


def _effect_image(world_low: np.ndarray, downgrade: np.ndarray,
                  next_low: np.ndarray, bounds: tuple) -> np.ndarray:
    """Show which candidates are supported or contradicted by next GT."""
    image = np.full((*world_low.shape, 3), (24, 26, 30), dtype=np.uint8)
    image[world_low == STATIC_OCCUPIED] = (70, 80, 225)
    image[downgrade] = (40, 220, 250)
    if next_low is not None:
        image[downgrade & (next_low == FREE)] = (80, 220, 80)
        image[downgrade & (next_low == STATIC_OCCUPIED)] = (230, 130, 40)
        image[downgrade & (next_low == UNKNOWN)] = (205, 105, 205)
        image[downgrade & (next_low == INSTANCE_OCCUPIED)] = (240, 120, 65)
    row_min, row_max, col_min, col_max = bounds
    return cv2.resize(image[row_min:row_max, col_min:col_max], None,
                      fx=6, fy=6, interpolation=cv2.INTER_NEAREST)


def _legend(width: int) -> np.ndarray:
    image = np.full((58, width, 3), (28, 30, 34), dtype=np.uint8)
    entries = (
        ('static kept', (70, 80, 225)),
        ('candidate', (40, 220, 250)),
        ('next free', (80, 220, 80)),
        ('next static', (230, 130, 40)),
        ('next unknown', (205, 105, 205)),
    )
    x = 12
    for label, color in entries:
        cv2.rectangle(image, (x, 19), (x + 18, 37), color, -1)
        cv2.putText(image, label, (x + 24, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (238, 238, 238), 1, cv2.LINE_AA)
        x += 36 + max(72, len(label) * 8)
    return image


def _sheet(panels: list) -> np.ndarray:
    rows = []
    for start in range(0, len(panels), 2):
        rows.append(np.concatenate(panels[start:start + 2], axis=1))
    separator = np.full((8, rows[0].shape[1], 3), (18, 20, 24),
                        dtype=np.uint8)
    return np.concatenate(
        [rows[0], separator, rows[1], separator, rows[2],
         _legend(rows[0].shape[1])], axis=0)


def _parse_frame(value: str) -> tuple:
    try:
        reference, horizon = value.split(':', 1)
        return int(reference), int(horizon)
    except ValueError as error:
        raise ValueError(
            f'Frame must use REFERENCE:HORIZON, got {value!r}') from error


def _select_frames(result: dict, manual: list,
                   top_per_group: int) -> list:
    selected = [_parse_frame(value) for value in manual]
    for key in (
            'top_frames_by_downgraded_voxels',
            'top_frames_by_downgraded_static_next'):
        for row in result['labels'][key][:top_per_group]:
            selected.append((int(row['reference_index']),
                             int(row['source_horizon'])))
    unique = []
    for item in selected:
        if item not in unique:
            unique.append(item)
    return unique


def _source_point_statistics(evidence: dict, builder) -> tuple:
    points = np.asarray(evidence['points'], dtype=np.float32)
    voxels = builder.builder.coord_to_index_floor(points[:, :3])
    valid = np.all(
        (voxels >= 0) & (voxels < builder.builder.occ_size[None, :]), axis=1)
    ground, _ = builder.builder.estimate_ground_height(
        points[valid, :3], voxels[valid])
    return _cell_point_statistics(points, builder.builder, ground)


def _transform_points(points: np.ndarray, source_pose: np.ndarray,
                      reference_pose: np.ndarray) -> np.ndarray:
    transform = np.linalg.inv(reference_pose) @ source_pose
    homogeneous = np.concatenate([
        points[:, :3], np.ones((len(points), 1), dtype=np.float32)], axis=1)
    transformed = points.copy()
    transformed[:, :3] = (homogeneous @ transform.T)[:, :3]
    return transformed


def _render_frame(reference: int, horizon: int, sequences: dict,
                  infos: list, builder, low_index: int, threshold_m: float,
                  output_dir: Path) -> dict:
    path = sequences[reference]
    with np.load(path, allow_pickle=False) as archive:
        world = np.asarray(archive['world_target_state_3d'], dtype=np.uint8)
        direct = np.asarray(archive['direct_observation_state_3d'], dtype=np.uint8)
        target_indices = np.asarray(archive['target_indices'], dtype=np.int64)
        pc_range = np.asarray(archive['pc_range'], dtype=np.float32)
        occ_size = np.asarray(archive['occ_size'], dtype=np.int64)
    if not 0 <= horizon < len(target_indices):
        raise IndexError(f'Horizon {horizon} is outside reference {reference}')
    target_index = int(target_indices[horizon])
    evidence = builder.build(
        infos[target_index], diagnostics=False, return_points=True)
    point_stats = _source_point_statistics(evidence, builder)
    p90_hw = _xy_to_hw(point_stats['low_p90_xy'])
    static_3d = np.asarray(evidence['static_obstacle_3d'], dtype=bool)
    source_static = static_3d[low_index]
    higher_static = np.any(static_3d[low_index + 1:], axis=0)
    source_weak = point_height_downgrade_mask(
        source_static, p90_hw, higher_static, threshold_m)
    source_mask = np.zeros_like(static_3d, dtype=bool)
    source_mask[low_index] = source_weak
    source_pose = np.asarray(infos[target_index]['ego2global'], dtype=np.float64)
    reference_pose = np.asarray(infos[reference]['ego2global'], dtype=np.float64)
    warped = _xyz_to_zhw(_warp_mask_to_reference_xyz(
        source_mask, source_pose, reference_pose, pc_range, occ_size,
        radius_xy=0, radius_z=0))[low_index]
    world_low = world[horizon, low_index]
    direct_low = direct[horizon, low_index]
    downgrade = (
        (world_low == STATIC_OCCUPIED) &
        (direct_low == STATIC_OCCUPIED) & warped)
    candidate_low = apply_point_height_downgrade(
        world_low, direct_low, downgrade)
    if not np.any(downgrade):
        raise ValueError(
            f'Reference {reference}, horizon {horizon} has no candidate voxels')
    source_bounds = _crop_bounds(source_weak, margin=6)
    reference_bounds = _crop_bounds(downgrade, margin=6)
    transformed_points = _transform_points(
        point_stats['points'], source_pose, reference_pose)
    next_low = (
        world[horizon + 1, low_index]
        if horizon + 1 < world.shape[0] else None)
    relative_cloud = _relative_pointcloud_image(
        transformed_points, point_stats['relative_height'], pc_range,
        occ_size, reference_bounds, downgrade)
    source_p90 = _heatmap_image(
        p90_hw, np.isfinite(p90_hw), source_bounds, 0.0, 1.5, source_weak)
    panels = [
        _panel(relative_cloud,
               'reference-aligned points | color: height above local ground'),
        _panel(source_p90,
               'source LiDAR | z=0 point P90 above local ground'),
        _panel(_semantic_image(world_low, downgrade, reference_bounds),
               'original world GT | red: static'),
        _panel(_semantic_image(candidate_low, downgrade, reference_bounds,
                               downgrade),
               'candidate GT | yellow: static -> unknown'),
        _panel(_effect_image(world_low, downgrade, next_low, reference_bounds),
               'candidate outcome | green next-free, blue next-static'),
        _panel(_semantic_image(
            next_low if next_low is not None else world_low,
            downgrade, reference_bounds),
            'next world GT' if next_low is not None else 'no next horizon'),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f'ref_{reference:06d}__h{horizon}_point_height.png'
    if not cv2.imwrite(str(image_path), _sheet(panels)):
        raise OSError(f'Failed to write {image_path}')
    next_counts = None if next_low is None else {
        name: int(np.count_nonzero(downgrade & (next_low == state)))
        for state, name in enumerate(('unknown', 'free', 'static', 'instance'))}
    return {
        'reference_index': reference,
        'source_horizon': horizon,
        'target_index': target_index,
        'source_weak_voxels': int(np.count_nonzero(source_weak)),
        'downgraded_voxels': int(np.count_nonzero(downgrade)),
        'next_state_of_downgraded': next_counts,
        'image': str(image_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--result-json', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_low_static_point_height_ablation_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_full_train_v1'))
    parser.add_argument('--ann-file', type=Path,
                        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument('--frames', nargs='*', default=['25267:0'])
    parser.add_argument('--top-per-group', type=int, default=3)
    parser.add_argument('--point-height-threshold-m', type=float,
                        default=0.55)
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_low_static_point_height_visuals_v1'))
    return parser.parse_args()


def main():
    args = parse_args()
    with args.result_json.open() as source:
        result = json.load(source)
    frames = _select_frames(result, args.frames, args.top_per_group)
    sequences = _sequence_paths(args.sequence_root)
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    grid = result['labels']['grid']
    builder = MultiLidarOccLabelBuilder(
        grid['pc_range'], (int(grid['occ_size'][1]), int(grid['occ_size'][0])),
        grid['occ_size'],
        target_frame=str(metainfo.get('lidar_coord_frame', 'FLU')),
        collision_z=grid['collision_z'])
    rows = []
    for position, (reference, horizon) in enumerate(frames, start=1):
        if reference not in sequences:
            raise KeyError(f'Reference {reference} is absent from sequence root')
        row = _render_frame(
            reference, horizon, sequences, infos, builder,
            int(grid['low_static_z_index']),
            args.point_height_threshold_m, args.out_dir)
        rows.append(row)
        print(f'[{position}/{len(frames)}] {row}', flush=True)
    summary_path = args.out_dir / 'summary.json'
    with summary_path.open('w') as destination:
        json.dump({
            'schema_version': 1,
            'analysis_type': 'offline_candidate_visualization',
            'result_json': str(args.result_json),
            'rule': result['rule'],
            'frames': rows,
        }, destination, ensure_ascii=False, indent=2)
        destination.write('\n')


if __name__ == '__main__':
    main()
