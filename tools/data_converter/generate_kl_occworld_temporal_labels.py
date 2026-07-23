#!/usr/bin/env python
"""Generate causal temporal-promotion labels for KL OccWorld evidence."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.data_converter.generate_kl_occworld_labels import (
    ENDPOINT_RELIABLE_STATIC,
    ENDPOINT_UNCERTAIN_OBSTACLE,
    INSTANCE_OCCUPIED,
    STATIC_OCCUPIED,
    MultiLidarOccLabelBuilder,
    _load_infos,
    _output_stem,
    _project_observed_state,
    _resolve_path,
    _save_label,
    _save_preview,
    _xyz_to_zhw,
)


def _mask_zhw_to_xyz_points(mask_zhw: np.ndarray,
                            pc_range: Sequence[float],
                            occ_size: Sequence[int]) -> np.ndarray:
    """Convert active [Z,H,W] voxels to XYZ voxel-center coordinates."""
    indices = np.argwhere(mask_zhw)
    if indices.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    z_idx = indices[:, 0]
    y_idx = mask_zhw.shape[1] - 1 - indices[:, 1]
    x_idx = indices[:, 2]
    xyz_indices = np.stack([x_idx, y_idx, z_idx], axis=1)
    pc_range = np.asarray(pc_range, dtype=np.float32)
    occ_size = np.asarray(occ_size, dtype=np.int64)
    voxel_size = ((pc_range[3:] - pc_range[:3]) /
                  occ_size.astype(np.float32))
    return (pc_range[:3] +
            (xyz_indices.astype(np.float32) + 0.5) * voxel_size)


def _warp_mask_to_reference_xyz(
        mask_zhw: np.ndarray,
        source_ego2global: np.ndarray,
        reference_ego2global: np.ndarray,
        pc_range: Sequence[float],
        occ_size: Sequence[int],
        radius_xy: int = 0,
        radius_z: int = 0) -> np.ndarray:
    """Warp one frame's voxel mask into the reference XYZ grid."""
    pc_range = np.asarray(pc_range, dtype=np.float32)
    occ_size = np.asarray(occ_size, dtype=np.int64)
    points = _mask_zhw_to_xyz_points(mask_zhw, pc_range, occ_size)
    support = np.zeros(tuple(occ_size.tolist()), dtype=bool)
    if points.shape[0] == 0:
        return support

    source_to_reference = (
        np.linalg.inv(np.asarray(reference_ego2global, dtype=np.float64)) @
        np.asarray(source_ego2global, dtype=np.float64))
    homogeneous = np.concatenate([
        points.astype(np.float64),
        np.ones((points.shape[0], 1), dtype=np.float64),
    ], axis=1)
    reference_points = (source_to_reference @ homogeneous.T).T[:, :3]
    voxel_size = ((pc_range[3:] - pc_range[:3]) /
                  occ_size.astype(np.float32))
    indices = np.floor(
        (reference_points - pc_range[:3]) / voxel_size).astype(np.int64)

    for delta_x in range(-radius_xy, radius_xy + 1):
        for delta_y in range(-radius_xy, radius_xy + 1):
            for delta_z in range(-radius_z, radius_z + 1):
                shifted = indices + np.asarray(
                    [delta_x, delta_y, delta_z], dtype=np.int64)
                valid = np.all(
                    (shifted >= 0) & (shifted < occ_size), axis=1)
                shifted = shifted[valid]
                if shifted.shape[0] > 0:
                    support[tuple(shifted.T)] = True
    return support


def _promote_uncertain(
        endpoint_type_3d: np.ndarray,
        temporal_support_count_3d: np.ndarray,
        min_support_frames: int) -> np.ndarray:
    """Promote only current uncertain candidates with temporal support."""
    return ((endpoint_type_3d == ENDPOINT_UNCERTAIN_OBSTACLE) &
            (temporal_support_count_3d >= min_support_frames))


def _temporal_source_indices(infos: List[dict], reference_index: int,
                             offsets: Sequence[int]) -> List[Tuple[int, int]]:
    reference_scene = infos[reference_index].get('scene_token')
    sources = []
    for offset in offsets:
        source_index = reference_index + int(offset)
        if source_index < 0 or source_index >= len(infos):
            continue
        if infos[source_index].get('scene_token') != reference_scene:
            continue
        sources.append((int(offset), source_index))
    if not any(offset == 0 for offset, _ in sources):
        sources.append((0, reference_index))
    return sorted(set(sources))


def _save_highlight_preview(path: Path, state: np.ndarray,
                            promoted_bev: np.ndarray, scale: int = 4):
    palette = np.asarray([
        [40, 40, 40],
        [70, 180, 90],
        [225, 80, 70],
        [65, 120, 240],
    ], dtype=np.uint8)
    rgb = palette[state]
    highlight = (promoted_bev > 0) & (state != INSTANCE_OCCUPIED)
    rgb[highlight] = (235, 190, 55)
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _contact_tile(path: Path, title: str,
                  size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    header = np.full((28, size[0], 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def _save_contact_sheet(out_dir: Path, summary_rows: List[dict]):
    rows = []
    for summary in summary_rows:
        stem = summary['stem']
        index = summary['frame_index']
        promoted = summary['promoted_3d']
        rows.append(np.concatenate([
            _contact_tile(
                out_dir / f'{stem}__G_filtered.png', f'#{index} G'),
            _contact_tile(
                out_dir / f'{stem}__H_temporal.png', f'#{index} H'),
            _contact_tile(
                out_dir / f'{stem}__H_highlight.png',
                f'#{index} promoted={promoted}'),
        ], axis=1))
    cv2.imwrite(
        str(out_dir / 'temporal_contact_sheet.png'),
        np.concatenate(rows, axis=0))


def _write_summary(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    parser.add_argument(
        '--offsets', type=int, nargs='+', default=[-2, -1, 0],
        help='Causal frame offsets relative to each reference.')
    parser.add_argument('--min-support-frames', type=int, default=2)
    parser.add_argument('--match-radius-xy', type=int, default=0)
    parser.add_argument('--match-radius-z', type=int, default=0)
    parser.add_argument(
        '--out-dir',
        default='outputs/patent_2026_occ/occworld_temporal_debug')
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument('--occ-size', type=int, nargs=3,
                        default=[160, 120, 10])
    parser.add_argument('--collision-z', type=float, nargs=2,
                        default=[0.3, 2.5])
    parser.add_argument('--target-frame', choices=['auto', 'FLU'],
                        default='auto')
    parser.add_argument(
        '--save-per-sensor-diagnostics', action='store_true',
        help='Save current-frame per-sensor free/hit masks for occlusion audit.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_support_frames < 2:
        raise ValueError('min-support-frames must be at least 2')
    if 0 not in args.offsets:
        args.offsets.append(0)

    ann_path = _resolve_path(args.ann_file)
    infos, metainfo = _load_infos(ann_path)
    invalid = [index for index in args.indices
               if index < 0 or index >= len(infos)]
    if invalid:
        raise ValueError(f'Invalid frame indices: {invalid}')
    target_frame = args.target_frame
    if target_frame == 'auto':
        target_frame = str(metainfo.get('lidar_coord_frame', 'FLU'))
    if target_frame != 'FLU':
        raise ValueError(
            'Temporal ego2global alignment currently requires FLU labels')

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_builder = MultiLidarOccLabelBuilder(
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size,
        target_frame=target_frame,
        collision_z=args.collision_z,
    )
    cache: Dict[int, dict] = {}

    def build(index: int, diagnostics: bool = False) -> dict:
        needs_rebuild = (
            index not in cache or
            (diagnostics and 'per_sensor_free_3d' not in cache[index]))
        if needs_rebuild:
            cache[index] = label_builder.build(
                infos[index], diagnostics=diagnostics)
        return cache[index]

    summary_rows = []
    for reference_index in list(dict.fromkeys(args.indices)):
        info = infos[reference_index]
        stem = _output_stem(info)
        label_path = out_dir / f'{stem}__temporal.npz'
        if label_path.exists() and not args.overwrite:
            print(f'[{reference_index}] skip existing {label_path.name}')
            continue

        current = build(
            reference_index,
            diagnostics=args.save_per_sensor_diagnostics)
        source_pairs = _temporal_source_indices(
            infos, reference_index, args.offsets)
        support_count_xyz = np.zeros(
            tuple(np.asarray(args.occ_size, dtype=np.int64).tolist()),
            dtype=np.uint8)
        source_transforms = []
        reference_ego2global = np.asarray(
            info['ego2global'], dtype=np.float64)
        for offset, source_index in source_pairs:
            source = build(source_index)
            source_static_candidate = (
                (source['endpoint_type_3d'] ==
                 ENDPOINT_UNCERTAIN_OBSTACLE) |
                (source['endpoint_type_3d'] ==
                 ENDPOINT_RELIABLE_STATIC))
            source_ego2global = np.asarray(
                infos[source_index]['ego2global'], dtype=np.float64)
            warped = _warp_mask_to_reference_xyz(
                source_static_candidate,
                source_ego2global,
                reference_ego2global,
                args.pc_range,
                args.occ_size,
                radius_xy=args.match_radius_xy,
                radius_z=args.match_radius_z,
            )
            support_count_xyz += warped.astype(np.uint8)
            source_transforms.append(
                np.linalg.inv(reference_ego2global) @ source_ego2global)

        temporal_support_count_3d = _xyz_to_zhw(support_count_xyz)
        promoted = _promote_uncertain(
            current['endpoint_type_3d'], temporal_support_count_3d,
            args.min_support_frames)
        temporal_state_3d = current['filtered_state_3d'].copy()
        temporal_visibility_mask = current[
            'filtered_visibility_mask'].copy()
        temporal_occupancy_target = current[
            'filtered_occupancy_target'].copy()
        temporal_state_3d[promoted] = STATIC_OCCUPIED
        temporal_visibility_mask[promoted] = 1
        temporal_occupancy_target[promoted] = 1

        z_centers = current['z_centers']
        instance_bev = np.any(
            current['box_occupied_3d'] > 0, axis=0).astype(np.uint8)
        g_state = _project_observed_state(
            current['filtered_state_3d'], z_centers,
            args.collision_z, instance_bev,
            blocking_unknown_3d=current['uncertain_obstacle_3d'])
        unresolved_uncertain = (
            (current['endpoint_type_3d'] ==
             ENDPOINT_UNCERTAIN_OBSTACLE) & ~promoted)
        h_state = _project_observed_state(
            temporal_state_3d, z_centers,
            args.collision_z, instance_bev,
            blocking_unknown_3d=unresolved_uncertain)
        collision_keep = ((z_centers >= args.collision_z[0]) &
                          (z_centers <= args.collision_z[1]))
        promoted_bev = np.any(promoted[collision_keep], axis=0).astype(
            np.uint8)

        temporal_result = dict(current)
        temporal_result.update(
            temporal_state_3d=temporal_state_3d,
            temporal_visibility_mask=temporal_visibility_mask,
            temporal_occupancy_target=temporal_occupancy_target,
            temporal_support_count_3d=temporal_support_count_3d,
            promoted_uncertain_mask_3d=promoted.astype(np.uint8),
            unresolved_uncertain_mask_3d=unresolved_uncertain.astype(
                np.uint8),
            state_filtered_3d_projection=g_state,
            state_temporal_3d_projection=h_state,
            promoted_uncertain_bev=promoted_bev,
            temporal_offsets=np.asarray(
                [offset for offset, _ in source_pairs], dtype=np.int16),
            temporal_source_indices=np.asarray(
                [index for _, index in source_pairs], dtype=np.int64),
            temporal_source_timestamps=np.asarray([
                infos[index]['timestamp'] for _, index in source_pairs
            ], dtype=np.float64),
            temporal_source_to_reference=np.asarray(
                source_transforms, dtype=np.float64),
            temporal_min_support_frames=np.int16(
                args.min_support_frames),
            temporal_match_radius_xyz=np.asarray([
                args.match_radius_xy, args.match_radius_xy,
                args.match_radius_z,
            ], dtype=np.int16),
        )
        _save_label(
            label_path, temporal_result, info,
            args.pc_range, args.occ_size)
        _save_preview(out_dir / f'{stem}__G_filtered.png', g_state)
        _save_preview(out_dir / f'{stem}__H_temporal.png', h_state)
        _save_highlight_preview(
            out_dir / f'{stem}__H_highlight.png', h_state,
            promoted_bev)

        current_uncertain = int(np.count_nonzero(
            current['endpoint_type_3d'] ==
            ENDPOINT_UNCERTAIN_OBSTACLE))
        promoted_count = int(np.count_nonzero(promoted))
        summary = {
            'frame_index': reference_index,
            'stem': stem,
            'source_indices': ';'.join(
                str(index) for _, index in source_pairs),
            'source_offsets': ';'.join(
                str(offset) for offset, _ in source_pairs),
            'current_uncertain_3d': current_uncertain,
            'promoted_3d': promoted_count,
            'promotion_ratio': promoted_count / max(current_uncertain, 1),
            'promoted_collision_bev': int(promoted_bev.sum()),
            'g_h_changed_bev': int(np.count_nonzero(g_state != h_state)),
        }
        summary_rows.append(summary)
        print(
            f'[{reference_index}] sources={summary["source_offsets"]} '
            f'uncertain={current_uncertain} promoted={promoted_count} '
            f'({summary["promotion_ratio"]:.1%}) '
            f'G/H changed={summary["g_h_changed_bev"]}')

    if not summary_rows:
        return
    _write_summary(out_dir / 'temporal_summary.csv', summary_rows)
    _save_contact_sheet(out_dir, summary_rows)
    print(f'summary={out_dir / "temporal_summary.csv"}')
    print(f'contact={out_dir / "temporal_contact_sheet.png"}')


if __name__ == '__main__':
    main()
