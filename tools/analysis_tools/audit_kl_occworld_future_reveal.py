#!/usr/bin/env python
"""Use future LiDAR frames to reveal regions removed by occlusion audit."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.data_converter.generate_kl_occworld_labels import (
    ENDPOINT_INSTANCE,
    ENDPOINT_RELIABLE_STATIC,
    ENDPOINT_UNCERTAIN_OBSTACLE,
    FREE,
    MultiLidarOccLabelBuilder,
    _load_infos,
    _resolve_path,
    _xyz_to_zhw,
)
from tools.data_converter.generate_kl_occworld_temporal_labels import (
    _warp_mask_to_reference_xyz,
)


REVEAL_UNKNOWN = 0
REVEAL_STATIC_OCCUPIED = 1
REVEAL_DYNAMIC_OCCUPIED = 2
REVEAL_REPEATED_FREE = 3
REVEAL_SINGLE_FREE = 4


def _classify_reveal(static_count: np.ndarray,
                     dynamic_count: np.ndarray,
                     free_count: np.ndarray,
                     removed: np.ndarray) -> np.ndarray:
    reveal = np.full(removed.shape, REVEAL_UNKNOWN, dtype=np.uint8)
    reveal[removed & (free_count == 1)] = REVEAL_SINGLE_FREE
    reveal[removed & (free_count >= 2)] = REVEAL_REPEATED_FREE
    reveal[removed & (dynamic_count > 0)] = REVEAL_DYNAMIC_OCCUPIED
    reveal[removed & (static_count > 0)] = REVEAL_STATIC_OCCUPIED
    return reveal


def _temporal_index_map(temporal_dir: Path) -> Dict[int, Path]:
    mapping = {}
    for path in temporal_dir.glob('*__temporal.npz'):
        with np.load(path, allow_pickle=False) as label:
            offsets = label['temporal_offsets']
            indices = label['temporal_source_indices']
            current = np.flatnonzero(offsets == 0)
            if current.size != 1:
                continue
            mapping[int(indices[current[0]])] = path
    return mapping


def _occlusion_index_map(occlusion_dir: Path) -> Dict[str, Path]:
    mapping = {}
    suffix = '__occlusion'
    for path in occlusion_dir.glob('*__occlusion.npz'):
        stem = path.stem
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
        mapping[stem] = path
    return mapping


def _future_indices(infos: List[dict], reference_index: int,
                    offsets: Sequence[int]) -> List[Tuple[int, int]]:
    scene = infos[reference_index].get('scene_token')
    result = []
    for offset in offsets:
        index = reference_index + int(offset)
        if index < 0 or index >= len(infos):
            continue
        if infos[index].get('scene_token') != scene:
            continue
        result.append((int(offset), index))
    return result


def _project_future_frame(source: dict,
                          source_ego2global: np.ndarray,
                          reference_ego2global: np.ndarray,
                          pc_range: Sequence[float],
                          occ_size: Sequence[int],
                          collision_keep: np.ndarray):
    endpoint_type = source['endpoint_type_3d']
    masks = {
        'static': endpoint_type == ENDPOINT_RELIABLE_STATIC,
        'dynamic': (
            (endpoint_type == ENDPOINT_INSTANCE) |
            (source['box_occupied_3d'] > 0)),
        'uncertain': endpoint_type == ENDPOINT_UNCERTAIN_OBSTACLE,
        'free': source['filtered_state_3d'] == FREE,
    }
    warped = {}
    for key, mask in masks.items():
        xyz = _warp_mask_to_reference_xyz(
            mask, source_ego2global, reference_ego2global,
            pc_range, occ_size, radius_xy=0, radius_z=0)
        warped[key] = _xyz_to_zhw(xyz)
    static_bev = np.any(warped['static'][collision_keep], axis=0)
    dynamic_bev = np.any(warped['dynamic'][collision_keep], axis=0)
    uncertain_bev = np.any(warped['uncertain'][collision_keep], axis=0)
    free_bev = np.any(warped['free'][collision_keep], axis=0)
    free_bev &= ~static_bev & ~dynamic_bev & ~uncertain_bev
    return static_bev, dynamic_bev, free_bev


def _save_reveal_preview(path: Path, k_state: np.ndarray,
                         reveal: np.ndarray, removed: np.ndarray,
                         scale: int = 4):
    state_palette = np.asarray([
        [40, 40, 40], [70, 180, 90],
        [225, 80, 70], [65, 120, 240],
    ], dtype=np.uint8)
    rgb = state_palette[k_state]
    rgb[removed] = (150, 80, 170)
    rgb[reveal == REVEAL_SINGLE_FREE] = (235, 190, 55)
    rgb[reveal == REVEAL_REPEATED_FREE] = (55, 215, 215)
    rgb[reveal == REVEAL_DYNAMIC_OCCUPIED] = (65, 120, 240)
    rgb[reveal == REVEAL_STATIC_OCCUPIED] = (235, 65, 70)
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _point_indices(points: np.ndarray, pc_range: Sequence[float],
                   image_shape: Tuple[int, int]):
    height, width = image_shape
    pc_range = np.asarray(pc_range, dtype=np.float32)
    cols = np.floor(
        (points[:, 0] - pc_range[0]) /
        (pc_range[3] - pc_range[0]) * width).astype(np.int32)
    rows = np.floor(
        (pc_range[4] - points[:, 1]) /
        (pc_range[4] - pc_range[1]) * height).astype(np.int32)
    valid = (
        (cols >= 0) & (cols < width) &
        (rows >= 0) & (rows < height) &
        (points[:, 2] >= pc_range[2]) &
        (points[:, 2] < pc_range[5]))
    return rows[valid], cols[valid], points[valid, 2]


def _paint_point_band(image: np.ndarray, rows: np.ndarray,
                      cols: np.ndarray, color: Tuple[int, int, int]):
    if rows.size == 0:
        return
    density = np.zeros(image.shape[:2], dtype=np.uint16)
    np.add.at(density, (rows, cols), 1)
    active = density > 0
    strength = np.clip(
        np.log1p(density[active]) / np.log(6.0), 0.45, 1.0)
    background = image[active].astype(np.float32)
    foreground = np.asarray(color, dtype=np.float32)[None, :]
    image[active] = (
        background * (1.0 - strength[:, None]) +
        foreground * strength[:, None]).astype(np.uint8)


def _save_pointcloud_bev(path: Path, points: np.ndarray,
                         pc_range: Sequence[float],
                         bev_shape: Tuple[int, int],
                         collision_z: Sequence[float], scale: int = 4):
    image_shape = (bev_shape[0] * scale, bev_shape[1] * scale)
    image = np.full((*image_shape, 3), 40, dtype=np.uint8)
    rows, cols, heights = _point_indices(points, pc_range, image_shape)
    low = heights < collision_z[0]
    collision = ((heights >= collision_z[0]) &
                 (heights <= collision_z[1]))
    high = heights > collision_z[1]
    # BGR colors. Collision-height points are painted last so thin obstacle
    # structures remain visible over dense ground returns.
    _paint_point_band(image, rows[low], cols[low], (165, 135, 95))
    _paint_point_band(image, rows[high], cols[high], (195, 105, 190))
    _paint_point_band(image, rows[collision], cols[collision],
                      (70, 205, 245))
    cv2.imwrite(str(path), image)


def _bev_pixel(x: float, y: float, pc_range: Sequence[float],
               shape: Tuple[int, int]):
    height, width = shape
    col = int(round(
        (x - pc_range[0]) / (pc_range[3] - pc_range[0]) * width))
    row = int(round(
        (pc_range[4] - y) / (pc_range[4] - pc_range[1]) * height))
    return col, row


def _draw_bev_guides(image: np.ndarray, pc_range: Sequence[float]):
    overlay = image.copy()
    x_ticks = np.arange(
        np.ceil(pc_range[0] / 10.0) * 10.0, pc_range[3], 10.0)
    y_ticks = np.arange(
        np.ceil(pc_range[1] / 10.0) * 10.0, pc_range[4], 10.0)
    for x in x_ticks:
        col, _ = _bev_pixel(x, 0.0, pc_range, image.shape[:2])
        cv2.line(overlay, (col, 0), (col, image.shape[0] - 1),
                 (105, 105, 105), 1)
    for y in y_ticks:
        _, row = _bev_pixel(0.0, y, pc_range, image.shape[:2])
        cv2.line(overlay, (0, row), (image.shape[1] - 1, row),
                 (105, 105, 105), 1)
    cv2.addWeighted(overlay, 0.22, image, 0.78, 0.0, image)
    ego = _bev_pixel(0.0, 0.0, pc_range, image.shape[:2])
    cv2.drawMarker(image, ego, (245, 245, 245), cv2.MARKER_CROSS,
                   13, 1, cv2.LINE_AA)


def _image_panel(image: np.ndarray, title: str,
                 size: Tuple[int, int]) -> np.ndarray:
    image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    header = np.full((28, size[0], 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def _save_pointcloud_comparison(path: Path, reveal_path: Path,
                                point_path: Path,
                                pc_range: Sequence[float]):
    reveal = cv2.imread(str(reveal_path))
    points = cv2.imread(str(point_path))
    if reveal is None or points is None:
        raise FileNotFoundError('Missing reveal or point-cloud preview')
    if reveal.shape != points.shape:
        raise ValueError(
            f'Preview shapes differ: {reveal.shape} vs {points.shape}')
    _draw_bev_guides(reveal, pc_range)
    _draw_bev_guides(points, pc_range)
    left = _image_panel(reveal, 'L future reveal | x right, y left up',
                        (reveal.shape[1], reveal.shape[0]))
    right = _image_panel(
        points, 'Current 8-LiDAR points | same coordinates',
        (points.shape[1], points.shape[0]))
    separator = np.full((left.shape[0], 8, 3), 28, dtype=np.uint8)
    canvas = np.concatenate([left, separator, right], axis=1)
    legend = np.full((42, canvas.shape[1], 3), 28, dtype=np.uint8)
    entries = [
        ('point z < 0.3m', (165, 135, 95)),
        ('point z 0.3-2.5m', (70, 205, 245)),
        ('point z > 2.5m', (195, 105, 190)),
        ('white cross: ego', (245, 245, 245)),
    ]
    x = 12
    for label, color in entries:
        cv2.rectangle(legend, (x, 12), (x + 18, 30), color, -1)
        cv2.putText(legend, label, (x + 24, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (235, 235, 235), 1, cv2.LINE_AA)
        x += 24 + max(150, len(label) * 8)
    cv2.imwrite(str(path), np.concatenate([canvas, legend], axis=0))


def _tile(path: Path, title: str,
          size: Tuple[int, int] = (480, 360)) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return _image_panel(image, title, size)


def _save_contact_sheet(out_dir: Path, rows: List[dict]):
    panels = []
    for row in rows:
        panels.append(_tile(
            out_dir / f"{row['stem']}__L_future_reveal.png",
            f"#{row['frame_index']} occ={row['static_occupied_ratio']:.1%} "
            f"free2={row['repeated_free_ratio']:.1%}"))
    columns = 2
    grid = []
    for start in range(0, len(panels), columns):
        grid.append(np.concatenate(panels[start:start + columns], axis=1))
    sheet = np.concatenate(grid, axis=0)
    legend = np.full((44, sheet.shape[1], 3), 28, dtype=np.uint8)
    entries = [
        ('static occupied', (70, 65, 235)),
        ('dynamic occupied', (240, 120, 65)),
        ('repeated free', (215, 215, 55)),
        ('single free', (55, 190, 235)),
        ('still unknown', (170, 80, 150)),
    ]
    x = 10
    for text, color in entries:
        cv2.rectangle(legend, (x, 13), (x + 18, 31), color, -1)
        cv2.putText(legend, text, (x + 24, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (235, 235, 235), 1, cv2.LINE_AA)
        x += 24 + max(105, len(text) * 8)
    cv2.imwrite(str(out_dir / 'future_reveal_contact_sheet.png'),
                np.concatenate([legend, sheet], axis=0))


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--temporal-dir', required=True)
    parser.add_argument('--occlusion-dir', required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--future-offsets', type=int, nargs='+',
                        default=[1, 2, 3, 4])
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument('--occ-size', type=int, nargs=3,
                        default=[160, 120, 10])
    parser.add_argument('--collision-z', type=float, nargs=2,
                        default=[0.3, 2.5])
    parser.add_argument(
        '--save-pointcloud-bev', action='store_true',
        help='Save current fused 8-LiDAR BEV and an L/point comparison.')
    return parser.parse_args()


def main():
    args = parse_args()
    ann_path = _resolve_path(args.ann_file)
    infos, metainfo = _load_infos(ann_path)
    temporal_dir = Path(args.temporal_dir)
    occlusion_dir = Path(args.occlusion_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporal_map = _temporal_index_map(temporal_dir)
    occlusion_map = _occlusion_index_map(occlusion_dir)
    if args.indices:
        indices = list(dict.fromkeys(args.indices))
    else:
        with (occlusion_dir / 'summary.json').open() as f:
            indices = json.load(f)['worst8_frame_indices']

    target_frame = str(metainfo.get('lidar_coord_frame', 'FLU'))
    if target_frame != 'FLU':
        raise ValueError('Future reveal currently requires FLU labels')
    builder = MultiLidarOccLabelBuilder(
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size,
        target_frame=target_frame,
        collision_z=args.collision_z,
    )
    cache: Dict[int, dict] = {}

    def build(index: int) -> dict:
        if index not in cache:
            cache[index] = builder.build(infos[index], diagnostics=False)
        return cache[index]

    rows = []
    for reference_index in indices:
        temporal_path = temporal_map[reference_index]
        with np.load(temporal_path, allow_pickle=False) as temporal:
            offsets = temporal['temporal_offsets']
            source_indices = temporal['temporal_source_indices']
            current_pos = np.flatnonzero(offsets == 0)
            if current_pos.size != 1:
                raise ValueError(f'Missing current source in {temporal_path}')
            stem_suffix = '__temporal'
            stem = temporal_path.stem
            if stem.endswith(stem_suffix):
                stem = stem[:-len(stem_suffix)]
            occlusion_path = occlusion_map[stem]
            z_centers = temporal['z_centers']
            collision_z = temporal['collision_z']
            collision_keep = ((z_centers >= collision_z[0]) &
                              (z_centers <= collision_z[1]))
        with np.load(occlusion_path, allow_pickle=False) as occlusion:
            k_state = occlusion['state_occlusion_corrected']
            removed = occlusion['occlusion_removed_bev'] > 0

        future_pairs = _future_indices(
            infos, reference_index, args.future_offsets)
        static_count = np.zeros(removed.shape, dtype=np.uint8)
        dynamic_count = np.zeros(removed.shape, dtype=np.uint8)
        free_count = np.zeros(removed.shape, dtype=np.uint8)
        reference_ego2global = np.asarray(
            infos[reference_index]['ego2global'], dtype=np.float64)
        for _, future_index in future_pairs:
            source = build(future_index)
            static_bev, dynamic_bev, free_bev = _project_future_frame(
                source,
                np.asarray(infos[future_index]['ego2global'],
                           dtype=np.float64),
                reference_ego2global,
                args.pc_range,
                args.occ_size,
                collision_keep,
            )
            static_count += static_bev.astype(np.uint8)
            dynamic_count += dynamic_bev.astype(np.uint8)
            free_count += free_bev.astype(np.uint8)

        reveal = _classify_reveal(
            static_count, dynamic_count, free_count, removed)
        removed_count = max(int(removed.sum()), 1)
        counts = {
            'static_occupied': int(np.count_nonzero(
                reveal == REVEAL_STATIC_OCCUPIED)),
            'dynamic_occupied': int(np.count_nonzero(
                reveal == REVEAL_DYNAMIC_OCCUPIED)),
            'repeated_free': int(np.count_nonzero(
                reveal == REVEAL_REPEATED_FREE)),
            'single_free': int(np.count_nonzero(
                reveal == REVEAL_SINGLE_FREE)),
            'still_unknown': int(np.count_nonzero(
                removed & (reveal == REVEAL_UNKNOWN))),
        }
        row = {
            'frame_index': reference_index,
            'stem': stem,
            'future_indices': ';'.join(
                str(index) for _, index in future_pairs),
            'future_offsets': ';'.join(
                str(offset) for offset, _ in future_pairs),
            'removed_cells': int(removed.sum()),
            **counts,
            'static_occupied_ratio': counts['static_occupied'] / removed_count,
            'dynamic_occupied_ratio': counts['dynamic_occupied'] / removed_count,
            'repeated_free_ratio': counts['repeated_free'] / removed_count,
            'single_free_ratio': counts['single_free'] / removed_count,
            'still_unknown_ratio': counts['still_unknown'] / removed_count,
        }
        rows.append(row)
        reveal_path = out_dir / f'{stem}__L_future_reveal.png'
        _save_reveal_preview(reveal_path, k_state, reveal, removed)
        if args.save_pointcloud_bev:
            current = builder.build(
                infos[reference_index], diagnostics=False,
                return_points=True)
            point_path = out_dir / f'{stem}__pointcloud_bev.png'
            comparison_path = out_dir / f'{stem}__L_pointcloud_compare.png'
            _save_pointcloud_bev(
                point_path, current['points'], args.pc_range,
                tuple(k_state.shape), args.collision_z)
            _save_pointcloud_comparison(
                comparison_path, reveal_path, point_path, args.pc_range)
            print(f'pointcloud_comparison={comparison_path}')
        np.savez_compressed(
            out_dir / f'{stem}__future_reveal.npz',
            reveal_state=reveal,
            future_static_count=static_count,
            future_dynamic_count=dynamic_count,
            future_free_count=free_count,
            occlusion_removed_bev=removed.astype(np.uint8),
            future_indices=np.asarray(
                [index for _, index in future_pairs], dtype=np.int64),
            future_offsets=np.asarray(
                [offset for offset, _ in future_pairs], dtype=np.int16),
        )
        print(
            f'[{reference_index}] removed={row["removed_cells"]} '
            f'static_occ={row["static_occupied_ratio"]:.1%} '
            f'free2={row["repeated_free_ratio"]:.1%} '
            f'unknown={row["still_unknown_ratio"]:.1%}')

    _write_csv(out_dir / 'future_reveal_metrics.csv', rows)
    total_removed = max(sum(row['removed_cells'] for row in rows), 1)
    totals = {key: sum(row[key] for row in rows) for key in (
        'static_occupied', 'dynamic_occupied', 'repeated_free',
        'single_free', 'still_unknown')}
    summary = {
        'frame_count': len(rows),
        'total_removed_cells': total_removed,
        'totals': totals,
        'ratios': {key: value / total_removed
                   for key, value in totals.items()},
    }
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _save_contact_sheet(out_dir, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
