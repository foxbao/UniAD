#!/usr/bin/env python
"""Visualize top low-static downgrade components against aligned LiDAR."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.analysis_tools.audit_kl_occworld_future_reveal import (
    _paint_point_band,
    _point_indices,
)
from tools.analysis_tools.audit_kl_occworld_low_static_stability import (
    _grid_metadata,
    _sequence_paths,
)
from tools.analysis_tools.evaluate_kl_occworld_low_static_downgrade import (
    apply_low_static_conflict_downgrade,
)
from tools.data_converter.generate_kl_occworld_labels import (
    MultiLidarOccLabelBuilder,
    _load_infos,
    _resolve_path,
)


STATE_PALETTE_BGR = np.asarray([
    [24, 26, 30],
    [90, 180, 70],
    [70, 80, 225],
    [240, 120, 65],
], dtype=np.uint8)


def _component_mask(source_static: np.ndarray, expected: dict,
                    x_centers: np.ndarray,
                    row_y_centers: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        source_static.astype(np.uint8), connectivity=8)
    matches = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area != int(expected['area_cells']):
            continue
        selection = labels == component
        rows, columns = np.where(selection)
        bbox_x = [
            float(x_centers[columns].min()),
            float(x_centers[columns].max())]
        bbox_y = [
            float(row_y_centers[rows].min()),
            float(row_y_centers[rows].max())]
        if (np.allclose(bbox_x, expected['bbox_x_m']) and
                np.allclose(bbox_y, expected['bbox_y_m'])):
            matches.append(selection)
    if len(matches) != 1:
        raise ValueError(
            f'Expected exactly one component match, found {len(matches)} '
            f'for reference {expected["reference_index"]}')
    return matches[0]


def _crop_bounds(mask: np.ndarray, margin_cells: int,
                 minimum_cells: int = 36) -> tuple:
    rows, columns = np.where(mask)
    if not len(rows):
        raise ValueError('Cannot crop an empty component')
    row_min = max(0, int(rows.min()) - margin_cells)
    row_max = min(mask.shape[0], int(rows.max()) + margin_cells + 1)
    col_min = max(0, int(columns.min()) - margin_cells)
    col_max = min(mask.shape[1], int(columns.max()) + margin_cells + 1)

    def expand(lower, upper, limit):
        if upper - lower >= minimum_cells:
            return lower, upper
        missing = max(0, minimum_cells - (upper - lower))
        lower = max(0, lower - missing // 2)
        upper = min(limit, upper + missing - missing // 2)
        lower = max(0, upper - minimum_cells)
        return lower, upper

    row_min, row_max = expand(row_min, row_max, mask.shape[0])
    col_min, col_max = expand(col_min, col_max, mask.shape[1])
    return row_min, row_max, col_min, col_max


def _draw_orientation(image: np.ndarray) -> None:
    origin = (34, image.shape[0] - 30)
    x_end = (84, image.shape[0] - 30)
    y_end = (34, image.shape[0] - 80)
    for end, color in ((x_end, (45, 55, 225)),
                       (y_end, (70, 190, 75))):
        cv2.arrowedLine(
            image, origin, end, (8, 10, 14), 5, cv2.LINE_AA,
            tipLength=0.22)
        cv2.arrowedLine(
            image, origin, end, color, 3, cv2.LINE_AA,
            tipLength=0.22)
    cv2.putText(
        image, '+x', (88, image.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (65, 75, 245), 1,
        cv2.LINE_AA)
    cv2.putText(
        image, '+y', (39, image.shape[0] - 84),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (85, 220, 90), 1,
        cv2.LINE_AA)


def _fit_image(image: np.ndarray, size=(480, 360)) -> np.ndarray:
    target_width, target_height = size
    scale = min(target_width / image.shape[1],
                target_height / image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((target_height, target_width, 3), 24, dtype=np.uint8)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    canvas[top:top + height, left:left + width] = resized
    _draw_orientation(canvas)
    return canvas


def _panel(image: np.ndarray, title: str,
           size=(480, 360)) -> np.ndarray:
    content = _fit_image(image, size)
    header = np.full((32, size[0], 3), 28, dtype=np.uint8)
    cv2.putText(
        header, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.50, (238, 238, 238), 1, cv2.LINE_AA)
    return np.concatenate([header, content], axis=0)


def _semantic_crop(state: np.ndarray, bounds: tuple,
                   component: np.ndarray,
                   downgraded: np.ndarray = None,
                   scale: int = 5) -> np.ndarray:
    row_min, row_max, col_min, col_max = bounds
    image = STATE_PALETTE_BGR[state]
    image = image[row_min:row_max, col_min:col_max]
    image = cv2.resize(
        image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_NEAREST)
    local_component = component[
        row_min:row_max, col_min:col_max].astype(np.uint8)
    local_component = cv2.resize(
        local_component, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(
        local_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (250, 245, 80), 2)
    if downgraded is not None:
        local_downgraded = downgraded[
            row_min:row_max, col_min:col_max]
        local_downgraded = cv2.resize(
            local_downgraded.astype(np.uint8), None,
            fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
        overlay = image.copy()
        overlay[local_downgraded] = (40, 220, 250)
        image = cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0)
    return image


def _effect_crop(component: np.ndarray, downgraded: np.ndarray,
                 original_next: np.ndarray, bounds: tuple,
                 scale: int = 5) -> np.ndarray:
    row_min, row_max, col_min, col_max = bounds
    image = np.full((*component.shape, 3), (24, 26, 30), dtype=np.uint8)
    remaining = component & ~downgraded
    next_free = component & (original_next == 1)
    fixed_conflict = downgraded & next_free
    image[component] = (92, 96, 104)
    image[remaining] = (70, 80, 225)
    image[downgraded] = (40, 220, 250)
    image[next_free] = (235, 200, 65)
    image[fixed_conflict] = (80, 220, 80)
    image = image[row_min:row_max, col_min:col_max]
    return cv2.resize(
        image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_NEAREST)


def _aligned_points(builder, info: dict,
                    transform: np.ndarray) -> np.ndarray:
    result = builder.build(
        info, diagnostics=False, return_points=True)
    points = np.asarray(result['points'], dtype=np.float32)
    homogeneous = np.concatenate([
        points[:, :3], np.ones((len(points), 1), dtype=np.float32)
    ], axis=1)
    aligned = points.copy()
    aligned[:, :3] = (homogeneous @ transform.T)[:, :3]
    return aligned


def _pointcloud_crop(points: np.ndarray, pc_range: np.ndarray,
                     occ_size: np.ndarray, collision_z: np.ndarray,
                     component: np.ndarray, downgraded: np.ndarray,
                     bounds: tuple, scale: int = 5) -> np.ndarray:
    image_shape = (
        int(occ_size[1]) * scale, int(occ_size[0]) * scale)
    image = np.full((*image_shape, 3), (20, 22, 26), dtype=np.uint8)
    rows, columns, heights = _point_indices(
        points, pc_range, image_shape)
    low = heights < collision_z[0]
    collision = ((heights >= collision_z[0]) &
                 (heights <= collision_z[1]))
    high = heights > collision_z[1]
    _paint_point_band(image, rows[low], columns[low], (180, 145, 100))
    _paint_point_band(image, rows[high], columns[high], (205, 105, 205))
    _paint_point_band(
        image, rows[collision], columns[collision], (70, 220, 250))
    component_high = cv2.resize(
        component.astype(np.uint8), None, fx=scale, fy=scale,
        interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(
        component_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (250, 245, 80), 2)
    downgrade_high = cv2.resize(
        downgraded.astype(np.uint8), None, fx=scale, fy=scale,
        interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(
        downgrade_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (40, 220, 250), 2)
    row_min, row_max, col_min, col_max = bounds
    return image[
        row_min * scale:row_max * scale,
        col_min * scale:col_max * scale]


def _legend(width: int) -> np.ndarray:
    image = np.full((56, width, 3), 28, dtype=np.uint8)
    entries = (
        ('unknown', (24, 26, 30)), ('free', (90, 180, 70)),
        ('static', (70, 80, 225)), ('instance', (240, 120, 65)),
        ('component outline', (250, 245, 80)),
        ('downgraded', (40, 220, 250)),
        ('remaining hard conflict', (235, 200, 65)),
        ('fixed static->free', (80, 220, 80)))
    x = 12
    for label, color in entries:
        cv2.rectangle(image, (x, 17), (x + 18, 35), color, -1)
        cv2.putText(
            image, label, (x + 24, 32), cv2.FONT_HERSHEY_SIMPLEX,
            0.43, (235, 235, 235), 1, cv2.LINE_AA)
        x += 32 + max(82, len(label) * 8)
    return image


def _contact_sheet(panels: list) -> np.ndarray:
    top = np.concatenate(panels[:3], axis=1)
    bottom = np.concatenate(panels[3:], axis=1)
    separator = np.full((8, top.shape[1], 3), 28, dtype=np.uint8)
    body = np.concatenate([top, separator, bottom], axis=0)
    return np.concatenate([body, _legend(body.shape[1])], axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ablation-json', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_low_static_downgrade_ablation_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_full_train_v1'))
    parser.add_argument(
        '--ann-file', type=Path,
        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument('--top-count', type=int, default=20)
    parser.add_argument('--margin-cells', type=int, default=6)
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_low_static_downgrade_visuals_v1'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_count < 1 or args.margin_cells < 0:
        raise ValueError('Visualization limits are invalid')
    ablation = json.loads(args.ablation_json.read_text())
    rule = ablation['rule']
    records = ablation['labels'][
        'top_components_by_original_next_free'][:args.top_count]
    sequences = _sequence_paths(args.sequence_root)
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    grid_payload = ablation['labels']['grid']
    pc_range = np.asarray(grid_payload['pc_range'], dtype=np.float32)
    occ_size = np.asarray(grid_payload['occ_size'], dtype=np.int64)
    collision_z = np.asarray(grid_payload['collision_z'], dtype=np.float32)
    grid = _grid_metadata(pc_range, occ_size, collision_z)
    builder = MultiLidarOccLabelBuilder(
        pc_range, (int(occ_size[1]), int(occ_size[0])), occ_size,
        target_frame=str(metainfo.get('lidar_coord_frame', 'FLU')),
        collision_z=collision_z)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    contacts = []

    for rank, record in enumerate(records, start=1):
        reference = int(record['reference_index'])
        source_horizon = int(record['source_horizon'])
        with np.load(sequences[reference], allow_pickle=False) as archive:
            world = np.asarray(
                archive['world_target_state_3d'], dtype=np.uint8)
            direct = np.asarray(
                archive['direct_observation_state_3d'], dtype=np.uint8)
            future_free = np.asarray(
                archive['future_free_count_3d'], dtype=np.uint8)
            target_indices = np.asarray(
                archive['target_indices'], dtype=np.int64)
            transforms = np.asarray(
                archive['target_to_reference'], dtype=np.float64)
            target_times = np.asarray(
                archive['target_times_s'], dtype=np.float32)
        low_index = int(rule['target_z_index'])
        candidate, downgrade = apply_low_static_conflict_downgrade(
            world, direct, future_free, low_index,
            int(rule['future_free_count_minimum']))
        source_static = world[source_horizon, low_index] == 2
        component = _component_mask(
            source_static, record, grid['x_centers'],
            grid['row_y_centers'])
        component_downgrade = downgrade[source_horizon] & component
        bounds = _crop_bounds(
            component, args.margin_cells)
        target_index = int(target_indices[source_horizon])
        points = _aligned_points(
            builder, infos[target_index], transforms[source_horizon])
        next_horizon = source_horizon + 1
        panels = [
            _panel(_pointcloud_crop(
                points, pc_range, occ_size, collision_z,
                component, component_downgrade, bounds),
                f'aligned 8-LiDAR | t={target_times[source_horizon]:.1f}s'),
            _panel(_semantic_crop(
                world[source_horizon, low_index], bounds, component),
                'original source GT | yellow: audited component'),
            _panel(_semantic_crop(
                candidate[source_horizon], bounds, component,
                component_downgrade),
                'candidate source GT | yellow fill: downgraded'),
            _panel(_semantic_crop(
                world[next_horizon, low_index], bounds, component),
                f'original next GT | t={target_times[next_horizon]:.1f}s'),
            _panel(_semantic_crop(
                candidate[next_horizon], bounds, component,
                downgrade[next_horizon]),
                'candidate next GT'),
            _panel(_effect_crop(
                component, component_downgrade,
                world[next_horizon, low_index], bounds),
                'rule effect | green: removed hard conflict'),
        ]
        contact = _contact_sheet(panels)
        path = args.out_dir / (
            f'rank_{rank:02d}__ref_{reference:06d}__h{source_horizon}.png')
        if not cv2.imwrite(str(path), contact):
            raise OSError(f'Failed to write {path}')
        contacts.append(contact)
        rendered.append({
            'rank': rank,
            'reference_index': reference,
            'source_horizon': source_horizon,
            'target_frame_index': target_index,
            'point_count': int(len(points)),
            'component_area_cells': int(record['area_cells']),
            'downgraded_cells': int(record['downgraded_cells']),
            'original_next_free_cells': int(
                record['original_transition']['next_free_voxels']),
            'candidate_next_free_cells': int(
                record['candidate_transition']['next_free_voxels']),
            'contact_path': str(path),
        })
        print(
            f'[{rank}/{len(records)}] ref={reference} h={source_horizon} '
            f'points={len(points)} path={path.name}')

    overview_tiles = [
        cv2.resize(contact, (960, 540), interpolation=cv2.INTER_AREA)
        for contact in contacts]
    blank = np.full_like(overview_tiles[0], 28)
    overview_rows = []
    for start in range(0, len(overview_tiles), 2):
        row = overview_tiles[start:start + 2]
        if len(row) == 1:
            row.append(blank)
        overview_rows.append(np.concatenate(row, axis=1))
    overview = np.concatenate(overview_rows, axis=0)
    overview_path = args.out_dir / 'top20_low_static_downgrade_overview.png'
    if not cv2.imwrite(str(overview_path), overview):
        raise OSError(f'Failed to write {overview_path}')
    summary = {
        'schema_version': 1,
        'ablation_json': str(args.ablation_json),
        'rule': rule,
        'record_count': len(rendered),
        'overview_path': str(overview_path),
        'records': rendered,
    }
    summary_path = args.out_dir / 'summary.json'
    with summary_path.open('w') as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    print(json.dumps({
        'summary_path': str(summary_path),
        'overview_path': str(overview_path),
        'record_count': len(rendered),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
