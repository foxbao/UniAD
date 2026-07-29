#!/usr/bin/env python
"""Audit spatial expansion of one ground-level static OccWorld component."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    grouped_linear_percentile,
)
from tools.analysis_tools.audit_kl_occworld_low_static_stability import (
    _distribution,
    _grid_metadata,
    _prediction_paths,
    _sequence_paths,
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


def _xyz_to_zhw(volume: np.ndarray) -> np.ndarray:
    return np.transpose(volume, (2, 1, 0))[:, ::-1, :]


def _xy_to_hw(volume: np.ndarray) -> np.ndarray:
    return volume.T[::-1, :]


def _front_left_component(state_low: np.ndarray,
                          x_centers: np.ndarray,
                          row_y_centers: np.ndarray) -> np.ndarray:
    front_left = (
        (row_y_centers[:, None] >= 0) &
        (x_centers[None, :] >= 0))
    mask = (state_low == 2) & front_left
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        raise ValueError('No front-left static component was found')
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def _crop_bounds(mask: np.ndarray, margin: int,
                 minimum_cells: int = 40) -> tuple:
    rows, columns = np.where(mask)
    if not len(rows):
        raise ValueError('Cannot crop an empty component')
    row_min = max(0, int(rows.min()) - margin)
    row_max = min(mask.shape[0], int(rows.max()) + margin + 1)
    col_min = max(0, int(columns.min()) - margin)
    col_max = min(mask.shape[1], int(columns.max()) + margin + 1)

    def expand(lower, upper, limit):
        if upper - lower >= minimum_cells:
            return lower, upper
        missing = minimum_cells - (upper - lower)
        lower = max(0, lower - missing // 2)
        upper = min(limit, upper + missing - missing // 2)
        lower = max(0, upper - minimum_cells)
        return lower, upper

    row_min, row_max = expand(row_min, row_max, mask.shape[0])
    col_min, col_max = expand(col_min, col_max, mask.shape[1])
    return row_min, row_max, col_min, col_max


def _finite_distribution(values: np.ndarray,
                         selection: np.ndarray) -> dict:
    selected = np.asarray(values)[selection]
    return _distribution(selected[np.isfinite(selected)])


def _cell_point_statistics(points: np.ndarray, builder,
                           ground_estimate: np.ndarray) -> dict:
    point_voxels = builder.coord_to_index_floor(points[:, :3])
    valid = np.all(
        (point_voxels >= 0) &
        (point_voxels < builder.occ_size[None, :]), axis=1)
    points = points[valid]
    point_voxels = point_voxels[valid]
    ground = ground_estimate[
        point_voxels[:, 0], point_voxels[:, 1]]
    relative_height = points[:, 2] - ground
    finite = np.isfinite(relative_height)
    points = points[finite]
    point_voxels = point_voxels[finite]
    relative_height = relative_height[finite]
    cell_count = int(builder.occ_size[0] * builder.occ_size[1])
    flat_cell = (
        point_voxels[:, 0].astype(np.int64) * int(builder.occ_size[1]) +
        point_voxels[:, 1].astype(np.int64))
    p50, _ = grouped_linear_percentile(
        flat_cell, relative_height, 0.50, cell_count)
    p90, _ = grouped_linear_percentile(
        flat_cell, relative_height, 0.90, cell_count)
    minimum = np.full(cell_count, np.nan, dtype=np.float32)
    maximum = np.full(cell_count, np.nan, dtype=np.float32)
    order = np.argsort(flat_cell)
    sorted_cells = flat_cell[order]
    sorted_height = relative_height[order]
    unique, first = np.unique(sorted_cells, return_index=True)
    minimum[unique] = np.minimum.reduceat(sorted_height, first)
    maximum[unique] = np.maximum.reduceat(sorted_height, first)
    p50 = p50.reshape(tuple(builder.occ_size[:2]))
    p90 = p90.reshape(tuple(builder.occ_size[:2]))
    minimum = minimum.reshape(tuple(builder.occ_size[:2]))
    maximum = maximum.reshape(tuple(builder.occ_size[:2]))

    low_z_index = int(np.argmin(np.abs(builder.voxel_centers(
        2, np.arange(builder.occ_size[2])))))
    low = point_voxels[:, 2] == low_z_index
    low_flat = flat_cell[low]
    low_height = relative_height[low]
    low_p50, _ = grouped_linear_percentile(
        low_flat, low_height, 0.50, cell_count)
    low_p90, _ = grouped_linear_percentile(
        low_flat, low_height, 0.90, cell_count)
    low_p50 = low_p50.reshape(tuple(builder.occ_size[:2]))
    low_p90 = low_p90.reshape(tuple(builder.occ_size[:2]))
    return {
        'points': points,
        'point_voxels': point_voxels,
        'relative_height': relative_height,
        'p50_xy': p50,
        'p90_xy': p90,
        'min_xy': minimum,
        'max_xy': maximum,
        'vertical_span_xy': maximum - minimum,
        'low_p50_xy': low_p50,
        'low_p90_xy': low_p90,
        'low_z_index': low_z_index,
    }


def _distance_from_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.full(mask.shape, np.nan, dtype=np.float32)
    return cv2.distanceTransform(
        (~mask).astype(np.uint8), cv2.DIST_L2, 5)


def _outline(image: np.ndarray, mask: np.ndarray,
             color=(250, 245, 80), thickness=2) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)


def _crop_scale(image: np.ndarray, bounds: tuple,
                scale: int = 6) -> np.ndarray:
    row_min, row_max, col_min, col_max = bounds
    crop = image[row_min:row_max, col_min:col_max]
    return cv2.resize(
        crop, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_NEAREST)


def _semantic_image(state: np.ndarray, component: np.ndarray,
                    bounds: tuple, overlay: np.ndarray = None) -> np.ndarray:
    image = STATE_PALETTE_BGR[state]
    image = _crop_scale(image, bounds)
    component_crop = _crop_scale(
        component.astype(np.uint8), bounds) > 0
    _outline(image, component_crop)
    if overlay is not None:
        overlay_crop = _crop_scale(
            overlay.astype(np.uint8), bounds) > 0
        blended = image.copy()
        blended[overlay_crop] = (40, 220, 250)
        image = cv2.addWeighted(blended, 0.72, image, 0.28, 0.0)
    return image


def _binary_image(mask: np.ndarray, component: np.ndarray,
                  bounds: tuple, color=(70, 80, 225)) -> np.ndarray:
    image = np.full((*mask.shape, 3), (24, 26, 30), dtype=np.uint8)
    image[mask] = color
    image = _crop_scale(image, bounds)
    component_crop = _crop_scale(
        component.astype(np.uint8), bounds) > 0
    _outline(image, component_crop)
    return image


def _heatmap_image(values: np.ndarray, valid: np.ndarray,
                   bounds: tuple, minimum: float, maximum: float,
                   component: np.ndarray) -> np.ndarray:
    normalized = np.clip(
        (values - minimum) / max(maximum - minimum, 1e-6), 0.0, 1.0)
    heatmap = cv2.applyColorMap(
        (np.nan_to_num(normalized, nan=0.0) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO)
    heatmap[~valid] = (24, 26, 30)
    heatmap = _crop_scale(heatmap, bounds)
    component_crop = _crop_scale(
        component.astype(np.uint8), bounds) > 0
    _outline(heatmap, component_crop)
    return heatmap


def _relative_pointcloud_image(points: np.ndarray,
                               relative_height: np.ndarray,
                               pc_range: np.ndarray,
                               occ_size: np.ndarray,
                               bounds: tuple,
                               component: np.ndarray,
                               scale: int = 6) -> np.ndarray:
    height = int(occ_size[1]) * scale
    width = int(occ_size[0]) * scale
    columns = np.floor(
        (points[:, 0] - pc_range[0]) /
        (pc_range[3] - pc_range[0]) * width).astype(np.int32)
    rows = np.floor(
        (pc_range[4] - points[:, 1]) /
        (pc_range[4] - pc_range[1]) * height).astype(np.int32)
    valid = (
        (columns >= 0) & (columns < width) &
        (rows >= 0) & (rows < height) &
        np.isfinite(relative_height))
    columns = columns[valid]
    rows = rows[valid]
    values = relative_height[valid]
    maximum = np.full(height * width, -np.inf, dtype=np.float32)
    flat = rows.astype(np.int64) * width + columns
    np.maximum.at(maximum, flat, values)
    active = np.isfinite(maximum)
    normalized = np.clip((maximum + 0.2) / 2.2, 0.0, 1.0)
    colors = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8).reshape(height, width),
        cv2.COLORMAP_TURBO)
    image = np.full((height, width, 3), (20, 22, 26), dtype=np.uint8)
    image.reshape(-1, 3)[active] = colors.reshape(-1, 3)[active]
    component_high = cv2.resize(
        component.astype(np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST)
    _outline(image, component_high, thickness=2)
    row_min, row_max, col_min, col_max = bounds
    return image[
        row_min * scale:row_max * scale,
        col_min * scale:col_max * scale]


def _fit_image(image: np.ndarray, size=(420, 320)) -> np.ndarray:
    width, height = size
    ratio = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * ratio))),
         max(1, int(round(image.shape[0] * ratio)))),
        interpolation=cv2.INTER_NEAREST)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    left = (width - resized.shape[1]) // 2
    top = (height - resized.shape[0]) // 2
    canvas[top:top + resized.shape[0],
           left:left + resized.shape[1]] = resized
    return canvas


def _panel(image: np.ndarray, title: str,
           size=(420, 320)) -> np.ndarray:
    content = _fit_image(image, size)
    header = np.full((32, size[0], 3), 28, dtype=np.uint8)
    cv2.putText(
        header, title, (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (238, 238, 238), 1, cv2.LINE_AA)
    return np.concatenate([header, content], axis=0)


def _evidence_legend(width: int) -> np.ndarray:
    legend = np.full((64, width, 3), 28, dtype=np.uint8)
    entries = (
        ('unknown', (24, 26, 30)), ('free', (90, 180, 70)),
        ('static', (70, 80, 225)), ('instance', (240, 120, 65)),
        ('audited component', (250, 245, 80)),
        ('completion/model-only', (40, 220, 250)))
    x = 12
    for label, color in entries:
        cv2.rectangle(legend, (x, 17), (x + 18, 35), color, -1)
        cv2.putText(
            legend, label, (x + 24, 32), cv2.FONT_HERSHEY_SIMPLEX,
            0.44, (235, 235, 235), 1, cv2.LINE_AA)
        x += 32 + max(90, len(label) * 8)
    cv2.putText(
        legend,
        'heatmaps: TURBO blue=low, red=high; ranges are written in titles',
        (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (205, 205, 205), 1, cv2.LINE_AA)
    return legend


def _evidence_sheet(panels: list) -> np.ndarray:
    rows = [
        np.concatenate(panels[start:start + 4], axis=1)
        for start in range(0, 12, 4)]
    separator = np.full((8, rows[0].shape[1], 3), 28, dtype=np.uint8)
    body = rows[0]
    for row in rows[1:]:
        body = np.concatenate([body, separator, row], axis=0)
    return np.concatenate([body, _evidence_legend(body.shape[1])], axis=0)


def _section_images(point_voxels: np.ndarray,
                    component: np.ndarray,
                    world_state: np.ndarray,
                    grid: dict,
                    bounds: tuple) -> tuple:
    component_rows = component.sum(axis=1)
    component_columns = component.sum(axis=0)
    row = int(np.argmax(component_rows))
    column = int(np.argmax(component_columns))
    y_index = int(world_state.shape[1] - 1 - row)
    x_band = np.abs(point_voxels[:, 1] - y_index) <= 1
    y_band = np.abs(point_voxels[:, 0] - column) <= 1
    xz_count = np.zeros(
        (world_state.shape[0], world_state.shape[2]), dtype=np.uint32)
    yz_count = np.zeros(
        (world_state.shape[0], world_state.shape[1]), dtype=np.uint32)
    np.add.at(xz_count, (
        point_voxels[x_band, 2], point_voxels[x_band, 0]), 1)
    np.add.at(yz_count, (
        point_voxels[y_band, 2],
        world_state.shape[1] - 1 - point_voxels[y_band, 1]), 1)
    xz_heat = cv2.applyColorMap(
        np.clip(np.log1p(xz_count) / np.log(20.0) * 255, 0, 255
                ).astype(np.uint8), cv2.COLORMAP_TURBO)
    yz_heat = cv2.applyColorMap(
        np.clip(np.log1p(yz_count) / np.log(20.0) * 255, 0, 255
                ).astype(np.uint8), cv2.COLORMAP_TURBO)
    xz_heat[xz_count == 0] = (24, 26, 30)
    yz_heat[yz_count == 0] = (24, 26, 30)
    xz_state = STATE_PALETTE_BGR[world_state[:, row, :]]
    yz_state = STATE_PALETTE_BGR[world_state[:, :, column]]
    row_min, row_max, col_min, col_max = bounds
    xz_heat = xz_heat[:, col_min:col_max]
    xz_state = xz_state[:, col_min:col_max]
    yz_heat = yz_heat[:, row_min:row_max]
    yz_state = yz_state[:, row_min:row_max]
    scale_y = 20
    scale_x = 5
    images = tuple(cv2.resize(
        image, None, fx=scale_x, fy=scale_y,
        interpolation=cv2.INTER_NEAREST)
        for image in (xz_heat, xz_state, yz_heat, yz_state))
    metadata = {
        'xz_section_row': row,
        'xz_section_y_m': float(grid['row_y_centers'][row]),
        'yz_section_column': column,
        'yz_section_x_m': float(grid['x_centers'][column]),
        'section_point_band_cells': 1,
    }
    return images, metadata


def _section_sheet(images: tuple, metadata: dict) -> np.ndarray:
    titles = (
        f'X-Z point density | y={metadata["xz_section_y_m"]:.1f}m +/- 1 cell',
        f'X-Z world GT | y={metadata["xz_section_y_m"]:.1f}m',
        f'Y-Z point density | x={metadata["yz_section_x_m"]:.1f}m +/- 1 cell',
        f'Y-Z world GT | x={metadata["yz_section_x_m"]:.1f}m')
    panels = []
    for index, (image, title) in enumerate(zip(images, titles)):
        panel = _panel(image, title, (720, 300))
        origin = (40, panel.shape[0] - 28)
        horizontal_end = (98, panel.shape[0] - 28)
        vertical_end = (40, panel.shape[0] - 88)
        cv2.arrowedLine(
            panel, origin, horizontal_end, (45, 55, 225), 3,
            cv2.LINE_AA, tipLength=0.22)
        cv2.arrowedLine(
            panel, origin, vertical_end, (235, 210, 70), 3,
            cv2.LINE_AA, tipLength=0.22)
        horizontal_label = '+x' if index < 2 else '+y'
        cv2.putText(
            panel, horizontal_label,
            (102, panel.shape[0] - 23), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (65, 75, 245), 1, cv2.LINE_AA)
        cv2.putText(
            panel, '+z', (45, panel.shape[0] - 91),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (250, 225, 90), 1, cv2.LINE_AA)
        panels.append(panel)
    top = np.concatenate(panels[:2], axis=1)
    bottom = np.concatenate(panels[2:], axis=1)
    separator = np.full((8, top.shape[1], 3), 28, dtype=np.uint8)
    return np.concatenate([top, separator, bottom], axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=int, default=25267)
    parser.add_argument('--source-horizon', type=int, default=0)
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_full_train_v1'))
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b24_internal_epochs1_3_fixed_v1/'
            'internal_dev/epoch_003'))
    parser.add_argument(
        '--ann-file', type=Path,
        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument('--margin-cells', type=int, default=6)
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_low_static_spatial_expansion_case025267_v1'))
    return parser.parse_args()


def main():
    args = parse_args()
    sequences = _sequence_paths(args.sequence_root)
    predictions = _prediction_paths(args.prediction_root)
    if args.reference not in sequences or args.reference not in predictions:
        raise FileNotFoundError('Reference label or prediction is missing')
    with np.load(sequences[args.reference], allow_pickle=False) as archive:
        world = np.asarray(
            archive['world_target_state_3d'], dtype=np.uint8)
        direct = np.asarray(
            archive['direct_observation_state_3d'], dtype=np.uint8)
        completion = np.asarray(
            archive['completion_source_3d'], dtype=np.uint8)
        target_indices = np.asarray(
            archive['target_indices'], dtype=np.int64)
        transforms = np.asarray(
            archive['target_to_reference'], dtype=np.float64)
        pc_range = np.asarray(archive['pc_range'], dtype=np.float32)
        occ_size = np.asarray(archive['occ_size'], dtype=np.int64)
        collision_z = np.asarray(
            archive['collision_z'], dtype=np.float32)
    with np.load(predictions[args.reference], allow_pickle=False) as archive:
        raw = np.asarray(
            archive['raw_world_pred_class_3d'], dtype=np.uint8) + 1
        final = np.asarray(
            archive['world_pred_class_3d'], dtype=np.uint8) + 1
    grid = _grid_metadata(pc_range, occ_size, collision_z)
    low_index = grid['low_static_z_index']
    source_horizon = args.source_horizon
    if not np.allclose(transforms[source_horizon], np.eye(4), atol=1e-5):
        raise ValueError(
            'This evidence audit currently requires the reference horizon')
    component = _front_left_component(
        world[source_horizon, low_index], grid['x_centers'],
        grid['row_y_centers'])
    bounds = _crop_bounds(component, args.margin_cells)

    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    builder = MultiLidarOccLabelBuilder(
        pc_range, (int(occ_size[1]), int(occ_size[0])), occ_size,
        target_frame=str(metainfo.get('lidar_coord_frame', 'FLU')),
        collision_z=collision_z)
    target_index = int(target_indices[source_horizon])
    evidence = builder.build(
        infos[target_index], diagnostics=True, return_points=True)
    points = np.asarray(evidence['points'], dtype=np.float32)
    transform = transforms[source_horizon]
    homogeneous = np.concatenate([
        points[:, :3], np.ones((len(points), 1), dtype=np.float32)
    ], axis=1)
    points = points.copy()
    points[:, :3] = (homogeneous @ transform.T)[:, :3]
    point_voxels = builder.builder.coord_to_index_floor(points[:, :3])
    point_valid = np.all(
        (point_voxels >= 0) & (point_voxels < occ_size[None]), axis=1)
    ground_estimate, _ = builder.builder.estimate_ground_height(
        points[point_valid, :3], point_voxels[point_valid])
    point_stats = _cell_point_statistics(
        points, builder.builder, ground_estimate)

    ground_hw = _xy_to_hw(ground_estimate)
    low_p90_hw = _xy_to_hw(point_stats['low_p90_xy'])
    hit_count = np.asarray(
        evidence['hit_point_count_3d'][low_index], dtype=np.float32)
    sensor_count = np.asarray(
        evidence['occupied_sensor_count_3d'][low_index], dtype=np.float32)
    raw_obstacle = np.asarray(
        evidence['raw_obstacle_3d'][low_index], dtype=bool)
    filtered_static = np.asarray(
        evidence['static_obstacle_3d'][low_index], dtype=bool)
    direct_static = direct[source_horizon, low_index] == 2
    completion_static = completion[source_horizon, low_index] == 3
    world_low = world[source_horizon, low_index]
    raw_low = raw[source_horizon, low_index]
    final_low = final[source_horizon, low_index]
    known = world_low != 0
    raw_false_static = known & (world_low == 1) & (raw_low == 2)
    final_false_static = known & (world_low == 1) & (final_low == 2)
    final_added_static = (final_low == 2) & (raw_low != 2)
    final_removed_static = (raw_low == 2) & (final_low != 2)
    expansion = np.full(world_low.shape, 0, dtype=np.uint8)
    expansion[world_low == 2] = 2
    expansion[raw_false_static] = 3
    expansion[final_added_static] = 1

    relative_cloud = _relative_pointcloud_image(
        point_stats['points'], point_stats['relative_height'],
        pc_range, occ_size, bounds, component)
    panels = [
        _panel(relative_cloud,
               'points: height above local ground | -0.2 to 2.0m'),
        _panel(_heatmap_image(
            ground_hw, np.isfinite(ground_hw), bounds,
            -1.6, 0.4, component),
            'local ground estimate | -1.6 to 0.4m'),
        _panel(_heatmap_image(
            low_p90_hw, np.isfinite(low_p90_hw), bounds,
            0.0, 2.0, component),
            'z=0 point P90 above ground | 0 to 2m'),
        _panel(_heatmap_image(
            np.log1p(hit_count), hit_count > 0, bounds,
            0.0, np.log(200.0), component),
            'z=0 endpoint point count | log(1+count), max 200'),
        _panel(_heatmap_image(
            sensor_count, sensor_count > 0, bounds,
            0.0, 8.0, component),
            'z=0 occupied sensor support | 0 to 8'),
        _panel(_binary_image(
            raw_obstacle, component, bounds),
            'raw obstacle endpoints | before component filtering'),
        _panel(_binary_image(
            filtered_static, component, bounds),
            'filtered static endpoints | direct classifier output'),
        _panel(_semantic_image(
            direct[source_horizon, low_index], component, bounds,
            completion_static),
            'direct state | yellow overlay: completion-only static'),
        _panel(_semantic_image(world_low, component, bounds),
               'world GT z=0'),
        _panel(_semantic_image(raw_low, component, bounds,
                               raw_false_static),
               'B17A Raw | yellow: known-free predicted static'),
        _panel(_semantic_image(final_low, component, bounds,
                               final_false_static),
               'B24 Final | yellow: known-free predicted static'),
        _panel(_semantic_image(expansion, component, bounds),
               'expansion: red GT static, blue model false-static'),
    ]
    evidence_sheet = _evidence_sheet(panels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.out_dir / 'case025267_spatial_evidence.png'
    if not cv2.imwrite(str(evidence_path), evidence_sheet):
        raise OSError(f'Failed to write {evidence_path}')

    section_images, section_metadata = _section_images(
        point_stats['point_voxels'], component,
        world[source_horizon], grid, bounds)
    section_path = args.out_dir / 'case025267_vertical_sections.png'
    if not cv2.imwrite(
            str(section_path),
            _section_sheet(section_images, section_metadata)):
        raise OSError(f'Failed to write {section_path}')

    rows, columns = np.where(component)
    component_raw_obstacle = component & raw_obstacle
    component_filtered = component & filtered_static
    component_direct = component & direct_static
    component_completion = component & completion_static
    component_hit = component & (hit_count > 0)
    component_ground = component & np.isfinite(ground_hw)
    component_center_relative = -ground_hw
    p90_valid = component & np.isfinite(low_p90_hw)
    vertical_span_hw = _xy_to_hw(point_stats['vertical_span_xy'])
    vertical_span_valid = component & np.isfinite(vertical_span_hw)
    higher_static_support = np.any(
        np.asarray(evidence['static_obstacle_3d'][low_index + 1:],
                   dtype=bool), axis=0)
    low_point_height = p90_valid & (low_p90_hw <= 0.55)
    weak_low_static = (
        component & low_point_height & ~higher_static_support)
    crop_mask = np.zeros_like(component)
    row_min, row_max, col_min, col_max = bounds
    crop_mask[row_min:row_max, col_min:col_max] = True
    direct_distance = _distance_from_mask(direct_static)
    completion_distance = direct_distance[component_completion]
    raw_false_distance = direct_distance[crop_mask & raw_false_static]
    final_false_distance = direct_distance[crop_mask & final_false_static]
    voxel_xy = float(grid['voxel_size'][0])
    crop_known = crop_mask & known
    summary = {
        'schema_version': 1,
        'reference_index': args.reference,
        'source_horizon': source_horizon,
        'target_frame_index': target_index,
        'analysis_type': 'evidence_only_no_label_change',
        'grid': {
            key: value for key, value in grid.items()
            if key not in ('x_centers', 'row_y_centers', 'quadrant_masks')
        },
        'component': {
            'area_cells': int(np.count_nonzero(component)),
            'area_m2': float(np.count_nonzero(component) * voxel_xy ** 2),
            'bbox_x_m': [
                float(grid['x_centers'][columns].min()),
                float(grid['x_centers'][columns].max())],
            'bbox_y_m': [
                float(grid['row_y_centers'][rows].min()),
                float(grid['row_y_centers'][rows].max())],
            'raw_obstacle_cells': int(np.count_nonzero(
                component_raw_obstacle)),
            'filtered_static_cells': int(np.count_nonzero(
                component_filtered)),
            'direct_static_cells': int(np.count_nonzero(component_direct)),
            'completion_only_static_cells': int(np.count_nonzero(
                component_completion)),
            'endpoint_hit_supported_cells': int(np.count_nonzero(
                component_hit)),
            'finite_ground_estimate_cells': int(np.count_nonzero(
                component_ground)),
            'z0_endpoint_point_count': int(hit_count[component].sum()),
            'occupied_sensor_support_distribution': _distribution(
                sensor_count[component_hit]),
            'ground_estimate_m_distribution': _finite_distribution(
                ground_hw, component_ground),
            'voxel_center_height_above_ground_m_distribution': (
                _finite_distribution(
                    component_center_relative, component_ground)),
            'z0_point_p90_height_above_ground_m_distribution': (
                _finite_distribution(low_p90_hw, p90_valid)),
            'point_column_vertical_span_m_distribution': (
                _finite_distribution(
                    vertical_span_hw, vertical_span_valid)),
            'cells_point_p90_at_or_below_0_55m': int(np.count_nonzero(
                p90_valid & (low_p90_hw <= 0.55))),
            'cells_point_p90_above_0_55m': int(np.count_nonzero(
                p90_valid & (low_p90_hw > 0.55))),
            'cells_with_filtered_static_support_above_z0': int(
                np.count_nonzero(component & higher_static_support)),
            'cells_without_filtered_static_support_above_z0': int(
                np.count_nonzero(component & ~higher_static_support)),
            'cells_low_point_p90_and_no_higher_static_support': int(
                np.count_nonzero(weak_low_static)),
        },
        'crop': {
            'row_col_bounds_exclusive': list(bounds),
            'known_cells': int(np.count_nonzero(crop_known)),
            'gt_static_cells': int(np.count_nonzero(
                crop_known & (world_low == 2))),
            'gt_free_cells': int(np.count_nonzero(
                crop_known & (world_low == 1))),
            'b17_raw_static_cells_on_known': int(np.count_nonzero(
                crop_known & (raw_low == 2))),
            'b24_final_static_cells_on_known': int(np.count_nonzero(
                crop_known & (final_low == 2))),
            'b17_known_free_to_static_cells': int(np.count_nonzero(
                crop_mask & raw_false_static)),
            'b24_known_free_to_static_cells': int(np.count_nonzero(
                crop_mask & final_false_static)),
            'b17_gt_static_to_free_cells': int(np.count_nonzero(
                crop_known & (world_low == 2) & (raw_low == 1))),
            'b24_gt_static_to_free_cells': int(np.count_nonzero(
                crop_known & (world_low == 2) & (final_low == 1))),
            'raw_final_changed_cells': int(np.count_nonzero(
                crop_mask & (raw_low != final_low))),
        },
        'spatial_distance_from_direct_static_m': {
            'completion_only_static': _distribution(
                completion_distance * voxel_xy),
            'b17_known_free_to_static': _distribution(
                raw_false_distance * voxel_xy),
            'b24_known_free_to_static': _distribution(
                final_false_distance * voxel_xy),
        },
        'section': section_metadata,
        'new_model_inference_performed': False,
        'formal_labels_modified': False,
        'evidence_sheet': str(evidence_path),
        'vertical_sections': str(section_path),
    }
    summary_path = args.out_dir / 'summary.json'
    with summary_path.open('w') as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
