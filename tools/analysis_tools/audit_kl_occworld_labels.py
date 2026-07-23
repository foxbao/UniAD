#!/usr/bin/env python
"""Audit generated KL OccWorld labels and build compact review sheets."""

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _load_infos(path: Path) -> Tuple[List[dict], dict]:
    with path.open('rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list'], data.get('metainfo', {})
    if isinstance(data, dict) and 'infos' in data:
        return data['infos'], data.get('metadata', {})
    if isinstance(data, list):
        return data, {}
    raise KeyError(f'Unsupported info container: {path}')


def _class_counts(info: dict, id_to_name: Dict[int, str]) -> Counter:
    counts = Counter()
    for instance in info.get('instances', []):
        if not bool(instance.get('bbox_3d_isvalid', True)):
            continue
        label = int(instance.get(
            'bbox_label_3d', instance.get('bbox_label', -1)))
        counts[id_to_name.get(label, f'label_{label}')] += 1
    return counts


def _quantiles(values: List[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        'min': float(np.min(values)),
        'p25': float(np.percentile(values, 25)),
        'median': float(np.median(values)),
        'p75': float(np.percentile(values, 75)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
    }


def _analyze_label(path: Path, frame_index: int, info: dict,
                   id_to_name: Dict[int, str]) -> dict:
    with np.load(path, allow_pickle=False) as label:
        raw_target = label['occupancy_target']
        filtered_target = label['filtered_occupancy_target']
        filtered_valid = label['filtered_visibility_mask']
        rejected = label['unreliable_endpoint_mask']
        instance_endpoint = label['instance_endpoint_3d']
        endpoint_type = label['endpoint_type_3d']
        uncertain = endpoint_type == 2
        e_state = label['state_collision_band']
        f_state = label['state_observed_3d_projection']
        g_state = label['state_filtered_3d_projection']
        raw_occupied = int(raw_target.sum())
        filtered_occupied = int(filtered_target.sum())
        classes = _class_counts(info, id_to_name)
        return {
            'frame_index': frame_index,
            'token': str(label['token'].item()),
            'scene_token': str(label['scene_token'].item()),
            'stem': path.stem,
            'instance_count': int(sum(classes.values())),
            'classes': ';'.join(
                f'{name}:{count}' for name, count in sorted(classes.items())),
            'raw_visible_3d': int(label['visibility_mask'].sum()),
            'filtered_visible_3d': int(filtered_valid.sum()),
            'raw_occupied_3d': raw_occupied,
            'filtered_occupied_3d': filtered_occupied,
            'occupied_keep_ratio': (
                filtered_occupied / max(raw_occupied, 1)),
            'unreliable_endpoint_3d': int(rejected.sum()),
            'rejected_still_visible': int(np.count_nonzero(
                (rejected > 0) & (filtered_valid > 0))),
            'instance_endpoint_not_protected': int(np.count_nonzero(
                (instance_endpoint > 0) & (filtered_target == 0))),
            'ground_endpoint_3d': int(np.count_nonzero(endpoint_type == 1)),
            'uncertain_obstacle_3d': int(np.count_nonzero(uncertain)),
            'reliable_static_endpoint_3d': int(np.count_nonzero(
                endpoint_type == 3)),
            'instance_endpoint_3d': int(np.count_nonzero(
                endpoint_type == 4)),
            'uncertain_multi_sensor_3d': int(np.count_nonzero(
                uncertain & (label['occupied_sensor_count_3d'] >= 2))),
            'uncertain_multi_point_3d': int(np.count_nonzero(
                uncertain & (label['hit_point_count_3d'] >= 2))),
            'uncertain_five_point_3d': int(np.count_nonzero(
                uncertain & (label['hit_point_count_3d'] >= 5))),
            'e_g_changed_cells': int(np.count_nonzero(e_state != g_state)),
            'f_g_changed_cells': int(np.count_nonzero(f_state != g_state)),
            'e_occupied_cells': int(np.count_nonzero(e_state == 2)),
            'f_occupied_cells': int(np.count_nonzero(f_state == 2)),
            'g_occupied_cells': int(np.count_nonzero(g_state == 2)),
        }


def _read_preview(label_dir: Path, stem: str, suffix: str,
                  image_size: Tuple[int, int]) -> np.ndarray:
    path = label_dir / f'{stem}{suffix}'
    image = cv2.imread(str(path))
    if image is None:
        image = np.full(
            (image_size[1], image_size[0], 3), 40, dtype=np.uint8)
        cv2.putText(image, 'missing', (20, image_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return image
    return cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)


def _tile(label_dir: Path, row: dict, suffix: str, title: str,
          image_size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    image = _read_preview(label_dir, row['stem'], suffix, image_size)
    return _tile_image(image, row, title, image_size)


def _tile_image(image: np.ndarray, row: dict, title: str,
                image_size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)
    header = np.full((28, image_size[0], 3), 28, dtype=np.uint8)
    text = f"#{row['frame_index']} {title} dEG={row['e_g_changed_cells']}"
    cv2.putText(header, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def _collision_projection(label, key: str) -> np.ndarray:
    z_centers = label['z_centers']
    collision_z = label['collision_z']
    keep = ((z_centers >= collision_z[0]) &
            (z_centers <= collision_z[1]))
    return np.any(label[key][keep] > 0, axis=0)


def _draw_boxes(image: np.ndarray, info: dict, pc_range: np.ndarray,
                color=(255, 255, 0), thickness: int = 1) -> np.ndarray:
    image = image.copy()
    height, width = image.shape[:2]
    x_min, y_min, _, x_max, y_max, _ = pc_range.astype(np.float32)
    base = np.asarray([
        [-0.5, -0.5], [0.5, -0.5],
        [0.5, 0.5], [-0.5, 0.5],
    ], dtype=np.float32)
    for instance in info.get('instances', []):
        if not bool(instance.get('bbox_3d_isvalid', True)):
            continue
        box = np.asarray(instance.get('bbox_3d', []), dtype=np.float32)
        if box.size < 7 or np.any(box[3:5] <= 0):
            continue
        yaw = float(box[6])
        rotation = np.asarray([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ], dtype=np.float32)
        corners = base * box[None, 3:5]
        corners = corners @ rotation.T + box[None, :2]
        cols = (corners[:, 0] - x_min) / (x_max - x_min) * width
        rows = (y_max - corners[:, 1]) / (y_max - y_min) * height
        pixels = np.round(np.stack([cols, rows], axis=1)).astype(np.int32)
        cv2.polylines(image, [pixels], True, color, thickness,
                      cv2.LINE_AA)
    return image


def _point_evidence_panels(label_path: Path, info: dict):
    with np.load(label_path, allow_pickle=False) as label:
        z_centers = label['z_centers']
        collision_z = label['collision_z']
        keep = ((z_centers >= collision_z[0]) &
                (z_centers <= collision_z[1]))
        hit_count = label['hit_point_count_3d'][keep].sum(axis=0)
        log_count = np.log1p(hit_count.astype(np.float32))
        scale = max(float(np.percentile(log_count, 99)), 1e-6)
        density = np.clip(log_count / scale * 255, 0, 255).astype(np.uint8)
        density = cv2.applyColorMap(density, cv2.COLORMAP_BONE)
        density[hit_count == 0] = (28, 28, 28)

        ground = _collision_projection(label, 'observed_ground_3d')
        raw_obstacle = _collision_projection(label, 'raw_obstacle_3d')
        reliable_static = _collision_projection(
            label, 'static_obstacle_3d')
        instance_endpoint = _collision_projection(
            label, 'instance_endpoint_3d')
        rejected = _collision_projection(
            label, 'unreliable_endpoint_mask')
        rejected_obstacle = raw_obstacle & ~reliable_static & rejected

        evidence = np.full((*ground.shape, 3), 28, dtype=np.uint8)
        # BGR colors: orange ground, magenta rejected obstacle, red reliable
        # static, blue instance endpoint.
        evidence[ground & rejected] = (0, 165, 255)
        evidence[rejected_obstacle] = (255, 0, 255)
        evidence[reliable_static] = (60, 70, 225)
        evidence[instance_endpoint] = (240, 120, 65)

        pc_range = label['pc_range']
        density = _draw_boxes(density, info, pc_range)
        evidence = _draw_boxes(evidence, info, pc_range)
        metrics = {
            'collision_hit_cells': int(np.count_nonzero(hit_count)),
            'collision_ground_cells': int(np.count_nonzero(ground & rejected)),
            'collision_rejected_obstacle_cells': int(np.count_nonzero(
                rejected_obstacle)),
            'collision_reliable_static_cells': int(np.count_nonzero(
                reliable_static)),
            'collision_instance_endpoint_cells': int(np.count_nonzero(
                instance_endpoint)),
        }
    return density, evidence, metrics


def _save_g_contact_sheet(label_dir: Path, rows: List[dict], path: Path,
                          columns: int = 6):
    tiles = [_tile(
        label_dir, row, '__G_filtered_3d_projection.png', 'G')
             for row in rows]
    blank = np.full_like(tiles[0], 28)
    grid_rows = []
    for start in range(0, len(tiles), columns):
        chunk = tiles[start:start + columns]
        chunk.extend([blank] * (columns - len(chunk)))
        grid_rows.append(np.concatenate(chunk, axis=1))
    cv2.imwrite(str(path), np.concatenate(grid_rows, axis=0))


def _save_worst_triptych(label_dir: Path, rows: List[dict], path: Path,
                         count: int = 6):
    worst = sorted(
        rows, key=lambda row: row['e_g_changed_cells'], reverse=True)[:count]
    grid_rows = []
    suffixes = (
        ('__E_collision_band.png', 'E'),
        ('__F_observed_3d_projection.png', 'F'),
        ('__G_filtered_3d_projection.png', 'G'),
    )
    for row in worst:
        grid_rows.append(np.concatenate([
            _tile(label_dir, row, suffix, title)
            for suffix, title in suffixes
        ], axis=1))
    cv2.imwrite(str(path), np.concatenate(grid_rows, axis=0))


def _save_worst_point_evidence(label_dir: Path, rows: List[dict], path: Path,
                               info_by_index: Dict[int, dict], count: int = 6):
    worst = sorted(
        rows, key=lambda row: row['e_g_changed_cells'], reverse=True)[:count]
    grid_rows = []
    metric_rows = []
    for row in worst:
        label_path = label_dir / f"{row['stem']}.npz"
        info = info_by_index[row['frame_index']]
        density, evidence, metrics = _point_evidence_panels(label_path, info)
        g_image = cv2.imread(str(
            label_dir / f"{row['stem']}__G_filtered_3d_projection.png"))
        if g_image is None:
            raise FileNotFoundError(f'Missing G preview for {row["stem"]}')
        with np.load(label_path, allow_pickle=False) as label:
            g_image = _draw_boxes(g_image, info, label['pc_range'],
                                  thickness=2)
        grid_rows.append(np.concatenate([
            _tile_image(density, row, 'raw hits + boxes'),
            _tile_image(evidence, row, 'endpoint classes'),
            _tile_image(g_image, row, 'G + boxes'),
        ], axis=1))
        metric_rows.append({'frame_index': row['frame_index'], **metrics})

    sheet = np.concatenate(grid_rows, axis=0)
    legend = np.full((44, sheet.shape[1], 3), 28, dtype=np.uint8)
    entries = [
        ('ground', (0, 165, 255)),
        ('rejected obstacle', (255, 0, 255)),
        ('reliable static', (60, 70, 225)),
        ('instance endpoint', (240, 120, 65)),
        ('3D box', (255, 255, 0)),
    ]
    x = 12
    for text, color in entries:
        cv2.rectangle(legend, (x, 13), (x + 18, 31), color, -1)
        cv2.putText(legend, text, (x + 24, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
        x += 24 + max(105, len(text) * 8)
    cv2.imwrite(str(path), np.concatenate([legend, sheet], axis=0))
    _write_csv(path.with_suffix('.csv'), metric_rows)


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument(
        '--label-dir', required=True,
        help='Directory containing generated NPZ and E/F/G previews.')
    parser.add_argument(
        '--out-dir', help='Defaults to <label-dir>/audit.')
    return parser.parse_args()


def main():
    args = parse_args()
    ann_path = Path(args.ann_file)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir) if args.out_dir else label_dir / 'audit'
    out_dir.mkdir(parents=True, exist_ok=True)

    infos, metainfo = _load_infos(ann_path)
    token_to_info = {
        str(info.get('token', '')): (index, info)
        for index, info in enumerate(infos)
    }
    info_by_index = {index: info for index, info in enumerate(infos)}
    categories = metainfo.get('categories', {})
    id_to_name = {int(label): str(name)
                  for name, label in categories.items()}

    rows = []
    category_frames = Counter()
    for path in sorted(label_dir.glob('*.npz')):
        with np.load(path, allow_pickle=False) as label:
            token = str(label['token'].item())
        if token not in token_to_info:
            raise KeyError(f'Label token not found in annotations: {token}')
        frame_index, info = token_to_info[token]
        row = _analyze_label(path, frame_index, info, id_to_name)
        rows.append(row)
        category_frames.update(_class_counts(info, id_to_name).keys())
    if not rows:
        raise RuntimeError(f'No NPZ labels found in {label_dir}')
    rows.sort(key=lambda row: row['frame_index'])

    endpoint_totals = {
        'ground': int(sum(row['ground_endpoint_3d'] for row in rows)),
        'uncertain_obstacle': int(sum(
            row['uncertain_obstacle_3d'] for row in rows)),
        'reliable_static': int(sum(
            row['reliable_static_endpoint_3d'] for row in rows)),
        'instance': int(sum(row['instance_endpoint_3d'] for row in rows)),
    }
    endpoint_total = max(sum(endpoint_totals.values()), 1)
    uncertain_total = max(endpoint_totals['uncertain_obstacle'], 1)
    uncertain_support = {
        'multi_sensor_count': int(sum(
            row['uncertain_multi_sensor_3d'] for row in rows)),
        'two_or_more_points_count': int(sum(
            row['uncertain_multi_point_3d'] for row in rows)),
        'five_or_more_points_count': int(sum(
            row['uncertain_five_point_3d'] for row in rows)),
    }
    uncertain_support['multi_sensor_ratio'] = (
        uncertain_support['multi_sensor_count'] / uncertain_total)
    uncertain_support['two_or_more_points_ratio'] = (
        uncertain_support['two_or_more_points_count'] / uncertain_total)
    uncertain_support['five_or_more_points_ratio'] = (
        uncertain_support['five_or_more_points_count'] / uncertain_total)

    summary = {
        'frame_count': len(rows),
        'unique_scene_count': len({row['scene_token'] for row in rows}),
        'category_frame_coverage': dict(sorted(category_frames.items())),
        'occupied_keep_ratio': _quantiles([
            row['occupied_keep_ratio'] for row in rows]),
        'e_g_changed_cells': _quantiles([
            row['e_g_changed_cells'] for row in rows]),
        'f_g_changed_cells': _quantiles([
            row['f_g_changed_cells'] for row in rows]),
        'raw_occupied_3d': _quantiles([
            row['raw_occupied_3d'] for row in rows]),
        'filtered_occupied_3d': _quantiles([
            row['filtered_occupied_3d'] for row in rows]),
        'total_rejected_still_visible': int(sum(
            row['rejected_still_visible'] for row in rows)),
        'total_instance_endpoint_not_protected': int(sum(
            row['instance_endpoint_not_protected'] for row in rows)),
        'endpoint_type_totals': endpoint_totals,
        'endpoint_type_ratios': {
            key: value / endpoint_total
            for key, value in endpoint_totals.items()
        },
        'uncertain_obstacle_support': uncertain_support,
        'worst_e_g_frames': [
            {'frame_index': row['frame_index'],
             'scene_token': row['scene_token'],
             'changed_cells': row['e_g_changed_cells']}
            for row in sorted(
                rows, key=lambda item: item['e_g_changed_cells'],
                reverse=True)[:6]
        ],
    }

    _write_csv(out_dir / 'frame_metrics.csv', rows)
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _save_g_contact_sheet(
        label_dir, rows, out_dir / 'g_contact_sheet.png')
    _save_worst_triptych(
        label_dir, rows, out_dir / 'worst6_e_f_g.png')
    _save_worst_point_evidence(
        label_dir, rows, out_dir / 'worst6_point_evidence.png',
        info_by_index)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'csv={out_dir / "frame_metrics.csv"}')
    print(f'contact={out_dir / "g_contact_sheet.png"}')
    print(f'worst={out_dir / "worst6_e_f_g.png"}')
    print(f'evidence={out_dir / "worst6_point_evidence.png"}')


if __name__ == '__main__':
    main()
