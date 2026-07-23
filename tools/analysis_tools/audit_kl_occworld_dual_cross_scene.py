#!/usr/bin/env python
"""Promote non-drivable candidates repeated at the same global location."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from scipy.spatial import cKDTree

from tools.analysis_tools.audit_kl_occworld_dual_representation import (
    NAVIGABILITY_STATE_NAMES,
    NON_DRIVABLE,
    OCCUPANCY_STATE_NAMES,
    TRAVERSABILITY_STATE_NAMES,
    TRAVERSABILITY_UNKNOWN,
    _compose_navigability,
    _save_state,
    _tile,
)
from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _resolve_path,
)


CONFIDENCE_STATE_NAMES = np.asarray([
    'none', 'single_frame_candidate', 'cross_scene_confirmed'])
CONFIDENCE_PALETTE = np.asarray([
    [40, 40, 40], [235, 190, 55], [235, 115, 55],
], dtype=np.uint8)
TRAVERSABILITY_PALETTE = np.asarray([
    [40, 40, 40], [55, 185, 95], [235, 155, 55],
], dtype=np.uint8)
NAVIGABILITY_PALETTE = np.asarray([
    [40, 40, 40], [55, 195, 100], [230, 75, 65],
], dtype=np.uint8)


def _bev_mask_to_global_xy(mask: np.ndarray,
                           ego2global: np.ndarray,
                           pc_range: Sequence[float]) -> np.ndarray:
    """Convert active image-aligned BEV cells to global XY centers."""
    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pc_range = np.asarray(pc_range, dtype=np.float64)
    voxel_x = (pc_range[3] - pc_range[0]) / mask.shape[1]
    voxel_y = (pc_range[4] - pc_range[1]) / mask.shape[0]
    x = pc_range[0] + (cols.astype(np.float64) + 0.5) * voxel_x
    y = pc_range[4] - (rows.astype(np.float64) + 0.5) * voxel_y
    points = np.stack([x, y, np.zeros_like(x), np.ones_like(x)], axis=1)
    global_points = (
        np.asarray(ego2global, dtype=np.float64) @ points.T).T
    return global_points[:, :2]


def _cross_frame_support(point_sets: List[np.ndarray], radius: float,
                         group_ids: Sequence[str] = None,
                         source_count: int = None
                         ) -> List[np.ndarray]:
    """Count distinct frames containing a nearby candidate for each point."""
    if radius <= 0:
        raise ValueError('radius must be positive')
    if group_ids is None:
        group_ids = [str(index) for index in range(len(point_sets))]
    if len(group_ids) != len(point_sets):
        raise ValueError('group_ids and point_sets must have equal length')
    if source_count is None:
        source_count = len(point_sets)
    if not 0 <= source_count <= len(point_sets):
        raise ValueError('source_count is outside the point-set range')
    trees = [cKDTree(points) if points.shape[0] else None
             for points in point_sets]
    support = [np.ones(points.shape[0], dtype=np.uint8)
               for points in point_sets[:source_count]]
    for source_index, points in enumerate(point_sets[:source_count]):
        if points.shape[0] == 0:
            continue
        for target_index, tree in enumerate(trees):
            if (group_ids[source_index] == group_ids[target_index] or
                    tree is None):
                continue
            distance, _ = tree.query(
                points, k=1, distance_upper_bound=radius)
            support[source_index] += np.isfinite(distance).astype(np.uint8)
    return support


def _index_dual_labels(dual_dir: Path):
    result = {}
    for path in dual_dir.glob('*__dual_audit.npz'):
        with np.load(path, allow_pickle=False) as label:
            result[int(label['frame_index'])] = path
    return result


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def _save_contact_sheet(out_dir: Path, rows: List[dict],
                        image_suffix: str, output_name: str,
                        value_key: str, columns: int = 6):
    tiles = []
    for row in rows:
        path = out_dir / f"{row['stem']}__{image_suffix}.png"
        tiles.append(_tile(
            _read(path),
            f"#{row['frame_index']} {value_key}={row[value_key]}",
            (300, 225)))
    blank = np.full_like(tiles[0], 28)
    grid = []
    for start in range(0, len(tiles), columns):
        chunk = tiles[start:start + columns]
        chunk.extend([blank] * (columns - len(chunk)))
        grid.append(np.concatenate(chunk, axis=1))
    cv2.imwrite(str(out_dir / output_name),
                np.concatenate(grid, axis=0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--dual-dir', required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument(
        '--support-dual-dir',
        help=(
            'Optional frozen evidence pool. Labels in this directory only '
            'contribute cross-scene support and are never written.'))
    parser.add_argument('--support-indices', type=int, nargs='+')
    parser.add_argument('--match-radius', type=float, default=1.2)
    parser.add_argument('--min-support-frames', type=int, default=2)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_support_frames < 2:
        raise ValueError('min-support-frames must be at least 2')
    infos, _ = _load_infos(_resolve_path(args.ann_file))
    dual_map = _index_dual_labels(Path(args.dual_dir))
    indices = sorted(dual_map) if args.indices is None else list(
        dict.fromkeys(args.indices))
    missing = [index for index in indices if index not in dual_map]
    if missing:
        raise ValueError(f'Missing dual labels for indices: {missing}')

    labels = []
    point_sets = []
    group_ids = []
    for index in indices:
        with np.load(dual_map[index], allow_pickle=False) as label:
            candidate = (
                (label['selected_non_drivable_evidence_bev'] > 0) &
                (label['map_drivable_prior_bev'] == 0))
            labels.append({
                'index': index,
                'path': dual_map[index],
                'stem': dual_map[index].stem.replace('__dual_audit', ''),
                'candidate': candidate,
                'traversability': label['traversability_state_bev'],
                'observation': label['observation_state_bev'],
                'occupancy_3d': label['occupancy_state_3d'],
                'occupancy_visibility_3d': label[
                    'occupancy_visibility_3d'],
                'pc_range': label['pc_range'],
            })
        point_sets.append(_bev_mask_to_global_xy(
            candidate, infos[index]['ego2global'], labels[-1]['pc_range']))
        group_ids.append(str(infos[index].get('scene_token', index)))

    target_count = len(point_sets)
    if bool(args.support_dual_dir) != bool(args.support_indices):
        raise ValueError(
            '--support-dual-dir and --support-indices must be used together')
    if args.support_dual_dir:
        support_map = _index_dual_labels(Path(args.support_dual_dir))
        support_indices = list(dict.fromkeys(args.support_indices))
        missing_support = [
            index for index in support_indices if index not in support_map
        ]
        if missing_support:
            raise ValueError(
                f'Missing support dual labels: {missing_support}')
        target_scenes = set(group_ids)
        support_scenes = {
            str(infos[index].get('scene_token', index))
            for index in support_indices
        }
        if target_scenes.intersection(support_scenes):
            raise ValueError(
                'Frozen support pool overlaps target holdout scenes')
        for index in support_indices:
            with np.load(
                    support_map[index], allow_pickle=False) as support_label:
                candidate = (
                    (support_label[
                        'selected_non_drivable_evidence_bev'] > 0) &
                    (support_label['map_drivable_prior_bev'] == 0))
                pc_range = support_label['pc_range']
            point_sets.append(_bev_mask_to_global_xy(
                candidate, infos[index]['ego2global'], pc_range))
            group_ids.append(
                str(infos[index].get('scene_token', index)))

    support_sets = _cross_frame_support(
        point_sets, radius=args.match_radius,
        group_ids=group_ids, source_count=target_count)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, support in zip(labels, support_sets):
        candidate = label['candidate']
        support_bev = np.zeros(candidate.shape, dtype=np.uint8)
        support_bev[candidate] = support
        confirmed = (
            candidate & (support_bev >= args.min_support_frames))
        single = candidate & ~confirmed
        confidence = np.zeros(candidate.shape, dtype=np.uint8)
        confidence[single] = 1
        confidence[confirmed] = 2
        conservative_traversability = label['traversability'].copy()
        conservative_traversability[
            (conservative_traversability == NON_DRIVABLE) & ~confirmed
        ] = TRAVERSABILITY_UNKNOWN
        conservative_traversability_valid = (
            conservative_traversability != TRAVERSABILITY_UNKNOWN)
        conservative_navigability, conservative_navigability_valid = (
            _compose_navigability(
                conservative_traversability, label['observation']))

        stem = label['stem']
        np.savez_compressed(
            out_dir / f'{stem}__cross_scene.npz',
            cross_scene_support_count_bev=support_bev,
            cross_scene_confirmed_non_drivable_bev=confirmed.astype(
                np.uint8),
            cross_scene_confidence_state_bev=confidence,
            cross_scene_confidence_state_names=CONFIDENCE_STATE_NAMES,
            occupancy_state_3d=label['occupancy_3d'],
            occupancy_visibility_3d=label['occupancy_visibility_3d'],
            occupancy_state_names=OCCUPANCY_STATE_NAMES,
            conservative_traversability_state_bev=(
                conservative_traversability),
            conservative_traversability_valid_bev=(
                conservative_traversability_valid.astype(np.uint8)),
            traversability_state_names=TRAVERSABILITY_STATE_NAMES,
            conservative_navigability_state_bev=(
                conservative_navigability),
            conservative_navigability_valid_bev=(
                conservative_navigability_valid),
            navigability_state_names=NAVIGABILITY_STATE_NAMES,
            observation_state_bev=label['observation'],
            frame_index=np.int64(label['index']),
            match_radius_m=np.float32(args.match_radius),
            min_support_frames=np.int16(args.min_support_frames),
        )
        confidence_path = (
            out_dir / f'{stem}__cross_scene_confidence.png')
        conservative_path = (
            out_dir / f'{stem}__conservative_traversability.png')
        navigability_path = (
            out_dir / f'{stem}__conservative_navigability.png')
        _save_state(confidence_path, confidence, CONFIDENCE_PALETTE)
        _save_state(
            conservative_path, conservative_traversability,
            TRAVERSABILITY_PALETTE)
        _save_state(
            navigability_path, conservative_navigability,
            NAVIGABILITY_PALETTE)
        row = {
            'frame_index': label['index'],
            'stem': stem,
            'candidate_cells': int(candidate.sum()),
            'confirmed_cells': int(confirmed.sum()),
            'single_frame_cells': int(single.sum()),
            'confirmed_ratio': float(
                confirmed.sum() / max(candidate.sum(), 1)),
            'max_support_frames': int(support.max()) if support.size else 0,
            'navigable_cells': int(np.count_nonzero(
                conservative_navigability == 1)),
            'blocked_cells': int(np.count_nonzero(
                conservative_navigability == 2)),
        }
        rows.append(row)
        print(
            f'[{label["index"]}] candidate={row["candidate_cells"]} '
            f'confirmed={row["confirmed_cells"]} '
            f'max_support={row["max_support_frames"]}')

    _write_csv(out_dir / 'cross_scene_metrics.csv', rows)
    total_candidate = max(sum(row['candidate_cells'] for row in rows), 1)
    total_confirmed = sum(row['confirmed_cells'] for row in rows)
    summary = {
        'frame_count': len(rows),
        'match_radius_m': args.match_radius,
        'min_support_frames': args.min_support_frames,
        'candidate_cells': total_candidate,
        'confirmed_cells': total_confirmed,
        'confirmed_ratio': total_confirmed / total_candidate,
        'frames_with_confirmed': sum(
            row['confirmed_cells'] > 0 for row in rows),
        'top8_frame_indices': [
            row['frame_index'] for row in sorted(
                rows, key=lambda item: item['confirmed_cells'],
                reverse=True)[:8]
        ],
    }
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _save_contact_sheet(
        out_dir, rows, 'cross_scene_confidence',
        'cross_scene_contact_sheet.png', 'confirmed_cells')
    _save_contact_sheet(
        out_dir, rows, 'conservative_traversability',
        'conservative_traversability_contact_sheet.png',
        'confirmed_cells')
    _save_contact_sheet(
        out_dir, rows, 'conservative_navigability',
        'conservative_navigability_contact_sheet.png',
        'blocked_cells')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
