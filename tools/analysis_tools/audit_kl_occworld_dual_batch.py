#!/usr/bin/env python
"""Batch-audit KL 3D occupancy and 2D traversability candidates."""

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

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    build_map_mask,
    load_clean_drivable_geometry,
)
from tools.analysis_tools.audit_kl_occworld_dual_representation import (
    DRIVABLE,
    NAVIGABILITY_STATE_NAMES,
    NON_DRIVABLE,
    OCCUPANCY_STATE_NAMES,
    TRAVERSABILITY_STATE_NAMES,
    _compose_navigability,
    _compose_occupancy_3d,
    _compose_traversability,
    _filter_small_components,
    _opposing_obstacle_surface,
    _save_state,
    _tile,
)
from tools.data_converter.generate_kl_occworld_labels import (
    UNKNOWN,
    _load_infos,
    _resolve_path,
)


OCCUPANCY_PALETTE = np.asarray([
    [40, 40, 40], [70, 180, 90],
    [225, 80, 70], [65, 120, 240],
], dtype=np.uint8)
TRAVERSABILITY_PALETTE = np.asarray([
    [40, 40, 40], [55, 185, 95], [235, 155, 55],
], dtype=np.uint8)
NAVIGABILITY_PALETTE = np.asarray([
    [40, 40, 40], [55, 195, 100], [230, 75, 65],
], dtype=np.uint8)


def _temporal_index_map(temporal_dir: Path) -> Dict[int, Path]:
    mapping = {}
    for path in temporal_dir.glob('*__temporal.npz'):
        with np.load(path, allow_pickle=False) as label:
            current = np.flatnonzero(label['temporal_offsets'] == 0)
            if current.size != 1:
                raise ValueError(f'Expected one offset=0 in {path.name}')
            index = int(label['temporal_source_indices'][current[0]])
        mapping[index] = path
    return mapping


def _occlusion_stem_map(occlusion_dir: Path) -> Dict[str, Path]:
    mapping = {}
    suffix = '__occlusion'
    for path in occlusion_dir.glob('*__occlusion.npz'):
        stem = path.stem
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
        mapping[stem] = path
    return mapping


def _stem_from_temporal(path: Path) -> str:
    suffix = '__temporal'
    stem = path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def _component_sizes(mask: np.ndarray) -> List[int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return []
    return sorted(
        stats[1:, cv2.CC_STAT_AREA].astype(int).tolist(), reverse=True)


def _read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def _save_frame_preview(path: Path, k_path: Path, raw_path: Path,
                        selected_path: Path, nav_path: Path,
                        selected_min: int):
    panels = [
        _tile(_read(k_path), 'K observation', (320, 240)),
        _tile(_read(raw_path), 'raw opposing-boundary evidence', (320, 240)),
        _tile(_read(selected_path),
              f'traversability | component >= {selected_min}', (320, 240)),
        _tile(_read(nav_path), 'current navigability', (320, 240)),
    ]
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def _save_contact_sheet(out_dir: Path, rows: List[dict], columns: int = 6):
    tiles = []
    for row in rows:
        image = _read(out_dir / f"{row['stem']}__traversability.png")
        tiles.append(_tile(
            image,
            f"#{row['frame_index']} non={row['selected_non_drivable_cells']}",
            (300, 225)))
    blank = np.full_like(tiles[0], 28)
    grid = []
    for start in range(0, len(tiles), columns):
        chunk = tiles[start:start + columns]
        chunk.extend([blank] * (columns - len(chunk)))
        grid.append(np.concatenate(chunk, axis=1))
    cv2.imwrite(str(out_dir / 'dual24_contact_sheet.png'),
                np.concatenate(grid, axis=0))


def _save_largest_contact(out_dir: Path, rows: List[dict], count: int = 8):
    selected = sorted(
        rows, key=lambda row: row['selected_non_drivable_cells'],
        reverse=True)[:count]
    grid = []
    for row in selected:
        stem = row['stem']
        grid.append(np.concatenate([
            _tile(_read(out_dir / f'{stem}__K.png'),
                  f"#{row['frame_index']} K", (320, 240)),
            _tile(_read(out_dir / f'{stem}__non_drivable_raw.png'),
                  f"raw={row['raw_non_drivable_cells']}", (320, 240)),
            _tile(_read(out_dir / f'{stem}__traversability.png'),
                  f"kept={row['selected_non_drivable_cells']}", (320, 240)),
        ], axis=1))
    cv2.imwrite(str(out_dir / 'largest8_raw_filtered.png'),
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
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--temporal-dir', required=True)
    parser.add_argument('--occlusion-dir', required=True)
    parser.add_argument(
        '--clean-map-file',
        default='data/kl_8/map/base_map_drivable_clean.pkl')
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--component-thresholds', type=int, nargs='+',
                        default=[4, 8, 16])
    parser.add_argument('--selected-min-component', type=int, default=8)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = sorted(set(
        [1, args.selected_min_component] + args.component_thresholds))
    if any(value < 1 for value in thresholds):
        raise ValueError('Component thresholds must be positive')
    infos, _ = _load_infos(_resolve_path(args.ann_file))
    temporal_map = _temporal_index_map(Path(args.temporal_dir))
    occlusion_map = _occlusion_stem_map(Path(args.occlusion_dir))
    indices = sorted(temporal_map) if args.indices is None else list(
        dict.fromkeys(args.indices))
    missing = [index for index in indices if index not in temporal_map]
    if missing:
        raise ValueError(f'Missing temporal labels for indices: {missing}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drivable_global = load_clean_drivable_geometry(
        str(_resolve_path(args.clean_map_file)))
    rows = []
    for index in indices:
        temporal_path = temporal_map[index]
        stem = _stem_from_temporal(temporal_path)
        if stem not in occlusion_map:
            raise KeyError(f'Missing occlusion label for {stem}')
        with np.load(temporal_path, allow_pickle=False) as temporal, \
                np.load(occlusion_map[stem], allow_pickle=False) as occlusion:
            required = 'valid_free_sensor_count_3d'
            if required not in occlusion:
                raise KeyError(
                    f'{occlusion_map[stem].name} lacks {required}')
            occupancy_3d, occupancy_visibility = _compose_occupancy_3d(
                temporal['temporal_state_3d'],
                temporal['box_occupied_3d'],
                occlusion['valid_free_sensor_count_3d'])
            ground_surface = (
                (temporal['filled_ground'] > 0) |
                np.any(temporal['ground_endpoint_3d'] > 0, axis=0))
            k_state = occlusion['state_occlusion_corrected']
            pc_range = temporal['pc_range']
            z_centers = temporal['z_centers']

        map_drivable = build_map_mask(
            drivable_global,
            np.asarray(infos[index]['ego2global'], dtype=np.float64),
            pc_range, k_state.shape)
        raw_non_drivable = _opposing_obstacle_surface(
            k_state, ground_surface) & ~(map_drivable > 0)
        filtered = {
            threshold: _filter_small_components(
                raw_non_drivable, threshold)
            for threshold in thresholds
        }
        selected_non_drivable = filtered[args.selected_min_component]
        traversability, traversability_valid = _compose_traversability(
            map_drivable, ground_surface, k_state != UNKNOWN,
            selected_non_drivable)
        navigability, navigability_valid = _compose_navigability(
            traversability, k_state)
        sizes = _component_sizes(raw_non_drivable)

        np.savez_compressed(
            out_dir / f'{stem}__dual_audit.npz',
            occupancy_state_3d=occupancy_3d,
            occupancy_visibility_3d=occupancy_visibility,
            occupancy_state_names=OCCUPANCY_STATE_NAMES,
            traversability_state_bev=traversability,
            traversability_valid_bev=traversability_valid,
            traversability_state_names=TRAVERSABILITY_STATE_NAMES,
            navigability_state_bev=navigability,
            navigability_valid_bev=navigability_valid,
            navigability_state_names=NAVIGABILITY_STATE_NAMES,
            map_drivable_prior_bev=map_drivable.astype(np.uint8),
            lidar_ground_surface_bev=ground_surface.astype(np.uint8),
            raw_non_drivable_evidence_bev=raw_non_drivable.astype(np.uint8),
            selected_non_drivable_evidence_bev=(
                selected_non_drivable.astype(np.uint8)),
            selected_min_component_cells=np.int16(
                args.selected_min_component),
            observation_state_bev=k_state,
            z_centers=z_centers,
            pc_range=pc_range,
            frame_index=np.int64(index),
        )

        raw_state = np.zeros_like(k_state)
        raw_state[raw_non_drivable] = NON_DRIVABLE
        k_path = out_dir / f'{stem}__K.png'
        raw_path = out_dir / f'{stem}__non_drivable_raw.png'
        trav_path = out_dir / f'{stem}__traversability.png'
        nav_path = out_dir / f'{stem}__navigability.png'
        _save_state(k_path, k_state, OCCUPANCY_PALETTE)
        _save_state(raw_path, raw_state, TRAVERSABILITY_PALETTE)
        _save_state(trav_path, traversability, TRAVERSABILITY_PALETTE)
        _save_state(nav_path, navigability, NAVIGABILITY_PALETTE)
        _save_frame_preview(
            out_dir / f'{stem}__dual_compare.png',
            k_path, raw_path, trav_path, nav_path,
            args.selected_min_component)

        row = {
            'frame_index': index,
            'stem': stem,
            'map_drivable_cells': int(map_drivable.sum()),
            'ground_surface_cells': int(ground_surface.sum()),
            'raw_non_drivable_cells': int(raw_non_drivable.sum()),
            'raw_component_count': len(sizes),
            'largest_component_cells': sizes[0] if sizes else 0,
            **{
                f'component_ge_{threshold}_cells': int(mask.sum())
                for threshold, mask in filtered.items()
            },
            'selected_non_drivable_cells': int(
                selected_non_drivable.sum()),
            'selected_retention_ratio': float(
                selected_non_drivable.sum() /
                max(raw_non_drivable.sum(), 1)),
            'drivable_cells': int(np.count_nonzero(
                traversability == DRIVABLE)),
            'non_drivable_cells': int(np.count_nonzero(
                traversability == NON_DRIVABLE)),
            'navigable_cells': int(np.count_nonzero(navigability == 1)),
            'blocked_cells': int(np.count_nonzero(navigability == 2)),
        }
        rows.append(row)
        print(
            f'[{index}] raw={row["raw_non_drivable_cells"]} '
            f'cc{args.selected_min_component}='
            f'{row["selected_non_drivable_cells"]} '
            f'largest={row["largest_component_cells"]} '
            f'map={row["map_drivable_cells"]}')

    _write_csv(out_dir / 'dual_metrics.csv', rows)
    summary = {
        'frame_count': len(rows),
        'selected_min_component_cells': args.selected_min_component,
        'raw_non_drivable_cells': _quantiles(
            rows, 'raw_non_drivable_cells'),
        'selected_non_drivable_cells': _quantiles(
            rows, 'selected_non_drivable_cells'),
        'selected_retention_ratio': _quantiles(
            rows, 'selected_retention_ratio'),
        'largest_component_cells': _quantiles(
            rows, 'largest_component_cells'),
        'map_drivable_cells': _quantiles(rows, 'map_drivable_cells'),
        'largest8_frame_indices': [
            row['frame_index'] for row in sorted(
                rows,
                key=lambda item: item['selected_non_drivable_cells'],
                reverse=True)[:8]
        ],
    }
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _save_contact_sheet(out_dir, rows)
    _save_largest_contact(out_dir, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
