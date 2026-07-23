#!/usr/bin/env python
"""Cache full-height per-frame KL K observations for sequence generation."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.data_converter.generate_kl_occworld_labels import (
    FREE,
    INSTANCE_OCCUPIED,
    STATIC_OCCUPIED,
    MultiLidarOccLabelBuilder,
    _load_infos,
    _output_stem,
    _resolve_path,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    WORLD_STATE_NAMES,
    _correct_free_visibility_3d,
)


def _expand_indices(infos: List[dict], indices: List[int],
                    reference_indices: List[int],
                    offsets: List[int]) -> List[int]:
    result = set(indices)
    for reference_index in reference_indices:
        if reference_index < 0 or reference_index >= len(infos):
            raise ValueError(f'Invalid reference index {reference_index}')
        scene = infos[reference_index].get('scene_token')
        for offset in offsets:
            index = reference_index + offset
            if index < 0 or index >= len(infos):
                raise ValueError(
                    f'Reference {reference_index} offset {offset} is invalid')
            if infos[index].get('scene_token') != scene:
                raise ValueError(
                    f'Reference {reference_index} offset {offset} '
                    'crosses the scene boundary')
            result.add(index)
    return sorted(result)


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--indices', type=int, nargs='*', default=[])
    parser.add_argument('--reference-indices', type=int, nargs='*',
                        default=[])
    parser.add_argument('--offsets', type=int, nargs='+',
                        default=list(range(9)))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument('--occ-size', type=int, nargs=3,
                        default=[160, 120, 10])
    parser.add_argument('--collision-z', type=float, nargs=2,
                        default=[0.3, 2.5])
    return parser.parse_args()


def generate_observation_cache(args, infos: List[dict], metainfo: dict):
    """Populate a resumable observation cache using already-loaded infos."""
    if not args.indices and not args.reference_indices:
        raise ValueError('Provide --indices or --reference-indices')
    indices = _expand_indices(
        infos, args.indices, args.reference_indices, args.offsets)
    target_frame = str(metainfo.get('lidar_coord_frame', 'FLU'))
    builder = MultiLidarOccLabelBuilder(
        args.pc_range, args.bev_size, args.occ_size,
        target_frame=target_frame, collision_z=args.collision_z)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    start_all = time.perf_counter()
    for position, index in enumerate(indices, start=1):
        stem = _output_stem(infos[index])
        path = out_dir / f'{index:06d}__{stem}__observation.npz'
        start = time.perf_counter()
        if path.exists() and not args.overwrite:
            with np.load(path, allow_pickle=False) as cached:
                state = cached['observation_state_3d']
            elapsed = time.perf_counter() - start
            source = 'cache'
        else:
            current = builder.build(infos[index], diagnostics=True)
            state, valid_count, blocked_count = (
                _correct_free_visibility_3d(current, args.pc_range))
            np.savez_compressed(
                path,
                observation_state_3d=state,
                observation_valid_3d=(state > 0).astype(np.uint8),
                valid_free_sensor_count_3d=valid_count,
                blocked_free_sensor_count_3d=blocked_count,
                observation_state_names=WORLD_STATE_NAMES,
                sensor_names=current['sensor_names'],
                sensor_origins=current['sensor_origins'],
                frame_index=np.int64(index),
                timestamp=np.float64(infos[index]['timestamp']),
                ego2global=np.asarray(
                    infos[index]['ego2global'], dtype=np.float64),
                pc_range=np.asarray(args.pc_range, dtype=np.float32),
                occ_size=np.asarray(args.occ_size, dtype=np.int16),
                collision_z=np.asarray(args.collision_z, dtype=np.float32),
            )
            elapsed = time.perf_counter() - start
            source = 'built'
        row = {
            'frame_index': index,
            'stem': stem,
            'source': source,
            'elapsed_s': elapsed,
            'known_voxels': int(np.count_nonzero(state)),
            'free_voxels': int(np.count_nonzero(state == FREE)),
            'static_voxels': int(np.count_nonzero(
                state == STATIC_OCCUPIED)),
            'instance_voxels': int(np.count_nonzero(
                state == INSTANCE_OCCUPIED)),
            'path': str(path),
        }
        rows.append(row)
        print(
            f'[{position}/{len(indices)}] frame={index} {source} '
            f'time={elapsed:.2f}s known={row["known_voxels"]:,}')

    _write_csv(out_dir / 'observation_cache_metrics.csv', rows)
    summary = {
        'frame_count': len(rows),
        'built_frames': sum(row['source'] == 'built' for row in rows),
        'cached_frames': sum(row['source'] == 'cache' for row in rows),
        'total_elapsed_s': time.perf_counter() - start_all,
        'mean_built_elapsed_s': float(np.mean([
            row['elapsed_s'] for row in rows if row['source'] == 'built'
        ])) if any(row['source'] == 'built' for row in rows) else 0.0,
        'indices': indices,
    }
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    args = parse_args()
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    generate_observation_cache(args, infos, metainfo)


if __name__ == '__main__':
    main()
