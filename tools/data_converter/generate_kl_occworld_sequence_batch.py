#!/usr/bin/env python
"""Generate many KL OccWorld sequences in one resumable process."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.data_converter.generate_kl_occworld_labels import (
    MultiLidarOccLabelBuilder,
    _load_infos,
    _output_stem,
    _resolve_path,
)
from tools.data_converter.generate_kl_occworld_observation_cache import (
    generate_observation_cache,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    _observation_cache_index,
    _same_scene_indices,
    _validate_fixed_frame_times,
    generate_sequence,
)


def _reference_indices(eligibility_summary: Path,
                       explicit_indices: List[int]) -> List[int]:
    if explicit_indices:
        return sorted(set(explicit_indices))
    if not eligibility_summary:
        raise ValueError(
            'Provide --reference-indices or --eligibility-summary')
    with eligibility_summary.open() as f:
        summary = json.load(f)
    return sorted(set(int(index) for index in summary['eligible_indices']))


def _dual_label_index(dual_dir: Path) -> Dict[int, Path]:
    required_keys = {
        'frame_index',
        'occupancy_state_3d',
        'conservative_traversability_state_bev',
        'conservative_traversability_valid_bev',
    }
    mapping = {}
    for path in sorted(dual_dir.glob('*__cross_scene.npz')):
        with np.load(path, allow_pickle=False) as label:
            missing_keys = sorted(required_keys.difference(label.files))
            if missing_keys:
                raise ValueError(
                    f'Dual label {path} is missing required keys '
                    f'{missing_keys}')
            index = int(label['frame_index'])
        if index in mapping:
            raise ValueError(f'Duplicate dual label for frame {index}')
        mapping[index] = path
    return mapping


def _completed_label(out_dir: Path, info: dict,
                     reference_index: int) -> Path:
    path = out_dir / f'{_output_stem(info)}__occworld_sequence.npz'
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as label:
        if int(label['reference_index']) != reference_index:
            raise ValueError(f'Reference mismatch in existing label {path}')
        if label['world_target_state_3d'].ndim != 4:
            raise ValueError(f'Invalid existing sequence label {path}')
    return path


def _write_progress(path: Path, rows: List[dict], reference_count: int):
    payload = {
        'reference_count': reference_count,
        'completed_count': sum(
            row['status'] in ('generated', 'existing') for row in rows),
        'failed_count': sum(row['status'] == 'failed' for row in rows),
        'rows': rows,
    }
    with path.open('w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--eligibility-summary', type=Path)
    parser.add_argument('--reference-indices', type=int, nargs='*',
                        default=[])
    parser.add_argument('--dual-dir', type=Path, required=True)
    parser.add_argument('--observation-cache-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--target-offsets', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--reveal-offsets', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--min-free-frames', type=int, default=2)
    parser.add_argument('--min-static-frames', type=int, default=2)
    parser.add_argument('--expected-step-s', type=float, default=0.5)
    parser.add_argument('--max-time-error-s', type=float, default=0.2)
    parser.add_argument('--overwrite-cache', action='store_true')
    parser.add_argument('--overwrite-sequences', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument('--occ-size', type=int, nargs=3,
                        default=[160, 120, 10])
    parser.add_argument('--collision-z', type=float, nargs=2,
                        default=[0.3, 2.5])
    return parser.parse_args()


def main():
    args = parse_args()
    references = _reference_indices(
        args.eligibility_summary, args.reference_indices)
    if not references:
        raise ValueError('No reference indices selected')

    # This is intentionally the only annotation load in the batch process.
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    target_offsets = list(dict.fromkeys(args.target_offsets))
    reveal_offsets = sorted(set(args.reveal_offsets))
    required_offsets = sorted(set(
        target + reveal
        for target in target_offsets
        for reveal in reveal_offsets))
    for index in references:
        _same_scene_indices(infos, index, required_offsets)
        _validate_fixed_frame_times(
            infos, index, target_offsets, reveal_offsets,
            args.expected_step_s, args.max_time_error_s)

    dual_labels = _dual_label_index(args.dual_dir)
    missing_dual = [index for index in references if index not in dual_labels]
    if missing_dual:
        raise FileNotFoundError(
            f'Missing dual labels for reference indices {missing_dual}')

    cache_args = Namespace(
        indices=[],
        reference_indices=references,
        offsets=required_offsets,
        out_dir=str(args.observation_cache_dir),
        overwrite=args.overwrite_cache,
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size,
        collision_z=args.collision_z,
    )
    print(
        f'Preparing observation cache for {len(references)} sequences and '
        f'{len(required_offsets)} offsets')
    generate_observation_cache(cache_args, infos, metainfo)
    observation_cache = _observation_cache_index(
        args.observation_cache_dir)

    target_frame = str(metainfo.get('lidar_coord_frame', 'FLU'))
    builder = MultiLidarOccLabelBuilder(
        args.pc_range, args.bev_size, args.occ_size,
        target_frame=target_frame, collision_z=args.collision_z)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / 'batch_generation_summary.json'
    rows = []
    for position, reference_index in enumerate(references, start=1):
        sequence_dir = args.out_dir / f'{reference_index:06d}'
        try:
            existing = None
            if not args.overwrite_sequences:
                existing = _completed_label(
                    sequence_dir, infos[reference_index], reference_index)
            if existing is not None:
                row = {
                    'reference_index': reference_index,
                    'status': 'existing',
                    'label_path': str(existing),
                    'error': '',
                }
                print(
                    f'[{position}/{len(references)}] reference='
                    f'{reference_index} existing')
            else:
                sequence_args = Namespace(
                    reference_index=reference_index,
                    target_offsets=target_offsets,
                    reveal_offsets=reveal_offsets,
                    min_free_frames=args.min_free_frames,
                    min_static_frames=args.min_static_frames,
                    expected_step_s=args.expected_step_s,
                    max_time_error_s=args.max_time_error_s,
                    reference_dual_label=str(dual_labels[reference_index]),
                    observation_cache_dir=str(
                        args.observation_cache_dir),
                    out_dir=str(sequence_dir),
                    pc_range=args.pc_range,
                    bev_size=args.bev_size,
                    occ_size=args.occ_size,
                    collision_z=args.collision_z,
                )
                print(
                    f'[{position}/{len(references)}] reference='
                    f'{reference_index} generating')
                result = generate_sequence(
                    sequence_args, infos, metainfo, builder=builder,
                    observation_cache=observation_cache)
                row = {
                    'reference_index': reference_index,
                    'status': 'generated',
                    'label_path': str(result['label_path']),
                    'error': '',
                }
        except Exception as error:
            row = {
                'reference_index': reference_index,
                'status': 'failed',
                'label_path': '',
                'error': str(error),
            }
            print(
                f'[{position}/{len(references)}] reference='
                f'{reference_index} failed: {error}')
            if args.fail_fast:
                rows.append(row)
                _write_progress(progress_path, rows, len(references))
                raise
        rows.append(row)
        _write_progress(progress_path, rows, len(references))

    _write_progress(progress_path, rows, len(references))
    failed = [row for row in rows if row['status'] == 'failed']
    print(
        f'Batch complete: {len(rows) - len(failed)}/{len(rows)} '
        f'sequences ready, {len(failed)} failed')
    if failed:
        raise RuntimeError(
            f'{len(failed)} sequences failed; see {progress_path}')


if __name__ == '__main__':
    main()
