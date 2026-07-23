#!/usr/bin/env python
"""Generate fixed-time, reference-aligned KL OccWorld history queues."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.analysis_tools.audit_kl_occworld_dual_representation import (
    _save_state,
    _tile,
)
from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _output_stem,
    _resolve_path,
)
from tools.data_converter.generate_kl_occworld_observation_cache import (
    generate_observation_cache,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    OCCUPANCY_PALETTE,
    _observation_cache_index,
    _project_world_to_bev,
    _same_scene_indices,
    _validate_frame_offsets,
    _warp_state_to_reference,
)


def _reference_indices(summary_path: Path,
                       explicit_indices: List[int]) -> List[int]:
    if explicit_indices:
        return sorted(set(explicit_indices))
    if not summary_path:
        raise ValueError(
            'Provide --reference-indices or --history-eligibility-summary')
    with summary_path.open() as f:
        summary = json.load(f)
    return sorted(set(int(index) for index in summary['eligible_indices']))


def _voxel_z_centers(pc_range, occ_size) -> np.ndarray:
    z_size = int(occ_size[2])
    step = (float(pc_range[5]) - float(pc_range[2])) / z_size
    return float(pc_range[2]) + (np.arange(z_size) + 0.5) * step


def _save_history_contact(path: Path, image_paths: List[Path],
                          history_times: np.ndarray):
    canvas = np.concatenate([
        _tile(cv2.imread(str(image_path)), f't={time_value:.1f}s',
              (300, 225))
        for image_path, time_value in zip(image_paths, history_times)
    ], axis=1)
    cv2.imwrite(str(path), canvas)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--history-eligibility-summary', type=Path)
    parser.add_argument('--reference-indices', type=int, nargs='*', default=[])
    parser.add_argument('--history-offsets', type=int, nargs='+',
                        default=[-4, -3, -2, -1, 0])
    parser.add_argument('--expected-step-s', type=float, default=0.5)
    parser.add_argument('--max-time-error-s', type=float, default=0.2)
    parser.add_argument('--observation-cache-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--overwrite-cache', action='store_true')
    parser.add_argument('--overwrite-history', action='store_true')
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
    offsets = sorted(set(args.history_offsets))
    if not offsets or offsets[-1] != 0 or any(offset > 0 for offset in offsets):
        raise ValueError(
            'history-offsets must end at 0 and contain no future offsets')
    references = _reference_indices(
        args.history_eligibility_summary, args.reference_indices)
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    for reference_index in references:
        _same_scene_indices(infos, reference_index, offsets)
        _validate_frame_offsets(
            infos, reference_index, offsets, args.expected_step_s,
            args.max_time_error_s, window_name='history')

    cache_args = Namespace(
        indices=[],
        reference_indices=references,
        offsets=offsets,
        out_dir=str(args.observation_cache_dir),
        overwrite=args.overwrite_cache,
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size,
        collision_z=args.collision_z,
    )
    generate_observation_cache(cache_args, infos, metainfo)
    observation_cache: Dict[int, Path] = _observation_cache_index(
        args.observation_cache_dir)
    z_centers = _voxel_z_centers(args.pc_range, args.occ_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for position, reference_index in enumerate(references, start=1):
        history_indices = _same_scene_indices(
            infos, reference_index, offsets)
        reference_ego2global = np.asarray(
            infos[reference_index]['ego2global'], dtype=np.float64)
        history_states = []
        for history_index in history_indices:
            if history_index not in observation_cache:
                raise FileNotFoundError(
                    f'Missing observation cache for frame {history_index}')
            with np.load(
                    observation_cache[history_index],
                    allow_pickle=False) as cached:
                if not np.allclose(cached['pc_range'], args.pc_range):
                    raise ValueError(
                        f'Cache pc_range mismatch for frame {history_index}')
                if not np.array_equal(cached['occ_size'], args.occ_size):
                    raise ValueError(
                        f'Cache occ_size mismatch for frame {history_index}')
                state = np.array(
                    cached['observation_state_3d'], copy=True)
            history_ego2global = np.asarray(
                infos[history_index]['ego2global'], dtype=np.float64)
            history_states.append(_warp_state_to_reference(
                state, history_ego2global, reference_ego2global,
                args.pc_range, args.occ_size))
        history_states = np.stack(history_states, axis=0)
        reference_time = float(infos[reference_index]['timestamp'])
        history_times = np.asarray([
            float(infos[index]['timestamp']) - reference_time
            for index in history_indices
        ], dtype=np.float32)
        history_ego2global = np.asarray([
            infos[index]['ego2global'] for index in history_indices
        ], dtype=np.float64)
        history_to_reference = np.asarray([
            np.linalg.inv(reference_ego2global) @ transform
            for transform in history_ego2global
        ], dtype=np.float64)
        history_timestamps = np.asarray([
            infos[index]['timestamp'] for index in history_indices
        ], dtype=np.float64)

        sequence_dir = args.out_dir / f'{reference_index:06d}'
        sequence_dir.mkdir(parents=True, exist_ok=True)
        stem = _output_stem(infos[reference_index])
        output_path = sequence_dir / f'{stem}__occworld_history.npz'
        status = 'existing'
        if args.overwrite_history or not output_path.exists():
            np.savez_compressed(
                output_path,
                history_observation_state_3d=history_states,
                history_observation_valid_3d=(
                    history_states != 0).astype(np.uint8),
                history_offsets=np.asarray(offsets, dtype=np.int16),
                history_indices=np.asarray(history_indices, dtype=np.int64),
                history_times_s=history_times,
                nominal_history_times_s=(
                    np.asarray(offsets, dtype=np.float32) *
                    np.float32(args.expected_step_s)),
                history_timestamps=history_timestamps,
                history_ego2global=history_ego2global,
                history_to_reference=history_to_reference,
                reference_index=np.int64(reference_index),
                reference_ego2global=reference_ego2global,
                pc_range=np.asarray(args.pc_range, dtype=np.float32),
                occ_size=np.asarray(args.occ_size, dtype=np.int16),
                collision_z=np.asarray(args.collision_z, dtype=np.float32),
            )
            image_paths = []
            for history_position, state in enumerate(history_states):
                bev = _project_world_to_bev(
                    state, z_centers, args.collision_z)
                image_path = sequence_dir / (
                    f'{stem}__history_{history_position}.png')
                _save_state(image_path, bev, OCCUPANCY_PALETTE)
                image_paths.append(image_path)
            _save_history_contact(
                sequence_dir / f'{stem}__history_contact_sheet.png',
                image_paths, history_times)
            status = 'generated'
        row = {
            'reference_index': reference_index,
            'status': status,
            'history_shape': list(history_states.shape),
            'history_times_s': history_times.tolist(),
            'known_voxels_by_history': [
                int(np.count_nonzero(state)) for state in history_states],
            'output_path': str(output_path),
        }
        rows.append(row)
        print(
            f'[{position}/{len(references)}] reference={reference_index} '
            f'{status} shape={tuple(history_states.shape)}')

    summary = {
        'reference_count': len(references),
        'generated_count': sum(row['status'] == 'generated' for row in rows),
        'existing_count': sum(row['status'] == 'existing' for row in rows),
        'history_offsets': offsets,
        'history_shape': rows[0]['history_shape'],
        'rows': rows,
    }
    with (args.out_dir / 'batch_generation_summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
