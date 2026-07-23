#!/usr/bin/env python
"""Audit online OccWorld preprocessing against frozen cached inputs."""

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _resolve_path,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    _same_scene_indices,
)
from tools.data_converter.kl_occworld_online import (
    KLOccWorldOnlineBuilder,
)


def _single_file(root: Path, reference_index: int, pattern: str) -> Path:
    paths = sorted((root / f'{reference_index:06d}').glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(
            f'Expected one {pattern} for reference {reference_index}, '
            f'found {len(paths)} below {root}')
    return paths[0]


def _difference(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        return {
            'shape_match': False,
            'actual_shape': list(actual.shape),
            'expected_shape': list(expected.shape),
            'mismatch_count': None,
            'exact_match': False,
        }
    mismatch = int(np.count_nonzero(actual != expected))
    return {
        'shape_match': True,
        'actual_shape': list(actual.shape),
        'expected_shape': list(expected.shape),
        'mismatch_count': mismatch,
        'exact_match': mismatch == 0,
    }


def audit_reference(
        infos,
        metainfo: dict,
        reference_index: int,
        sequence_root: Path,
        history_root: Path,
        args) -> dict:
    indices = _same_scene_indices(
        infos, reference_index, [-4, -3, -2, -1, 0])
    online = KLOccWorldOnlineBuilder(
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size,
        target_frame=str(metainfo.get('lidar_coord_frame', 'FLU')),
        collision_z=args.collision_z,
    )

    frame_times = []
    output = None
    for index in indices:
        start = time.perf_counter()
        output = online.push_info(infos[index])
        frame_times.append(time.perf_counter() - start)
    if output is None or not output.ready:
        raise RuntimeError('Online queue did not produce a full history')

    sequence_path = _single_file(
        sequence_root, reference_index, '*__occworld_sequence.npz')
    history_path = _single_file(
        history_root, reference_index, '*__occworld_history.npz')
    with np.load(sequence_path, allow_pickle=False) as sequence:
        expected_current = np.asarray(
            sequence['current_observation_state_3d'], dtype=np.uint8)
        expected_current_valid = np.asarray(
            sequence['current_observation_valid_3d'], dtype=np.bool_)
        expected_direct = np.asarray(
            sequence['direct_observation_state_3d'][0], dtype=np.uint8)
    with np.load(history_path, allow_pickle=False) as history:
        expected_history = np.asarray(
            history['history_observation_state_3d'], dtype=np.uint8)
        expected_history_valid = np.asarray(
            history['history_observation_valid_3d'], dtype=np.bool_)
        expected_history_times = np.asarray(
            history['history_times_s'], dtype=np.float32)

    comparisons = {
        'current_state': _difference(
            output.current_observation_state_3d, expected_current),
        'current_valid': _difference(
            output.current_observation_valid_3d, expected_current_valid),
        'direct_current_state': _difference(
            output.direct_observation_state_3d, expected_direct),
        'history_state': _difference(
            output.history_observation_state_3d, expected_history),
        'history_valid': _difference(
            output.history_observation_valid_3d,
            expected_history_valid),
        'history_times_s': _difference(
            np.round(output.history_times_s, 5),
            np.round(expected_history_times, 5)),
    }
    return {
        'reference_index': int(reference_index),
        'frame_indices': [int(index) for index in indices],
        'scene_token': output.scene_token,
        'box_source': 'dataset_annotations_for_replay_only',
        'online_ready': output.ready,
        'promoted_uncertain_voxels': int(np.count_nonzero(
            output.promoted_uncertain_mask_3d)),
        'frame_latency_s': frame_times,
        'mean_frame_latency_s': float(np.mean(frame_times)),
        'max_frame_latency_s': float(np.max(frame_times)),
        'comparisons': comparisons,
        'exact_match': all(
            comparison['exact_match']
            for comparison in comparisons.values()),
        'sequence_path': str(sequence_path),
        'history_path': str(history_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument(
        '--reference-indices', type=int, nargs='+', default=[31669])
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument(
        '--history-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_history_expanded70'))
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_online_replay_audit_v1.json'))
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument(
        '--occ-size', type=int, nargs=3, default=[160, 120, 10])
    parser.add_argument(
        '--collision-z', type=float, nargs=2, default=[0.3, 2.5])
    return parser.parse_args()


def main():
    args = parse_args()
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    rows = [
        audit_reference(
            infos=infos,
            metainfo=metainfo,
            reference_index=index,
            sequence_root=args.sequence_root,
            history_root=args.history_root,
            args=args,
        )
        for index in args.reference_indices
    ]
    report = {
        'schema_version': 1,
        'purpose': (
            'Validate CPU online preprocessing parity without reading '
            'future labels or the sealed final holdout.'),
        'reference_count': len(rows),
        'all_exact_match': all(row['exact_match'] for row in rows),
        'mean_frame_latency_s': float(np.mean([
            latency
            for row in rows
            for latency in row['frame_latency_s']
        ])),
        'rows': rows,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['all_exact_match']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
