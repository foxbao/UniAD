#!/usr/bin/env python
"""Build B15 online inputs from five sequential TrackFormer predictions."""

import argparse
import json
import sys
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
from tools.data_converter.kl_occworld_track_queue import (
    unpack_track_queue_frame,
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
            'mismatch_ratio': None,
            'exact_match': False,
        }
    mismatch = int(np.count_nonzero(actual != expected))
    return {
        'shape_match': True,
        'actual_shape': list(actual.shape),
        'expected_shape': list(expected.shape),
        'mismatch_count': mismatch,
        'mismatch_ratio': float(np.mean(actual != expected)),
        'exact_match': mismatch == 0,
    }


def _mask_overlap(predicted: np.ndarray, reference: np.ndarray) -> dict:
    predicted = np.asarray(predicted, dtype=np.bool_)
    reference = np.asarray(reference, dtype=np.bool_)
    intersection = int(np.count_nonzero(predicted & reference))
    union = int(np.count_nonzero(predicted | reference))
    predicted_count = int(np.count_nonzero(predicted))
    reference_count = int(np.count_nonzero(reference))
    return {
        'predicted_voxels': predicted_count,
        'reference_voxels': reference_count,
        'intersection_voxels': intersection,
        'union_voxels': union,
        'iou': intersection / union if union else 1.0,
        'precision': (
            intersection / predicted_count if predicted_count else
            (1.0 if reference_count == 0 else 0.0)),
        'recall': (
            intersection / reference_count if reference_count else
            (1.0 if predicted_count == 0 else 0.0)),
    }


def _load_track_queue(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as queue:
        return {key: np.array(queue[key], copy=True) for key in queue.files}


def build_reference(infos, metainfo: dict, reference_index: int,
                    track_queue_root: Path, sequence_root: Path,
                    history_root: Path, output_root: Path, args) -> dict:
    track_path = track_queue_root / (
        f'{reference_index:06d}/occworld_track_queue.npz')
    if not track_path.exists():
        raise FileNotFoundError(f'Missing TrackFormer queue: {track_path}')
    queue = _load_track_queue(track_path)
    if int(queue['reference_index']) != reference_index:
        raise ValueError(f'Wrong reference index in {track_path}')
    queue_indices = np.asarray(
        queue['queue_frame_indices'], dtype=np.int64)
    expected_indices = np.asarray(
        _same_scene_indices(infos, reference_index, [-4, -3, -2, -1, 0]),
        dtype=np.int64)
    if not np.array_equal(queue_indices, expected_indices):
        raise ValueError(
            f'Track queue indices {queue_indices.tolist()} do not match '
            f'expected history {expected_indices.tolist()}')

    online = KLOccWorldOnlineBuilder(
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size,
        target_frame=str(metainfo.get('lidar_coord_frame', 'FLU')),
        collision_z=args.collision_z,
        expected_step_s=args.expected_step_s,
        max_time_error_s=args.max_time_error_s,
    )
    per_frame_instance = []
    output = None
    for frame_position, frame_index in enumerate(queue_indices):
        info = infos[int(frame_index)]
        boxes, instances = unpack_track_queue_frame(queue, frame_position)
        predicted_evidence = online.label_builder.build(
            info,
            diagnostics=True,
            boxes=boxes,
            instances=instances)
        overlap = {
            'frame_index': int(frame_index),
            'track_box_count': int(boxes.shape[0]),
        }
        if not args.skip_instance_overlap_audit:
            annotation_evidence = online.label_builder.build(
                info, diagnostics=True)
            overlap.update(_mask_overlap(
                predicted_evidence['box_occupied_3d'] > 0,
                annotation_evidence['box_occupied_3d'] > 0))
        per_frame_instance.append(overlap)
        output = online.push_evidence(
            evidence=predicted_evidence,
            timestamp=float(info['timestamp']),
            ego2global=np.asarray(info['ego2global'], dtype=np.float64),
            scene_token=str(info.get('scene_token', '')))
    if output is None or not output.ready:
        raise RuntimeError('Predicted TrackFormer queue did not fill history')

    sequence_path = _single_file(
        sequence_root, reference_index, '*__occworld_sequence.npz')
    history_path = _single_file(
        history_root, reference_index, '*__occworld_history.npz')
    with np.load(sequence_path, allow_pickle=False) as sequence:
        expected_current = np.asarray(
            sequence['current_observation_state_3d'], dtype=np.uint8)
        expected_current_valid = np.asarray(
            sequence['current_observation_valid_3d'], dtype=np.bool_)
    with np.load(history_path, allow_pickle=False) as history:
        expected_history = np.asarray(
            history['history_observation_state_3d'], dtype=np.uint8)
        expected_history_valid = np.asarray(
            history['history_observation_valid_3d'], dtype=np.bool_)

    output_dir = output_root / f'{reference_index:06d}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'occworld_online_input.npz'
    np.savez_compressed(
        output_path,
        reference_index=np.int64(reference_index),
        current_world_state_3d=output.current_observation_state_3d,
        current_world_valid_3d=output.current_observation_valid_3d,
        history_world_state_3d=output.history_observation_state_3d,
        history_world_valid_3d=output.history_observation_valid_3d,
        history_frame_valid=output.history_frame_valid,
        history_times_s=output.history_times_s,
        history_to_reference=output.history_to_reference,
        queue_frame_indices=queue_indices,
        queue_timestamps=queue['queue_timestamps'],
        queue_ego2global=queue['queue_ego2global'],
        track_box_offsets=queue['track_box_offsets'],
        track_kept_counts=queue['track_kept_counts'],
        track_queue_path=np.asarray(str(track_path)),
        input_contract=np.asarray(
            'five_frame_sequential_trackformer_predicted_boxes'))
    comparisons = {
        'current_state_vs_annotation': _difference(
            output.current_observation_state_3d, expected_current),
        'current_valid_vs_annotation': _difference(
            output.current_observation_valid_3d, expected_current_valid),
        'history_state_vs_annotation': _difference(
            output.history_observation_state_3d, expected_history),
        'history_valid_vs_annotation': _difference(
            output.history_observation_valid_3d, expected_history_valid),
    }
    return {
        'reference_index': reference_index,
        'scene_token': output.scene_token,
        'queue_frame_indices': queue_indices.tolist(),
        'track_kept_counts': queue['track_kept_counts'].tolist(),
        'per_frame_instance_overlap': per_frame_instance,
        'comparisons': comparisons,
        'track_queue_path': str(track_path),
        'online_input_path': str(output_path),
        'sequence_path': str(sequence_path),
        'history_path': str(history_path),
    }


def _aggregate(rows: list) -> dict:
    overlaps = [
        overlap for row in rows
        for overlap in row['per_frame_instance_overlap']
        if 'intersection_voxels' in overlap
    ]
    intersection = sum(
        item['intersection_voxels'] for item in overlaps)
    union = sum(item['union_voxels'] for item in overlaps)
    predicted = sum(item['predicted_voxels'] for item in overlaps)
    reference = sum(item['reference_voxels'] for item in overlaps)
    return {
        'reference_count': len(rows),
        'frame_count': sum(
            len(row['per_frame_instance_overlap']) for row in rows),
        'instance_overlap_audited_frame_count': len(overlaps),
        'instance_iou': (
            intersection / union if overlaps and union else
            (1.0 if overlaps else None)),
        'instance_precision': (
            intersection / predicted if overlaps and predicted else
            (0.0 if overlaps else None)),
        'instance_recall': (
            intersection / reference if overlaps and reference else
            (0.0 if overlaps else None)),
        'mean_current_state_mismatch_ratio': float(np.mean([
            row['comparisons']['current_state_vs_annotation'][
                'mismatch_ratio'] for row in rows
        ])),
        'mean_history_state_mismatch_ratio': float(np.mean([
            row['comparisons']['history_state_vs_annotation'][
                'mismatch_ratio'] for row in rows
        ])),
    }


def _write_report(report: dict, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')


def _merge_reports(paths, out_file: Path):
    reports = [
        json.loads(path.read_text(encoding='utf-8')) for path in paths
    ]
    if not reports:
        raise ValueError('At least one report is required for merging')
    metadata_keys = (
        'protocol', 'ann_file', 'track_queue_root', 'sequence_root',
        'history_root', 'output_root', 'instance_overlap_audit',
        'expected_step_s', 'max_time_error_s')
    baseline_metadata = {
        key: reports[0].get(key) for key in metadata_keys
    }
    for report in reports[1:]:
        metadata = {key: report.get(key) for key in metadata_keys}
        if metadata != baseline_metadata:
            raise ValueError('Shard reports use different input protocols')
    rows = [row for report in reports for row in report['rows']]
    references = [int(row['reference_index']) for row in rows]
    if len(references) != len(set(references)):
        raise ValueError('Merged reports contain duplicate references')
    rows.sort(key=lambda row: int(row['reference_index']))
    report = baseline_metadata
    report['aggregate'] = _aggregate(rows)
    report['rows'] = rows
    report['merged_from'] = [str(path) for path in paths]
    _write_report(report, out_file)
    print(json.dumps(report['aggregate'], ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument(
        '--track-queue-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_track_queues_b15_validation_v1'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument(
        '--history-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_history_expanded70'))
    parser.add_argument(
        '--output-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_online_inputs_full_predicted_history_validation_v1'))
    parser.add_argument('--reference-indices', type=int, nargs='*')
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_full_predicted_history_audit_v1.json'))
    parser.add_argument(
        '--merge-report-files', type=Path, nargs='+',
        help='Merge existing shard reports without rebuilding inputs.')
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument(
        '--occ-size', type=int, nargs=3, default=[160, 120, 10])
    parser.add_argument(
        '--collision-z', type=float, nargs=2, default=[0.3, 2.5])
    parser.add_argument('--expected-step-s', type=float, default=0.5)
    parser.add_argument('--max-time-error-s', type=float, default=0.2)
    parser.add_argument(
        '--skip-instance-overlap-audit', action='store_true',
        help=(
            'Skip the second annotation-box rasterization per frame; frozen '
            'current/history mismatch metrics are still computed.'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.merge_report_files is not None:
        _merge_reports(args.merge_report_files, args.out_file)
        return
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    references = args.reference_indices
    if references is None:
        references = sorted(
            int(path.parent.name)
            for path in args.track_queue_root.glob(
                '*/occworld_track_queue.npz'))
    if not references:
        raise FileNotFoundError(
            f'No TrackFormer queues below {args.track_queue_root}')
    if args.shard_count < 1:
        raise ValueError('shard-count must be positive')
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError(
            f'shard-index must be in [0,{args.shard_count}), got '
            f'{args.shard_index}')
    references = references[args.shard_index::args.shard_count]
    if not references:
        raise ValueError('Selected shard contains no references')
    rows = []
    for position, reference_index in enumerate(references, start=1):
        row = build_reference(
            infos=infos,
            metainfo=metainfo,
            reference_index=int(reference_index),
            track_queue_root=args.track_queue_root,
            sequence_root=args.sequence_root,
            history_root=args.history_root,
            output_root=args.output_root,
            args=args)
        rows.append(row)
        print(
            f'[{position}/{len(references)}] reference={reference_index} '
            f"history_mismatch={row['comparisons']['history_state_vs_annotation']['mismatch_ratio']:.6f}")
    report = {
        'protocol': (
            'five-frame sequential TrackFormer predicted boxes; annotation '
            'sequence/history used only for offline audit metrics'),
        'instance_overlap_audit': not args.skip_instance_overlap_audit,
        'expected_step_s': args.expected_step_s,
        'max_time_error_s': args.max_time_error_s,
        'ann_file': str(args.ann_file),
        'track_queue_root': str(args.track_queue_root),
        'sequence_root': str(args.sequence_root),
        'history_root': str(args.history_root),
        'output_root': str(args.output_root),
        'shard_count': args.shard_count,
        'shard_index': args.shard_index,
        'aggregate': _aggregate(rows),
        'rows': rows,
    }
    _write_report(report, args.out_file)
    print(json.dumps(report['aggregate'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
