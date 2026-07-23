#!/usr/bin/env python
"""Measure current OccWorld-anchor drift after replacing GT with track boxes."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.data_converter.generate_kl_occworld_labels import (
    INSTANCE_OCCUPIED,
    _collect_boxes,
    _load_infos,
    _resolve_path,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    _same_scene_indices,
)
from tools.data_converter.kl_occworld_online import (
    KLOccWorldOnlineBuilder,
)
from tools.data_converter.kl_occworld_track_adapter import (
    track_boxes_to_occworld_z_convention,
)


def _prediction_index(root: Path) -> dict:
    paths = sorted(root.glob('*/*__occworld_prediction.npz'))
    mapping = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as prediction:
            reference = int(prediction['reference_index'])
            required = {
                'track_boxes_3d', 'track_scores_3d',
                'track_runtime_scores', 'track_labels_3d', 'track_ids',
            }
            missing = sorted(required.difference(prediction.files))
        if missing:
            raise KeyError(
                f'{path} lacks track-box fields; rerun exporter with '
                f'--save-track-boxes: {missing}')
        if reference in mapping:
            raise ValueError(f'Duplicate prediction for {reference}')
        mapping[reference] = path
    if not mapping:
        raise FileNotFoundError(f'No predictions below {root}')
    return mapping


def _instances_from_prediction(prediction) -> tuple:
    boxes = np.asarray(prediction['track_boxes_3d'], dtype=np.float32)
    source_z_origin = (
        str(prediction['track_box_z_origin'].item())
        if 'track_box_z_origin' in prediction.files else 'bottom')
    boxes = track_boxes_to_occworld_z_convention(
        boxes, source_z_origin=source_z_origin)
    scores = np.asarray(prediction['track_scores_3d'], dtype=np.float32)
    runtime_scores = np.asarray(
        prediction['track_runtime_scores'], dtype=np.float32)
    labels = np.asarray(prediction['track_labels_3d'], dtype=np.int64)
    track_ids = np.asarray(prediction['track_ids'], dtype=np.int64)
    count = boxes.shape[0]
    if (boxes.ndim != 2 or boxes.shape[1] != 7 or any(
            values.shape != (count, )
            for values in (scores, runtime_scores, labels, track_ids))):
        raise ValueError('Malformed saved TrackFormer box payload')
    instances = [
        {
            'bbox_3d': box,
            'bbox_3d_isvalid': True,
            'bbox_label_3d': int(label),
            'track_id': int(track_id),
            'score_3d': float(score),
            'track_score': float(runtime_score),
        }
        for box, score, runtime_score, label, track_id in zip(
            boxes, scores, runtime_scores, labels, track_ids)
    ]
    return boxes, instances, source_z_origin


def _overlap_metrics(reference_state: np.ndarray,
                     predicted_state: np.ndarray) -> dict:
    reference_instance = reference_state == INSTANCE_OCCUPIED
    predicted_instance = predicted_state == INSTANCE_OCCUPIED
    intersection = int(np.count_nonzero(reference_instance & predicted_instance))
    union = int(np.count_nonzero(reference_instance | predicted_instance))
    predicted_count = int(np.count_nonzero(predicted_instance))
    reference_count = int(np.count_nonzero(reference_instance))
    return {
        'reference_instance_voxels': reference_count,
        'predicted_instance_voxels': predicted_count,
        'intersection_voxels': intersection,
        'union_voxels': union,
        'instance_iou': intersection / union if union else 1.0,
        'instance_precision': (
            intersection / predicted_count if predicted_count else
            (1.0 if reference_count == 0 else 0.0)),
        'instance_recall': (
            intersection / reference_count if reference_count else
            (1.0 if predicted_count == 0 else 0.0)),
        'state_mismatch_voxels': int(np.count_nonzero(
            reference_state != predicted_state)),
        'state_mismatch_ratio': float(np.mean(
            reference_state != predicted_state)),
    }


def audit_reference(infos, metainfo, reference_index: int,
                    prediction_path: Path, sequence_root: Path,
                    args, state_out_dir: Path = None,
                    box_source: str = 'predicted') -> dict:
    sequence_paths = sorted(
        (sequence_root / f'{reference_index:06d}').glob(
            '*__occworld_sequence.npz'))
    if len(sequence_paths) != 1:
        raise FileNotFoundError(
            f'Expected one sequence for {reference_index}, got '
            f'{len(sequence_paths)}')
    with np.load(sequence_paths[0], allow_pickle=False) as sequence:
        reference_state = np.asarray(
            sequence['current_observation_state_3d'], dtype=np.uint8)
    if box_source == 'predicted':
        with np.load(prediction_path, allow_pickle=False) as prediction:
            boxes, instances, source_z_origin = _instances_from_prediction(
                prediction)
    elif box_source == 'ray_only':
        boxes = np.empty((0, 7), dtype=np.float32)
        instances = []
        source_z_origin = 'not_applicable'
    else:
        raise ValueError(f'Unsupported box_source={box_source}')

    # This is intentionally a current-frame substitution audit. The two
    # causal support frames remain annotation replay, while the reference
    # frame uses TrackFormer boxes. A full five-frame predicted-box replay is
    # a separate follow-up because it needs sequential tracker inference.
    indices = _same_scene_indices(infos, reference_index, [-2, -1, 0])
    online = KLOccWorldOnlineBuilder(
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size,
        target_frame=str(metainfo.get('lidar_coord_frame', 'FLU')),
        collision_z=args.collision_z,
    )
    for index in indices[:-1]:
        online.push_info(infos[index])
    reference_info = infos[reference_index]
    evidence = online.label_builder.build(
        reference_info,
        diagnostics=True,
        boxes=boxes,
        instances=instances,
    )
    predicted_output = online.push_evidence(
        evidence=evidence,
        timestamp=float(reference_info['timestamp']),
        ego2global=np.asarray(reference_info['ego2global'], dtype=np.float64),
        scene_token=str(reference_info.get('scene_token', '')),
    )
    annotation_boxes, _ = _collect_boxes(reference_info)
    metrics = _overlap_metrics(
        reference_state, predicted_output.current_observation_state_3d)
    anchor_path = None
    if state_out_dir is not None:
        anchor_dir = state_out_dir / f'{reference_index:06d}'
        anchor_dir.mkdir(parents=True, exist_ok=True)
        anchor_path = anchor_dir / 'occworld_current_anchor.npz'
        np.savez_compressed(
            anchor_path,
            reference_index=np.int64(reference_index),
            current_world_state_3d=(
                predicted_output.current_observation_state_3d),
            current_world_valid_3d=(
                predicted_output.current_observation_valid_3d),
            track_boxes_3d=boxes,
            track_box_z_origin=np.asarray('occworld_legacy_center'),
            causal_support_indices=np.asarray(indices[:-1], dtype=np.int64),
        )
    return {
        'reference_index': int(reference_index),
        'scene_token': str(reference_info.get('scene_token', '')),
        'causal_support_indices': [int(index) for index in indices[:-1]],
        'substitution_scope': (
            f'reference-frame boxes {box_source}; two support frames replay '
            'their annotation boxes'),
        'annotation_box_count': int(annotation_boxes.shape[0]),
        'predicted_track_box_count': int(boxes.shape[0]),
        'predicted_box_source_z_origin': source_z_origin,
        'occworld_box_z_origin': 'occworld_legacy_center',
        'prediction_path': (
            None if box_source == 'ray_only' else str(prediction_path)),
        'sequence_path': str(sequence_paths[0]),
        'current_anchor_path': (
            None if anchor_path is None else str(anchor_path)),
        **metrics,
    }


def _aggregate(rows: list) -> dict:
    keys = (
        'reference_instance_voxels', 'predicted_instance_voxels',
        'intersection_voxels', 'union_voxels', 'state_mismatch_voxels')
    totals = {key: sum(row[key] for row in rows) for key in keys}
    return {
        **totals,
        'instance_iou': (
            totals['intersection_voxels'] / totals['union_voxels']
            if totals['union_voxels'] else 1.0),
        'instance_precision': (
            totals['intersection_voxels'] /
            totals['predicted_instance_voxels']
            if totals['predicted_instance_voxels'] else 0.0),
        'instance_recall': (
            totals['intersection_voxels'] /
            totals['reference_instance_voxels']
            if totals['reference_instance_voxels'] else 0.0),
        'mean_state_mismatch_ratio': float(np.mean([
            row['state_mismatch_ratio'] for row in rows
        ])),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b15_epoch7_validation_track_boxes_v1/'
            'validation/epoch_007'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument('--reference-indices', type=int, nargs='*')
    parser.add_argument(
        '--box-source', choices=('predicted', 'ray_only'),
        default='predicted')
    parser.add_argument(
        '--state-out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predicted_current_box_anchors_v1'))
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b15_predicted_current_box_anchor_v1.json'))
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
    prediction_map = _prediction_index(args.prediction_root)
    selected = (
        sorted(prediction_map) if args.reference_indices is None
        else sorted(set(args.reference_indices)))
    missing = sorted(set(selected).difference(prediction_map))
    if missing:
        raise KeyError(f'Missing track-box predictions for {missing}')
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    rows = [
        audit_reference(
            infos, metainfo, reference, prediction_map[reference],
            args.sequence_root, args, state_out_dir=args.state_out_dir,
            box_source=args.box_source)
        for reference in selected
    ]
    report = {
        'schema_version': 1,
        'purpose': (
            'Current-frame box-source substitution audit. This is not a '
            'full historical predicted-box or final-holdout evaluation.'),
        'box_source': args.box_source,
        'reference_count': len(rows),
        'aggregate': _aggregate(rows),
        'rows': rows,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
