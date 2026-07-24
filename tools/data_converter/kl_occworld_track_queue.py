"""Pack and unpack sequential TrackFormer queue predictions."""

from typing import Mapping, Sequence

import numpy as np

from tools.data_converter.kl_occworld_track_adapter import (
    track_result_to_occworld_instances,
)


def pack_track_queue_results(
        queue_results: Sequence[Mapping],
        score_threshold: float = 0.1,
        class_count: int = None) -> dict:
    if not queue_results:
        raise ValueError('Track queue must contain at least one frame')
    box_chunks = []
    score_chunks = []
    runtime_score_chunks = []
    label_chunks = []
    id_chunks = []
    offsets = [0]
    input_counts = []
    kept_counts = []
    frame_indices = []
    timestamps = []
    ego2globals = []
    scene_tokens = []
    tokens = []
    for result in queue_results:
        boxes, instances, summary = track_result_to_occworld_instances(
            result,
            score_threshold=score_threshold,
            track_score_threshold=score_threshold,
            class_count=class_count)
        box_chunks.append(boxes)
        score_chunks.append(np.asarray([
            instance['score_3d'] for instance in instances
        ], dtype=np.float32))
        runtime_score_chunks.append(np.asarray([
            instance['track_score'] for instance in instances
        ], dtype=np.float32))
        label_chunks.append(np.asarray([
            instance['bbox_label_3d'] for instance in instances
        ], dtype=np.int64))
        id_chunks.append(np.asarray([
            instance['track_id'] for instance in instances
        ], dtype=np.int64))
        offsets.append(offsets[-1] + boxes.shape[0])
        input_counts.append(summary['input_count'])
        kept_counts.append(summary['kept_count'])
        frame_indices.append(int(result['sample_idx']))
        timestamps.append(float(result['timestamp']))
        ego2global = np.asarray(result['ego2global'], dtype=np.float64)
        if ego2global.shape != (4, 4):
            raise ValueError(f'ego2global must be [4,4], got {ego2global.shape}')
        ego2globals.append(ego2global)
        scene_tokens.append(str(result.get('scene_token', '')))
        tokens.append(str(result.get('token', '')))
    if len(set(scene_tokens)) != 1:
        raise ValueError('Track queue crosses a scene boundary')
    if np.any(np.diff(np.asarray(timestamps, dtype=np.float64)) <= 0):
        raise ValueError('Track queue timestamps must increase')

    concatenate = lambda chunks, shape, dtype: (
        np.concatenate(chunks, axis=0)
        if any(chunk.shape[0] for chunk in chunks)
        else np.empty(shape, dtype=dtype))
    return {
        'queue_frame_indices': np.asarray(frame_indices, dtype=np.int64),
        'queue_timestamps': np.asarray(timestamps, dtype=np.float64),
        'queue_ego2global': np.stack(ego2globals),
        'queue_scene_tokens': np.asarray(scene_tokens),
        'queue_tokens': np.asarray(tokens),
        'track_box_offsets': np.asarray(offsets, dtype=np.int64),
        'track_boxes_3d': concatenate(
            box_chunks, (0, 7), np.float32),
        'track_scores_3d': concatenate(
            score_chunks, (0,), np.float32),
        'track_runtime_scores': concatenate(
            runtime_score_chunks, (0,), np.float32),
        'track_labels_3d': concatenate(
            label_chunks, (0,), np.int64),
        'track_ids': concatenate(id_chunks, (0,), np.int64),
        'track_input_counts': np.asarray(input_counts, dtype=np.int64),
        'track_kept_counts': np.asarray(kept_counts, dtype=np.int64),
        'track_box_z_origin': np.asarray('occworld_legacy_center'),
        'track_score_threshold': np.float32(score_threshold),
    }


def unpack_track_queue_frame(payload: Mapping, frame_position: int) -> tuple:
    frame_indices = np.asarray(payload['queue_frame_indices'], dtype=np.int64)
    frame_count = frame_indices.shape[0]
    if not 0 <= frame_position < frame_count:
        raise IndexError(
            f'frame_position must be in [0,{frame_count}), got '
            f'{frame_position}')
    offsets = np.asarray(payload['track_box_offsets'], dtype=np.int64)
    boxes_all = np.asarray(payload['track_boxes_3d'], dtype=np.float32)
    scores_all = np.asarray(payload['track_scores_3d'], dtype=np.float32)
    runtime_scores_all = np.asarray(
        payload['track_runtime_scores'], dtype=np.float32)
    labels_all = np.asarray(payload['track_labels_3d'], dtype=np.int64)
    track_ids_all = np.asarray(payload['track_ids'], dtype=np.int64)
    total = boxes_all.shape[0]
    if (offsets.shape != (frame_count + 1,) or offsets[0] != 0 or
            offsets[-1] != total or np.any(np.diff(offsets) < 0)):
        raise ValueError('Malformed track_box_offsets')
    if boxes_all.ndim != 2 or boxes_all.shape[1] != 7:
        raise ValueError('track_boxes_3d must have shape [N,7]')
    if not all(values.shape == (total,) for values in (
            scores_all, runtime_scores_all, labels_all, track_ids_all)):
        raise ValueError('Track queue field lengths do not match boxes')
    start, end = int(offsets[frame_position]), int(offsets[frame_position + 1])
    boxes = boxes_all[start:end]
    scores = scores_all[start:end]
    runtime_scores = runtime_scores_all[start:end]
    labels = labels_all[start:end]
    track_ids = track_ids_all[start:end]
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
    return boxes, instances
