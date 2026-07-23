"""Convert UniAD TrackFormer results into OccWorld instance-box inputs."""

from typing import Mapping, Tuple

import numpy as np


def _array(value, dtype=None) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'cpu') and hasattr(value, 'numpy'):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _box_tensor(boxes_3d) -> np.ndarray:
    tensor = getattr(boxes_3d, 'tensor', boxes_3d)
    boxes = _array(tensor, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(
            f'boxes_3d must have shape [N,>=7], got {boxes.shape}')
    return boxes[:, :7]


def track_boxes_to_occworld_z_convention(
        boxes: np.ndarray,
        source_z_origin: str = 'bottom') -> np.ndarray:
    """Match the legacy B15 OccWorld box-z convention.

    LiDARInstance3DBoxes expose bottom-z, while B15's frozen label builder
    rasterized the original KL ``bbox_3d`` center-z directly as a box floor.
    The conversion is deliberately explicit: correcting the label convention
    itself would require regenerated GT and a new model, whereas this adapter
    preserves B15's learned input contract for online predicted boxes.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f'boxes must be [N,7], got {boxes.shape}')
    if source_z_origin == 'occworld_legacy_center':
        return boxes.copy()
    if source_z_origin != 'bottom':
        raise ValueError(
            'source_z_origin must be bottom or occworld_legacy_center')
    converted = boxes.copy()
    converted[:, 2] += converted[:, 5] * 0.5
    return converted


def track_result_to_occworld_instances(
        result: Mapping,
        score_threshold: float = 0.1,
        track_score_threshold: float = 0.0,
        class_count: int = None,
        source_z_origin: str = 'bottom') -> Tuple[np.ndarray, list, dict]:
    """Convert one ``pts_bbox`` result to boxes and instance metadata.

    UniAD's tracker result already excludes inactive and SDC queries.  The
    adapter still filters finite, positive-size boxes and configurable score
    thresholds so it remains safe for exported or hand-built result dicts.
    ``boxes_3d`` is assumed to be in the same FLU ego frame as the LiDAR
    points consumed by :class:`MultiLidarOccLabelBuilder`. Its bottom-z is
    converted to B15's frozen legacy box-z convention by default.
    """
    if score_threshold < 0 or track_score_threshold < 0:
        raise ValueError('score thresholds must be non-negative')
    boxes = _box_tensor(result.get('boxes_3d'))
    count = boxes.shape[0]
    scores = _array(result.get('scores_3d'), dtype=np.float32).reshape(-1)
    track_scores = _array(
        result.get('track_scores'), dtype=np.float32).reshape(-1)
    labels = _array(result.get('labels_3d'), dtype=np.int64).reshape(-1)
    track_ids = _array(result.get('track_ids'), dtype=np.int64).reshape(-1)
    for name, values in (
            ('scores_3d', scores),
            ('track_scores', track_scores),
            ('labels_3d', labels),
            ('track_ids', track_ids)):
        if values.size not in (0, count):
            raise ValueError(
                f'{name} count {values.size} does not match boxes {count}')
    if scores.size == 0:
        scores = np.ones(count, dtype=np.float32)
    if track_scores.size == 0:
        track_scores = scores.copy()
    if labels.size == 0:
        labels = np.zeros(count, dtype=np.int64)
    if track_ids.size == 0:
        track_ids = np.arange(count, dtype=np.int64)

    keep = np.isfinite(boxes).all(axis=1)
    keep &= np.all(boxes[:, 3:6] > 0, axis=1)
    keep &= np.isfinite(scores) & (scores >= score_threshold)
    keep &= np.isfinite(track_scores) & (
        track_scores >= track_score_threshold)
    keep &= track_ids >= 0
    if class_count is not None:
        if class_count < 1:
            raise ValueError('class_count must be positive')
        keep &= (labels >= 0) & (labels < class_count)

    kept_boxes = track_boxes_to_occworld_z_convention(
        boxes[keep], source_z_origin=source_z_origin)
    instances = []
    for box, score, track_score, label, track_id in zip(
            kept_boxes, scores[keep], track_scores[keep], labels[keep],
            track_ids[keep]):
        instances.append({
            'bbox_3d': box,
            'bbox_3d_isvalid': True,
            'bbox_label_3d': int(label),
            'track_id': int(track_id),
            'score_3d': float(score),
            'track_score': float(track_score),
        })
    summary = {
        'input_count': int(count),
        'kept_count': int(kept_boxes.shape[0]),
        'dropped_count': int(count - kept_boxes.shape[0]),
        'score_threshold': float(score_threshold),
        'track_score_threshold': float(track_score_threshold),
        'source_z_origin': source_z_origin,
        'output_z_origin': 'occworld_legacy_center',
    }
    return kept_boxes, instances, summary
