import numpy as np
import pytest

from tools.data_converter.kl_occworld_track_queue import (
    pack_track_queue_results,
    unpack_track_queue_frame,
)


def _frame(index, box_count):
    boxes = np.zeros((box_count, 7), dtype=np.float32)
    boxes[:, 2] = 1.0
    boxes[:, 3:6] = 2.0
    return {
        'boxes_3d': boxes,
        'scores_3d': np.full(box_count, 0.9, dtype=np.float32),
        'track_scores': np.full(box_count, 0.8, dtype=np.float32),
        'labels_3d': np.arange(box_count, dtype=np.int64),
        'track_ids': np.arange(index * 10, index * 10 + box_count),
        'sample_idx': index,
        'timestamp': float(index),
        'ego2global': np.eye(4),
        'scene_token': 'scene',
        'token': f'token-{index}',
    }


def test_track_queue_pack_round_trip_keeps_variable_frame_counts():
    payload = pack_track_queue_results([
        _frame(1, 2), _frame(2, 0), _frame(3, 1)
    ], score_threshold=0.1, class_count=4)

    assert payload['track_box_offsets'].tolist() == [0, 2, 2, 3]
    assert payload['queue_frame_indices'].tolist() == [1, 2, 3]
    assert payload['track_boxes_3d'].shape == (3, 7)
    boxes, instances = unpack_track_queue_frame(payload, 2)
    assert boxes.shape == (1, 7)
    assert instances[0]['track_id'] == 30
    assert instances[0]['bbox_3d'][2] == 2.0


def test_track_queue_unpack_rejects_offsets_past_flat_box_count():
    payload = pack_track_queue_results(
        [_frame(1, 1)], score_threshold=0.1, class_count=4)
    payload['track_box_offsets'][-1] = 2

    with pytest.raises(ValueError, match='track_box_offsets'):
        unpack_track_queue_frame(payload, 0)
