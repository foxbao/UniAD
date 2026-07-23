import numpy as np
import pytest

from tools.data_converter.kl_occworld_track_adapter import (
    track_result_to_occworld_instances,
)


class _Boxes:
    def __init__(self, tensor):
        self.tensor = tensor


def test_track_result_adapter_filters_scores_ids_and_invalid_boxes():
    result = {
        'boxes_3d': _Boxes(np.asarray([
            [1, 2, 1, 2, 2, 1, 0, 0, 0],
            [2, 2, 1, -1, 2, 1, 0, 0, 0],
            [3, 2, 1, 2, 2, 1, 0, 0, 0],
            [4, 2, 1, 2, 2, 1, 0, 0, 0],
        ], dtype=np.float32)),
        'scores_3d': np.asarray([0.9, 0.9, 0.05, 0.9]),
        'track_scores': np.asarray([0.8, 0.8, 0.8, 0.8]),
        'labels_3d': np.asarray([1, 1, 1, 8]),
        'track_ids': np.asarray([10, 11, -1, 13]),
    }

    boxes, instances, summary = track_result_to_occworld_instances(
        result, score_threshold=0.1, class_count=5)

    assert boxes.shape == (1, 7)
    assert boxes[0, 2] == 1.5
    assert instances[0]['track_id'] == 10
    assert instances[0]['bbox_label_3d'] == 1
    assert summary == {
        'input_count': 4,
        'kept_count': 1,
        'dropped_count': 3,
        'score_threshold': 0.1,
        'track_score_threshold': 0.0,
        'source_z_origin': 'bottom',
        'output_z_origin': 'occworld_legacy_center',
    }


def test_track_result_adapter_accepts_empty_optional_fields():
    boxes, instances, summary = track_result_to_occworld_instances({})

    assert boxes.shape == (0, 7)
    assert instances == []
    assert summary['input_count'] == 0


def test_track_result_adapter_rejects_mismatched_metadata():
    result = {
        'boxes_3d': _Boxes(np.ones((2, 7), dtype=np.float32)),
        'scores_3d': np.ones(1, dtype=np.float32),
    }
    with pytest.raises(ValueError, match='does not match boxes'):
        track_result_to_occworld_instances(result)
