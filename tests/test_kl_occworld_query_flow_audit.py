import numpy as np

from tools.analysis_tools.audit_kl_occworld_query_flow_overlay import (
    event_metrics,
    qualifying_query_thresholds,
    semantic_metrics,
)


def test_event_metrics_counts_only_evaluable_voxels():
    prediction = np.asarray([True, True, True, False])
    target = np.asarray([True, False, True, True])
    evaluable = np.asarray([True, True, False, True])

    metrics = event_metrics(prediction, target, evaluable)

    assert metrics['predicted_count'] == 2
    assert metrics['target_count'] == 2
    assert metrics['true_positive'] == 1
    assert metrics['false_positive'] == 1
    assert metrics['false_negative'] == 1
    assert metrics['f1'] == 0.5


def test_semantic_metrics_uses_target_row_prediction_column_confusion():
    confusion = np.asarray([
        [4, 1, 0],
        [1, 3, 0],
        [0, 0, 2],
    ], dtype=np.int64)

    metrics = semantic_metrics(confusion)

    assert metrics['voxel_count'] == 11
    assert metrics['iou']['free'] == 4 / 6
    assert metrics['iou']['static_occupied'] == 3 / 5
    assert metrics['iou']['instance_occupied'] == 1.0


def test_query_gate_must_improve_corrections_and_preserve_transitions():
    result = {
        'query_thresholds': [0.1, 0.3],
        'methods': {
            'recreated_ungated': {
                'visible_transition': {'mean_iou': 0.2},
                'instance_transition': {'mean_iou': 0.3},
            },
            'query_0.1': {
                'visible_transition': {'mean_iou': 0.21},
                'instance_transition': {'mean_iou': 0.31},
            },
            'query_0.3': {
                'visible_transition': {'mean_iou': 0.19},
                'instance_transition': {'mean_iou': 0.31},
            },
        },
        'corrections_vs_raw': {
            'query_0.1': {'net_correct_voxels': 1},
            'query_0.3': {'net_correct_voxels': 2},
        },
    }

    assert qualifying_query_thresholds(result) == [0.1]
