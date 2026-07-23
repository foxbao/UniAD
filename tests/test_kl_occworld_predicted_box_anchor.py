import numpy as np

from tools.analysis_tools.audit_kl_occworld_predicted_box_anchor import (
    _overlap_metrics,
)


def test_predicted_box_anchor_overlap_reports_voxel_precision_recall():
    reference = np.asarray([[[3, 3, 0, 1]]], dtype=np.uint8)
    predicted = np.asarray([[[3, 0, 3, 1]]], dtype=np.uint8)

    metrics = _overlap_metrics(reference, predicted)

    assert metrics['reference_instance_voxels'] == 2
    assert metrics['predicted_instance_voxels'] == 2
    assert metrics['intersection_voxels'] == 1
    assert metrics['union_voxels'] == 3
    assert metrics['instance_iou'] == 1 / 3
    assert metrics['instance_precision'] == 0.5
    assert metrics['instance_recall'] == 0.5
    assert metrics['state_mismatch_voxels'] == 2
