import json

import numpy as np

from tools.analysis_tools.audit_kl_occworld_predicted_box_anchor import (
    _overlap_metrics,
    _select_reference_shard,
)
from tools.analysis_tools.merge_kl_occworld_anchor_audits import (
    merge_reports,
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


def test_anchor_audit_reference_shards_are_disjoint_and_complete():
    references = list(range(11))
    shards = [
        _select_reference_shard(references, 3, shard_index)
        for shard_index in range(3)
    ]

    assert shards == [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8]]
    assert sorted(value for shard in shards for value in shard) == references


def test_anchor_audit_merger_recomputes_global_overlap(tmp_path):
    paths = []
    for shard_index, row in enumerate((
            {
                'reference_index': 1,
                'reference_instance_voxels': 4,
                'predicted_instance_voxels': 3,
                'intersection_voxels': 2,
                'union_voxels': 5,
                'state_mismatch_voxels': 1,
                'state_mismatch_ratio': 0.1,
            },
            {
                'reference_index': 2,
                'reference_instance_voxels': 6,
                'predicted_instance_voxels': 5,
                'intersection_voxels': 3,
                'union_voxels': 8,
                'state_mismatch_voxels': 2,
                'state_mismatch_ratio': 0.2,
            })):
        path = tmp_path / f'shard_{shard_index}.json'
        path.write_text(json.dumps({
            'box_source': 'predicted',
            'rows': [row],
        }))
        paths.append(path)

    report = merge_reports(paths)

    assert report['reference_count'] == 2
    assert report['aggregate']['instance_iou'] == 5 / 13
    assert report['aggregate']['instance_precision'] == 5 / 8
    assert report['aggregate']['instance_recall'] == 0.5
    assert np.isclose(
        report['aggregate']['mean_state_mismatch_ratio'], 0.15)
