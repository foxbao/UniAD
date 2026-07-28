import numpy as np
import pytest

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    RaycastDrivableBuilder,
)
from tools.analysis_tools.audit_kl_occworld_motion_actor_support import (
    align_motion_steps,
    motion_actor_support,
    summarize_counts,
    support_correction_counts,
    support_counts,
)


def test_align_motion_steps_uses_nominal_half_second_horizons():
    indices = align_motion_steps(
        np.arange(1, 5, dtype=np.float32) * 0.5,
        np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32))

    assert indices == [0, 1, 2, 3]


def test_align_motion_steps_rejects_unmatched_time_contract():
    with pytest.raises(ValueError, match='No motion step matches'):
        align_motion_steps(
            np.asarray([0.5, 1.0], dtype=np.float32),
            np.asarray([0.75], dtype=np.float32),
            tolerance_s=0.05)


def test_motion_actor_support_moves_full_3d_box_and_filters_score():
    builder = RaycastDrivableBuilder(
        pc_range=[0.0, 0.0, 0.0, 4.0, 4.0, 2.0],
        bev_size=[4, 4], occ_size=[4, 4, 2])
    future = np.asarray([
        [[0.5, 0.5], [1.5, 0.5]],
        [[3.5, 3.5], [3.5, 2.5]],
    ], dtype=np.float32)
    payload = {
        'motion_actor_future_xy': future,
        'motion_actor_boxes_3d': np.asarray([
            [0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0],
            [3.5, 3.5, 0.0, 1.0, 1.0, 1.0, 0.0],
        ], dtype=np.float32),
        'motion_actor_scores': np.asarray([0.8, 0.05], dtype=np.float32),
        'motion_actor_valid': np.asarray([True, True]),
        'motion_actor_step_times_s': np.asarray([0.5, 1.0]),
        'motion_actor_box_z_origin': np.asarray('bottom'),
    }

    support, metadata = motion_actor_support(
        payload, builder, target_times=np.asarray([0.5, 1.0]),
        score_threshold=0.1)

    assert support.shape == (2, 2, 4, 4)
    assert metadata['kept_actor_count'] == 1
    assert metadata['motion_step_indices'] == [0, 1]
    assert np.count_nonzero(support[0]) > 0
    assert not np.array_equal(support[0], support[1])


def test_support_counts_reports_visible_arrival_coverage():
    support = np.asarray([[[[False, True, False]]]])
    target_state = np.asarray([[[[1, 3, 3]]]], dtype=np.uint8)
    target_valid = np.ones_like(target_state, dtype=bool)
    current_state = np.asarray([[[1, 1, 1]]], dtype=np.uint8)
    current_valid = np.ones_like(current_state, dtype=bool)

    metrics = summarize_counts(support_counts(
        support, target_state, target_valid,
        current_state, current_valid))

    assert metrics['known_support_voxels'] == 1
    assert metrics['target_instance_voxels'] == 2
    assert metrics['instance_precision_known'] == 1.0
    assert metrics['instance_recall'] == 0.5
    assert metrics['instance_iou_known'] == 0.5
    assert metrics['arrival_recall'] == 0.5


def test_support_correction_counts_pairs_motion_with_stationary_boxes():
    stationary = np.asarray([[[[True, False, False]]]])
    motion = np.asarray([[[[False, True, False]]]])
    target_state = np.asarray([[[[1, 3, 1]]]], dtype=np.uint8)
    target_valid = np.ones_like(target_state, dtype=bool)

    counts = support_correction_counts(
        motion, stationary, target_state, target_valid)

    assert counts == {
        'changed_known_voxels': 2,
        'improved_voxels': 2,
        'harmed_voxels': 0,
        'net_correct_voxels': 2,
    }
