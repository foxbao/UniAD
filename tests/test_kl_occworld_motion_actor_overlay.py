import numpy as np

from tools.analysis_tools.audit_kl_occworld_motion_actor_overlay import (
    apply_motion_actor_arrival_overlay,
    qualification_checks,
)


def test_actor_arrival_overlay_preserves_current_and_non_arrival_voxels():
    raw = np.asarray([
        [[[0, 1, 2]]],
        [[[0, 1, 0]]],
    ], dtype=np.uint8)
    motion = np.asarray([[[[True, True, False]]]])
    stationary = np.asarray([[[[True, False, False]]]])

    candidate, arrival, preservation = apply_motion_actor_arrival_overlay(
        raw, motion, stationary)

    assert arrival.tolist() == [[[[False, True, False]]]]
    assert np.array_equal(candidate[0], raw[0])
    assert candidate[1].tolist() == [[[0, 2, 0]]]
    assert preservation == {
        'arrival_voxels': 1,
        'effective_changed_voxels': 1,
        'outside_arrival_difference_voxels': 0,
        'current_difference_voxels': 0,
    }


def test_actor_arrival_overlay_can_gate_on_raw_free_only():
    raw = np.asarray([
        [[[0, 0, 0]]],
        [[[0, 1, 2]]],
    ], dtype=np.uint8)
    motion = np.ones((1, 1, 1, 3), dtype=bool)
    stationary = np.zeros_like(motion)

    candidate, arrival, _ = apply_motion_actor_arrival_overlay(
        raw, motion, stationary, raw_class_gate=[0])

    assert arrival.tolist() == [[[[True, False, False]]]]
    assert candidate[1].tolist() == [[[2, 1, 2]]]


def _method(future_mean, future_instance, current_visible,
            visible_transition, instance_transition):
    return {
        'future': {
            'mean_iou': future_mean,
            'iou': {'instance_occupied': future_instance},
        },
        'current_visible': {'mean_iou': current_visible},
        'visible_transition': {'mean_iou': visible_transition},
        'instance_transition': {'mean_iou': instance_transition},
    }


def test_b22_qualification_enforces_raw_and_final_dynamic_guards():
    methods = {
        'raw': _method(0.80, 0.50, 0.90, 0.20, 0.30),
        'b17_final': _method(0.80, 0.501, 0.90, 0.201, 0.301),
        'actor_arrival': _method(
            0.7996, 0.502, 0.8996, 0.202, 0.302),
    }
    corrections = {'net_correct_voxels': 4}
    preservation = {
        'outside_arrival_difference_voxels': 0,
        'current_difference_voxels': 0,
    }

    result = qualification_checks(
        methods, corrections, preservation)

    assert result['qualified']
    assert all(result['checks'].values())


def test_b22_qualification_rejects_candidate_below_current_final():
    methods = {
        'raw': _method(0.80, 0.50, 0.90, 0.20, 0.30),
        'b17_final': _method(0.80, 0.51, 0.90, 0.21, 0.31),
        'actor_arrival': _method(0.80, 0.502, 0.90, 0.202, 0.302),
    }

    result = qualification_checks(
        methods, {'net_correct_voxels': 4}, {
            'outside_arrival_difference_voxels': 0,
            'current_difference_voxels': 0,
        })

    assert not result['qualified']
    assert not result['checks']['not_below_b17_final_dynamic_metrics']
