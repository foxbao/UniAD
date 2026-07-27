import numpy as np

from tools.analysis_tools.select_kl_occworld_query_residual import (
    acceptance_checks,
    correction_counts,
)


def test_correction_counts_reports_net_gain_on_changed_valid_voxels():
    baseline = np.asarray([0, 0, 1, 2, 2], dtype=np.uint8)
    candidate = np.asarray([0, 1, 1, 1, 0], dtype=np.uint8)
    target = np.asarray([0, 1, 1, 2, 0], dtype=np.uint8)
    valid = np.asarray([True, True, True, True, False])

    result = correction_counts(candidate, baseline, target, valid)

    assert result == {
        'changed_voxels': 2,
        'improved_voxels': 1,
        'harmed_voxels': 1,
        'net_correct_voxels': 0,
    }


def test_acceptance_requires_target_gain_transition_safety_and_parity():
    result = {
        'external_b17a_replay_parity': {
            'semantic_difference_voxels_vs_b17a': 0,
            'visibility_max_abs_difference_vs_b17a': 0.0,
        },
        'corrections_vs_scale_zero': {'net_correct_voxels': 3},
        'delta_percentage_points': {
            'future_instance_iou': 0.11,
            'visible_transition_miou': 0.0,
            'instance_transition_miou': 0.01,
            'future_semantic_miou': -0.05,
            'current_visible_miou': -0.04,
        },
    }

    checks = acceptance_checks(result)

    assert checks['all_passed']
    result['delta_percentage_points']['visible_transition_miou'] = -0.001
    assert not acceptance_checks(result)['all_passed']
