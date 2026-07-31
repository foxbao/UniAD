import numpy as np
import pytest

from tools.analysis_tools.evaluate_kl_occworld_low_static_point_height import (
    STATIC_OCCUPIED,
    UNKNOWN,
    apply_point_height_downgrade,
    point_height_downgrade_mask,
)


def test_point_height_rule_requires_low_actual_points_and_no_upper_support():
    direct_static = np.asarray([[True, True, True, False]])
    p90 = np.asarray([[0.55, 0.56, np.nan, 0.10]], dtype=np.float32)
    higher_static = np.asarray([[False, False, False, False]])
    result = point_height_downgrade_mask(
        direct_static, p90, higher_static, threshold_m=0.55)
    np.testing.assert_array_equal(
        result, np.asarray([[True, False, False, False]]))


def test_point_height_rule_preserves_vertical_static_support():
    direct_static = np.asarray([[True, True]])
    p90 = np.asarray([[0.20, 0.20]], dtype=np.float32)
    higher_static = np.asarray([[True, False]])
    result = point_height_downgrade_mask(
        direct_static, p90, higher_static, threshold_m=0.55)
    np.testing.assert_array_equal(result, np.asarray([[False, True]]))


def test_application_changes_only_direct_world_static_to_unknown():
    world = np.asarray([[STATIC_OCCUPIED, STATIC_OCCUPIED, 1, 0]], dtype=np.uint8)
    direct = np.asarray([[STATIC_OCCUPIED, 0, 1, 0]], dtype=np.uint8)
    mask = np.asarray([[True, True, True, True]])
    candidate = apply_point_height_downgrade(world, direct, mask)
    np.testing.assert_array_equal(
        candidate, np.asarray([[UNKNOWN, STATIC_OCCUPIED, 1, 0]], dtype=np.uint8))


def test_point_height_rule_rejects_shape_mismatch():
    with pytest.raises(ValueError, match='share one shape'):
        point_height_downgrade_mask(
            np.zeros((1, 2), dtype=bool), np.zeros((2, 1)),
            np.zeros((1, 2), dtype=bool), 0.55)
