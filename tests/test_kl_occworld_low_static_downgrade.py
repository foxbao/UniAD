import numpy as np
import pytest

from tools.analysis_tools.evaluate_kl_occworld_low_static_downgrade import (
    apply_low_static_conflict_downgrade,
)


def _arrays():
    world = np.zeros((2, 3, 2, 3), dtype=np.uint8)
    direct = np.zeros_like(world)
    future_free = np.zeros_like(world)
    world[0, 1] = np.asarray([[2, 2, 2], [2, 2, 1]])
    direct[0, 1] = np.asarray([[2, 2, 2], [2, 0, 1]])
    future_free[0, 1] = np.asarray([[0, 1, 2], [3, 4, 5]])
    return world, direct, future_free


def test_rule_only_downgrades_direct_static_with_repeated_free():
    world, direct, future_free = _arrays()
    candidate, mask = apply_low_static_conflict_downgrade(
        world, direct, future_free, low_z_index=1,
        repeated_free_count=2)
    expected = np.asarray([[False, False, True],
                           [True, False, False]])
    np.testing.assert_array_equal(mask[0], expected)
    assert candidate[0, 0, 1] == 2
    assert candidate[0, 0, 2] == 0
    assert candidate[0, 1, 0] == 0
    assert candidate[0, 1, 1] == 2
    assert candidate[0, 1, 2] == 1
    assert np.count_nonzero(candidate[1]) == 0


def test_rule_does_not_mutate_source_arrays():
    world, direct, future_free = _arrays()
    original = world.copy()
    apply_low_static_conflict_downgrade(
        world, direct, future_free, low_z_index=1,
        repeated_free_count=2)
    np.testing.assert_array_equal(world, original)


def test_rule_rejects_single_frame_free_threshold():
    world, direct, future_free = _arrays()
    with pytest.raises(ValueError, match='at least 2'):
        apply_low_static_conflict_downgrade(
            world, direct, future_free, low_z_index=1,
            repeated_free_count=1)
