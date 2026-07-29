import numpy as np

from tools.analysis_tools.audit_kl_occworld_low_static_spatial_expansion import (
    _crop_bounds,
    _distance_from_mask,
    _front_left_component,
)


def test_front_left_component_selects_largest_candidate():
    state = np.zeros((6, 8), dtype=np.uint8)
    state[1:3, 5:7] = 2
    state[4:, 5:] = 2
    x_centers = np.arange(8, dtype=np.float32) - 3.5
    row_y_centers = np.asarray([3, 2, 1, 0, -1, -2], dtype=np.float32)
    component = _front_left_component(
        state, x_centers, row_y_centers)
    assert component.sum() == 4
    assert component[1:3, 5:7].all()


def test_crop_bounds_preserve_large_extent_and_expand_small_extent():
    large = np.zeros((120, 160), dtype=bool)
    large[5:110, 20:80] = True
    assert _crop_bounds(large, margin=6) == (0, 116, 14, 86)
    small = np.zeros((120, 160), dtype=bool)
    small[50:55, 70:75] = True
    row_min, row_max, col_min, col_max = _crop_bounds(small, margin=6)
    assert row_max - row_min >= 40
    assert col_max - col_min >= 40


def test_distance_from_mask_is_zero_on_seed_and_positive_outside():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    distance = _distance_from_mask(mask)
    assert distance[2, 2] == 0
    assert distance[2, 3] > 0
    assert distance[0, 0] > distance[2, 3]
