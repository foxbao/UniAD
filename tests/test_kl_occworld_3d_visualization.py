import numpy as np

from tools.analysis_tools.visualize_kl_occworld_3d import (
    _free_footprint_xyz,
    _surface_mask,
    _zhw_mask_to_xyz,
)


def test_3d_visualization_converts_image_aligned_voxel_to_xyz():
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[1, 0, 2] = True

    points = _zhw_mask_to_xyz(
        mask,
        pc_range=np.asarray([0, 0, 0, 4, 3, 2], dtype=np.float32),
        occ_size=np.asarray([4, 3, 2], dtype=np.int64))

    np.testing.assert_allclose(points, [[2.5, 2.5, 1.5]])


def test_3d_visualization_removes_only_fully_enclosed_voxels():
    mask = np.ones((3, 3, 3), dtype=bool)

    surface = _surface_mask(mask)

    assert surface.sum() == 26
    assert not surface[1, 1, 1]


def test_3d_visualization_free_footprint_excludes_occupied_columns():
    state = np.zeros((2, 2, 3), dtype=np.uint8)
    state[0, 0, 0] = 1
    state[0, 0, 1] = 1
    state[1, 0, 1] = 2

    points = _free_footprint_xyz(
        state,
        pc_range=np.asarray([0, 0, 0, 3, 2, 2], dtype=np.float32),
        occ_size=np.asarray([3, 2, 2], dtype=np.int64),
        surface_z=0.25)

    np.testing.assert_allclose(points, [[0.5, 1.5, 0.25]])
