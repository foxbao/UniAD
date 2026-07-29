import numpy as np

from tools.analysis_tools.audit_kl_occworld_low_static_stability import (
    _connected_component_rows,
    _pose_delta_metrics,
    _transition_counts,
    _with_rates,
)


def test_transition_counts_and_rates():
    source = np.asarray([
        [True, True, False],
        [True, True, True],
    ])
    target = np.asarray([
        [0, 1, 2],
        [2, 3, 2],
    ], dtype=np.uint8)
    result = _with_rates(_transition_counts(source, target))
    assert result['source_static_voxels'] == 5
    assert result['next_unknown_voxels'] == 1
    assert result['next_free_voxels'] == 1
    assert result['next_static_voxels'] == 2
    assert result['next_instance_voxels'] == 1
    assert result['next_nonstatic_voxels'] == 3
    assert result['next_free_fraction'] == 0.2
    assert result['next_nonstatic_fraction'] == 0.6


def test_connected_components_keep_physical_bounds_and_sources():
    source = np.zeros((5, 6), dtype=bool)
    source[1:3, 2:5] = True
    target = np.zeros_like(source, dtype=np.uint8)
    target[source] = 1
    target[1, 2] = 2
    completion = np.zeros_like(target)
    completion[source] = 1
    completion[2, 4] = 3
    x_centers = np.arange(6, dtype=np.float64) + 0.5
    row_y_centers = np.asarray([2.0, 1.0, 0.0, -1.0, -2.0])
    rows = _connected_component_rows(
        source, target, completion, x_centers, row_y_centers,
        reference=7, source_horizon=0, min_cells=2)
    assert len(rows) == 1
    row = rows[0]
    assert row['reference_index'] == 7
    assert row['area_cells'] == 6
    assert row['bbox_x_m'] == [2.5, 4.5]
    assert row['bbox_y_m'] == [0.0, 1.0]
    assert row['source_direct_cells'] == 5
    assert row['source_future_static_fill_cells'] == 1
    assert row['next_free_voxels'] == 5
    assert row['next_static_voxels'] == 1


def test_pose_delta_reports_tilt_and_translation():
    first = np.eye(4)
    angle = np.deg2rad(2.0)
    second = np.eye(4)
    second[:3, :3] = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
    ])
    second[:3, 3] = (3.0, 4.0, 0.2)
    result = _pose_delta_metrics(first, second)
    assert np.isclose(result['roll_deg'], 2.0)
    assert np.isclose(result['pitch_deg'], 0.0)
    assert np.isclose(result['tilt_deg'], 2.0)
    assert np.isclose(result['translation_xy_m'], 5.0)
    assert np.isclose(result['translation_z_m'], 0.2)
