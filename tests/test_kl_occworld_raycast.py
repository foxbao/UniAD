import numpy as np

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    RaycastDrivableBuilder,
    visible_ground_recovery_mask,
)
from tools.data_converter.generate_kl_occworld_labels import (
    FREE,
    STATIC_OCCUPIED,
    ENDPOINT_GROUND,
    ENDPOINT_INSTANCE,
    ENDPOINT_RELIABLE_STATIC,
    ENDPOINT_UNCERTAIN_OBSTACLE,
    _classify_endpoint_evidence_xyz,
    _compose_filtered_state_xyz,
    _compose_observed_state_xyz,
    _project_observed_state,
    _xyz_to_zhw,
)
from tools.data_converter.generate_kl_occworld_temporal_labels import (
    _promote_uncertain,
    _warp_mask_to_reference_xyz,
)
from tools.analysis_tools.audit_kl_occworld_occlusion import (
    _ray_blocked_by_occupied,
)
from tools.analysis_tools.audit_kl_occworld_future_reveal import (
    REVEAL_DYNAMIC_OCCUPIED,
    REVEAL_REPEATED_FREE,
    REVEAL_SINGLE_FREE,
    REVEAL_STATIC_OCCUPIED,
    REVEAL_UNKNOWN,
    _classify_reveal,
)
from tools.analysis_tools.audit_kl_occworld_dual_representation import (
    BLOCKED,
    DRIVABLE,
    NAVIGABILITY_UNKNOWN,
    NAVIGABLE,
    NON_DRIVABLE,
    TRAVERSABILITY_UNKNOWN,
    _compose_navigability,
    _compose_occupancy_3d,
    _compose_traversability,
    _filter_small_components,
    _opposing_obstacle_surface,
)
from tools.analysis_tools.audit_kl_occworld_dual_cross_scene import (
    _bev_mask_to_global_xy,
    _cross_frame_support,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    _compose_world_target,
    _validate_frame_offsets,
    _validate_fixed_frame_times,
    _warp_state_to_reference,
)
from tools.data_converter.generate_kl_occworld_observation_cache import (
    _expand_indices,
)
from tools.data_converter.generate_kl_occworld_sequence_batch import (
    _dual_label_index,
    _reference_indices,
)
from tools.analysis_tools.audit_kl_occworld_sequence_batch import (
    _contact_page,
)


def _builder():
    return RaycastDrivableBuilder(
        pc_range=[-4, -4, -2, 4, 4, 2],
        bev_size=[8, 8],
        occ_size=[8, 8, 4],
        fill_ground=False,
        obstacle_min_component_voxels=1,
        ego_ignore_range=None,
    )


def test_visible_ground_recovery_requires_visible_local_component():
    base = np.zeros((7, 7), dtype=np.uint8)
    base[3, 3] = 1
    relaxed = np.zeros_like(base)
    relaxed[3, 4] = 1
    relaxed[2, 4] = 1
    relaxed[1, 1:3] = 1
    free = np.ones_like(base)
    blocked = np.zeros_like(base)
    blocked[3, 2] = 1

    recovered = visible_ground_recovery_mask(
        base, relaxed, free, blocked, max_distance=1.5,
        min_component_cells=2)

    # The blocked neighbour and the isolated two-cell component must not be
    # restored, while the visible, connected cell on the other side is kept.
    assert recovered[3, 2] == 0
    assert recovered[3, 4] == 1
    assert recovered[2, 4] == 1
    assert recovered[1, 1] == 0
    assert recovered[1, 2] == 0


def test_visible_ground_recovery_geodesic_does_not_cross_candidate_gap():
    base = np.zeros((7, 7), dtype=np.uint8)
    base[3, 1] = 1
    relaxed = np.zeros_like(base)
    relaxed[3, 2] = 1
    relaxed[3, 4] = 1
    free = np.ones_like(base)
    blocked = np.zeros_like(base)

    euclidean = visible_ground_recovery_mask(
        base, relaxed, free, blocked, max_distance=4.0,
        min_component_cells=1, distance_mode='euclidean')
    geodesic = visible_ground_recovery_mask(
        base, relaxed, free, blocked, max_distance=4.0,
        min_component_cells=1, distance_mode='geodesic')

    assert euclidean[3, 2] == 1
    assert euclidean[3, 4] == 1
    assert geodesic[3, 2] == 1
    # Cell (3, 3) is not a valid candidate, so geodesic cannot jump to 4.
    assert geodesic[3, 4] == 0


def test_explicit_zero_origin_preserves_legacy_output():
    builder = _builder()
    points = np.asarray([
        [2.0, 0.0, 0.0, 1.0],
        [-2.0, 1.0, 0.0, 1.0],
        [1.0, 2.0, 1.0, 1.0],
    ], dtype=np.float32)
    boxes = np.empty((0, 7), dtype=np.float32)

    legacy = builder.build(points, boxes)
    explicit = builder.build(
        points, boxes, ray_origin=np.zeros(3, dtype=np.float32))

    assert legacy.keys() == explicit.keys()
    for key in legacy:
        np.testing.assert_array_equal(legacy[key], explicit[key])


def test_precomputed_sensor_free_voxels_are_reused():
    builder = _builder()
    points = np.asarray([
        [2.0, 0.0, 0.0, 1.0],
        [-2.0, 1.0, 0.0, 1.0],
    ], dtype=np.float32)
    boxes = np.empty((0, 7), dtype=np.float32)
    hit_voxels = builder.coord_to_index_floor(points[:, :3])
    sensor_origin = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    free_voxels = builder.raycast_free_voxels(
        hit_voxels, ray_origin=sensor_origin)

    result = builder.build(
        points, boxes, free_voxels=free_voxels)

    expected_free = builder.voxels_to_bev(free_voxels)
    np.testing.assert_array_equal(result['free'], expected_free)


def test_optional_voxel_evidence_does_not_change_bev_result():
    builder = _builder()
    points = np.asarray([
        [2.0, 0.0, 0.0, 1.0],
        [-2.0, 1.0, 0.0, 1.0],
    ], dtype=np.float32)
    boxes = np.empty((0, 7), dtype=np.float32)

    bev_only = builder.build(points, boxes)
    with_voxels = builder.build(points, boxes, return_voxels=True)

    for key, value in bev_only.items():
        np.testing.assert_array_equal(with_voxels[key], value)
    for key in ['ground_voxels', 'obstacle_voxels',
                'raw_obstacle_voxels', 'semantic_voxels', 'free_voxels']:
        assert with_voxels[key].ndim == 2
        assert with_voxels[key].shape[1] == 3


def test_endpoint_point_percentile_avoids_voxel_center_ground_error():
    common = dict(
        pc_range=[0, 0, -1, 2, 2, 1],
        bev_size=[2, 2],
        occ_size=[2, 2, 2],
        ground_height_threshold=0.10,
        ground_smooth_radius=0,
        fill_ground=False,
        obstacle_min_component_voxels=1,
        ego_ignore_range=None,
    )
    points = np.asarray([
        [0.2, 0.2, -0.65, 1.0],
        [0.3, 0.3, -0.62, 1.0],
    ], dtype=np.float32)
    boxes = np.empty((0, 7), dtype=np.float32)
    voxel = np.asarray([[0, 0, 0]], dtype=np.int64)

    center_builder = RaycastDrivableBuilder(**common)
    center_result = center_builder.build(points, boxes, return_voxels=True)
    assert np.any(np.all(center_result['raw_obstacle_voxels'] == voxel,
                         axis=1))

    point_builder = RaycastDrivableBuilder(
        **common, ground_endpoint_height_mode='point_percentile',
        ground_endpoint_point_percentile=0.90)
    point_result = point_builder.build(points, boxes, return_voxels=True)
    assert np.any(np.all(point_result['ground_voxels'] == voxel, axis=1))
    assert point_result['raw_obstacle_voxels'].shape == (0, 3)

    rescue_builder = RaycastDrivableBuilder(
        **common, ground_endpoint_height_mode='point_percentile_rescue',
        ground_endpoint_point_percentile=0.90)
    rescue_result = rescue_builder.build(points, boxes, return_voxels=True)
    assert np.any(np.all(rescue_result['ground_voxels'] == voxel, axis=1))


def test_endpoint_point_percentile_rescue_does_not_add_obstacles():
    common = dict(
        pc_range=[0, 0, -1, 2, 2, 1],
        bev_size=[2, 2],
        occ_size=[2, 2, 2],
        ground_height_threshold=0.55,
        ground_smooth_radius=0,
        fill_ground=False,
        obstacle_min_component_voxels=1,
        ego_ignore_range=None,
    )
    points = np.asarray([
        [0.2, 0.2, -0.95, 1.0],
        [0.3, 0.3, -0.05, 1.0],
    ], dtype=np.float32)
    boxes = np.empty((0, 7), dtype=np.float32)

    point_result = RaycastDrivableBuilder(
        **common, ground_endpoint_height_mode='point_percentile').build(
            points, boxes, return_voxels=True)
    assert point_result['raw_obstacle_voxels'].shape == (1, 3)

    rescue_result = RaycastDrivableBuilder(
        **common, ground_endpoint_height_mode='point_percentile_rescue').build(
            points, boxes, return_voxels=True)
    assert rescue_result['raw_obstacle_voxels'].shape == (0, 3)
    assert rescue_result['ground_voxels'].shape == (1, 3)


def test_endpoint_point_percentile_rescue_keeps_vertical_structure_base():
    builder = RaycastDrivableBuilder(
        pc_range=[0, 0, -1, 2, 2, 1],
        bev_size=[2, 2],
        occ_size=[2, 2, 2],
        ground_height_threshold=0.10,
        ground_smooth_radius=0,
        fill_ground=False,
        obstacle_min_component_voxels=1,
        ego_ignore_range=None,
        ground_endpoint_height_mode='point_percentile_rescue',
        ground_endpoint_point_percentile=0.90,
    )
    points = np.asarray([
        [0.2, 0.2, -0.65, 1.0],
        [0.3, 0.3, -0.62, 1.0],
        [0.2, 0.2, 0.70, 1.0],
    ], dtype=np.float32)
    result = builder.build(
        points, np.empty((0, 7), dtype=np.float32), return_voxels=True)
    low_voxel = np.asarray([[0, 0, 0]], dtype=np.int64)
    assert np.any(np.all(result['raw_obstacle_voxels'] == low_voxel, axis=1))


def test_endpoint_rescue_does_not_cascade_through_component_filtering():
    """Only the verified P90 endpoint may leave a valid old component."""
    builder = RaycastDrivableBuilder(
        pc_range=[0, 0, -1, 3, 1, 1],
        bev_size=[1, 3],
        occ_size=[3, 1, 2],
        ground_height_threshold=0.10,
        ground_smooth_radius=0,
        fill_ground=False,
        obstacle_min_points_per_voxel=1,
        obstacle_min_component_voxels=3,
        ego_ignore_range=None,
        ground_endpoint_height_mode='point_percentile_rescue',
        ground_endpoint_point_percentile=0.90,
    )
    # All endpoints lie in the same z voxel.  The first is ground-like by
    # its actual point height; the other two are genuine high returns.  The
    # historical three-voxel component passes the component-size filter.
    points = np.asarray([
        [0.1, 0.2, -0.95],
        [1.1, 0.2, -0.10],
        [2.1, 0.2, -0.10],
    ], dtype=np.float32)
    voxels = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int64)
    builder.estimate_ground_height = lambda _points, _voxels: (
        np.full((3, 1), -1.0, dtype=np.float32),
        np.ones((3, 1), dtype=bool))

    ground, obstacle, raw_obstacle, historical_obstacle = (
        builder.split_scene_ground_obstacle(
        points, voxels, voxels, np.empty((0, 3), dtype=np.int64))
    )

    np.testing.assert_array_equal(ground, np.asarray([[0, 0, 0]]))
    np.testing.assert_array_equal(
        raw_obstacle, np.asarray([[1, 0, 0], [2, 0, 0]]))
    np.testing.assert_array_equal(obstacle, raw_obstacle)
    np.testing.assert_array_equal(historical_obstacle, np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]]))

    result = builder.build(
        points, np.empty((0, 7), dtype=np.float32), return_voxels=True)
    np.testing.assert_array_equal(result['raw_obstacle_voxels'], raw_obstacle)
    np.testing.assert_array_equal(result['obstacle_voxels'], raw_obstacle)


def test_xyz_to_zhw_keeps_z_and_reverses_bev_rows():
    volume_xyz = np.zeros((2, 3, 2), dtype=np.uint8)
    volume_xyz[1, 2, 0] = 7
    volume_xyz[0, 0, 1] = 9

    volume_zhw = _xyz_to_zhw(volume_xyz)

    assert volume_zhw.shape == (2, 3, 2)
    assert volume_zhw[0, 0, 1] == 7
    assert volume_zhw[1, 2, 0] == 9


def test_observed_target_uses_visibility_and_occupied_precedence():
    free_count = np.zeros((2, 2, 2), dtype=np.uint8)
    occupied_count = np.zeros_like(free_count)
    free_count[0, 0, 0] = 2
    free_count[1, 1, 1] = 1
    occupied_count[0, 0, 0] = 1

    state, visibility, occupancy = _compose_observed_state_xyz(
        free_count, occupied_count)

    assert state[0, 0, 0] == STATIC_OCCUPIED
    assert state[1, 1, 1] == FREE
    assert state[0, 1, 1] == 0
    assert visibility[0, 0, 0] == 1
    assert visibility[0, 1, 1] == 0
    assert occupancy[0, 0, 0] == 1
    assert occupancy[1, 1, 1] == 0


def test_filtered_target_ignores_rejected_endpoint_instead_of_marking_free():
    free_count = np.ones((3, 1, 1), dtype=np.uint8)
    occupied_count = np.zeros_like(free_count)
    occupied_count[1, 0, 0] = 1
    occupied_count[2, 0, 0] = 1
    reliable_occupied = np.zeros_like(free_count)
    reliable_occupied[2, 0, 0] = 1

    state, visibility, occupancy, rejected = _compose_filtered_state_xyz(
        free_count, occupied_count, reliable_occupied)

    assert state[:, 0, 0].tolist() == [FREE, 0, STATIC_OCCUPIED]
    assert visibility[:, 0, 0].tolist() == [1, 0, 1]
    assert occupancy[:, 0, 0].tolist() == [0, 0, 1]
    assert rejected[:, 0, 0].tolist() == [0, 1, 0]


def test_endpoint_evidence_keeps_uncertain_obstacle_separate_from_ground():
    occupied = np.ones((4, 1, 1), dtype=np.uint8)
    raw_obstacle = np.zeros_like(occupied)
    raw_obstacle[1:3] = 1
    reliable_static = np.zeros_like(occupied)
    reliable_static[2] = 1
    instance_endpoint = np.zeros_like(occupied)
    instance_endpoint[3] = 1

    ground, uncertain, endpoint_type = _classify_endpoint_evidence_xyz(
        occupied, raw_obstacle, reliable_static, instance_endpoint)

    assert ground[:, 0, 0].tolist() == [1, 0, 0, 0]
    assert uncertain[:, 0, 0].tolist() == [0, 1, 0, 0]
    assert endpoint_type[:, 0, 0].tolist() == [
        ENDPOINT_GROUND,
        ENDPOINT_UNCERTAIN_OBSTACLE,
        ENDPOINT_RELIABLE_STATIC,
        ENDPOINT_INSTANCE,
    ]


def test_temporal_identity_warp_preserves_voxel_location():
    mask_zhw = np.zeros((4, 4, 4), dtype=bool)
    mask_zhw[2, 1, 3] = True
    identity = np.eye(4, dtype=np.float64)

    warped_xyz = _warp_mask_to_reference_xyz(
        mask_zhw, identity, identity,
        pc_range=[-2, -2, -2, 2, 2, 2],
        occ_size=[4, 4, 4])

    assert warped_xyz[3, 2, 2]
    assert int(warped_xyz.sum()) == 1


def test_temporal_warp_uses_source_to_reference_transform_direction():
    mask_zhw = np.zeros((4, 4, 4), dtype=bool)
    mask_zhw[2, 1, 1] = True
    source_ego2global = np.eye(4, dtype=np.float64)
    source_ego2global[0, 3] = 1.0
    reference_ego2global = np.eye(4, dtype=np.float64)

    warped_xyz = _warp_mask_to_reference_xyz(
        mask_zhw, source_ego2global, reference_ego2global,
        pc_range=[-2, -2, -2, 2, 2, 2],
        occ_size=[4, 4, 4])

    assert warped_xyz[2, 2, 2]
    assert int(warped_xyz.sum()) == 1


def test_temporal_promotion_requires_current_uncertain_and_support():
    endpoint_type = np.zeros((2, 2, 2), dtype=np.uint8)
    endpoint_type[0, 0, 0] = ENDPOINT_UNCERTAIN_OBSTACLE
    endpoint_type[0, 0, 1] = ENDPOINT_RELIABLE_STATIC
    support = np.zeros_like(endpoint_type)
    support[0, 0, 0] = 2
    support[0, 0, 1] = 2

    promoted = _promote_uncertain(endpoint_type, support, 2)

    assert promoted[0, 0, 0]
    assert not promoted[0, 0, 1]


def test_occlusion_audit_detects_occupied_voxel_before_free_target():
    occupied = np.zeros((4, 4, 4), dtype=bool)
    occupied[2, 2, 2] = True
    target = np.asarray([3, 2, 2])
    origin = np.asarray([-1.5, 0.5, 0.5])
    pc_range = [-2, -2, -2, 2, 2, 2]

    assert _ray_blocked_by_occupied(
        target, origin, occupied, pc_range)

    occupied[2, 2, 2] = False
    assert not _ray_blocked_by_occupied(
        target, origin, occupied, pc_range)


def test_future_reveal_uses_occupied_precedence_and_repeated_free():
    removed = np.ones((1, 5), dtype=bool)
    static_count = np.zeros((1, 5), dtype=np.uint8)
    dynamic_count = np.zeros_like(static_count)
    free_count = np.asarray([[0, 1, 2, 2, 2]], dtype=np.uint8)
    dynamic_count[0, 3] = 1
    static_count[0, 4] = 1

    reveal = _classify_reveal(
        static_count, dynamic_count, free_count, removed)

    assert reveal.tolist() == [[
        REVEAL_UNKNOWN,
        REVEAL_SINGLE_FREE,
        REVEAL_REPEATED_FREE,
        REVEAL_DYNAMIC_OCCUPIED,
        REVEAL_STATIC_OCCUPIED,
    ]]


def test_observed_3d_projection_has_occupied_precedence():
    observed = np.zeros((3, 2, 2), dtype=np.uint8)
    observed[1, 0, 0] = FREE
    observed[2, 0, 0] = STATIC_OCCUPIED
    observed[1, 1, 1] = FREE
    instance = np.zeros((2, 2), dtype=np.uint8)

    projected = _project_observed_state(
        observed, np.asarray([-0.5, 0.5, 1.5]), [0.3, 2.0], instance)

    assert projected[0, 0] == STATIC_OCCUPIED
    assert projected[1, 1] == FREE


def test_projection_does_not_mark_unresolved_obstacle_column_free():
    observed = np.zeros((2, 1, 2), dtype=np.uint8)
    observed[0, 0, :] = FREE
    observed[1, 0, 1] = STATIC_OCCUPIED
    blocking = np.zeros_like(observed)
    blocking[1, 0, 0] = 1
    instance = np.zeros((1, 2), dtype=np.uint8)

    projected = _project_observed_state(
        observed, np.asarray([0.5, 1.5]), [0.3, 2.0], instance,
        blocking_unknown_3d=blocking)

    assert projected[0, 0] == 0
    assert projected[0, 1] == STATIC_OCCUPIED


def test_dual_occupancy_requires_unblocked_free_and_keeps_instance():
    temporal = np.asarray([[[FREE, FREE, STATIC_OCCUPIED]]], dtype=np.uint8)
    boxes = np.zeros_like(temporal)
    boxes[0, 0, 2] = 1
    valid_free = np.asarray([[[1, 0, 0]]], dtype=np.uint8)

    state, visibility = _compose_occupancy_3d(
        temporal, boxes, valid_free)

    assert state.tolist() == [[[FREE, 0, 3]]]
    assert visibility.tolist() == [[[1, 0, 1]]]


def test_traversability_and_navigability_stay_separate():
    map_drivable = np.asarray([[1, 1, 0, 0]], dtype=np.uint8)
    ground = np.asarray([[0, 1, 1, 0]], dtype=np.uint8)
    observed = np.asarray([[1, 1, 1, 0]], dtype=np.uint8)
    non_drivable = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)

    traversability, valid = _compose_traversability(
        map_drivable, ground, observed, non_drivable)

    assert traversability.tolist() == [[
        DRIVABLE, DRIVABLE, NON_DRIVABLE, TRAVERSABILITY_UNKNOWN]]
    assert valid.tolist() == [[1, 1, 1, 0]]

    occupancy_bev = np.asarray(
        [[FREE, STATIC_OCCUPIED, FREE, FREE]], dtype=np.uint8)
    navigability, nav_valid = _compose_navigability(
        traversability, occupancy_bev)

    assert navigability.tolist() == [[
        NAVIGABLE, BLOCKED, BLOCKED, NAVIGABILITY_UNKNOWN]]
    assert nav_valid.tolist() == [[1, 1, 1, 0]]


def test_opposing_obstacles_mark_only_the_surface_between_them():
    state = np.zeros((5, 5), dtype=np.uint8)
    state[1, 1:4] = STATIC_OCCUPIED
    state[2, 1:4] = FREE
    state[3, 1:4] = STATIC_OCCUPIED
    ground = np.zeros_like(state)
    ground[2, 1:4] = 1

    enclosed = _opposing_obstacle_surface(state, ground, max_gap=1)

    expected = np.zeros_like(state, dtype=bool)
    expected[2, 1:4] = True
    np.testing.assert_array_equal(enclosed, expected)


def test_small_non_drivable_components_are_removed_without_merging():
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1, 1:4] = 1
    mask[3, 5] = 1

    filtered = _filter_small_components(mask, min_component_cells=3)

    expected = np.zeros_like(mask, dtype=bool)
    expected[1, 1:4] = True
    np.testing.assert_array_equal(filtered, expected)


def test_cross_scene_support_counts_distinct_frames_in_global_coordinates():
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    identity = np.eye(4, dtype=np.float64)
    shifted = np.eye(4, dtype=np.float64)
    shifted[0, 3] = 0.5
    points_a = _bev_mask_to_global_xy(
        mask, identity, pc_range=[-1, -1, -1, 1, 1, 1])
    points_b = _bev_mask_to_global_xy(
        mask, shifted, pc_range=[-1, -1, -1, 1, 1, 1])
    points_far = points_a + np.asarray([[10.0, 0.0]])

    support = _cross_frame_support(
        [points_a, points_b, points_far], radius=0.75,
        group_ids=['scene_a', 'scene_b', 'scene_c'])

    assert [item.tolist() for item in support] == [[2], [2], [1]]

    target_only = _cross_frame_support(
        [points_a, points_b, points_far], radius=0.75,
        group_ids=['holdout', 'train_a', 'train_b'], source_count=1)
    assert [item.tolist() for item in target_only] == [[2]]

    same_scene = _cross_frame_support(
        [points_a, points_b], radius=0.75,
        group_ids=['same_scene', 'same_scene'])
    assert [item.tolist() for item in same_scene] == [[1], [1]]


def test_world_target_fills_repeated_evidence_but_ignores_conflicts():
    direct = np.asarray(
        [[[0, 0, 0, FREE, 3]]], dtype=np.uint8)
    free_count = np.asarray(
        [[[2, 0, 2, 0, 2]]], dtype=np.uint8)
    static_count = np.asarray(
        [[[0, 2, 1, 2, 0]]], dtype=np.uint8)

    world, source = _compose_world_target(
        direct, free_count, static_count)

    assert world.tolist() == [[[FREE, STATIC_OCCUPIED, 0, FREE, 3]]]
    assert source.tolist() == [[[2, 3, 0, 1, 1]]]


def test_categorical_state_identity_warp_preserves_precedence():
    state = np.zeros((2, 2, 2), dtype=np.uint8)
    state[0, 0, 0] = FREE
    state[0, 0, 1] = STATIC_OCCUPIED
    state[1, 1, 1] = 3
    identity = np.eye(4, dtype=np.float64)

    warped = _warp_state_to_reference(
        state, identity, identity,
        pc_range=[-1, -1, -1, 1, 1, 1],
        occ_size=[2, 2, 2])

    np.testing.assert_array_equal(warped, state)


def test_observation_cache_expansion_deduplicates_overlapping_windows():
    infos = [
        {'scene_token': 'a'},
        {'scene_token': 'a'},
        {'scene_token': 'a'},
        {'scene_token': 'b'},
    ]

    indices = _expand_indices(
        infos, indices=[0], reference_indices=[0, 1], offsets=[0, 1])

    assert indices == [0, 1, 2]


def test_observation_cache_expansion_accepts_history_offsets():
    infos = [{'scene_token': 'a'} for _ in range(4)]

    indices = _expand_indices(
        infos, indices=[], reference_indices=[2], offsets=[-2, -1, 0])

    assert indices == [0, 1, 2]


def test_fixed_time_validation_rejects_a_dropped_frame():
    regular = [
        {'timestamp': 10.0},
        {'timestamp': 10.5},
        {'timestamp': 11.0},
    ]
    _validate_fixed_frame_times(
        regular, 0, target_offsets=[0, 1], reveal_offsets=[0, 1],
        expected_step_s=0.5, max_time_error_s=0.1)

    irregular = [
        {'timestamp': 10.0},
        {'timestamp': 10.5},
        {'timestamp': 11.5},
    ]
    try:
        _validate_fixed_frame_times(
            irregular, 0, target_offsets=[0, 1, 2], reveal_offsets=[0],
            expected_step_s=0.5, max_time_error_s=0.1)
    except ValueError as error:
        assert 'irregular timestamp window' in str(error)
    else:
        raise AssertionError('Dropped frame must fail fixed-time validation')


def test_history_time_validation_supports_negative_offsets():
    regular = [
        {'timestamp': 10.0},
        {'timestamp': 10.5},
        {'timestamp': 11.0},
    ]
    _validate_frame_offsets(
        regular, 2, offsets=[-2, -1, 0],
        expected_step_s=0.5, max_time_error_s=0.1,
        window_name='history')

    irregular = [
        {'timestamp': 10.0},
        {'timestamp': 10.5},
        {'timestamp': 11.5},
    ]
    try:
        _validate_frame_offsets(
            irregular, 2, offsets=[-2, -1, 0],
            expected_step_s=0.5, max_time_error_s=0.1,
            window_name='history')
    except ValueError as error:
        assert 'irregular history timestamp window' in str(error)
    else:
        raise AssertionError('Irregular history must fail validation')


def test_batch_references_use_only_eligible_indices(tmp_path):
    summary_path = tmp_path / 'eligibility.json'
    summary_path.write_text('{"eligible_indices": [8, 3, 8]}')

    assert _reference_indices(summary_path, []) == [3, 8]
    assert _reference_indices(summary_path, [9, 2, 9]) == [2, 9]


def test_sequence_contact_page_pads_an_incomplete_grid():
    images = [
        np.full((2, 3, 3), value, dtype=np.uint8)
        for value in (1, 2, 3)
    ]

    page = _contact_page(images, columns=2)

    assert page.shape == (4, 6, 3)
    np.testing.assert_array_equal(page[:2, :3], images[0])
    np.testing.assert_array_equal(page[:2, 3:], images[1])
    np.testing.assert_array_equal(page[2:, :3], images[2])
    assert np.all(page[2:, 3:] == 28)


def test_batch_rejects_legacy_dual_labels_before_generation(tmp_path):
    path = tmp_path / 'legacy__cross_scene.npz'
    np.savez_compressed(path, frame_index=np.int64(7))

    try:
        _dual_label_index(tmp_path)
    except ValueError as error:
        assert 'occupancy_state_3d' in str(error)
    else:
        raise AssertionError('A legacy dual label must fail preflight')
