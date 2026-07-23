import numpy as np
import pytest

from tools.data_converter.generate_kl_occworld_labels import (
    ENDPOINT_UNCERTAIN_OBSTACLE,
    INSTANCE_OCCUPIED,
    STATIC_OCCUPIED,
    MultiLidarOccLabelBuilder,
)
from tools.data_converter.kl_occworld_online import (
    KLOccWorldOnlineBuilder,
)


def test_label_builder_accepts_target_frame_sensor_inputs():
    builder = MultiLidarOccLabelBuilder(
        pc_range=[-4, -4, -2, 4, 4, 2],
        bev_size=[8, 8],
        occ_size=[8, 8, 4],
        target_frame='FLU',
        collision_z=[-1, 1],
    )
    result = builder.build(
        diagnostics=True,
        sensor_inputs={
            'lidar_a': {
                'points': np.asarray([
                    [2.0, 0.0, 0.0],
                    [2.0, 1.0, 0.5],
                ], dtype=np.float32),
                'origin': np.asarray([0.0, 0.0, 0.0]),
            },
            'lidar_b': {
                'points': np.asarray([
                    [-2.0, 0.0, 0.0],
                    [-2.0, -1.0, 0.5],
                ], dtype=np.float32),
                'origin': np.asarray([0.5, 0.0, 0.0]),
            },
        },
        boxes=np.empty((0, 7), dtype=np.float32),
    )

    assert result['sensor_names'].tolist() == ['lidar_a', 'lidar_b']
    assert result['point_count'] == 4
    assert result['sensor_origins'].shape == (2, 3)
    assert result['per_sensor_free_3d'].shape == (2, 4, 8, 8)


def _evidence(shape=(2, 4, 4), instance=False):
    filtered = np.zeros(shape, dtype=np.uint8)
    endpoint_type = np.zeros(shape, dtype=np.uint8)
    endpoint_type[0, 1, 1] = ENDPOINT_UNCERTAIN_OBSTACLE
    boxes = np.zeros(shape, dtype=np.uint8)
    if instance:
        boxes[1, 2, 2] = 1
    return {
        'filtered_state_3d': filtered,
        'filtered_occupancy_target': np.zeros(shape, dtype=np.uint8),
        'box_occupied_3d': boxes,
        'endpoint_type_3d': endpoint_type,
        'per_sensor_free_3d': np.zeros((0,) + shape, dtype=np.uint8),
        'sensor_origins': np.zeros((0, 3), dtype=np.float32),
    }


def _online_builder():
    return KLOccWorldOnlineBuilder(
        pc_range=[-2, -2, -1, 2, 2, 1],
        bev_size=[4, 4],
        occ_size=[4, 4, 2],
        target_frame='FLU',
        collision_z=[-0.5, 0.5],
    )


def test_online_builder_promotes_past_supported_current_obstacle():
    online = _online_builder()
    identity = np.eye(4, dtype=np.float64)

    first = online.push_evidence(
        _evidence(), 0.0, identity, 'scene_a')
    assert not first.ready
    assert first.history_frame_valid.tolist() == [
        False, False, False, False, True]
    assert not first.promoted_uncertain_mask_3d[0, 1, 1]

    output = first
    for index in range(1, 5):
        output = online.push_evidence(
            _evidence(instance=index == 4),
            index * 0.5,
            identity,
            'scene_a',
        )

    assert output.ready
    assert output.history_times_s.tolist() == [
        -2.0, -1.5, -1.0, -0.5, 0.0]
    assert output.promoted_uncertain_mask_3d[0, 1, 1]
    assert output.current_observation_state_3d[0, 1, 1] == (
        STATIC_OCCUPIED)
    assert output.direct_observation_state_3d[0, 1, 1] == 0
    assert output.history_observation_state_3d[-1, 0, 1, 1] == 0
    assert output.current_observation_state_3d[1, 2, 2] == (
        INSTANCE_OCCUPIED)
    assert output.history_observation_state_3d[-1, 1, 2, 2] == (
        INSTANCE_OCCUPIED)
    assert set(output.model_inputs()) == {
        'current_world_state',
        'current_world_valid',
        'history_world_state',
        'history_world_valid',
    }
    torch_inputs = output.to_torch()
    assert tuple(torch_inputs['current_world_state'].shape) == (
        1, 2, 4, 4)
    assert tuple(torch_inputs['history_world_state'].shape) == (
        1, 5, 2, 4, 4)


def test_online_builder_resets_history_at_scene_boundary():
    online = _online_builder()
    identity = np.eye(4, dtype=np.float64)
    for index in range(3):
        online.push_evidence(
            _evidence(), index * 0.5, identity, 'scene_a')

    output = online.push_evidence(
        _evidence(), 0.0, identity, 'scene_b')

    assert output.scene_token == 'scene_b'
    assert output.history_frame_valid.tolist() == [
        False, False, False, False, True]


def test_online_builder_rejects_wrong_frame_cadence():
    online = _online_builder()
    identity = np.eye(4, dtype=np.float64)
    online.push_evidence(_evidence(), 0.0, identity, 'scene_a')

    with pytest.raises(ValueError, match='Frame interval'):
        online.push_evidence(_evidence(), 1.0, identity, 'scene_a')
