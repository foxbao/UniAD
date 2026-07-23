"""Build causal OccWorld model inputs from a live multi-LiDAR stream."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Mapping, Sequence

import numpy as np

from tools.analysis_tools.audit_kl_occworld_dual_representation import (
    _compose_occupancy_3d,
)
from tools.analysis_tools.audit_kl_occworld_occlusion import (
    _ray_blocked_by_occupied,
    _zhw_to_xyz,
)
from tools.data_converter.generate_kl_occworld_labels import (
    ENDPOINT_RELIABLE_STATIC,
    ENDPOINT_UNCERTAIN_OBSTACLE,
    FREE,
    STATIC_OCCUPIED,
    MultiLidarOccLabelBuilder,
    _project_observed_state,
    _xyz_to_zhw,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    _correct_free_visibility_3d,
    _warp_state_to_reference,
)
from tools.data_converter.generate_kl_occworld_temporal_labels import (
    _promote_uncertain,
    _warp_mask_to_reference_xyz,
)


@dataclass(frozen=True)
class OnlineOccWorldInputs:
    """Fixed-shape causal inputs consumed by the B15 world decoder."""

    current_observation_state_3d: np.ndarray
    current_observation_valid_3d: np.ndarray
    history_observation_state_3d: np.ndarray
    history_observation_valid_3d: np.ndarray
    history_frame_valid: np.ndarray
    history_times_s: np.ndarray
    history_to_reference: np.ndarray
    direct_observation_state_3d: np.ndarray
    promoted_uncertain_mask_3d: np.ndarray
    temporal_support_count_3d: np.ndarray
    timestamp: float
    ego2global: np.ndarray
    scene_token: str

    @property
    def ready(self) -> bool:
        """Whether all five history positions contain real observations."""
        return bool(np.all(self.history_frame_valid))

    def model_inputs(self) -> dict:
        """Return only the arrays required by OccWorldHead inference."""
        return {
            'current_world_state':
                self.current_observation_state_3d,
            'current_world_valid':
                self.current_observation_valid_3d,
            'history_world_state':
                self.history_observation_state_3d,
            'history_world_valid':
                self.history_observation_valid_3d,
        }

    def to_torch(self, device=None) -> dict:
        """Return batched tensors ready for ``UniADMotionLidar.simple_test``."""
        import torch

        return {
            'current_world_state': torch.from_numpy(
                self.current_observation_state_3d[None]).long().to(device),
            'current_world_valid': torch.from_numpy(
                self.current_observation_valid_3d[None]).bool().to(device),
            'history_world_state': torch.from_numpy(
                self.history_observation_state_3d[None]).long().to(device),
            'history_world_valid': torch.from_numpy(
                self.history_observation_valid_3d[None]).bool().to(device),
        }


@dataclass(frozen=True)
class _BufferedFrame:
    timestamp: float
    ego2global: np.ndarray
    direct_state_3d: np.ndarray
    static_candidate_3d: np.ndarray


class KLOccWorldOnlineBuilder:
    """Reproduce B15 causal observation and history preprocessing online.

    The caller feeds frames at the same nominal 0.5-second cadence used for
    training. Missing startup history is left-padded with unknown voxels and
    marked by ``history_frame_valid``; ``ready`` becomes true after five
    frames. A scene-token change clears all temporal state.
    """

    def __init__(
            self,
            pc_range: Sequence[float] = (-64.0, -48.0, -2.0,
                                         64.0, 48.0, 6.0),
            bev_size: Sequence[int] = (120, 160),
            occ_size: Sequence[int] = (160, 120, 10),
            target_frame: str = 'FLU',
            collision_z: Sequence[float] = (0.3, 2.5),
            history_count: int = 5,
            temporal_count: int = 3,
            temporal_min_support: int = 2,
            temporal_radius_xy: int = 0,
            temporal_radius_z: int = 0,
            expected_step_s: float = 0.5,
            max_time_error_s: float = 0.2,
            validate_timing: bool = True):
        if history_count < 1:
            raise ValueError('history_count must be positive')
        if not 1 <= temporal_count <= history_count:
            raise ValueError(
                'temporal_count must be within the history window')
        if temporal_min_support < 1:
            raise ValueError('temporal_min_support must be positive')
        if expected_step_s <= 0 or max_time_error_s < 0:
            raise ValueError('Invalid frame timing tolerances')

        self.pc_range = np.asarray(pc_range, dtype=np.float32)
        self.occ_size = np.asarray(occ_size, dtype=np.int64)
        self.history_count = int(history_count)
        self.temporal_count = int(temporal_count)
        self.temporal_min_support = int(temporal_min_support)
        self.temporal_radius_xy = int(temporal_radius_xy)
        self.temporal_radius_z = int(temporal_radius_z)
        self.expected_step_s = float(expected_step_s)
        self.max_time_error_s = float(max_time_error_s)
        self.validate_timing = bool(validate_timing)
        self.label_builder = MultiLidarOccLabelBuilder(
            pc_range=pc_range,
            bev_size=bev_size,
            occ_size=occ_size,
            target_frame=target_frame,
            collision_z=collision_z,
        )
        self._frames: Deque[_BufferedFrame] = deque(
            maxlen=self.history_count)
        self._scene_token = None

    def reset(self, scene_token: str = None):
        """Clear all temporal state, for example at a scene boundary."""
        self._frames.clear()
        self._scene_token = scene_token

    def push_info(self, info: dict) -> OnlineOccWorldInputs:
        """Replay one dataset frame using its raw point files and boxes."""
        evidence = self.label_builder.build(info, diagnostics=True)
        return self.push_evidence(
            evidence=evidence,
            timestamp=float(info['timestamp']),
            ego2global=np.asarray(info['ego2global'], dtype=np.float64),
            scene_token=str(info.get('scene_token', '')),
        )

    def push_sensor_frame(
            self,
            sensor_inputs: Mapping[str, dict],
            timestamp: float,
            ego2global: np.ndarray,
            scene_token: str,
            boxes: np.ndarray = None,
            instances: Sequence[dict] = None) -> OnlineOccWorldInputs:
        """Process target-frame point clouds and optional tracker boxes."""
        evidence = self.label_builder.build(
            diagnostics=True,
            sensor_inputs=sensor_inputs,
            boxes=boxes,
            instances=instances,
        )
        return self.push_evidence(
            evidence=evidence,
            timestamp=timestamp,
            ego2global=ego2global,
            scene_token=scene_token,
        )

    def push_evidence(
            self,
            evidence: dict,
            timestamp: float,
            ego2global: np.ndarray,
            scene_token: str) -> OnlineOccWorldInputs:
        """Update the queue from one diagnostic label-builder result."""
        required = {
            'filtered_state_3d',
            'filtered_occupancy_target',
            'box_occupied_3d',
            'endpoint_type_3d',
            'per_sensor_free_3d',
            'sensor_origins',
        }
        missing = sorted(required.difference(evidence))
        if missing:
            raise KeyError(f'Online evidence is missing keys: {missing}')

        scene_token = str(scene_token)
        if self._scene_token is None:
            self._scene_token = scene_token
        elif scene_token != self._scene_token:
            self.reset(scene_token)

        timestamp = float(timestamp)
        ego2global = np.asarray(ego2global, dtype=np.float64)
        if ego2global.shape != (4, 4) or not np.isfinite(ego2global).all():
            raise ValueError('ego2global must be a finite [4,4] matrix')
        if self._frames:
            delta = timestamp - self._frames[-1].timestamp
            if delta <= 0:
                raise ValueError('Online timestamps must be increasing')
            if (self.validate_timing and
                    abs(delta - self.expected_step_s) >
                    self.max_time_error_s):
                raise ValueError(
                    f'Frame interval {delta:.3f}s differs from expected '
                    f'{self.expected_step_s:.3f}s')

        direct_state, _, _ = _correct_free_visibility_3d(
            evidence, self.pc_range)
        endpoint_type = np.asarray(
            evidence['endpoint_type_3d'], dtype=np.uint8)
        expected_shape = (
            int(self.occ_size[2]),
            int(self.occ_size[1]),
            int(self.occ_size[0]),
        )
        if direct_state.shape != expected_shape:
            raise ValueError(
                f'Expected ZHW state {expected_shape}, got '
                f'{direct_state.shape}')
        static_candidate = (
            (endpoint_type == ENDPOINT_UNCERTAIN_OBSTACLE) |
            (endpoint_type == ENDPOINT_RELIABLE_STATIC))

        prior_count = self.temporal_count - 1
        source_frames = (
            list(self._frames)[-prior_count:] if prior_count else [])
        support_xyz = np.zeros(
            tuple(self.occ_size.tolist()), dtype=np.uint8)
        for frame in source_frames:
            warped = _warp_mask_to_reference_xyz(
                frame.static_candidate_3d,
                frame.ego2global,
                ego2global,
                self.pc_range,
                self.occ_size,
                radius_xy=self.temporal_radius_xy,
                radius_z=self.temporal_radius_z,
            )
            support_xyz += warped.astype(np.uint8)
        current_support = _warp_mask_to_reference_xyz(
            static_candidate,
            ego2global,
            ego2global,
            self.pc_range,
            self.occ_size,
            radius_xy=self.temporal_radius_xy,
            radius_z=self.temporal_radius_z,
        )
        support_xyz += current_support.astype(np.uint8)
        support_3d = _xyz_to_zhw(support_xyz)
        promoted = _promote_uncertain(
            endpoint_type, support_3d, self.temporal_min_support)

        temporal_state = np.asarray(
            evidence['filtered_state_3d'], dtype=np.uint8).copy()
        temporal_state[promoted] = STATIC_OCCUPIED
        valid_free_count = self._temporal_valid_free_count(
            evidence, temporal_state, promoted)
        current_state, current_valid = _compose_occupancy_3d(
            temporal_state,
            np.asarray(evidence['box_occupied_3d'], dtype=np.uint8),
            valid_free_count,
        )

        self._frames.append(_BufferedFrame(
            timestamp=timestamp,
            ego2global=ego2global.copy(),
            direct_state_3d=direct_state.copy(),
            static_candidate_3d=static_candidate.copy(),
        ))
        return self._build_inputs(
            current_state=current_state,
            current_valid=current_valid,
            direct_state=direct_state,
            promoted=promoted,
            support_3d=support_3d,
            timestamp=timestamp,
            ego2global=ego2global,
            scene_token=scene_token,
        )

    def _temporal_valid_free_count(
            self,
            evidence: dict,
            temporal_state: np.ndarray,
            promoted: np.ndarray) -> np.ndarray:
        """Recheck free rays after temporal obstacle promotion."""
        box_occupied = np.asarray(
            evidence['box_occupied_3d'], dtype=np.uint8)
        instance_bev = np.any(box_occupied > 0, axis=0).astype(np.uint8)
        unresolved = (
            (np.asarray(evidence['endpoint_type_3d']) ==
             ENDPOINT_UNCERTAIN_OBSTACLE) & ~promoted)
        z_centers = self.label_builder.builder.voxel_centers(
            2, np.arange(int(self.occ_size[2]), dtype=np.int64))
        projected = _project_observed_state(
            temporal_state,
            z_centers,
            self.label_builder.collision_z,
            instance_bev,
            blocking_unknown_3d=unresolved,
        )
        projected_free = projected == FREE
        occupied_xyz = _zhw_to_xyz(
            (temporal_state == STATIC_OCCUPIED) | (box_occupied > 0))
        per_sensor_free = np.asarray(
            evidence['per_sensor_free_3d']) > 0
        valid_count = np.zeros_like(temporal_state, dtype=np.uint8)
        for sensor_index, sensor_origin in enumerate(
                evidence['sensor_origins']):
            active = (
                per_sensor_free[sensor_index] &
                (temporal_state == FREE) &
                projected_free[None, :, :])
            for z_index, row, col in np.argwhere(active):
                xyz_index = np.asarray([
                    col,
                    temporal_state.shape[1] - 1 - row,
                    z_index,
                ], dtype=np.int64)
                if not _ray_blocked_by_occupied(
                        xyz_index,
                        sensor_origin,
                        occupied_xyz,
                        self.pc_range):
                    valid_count[z_index, row, col] += 1
        return valid_count

    def _build_inputs(
            self,
            current_state: np.ndarray,
            current_valid: np.ndarray,
            direct_state: np.ndarray,
            promoted: np.ndarray,
            support_3d: np.ndarray,
            timestamp: float,
            ego2global: np.ndarray,
            scene_token: str) -> OnlineOccWorldInputs:
        frames = list(self._frames)
        pad_count = self.history_count - len(frames)
        state_shape = tuple(current_state.shape)
        history_states = [
            np.zeros(state_shape, dtype=np.uint8)
            for _ in range(pad_count)
        ]
        history_times = [np.nan] * pad_count
        history_transforms = [
            np.eye(4, dtype=np.float64) for _ in range(pad_count)
        ]
        for frame in frames:
            history_states.append(_warp_state_to_reference(
                frame.direct_state_3d,
                frame.ego2global,
                ego2global,
                self.pc_range,
                self.occ_size,
            ))
            history_times.append(frame.timestamp - timestamp)
            history_transforms.append(
                np.linalg.inv(ego2global) @ frame.ego2global)
        history_states = np.stack(history_states, axis=0)
        history_valid = history_states != 0
        frame_valid = np.asarray(
            [False] * pad_count + [True] * len(frames), dtype=np.bool_)

        return OnlineOccWorldInputs(
            current_observation_state_3d=current_state.astype(
                np.uint8, copy=False),
            current_observation_valid_3d=current_valid.astype(
                np.bool_, copy=False),
            history_observation_state_3d=history_states,
            history_observation_valid_3d=history_valid,
            history_frame_valid=frame_valid,
            history_times_s=np.asarray(history_times, dtype=np.float32),
            history_to_reference=np.stack(history_transforms, axis=0),
            direct_observation_state_3d=direct_state.astype(
                np.uint8, copy=False),
            promoted_uncertain_mask_3d=promoted.astype(
                np.bool_, copy=False),
            temporal_support_count_3d=support_3d.astype(
                np.uint8, copy=False),
            timestamp=float(timestamp),
            ego2global=ego2global.copy(),
            scene_token=scene_token,
        )
