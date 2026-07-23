#!/usr/bin/env python
"""Generate KL multi-LiDAR occupancy evidence labels.

This is the first, deliberately small OccWorld data milestone.  It keeps the
existing merged LiDAR as the model input, but builds supervision from the raw
per-sensor point clouds so every free-space ray starts at its calibrated
sensor origin.

The saved ``state`` array is the conservative ``[H, W]`` planning view:

    0: unknown (not observed by any valid ray)
    1: collision-band ray free
    2: static occupied evidence
    3: annotated instance occupied

The strict sensor-observation target is stored separately:

    observed_state_3d [Z, H, W]: 0 unknown, 1 free, 2 occupied
    visibility_mask   [Z, H, W]: loss-valid sensor evidence
    occupancy_target  [Z, H, W]: binary occupied target
    sensor_count_3d   [Z, H, W]: number of supporting LiDARs

The raw endpoint state is retained for traceability.  A second conservative
layer reuses the established KL ground/obstacle filters:

    filtered_state_3d          [Z, H, W]
    filtered_visibility_mask   [Z, H, W]
    filtered_occupancy_target  [Z, H, W]
    unreliable_endpoint_mask   [Z, H, W]

``box_instance_id_3d`` and ``box_class_id_3d`` preserve annotated box volumes
without pretending that every voxel inside a box was directly observed.  The
first version generates current-frame labels only.  Future frames can later be
transformed into the current reference pose and stacked into ``[T,Z,H,W]``.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    RaycastDrivableBuilder,
)
from tools.data_converter.kl_converter import (
    get_transform_matrix,
    read_pc,
    transform_points_numba,
)


UNKNOWN = 0
FREE = 1
STATIC_OCCUPIED = 2
INSTANCE_OCCUPIED = 3

ENDPOINT_NONE = 0
ENDPOINT_GROUND = 1
ENDPOINT_UNCERTAIN_OBSTACLE = 2
ENDPOINT_RELIABLE_STATIC = 3
ENDPOINT_INSTANCE = 4

STATE_NAMES = {
    UNKNOWN: 'unknown',
    FREE: 'free',
    STATIC_OCCUPIED: 'static_occupied',
    INSTANCE_OCCUPIED: 'instance_occupied',
}

OBSERVED_STATE_NAMES = np.asarray(['unknown', 'free', 'occupied'])
ENDPOINT_TYPE_NAMES = np.asarray([
    'none', 'ground', 'uncertain_obstacle',
    'reliable_static', 'instance',
])


def _xyz_to_zhw(volume: np.ndarray) -> np.ndarray:
    """Convert builder-native [X,Y,Z] to image-aligned [Z,H,W]."""
    if volume.ndim != 3:
        raise ValueError(f'Expected a 3D [X,Y,Z] array, got {volume.shape}')
    return np.transpose(volume, (2, 1, 0))[:, ::-1, :]


def _ego_ignore_xyz_mask(builder: RaycastDrivableBuilder) -> np.ndarray:
    """Return the existing ego-ignore cuboid in builder-native XYZ layout."""
    mask = np.zeros(tuple(builder.occ_size.tolist()), dtype=bool)
    if builder.ego_ignore_range is None:
        return mask

    lo = np.maximum(builder.ego_ignore_range[:3], builder.pc_range[:3])
    hi = np.minimum(builder.ego_ignore_range[3:], builder.pc_range[3:])
    if np.any(hi <= lo):
        return mask

    keep = []
    for axis in range(3):
        indices = np.arange(builder.occ_size[axis], dtype=np.int64)
        centers = builder.voxel_centers(axis, indices)
        keep.append(np.flatnonzero(
            (centers >= lo[axis]) & (centers <= hi[axis])))
    if any(indices.size == 0 for indices in keep):
        return mask
    mask[np.ix_(*keep)] = True
    return mask


def _compose_observed_state_xyz(
        free_sensor_count: np.ndarray,
        occupied_sensor_count: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray]:
    """Build strict 3D targets; occupied endpoints override free rays."""
    if free_sensor_count.shape != occupied_sensor_count.shape:
        raise ValueError(
            'free and occupied counts must have the same shape, got '
            f'{free_sensor_count.shape} and {occupied_sensor_count.shape}')
    state = np.full(free_sensor_count.shape, UNKNOWN, dtype=np.uint8)
    state[free_sensor_count > 0] = FREE
    state[occupied_sensor_count > 0] = STATIC_OCCUPIED
    visibility = (free_sensor_count > 0) | (occupied_sensor_count > 0)
    occupancy = (state == STATIC_OCCUPIED).astype(np.uint8)
    return state, visibility.astype(np.uint8), occupancy


def _compose_filtered_state_xyz(
        free_sensor_count: np.ndarray,
        occupied_sensor_count: np.ndarray,
        reliable_occupied: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a clean target without turning rejected endpoints into free."""
    if not (free_sensor_count.shape == occupied_sensor_count.shape ==
            reliable_occupied.shape):
        raise ValueError('All filtered-state inputs must share one shape')
    has_endpoint = occupied_sensor_count > 0
    reliable_occupied = reliable_occupied.astype(bool)
    reliable_free = (free_sensor_count > 0) & ~has_endpoint
    unreliable_endpoint = has_endpoint & ~reliable_occupied

    state = np.full(free_sensor_count.shape, UNKNOWN, dtype=np.uint8)
    state[reliable_free] = FREE
    state[reliable_occupied] = STATIC_OCCUPIED
    visibility = reliable_free | reliable_occupied
    occupancy = reliable_occupied.astype(np.uint8)
    return (state, visibility.astype(np.uint8), occupancy,
            unreliable_endpoint.astype(np.uint8))


def _classify_endpoint_evidence_xyz(
        occupied_sensor_count: np.ndarray,
        raw_obstacle: np.ndarray,
        reliable_static: np.ndarray,
        instance_endpoint: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split raw endpoints into ground, uncertain, and reliable evidence."""
    if not (occupied_sensor_count.shape == raw_obstacle.shape ==
            reliable_static.shape == instance_endpoint.shape):
        raise ValueError('All endpoint evidence inputs must share one shape')
    has_endpoint = occupied_sensor_count > 0
    raw_obstacle = (raw_obstacle > 0) & has_endpoint
    reliable_static = (reliable_static > 0) & has_endpoint
    instance_endpoint = (instance_endpoint > 0) & has_endpoint
    ground_endpoint = has_endpoint & ~raw_obstacle & ~instance_endpoint
    uncertain_obstacle = raw_obstacle & ~reliable_static

    endpoint_type = np.full(has_endpoint.shape, ENDPOINT_NONE, dtype=np.uint8)
    endpoint_type[ground_endpoint] = ENDPOINT_GROUND
    endpoint_type[uncertain_obstacle] = ENDPOINT_UNCERTAIN_OBSTACLE
    endpoint_type[reliable_static] = ENDPOINT_RELIABLE_STATIC
    endpoint_type[instance_endpoint] = ENDPOINT_INSTANCE
    return (ground_endpoint.astype(np.uint8),
            uncertain_obstacle.astype(np.uint8), endpoint_type)


def _project_observed_state(
        observed_state_3d: np.ndarray,
        z_centers: np.ndarray,
        collision_z: Sequence[float],
        instance_occupied: np.ndarray,
        blocking_unknown_3d: np.ndarray = None) -> np.ndarray:
    """Conservatively project strict 3D evidence into the collision band."""
    z_centers = np.asarray(z_centers, dtype=np.float32)
    collision_z = np.asarray(collision_z, dtype=np.float32)
    if observed_state_3d.ndim != 3:
        raise ValueError(
            f'observed_state_3d must be [Z,H,W], got '
            f'{observed_state_3d.shape}')
    if z_centers.shape != (observed_state_3d.shape[0], ):
        raise ValueError(
            f'z_centers must have shape ({observed_state_3d.shape[0]},), '
            f'got {z_centers.shape}')
    keep = ((z_centers >= collision_z[0]) &
            (z_centers <= collision_z[1]))
    if not np.any(keep):
        raise ValueError(
            f'No voxel centers fall inside collision_z={collision_z}')
    band = observed_state_3d[keep]
    free = np.any(band == FREE, axis=0).astype(np.uint8)
    occupied = np.any(
        band == STATIC_OCCUPIED, axis=0).astype(np.uint8)
    state = _compose_state(free, occupied, instance_occupied)
    if blocking_unknown_3d is not None:
        if blocking_unknown_3d.shape != observed_state_3d.shape:
            raise ValueError(
                'blocking_unknown_3d must match observed_state_3d, got '
                f'{blocking_unknown_3d.shape} and '
                f'{observed_state_3d.shape}')
        blocked = np.any(blocking_unknown_3d[keep] > 0, axis=0)
        state[blocked & (state == FREE)] = UNKNOWN
    return state


def _load_infos(path: Path):
    with path.open('rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list'], data.get('metainfo', {})
    if isinstance(data, dict) and 'infos' in data:
        return data['infos'], data.get('metadata', {})
    if isinstance(data, list):
        return data, {}
    raise KeyError(f'Unsupported info container in {path}')


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = REPO_ROOT / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_value)


def _find_extrinsics_path(point_path: Path) -> Path:
    # <record>/lidar/<sensor>/<timestamp>.pcd
    record_dir = point_path.parents[2]
    candidates = (
        record_dir / 'extrinsics.json',
        record_dir.parent / 'extrinsics.json',
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'No extrinsics.json near raw LiDAR file {point_path}')


def _load_extrinsics(path: Path) -> Dict[str, np.ndarray]:
    with path.open() as f:
        raw = json.load(f)
    prefix = 'Tx_baselink_lidar_'
    transforms = {}
    for key, value in raw.items():
        sensor_name = key[len(prefix):] if key.startswith(prefix) else key
        transforms[sensor_name] = get_transform_matrix(value)
    return transforms


def _rotate_flu_to_rfu(points: np.ndarray) -> np.ndarray:
    """Match ``kl_converter.merge_lidar_points`` RFU conversion."""
    out = points.copy()
    out[:, 0] = -points[:, 1]
    out[:, 1] = points[:, 0]
    return out


def _transform_sensor_points(points: np.ndarray, transform: np.ndarray,
                             target_frame: str
                             ) -> Tuple[np.ndarray, np.ndarray]:
    transformed = transform_points_numba(
        points, transform[:3, :3], transform[:3, 3])
    origin = transform[:3, 3].astype(np.float32).copy()
    if target_frame == 'RFU':
        transformed = _rotate_flu_to_rfu(transformed)
        origin = np.asarray([-origin[1], origin[0], origin[2]],
                            dtype=np.float32)
    elif target_frame != 'FLU':
        raise ValueError(f'Unsupported target LiDAR frame: {target_frame}')
    return transformed, origin


def _valid_point_voxels(builder: RaycastDrivableBuilder,
                        points: np.ndarray) -> np.ndarray:
    voxels = builder.coord_to_index_floor(points[:, :3])
    valid = np.all((voxels >= 0) & (voxels < builder.occ_size), axis=1)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.int64)
    return voxels[valid]


def _voxel_mask_xyz(builder: RaycastDrivableBuilder,
                    voxels: np.ndarray) -> np.ndarray:
    mask = np.zeros(tuple(builder.occ_size.tolist()), dtype=np.uint8)
    if voxels.shape[0] > 0:
        mask[tuple(voxels.T)] = 1
    return mask


def _collect_boxes(info: dict) -> Tuple[np.ndarray, List[dict]]:
    boxes = []
    instances = []
    for instance in info.get('instances', []):
        if not bool(instance.get('bbox_3d_isvalid', True)):
            continue
        box = np.asarray(instance.get('bbox_3d', []), dtype=np.float32)
        if box.size < 7 or not np.isfinite(box[:7]).all():
            continue
        if np.any(box[3:6] <= 0):
            continue
        boxes.append(box[:7])
        instances.append(instance)
    if not boxes:
        return np.empty((0, 7), dtype=np.float32), []
    return np.stack(boxes), instances


def _instance_bev_maps(builder: RaycastDrivableBuilder,
                       boxes: np.ndarray,
                       instances: Sequence[dict]
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = np.zeros((builder.bev_h, builder.bev_w), dtype=np.uint8)
    instance_map = np.full((builder.bev_h, builder.bev_w), -1,
                           dtype=np.int32)
    class_map = np.full((builder.bev_h, builder.bev_w), -1,
                        dtype=np.int16)

    for box, instance in zip(boxes, instances):
        indices = builder.box_voxel_indices(box)
        if indices is None:
            continue
        x_idx, y_idx, _ = indices
        rows = builder.bev_h - 1 - y_idx
        cols = x_idx
        valid = ((rows >= 0) & (rows < builder.bev_h) &
                 (cols >= 0) & (cols < builder.bev_w))
        rows = rows[valid]
        cols = cols[valid]
        occupied[rows, cols] = 1
        instance_map[rows, cols] = int(instance.get('track_id', -1))
        class_map[rows, cols] = int(
            instance.get('bbox_label_3d', instance.get('bbox_label', -1)))

    return occupied, instance_map, class_map


def _instance_3d_maps(builder: RaycastDrivableBuilder,
                      boxes: np.ndarray,
                      instances: Sequence[dict]
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize annotated box volumes separately from ray observations."""
    shape_xyz = tuple(builder.occ_size.tolist())
    occupied_xyz = np.zeros(shape_xyz, dtype=np.uint8)
    instance_xyz = np.full(shape_xyz, -1, dtype=np.int32)
    class_xyz = np.full(shape_xyz, -1, dtype=np.int16)

    for box, instance in zip(boxes, instances):
        indices = builder.box_voxel_indices(box)
        if indices is None:
            continue
        x_idx, y_idx, z_idx = indices
        xyz = (x_idx[:, None], y_idx[:, None], z_idx[None, :])
        occupied_xyz[xyz] = 1
        instance_xyz[xyz] = int(instance.get('track_id', -1))
        class_xyz[xyz] = int(
            instance.get('bbox_label_3d', instance.get('bbox_label', -1)))

    return tuple(_xyz_to_zhw(volume) for volume in (
        occupied_xyz, instance_xyz, class_xyz))


def _compose_state(free: np.ndarray, static_occupied: np.ndarray,
                   instance_occupied: np.ndarray) -> np.ndarray:
    """Compose a BEV state with conservative occupied precedence."""
    state = np.full(free.shape, UNKNOWN, dtype=np.uint8)
    state[free > 0] = FREE
    state[static_occupied > 0] = STATIC_OCCUPIED
    state[instance_occupied > 0] = INSTANCE_OCCUPIED
    return state


class MultiLidarOccLabelBuilder:
    """Fuse calibrated KL LiDAR rays into 3D and BEV evidence labels."""

    def __init__(self, pc_range: Sequence[float], bev_size: Sequence[int],
                 occ_size: Sequence[int], target_frame: str,
                 collision_z: Sequence[float] = (0.3, 2.5)):
        self.builder = RaycastDrivableBuilder(
            pc_range=pc_range,
            bev_size=bev_size,
            occ_size=occ_size,
        )
        # Same classifier without morphological ground filling.  It is only
        # invoked for diagnostics, so normal label generation keeps its
        # original runtime.
        self.raw_ground_builder = RaycastDrivableBuilder(
            pc_range=pc_range,
            bev_size=bev_size,
            occ_size=occ_size,
            fill_ground=False,
        )
        self.target_frame = target_frame
        self.collision_z = np.asarray(collision_z, dtype=np.float32)
        if (self.collision_z.shape != (2, ) or
                self.collision_z[1] <= self.collision_z[0]):
            raise ValueError(
                f'collision_z must be [min, max], got {collision_z}')
        self._extrinsics_cache: Dict[Path, Dict[str, np.ndarray]] = {}

    def _extrinsics_for(self, point_path: Path):
        extrinsics_path = _find_extrinsics_path(point_path)
        if extrinsics_path not in self._extrinsics_cache:
            self._extrinsics_cache[extrinsics_path] = _load_extrinsics(
                extrinsics_path)
        return self._extrinsics_cache[extrinsics_path]

    def _collision_band_voxels(self, voxels: np.ndarray) -> np.ndarray:
        if voxels.shape[0] == 0:
            return voxels
        z_centers = self.builder.voxel_centers(2, voxels[:, 2])
        keep = ((z_centers >= self.collision_z[0]) &
                (z_centers <= self.collision_z[1]))
        return voxels[keep]

    def build(self, info: dict, diagnostics: bool = False,
              return_points: bool = False) -> dict:
        all_points = []
        free_voxels = []
        collision_free_voxels = []
        sensor_names = []
        sensor_origins = []
        sensor_free_voxels = []
        sensor_hit_voxels = []
        shape_xyz = tuple(self.builder.occ_size.tolist())
        free_sensor_count_xyz = np.zeros(shape_xyz, dtype=np.uint8)
        occupied_sensor_count_xyz = np.zeros(shape_xyz, dtype=np.uint8)
        sensor_count_xyz = np.zeros(shape_xyz, dtype=np.uint8)
        hit_point_count_xyz = np.zeros(shape_xyz, dtype=np.uint16)
        free_sensor_count = np.zeros(
            (self.builder.bev_h, self.builder.bev_w), dtype=np.uint8)
        collision_free_sensor_count = np.zeros_like(free_sensor_count)

        lidar_entries = info.get('sync_info', {}).get('lidars', {})
        for sensor_name, entry in sorted(lidar_entries.items()):
            if not entry.get('valid', False) or not entry.get('path'):
                continue
            point_path = _resolve_path(entry['path'])
            transforms = self._extrinsics_for(point_path)
            if sensor_name not in transforms:
                raise KeyError(
                    f'{sensor_name} missing from extrinsics near {point_path}')

            raw_points = read_pc(point_path)
            points, origin = _transform_sensor_points(
                raw_points, transforms[sensor_name], self.target_frame)
            point_voxels = _valid_point_voxels(self.builder, points)
            hits = np.unique(point_voxels, axis=0)
            if hits.shape[0] == 0:
                continue

            free = self.builder.raycast_free_voxels(
                hits, ray_origin=origin)
            collision_free = self._collision_band_voxels(free)
            free_sensor_count_xyz[tuple(free.T)] += 1
            occupied_sensor_count_xyz[tuple(hits.T)] += 1
            np.add.at(hit_point_count_xyz, tuple(point_voxels.T), 1)
            observed_by_sensor = np.unique(
                np.concatenate([free, hits], axis=0), axis=0)
            sensor_count_xyz[tuple(observed_by_sensor.T)] += 1
            all_points.append(points)
            free_voxels.append(free)
            collision_free_voxels.append(collision_free)
            sensor_names.append(sensor_name)
            sensor_origins.append(origin)
            if diagnostics:
                sensor_free_voxels.append(free)
                sensor_hit_voxels.append(hits)
                free_sensor_count += self.builder.voxels_to_bev(free)
                collision_free_sensor_count += self.builder.voxels_to_bev(
                    collision_free)

        if not all_points:
            raise RuntimeError(
                f'Frame {info.get("token")} has no valid raw LiDAR points')

        points = np.concatenate(all_points, axis=0)
        if free_voxels:
            free_voxels = np.unique(
                np.concatenate(free_voxels, axis=0), axis=0)
        else:
            free_voxels = np.empty((0, 3), dtype=np.int64)
        if collision_free_voxels:
            collision_free_voxels = np.unique(
                np.concatenate(collision_free_voxels, axis=0), axis=0)
        else:
            collision_free_voxels = np.empty((0, 3), dtype=np.int64)

        boxes, instances = _collect_boxes(info)
        evidence = self.builder.build(
            points, boxes, free_voxels=free_voxels, return_voxels=True)
        instance_occupied, instance_map, class_map = _instance_bev_maps(
            self.builder, boxes, instances)
        (box_occupied_3d, box_instance_id_3d,
         box_class_id_3d) = _instance_3d_maps(
             self.builder, boxes, instances)

        # Ignore self-return evidence in the same cuboid already used by the
        # established KL BEV generator. Counts are zeroed too, so every saved
        # supervision array follows the same validity convention.
        ego_ignore_xyz = _ego_ignore_xyz_mask(self.builder)
        free_sensor_count_xyz[ego_ignore_xyz] = 0
        occupied_sensor_count_xyz[ego_ignore_xyz] = 0
        sensor_count_xyz[ego_ignore_xyz] = 0
        hit_point_count_xyz[ego_ignore_xyz] = 0
        (observed_state_xyz, visibility_xyz,
         occupancy_target_xyz) = _compose_observed_state_xyz(
             free_sensor_count_xyz, occupied_sensor_count_xyz)

        static_obstacle_xyz = _voxel_mask_xyz(
            self.builder, evidence['obstacle_voxels'])
        raw_obstacle_xyz = _voxel_mask_xyz(
            self.builder, evidence['raw_obstacle_voxels'])
        instance_endpoint_xyz = _voxel_mask_xyz(
            self.builder, evidence['semantic_voxels'])
        static_obstacle_xyz[ego_ignore_xyz] = 0
        raw_obstacle_xyz[ego_ignore_xyz] = 0
        instance_endpoint_xyz[ego_ignore_xyz] = 0
        reliable_occupied_xyz = np.maximum(
            static_obstacle_xyz, instance_endpoint_xyz).astype(np.uint8)
        reliable_occupied_xyz[ego_ignore_xyz] = 0
        (ground_endpoint_xyz, uncertain_obstacle_xyz,
         endpoint_type_xyz) = _classify_endpoint_evidence_xyz(
             occupied_sensor_count_xyz, raw_obstacle_xyz,
             static_obstacle_xyz, instance_endpoint_xyz)
        (filtered_state_xyz, filtered_visibility_xyz,
         filtered_occupancy_xyz,
         unreliable_endpoint_xyz) = _compose_filtered_state_xyz(
             free_sensor_count_xyz, occupied_sensor_count_xyz,
             reliable_occupied_xyz)

        observed_state_3d = _xyz_to_zhw(observed_state_xyz)
        visibility_mask = _xyz_to_zhw(visibility_xyz)
        occupancy_target = _xyz_to_zhw(occupancy_target_xyz)
        filtered_state_3d = _xyz_to_zhw(filtered_state_xyz)
        filtered_visibility_mask = _xyz_to_zhw(
            filtered_visibility_xyz)
        filtered_occupancy_target = _xyz_to_zhw(
            filtered_occupancy_xyz)
        unreliable_endpoint_mask = _xyz_to_zhw(
            unreliable_endpoint_xyz)
        static_obstacle_3d = _xyz_to_zhw(static_obstacle_xyz)
        raw_obstacle_3d = _xyz_to_zhw(raw_obstacle_xyz)
        instance_endpoint_3d = _xyz_to_zhw(instance_endpoint_xyz)
        ground_endpoint_3d = _xyz_to_zhw(ground_endpoint_xyz)
        uncertain_obstacle_3d = _xyz_to_zhw(uncertain_obstacle_xyz)
        endpoint_type_3d = _xyz_to_zhw(endpoint_type_xyz)
        free_sensor_count_3d = _xyz_to_zhw(free_sensor_count_xyz)
        occupied_sensor_count_3d = _xyz_to_zhw(
            occupied_sensor_count_xyz)
        sensor_count_3d = _xyz_to_zhw(sensor_count_xyz)
        hit_point_count_3d = _xyz_to_zhw(hit_point_count_xyz)
        z_centers = self.builder.voxel_centers(
            2, np.arange(self.builder.occ_size[2], dtype=np.int64))

        ray_free = evidence['free'].astype(np.uint8)
        collision_band_free = self.builder.voxels_to_bev(
            collision_free_voxels).astype(np.uint8)
        filled_ground = evidence['ground'].astype(np.uint8)
        # Keep the original optimistic composition for A/B/C diagnostics, but
        # use collision-height ray evidence as the primary 2D state.  Ground
        # completion must not masquerade as sensor visibility.
        legacy_free = np.maximum(ray_free, filled_ground).astype(np.uint8)
        static_occupied = evidence['obstacle'].astype(np.uint8)

        state = _compose_state(
            collision_band_free, static_occupied, instance_occupied)
        legacy_state = _compose_state(
            legacy_free, static_occupied, instance_occupied)
        observed_3d_projection = _project_observed_state(
            observed_state_3d, z_centers, self.collision_z,
            instance_occupied)
        filtered_3d_projection = _project_observed_state(
            filtered_state_3d, z_centers, self.collision_z,
            instance_occupied,
            blocking_unknown_3d=uncertain_obstacle_3d)

        # Do not expose stale IDs where a later state-precedence decision did
        # not leave an annotated instance.
        instance_map[state != INSTANCE_OCCUPIED] = -1
        class_map[state != INSTANCE_OCCUPIED] = -1

        result = dict(
            state=state,
            instance_id=instance_map,
            class_id=class_map,
            sensor_names=np.asarray(sensor_names),
            sensor_origins=np.asarray(sensor_origins, dtype=np.float32),
            point_count=np.int64(points.shape[0]),
            free_voxel_count=np.int64(free_voxels.shape[0]),
            collision_band_free=collision_band_free,
            collision_z=self.collision_z.copy(),
            observed_state_3d=observed_state_3d,
            visibility_mask=visibility_mask,
            occupancy_target=occupancy_target,
            filtered_state_3d=filtered_state_3d,
            filtered_visibility_mask=filtered_visibility_mask,
            filtered_occupancy_target=filtered_occupancy_target,
            unreliable_endpoint_mask=unreliable_endpoint_mask,
            sensor_count_3d=sensor_count_3d,
            free_sensor_count_3d=free_sensor_count_3d,
            occupied_sensor_count_3d=occupied_sensor_count_3d,
            hit_point_count_3d=hit_point_count_3d,
            static_obstacle_3d=static_obstacle_3d,
            raw_obstacle_3d=raw_obstacle_3d,
            instance_endpoint_3d=instance_endpoint_3d,
            ground_endpoint_3d=ground_endpoint_3d,
            observed_ground_3d=ground_endpoint_3d,
            uncertain_obstacle_3d=uncertain_obstacle_3d,
            endpoint_type_3d=endpoint_type_3d,
            endpoint_type_names=ENDPOINT_TYPE_NAMES,
            box_occupied_3d=box_occupied_3d,
            box_instance_id_3d=box_instance_id_3d,
            box_class_id_3d=box_class_id_3d,
            z_centers=z_centers.astype(np.float32),
            volume_layout=np.asarray('ZHW'),
        )
        if return_points:
            # Reuse the exact calibrated, target-frame points used to build
            # the label without storing them in normal label artifacts.
            result['points'] = points.astype(np.float32, copy=False)
        if diagnostics:
            per_sensor_free_3d = []
            per_sensor_hit_3d = []
            for sensor_free, sensor_hits in zip(
                    sensor_free_voxels, sensor_hit_voxels):
                free_xyz = _voxel_mask_xyz(self.builder, sensor_free)
                hit_xyz = _voxel_mask_xyz(self.builder, sensor_hits)
                free_xyz[ego_ignore_xyz] = 0
                hit_xyz[ego_ignore_xyz] = 0
                per_sensor_free_3d.append(_xyz_to_zhw(free_xyz))
                per_sensor_hit_3d.append(_xyz_to_zhw(hit_xyz))
            per_sensor_free_3d = np.stack(
                per_sensor_free_3d, axis=0).astype(np.uint8)
            per_sensor_hit_3d = np.stack(
                per_sensor_hit_3d, axis=0).astype(np.uint8)
            raw_evidence = self.raw_ground_builder.build(
                points, boxes, free_voxels=free_voxels)
            raw_ground = raw_evidence['ground'].astype(np.uint8)
            ray_raw_ground = np.maximum(ray_free, raw_ground).astype(np.uint8)

            result.update(
                ray_free=ray_free,
                raw_ground=raw_ground,
                filled_ground=filled_ground,
                static_occupied=static_occupied,
                instance_occupied=instance_occupied,
                free_sensor_count=free_sensor_count,
                collision_free_sensor_count=collision_free_sensor_count,
                per_sensor_free_3d=per_sensor_free_3d,
                per_sensor_hit_3d=per_sensor_hit_3d,
                state_ray_only=_compose_state(
                    ray_free, static_occupied, instance_occupied),
                state_ray_raw_ground=_compose_state(
                    ray_raw_ground, static_occupied, instance_occupied),
                state_ray_filled_ground=legacy_state,
                state_collision_band=state.copy(),
                state_observed_3d_projection=observed_3d_projection,
                state_filtered_3d_projection=filtered_3d_projection,
            )
        return result


def _output_stem(info: dict) -> str:
    scene = str(info.get('scene_token', 'scene')).replace('/', '__')
    timestamp = float(info.get('timestamp', 0.0))
    return f'{scene}__{timestamp:.6f}'


def _save_preview(path: Path, state: np.ndarray, scale: int = 4):
    # RGB: unknown dark gray, free green, static red, instance blue.
    palette = np.asarray([
        [40, 40, 40],
        [70, 180, 90],
        [225, 80, 70],
        [65, 120, 240],
    ], dtype=np.uint8)
    rgb = palette[state]
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _save_sensor_count_preview(path: Path, sensor_count: np.ndarray,
                               static_occupied: np.ndarray,
                               instance_occupied: np.ndarray,
                               scale: int = 4):
    max_count = max(int(sensor_count.max()), 1)
    normalized = np.round(
        sensor_count.astype(np.float32) / max_count * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    bgr[sensor_count == 0] = (40, 40, 40)
    # Keep occupied evidence recognizable while the free support count uses
    # the heatmap: static is red, annotated instance is blue.
    bgr[static_occupied > 0] = (70, 80, 225)
    bgr[instance_occupied > 0] = (240, 120, 65)
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), bgr)


def _save_label(path: Path, result: dict, info: dict,
                pc_range: Sequence[float], occ_size: Sequence[int]):
    np.savez_compressed(
        path,
        **result,
        timestamp=np.float64(info.get('timestamp', 0.0)),
        token=np.asarray(str(info.get('token', ''))),
        scene_token=np.asarray(str(info.get('scene_token', ''))),
        pc_range=np.asarray(pc_range, dtype=np.float32),
        occ_size=np.asarray(occ_size, dtype=np.int64),
        state_names=np.asarray([
            STATE_NAMES[i] for i in range(len(STATE_NAMES))]),
        observed_state_names=OBSERVED_STATE_NAMES,
    )


def _state_stats(state: np.ndarray) -> str:
    counts = np.bincount(state.reshape(-1), minlength=len(STATE_NAMES))
    total = max(int(state.size), 1)
    return ', '.join(
        f'{STATE_NAMES[i]}={int(counts[i])} ({counts[i] / total:.1%})'
        for i in range(len(STATE_NAMES)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate KL multi-LiDAR OccWorld occupancy labels.')
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument(
        '--out-dir',
        default='outputs/patent_2026_occ/occworld_labels_debug')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num-frames', type=int, default=1)
    parser.add_argument(
        '--indices', type=int, nargs='+',
        help='Explicit frame indices; overrides --start/--num-frames.')
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument('--occ-size', type=int, nargs=3,
                        default=[160, 120, 10],
                        help='Voxel counts in X,Y,Z order.')
    parser.add_argument('--target-frame', choices=['auto', 'FLU', 'RFU'],
                        default='auto')
    parser.add_argument('--collision-z', type=float, nargs=2,
                        default=[0.3, 2.5],
                        help='Height band used by the conservative BEV view.')
    parser.add_argument('--save-preview', action='store_true')
    parser.add_argument('--save-diagnostics', action='store_true',
                        help='Save separated A/B/C/D/E/F/G evidence previews.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    ann_path = _resolve_path(args.ann_file)
    infos, metainfo = _load_infos(ann_path)
    target_frame = args.target_frame
    if target_frame == 'auto':
        target_frame = str(metainfo.get('lidar_coord_frame', 'FLU'))

    if args.indices:
        frame_indices = list(dict.fromkeys(args.indices))
        invalid = [index for index in frame_indices
                   if index < 0 or index >= len(infos)]
        if invalid:
            raise ValueError(
                f'Frame indices out of range for {len(infos)} infos: '
                f'{invalid}')
        frame_description = str(frame_indices)
    else:
        start = max(args.start, 0)
        end = min(start + max(args.num_frames, 0), len(infos))
        if start >= end:
            raise ValueError(
                f'Empty frame range [{start}, {end}) for {len(infos)} infos')
        frame_indices = range(start, end)
        frame_description = f'[{start}, {end})'

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_builder = MultiLidarOccLabelBuilder(
        pc_range=args.pc_range,
        bev_size=args.bev_size,
        occ_size=args.occ_size,
        target_frame=target_frame,
        collision_z=args.collision_z,
    )

    print(f'ann={ann_path}')
    print(f'frames={frame_description}, target_frame={target_frame}')
    print(f'out={out_dir}')

    for frame_index in frame_indices:
        info = infos[frame_index]
        stem = _output_stem(info)
        label_path = out_dir / f'{stem}.npz'
        preview_path = out_dir / f'{stem}.png'
        if label_path.exists() and not args.overwrite:
            print(f'[{frame_index}] skip existing {label_path.name}')
            continue

        result = label_builder.build(
            info, diagnostics=args.save_diagnostics)
        _save_label(
            label_path, result, info, args.pc_range, args.occ_size)
        if args.save_preview:
            _save_preview(preview_path, result['state'])
        if args.save_diagnostics:
            _save_preview(
                out_dir / f'{stem}__A_ray_only.png',
                result['state_ray_only'])
            _save_preview(
                out_dir / f'{stem}__B_ray_raw_ground.png',
                result['state_ray_raw_ground'])
            _save_preview(
                out_dir / f'{stem}__C_ray_filled_ground.png',
                result['state_ray_filled_ground'])
            _save_sensor_count_preview(
                out_dir / f'{stem}__D_free_sensor_count.png',
                result['free_sensor_count'],
                result['static_occupied'], result['instance_occupied'])
            _save_preview(
                out_dir / f'{stem}__E_collision_band.png',
                result['state_collision_band'])
            _save_preview(
                out_dir / f'{stem}__F_observed_3d_projection.png',
                result['state_observed_3d_projection'])
            _save_preview(
                out_dir / f'{stem}__G_filtered_3d_projection.png',
                result['state_filtered_3d_projection'])

        print(
            f'[{frame_index}] sensors={len(result["sensor_names"])} '
            f'points={int(result["point_count"])} '
            f'free_voxels={int(result["free_voxel_count"])}; '
            f'{_state_stats(result["state"])}')
        print(
            f'  3d: shape={result["observed_state_3d"].shape} '
            f'visible={int(result["visibility_mask"].sum())} '
            f'occupied={int(result["occupancy_target"].sum())} '
            f'filtered_visible='
            f'{int(result["filtered_visibility_mask"].sum())} '
            f'filtered_occupied='
            f'{int(result["filtered_occupancy_target"].sum())} '
            f'box_voxels={int(result["box_occupied_3d"].sum())}')
        if args.save_diagnostics:
            fill_added = np.count_nonzero(
                (result['filled_ground'] > 0) &
                (result['raw_ground'] == 0))
            print(
                f'  diagnostics: ray_free={int(result["ray_free"].sum())} '
                f'raw_ground={int(result["raw_ground"].sum())} '
                f'filled_ground={int(result["filled_ground"].sum())} '
                f'fill_added={fill_added} '
                f'collision_band_free='
                f'{int(result["collision_band_free"].sum())}')


if __name__ == '__main__':
    main()
