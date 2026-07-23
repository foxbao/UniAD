#!/usr/bin/env python
"""Generate a reference-aligned KL OccWorld sequence-label prototype."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.analysis_tools.audit_kl_occworld_dual_representation import (
    NAVIGABILITY_STATE_NAMES,
    TRAVERSABILITY_STATE_NAMES,
    _compose_navigability,
    _save_state,
    _tile,
)
from tools.analysis_tools.audit_kl_occworld_occlusion import (
    _ray_blocked_by_occupied,
    _zhw_to_xyz,
)
from tools.data_converter.generate_kl_occworld_labels import (
    FREE,
    INSTANCE_OCCUPIED,
    STATIC_OCCUPIED,
    UNKNOWN,
    MultiLidarOccLabelBuilder,
    _load_infos,
    _output_stem,
    _project_observed_state,
    _resolve_path,
    _xyz_to_zhw,
)
from tools.data_converter.generate_kl_occworld_temporal_labels import (
    _warp_mask_to_reference_xyz,
)


WORLD_STATE_NAMES = np.asarray([
    'unknown', 'free', 'static_occupied', 'instance_occupied'])
COMPLETION_SOURCE_NAMES = np.asarray([
    'unknown', 'direct_observation', 'future_repeated_free',
    'future_repeated_static'])

OCCUPANCY_PALETTE = np.asarray([
    [40, 40, 40], [70, 180, 90],
    [225, 80, 70], [65, 120, 240],
], dtype=np.uint8)
NAVIGABILITY_PALETTE = np.asarray([
    [40, 40, 40], [55, 195, 100], [230, 75, 65],
], dtype=np.uint8)


def _correct_free_visibility_3d(label: dict,
                                pc_range: Sequence[float]) -> Tuple[
                                    np.ndarray, np.ndarray, np.ndarray]:
    """Build full-height K state from per-sensor rays and blockers."""
    state = label['filtered_state_3d'].astype(np.uint8, copy=True)
    per_sensor_free = label['per_sensor_free_3d'] > 0
    occupied_zhw = (
        (label['filtered_occupancy_target'] > 0) |
        (label['box_occupied_3d'] > 0))
    occupied_xyz = _zhw_to_xyz(occupied_zhw)
    valid_count = np.zeros_like(state, dtype=np.uint8)
    blocked_count = np.zeros_like(state, dtype=np.uint8)
    for sensor_index, sensor_origin in enumerate(label['sensor_origins']):
        active = per_sensor_free[sensor_index] & (state == FREE)
        for z_index, row, col in np.argwhere(active):
            xyz_index = np.asarray([
                col, state.shape[1] - 1 - row, z_index,
            ], dtype=np.int64)
            if _ray_blocked_by_occupied(
                    xyz_index, sensor_origin, occupied_xyz, pc_range):
                blocked_count[z_index, row, col] += 1
            else:
                valid_count[z_index, row, col] += 1
    state[(state == FREE) & (valid_count == 0)] = UNKNOWN
    state[label['box_occupied_3d'] > 0] = INSTANCE_OCCUPIED
    return state, valid_count, blocked_count


def _warp_state_to_reference(
        state_3d: np.ndarray,
        source_ego2global: np.ndarray,
        reference_ego2global: np.ndarray,
        pc_range: Sequence[float],
        occ_size: Sequence[int]) -> np.ndarray:
    """Warp a categorical ZHW state with occupied precedence."""
    warped = np.full(state_3d.shape, UNKNOWN, dtype=np.uint8)
    for state_value in (FREE, STATIC_OCCUPIED, INSTANCE_OCCUPIED):
        xyz = _warp_mask_to_reference_xyz(
            state_3d == state_value,
            source_ego2global,
            reference_ego2global,
            pc_range,
            occ_size,
            radius_xy=0,
            radius_z=0,
        )
        warped[_xyz_to_zhw(xyz)] = state_value
    return warped


def _compose_world_target(
        direct_state: np.ndarray,
        future_free_count: np.ndarray,
        future_static_count: np.ndarray,
        min_free_frames: int = 2,
        min_static_frames: int = 2
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Fill only direct unknowns using consistent future static/free evidence."""
    if not (direct_state.shape == future_free_count.shape ==
            future_static_count.shape):
        raise ValueError('World target inputs must have equal shape')
    if min_free_frames < 2 or min_static_frames < 2:
        raise ValueError('Repeated-evidence thresholds must be at least 2')
    world = direct_state.astype(np.uint8, copy=True)
    completion_source = np.zeros(direct_state.shape, dtype=np.uint8)
    completion_source[direct_state != UNKNOWN] = 1
    unknown = direct_state == UNKNOWN
    repeated_free = future_free_count >= min_free_frames
    repeated_static = future_static_count >= min_static_frames
    # Any evidence of the opposite state leaves the voxel ignored. This is
    # deliberately stricter than choosing a fixed occupied/free precedence.
    fill_free = unknown & repeated_free & (future_static_count == 0)
    fill_static = unknown & repeated_static & (future_free_count == 0)
    world[fill_free] = FREE
    world[fill_static] = STATIC_OCCUPIED
    completion_source[fill_free] = 2
    completion_source[fill_static] = 3
    return world, completion_source


def _same_scene_indices(infos: List[dict], reference_index: int,
                        offsets: Sequence[int]) -> List[int]:
    scene = infos[reference_index].get('scene_token')
    indices = []
    for offset in offsets:
        index = reference_index + int(offset)
        if index < 0 or index >= len(infos):
            raise ValueError(f'Frame offset {offset} is outside the dataset')
        if infos[index].get('scene_token') != scene:
            raise ValueError(
                f'Frame offset {offset} crosses the reference scene')
        indices.append(index)
    return indices


def _validate_fixed_frame_times(
        infos: List[dict], reference_index: int,
        target_offsets: Sequence[int], reveal_offsets: Sequence[int],
        expected_step_s: float, max_time_error_s: float):
    """Reject index windows whose timestamps do not match fixed horizons."""
    if expected_step_s <= 0 or max_time_error_s < 0:
        raise ValueError('Invalid fixed-time validation parameters')
    reference_time = float(infos[reference_index]['timestamp'])
    errors = []
    for target_offset in target_offsets:
        target_index = reference_index + target_offset
        actual = float(infos[target_index]['timestamp']) - reference_time
        expected = target_offset * expected_step_s
        if abs(actual - expected) > max_time_error_s:
            errors.append(
                f'target offset {target_offset}: actual={actual:.3f}s '
                f'expected={expected:.3f}s')
        target_time = float(infos[target_index]['timestamp'])
        for reveal_offset in reveal_offsets:
            reveal_index = target_index + reveal_offset
            reveal_actual = (
                float(infos[reveal_index]['timestamp']) - target_time)
            reveal_expected = reveal_offset * expected_step_s
            if abs(reveal_actual - reveal_expected) > max_time_error_s:
                errors.append(
                    f'target {target_offset} reveal {reveal_offset}: '
                    f'actual={reveal_actual:.3f}s '
                    f'expected={reveal_expected:.3f}s')
    if errors:
        raise ValueError(
            f'Reference {reference_index} has an irregular timestamp window: '
            + '; '.join(errors))


def _validate_frame_offsets(
        infos: List[dict], reference_index: int, offsets: Sequence[int],
        expected_step_s: float, max_time_error_s: float,
        window_name: str = 'frame'):
    """Validate arbitrary positive or negative offsets against timestamps."""
    if expected_step_s <= 0 or max_time_error_s < 0:
        raise ValueError('Invalid fixed-time validation parameters')
    reference_time = float(infos[reference_index]['timestamp'])
    errors = []
    for offset in offsets:
        index = reference_index + int(offset)
        if index < 0 or index >= len(infos):
            errors.append(f'{window_name} offset {offset}: outside dataset')
            continue
        actual = float(infos[index]['timestamp']) - reference_time
        expected = float(offset) * expected_step_s
        if abs(actual - expected) > max_time_error_s:
            errors.append(
                f'{window_name} offset {offset}: actual={actual:.3f}s '
                f'expected={expected:.3f}s')
    if errors:
        raise ValueError(
            f'Reference {reference_index} has an irregular {window_name} '
            'timestamp window: ' + '; '.join(errors))


def _observation_cache_index(cache_dir: Path) -> Dict[int, Path]:
    mapping = {}
    for path in cache_dir.glob('*__observation.npz'):
        with np.load(path, allow_pickle=False) as cached:
            mapping[int(cached['frame_index'])] = path
    return mapping


def _project_world_to_bev(state_3d: np.ndarray,
                          z_centers: np.ndarray,
                          collision_z: Sequence[float]) -> np.ndarray:
    keep = ((z_centers >= collision_z[0]) &
            (z_centers <= collision_z[1]))
    instance_bev = np.any(
        state_3d[keep] == INSTANCE_OCCUPIED, axis=0).astype(np.uint8)
    observed = state_3d.copy()
    observed[observed == INSTANCE_OCCUPIED] = UNKNOWN
    return _project_observed_state(
        observed, z_centers, collision_z, instance_bev)


def _save_completion_preview(path: Path, state: np.ndarray,
                             source: np.ndarray, scale: int = 4):
    rgb = OCCUPANCY_PALETTE[state]
    rgb[source == 2] = (55, 215, 215)
    rgb[source == 3] = (235, 190, 55)
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _save_sequence_contact(path: Path, world_paths: Sequence[Path],
                           completion_paths: Sequence[Path],
                           navigability_paths: Sequence[Path],
                           target_times: Sequence[float]):
    rows = []
    specifications = (
        (world_paths, 'world occupancy'),
        (completion_paths, 'completion: cyan free / yellow static'),
        (navigability_paths, 'navigability'),
    )
    for paths, title in specifications:
        rows.append(np.concatenate([
            _tile(cv2.imread(str(item)),
                  f'{title} | t={time_value:.1f}s', (300, 225))
            for item, time_value in zip(paths, target_times)
        ], axis=1))
    separator = np.full((8, rows[0].shape[1], 3), 28, dtype=np.uint8)
    canvas = rows[0]
    for row in rows[1:]:
        canvas = np.concatenate([canvas, separator, row], axis=0)
    cv2.imwrite(str(path), canvas)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--reference-index', type=int, required=True)
    parser.add_argument('--target-offsets', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--reveal-offsets', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--min-free-frames', type=int, default=2)
    parser.add_argument('--min-static-frames', type=int, default=2)
    parser.add_argument('--expected-step-s', type=float, default=0.5)
    parser.add_argument('--max-time-error-s', type=float, default=0.2)
    parser.add_argument(
        '--reference-dual-label', required=True,
        help='Cross-scene NPZ containing current occupancy/traversability.')
    parser.add_argument(
        '--observation-cache-dir',
        help='Optional directory of precomputed full-3D K observations.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument('--occ-size', type=int, nargs=3,
                        default=[160, 120, 10])
    parser.add_argument('--collision-z', type=float, nargs=2,
                        default=[0.3, 2.5])
    return parser.parse_args()


def generate_sequence(args, infos: List[dict], metainfo: dict,
                      builder=None,
                      observation_cache: Dict[int, Path] = None):
    """Generate one sequence while allowing batch callers to reuse state."""
    target_offsets = list(dict.fromkeys(args.target_offsets))
    reveal_offsets = sorted(set(args.reveal_offsets))
    if 0 not in reveal_offsets or any(offset < 0 for offset in reveal_offsets):
        raise ValueError('reveal-offsets must be non-negative and include 0')
    target_indices = _same_scene_indices(
        infos, args.reference_index, target_offsets)
    required_offsets = sorted(set(
        target_offset + reveal_offset
        for target_offset in target_offsets
        for reveal_offset in reveal_offsets))
    required_indices = _same_scene_indices(
        infos, args.reference_index, required_offsets)
    _validate_fixed_frame_times(
        infos, args.reference_index, target_offsets, reveal_offsets,
        args.expected_step_s, args.max_time_error_s)
    offset_to_index = dict(zip(required_offsets, required_indices))
    reference_ego2global = np.asarray(
        infos[args.reference_index]['ego2global'], dtype=np.float64)
    target_frame = str(metainfo.get('lidar_coord_frame', 'FLU'))
    if builder is None:
        builder = MultiLidarOccLabelBuilder(
            args.pc_range, args.bev_size, args.occ_size,
            target_frame=target_frame, collision_z=args.collision_z)
    if observation_cache is None:
        observation_cache = (
            _observation_cache_index(Path(args.observation_cache_dir))
            if args.observation_cache_dir else {})

    corrected_cache: Dict[int, dict] = {}
    for position, index in enumerate(required_indices, start=1):
        if index in observation_cache:
            with np.load(
                    observation_cache[index], allow_pickle=False) as cached:
                state = cached['observation_state_3d']
                valid_count = cached['valid_free_sensor_count_3d']
                blocked_count = cached['blocked_free_sensor_count_3d']
                if not np.allclose(cached['pc_range'], args.pc_range):
                    raise ValueError(
                        f'Cache pc_range mismatch for frame {index}')
                if not np.array_equal(cached['occ_size'], args.occ_size):
                    raise ValueError(
                        f'Cache occ_size mismatch for frame {index}')
            source = 'cache'
        else:
            current = builder.build(infos[index], diagnostics=True)
            state, valid_count, blocked_count = (
                _correct_free_visibility_3d(current, args.pc_range))
            source = 'built'
        corrected_cache[index] = {
            'state': state,
            'valid_count': valid_count,
            'blocked_count': blocked_count,
        }
        print(
            f'[{position}/{len(required_indices)}] frame={index} '
            f'source={source} '
            f'known={np.count_nonzero(state):,} '
            f'free={np.count_nonzero(state == FREE):,}')

    direct_sequence = []
    world_sequence = []
    completion_sequence = []
    free_count_sequence = []
    static_count_sequence = []
    summary_rows = []
    for target_offset, target_index in zip(target_offsets, target_indices):
        target_ego2global = np.asarray(
            infos[target_index]['ego2global'], dtype=np.float64)
        direct = _warp_state_to_reference(
            corrected_cache[target_index]['state'],
            target_ego2global, reference_ego2global,
            args.pc_range, args.occ_size)
        free_count = np.zeros(direct.shape, dtype=np.uint8)
        static_count = np.zeros(direct.shape, dtype=np.uint8)
        for reveal_offset in reveal_offsets:
            reveal_index = offset_to_index[target_offset + reveal_offset]
            reveal_state = corrected_cache[reveal_index]['state']
            reveal_ego2global = np.asarray(
                infos[reveal_index]['ego2global'], dtype=np.float64)
            warped = _warp_state_to_reference(
                reveal_state, reveal_ego2global, reference_ego2global,
                args.pc_range, args.occ_size)
            free_count += (warped == FREE).astype(np.uint8)
            static_count += (warped == STATIC_OCCUPIED).astype(np.uint8)
        world, completion = _compose_world_target(
            direct, free_count, static_count,
            min_free_frames=args.min_free_frames,
            min_static_frames=args.min_static_frames)
        direct_sequence.append(direct)
        world_sequence.append(world)
        completion_sequence.append(completion)
        free_count_sequence.append(free_count)
        static_count_sequence.append(static_count)
        summary_rows.append({
            'target_offset': target_offset,
            'target_index': target_index,
            'target_time_s': float(
                infos[target_index]['timestamp'] -
                infos[args.reference_index]['timestamp']),
            'direct_known_voxels': int(np.count_nonzero(direct)),
            'world_known_voxels': int(np.count_nonzero(world)),
            'future_free_filled_voxels': int(np.count_nonzero(
                completion == 2)),
            'future_static_filled_voxels': int(np.count_nonzero(
                completion == 3)),
            'conflicting_unknown_voxels': int(np.count_nonzero(
                (direct == UNKNOWN) & (free_count > 0) &
                (static_count > 0))),
        })

    direct_sequence = np.stack(direct_sequence, axis=0)
    world_sequence = np.stack(world_sequence, axis=0)
    completion_sequence = np.stack(completion_sequence, axis=0)
    free_count_sequence = np.stack(free_count_sequence, axis=0)
    static_count_sequence = np.stack(static_count_sequence, axis=0)
    target_times = np.asarray(
        [row['target_time_s'] for row in summary_rows], dtype=np.float32)
    target_ego2global = np.asarray([
        infos[index]['ego2global'] for index in target_indices
    ], dtype=np.float64)
    target_to_reference = np.asarray([
        np.linalg.inv(reference_ego2global) @ transform
        for transform in target_ego2global
    ], dtype=np.float64)
    target_timestamps = np.asarray([
        infos[index]['timestamp'] for index in target_indices
    ], dtype=np.float64)

    with np.load(args.reference_dual_label, allow_pickle=False) as dual:
        current_observation = dual['occupancy_state_3d']
        traversability = dual[
            'conservative_traversability_state_bev']
        traversability_valid = dual[
            'conservative_traversability_valid_bev']
    traversability_sequence = np.repeat(
        traversability[None], len(target_offsets), axis=0)
    traversability_valid_sequence = np.repeat(
        traversability_valid[None], len(target_offsets), axis=0)
    world_bev_sequence = np.stack([
        _project_world_to_bev(state, builder.builder.voxel_centers(
            2, np.arange(args.occ_size[2])), args.collision_z)
        for state in world_sequence
    ], axis=0)
    navigability = []
    navigability_valid = []
    for trav, world_bev in zip(
            traversability_sequence, world_bev_sequence):
        state, valid = _compose_navigability(trav, world_bev)
        navigability.append(state)
        navigability_valid.append(valid)
    navigability = np.stack(navigability, axis=0)
    navigability_valid = np.stack(navigability_valid, axis=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(infos[args.reference_index])
    output_path = out_dir / f'{stem}__occworld_sequence.npz'
    np.savez_compressed(
        output_path,
        current_observation_state_3d=current_observation,
        current_observation_valid_3d=(
            current_observation != UNKNOWN).astype(np.uint8),
        direct_observation_state_3d=direct_sequence,
        direct_observation_valid_3d=(
            direct_sequence != UNKNOWN).astype(np.uint8),
        world_target_state_3d=world_sequence,
        world_target_valid_3d=(world_sequence != UNKNOWN).astype(np.uint8),
        completion_source_3d=completion_sequence,
        future_free_count_3d=free_count_sequence,
        future_static_count_3d=static_count_sequence,
        world_state_names=WORLD_STATE_NAMES,
        completion_source_names=COMPLETION_SOURCE_NAMES,
        world_target_state_bev=world_bev_sequence,
        traversability_state_bev=traversability_sequence,
        traversability_valid_bev=traversability_valid_sequence,
        traversability_state_names=TRAVERSABILITY_STATE_NAMES,
        navigability_state_bev=navigability,
        navigability_valid_bev=navigability_valid,
        navigability_state_names=NAVIGABILITY_STATE_NAMES,
        target_offsets=np.asarray(target_offsets, dtype=np.int16),
        target_indices=np.asarray(target_indices, dtype=np.int64),
        target_times_s=target_times,
        nominal_target_times_s=(
            np.asarray(target_offsets, dtype=np.float32) *
            np.float32(args.expected_step_s)),
        target_timestamps=target_timestamps,
        target_ego2global=target_ego2global,
        target_to_reference=target_to_reference,
        reveal_offsets=np.asarray(reveal_offsets, dtype=np.int16),
        reference_index=np.int64(args.reference_index),
        reference_ego2global=reference_ego2global,
        pc_range=np.asarray(args.pc_range, dtype=np.float32),
        occ_size=np.asarray(args.occ_size, dtype=np.int16),
        collision_z=np.asarray(args.collision_z, dtype=np.float32),
    )

    world_paths = []
    completion_paths = []
    nav_paths = []
    for sequence_index, row in enumerate(summary_rows):
        suffix = f't{sequence_index}_{row["target_time_s"]:.1f}s'
        world_path = out_dir / f'{stem}__world_{suffix}.png'
        completion_path = out_dir / f'{stem}__completion_{suffix}.png'
        nav_path = out_dir / f'{stem}__navigability_{suffix}.png'
        _save_state(
            world_path, world_bev_sequence[sequence_index],
            OCCUPANCY_PALETTE)
        collision_keep = (
            (builder.builder.voxel_centers(
                2, np.arange(args.occ_size[2])) >= args.collision_z[0]) &
            (builder.builder.voxel_centers(
                2, np.arange(args.occ_size[2])) <= args.collision_z[1]))
        completion_bev = np.max(
            completion_sequence[sequence_index][collision_keep], axis=0)
        _save_completion_preview(
            completion_path, world_bev_sequence[sequence_index],
            completion_bev)
        _save_state(
            nav_path, navigability[sequence_index],
            NAVIGABILITY_PALETTE)
        world_paths.append(world_path)
        completion_paths.append(completion_path)
        nav_paths.append(nav_path)
    contact_path = out_dir / f'{stem}__sequence_contact_sheet.png'
    _save_sequence_contact(
        contact_path, world_paths, completion_paths, nav_paths, target_times)

    summary = {
        'reference_index': args.reference_index,
        'stem': stem,
        'sequence_shape': list(world_sequence.shape),
        'target_offsets': target_offsets,
        'target_times_s': target_times.tolist(),
        'reveal_offsets': reveal_offsets,
        'min_free_frames': args.min_free_frames,
        'min_static_frames': args.min_static_frames,
        'expected_step_s': args.expected_step_s,
        'max_time_error_s': args.max_time_error_s,
        'targets': summary_rows,
    }
    with (out_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'label={output_path}')
    print(f'contact={contact_path}')
    return {
        'summary': summary,
        'label_path': output_path,
        'contact_path': contact_path,
    }


def main():
    args = parse_args()
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    generate_sequence(args, infos, metainfo)


if __name__ == '__main__':
    main()
