#!/usr/bin/env python
"""Add self-driving-car (SDC) annotations to KL dataset pkl files.

The generated fields are dormant until a config explicitly collects them:

  - gt_sdc_bbox: [[x, y, z, l, w, h, yaw, vx, vy]]
  - gt_sdc_label: [label_id]  (post label_mapping if --sdc-label-mapping given)
  - gt_sdc_fut_traj: [[[x, y], ...]]
  - gt_sdc_fut_traj_mask: [[[1, 1], ...]]

The future trajectory is the ego vehicle origin in future frames, transformed
into the current LiDAR frame.  This matches the local-coordinate convention
used by UniAD's SDC trajectory labels.

Defaults are tuned for the KL dataset (large IGV ego vehicle, ~14.6 x 3.0 x
2.16 m, LiDAR mounted near ground level so box center sits at z=0.98).
"""

import argparse
import os
from pathlib import Path

import mmcv
import numpy as np
from tqdm import tqdm


def _get_infos(data):
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, list):
        return data
    raise KeyError('Expected pkl to contain "data_list" or "infos".')


def _get_metainfo(data):
    if not isinstance(data, dict):
        return {}
    return data.get('metainfo', data.get('metadata', {}))


def _lidar2ego_from_frame(frame: str) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    if frame == 'RFU':
        mat[:2, :2] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    elif frame != 'FLU':
        raise ValueError(f'unknown lidar_coord_frame: {frame!r}')
    return mat


def _label_from_name(metainfo, label_name, label_id, label_mapping=None):
    """Resolve dataset class id, then optionally remap through label_mapping.

    KL configs collapse 15 raw classes to 13 trained classes via label_mapping.
    The dataset applies that mapping when emitting GT labels, so the SDC label
    we store must be the *post-mapping* id to stay consistent.
    """
    if label_id is not None:
        raw_id = int(label_id)
    elif isinstance(metainfo.get('categories'), dict) and label_name in metainfo['categories']:
        raw_id = int(metainfo['categories'][label_name])
    else:
        classes = metainfo.get('classes', None)
        if isinstance(classes, (list, tuple)) and label_name in classes:
            raw_id = int(classes.index(label_name))
        elif label_name == 'IGV-Empty':
            raw_id = 6
        elif label_name == 'Car':
            raw_id = 1
        else:
            raise ValueError(
                f'Could not infer label id for {label_name!r}; pass --sdc-label-id.')

    if label_mapping is not None and raw_id < len(label_mapping):
        return int(label_mapping[raw_id])
    return raw_id


def _localization_valid(info):
    loc = info.get('sync_info', {}).get('localization', None)
    if loc is None:
        return True
    return bool(loc.get('valid', True))


def _ego2global(info):
    return np.asarray(info.get('ego2global', np.eye(4)), dtype=np.float64)


def _global_origin(info):
    return _ego2global(info)[:3, 3]


def _global_to_lidar_matrix(info, lidar2ego):
    lidar2global = _ego2global(info) @ lidar2ego
    return np.linalg.inv(lidar2global)


def _velocity_xy(info,
                 token_to_info,
                 lidar2ego,
                 min_dt,
                 max_time_diff,
                 max_speed,
                 require_valid_localization):
    """Estimate SDC velocity in the current LiDAR frame."""
    if require_valid_localization and not _localization_valid(info):
        return np.zeros(2, dtype=np.float32), 'invalid_localization'

    curr_scene = info.get('scene_token', '')
    curr_time = float(info.get('timestamp', 0.0))
    curr_global = _global_origin(info)

    candidates = []
    next_token = info.get('next', '')
    if next_token in token_to_info:
        next_info = token_to_info[next_token]
        candidates.append((info, next_info))

    prev_token = info.get('prev', '')
    if prev_token in token_to_info:
        prev_info = token_to_info[prev_token]
        candidates.append((prev_info, info))

    lidar2global = _ego2global(info) @ lidar2ego
    global2lidar_rot = np.linalg.inv(lidar2global[:3, :3])

    for start_info, end_info in candidates:
        if curr_scene and start_info.get('scene_token', '') != curr_scene:
            continue
        if curr_scene and end_info.get('scene_token', '') != curr_scene:
            continue
        if require_valid_localization and (
                not _localization_valid(start_info)
                or not _localization_valid(end_info)):
            continue

        dt = float(end_info.get('timestamp', 0.0)) - float(
            start_info.get('timestamp', 0.0))
        if dt <= min_dt or dt > max_time_diff:
            continue
        vel_global = (_global_origin(end_info) -
                      _global_origin(start_info)) / dt
        if float(np.linalg.norm(vel_global[:2])) > max_speed:
            continue
        vel_lidar = global2lidar_rot @ vel_global
        return vel_lidar[:2].astype(np.float32), 'estimated'

    return np.zeros(2, dtype=np.float32), 'fallback_zero'


def _future_traj(info,
                 token_to_info,
                 lidar2ego,
                 future_steps,
                 max_step_time_diff,
                 max_displacement,
                 require_valid_localization):
    traj = np.zeros((1, future_steps, 2), dtype=np.float32)
    mask = np.zeros((1, future_steps, 2), dtype=np.float32)

    if require_valid_localization and not _localization_valid(info):
        return traj, mask, 'invalid_current_localization'

    curr_scene = info.get('scene_token', '')
    global2lidar_curr = _global_to_lidar_matrix(info, lidar2ego)

    prev_info = info
    future_token = info.get('next', '')
    stop_reason = 'ok'
    for step in range(future_steps):
        if not future_token or future_token not in token_to_info:
            stop_reason = 'chain_end'
            break

        future_info = token_to_info[future_token]
        if curr_scene and future_info.get('scene_token', '') != curr_scene:
            stop_reason = 'scene_boundary'
            break
        if require_valid_localization and not _localization_valid(future_info):
            stop_reason = 'invalid_future_localization'
            break

        dt = float(future_info.get('timestamp', 0.0)) - float(
            prev_info.get('timestamp', 0.0))
        if dt <= 0.0 or dt > max_step_time_diff:
            stop_reason = 'time_gap'
            break

        pos_global = np.concatenate([_global_origin(future_info), [1.0]])
        pos_curr = global2lidar_curr @ pos_global
        xy = pos_curr[:2].astype(np.float32)
        if float(np.linalg.norm(xy)) > max_displacement:
            stop_reason = 'displacement_guard'
            break

        traj[0, step] = xy
        mask[0, step] = 1.0
        prev_info = future_info
        future_token = future_info.get('next', '')

    return traj, mask, stop_reason


def _atomic_dump(data, out_path):
    out_path = Path(out_path)
    mmcv.mkdir_or_exist(str(out_path.parent))
    tmp_path = out_path.with_name(out_path.name + '.tmp.pkl')
    mmcv.dump(data, str(tmp_path))
    os.replace(str(tmp_path), str(out_path))


def add_sdc_to_pkl(pkl_path,
                   out_path=None,
                   in_place=False,
                   future_steps=6,
                   sdc_label_name='IGV-Empty',
                   sdc_label_id=None,
                   sdc_label_mapping=None,
                   sdc_size=(14.6, 3.0, 2.16),
                   sdc_z=0.98,
                   sdc_yaw=0.0,
                   min_dt=1e-3,
                   max_time_diff=1.5,
                   max_step_time_diff=1.5,
                   max_speed=60.0,
                   max_displacement=100.0,
                   require_valid_localization=True):
    pkl_path = Path(pkl_path)
    if in_place:
        out_path = pkl_path
    elif out_path is None:
        out_path = pkl_path.with_name(f'{pkl_path.stem}_with_sdc.pkl')
    else:
        out_path = Path(out_path)

    print(f'\n{"=" * 60}')
    print(f'Processing: {pkl_path}')
    print(f'Output:     {out_path}')
    print(f'Future steps: {future_steps}')

    data = mmcv.load(str(pkl_path))
    infos = _get_infos(data)
    metainfo = _get_metainfo(data)
    frame = metainfo.get('lidar_coord_frame', 'FLU')
    lidar2ego = _lidar2ego_from_frame(frame)
    sdc_label = _label_from_name(metainfo, sdc_label_name, sdc_label_id,
                                 label_mapping=sdc_label_mapping)
    token_to_info = {info['token']: info for info in infos}

    print(f'Total frames: {len(infos)}  lidar_coord_frame: {frame}')
    if sdc_label_mapping is not None:
        print(f'SDC label: {sdc_label_name} -> {sdc_label} (mapped)')
    else:
        print(f'SDC label: {sdc_label_name} -> {sdc_label}')
    print(f'SDC size lwh: {sdc_size}, z: {sdc_z}')

    reason_counts = {}
    velocity_counts = {}
    valid_steps = []
    invalid_localization_frames = 0

    for info in tqdm(infos, desc='Adding SDC'):
        if require_valid_localization and not _localization_valid(info):
            invalid_localization_frames += 1

        vel_xy, vel_reason = _velocity_xy(
            info,
            token_to_info,
            lidar2ego,
            min_dt=min_dt,
            max_time_diff=max_time_diff,
            max_speed=max_speed,
            require_valid_localization=require_valid_localization)
        velocity_counts[vel_reason] = velocity_counts.get(vel_reason, 0) + 1

        bbox = [
            0.0, 0.0, float(sdc_z),
            float(sdc_size[0]), float(sdc_size[1]), float(sdc_size[2]),
            float(sdc_yaw), float(vel_xy[0]), float(vel_xy[1])
        ]
        traj, mask, reason = _future_traj(
            info,
            token_to_info,
            lidar2ego,
            future_steps=future_steps,
            max_step_time_diff=max_step_time_diff,
            max_displacement=max_displacement,
            require_valid_localization=require_valid_localization)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        valid_steps.append(int(mask[0, :, 0].sum()))

        info['gt_sdc_bbox'] = np.asarray([bbox], dtype=np.float32)
        info['gt_sdc_label'] = np.asarray([sdc_label], dtype=np.int64)
        info['gt_sdc_fut_traj'] = traj
        info['gt_sdc_fut_traj_mask'] = mask

    valid_steps = np.asarray(valid_steps, dtype=np.float32)
    print(f'Invalid localization frames: {invalid_localization_frames}')
    print('Future stop reasons:')
    for key in sorted(reason_counts):
        print(f'  {key:30s} {reason_counts[key]}')
    print('Velocity reasons:')
    for key in sorted(velocity_counts):
        print(f'  {key:30s} {velocity_counts[key]}')
    print('Valid future steps per frame: '
          f'mean={valid_steps.mean():.2f}, '
          f'p50={np.percentile(valid_steps, 50):.1f}, '
          f'p95={np.percentile(valid_steps, 95):.1f}, '
          f'max={valid_steps.max():.0f}')

    _atomic_dump(data, out_path)
    print(f'Saved to: {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Add SDC annotations to KL pkl files.')
    parser.add_argument(
        '--pkl-path', nargs='+', required=True,
        help='Path(s) to KL pkl files.')
    parser.add_argument(
        '--out-path', default=None,
        help='Output path. Only valid when one --pkl-path is provided.')
    parser.add_argument(
        '--in-place', action='store_true',
        help='Overwrite input pkl files instead of writing *_with_sdc.pkl.')
    parser.add_argument(
        '--future-steps', type=int, default=6,
        help='Number of future SDC steps to store.')
    parser.add_argument(
        '--sdc-label-name', default='IGV-Empty',
        help='Class name used as SDC label when --sdc-label-id is not set.')
    parser.add_argument(
        '--sdc-label-id', type=int, default=None,
        help='Explicit SDC class id (raw, before label_mapping).')
    parser.add_argument(
        '--sdc-label-mapping', type=int, nargs='+', default=None,
        help='Optional label_mapping list to remap raw class id to trained id.')
    parser.add_argument(
        '--sdc-size', type=float, nargs=3, default=(14.6, 3.0, 2.16),
        metavar=('L', 'W', 'H'),
        help='SDC box size in metres as length width height.')
    parser.add_argument('--sdc-z', type=float, default=0.98)
    parser.add_argument('--sdc-yaw', type=float, default=0.0)
    parser.add_argument('--min-dt', type=float, default=1e-3)
    parser.add_argument('--max-time-diff', type=float, default=1.5)
    parser.add_argument('--max-step-time-diff', type=float, default=1.5)
    parser.add_argument('--max-speed', type=float, default=60.0)
    parser.add_argument('--max-displacement', type=float, default=100.0)
    parser.add_argument(
        '--allow-invalid-localization', action='store_true',
        help='Use ego2global even when sync_info.localization.valid is false.')
    args = parser.parse_args()

    if args.out_path is not None and len(args.pkl_path) != 1:
        raise ValueError('--out-path can only be used with one --pkl-path.')

    for pkl_path in args.pkl_path:
        add_sdc_to_pkl(
            pkl_path,
            out_path=args.out_path,
            in_place=args.in_place,
            future_steps=args.future_steps,
            sdc_label_name=args.sdc_label_name,
            sdc_label_id=args.sdc_label_id,
            sdc_label_mapping=args.sdc_label_mapping,
            sdc_size=tuple(args.sdc_size),
            sdc_z=args.sdc_z,
            sdc_yaw=args.sdc_yaw,
            min_dt=args.min_dt,
            max_time_diff=args.max_time_diff,
            max_step_time_diff=args.max_step_time_diff,
            max_speed=args.max_speed,
            max_displacement=args.max_displacement,
            require_valid_localization=not args.allow_invalid_localization)

    print('\nDone.')


if __name__ == '__main__':
    main()
