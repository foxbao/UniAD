#!/usr/bin/env python
"""Add UniAD trajectory labels to KL dataset v2 pkl files.

For each object with a track_id, follows the temporal frame chain and finds
the same track_id in past/future frames. Positions are transformed into the
current LiDAR frame and stored as per-instance fields:

  - gt_fut_traj_locs : list[list[float]]  (future_steps, 2)  — [dx, dy]
        relative displacement from the object's current (x, y) position,
        expressed in the current LiDAR frame.
  - gt_fut_traj_mask : list[bool]  (future_steps,)
        True where the track_id was found in the future frame.
  - gt_track_traj_locs : list[list[float]]
        tracking branch trajectory labels in UniAD stage-1 order:
        nearest past frames first, followed by future frames.
  - gt_track_traj_mask : list[bool]
        True where the track_id was found for the corresponding track step.

Usage:
    python tools/dataset_converters/add_forecasting.py \
        --pkl-path data/kl_8/kl_infos_train.pkl \
        --future-steps 12 --track-past-steps 4 --track-fut-steps 4

    # Multiple files:
    python tools/dataset_converters/add_forecasting.py \
        --pkl-path data/kl_8/kl_infos_train.pkl data/kl_8/kl_infos_val.pkl
"""

import argparse
import numpy as np
import mmcv
from tqdm import tqdm


MAX_DISPLACEMENT = 100.0  # metres; reject per-step displacements beyond this

# KL LiDAR frame is determined by pkl metainfo['lidar_coord_frame'] ∈
# {'FLU', 'RFU'}. LIDAR2EGO is identity for FLU (sensor=ego) and a 90°
# Z-rotation for legacy RFU pkls. The matrix is (re)built per pkl by
# `_lidar2ego_from_frame` inside `add_forecasting_to_pkl`.
def _lidar2ego_from_frame(frame: str) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    if frame == 'RFU':
        M[:2, :2] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    elif frame != 'FLU':
        raise ValueError(f'unknown lidar_coord_frame: {frame!r}')
    return M


def build_indices(infos):
    """Build lookup dicts from a data_list.

    Returns:
        token_to_idx: {token_str: index_in_infos}
        token_tid_to_box: {(token_str, track_id): [x, y, z, ...]}
    """
    token_to_idx = {}
    token_tid_to_box = {}

    for idx, info in enumerate(infos):
        tok = info['token']
        token_to_idx[tok] = idx
        for inst in info.get('instances', []):
            tid = inst.get('track_id', -1)
            if tid < 0:
                continue
            token_tid_to_box[(tok, tid)] = inst['bbox_3d']

    return token_to_idx, token_tid_to_box


def _merge_stats(total, new):
    for key, value in new.items():
        total[key] = total.get(key, 0) + value


def _collect_offsets(infos,
                     token_to_idx,
                     token_tid_to_box,
                     info,
                     tid,
                     curr_xy,
                     global2lidar_curr,
                     direction,
                     steps,
                     curr_scene,
                     lidar2ego):
    locs = []
    mask = []
    stats = dict(found=0, total=0, rejected_scene=0, rejected_range=0)
    target_token = info.get(direction, '')

    for _ in range(steps):
        if not target_token or target_token not in token_to_idx:
            locs.append([0.0, 0.0])
            mask.append(False)
            target_token = ''
            continue

        target_idx = token_to_idx[target_token]
        target_info = infos[target_idx]

        if curr_scene and target_info.get('scene_token', '') != curr_scene:
            locs.append([0.0, 0.0])
            mask.append(False)
            target_token = ''
            stats['rejected_scene'] += 1
            continue

        key = (target_token, tid)
        stats['total'] += 1
        if tid >= 0 and key in token_tid_to_box:
            target_box = token_tid_to_box[key]
            pos_target_lidar = np.array(
                [target_box[0], target_box[1], target_box[2], 1.0],
                dtype=np.float64)

            ego2global_target = np.array(
                target_info['ego2global'], dtype=np.float64)
            lidar2global_target = ego2global_target @ lidar2ego
            pos_global = lidar2global_target @ pos_target_lidar
            pos_curr = global2lidar_curr @ pos_global

            dx = float(pos_curr[0] - curr_xy[0])
            dy = float(pos_curr[1] - curr_xy[1])
            if abs(dx) > MAX_DISPLACEMENT or abs(dy) > MAX_DISPLACEMENT:
                locs.append([0.0, 0.0])
                mask.append(False)
                stats['rejected_range'] += 1
            else:
                locs.append([dx, dy])
                mask.append(True)
                stats['found'] += 1
        else:
            locs.append([0.0, 0.0])
            mask.append(False)

        target_token = target_info.get(direction, '')

    return locs, mask, stats


def compute_forecasting(infos, token_to_idx, token_tid_to_box,
                         future_steps=12, track_past_steps=4,
                         track_fut_steps=None, lidar2ego=None):
    """Add UniAD future and stage-1 tracking trajectory labels."""
    if lidar2ego is None:
        lidar2ego = np.eye(4, dtype=np.float64)
    if track_fut_steps is None:
        track_fut_steps = 4
    future_stats = dict(found=0, total=0, rejected_scene=0, rejected_range=0)
    track_stats = dict(found=0, total=0, rejected_scene=0, rejected_range=0)

    for info in tqdm(infos, desc='Adding forecasting'):
        ego2global_curr = np.array(info['ego2global'], dtype=np.float64)
        lidar2global_curr = ego2global_curr @ lidar2ego
        global2lidar_curr = np.linalg.inv(lidar2global_curr)
        curr_scene = info.get('scene_token', '')

        for inst in info.get('instances', []):
            tid = inst.get('track_id', -1)
            curr_xy = inst['bbox_3d'][:2]  # (x, y) in current LiDAR frame

            locs, mask, stats = _collect_offsets(
                infos, token_to_idx, token_tid_to_box, info, tid, curr_xy,
                global2lidar_curr, 'next', future_steps, curr_scene,
                lidar2ego)
            _merge_stats(future_stats, stats)

            inst['gt_fut_traj_locs'] = locs
            inst['gt_fut_traj_mask'] = mask
            inst.pop('gt_forecasting_locs', None)
            inst.pop('gt_forecasting_mask', None)

            past_locs, past_mask, stats = _collect_offsets(
                infos, token_to_idx, token_tid_to_box, info, tid, curr_xy,
                global2lidar_curr, 'prev', track_past_steps, curr_scene,
                lidar2ego)
            _merge_stats(track_stats, stats)
            fut_locs, fut_mask, stats = _collect_offsets(
                infos, token_to_idx, token_tid_to_box, info, tid, curr_xy,
                global2lidar_curr, 'next', track_fut_steps, curr_scene,
                lidar2ego)
            _merge_stats(track_stats, stats)
            inst['gt_track_traj_locs'] = past_locs + fut_locs
            inst['gt_track_traj_mask'] = past_mask + fut_mask

    return future_stats, track_stats


def add_forecasting_to_pkl(pkl_path, future_steps=12, track_past_steps=4,
                           track_fut_steps=None):
    """Process a single pkl file."""
    pkl_path = str(pkl_path)
    print(f'\n{"="*60}')
    print(f'Processing: {pkl_path}')
    print(f'Future trajectory steps: {future_steps}')
    if track_fut_steps is None:
        track_fut_steps = 4
    print(f'Track trajectory steps: past={track_past_steps} '
          f'future={track_fut_steps}')

    data = mmcv.load(pkl_path)
    infos = data['data_list']
    frame = data.get('metainfo', {}).get('lidar_coord_frame', 'FLU')
    lidar2ego = _lidar2ego_from_frame(frame)
    print(f'Total frames: {len(infos)}  lidar_coord_frame: {frame}')

    # Build indices
    token_to_idx, token_tid_to_box = build_indices(infos)
    print(f'Unique (token, track_id) pairs: {len(token_tid_to_box)}')

    # Compute forecasting
    future_stats, track_stats = compute_forecasting(
        infos,
        token_to_idx,
        token_tid_to_box,
        future_steps=future_steps,
        track_past_steps=track_past_steps,
        track_fut_steps=track_fut_steps,
        lidar2ego=lidar2ego)

    future_hit_rate = future_stats['found'] / max(future_stats['total'],
                                                  1) * 100
    track_hit_rate = track_stats['found'] / max(track_stats['total'], 1) * 100
    print(f'Future track matches: {future_stats["found"]}/'
          f'{future_stats["total"]} ({future_hit_rate:.1f}%)')
    print(f'Track-branch traj matches: {track_stats["found"]}/'
          f'{track_stats["total"]} ({track_hit_rate:.1f}%)')
    print(f'Rejected by scene guard: future={future_stats["rejected_scene"]} '
          f'track={track_stats["rejected_scene"]}')
    print(f'Rejected by range guard (>{int(MAX_DISPLACEMENT)}m): '
          f'future={future_stats["rejected_range"]} '
          f'track={track_stats["rejected_range"]}')

    # Save back
    mmcv.dump(data, pkl_path)
    print(f'Saved to: {pkl_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Add UniAD trajectory labels to KL pkl files.')
    parser.add_argument(
        '--pkl-path', nargs='+', required=True,
        help='Path(s) to v2 pkl files.')
    parser.add_argument(
        '--future-steps', type=int, default=None,
        help='Number of future frames for gt_fut_traj_locs. Default: 12.')
    parser.add_argument(
        '--forecast-steps', type=int, default=None,
        help='Deprecated alias for --future-steps.')
    parser.add_argument(
        '--track-past-steps', type=int, default=4,
        help='Number of past frames for gt_track_traj_locs.')
    parser.add_argument(
        '--track-fut-steps', type=int, default=4,
        help='Number of future frames for gt_track_traj_locs.')
    args = parser.parse_args()
    future_steps = args.future_steps
    if future_steps is None:
        future_steps = (
            args.forecast_steps if args.forecast_steps is not None else 12)

    for p in args.pkl_path:
        add_forecasting_to_pkl(
            p,
            future_steps=future_steps,
            track_past_steps=args.track_past_steps,
            track_fut_steps=args.track_fut_steps)

    print('\nDone.')


if __name__ == '__main__':
    main()
