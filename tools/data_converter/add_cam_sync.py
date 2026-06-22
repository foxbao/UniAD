#!/usr/bin/env python
"""Add front-camera image paths to KL dataset pkl files (post-processing).

The KL pkl files were built with camera_processing disabled
(projects/KL8/configs/kl8_lidar_bevformer.py: camera_processing_cfg.enable=
False), so sync_info has no 'cameras' entry. This script fills it in by
nearest-timestamp matching each LiDAR frame to the camera_undist images on
disk, mirroring kl_converter.process_cameras' output location
(info['sync_info']['cameras'][<view>]) and entry schema (make_sync_entry).

Scope: front view only by default (see documents/llm_integration_plan.md;
the cross-modal VLM-teacher pipeline starts with the front camera). Adding
more views later is just extending --views.

Like add_sdc.py / add_forecasting.py, this is an incremental post-processor:
it only adds sync_info['cameras'] and leaves all other fields untouched.
"""

import argparse
import glob
import os
import os.path as osp

import mmcv
import numpy as np
from tqdm import tqdm

# Mirrors kl_converter.CAM_NAME_MAP: on-disk view dir -> nuScenes-style name.
CAM_NAME_MAP = {
    'front': 'CAM_FRONT',
    'left_front': 'CAM_FRONT_LEFT',
    'left_rear': 'CAM_BACK_LEFT',
    'rear': 'CAM_BACK',
    'right_front': 'CAM_FRONT_RIGHT',
    'right_rear': 'CAM_BACK_RIGHT',
}


def _get_infos(data):
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, list):
        return data
    raise KeyError('Expected pkl to contain "data_list" or "infos".')


def _view_dir(data_root, sample_prefix, scene_token, view):
    """On-disk camera dir: <root>/<prefix>/<scene_token>/camera_undist/<view>_image."""
    return osp.join(data_root, sample_prefix, scene_token,
                    'camera_undist', f'{view}_image')


def _index_view(view_dir):
    """Glob a view dir once -> (sorted_ts[np], {ts: relpath_from_cwd})."""
    files = glob.glob(osp.join(view_dir, '*.jpg'))
    if not files:
        return np.empty((0,), dtype=np.float64), {}
    ts_to_path = {}
    ts_list = []
    for f in files:
        try:
            ts = float(osp.splitext(osp.basename(f))[0])
        except ValueError:
            continue
        ts_list.append(ts)
        ts_to_path[ts] = f
    ts_arr = np.array(sorted(ts_list), dtype=np.float64)
    return ts_arr, ts_to_path


def _make_cam_entry(nearest_ts, frame_ts, valid, reason=''):
    """Mirror kl_converter.make_sync_entry's schema (offset=0, no correction)."""
    entry = {
        'timestamp': float(nearest_ts) if nearest_ts is not None else None,
        'dt': (float(nearest_ts - frame_ts)
               if nearest_ts is not None else None),
        'offset': 0.0,
        'dt_corrected': (float(nearest_ts - frame_ts)
                         if nearest_ts is not None else None),
        'valid': bool(valid),
    }
    if reason:
        entry['reason'] = reason
    return entry


def add_cam_sync_to_pkl(pkl_path, out_path=None, in_place=False,
                        data_root='data/kl_8/',
                        sample_prefix='v1.0-trainval/sample',
                        views=('front',), max_diff=0.05):
    data = mmcv.load(pkl_path)
    infos = _get_infos(data)

    # Cache per (scene_token, view) so each dir is globbed once.
    index_cache = {}
    gaps = {v: [] for v in views}
    n_valid = {v: 0 for v in views}

    for info in tqdm(infos, desc=osp.basename(pkl_path)):
        scene_token = info.get('scene_token', '')
        frame_ts = float(info.get('timestamp', 0.0))
        sync = info.setdefault('sync_info', {})
        cams = sync.setdefault('cameras', {})
        for view in views:
            key = (scene_token, view)
            if key not in index_cache:
                index_cache[key] = _index_view(
                    _view_dir(data_root, sample_prefix, scene_token, view))
            ts_arr, ts_to_path = index_cache[key]
            cam_name = CAM_NAME_MAP.get(view, view)
            if ts_arr.size == 0:
                cams[cam_name] = _make_cam_entry(
                    None, frame_ts, False, 'view dir empty/missing')
                continue
            j = int(np.argmin(np.abs(ts_arr - frame_ts)))
            nearest_ts = float(ts_arr[j])
            gap = abs(nearest_ts - frame_ts)
            gaps[view].append(gap * 1000.0)
            if gap > max_diff:
                cams[cam_name] = _make_cam_entry(
                    nearest_ts, frame_ts, False, 'exceeds max diff')
                continue
            entry = _make_cam_entry(nearest_ts, frame_ts, True)
            entry['path'] = osp.relpath(ts_to_path[nearest_ts])
            cams[cam_name] = entry
            n_valid[view] += 1

    _report(pkl_path, len(infos), views, gaps, n_valid, max_diff)
    _write(data, pkl_path, out_path, in_place)
    return data


def _report(pkl_path, n, views, gaps, n_valid, max_diff):
    print(f'[{osp.basename(pkl_path)}] {n} frames, max_diff={max_diff*1000:.0f}ms')
    for view in views:
        g = np.array(gaps[view]) if gaps[view] else np.empty((0,))
        cov = 100.0 * n_valid[view] / max(n, 1)
        line = (f'  {view}: valid={n_valid[view]}/{n} ({cov:.2f}%)')
        if g.size:
            line += (f'  gap ms median={np.median(g):.1f} '
                     f'p95={np.percentile(g, 95):.1f} max={g.max():.1f}')
        print(line)


def _write(data, pkl_path, out_path, in_place):
    if in_place:
        dst = pkl_path
    elif out_path is not None:
        dst = out_path
    else:
        root, ext = osp.splitext(pkl_path)
        dst = f'{root}_with_cam{ext}'
    mmcv.dump(data, dst)
    print(f'  -> wrote {dst}')


def main():
    parser = argparse.ArgumentParser(
        description='Add front-camera image paths to KL pkl files.')
    parser.add_argument('--pkl-path', nargs='+', required=True,
                        help='Path(s) to KL pkl files.')
    parser.add_argument('--out-path', default=None,
                        help='Output path (only with a single --pkl-path).')
    parser.add_argument('--in-place', action='store_true',
                        help='Overwrite input instead of *_with_cam.pkl.')
    parser.add_argument('--data-root', default='data/kl_8/')
    parser.add_argument('--sample-prefix', default='v1.0-trainval/sample')
    parser.add_argument('--views', nargs='+', default=['front'],
                        choices=list(CAM_NAME_MAP.keys()),
                        help='Camera views to sync (default: front only).')
    parser.add_argument('--max-diff', type=float, default=0.05,
                        help='Max |lidar-cam| gap in seconds (default 0.05).')
    args = parser.parse_args()

    if args.out_path is not None and len(args.pkl_path) != 1:
        raise ValueError('--out-path can only be used with one --pkl-path.')
    for pkl_path in args.pkl_path:
        add_cam_sync_to_pkl(
            pkl_path, out_path=args.out_path, in_place=args.in_place,
            data_root=args.data_root, sample_prefix=args.sample_prefix,
            views=tuple(args.views), max_diff=args.max_diff)


if __name__ == '__main__':
    main()



