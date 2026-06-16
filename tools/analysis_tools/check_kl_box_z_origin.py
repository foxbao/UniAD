# Copyright (c) OpenMMLab. All rights reserved.
"""Empirically verify the box z convention used by KL drivable labels.

``GenerateKLDrivableMapLabels._boxes_numpy`` interprets ``box_z_origin``:

* ``'center'`` -> subtract ``h/2`` from ``tensor[:, 2]`` (assumes the stored
  z is the box *centre*).
* ``'bottom'`` -> leave ``tensor[:, 2]`` as is (assumes it is the box floor).

mmdet3d stores ``LiDARInstance3DBoxes`` bottom-centred, so if the dataset's
stored z is already the floor, ``'center'`` double-shifts every obstacle box
down by ``h/2``. This tool settles the question with data, not assumptions:
for each box it gathers the LiDAR points whose XY falls inside the box
footprint and compares their z distribution against the two candidate floors
(``z`` and ``z - h/2``). Whichever candidate sits near the *bottom* of the
in-box points is the true floor, which tells us the correct ``box_z_origin``.

Read-only: never writes labels, never touches training.

Example
-------
    PYTHONPATH=$(pwd) python3 tools/analysis_tools/check_kl_box_z_origin.py \
        projects/configs/stage1_track_map_lidar/base_track_drivable_lidar.py \
        --split train --num-frames 30
"""
import argparse
import os.path as osp
import importlib

import numpy as np
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Empirically check KL box_z_origin against point clouds')
    parser.add_argument('config', help='train config path')
    parser.add_argument(
        '--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--num-frames', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument(
        '--min-pts', type=int, default=20,
        help='skip boxes with fewer in-footprint LiDAR points')
    return parser.parse_args()


def import_plugin(cfg):
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if getattr(cfg, 'plugin', False):
        if hasattr(cfg, 'plugin_dir'):
            module_dir = osp.dirname(cfg.plugin_dir).split('/')
            importlib.import_module('.'.join(module_dir))


def build_split_dataset(cfg, split):
    from third_party.uniad_mmdet3d.datasets.builder import build_dataset
    ds_cfg = dict(cfg.data[split])
    ds_cfg.setdefault('test_mode', split != 'train')
    return build_dataset(ds_cfg)


def steps_before_generator(pipeline_cfg):
    from mmdet.datasets.pipelines import Compose
    prefix = []
    for step in pipeline_cfg:
        if step.get('type') == 'GenerateKLDrivableMapLabels':
            break
        prefix.append(step)
    return Compose(prefix)


def points_in_footprint(pts_xy, box_arr):
    """Boolean mask of points whose XY lies inside the box's rotated rect."""
    cx, cy, length, width, yaw = (
        box_arr[0], box_arr[1], box_arr[3], box_arr[4], box_arr[6])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rel_x = pts_xy[:, 0] - cx
    rel_y = pts_xy[:, 1] - cy
    local_x = rel_x * cos_y + rel_y * sin_y
    local_y = -rel_x * sin_y + rel_y * cos_y
    return (np.abs(local_x) <= length * 0.5) & (np.abs(local_y) <= width * 0.5)


def analyse_frame(points, boxes, min_pts):
    """Per-box floor residuals for the two box_z_origin candidates.

    Returns lists of (z_pts_min - candidate_floor): a value near 0 means the
    candidate matches the real box floor. ``stored`` uses tensor[:,2] as is
    (box_z_origin='bottom'); ``minus_half_h`` uses z - h/2 ('center').
    """
    res_stored, res_minus = [], []
    pts_xy = points[:, :2]
    pts_z = points[:, 2]
    for box_arr in boxes:
        h = box_arr[5]
        if h <= 0:
            continue
        inside = points_in_footprint(pts_xy, box_arr)
        if inside.sum() < min_pts:
            continue
        z_in = pts_z[inside]
        # Use a low percentile rather than raw min to resist outliers/ground.
        z_floor_obs = np.percentile(z_in, 5)
        z_stored = box_arr[2]
        res_stored.append(z_floor_obs - z_stored)
        res_minus.append(z_floor_obs - (z_stored - h * 0.5))
    return res_stored, res_minus


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)

    pipeline_cfg = cfg.data[args.split].pipeline
    pre = steps_before_generator(pipeline_cfg)
    dataset = build_split_dataset(cfg, args.split)
    num_total = len(dataset.data_infos)
    indices = list(range(0, num_total, args.stride))[:args.num_frames]

    all_stored, all_minus = [], []
    n_boxes = 0
    for idx in indices:
        info = dataset.get_data_info(idx)
        if info is None:
            continue
        dataset.pre_pipeline(info)
        results = pre(info)
        if results is None or 'points' not in results:
            continue
        pts = results['points']
        pts = pts.tensor.numpy() if hasattr(pts, 'tensor') else np.asarray(pts)
        boxes = results.get('gt_bboxes_3d')
        if boxes is None:
            continue
        boxes = boxes.tensor.numpy() if hasattr(boxes, 'tensor') \
            else np.asarray(boxes)
        if boxes.size == 0:
            continue
        rs, rm = analyse_frame(pts.astype(np.float32),
                               boxes.astype(np.float32), args.min_pts)
        all_stored.extend(rs)
        all_minus.extend(rm)
        n_boxes += len(rs)

    if n_boxes == 0:
        print('No boxes with enough in-footprint points; try a larger '
              '--num-frames or smaller --min-pts.')
        return

    stored = np.asarray(all_stored)
    minus = np.asarray(all_minus)
    print('=== box z convention check over %d boxes ===' % n_boxes)
    print('Residual = (5th-pct z of in-footprint points) - candidate floor.')
    print('A floor that matches the data has residual near 0 (points start '
          'at the floor); a positive residual ~h/2 means that candidate sits '
          'half a box too low.\n')
    print('candidate            mean     median   std    |median|')
    print("box_z_origin='bottom' %7.3f %7.3f %6.3f  %6.3f"
          % (stored.mean(), np.median(stored), stored.std(),
             abs(np.median(stored))))
    print("box_z_origin='center' %7.3f %7.3f %6.3f  %6.3f"
          % (minus.mean(), np.median(minus), minus.std(),
             abs(np.median(minus))))
    winner = ('bottom' if abs(np.median(stored)) <= abs(np.median(minus))
              else 'center')
    print("\n=> data favours box_z_origin='%s' "
          "(its floor sits closest to where points actually start)." % winner)


if __name__ == '__main__':
    main()
