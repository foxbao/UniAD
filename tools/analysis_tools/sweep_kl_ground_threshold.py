# Copyright (c) OpenMMLab. All rights reserved.
"""Sweep raycast ground-height threshold vs HD-map drivable agreement.

For a fixed set of frames, rebuild the raycast ground mask at several
``raycast_ground_height_threshold`` values and report, per threshold:

* ``ray_px``   -- mean raycast ground area (pixels)
* ``iou``      -- mean IoU with the HD-map drivable mask
* ``ray_only`` -- mean fraction of raycast ground falling OUTSIDE the map
                  (proxy for over-coverage: hard flat ground wrongly called
                  drivable)
* ``map_cov``  -- mean fraction of the HD map that raycast also covers
                  (proxy for under-coverage if the threshold is too tight)

Tightening the threshold should drop ``ray_px`` and ``ray_only`` while
keeping ``map_cov`` high; the knee of that trade-off is the value to pick.

Read-only: builds masks in memory, writes nothing.

Example
-------
    PYTHONPATH=$(pwd) python3 tools/analysis_tools/sweep_kl_ground_threshold.py \
        projects/configs/stage1_track_map_lidar/base_track_drivable_lidar.py \
        --split train --num-frames 30 --stride 10 \
        --thresholds 0.25 0.35 0.45 0.55
"""
import argparse

import numpy as np
from mmcv import Config

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    RaycastDrivableBuilder, build_map_mask)
# Reuse the alignment tool's plumbing so behaviour stays consistent.
from tools.analysis_tools.check_kl_drivable_alignment import (
    build_generator, build_split_dataset, find_generator, import_plugin,
    steps_before_generator)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sweep raycast ground threshold vs HD-map agreement')
    parser.add_argument('config', help='train config path')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--num-frames', type=int, default=30)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument(
        '--thresholds', type=float, nargs='+',
        default=[0.25, 0.35, 0.45, 0.55],
        help='ground_height_threshold values to compare')
    return parser.parse_args()


def builder_with_threshold(gen, threshold):
    """Clone the generator's raycast builder with one threshold overridden.

    We copy every constructor knob the generator already resolved so the
    only thing that changes across the sweep is the ground-height cutoff.
    """
    b = gen.raycast_builder
    return RaycastDrivableBuilder(
        b.pc_range, (b.bev_h, b.bev_w),
        occ_size=tuple(int(v) for v in b.occ_size),
        ground_height_threshold=threshold,
        ground_smooth_radius=b.ground_smooth_radius,
        fill_ground=b.fill_ground,
        ground_fill_radius=b.ground_fill_radius,
        ground_fill_min_neighbors=b.ground_fill_min_neighbors,
        remove_ground_under_obstacle=b.remove_ground_under_obstacle,
        obstacle_min_points_per_voxel=b.obstacle_min_points_per_voxel,
        obstacle_min_component_voxels=b.obstacle_min_component_voxels,
        obstacle_small_component_keep_min_points=(
            b.obstacle_small_component_keep_min_points),
        obstacle_thin_component_min_major_span=(
            b.obstacle_thin_component_min_major_span),
        obstacle_thin_component_max_minor_span=(
            b.obstacle_thin_component_max_minor_span),
        obstacle_thin_component_max_z_span=(
            b.obstacle_thin_component_max_z_span),
        obstacle_thin_component_keep_min_points=(
            b.obstacle_thin_component_keep_min_points),
        obstacle_box_ignore_margin=b.obstacle_box_ignore_margin,
        ego_ignore_range=(
            None if b.ego_ignore_range is None
            else tuple(float(v) for v in b.ego_ignore_range)))


def frame_inputs(dataset, pre, gen, indices):
    """Yield (map_mask, points, boxes) per usable frame."""
    for idx in indices:
        info = dataset.get_data_info(idx)
        if info is None:
            continue
        dataset.pre_pipeline(info)
        results = pre(info)
        if results is None or 'points' not in results:
            continue
        map_mask = build_map_mask(
            gen.drivable_global, gen._ego2global(results),
            gen.point_cloud_range, gen.bev_size)
        pts = gen._points_numpy(results['points'])
        boxes = gen._boxes_numpy(results.get('gt_bboxes_3d'))
        yield map_mask, pts, boxes


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)

    pipeline_cfg = cfg.data[args.split].pipeline
    gen = build_generator(find_generator(pipeline_cfg))
    pre = steps_before_generator(pipeline_cfg)
    dataset = build_split_dataset(cfg, args.split)

    num_total = len(dataset.data_infos)
    indices = list(range(0, num_total, args.stride))[:args.num_frames]

    builders = {t: builder_with_threshold(gen, t) for t in args.thresholds}
    # Cache per-frame inputs once; rebuild only the (cheap) raycast per thr.
    frames = list(frame_inputs(dataset, pre, gen, indices))
    if not frames:
        print('No usable frames; nothing to sweep.')
        return

    print('frames=%d  thresholds=%s\n' % (len(frames), args.thresholds))
    header = ('thr   ray_px   iou    ray_only(out-of-map)   '
              'map_cov(map also seen)')
    print(header)
    print('-' * len(header))
    for t in args.thresholds:
        b = builders[t]
        ray_px, ious, ray_only, map_cov = [], [], [], []
        for map_mask, pts, boxes in frames:
            ray = b.build(pts, boxes)['ground'] > 0
            mp = map_mask > 0
            inter = np.logical_and(mp, ray).sum()
            union = np.logical_or(mp, ray).sum()
            ray_px.append(int(ray.sum()))
            ious.append(float(inter) / float(union) if union else 1.0)
            ray_only.append(
                float(np.logical_and(ray, ~mp).sum()) / float(ray.sum() or 1))
            map_cov.append(float(inter) / float(mp.sum() or 1))
        print('%.2f  %7.0f  %.3f       %.3f               %.3f'
              % (t, np.mean(ray_px), np.mean(ious),
                 np.mean(ray_only), np.mean(map_cov)))


if __name__ == '__main__':
    main()
