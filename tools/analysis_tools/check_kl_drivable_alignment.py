# Copyright (c) OpenMMLab. All rights reserved.
"""Diagnose alignment between the two KL drivable-label sources.

``GenerateKLDrivableMapLabels`` fuses two independently-derived masks into
the seg-head target:

* ``build_map_mask`` -- the static HD-map drivable polygon, rasterised into
  the BEV via the per-frame ``ego2global`` transform.
* ``RaycastDrivableBuilder`` -- the per-frame LiDAR raycast ground estimate,
  computed in the point cloud's own frame.

If ``ego2global`` (or ``box_z_origin``) is even slightly off, the two masks
drift apart and the fused target develops a doubled / smeared boundary that
silently degrades training. This tool renders the two sources *separately*
for a bounded set of frames and reports overlap metrics (IoU, map-only and
raycast-only fractions, centroid offset in metres), optionally dumping an
RGB overlay PNG per frame for visual inspection.

It is read-only: it never writes labels and does not touch training.

Example
-------
    PYTHONPATH=$(pwd) python3 tools/analysis_tools/check_kl_drivable_alignment.py \
        projects/configs/stage1_track_map_lidar/base_track_drivable_lidar.py \
        --split train --num-frames 20 --out-dir /tmp/kl_align
"""
import argparse
import os
import os.path as osp

import numpy as np
from mmcv import Config

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    GenerateKLDrivableMapLabels, build_map_mask)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check KL map vs raycast drivable-label alignment')
    parser.add_argument('config', help='train config path')
    parser.add_argument(
        '--split', default='train', choices=['train', 'val', 'test'],
        help='which data split to pull frames from')
    parser.add_argument(
        '--num-frames', type=int, default=20,
        help='max number of frames to evaluate')
    parser.add_argument(
        '--stride', type=int, default=1,
        help='sample every Nth frame (after the split is built)')
    parser.add_argument(
        '--out-dir', default=None,
        help='if set, write an overlay PNG per frame here')
    parser.add_argument(
        '--save-worst', type=int, default=0,
        help='only dump the N worst-IoU overlays (0 = dump all)')
    return parser.parse_args()


def find_generator(pipeline_cfg):
    """Locate the GenerateKLDrivableMapLabels dict inside a pipeline cfg."""
    for step in pipeline_cfg:
        if step.get('type') == 'GenerateKLDrivableMapLabels':
            return dict(step)
    raise ValueError(
        'No GenerateKLDrivableMapLabels step found in the pipeline; this '
        'tool only applies to KL drivable configs.')


def build_generator(gen_cfg):
    """Instantiate the transform, stripping the registry-only ``type`` key."""
    kwargs = {k: v for k, v in gen_cfg.items() if k != 'type'}
    return GenerateKLDrivableMapLabels(**kwargs)


def mask_metrics(map_mask, ray_mask, voxel_m):
    """Overlap metrics between two binary BEV masks.

    ``voxel_m`` is (metres-per-col, metres-per-row) so the centroid offset
    is reported in physical units rather than pixels.
    """
    map_b = map_mask > 0
    ray_b = ray_mask > 0
    inter = np.logical_and(map_b, ray_b).sum()
    union = np.logical_or(map_b, ray_b).sum()
    map_only = np.logical_and(map_b, ~ray_b).sum()
    ray_only = np.logical_and(ray_b, ~map_b).sum()
    iou = float(inter) / float(union) if union > 0 else 1.0

    def centroid(mask_b):
        if mask_b.sum() == 0:
            return None
        ys, xs = np.nonzero(mask_b)
        return np.array([xs.mean(), ys.mean()], dtype=np.float64)

    c_map = centroid(map_b)
    c_ray = centroid(ray_b)
    if c_map is None or c_ray is None:
        offset_m = float('nan')
    else:
        d = (c_map - c_ray) * np.asarray(voxel_m, dtype=np.float64)
        offset_m = float(np.hypot(d[0], d[1]))

    return dict(
        iou=iou,
        map_area=int(map_b.sum()),
        ray_area=int(ray_b.sum()),
        map_only_frac=float(map_only) / float(map_b.sum() or 1),
        ray_only_frac=float(ray_only) / float(ray_b.sum() or 1),
        centroid_offset_m=offset_m)


def save_overlay(path, map_mask, ray_mask):
    """RGB overlay: red = map-only, green = raycast-only, yellow = both."""
    import cv2
    h, w = map_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    map_b = map_mask > 0
    ray_b = ray_mask > 0
    rgb[..., 2] = (map_b & ~ray_b).astype(np.uint8) * 255   # red   (B,G,R)
    rgb[..., 1] = (ray_b & ~map_b).astype(np.uint8) * 255   # green
    both = map_b & ray_b
    rgb[both] = (0, 255, 255)                               # yellow
    cv2.imwrite(path, rgb)

def import_plugin(cfg):
    """Mirror tools/train.py plugin loading so registries are populated."""
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if getattr(cfg, 'plugin', False):
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            module_dir = osp.dirname(cfg.plugin_dir).split('/')
            module_path = '.'.join(module_dir)
            importlib.import_module(module_path)


def build_split_dataset(cfg, split):
    """Build the dataset for the requested split with its real pipeline."""
    from third_party.uniad_mmdet3d.datasets.builder import build_dataset
    ds_cfg = dict(cfg.data[split])
    ds_cfg.setdefault('test_mode', split != 'train')
    return build_dataset(ds_cfg)


def steps_before_generator(pipeline_cfg):
    """Pipeline steps preceding GenerateKLDrivableMapLabels.

    Running only these gives us a results dict that still holds raw
    ``points`` / ``gt_bboxes_3d`` / ``ego2global`` -- exactly the inputs the
    generator fuses -- without the generator having merged them yet.
    """
    from mmdet.datasets.pipelines import Compose
    prefix = []
    for step in pipeline_cfg:
        if step.get('type') == 'GenerateKLDrivableMapLabels':
            break
        # DefaultFormatBundle / Collect would wrap tensors in containers;
        # they live after the generator anyway, so the break above skips
        # them. Everything before is a plain transform we can replay.
        prefix.append(step)
    return Compose(prefix)


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

    voxel_m = (
        float(gen.raycast_builder.voxel_size[0]),
        float(gen.raycast_builder.voxel_size[1]))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    records = []
    for idx in indices:
        info = dataset.get_data_info(idx)
        if info is None:
            continue
        dataset.pre_pipeline(info)
        results = pre(info)
        if results is None or 'points' not in results:
            continue

        map_mask = build_map_mask(
            gen.drivable_global,
            gen._ego2global(results),
            gen.point_cloud_range,
            gen.bev_size)
        raycast = gen.raycast_builder.build(
            gen._points_numpy(results['points']),
            gen._boxes_numpy(results.get('gt_bboxes_3d')))
        ray_mask = raycast['ground']

        m = mask_metrics(map_mask, ray_mask, voxel_m)
        m['index'] = idx
        m['token'] = results.get('token', str(idx))
        records.append((m, map_mask, ray_mask))
        print(
            f"[{idx:5d}] IoU={m['iou']:.3f}  "
            f"map_only={m['map_only_frac']:.2f}  "
            f"ray_only={m['ray_only_frac']:.2f}  "
            f"offset={m['centroid_offset_m']:.2f}m  "
            f"(map={m['map_area']} ray={m['ray_area']} px)")

    if not records:
        print('No frames produced a valid points mask; nothing to report.')
        return

    ious = np.array([r[0]['iou'] for r in records])
    offs = np.array([r[0]['centroid_offset_m'] for r in records])
    print('\n=== summary over %d frames ===' % len(records))
    print('IoU      mean=%.3f  min=%.3f  median=%.3f'
          % (np.nanmean(ious), np.nanmin(ious), np.nanmedian(ious)))
    print('offset_m mean=%.2f  max=%.2f'
          % (np.nanmean(offs), np.nanmax(offs)))
    low = (ious < 0.5).sum()
    print('frames with IoU < 0.5: %d / %d  (likely misalignment if high)'
          % (low, len(records)))

    if args.out_dir:
        order = list(range(len(records)))
        if args.save_worst > 0:
            order = sorted(order, key=lambda i: records[i][0]['iou'])
            order = order[:args.save_worst]
        for i in order:
            m, map_mask, ray_mask = records[i]
            path = osp.join(
                args.out_dir,
                'align_%05d_iou%.2f.png' % (m['index'], m['iou']))
            save_overlay(path, map_mask, ray_mask)
        print('wrote %d overlay PNG(s) to %s' % (len(order), args.out_dir))


if __name__ == '__main__':
    main()
