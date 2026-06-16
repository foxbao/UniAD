# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize the fused KL drivable seg-head ground-truth.

Renders the target that ``GenerateKLDrivableMapLabels`` feeds to the
Pansegformer seg-head, for a bounded set of frames. Each frame is a 2x2
grid; every panel has its own title bar and the colours are explained by
two legend strips along the bottom. Panels read input -> output:

* **D: raw LiDAR (colour = height)** -- the raw point cloud coloured by z
  via a JET colormap (top-left); the model's actual input.
* **A: final drivable GT + LiDAR** -- final drivable GT (green) over a
  LiDAR point-density backdrop (gray), with the ego centre marked (red).
* **B: where the GT comes from** -- source decomposition: red = HD-map only,
  green = raycast-ground only, yellow = both agree, blue = raycast obstacle
  (subtracted from the GT).
* **C: final mask (Dice target)** -- the binary target actually used for the
  Dice loss.

Optionally muxes the frames into a ``.webm`` (libvpx-vp9, matching
run_e2e_subset.py) for browser inspection.

Read-only: renders images, never writes labels or touches training.

Example
-------
    PYTHONPATH=$(pwd) python3 tools/analysis_tools/visualize_kl_drivable_gt.py \
        projects/configs/stage1_track_map_lidar/base_track_drivable_lidar.py \
        --split train --num-frames 40 --stride 5 \
        --out-dir /tmp/kl_gt_vis --video /tmp/kl_gt_vis/drivable_gt.webm
"""
import argparse
import os
import os.path as osp
import subprocess

import cv2
import numpy as np
from mmcv import Config

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    build_map_mask)
from tools.analysis_tools.check_kl_drivable_alignment import (
    build_generator, build_split_dataset, find_generator, import_plugin,
    steps_before_generator)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize fused KL drivable seg-head GT')
    parser.add_argument('config', help='train config path')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--num-frames', type=int, default=40)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--out-dir', required=True,
                        help='folder for rendered PNG frames')
    parser.add_argument('--video', default=None,
                        help='optional .webm output path')
    parser.add_argument('--scale', type=int, default=3,
                        help='nearest-neighbour upscale factor for legibility')
    parser.add_argument('--fps', type=int, default=4)
    return parser.parse_args()


def lidar_density_bev(pts, pc_range, bev_size):
    """Gray backdrop: clipped LiDAR point count per BEV pixel."""
    h, w = bev_size
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    cols = ((pts[:, 0] - x_min) / (x_max - x_min) * w).astype(int)
    rows = ((y_max - pts[:, 1]) / (y_max - y_min) * h).astype(int)
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    bg = np.zeros((h, w), dtype=np.uint8)
    np.add.at(bg, (rows[ok], cols[ok]), 1)
    return (np.clip(bg, 0, 3) * 70).astype(np.uint8)


def lidar_height_bev(pts, pc_range, bev_size):
    """BGR panel of raw LiDAR coloured by height (z), JET colormap.

    Higher points are warmer; the topmost point per pixel wins so tall
    structures (stacks, gantry) stand out over ground. This is a richer
    view of the raw input than the gray density backdrop in panel A.
    """
    h, w = bev_size
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in pc_range]
    cols = ((pts[:, 0] - x_min) / (x_max - x_min) * w).astype(int)
    rows = ((y_max - pts[:, 1]) / (y_max - y_min) * h).astype(int)
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    cols, rows, z = cols[ok], rows[ok], pts[ok, 2]
    # Keep the highest point per pixel so foreground structure is visible.
    order = np.argsort(z)
    cols, rows, z = cols[order], rows[order], z[order]
    z_norm = np.clip((z - z_min) / max(z_max - z_min, 1e-6), 0, 1)
    color = cv2.applyColorMap((z_norm * 255).astype(np.uint8),
                              cv2.COLORMAP_JET).reshape(-1, 3)
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[rows, cols] = color
    return panel


def _titled_panel(panel, title, scale):
    """Upscale a panel and stack a dark title bar on top of it."""
    big = cv2.resize(panel, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    h, w = big.shape[:2]
    bar = np.full((26, w, 3), 30, dtype=np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, big], axis=0)


def _legend_bar(width, items):
    """A bottom strip of colour swatches + labels (BGR colours)."""
    bar = np.full((34, width, 3), 30, dtype=np.uint8)
    x = 8
    for color, label in items:
        cv2.rectangle(bar, (x, 9), (x + 18, 25), color, -1)
        cv2.rectangle(bar, (x, 9), (x + 18, 25), (200, 200, 200), 1)
        x += 24
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(bar, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (235, 235, 235), 1, cv2.LINE_AA)
        x += tw + 22
    return bar


def render_frame(map_mask, ground, blocked, pts, pc_range, bev_size,
                 scale=3, frame_idx=None):
    """Build the labelled 2x2 panel figure for one frame.

    Panels read input -> output, left-to-right then top-to-bottom:
    D (raw LiDAR by height) | A (final GT on density) on the top row,
    B (source decomposition) | C (final Dice mask) on the bottom row.
    Each panel has its own title bar; legend strips along the bottom
    explain every colour so the figure stands on its own.
    """
    h, w = bev_size
    final = np.maximum(map_mask, ground).astype(np.uint8)
    final[blocked > 0] = 0

    # Panel D: raw LiDAR coloured by height (the model's actual input).
    panel_d = lidar_height_bev(pts, pc_range, bev_size)
    cv2.drawMarker(panel_d, (w // 2, h // 2), (255, 255, 255),
                   cv2.MARKER_CROSS, 8, 1)

    # Panel A: final GT over lidar density, ego centre cross.
    bg = lidar_density_bev(pts, pc_range, bev_size)
    panel_a = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    panel_a[final > 0] = (60, 200, 60)
    cv2.drawMarker(panel_a, (w // 2, h // 2), (0, 0, 255),
                   cv2.MARKER_CROSS, 8, 1)

    # Panel B: source decomposition.
    map_b = map_mask > 0
    ground_b = ground > 0
    panel_b = np.zeros((h, w, 3), dtype=np.uint8)
    panel_b[..., 2] = ((map_b & ~ground_b) * 255).astype(np.uint8)  # red
    panel_b[..., 1] = ((ground_b & ~map_b) * 255).astype(np.uint8)  # green
    panel_b[map_b & ground_b] = (0, 255, 255)                       # yellow
    panel_b[blocked > 0] = (255, 80, 0)                             # blue

    # Panel C: final binary target.
    panel_c = cv2.cvtColor((final * 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)

    d = _titled_panel(panel_d, 'D: raw LiDAR (colour = height)', scale)
    a = _titled_panel(panel_a, 'A: final drivable GT + LiDAR', scale)
    b = _titled_panel(panel_b, 'B: where the GT comes from', scale)
    c = _titled_panel(panel_c, 'C: final mask (Dice target)', scale)

    hpad = np.full((d.shape[0], 6, 3), 30, dtype=np.uint8)
    top = np.concatenate([d, hpad, a], axis=1)
    bottom = np.concatenate([b, hpad, c], axis=1)
    vpad = np.full((6, top.shape[1], 3), 30, dtype=np.uint8)
    grid = np.concatenate([top, vpad, bottom], axis=0)

    # Legend rows: D/A semantics, then B semantics.
    legend_da = _legend_bar(grid.shape[1], [
        ((0, 0, 255), 'low'),
        ((0, 255, 255), 'mid height'),
        ((0, 0, 128), 'high (LiDAR z)'),
        ((60, 200, 60), 'drivable GT'),
        ((120, 120, 120), 'LiDAR density'),
        ((255, 255, 255), 'ego'),
    ])
    legend_b = _legend_bar(grid.shape[1], [
        ((0, 0, 255), 'HD-map only'),
        ((0, 255, 0), 'raycast ground only'),
        ((0, 255, 255), 'both agree'),
        ((255, 80, 0), 'obstacle (removed)'),
    ])
    out = np.concatenate([grid, legend_da, legend_b], axis=0)
    if frame_idx is not None:
        cv2.putText(out, 'frame %d' % frame_idx,
                    (out.shape[1] - 96, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 220, 255), 1, cv2.LINE_AA)

    stats = dict(
        final=int((final > 0).sum()),
        map=int(map_b.sum()),
        ray=int(ground_b.sum()),
        blocked=int((blocked > 0).sum()))
    return out, stats

def write_webm(frame_dir, out_path, fps):
    """Mux numbered PNG frames into a VP9 webm (matches run_e2e_subset.py)."""
    if osp.splitext(out_path)[1].lower() != '.webm':
        raise ValueError('--video must end in .webm')
    out_parent = osp.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    # Frames are already even-sized after the integer upscale; still guard
    # the encoder with a trunc-to-even scale filter.
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', osp.join(frame_dir, 'gt_%05d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libvpx-vp9', '-crf', '32', '-b:v', '0',
        '-pix_fmt', 'yuv420p',
        out_path,
    ]
    subprocess.run(cmd, check=True)


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
    os.makedirs(args.out_dir, exist_ok=True)

    bev_size = gen.bev_size
    pc_range = gen.point_cloud_range
    seq = 0
    for idx in indices:
        info = dataset.get_data_info(idx)
        if info is None:
            continue
        dataset.pre_pipeline(info)
        results = pre(info)
        if results is None or 'points' not in results:
            continue

        pts = gen._points_numpy(results['points'])
        boxes = gen._boxes_numpy(results.get('gt_bboxes_3d'))
        map_mask = build_map_mask(
            gen.drivable_global, gen._ego2global(results), pc_range, bev_size)
        raycast = gen.raycast_builder.build(pts, boxes)
        strip, stats = render_frame(
            map_mask, raycast['ground'], raycast['blocked'],
            pts, pc_range, bev_size, scale=args.scale, frame_idx=idx)

        # Sequential filenames so ffmpeg sees a contiguous frame range even
        # when --stride skips dataset indices.
        cv2.imwrite(osp.join(args.out_dir, 'gt_%05d.png' % seq), strip)
        seq += 1
        print('frame %5d: final=%d px (map=%d ray=%d blocked=%d)'
              % (idx, stats['final'], stats['map'], stats['ray'],
                 stats['blocked']))

    if seq == 0:
        print('No usable frames; nothing rendered.')
        return
    print('rendered %d frames to %s' % (seq, args.out_dir))

    if args.video:
        write_webm(args.out_dir, args.video, args.fps)
        print('video saved to %s' % args.video)


if __name__ == '__main__':
    main()
