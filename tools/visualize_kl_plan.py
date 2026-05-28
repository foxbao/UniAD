#!/usr/bin/env python
"""Visualize KL SDC planning ground truth in BEV.

GT-only viewer for the sdc_planning / sdc_planning_mask / command fields
written by tools/data_converter/add_sdc.py.  Once a planning checkpoint
is available, --checkpoint will overlay the predicted trajectory; for
now it reads only from the pkl.

Example:
    python tools/visualize_kl_plan.py \
        --config projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan.py \
        --out-dir projects/work_dirs/vis_plan_gt \
        --max-per-command 5 --max-total 20

The default sampling picks a balanced mix of Straight / Left / Right
frames so turning failures are easy to spot.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Sequence

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmcv import Config

from visualize_kl_velocity import (compute_box_corners_bev, get_class_names,
                                   get_dataset_cfg, get_label_mapping,
                                   lidar_xy_to_display, load_points)


COMMAND_NAME = {0: 'Right', 1: 'Left', 2: 'Straight'}
COMMAND_COLOR = {0: 'tab:orange', 1: 'tab:purple', 2: 'tab:cyan'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize KL SDC planning GT (and predictions when'
                    ' a checkpoint is provided).')
    parser.add_argument('--config', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--split', default='val', choices=['val', 'test'])
    parser.add_argument(
        '--max-per-command',
        type=int,
        default=5,
        help='Cap per command class when sampling.')
    parser.add_argument(
        '--max-total',
        type=int,
        default=20,
        help='Hard cap on rendered frames.')
    parser.add_argument('--point-stride', type=int, default=4)
    parser.add_argument(
        '--range',
        type=float,
        nargs=4,
        default=None,
        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
        help='Override BEV plot range; defaults to point_cloud_range '
             'in the config.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--min-displacement',
        type=float,
        default=0.0,
        help='Drop frames whose final-step planning displacement is '
             'below this many metres. Useful for filtering the long '
             'tail of stationary/idling frames.')
    return parser.parse_args()


def gather_frames_by_command(infos: Sequence[dict],
                             min_displacement: float = 0.0
                             ) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for idx, info in enumerate(infos):
        if 'command' not in info:
            continue
        c = int(np.asarray(info['command']).reshape(-1)[0])
        if c not in buckets:
            continue
        if min_displacement > 0:
            plan = np.asarray(info['sdc_planning'], dtype=np.float32)
            mask = np.asarray(info['sdc_planning_mask'], dtype=bool)
            if plan.ndim != 3 or mask.ndim != 3:
                continue
            valid = mask[0, :, 0]
            if not valid.any():
                continue
            last = int(np.where(valid)[0].max())
            disp = float(np.linalg.norm(plan[0, last, :2]))
            if disp < min_displacement:
                continue
        buckets[c].append(idx)
    return buckets


def sample_balanced(buckets: Dict[int, List[int]], max_per_command: int,
                    max_total: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    # Ordered Left / Right / Straight so the Left/Right minority gets
    # picked first when the total cap binds.
    for cmd in (1, 0, 2):
        pool = buckets.get(cmd, [])
        if not pool:
            continue
        take = min(max_per_command, len(pool), max_total - len(chosen))
        if take <= 0:
            continue
        picked = rng.choice(pool, size=take, replace=False)
        chosen.extend(int(x) for x in picked)
        if len(chosen) >= max_total:
            break
    return chosen


def draw_boxes(ax, boxes: np.ndarray, color: str, alpha: float = 0.8,
               linewidth: float = 1.0) -> None:
    if boxes.size == 0:
        return
    corners = compute_box_corners_bev(boxes)
    for poly in corners:
        poly_disp = lidar_xy_to_display(poly)
        closed = np.concatenate([poly_disp, poly_disp[:1]], axis=0)
        ax.plot(closed[:, 0], closed[:, 1],
                color=color, linewidth=linewidth, alpha=alpha)


def draw_other_futures(ax, info: dict) -> None:
    # add_forecasting.py writes per-instance gt_fut_traj_locs / _mask under
    # info['instances'][i] (not at the info top level), so iterate the
    # instance list rather than indexing a top-level array.
    instances = info.get('instances') or []
    if not instances:
        return
    # Future trajectory labels are relative to the object's current
    # position. Add the box center to get world-aligned points in the
    # current LiDAR frame.
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        bbox = inst.get('bbox_3d')
        locs = inst.get('gt_fut_traj_locs', inst.get('gt_forecasting_locs'))
        mask = inst.get('gt_fut_traj_mask', inst.get('gt_forecasting_mask'))
        if bbox is None or locs is None or mask is None:
            continue
        traj = np.asarray(locs, dtype=np.float32)
        m = np.asarray(mask).astype(bool)
        if traj.ndim != 2 or traj.shape[-1] < 2:
            continue
        valid = m[:traj.shape[0]] if m.ndim == 1 else m.any(axis=-1)[:traj.shape[0]]
        if not valid.any():
            continue
        cx, cy = float(bbox[0]), float(bbox[1])
        pts = traj[valid, :2] + np.array([cx, cy], dtype=np.float32)
        pts_disp = lidar_xy_to_display(pts)
        ax.plot(pts_disp[:, 0], pts_disp[:, 1],
                color='tab:blue', linewidth=0.8, alpha=0.5)


def draw_planning(ax, plan: np.ndarray, mask: np.ndarray,
                  color: str, label: str) -> None:
    if plan.size == 0:
        return
    valid = mask[0, :, 0].astype(bool) if mask.ndim == 3 else mask.astype(bool)
    if not valid.any():
        return
    xy = plan[0, valid, :2] if plan.ndim == 3 else plan[valid, :2]
    # Prepend ego origin so the trajectory clearly emerges from the
    # vehicle.
    xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), xy], axis=0)
    disp = lidar_xy_to_display(xy)
    ax.plot(disp[:, 0], disp[:, 1],
            color=color, linewidth=2.4, label=label)
    ax.scatter(disp[1:, 0], disp[1:, 1],
               s=22, color=color, zorder=5)


def render_frame(ax, info: dict, points: np.ndarray, point_stride: int,
                 plot_range, sdc_box: np.ndarray) -> None:
    # Points
    if points.size and point_stride > 1:
        points = points[::point_stride]
    if points.size:
        pts_disp = lidar_xy_to_display(points[:, :2])
        ax.scatter(pts_disp[:, 0], pts_disp[:, 1],
                   s=0.25, c='lightgray', alpha=0.6, linewidths=0)

    # Object boxes (current frame)
    instances = info.get('instances') or []
    if instances:
        boxes = np.asarray(
            [inst['bbox_3d'][:7] for inst in instances if 'bbox_3d' in inst],
            dtype=np.float32)
        # bbox_3d here is [x, y, z, l, w, h, yaw] -- pad to 9 for the
        # helper that expects (x,y,z,l,w,h,yaw,vx,vy).
        if boxes.size:
            pad = np.zeros((boxes.shape[0], 9 - boxes.shape[1]),
                           dtype=np.float32)
            boxes = np.concatenate([boxes[:, :7], pad], axis=-1)
            draw_boxes(ax, boxes, color='tab:blue', alpha=0.6,
                       linewidth=0.9)

    # SDC ego box (yellow)
    if sdc_box.size:
        draw_boxes(ax, sdc_box, color='gold', alpha=1.0, linewidth=1.8)

    # Other-vehicle futures (faint blue) for context
    draw_other_futures(ax, info)

    # SDC planning GT
    plan = np.asarray(info['sdc_planning'], dtype=np.float32)
    plan_mask = np.asarray(info['sdc_planning_mask'], dtype=np.float32)
    draw_planning(ax, plan, plan_mask,
                  color='tab:green', label='Plan GT')

    # Frame composition
    xmin, ymin, xmax, ymax = plot_range
    # display axes use disp_x = -y (lidar), disp_y = x (lidar).
    ax.set_xlim(-ymax, -ymin)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset_cfg = get_dataset_cfg(cfg, args.split)
    pkl = osp.join(dataset_cfg.get('data_root', 'data/kl_8/'),
                   dataset_cfg['ann_file'])
    print(f'Loading pkl: {pkl}')
    data = mmcv.load(pkl)
    infos = data.get('data_list', data.get('infos', data))

    if not infos or 'sdc_planning' not in infos[0]:
        raise SystemExit(
            f'pkl missing sdc_planning; rerun add_sdc.py on {pkl}.')

    buckets = gather_frames_by_command(infos, args.min_displacement)
    counts = {COMMAND_NAME[k]: len(v) for k, v in buckets.items()}
    print(f'Frames per command (min_disp={args.min_displacement}m): '
          f'{counts}')

    chosen = sample_balanced(buckets, args.max_per_command,
                             args.max_total, args.seed)
    print(f'Sampled {len(chosen)} frames.')

    if args.range is None:
        pcr = cfg.point_cloud_range
        plot_range = (pcr[0], pcr[1], pcr[3], pcr[4])
    else:
        plot_range = tuple(args.range)
    print(f'BEV plot range (lidar x/y): {plot_range}')

    mmcv.mkdir_or_exist(args.out_dir)
    sdc_box_template = None  # filled per-frame from gt_sdc_bbox

    for n, idx in enumerate(chosen):
        info = infos[idx]
        cmd = int(np.asarray(info['command']).reshape(-1)[0])
        token = info.get('token', f'idx{idx}')
        try:
            points = load_points(cfg, dataset_cfg, info)
        except Exception as e:
            print(f'  [{n + 1}/{len(chosen)}] {token}: load_points '
                  f'failed: {e}')
            points = np.zeros((0, 4), dtype=np.float32)

        sdc_box = np.asarray(info.get('gt_sdc_bbox', []), dtype=np.float32)
        if sdc_box.size and sdc_box.shape[-1] < 9:
            pad = np.zeros((sdc_box.shape[0], 9 - sdc_box.shape[-1]),
                           dtype=np.float32)
            sdc_box = np.concatenate([sdc_box, pad], axis=-1)

        fig, ax = plt.subplots(figsize=(8, 6))
        render_frame(ax, info, points, args.point_stride, plot_range,
                     sdc_box)

        title = (f'{COMMAND_NAME[cmd]} (cmd={cmd})  idx={idx}  '
                 f'token={token[:12]}')
        ax.set_title(title, fontsize=10)
        ax.legend(loc='lower right', fontsize=8)
        out_path = osp.join(
            args.out_dir,
            f'{n:02d}_{COMMAND_NAME[cmd][0]}_{token[:12]}.png')
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'  [{n + 1}/{len(chosen)}] {COMMAND_NAME[cmd]:8s} '
              f'-> {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
