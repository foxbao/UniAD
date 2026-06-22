#!/usr/bin/env python
"""Visualize KL LiDAR track + drivable-map evaluation outputs.

The normal eval pickle intentionally strips per-frame map logits to avoid
GPU memory growth. This script runs a small sequential inference pass and
renders comparable PNG frames plus an optional WEBM.
"""

from __future__ import annotations

import argparse
import importlib
import os
import os.path as osp
import subprocess
from typing import Iterable, Tuple

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.apis import set_random_seed
from third_party.uniad_mmdet3d.datasets.builder import (
    build_dataloader,
    build_dataset,
)
from third_party.uniad_mmdet3d.models.builder import build_model


PALETTE = np.array(
    [
        (239, 83, 80),
        (255, 167, 38),
        (255, 238, 88),
        (102, 187, 106),
        (38, 166, 154),
        (41, 182, 246),
        (92, 107, 192),
        (171, 71, 188),
        (141, 110, 99),
        (120, 144, 156),
    ],
    dtype=np.uint8,
)

BG_COLOR = (10, 13, 15)
PANEL_BG = (6, 8, 10)
TITLE_BG = (24, 27, 30)
GRID_COLOR = (22, 26, 28)
POINT_COLOR = (230, 232, 235)
POINT_ALPHA = 0.28
TITLE_HEIGHT = 44


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize stage-1 KL LiDAR track + drivable eval.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--num-frames', type=int, default=48)
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--topk', type=int, default=60)
    parser.add_argument('--label-topk', type=int, default=12)
    parser.add_argument('--scale', type=int, default=5)
    parser.add_argument('--fps', type=float, default=6.0)
    parser.add_argument('--video', default=None)
    parser.add_argument('--no-points', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def import_plugin(cfg):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', None)
    if plugin_dir is None:
        return
    module_parts = osp.dirname(plugin_dir).split('/')
    module_path = module_parts[0]
    for part in module_parts[1:]:
        module_path += f'.{part}'
    importlib.import_module(module_path)


def tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        return value.numpy()
    return np.asarray(value)


def dc_data(value):
    return value.data if hasattr(value, 'data') else value


def current_meta(img_metas):
    data = dc_data(img_metas)
    meta = data[0][0]
    if isinstance(meta, dict) and meta and all(
            isinstance(k, int) for k in meta.keys()):
        return meta[max(meta.keys())]
    return meta


def current_points(points):
    data = dc_data(points)
    pts = data[0][0]
    if isinstance(pts, (list, tuple)):
        pts = pts[-1]
    return tensor_to_numpy(pts)


def current_gt_mask(gt_lane_masks):
    mask = tensor_to_numpy(gt_lane_masks)
    while mask.ndim > 2:
        mask = mask[0]
    return (mask > 0)


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    color = PALETTE[int(track_id) % len(PALETTE)]
    return int(color[2]), int(color[1]), int(color[0])  # BGR


def bev_indices(points, pc_range, shape):
    h, w = shape
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    cols = ((points[:, 0] - x_min) / (x_max - x_min) * w).astype(np.int32)
    rows = ((y_max - points[:, 1]) / (y_max - y_min) * h).astype(np.int32)
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    return rows[ok], cols[ok]


def lidar_density(points, pc_range, shape):
    h, w = shape
    rows, cols = bev_indices(points, pc_range, shape)
    panel = np.zeros((h, w), dtype=np.uint8)
    if len(rows):
        np.add.at(panel, (rows, cols), 1)
    return (np.clip(panel, 0, 4) * 55).astype(np.uint8)


def overlay_points(panel, points, pc_range):
    rows, cols = bev_indices(points, pc_range, panel.shape[:2])
    panel[rows, cols] = (165, 165, 165)


def overlay_points_scaled(panel, points, pc_range, base_shape, scale,
                          title_height=TITLE_HEIGHT, alpha=POINT_ALPHA):
    if points is None or len(points) == 0:
        return
    base_h, base_w = base_shape
    width = base_w * scale
    height = base_h * scale
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    cols = ((points[:, 0] - x_min) / (x_max - x_min) * width).astype(np.int32)
    rows = ((y_max - points[:, 1]) / (y_max - y_min) * height).astype(np.int32)
    rows = rows + title_height
    ok = (
        (rows >= title_height) & (rows < panel.shape[0]) &
        (cols >= 0) & (cols < panel.shape[1])
    )
    rows = rows[ok]
    cols = cols[ok]
    base = panel[rows, cols].astype(np.float32)
    point = np.array(POINT_COLOR, dtype=np.float32)
    panel[rows, cols] = (base * (1.0 - alpha) +
                         point * alpha).astype(np.uint8)


def xy_to_px(xy, pc_range, shape):
    h, w = shape[:2]
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    col = (xy[:, 0] - x_min) / (x_max - x_min) * w
    row = (y_max - xy[:, 1]) / (y_max - y_min) * h
    return np.stack([col, row], axis=1).astype(np.int32)


def xy_to_px_scaled(xy, pc_range, base_shape, scale,
                    title_height=TITLE_HEIGHT):
    h, w = base_shape
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    col = (xy[:, 0] - x_min) / (x_max - x_min) * (w * scale)
    row = (y_max - xy[:, 1]) / (y_max - y_min) * (h * scale)
    row = row + title_height
    return np.stack([col, row], axis=1).astype(np.int32)


def boxes_bev_corners_kl(boxes):
    """Return BEV corners using the KL [x,y,z,length,width,height,yaw] convention."""
    if boxes is None:
        return np.zeros((0, 4, 2), dtype=np.float32)
    tensor = tensor_to_numpy(boxes.tensor if hasattr(boxes, 'tensor') else boxes)
    if tensor is None or len(tensor) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    centers = tensor[:, :2].astype(np.float32)
    length = tensor[:, 3].astype(np.float32)
    width = tensor[:, 4].astype(np.float32)
    yaw = tensor[:, 6].astype(np.float32)
    c = np.cos(yaw)
    s = np.sin(yaw)
    dx = np.stack(
        [length * 0.5, length * 0.5, -length * 0.5, -length * 0.5],
        axis=1)
    dy = np.stack(
        [width * 0.5, -width * 0.5, -width * 0.5, width * 0.5],
        axis=1)
    x = centers[:, 0:1] + dx * c[:, None] - dy * s[:, None]
    y = centers[:, 1:2] + dx * s[:, None] + dy * c[:, None]
    return np.stack([x, y], axis=-1)


def draw_boxes(panel, boxes, scores, labels, track_ids, pc_range, score_thr,
               topk, label_topk, class_names, prefix):
    if boxes is None or len(scores) == 0:
        return
    order = np.argsort(-scores)
    order = order[scores[order] >= score_thr][:topk]
    if len(order) == 0:
        return
    corners = boxes_bev_corners_kl(boxes)
    for rank, idx in enumerate(order):
        pts = xy_to_px(corners[idx], pc_range, panel.shape)
        track_id = int(track_ids[idx]) if track_ids is not None and idx < len(track_ids) else idx
        color = color_for_id(track_id if prefix == 'T' else int(labels[idx]))
        cv2.polylines(panel, [pts.reshape(-1, 1, 2)], True, color, 1,
                      cv2.LINE_AA)
        if rank >= label_topk:
            continue
        center = pts.mean(axis=0).astype(np.int32)
        label = class_names[int(labels[idx])] if int(labels[idx]) < len(class_names) else str(int(labels[idx]))
        text = f'{prefix}{track_id}:{label[:4]} {scores[idx]:.2f}'
        org = tuple(center)
        cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    color, 1, cv2.LINE_AA)


def draw_gt_boxes_scaled(panel, boxes, labels, track_ids, pc_range, base_shape,
                         scale, label_topk, class_names):
    if boxes is None or labels is None:
        return
    labels = tensor_to_numpy(labels)
    track_ids = tensor_to_numpy(track_ids)
    if labels is None or len(labels) == 0:
        return
    corners = boxes_bev_corners_kl(boxes)
    for rank, idx in enumerate(range(len(labels))):
        pts = xy_to_px_scaled(corners[idx], pc_range, base_shape, scale)
        in_view = (
            (pts[:, 0] >= 0) & (pts[:, 0] < panel.shape[1]) &
            (pts[:, 1] >= TITLE_HEIGHT) & (pts[:, 1] < panel.shape[0])
        )
        if not in_view.any():
            continue
        track_id = int(track_ids[idx]) if track_ids is not None and idx < len(track_ids) else idx
        color = color_for_id(track_id)
        cv2.polylines(panel, [pts.reshape(-1, 1, 2)], True, color, 2,
                      cv2.LINE_AA)
        if rank >= label_topk:
            continue
        center = pts.mean(axis=0).astype(np.int32)
        label = class_names[int(labels[idx])] if int(labels[idx]) < len(class_names) else str(int(labels[idx]))
        text = f'GT{track_id}:{label[:4]}'
        cv2.putText(panel, text, tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, text, tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)


def make_mask_panel(mask, color):
    panel = np.full((*mask.shape, 3), PANEL_BG, dtype=np.uint8)
    panel[mask] = color
    return panel


def make_diff_panel(gt, pred):
    panel = np.full((*gt.shape, 3), PANEL_BG, dtype=np.uint8)
    tp = gt & pred
    fp = ~gt & pred
    fn = gt & ~pred
    panel[tp] = (80, 190, 80)
    panel[fp] = (70, 70, 255)
    panel[fn] = (255, 90, 40)
    return panel


def make_bev_panel(shape, grid=20):
    panel = np.full((*shape, 3), PANEL_BG, dtype=np.uint8)
    h, w = shape
    for row in range(grid, h, grid):
        cv2.line(panel, (0, row), (w - 1, row), GRID_COLOR, 1)
    for col in range(grid, w, grid):
        cv2.line(panel, (col, 0), (col, h - 1), GRID_COLOR, 1)
    return panel


def title_panel(panel, title, scale):
    panel = cv2.resize(panel, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)
    bar = np.full((TITLE_HEIGHT, panel.shape[1], 3), TITLE_BG,
                  dtype=np.uint8)
    cv2.putText(bar, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.74,
                (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([bar, panel], axis=0)


def legend(width):
    items = [
        ((80, 190, 80), 'TP / drivable'),
        ((70, 70, 255), 'FP pred-only'),
        ((255, 90, 40), 'FN gt-only'),
        (POINT_COLOR, 'LiDAR points'),
        ((255, 167, 38), 'GT track/det boxes'),
        ((255, 255, 255), 'ego'),
    ]
    bar = np.full((34, width, 3), TITLE_BG, dtype=np.uint8)
    x = 8
    for color, label in items:
        cv2.rectangle(bar, (x, 9), (x + 18, 25), color, -1)
        cv2.rectangle(bar, (x, 9), (x + 18, 25), (210, 210, 210), 1)
        x += 24
        cv2.putText(bar, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (235, 235, 235), 1, cv2.LINE_AA)
        x += cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 18
    return bar


def render_frame(meta, points, gt, pred, result, gt_ann, pc_range,
                 class_names, args, frame_id):
    shape = gt.shape
    gt_panel = make_mask_panel(gt, (80, 190, 80))
    pred_panel = make_mask_panel(pred, (70, 210, 210))
    diff_panel = make_diff_panel(gt, pred)
    point_panel = make_bev_panel(shape)

    h, w = shape
    ego = (w // 2, h // 2)
    for panel in (gt_panel, pred_panel, diff_panel, point_panel):
        cv2.drawMarker(panel, ego, (255, 255, 255), cv2.MARKER_CROSS, 10, 1)

    inter = int((gt & pred).sum())
    union = int((gt | pred).sum())
    iou = inter / union if union else 0.0
    timestamp = meta.get('timestamp', '')
    titles = [
        f'GT drivable | idx={frame_id}',
        f'Pred drivable | IoU={iou:.3f}',
        'Diff: green TP, red FP, blue FN',
        f'LiDAR points + GT track/det boxes | {timestamp}',
    ]
    panels = [
        title_panel(gt_panel, titles[0], args.scale),
        title_panel(pred_panel, titles[1], args.scale),
        title_panel(diff_panel, titles[2], args.scale),
        title_panel(point_panel, titles[3], args.scale),
    ]
    if not args.no_points:
        overlay_points_scaled(panels[3], points, pc_range, shape, args.scale,
                              alpha=0.8)
    draw_gt_boxes_scaled(
        panels[3],
        gt_ann.get('gt_bboxes_3d'),
        gt_ann.get('gt_labels_3d'),
        gt_ann.get('gt_inds'),
        pc_range,
        shape,
        args.scale,
        args.label_topk,
        class_names,
    )

    pad_v = np.full((panels[0].shape[0], 8, 3), TITLE_BG, dtype=np.uint8)
    top = np.concatenate([panels[0], pad_v, panels[1]], axis=1)
    bottom = np.concatenate([panels[2], pad_v, panels[3]], axis=1)
    pad_h = np.full((8, top.shape[1], 3), TITLE_BG, dtype=np.uint8)
    canvas = np.concatenate([top, pad_h, bottom], axis=0)
    canvas = np.concatenate([canvas, legend(canvas.shape[1])], axis=0)
    return canvas, iou


def write_webm(out_dir, video, fps):
    if not video:
        return
    os.makedirs(osp.dirname(video) or '.', exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', osp.join(out_dir, 'frame_%05d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libvpx-vp9', '-crf', '32', '-b:v', '0',
        '-pix_fmt', 'yuv420p', video,
    ]
    subprocess.run(cmd, check=True)


def reset_model_sequence_state(model):
    module = model.module if hasattr(model, 'module') else model
    for name in ('test_track_instances', 'scene_token', 'timestamp', 'l2g_t',
                 'l2g_r_mat', '_test_track_instances', '_test_prev_bev',
                 '_test_scene_token', 'test_frame_token', 'prev_bev'):
        if hasattr(module, name):
            setattr(module, name, None)
    if hasattr(module, 'track_base') and hasattr(module.track_base, 'clear'):
        module.track_base.clear()


def is_sequence_break(prev_meta, curr_meta, max_time_gap=1.0):
    if prev_meta is None:
        return True
    if curr_meta.get('scene_token') != prev_meta.get('scene_token'):
        return True
    prev_token = curr_meta.get('prev', None)
    if prev_token == '':
        return True
    if prev_token is not None and prev_token != prev_meta.get('token'):
        return True
    try:
        time_gap = abs(
            float(curr_meta.get('timestamp', 0.0)) -
            float(prev_meta.get('timestamp', 0.0)))
    except (TypeError, ValueError):
        return False
    return time_gap > max_time_gap


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    set_random_seed(args.seed, deterministic=True)

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', None),
    )

    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    pc_range = cfg.get('point_cloud_range', cfg.model.point_cloud_range)
    class_names = list(cfg.get('class_names', []))
    rendered = 0
    ious = []
    prev_meta = None
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            if idx < args.start_index:
                continue
            if rendered >= args.num_frames:
                break
            meta = current_meta(data['img_metas'])
            if is_sequence_break(prev_meta, meta):
                reset_model_sequence_state(model)
            points = current_points(data['points'])
            gt = current_gt_mask(data['gt_lane_masks'])
            gt_ann = dataset.get_ann_info(idx)
            result = model(return_loss=False, rescale=True, **data)
            pts_bbox = result[0]['pts_bbox']
            pred = tensor_to_numpy(pts_bbox['map']['drivable']).astype(bool)
            frame, iou = render_frame(meta, points, gt, pred, pts_bbox,
                                      gt_ann, pc_range, class_names, args,
                                      idx)
            out_path = osp.join(args.out_dir, f'frame_{rendered:05d}.png')
            cv2.imwrite(out_path, frame)
            ious.append(float(iou))
            rendered += 1
            prev_meta = meta
            torch.cuda.empty_cache()

    summary = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        frames=rendered,
        scale=args.scale,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        ious=ious,
    )
    mmcv.dump(summary, osp.join(args.out_dir, 'summary.json'))
    write_webm(args.out_dir, args.video, args.fps)
    print(summary)


if __name__ == '__main__':
    main()
