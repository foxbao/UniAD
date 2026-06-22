#!/usr/bin/env python
"""Visualize LiDAR E2E tracking and motion predictions in BEV."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import os.path as osp
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model

from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from third_party.uniad_mmdet3d.models.builder import build_model


PALETTE = np.array([
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
    (244, 143, 177),
    (77, 208, 225),
], dtype=np.float32) / 255.0

NUM_RE = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
POINT_RE = re.compile(
    r'point\s*\{[^{}]*?\bx:\s*(' + NUM_RE + r')'
    r'[^{}]*?\by:\s*(' + NUM_RE + r')',
    re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize LiDAR E2E BEV boxes, track IDs, and motion.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--token', default=None)
    parser.add_argument('--scene-token', default=None)
    parser.add_argument('--max-frames', type=int, default=32)
    parser.add_argument('--score-thr', type=float, default=0.25)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--point-stride', type=int, default=4)
    parser.add_argument('--annotate-topk', type=int, default=18)
    parser.add_argument('--webm-fps', type=float, default=6.0)
    parser.add_argument('--webm-crf', type=int, default=32)
    parser.add_argument('--gt-map-overlay', default='none',
                        choices=['none', 'hdmap'],
                        help='overlay map information on the left GT panel')
    parser.add_argument('--hdmap-path', default=None,
                        help='HDMap text path; default uses config map_path or data/kl_8/map/base_map.txt')
    parser.add_argument('--hdmap-max-lanes', type=int, default=96)
    parser.add_argument('--hdmap-margin', type=float, default=8.0,
                        help='extra crop margin in metres around point_cloud_range')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if args.max_frames <= 0:
        raise ValueError('--max-frames must be positive.')
    if args.point_stride <= 0:
        raise ValueError('--point-stride must be positive.')
    if args.hdmap_max_lanes <= 0:
        raise ValueError('--hdmap-max-lanes must be positive.')
    return args


def resolve_repo_path(path: str) -> str:
    if osp.isabs(path) or osp.exists(path):
        return path
    return osp.join(REPO_ROOT, path)


def import_cfg_modules(cfg: Config, config_path: str) -> None:
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**custom_imports)
    if cfg.get('plugin', False):
        module_dir = osp.dirname(cfg.get('plugin_dir', osp.dirname(config_path)))
        module_path = module_dir.replace('/', '.').strip('.')
        if module_path:
            importlib.import_module(module_path)


def get_dataset_cfg(cfg: Config, split: str):
    return cfg.data.val if split == 'val' else cfg.data.test


def raw_info_from_dataset(dataset, index: int) -> dict:
    raw_index = dataset._to_raw_index(index) if hasattr(
        dataset, '_to_raw_index') else index
    return dataset.data_infos[raw_index]


def resolve_start_index(dataset, args: argparse.Namespace) -> int:
    if args.token is None and args.scene_token is None:
        return args.start_index
    for idx in range(len(dataset)):
        info = raw_info_from_dataset(dataset, idx)
        if args.token is not None and info.get('token') == args.token:
            return idx
        if args.scene_token is not None and (
                info.get('scene_token') == args.scene_token):
            return idx
    key = args.token if args.token is not None else args.scene_token
    raise KeyError(f'Cannot find requested token/scene: {key}')


def consecutive_scene_indices(dataset, start_index: int,
                              max_frames: int) -> List[int]:
    start_info = raw_info_from_dataset(dataset, start_index)
    scene_token = start_info.get('scene_token')
    indices = []
    for idx in range(start_index, len(dataset)):
        info = raw_info_from_dataset(dataset, idx)
        if info.get('scene_token') != scene_token:
            break
        indices.append(idx)
        if len(indices) >= max_frames:
            break
    return indices


def resolve_lidar_path(cfg: Config, dataset_cfg, info: dict) -> str:
    data_root = dataset_cfg.get('data_root', cfg.get('data_root', ''))
    data_prefix = dataset_cfg.get('data_prefix', cfg.get('data_prefix', {}))
    pts_prefix = data_prefix.get('pts', '')
    lidar_info = info.get('lidar_points', {})
    lidar_path = lidar_info.get('lidar_path', info.get('lidar_path'))
    if lidar_path is None:
        raise KeyError('info does not contain lidar path')
    if osp.isabs(lidar_path) or osp.exists(lidar_path):
        return lidar_path
    return osp.join(resolve_repo_path(data_root), pts_prefix, lidar_path)


def load_points(cfg: Config, dataset_cfg, info: dict) -> np.ndarray:
    lidar_path = resolve_lidar_path(cfg, dataset_cfg, info)
    lidar_info = info.get('lidar_points', {})
    num_feats = int(lidar_info.get('num_pts_feats',
                                   info.get('num_features', 4)))
    points = np.fromfile(lidar_path, dtype=np.float32)
    return points.reshape(-1, num_feats)


def tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, 'tensor'):
        return value.tensor.detach().cpu().numpy()
    return np.asarray(value)


def boxes_to_numpy(boxes_3d) -> np.ndarray:
    if boxes_3d is None:
        return np.zeros((0, 9), dtype=np.float32)
    boxes = tensor_to_numpy(boxes_3d).astype(np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[1] < 9:
        boxes = np.pad(boxes, ((0, 0), (0, 9 - boxes.shape[1])))
    return boxes[:, :9]


def color_for_id(track_id: int):
    return PALETTE[int(track_id) % len(PALETTE)]


def box_corners_bev(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    corners = []
    for box in boxes:
        x, y, _, length, width, _, yaw = box[:7]
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        local = np.array([
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ], dtype=np.float32)
        corners.append(local @ rot.T + np.array([x, y], dtype=np.float32))
    return np.stack(corners, axis=0)


def lidar_xy_to_display(xy: np.ndarray) -> np.ndarray:
    disp = np.empty_like(xy, dtype=np.float32)
    disp[..., 0] = -xy[..., 1]
    disp[..., 1] = xy[..., 0]
    return disp


def get_class_names(cfg: Config, dataset_cfg) -> Sequence[str]:
    if dataset_cfg.get('classes', None) is not None:
        return list(dataset_cfg.classes)
    return list(cfg.get('class_names', []))


def get_label_mapping(cfg: Config, dataset_cfg) -> Optional[List[int]]:
    mapping = dataset_cfg.get('label_mapping', cfg.get('label_mapping', None))
    if mapping is None:
        return None
    return [int(x) for x in mapping]


def map_label(label: int, label_mapping: Optional[List[int]],
              num_classes: int) -> int:
    label = int(label)
    if label_mapping is not None:
        if label < 0 or label >= len(label_mapping):
            return -1
        label = int(label_mapping[label])
    if label < 0 or label >= num_classes:
        return -1
    return label


def iter_named_blocks(text: str, name: str):
    needle = name + ' {'
    cursor = 0
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            return
        brace = text.find('{', start)
        if brace < 0:
            return
        depth = 0
        for idx in range(brace, len(text)):
            ch = text[idx]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    yield text[brace + 1:idx]
                    cursor = idx + 1
                    break
        else:
            return


def find_field_block(block: str, field: str) -> Optional[str]:
    match = re.search(r'\b' + re.escape(field) + r'\s*\{', block)
    if match is None:
        return None
    brace = match.end() - 1
    depth = 0
    for idx in range(brace, len(block)):
        ch = block[idx]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return block[brace + 1:idx]
    return None


def extract_hdmap_points(block: Optional[str]) -> Optional[np.ndarray]:
    if not block:
        return None
    points = [(float(x), float(y)) for x, y in POINT_RE.findall(block)]
    if len(points) < 2:
        return None
    return np.asarray(points, dtype=np.float32)


def load_hdmap_lanes(path: str) -> List[Dict[str, Optional[np.ndarray]]]:
    with open(path, 'r') as f:
        text = f.read()

    lanes = []
    for lane_block in iter_named_blocks(text, 'lane'):
        central = extract_hdmap_points(
            find_field_block(lane_block, 'central_curve'))
        if central is None:
            continue
        lanes.append(dict(
            central=central,
            left=extract_hdmap_points(
                find_field_block(lane_block, 'left_boundary')),
            right=extract_hdmap_points(
                find_field_block(lane_block, 'right_boundary'))))
    return lanes


def resolve_hdmap_path(cfg: Config, args: argparse.Namespace) -> str:
    if args.hdmap_path is not None:
        return resolve_repo_path(args.hdmap_path)
    map_lane_encoder = cfg.model.get('map_lane_encoder', None)
    if map_lane_encoder is not None:
        map_path = map_lane_encoder.get('map_path', None)
        if map_path is not None:
            return resolve_repo_path(map_path)
    return resolve_repo_path('data/kl_8/map/base_map.txt')


def transform_hdmap_polyline(polyline: Optional[np.ndarray],
                             g2e: np.ndarray) -> Optional[np.ndarray]:
    if polyline is None or len(polyline) < 2:
        return None
    rotation = g2e[:2, :2]
    translation = g2e[:2, 3]
    return polyline.astype(np.float32) @ rotation.T + translation


def hdmap_lanes_for_frame(
        lanes: Sequence[Dict[str, Optional[np.ndarray]]],
        ego2global,
        pc_range: Sequence[float],
        max_lanes: int,
        margin: float) -> List[Dict[str, Optional[np.ndarray]]]:
    if ego2global is None:
        return []
    e2g = np.asarray(ego2global, dtype=np.float64)
    if e2g.shape != (4, 4):
        return []
    g2e = np.linalg.inv(e2g).astype(np.float32)
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin

    candidates = []
    for lane in lanes:
        central = transform_hdmap_polyline(lane.get('central'), g2e)
        if central is None:
            continue
        cmin = central.min(axis=0)
        cmax = central.max(axis=0)
        if cmax[0] < x_min or cmin[0] > x_max:
            continue
        if cmax[1] < y_min or cmin[1] > y_max:
            continue
        min_dist = float(np.linalg.norm(central, axis=1).min())
        candidates.append((
            min_dist,
            dict(
                central=central,
                left=transform_hdmap_polyline(lane.get('left'), g2e),
                right=transform_hdmap_polyline(lane.get('right'), g2e))))

    candidates.sort(key=lambda item: item[0])
    return [lane for _, lane in candidates[:max_lanes]]


def gt_from_ann(ann: dict, class_names: Sequence[str]):
    boxes = boxes_to_numpy(ann.get('gt_bboxes_3d'))
    labels = np.asarray(ann.get('gt_labels_3d', []), dtype=np.int64)
    track_ids = np.asarray(ann.get('gt_inds', np.arange(len(labels))),
                           dtype=np.int64)
    fut = np.asarray(ann.get('gt_fut_traj', np.zeros((0, 0, 2))),
                     dtype=np.float32)
    fut_mask = np.asarray(
        ann.get('gt_fut_traj_mask', np.zeros((0, 0, 2))),
        dtype=np.float32)
    num = min(len(boxes), len(labels), len(track_ids), len(fut))
    return dict(
        boxes=boxes[:num],
        labels=labels[:num],
        track_ids=track_ids[:num],
        fut=fut[:num],
        fut_mask=fut_mask[:num],
        scores=np.ones((num,), dtype=np.float32))


def unwrap_model_result(result):
    if isinstance(result, (list, tuple)):
        result = result[0] if result else {}
    if isinstance(result, dict) and 'pts_bbox' in result:
        return result['pts_bbox']
    return result if isinstance(result, dict) else {}


def pred_from_result(result: dict, score_thr: float, topk: int):
    result = unwrap_model_result(result)
    boxes = boxes_to_numpy(
        result.get('track_boxes_3d', result.get('boxes_3d', None)))
    labels = tensor_to_numpy(
        result.get('track_labels_3d', result.get('labels_3d', None)))
    scores = tensor_to_numpy(
        result.get('track_scores', result.get('scores_3d', None)))
    track_ids = tensor_to_numpy(result.get('track_ids', None))
    traj = tensor_to_numpy(result.get('traj', None))
    traj_scores = tensor_to_numpy(result.get('traj_scores', None))
    if labels is None:
        labels = np.zeros((len(boxes),), dtype=np.int64)
    if scores is None:
        scores = np.ones((len(boxes),), dtype=np.float32)
    if track_ids is None:
        track_ids = np.arange(len(boxes), dtype=np.int64)
    labels = labels.reshape(-1).astype(np.int64)
    scores = scores.reshape(-1).astype(np.float32)
    track_ids = track_ids.reshape(-1).astype(np.int64)
    num = min(len(boxes), len(labels), len(scores), len(track_ids))
    if traj is not None:
        num = min(num, len(traj))
    keep = np.argsort(-scores[:num])
    keep = keep[scores[keep] >= score_thr][:topk]
    return dict(
        boxes=boxes[keep],
        labels=labels[keep],
        scores=scores[keep],
        track_ids=track_ids[keep],
        traj=None if traj is None else traj[keep],
        traj_scores=None if traj_scores is None else traj_scores[keep])


def best_traj(traj: Optional[np.ndarray],
              traj_scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if traj is None or len(traj) == 0:
        return None
    # traj: (N, modes, steps, 5 or 2). Use highest-score mode per prediction.
    if traj.ndim != 4:
        return None
    if traj_scores is None:
        mode_idx = np.zeros((traj.shape[0],), dtype=np.int64)
    else:
        mode_idx = np.argmax(traj_scores, axis=1).astype(np.int64)
    return traj[np.arange(traj.shape[0]), mode_idx, :, :2]


def setup_axis(ax, pc_range: Sequence[float], title: str) -> None:
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    ax.set_facecolor('#050608')
    ax.set_xlim(-y_max, -y_min)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=7)
    ax.grid(color='#444444', linestyle='--', linewidth=0.5, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color('#888888')
    ax.plot(0, 0, marker='o', markersize=4, color='#ffd166')
    ax.arrow(0, 0, 0, 3.0, color='#ffd166', width=0.03,
             head_width=0.5, head_length=0.6, length_includes_head=True)


def draw_points(ax, points: np.ndarray, stride: int) -> None:
    pts = points[::stride, :2]
    if len(pts):
        disp = lidar_xy_to_display(pts)
        ax.scatter(disp[:, 0], disp[:, 1], s=0.12, c='white', alpha=0.26,
                   linewidths=0)


def draw_hdmap_lanes(ax,
                     lanes: Optional[Sequence[Dict[str, Optional[np.ndarray]]]]
                     ) -> None:
    if not lanes:
        return
    for lane in lanes:
        for key, color, linewidth, alpha, linestyle in (
                ('left', '#8aa0a8', 0.55, 0.28, ':'),
                ('right', '#8aa0a8', 0.55, 0.28, ':'),
                ('central', '#3dd6d0', 0.95, 0.55, '-')):
            pts = lane.get(key)
            if pts is None or len(pts) < 2:
                continue
            disp = lidar_xy_to_display(pts[:, :2])
            ax.plot(disp[:, 0], disp[:, 1], color=color,
                    linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                    solid_capstyle='round', zorder=1)


def draw_boxes(ax,
               data: Dict[str, np.ndarray],
               class_names: Sequence[str],
               prefix: str,
               annotate_topk: int,
               alpha: float = 0.95) -> None:
    boxes = data['boxes']
    if boxes.size == 0:
        return
    corners = box_corners_bev(boxes)
    for rank, box_idx in enumerate(range(len(boxes))):
        track_id = int(data['track_ids'][box_idx])
        color = color_for_id(track_id)
        poly = lidar_xy_to_display(corners[box_idx])
        closed = np.concatenate([poly, poly[:1]], axis=0)
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.4,
                alpha=alpha)
        center = lidar_xy_to_display(boxes[box_idx:box_idx + 1, :2])[0]
        yaw = float(boxes[box_idx, 6])
        heading = lidar_xy_to_display(
            np.array([[math.cos(yaw), math.sin(yaw)]], dtype=np.float32))[0]
        ax.plot([center[0], center[0] + heading[0] * 1.2],
                [center[1], center[1] + heading[1] * 1.2],
                color=color, linewidth=0.9, linestyle='--', alpha=0.75)
        if rank >= annotate_topk:
            continue
        label = int(data['labels'][box_idx])
        label_name = class_names[label] if 0 <= label < len(class_names) else str(label)
        score = ''
        if 'scores' in data and data['scores'] is not None:
            score = f' {float(data["scores"][box_idx]):.2f}'
        ax.text(center[0], center[1], f'{prefix}{track_id}:{label_name[:5]}{score}',
                color=color, fontsize=6, ha='left', va='bottom')


def draw_gt_future(ax, gt_data: Dict[str, np.ndarray]) -> None:
    boxes = gt_data['boxes']
    fut = gt_data['fut']
    mask = gt_data['fut_mask']
    for idx in range(min(len(boxes), len(fut))):
        valid = mask[idx]
        if valid.ndim == 2:
            valid = valid[:, 0] > 0
        else:
            valid = valid > 0
        pts = fut[idx, valid, :2]
        if len(pts) == 0:
            continue
        abs_pts = np.concatenate(
            [boxes[idx:idx + 1, :2], boxes[idx, None, :2] + pts], axis=0)
        disp = lidar_xy_to_display(abs_pts)
        color = color_for_id(int(gt_data['track_ids'][idx]))
        ax.plot(disp[:, 0], disp[:, 1], color=color, linewidth=1.6,
                alpha=0.95)
        ax.scatter(disp[-1:, 0], disp[-1:, 1], color=color, s=12,
                   marker='o', alpha=0.95)


def draw_pred_traj(ax, pred_data: Dict[str, np.ndarray]) -> None:
    traj = best_traj(pred_data.get('traj'), pred_data.get('traj_scores'))
    if traj is None:
        return
    boxes = pred_data['boxes']
    for idx in range(min(len(boxes), len(traj))):
        abs_pts = np.concatenate(
            [boxes[idx:idx + 1, :2], boxes[idx, None, :2] + traj[idx, :, :2]],
            axis=0)
        disp = lidar_xy_to_display(abs_pts)
        color = color_for_id(int(pred_data['track_ids'][idx]))
        ax.plot(disp[:, 0], disp[:, 1], color=color, linewidth=1.6,
                alpha=0.95)
        ax.scatter(disp[-1:, 0], disp[-1:, 1], color=color, s=12,
                   marker='x', alpha=0.95)


def render_frame(points: np.ndarray,
                 gt_data: Dict[str, np.ndarray],
                 pred_data: Dict[str, np.ndarray],
                 class_names: Sequence[str],
                 pc_range: Sequence[float],
                 title: str,
                 out_path: str,
                 point_stride: int,
                 annotate_topk: int,
                 hdmap_lanes: Optional[Sequence[
                     Dict[str, Optional[np.ndarray]]]] = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 8.6), dpi=160)
    fig.patch.set_facecolor('black')
    left, right = axes
    gt_title = f'GT boxes + GT future paths ({len(gt_data["boxes"])})'
    if hdmap_lanes:
        gt_title += f' + HDMap lanes ({len(hdmap_lanes)})'
    setup_axis(left, pc_range,
               gt_title)
    setup_axis(right, pc_range,
               f'Pred tracks + predicted future paths ({len(pred_data["boxes"])})')
    for ax in axes:
        draw_points(ax, points, point_stride)
    draw_hdmap_lanes(left, hdmap_lanes)
    draw_boxes(left, gt_data, class_names, 'GT#', annotate_topk, alpha=0.75)
    draw_gt_future(left, gt_data)
    draw_boxes(right, pred_data, class_names, 'P#', annotate_topk, alpha=0.95)
    draw_pred_traj(right, pred_data)
    fig.suptitle(title, color='white', fontsize=11)
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.06, top=0.91, wspace=0.06)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def safe_token(token: str) -> str:
    return str(token).replace('/', '_').replace('\\', '_')


def write_html(out_dir: str, frame_files: List[str]) -> None:
    frames = [osp.basename(path) for path in frame_files]
    if not frames:
        return
    frame_items = ',\n      '.join(f'"{name}"' for name in frames)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>LiDAR E2E Motion</title>
<style>
body {{ margin:0; background:#101114; color:#f2f5fa; font-family:system-ui,sans-serif; }}
main {{ width:min(1500px, calc(100vw - 32px)); margin:0 auto; padding:18px 0 28px; }}
h1 {{ margin:0 0 12px; font-size:18px; }}
.stage {{ display:grid; place-items:center; min-height:360px; background:#050608; border:1px solid #343844; border-radius:6px; overflow:hidden; }}
img {{ max-width:100%; max-height:calc(100vh - 190px); }}
.controls {{ display:flex; gap:10px; align-items:center; margin-top:12px; padding:12px; background:#181a20; border:1px solid #343844; border-radius:6px; }}
button {{ height:34px; padding:0 12px; background:#222631; border:1px solid #343844; border-radius:5px; color:#f2f5fa; }}
input[type=range] {{ flex:1; accent-color:#64d2ff; }}
#frameText {{ color:#9aa4b2; font-size:13px; min-width:170px; }}
</style></head>
<body><main><h1>LiDAR E2E Motion Visualization</h1>
<section class="stage"><img id="frameImage"></section>
<section class="controls"><button id="prev">Prev</button><button id="play">Play</button><button id="next">Next</button><input id="slider" type="range" min="0" max="{len(frames)-1}" value="0"><span id="frameText"></span></section>
</main><script>
const frames=[{frame_items}]; let idx=0; let timer=null;
const img=document.getElementById('frameImage'), slider=document.getElementById('slider'), text=document.getElementById('frameText'), playBtn=document.getElementById('play');
function setFrame(i){{ idx=Math.max(0,Math.min(frames.length-1,i)); img.src=frames[idx]; slider.value=idx; text.textContent=`Frame ${{idx+1}} / ${{frames.length}}`; }}
function stop(){{ if(timer) clearInterval(timer); timer=null; playBtn.textContent='Play'; }}
function play(){{ stop(); playBtn.textContent='Pause'; timer=setInterval(()=>setFrame(idx>=frames.length-1?0:idx+1), 170); }}
playBtn.onclick=()=>timer?stop():play(); document.getElementById('prev').onclick=()=>{{stop();setFrame(idx-1)}}; document.getElementById('next').onclick=()=>{{stop();setFrame(idx+1)}}; slider.oninput=()=>{{stop();setFrame(Number(slider.value))}}; setFrame(0);
</script></body></html>"""
    with open(osp.join(out_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)


def write_webm(out_dir: str, fps: float, crf: int) -> None:
    if fps <= 0:
        return
    cmd = [
        'ffmpeg', '-y', '-framerate', str(float(fps)), '-pattern_type',
        'glob', '-i', osp.join(out_dir, '*.png'), '-vf',
        'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libvpx-vp9', '-b:v', '0',
        '-crf', str(int(crf)), osp.join(out_dir, 'pytorch_bev_vis.webm')
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print('[WARN] ffmpeg not found; skipped WebM generation.')
    except subprocess.CalledProcessError as exc:
        print(f'[WARN] ffmpeg failed with exit code {exc.returncode}.')


def build_pytorch_model(cfg: Config, checkpoint: str, device: str, dataset):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    ckpt = load_checkpoint(model, checkpoint, map_location='cpu')
    if 'CLASSES' in ckpt.get('meta', {}):
        model.CLASSES = ckpt['meta']['CLASSES']
    elif hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES
    torch_device = torch.device(device)
    if torch_device.type == 'cuda':
        if torch_device.index is not None:
            torch.cuda.set_device(torch_device.index)
        model = model.cuda(torch_device.index)
    else:
        model = model.to(torch_device)
    model.eval()
    return model


def scatter_batch(data, device: str):
    torch_device = torch.device(device)
    if torch_device.type != 'cuda':
        return data
    target = torch_device.index
    if target is None:
        target = torch.cuda.current_device()
    return scatter(data, [target])[0]


def model_forward_one(model, dataset, index: int, device: str):
    data = collate([dataset[index]], samples_per_gpu=1)
    data = scatter_batch(data, device)
    with torch.no_grad():
        output = model(return_loss=False, rescale=True, **data)
    if isinstance(output, (list, tuple)):
        return output[0] if output else {}
    return output


def reset_model_sequence_state(model) -> None:
    for name in ('test_track_instances', 'scene_token', 'timestamp', 'l2g_t',
                 'l2g_r_mat', '_test_track_instances', '_test_prev_bev',
                 '_test_scene_token', 'test_frame_token', 'prev_bev'):
        if hasattr(model, name):
            setattr(model, name, None)
    if hasattr(model, 'track_base') and hasattr(model.track_base, 'clear'):
        model.track_base.clear()


def is_sequence_break(prev_info: Optional[dict],
                      curr_info: dict,
                      max_time_gap: float = 1.0) -> bool:
    if prev_info is None:
        return True
    if curr_info.get('scene_token') != prev_info.get('scene_token'):
        return True
    prev_token = curr_info.get('prev', None)
    if prev_token == '':
        return True
    if prev_token is not None and prev_token != prev_info.get('token'):
        return True
    try:
        time_gap = abs(
            float(curr_info.get('timestamp', 0.0)) -
            float(prev_info.get('timestamp', 0.0)))
    except (TypeError, ValueError):
        return False
    return time_gap > max_time_gap


def run_visualization(cfg: Config, args: argparse.Namespace) -> None:
    dataset_cfg = get_dataset_cfg(cfg, args.split)
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    class_names = get_class_names(cfg, dataset_cfg)

    start_index = resolve_start_index(dataset, args)
    indices = consecutive_scene_indices(dataset, start_index, args.max_frames)
    if not indices:
        raise RuntimeError('No frames selected for visualization.')

    checkpoint = resolve_repo_path(args.checkpoint)
    model = build_pytorch_model(cfg, checkpoint, args.device, dataset)
    reset_model_sequence_state(model)

    hdmap_lanes = None
    hdmap_path = None
    if args.gt_map_overlay == 'hdmap':
        hdmap_path = resolve_hdmap_path(cfg, args)
        hdmap_lanes = load_hdmap_lanes(hdmap_path)
        print(f'[INFO] Loaded {len(hdmap_lanes)} HDMap lanes from {hdmap_path}')

    mmcv.mkdir_or_exist(args.out_dir)
    summary = []
    frame_files = []
    prev_info = None
    for frame_id, idx in enumerate(indices):
        info = raw_info_from_dataset(dataset, idx)
        sequence_reset = is_sequence_break(prev_info, info)
        if sequence_reset:
            reset_model_sequence_state(model)
        token = str(info.get('token', idx))
        save_path = osp.join(
            args.out_dir, f'{frame_id:03d}_{idx:06d}_{safe_token(token)}.png')
        if args.skip_existing and osp.exists(save_path):
            frame_files.append(save_path)
            prev_info = info
            continue

        points = load_points(cfg, dataset_cfg, info)
        result = model_forward_one(model, dataset, idx, args.device)
        ann = dataset.get_ann_info(idx)
        gt_data = gt_from_ann(ann, class_names)
        pred_data = pred_from_result(result, args.score_thr, args.topk)
        frame_hdmap_lanes = None
        if hdmap_lanes is not None:
            frame_hdmap_lanes = hdmap_lanes_for_frame(
                hdmap_lanes, info.get('ego2global', None),
                cfg.point_cloud_range, args.hdmap_max_lanes,
                args.hdmap_margin)
        scene = str(info.get('scene_token', ''))
        title = (
            f'base_e2e_lidar | frame={frame_id} index={idx} '
            f'token={token[:8]} scene={scene[-8:]} | '
            f'GT={len(gt_data["boxes"])} Pred={len(pred_data["boxes"])}')
        render_frame(
            points, gt_data, pred_data, class_names, cfg.point_cloud_range,
            title, save_path, args.point_stride, args.annotate_topk,
            hdmap_lanes=frame_hdmap_lanes)
        frame_files.append(save_path)
        summary.append(dict(
            frame=int(frame_id),
            index=int(idx),
            token=token,
            scene_token=info.get('scene_token'),
            sequence_reset=bool(sequence_reset),
            out_file=save_path,
            num_gt=int(len(gt_data['boxes'])),
            num_pred=int(len(pred_data['boxes'])),
            hdmap_path=hdmap_path,
            num_hdmap_lanes=0 if frame_hdmap_lanes is None else int(
                len(frame_hdmap_lanes)),
            gt_track_ids=[int(x) for x in gt_data['track_ids']],
            pred_track_ids=[int(x) for x in pred_data['track_ids']]))
        print(f'[OK] {save_path}')
        prev_info = info

    with open(osp.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    write_html(args.out_dir, frame_files)
    write_webm(args.out_dir, args.webm_fps, args.webm_crf)


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    cfg = Config.fromfile(config_path)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_cfg_modules(cfg, config_path)
    run_visualization(cfg, args)


if __name__ == '__main__':
    main()
