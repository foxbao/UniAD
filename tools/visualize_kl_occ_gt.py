#!/usr/bin/env python
"""Generate and visualize KL occupancy GT labels.

This script is a data-side smoke test for wiring OccHead later. It builds the
same core fields consumed by UniAD's GenerateOccFlowLabels from KL infos:
current + future boxes, instance ids, temporal validity, and ego poses.

Example:
    python tools/visualize_kl_occ_gt.py \
        --ann-file data/kl_8/kl_infos_val.pkl \
        --out-dir projects/work_dirs/vis_kl_occ_gt \
        --scene-token 202511191447_record --max-samples 24 \
        --skip-png
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from projects.mmdet3d_plugin.datasets.pipelines.occflow_label import (
    GenerateOccFlowLabels,
)


KL_CLASSES = (
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
)
DEFAULT_LABEL_MAPPING = (
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9,
    10, 11, 8, 8, 12,
)
DEFAULT_VEHICLE_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12)
DEFAULT_CLASS_NAMES = (
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
)
DEFAULT_GRID_CONF = {
    'xbound': [-80.0, 80.0, 0.8],
    'ybound': [-48.0, 48.0, 0.8],
    'zbound': [-10.0, 10.0, 20.0],
}
LABEL_COLORS = {
    '1': '#2563eb',
    '2': '#7c3aed',
    '3': '#f97316',
    '4': '#ec4899',
    '5': '#dc2626',
    '6': '#38bdf8',
    '7': '#a855f7',
    '8': '#64748b',
    '9': '#facc15',
    '10': '#06b6d4',
    '11': '#14b8a6',
    '12': '#8b5cf6',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize KL occupancy GT labels for OccHead.')
    parser.add_argument(
        '--ann-file',
        default='data/kl_8/kl_infos_val.pkl',
        help='KL info pickle, e.g. data/kl_8/kl_infos_val.pkl')
    parser.add_argument(
        '--out-dir',
        default='projects/work_dirs/vis_kl_occ_gt',
        help='directory for PNG/NPZ outputs')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--token', default=None)
    parser.add_argument('--scene-token', default=None)
    parser.add_argument('--max-samples', type=int, default=4)
    parser.add_argument('--future-frames', type=int, default=4)
    parser.add_argument('--receptive-field', type=int, default=3)
    parser.add_argument(
        '--grid-conf',
        default=None,
        help='JSON dict with xbound/ybound/zbound. Defaults to KL BEV 200x120.')
    parser.add_argument(
        '--vehicle-class-ids',
        default=','.join(str(x) for x in DEFAULT_VEHICLE_CLASS_IDS),
        help='comma-separated mapped KL class ids kept in occupancy labels')
    parser.add_argument(
        '--instance-id-offset',
        type=int,
        default=0,
        help='offset added to track_id before rasterizing; use 1 if ids contain 0')
    parser.add_argument(
        '--save-npz',
        action='store_true',
        help='save gt_segmentation/gt_instance arrays next to PNGs')
    parser.add_argument(
        '--skip-png',
        action='store_true',
        help='skip static PNG panels and only write the interactive HTML')
    parser.add_argument(
        '--html-name',
        default='index.html',
        help='interactive HTML filename written inside --out-dir')
    parser.add_argument(
        '--max-points-per-frame',
        type=int,
        default=50000,
        help='cap points rendered in the HTML right panel')
    parser.add_argument(
        '--cube-scale',
        type=float,
        default=0.9,
        help='cuboid size relative to the BEV cell size in the HTML')
    args = parser.parse_args()
    if args.max_samples <= 0:
        raise ValueError('--max-samples must be positive.')
    if args.future_frames < 0:
        raise ValueError('--future-frames must be non-negative.')
    if args.receptive_field < 1:
        raise ValueError('--receptive-field must be positive.')
    if args.max_points_per_frame < 0:
        raise ValueError('--max-points-per-frame must be non-negative.')
    if not 0.0 < args.cube_scale <= 1.0:
        raise ValueError('--cube-scale must be in (0, 1].')
    return args


def load_infos(ann_file: str) -> List[dict]:
    data = mmcv.load(ann_file)
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, list):
        return data
    raise TypeError(f'Unsupported info file format: {type(data)}')


def parse_grid_conf(raw: Optional[str]) -> Dict[str, List[float]]:
    if raw is None:
        return {k: list(v) for k, v in DEFAULT_GRID_CONF.items()}
    conf = json.loads(raw)
    for key in ('xbound', 'ybound', 'zbound'):
        if key not in conf or len(conf[key]) != 3:
            raise ValueError(f'grid_conf must contain {key}=[min,max,step]')
    return conf


def parse_int_list(raw: str) -> Tuple[int, ...]:
    if raw.strip() == '':
        return tuple()
    return tuple(int(part.strip()) for part in raw.split(','))


def build_token_index(infos: Sequence[dict]) -> Dict[str, int]:
    return {
        info.get('token'): idx
        for idx, info in enumerate(infos)
        if info.get('token')
    }


def resolve_start_index(infos: Sequence[dict], args: argparse.Namespace) -> int:
    if args.token is None and args.scene_token is None:
        if args.start_index < 0 or args.start_index >= len(infos):
            raise IndexError(f'--start-index {args.start_index} out of range')
        return args.start_index

    for idx, info in enumerate(infos):
        if args.token is not None and info.get('token') == args.token:
            return idx
        if args.scene_token is not None and (
                info.get('scene_token') == args.scene_token):
            return idx
    key = args.token if args.token is not None else args.scene_token
    raise KeyError(f'Cannot find token/scene-token: {key}')


def map_label(raw_label: int, label_mapping: Sequence[int]) -> int:
    raw_label = int(raw_label)
    if raw_label < 0 or raw_label >= len(label_mapping):
        return -1
    label = int(label_mapping[raw_label])
    return label if label >= 0 else -1


def valid_instance(inst: dict) -> bool:
    if not bool(inst.get('bbox_3d_isvalid', True)):
        return False
    return int(inst.get('num_lidar_pts', 1)) > 0


def ann_info_from_kl_info(info: dict,
                          label_mapping: Sequence[int],
                          instance_id_offset: int = 0) -> dict:
    bboxes = []
    labels = []
    track_ids = []

    for inst in info.get('instances', []):
        if not valid_instance(inst):
            continue
        label = map_label(
            inst.get('bbox_label_3d', inst.get('bbox_label', -1)),
            label_mapping)
        if label < 0:
            continue

        bbox = list(inst['bbox_3d'])
        if len(bbox) == 7:
            bbox.extend(inst.get('velocity', [0.0, 0.0]))
        bboxes.append(bbox)
        labels.append(label)
        track_ids.append(int(inst.get('track_id', -1)) + instance_id_offset)

    box_dim = 9
    if bboxes:
        bboxes_np = np.asarray(bboxes, dtype=np.float32)
    else:
        bboxes_np = np.zeros((0, box_dim), dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)
    track_ids_np = np.asarray(track_ids, dtype=np.int64)
    boxes = LiDARInstance3DBoxes(
        bboxes_np, box_dim=bboxes_np.shape[-1], origin=(0.5, 0.5, 0.5))
    return dict(
        gt_bboxes_3d=boxes,
        gt_labels_3d=labels_np,
        gt_inds=track_ids_np,
        gt_vis_tokens=None)


def collect_prev_indices(infos: Sequence[dict], token2index: Dict[str, int],
                         index: int, receptive_field: int) -> List[int]:
    scene_token = infos[index].get('scene_token')
    out = []
    cursor = infos[index]
    prev_token = cursor.get('prev', '')
    for _ in range(receptive_field - 1):
        prev_idx = token2index.get(prev_token, -1) if prev_token else -1
        if prev_idx < 0 or infos[prev_idx].get('scene_token') != scene_token:
            out.append(-1)
            prev_token = ''
            continue
        out.append(prev_idx)
        prev_token = infos[prev_idx].get('prev', '')
    out.reverse()
    return out


def collect_future_indices(infos: Sequence[dict], token2index: Dict[str, int],
                           index: int, n_future: int) -> List[int]:
    scene_token = infos[index].get('scene_token')
    out = []
    cursor = infos[index]
    next_token = cursor.get('next', '')
    for _ in range(n_future):
        next_idx = token2index.get(next_token, -1) if next_token else -1
        if next_idx < 0 or infos[next_idx].get('scene_token') != scene_token:
            out.append(-1)
            next_token = ''
            continue
        out.append(next_idx)
        next_token = infos[next_idx].get('next', '')
    return out


def pose_parts(info: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    ego2global = np.asarray(info.get('ego2global', np.eye(4)),
                            dtype=np.float32)
    return (torch.from_numpy(ego2global[:3, :3]),
            torch.from_numpy(ego2global[:3, 3]))


def build_occ_results(infos: Sequence[dict],
                      token2index: Dict[str, int],
                      index: int,
                      n_future: int,
                      receptive_field: int,
                      label_mapping: Sequence[int],
                      instance_id_offset: int = 0) -> dict:
    prev_indices = collect_prev_indices(infos, token2index, index,
                                        receptive_field)
    future_indices = collect_future_indices(infos, token2index, index,
                                            n_future)
    all_validity_frames = prev_indices + [index] + future_indices
    future_frames = [index] + future_indices

    ann_infos = []
    l2e_r_mats = []
    l2e_t_vecs = []
    e2g_r_mats = []
    e2g_t_vecs = []

    for frame_idx in future_frames:
        if frame_idx < 0:
            ann_infos.append(None)
            l2e_r_mats.append(None)
            l2e_t_vecs.append(None)
            e2g_r_mats.append(None)
            e2g_t_vecs.append(None)
            continue
        ann_infos.append(
            ann_info_from_kl_info(infos[frame_idx], label_mapping,
                                  instance_id_offset))
        e2g_r, e2g_t = pose_parts(infos[frame_idx])
        l2e_r_mats.append(torch.eye(3, dtype=torch.float32))
        l2e_t_vecs.append(torch.zeros(3, dtype=torch.float32))
        e2g_r_mats.append(e2g_r)
        e2g_t_vecs.append(e2g_t)

    return dict(
        future_gt_bboxes_3d=[
            None if ann is None else ann['gt_bboxes_3d'] for ann in ann_infos
        ],
        future_gt_labels_3d=[
            None if ann is None else ann['gt_labels_3d'] for ann in ann_infos
        ],
        future_gt_inds=[
            None if ann is None else ann['gt_inds'] for ann in ann_infos
        ],
        future_gt_vis_tokens=[
            None if ann is None else ann['gt_vis_tokens'] for ann in ann_infos
        ],
        occ_l2e_r_mats=l2e_r_mats,
        occ_l2e_t_vecs=l2e_t_vecs,
        occ_e2g_r_mats=e2g_r_mats,
        occ_e2g_t_vecs=e2g_t_vecs,
        occ_has_invalid_frame=any(idx < 0 for idx in all_validity_frames),
        occ_img_is_valid=np.asarray(
            [idx >= 0 for idx in all_validity_frames], dtype=np.bool_),
        _occ_prev_indices=prev_indices,
        _occ_future_indices=future_indices,
        _occ_future_frames=future_frames)


def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def find_lidar_path(info: dict, ann_file: str) -> Optional[str]:
    lidar_info = info.get('lidar_points', {})
    lidar_path = lidar_info.get('lidar_path', info.get('lidar_path'))
    if lidar_path is None:
        return None
    if osp.isabs(lidar_path) and osp.exists(lidar_path):
        return lidar_path
    if osp.exists(lidar_path):
        return lidar_path

    data_root = osp.dirname(ann_file)
    candidates = [
        osp.join(data_root, lidar_path),
        osp.join(data_root, 'samples', lidar_path),
        osp.join(data_root, 'v1.0-trainval', 'samples', lidar_path),
        osp.join(data_root, 'v1.0-mini', 'samples', lidar_path),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return None


def load_points_for_info(info: dict, ann_file: str,
                         max_points: int) -> Tuple[np.ndarray, int]:
    path = find_lidar_path(info, ann_file)
    if path is None:
        return np.zeros((0, 3), dtype=np.float32), 0
    num_feats = int(info.get('lidar_points', {}).get('num_pts_feats', 4))
    points = np.fromfile(path, dtype=np.float32)
    if points.size % num_feats != 0:
        points = points[:points.size - points.size % num_feats]
    points = points.reshape(-1, num_feats)[:, :3].astype(np.float32)
    total_points = int(points.shape[0])
    if max_points > 0 and total_points > max_points:
        keep = np.linspace(0, total_points - 1, max_points, dtype=np.int64)
        points = points[keep]
    return points, total_points


def box_corners_3d(box: Sequence[float]) -> List[List[float]]:
    box_np = np.asarray(box, dtype=np.float32).reshape(1, -1)
    boxes = LiDARInstance3DBoxes(
        box_np, box_dim=box_np.shape[-1], origin=(0.5, 0.5, 0.5))
    # Three.js edge code expects the first four corners to be the bottom face
    # in perimeter order, followed by the matching top-face corners.
    corner_order = [0, 3, 7, 4, 1, 2, 6, 5]
    return boxes.corners[0, corner_order].detach().cpu().numpy().tolist()


def points_boxes_frame(info: dict, ann_file: str,
                       label_mapping: Sequence[int],
                       max_points: int) -> dict:
    points, total_points = load_points_for_info(info, ann_file, max_points)
    boxes = []
    for inst in info.get('instances', []):
        if not valid_instance(inst):
            continue
        label = map_label(
            inst.get('bbox_label_3d', inst.get('bbox_label', -1)),
            label_mapping)
        if label < 0:
            continue
        track_id = int(inst.get('track_id', -1))
        boxes.append(
            dict(
                label=track_id,
                track_id=track_id,
                class_id=label,
                name=(DEFAULT_CLASS_NAMES[label]
                      if label < len(DEFAULT_CLASS_NAMES) else str(label)),
                corners=box_corners_3d(inst['bbox_3d'])))
    return dict(
        points=np.round(points, 3).reshape(-1).tolist(),
        numPoints=int(points.shape[0]),
        totalPoints=total_points,
        boxes=boxes,
        numBoxes=len(boxes))


def grid_payload(grid_conf: dict, cube_scale: float) -> dict:
    x0, x1, dx = [float(v) for v in grid_conf['xbound']]
    y0, y1, dy = [float(v) for v in grid_conf['ybound']]
    x_size = int(round((x1 - x0) / dx))
    y_size = int(round((y1 - y0) / dy))
    z_height = min(0.25, max(0.05, 0.25 * max(dx, dy)))
    return dict(
        pointCloudRange=[x0, y0, -0.02, x1, y1, z_height],
        occSize=[x_size, y_size, 1],
        voxelSize=[dx, dy, z_height],
        cubeScale=float(cube_scale))


def occ_step_to_frame(segmentation: np.ndarray, instance: np.ndarray,
                      future_step: int) -> dict:
    height, width = instance.shape
    labels = {}
    ids = np.unique(instance)
    ids = ids[(ids > 0) & (ids != 255)]
    for ins_id in ids:
        ys, xs = np.nonzero(instance == ins_id)
        if len(xs) == 0:
            continue
        flats = (xs.astype(np.int64) * height + ys.astype(np.int64))
        labels[str(int(ins_id))] = flats.astype(np.int32).tolist()

    if not labels and (segmentation == 1).any():
        ys, xs = np.nonzero(segmentation == 1)
        flats = (xs.astype(np.int64) * height + ys.astype(np.int64))
        labels['1'] = flats.astype(np.int32).tolist()

    unique, counts = np.unique(segmentation, return_counts=True)
    stats = {
        str(int(label)): int(count)
        for label, count in zip(unique, counts)
    }
    return dict(
        futureStep=int(future_step),
        stats=stats,
        occPixels=int((segmentation == 1).sum()),
        numInstances=int(len(ids)),
        labels=labels)


def sample_to_html_frame(info: dict,
                         ann_file: str,
                         raw_index: int,
                         segmentation: np.ndarray,
                         instance: np.ndarray,
                         valid_mask: Sequence[bool],
                         prev_indices: Sequence[int],
                         future_indices: Sequence[int],
                         max_points: int) -> dict:
    occ_steps = [
        occ_step_to_frame(segmentation[step], instance[step], step)
        for step in range(segmentation.shape[0])
    ]
    return dict(
        raw_index=int(raw_index),
        token=info.get('token', ''),
        timestamp=float(info.get('timestamp', 0.0)),
        scene_token=info.get('scene_token', ''),
        validMask=[bool(v) for v in valid_mask],
        prevIndices=[int(v) for v in prev_indices],
        futureIndices=[int(v) for v in future_indices],
        occSteps=occ_steps,
        pointsBoxes=points_boxes_frame(
            info, ann_file, DEFAULT_LABEL_MAPPING, max_points))


def build_occ_gt_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(',', ':'))
    html = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KL UniAD Occ GT</title>
<style>
html, body {
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #f8fafc;
  font-family: Arial, Helvetica, sans-serif;
}
#viewport { position: fixed; inset: 0; }
.divider {
  position: fixed;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 1px;
  background: rgba(15, 23, 42, 0.22);
  pointer-events: none;
}
.side-label {
  position: fixed;
  top: 12px;
  padding: 6px 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
  pointer-events: none;
}
.side-label.left { left: 14px; }
.side-label.right { left: calc(50% + 14px); }
.panel {
  position: fixed;
  left: 14px;
  right: 14px;
  bottom: 12px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(360px, 520px) minmax(0, 1fr);
  gap: 10px;
  align-items: end;
  pointer-events: none;
}
.box {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 8px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
  color: #0f172a;
  padding: 8px 10px;
  min-width: 0;
  box-sizing: border-box;
  pointer-events: auto;
}
.title {
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 6px;
}
.row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
}
button {
  width: 38px;
  height: 30px;
  border: 1px solid rgba(15, 23, 42, 0.22);
  border-radius: 6px;
  background: #ffffff;
  color: #0f172a;
  cursor: pointer;
  font-size: 13px;
}
input[type="range"] {
  flex: 1;
  min-width: 0;
}
.stats {
  font-size: 12px;
  line-height: 1.45;
  color: #334155;
  word-break: break-word;
}
.legend {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 4px 10px;
  font-size: 12px;
  margin-top: 6px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex: 0 0 auto;
}
.legend-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.error {
  position: fixed;
  inset: 20px;
  display: none;
  align-items: center;
  justify-content: center;
  color: #991b1b;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(153, 27, 27, 0.25);
  border-radius: 8px;
  font-size: 14px;
  padding: 20px;
  box-sizing: border-box;
}
@media (max-width: 900px) {
  .panel { grid-template-columns: minmax(0, 1fr); }
  .legend { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
</head>
<body>
<div id="viewport"></div>
<div class="divider"></div>
<div class="side-label left">Occ GT</div>
<div class="side-label right">Points + GT Boxes</div>
<div class="panel">
  <div id="occStats" class="box stats"></div>
  <div class="box">
    <div id="title" class="title"></div>
    <div class="row">
      <button id="play" title="Play or pause">Play</button>
      <input id="frame" type="range" min="0" value="0">
      <span id="frameText"></span>
    </div>
    <div class="row">
      <span>t+</span>
      <input id="future" type="range" min="0" value="0">
      <span id="futureText"></span>
    </div>
    <div id="legend" class="legend"></div>
  </div>
  <div id="pointsStats" class="box stats"></div>
</div>
<div id="error" class="error"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js">
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js">
</script>
<script>
const DATA = __DATA_JSON__;
const viewport = document.getElementById('viewport');
const errorBox = document.getElementById('error');
const frameSlider = document.getElementById('frame');
const futureSlider = document.getElementById('future');
const playButton = document.getElementById('play');
const frameText = document.getElementById('frameText');
const futureText = document.getElementById('futureText');
const titleBox = document.getElementById('title');
const occStatsBox = document.getElementById('occStats');
const pointsStatsBox = document.getElementById('pointsStats');
const legendBox = document.getElementById('legend');

if (!window.THREE || !THREE.OrbitControls) {
  errorBox.style.display = 'flex';
  errorBox.textContent = 'Three.js failed to load. Check network access.';
  throw new Error('Three.js failed to load.');
}

const frames = DATA.frames || [];
const range = DATA.pointCloudRange;
const occSize = DATA.occSize;
const voxelSize = DATA.voxelSize;
const scale = DATA.cubeScale;
const frameCount = frames.length;
const futureCount = DATA.futureSteps || 1;
frameSlider.max = Math.max(0, frameCount - 1);
futureSlider.max = Math.max(0, futureCount - 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setScissorTest(true);
viewport.appendChild(renderer.domElement);

const center = new THREE.Vector3(
  (range[0] + range[3]) * 0.5,
  (range[1] + range[4]) * 0.5,
  (range[2] + range[5]) * 0.5);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1200);
camera.up.set(0, 0, 1);
camera.position.set(center.x + 80, center.y - 125, center.z + 58);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

function makeScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8fafc);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb6c2cf, 0.92));
  const sun = new THREE.DirectionalLight(0xffffff, 0.72);
  sun.position.set(center.x - 30, center.y - 40, center.z + 120);
  scene.add(sun);

  const gridSize = Math.max(range[3] - range[0], range[4] - range[1]);
  const grid = new THREE.GridHelper(gridSize, 32, 0x94a3b8, 0xdbe3ee);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(center.x, center.y, 0);
  scene.add(grid);

  function addAxis(start, end, color) {
    const material = new THREE.LineBasicMaterial({ color });
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    scene.add(new THREE.Line(geometry, material));
  }
  addAxis(
    new THREE.Vector3(range[0], 0, 0),
    new THREE.Vector3(range[3], 0, 0),
    0xef4444);
  addAxis(
    new THREE.Vector3(0, range[1], 0),
    new THREE.Vector3(0, range[4], 0),
    0x22c55e);
  addAxis(
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, range[5]),
    0x2563eb);
  return scene;
}

const occScene = makeScene();
const pointsScene = makeScene();
const cubeGeometry = new THREE.BoxGeometry(
  voxelSize[0] * scale, voxelSize[1] * scale, voxelSize[2] * scale);
const dummy = new THREE.Object3D();
let occMeshes = [];
let rightObjects = [];
let activeFrame = 0;
let activeFuture = 0;
let timer = null;

function hashColor(label) {
  const hue = (Number(label) * 47) % 360;
  return `hsl(${hue}, 72%, 52%)`;
}

function labelColor(label) {
  return DATA.labelColors[String(label)] || hashColor(label);
}

function labelName(label) {
  return DATA.labelNames[String(label)] || `track ${label}`;
}

function decodeFlat(flat) {
  const yz = occSize[1] * occSize[2];
  const x = Math.floor(flat / yz);
  const rem = flat - x * yz;
  const y = Math.floor(rem / occSize[2]);
  const z = rem - y * occSize[2];
  return [x, y, z];
}

function voxelCenter(index) {
  return [
    range[0] + (index[0] + 0.5) * voxelSize[0],
    range[1] + (index[1] + 0.5) * voxelSize[1],
    range[2] + (index[2] + 0.5) * voxelSize[2],
  ];
}

function disposeObjects(objects, scene, disposeGeometry) {
  for (const obj of objects) {
    scene.remove(obj);
    if (disposeGeometry && obj.geometry) {
      obj.geometry.dispose();
    }
    if (obj.material) {
      obj.material.dispose();
    }
  }
  objects.length = 0;
}

function addLabelMesh(scene, meshes, label, flats) {
  if (!flats || flats.length === 0) {
    return;
  }
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(labelColor(label)),
    roughness: 0.82,
    metalness: 0.02
  });
  const mesh = new THREE.InstancedMesh(cubeGeometry, material, flats.length);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < flats.length; i++) {
    const ctr = voxelCenter(decodeFlat(flats[i]));
    dummy.position.set(ctr[0], ctr[1], ctr[2]);
    dummy.rotation.set(0, 0, 0);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  meshes.push(mesh);
}

function addOccStep(step) {
  const labels = Object.keys(step.labels || {})
    .map((label) => Number(label))
    .sort((a, b) => a - b);
  for (const label of labels) {
    addLabelMesh(occScene, occMeshes, label, step.labels[String(label)]);
  }
}

function addPoints(pointsFrame) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(pointsFrame.points), 3));
  const material = new THREE.PointsMaterial({
    color: 0x111827,
    size: 0.09,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.72
  });
  const cloud = new THREE.Points(geometry, material);
  pointsScene.add(cloud);
  rightObjects.push(cloud);
}

function addBox(box) {
  const c = box.corners;
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
  ];
  const positions = [];
  for (const edge of edges) {
    positions.push(...c[edge[0]], ...c[edge[1]]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(positions), 3));
  const material = new THREE.LineBasicMaterial({
    color: new THREE.Color(labelColor(box.label))
  });
  const line = new THREE.LineSegments(geometry, material);
  pointsScene.add(line);
  rightObjects.push(line);
}

function addPointsBoxes(pointsFrame) {
  addPoints(pointsFrame);
  for (const box of pointsFrame.boxes || []) {
    addBox(box);
  }
}

function formatOccStats(frame, step) {
  const stats = step.stats || {};
  const free = stats['0'] || 0;
  const occ = step.occPixels || 0;
  return `Occ GT: raw_index=${frame.raw_index} | t+${step.futureStep}<br>` +
    `occupied_cells=${occ} | free_cells=${free} | ` +
    `instances=${step.numInstances}<br>` +
    `valid_mask=${JSON.stringify(frame.validMask)}<br>` +
    `future_indices=${JSON.stringify(frame.futureIndices)}`;
}

function formatPointsStats(frame) {
  const pb = frame.pointsBoxes;
  return `Points + boxes: raw_index=${frame.raw_index}<br>` +
    `points=${pb.numPoints}/${pb.totalPoints} | boxes=${pb.numBoxes}<br>` +
    `scene=${frame.scene_token}`;
}

function renderLegend(frame, step) {
  const labels = new Set();
  for (const label of Object.keys(step.labels || {})) {
    labels.add(Number(label));
  }
  for (const box of frame.pointsBoxes.boxes || []) {
    labels.add(Number(box.label));
  }
  const items = Array.from(labels).sort((a, b) => a - b).slice(0, 36).map(
    (label) =>
      `<div class="legend-item" title="${labelName(label)}">` +
      `<span class="swatch" style="background:${labelColor(label)}"></span>` +
      `<span class="legend-text">${label}: ${labelName(label)}</span></div>`);
  legendBox.innerHTML = items.join('');
}

function renderFrame(index, futureIndex) {
  if (frameCount === 0) {
    titleBox.textContent = 'No frames found';
    return;
  }
  activeFrame = Math.max(0, Math.min(frameCount - 1, index));
  const frame = frames[activeFrame];
  const maxFuture = Math.max(0, frame.occSteps.length - 1);
  activeFuture = Math.max(0, Math.min(maxFuture, futureIndex));
  const step = frame.occSteps[activeFuture];

  disposeObjects(occMeshes, occScene, false);
  disposeObjects(rightObjects, pointsScene, true);
  addOccStep(step);
  addPointsBoxes(frame.pointsBoxes);

  frameSlider.value = String(activeFrame);
  futureSlider.value = String(activeFuture);
  frameText.textContent = `${activeFrame + 1} / ${frameCount}`;
  futureText.textContent = `${activeFuture} / ${maxFuture}`;
  titleBox.textContent =
    `raw_index=${frame.raw_index} | token=${frame.token || ''}`;
  occStatsBox.innerHTML = formatOccStats(frame, step);
  pointsStatsBox.innerHTML = formatPointsStats(frame);
  renderLegend(frame, step);
}

function draw() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const half = Math.floor(width / 2);
  renderer.clear();

  camera.aspect = half / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(0, 0, half, height);
  renderer.setScissor(0, 0, half, height);
  renderer.render(occScene, camera);

  const rightWidth = width - half;
  camera.aspect = rightWidth / Math.max(1, height);
  camera.updateProjectionMatrix();
  renderer.setViewport(half, 0, rightWidth, height);
  renderer.setScissor(half, 0, rightWidth, height);
  renderer.render(pointsScene, camera);
}

frameSlider.addEventListener('input', () => {
  renderFrame(Number(frameSlider.value), activeFuture);
});

futureSlider.addEventListener('input', () => {
  renderFrame(activeFrame, Number(futureSlider.value));
});

playButton.addEventListener('click', () => {
  if (timer) {
    clearInterval(timer);
    timer = null;
    playButton.textContent = 'Play';
    return;
  }
  playButton.textContent = 'Pause';
  timer = setInterval(() => {
    renderFrame((activeFrame + 1) % frameCount, activeFuture);
  }, 650);
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight') {
    renderFrame((activeFrame + 1) % frameCount, activeFuture);
  }
  if (event.key === 'ArrowLeft') {
    renderFrame((activeFrame - 1 + frameCount) % frameCount, activeFuture);
  }
  if (event.key === 'ArrowUp') {
    renderFrame(activeFrame, activeFuture + 1);
  }
  if (event.key === 'ArrowDown') {
    renderFrame(activeFrame, activeFuture - 1);
  }
});

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  draw();
}

renderFrame(0, 0);
animate();
</script>
</body>
</html>
"""
    return html.replace('__DATA_JSON__', data_json)


def make_instance_rgb(instance: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*instance.shape, 3), dtype=np.float32)
    ids = np.unique(instance)
    ids = ids[(ids > 0) & (ids != 255)]
    for ins_id in ids:
        rng = np.random.default_rng(int(ins_id) * 7919)
        rgb[instance == ins_id] = rng.uniform(0.2, 1.0, size=3)
    ignore = instance == 255
    rgb[ignore] = np.asarray([0.35, 0.35, 0.35])
    return rgb


def draw_occ_panels(segmentation: np.ndarray,
                    instance: np.ndarray,
                    grid_conf: dict,
                    out_path: str,
                    title: str) -> None:
    num_frames = segmentation.shape[0]
    x0, x1, _ = grid_conf['xbound']
    y0, y1, _ = grid_conf['ybound']
    extent = [x0, x1, y0, y1]

    fig, axes = plt.subplots(
        2, num_frames, figsize=(4.0 * num_frames, 7.5), squeeze=False)
    for frame_idx in range(num_frames):
        seg = segmentation[frame_idx]
        ins = instance[frame_idx]

        ax = axes[0, frame_idx]
        ax.imshow(seg, origin='lower', extent=extent, cmap='gray_r', vmin=0,
                  vmax=1)
        ax.scatter([0], [0], c='tab:red', s=18)
        ax.set_title(f't+{frame_idx} occupancy')
        ax.set_aspect('equal')
        ax.grid(color='0.85', linewidth=0.4)

        ax = axes[1, frame_idx]
        ax.imshow(make_instance_rgb(ins), origin='lower', extent=extent)
        ax.scatter([0], [0], c='tab:red', s=18)
        ax.set_title(f't+{frame_idx} instance')
        ax.set_aspect('equal')
        ax.grid(color='0.85', linewidth=0.4)

    for ax in axes.reshape(-1):
        ax.set_xlabel('x / m')
        ax.set_ylabel('y / m')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def summarize_label(segmentation: np.ndarray, instance: np.ndarray) -> dict:
    frames = []
    for frame_idx in range(segmentation.shape[0]):
        seg = segmentation[frame_idx]
        ins = instance[frame_idx]
        ids = np.unique(ins)
        ids = ids[(ids > 0) & (ids != 255)]
        frames.append(
            dict(
                frame=frame_idx,
                occ_pixels=int((seg == 1).sum()),
                num_instances=int(len(ids)),
                instance_ids=[int(x) for x in ids[:30]]))
    return dict(frames=frames)


def main() -> None:
    args = parse_args()
    infos = load_infos(args.ann_file)
    token2index = build_token_index(infos)
    start_index = resolve_start_index(infos, args)
    grid_conf = parse_grid_conf(args.grid_conf)
    vehicle_class_ids = parse_int_list(args.vehicle_class_ids)
    generator = GenerateOccFlowLabels(
        grid_conf=grid_conf,
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=False,
        filter_cls_ids=vehicle_class_ids)

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    html_frames = []
    label_names = {}
    index = start_index
    scene_token = infos[start_index].get('scene_token')
    for sample_id in range(args.max_samples):
        if index >= len(infos):
            break
        info = infos[index]
        if info.get('scene_token') != scene_token:
            break

        results = build_occ_results(
            infos=infos,
            token2index=token2index,
            index=index,
            n_future=args.future_frames,
            receptive_field=args.receptive_field,
            label_mapping=DEFAULT_LABEL_MAPPING,
            instance_id_offset=args.instance_id_offset)
        occ = generator(results)
        segmentation = tensor_to_numpy(occ['gt_segmentation'])
        instance = tensor_to_numpy(occ['gt_instance'])
        token = info.get('token', str(index))
        stem = f'{sample_id:03d}_idx_{index}_token_{token[:8]}'

        out_png = None
        if not args.skip_png:
            out_png = osp.join(args.out_dir, f'{stem}_occ_gt.png')
            draw_occ_panels(
                segmentation,
                instance,
                grid_conf,
                out_png,
                title=f'{info.get("scene_token", "")} / {token}')

        if args.save_npz:
            np.savez_compressed(
                osp.join(args.out_dir, f'{stem}_occ_gt.npz'),
                gt_segmentation=segmentation,
                gt_instance=instance,
                gt_occ_img_is_valid=tensor_to_numpy(
                    occ['gt_occ_img_is_valid']))

        summary = dict(
            index=int(index),
            token=token,
            scene_token=info.get('scene_token', ''),
            prev_indices=[int(x) for x in results['_occ_prev_indices']],
            future_indices=[int(x) for x in results['_occ_future_indices']],
            valid_mask=[
                bool(x) for x in tensor_to_numpy(occ['gt_occ_img_is_valid'])
            ],
            output_png=out_png)
        summary.update(summarize_label(segmentation, instance))
        summaries.append(summary)
        html_frame = sample_to_html_frame(
            info=info,
            ann_file=args.ann_file,
            raw_index=index,
            segmentation=segmentation,
            instance=instance,
            valid_mask=summary['valid_mask'],
            prev_indices=results['_occ_prev_indices'],
            future_indices=results['_occ_future_indices'],
            max_points=args.max_points_per_frame)
        html_frames.append(html_frame)
        for step in html_frame['occSteps']:
            for label in step['labels'].keys():
                label_names.setdefault(label, f'track {label}')
        for box in html_frame['pointsBoxes']['boxes']:
            label_names.setdefault(str(box['label']), f'track {box["label"]}')

        next_token = info.get('next', '')
        next_index = token2index.get(next_token, -1) if next_token else -1
        if next_index < 0:
            break
        index = next_index

    summary_path = osp.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    payload = grid_payload(grid_conf, args.cube_scale)
    payload.update(
        frames=html_frames,
        futureSteps=args.future_frames + 1,
        labelNames=label_names,
        labelColors=LABEL_COLORS,
        hiddenLabels=[0, 255])
    html_path = osp.join(args.out_dir, args.html_name)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(build_occ_gt_html(payload))
    print(f'Wrote {len(summaries)} samples to {args.out_dir}')
    print(f'Summary: {summary_path}')
    print(f'HTML: {html_path}')


if __name__ == '__main__':
    main()
