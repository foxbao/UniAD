#!/usr/bin/env python
"""Visualize BEVFormer-LiDAR pure PyTorch inference in BEV.

This script is adapted from the KL visualization helpers. It directly builds
the UniAD/OpenMMLab dataset and model, runs normal PyTorch inference, then
renders point cloud, GT boxes, and predicted boxes as BEV PNG frames.

Example:
    conda activate uniad_train
    python tools/visualize_bevformer_lidar_pytorch.py \
        --config projects/configs/bevformer_lidar/base_bevformer_lidar.py \
        --checkpoint projects/work_dirs/bevformer_lidar/base_bevformer_lidar/epoch_2.pth \
        --out-dir projects/work_dirs/vis_bevformer_lidar_pytorch \
        --start-index 0 --max-frames 4 --score-thr 0.05 --annotate
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import os.path as osp
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model

from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from third_party.uniad_mmdet3d.models.builder import build_model


KL_CLASSES = (
    "Pedestrian",
    "Car",
    "IGV-Full",
    "Truck",
    "Trailer-Empty",
    "Trailer-Full",
    "IGV-Empty",
    "Crane",
    "OtherVehicle",
    "Cone",
    "ContainerForklift",
    "Forklift",
    "Lorry",
    "ConstructionVehicle",
    "WheelCrane",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize pure PyTorch BEVFormer-LiDAR predictions.")
    parser.add_argument(
        "--config",
        default="projects/configs/bevformer_lidar/base_bevformer_lidar.py",
        help="UniAD LiDAR config path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Which dataset split from the config to visualize.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--token",
        default=None,
        help="Start from this sample token instead of --start-index.")
    parser.add_argument(
        "--scene-token",
        default=None,
        help="Start from the first sample of this scene token.")
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=120)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--vel-scale", type=float, default=1.2)
    parser.add_argument("--min-vel-draw", type=float, default=0.2)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--webm-fps",
        type=float,
        default=0.0,
        help="FPS for optional pytorch_bev_vis.webm. Set <=0 to skip.")
    parser.add_argument("--webm-crf", type=int, default=34)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. model.score_thresh=0.7")
    args = parser.parse_args()
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.point_stride <= 0:
        raise ValueError("--point-stride must be positive.")
    if args.topk <= 0:
        raise ValueError("--topk must be positive.")
    return args


def resolve_repo_path(path: str) -> str:
    if osp.isabs(path) or osp.exists(path):
        return path
    return osp.join(REPO_ROOT, path)


def import_cfg_modules(cfg: Config, config_path: str) -> None:
    custom_imports = cfg.get("custom_imports")
    if custom_imports:
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**custom_imports)

    if hasattr(cfg, "plugin") and cfg.plugin:
        if hasattr(cfg, "plugin_dir"):
            module_dir = osp.dirname(cfg.plugin_dir)
        else:
            module_dir = osp.dirname(config_path)
        module_path = module_dir.replace("/", ".").strip(".")
        if module_path:
            importlib.import_module(module_path)


def get_dataset_cfg(cfg: Config, split: str):
    return cfg.data.val if split == "val" else cfg.data.test


def get_class_names(cfg: Config, dataset_cfg) -> Sequence[str]:
    if dataset_cfg.get("classes", None) is not None:
        return list(dataset_cfg.classes)
    if cfg.get("class_names", None) is not None:
        return list(cfg.class_names)
    return list(KL_CLASSES)


def get_label_mapping(cfg: Config, dataset_cfg) -> Optional[List[int]]:
    mapping = dataset_cfg.get("label_mapping", None)
    if mapping is None:
        mapping = cfg.get("label_mapping", None)
    if mapping is None:
        return None
    return [int(x) for x in mapping]


def raw_info_from_dataset(dataset, index: int) -> dict:
    raw_index = dataset._to_raw_index(index) if hasattr(
        dataset, "_to_raw_index") else index
    return dataset.data_infos[raw_index]


def resolve_start_index(dataset, args: argparse.Namespace) -> int:
    if args.token is None and args.scene_token is None:
        if args.start_index < 0 or args.start_index >= len(dataset):
            raise IndexError(
                f"--start-index {args.start_index} is out of range.")
        return args.start_index

    for idx in range(len(dataset)):
        info = raw_info_from_dataset(dataset, idx)
        if args.token is not None and info.get("token") == args.token:
            return idx
        if args.scene_token is not None and (
                info.get("scene_token") == args.scene_token):
            return idx
    key = args.token if args.token is not None else args.scene_token
    raise KeyError(f"Cannot find requested token/scene: {key}")


def consecutive_scene_indices(
        dataset, start_index: int, max_frames: int) -> List[int]:
    start_info = raw_info_from_dataset(dataset, start_index)
    scene_token = start_info.get("scene_token")
    indices = []
    for idx in range(start_index, len(dataset)):
        info = raw_info_from_dataset(dataset, idx)
        if info.get("scene_token") != scene_token:
            break
        indices.append(idx)
        if len(indices) >= max_frames:
            break
    return indices


def resolve_lidar_path(cfg: Config, dataset_cfg, info: dict) -> str:
    data_root = dataset_cfg.get("data_root", cfg.get("data_root", ""))
    data_prefix = dataset_cfg.get("data_prefix", cfg.get("data_prefix", {}))
    pts_prefix = data_prefix.get("pts", "")
    lidar_info = info.get("lidar_points", {})
    lidar_path = lidar_info.get("lidar_path", info.get("lidar_path"))
    if lidar_path is None:
        raise KeyError("info does not contain lidar path")
    if osp.isabs(lidar_path) or osp.exists(lidar_path):
        return lidar_path
    return osp.join(resolve_repo_path(data_root), pts_prefix, lidar_path)


def load_points(cfg: Config, dataset_cfg, info: dict) -> np.ndarray:
    lidar_path = resolve_lidar_path(cfg, dataset_cfg, info)
    lidar_info = info.get("lidar_points", {})
    num_feats = int(lidar_info.get("num_pts_feats",
                                   info.get("num_features", 4)))
    points = np.fromfile(lidar_path, dtype=np.float32)
    return points.reshape(-1, num_feats)


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


def empty_arrays() -> Dict[str, np.ndarray]:
    return dict(
        boxes=np.zeros((0, 9), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.int64),
        scores=np.zeros((0,), dtype=np.float32))


def gt_arrays_from_info(info: dict, dataset_cfg, class_names: Sequence[str],
                        label_mapping: Optional[List[int]]
                        ) -> Dict[str, np.ndarray]:
    use_valid_flag = bool(dataset_cfg.get("use_valid_flag", False))
    boxes = []
    labels = []

    if info.get("instances"):
        for inst in info["instances"]:
            raw_label = inst.get("bbox_label_3d", inst.get("bbox_label", -1))
            label = map_label(raw_label, label_mapping, len(class_names))
            if label < 0:
                continue
            if use_valid_flag:
                keep = bool(inst.get("bbox_3d_isvalid", False))
            else:
                keep = int(inst.get("num_lidar_pts", 0)) > 0
            if not keep:
                continue
            box = np.asarray(inst["bbox_3d"], dtype=np.float32)
            vel = np.asarray(inst.get("velocity", [0.0, 0.0]),
                             dtype=np.float32)
            if box.shape[0] == 7:
                box = np.concatenate([box, vel[:2]], axis=0)
            elif box.shape[0] >= 9:
                box = box.copy()
                box[7:9] = vel[:2]
            if box.shape[0] < 9:
                box = np.pad(box, (0, 9 - box.shape[0]))
            boxes.append(box[:9])
            labels.append(label)
    elif "gt_boxes" in info:
        gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float32)
        gt_names = info.get("gt_names", [])
        valid_flags = info.get("valid_flag", [True] * len(gt_boxes))
        name_to_label = {name: idx for idx, name in enumerate(class_names)}
        for box, name, valid in zip(gt_boxes, gt_names, valid_flags):
            if use_valid_flag and not valid:
                continue
            label = name_to_label.get(str(name), -1)
            if label < 0:
                continue
            if box.shape[0] < 9:
                box = np.pad(box, (0, 9 - box.shape[0]))
            boxes.append(box[:9])
            labels.append(label)

    if not boxes:
        return empty_arrays()
    return dict(
        boxes=np.asarray(boxes, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scores=np.ones((len(boxes),), dtype=np.float32))


def tensor_to_numpy(value) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def boxes_to_numpy(boxes_3d) -> np.ndarray:
    if boxes_3d is None:
        return np.zeros((0, 9), dtype=np.float32)
    if hasattr(boxes_3d, "tensor"):
        boxes = boxes_3d.tensor.detach().cpu().numpy()
    else:
        boxes = tensor_to_numpy(boxes_3d)
    boxes = boxes.astype(np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[1] < 9:
        pad = np.zeros((boxes.shape[0], 9 - boxes.shape[1]),
                       dtype=boxes.dtype)
        boxes = np.concatenate([boxes, pad], axis=1)
    return boxes[:, :9]


def unwrap_model_result(result):
    if isinstance(result, (list, tuple)):
        result = result[0] if result else {}
    if isinstance(result, dict) and "pts_bbox" in result:
        return result["pts_bbox"]
    return result if isinstance(result, dict) else {}


def pred_arrays_from_result(result, score_thr: float,
                            topk: int) -> Dict[str, np.ndarray]:
    result = unwrap_model_result(result)
    if not isinstance(result, dict):
        return empty_arrays()
    if "boxes_3d_det" in result and (
            "boxes_3d" not in result or len(result.get("boxes_3d", [])) == 0):
        boxes = boxes_to_numpy(result.get("boxes_3d_det"))
        scores = tensor_to_numpy(result.get("scores_3d_det"))
        labels = tensor_to_numpy(result.get("labels_3d_det"))
    else:
        boxes = boxes_to_numpy(result.get("boxes_3d"))
        scores = tensor_to_numpy(result.get("scores_3d"))
        labels = tensor_to_numpy(result.get("labels_3d"))
    if len(boxes) == 0:
        return empty_arrays()
    scores = scores.astype(np.float32).reshape(-1)
    labels = labels.astype(np.int64).reshape(-1)
    num = min(len(boxes), len(scores), len(labels))
    boxes = boxes[:num]
    scores = scores[:num]
    labels = labels[:num]
    order = np.argsort(-scores)
    keep = order[scores[order] >= score_thr][:topk]
    return dict(boxes=boxes[keep], labels=labels[keep], scores=scores[keep])


def compute_box_corners_bev(boxes: np.ndarray) -> np.ndarray:
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


def draw_points_and_ego(ax, points: np.ndarray, point_stride: int) -> None:
    pts = points[::point_stride, :3]
    if len(pts):
        pts_disp = lidar_xy_to_display(pts[:, :2])
        ax.scatter(
            pts_disp[:, 0],
            pts_disp[:, 1],
            s=0.12,
            c="white",
            alpha=0.32,
            linewidths=0)
    ax.plot(0.0, 0.0, marker="o", markersize=4, color="#ffd166")
    ax.arrow(
        0.0,
        0.0,
        0.0,
        3.0,
        color="#ffd166",
        width=0.03,
        head_width=0.5,
        head_length=0.6,
        length_includes_head=True)


def draw_boxes(ax,
               boxes: np.ndarray,
               labels: np.ndarray,
               scores: Optional[np.ndarray],
               class_names: Sequence[str],
               color: str,
               vel_scale: float,
               min_vel_draw: float,
               annotate: bool,
               alpha: float = 1.0) -> None:
    if boxes.size == 0:
        return
    corners = compute_box_corners_bev(boxes)
    for idx, box in enumerate(boxes):
        poly_disp = lidar_xy_to_display(corners[idx])
        closed = np.concatenate([poly_disp, poly_disp[:1]], axis=0)
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.4,
                alpha=alpha)

        center_disp = lidar_xy_to_display(
            np.array([[box[0], box[1]]], dtype=np.float32))[0]
        yaw = float(box[6])
        heading_len = min(max(float(box[3]) * 0.22, 0.45), 1.35)
        heading = lidar_xy_to_display(
            np.array([[math.cos(yaw), math.sin(yaw)]],
                     dtype=np.float32))[0]
        ax.plot(
            [center_disp[0], center_disp[0] + heading[0] * heading_len],
            [center_disp[1], center_disp[1] + heading[1] * heading_len],
            color=color,
            linewidth=0.9,
            linestyle="--",
            alpha=0.65)

        vx = float(box[7])
        vy = float(box[8])
        speed = math.hypot(vx, vy)
        if speed >= min_vel_draw:
            vel_disp = lidar_xy_to_display(
                np.array([[vx, vy]], dtype=np.float32))[0]
            ax.arrow(
                float(center_disp[0]),
                float(center_disp[1]),
                float(vel_disp[0]) * vel_scale,
                float(vel_disp[1]) * vel_scale,
                color=color,
                width=0.025,
                head_width=0.45,
                head_length=0.55,
                length_includes_head=True,
                alpha=alpha)

        if annotate:
            label = int(labels[idx])
            label_name = class_names[label] if 0 <= label < len(class_names) \
                else str(label)
            score_text = "" if scores is None else f" {float(scores[idx]):.2f}"
            ax.text(
                float(center_disp[0]),
                float(center_disp[1]),
                f"{label_name}{score_text} {speed:.1f}m/s",
                color=color,
                fontsize=6,
                ha="left",
                va="bottom",
                alpha=alpha)


def setup_bev_axis(ax, pc_range: Sequence[float], subtitle: str) -> None:
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    ax.set_facecolor("black")
    ax.set_xlim(-y_max, -y_min)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Lateral (m)", color="white")
    ax.set_ylabel("Forward (m)", color="white")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#888888")
    ax.grid(color="#444444", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title(subtitle, color="white", fontsize=10)


def render_frame(points: np.ndarray,
                 gt_data: Dict[str, np.ndarray],
                 pred_data: Dict[str, np.ndarray],
                 class_names: Sequence[str],
                 save_path: str,
                 title: str,
                 pc_range: Sequence[float],
                 point_stride: int,
                 vel_scale: float,
                 min_vel_draw: float,
                 annotate: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 8.8), dpi=160)
    fig.patch.set_facecolor("black")
    left_ax, right_ax = axes

    setup_bev_axis(left_ax, pc_range, f"GT boxes ({len(gt_data['boxes'])})")
    setup_bev_axis(
        right_ax, pc_range,
        f"PyTorch prediction ({len(pred_data['boxes'])})")
    draw_points_and_ego(left_ax, points, point_stride)
    draw_points_and_ego(right_ax, points, point_stride)
    draw_boxes(
        left_ax,
        gt_data["boxes"],
        gt_data["labels"],
        None,
        class_names,
        color="#9cffbf",
        vel_scale=vel_scale,
        min_vel_draw=min_vel_draw,
        annotate=False,
        alpha=0.65)
    draw_boxes(
        right_ax,
        pred_data["boxes"],
        pred_data["labels"],
        pred_data["scores"],
        class_names,
        color="#ff5b5b",
        vel_scale=vel_scale,
        min_vel_draw=min_vel_draw,
        annotate=annotate,
        alpha=0.95)
    fig.suptitle(title, color="white", fontsize=11)
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.07, top=0.91, wspace=0.06)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def safe_token(token: str) -> str:
    return str(token).replace("/", "_").replace("\\", "_")


def write_html_player(out_dir: str, frame_files: List[str]) -> None:
    frames = [osp.basename(path) for path in frame_files]
    if not frames:
        return
    frame_items = ",\n      ".join(f'"{name}"' for name in frames)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BEVFormer-LiDAR PyTorch Visualization</title>
  <style>
    body {{ margin: 0; background: #101114; color: #f2f5fa; font-family: system-ui, sans-serif; }}
    main {{ width: min(1500px, calc(100vw - 32px)); margin: 0 auto; padding: 18px 0 28px; }}
    header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 12px; }}
    h1 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    #frameText {{ color: #9aa4b2; font-size: 13px; white-space: nowrap; }}
    .stage {{ display: grid; place-items: center; min-height: 360px; background: #050608; border: 1px solid #343844; border-radius: 6px; overflow: hidden; }}
    #frameImage {{ display: block; max-width: 100%; max-height: calc(100vh - 210px); width: auto; height: auto; }}
    .controls {{ display: grid; gap: 10px; margin-top: 12px; padding: 12px; background: #181a20; border: 1px solid #343844; border-radius: 6px; }}
    .row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    button {{ height: 34px; min-width: 42px; padding: 0 12px; border: 1px solid #343844; border-radius: 5px; background: #222631; color: #f2f5fa; font: inherit; cursor: pointer; }}
    button.primary {{ background: #113548; border-color: #26637d; }}
    input[type="range"] {{ flex: 1 1 300px; min-width: 160px; accent-color: #64d2ff; }}
    input[type="number"] {{ width: 72px; height: 32px; border: 1px solid #343844; border-radius: 5px; background: #11141a; color: #f2f5fa; padding: 0 8px; font: inherit; }}
    label {{ display: inline-flex; align-items: center; gap: 6px; color: #9aa4b2; font-size: 13px; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>BEVFormer-LiDAR PyTorch Visualization</h1>
      <div id="frameText"></div>
    </header>
    <section class="stage"><img id="frameImage" alt="PyTorch BEV frame"></section>
    <section class="controls">
      <div class="row">
        <button id="prevBtn">Prev</button>
        <button id="playBtn" class="primary">Play</button>
        <button id="nextBtn">Next</button>
        <label>FPS <input id="fpsInput" type="number" value="3" min="0.5" max="30" step="0.5"></label>
        <label><input id="loopInput" type="checkbox" checked> Loop</label>
      </div>
      <div class="row"><input id="frameSlider" type="range" min="0" max="{len(frames) - 1}" value="0" step="1"></div>
    </section>
  </main>
  <script>
    const frames = [
      {frame_items}
    ];
    const image = document.getElementById("frameImage");
    const text = document.getElementById("frameText");
    const slider = document.getElementById("frameSlider");
    const playBtn = document.getElementById("playBtn");
    const fpsInput = document.getElementById("fpsInput");
    const loopInput = document.getElementById("loopInput");
    let index = 0;
    let timer = null;
    function setFrame(nextIndex) {{
      index = Math.max(0, Math.min(frames.length - 1, nextIndex));
      image.src = frames[index];
      slider.value = index;
      text.textContent = `Frame ${{index + 1}} / ${{frames.length}} | ${{frames[index]}}`;
    }}
    function stop() {{
      if (timer !== null) {{ clearInterval(timer); timer = null; }}
      playBtn.textContent = "Play";
    }}
    function play() {{
      stop();
      playBtn.textContent = "Pause";
      const fps = Math.max(0.5, Number(fpsInput.value) || 3);
      timer = setInterval(() => {{
        if (index >= frames.length - 1) {{
          if (!loopInput.checked) {{ stop(); return; }}
          setFrame(0); return;
        }}
        setFrame(index + 1);
      }}, 1000 / fps);
    }}
    function togglePlay() {{ timer === null ? play() : stop(); }}
    document.getElementById("prevBtn").addEventListener("click", () => {{ stop(); setFrame(index - 1); }});
    document.getElementById("nextBtn").addEventListener("click", () => {{ stop(); setFrame(index + 1); }});
    playBtn.addEventListener("click", togglePlay);
    fpsInput.addEventListener("change", () => {{ if (timer !== null) play(); }});
    slider.addEventListener("input", () => {{ stop(); setFrame(Number(slider.value)); }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === " ") {{ event.preventDefault(); togglePlay(); }}
      else if (event.key === "ArrowLeft") {{ stop(); setFrame(index - 1); }}
      else if (event.key === "ArrowRight") {{ stop(); setFrame(index + 1); }}
    }});
    setFrame(0);
  </script>
</body>
</html>
"""
    with open(osp.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def write_webm_video(out_dir: str, fps: float, crf: int) -> None:
    if fps <= 0:
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(float(fps)),
        "-pattern_type",
        "glob",
        "-i",
        osp.join(out_dir, "*.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        str(int(crf)),
        osp.join(out_dir, "pytorch_bev_vis.webm"),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[WARN] ffmpeg not found; skipped WebM generation.")
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] ffmpeg failed with exit code {exc.returncode}; "
              "skipped WebM generation.")


def build_pytorch_model(cfg: Config, checkpoint: str, device: str, dataset):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    ckpt = load_checkpoint(model, checkpoint, map_location="cpu")
    if "CLASSES" in ckpt.get("meta", {}):
        model.CLASSES = ckpt["meta"]["CLASSES"]
    elif hasattr(dataset, "CLASSES"):
        model.CLASSES = dataset.CLASSES

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if torch_device.index is not None:
            torch.cuda.set_device(torch_device.index)
        model = model.cuda(torch_device.index)
    else:
        model = model.to(torch_device)
    model.eval()
    return model


def scatter_batch(data, device: str):
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return data
    target = torch_device.index
    if target is None:
        target = torch.cuda.current_device()
    return scatter(data, [target])[0]


def model_forward_one(model, dataset, index: int, device: str):
    data = dataset[index]
    data = collate([data], samples_per_gpu=1)
    data = scatter_batch(data, device)
    with torch.no_grad():
        output = model(return_loss=False, rescale=True, **data)
    if isinstance(output, (list, tuple)):
        return output[0] if output else {}
    return output


def reset_model_sequence_state(model) -> None:
    for name in (
            "test_track_instances",
            "scene_token",
            "timestamp",
            "l2g_t",
            "l2g_r_mat",
            "_test_track_instances",
            "_test_prev_bev",
            "_test_scene_token",
            "prev_bev",
    ):
        if hasattr(model, name):
            setattr(model, name, None)
    if hasattr(model, "track_base") and hasattr(model.track_base, "clear"):
        model.track_base.clear()


def run_visualization(cfg: Config, args: argparse.Namespace) -> None:
    dataset_cfg = get_dataset_cfg(cfg, args.split)
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    class_names = get_class_names(cfg, dataset_cfg)
    label_mapping = get_label_mapping(cfg, dataset_cfg)

    start_index = resolve_start_index(dataset, args)
    indices = consecutive_scene_indices(dataset, start_index, args.max_frames)
    if not indices:
        raise RuntimeError("No frames selected for visualization.")

    checkpoint = resolve_repo_path(args.checkpoint)
    model = build_pytorch_model(cfg, checkpoint, args.device, dataset)
    reset_model_sequence_state(model)

    mmcv.mkdir_or_exist(args.out_dir)
    summary = []
    frame_files = []
    for frame_id, idx in enumerate(indices):
        info = raw_info_from_dataset(dataset, idx)
        token = str(info.get("token", idx))
        save_path = osp.join(
            args.out_dir, f"{frame_id:03d}_{idx:06d}_{safe_token(token)}.png")
        if args.skip_existing and osp.exists(save_path):
            print(f"[SKIP] {save_path}")
            frame_files.append(save_path)
            continue

        points = load_points(cfg, dataset_cfg, info)
        result = model_forward_one(model, dataset, idx, args.device)
        pred_data = pred_arrays_from_result(
            result, args.score_thr, args.topk)
        gt_data = gt_arrays_from_info(
            info, dataset_cfg, class_names, label_mapping)
        scene = str(info.get("scene_token", ""))
        title = (
            f"BEVFormer-LiDAR PyTorch | frame={frame_id} index={idx} "
            f"token={token[:8]} scene={scene[-8:]}\n"
            f"GT={len(gt_data['boxes'])} Pred={len(pred_data['boxes'])} "
            f"score_thr={args.score_thr}")
        render_frame(
            points=points,
            gt_data=gt_data,
            pred_data=pred_data,
            class_names=class_names,
            save_path=save_path,
            title=title,
            pc_range=cfg.point_cloud_range,
            point_stride=args.point_stride,
            vel_scale=args.vel_scale,
            min_vel_draw=args.min_vel_draw,
            annotate=args.annotate)
        frame_files.append(save_path)
        summary.append(dict(
            frame=int(frame_id),
            index=int(idx),
            token=token,
            scene_token=info.get("scene_token"),
            out_file=save_path,
            num_gt=int(len(gt_data["boxes"])),
            num_pred=int(len(pred_data["boxes"]))))
        print(f"[OK] {save_path}")

    with open(osp.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_html_player(args.out_dir, frame_files)
    write_webm_video(args.out_dir, fps=args.webm_fps, crf=args.webm_crf)


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    cfg = Config.fromfile(config_path)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_cfg_modules(cfg, config_path)
    run_visualization(cfg, args)


if __name__ == "__main__":
    main()
