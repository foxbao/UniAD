#!/usr/bin/env python
"""Visualize BEVFormer detection results from a saved result pickle.

This script intentionally avoids the NuScenes table loader. It uses the
BEVFormer info pickle for sample tokens, image paths, and lidar-to-camera
calibration, which makes it suitable for detection-only results.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config
from third_party.uniad_mmdet3d.core.bbox.structures.lidar_box3d import (
    LiDARInstance3DBoxes,
)


CAMERA_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)

BOX_EDGES = (
    (0, 1),
    (3, 0),
    (0, 4),
    (1, 2),
    (1, 5),
    (3, 2),
    (3, 7),
    (4, 5),
    (7, 4),
    (2, 6),
    (5, 6),
    (6, 7),
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize pure BEVFormer 3D detection result pickles."
    )
    parser.add_argument(
        "--config",
        default="projects/configs/bevformer_tiny/bevformer_tiny_imgx0.25.py",
        help="Config used to infer ann_file, data_root, and class names.",
    )
    parser.add_argument(
        "--predroot",
        required=True,
        help="Path to result pkl. Supports plain list or dict['bbox_results'].",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where visualization frames will be written.",
    )
    parser.add_argument(
        "--ann-file",
        default=None,
        help="Override the validation/test info pickle from the config.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override dataset data_root from the config.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument(
        "--result-mode",
        default="auto",
        choices=["auto", "det", "track"],
        help="Which box fields to draw when both det and track outputs exist.",
    )
    parser.add_argument("--bev-range", type=float, default=55.0)
    parser.add_argument("--cam-width", type=int, default=800)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--draw-gt", action="store_true")
    parser.add_argument("--only-bev", action="store_true")
    parser.add_argument("--only-cam", action="store_true")
    parser.add_argument(
        "--video",
        default=None,
        help="Optional .webm output video path. Set empty to skip.",
    )
    parser.add_argument("--fps", type=float, default=4.0)
    args = parser.parse_args()

    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.sample_step <= 0:
        raise ValueError("--sample-step must be positive.")
    if args.topk <= 0:
        raise ValueError("--topk must be positive.")
    if args.only_bev and args.only_cam:
        raise ValueError("--only-bev and --only-cam cannot both be set.")
    return args


def tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def resolve_path(path: str, data_root: Optional[str] = None) -> str:
    candidates = [path]
    if not osp.isabs(path):
        candidates.append(osp.join(REPO_ROOT, path))
    if data_root is not None and not osp.isabs(path):
        candidates.append(osp.join(data_root, path))
        candidates.append(osp.join(REPO_ROOT, data_root, path))
        if path.startswith("./"):
            candidates.append(osp.join(REPO_ROOT, path[2:]))
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return path


def load_cfg_data(args: argparse.Namespace) -> Tuple[str, Optional[str], List[str]]:
    cfg = Config.fromfile(resolve_path(args.config))
    test_cfg = cfg.data.get("test", cfg.data.get("val"))
    ann_file = args.ann_file or test_cfg.ann_file
    data_root = args.data_root or test_cfg.get("data_root", cfg.get("data_root", None))
    class_names = list(test_cfg.get("classes", cfg.get("class_names", [])))
    ann_file = resolve_path(ann_file, data_root=None)
    if data_root is not None and not osp.isabs(data_root):
        data_root = osp.join(REPO_ROOT, data_root)
    return ann_file, data_root, class_names


def load_infos(ann_file: str) -> List[dict]:
    data = mmcv.load(ann_file)
    if isinstance(data, dict) and "infos" in data:
        return list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    if isinstance(data, list):
        return list(sorted(data, key=lambda e: e["timestamp"]))
    raise TypeError(f"Unsupported info file format: {type(data)}")


def load_results(predroot: str) -> List[dict]:
    outputs = mmcv.load(predroot)
    if isinstance(outputs, dict) and "bbox_results" in outputs:
        outputs = outputs["bbox_results"]
    if not isinstance(outputs, list):
        raise TypeError(f"Unsupported prediction format: {type(outputs)}")
    return outputs


def unwrap_det_result(result: dict, result_mode: str = "auto") -> dict:
    if isinstance(result, dict) and "pts_bbox" in result:
        return result["pts_bbox"]
    if isinstance(result, dict) and "boxes_3d" in result:
        if result_mode == "det" and "boxes_3d_det" in result:
            return dict(
                boxes_3d=result["boxes_3d_det"],
                scores_3d=result["scores_3d_det"],
                labels_3d=result["labels_3d_det"],
            )
        if result_mode == "track":
            return result
        if len(result["boxes_3d"]) == 0 and "boxes_3d_det" in result:
            return dict(
                boxes_3d=result["boxes_3d_det"],
                scores_3d=result["scores_3d_det"],
                labels_3d=result["labels_3d_det"],
            )
        return result
    if isinstance(result, dict) and "boxes_3d_det" in result:
        return dict(
            boxes_3d=result["boxes_3d_det"],
            scores_3d=result["scores_3d_det"],
            labels_3d=result["labels_3d_det"],
        )
    raise KeyError(f"Cannot find boxes_3d in result keys: {list(result.keys())}")


def sort_and_filter_result(
    result: dict, score_thr: float, topk: int, result_mode: str = "auto"
) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    det = unwrap_det_result(result, result_mode)
    boxes = det["boxes_3d"]
    scores = tensor_to_numpy(det["scores_3d"]).reshape(-1)
    labels = tensor_to_numpy(det["labels_3d"]).reshape(-1).astype(np.int64)

    order = np.argsort(-scores)
    keep = order[scores[order] >= score_thr][:topk]
    if hasattr(boxes, "__getitem__"):
        boxes = boxes[keep]
    else:
        raise TypeError(f"Unsupported boxes_3d type: {type(boxes)}")
    return boxes, scores[keep], labels[keep], keep


def boxes_to_arrays(boxes) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = tensor_to_numpy(boxes.gravity_center)
    dims = tensor_to_numpy(boxes.dims)
    yaw = tensor_to_numpy(boxes.yaw).reshape(-1)
    corners = tensor_to_numpy(boxes.corners)
    return centers, dims, yaw, corners


def label_color(label: int, bgr: bool = False) -> Tuple[int, int, int]:
    color = PALETTE[int(label) % len(PALETTE)]
    if bgr:
        color = color[::-1]
    return tuple(int(v) for v in color)


def draw_bev_boxes(
    boxes,
    scores: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    class_names: Sequence[str],
    bev_range: float,
    gt_boxes: Optional[np.ndarray] = None,
) -> None:
    centers, dims, yaw, corners = boxes_to_arrays(boxes)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.set_facecolor("#f7f7f5")
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect("equal")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.set_xlabel("y left (m)")
    ax.set_ylabel("x forward (m)")
    ax.scatter([0], [0], marker="s", s=60, color="#222222", zorder=5)

    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_lidar_boxes = make_lidar_boxes(gt_boxes)
        _, _, _, gt_corners = boxes_to_arrays(gt_lidar_boxes)
        for gt_corner in gt_corners:
            xy = gt_corner[[0, 3, 7, 4, 0], :2]
            ax.plot(xy[:, 1], xy[:, 0], color="#4caf50", linewidth=0.8, alpha=0.45)

    for i in range(len(scores)):
        color = np.asarray(label_color(labels[i])) / 255.0
        xy = corners[i, [0, 3, 7, 4, 0], :2]
        ax.plot(xy[:, 1], xy[:, 0], color=color, linewidth=1.4)
        heading = centers[i, :2] + np.array([np.cos(yaw[i]), np.sin(yaw[i])]) * (
            dims[i, 0] * 0.5
        )
        ax.plot(
            [centers[i, 1], heading[1]],
            [centers[i, 0], heading[0]],
            color=color,
            linewidth=1.2,
        )
        if i < 20:
            name = (
                class_names[labels[i]]
                if 0 <= labels[i] < len(class_names)
                else str(labels[i])
            )
            ax.text(
                centers[i, 1],
                centers[i, 0],
                f"{name} {scores[i]:.2f}",
                fontsize=5,
                color=color,
            )

    ax.set_title("BEV detection", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def lidar2img_from_cam(cam_info: Dict) -> np.ndarray:
    lidar2cam_r = np.linalg.inv(np.asarray(cam_info["sensor2lidar_rotation"]))
    lidar2cam_t = np.asarray(cam_info["sensor2lidar_translation"]) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.asarray(cam_info["cam_intrinsic"])
    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam_rt.T


def project_corners(corners: np.ndarray, lidar2img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts_4d = np.concatenate([corners.reshape(-1, 3), np.ones((corners.size // 3, 1))], axis=1)
    pts_2d = pts_4d @ lidar2img.T
    depth = pts_2d[:, 2].copy()
    safe_depth = np.clip(depth, 1e-5, 1e5)
    pts_2d[:, 0] /= safe_depth
    pts_2d[:, 1] /= safe_depth
    return pts_2d[:, :2].reshape(corners.shape[0], 8, 2), depth.reshape(corners.shape[0], 8)


def draw_camera_boxes(
    image: np.ndarray,
    boxes,
    scores: np.ndarray,
    labels: np.ndarray,
    lidar2img: np.ndarray,
    class_names: Sequence[str],
    thickness: int,
    fixed_color_bgr: Optional[Tuple[int, int, int]] = None,
    draw_text: bool = True,
) -> np.ndarray:
    out = image.copy()
    _, _, _, corners = boxes_to_arrays(boxes)
    corners_2d, depths = project_corners(corners, lidar2img)
    h, w = out.shape[:2]

    for box_idx in range(corners_2d.shape[0]):
        if not np.any(depths[box_idx] > 0.1):
            continue
        in_img = (
            (corners_2d[box_idx, :, 0] >= 0)
            & (corners_2d[box_idx, :, 0] < w)
            & (corners_2d[box_idx, :, 1] >= 0)
            & (corners_2d[box_idx, :, 1] < h)
            & (depths[box_idx] > 0.1)
        )
        if not np.any(in_img):
            continue

        color = fixed_color_bgr or label_color(labels[box_idx], bgr=True)
        for start, end in BOX_EDGES:
            if depths[box_idx, start] <= 0.1 or depths[box_idx, end] <= 0.1:
                continue
            p1 = tuple(np.round(corners_2d[box_idx, start]).astype(int))
            p2 = tuple(np.round(corners_2d[box_idx, end]).astype(int))
            clipped = cv2.clipLine((0, 0, w, h), p1, p2)
            if clipped[0]:
                cv2.line(out, clipped[1], clipped[2], color, thickness, cv2.LINE_AA)

        if draw_text and box_idx < 20:
            text_xy = corners_2d[box_idx, in_img].mean(axis=0).astype(int)
            name = (
                class_names[labels[box_idx]]
                if 0 <= labels[box_idx] < len(class_names)
                else str(labels[box_idx])
            )
            cv2.putText(
                out,
                f"{name} {scores[box_idx]:.2f}",
                tuple(text_xy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return out


def make_lidar_boxes(box_array: np.ndarray):
    box_array = np.asarray(box_array, dtype=np.float32)
    if box_array.size == 0:
        return None
    if box_array.ndim != 2 or box_array.shape[1] < 7:
        raise ValueError(f"Expected gt boxes with shape Nx7+, got {box_array.shape}")
    return LiDARInstance3DBoxes(
        torch.from_numpy(box_array), box_dim=box_array.shape[1], origin=(0.5, 0.5, 0.5)
    )


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or image.shape[1] == width:
        return image
    scale = width / float(image.shape[1])
    return cv2.resize(image, (width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def render_camera_mosaic(
    info: dict,
    boxes,
    scores: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    class_names: Sequence[str],
    data_root: Optional[str],
    cam_width: int,
    thickness: int,
    gt_boxes: Optional[np.ndarray] = None,
) -> None:
    gt_lidar_boxes = make_lidar_boxes(gt_boxes) if gt_boxes is not None else None
    if gt_lidar_boxes is not None:
        gt_scores = np.zeros((len(gt_lidar_boxes),), dtype=np.float32)
        gt_labels = np.zeros((len(gt_lidar_boxes),), dtype=np.int64)

    rendered = []
    for cam_name in CAMERA_ORDER:
        cam_info = info["cams"][cam_name]
        img_path = resolve_path(cam_info["data_path"], data_root)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read camera image: {img_path}")
        image = draw_camera_boxes(
            image,
            boxes,
            scores,
            labels,
            lidar2img_from_cam(cam_info),
            class_names,
            thickness,
        )
        if gt_lidar_boxes is not None:
            image = draw_camera_boxes(
                image,
                gt_lidar_boxes,
                gt_scores,
                gt_labels,
                lidar2img_from_cam(cam_info),
                class_names,
                thickness,
                fixed_color_bgr=(0, 220, 0),
                draw_text=False,
            )
        cv2.putText(
            image,
            cam_name,
            (28, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            cam_name,
            (28, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        rendered.append(resize_width(image, cam_width))

    row1 = cv2.hconcat(rendered[:3])
    row2 = cv2.hconcat(rendered[3:])
    cv2.imwrite(out_path, cv2.vconcat([row1, row2]))


def resize_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / float(image.shape[0])
    return cv2.resize(image, (int(round(image.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)


def combine_images(cam_path: Optional[str], bev_path: Optional[str], out_path: str) -> None:
    if cam_path is None:
        image = cv2.imread(bev_path)
        cv2.imwrite(out_path, image)
        return
    if bev_path is None:
        image = cv2.imread(cam_path)
        cv2.imwrite(out_path, image)
        return

    cam_img = cv2.imread(cam_path)
    bev_img = cv2.imread(bev_path)
    if cam_img is None or bev_img is None:
        raise FileNotFoundError("Cannot read intermediate camera or BEV image.")
    bev_img = resize_height(bev_img, cam_img.shape[0])
    cv2.imwrite(out_path, cv2.hconcat([cam_img, bev_img]))


def iter_sample_indices(start: int, max_samples: int, step: int, total: int) -> Iterable[int]:
    idx = start
    produced = 0
    while idx < total and produced < max_samples:
        yield idx
        idx += step
        produced += 1


def result_token_map(results: Sequence[dict]) -> Dict[str, dict]:
    mapping = {}
    for result in results:
        if isinstance(result, dict) and isinstance(result.get("token", None), str):
            mapping[result["token"]] = result
    return mapping


def make_video(frame_paths: Sequence[str], video_path: str, fps: float) -> None:
    if not frame_paths:
        return
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise FileNotFoundError(frame_paths[0])
    size = (first.shape[1], first.shape[0])
    if osp.splitext(video_path)[1].lower() != ".webm":
        raise ValueError(f"--video must end with .webm, got: {video_path}")
    make_webm_video(frame_paths, video_path, fps, size)


def make_webm_video(
    frame_paths: Sequence[str], video_path: str, fps: float, size: Tuple[int, int]
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Writing .webm requires ffmpeg, but ffmpeg was not found.")

    # VP8/VP9 yuv420p encoders require even dimensions.
    webm_size = (size[0] - size[0] % 2, size[1] - size[1] % 2)
    if webm_size[0] <= 0 or webm_size[1] <= 0:
        raise ValueError(f"Invalid video frame size: {size}")

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{webm_size[0]}x{webm_size[1]}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "0",
        "-crf",
        "32",
        video_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is None:
                continue
            if (frame.shape[1], frame.shape[0]) != webm_size:
                frame = cv2.resize(frame, webm_size, interpolation=cv2.INTER_AREA)
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass
    finally:
        proc.stdin.close()

    stderr = proc.stderr.read() if proc.stderr is not None else b""
    return_code = proc.wait()
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to write {video_path}: {message}")


def main() -> None:
    args = parse_args()
    ann_file, data_root, class_names = load_cfg_data(args)
    infos = load_infos(ann_file)
    results = load_results(resolve_path(args.predroot))
    results_by_token = result_token_map(results)
    total = len(infos) if results_by_token else min(len(infos), len(results))
    if args.start_index >= total:
        raise IndexError(f"--start-index {args.start_index} >= available samples {total}.")

    mmcv.mkdir_or_exist(args.out_dir)
    frame_paths = []
    for idx in iter_sample_indices(args.start_index, args.max_samples, args.sample_step, total):
        info = infos[idx]
        result = results_by_token.get(info.get("token"), results[idx] if idx < len(results) else None)
        if result is None:
            print(f"[{idx}] {info.get('token', '')}: no prediction result, skipped")
            continue
        boxes, scores, labels, _ = sort_and_filter_result(
            result, args.score_thr, args.topk, args.result_mode
        )
        token = info.get("token", str(idx))
        prefix = osp.join(args.out_dir, f"{idx:06d}_{token[:8]}")
        cam_path = None
        bev_path = None

        if not args.only_cam:
            bev_path = prefix + "_bev.jpg"
            gt_boxes = np.asarray(info.get("gt_boxes", [])) if args.draw_gt else None
            draw_bev_boxes(
                boxes,
                scores,
                labels,
                bev_path,
                class_names,
                args.bev_range,
                gt_boxes=gt_boxes,
            )

        if not args.only_bev:
            cam_path = prefix + "_cams.jpg"
            render_camera_mosaic(
                info,
                boxes,
                scores,
                labels,
                cam_path,
                class_names,
                data_root,
                args.cam_width,
                args.thickness,
                gt_boxes=np.asarray(info.get("gt_boxes", [])) if args.draw_gt else None,
            )

        merged_path = prefix + ".jpg"
        combine_images(cam_path, bev_path, merged_path)
        frame_paths.append(merged_path)
        print(
            f"[{idx}] {token}: kept {len(scores)} boxes >= {args.score_thr:.2f}, "
            f"wrote {merged_path}"
        )

    if args.video:
        video_path = args.video
        if not osp.isabs(video_path):
            video_path = osp.join(args.out_dir, video_path)
        make_video(frame_paths, video_path, args.fps)
        print(f"wrote video {video_path}")


if __name__ == "__main__":
    main()
