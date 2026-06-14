#!/usr/bin/env python
"""Prepare BEVFormer-LiDAR deployment input data without model inference.

This script only reads the configured dataset and writes raw points, metadata,
and optional GT boxes in the format consumed by ``inference_app/sparse_lidar``.
It does not load a checkpoint and does not run PyTorch forward.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import os.path as osp
import sys
from typing import List, Optional

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from third_party.uniad_mmdet3d.datasets.builder import build_dataset  # noqa: E402


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
        description="Prepare raw points and metadata for LiDAR deployment.")
    parser.add_argument(
        "--config",
        default="projects/configs/bevformer_lidar/base_bevformer_lidar.py",
        help="UniAD LiDAR config path.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split from cfg.data.")
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
    parser.add_argument(
        "--out-dir",
        default="dumped_inputs/bevformer_lidar_deploy_data",
        help="Output directory for deployment input files.")
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Do not write gt_detections_XXXXXX.txt files.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. data.test.data_root=/path.")
    args = parser.parse_args()
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
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

    if getattr(cfg, "plugin", False):
        if hasattr(cfg, "plugin_dir"):
            module_dir = cfg.plugin_dir
        else:
            module_dir = osp.dirname(config_path)
        module_path = module_dir.rstrip("/").replace("/", ".")
        if module_path:
            importlib.import_module(module_path)


def get_dataset_cfg(cfg: Config, split: str):
    dataset_cfg = cfg.data[split].copy()
    dataset_cfg.test_mode = split != "train"
    dataset_cfg.pop("samples_per_gpu", None)
    return dataset_cfg


def get_class_names(cfg: Config, dataset_cfg) -> List[str]:
    if dataset_cfg.get("classes", None) is not None:
        return list(dataset_cfg.classes)
    if cfg.get("class_names", None) is not None:
        return list(cfg.class_names)
    return list(KL_CLASSES)


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


def unwrap_data(value):
    return value.data if isinstance(value, DataContainer) else value


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def points_to_numpy(points) -> np.ndarray:
    points = unwrap_data(points)
    if isinstance(points, (list, tuple)):
        if len(points) == 0:
            raise ValueError("Expected at least one point tensor, got 0.")
        # Track datasets return a temporal queue. Deployment consumes the
        # current frame, which is stored as the last queue item.
        points = points[-1]
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected NxC point array, got {points.shape}.")
    return np.ascontiguousarray(points)


def current_img_metas(img_metas):
    img_metas = unwrap_data(img_metas)
    if isinstance(img_metas, dict):
        return [img_metas]
    if isinstance(img_metas, (list, tuple)):
        if len(img_metas) == 0:
            return []
        return [img_metas[-1]]
    return img_metas


def numeric_key(key):
    try:
        return (0, int(key))
    except (TypeError, ValueError):
        return (1, str(key))


def current_meta_entry(img_metas):
    if not img_metas:
        return None
    meta = img_metas[0]
    if not isinstance(meta, dict):
        return None
    if isinstance(meta.get("queue_metas"), dict):
        meta = meta["queue_metas"]
    if "ego_motion_delta" in meta:
        return meta

    dict_items = [
        (key, value) for key, value in meta.items() if isinstance(value, dict)
    ]
    if not dict_items:
        return None
    key, value = sorted(dict_items, key=lambda item: numeric_key(item[0]))[-1]
    return value


def attach_optional_planning_metadata(img_metas, sample):
    current_meta = current_meta_entry(img_metas)
    if current_meta is None:
        return
    for key in ("command", "sdc_planning", "sdc_planning_mask"):
        if key in sample:
            current_meta[key] = to_jsonable(unwrap_data(sample[key]))


def boxes_to_numpy(boxes_3d) -> np.ndarray:
    if boxes_3d is None:
        return np.zeros((0, 9), dtype=np.float32)
    if hasattr(boxes_3d, "tensor"):
        boxes = boxes_3d.tensor.detach().cpu().numpy()
    elif torch.is_tensor(boxes_3d):
        boxes = boxes_3d.detach().cpu().numpy()
    else:
        boxes = np.asarray(boxes_3d)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[1] < 9:
        pad = np.zeros((boxes.shape[0], 9 - boxes.shape[1]),
                       dtype=boxes.dtype)
        boxes = np.concatenate([boxes, pad], axis=1)
    return np.ascontiguousarray(boxes[:, :9])


def write_gt_detections(path: str, boxes: np.ndarray, labels: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("# x y z length width height yaw vx vy label score query_index\n")
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            vx = float(box[7]) if np.isfinite(box[7]) else 0.0
            vy = float(box[8]) if np.isfinite(box[8]) else 0.0
            f.write(
                f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f} "
                f"{box[3]:.6f} {box[4]:.6f} {box[5]:.6f} "
                f"{box[6]:.6f} {vx:.6f} {vy:.6f} "
                f"{int(label)} 1.000000000 {idx}\n")


def frame_name(prefix: str, frame_id: int, suffix: str) -> str:
    return f"{prefix}_{frame_id:06d}.{suffix}"


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    cfg = Config.fromfile(config_path)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_cfg_modules(cfg, config_path)

    dataset_cfg = get_dataset_cfg(cfg, args.split)
    dataset = build_dataset(dataset_cfg)
    start_index = resolve_start_index(dataset, args)
    indices = consecutive_scene_indices(dataset, start_index, args.max_frames)
    if not indices:
        raise RuntimeError("No frames selected for deployment data.")

    out_dir = resolve_repo_path(args.out_dir)
    mmcv.mkdir_or_exist(out_dir)

    class_names = get_class_names(cfg, dataset_cfg)
    manifest = {
        "config": osp.abspath(config_path),
        "split": args.split,
        "start_index": int(start_index),
        "num_frames": len(indices),
        "dataset_type": type(dataset).__name__,
        "dataset_length": len(dataset),
        "class_names": class_names,
        "frames": [],
    }

    for frame_id, index in enumerate(indices):
        sample = dataset[index]
        info = raw_info_from_dataset(dataset, index)
        points = points_to_numpy(sample["points"])
        raw_bin = frame_name("raw_points", frame_id, "bin")
        raw_npy = frame_name("raw_points", frame_id, "npy")
        points.tofile(osp.join(out_dir, raw_bin))
        np.save(osp.join(out_dir, raw_npy), points)

        img_metas = current_img_metas(sample["img_metas"])
        attach_optional_planning_metadata(img_metas, sample)
        meta_json = frame_name("img_metas", frame_id, "json")
        with open(osp.join(out_dir, meta_json), "w") as f:
            json.dump(to_jsonable(img_metas), f, indent=2)

        gt_txt: Optional[str] = None
        num_gt = None
        if not args.no_gt and hasattr(dataset, "get_ann_info"):
            ann = dataset.get_ann_info(index)
            boxes = boxes_to_numpy(ann.get("gt_bboxes_3d"))
            labels = np.asarray(ann.get("gt_labels_3d", []), dtype=np.int64)
            gt_txt = frame_name("gt_detections", frame_id, "txt")
            write_gt_detections(osp.join(out_dir, gt_txt), boxes, labels)
            num_gt = int(len(boxes))

        manifest["frames"].append({
            "frame": int(frame_id),
            "index": int(index),
            "token": info.get("token"),
            "scene_token": info.get("scene_token"),
            "sample_idx": to_jsonable(img_metas[0].get("sample_idx"))
            if img_metas else None,
            "raw_points": raw_bin,
            "raw_points_npy": raw_npy,
            "img_metas": meta_json,
            "command": to_jsonable(unwrap_data(sample["command"]))
            if "command" in sample else None,
            "gt_detections": gt_txt,
            "num_points": int(points.shape[0]),
            "point_dim": int(points.shape[1]),
            "num_gt": num_gt,
        })
        print(
            f"[OK] frame={frame_id} index={index} "
            f"points={points.shape[0]} meta={meta_json} gt={num_gt}")

    with open(osp.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote deployment data: {out_dir}")
    print(f"Wrote manifest: {osp.join(out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
