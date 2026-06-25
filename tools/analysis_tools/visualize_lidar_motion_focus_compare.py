#!/usr/bin/env python
"""Focused per-object motion comparison from UniAD LiDAR eval outputs."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
ANALYSIS_ROOT = osp.dirname(__file__)
if ANALYSIS_ROOT not in sys.path:
    sys.path.insert(0, ANALYSIS_ROOT)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmcv import Config, DictAction

from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from compute_turn_bucket_metrics import (
    classify_bucket,
    motion_valid_mask,
    traj_error_detail,
    valid_points,
)
from visualize_lidar_e2e_motion import (
    box_corners_bev,
    draw_hdmap_lanes,
    draw_points,
    get_class_names,
    get_dataset_cfg,
    hdmap_lanes_for_frame,
    import_cfg_modules,
    lidar_xy_to_display,
    load_hdmap_lanes,
    load_points,
    raw_info_from_dataset,
    resolve_hdmap_path,
    resolve_repo_path,
    safe_token,
    setup_axis,
    tensor_to_numpy,
    write_html,
)


MODEL_COLORS = {
    'base': '#4c78a8',
    'turnaware': '#f58518',
    'turnloss': '#54a24b',
}


class BucketArgs:
    static_path_thr = 2.0
    straight_deg = 15.0
    mild_deg = 45.0
    straight_lateral_ratio = 0.15
    mild_lateral_ratio = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Draw focused GT/top1/oracle motion comparisons.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', action='append', nargs=2, metavar=('NAME', 'PKL'),
                        required=True,
                        help='Model name and eval output pkl. Repeatable.')
    parser.add_argument('--case', action='append', required=True,
                        help='sample_idx:class_name:track_id[:note]. Repeatable.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--dist-thr', type=float, default=2.0)
    parser.add_argument('--miss-thr', type=float, default=2.0)
    parser.add_argument('--point-stride', type=int, default=5)
    parser.add_argument('--zoom-margin', type=float, default=0.0,
                        help='If positive, crop around the focused object and trajectories.')
    parser.add_argument('--gt-map-overlay', default='none',
                        choices=['none', 'hdmap'])
    parser.add_argument('--hdmap-path', default=None)
    parser.add_argument('--hdmap-max-lanes', type=int, default=96)
    parser.add_argument('--hdmap-margin', type=float, default=8.0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def unwrap_result(result: dict) -> dict:
    if isinstance(result, dict) and 'pts_bbox' in result:
        return result['pts_bbox']
    return result


def boxes_to_numpy(boxes_3d) -> np.ndarray:
    if boxes_3d is None:
        return np.zeros((0, 9), dtype=np.float32)
    if hasattr(boxes_3d, 'tensor'):
        boxes = boxes_3d.tensor.detach().cpu().numpy()
    else:
        boxes = tensor_to_numpy(boxes_3d)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[1] < 9:
        boxes = np.pad(boxes, ((0, 0), (0, 9 - boxes.shape[1])))
    return boxes[:, :9]


def parse_case(text: str) -> Tuple[int, str, int, str]:
    parts = text.split(':', 3)
    if len(parts) < 3:
        raise ValueError(
            f'Invalid case {text!r}; expected sample_idx:class_name:track_id[:note]')
    note = parts[3] if len(parts) > 3 else ''
    return int(parts[0]), parts[1], int(parts[2]), note


def find_gt_index(ann: dict,
                  class_names: Sequence[str],
                  class_name: str,
                  track_id: int) -> int:
    labels = np.asarray(ann.get('gt_labels_3d', []), dtype=np.int64)
    track_ids = np.asarray(ann.get('gt_inds', np.full((len(labels),), -1)),
                           dtype=np.int64)
    for idx, (label, gt_track_id) in enumerate(zip(labels, track_ids)):
        label_name = class_names[int(label)] if 0 <= int(label) < len(class_names) else str(label)
        if label_name == class_name and int(gt_track_id) == track_id:
            return int(idx)
    raise KeyError(f'Cannot find GT {class_name} track_id={track_id}')


def match_prediction(result: dict,
                     dataset,
                     gt_box: np.ndarray,
                     gt_label: int,
                     gt_traj: np.ndarray,
                     gt_mask: np.ndarray,
                     score_thr: float,
                     dist_thr: float,
                     miss_thr: float) -> Optional[dict]:
    result = unwrap_result(result)
    pred_boxes = boxes_to_numpy(
        result.get('track_boxes_3d', result.get('boxes_3d', None)))
    pred_labels = tensor_to_numpy(
        result.get('track_labels_3d', result.get('labels_3d', None)))
    pred_scores = tensor_to_numpy(
        result.get('track_scores', result.get('scores_3d', None)))
    pred_trajs = tensor_to_numpy(result.get('traj', None))
    pred_traj_scores = tensor_to_numpy(result.get('traj_scores', None))
    if pred_labels is None or pred_trajs is None or len(pred_boxes) == 0:
        return None
    pred_labels = np.asarray(pred_labels).reshape(-1).astype(np.int64)
    if pred_scores is None:
        pred_scores = np.ones((len(pred_boxes),), dtype=np.float32)
    pred_scores = np.asarray(pred_scores).reshape(-1).astype(np.float32)
    num = min(len(pred_boxes), len(pred_labels), len(pred_scores), len(pred_trajs))

    best = None
    best_dist = float('inf')
    for pred_idx in range(num):
        if int(pred_labels[pred_idx]) != int(gt_label):
            continue
        if float(pred_scores[pred_idx]) < score_thr:
            continue
        dist = float(dataset._center_distance(gt_box, pred_boxes[pred_idx]))
        if dist < best_dist:
            best = pred_idx
            best_dist = dist
    if best is None or best_dist >= dist_thr:
        return None

    min_detail = traj_error_detail(pred_trajs[best], gt_traj, gt_mask,
                                   miss_threshold=miss_thr)
    if min_detail is None:
        return None
    top1 = 0
    if pred_traj_scores is not None and best < len(pred_traj_scores):
        top1 = int(np.argmax(pred_traj_scores[best]))
    top1_detail = traj_error_detail(pred_trajs[best], gt_traj, gt_mask,
                                    miss_threshold=miss_thr,
                                    mode_index=top1)
    return dict(
        pred_idx=int(best),
        box=pred_boxes[best],
        score=float(pred_scores[best]),
        dist=best_dist,
        traj=np.asarray(pred_trajs[best], dtype=np.float32),
        top1=int(top1),
        oracle=int(min_detail['oracle_fde_mode']),
        minFDE=float(min_detail['minFDE']),
        top1FDE=float('nan') if top1_detail is None else float(top1_detail['minFDE']),
    )


def draw_single_box(ax, box: np.ndarray, color: str, label: str,
                    linestyle: str = '-') -> None:
    corners = box_corners_bev(np.asarray([box], dtype=np.float32))[0]
    poly = lidar_xy_to_display(corners)
    closed = np.concatenate([poly, poly[:1]], axis=0)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.7,
            linestyle=linestyle, alpha=0.95)
    center = lidar_xy_to_display(np.asarray([box[:2]], dtype=np.float32))[0]
    ax.text(center[0], center[1], label, color=color, fontsize=8,
            ha='left', va='bottom')


def absolute_traj(box_xy: np.ndarray, rel_traj: np.ndarray,
                  steps: Optional[int] = None) -> np.ndarray:
    rel = np.asarray(rel_traj, dtype=np.float32)[..., :2]
    if steps is not None:
        rel = rel[:steps]
    return np.concatenate([box_xy[None, :2], box_xy[None, :2] + rel], axis=0)


def plot_path(ax, points_xy: np.ndarray, color: str, label: str,
              linestyle: str, marker: str, linewidth: float) -> None:
    disp = lidar_xy_to_display(points_xy)
    ax.plot(disp[:, 0], disp[:, 1], color=color, linewidth=linewidth,
            linestyle=linestyle, alpha=0.95, label=label)
    ax.scatter(disp[-1:, 0], disp[-1:, 1], color=color, s=24,
               marker=marker, alpha=0.95)


def apply_zoom(ax, xy_chunks: Sequence[np.ndarray],
               pc_range: Sequence[float],
               margin: float) -> None:
    if margin <= 0:
        return
    valid_chunks = [
        np.asarray(chunk, dtype=np.float32).reshape(-1, 2)
        for chunk in xy_chunks if chunk is not None and np.asarray(chunk).size
    ]
    if not valid_chunks:
        return
    xy = np.concatenate(valid_chunks, axis=0)
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    crop_x_min = max(x_min, float(np.min(xy[:, 0])) - margin)
    crop_x_max = min(x_max, float(np.max(xy[:, 0])) + margin)
    crop_y_min = max(y_min, float(np.min(xy[:, 1])) - margin)
    crop_y_max = min(y_max, float(np.max(xy[:, 1])) + margin)
    ax.set_xlim(-crop_y_max, -crop_y_min)
    ax.set_ylim(crop_x_min, crop_x_max)


def render_case(points: np.ndarray,
                hdmap_lanes,
                class_name: str,
                track_id: int,
                bucket: str,
                bucket_meta: Dict[str, float],
                gt_box: np.ndarray,
                gt_traj: np.ndarray,
                gt_mask: np.ndarray,
                matches: Dict[str, Optional[dict]],
                pc_range: Sequence[float],
                title: str,
                out_path: str,
                point_stride: int,
                note: str,
                zoom_margin: float) -> dict:
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 9.2), dpi=170)
    fig.patch.set_facecolor('black')
    setup_axis(ax, pc_range, title)
    draw_points(ax, points, point_stride)
    draw_hdmap_lanes(ax, hdmap_lanes)

    draw_single_box(ax, gt_box, '#ffffff', f'GT#{track_id}:{class_name}', '-')
    zoom_chunks = [gt_box[None, :2]]
    valid = motion_valid_mask(gt_mask)
    steps = min(len(gt_traj), len(valid))
    valid = valid[:steps]
    gt_abs = absolute_traj(gt_box[:2], gt_traj[:steps][valid])
    zoom_chunks.append(gt_abs)
    plot_path(ax, gt_abs, '#ffffff', 'GT future', '-', 'o', 2.2)

    summary = []
    for name, match in matches.items():
        color = MODEL_COLORS.get(name, None)
        if color is None:
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(summary) % 10]
        if match is None:
            summary.append(dict(model=name, matched=False))
            continue
        draw_single_box(ax, match['box'], color, name, ':')
        zoom_chunks.append(match['box'][None, :2])
        top1_path = absolute_traj(match['box'][:2], match['traj'][match['top1']])
        zoom_chunks.append(top1_path)
        plot_path(ax, top1_path, color, f'{name} top1', '-', 'x', 1.8)
        if match['oracle'] != match['top1']:
            oracle_path = absolute_traj(
                match['box'][:2], match['traj'][match['oracle']])
            zoom_chunks.append(oracle_path)
            plot_path(ax, oracle_path, color, f'{name} oracle', '--', 's', 1.2)
        summary.append(dict(
            model=name,
            matched=True,
            pred_idx=match['pred_idx'],
            score=match['score'],
            center_dist=match['dist'],
            top1_mode=match['top1'],
            oracle_mode=match['oracle'],
            top1FDE=match['top1FDE'],
            minFDE=match['minFDE'],
        ))

    lines = [
        f'bucket={bucket}  heading={bucket_meta["heading_change"]:.1f}deg  path={bucket_meta["path"]:.1f}m',
    ]
    if note:
        lines.append(note)
    lines.append('model       top1FDE  minFDE  top1/oracle  score')
    for item in summary:
        if not item.get('matched'):
            lines.append(f'{item["model"]:<10} missed')
            continue
        lines.append(
            f'{item["model"]:<10} {item["top1FDE"]:>7.2f} {item["minFDE"]:>7.2f} '
            f'{item["top1_mode"]:>2}/{item["oracle_mode"]:<2}      {item["score"]:.2f}')
    ax.text(
        0.012, 0.018, '\n'.join(lines), transform=ax.transAxes,
        color='white', fontsize=8, family='monospace', ha='left',
        va='bottom',
        bbox=dict(facecolor='#050608', edgecolor='#555555', alpha=0.86,
                  boxstyle='round,pad=0.38'))
    ax.legend(loc='upper right', fontsize=7, facecolor='#050608',
              edgecolor='#555555', labelcolor='white')
    apply_zoom(ax, zoom_chunks, pc_range, zoom_margin)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return dict(bucket=bucket, bucket_meta=bucket_meta, matches=summary)


def main() -> None:
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)

    cfg_path = resolve_repo_path(args.config)
    cfg = Config.fromfile(cfg_path)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_cfg_modules(cfg, cfg_path)
    dataset_cfg = get_dataset_cfg(cfg, args.split)
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    class_names = get_class_names(cfg, dataset_cfg)

    result_paths = [(name, resolve_repo_path(path)) for name, path in args.result]
    results = {name: mmcv.load(path) for name, path in result_paths}
    for name, rows in results.items():
        if len(rows) != len(dataset):
            raise ValueError(
                f'Result length mismatch for {name}: {len(rows)} != {len(dataset)}')

    hdmap_lanes = None
    hdmap_path = None
    if args.gt_map_overlay == 'hdmap':
        hdmap_path = resolve_hdmap_path(cfg, args)
        hdmap_lanes = load_hdmap_lanes(hdmap_path)
        print(f'[INFO] Loaded {len(hdmap_lanes)} HDMap lanes from {hdmap_path}')

    frame_files: List[str] = []
    summary_rows = []
    for frame_id, case_text in enumerate(args.case):
        sample_idx, class_name, track_id, note = parse_case(case_text)
        info = raw_info_from_dataset(dataset, sample_idx)
        ann = dataset.get_ann_info(sample_idx)
        gt_idx = find_gt_index(ann, class_names, class_name, track_id)

        gt_boxes = boxes_to_numpy(ann['gt_bboxes_3d'])
        gt_labels = np.asarray(ann['gt_labels_3d'], dtype=np.int64)
        gt_trajs = np.asarray(ann.get('gt_fut_traj', np.zeros((0, 0, 2))),
                              dtype=np.float32)
        gt_masks = np.asarray(ann.get('gt_fut_traj_mask', np.zeros((0, 0, 2))),
                              dtype=np.float32)
        bucket_points = valid_points(gt_trajs[gt_idx], gt_masks[gt_idx])
        bucket, bucket_meta = classify_bucket(bucket_points, BucketArgs())

        points = load_points(cfg, dataset_cfg, info)
        frame_hdmap_lanes = None
        if hdmap_lanes is not None:
            frame_hdmap_lanes = hdmap_lanes_for_frame(
                hdmap_lanes, info.get('ego2global', None),
                cfg.point_cloud_range, args.hdmap_max_lanes,
                args.hdmap_margin)

        matches = {}
        for name, rows in results.items():
            matches[name] = match_prediction(
                rows[sample_idx], dataset, gt_boxes[gt_idx],
                int(gt_labels[gt_idx]), gt_trajs[gt_idx], gt_masks[gt_idx],
                args.score_thr, args.dist_thr, args.miss_thr)

        token = str(info.get('token', sample_idx))
        scene = str(info.get('scene_token', ''))
        out_path = osp.join(
            args.out_dir, f'{frame_id:03d}_{sample_idx:06d}_{safe_token(token)}.png')
        title = (
            f'epoch6 focus | index={sample_idx} token={token[:8]} '
            f'scene={scene[-8:]} | GT#{track_id}:{class_name}')
        case_summary = render_case(
            points, frame_hdmap_lanes, class_name, track_id, bucket,
            bucket_meta, gt_boxes[gt_idx], gt_trajs[gt_idx], gt_masks[gt_idx],
            matches, cfg.point_cloud_range, title, out_path,
            args.point_stride, note, args.zoom_margin)
        frame_files.append(out_path)
        summary_rows.append(dict(
            frame=frame_id,
            sample_idx=sample_idx,
            token=token,
            scene_token=info.get('scene_token', ''),
            gt_idx=gt_idx,
            class_name=class_name,
            track_id=track_id,
            note=note,
            out_file=out_path,
            hdmap_path=hdmap_path,
            num_hdmap_lanes=0 if frame_hdmap_lanes is None else len(frame_hdmap_lanes),
            **case_summary,
        ))
        print(f'[OK] {out_path}')

    with open(osp.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary_rows, f, indent=2)
    write_html(args.out_dir, frame_files)
    print(osp.join(args.out_dir, 'index.html'))


if __name__ == '__main__':
    main()
