#!/usr/bin/env python
"""Compare two LiDAR E2E motion eval outputs in BEV."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Sequence

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
from visualize_lidar_e2e_motion import (
    draw_boxes,
    draw_gt_future,
    draw_hdmap_lanes,
    draw_points,
    draw_pred_traj,
    get_class_names,
    get_dataset_cfg,
    gt_from_ann,
    hdmap_lanes_for_frame,
    import_cfg_modules,
    load_hdmap_lanes,
    load_points,
    pred_from_result,
    raw_info_from_dataset,
    resolve_hdmap_path,
    resolve_repo_path,
    safe_token,
    setup_axis,
    write_html,
    write_webm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize base vs turn-aware LiDAR E2E motion outputs.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--base-results', required=True)
    parser.add_argument('--compare-results', required=True)
    parser.add_argument('--base-work-dir', default=None)
    parser.add_argument('--compare-work-dir', default=None)
    parser.add_argument('--base-turn-csv', default=None)
    parser.add_argument('--compare-turn-csv', default=None)
    parser.add_argument('--base-name', default='base_e2e_lidar')
    parser.add_argument('--compare-name', default='base_e2e_lidar_turnaware')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--epoch', type=int, default=6)
    parser.add_argument('--indices', nargs='*', type=int, default=None)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=8)
    parser.add_argument('--score-thr', type=float, default=0.25)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--point-stride', type=int, default=5)
    parser.add_argument('--annotate-topk', type=int, default=16)
    parser.add_argument('--webm-fps', type=float, default=3.0)
    parser.add_argument('--webm-crf', type=int, default=32)
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


def read_latest_val_metrics(work_dir: Optional[str],
                            epoch: int) -> Dict[str, float]:
    if not work_dir:
        return {}
    rows = []
    for path in sorted(glob.glob(osp.join(resolve_repo_path(work_dir),
                                          '*.log.json'))):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('mode') == 'val' and int(row.get('epoch', -1)) == epoch:
                    rows.append(row)
    return rows[-1] if rows else {}


def plot_overall_metrics(base: Dict[str, float],
                         compare: Dict[str, float],
                         base_name: str,
                         compare_name: str,
                         out_path: str) -> None:
    keys = [
        ('motion_min_ade', 'minADE'),
        ('motion_min_fde', 'minFDE'),
        ('motion_mr', 'MR'),
        ('motion_recall', 'Recall'),
        ('mAP', 'mAP'),
        ('AMOTA', 'AMOTA'),
    ]
    labels = [label for _, label in keys]
    base_values = [base.get(key, np.nan) for key, _ in keys]
    compare_values = [compare.get(key, np.nan) for key, _ in keys]
    x = np.arange(len(keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    ax.bar(x - width / 2, base_values, width, label=base_name,
           color='#4c78a8')
    ax.bar(x + width / 2, compare_values, width, label=compare_name,
           color='#f58518')
    for xpos, values in ((x - width / 2, base_values),
                         (x + width / 2, compare_values)):
        for xx, value in zip(xpos, values):
            if not np.isfinite(value):
                continue
            ax.text(xx, value, f'{value:.3f}', ha='center', va='bottom',
                    fontsize=8)
    ax.set_title('Epoch 6 Overall Validation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_overall_improvement(base: Dict[str, float],
                             compare: Dict[str, float],
                             base_name: str,
                             compare_name: str,
                             out_path: str) -> None:
    keys = [
        ('motion_min_ade', 'minADE', 'lower'),
        ('motion_min_fde', 'minFDE', 'lower'),
        ('motion_mr', 'MR', 'lower'),
        ('motion_recall', 'Recall', 'higher'),
        ('mAP', 'mAP', 'higher'),
        ('AMOTA', 'AMOTA', 'higher'),
    ]
    labels = []
    improvements = []
    for key, label, direction in keys:
        base_value = base.get(key, np.nan)
        compare_value = compare.get(key, np.nan)
        labels.append(label)
        if not np.isfinite(base_value) or not np.isfinite(compare_value) or abs(base_value) < 1e-12:
            improvements.append(np.nan)
            continue
        if direction == 'lower':
            improvements.append((base_value - compare_value) / base_value * 100.0)
        else:
            improvements.append((compare_value - base_value) / base_value * 100.0)

    colors = ['#2ca02c' if value >= 0 else '#d62728'
              for value in improvements]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    ax.axhline(0.0, color='#333333', linewidth=1.0)
    ax.bar(x, improvements, color=colors)
    for xx, value in zip(x, improvements):
        if not np.isfinite(value):
            continue
        va = 'bottom' if value >= 0 else 'top'
        offset = 0.04 if value >= 0 else -0.04
        ax.text(xx, value + offset, f'{value:+.2f}%', ha='center',
                va=va, fontsize=8)
    ax.set_title(f'{compare_name} Relative Improvement vs {base_name}')
    ax.set_ylabel('Improvement (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def read_turn_csv(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    rows = {}
    with open(resolve_repo_path(path), newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('group') != 'Overall':
                continue
            bucket = row['bucket']
            rows[bucket] = {}
            for key, value in row.items():
                if key in ('group', 'bucket'):
                    continue
                try:
                    rows[bucket][key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def plot_turn_bucket_metrics(base_rows: Dict[str, Dict[str, float]],
                             compare_rows: Dict[str, Dict[str, float]],
                             base_name: str,
                             compare_name: str,
                             out_path: str) -> None:
    buckets = ['static_slow', 'straight', 'mild_turn', 'sharp_turn']
    panels = [
        ('minFDE', 'Oracle minFDE'),
        ('top1_FDE', 'Top-1 FDE'),
        ('MR', 'Miss Rate'),
        ('recall', 'Motion Recall'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), dpi=160)
    axes = axes.reshape(-1)
    x = np.arange(len(buckets))
    width = 0.36
    for ax, (key, title) in zip(axes, panels):
        base_values = [base_rows.get(bucket, {}).get(key, np.nan)
                       for bucket in buckets]
        compare_values = [compare_rows.get(bucket, {}).get(key, np.nan)
                          for bucket in buckets]
        ax.bar(x - width / 2, base_values, width, label=base_name,
               color='#4c78a8')
        ax.bar(x + width / 2, compare_values, width, label=compare_name,
               color='#f58518')
        for xpos, values in ((x - width / 2, base_values),
                             (x + width / 2, compare_values)):
            for xx, value in zip(xpos, values):
                if not np.isfinite(value):
                    continue
                ax.text(xx, value, f'{value:.2f}', ha='center',
                        va='bottom', fontsize=7)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=18, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
    axes[0].legend(fontsize=8)
    fig.suptitle('Epoch 6 Turn-Bucket Motion Metrics', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def plot_turn_bucket_improvement(base_rows: Dict[str, Dict[str, float]],
                                 compare_rows: Dict[str, Dict[str, float]],
                                 base_name: str,
                                 compare_name: str,
                                 out_path: str) -> None:
    buckets = ['static_slow', 'straight', 'mild_turn', 'sharp_turn']
    panels = [
        ('minFDE', 'Oracle minFDE', 'lower'),
        ('top1_FDE', 'Top-1 FDE', 'lower'),
        ('MR', 'Miss Rate', 'lower'),
        ('recall', 'Motion Recall', 'higher'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), dpi=160)
    axes = axes.reshape(-1)
    x = np.arange(len(buckets))
    for ax, (key, title, direction) in zip(axes, panels):
        values = []
        for bucket in buckets:
            base_value = base_rows.get(bucket, {}).get(key, np.nan)
            compare_value = compare_rows.get(bucket, {}).get(key, np.nan)
            if not np.isfinite(base_value) or not np.isfinite(compare_value) or abs(base_value) < 1e-12:
                values.append(np.nan)
            elif direction == 'lower':
                values.append((base_value - compare_value) / base_value * 100.0)
            else:
                values.append((compare_value - base_value) / base_value * 100.0)
        colors = ['#2ca02c' if value >= 0 else '#d62728' for value in values]
        ax.axhline(0.0, color='#333333', linewidth=1.0)
        ax.bar(x, values, color=colors)
        for xx, value in zip(x, values):
            if not np.isfinite(value):
                continue
            va = 'bottom' if value >= 0 else 'top'
            offset = 0.2 if value >= 0 else -0.2
            ax.text(xx, value + offset, f'{value:+.1f}%', ha='center',
                    va=va, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel('Improvement (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=18, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.35)
    fig.suptitle(f'{compare_name} Turn-Bucket Improvement vs {base_name}',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def render_compare_frame(points: np.ndarray,
                         gt_data: Dict[str, np.ndarray],
                         base_pred: Dict[str, np.ndarray],
                         compare_pred: Dict[str, np.ndarray],
                         class_names: Sequence[str],
                         pc_range: Sequence[float],
                         title: str,
                         base_name: str,
                         compare_name: str,
                         out_path: str,
                         point_stride: int,
                         annotate_topk: int,
                         hdmap_lanes: Optional[Sequence[dict]] = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 8.4), dpi=150)
    fig.patch.set_facecolor('black')

    gt_title = f'GT boxes + GT future ({len(gt_data["boxes"])})'
    if hdmap_lanes:
        gt_title += f' + HDMap ({len(hdmap_lanes)})'
    panel_titles = [
        gt_title,
        f'{base_name} pred ({len(base_pred["boxes"])})',
        f'{compare_name} pred ({len(compare_pred["boxes"])})',
    ]
    for ax, panel_title in zip(axes, panel_titles):
        setup_axis(ax, pc_range, panel_title)
        draw_points(ax, points, point_stride)

    draw_hdmap_lanes(axes[0], hdmap_lanes)
    draw_boxes(axes[0], gt_data, class_names, 'GT#', annotate_topk,
               alpha=0.75)
    draw_gt_future(axes[0], gt_data)

    draw_boxes(axes[1], base_pred, class_names, 'B#', annotate_topk,
               alpha=0.95)
    draw_pred_traj(axes[1], base_pred)

    draw_boxes(axes[2], compare_pred, class_names, 'T#', annotate_topk,
               alpha=0.95)
    draw_pred_traj(axes[2], compare_pred)

    fig.suptitle(title, color='white', fontsize=11)
    fig.subplots_adjust(
        left=0.04, right=0.99, bottom=0.06, top=0.91, wspace=0.05)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def selected_indices(args: argparse.Namespace, dataset_len: int) -> List[int]:
    if args.indices:
        indices = args.indices
    else:
        end = min(dataset_len, args.start_index + args.max_frames)
        indices = list(range(args.start_index, end))
    valid = []
    for idx in indices[:args.max_frames]:
        if 0 <= idx < dataset_len:
            valid.append(idx)
    if not valid:
        raise ValueError('No valid frame indices selected.')
    return valid


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

    base_results = mmcv.load(resolve_repo_path(args.base_results))
    compare_results = mmcv.load(resolve_repo_path(args.compare_results))
    if len(base_results) != len(dataset) or len(compare_results) != len(dataset):
        raise ValueError(
            f'Result length mismatch: dataset={len(dataset)}, '
            f'base={len(base_results)}, compare={len(compare_results)}')

    base_metrics = read_latest_val_metrics(args.base_work_dir, args.epoch)
    compare_metrics = read_latest_val_metrics(args.compare_work_dir, args.epoch)
    if base_metrics and compare_metrics:
        plot_overall_metrics(
            base_metrics, compare_metrics, args.base_name, args.compare_name,
            osp.join(args.out_dir, 'overall_epoch6_metrics.png'))
        plot_overall_improvement(
            base_metrics, compare_metrics, args.base_name, args.compare_name,
            osp.join(args.out_dir, 'overall_epoch6_improvement.png'))

    base_turn = read_turn_csv(args.base_turn_csv)
    compare_turn = read_turn_csv(args.compare_turn_csv)
    if base_turn and compare_turn:
        plot_turn_bucket_metrics(
            base_turn, compare_turn, args.base_name, args.compare_name,
            osp.join(args.out_dir, 'turn_bucket_epoch6_metrics.png'))
        plot_turn_bucket_improvement(
            base_turn, compare_turn, args.base_name, args.compare_name,
            osp.join(args.out_dir, 'turn_bucket_epoch6_improvement.png'))

    hdmap_lanes = None
    hdmap_path = None
    if args.gt_map_overlay == 'hdmap':
        hdmap_path = resolve_hdmap_path(cfg, args)
        hdmap_lanes = load_hdmap_lanes(hdmap_path)
        print(f'[INFO] Loaded {len(hdmap_lanes)} HDMap lanes from {hdmap_path}')

    frame_dir = osp.join(args.out_dir, 'frames')
    mmcv.mkdir_or_exist(frame_dir)
    indices = selected_indices(args, len(dataset))
    frame_files = []
    summary = []
    for frame_id, idx in enumerate(indices):
        info = raw_info_from_dataset(dataset, idx)
        points = load_points(cfg, dataset_cfg, info)
        ann = dataset.get_ann_info(idx)
        gt_data = gt_from_ann(ann, class_names)
        base_pred = pred_from_result(
            unwrap_result(base_results[idx]), args.score_thr, args.topk)
        compare_pred = pred_from_result(
            unwrap_result(compare_results[idx]), args.score_thr, args.topk)

        frame_hdmap = None
        if hdmap_lanes is not None:
            frame_hdmap = hdmap_lanes_for_frame(
                hdmap_lanes, info.get('ego2global', None),
                cfg.point_cloud_range, args.hdmap_max_lanes,
                args.hdmap_margin)

        token = str(info.get('token', idx))
        scene = str(info.get('scene_token', ''))
        out_path = osp.join(
            frame_dir, f'{frame_id:03d}_{idx:06d}_{safe_token(token)}.png')
        title = (
            f'epoch={args.epoch} frame={frame_id} index={idx} '
            f'token={token[:8]} scene={scene[-8:]}')
        render_compare_frame(
            points, gt_data, base_pred, compare_pred, class_names,
            cfg.point_cloud_range, title, args.base_name, args.compare_name,
            out_path, args.point_stride, args.annotate_topk,
            hdmap_lanes=frame_hdmap)
        frame_files.append(out_path)
        summary.append(dict(
            frame=int(frame_id),
            index=int(idx),
            token=token,
            scene_token=info.get('scene_token'),
            out_file=out_path,
            num_gt=int(len(gt_data['boxes'])),
            num_base_pred=int(len(base_pred['boxes'])),
            num_compare_pred=int(len(compare_pred['boxes'])),
            hdmap_path=hdmap_path,
            num_hdmap_lanes=0 if frame_hdmap is None else int(len(frame_hdmap)),
        ))
        print(f'[OK] {out_path}')

    with open(osp.join(frame_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    write_html(frame_dir, frame_files)
    write_webm(frame_dir, args.webm_fps, args.webm_crf)

    print(osp.join(args.out_dir, 'overall_epoch6_metrics.png'))
    print(osp.join(args.out_dir, 'overall_epoch6_improvement.png'))
    print(osp.join(args.out_dir, 'turn_bucket_epoch6_metrics.png'))
    print(osp.join(args.out_dir, 'turn_bucket_epoch6_improvement.png'))
    print(osp.join(frame_dir, 'index.html'))
    print(osp.join(frame_dir, 'pytorch_bev_vis.webm'))


if __name__ == '__main__':
    main()
