#!/usr/bin/env python
"""Plot LiDAR E2E training and validation metrics from MMDet log.json files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import os.path as osp
from collections import OrderedDict

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot LiDAR E2E training metrics from log.json files.')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--iters-per-epoch', type=float, default=6283.0)
    parser.add_argument('--prefix', default=None)
    return parser.parse_args()


def read_logs(work_dir):
    train_rows = []
    val_by_epoch = OrderedDict()
    log_paths = sorted(glob.glob(osp.join(work_dir, '*.log.json')))
    for path in log_paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('mode') == 'train':
                    row['_log'] = osp.basename(path)
                    train_rows.append(row)
                elif row.get('mode') == 'val':
                    val_by_epoch[int(row['epoch'])] = row
    return log_paths, train_rows, list(val_by_epoch.values())


def add_epoch_progress(train_rows, iters_per_epoch):
    for row in train_rows:
        row['_x'] = float(row['epoch']) + float(row.get('iter', 0)) / iters_per_epoch


def plot_train_curves(train_rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), dpi=150)
    fig.suptitle('base_e2e_lidar training curves (segmented by log file)',
                 fontsize=13)
    series = [
        (axes[0, 0], [('loss', 'loss'), ('motion.loss_traj', 'motion loss_traj'),
                     ('map.loss_mask_stuff', 'map loss')], 'Loss'),
        (axes[0, 1], [('motion.min_ade', 'train minADE'),
                     ('motion.min_fde', 'train minFDE'), ('motion.mr', 'train MR')],
         'Train Motion Metrics'),
        (axes[1, 0], [('track.frame_4_loss_past_trajs_5', 'track f4 past traj d5'),
                     ('track.frame_0_loss_past_trajs_5', 'track f0 past traj d5')],
         'Track Past Trajectory Loss'),
        (axes[1, 1], [('grad_norm', 'grad_norm')], 'Gradient Norm'),
    ]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    log_names = sorted({row['_log'] for row in train_rows})
    for ax, items, title in series:
        for item_idx, (key, label) in enumerate(items):
            color = colors[item_idx % len(colors)]
            label_used = False
            for log_name in log_names:
                rows = [row for row in train_rows if row['_log'] == log_name]
                rows = sorted(rows, key=lambda row: row['_x'])
                points = [(row['_x'], row.get(key)) for row in rows
                          if isinstance(row.get(key), (int, float))]
                if not points:
                    continue
                ax.plot([p[0] for p in points], [p[1] for p in points],
                        linewidth=1.2, color=color,
                        label=label if not label_used else None)
                label_used = True
        ax.set_title(title)
        ax.set_xlabel('epoch + iter/epoch')
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def plot_val_metrics(val_rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), dpi=150)
    fig.suptitle('base_e2e_lidar validation metrics by checkpoint epoch',
                 fontsize=13)
    series = [
        (axes[0, 0], [('mAP', 'mAP'), ('NDS', 'NDS')], 'Detection'),
        (axes[0, 1], [('AMOTA', 'AMOTA'), ('AMOTP', 'AMOTP')], 'Tracking'),
        (axes[1, 0], [('motion_min_ade', 'motion minADE'),
                     ('motion_min_fde', 'motion minFDE')], 'Motion ADE/FDE'),
        (axes[1, 1], [('motion_mr', 'motion MR'), ('motion_recall',
                                                   'motion recall')],
         'Motion MR/Recall'),
    ]
    epochs = [int(r['epoch']) for r in val_rows]
    for ax, items, title in series:
        for key, label in items:
            ys = [r.get(key) for r in val_rows]
            points = [(e, y) for e, y in zip(epochs, ys)
                      if isinstance(y, (int, float))]
            if not points:
                continue
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    marker='o', linewidth=1.6, label=label)
            for e, y in points:
                ax.text(e, y, f'{y:.3f}', fontsize=7, ha='center',
                        va='bottom')
        ax.set_title(title)
        ax.set_xlabel('checkpoint epoch')
        if epochs:
            ax.set_xticks(epochs)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def write_val_csv(val_rows, out_path):
    fields = [
        'epoch', 'mAP', 'NDS', 'AMOTA', 'AMOTP', 'motion_min_ade',
        'motion_min_fde', 'motion_mr', 'motion_recall', 'map_drivable_iou',
        'map_lanes_iou'
    ]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in val_rows:
            writer.writerow({key: row.get(key, '') for key in fields})
    return fields


def write_summary(log_paths, train_rows, val_rows, fields, out_path):
    latest = train_rows[-1] if train_rows else {}
    with open(out_path, 'w') as f:
        json.dump({
            'logs': [osp.basename(path) for path in log_paths],
            'num_train_points': len(train_rows),
            'val_epochs': [int(row['epoch']) for row in val_rows],
            'latest_train': {
                key: latest.get(key)
                for key in [
                    'epoch', 'iter', 'lr', 'loss', 'motion.min_ade',
                    'motion.min_fde', 'motion.mr',
                    'track.frame_4_loss_past_trajs_5', 'grad_norm'
                ]
            },
            'val_metrics': [{
                key: row.get(key)
                for key in fields
            } for row in val_rows],
        }, f, indent=2)


def main():
    args = parse_args()
    out_dir = args.out_dir or osp.join(args.work_dir, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)
    prefix = args.prefix or osp.basename(osp.normpath(args.work_dir))

    log_paths, train_rows, val_rows = read_logs(args.work_dir)
    add_epoch_progress(train_rows, args.iters_per_epoch)

    train_png = osp.join(out_dir, f'{prefix}_train_curves_epoch1_latest.png')
    val_png = osp.join(out_dir, f'{prefix}_val_metrics_epoch1_latest.png')
    csv_path = osp.join(out_dir, f'{prefix}_val_metrics_epoch1_latest.csv')
    summary_path = osp.join(out_dir, f'{prefix}_training_summary.json')

    plot_train_curves(train_rows, train_png)
    plot_val_metrics(val_rows, val_png)
    fields = write_val_csv(val_rows, csv_path)
    write_summary(log_paths, train_rows, val_rows, fields, summary_path)

    print(train_png)
    print(val_png)
    print(csv_path)
    print(summary_path)


if __name__ == '__main__':
    main()
