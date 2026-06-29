#!/usr/bin/env python
"""Analyze KL SDC planning GT distribution.

This is meant to catch data bias that is easy to hide in average planning
metrics, especially parked/low-speed collection segments.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import os.path as osp
import pickle
from collections import Counter, defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COMMAND_NAMES = {
    0: 'Right',
    1: 'Left',
    2: 'Straight',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze SDC planning GT speed/turn distribution.')
    parser.add_argument(
        '--infos',
        nargs='+',
        default=['data/kl_8/kl_infos_train.pkl',
                 'data/kl_8/kl_infos_val.pkl'])
    parser.add_argument(
        '--splits',
        nargs='*',
        default=None,
        help='Optional split names matching --infos. Defaults to file names.')
    parser.add_argument(
        '--out-dir',
        default='projects/work_dirs/stage2_e2e_lidar/planning_gt_distribution')
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--static-final-thr', type=float, default=0.5)
    parser.add_argument('--slow-final-thr', type=float, default=2.0)
    parser.add_argument('--turn-deg-thr', type=float, default=15.0)
    parser.add_argument('--lateral-ratio-thr', type=float, default=0.15)
    parser.add_argument('--min-segment-disp', type=float, default=0.05)
    parser.add_argument('--top-scenes', type=int, default=30)
    return parser.parse_args()


def split_name_from_path(path):
    name = osp.splitext(osp.basename(path))[0]
    for prefix in ('kl_infos_', 'infos_'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def load_infos(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        if 'data_list' in data:
            return data['data_list']
        if 'infos' in data:
            return data['infos']
    if isinstance(data, list):
        return data
    raise TypeError(f'Unsupported info file structure in {path}: {type(data)}')


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def valid_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0, :, 0]
    elif mask.ndim == 2:
        mask = mask[:, 0]
    else:
        mask = mask.reshape(-1)
    return mask.astype(bool)


def planning_xyyaw(info):
    traj = np.asarray(info.get('sdc_planning'), dtype=np.float64)
    mask = np.asarray(info.get('sdc_planning_mask'), dtype=np.float64)
    if traj.ndim == 3:
        traj = traj[0]
    elif traj.ndim != 2:
        return None, None
    return traj, valid_mask(mask)


def path_length(points):
    if len(points) == 0:
        return 0.0
    pts = np.concatenate([np.zeros((1, 2), dtype=np.float64),
                          points[:, :2]], axis=0)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def heading_change_deg(points, min_segment_disp):
    if len(points) < 2:
        return 0.0
    pts = np.concatenate([np.zeros((1, 2), dtype=np.float64),
                          points[:, :2]], axis=0)
    deltas = np.diff(pts, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = np.where(norms >= min_segment_disp)[0]
    if len(valid) < 2:
        return 0.0
    v0 = deltas[valid[0]]
    v1 = deltas[valid[-1]]
    a0 = math.atan2(float(v0[1]), float(v0[0]))
    a1 = math.atan2(float(v1[1]), float(v1[0]))
    return abs(math.degrees(wrap_pi(a1 - a0)))


def yaw_change_deg(traj, valid):
    if traj.shape[1] < 3:
        return 0.0
    idx = np.where(valid[:len(traj)])[0]
    if len(idx) < 2:
        return 0.0
    yaw = traj[idx, 2]
    if not np.isfinite(yaw).all():
        return 0.0
    return abs(math.degrees(wrap_pi(float(yaw[-1] - yaw[0]))))


def lateral_ratio(points):
    if len(points) < 2:
        return 0.0
    end = points[-1, :2]
    net = float(np.linalg.norm(end))
    if net < 1e-6:
        return 0.0
    direction = end / net
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    lateral = float(np.max(np.abs(points[:, :2] @ normal)))
    return lateral / max(net, 1e-6)


def classify(final_disp, turn_angle, lat_ratio, args):
    if final_disp < args.static_final_thr:
        return 'static', 'static'
    if final_disp < args.slow_final_thr:
        turn_bucket = 'slow_turn' if turn_angle >= args.turn_deg_thr else 'slow'
        return 'slow', turn_bucket
    if (turn_angle >= args.turn_deg_thr
            or lat_ratio >= args.lateral_ratio_thr):
        return 'moving', 'turning'
    return 'moving', 'straight'


def analyze_info(info, split, args):
    traj, mask = planning_xyyaw(info)
    if traj is None or mask is None:
        return None
    steps = min(len(traj), len(mask))
    mask = mask[:steps]
    if not np.any(mask):
        return None

    valid_idx = np.where(mask)[0]
    last = int(valid_idx[-1])
    valid_points = traj[:last + 1][mask[:last + 1]]
    if len(valid_points) == 0:
        return None

    final_xy = valid_points[-1, :2]
    final_disp = float(np.linalg.norm(final_xy))
    path = path_length(valid_points)
    horizon = float((last + 1) * args.dt)
    avg_speed = path / horizon if horizon > 1e-6 else 0.0
    xy_turn = heading_change_deg(valid_points, args.min_segment_disp)
    yaw_turn = yaw_change_deg(traj[:last + 1], mask[:last + 1])
    turn_angle = max(xy_turn, yaw_turn)
    lat_ratio = lateral_ratio(valid_points)
    motion_bucket, turn_bucket = classify(
        final_disp, turn_angle, lat_ratio, args)

    command_raw = np.asarray(info.get('command', [-1])).reshape(-1)
    command = int(command_raw[0]) if len(command_raw) else -1
    scene = info.get('scene_token', '')
    token = info.get('token', info.get('sample_idx', ''))
    return dict(
        split=split,
        token=token,
        scene_token=scene,
        command=command,
        command_name=COMMAND_NAMES.get(command, f'Unknown_{command}'),
        valid_steps=int(mask.sum()),
        horizon_s=horizon,
        final_disp_m=final_disp,
        path_length_m=path,
        avg_speed_mps=avg_speed,
        xy_heading_change_deg=xy_turn,
        yaw_change_deg=yaw_turn,
        turn_angle_deg=turn_angle,
        lateral_ratio=lat_ratio,
        final_x_m=float(final_xy[0]),
        final_y_m=float(final_xy[1]),
        motion_bucket=motion_bucket,
        turn_bucket=turn_bucket,
    )


def write_csv(path, rows, fieldnames):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def percentile(rows, key, q):
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    if len(vals) == 0:
        return float('nan')
    return float(np.percentile(vals, q))


def split_summary(split, rows):
    n = len(rows)
    motion = Counter(r['motion_bucket'] for r in rows)
    turns = Counter(r['turn_bucket'] for r in rows)
    commands = Counter(r['command_name'] for r in rows)
    return dict(
        split=split,
        total=n,
        static=motion['static'],
        static_pct=pct(motion['static'], n),
        slow=motion['slow'],
        slow_pct=pct(motion['slow'], n),
        moving=motion['moving'],
        moving_pct=pct(motion['moving'], n),
        turning=turns['turning'],
        turning_pct=pct(turns['turning'], n),
        slow_turn=turns['slow_turn'],
        slow_turn_pct=pct(turns['slow_turn'], n),
        straight=turns['straight'],
        straight_pct=pct(turns['straight'], n),
        right=commands['Right'],
        right_pct=pct(commands['Right'], n),
        left=commands['Left'],
        left_pct=pct(commands['Left'], n),
        command_straight=commands['Straight'],
        command_straight_pct=pct(commands['Straight'], n),
        final_disp_mean=(
            float(np.mean([r['final_disp_m'] for r in rows])) if rows
            else float('nan')),
        final_disp_p50=percentile(rows, 'final_disp_m', 50),
        final_disp_p90=percentile(rows, 'final_disp_m', 90),
        avg_speed_mean=(
            float(np.mean([r['avg_speed_mps'] for r in rows])) if rows
            else float('nan')),
        avg_speed_p50=percentile(rows, 'avg_speed_mps', 50),
        turn_angle_p90=percentile(rows, 'turn_angle_deg', 90),
    )


def scene_summary(rows, top_k):
    agg = defaultdict(list)
    for row in rows:
        agg[row['scene_token']].append(row)
    out = []
    for scene, items in agg.items():
        n = len(items)
        motion = Counter(r['motion_bucket'] for r in items)
        turns = Counter(r['turn_bucket'] for r in items)
        out.append(dict(
            scene_token=scene,
            total=n,
            static=motion['static'],
            static_pct=pct(motion['static'], n),
            slow=motion['slow'],
            slow_pct=pct(motion['slow'], n),
            moving=motion['moving'],
            moving_pct=pct(motion['moving'], n),
            turning=turns['turning'],
            turning_pct=pct(turns['turning'], n),
            final_disp_mean=float(np.mean(
                [r['final_disp_m'] for r in items])),
            avg_speed_mean=float(np.mean(
                [r['avg_speed_mps'] for r in items])),
        ))
    out.sort(key=lambda r: (-r['static_pct'], -r['total'], r['scene_token']))
    return out[:top_k], out


def save_bar(path, labels, values, title, ylabel='Count'):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_hist(path, values, title, xlabel, bins=60):
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_plots(out_dir, split, rows):
    if not rows:
        return
    split_dir = osp.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    save_hist(
        osp.join(split_dir, 'final_displacement_hist.png'),
        [r['final_disp_m'] for r in rows],
        f'{split} SDC planning final displacement',
        'Final displacement at 3s (m)')
    save_hist(
        osp.join(split_dir, 'avg_speed_hist.png'),
        [r['avg_speed_mps'] for r in rows],
        f'{split} SDC planning average speed',
        'Average speed over valid horizon (m/s)')
    motion = Counter(r['motion_bucket'] for r in rows)
    save_bar(
        osp.join(split_dir, 'motion_bucket_bar.png'),
        ['static', 'slow', 'moving'],
        [motion['static'], motion['slow'], motion['moving']],
        f'{split} motion buckets')
    command = Counter(r['command_name'] for r in rows)
    labels = ['Left', 'Right', 'Straight']
    save_bar(
        osp.join(split_dir, 'command_bar.png'),
        labels,
        [command[label] for label in labels],
        f'{split} command distribution')


def main():
    args = parse_args()
    if args.splits is not None and len(args.splits) != len(args.infos):
        raise ValueError('--splits length must match --infos length.')
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    summaries = []
    scene_top_rows = []

    for idx, info_path in enumerate(args.infos):
        split = (args.splits[idx] if args.splits is not None
                 else split_name_from_path(info_path))
        infos = load_infos(info_path)
        rows = []
        for info in infos:
            row = analyze_info(info, split, args)
            if row is not None:
                rows.append(row)
        all_rows.extend(rows)
        summaries.append(split_summary(split, rows))

        fieldnames = list(rows[0].keys()) if rows else [
            'split', 'token', 'scene_token']
        write_csv(osp.join(args.out_dir, f'{split}_samples.csv'),
                  rows, fieldnames)
        top_scenes, all_scenes = scene_summary(rows, args.top_scenes)
        for row in top_scenes:
            row = dict(row)
            row['split'] = split
            scene_top_rows.append(row)
        if all_scenes:
            scene_fields = ['scene_token', 'total', 'static', 'static_pct',
                            'slow', 'slow_pct', 'moving', 'moving_pct',
                            'turning', 'turning_pct', 'final_disp_mean',
                            'avg_speed_mean']
            write_csv(osp.join(args.out_dir, f'{split}_scenes.csv'),
                      all_scenes, scene_fields)
        save_plots(args.out_dir, split, rows)

    summary_fields = [
        'split', 'total', 'static', 'static_pct', 'slow', 'slow_pct',
        'moving', 'moving_pct', 'turning', 'turning_pct',
        'slow_turn', 'slow_turn_pct', 'straight', 'straight_pct',
        'left', 'left_pct', 'right', 'right_pct',
        'command_straight', 'command_straight_pct',
        'final_disp_mean', 'final_disp_p50', 'final_disp_p90',
        'avg_speed_mean', 'avg_speed_p50', 'turn_angle_p90',
    ]
    write_csv(osp.join(args.out_dir, 'summary.csv'), summaries,
              summary_fields)
    if scene_top_rows:
        scene_top_fields = ['split', 'scene_token', 'total', 'static',
                            'static_pct', 'slow', 'slow_pct', 'moving',
                            'moving_pct', 'turning', 'turning_pct',
                            'final_disp_mean', 'avg_speed_mean']
        write_csv(osp.join(args.out_dir, 'top_static_scenes.csv'),
                  scene_top_rows, scene_top_fields)

    print('Wrote:', args.out_dir)
    for summary in summaries:
        print(
            '{split}: N={total} static={static_pct:.1f}% '
            'slow={slow_pct:.1f}% moving={moving_pct:.1f}% '
            'turning={turning_pct:.1f}% final_p50={final_disp_p50:.2f}m '
            'speed_p50={avg_speed_p50:.2f}m/s'.format(**summary))


if __name__ == '__main__':
    main()
