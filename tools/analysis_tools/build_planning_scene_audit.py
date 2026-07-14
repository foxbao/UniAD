#!/usr/bin/env python3
"""Build a scene-level audit bundle for planning demonstration quality."""

import argparse
import csv
import html
import json
import math
import os
import os.path as osp
import pickle
import re
import textwrap
from collections import Counter, defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = (
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
)
FOCUS_LABELS = {7, 12}
COMMAND_NAMES = {0: 'Right', 1: 'Left', 2: 'Straight'}
HUMAN_FIELDS = (
    'human_label', 'human_control_mode', 'planning_usable', 'review_status',
    'reviewer', 'notes',
)
MANIFEST_LABELS = ('NaturalRun', 'OperationalStop', 'DetectionProbe',
                   'Uncertain')
CONTROL_MODES = ('Auto', 'Manual', 'Mixed', 'Unknown')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--infos', nargs='+', required=True,
        help='Inputs as SPLIT=path.pkl, for example train=data/foo.pkl.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--stop-speed', type=float, default=0.2)
    parser.add_argument('--slow-speed', type=float, default=0.8)
    parser.add_argument('--max-time-gap', type=float, default=2.0)
    parser.add_argument('--pose-cell', type=float, default=2.0)
    parser.add_argument('--pose-yaw-bin-deg', type=float, default=30.0)
    parser.add_argument('--target-static-radius', type=float, default=3.0)
    parser.add_argument('--target-max-distance', type=float, default=60.0)
    parser.add_argument('--target-min-observations', type=int, default=8)
    parser.add_argument(
        '--plot-mode', choices=('none', 'val', 'review', 'all'),
        default='review')
    parser.add_argument('--max-train-plots', type=int, default=160)
    return parser.parse_args()


def parse_named_path(value):
    if '=' not in value:
        raise ValueError(f'Expected SPLIT=path, got {value}')
    split, path = value.split('=', 1)
    split = split.strip()
    path = path.strip()
    if not split or not path:
        raise ValueError(f'Invalid SPLIT=path value: {value}')
    return split, path


def load_infos(path):
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        if 'data_list' in payload:
            return payload['data_list']
        if 'infos' in payload:
            return payload['infos']
    if isinstance(payload, list):
        return payload
    raise TypeError(f'Unsupported info payload in {path}: {type(payload)}')


def write_csv(path, rows, fieldnames):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def read_existing_manifest(path):
    if not osp.exists(path):
        return {}
    with open(path, newline='', encoding='utf-8') as handle:
        return {
            (row.get('split', ''), row.get('scene_token', '')): row
            for row in csv.DictReader(handle)
        }


def safe_name(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '__', value)
    return value.strip('_') or 'scene'


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def ego_pose(info):
    matrix = np.asarray(info.get('ego2global', np.eye(4)), dtype=np.float64)
    if matrix.shape != (4, 4):
        return None
    position = matrix[:2, 3].copy()
    yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return position, yaw, matrix


def planning_mask(value):
    mask = np.asarray(value)
    if mask.size == 0:
        return None
    if mask.ndim >= 2:
        mask = mask.any(axis=-1).reshape(-1)
    else:
        mask = mask.reshape(-1)
    return mask.astype(bool)


def planning_trajectory(value):
    trajectory = np.asarray(value, dtype=np.float64)
    if trajectory.size == 0:
        return None
    return trajectory.reshape(-1, trajectory.shape[-1])


def heading_change_deg(points, min_displacement=0.05):
    if len(points) < 2:
        return 0.0
    points = np.concatenate([
        np.zeros((1, 2), dtype=np.float64), points[:, :2]], axis=0)
    delta = np.diff(points, axis=0)
    norm = np.linalg.norm(delta, axis=1)
    indices = np.where(norm >= min_displacement)[0]
    if len(indices) < 2:
        return 0.0
    first = delta[indices[0]]
    last = delta[indices[-1]]
    return abs(math.degrees(wrap_pi(
        math.atan2(float(last[1]), float(last[0]))
        - math.atan2(float(first[1]), float(first[0])))))


def lateral_ratio(points):
    if len(points) < 2:
        return 0.0
    endpoint = points[-1, :2]
    distance = float(np.linalg.norm(endpoint))
    if distance < 1e-6:
        return 0.0
    direction = endpoint / distance
    normal = np.asarray([-direction[1], direction[0]])
    return float(np.max(np.abs(points[:, :2] @ normal)) / distance)


def planning_bucket(info):
    trajectory = planning_trajectory(info.get('sdc_planning', []))
    valid = planning_mask(info.get('sdc_planning_mask', []))
    if trajectory is None or valid is None:
        return None, None
    length = min(len(trajectory), len(valid))
    trajectory = trajectory[:length]
    valid = valid[:length]
    indices = np.where(valid)[0]
    if len(indices) == 0:
        return None, None
    last = int(indices[-1])
    points = trajectory[:last + 1][valid[:last + 1]]
    final_disp = float(np.linalg.norm(points[-1, :2]))
    if final_disp < 0.5:
        bucket = 'Static'
    elif final_disp < 2.0:
        bucket = 'Slow'
    else:
        yaw_change = 0.0
        if trajectory.shape[1] >= 3 and len(indices) >= 2:
            yaw = trajectory[indices, 2]
            if np.isfinite(yaw).all():
                yaw_change = abs(math.degrees(wrap_pi(
                    float(yaw[-1] - yaw[0]))))
        turning = max(heading_change_deg(points), yaw_change) >= 15.0 \
            or lateral_ratio(points) >= 0.15
        bucket = 'Turning' if turning else 'MovingStraight'
    signature = tuple(np.round(trajectory[:, :2].reshape(-1) / 0.5).astype(
        np.int32).tolist())
    return bucket, signature


def longest_true_duration(flags, durations):
    longest = 0.0
    current = 0.0
    for flag, duration in zip(flags, durations):
        if flag:
            current += float(duration)
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def infer_control_mode_proxy(row, slow_speed):
    """Infer a review hint; the info files contain no true control mode."""
    if (row['path_length_m'] < 3.0 or row['stop_ratio'] >= 0.75
            or row['speed_p90_mps'] < 0.2):
        return 'StoppedOrUnknown', 0.0
    if (row['speed_p50_mps'] < slow_speed
            and row['slow_ratio'] >= 0.60
            and row['speed_p90_mps'] <= 1.5):
        score = min(1.0, 0.5 + 0.5 * row['slow_ratio'])
        return 'LikelyManual', score
    if (row['speed_p50_mps'] >= 1.2
            and row['speed_p90_mps'] >= 2.0
            and row['slow_ratio'] <= 0.40):
        score = min(1.0, 0.5 + 0.5 * (1.0 - row['slow_ratio']))
        return 'LikelyAuto', score
    return 'MixedOrUnknown', 0.0


def summarize_target_tracks(frames, args):
    tracks = defaultdict(list)
    label_frame_count = Counter()
    object_counts = []
    for frame_index, item in enumerate(frames):
        pose = ego_pose(item)
        instances = item.get('instances') or []
        object_counts.append(len(instances))
        labels_in_frame = set()
        if pose is None:
            continue
        ego_xy, _, matrix = pose
        for instance in instances:
            label = instance.get('bbox_label_3d', instance.get('bbox_label'))
            bbox = instance.get('bbox_3d')
            track_id = instance.get('track_id')
            if label is None or bbox is None or track_id is None:
                continue
            label = int(label)
            bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
            if len(bbox) < 2:
                continue
            local = np.asarray([bbox[0], bbox[1], 0.0, 1.0])
            global_xy = (matrix @ local)[:2]
            tracks[(label, int(track_id))].append(
                (frame_index, ego_xy, global_xy))
            labels_in_frame.add(label)
        for label in labels_in_frame:
            label_frame_count[label] += 1

    target_rows = []
    for (label, track_id), observations in tracks.items():
        if len(observations) < args.target_min_observations:
            continue
        object_xy = np.stack([row[2] for row in observations])
        object_center = np.median(object_xy, axis=0)
        object_radius = float(np.percentile(
            np.linalg.norm(object_xy - object_center, axis=1), 90))
        if object_radius > args.target_static_radius:
            continue
        ego_xy = np.stack([row[1] for row in observations])
        relative = ego_xy - object_center
        distance = np.linalg.norm(relative, axis=1)
        keep = distance <= args.target_max_distance
        if keep.sum() < args.target_min_observations:
            continue
        bearing = np.unwrap(np.arctan2(relative[keep, 1], relative[keep, 0]))
        total_angle = float(np.degrees(np.abs(np.diff(bearing)).sum())) \
            if len(bearing) > 1 else 0.0
        span_angle = float(np.degrees(np.ptp(bearing))) \
            if len(bearing) > 1 else 0.0
        target_rows.append(dict(
            label=label,
            class_name=(CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES)
                        else f'Class_{label}'),
            track_id=track_id,
            observations=int(keep.sum()),
            object_center=object_center,
            object_radius_m=object_radius,
            median_distance_m=float(np.median(distance[keep])),
            total_orbit_deg=total_angle,
            orbit_span_deg=span_angle,
            is_focus=int(label in FOCUS_LABELS),
        ))
    target_rows.sort(key=lambda row: (
        row['is_focus'], row['total_orbit_deg'], row['observations']),
        reverse=True)
    median_objects = float(np.median(object_counts)) if object_counts else 0.0
    p90_objects = float(np.percentile(object_counts, 90)) \
        if object_counts else 0.0
    return target_rows, label_frame_count, median_objects, p90_objects


def auto_classify(row):
    reasons = []
    probe_score = 0.0
    natural_score = 0.0
    path = row['path_length_m']
    progress = row['progress_ratio']
    stop_ratio = row['stop_ratio']
    revisit = row['pose_revisit_ratio']
    focus_orbit = row['focus_target_orbit_deg']
    any_orbit = row['static_target_orbit_deg']

    if row['control_mode_proxy'] == 'LikelyManual':
        reasons.append('likely_manual_control_proxy')
    if row['duration_s'] >= 20.0 and stop_ratio >= 0.75:
        probe_score += 0.25
        reasons.append('long_stationary')
    if path >= 10.0 and progress < 0.25:
        probe_score += 0.25
        reasons.append('low_route_progress')
    if path >= 20.0 and row['net_displacement_m'] < 5.0:
        probe_score += 0.15
        reasons.append('closed_loop')
    if row['frame_count'] >= 30 and revisit >= 0.55:
        probe_score += 0.15
        reasons.append('pose_revisit')
    if focus_orbit >= 120.0:
        probe_score += 0.45
        reasons.append('focus_target_orbit')
    elif any_orbit >= 180.0:
        probe_score += 0.25
        reasons.append('static_target_orbit')
    if path >= 10.0 and row['turn_deg_per_m'] >= 8.0:
        probe_score += 0.10
        reasons.append('high_turn_density')
    if row['planning_repeat_ratio'] >= 0.75:
        probe_score += 0.10
        reasons.append('repeated_planning_labels')
    if (row['static_slow_ratio'] >= 0.85
            and row['median_objects_per_frame'] >= 5.0):
        probe_score += 0.15
        reasons.append('stationary_dense_objects')
    probe_score = min(probe_score, 1.0)

    if path >= 20.0:
        natural_score += 0.20
    if progress >= 0.65:
        natural_score += 0.25
    if stop_ratio <= 0.35:
        natural_score += 0.15
    if revisit <= 0.35:
        natural_score += 0.15
    if focus_orbit < 60.0 and any_orbit < 120.0:
        natural_score += 0.15
    if row['moving_ratio'] >= 0.50:
        natural_score += 0.10
    natural_score = min(natural_score, 1.0)

    if probe_score >= 0.55:
        label = 'DetectionProbe'
        confidence = probe_score
    elif natural_score >= 0.65 and probe_score < 0.30:
        label = 'NaturalRun'
        confidence = natural_score
    else:
        label = 'Uncertain'
        confidence = abs(probe_score - natural_score)
        reasons.append('intent_requires_review')
    return label, confidence, probe_score, natural_score, sorted(set(reasons))


def analyze_scene(split, scene_token, frames, args):
    frames = sorted(frames, key=lambda item: float(item.get('timestamp', 0.0)))
    poses = []
    kept_frames = []
    for item in frames:
        pose = ego_pose(item)
        if pose is not None:
            poses.append((float(item.get('timestamp', 0.0)), *pose[:2]))
            kept_frames.append(item)
    if not poses:
        return None, None
    timestamps = np.asarray([row[0] for row in poses], dtype=np.float64)
    positions = np.stack([row[1] for row in poses])
    yaws = np.asarray([row[2] for row in poses], dtype=np.float64)
    delta_t = np.diff(timestamps)
    delta_xy = np.diff(positions, axis=0)
    distance = np.linalg.norm(delta_xy, axis=1)
    valid_transition = ((delta_t > 1e-4)
                        & (delta_t <= args.max_time_gap))
    distance_valid = distance[valid_transition]
    duration_valid = delta_t[valid_transition]
    speed = np.divide(
        distance_valid, duration_valid,
        out=np.zeros_like(distance_valid), where=duration_valid > 0)
    yaw_delta = np.asarray([
        abs(wrap_pi(value)) for value in np.diff(yaws)], dtype=np.float64)
    yaw_delta = yaw_delta[valid_transition]
    path_length = float(distance_valid.sum())
    net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    progress_ratio_raw = net_displacement / max(path_length, 1e-6)
    progress_ratio = min(progress_ratio_raw, 1.0)
    stop_flags = speed < args.stop_speed
    slow_flags = speed < args.slow_speed
    stop_ratio = float(stop_flags.mean()) if len(stop_flags) else 1.0
    slow_ratio = float(slow_flags.mean()) if len(slow_flags) else 1.0
    max_stop_s = longest_true_duration(stop_flags, duration_valid)
    duration_s = float(timestamps[-1] - timestamps[0]) \
        if len(timestamps) > 1 else 0.0
    cumulative_yaw_deg = float(np.degrees(yaw_delta.sum()))
    net_yaw_deg = abs(math.degrees(wrap_pi(float(yaws[-1] - yaws[0]))))

    xy_bins = np.round(positions / args.pose_cell).astype(np.int64)
    yaw_bin_size = math.radians(args.pose_yaw_bin_deg)
    yaw_bins = np.round(yaws / yaw_bin_size).astype(np.int64)
    pose_bins = np.column_stack([xy_bins, yaw_bins])
    unique_xy = len({tuple(row) for row in xy_bins})
    unique_pose = len({tuple(row) for row in pose_bins})
    position_revisit = 1.0 - unique_xy / max(len(xy_bins), 1)
    pose_revisit = 1.0 - unique_pose / max(len(pose_bins), 1)

    bucket_count = Counter()
    command_count = Counter()
    signatures = []
    for item in frames:
        bucket, signature = planning_bucket(item)
        if bucket is not None:
            bucket_count[bucket] += 1
        if signature is not None:
            signatures.append(signature)
        command = np.asarray(item.get('command', [-1])).reshape(-1)
        command_count[int(command[0]) if len(command) else -1] += 1
    planning_n = sum(bucket_count.values())
    static_slow = bucket_count['Static'] + bucket_count['Slow']
    moving = bucket_count['MovingStraight'] + bucket_count['Turning']
    repeat_ratio = 1.0 - len(set(signatures)) / max(len(signatures), 1)

    targets, label_frames, median_objects, p90_objects = \
        summarize_target_tracks(kept_frames, args)
    dominant = targets[0] if targets else None
    focus_targets = [row for row in targets if row['is_focus']]
    dominant_focus = focus_targets[0] if focus_targets else None
    frame_count = len(frames)

    row = dict(
        split=split,
        scene_token=scene_token,
        frame_count=frame_count,
        planning_valid_frames=planning_n,
        transition_count=max(len(timestamps) - 1, 0),
        valid_transition_count=int(valid_transition.sum()),
        disconnected_transition_count=int((~valid_transition).sum()),
        valid_transition_ratio=float(valid_transition.mean())
        if len(valid_transition) else 0.0,
        duration_s=duration_s,
        path_length_m=path_length,
        net_displacement_m=net_displacement,
        progress_ratio=progress_ratio,
        progress_ratio_raw=progress_ratio_raw,
        stop_ratio=stop_ratio,
        slow_ratio=slow_ratio,
        max_continuous_stop_s=max_stop_s,
        speed_mean_mps=float(speed.mean()) if len(speed) else 0.0,
        speed_p50_mps=float(np.median(speed)) if len(speed) else 0.0,
        speed_p90_mps=float(np.percentile(speed, 90)) if len(speed) else 0.0,
        cumulative_yaw_deg=cumulative_yaw_deg,
        net_yaw_deg=net_yaw_deg,
        turn_deg_per_m=cumulative_yaw_deg / max(path_length, 1e-6),
        position_revisit_ratio=position_revisit,
        pose_revisit_ratio=pose_revisit,
        planning_repeat_ratio=repeat_ratio,
        static_frames=bucket_count['Static'],
        slow_frames=bucket_count['Slow'],
        moving_straight_frames=bucket_count['MovingStraight'],
        turning_frames=bucket_count['Turning'],
        static_slow_ratio=static_slow / max(planning_n, 1),
        moving_ratio=moving / max(planning_n, 1),
        turning_ratio=bucket_count['Turning'] / max(planning_n, 1),
        command_straight_ratio=command_count[2] / max(frame_count, 1),
        command_left_ratio=command_count[1] / max(frame_count, 1),
        command_right_ratio=command_count[0] / max(frame_count, 1),
        median_objects_per_frame=median_objects,
        p90_objects_per_frame=p90_objects,
        crane_frame_ratio=label_frames[7] / max(frame_count, 1),
        wheelcrane_frame_ratio=label_frames[12] / max(frame_count, 1),
        static_target_orbit_deg=(
            dominant['total_orbit_deg'] if dominant else 0.0),
        static_target_orbit_span_deg=(
            dominant['orbit_span_deg'] if dominant else 0.0),
        focus_target_orbit_deg=(
            dominant_focus['total_orbit_deg'] if dominant_focus else 0.0),
        focus_target_orbit_span_deg=(
            dominant_focus['orbit_span_deg'] if dominant_focus else 0.0),
        dominant_target_class=(
            dominant['class_name'] if dominant else ''),
        dominant_target_track_id=(
            dominant['track_id'] if dominant else ''),
        dominant_target_observations=(
            dominant['observations'] if dominant else 0),
        dominant_target_distance_m=(
            dominant['median_distance_m'] if dominant else 0.0),
    )
    control_mode_proxy, control_mode_proxy_score = \
        infer_control_mode_proxy(row, args.slow_speed)
    row.update(
        control_mode_proxy=control_mode_proxy,
        control_mode_proxy_score=control_mode_proxy_score,
    )
    label, confidence, probe_score, natural_score, reasons = auto_classify(row)
    row.update(
        auto_label=label,
        auto_confidence=confidence,
        probe_score=probe_score,
        natural_score=natural_score,
        reason_codes=';'.join(reasons),
        manual_review_required=int(
            split.lower() == 'val' or label != 'NaturalRun'
            or confidence < 0.80),
    )
    plot_data = dict(
        timestamps=timestamps,
        positions=positions,
        speed=speed,
        speed_time=timestamps[1:][valid_transition],
        yaws=yaws,
        bucket_count=bucket_count,
        command_count=command_count,
        dominant_target=(dominant['object_center'] if dominant else None),
    )
    return row, plot_data


def render_scene_plot(path, row, data, args):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    position = data['positions'] - data['positions'][0]
    ax = axes[0, 0]
    if len(position) > 1:
        color = np.arange(len(position))
        scatter = ax.scatter(position[:, 0], position[:, 1], c=color,
                             s=13, cmap='viridis')
        fig.colorbar(scatter, ax=ax, label='frame')
    else:
        ax.scatter(position[:, 0], position[:, 1], s=20)
    ax.scatter(position[0, 0], position[0, 1], marker='o', s=70,
               label='start')
    ax.scatter(position[-1, 0], position[-1, 1], marker='x', s=80,
               label='end')
    target = data.get('dominant_target')
    if target is not None:
        target = np.asarray(target) - data['positions'][0]
        ax.scatter(target[0], target[1], marker='*', s=140,
                   label=row['dominant_target_class'])
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title('Ego path in global frame (relative to start)', fontsize=10)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='best')

    ax = axes[0, 1]
    if len(data['speed']):
        time = data['speed_time'] - data['timestamps'][0]
        ax.plot(time, data['speed'], linewidth=1.2)
    ax.axhline(args.stop_speed, color='r', linestyle='--', linewidth=0.8,
               label='stop threshold')
    ax.axhline(args.slow_speed, color='orange', linestyle='--', linewidth=0.8,
               label='slow threshold')
    ax.set_title('Ego speed')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('m/s')
    ax.legend(loc='best')

    ax = axes[1, 0]
    names = ('Static', 'Slow', 'MovingStraight', 'Turning')
    ax.bar(names, [data['bucket_count'][name] for name in names])
    ax.set_title('3-second planning GT buckets')
    ax.tick_params(axis='x', rotation=20)

    ax = axes[1, 1]
    ax.axis('off')
    lines = [
        f'scene: {row["scene_token"]}',
        f'auto: {row["auto_label"]}  confidence={row["auto_confidence"]:.2f}',
        f'probe/natural score: {row["probe_score"]:.2f}/{row["natural_score"]:.2f}',
        f'path/net/progress: {row["path_length_m"]:.1f}m / '
        f'{row["net_displacement_m"]:.1f}m / {row["progress_ratio"]:.2f}',
        f'stop/slow/revisit: {row["stop_ratio"]:.2f} / '
        f'{row["slow_ratio"]:.2f} / {row["pose_revisit_ratio"]:.2f}',
        f'control mode proxy: {row["control_mode_proxy"]}  '
        f'score={row["control_mode_proxy_score"]:.2f}',
        f'yaw total/per m: {row["cumulative_yaw_deg"]:.0f}deg / '
        f'{row["turn_deg_per_m"]:.1f}deg/m',
        f'target: {row["dominant_target_class"] or "none"}  '
        f'orbit={row["static_target_orbit_deg"]:.0f}deg',
    ]
    reasons = (row['reason_codes'] or 'none').replace(';', '; ')
    lines.extend(textwrap.wrap(
        f'reasons: {reasons}', width=58,
        subsequent_indent='         ', break_long_words=False,
        break_on_hyphens=False))
    ax.text(0.0, 1.0, '\n'.join(lines), va='top', family='monospace',
            fontsize=9)
    fig.suptitle(f'{row["split"]}: {row["scene_token"]}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(osp.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_review_html(path, rows):
    columns = (
        'split', 'scene_token', 'auto_label', 'auto_confidence',
        'probe_score', 'natural_score', 'path_length_m', 'progress_ratio',
        'stop_ratio', 'control_mode_proxy', 'pose_revisit_ratio',
        'focus_target_orbit_deg', 'reason_codes',
    )
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('<!doctype html><meta charset="utf-8">')
        handle.write('<title>Planning scene audit</title>')
        handle.write('<style>body{font-family:sans-serif}table{border-collapse:'
                     'collapse;font-size:12px}td,th{border:1px solid #bbb;'
                     'padding:4px}tr:nth-child(even){background:#f5f5f5}'
                     'img{width:220px}</style>')
        handle.write('<h1>Planning scene audit</h1><table><thead><tr>')
        handle.write('<th>plot</th>')
        for column in columns:
            handle.write(f'<th>{html.escape(column)}</th>')
        handle.write('</tr></thead><tbody>')
        for row in rows:
            handle.write('<tr>')
            plot_path = row.get('plot_path', '')
            if plot_path:
                escaped = html.escape(plot_path)
                handle.write(
                    f'<td><a href="{escaped}"><img src="{escaped}"></a></td>')
            else:
                handle.write('<td></td>')
            for column in columns:
                value = row.get(column, '')
                if isinstance(value, float):
                    value = f'{value:.4f}'
                handle.write(f'<td>{html.escape(str(value))}</td>')
            handle.write('</tr>')
        handle.write('</tbody></table>')


def write_report(path, rows, source_paths):
    split_rows = defaultdict(list)
    for row in rows:
        split_rows[row['split']].append(row)
    lines = [
        '# Planning Scene Audit', '',
        'Automatic labels are review suggestions, not ground truth.', '',
        '## Sources', '',
    ]
    for split, source in source_paths.items():
        lines.append(f'- `{split}`: `{source}`')
    lines.extend(['', '## Summary', '',
                  '| split | scenes | frames | NaturalRun | DetectionProbe | Uncertain | manual review |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for split, items in split_rows.items():
        labels = Counter(row['auto_label'] for row in items)
        lines.append(
            f'| {split} | {len(items)} | '
            f'{sum(row["frame_count"] for row in items)} | '
            f'{labels["NaturalRun"]} | {labels["DetectionProbe"]} | '
            f'{labels["Uncertain"]} | '
            f'{sum(row["manual_review_required"] for row in items)} |')
    lines.extend([
        '', '## Human Review', '',
        'Edit `scene_manifest.csv`. Fill `human_label` with one of:', '',
        '- `NaturalRun`',
        '- `OperationalStop`',
        '- `DetectionProbe`',
        '- `Uncertain`', '',
        'Fill `human_control_mode` with `Auto`, `Manual`, `Mixed`, or '
        '`Unknown`. The automatic control-mode proxy is only a speed-based '
        'review hint because the current info files contain no control-mode '
        'field.', '',
        'Set `planning_usable` to `1` for demonstrations that should supervise '
        'planning, otherwise `0`.', '',
        'All validation scenes, DetectionProbe candidates, and uncertain or '
        'low-confidence training scenes require review.', '',
    ])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    named_paths = dict(parse_named_path(value) for value in args.infos)
    grouped = defaultdict(list)
    for split, path in named_paths.items():
        for info in load_infos(path):
            scene = str(info.get('scene_token', '') or 'unknown_scene')
            grouped[(split, scene)].append(info)

    rows = []
    plot_data = {}
    for (split, scene), frames in grouped.items():
        row, data = analyze_scene(split, scene, frames, args)
        if row is None:
            continue
        rows.append(row)
        plot_data[(split, scene)] = data
    rows.sort(key=lambda row: (row['split'], row['scene_token']))

    manifest_path = osp.join(args.out_dir, 'scene_manifest.csv')
    existing = read_existing_manifest(manifest_path)
    for row in rows:
        previous = existing.get((row['split'], row['scene_token']), {})
        for field in HUMAN_FIELDS:
            row[field] = previous.get(field, '')

    review_rows = sorted(rows, key=lambda row: (
        0 if row['split'].lower() == 'val' else 1,
        0 if row['manual_review_required'] else 1,
        -row['probe_score'], row['scene_token']))
    train_plot_count = 0
    for row in review_rows:
        split = row['split'].lower()
        should_plot = args.plot_mode == 'all'
        if args.plot_mode == 'val':
            should_plot = split == 'val'
        elif args.plot_mode == 'review':
            should_plot = split == 'val' or (
                row['manual_review_required']
                and train_plot_count < args.max_train_plots)
        if split != 'val' and should_plot:
            train_plot_count += 1
        if should_plot:
            relative = osp.join(
                'plots', row['split'], safe_name(row['scene_token']) + '.png')
            render_scene_plot(
                osp.join(args.out_dir, relative), row,
                plot_data[(row['split'], row['scene_token'])], args)
            row['plot_path'] = relative
        else:
            row['plot_path'] = ''

    feature_fields = list(rows[0].keys()) if rows else []
    write_csv(osp.join(args.out_dir, 'scene_features.csv'), rows,
              feature_fields)
    manifest_fields = [
        'split', 'scene_token', 'frame_count', 'auto_label',
        'auto_confidence', 'probe_score', 'natural_score', 'reason_codes',
        'control_mode_proxy', 'control_mode_proxy_score',
        'dominant_target_class', 'focus_target_orbit_deg',
        'manual_review_required', 'plot_path', *HUMAN_FIELDS,
    ]
    write_csv(manifest_path, rows, manifest_fields)
    write_csv(osp.join(args.out_dir, 'review_queue.csv'), review_rows,
              manifest_fields)
    write_review_html(osp.join(args.out_dir, 'review_index.html'), review_rows)
    write_report(
        osp.join(args.out_dir, 'README.md'), rows, named_paths)
    summary = dict(
        sources=named_paths,
        scenes=len(rows),
        frames=int(sum(row['frame_count'] for row in rows)),
        auto_labels=dict(Counter(row['auto_label'] for row in rows)),
        manual_review_required=int(sum(
            row['manual_review_required'] for row in rows)),
        allowed_human_labels=MANIFEST_LABELS,
        allowed_control_modes=CONTROL_MODES,
    )
    with open(osp.join(args.out_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    print(json.dumps(summary, indent=2))
    print(f'Wrote audit bundle to {args.out_dir}')


if __name__ == '__main__':
    main()
