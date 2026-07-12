#!/usr/bin/env python3
"""Evaluate a lane-chain x speed-profile planning candidate oracle."""

import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from projects.mmdet3d_plugin.uniad.dense_heads.motion_head_plugin.map_lane_encoder import (  # noqa: E402
    HDMapParser,
)


EVAL_INDICES = (1, 3, 5)


def load_infos(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data.get('data_list', data.get('infos', []))
    return data


def load_profiles(path):
    data = np.load(path, allow_pickle=True)
    profiles = np.asarray(data['profiles'], dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError(f'Expected [K,T] profiles, got {profiles.shape}')
    return profiles


def valid_plan(info, steps):
    traj = np.asarray(info.get('sdc_planning'))
    mask = np.asarray(info.get('sdc_planning_mask'))
    if traj.size == 0 or mask.size == 0:
        return None, None
    traj = traj.reshape(-1, traj.shape[-2], traj.shape[-1])[0]
    mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
    length = min(steps, len(traj), len(mask))
    valid = mask[:length].any(axis=-1)
    if length == 0 or not valid.any():
        return None, None
    return traj[:length, :2].astype(np.float64), valid


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def motion_bucket(traj, valid):
    points = traj[valid]
    if len(points) == 0:
        return 'unknown'
    final_disp = float(np.linalg.norm(points[-1]))
    if final_disp < 0.5:
        return 'static'
    if final_disp < 2.0:
        return 'slow'

    xy = np.concatenate(
        [np.zeros((1, 2), dtype=np.float64), points], axis=0)
    deltas = np.diff(xy, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    heading_idx = np.where(norms >= 0.05)[0]
    heading_change = 0.0
    if len(heading_idx) >= 2:
        first = deltas[heading_idx[0]]
        last = deltas[heading_idx[-1]]
        heading_change = abs(math.degrees(wrap_pi(
            math.atan2(last[1], last[0])
            - math.atan2(first[1], first[0]))))
    end = points[-1]
    net = float(np.linalg.norm(end))
    lateral_ratio = 0.0
    if net >= 1e-6:
        normal = np.array([-end[1] / net, end[0] / net])
        lateral_ratio = float(np.max(np.abs(points @ normal)) / net)
    if heading_change >= 15.0 or lateral_ratio >= 0.15:
        return 'turning'
    return 'moving_straight'


def cumulative_distance(traj, valid=None):
    if valid is None:
        xy = np.concatenate(
            [np.zeros((1, 2), dtype=np.float64), traj], axis=0)
        return np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    distance = np.zeros(len(traj), dtype=np.float64)
    previous = np.zeros(2, dtype=np.float64)
    total = 0.0
    for index, point in enumerate(traj):
        if valid[index]:
            total += float(np.linalg.norm(point - previous))
            previous = point
        distance[index] = total
    return distance


def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def nearest_point_and_heading(points, position):
    index = int(np.linalg.norm(points - position[None], axis=1).argmin())
    left = max(0, index - 1)
    right = min(len(points) - 1, index + 1)
    tangent = points[right] - points[left]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        tangent = np.array([1.0, 0.0], dtype=np.float64)
    else:
        tangent = tangent / norm
    distance = float(np.linalg.norm(points[index] - position))
    return index, distance, tangent


def select_start_lanes(map_parser, ego2global, num_start_lanes,
                       max_start_distance, max_heading_error_deg):
    ego2global = np.asarray(ego2global, dtype=np.float64)
    position = ego2global[:2, 3]
    heading = ego2global[:2, 0]
    heading = heading / max(float(np.linalg.norm(heading)), 1e-6)
    min_alignment = math.cos(math.radians(max_heading_error_deg))
    candidates = []
    nearest = []
    for lane in map_parser.lanes:
        points = np.asarray(lane['central'], dtype=np.float64)
        index, distance, tangent = nearest_point_and_heading(
            points, position)
        signed_alignment = float(np.dot(tangent, heading))
        alignment = abs(signed_alignment)
        row = (distance + 5.0 * (1.0 - alignment), distance,
               -alignment, str(lane['id']), index)
        nearest.append(row)
        if distance <= max_start_distance and alignment >= min_alignment:
            candidates.append(row)
    used_fallback = False
    if not candidates:
        candidates = nearest
        used_fallback = True
    candidates.sort()
    oriented = []
    for score, distance, neg_alignment, lane_id, index in \
            candidates[:num_start_lanes]:
        lane_size = len(map_parser.lane_by_id[lane_id]['central'])
        oriented.append((
            score, distance, neg_alignment, lane_id, index, False))
        oriented.append((
            score, distance, neg_alignment, lane_id,
            lane_size - 1 - index, True))
    return oriented, used_fallback


def join_lane_points(points, extension, max_join_gap):
    if len(points) == 0:
        return extension.copy()
    gap = float(np.linalg.norm(points[-1] - extension[0]))
    if gap > max_join_gap:
        return None
    start = 1 if gap < 0.2 else 0
    return np.concatenate([points, extension[start:]], axis=0)


def extend_lane_chain(map_parser, lane_id, reverse, points, sequence,
                      required_length, max_depth, max_join_gap, output):
    if (polyline_length(points) >= required_length
            or len(sequence) >= max_depth):
        output.append((tuple(sequence), points))
        return
    lane = map_parser.lane_by_id[lane_id]
    next_field = 'predecessor_ids' if reverse else 'successor_ids'
    successors = [
        successor for successor in lane.get(next_field, [])
        if (successor in map_parser.lane_by_id
            and (successor, reverse) not in sequence)
    ]
    extended = False
    for successor in successors:
        successor_points = np.asarray(
            map_parser.lane_by_id[successor]['central'], dtype=np.float64)
        if reverse:
            successor_points = successor_points[::-1]
        joined = join_lane_points(points, successor_points, max_join_gap)
        if joined is None:
            continue
        extended = True
        extend_lane_chain(
            map_parser, successor, reverse, joined,
            sequence + [(successor, reverse)],
            required_length, max_depth, max_join_gap, output)
    if not extended:
        output.append((tuple(sequence), points))


def build_route_paths(map_parser, ego2global, num_start_lanes, max_paths,
                      required_length, max_start_distance,
                      max_heading_error_deg, max_depth, max_join_gap):
    start_lanes, used_fallback = select_start_lanes(
        map_parser, ego2global, num_start_lanes, max_start_distance,
        max_heading_error_deg)
    paths = []
    for (score, _distance, _neg_alignment, lane_id, start_index,
         reverse) in start_lanes:
        lane_points = np.asarray(
            map_parser.lane_by_id[lane_id]['central'], dtype=np.float64)
        if reverse:
            lane_points = lane_points[::-1]
        lane_points = lane_points[start_index:]
        if len(lane_points) < 2:
            continue
        expanded = []
        extend_lane_chain(
            map_parser, lane_id, reverse, lane_points,
            [(lane_id, reverse)], required_length, max_depth, max_join_gap,
            expanded)
        for sequence, points in expanded:
            paths.append((score, sequence, points))
    paths.sort(key=lambda row: (row[0], row[1]))
    unique = []
    seen = set()
    for row in paths:
        if row[1] in seen:
            continue
        seen.add(row[1])
        unique.append(row)
        if len(unique) >= max_paths:
            break
    return unique, used_fallback


def interpolate_polyline(points, distances):
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    if cumulative[-1] < 1e-6:
        return np.repeat(points[:1], len(distances), axis=0), True
    clamped = bool(np.any(distances > cumulative[-1]))
    distances = np.clip(distances, 0.0, cumulative[-1])
    right = np.searchsorted(cumulative, distances, side='left')
    right = np.clip(right, 1, len(points) - 1)
    left = right - 1
    denom = np.maximum(cumulative[right] - cumulative[left], 1e-6)
    alpha = ((distances - cumulative[left]) / denom)[:, None]
    samples = points[left] * (1.0 - alpha) + points[right] * alpha
    return samples, clamped


def sample_path(points, distances, ego2global):
    samples, clamped = interpolate_polyline(points, distances)
    origin = points[0]
    relative_global = samples - origin[None]
    global2ego = np.linalg.inv(np.asarray(ego2global, dtype=np.float64))
    relative_ego = relative_global @ global2ego[:2, :2].T
    return relative_ego, clamped


def apply_lateral_offset(path, offset):
    if abs(offset) < 1e-8:
        return path.copy()
    points = np.concatenate(
        [np.zeros((1, 2), dtype=np.float64), path], axis=0)
    tangent = np.diff(points, axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(norms, 1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    distance = np.cumsum(norms[:, 0])
    progress = distance / max(float(distance[-1]), 1e-6)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    return path + normal * (float(offset) * smooth[:, None])


def candidate_cost(candidates, gt, valid):
    indices = [
        index for index in EVAL_INDICES
        if index < len(gt) and index < candidates.shape[1] and valid[index]
    ]
    if not indices:
        indices = np.where(valid[:min(len(gt), candidates.shape[1])])[0]
    if len(indices) == 0:
        return None
    error = np.linalg.norm(
        candidates[:, indices] - gt[None, indices], axis=-1)
    return error.mean(axis=1)


def best_candidate(paths, distance_profiles, lateral_offsets, ego2global,
                   gt, valid):
    candidates = []
    clamped = 0
    for _score, _sequence, points in paths:
        for profile in distance_profiles:
            candidate, was_clamped = sample_path(
                points, profile[:len(gt)], ego2global)
            for offset in lateral_offsets:
                candidates.append(apply_lateral_offset(candidate, offset))
                clamped += int(was_clamped)
    if not candidates:
        return None
    candidates = np.asarray(candidates, dtype=np.float64)
    costs = candidate_cost(candidates, gt, valid)
    if costs is None:
        return None
    best = int(costs.argmin())
    return candidates[best], float(costs[best]), len(candidates), clamped


def add_metric(agg, prediction, gt, valid, path_count, candidate_count,
               clamped_count):
    agg['n'] += 1
    agg['path_count_sum'] += path_count
    agg['candidate_count_sum'] += candidate_count
    agg['clamped_count_sum'] += clamped_count
    for index in EVAL_INDICES:
        if index < len(gt) and index < len(prediction) and valid[index]:
            error = float(np.linalg.norm(prediction[index] - gt[index]))
            agg[f'L2_{index}_sum'] += error
            agg[f'L2_{index}_n'] += 1


def summarize(agg):
    row = dict(n=int(agg['n']))
    horizons = []
    for index, name in zip(EVAL_INDICES, ('1s', '2s', '3s')):
        count = agg[f'L2_{index}_n']
        value = (agg[f'L2_{index}_sum'] / count
                 if count else float('nan'))
        row[f'L2_{name}'] = value
        if np.isfinite(value):
            horizons.append(value)
    row['avg.L2'] = float(np.mean(horizons)) if horizons else float('nan')
    denom = max(agg['n'], 1)
    row['paths_per_sample'] = agg['path_count_sum'] / denom
    row['candidates_per_sample'] = agg['candidate_count_sum'] / denom
    row['clamped_per_sample'] = agg['clamped_count_sum'] / denom
    return row


def print_rows(title, rows):
    print(title)
    for name, row in rows:
        print(
            f'{name:18s} N={row["n"]:5d} '
            f'L2@1/2/3s={row["L2_1s"]:.4f}/'
            f'{row["L2_2s"]:.4f}/{row["L2_3s"]:.4f} '
            f'avg.L2={row["avg.L2"]:.4f} '
            f'paths={row["paths_per_sample"]:.1f} '
            f'cands={row["candidates_per_sample"]:.1f} '
            f'clamped={row["clamped_per_sample"]:.1f}')


def plot_sample(path, paths, ego2global, gt, valid, factorized, exact,
                title, required_length):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    for _score, _sequence, points in paths:
        length = min(polyline_length(points), required_length)
        distances = np.linspace(0.0, length, 80)
        route, _ = sample_path(points, distances, ego2global)
        axis.plot(route[:, 0], route[:, 1], color='0.75', linewidth=0.8)
    axis.plot(
        factorized[:, 0], factorized[:, 1], 'o-', color='tab:red',
        linewidth=2.0, markersize=4, label='profile oracle')
    axis.plot(
        exact[:, 0], exact[:, 1], 'o-', color='tab:blue',
        linewidth=1.5, markersize=3, label='exact-speed oracle')
    axis.plot(
        gt[valid, 0], gt[valid, 1], 'o-', color='black', linewidth=2.0,
        markersize=4, label='GT')
    axis.scatter([0.0], [0.0], marker='x', color='tab:green', s=60)
    axis.set_aspect('equal', adjustable='box')
    axis.grid(True, alpha=0.25)
    axis.set_xlabel('forward x (m)')
    axis.set_ylabel('left y (m)')
    axis.set_title(title)
    axis.legend(loc='best')
    focus = np.concatenate([factorized, exact, gt[valid]], axis=0)
    minimum = focus.min(axis=0) - 2.0
    maximum = focus.max(axis=0) + 2.0
    axis.set_xlim(minimum[0], maximum[0])
    axis.set_ylim(minimum[1], maximum[1])
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate topology-path x speed-profile planning oracle.')
    parser.add_argument('--ann-file', default='data/kl_8/kl_infos_val.pkl')
    parser.add_argument('--map-path', default='data/kl_8/map/base_map.txt')
    parser.add_argument(
        '--profiles',
        default='data/others/planning_speed_profiles_d0_8.npz')
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--num-points-per-lane', type=int, default=80)
    parser.add_argument('--num-start-lanes', type=int, default=8)
    parser.add_argument('--max-paths', type=int, default=16)
    parser.add_argument('--path-margin', type=float, default=5.0)
    parser.add_argument('--max-start-distance', type=float, default=12.0)
    parser.add_argument('--max-heading-error-deg', type=float, default=80.0)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--max-join-gap', type=float, default=6.0)
    parser.add_argument(
        '--lateral-offsets', type=float, nargs='+',
        default=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument('--command', type=int, default=None)
    parser.add_argument('--plot-dir', default=None)
    parser.add_argument('--plot-limit', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    infos = load_infos(args.ann_file)
    if args.command is not None:
        infos = [
            info for info in infos
            if int(np.asarray(info.get('command', [-1])).reshape(-1)[0])
            == args.command
        ]
    if args.limit > 0:
        infos = infos[:args.limit]
    profiles = load_profiles(args.profiles)
    if profiles.shape[1] < args.steps:
        raise ValueError(
            f'Profiles have {profiles.shape[1]} steps, need {args.steps}')
    map_parser = HDMapParser(
        args.map_path, num_points_per_lane=args.num_points_per_lane)
    required_length = float(profiles[:, args.steps - 1].max()) \
        + args.path_margin

    factorized_all = defaultdict(float)
    exact_all = defaultdict(float)
    factorized_bucket = defaultdict(lambda: defaultdict(float))
    exact_bucket = defaultdict(lambda: defaultdict(float))
    factorized_command = defaultdict(lambda: defaultdict(float))
    exact_command = defaultdict(lambda: defaultdict(float))
    no_plan = 0
    no_path = 0
    heading_fallback = 0
    plotted = 0
    if args.plot_dir and args.plot_limit > 0:
        os.makedirs(args.plot_dir, exist_ok=True)

    for info in infos:
        gt, valid = valid_plan(info, args.steps)
        if gt is None:
            no_plan += 1
            continue
        paths, used_fallback = build_route_paths(
            map_parser, info.get('ego2global', np.eye(4)),
            args.num_start_lanes, args.max_paths, required_length,
            args.max_start_distance, args.max_heading_error_deg,
            args.max_depth, args.max_join_gap)
        heading_fallback += int(used_fallback)
        if not paths:
            no_path += 1
            continue

        factorized = best_candidate(
            paths, profiles, args.lateral_offsets,
            info.get('ego2global', np.eye(4)), gt, valid)
        exact_profile = cumulative_distance(gt, valid)[None]
        exact = best_candidate(
            paths, exact_profile, args.lateral_offsets,
            info.get('ego2global', np.eye(4)), gt, valid)
        if factorized is None or exact is None:
            no_path += 1
            continue

        bucket = motion_bucket(gt, valid)
        command_arr = np.asarray(info.get('command', [-1])).reshape(-1)
        command = int(command_arr[0]) if len(command_arr) else -1
        factorized_pred, _cost, factorized_count, factorized_clamped = \
            factorized
        exact_pred, _cost, exact_count, exact_clamped = exact
        if args.plot_dir and plotted < args.plot_limit:
            token = str(info.get('token', plotted))
            plot_sample(
                os.path.join(
                    args.plot_dir, f'{plotted:03d}_{token}.png'),
                paths, info.get('ego2global', np.eye(4)), gt, valid,
                factorized_pred, exact_pred,
                f'command={command} bucket={bucket}', required_length)
            plotted += 1
        for agg in (factorized_all, factorized_bucket[bucket],
                    factorized_command[command]):
            add_metric(
                agg, factorized_pred, gt, valid, len(paths),
                factorized_count, factorized_clamped)
        for agg in (exact_all, exact_bucket[bucket], exact_command[command]):
            add_metric(
                agg, exact_pred, gt, valid, len(paths), exact_count,
                exact_clamped)

    bucket_order = (
        'static', 'slow', 'moving_straight', 'turning', 'unknown')
    factorized_rows = [('ALL', summarize(factorized_all))]
    exact_rows = [('ALL', summarize(exact_all))]
    for name in bucket_order:
        if factorized_bucket[name]['n']:
            factorized_rows.append((name, summarize(factorized_bucket[name])))
            exact_rows.append((name, summarize(exact_bucket[name])))
    command_rows = []
    exact_command_rows = []
    for command in sorted(factorized_command):
        command_rows.append((
            f'command={command}', summarize(factorized_command[command])))
        exact_command_rows.append((
            f'command={command}', summarize(exact_command[command])))

    print('D0 topology multimodal oracle')
    print(
        f'samples={len(infos)} used={int(factorized_all["n"])} '
        f'no_plan={no_plan} no_path={no_path} '
        f'heading_fallback={heading_fallback}')
    print(
        f'profiles={len(profiles)} max_paths={args.max_paths} '
        f'required_length={required_length:.2f}m')
    print_rows('\nDeployable speed-profile oracle:', factorized_rows)
    print_rows('\nExact-speed geometry oracle:', exact_rows)
    print_rows('\nDeployable oracle by command:', command_rows)
    print_rows('\nExact-speed oracle by command:', exact_command_rows)

    result = dict(
        config=vars(args),
        counts=dict(
            samples=len(infos), used=int(factorized_all['n']),
            no_plan=no_plan, no_path=no_path,
            heading_fallback=heading_fallback),
        factorized={name: row for name, row in factorized_rows},
        exact_speed={name: row for name, row in exact_rows},
        factorized_by_command={name: row for name, row in command_rows},
        exact_speed_by_command={
            name: row for name, row in exact_command_rows},
    )
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f'\nwrote {args.output_json}')


if __name__ == '__main__':
    main()
