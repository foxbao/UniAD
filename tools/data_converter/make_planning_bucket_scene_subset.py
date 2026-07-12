#!/usr/bin/env python
"""Build a deterministic scene-complete subset with planning bucket quotas."""

import argparse
import math
import pickle
from collections import Counter, defaultdict

import numpy as np

from make_scene_subset import _records


BUCKETS = ('static', 'slow', 'moving_straight', 'turning')


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def planning_bucket(record):
    traj = np.asarray(record['sdc_planning'])[0]
    valid = np.asarray(record['sdc_planning_mask'])[0].any(axis=-1)
    points = traj[valid]
    if len(points) == 0:
        return 'unknown'

    final_disp = float(np.linalg.norm(points[-1, :2]))
    if final_disp < 0.5:
        return 'static'
    if final_disp < 2.0:
        return 'slow'

    xy = np.concatenate(
        [np.zeros((1, 2), dtype=np.float64), points[:, :2]], axis=0)
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
    yaw_change = abs(math.degrees(wrap_pi(
        float(points[-1, 2] - points[0, 2]))))
    end = points[-1, :2]
    net = float(np.linalg.norm(end))
    lateral_ratio = 0.0
    if net >= 1e-6:
        normal = np.array([-end[1] / net, end[0] / net])
        lateral_ratio = float(
            np.max(np.abs(points[:, :2] @ normal)) / net)
    if max(heading_change, yaw_change) >= 15.0 or lateral_ratio >= 0.15:
        return 'turning'
    return 'moving_straight'


def make_subset(input_path, output_path, min_samples, targets):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    records, key = _records(data)

    scene_records = defaultdict(list)
    scene_buckets = defaultdict(Counter)
    scene_order = []
    full_buckets = Counter()
    for record in records:
        scene = record.get('scene_token')
        if not scene:
            raise ValueError('Every record must have a non-empty scene_token')
        if scene not in scene_records:
            scene_order.append(scene)
        bucket = planning_bucket(record)
        scene_records[scene].append(record)
        scene_buckets[scene][bucket] += 1
        full_buckets[bucket] += 1

    selected = []
    selected_set = set()
    selected_buckets = Counter()
    selected_samples = 0

    while True:
        remaining = {
            bucket: max(0, targets[bucket] - selected_buckets[bucket])
            for bucket in BUCKETS
        }
        if selected_samples >= min_samples and not any(remaining.values()):
            break
        best_scene = None
        best_score = -1.0
        for scene in scene_order:
            if scene in selected_set:
                continue
            counts = scene_buckets[scene]
            quota_gain = sum(
                min(counts[bucket], remaining[bucket])
                / max(targets[bucket], 1)
                for bucket in BUCKETS)
            if any(remaining.values()):
                score = quota_gain / math.sqrt(len(scene_records[scene]))
            else:
                score = 1.0 / math.sqrt(len(scene_records[scene]))
            if score > best_score:
                best_score = score
                best_scene = scene
        if best_scene is None:
            raise RuntimeError('Unable to satisfy subset constraints')
        selected.append(best_scene)
        selected_set.add(best_scene)
        selected_samples += len(scene_records[best_scene])
        selected_buckets.update(scene_buckets[best_scene])

    subset_records = [
        record for record in records
        if record['scene_token'] in selected_set
    ]
    if isinstance(data, dict):
        subset = dict(data)
        subset[key] = subset_records
    else:
        subset = subset_records
    with open(output_path, 'wb') as f:
        pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f'total={len(records)} selected_scenes={len(selected)} '
        f'selected_samples={len(subset_records)} output={output_path}')
    print(f'full_buckets={dict(full_buckets)}')
    print(f'selected_buckets={dict(selected_buckets)}')


def main():
    parser = argparse.ArgumentParser(
        description='Create a scene-complete planning-bucket subset.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-samples', type=int, required=True)
    parser.add_argument('--target-static', type=int, default=200)
    parser.add_argument('--target-slow', type=int, default=350)
    parser.add_argument('--target-moving', type=int, default=1000)
    parser.add_argument('--target-turning', type=int, default=150)
    args = parser.parse_args()
    targets = dict(
        static=args.target_static,
        slow=args.target_slow,
        moving_straight=args.target_moving,
        turning=args.target_turning)
    if args.min_samples <= 0 or any(value < 0 for value in targets.values()):
        parser.error('sample count and bucket targets must be non-negative')
    make_subset(args.input, args.output, args.min_samples, targets)


if __name__ == '__main__':
    main()
