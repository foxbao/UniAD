#!/usr/bin/env python3
"""Cluster deployable ego speed profiles from planning training labels."""

import argparse
import json
import math
import pickle
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans


BUCKETS = ('static', 'slow', 'moving_straight', 'turning')


def load_infos(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data.get('data_list', data.get('infos', []))
    return data


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def motion_bucket(traj):
    final_disp = float(np.linalg.norm(traj[-1, :2]))
    if final_disp < 0.5:
        return 'static'
    if final_disp < 2.0:
        return 'slow'

    xy = np.concatenate(
        [np.zeros((1, 2), dtype=np.float64), traj[:, :2]], axis=0)
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

    end = traj[-1, :2]
    net = float(np.linalg.norm(end))
    lateral_ratio = 0.0
    if net >= 1e-6:
        normal = np.array([-end[1] / net, end[0] / net])
        lateral_ratio = float(
            np.max(np.abs(traj[:, :2] @ normal)) / net)
    if heading_change >= 15.0 or lateral_ratio >= 0.15:
        return 'turning'
    return 'moving_straight'


def cumulative_distance(traj):
    xy = np.concatenate(
        [np.zeros((1, 2), dtype=np.float64), traj[:, :2]], axis=0)
    return np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))


def collect_profiles(infos, steps):
    profiles = []
    buckets = []
    commands = []
    for info in infos:
        traj = np.asarray(info.get('sdc_planning'))
        mask = np.asarray(info.get('sdc_planning_mask'))
        if traj.size == 0 or mask.size == 0:
            continue
        traj = traj.reshape(-1, traj.shape[-2], traj.shape[-1])[0]
        mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
        valid = mask.any(axis=-1)
        if len(traj) < steps or not valid[:steps].all():
            continue
        traj = traj[:steps]
        profiles.append(cumulative_distance(traj))
        buckets.append(motion_bucket(traj))
        command = np.asarray(info.get('command', [-1])).reshape(-1)
        commands.append(int(command[0]) if len(command) else -1)
    return (np.asarray(profiles, dtype=np.float32),
            np.asarray(buckets), np.asarray(commands, dtype=np.int64))


def balanced_weights(buckets):
    counts = Counter(buckets.tolist())
    weights = np.asarray(
        [1.0 / counts[bucket] for bucket in buckets], dtype=np.float64)
    return weights / weights.mean(), counts


def assign_profiles(samples, centers):
    dist = np.square(samples[:, None] - centers[None]).mean(axis=-1)
    return dist.argmin(axis=1), np.sqrt(dist.min(axis=1))


def main():
    parser = argparse.ArgumentParser(
        description='Cluster cumulative ego distance profiles for planning.')
    parser.add_argument('--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument(
        '--output',
        default='data/others/planning_speed_profiles_d0_8.npz')
    parser.add_argument('--num-profiles', type=int, default=8)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--random-state', type=int, default=0)
    args = parser.parse_args()
    if args.num_profiles < 2:
        parser.error('--num-profiles must be at least 2')

    samples, buckets, commands = collect_profiles(
        load_infos(args.ann_file), args.steps)
    non_static = buckets != 'static'
    fit_samples = samples[non_static]
    fit_buckets = buckets[non_static]
    if len(fit_samples) < args.num_profiles - 1:
        raise RuntimeError('Not enough valid non-static planning samples')

    weights, bucket_counts = balanced_weights(fit_buckets)
    model = KMeans(
        n_clusters=args.num_profiles - 1,
        random_state=args.random_state,
        n_init=20)
    model.fit(fit_samples, sample_weight=weights)
    profiles = np.concatenate([
        np.zeros((1, args.steps), dtype=np.float32),
        model.cluster_centers_.astype(np.float32),
    ], axis=0)
    profiles = np.maximum.accumulate(profiles, axis=1)
    profiles = profiles[np.argsort(profiles[:, -1])]

    assignment, error = assign_profiles(samples, profiles)
    assignment_counts = np.bincount(
        assignment, minlength=len(profiles)).astype(np.int64)
    metadata = dict(
        ann_file=args.ann_file,
        steps=args.steps,
        num_samples=int(len(samples)),
        bucket_counts=dict(Counter(buckets.tolist())),
        fit_bucket_counts=dict(bucket_counts),
        command_counts={
            str(key): int(value)
            for key, value in Counter(commands.tolist()).items()
        },
        mean_assignment_rmse=float(error.mean()),
        p90_assignment_rmse=float(np.percentile(error, 90)),
    )
    np.savez(
        args.output,
        profiles=profiles,
        assignment_counts=assignment_counts,
        metadata=json.dumps(metadata, sort_keys=True))

    print(f'samples={len(samples)} output={args.output}')
    print(f'buckets={metadata["bucket_counts"]}')
    print(
        'assignment_rmse='
        f'{metadata["mean_assignment_rmse"]:.4f} '
        f'p90={metadata["p90_assignment_rmse"]:.4f}')
    for index, (profile, count) in enumerate(
            zip(profiles, assignment_counts)):
        values = ' '.join(f'{value:.3f}' for value in profile)
        print(
            f'profile[{index}] count={int(count):5d} '
            f'final={profile[-1]:.3f}: {values}')


if __name__ == '__main__':
    main()
