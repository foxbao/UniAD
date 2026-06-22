#!/usr/bin/env python3
"""Generate stratified K=6 KL motion anchors.

Unlike weighted k-means, this script reserves one vehicle-mode slot for each
motion family before averaging/cluster-centering trajectories inside that
family. The output schema matches MotionHead's anchor pkl:
grouped_classes, class_list, K_mode, anchors_all.
"""

import argparse
import pickle

import numpy as np


GROUPED_CLASSES = [
    ['Pedestrian'],
    ['Car', 'IGV-Full', 'Truck', 'Trailer-Empty', 'Trailer-Full',
     'IGV-Empty', 'Crane', 'OtherVehicle', 'ContainerForklift',
     'Forklift', 'WheelCrane'],
    ['Cone'],
]
CLASS_LIST = [[0], [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12], [9]]
MODE_NAMES = [
    'static_slow',
    'straight_short',
    'straight_long',
    'negative_lateral_turn',
    'positive_lateral_turn',
    'maneuver_sharp',
]


def rot_2d(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def wrap_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def heading_change_deg(traj, segment_min_disp=0.05):
    if len(traj) < 3:
        return 0.0
    deltas = np.diff(traj, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = np.where(norms >= segment_min_disp)[0]
    if len(valid) < 2:
        return 0.0
    v0 = deltas[valid[0]]
    v1 = deltas[valid[-1]]
    a0 = np.arctan2(v0[1], v0[0])
    a1 = np.arctan2(v1[1], v1[0])
    return float(abs(np.degrees(wrap_pi(a1 - a0))))


def lateral_ratio(traj):
    if len(traj) < 2:
        return 0.0
    end = traj[-1]
    net = float(np.linalg.norm(end))
    if net < 1e-6:
        return 0.0
    direction = end / net
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    lateral = float(np.max(np.abs(traj @ normal)))
    return lateral / max(net, 1e-6)


def path_length(traj):
    if len(traj) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def collect_group_trajs(data_list, class_ids, steps):
    class_ids = set(class_ids)
    trajs = []
    for sample in data_list:
        for inst in sample.get('instances', []):
            if inst.get('bbox_label_3d') not in class_ids:
                continue
            locs = np.asarray(inst.get('gt_fut_traj_locs'), dtype=np.float64)
            mask = np.asarray(inst.get('gt_fut_traj_mask'), dtype=np.float64)
            if locs.ndim != 2 or locs.shape[0] < steps:
                continue
            mask_step = mask.reshape(mask.shape[0], -1)[:, 0] if mask.ndim else mask
            if mask_step[:steps].sum() < steps:
                continue
            locs = locs[:steps]
            yaw = float(np.asarray(inst['bbox_3d'], dtype=np.float64)[6])
            # Same local convention as the original KL anchors: +Y forward.
            local = (rot_2d(np.pi / 2 - yaw) @ locs.T).T
            trajs.append(local)
    return np.asarray(trajs, dtype=np.float64)


def traj_stats(trajs):
    rows = []
    for traj in trajs:
        path = path_length(traj)
        net = float(np.linalg.norm(traj[-1]))
        hchg = heading_change_deg(traj)
        lat_ratio = lateral_ratio(traj)
        rows.append((path, net, hchg, lat_ratio, float(traj[-1, 0])))
    return np.asarray(rows, dtype=np.float64)


def make_masks(stats, args):
    path, net, hchg, lat_ratio, final_x = stats.T
    static = (path < args.static_path_thr) | (net < args.static_path_thr)
    straight = (
        ~static
        & (hchg <= args.straight_deg)
        & (lat_ratio <= args.straight_lateral_ratio))
    straight_net = net[straight]
    if straight_net.size:
        split = float(np.median(straight_net))
    else:
        split = args.straight_long_split
    straight_short = straight & (net <= split)
    straight_long = straight & (net > split)
    nonstraight = ~static & ~straight
    maneuver = (
        nonstraight
        & ((hchg >= args.sharp_deg)
           | (lat_ratio >= args.maneuver_lateral_ratio)
           | ((path / np.maximum(net, 1e-6)) >= args.maneuver_path_ratio)))
    neg_turn = nonstraight & (final_x < 0)
    pos_turn = nonstraight & (final_x >= 0)

    # If a side-specific mild turn bucket is starved, borrow side-specific
    # sharp turns before falling back to all non-straight trajectories.
    if neg_turn.sum() < args.min_bucket_samples:
        neg_turn = nonstraight & (final_x < 0)
    if pos_turn.sum() < args.min_bucket_samples:
        pos_turn = nonstraight & (final_x >= 0)
    return [
        static,
        straight_short,
        straight_long,
        neg_turn,
        pos_turn,
        maneuver,
    ]


def center_from_mask(trajs, stats, mask, fallback_mask, min_samples,
                     turn_center_weight=0.0,
                     representative='mean'):
    if mask.sum() < min_samples:
        mask = fallback_mask
    if mask.sum() == 0:
        order = np.argsort(np.linalg.norm(trajs[:, -1], axis=1))
        selected = trajs[order[:max(1, min(min_samples, len(order)))]]
        selected_stats = stats[order[:max(1, min(min_samples, len(order)))]]
    else:
        selected = trajs[mask]
        selected_stats = stats[mask]
    if representative == 'sharp_sample' and len(selected) > 0:
        hchg = selected_stats[:, 2]
        cutoff = np.percentile(hchg, 80.0)
        sharp_mask = hchg >= cutoff
        pool = selected[sharp_mask]
        pool_stats = selected_stats[sharp_mask]
        if len(pool) == 0:
            pool = selected
            pool_stats = selected_stats
        net = pool_stats[:, 1]
        path = pool_stats[:, 0]
        score = (
            np.abs(net - np.median(net))
            + 0.25 * np.abs(path - np.median(path))
            - 0.02 * pool_stats[:, 2])
        return pool[int(np.argmin(score))]
    if turn_center_weight > 0 and len(selected) > 0:
        hchg = selected_stats[:, 2]
        weights = 1.0 + float(turn_center_weight) * np.clip(hchg / 90.0, 0, 1)
        return np.average(selected, axis=0, weights=weights)
    return selected.mean(axis=0)


def build_group_anchors(trajs, args):
    if len(trajs) < args.k:
        raise RuntimeError(f'only {len(trajs)} trajectories, need >= {args.k}')
    stats = traj_stats(trajs)
    masks = make_masks(stats, args)
    path, net, hchg, _, _ = stats.T
    static = masks[0]
    straight = masks[1] | masks[2]
    nonstraight = ~static & ~straight
    sharp = nonstraight & (hchg >= args.sharp_deg)
    moving = ~static
    fallbacks = [
        static,
        straight if straight.any() else moving,
        straight if straight.any() else moving,
        nonstraight if nonstraight.any() else moving,
        nonstraight if nonstraight.any() else moving,
        sharp if sharp.any() else nonstraight if nonstraight.any() else moving,
    ]
    centers = []
    for mode_idx, (mask, fallback) in enumerate(zip(masks, fallbacks)):
        turn_weight = args.turn_center_weight if mode_idx >= 3 else 0.0
        representative = 'sharp_sample' if mode_idx == 5 else 'mean'
        centers.append(
            center_from_mask(trajs, stats, mask, fallback,
                             args.min_bucket_samples, turn_weight,
                             representative=representative))
    return np.asarray(centers, dtype=np.float32), masks, stats


def build_static_only_anchors(trajs, args):
    stats = traj_stats(trajs)
    path, net = stats[:, 0], stats[:, 1]
    static = (path < args.static_path_thr) | (net < args.static_path_thr)
    center = center_from_mask(trajs, stats, static, static,
                              args.min_bucket_samples)
    anchors = np.repeat(center[None, ...], args.k, axis=0).astype(np.float32)
    return anchors, [static for _ in range(args.k)], stats


def report(group_name, anchors, masks, stats):
    print(f'--- {group_name}: {anchors.shape} ---')
    for name, mask in zip(MODE_NAMES, masks):
        print(f'  {name:22s}: {int(mask.sum())} samples')
    for idx, (name, traj) in enumerate(zip(MODE_NAMES, anchors)):
        end = traj[-1]
        disp = float(np.linalg.norm(end))
        hchg = heading_change_deg(traj)
        print(f'  mode{idx} {name:22s}: '
              f'end=({end[0]:+6.1f},{end[1]:+6.1f}) '
              f'disp={disp:5.1f}m heading_chg={hchg:5.1f}deg')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate stratified K=6 KL motion anchors.')
    parser.add_argument('--info', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--out',
                        default='data/others/motion_anchor_infos_kl_stratified_k6_3grp.pkl')
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--static-path-thr', type=float, default=2.0)
    parser.add_argument('--straight-deg', type=float, default=15.0)
    parser.add_argument('--straight-lateral-ratio', type=float, default=0.15)
    parser.add_argument('--straight-long-split', type=float, default=15.0)
    parser.add_argument('--sharp-deg', type=float, default=60.0)
    parser.add_argument('--maneuver-lateral-ratio', type=float, default=0.35)
    parser.add_argument('--maneuver-path-ratio', type=float, default=1.25)
    parser.add_argument('--turn-center-weight', type=float, default=8.0,
                        help='extra center weight for high-heading-change turn samples')
    parser.add_argument('--min-bucket-samples', type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.k != 6:
        raise ValueError('This stratified layout is defined for k=6.')
    print(f'Loading {args.info} ...')
    with open(args.info, 'rb') as f:
        data = pickle.load(f)
    data_list = data.get('data_list', data.get('infos', data))
    print(f'  {len(data_list)} samples')

    anchors_all = []
    for group_idx, class_ids in enumerate(CLASS_LIST):
        trajs = collect_group_trajs(data_list, class_ids, args.steps)
        if class_ids == [9]:
            anchors, masks, stats = build_static_only_anchors(trajs, args)
        else:
            anchors, masks, stats = build_group_anchors(trajs, args)
        report(f'group {group_idx} ({GROUPED_CLASSES[group_idx][0]}...)',
               anchors, masks, stats)
        anchors_all.append(anchors)

    payload = dict(
        grouped_classes=GROUPED_CLASSES,
        class_list=CLASS_LIST,
        K_mode=args.k,
        mode_names=MODE_NAMES,
        anchors_all=anchors_all)
    with open(args.out, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
