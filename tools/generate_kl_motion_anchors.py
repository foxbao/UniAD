#!/usr/bin/env python3
"""Generate turn-aware motion anchors for the KL LiDAR MotionHead.

The original `motion_anchor_infos_kl.pkl` was produced by plain k-means over
ground-truth future trajectories.  Because straight-driving samples dominate the
KL dataset, all 6 vehicle anchors collapsed onto straight / slight-lane-change
templates (max |heading change| ~13 deg), so the MotionHead has no anchor that
represents a real turn and predicts turning vehicles poorly -- even though 21.7%
of moving vehicles in the training set are real turns (>45 deg).

This script regenerates the anchors with the SAME schema and frame convention,
but up-weights turning samples in the k-means so cluster centers must cover
turns.  K stays 6 (the model's num_anchor is unchanged).

Frame convention (must match the original pkl, verified empirically):
  - Anchors are agent-local cumulative xy offsets with +Y = forward.
  - The original anchors are reproduced by rotating the ground-frame GT
    offsets by rot(pi/2 - yaw) into the anchor-local frame (verified: this
    maps straight movers onto +Y, matching the original straight anchors at
    (0, +disp); rot(pi - yaw) is off by 90deg and does NOT match).

Output pkl keys mirror the original: grouped_classes, class_list, K_mode,
anchors_all (list of (K, steps, 2) float32 arrays, one per group).
"""
import argparse
import pickle

import numpy as np
from sklearn.cluster import KMeans


# Mirrors the original pkl (do not change without retraining: the head maps
# class id -> group via these lists).
GROUPED_CLASSES = [
    ['Pedestrian'],
    ['Car', 'IGV-Full', 'Truck', 'Trailer-Empty', 'Trailer-Full',
     'IGV-Empty', 'Crane', 'OtherVehicle', 'ContainerForklift',
     'Forklift', 'WheelCrane'],
]
CLASS_LIST = [[0], [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]]


def rot_2d(angle):
    """2D rotation matrix matching functional.rot_2d (CCW)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def heading_change_deg(traj):
    """Absolute heading change between the first and last segment (degrees)."""
    if traj.shape[0] < 3:
        return 0.0
    v0 = traj[1] - traj[0]
    v1 = traj[-1] - traj[-2]
    if np.linalg.norm(v0) < 1e-3 or np.linalg.norm(v1) < 1e-3:
        return 0.0
    ang = np.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0]))
    return abs((ang + 180.0) % 360.0 - 180.0)


def turn_weight(hc_deg, w_turn):
    """Ramp from 1 (<=30 deg) to w_turn (>=90 deg), linear in between."""
    if hc_deg <= 30.0:
        return 1.0
    if hc_deg >= 90.0:
        return float(w_turn)
    frac = (hc_deg - 30.0) / 60.0
    return 1.0 + frac * (float(w_turn) - 1.0)


def collect_group_trajs(data_list, class_ids, steps):
    """Return (trajs (N, steps, 2) local +Y-forward, weights (N,), raw hc)."""
    class_ids = set(class_ids)
    trajs, hcs = [], []
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
                continue  # require a fully-valid horizon for clean anchors
            locs = locs[:steps]
            yaw = float(np.asarray(inst['bbox_3d'], dtype=np.float64)[6])
            # ground relative offsets -> anchor-local (+Y forward).
            # rot(pi/2 - yaw) reproduces the original pkl convention.
            local = (rot_2d(np.pi / 2 - yaw) @ locs.T).T
            trajs.append(local)
            hcs.append(heading_change_deg(local))
    return np.asarray(trajs), np.asarray(hcs)


def weighted_kmeans(trajs, hcs, k, w_turn, seed=0):
    """K-means over flattened trajectories using per-sample turn weights."""
    n, steps, _ = trajs.shape
    flat = trajs.reshape(n, steps * 2)
    weights = np.array([turn_weight(h, w_turn) for h in hcs], dtype=np.float64)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(flat, sample_weight=weights)
    centers = km.cluster_centers_.reshape(k, steps, 2).astype(np.float32)
    # order modes by displacement for readability (does not affect training)
    order = np.argsort(np.linalg.norm(centers[:, -1], axis=1))
    return centers[order]


def report(name, anchors):
    print(f'--- {name}: {anchors.shape} ---')
    n_turn = 0
    for kk, traj in enumerate(anchors):
        end = traj[-1]
        disp = float(np.linalg.norm(end))
        hc = heading_change_deg(traj)
        n_turn += hc > 45.0
        print(f'  mode{kk}: end=({end[0]:+6.1f},{end[1]:+6.1f}) '
              f'disp={disp:5.1f}m heading_chg={hc:+6.1f}deg')
    return n_turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--info', default='data/kl_8/kl_infos_train.pkl')
    ap.add_argument('--out',
                    default='data/others/motion_anchor_infos_kl_turnaware.pkl')
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--turn-weight', type=float, default=8.0,
                    help='max kmeans sample weight for sharp (>=90deg) turns')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print(f'Loading {args.info} ...')
    data = pickle.load(open(args.info, 'rb'))
    data_list = data.get('data_list', data.get('infos', data))
    print(f'  {len(data_list)} samples')

    anchors_all = []
    total_turn_modes = 0
    for gi, class_ids in enumerate(CLASS_LIST):
        trajs, hcs = collect_group_trajs(data_list, class_ids, args.steps)
        print(f'group {gi} ({GROUPED_CLASSES[gi][0]}...): {len(trajs)} tracks, '
              f'{(hcs > 45).sum()} turns(>45deg)')
        if len(trajs) < args.k:
            raise RuntimeError(
                f'group {gi}: only {len(trajs)} tracks, need >= k={args.k}')
        anchors = weighted_kmeans(trajs, hcs, args.k, args.turn_weight,
                                  args.seed)
        total_turn_modes += report(f'group {gi} anchors', anchors)
        anchors_all.append(anchors)

    if total_turn_modes == 0:
        raise RuntimeError(
            'No anchor mode exceeds 45deg heading change after turn-weighting; '
            'increase --turn-weight. The whole point is to cover turns.')

    out = dict(grouped_classes=GROUPED_CLASSES, class_list=CLASS_LIST,
               K_mode=args.k, anchors_all=anchors_all)
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print(f'Wrote {args.out} ({total_turn_modes} turn modes total)')


if __name__ == '__main__':
    main()
