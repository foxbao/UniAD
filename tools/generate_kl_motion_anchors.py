"""Generate KL-dataset motion anchors via K-means on gt_fut_traj_locs.

Output format matches motion_anchor_infos_mode6.pkl:
  {
    'grouped_classes': list[list[str]],
    'class_list':      list[list[int]],
    'K_mode':          int,
    'anchors_all':     list[np.ndarray],  # one per group, shape (K, steps, 2)
  }

Usage:
  python tools/generate_kl_motion_anchors.py \
      --info data/kl_8/kl_infos_train.pkl \
      --out  data/others/motion_anchor_infos_kl.pkl \
      --k 6 --steps 12
"""

import argparse
import pickle
import numpy as np
from sklearn.cluster import KMeans

CLASS_NAMES = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
]

# group definitions: (group_name, class_ids)
GROUPS = [
    ('pedestrian',  [0]),
    ('vehicle',     [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]),
]


def collect_trajs(data_list, steps):
    """Return dict[group_idx -> list of (steps, 2) arrays].

    The trajectories are transformed from the LiDAR frame (where
    gt_fut_traj_locs lives) into the *object-local* frame, because
    UniAD's `anchor_coordinate_transform` rotates anchors by `yaw - pi`
    at runtime. Anchors must therefore be authored in the object-local
    convention used by nuScenes (heading aligned with +y after that rot).
    """
    group_trajs = {i: [] for i in range(len(GROUPS))}
    cls2group = {}
    for g_idx, (_, ids) in enumerate(GROUPS):
        for cid in ids:
            cls2group[cid] = g_idx

    for item in data_list:
        for inst in item.get('instances', []):
            label = inst.get('bbox_label_3d', inst.get('bbox_label', -1))
            if label not in cls2group:
                continue
            locs = inst.get('gt_fut_traj_locs',
                            inst.get('gt_forecasting_locs'))
            mask = inst.get('gt_fut_traj_mask',
                            inst.get('gt_forecasting_mask'))
            bbox = inst.get('bbox_3d')
            if locs is None or mask is None or bbox is None:
                continue
            locs = np.array(locs, dtype=np.float32)   # (T, 2) in LiDAR frame
            mask = np.array(mask, dtype=bool)
            if locs.shape[0] < steps or mask[:steps].sum() < steps:
                continue

            # rotate ego-frame deltas into the agent frame (forward = +y).
            # nuScenes fut_traj is already in agent frame; for KL data
            # we must convert from ego frame using R(pi/2 - yaw):
            # R(-yaw) aligns ego axes to object heading (+x = forward),
            # then R(pi/2) rotates +x to +y.
            yaw = float(bbox[6])
            angle = np.pi / 2.0 - yaw
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            local = locs[:steps] @ R.T
            group_trajs[cls2group[label]].append(local)

    return group_trajs


def kmeans_anchors(trajs, k, steps, lateral_thresh=1.0):
    """Fit K-means with stratified sampling to preserve turning modes.

    Splits trajectories into straight, left-turn, and right-turn groups,
    then allocates K-means clusters to each. This prevents left/right
    turns from cancelling each other out in the cluster centers.
    """
    if len(trajs) == 0:
        print('  WARNING: no trajectories, using zero anchors')
        return np.zeros((k, steps, 2), dtype=np.float32)

    X = np.stack(trajs)  # (N, steps, 2)
    lateral_end = X[:, -1, 0]
    left_mask = lateral_end < -lateral_thresh
    right_mask = lateral_end > lateral_thresh
    straight_mask = ~left_mask & ~right_mask

    n_left = left_mask.sum()
    n_right = right_mask.sum()
    n_straight = straight_mask.sum()
    n_turn = n_left + n_right
    print(f'    straight: {n_straight}, left: {n_left}, right: {n_right}')

    if n_turn < 4:
        X_flat = X.reshape(len(X), -1)
        km = KMeans(n_clusters=min(k, len(X)), random_state=0, n_init=10)
        km.fit(X_flat)
        centers = km.cluster_centers_.reshape(-1, steps, 2)
        if centers.shape[0] < k:
            pad = np.zeros((k - centers.shape[0], steps, 2))
            centers = np.concatenate([centers, pad], axis=0)
        return centers.astype(np.float32)

    # allocate: at least 1 left + 1 right, rest to straight
    k_left = max(1, round(k * n_left / len(X)))
    k_right = max(1, round(k * n_right / len(X)))
    k_straight = k - k_left - k_right
    if k_straight < 2:
        k_straight = 2
        excess = k_left + k_right - (k - k_straight)
        if k_left >= k_right:
            k_left -= excess
        else:
            k_right -= excess
    k_left = max(1, k_left)
    k_right = max(1, k_right)
    k_straight = k - k_left - k_right

    all_centers = []
    for name, mask, ki in [('straight', straight_mask, k_straight),
                           ('left', left_mask, k_left),
                           ('right', right_mask, k_right)]:
        subset = X[mask].reshape(mask.sum(), -1)
        km = KMeans(n_clusters=min(ki, mask.sum()),
                    random_state=0, n_init=10)
        km.fit(subset)
        all_centers.append(km.cluster_centers_)

    centers = np.concatenate(all_centers, axis=0).reshape(-1, steps, 2)
    if centers.shape[0] < k:
        pad = np.zeros((k - centers.shape[0], steps, 2))
        centers = np.concatenate([centers, pad], axis=0)

    print(f'    allocated: {k_straight} straight + '
          f'{k_left} left + {k_right} right')
    return centers[:k].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--out',  default='data/others/motion_anchor_infos_kl.pkl')
    parser.add_argument('--k',     type=int, default=6)
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()

    print(f'Loading {args.info} ...')
    with open(args.info, 'rb') as f:
        raw = pickle.load(f)
    data_list = raw['data_list'] if isinstance(raw, dict) else raw
    print(f'  {len(data_list)} frames')

    print('Collecting trajectories ...')
    group_trajs = collect_trajs(data_list, args.steps)
    for g_idx, (gname, _) in enumerate(GROUPS):
        print(f'  group {g_idx} ({gname}): {len(group_trajs[g_idx])} trajs')

    print(f'Running K-means (k={args.k}) ...')
    anchors_all = []
    for g_idx, (gname, _) in enumerate(GROUPS):
        centers = kmeans_anchors(group_trajs[g_idx], args.k, args.steps)
        anchors_all.append(centers)
        print(f'  group {g_idx} ({gname}): anchor range '
              f'x=[{centers[...,0].min():.3f}, {centers[...,0].max():.3f}] '
              f'y=[{centers[...,1].min():.3f}, {centers[...,1].max():.3f}]')

    out = {
        'grouped_classes': [[CLASS_NAMES[i] for i in ids] for _, ids in GROUPS],
        'class_list':      [ids for _, ids in GROUPS],
        'K_mode':          args.k,
        'anchors_all':     anchors_all,
    }

    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
