"""Generate UniAD motion anchors from nuScenes trajectories.

The output matches the format consumed by ``MotionHead``:

    {
        "grouped_classes": list[list[str]],
        "class_list": list[list[int]],
        "K_mode": int,
        "anchors_all": list[np.ndarray],  # (K, steps, 2), one array per group
    }

By default this script reads UniAD's generated temporal train info file, whose
``fut_traj`` field is already in the nuScenes agent frame. It can also read raw
nuScenes annotations through the nuScenes SDK.

Examples:
    python tools/generate_nuscenes_motion_anchors.py \
        --info data/infos/nuscenes_infos_temporal_train.pkl \
        --out data/others/motion_anchor_infos_mode6_generated.pkl

    python tools/generate_nuscenes_motion_anchors.py \
        --dataroot data/nuscenes --version v1.0-trainval --split train \
        --out data/others/motion_anchor_infos_mode6_generated.pkl
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional

import mmcv
import numpy as np
from sklearn.cluster import KMeans


CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

GROUP_ID_LIST = [[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]]

NUSCENES_NAME_MAPPING = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def build_class_to_group() -> Dict[str, int]:
    class_to_group = {}
    for group_idx, class_ids in enumerate(GROUP_ID_LIST):
        for class_id in class_ids:
            class_to_group[CLASS_NAMES[class_id]] = group_idx
    return class_to_group


def _load_info_records(info_path: str) -> List[dict]:
    raw = mmcv.load(info_path)
    if isinstance(raw, dict):
        if "infos" in raw:
            return raw["infos"]
        if "data_list" in raw:
            return raw["data_list"]
    if isinstance(raw, list):
        return raw
    raise TypeError(f"Unsupported info format in {info_path}")


def collect_from_info(
    info_path: str,
    steps: int,
    min_valid_steps: int,
    use_valid_flag: bool = True,
    max_samples: Optional[int] = None,
) -> DefaultDict[int, List[np.ndarray]]:
    """Collect agent-frame future trajectories from UniAD temporal infos."""
    class_to_group = build_class_to_group()
    group_trajs: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    records = _load_info_records(info_path)
    if max_samples is not None:
        records = records[:max_samples]

    for info in mmcv.track_iter_progress(records):
        names = np.asarray(info.get("gt_names", []))
        fut_traj = np.asarray(info.get("fut_traj", []), dtype=np.float64)
        fut_mask = np.asarray(info.get("fut_traj_valid_mask", []))
        valid_flag = np.asarray(
            info.get("valid_flag", np.ones(len(names), dtype=bool)), dtype=bool)

        if len(names) == 0 or fut_traj.ndim != 3 or fut_mask.ndim != 3:
            continue

        n = min(len(names), fut_traj.shape[0], fut_mask.shape[0], len(valid_flag))
        for name, traj, mask, is_valid in zip(
                names[:n], fut_traj[:n], fut_mask[:n], valid_flag[:n]):
            if use_valid_flag and not is_valid:
                continue
            group_idx = class_to_group.get(str(name))
            if group_idx is None:
                continue
            traj = np.asarray(traj[:steps], dtype=np.float64)
            mask = np.asarray(mask[:steps])
            if traj.shape != (steps, 2) or mask.shape != (steps, 2):
                continue
            if np.all(mask > 0, axis=-1).sum() < min_valid_steps:
                continue
            group_trajs[group_idx].append(traj)

    return group_trajs


def _iter_train_scene_tokens(split: str, version: str) -> Iterable[str]:
    from nuscenes.utils import splits

    if version == "v1.0-mini":
        scene_names = splits.mini_train if split == "train" else splits.mini_val
    elif split == "train":
        scene_names = splits.train
    elif split == "val":
        scene_names = splits.val
    elif split == "trainval":
        scene_names = splits.train + splits.val
    else:
        raise ValueError("--split must be train, val, or trainval")
    return scene_names


def collect_from_sdk(
    dataroot: str,
    version: str,
    split: str,
    steps: int,
    seconds: float,
    min_valid_steps: int,
    use_valid_flag: bool = True,
    max_samples: Optional[int] = None,
) -> DefaultDict[int, List[np.ndarray]]:
    """Collect agent-frame future trajectories directly from nuScenes."""
    from nuscenes import NuScenes
    from nuscenes.prediction import PredictHelper

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    predict_helper = PredictHelper(nusc)
    class_to_group = build_class_to_group()
    group_trajs: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    scene_names = set(_iter_train_scene_tokens(split, version))
    scene_tokens = {
        scene["token"]
        for scene in nusc.scene
        if scene["name"] in scene_names
    }

    records = [
        sample for sample in nusc.sample
        if sample["scene_token"] in scene_tokens
    ]
    if max_samples is not None:
        records = records[:max_samples]

    for sample in mmcv.track_iter_progress(records):
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            if use_valid_flag and (ann["num_lidar_pts"] + ann["num_radar_pts"]) <= 0:
                continue
            class_name = NUSCENES_NAME_MAPPING.get(ann["category_name"])
            group_idx = class_to_group.get(class_name)
            if group_idx is None:
                continue

            traj = predict_helper.get_future_for_agent(
                ann["instance_token"],
                sample["token"],
                seconds=seconds,
                in_agent_frame=True,
            )
            if traj.shape[0] < min_valid_steps:
                continue
            if traj.shape[0] < steps:
                padded = np.zeros((steps, 2), dtype=np.float64)
                padded[:traj.shape[0]] = traj
                traj = padded
            group_trajs[group_idx].append(np.asarray(traj[:steps], dtype=np.float64))

    return group_trajs


def fit_kmeans(trajs: List[np.ndarray], k: int, steps: int, random_state: int,
               n_init: int) -> np.ndarray:
    if len(trajs) == 0:
        raise RuntimeError("No trajectories collected for one anchor group")

    x = np.stack(trajs, axis=0).reshape(len(trajs), steps * 2)
    n_clusters = min(k, len(trajs))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    kmeans.fit(x)
    anchors = kmeans.cluster_centers_.reshape(n_clusters, steps, 2)

    if n_clusters < k:
        anchors = np.concatenate(
            [anchors, np.zeros((k - n_clusters, steps, 2), dtype=np.float64)],
            axis=0,
        )
    return anchors


def build_anchor_infos(group_trajs: Dict[int, List[np.ndarray]], k: int,
                       steps: int, random_state: int, n_init: int) -> dict:
    anchors_all = []
    for group_idx, class_ids in enumerate(GROUP_ID_LIST):
        trajs = group_trajs.get(group_idx, [])
        print(f"group {group_idx} {class_ids}: {len(trajs)} trajectories")
        anchors = fit_kmeans(trajs, k, steps, random_state, n_init)
        print(
            f"  anchors shape={anchors.shape}, "
            f"x=[{anchors[..., 0].min():.3f}, {anchors[..., 0].max():.3f}], "
            f"y=[{anchors[..., 1].min():.3f}, {anchors[..., 1].max():.3f}]")
        anchors_all.append(anchors)

    return {
        "grouped_classes": [[CLASS_NAMES[i] for i in ids] for ids in GROUP_ID_LIST],
        "class_list": GROUP_ID_LIST,
        "K_mode": k,
        "anchors_all": anchors_all,
    }


def _best_mode_match(reference: np.ndarray, generated: np.ndarray) -> tuple:
    import itertools

    k = reference.shape[0]
    pairwise_rmse = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            pairwise_rmse[i, j] = np.sqrt(np.mean((reference[i] - generated[j]) ** 2))

    best_score = None
    best_perm = None
    for perm in itertools.permutations(range(k)):
        score = sum(pairwise_rmse[i, perm[i]] for i in range(k)) / k
        if best_score is None or score < best_score:
            best_score = score
            best_perm = perm
    matched = np.stack([generated[j] for j in best_perm])
    return best_perm, best_score, matched


def compare_anchor_infos(generated: dict, reference_path: str) -> None:
    with open(reference_path, "rb") as f:
        reference = pickle.load(f)

    print(f"Comparing against {reference_path}")
    print(f"  grouped_classes: {generated['grouped_classes'] == reference['grouped_classes']}")
    print(f"  class_list: {generated['class_list'] == reference['class_list']}")
    print(f"  K_mode: {generated['K_mode'] == reference['K_mode']}")

    for group_idx, (ref, gen) in enumerate(
            zip(reference["anchors_all"], generated["anchors_all"])):
        ref = np.asarray(ref, dtype=np.float64)
        gen = np.asarray(gen, dtype=np.float64)
        perm, mean_rmse, matched = _best_mode_match(ref, gen)
        print(f"  group {group_idx} {reference['grouped_classes'][group_idx]}")
        print(f"    best_perm={perm}")
        print(f"    matched_mean_rmse={mean_rmse:.6f}")
        print(f"    matched_mean_abs={np.mean(np.abs(ref - matched)):.6f}")
        print(f"    matched_max_abs={np.max(np.abs(ref - matched)):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UniAD nuScenes motion anchor infos.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--info",
        help="UniAD temporal train info pkl, e.g. data/infos/nuscenes_infos_temporal_train.pkl.",
    )
    source.add_argument(
        "--dataroot",
        help="Raw nuScenes data root. Requires --version and uses nuScenes SDK.",
    )
    parser.add_argument(
        "--out",
        default="data/others/motion_anchor_infos_mode6_generated.pkl",
        help="Output pkl path.",
    )
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="train", choices=["train", "val", "trainval"])
    parser.add_argument("--k", type=int, default=6, help="Number of modes.")
    parser.add_argument("--steps", type=int, default=12, help="Future trajectory steps.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=6.0,
        help="Future seconds when reading raw nuScenes SDK.",
    )
    parser.add_argument(
        "--min-valid-steps",
        type=int,
        default=None,
        help="Minimum valid future steps. Defaults to --steps.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include boxes with no lidar/radar points. Default matches UniAD training filtering.",
    )
    parser.add_argument(
        "--compare-to",
        help="Optional reference pkl for a best-permutation numerical comparison.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug option: limit samples before collecting trajectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_valid_steps = args.steps if args.min_valid_steps is None else args.min_valid_steps

    if args.info:
        group_trajs = collect_from_info(
            args.info,
            steps=args.steps,
            min_valid_steps=min_valid_steps,
            use_valid_flag=not args.include_invalid,
            max_samples=args.max_samples,
        )
    else:
        group_trajs = collect_from_sdk(
            args.dataroot,
            version=args.version,
            split=args.split,
            steps=args.steps,
            seconds=args.seconds,
            min_valid_steps=min_valid_steps,
            use_valid_flag=not args.include_invalid,
            max_samples=args.max_samples,
        )

    anchor_infos = build_anchor_infos(
        group_trajs,
        k=args.k,
        steps=args.steps,
        random_state=args.random_state,
        n_init=args.n_init,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(anchor_infos, f)
    print(f"Saved {out_path}")
    if args.compare_to:
        compare_anchor_infos(anchor_infos, args.compare_to)


if __name__ == "__main__":
    main()
