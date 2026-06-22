#!/usr/bin/env python
"""Compute turn-bucket motion metrics from UniAD KL eval result pkl."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import os.path as osp
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import mmcv
import numpy as np
from mmcv import Config

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from third_party.uniad_mmdet3d.datasets.builder import build_dataset


DEFAULT_CORE_CLASSES = (
    'Truck', 'Trailer-Empty', 'Trailer-Full', 'IGV-Empty', 'IGV-Full',
    'ContainerForklift', 'Forklift', 'Crane',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Turn-bucket motion metrics for KL UniAD outputs.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--results', required=True,
                        help='Eval output pkl produced by tools/uniad_dist_eval.sh --out.')
    parser.add_argument('--out-prefix', required=True)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--dist-thr', type=float, default=2.0,
                        help='Center distance threshold for matching GT/pred boxes.')
    parser.add_argument('--miss-thr', type=float, default=2.0,
                        help='FDE threshold for MR, matching KlDataset._evaluate_motion.')
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--static-path-thr', type=float, default=2.0)
    parser.add_argument('--straight-deg', type=float, default=15.0)
    parser.add_argument('--mild-deg', type=float, default=45.0)
    parser.add_argument('--straight-lateral-ratio', type=float, default=0.15)
    parser.add_argument('--mild-lateral-ratio', type=float, default=0.35)
    parser.add_argument('--core-classes', nargs='*', default=list(DEFAULT_CORE_CLASSES))
    parser.add_argument('--failure-out', default=None,
                        help='Optional failure-mining CSV path. Defaults to <out-prefix>_failures.csv.')
    parser.add_argument('--failure-topk', type=int, default=50,
                        help='Write top-K rows per failure type; set 0 to disable.')
    parser.add_argument('--failure-fde-thr', type=float, default=5.0)
    parser.add_argument('--mode-gap-thr', type=float, default=2.0,
                        help='top1FDE-minFDE gap threshold for wrong_mode failures.')
    parser.add_argument('--failure-buckets', nargs='*',
                        default=['mild_turn', 'sharp_turn'],
                        help='Buckets included in failure mining. Empty list means all buckets.')
    return parser.parse_args()


def import_cfg_modules(cfg: Config, config_path: str) -> None:
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**custom_imports)
    if cfg.get('plugin', False):
        module_dir = osp.dirname(cfg.get('plugin_dir', osp.dirname(config_path)))
        module_path = module_dir.replace('/', '.').strip('.')
        if module_path:
            importlib.import_module(module_path)


def tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'tensor'):
        return value.tensor.detach().cpu().numpy()
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def unwrap_result(result: dict) -> dict:
    if isinstance(result, dict) and 'pts_bbox' in result:
        return result['pts_bbox']
    return result


def motion_valid_mask(mask) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = np.all(mask > 0, axis=-1)
    else:
        mask = mask > 0
    return mask.astype(np.bool_)


def traj_error_detail(pred_traj,
                      gt_traj,
                      gt_mask,
                      miss_threshold: float,
                      mode_index: Optional[int] = None) -> Optional[dict]:
    pred_traj = np.asarray(pred_traj, dtype=np.float64)[..., :2]
    gt_traj = np.asarray(gt_traj, dtype=np.float64)[..., :2]
    gt_mask = motion_valid_mask(gt_mask)
    if pred_traj.ndim == 2:
        pred_traj = pred_traj[None]
    if pred_traj.ndim != 3 or gt_traj.ndim != 2:
        return None
    if mode_index is not None:
        if mode_index < 0 or mode_index >= pred_traj.shape[0]:
            return None
        pred_traj = pred_traj[mode_index:mode_index + 1]

    steps = min(pred_traj.shape[1], gt_traj.shape[0], gt_mask.shape[0])
    if steps <= 0:
        return None
    pred_traj = np.nan_to_num(pred_traj[:, :steps], nan=0.0,
                              posinf=0.0, neginf=0.0)
    gt_traj = np.nan_to_num(gt_traj[:steps], nan=0.0,
                            posinf=0.0, neginf=0.0)
    valid = gt_mask[:steps]
    if not np.any(valid):
        return None

    dists = np.linalg.norm(pred_traj[:, valid] - gt_traj[None, valid], axis=-1)
    ade = dists.mean(axis=-1)
    final_step = np.where(valid)[0][-1]
    fde = np.linalg.norm(pred_traj[:, final_step] - gt_traj[final_step], axis=-1)
    min_ade_mode = int(np.argmin(ade))
    min_fde_mode = int(np.argmin(fde))
    min_ade = float(ade[min_ade_mode])
    min_fde = float(fde[min_fde_mode])
    return dict(
        minADE=min_ade,
        minFDE=min_fde,
        MR=float(min_fde > miss_threshold),
        oracle_ade_mode=min_ade_mode if mode_index is None else mode_index,
        oracle_fde_mode=min_fde_mode if mode_index is None else mode_index)


def traj_errors(pred_traj,
                gt_traj,
                gt_mask,
                miss_threshold: float,
                mode_index: Optional[int] = None) -> Optional[Tuple[float, float, float]]:
    detail = traj_error_detail(pred_traj, gt_traj, gt_mask, miss_threshold,
                               mode_index=mode_index)
    if detail is None:
        return None
    return detail['minADE'], detail['minFDE'], detail['MR']


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def valid_points(traj, mask) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float64)[..., :2]
    mask = motion_valid_mask(mask)
    steps = min(len(traj), len(mask))
    if steps <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    return traj[:steps][mask[:steps]]


def heading_change_deg(points: np.ndarray,
                       segment_min_disp: float = 0.05) -> float:
    if len(points) < 3:
        return 0.0
    deltas = np.diff(points, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = np.where(norms >= segment_min_disp)[0]
    if len(valid) < 2:
        return 0.0
    v0 = deltas[valid[0]]
    v1 = deltas[valid[-1]]
    a0 = math.atan2(float(v0[1]), float(v0[0]))
    a1 = math.atan2(float(v1[1]), float(v1[0]))
    return abs(math.degrees(wrap_pi(a1 - a0)))


def lateral_ratio(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    end = points[-1]
    net = float(np.linalg.norm(end))
    if net < 1e-6:
        return 0.0
    direction = end / net
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    lateral = float(np.max(np.abs(points @ normal)))
    return lateral / max(net, 1e-6)


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def classify_bucket(points: np.ndarray, args: argparse.Namespace) -> Tuple[str, Dict[str, float]]:
    if len(points) < 2:
        return 'invalid', dict(path=0.0, net=0.0, heading_change=0.0,
                               lateral_ratio=0.0)
    path = path_length(points)
    net = float(np.linalg.norm(points[-1]))
    hchg = heading_change_deg(points)
    lat_ratio = lateral_ratio(points)
    stats = dict(path=path, net=net, heading_change=hchg,
                 lateral_ratio=lat_ratio)
    if path < args.static_path_thr or net < args.static_path_thr:
        return 'static_slow', stats
    if hchg <= args.straight_deg and lat_ratio <= args.straight_lateral_ratio:
        return 'straight', stats
    if hchg <= args.mild_deg and lat_ratio <= args.mild_lateral_ratio:
        return 'mild_turn', stats
    return 'sharp_turn', stats


def mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float('nan')


def empty_stats():
    return dict(
        gt=0,
        matched=0,
        min_ade=[],
        min_fde=[],
        mr=[],
        top1_ade=[],
        top1_fde=[],
        top1_mr=[],
        heading_change=[],
        path=[],
        net=[],
        lateral_ratio=[],
    )


def update_gt_stats(stats: dict, bucket_info: Dict[str, float]) -> None:
    stats['gt'] += 1
    for key in ('heading_change', 'path', 'net', 'lateral_ratio'):
        stats[key].append(bucket_info[key])


def update_match_stats(stats: dict,
                       min_errors: Tuple[float, float, float],
                       top1_errors: Optional[Tuple[float, float, float]]) -> None:
    stats['matched'] += 1
    stats['min_ade'].append(min_errors[0])
    stats['min_fde'].append(min_errors[1])
    stats['mr'].append(min_errors[2])
    if top1_errors is not None:
        stats['top1_ade'].append(top1_errors[0])
        stats['top1_fde'].append(top1_errors[1])
        stats['top1_mr'].append(top1_errors[2])


def summarize(rows_by_key: Dict[Tuple[str, str], dict]) -> List[dict]:
    rows = []
    for (group, bucket), stats in sorted(rows_by_key.items()):
        gt = stats['gt']
        matched = stats['matched']
        rows.append(dict(
            group=group,
            bucket=bucket,
            gt=gt,
            matched=matched,
            recall=(matched / gt if gt else float('nan')),
            minADE=mean_or_nan(stats['min_ade']),
            minFDE=mean_or_nan(stats['min_fde']),
            MR=mean_or_nan(stats['mr']),
            top1_ADE=mean_or_nan(stats['top1_ade']),
            top1_FDE=mean_or_nan(stats['top1_fde']),
            top1_MR=mean_or_nan(stats['top1_mr']),
            avg_heading_change=mean_or_nan(stats['heading_change']),
            avg_path=mean_or_nan(stats['path']),
            avg_net=mean_or_nan(stats['net']),
            avg_lateral_ratio=mean_or_nan(stats['lateral_ratio']),
        ))
    return rows


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    dirname = osp.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[dict], group: str) -> None:
    selected = [r for r in rows if r['group'] == group]
    if not selected:
        return
    headers = [
        'bucket', 'GT', 'Match', 'Recall', 'minADE', 'minFDE', 'MR',
        'top1ADE', 'top1FDE', 'top1MR', 'hchg'
    ]
    table = [headers]
    for r in selected:
        table.append([
            r['bucket'],
            str(r['gt']),
            str(r['matched']),
            fmt(r['recall']),
            fmt(r['minADE']),
            fmt(r['minFDE']),
            fmt(r['MR']),
            fmt(r['top1_ADE']),
            fmt(r['top1_FDE']),
            fmt(r['top1_MR']),
            fmt(r['avg_heading_change']),
        ])
    from terminaltables import AsciiTable
    print(f'\n[{group}]')
    print(AsciiTable(table).table)


def fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return 'nan'
    return f'{value:.4f}'


def raw_info_from_dataset(dataset, index: int) -> Tuple[int, dict]:
    if hasattr(dataset, '_to_raw_index'):
        raw_index = dataset._to_raw_index(index)
    else:
        raw_index = index
    if hasattr(dataset, '_get_raw_info'):
        return raw_index, dataset._get_raw_info(index)
    return raw_index, dataset.data_infos[raw_index]


def failure_type(bucket: str,
                 min_detail: dict,
                 top1_detail: Optional[dict],
                 args: argparse.Namespace) -> Optional[Tuple[str, float]]:
    if min_detail['minFDE'] > args.failure_fde_thr:
        return 'poor_oracle', min_detail['minFDE']
    if top1_detail is not None:
        gap = top1_detail['minFDE'] - min_detail['minFDE']
        if top1_detail['minFDE'] > args.failure_fde_thr and \
                gap > args.mode_gap_thr:
            return 'wrong_mode', gap
        if bucket == 'sharp_turn' and top1_detail['minFDE'] > args.failure_fde_thr:
            return 'sharp_turn_fail', top1_detail['minFDE']
    if bucket == 'sharp_turn' and min_detail['minFDE'] > args.failure_fde_thr:
        return 'sharp_turn_fail', min_detail['minFDE']
    return None


def visualize_command(args: argparse.Namespace, sample_idx: int) -> str:
    return (
        'python tools/analysis_tools/visualize_lidar_e2e_motion.py '
        f'--config {args.config} --checkpoint <checkpoint.pth> '
        f'--out-dir <out_dir> --start-index {sample_idx} --max-frames 1')


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_cfg_modules(cfg, args.config)
    dataset_cfg = cfg.data.val if args.split == 'val' else cfg.data.test
    dataset = build_dataset(dataset_cfg)

    results = mmcv.load(args.results)
    if len(results) != len(dataset):
        raise ValueError(
            f'Result length {len(results)} != dataset length {len(dataset)}. '
            'Use the same config/split that produced the result pkl.')

    class_names = list(dataset.CLASSES)
    eval_range = dataset._active_eval_point_cloud_range()
    overall = defaultdict(empty_stats)
    by_class = defaultdict(empty_stats)
    core_set = set(args.core_classes)
    by_core = defaultdict(empty_stats)
    failure_rows = []
    failure_bucket_set = set(args.failure_buckets or [])

    for sample_idx, raw_result in enumerate(results):
        result = unwrap_result(raw_result)
        raw_idx, raw_info = raw_info_from_dataset(dataset, sample_idx)
        ann_info = dataset.get_ann_info(sample_idx)
        gt_boxes = ann_info['gt_bboxes_3d'].tensor.detach().cpu().numpy()
        gt_labels = np.asarray(ann_info['gt_labels_3d'])
        gt_track_ids = np.asarray(ann_info.get('gt_inds',
                                               np.full((len(gt_boxes),), -1)))
        gt_trajs = np.asarray(ann_info.get('gt_fut_traj', np.zeros((0, 0, 2))),
                              dtype=np.float32)
        gt_masks = np.asarray(ann_info.get('gt_fut_traj_mask',
                                           np.zeros((0, 0, 2))),
                              dtype=np.float32)
        if eval_range is not None:
            range_mask = dataset._box_bev_range_mask(gt_boxes, eval_range)
        else:
            range_mask = np.ones((len(gt_boxes),), dtype=np.bool_)

        buckets = {}
        bucket_meta = {}
        valid_gt = np.zeros((len(gt_boxes),), dtype=np.bool_)
        for gt_idx, (gt_traj, gt_mask) in enumerate(zip(gt_trajs, gt_masks)):
            if not range_mask[gt_idx]:
                continue
            if not motion_valid_mask(gt_mask).any():
                continue
            points = valid_points(gt_traj, gt_mask)
            bucket, meta = classify_bucket(points, args)
            if bucket == 'invalid':
                continue
            buckets[gt_idx] = bucket
            bucket_meta[gt_idx] = meta
            valid_gt[gt_idx] = True
            label = int(gt_labels[gt_idx])
            cls_name = class_names[label]
            update_gt_stats(overall[('Overall', bucket)], meta)
            update_gt_stats(by_class[(cls_name, bucket)], meta)
            if cls_name in core_set:
                update_gt_stats(by_core[('CoreVehicle', bucket)], meta)

        if 'traj' not in result:
            continue
        boxes = result.get('track_boxes_3d', result.get('boxes_3d', None))
        labels = result.get('track_labels_3d', result.get('labels_3d', None))
        scores = result.get('track_scores', result.get('scores_3d', None))
        if boxes is None or labels is None:
            continue
        pred_boxes = tensor_to_numpy(boxes)
        pred_labels = tensor_to_numpy(labels)
        pred_scores = tensor_to_numpy(scores)
        pred_trajs = tensor_to_numpy(result['traj'])
        pred_traj_scores = tensor_to_numpy(result.get('traj_scores', None))
        if pred_trajs is None:
            continue
        if eval_range is not None:
            pred_range_mask = dataset._box_bev_range_mask(pred_boxes, eval_range)
        else:
            pred_range_mask = np.ones((len(pred_boxes),), dtype=np.bool_)

        num_preds = min(len(pred_boxes), len(pred_labels), len(pred_trajs))
        if pred_scores is not None:
            num_preds = min(num_preds, len(pred_scores))

        candidates = []
        nearest_same_class = {
            gt_idx: float('inf') for gt_idx in range(len(gt_labels))
            if valid_gt[gt_idx]
        }
        for gt_idx, gt_label in enumerate(gt_labels):
            if not valid_gt[gt_idx]:
                continue
            for pred_idx in range(num_preds):
                if not pred_range_mask[pred_idx]:
                    continue
                if int(pred_labels[pred_idx]) != int(gt_label):
                    continue
                if pred_scores is not None and float(pred_scores[pred_idx]) < args.score_thr:
                    continue
                distance = dataset._center_distance(gt_boxes[gt_idx],
                                                    pred_boxes[pred_idx])
                nearest_same_class[gt_idx] = min(
                    nearest_same_class[gt_idx], distance)
                if distance < args.dist_thr:
                    candidates.append((distance, gt_idx, pred_idx))
        candidates.sort(key=lambda item: item[0])

        used_gts = set()
        used_preds = set()
        for match_dist, gt_idx, pred_idx in candidates:
            if gt_idx in used_gts or pred_idx in used_preds:
                continue
            min_detail = traj_error_detail(pred_trajs[pred_idx],
                                           gt_trajs[gt_idx],
                                           gt_masks[gt_idx],
                                           miss_threshold=args.miss_thr)
            if min_detail is None:
                continue
            min_errors = (
                min_detail['minADE'], min_detail['minFDE'], min_detail['MR'])
            top1_errors = None
            top1_detail = None
            top1 = -1
            if pred_traj_scores is not None and pred_idx < len(pred_traj_scores):
                top1 = int(np.argmax(pred_traj_scores[pred_idx]))
                top1_detail = traj_error_detail(
                    pred_trajs[pred_idx],
                    gt_trajs[gt_idx],
                    gt_masks[gt_idx],
                    miss_threshold=args.miss_thr,
                    mode_index=top1)
                if top1_detail is not None:
                    top1_errors = (
                        top1_detail['minADE'], top1_detail['minFDE'],
                        top1_detail['MR'])
            used_gts.add(gt_idx)
            used_preds.add(pred_idx)

            bucket = buckets[gt_idx]
            label = int(gt_labels[gt_idx])
            cls_name = class_names[label]
            update_match_stats(overall[('Overall', bucket)], min_errors,
                               top1_errors)
            update_match_stats(by_class[(cls_name, bucket)], min_errors,
                               top1_errors)
            if cls_name in core_set:
                update_match_stats(by_core[('CoreVehicle', bucket)], min_errors,
                                   top1_errors)

            include_failure_bucket = (
                not failure_bucket_set or bucket in failure_bucket_set)
            if include_failure_bucket and args.failure_topk > 0:
                failure = failure_type(bucket, min_detail, top1_detail, args)
                if failure is not None:
                    ftype, severity = failure
                    meta = bucket_meta[gt_idx]
                    failure_rows.append(dict(
                        failure_type=ftype,
                        severity=severity,
                        sample_idx=sample_idx,
                        raw_idx=raw_idx,
                        token=raw_info.get('token', ''),
                        scene_token=raw_info.get('scene_token', ''),
                        timestamp=raw_info.get('timestamp', ''),
                        group=cls_name,
                        bucket=bucket,
                        gt_idx=gt_idx,
                        track_id=int(gt_track_ids[gt_idx])
                        if gt_idx < len(gt_track_ids) else -1,
                        pred_idx=pred_idx,
                        pred_score=float(pred_scores[pred_idx])
                        if pred_scores is not None else float('nan'),
                        center_dist=float(match_dist),
                        nearest_same_class_dist=nearest_same_class.get(
                            gt_idx, float('inf')),
                        minADE=min_detail['minADE'],
                        minFDE=min_detail['minFDE'],
                        MR=min_detail['MR'],
                        oracle_ade_mode=min_detail['oracle_ade_mode'],
                        oracle_fde_mode=min_detail['oracle_fde_mode'],
                        top1_mode=top1,
                        top1_ADE=top1_detail['minADE']
                        if top1_detail is not None else float('nan'),
                        top1_FDE=top1_detail['minFDE']
                        if top1_detail is not None else float('nan'),
                        top1_MR=top1_detail['MR']
                        if top1_detail is not None else float('nan'),
                        heading_change=meta['heading_change'],
                        path=meta['path'],
                        net=meta['net'],
                        lateral_ratio=meta['lateral_ratio'],
                        visualize_cmd=visualize_command(args, sample_idx)))

        if args.failure_topk > 0:
            for gt_idx in range(len(gt_labels)):
                if not valid_gt[gt_idx] or gt_idx in used_gts:
                    continue
                bucket = buckets[gt_idx]
                if failure_bucket_set and bucket not in failure_bucket_set:
                    continue
                label = int(gt_labels[gt_idx])
                cls_name = class_names[label]
                meta = bucket_meta[gt_idx]
                nearest_dist = nearest_same_class.get(gt_idx, float('inf'))
                severity = 1000.0 if not np.isfinite(nearest_dist) else nearest_dist
                failure_rows.append(dict(
                    failure_type='missed_track',
                    severity=severity,
                    sample_idx=sample_idx,
                    raw_idx=raw_idx,
                    token=raw_info.get('token', ''),
                    scene_token=raw_info.get('scene_token', ''),
                    timestamp=raw_info.get('timestamp', ''),
                    group=cls_name,
                    bucket=bucket,
                    gt_idx=gt_idx,
                    track_id=int(gt_track_ids[gt_idx])
                    if gt_idx < len(gt_track_ids) else -1,
                    pred_idx=-1,
                    pred_score=float('nan'),
                    center_dist=float('nan'),
                    nearest_same_class_dist=nearest_dist,
                    minADE=float('nan'),
                    minFDE=float('nan'),
                    MR=float('nan'),
                    oracle_ade_mode=-1,
                    oracle_fde_mode=-1,
                    top1_mode=-1,
                    top1_ADE=float('nan'),
                    top1_FDE=float('nan'),
                    top1_MR=float('nan'),
                    heading_change=meta['heading_change'],
                    path=meta['path'],
                    net=meta['net'],
                    lateral_ratio=meta['lateral_ratio'],
                    visualize_cmd=visualize_command(args, sample_idx)))

    rows = summarize(overall)
    rows += summarize(by_core)
    rows += summarize(by_class)
    csv_path = args.out_prefix + '.csv'
    write_csv(csv_path, rows)
    print(f'Wrote {csv_path}')
    if args.failure_topk > 0:
        failure_path = args.failure_out or args.out_prefix + '_failures.csv'
        selected_failures = []
        ordered_types = [
            'missed_track', 'poor_oracle', 'wrong_mode', 'sharp_turn_fail']
        seen_types = set(ordered_types)
        seen_types.update(row['failure_type'] for row in failure_rows)
        for ftype in ordered_types + sorted(seen_types - set(ordered_types)):
            rows_for_type = [
                row for row in failure_rows if row['failure_type'] == ftype]
            rows_for_type = sorted(
                rows_for_type, key=lambda row: row['severity'], reverse=True)
            selected_failures.extend(rows_for_type[:args.failure_topk])
        write_csv(failure_path, selected_failures)
        print(f'Wrote {failure_path} ({len(selected_failures)} rows)')
    print_table(rows, 'Overall')
    print_table(rows, 'CoreVehicle')


if __name__ == '__main__':
    main()
