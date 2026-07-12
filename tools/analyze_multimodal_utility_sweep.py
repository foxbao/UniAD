#!/usr/bin/env python3
"""Sweep D1.1 utility thresholds from one compact evaluation result file."""

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


EVAL_INDICES = (1, 3, 5)


def to_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def motion_bucket(traj, valid):
    points = traj[valid]
    if len(points) == 0:
        return 'Unknown'
    final_disp = float(np.linalg.norm(points[-1, :2]))
    if final_disp < 0.5:
        return 'Static'
    if final_disp < 2.0:
        return 'Slow'
    xy = np.concatenate([np.zeros((1, 2)), points[:, :2]], axis=0)
    delta = np.diff(xy, axis=0)
    norms = np.linalg.norm(delta, axis=1)
    moving = np.where(norms >= 0.05)[0]
    heading_change = 0.0
    if len(moving) >= 2:
        first = delta[moving[0]]
        last = delta[moving[-1]]
        heading_change = abs(math.degrees(wrap_pi(
            math.atan2(last[1], last[0])
            - math.atan2(first[1], first[0]))))
    end = points[-1, :2]
    net = float(np.linalg.norm(end))
    lateral_ratio = 0.0
    if net >= 1e-6:
        normal = np.array([-end[1], end[0]]) / net
        lateral_ratio = float(np.max(np.abs(points[:, :2] @ normal)) / net)
    return ('Turning' if heading_change >= 15.0 or lateral_ratio >= 0.15
            else 'MovingStraight')


def plan_fields(result):
    planning = result.get('planning', {})
    pred = planning.get('result_planning', {})
    gt_blob = planning.get('planning_gt', {})
    utility_key = ('multimodal_utility_score'
                   if pred.get('multimodal_utility_score') is not None
                   else 'multimodal_utility_probability')
    required = (utility_key, 'multimodal_selected_map_traj',
                'multimodal_fallback_traj')
    if any(pred.get(key) is None for key in required):
        return None
    if gt_blob.get('sdc_planning') is None:
        return None
    gt = to_numpy(gt_blob['sdc_planning'])
    gt = gt.reshape(-1, gt.shape[-1])
    mask = to_numpy(gt_blob['sdc_planning_mask'])
    mask = (mask.any(axis=-1).reshape(-1) if mask.ndim >= 2
            else mask.reshape(-1)).astype(bool)
    map_traj = to_numpy(pred['multimodal_selected_map_traj'])
    fallback = to_numpy(pred['multimodal_fallback_traj'])
    map_traj = map_traj.reshape(-1, map_traj.shape[-1])
    fallback = fallback.reshape(-1, fallback.shape[-1])
    steps = min(len(gt), len(mask), len(map_traj), len(fallback))
    if steps == 0 or not mask[:steps].any():
        return None
    probability = float(to_numpy(pred[utility_key]).reshape(-1)[0])
    return (probability, map_traj[:steps, :2], fallback[:steps, :2],
            gt[:steps, :2], mask[:steps])


def cost(pred, gt, valid):
    indices = [index for index in EVAL_INDICES
               if index < len(pred) and index < len(gt) and valid[index]]
    if not indices:
        indices = np.where(valid[:min(len(pred), len(gt))])[0].tolist()
    if not indices:
        return float('nan')
    return float(np.linalg.norm(
        pred[indices] - gt[indices], axis=-1).mean())


def summarize(samples, threshold):
    def new_group():
        return dict(error_sum=np.zeros(len(EVAL_INDICES)),
                    error_n=np.zeros(len(EVAL_INDICES), dtype=np.int64),
                    map_selected=0, n=0)

    groups = {'ALL': new_group()}
    for probability, map_traj, fallback, gt, valid, bucket in samples:
        use_map = probability >= threshold
        prediction = map_traj if use_map else fallback
        groups.setdefault(bucket, new_group())
        for name in ('ALL', bucket):
            agg = groups[name]
            agg['map_selected'] += int(use_map)
            agg['n'] += 1
            for offset, index in enumerate(EVAL_INDICES):
                if index < len(prediction) and index < len(gt) and valid[index]:
                    agg['error_sum'][offset] += float(np.linalg.norm(
                        prediction[index] - gt[index]))
                    agg['error_n'][offset] += 1
    output = {}
    for name, agg in groups.items():
        horizons = np.divide(
            agg['error_sum'], agg['error_n'],
            out=np.full(len(EVAL_INDICES), np.nan),
            where=agg['error_n'] > 0)
        output[name] = dict(
            avg_L2=float(np.nanmean(horizons)),
            horizon_L2=horizons.tolist(),
            map_rate=agg['map_selected'] / max(agg['n'], 1),
            n=agg['n'])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--thresholds', type=float, nargs='+', default=None)
    args = parser.parse_args()
    with open(args.results, 'rb') as file:
        results = pickle.load(file)
    if isinstance(results, dict):
        results = results.get('bbox_results', results.get('results', []))
    samples = []
    for result in results:
        fields = plan_fields(result)
        if fields is None:
            continue
        probability, map_traj, fallback, gt, valid = fields
        samples.append((probability, map_traj, fallback, gt, valid,
                        motion_bucket(gt, valid)))
    thresholds = (args.thresholds if args.thresholds is not None else
                  np.linspace(0.0, 1.0, 51).tolist())
    rows = []
    for threshold in thresholds:
        summary = summarize(samples, threshold)
        rows.append(dict(threshold=float(threshold), metrics=summary))
        print(
            f'{threshold:5.2f}  ALL={summary["ALL"]["avg_L2"]:.4f}  '
            f'map={100 * summary["ALL"]["map_rate"]:5.1f}%  '
            f'Static={summary.get("Static", {}).get("avg_L2", float("nan")):.4f}  '
            f'Slow={summary.get("Slow", {}).get("avg_L2", float("nan")):.4f}  '
            f'Moving={summary.get("MovingStraight", {}).get("avg_L2", float("nan")):.4f}  '
            f'Turning={summary.get("Turning", {}).get("avg_L2", float("nan")):.4f}')
    fallback_summary = summarize(samples, float('inf'))
    oracle_samples = []
    for _, map_traj, fallback, gt, valid, bucket in samples:
        map_cost = cost(map_traj, gt, valid)
        fallback_cost = cost(fallback, gt, valid)
        use_map = map_cost < fallback_cost
        oracle_samples.append((
            float(use_map), map_traj, fallback, gt, valid, bucket))
    oracle_summary = summarize(oracle_samples, 0.5)
    output = dict(
        samples=len(samples), rows=rows,
        fallback=fallback_summary, top1_oracle=oracle_summary)
    print('fallback', fallback_summary['ALL'])
    print('top1 oracle', oracle_summary['ALL'])
    if args.output_json:
        with open(args.output_json, 'w') as file:
            json.dump(output, file, indent=2)


if __name__ == '__main__':
    main()
