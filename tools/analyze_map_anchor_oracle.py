#!/usr/bin/env python3
"""Offline oracle analysis for map-lane anchors in KL planning.

This does not run the detector or touch training. It asks a narrower question:
given the surveyed HD map and each validation sample's GT SDC plan, how close
can a local lane-centerline anchor get to the GT trajectory if the best map lane
were selected by oracle?

The result is an upper-bound diagnostic for the next route:
  - strong oracle gain: explicit map/goal conditioning is worth implementing;
  - weak oracle gain: the current map geometry is unlikely to help planning L2.
"""

import argparse
import math
import pickle
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from projects.mmdet3d_plugin.uniad.dense_heads.motion_head_plugin.map_lane_encoder import (
    HDMapParser,
)


def load_infos(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj.get('data_list', obj.get('infos', []))
    return obj


def valid_plan_xy(info, steps):
    plan = info.get('sdc_planning')
    mask = info.get('sdc_planning_mask')
    if plan is None or mask is None:
        return None, None
    plan = np.asarray(plan, dtype=np.float64).reshape(-1, plan.shape[-1])
    mask = np.asarray(mask)
    if mask.ndim >= 2:
        valid = mask.any(axis=-1).reshape(-1).astype(bool)
    else:
        valid = mask.reshape(-1).astype(bool)
    T = min(steps, plan.shape[0], valid.shape[0])
    if T <= 0 or not valid[:T].any():
        return None, None
    return plan[:T, :2], valid[:T]


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_change_deg(points, min_segment_disp=0.05):
    if len(points) < 2:
        return 0.0
    pts = np.concatenate([np.zeros((1, 2), dtype=np.float64),
                          points[:, :2]], axis=0)
    deltas = np.diff(pts, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = np.where(norms >= min_segment_disp)[0]
    if len(valid) < 2:
        return 0.0
    v0 = deltas[valid[0]]
    v1 = deltas[valid[-1]]
    a0 = math.atan2(float(v0[1]), float(v0[0]))
    a1 = math.atan2(float(v1[1]), float(v1[0]))
    return abs(math.degrees(wrap_pi(a1 - a0)))


def lateral_ratio(points):
    if len(points) < 2:
        return 0.0
    end = points[-1, :2]
    net = float(np.linalg.norm(end))
    if net < 1e-6:
        return 0.0
    direction = end / net
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    lateral = float(np.max(np.abs(points[:, :2] @ normal)))
    return lateral / max(net, 1e-6)


def motion_bucket(gt_xy, valid):
    idx = np.where(valid[:len(gt_xy)])[0]
    if len(idx) == 0:
        return 'unknown'
    points = gt_xy[:idx[-1] + 1][valid[:idx[-1] + 1]]
    if len(points) == 0:
        return 'unknown'
    final_disp = float(np.linalg.norm(points[-1, :2]))
    if final_disp < 0.5:
        return 'static'
    if final_disp < 2.0:
        return 'slow'
    if heading_change_deg(points) >= 15.0 or lateral_ratio(points) >= 0.15:
        return 'turning'
    return 'moving_straight'


def interp_polyline(points, sample_s):
    deltas = points[1:] - points[:-1]
    seg_len = np.linalg.norm(deltas, axis=-1)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])
    if cum_s[-1] < 1e-6:
        return np.repeat(points[:1], len(sample_s), axis=0)
    sample_s = np.minimum(sample_s, cum_s[-1])
    right = np.searchsorted(cum_s, sample_s, side='left')
    right = np.clip(right, 1, len(points) - 1)
    left = right - 1
    denom = np.maximum(cum_s[right] - cum_s[left], 1e-6)
    alpha = ((sample_s - cum_s[left]) / denom)[:, None]
    return points[left] * (1.0 - alpha) + points[right] * alpha


def local_forward_anchor(points, gt_xy, reference):
    origin = np.zeros((1, 2), dtype=np.float64)
    start_idx = int(np.linalg.norm(points, axis=-1).argmin())
    deltas = points[1:] - points[:-1]
    seg_len = np.linalg.norm(deltas, axis=-1)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])
    ref_pts = np.concatenate([origin, gt_xy], axis=0)
    ref_step = np.linalg.norm(ref_pts[1:] - ref_pts[:-1], axis=-1)
    target_s = np.cumsum(ref_step)
    sample_s = cum_s[start_idx] + target_s
    anchor = interp_polyline(points, sample_s)
    if reference == 'relative_start':
        anchor = anchor - points[start_idx:start_idx + 1]
    return anchor


def score_anchor(anchor, gt_xy, valid):
    err = np.linalg.norm(anchor - gt_xy, axis=-1)
    eval_idx = [1, 3, 5]
    eval_vals = [err[i] for i in eval_idx if i < len(err) and valid[i]]
    if eval_vals:
        avg_l2 = float(np.mean(eval_vals))
    else:
        avg_l2 = float(np.mean(err[valid]))
    endpoint = float(np.linalg.norm(anchor[-1] - gt_xy[-1]))
    traj = float(np.mean(err[valid]))
    # Match the in-model best_endpoint intent: endpoint first, trajectory tie
    # breaker. We still report planning avg.L2 separately.
    select_score = endpoint + 0.25 * traj
    return select_score, avg_l2, err


def add_metric(agg, key, value):
    agg[key + '_sum'] += float(value)
    agg[key + '_n'] += 1


def fmt(sum_v, n_v):
    return float(sum_v / n_v) if n_v else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-file', default='data/kl_8/kl_infos_val.pkl')
    parser.add_argument('--map-path', default='data/kl_8/map/base_map.txt')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--num-lanes', type=int, default=64)
    parser.add_argument('--num-points-per-lane', type=int, default=20)
    parser.add_argument('--candidate-k', type=int, default=16)
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--reference', choices=['relative_start', 'absolute'],
                        default='relative_start')
    parser.add_argument('--direction', choices=['bidirectional', 'forward'],
                        default='bidirectional')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    infos = load_infos(args.ann_file)
    if args.limit:
        infos = infos[:args.limit]
    map_parser = HDMapParser(args.map_path, args.num_points_per_lane)

    total = defaultdict(float)
    by_bucket = {name: defaultdict(float) for name in
                 ('static', 'slow', 'moving_straight', 'turning', 'unknown')}
    no_lane = 0
    no_plan = 0

    for info in infos:
        gt_xy, valid = valid_plan_xy(info, args.steps)
        if gt_xy is None:
            no_plan += 1
            continue
        features, _centroids, lane_valid = map_parser.crop_and_transform(
            np.asarray(info.get('ego2global', np.eye(4)), dtype=np.float64),
            args.pc_range, args.num_lanes)
        lane_points = features[..., :2].astype(np.float64)
        valid_lane_idx = np.where(lane_valid)[0]
        if len(valid_lane_idx) == 0:
            no_lane += 1
            continue

        lane_dist = np.linalg.norm(lane_points[valid_lane_idx], axis=-1).min(axis=-1)
        order = valid_lane_idx[np.argsort(lane_dist)[:args.candidate_k]]
        best = None
        for idx in order:
            candidates = [lane_points[idx]]
            if args.direction == 'bidirectional':
                candidates.append(lane_points[idx][::-1])
            for points in candidates:
                anchor = local_forward_anchor(points, gt_xy, args.reference)
                select_score, avg_l2, err = score_anchor(anchor, gt_xy, valid)
                if best is None or select_score < best[0]:
                    best = (select_score, avg_l2, err, anchor, idx)

        if best is None:
            no_lane += 1
            continue
        _select_score, avg_l2, err, anchor, _idx = best
        bucket = motion_bucket(gt_xy, valid)
        target_aggs = (total, by_bucket[bucket])
        final_disp = float(np.linalg.norm(gt_xy[valid][-1]))
        anchor_final_disp = float(np.linalg.norm(anchor[valid][-1]))
        for agg in target_aggs:
            agg['n'] += 1
            add_metric(agg, 'oracle_avg_l2', avg_l2)
            add_metric(agg, 'gt_final_disp', final_disp)
            add_metric(agg, 'anchor_final_disp', anchor_final_disp)
            if final_disp >= 0.5:
                add_metric(agg, 'final_disp_ratio',
                           anchor_final_disp / max(final_disp, 1e-6))
            for step in (1, 3, 5):
                if step < len(err) and valid[step]:
                    add_metric(agg, f'L2_{step}', err[step])

    print('Map-anchor oracle C0')
    print(f'ann_file: {args.ann_file}')
    print(f'map_path: {args.map_path}')
    print(f'samples: total={len(infos)} used={int(total["n"])} '
          f'no_plan={no_plan} no_lane={no_lane}')
    print(f'config: num_lanes={args.num_lanes} candidate_k={args.candidate_k} '
          f'reference={args.reference} direction={args.direction}')

    def print_row(name, agg):
        n = int(agg['n'])
        if n == 0:
            return
        l2_1 = fmt(agg['L2_1_sum'], agg['L2_1_n'])
        l2_2 = fmt(agg['L2_3_sum'], agg['L2_3_n'])
        l2_3 = fmt(agg['L2_5_sum'], agg['L2_5_n'])
        avg = fmt(agg['oracle_avg_l2_sum'], agg['oracle_avg_l2_n'])
        gt_disp = fmt(agg['gt_final_disp_sum'], agg['gt_final_disp_n'])
        anchor_disp = fmt(agg['anchor_final_disp_sum'],
                          agg['anchor_final_disp_n'])
        ratio = fmt(agg['final_disp_ratio_sum'],
                    agg['final_disp_ratio_n'])
        print(f'{name:16s} N={n:5d}  '
              f'L2@1/2/3s={l2_1:.4f}/{l2_2:.4f}/{l2_3:.4f}  '
              f'avg.L2={avg:.4f}  '
              f'disp={anchor_disp:.3f}/{gt_disp:.3f} ratio={ratio:.3f}')

    print_row('ALL', total)
    for name in ('static', 'slow', 'moving_straight', 'turning', 'unknown'):
        print_row(name, by_bucket[name])

    # Current best measured model for quick readout. Keep this script offline
    # and model-agnostic; these constants are printed only as context.
    champion_avg_l2 = 0.604951604902982
    oracle_avg = fmt(total['oracle_avg_l2_sum'], total['oracle_avg_l2_n'])
    if np.isfinite(oracle_avg):
        print(f'vs A_epoch1_champion avg.L2={champion_avg_l2:.4f}: '
              f'oracle_delta={oracle_avg - champion_avg_l2:+.4f}')


if __name__ == '__main__':
    main()
