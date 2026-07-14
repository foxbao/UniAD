#!/usr/bin/env python3
"""Replay D3-A.1 relative fallback guards from one saved eval result."""

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.analysis_tools.analyze_d21_collision_audit import (  # noqa: E402
    summarize,
    transition_summary,
    valid_collisions,
)
from tools.analysis_tools.planning_ir_audit_utils import (  # noqa: E402
    motion_bucket,
    planning_mask,
    planning_trajectory,
    to_numpy,
)


DEFAULT_PC_RANGE = (-64.0, -48.0, -2.0, 64.0, 48.0, 6.0)


def load_results(path):
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        payload = payload.get('bbox_results', payload.get('results'))
    if not isinstance(payload, list):
        raise TypeError(f'Unsupported result payload in {path}')
    return payload


def sigmoid(value):
    value = np.clip(value, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-value))


def extract_sample(result, index, pc_range, cell_size, fallback_tiebreak):
    planning = result.get('planning', {})
    pred = planning.get('result_planning', {})
    gt_blob = planning.get('planning_gt', {})
    required = (
        'multimodal_set_candidates',
        'multimodal_set_valid',
        'multimodal_set_variant',
        'multimodal_set_indices',
        'multimodal_set_predicted_horizon_costs',
        'multimodal_set_collision_logits',
        'multimodal_set_selected_position',
        'sdc_traj',
    )
    if any(pred.get(key) is None for key in required):
        return None

    gt = planning_trajectory(gt_blob.get('sdc_planning'))
    valid_gt = planning_mask(gt_blob.get('sdc_planning_mask'))
    segmentation = to_numpy(gt_blob.get('segmentation'))
    if gt is None or valid_gt is None or segmentation is None:
        return None

    candidates = to_numpy(pred['multimodal_set_candidates'])
    candidates = candidates.reshape(
        -1, candidates.shape[-2], candidates.shape[-1])[:, :, :2]
    set_valid = to_numpy(pred['multimodal_set_valid']).reshape(-1).astype(bool)
    variants = to_numpy(pred['multimodal_set_variant']).reshape(-1).astype(
        np.int64)
    indices = to_numpy(pred['multimodal_set_indices']).reshape(-1).astype(
        np.int64)
    horizon_cost = to_numpy(
        pred['multimodal_set_predicted_horizon_costs'])
    horizon_cost = horizon_cost.reshape(-1, horizon_cost.shape[-1])
    collision_logits = to_numpy(pred['multimodal_set_collision_logits'])
    collision_logits = collision_logits.reshape(
        -1, collision_logits.shape[-1])
    size = len(candidates)
    if not all(len(value) == size for value in (
            set_valid, variants, indices, horizon_cost, collision_logits)):
        raise ValueError(f'Inconsistent set payload at result {index}')
    if not np.isin(variants, (0, 1, 2)).all():
        raise ValueError(f'Unknown set variant at result {index}')
    fallback_positions = np.where(variants == 2)[0]
    if len(fallback_positions) != 1:
        raise ValueError(
            f'Result {index} has {len(fallback_positions)} fallbacks')
    fallback_position = int(fallback_positions[0])
    if not set_valid[fallback_position]:
        raise ValueError(f'Fallback is invalid at result {index}')

    selection_cost = horizon_cost.mean(axis=-1).astype(np.float64)
    selection_cost[~set_valid] = 1e4
    selection_cost[fallback_position] -= float(fallback_tiebreak)
    collision_risk = sigmoid(collision_logits).max(axis=-1)
    online_position = int(to_numpy(
        pred.get('multimodal_set_selected_position')).reshape(-1)[0])
    online_trajectory = planning_trajectory(pred.get('sdc_traj'))
    steps = min(len(gt), len(valid_gt), candidates.shape[1])
    if steps == 0 or not valid_gt[:steps].any():
        return None
    gt = gt[:steps]
    valid_gt = valid_gt[:steps]
    candidates = candidates[:, :steps]
    if segmentation.ndim == 4:
        segmentation = segmentation[0]

    return dict(
        result_index=index,
        gt=gt,
        valid=valid_gt,
        segmentation=segmentation,
        candidates=candidates,
        set_valid=set_valid,
        variants=variants,
        indices=indices,
        selection_cost=selection_cost,
        collision_risk=collision_risk,
        fallback_position=fallback_position,
        online_position=online_position,
        online_trajectory=online_trajectory[:steps, :2],
        motion_bucket=motion_bucket(gt, valid_gt),
        pc_range=pc_range,
        cell_size=cell_size,
    )


def select(sample, margin):
    cost = sample['selection_cost'].copy()
    guarded = np.zeros_like(sample['set_valid'])
    if margin is not None:
        fallback_risk = sample['collision_risk'][sample['fallback_position']]
        guarded = sample['set_valid'] & (
            sample['collision_risk'] > fallback_risk + float(margin))
        guarded[sample['fallback_position']] = False
        cost[guarded] += 1000.0
    position = int(np.argmin(cost))
    return position, guarded


def replay(samples, margin):
    trajectories = []
    map_decisions = []
    positions = []
    guarded_count = 0
    guard_valid_count = 0
    variant_count = np.zeros(3, dtype=np.int64)
    selected_risk_sum = 0.0
    for sample in samples:
        position, guarded = select(sample, margin)
        trajectories.append(sample['candidates'][position])
        variant = int(sample['variants'][position])
        map_decisions.append(variant != 2)
        positions.append(position)
        variant_count[variant] += 1
        selected_risk_sum += float(sample['collision_risk'][position])
        guard_valid = sample['set_valid'] & (sample['variants'] != 2)
        guarded_count += int(guarded[guard_valid].sum())
        guard_valid_count += int(guard_valid.sum())
    metrics = summarize(trajectories, samples, map_decisions)
    n = max(len(samples), 1)
    diagnostics = dict(
        raw_rate=float(variant_count[0] / n),
        refined_rate=float(variant_count[1] / n),
        fallback_rate=float(variant_count[2] / n),
        guarded_rate=float(guarded_count / max(guard_valid_count, 1)),
        selected_max_collision_probability=float(selected_risk_sum / n),
    )
    return metrics, diagnostics, positions, trajectories


def fallback_replay(samples):
    trajectories = [
        sample['candidates'][sample['fallback_position']]
        for sample in samples
    ]
    metrics = summarize(trajectories, samples, [False] * len(samples))
    return metrics, trajectories


def collision_matrix(trajectories, samples):
    return [
        valid_collisions(
            trajectory, sample, sample['pc_range'], sample['cell_size'])
        for trajectory, sample in zip(trajectories, samples)
    ]


def format_percent(value):
    return f'{100.0 * value:.4f}%'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--output-json', required=True)
    parser.add_argument(
        '--margins', nargs='*', type=float,
        default=[0.10, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0])
    parser.add_argument('--online-margin', type=float, default=0.05)
    parser.add_argument('--fallback-tiebreak', type=float, default=1e-4)
    parser.add_argument('--cell-size', type=float, default=0.8)
    parser.add_argument(
        '--pc-range', nargs=6, type=float, default=DEFAULT_PC_RANGE)
    args = parser.parse_args()

    results = load_results(args.results)
    pc_range = tuple(float(value) for value in args.pc_range)
    samples = []
    for index, result in enumerate(results):
        sample = extract_sample(
            result, index, pc_range, args.cell_size,
            args.fallback_tiebreak)
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise RuntimeError('No valid D3-A.1 samples found')

    rows = []
    trajectories_by_name = {}
    no_guard_metrics, no_guard_diag, _, no_guard_traj = replay(samples, None)
    trajectories_by_name['no_guard'] = no_guard_traj
    rows.append(dict(
        name='no_guard', margin=None, metrics=no_guard_metrics,
        diagnostics=no_guard_diag))
    online_positions = None
    for margin in args.margins:
        metrics, diagnostics, positions, trajectories = replay(
            samples, margin)
        name = f'margin_{margin:g}'
        rows.append(dict(
            name=name, margin=float(margin),
            metrics=metrics, diagnostics=diagnostics))
        trajectories_by_name[name] = trajectories
        if math.isclose(margin, args.online_margin, abs_tol=1e-12):
            online_positions = positions
    if online_positions is None:
        _, _, online_positions, online_trajectories = replay(
            samples, args.online_margin)
        trajectories_by_name['online_margin'] = online_trajectories

    mismatch = 0
    trajectory_mismatch = 0
    for sample, position in zip(samples, online_positions):
        mismatch += int(position != sample['online_position'])
        trajectory_mismatch += int(not np.allclose(
            sample['candidates'][position], sample['online_trajectory'],
            atol=1e-5, rtol=1e-5))
    if mismatch or trajectory_mismatch:
        raise RuntimeError(
            f'Online replay mismatch: positions={mismatch}, '
            f'trajectories={trajectory_mismatch}')

    fallback_metrics, fallback_trajectories = fallback_replay(samples)
    trajectories_by_name['fallback_only'] = fallback_trajectories
    rows.append(dict(
        name='fallback_only', margin=None, metrics=fallback_metrics,
        diagnostics=dict(
            raw_rate=0.0, refined_rate=0.0, fallback_rate=1.0,
            guarded_rate=0.0,
            selected_max_collision_probability=None)))

    collision_by_name = {
        name: collision_matrix(trajectories, samples)
        for name, trajectories in trajectories_by_name.items()
    }
    transition_names = [
        name for name in (
            f'margin_{args.online_margin:g}', 'margin_0.01', 'margin_0')
        if name in collision_by_name
    ]
    transitions = {
        f'{name}_vs_fallback': transition_summary(
            collision_by_name[name], collision_by_name['fallback_only'])
        for name in transition_names
    }
    payload = dict(
        source=os.path.abspath(args.results),
        sample_count=len(samples),
        online_margin=float(args.online_margin),
        online_replay_position_mismatches=mismatch,
        online_replay_trajectory_mismatches=trajectory_mismatch,
        transitions=transitions,
        rows=rows,
    )
    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')

    print(
        'name          avg.L2  collision events map      raw/ref/fb        '
        'guarded')
    for row in rows:
        overall = row['metrics']['ALL']
        diag = row['diagnostics']
        variants = (
            f'{100 * diag["raw_rate"]:.1f}/'
            f'{100 * diag["refined_rate"]:.1f}/'
            f'{100 * diag["fallback_rate"]:.1f}')
        print(
            f'{row["name"]:<13s} {overall["avg_l2"]:7.5f}  '
            f'{format_percent(overall["avg_collision"]):>9s} '
            f'{overall["collision_events"]:6d} '
            f'{format_percent(overall["map_rate"]):>8s} '
            f'{variants:>16s} '
            f'{format_percent(diag["guarded_rate"]):>9s}')
    for name, transition in transitions.items():
        print(
            f'{name}: added={transition["added_total"]} '
            f'removed={transition["removed_total"]} '
            f'net={transition["net_total"]:+d}')
    print(f'Wrote {args.output_json}')


if __name__ == '__main__':
    main()
