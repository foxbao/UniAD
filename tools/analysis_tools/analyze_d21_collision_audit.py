#!/usr/bin/env python3
"""Pair D2.1, fallback, raw-candidate, and D2 collision outcomes."""

import argparse
import json
import os
import pickle
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.analysis_tools.planning_ir_audit_utils import (  # noqa: E402
    EVAL_HORIZON_INDICES,
    front_obstacle_bucket,
    horizon_collisions,
    horizon_l2,
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


def load_metadata(path):
    if path is None:
        return {}
    output = {}
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            index = int(record['result_index'])
            sample = record.get('teacher_input', {}).get('sample', {})
            output[index] = dict(
                token=sample.get('token'),
                scene_token=sample.get('scene_token'),
                timestamp=sample.get('timestamp'),
                raw_dataset_index=sample.get('raw_dataset_index'),
            )
    return output


def scalar(blob, key, default=None):
    value = to_numpy(blob.get(key))
    if value is None or value.size == 0:
        return default
    return value.reshape(-1)[0].item()


def audit_selected_trajectory(blob, selected_index, field):
    indices = to_numpy(blob.get('multimodal_audit_indices'))
    candidates = to_numpy(blob.get(field))
    if indices is None or candidates is None:
        return None
    indices = indices.reshape(-1).astype(np.int64)
    candidates = candidates.reshape(
        -1, candidates.shape[-2], candidates.shape[-1])
    offsets = np.where(indices == int(selected_index))[0]
    if len(offsets) != 1:
        raise ValueError(
            f'Selected candidate {selected_index} appears '
            f'{len(offsets)} times in audit payload')
    return candidates[int(offsets[0]), :, :2]


def extract_sample(result, require_audit=False):
    planning = result.get('planning', {})
    pred = planning.get('result_planning', {})
    gt_blob = planning.get('planning_gt', {})
    gt = planning_trajectory(gt_blob.get('sdc_planning'))
    valid = planning_mask(gt_blob.get('sdc_planning_mask'))
    selected = planning_trajectory(pred.get('sdc_traj'))
    fallback = planning_trajectory(pred.get('multimodal_fallback_traj'))
    segmentation = gt_blob.get('segmentation')
    if any(value is None for value in (gt, valid, selected, fallback,
                                       segmentation)):
        return None
    steps = min(len(gt), len(valid), len(selected), len(fallback))
    if steps == 0 or not valid[:steps].any():
        return None
    gt = gt[:steps]
    valid = valid[:steps]
    selected = selected[:steps, :2]
    fallback = fallback[:steps, :2]
    selected_index = scalar(pred, 'multimodal_selected_index')
    fallback_index = scalar(pred, 'multimodal_fallback_index')
    if selected_index is None or fallback_index is None:
        return None
    raw_selected = audit_selected_trajectory(
        pred, selected_index, 'multimodal_audit_raw_candidates')
    refined_audit = audit_selected_trajectory(
        pred, selected_index, 'multimodal_audit_refined_candidates')
    if require_audit and (raw_selected is None or refined_audit is None):
        raise KeyError('D2.1 result does not contain an audit_topk payload')
    if refined_audit is not None and not np.allclose(
            refined_audit[:steps, :2], selected, atol=1e-5):
        raise ValueError('Audited refined candidate differs from sdc_traj')
    selected_cost = scalar(pred, 'multimodal_selected_predicted_cost')
    fallback_cost = scalar(pred, 'multimodal_fallback_predicted_cost')
    margin = (None if selected_cost is None or fallback_cost is None else
              float(fallback_cost - selected_cost))
    return dict(
        gt=gt,
        valid=valid,
        segmentation=segmentation,
        selected=selected,
        fallback=fallback,
        raw_selected=(None if raw_selected is None else
                      raw_selected[:steps, :2]),
        selected_index=int(selected_index),
        fallback_index=int(fallback_index),
        map_selected=int(selected_index) != int(fallback_index),
        margin=margin,
        motion_bucket=motion_bucket(gt, valid),
    )


def valid_collisions(trajectory, sample, pc_range, cell_size):
    values = horizon_collisions(
        trajectory, sample['segmentation'], pc_range, cell_size)
    return [
        (None if index >= len(sample['valid']) or not sample['valid'][index]
         else bool(value))
        for index, value in zip(EVAL_HORIZON_INDICES, values)
    ]


def trajectory_metrics(trajectory, sample, pc_range, cell_size):
    l2, _ = horizon_l2(trajectory, sample['gt'], sample['valid'])
    collisions = valid_collisions(
        trajectory, sample, pc_range, cell_size)
    return l2, collisions


def new_group():
    size = len(EVAL_HORIZON_INDICES)
    return dict(
        l2_sum=np.zeros(size, dtype=np.float64),
        l2_n=np.zeros(size, dtype=np.int64),
        collision_sum=np.zeros(size, dtype=np.int64),
        collision_n=np.zeros(size, dtype=np.int64),
        map_selected=0,
        n=0,
    )


def add_metrics(group, l2, collisions, map_selected):
    group['n'] += 1
    group['map_selected'] += int(map_selected)
    for offset, value in enumerate(l2):
        if value is not None:
            group['l2_sum'][offset] += float(value)
            group['l2_n'][offset] += 1
    for offset, value in enumerate(collisions):
        if value is not None:
            group['collision_sum'][offset] += int(value)
            group['collision_n'][offset] += 1


def finish_group(group):
    horizon_l2_values = np.divide(
        group['l2_sum'], group['l2_n'],
        out=np.full(len(EVAL_HORIZON_INDICES), np.nan),
        where=group['l2_n'] > 0)
    horizon_collision = np.divide(
        group['collision_sum'], group['collision_n'],
        out=np.full(len(EVAL_HORIZON_INDICES), np.nan),
        where=group['collision_n'] > 0)
    return dict(
        avg_l2=float(np.nanmean(horizon_l2_values)),
        avg_collision=float(np.nanmean(horizon_collision)),
        horizon_l2=horizon_l2_values.tolist(),
        horizon_collision=horizon_collision.tolist(),
        collision_events=int(group['collision_sum'].sum()),
        valid_horizon_events=int(group['collision_n'].sum()),
        map_rate=group['map_selected'] / max(group['n'], 1),
        n=int(group['n']),
    )


def summarize(trajectories, samples, map_decisions):
    groups = {'ALL': new_group()}
    for trajectory, sample, map_selected in zip(
            trajectories, samples, map_decisions):
        obstacle = front_obstacle_bucket(
            sample['segmentation'], sample['pc_range'], sample['cell_size'])
        names = ('ALL', sample['motion_bucket'], obstacle)
        l2, collisions = trajectory_metrics(
            trajectory, sample, sample['pc_range'], sample['cell_size'])
        for name in names:
            groups.setdefault(name, new_group())
            add_metrics(groups[name], l2, collisions, map_selected)
    return {name: finish_group(group) for name, group in groups.items()}


def transition_summary(first, second):
    added = np.zeros(len(EVAL_HORIZON_INDICES), dtype=np.int64)
    removed = np.zeros(len(EVAL_HORIZON_INDICES), dtype=np.int64)
    both = np.zeros(len(EVAL_HORIZON_INDICES), dtype=np.int64)
    neither = np.zeros(len(EVAL_HORIZON_INDICES), dtype=np.int64)
    for first_values, second_values in zip(first, second):
        for offset, (left, right) in enumerate(zip(
                first_values, second_values)):
            if left is None or right is None:
                continue
            if left and not right:
                added[offset] += 1
            elif right and not left:
                removed[offset] += 1
            elif left:
                both[offset] += 1
            else:
                neither[offset] += 1
    return dict(
        added=added.tolist(),
        removed=removed.tolist(),
        both=both.tolist(),
        neither=neither.tolist(),
        added_total=int(added.sum()),
        removed_total=int(removed.sum()),
        net_total=int(added.sum() - removed.sum()),
    )


def collision_rows(samples, system_trajectories, names):
    system_collisions = {
        name: [valid_collisions(
            trajectory, sample, sample['pc_range'], sample['cell_size'])
               for trajectory, sample in zip(trajectories, samples)]
        for name, trajectories in zip(names, system_trajectories)
    }
    transitions = {}
    for left, right in (
            ('d21_refined', 'fallback'),
            ('d21_refined', 'd2_full'),
            ('d21_refined', 'd21_raw'),
            ('d21_raw', 'fallback'),
            ('d21_raw', 'd2_full')):
        if left in system_collisions and right in system_collisions:
            transitions[f'{left}_vs_{right}'] = transition_summary(
                system_collisions[left], system_collisions[right])
    cases = []
    for index, sample in enumerate(samples):
        refined = system_collisions['d21_refined'][index]
        fallback = system_collisions['fallback'][index]
        d2 = system_collisions.get('d2_full', [None] * len(samples))[index]
        raw = system_collisions.get('d21_raw', [None] * len(samples))[index]
        for offset, horizon_index in enumerate(EVAL_HORIZON_INDICES):
            values = dict(
                d21_refined=refined[offset],
                fallback=fallback[offset],
                d2_full=(None if d2 is None else d2[offset]),
                d21_raw=(None if raw is None else raw[offset]),
            )
            if values['d21_refined'] and (
                    values['fallback'] is False
                    or values['d2_full'] is False
                    or values['d21_raw'] is False):
                cases.append(dict(
                    result_index=sample['result_index'],
                    sample=sample.get('metadata'),
                    horizon_step=int(horizon_index),
                    horizon_seconds=float((horizon_index + 1) * 0.5),
                    motion_bucket=sample['motion_bucket'],
                    map_selected=sample['map_selected'],
                    selected_index=sample['selected_index'],
                    fallback_index=sample['fallback_index'],
                    predicted_cost_margin=sample['margin'],
                    collisions=values,
                ))
    return transitions, cases


def threshold_summary(samples, thresholds, d2_summary, trajectory_key):
    rows = []
    for threshold in thresholds:
        trajectories = []
        decisions = []
        for sample in samples:
            use_map = (sample['map_selected'] and sample['margin'] is not None
                       and sample['margin'] >= threshold)
            trajectories.append(
                sample[trajectory_key] if use_map else sample['fallback'])
            decisions.append(use_map)
        metrics = summarize(trajectories, samples, decisions)
        promotion_pass = None
        if d2_summary is not None:
            promotion_pass = (
                metrics['ALL']['avg_l2'] < d2_summary['ALL']['avg_l2']
                and metrics['ALL']['avg_collision']
                <= d2_summary['ALL']['avg_collision']
                and all(
                    name not in d2_summary
                    or metrics[name]['avg_l2']
                    <= d2_summary[name]['avg_l2'] + 0.01
                    for name in ('static', 'slow', 'moving_straight',
                                 'turning')))
        rows.append(dict(
            threshold=float(threshold),
            metrics=metrics,
            promotion_pass=promotion_pass,
        ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('d21_results')
    parser.add_argument('--d2-results')
    parser.add_argument('--metadata-jsonl')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--cell-size', type=float, default=0.8)
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=DEFAULT_PC_RANGE)
    parser.add_argument(
        '--thresholds', type=float, nargs='+',
        default=(0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
                 0.125, 0.15, 0.2, 0.3))
    args = parser.parse_args()

    d21_results = load_results(args.d21_results)
    metadata = load_metadata(args.metadata_jsonl)
    d2_results = (load_results(args.d2_results)
                  if args.d2_results else None)
    if d2_results is not None and len(d2_results) != len(d21_results):
        raise ValueError('D2 and D2.1 result lengths differ')

    samples = []
    d2_trajectories = []
    d2_decisions = []
    for index, d21_result in enumerate(d21_results):
        sample = extract_sample(d21_result, require_audit=True)
        if sample is None:
            continue
        sample['result_index'] = index
        sample['metadata'] = metadata.get(index)
        sample['pc_range'] = tuple(args.pc_range)
        sample['cell_size'] = float(args.cell_size)
        if d2_results is not None:
            d2_sample = extract_sample(d2_results[index])
            if d2_sample is None:
                raise ValueError(f'D2 sample {index} has no planning payload')
            if not np.allclose(sample['gt'], d2_sample['gt'], atol=1e-5):
                raise ValueError(f'GT mismatch at result index {index}')
            d2_trajectories.append(d2_sample['selected'])
            d2_decisions.append(d2_sample['map_selected'])
        samples.append(sample)

    refined = [sample['selected'] for sample in samples]
    fallback = [sample['fallback'] for sample in samples]
    raw = [sample['raw_selected'] for sample in samples]
    map_decisions = [sample['map_selected'] for sample in samples]
    no_map = [False] * len(samples)
    systems = dict(
        d21_refined=summarize(refined, samples, map_decisions),
        d21_raw=summarize(raw, samples, map_decisions),
        fallback=summarize(fallback, samples, no_map),
    )
    if d2_results is not None:
        systems['d2_full'] = summarize(
            d2_trajectories, samples, d2_decisions)

    names = ['d21_refined', 'fallback', 'd21_raw']
    trajectories = [refined, fallback, raw]
    if d2_results is not None:
        names.append('d2_full')
        trajectories.append(d2_trajectories)
    transitions, cases = collision_rows(samples, trajectories, names)
    refined_rows = threshold_summary(
        samples, args.thresholds, systems.get('d2_full'), 'selected')
    raw_rows = threshold_summary(
        samples, args.thresholds, systems.get('d2_full'), 'raw_selected')
    output = dict(
        samples=len(samples),
        horizons=[float((index + 1) * 0.5)
                  for index in EVAL_HORIZON_INDICES],
        systems=systems,
        collision_transitions=transitions,
        collision_cases=cases,
        threshold_sweep_refined=refined_rows,
        threshold_sweep_raw=raw_rows,
    )
    parent = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(parent, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2, allow_nan=False)

    for name, summary in systems.items():
        overall = summary['ALL']
        print(
            f'{name:12s} L2={overall["avg_l2"]:.5f} '
            f'collision={100 * overall["avg_collision"]:.4f}% '
            f'map={100 * overall["map_rate"]:.2f}%')
    for name, transition in transitions.items():
        print(
            f'{name:28s} added={transition["added_total"]} '
            f'removed={transition["removed_total"]} '
            f'net={transition["net_total"]:+d}')
    for mode, rows in (
            ('refined', refined_rows), ('raw', raw_rows)):
        for row in rows:
            overall = row['metrics']['ALL']
            print(
                f'{mode:7s} threshold={row["threshold"]:.3f} '
                f'L2={overall["avg_l2"]:.5f} '
                f'collision={100 * overall["avg_collision"]:.4f}% '
                f'map={100 * overall["map_rate"]:.2f}% '
                f'pass={row["promotion_pass"]}')


if __name__ == '__main__':
    main()
