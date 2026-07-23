#!/usr/bin/env python
"""Compare incremental and current-source cumulative GT-flow oracles."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from mmcv import Config

import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.datasets.pipelines.occflow_label import (
    GenerateOccFlowLabels,
)
from projects.mmdet3d_plugin.uniad.dense_heads.occworld_head import (
    compose_incremental_flow_2d,
    flow_to_world_layout,
    forward_splat_2d,
)
from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from tools.analysis_tools.audit_kl_occworld_flow_oracle import (
    IGNORE_FLOW,
    _generate_gt_flow,
)
from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)


def _empty(step_count):
    return {
        'current_instance_voxels': np.zeros(step_count, dtype=np.int64),
        'cumulative_covered_voxels': np.zeros(step_count, dtype=np.int64),
        'sequential_intersection': np.zeros(step_count, dtype=np.int64),
        'sequential_union': np.zeros(step_count, dtype=np.int64),
        'cumulative_intersection': np.zeros(step_count, dtype=np.int64),
        'cumulative_union': np.zeros(step_count, dtype=np.int64),
        'target_instance_voxels': np.zeros(step_count, dtype=np.int64),
        'norm_values': [[] for _ in range(step_count)],
        'cumulative_norm_values': [[] for _ in range(step_count)],
    }


def _percentiles(values):
    if not values:
        return None
    array = np.concatenate(values)
    return {
        'count': int(array.size),
        'mean': float(array.mean()),
        'p50': float(np.percentile(array, 50)),
        'p90': float(np.percentile(array, 90)),
        'p95': float(np.percentile(array, 95)),
        'p99': float(np.percentile(array, 99)),
        'max': float(array.max()),
    }


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _accumulate(acc, incremental_flow, label_path):
    with np.load(label_path, allow_pickle=False) as label:
        anchor_state = torch.from_numpy(np.asarray(
            label['current_observation_state_3d'], dtype=np.int64))
        anchor_valid = torch.from_numpy(np.asarray(
            label['current_observation_valid_3d'], dtype=np.bool_))
        target_state = torch.from_numpy(np.asarray(
            label['world_target_state_3d'], dtype=np.int64))
        target_valid = torch.from_numpy(np.asarray(
            label['world_target_valid_3d'], dtype=np.bool_))
    incremental_flow = flow_to_world_layout(
        incremental_flow[:4], ignore_index=IGNORE_FLOW)
    cumulative_flow = compose_incremental_flow_2d(
        incremental_flow[None], ignore_index=IGNORE_FLOW)[0]
    incremental_valid = torch.all(
        incremental_flow != IGNORE_FLOW, dim=1)
    cumulative_valid = torch.all(
        cumulative_flow != IGNORE_FLOW, dim=1)
    safe_incremental = torch.where(
        incremental_valid[:, None], incremental_flow,
        torch.zeros_like(incremental_flow))
    safe_cumulative = torch.where(
        cumulative_valid[:, None], cumulative_flow,
        torch.zeros_like(cumulative_flow))
    current_instance = (
        anchor_valid & (anchor_state == 3)).float()[None]
    sequential = current_instance
    for horizon in range(4):
        incremental_norm = torch.linalg.vector_norm(
            incremental_flow[horizon], dim=0)
        acc['norm_values'][horizon].append(
            incremental_norm[incremental_valid[horizon]].numpy())
        cumulative_norm = torch.linalg.vector_norm(
            cumulative_flow[horizon], dim=0)
        acc['cumulative_norm_values'][horizon].append(
            cumulative_norm[cumulative_valid[horizon]].numpy())

        sequential = forward_splat_2d(
            sequential, safe_incremental[horizon][None])
        cumulative = forward_splat_2d(
            current_instance, safe_cumulative[horizon][None])
        sequential_prediction = sequential[0] >= 0.5
        cumulative_prediction = cumulative[0] >= 0.5
        known = target_valid[horizon + 1]
        target = known & (target_state[horizon + 1] == 3)
        for name, prediction in (
                ('sequential', sequential_prediction),
                ('cumulative', cumulative_prediction)):
            prediction = prediction & known
            acc[f'{name}_intersection'][horizon] += int(
                (prediction & target).sum())
            acc[f'{name}_union'][horizon] += int(
                (prediction | target).sum())
        current = current_instance[0].bool()
        coverage = cumulative_valid[horizon][None].expand_as(current)
        acc['current_instance_voxels'][horizon] += int(current.sum())
        acc['cumulative_covered_voxels'][horizon] += int(
            (current & coverage).sum())
        acc['target_instance_voxels'][horizon] += int(target.sum())


def _summarize(acc):
    rows = []
    for horizon in range(4):
        rows.append({
            'future_horizon_index': horizon + 1,
            'incremental_flow_norm_cells': _percentiles(
                acc['norm_values'][horizon]),
            'current_source_cumulative_norm_cells': _percentiles(
                acc['cumulative_norm_values'][horizon]),
            'current_instance_cumulative_target_coverage': _ratio(
                acc['cumulative_covered_voxels'][horizon],
                acc['current_instance_voxels'][horizon]),
            'sequential_incremental_oracle_instance_iou': _ratio(
                acc['sequential_intersection'][horizon],
                acc['sequential_union'][horizon]),
            'independent_cumulative_oracle_instance_iou': _ratio(
                acc['cumulative_intersection'][horizon],
                acc['cumulative_union'][horizon]),
            'target_instance_voxels': int(
                acc['target_instance_voxels'][horizon]),
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sequence-root', type=Path, required=True)
    parser.add_argument(
        '--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--out-file', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(str(args.config))
    dataset = build_dataset(cfg.data.train)
    with args.manifest.open() as source:
        manifest = json.load(source)
    references = [
        int(record['reference_index'])
        for record in manifest['splits'][args.split]
    ]
    labels = _sequence_mapping(args.sequence_root)
    flow_cfg = next(
        dict(item) for item in cfg.train_pipeline
        if item['type'] == 'GenerateOccFlowLabels')
    flow_cfg.pop('type')
    generator = GenerateOccFlowLabels(**flow_cfg)
    acc = _empty(step_count=4)
    for position, reference in enumerate(references, start=1):
        flow, _ = _generate_gt_flow(dataset, generator, reference)
        _accumulate(acc, flow, labels[reference])
        print(f'[{position}/{len(references)}] reference={reference}')
    result = {
        'schema_version': 1,
        'manifest_name': manifest.get('name'),
        'split': args.split,
        'reference_count': len(references),
        'gt_flow_contract': (
            'incremental[t] maps frame t to t+1; cumulative[t] maps '
            'the current source directly to t+1'),
        'by_horizon': _summarize(acc),
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
