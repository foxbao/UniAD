#!/usr/bin/env python
"""Audit KL GT-flow coordinates and the oracle ceiling of 2D XY warp."""

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
from projects.mmdet3d_plugin.datasets.kl_dataset import KlDataset
from projects.mmdet3d_plugin.datasets.pipelines.loading import (
    LoadAnnotations3D_E2E,
)
from projects.mmdet3d_plugin.datasets.pipelines.occflow_label import (
    GenerateOccFlowLabels,
)
from projects.mmdet3d_plugin.uniad.dense_heads.occworld_head import (
    bev_to_world_layout,
    flow_to_world_layout,
    forward_splat_2d,
)
from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)


IGNORE_FLOW = 255.0


def _empty_accumulators(horizon_count: int, z_count: int) -> dict:
    return {
        'reference_count': 0,
        'native_source_instance_cells': 0,
        'native_source_flow_covered_cells': 0,
        'aligned_source_instance_cells': 0,
        'aligned_source_flow_covered_cells': 0,
        'flow_valid_cells_by_transition': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'source_voxels_by_transition': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'flow_covered_source_voxels_by_transition': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'persistence_intersection_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'persistence_union_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'oracle_intersection_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'oracle_union_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'target_instance_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'persistence_true_positive_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'oracle_true_positive_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'arrival_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'persistence_arrival_true_positive_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'oracle_arrival_true_positive_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'departure_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'persistence_departure_true_negative_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'oracle_departure_true_negative_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'visible_transition_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'instance_related_transition_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'non_instance_transition_by_horizon': np.zeros(
            horizon_count - 1, dtype=np.int64),
        'target_instance_by_z': np.zeros(z_count, dtype=np.int64),
        'persistence_true_positive_by_z': np.zeros(
            z_count, dtype=np.int64),
        'oracle_true_positive_by_z': np.zeros(z_count, dtype=np.int64),
        'persistence_union_by_z': np.zeros(z_count, dtype=np.int64),
        'oracle_union_by_z': np.zeros(z_count, dtype=np.int64),
        'persistence_intersection_by_z': np.zeros(z_count, dtype=np.int64),
        'oracle_intersection_by_z': np.zeros(z_count, dtype=np.int64),
    }


def _ratio(numerator: int, denominator: int):
    return None if denominator == 0 else float(numerator / denominator)


def _summarize(acc: dict) -> dict:
    horizons = []
    for horizon in range(len(acc['target_instance_by_horizon'])):
        target = int(acc['target_instance_by_horizon'][horizon])
        arrival = int(acc['arrival_by_horizon'][horizon])
        departure = int(acc['departure_by_horizon'][horizon])
        source = int(acc['source_voxels_by_transition'][horizon])
        transition = int(acc['visible_transition_by_horizon'][horizon])
        horizons.append({
            'future_horizon_index': horizon + 1,
            'flow_valid_bev_cells': int(
                acc['flow_valid_cells_by_transition'][horizon]),
            'source_instance_voxels': source,
            'source_flow_coverage': _ratio(
                int(acc['flow_covered_source_voxels_by_transition'][horizon]),
                source),
            'target_instance_voxels': target,
            'persistence_instance_iou': _ratio(
                int(acc['persistence_intersection_by_horizon'][horizon]),
                int(acc['persistence_union_by_horizon'][horizon])),
            'oracle_xy_instance_iou': _ratio(
                int(acc['oracle_intersection_by_horizon'][horizon]),
                int(acc['oracle_union_by_horizon'][horizon])),
            'persistence_instance_recall': _ratio(
                int(acc['persistence_true_positive_by_horizon'][horizon]),
                target),
            'oracle_xy_instance_recall': _ratio(
                int(acc['oracle_true_positive_by_horizon'][horizon]),
                target),
            'arrival_voxels': arrival,
            'persistence_arrival_recall': _ratio(
                int(acc[
                    'persistence_arrival_true_positive_by_horizon'][horizon]),
                arrival),
            'oracle_xy_arrival_recall': _ratio(
                int(acc['oracle_arrival_true_positive_by_horizon'][horizon]),
                arrival),
            'departure_voxels': departure,
            'persistence_departure_accuracy': _ratio(
                int(acc[
                    'persistence_departure_true_negative_by_horizon'][
                        horizon]), departure),
            'oracle_xy_departure_accuracy': _ratio(
                int(acc[
                    'oracle_departure_true_negative_by_horizon'][horizon]),
                departure),
            'visible_transition_voxels': transition,
            'instance_related_transition_ratio': _ratio(
                int(acc[
                    'instance_related_transition_by_horizon'][horizon]),
                transition),
            'non_instance_transition_ratio': _ratio(
                int(acc['non_instance_transition_by_horizon'][horizon]),
                transition),
        })
    by_z = []
    for z_index in range(len(acc['target_instance_by_z'])):
        target = int(acc['target_instance_by_z'][z_index])
        by_z.append({
            'z_index': z_index,
            'target_instance_voxels': target,
            'persistence_instance_iou': _ratio(
                int(acc['persistence_intersection_by_z'][z_index]),
                int(acc['persistence_union_by_z'][z_index])),
            'oracle_xy_instance_iou': _ratio(
                int(acc['oracle_intersection_by_z'][z_index]),
                int(acc['oracle_union_by_z'][z_index])),
            'persistence_instance_recall': _ratio(
                int(acc['persistence_true_positive_by_z'][z_index]),
                target),
            'oracle_xy_instance_recall': _ratio(
                int(acc['oracle_true_positive_by_z'][z_index]), target),
        })
    native_total = int(acc['native_source_instance_cells'])
    aligned_total = int(acc['aligned_source_instance_cells'])
    return {
        'reference_count': int(acc['reference_count']),
        'coordinate_contract': {
            'native_flow_source_coverage': _ratio(
                int(acc['native_source_flow_covered_cells']), native_total),
            'h_flip_flow_source_coverage': _ratio(
                int(acc['aligned_source_flow_covered_cells']), aligned_total),
            'required_conversion': (
                'flip H; negate dy; keep dx'),
        },
        'by_horizon': horizons,
        'by_z': by_z,
    }


def _generate_gt_flow(dataset, generator, reference_index: int):
    results = KlDataset.get_data_info(dataset, reference_index)
    occ_inputs = dataset._build_occ_inputs(reference_index)
    if results is None or occ_inputs is None:
        raise ValueError(
            f'Reference {reference_index} has no valid OCC inputs')
    results.update(occ_inputs)
    dataset.pre_pipeline(results)
    loader = LoadAnnotations3D_E2E(
        with_bbox_3d=True,
        with_label_3d=True,
        with_future_anns=True,
        with_ins_inds_3d=True,
        ins_inds_add_1=True)
    results = loader(results)
    results = generator(results)
    return results['gt_flow'].float(), results['gt_segmentation'].bool()


def _accumulate_reference(acc: dict, flow: torch.Tensor,
                          segmentation: torch.Tensor,
                          label_path: Path):
    with np.load(label_path, allow_pickle=False) as label:
        anchor_state = torch.from_numpy(np.asarray(
            label['current_observation_state_3d'], dtype=np.int64))
        anchor_valid = torch.from_numpy(np.asarray(
            label['current_observation_valid_3d'], dtype=np.bool_))
        target_state = torch.from_numpy(np.asarray(
            label['world_target_state_3d'], dtype=np.int64))
        target_valid = torch.from_numpy(np.asarray(
            label['world_target_valid_3d'], dtype=np.bool_))
    horizon_count, z_count, height, width = target_state.shape
    if tuple(flow.shape) != (horizon_count, 2, height, width):
        raise ValueError(
            f'Flow/target shape mismatch for {label_path}')
    current_instance = anchor_valid & (anchor_state == 3)
    current_instance_bev = current_instance.any(dim=0)
    native_valid = torch.all(flow != IGNORE_FLOW, dim=1)
    aligned_valid = bev_to_world_layout(native_valid)
    acc['reference_count'] += 1
    acc['native_source_instance_cells'] += int(
        current_instance_bev.sum())
    acc['native_source_flow_covered_cells'] += int(
        (current_instance_bev & native_valid[0]).sum())
    acc['aligned_source_instance_cells'] += int(
        current_instance_bev.sum())
    acc['aligned_source_flow_covered_cells'] += int(
        (current_instance_bev & aligned_valid[0]).sum())

    safe_flow = torch.where(
        native_valid[:, None], flow, torch.zeros_like(flow))
    aligned_flow = flow_to_world_layout(safe_flow)
    warped_probability = current_instance[None].float()
    for transition in range(horizon_count - 1):
        source = warped_probability[0] >= 0.5
        flow_covered = aligned_valid[transition][None].expand_as(source)
        acc['flow_valid_cells_by_transition'][transition] += int(
            aligned_valid[transition].sum())
        acc['source_voxels_by_transition'][transition] += int(source.sum())
        acc['flow_covered_source_voxels_by_transition'][transition] += int(
            (source & flow_covered).sum())
        warped_probability = forward_splat_2d(
            warped_probability,
            aligned_flow[transition][None])
        oracle = warped_probability[0] >= 0.5
        persistence = current_instance
        known = target_valid[transition + 1]
        target = known & (target_state[transition + 1] == 3)
        oracle_known = oracle & known
        persistence_known = persistence & known
        for name, prediction in (
                ('persistence', persistence_known),
                ('oracle', oracle_known)):
            intersection = prediction & target
            union = prediction | target
            acc[f'{name}_intersection_by_horizon'][transition] += int(
                intersection.sum())
            acc[f'{name}_union_by_horizon'][transition] += int(union.sum())
            acc[f'{name}_true_positive_by_horizon'][transition] += int(
                intersection.sum())
            acc[f'{name}_intersection_by_z'] += (
                intersection.sum(dim=(1, 2)).numpy())
            acc[f'{name}_union_by_z'] += union.sum(dim=(1, 2)).numpy()
            acc[f'{name}_true_positive_by_z'] += (
                intersection.sum(dim=(1, 2)).numpy())
        acc['target_instance_by_horizon'][transition] += int(target.sum())
        acc['target_instance_by_z'] += target.sum(dim=(1, 2)).numpy()

        arrival = target & ~current_instance
        departure = known & current_instance & ~target
        acc['arrival_by_horizon'][transition] += int(arrival.sum())
        acc['departure_by_horizon'][transition] += int(departure.sum())
        acc['persistence_arrival_true_positive_by_horizon'][transition] += (
            int((persistence_known & arrival).sum()))
        acc['oracle_arrival_true_positive_by_horizon'][transition] += int(
            (oracle_known & arrival).sum())
        acc[
            'persistence_departure_true_negative_by_horizon'][transition
        ] += int((~persistence_known & departure).sum())
        acc['oracle_departure_true_negative_by_horizon'][transition] += int(
            (~oracle_known & departure).sum())

        current_known = anchor_valid & (anchor_state != 0)
        current_class = anchor_state - 1
        future_class = target_state[transition + 1] - 1
        visible_transition = (
            current_known & known & (current_class != future_class))
        instance_related = visible_transition & (
            (current_class == 2) | (future_class == 2))
        acc['visible_transition_by_horizon'][transition] += int(
            visible_transition.sum())
        acc['instance_related_transition_by_horizon'][transition] += int(
            instance_related.sum())
        acc['non_instance_transition_by_horizon'][transition] += int(
            (visible_transition & ~instance_related).sum())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path,
        default=Path(
            'projects/configs/stage2_e2e_lidar/'
            'base_e2e_lidar_occworld_split_v2_world_only_anchor_'
            'history_flow_pilot.py'))
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v2.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument(
        '--splits', nargs='+', choices=('train', 'validation'),
        default=['train', 'validation'])
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_flow_oracle_audit_v2.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(str(args.config))
    dataset = build_dataset(cfg.data.train)
    with args.manifest.open() as source:
        manifest = json.load(source)
    sequence_mapping = _sequence_mapping(args.sequence_root)
    flow_cfg = next(
        dict(item) for item in cfg.train_pipeline
        if item['type'] == 'GenerateOccFlowLabels')
    flow_cfg.pop('type')
    generator = GenerateOccFlowLabels(**flow_cfg)
    result = {
        'schema_version': 1,
        'manifest_name': manifest.get('name'),
        'splits': {},
    }
    for split in args.splits:
        references = [
            int(record['reference_index'])
            for record in manifest['splits'][split]
        ]
        acc = _empty_accumulators(
            horizon_count=5, z_count=10)
        for position, reference in enumerate(references, start=1):
            flow, segmentation = _generate_gt_flow(
                dataset, generator, reference)
            _accumulate_reference(
                acc, flow, segmentation, sequence_mapping[reference])
            print(
                f'[{split} {position}/{len(references)}] '
                f'reference={reference}')
        result['splits'][split] = {
            'reference_indices': references,
            **_summarize(acc),
        }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        split: {
            'reference_count': summary['reference_count'],
            'coordinate_contract': summary['coordinate_contract'],
            'by_horizon': summary['by_horizon'],
        }
        for split, summary in result['splits'].items()
    }, ensure_ascii=False, indent=2))
    print(f'out_file={args.out_file}')


if __name__ == '__main__':
    main()
