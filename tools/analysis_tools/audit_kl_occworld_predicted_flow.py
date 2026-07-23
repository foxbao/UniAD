#!/usr/bin/env python
"""Compare exported aligned OccWorld flow with official GT endpoints."""

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
)
from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from tools.analysis_tools.audit_kl_occworld_flow_oracle import (
    IGNORE_FLOW,
    _generate_gt_flow,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
    _split_references,
)


PREDICTION_FLOW_KEY = 'future_flow_2d'


def _empty_accumulators(horizon_count: int) -> dict:
    return {
        'valid_count': np.zeros(horizon_count, dtype=np.int64),
        'direction_count': np.zeros(horizon_count, dtype=np.int64),
        'in_bounds_count': np.zeros(horizon_count, dtype=np.int64),
        'epe_sum': np.zeros(horizon_count, dtype=np.float64),
        'zero_flow_epe_sum': np.zeros(horizon_count, dtype=np.float64),
        'predicted_norm_sum': np.zeros(horizon_count, dtype=np.float64),
        'target_norm_sum': np.zeros(horizon_count, dtype=np.float64),
        'cosine_sum': np.zeros(horizon_count, dtype=np.float64),
        'absolute_error_sum': np.zeros(
            (horizon_count, 2), dtype=np.float64),
        'sign_count': np.zeros(
            (horizon_count, 2), dtype=np.int64),
        'sign_match_count': np.zeros(
            (horizon_count, 2), dtype=np.int64),
    }


def _accumulate(acc: dict, prediction: torch.Tensor,
                target: torch.Tensor):
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError('Predicted and target flow must match [T,2,H,W]')
    horizon_count, _, height, width = target.shape
    valid = torch.all(target != IGNORE_FLOW, dim=1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=prediction.dtype),
        torch.arange(width, dtype=prediction.dtype),
        indexing='ij')
    for horizon in range(horizon_count):
        selection = valid[horizon]
        predicted_vectors = prediction[horizon, :, selection].transpose(0, 1)
        target_vectors = target[horizon, :, selection].transpose(0, 1)
        if predicted_vectors.numel() == 0:
            continue
        if not torch.isfinite(predicted_vectors).all():
            raise ValueError('Predicted flow contains non-finite values')
        error = predicted_vectors - target_vectors
        predicted_norm = torch.linalg.vector_norm(
            predicted_vectors, dim=1)
        target_norm = torch.linalg.vector_norm(target_vectors, dim=1)
        endpoint_error = torch.linalg.vector_norm(error, dim=1)
        direction_selection = (
            (predicted_norm > 1e-6) & (target_norm > 1e-6))
        cosine = torch.sum(
            predicted_vectors[direction_selection] *
            target_vectors[direction_selection], dim=1) / (
                predicted_norm[direction_selection] *
                target_norm[direction_selection])
        destination_y = grid_y[selection] + predicted_vectors[:, 0]
        destination_x = grid_x[selection] + predicted_vectors[:, 1]
        in_bounds = (
            (destination_y >= 0) & (destination_y < height) &
            (destination_x >= 0) & (destination_x < width))

        acc['valid_count'][horizon] += int(selection.sum())
        acc['direction_count'][horizon] += int(direction_selection.sum())
        acc['in_bounds_count'][horizon] += int(in_bounds.sum())
        acc['epe_sum'][horizon] += float(endpoint_error.sum())
        acc['zero_flow_epe_sum'][horizon] += float(target_norm.sum())
        acc['predicted_norm_sum'][horizon] += float(predicted_norm.sum())
        acc['target_norm_sum'][horizon] += float(target_norm.sum())
        acc['cosine_sum'][horizon] += float(cosine.sum())
        acc['absolute_error_sum'][horizon] += (
            error.abs().sum(dim=0).numpy())
        for component in range(2):
            sign_selection = target_vectors[:, component].abs() > 1e-6
            sign_match = (
                torch.sign(predicted_vectors[:, component]) ==
                torch.sign(target_vectors[:, component]))
            acc['sign_count'][horizon, component] += int(
                sign_selection.sum())
            acc['sign_match_count'][horizon, component] += int(
                (sign_selection & sign_match).sum())


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _summarize(acc: dict) -> list:
    summaries = []
    for horizon in range(len(acc['valid_count'])):
        count = int(acc['valid_count'][horizon])
        direction_count = int(acc['direction_count'][horizon])
        mean_epe = _ratio(acc['epe_sum'][horizon], count)
        zero_epe = _ratio(acc['zero_flow_epe_sum'][horizon], count)
        summaries.append({
            'future_horizon_index': horizon + 1,
            'valid_source_endpoints': count,
            'mean_endpoint_error_cells': mean_epe,
            'zero_flow_endpoint_error_cells': zero_epe,
            'epe_improvement_over_zero_cells': (
                None if mean_epe is None or zero_epe is None
                else zero_epe - mean_epe),
            'mean_predicted_flow_norm_cells': _ratio(
                acc['predicted_norm_sum'][horizon], count),
            'mean_target_flow_norm_cells': _ratio(
                acc['target_norm_sum'][horizon], count),
            'mean_direction_cosine': _ratio(
                acc['cosine_sum'][horizon], direction_count),
            'predicted_endpoint_in_bounds_ratio': _ratio(
                acc['in_bounds_count'][horizon], count),
            'mean_absolute_error_dy_dx': [
                _ratio(acc['absolute_error_sum'][horizon, component], count)
                for component in range(2)
            ],
            'nonzero_target_sign_accuracy_dy_dx': [
                _ratio(
                    acc['sign_match_count'][horizon, component],
                    acc['sign_count'][horizon, component])
                for component in range(2)
            ],
        })
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--prediction-root', type=Path, required=True)
    parser.add_argument(
        '--split', choices=('validation', 'test', 'blind'),
        default='validation')
    parser.add_argument('--out-file', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(str(args.config))
    flow_parameterization = cfg.model.occ_head.get(
        'world_flow_parameterization', 'incremental')
    if flow_parameterization not in ('incremental', 'cumulative_current'):
        raise ValueError(
            f'Unsupported flow parameterization {flow_parameterization}')
    dataset = build_dataset(cfg.data.train)
    manifest = _load_manifest(args.manifest)
    references = _split_references(manifest, args.split)
    predictions = _prediction_mapping(args.prediction_root)
    missing = sorted(set(references).difference(predictions))
    if missing:
        raise ValueError(f'Missing predictions for references {missing}')
    flow_cfg = next(
        dict(item) for item in cfg.train_pipeline
        if item['type'] == 'GenerateOccFlowLabels')
    flow_cfg.pop('type')
    generator = GenerateOccFlowLabels(**flow_cfg)
    acc = _empty_accumulators(horizon_count=4)
    for position, reference in enumerate(references, start=1):
        target, _ = _generate_gt_flow(dataset, generator, reference)
        target = flow_to_world_layout(target[:4], ignore_index=IGNORE_FLOW)
        if flow_parameterization == 'cumulative_current':
            target = compose_incremental_flow_2d(
                target[None], ignore_index=IGNORE_FLOW)[0]
        with np.load(predictions[reference], allow_pickle=False) as archive:
            if PREDICTION_FLOW_KEY not in archive.files:
                raise ValueError(
                    f'{predictions[reference]} has no {PREDICTION_FLOW_KEY}')
            prediction = torch.from_numpy(np.asarray(
                archive[PREDICTION_FLOW_KEY], dtype=np.float32))
        _accumulate(acc, prediction, target)
        print(
            f'[{position}/{len(references)}] reference={reference}')
    result = {
        'schema_version': 1,
        'manifest_name': manifest.get('name'),
        'split': args.split,
        'sample_count': len(references),
        'reference_indices': list(references),
        'coordinate_contract': 'image-aligned H; [dy,dx]',
        'flow_parameterization': flow_parameterization,
        'by_horizon': _summarize(acc),
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
