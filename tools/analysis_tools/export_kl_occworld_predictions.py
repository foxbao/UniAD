#!/usr/bin/env python
"""Export dense OccWorld predictions for frozen manifest splits."""

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel

import projects.mmdet3d_plugin  # noqa: F401
from third_party.uniad_mmdet3d.datasets.builder import (
    build_dataloader,
    build_dataset,
)
from third_party.uniad_mmdet3d.models.builder import build_model


def _checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
    if match is None:
        raise ValueError(f'Checkpoint has no epoch number: {path}')
    return int(match.group(1))


def _discover_checkpoints(args) -> list:
    if args.checkpoint:
        return [args.checkpoint]
    paths = sorted(
        args.checkpoint_dir.glob(args.checkpoint_glob),
        key=_checkpoint_epoch)
    if not paths:
        raise FileNotFoundError(
            f'No checkpoints below {args.checkpoint_dir}')
    return paths


def _reference_indices(dataset):
    return [
        int(dataset.data_infos[raw_index].get('sample_idx', raw_index))
        for raw_index in dataset.valid_data_indices
    ]


def _prediction_filename(label_path: Path) -> str:
    suffix = '__occworld_sequence.npz'
    if not label_path.name.endswith(suffix):
        raise ValueError(f'Unexpected OccWorld label name: {label_path}')
    return label_path.name[:-len(suffix)] + '__occworld_prediction.npz'


def _load_checkpoint(model, path: Path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict


def _reset_test_state(model):
    model.scene_token = None
    model._test_scene_token = None
    model._test_prev_bev = None
    model.test_frame_token = None
    model.timestamp = None
    model.l2g_r_mat = None
    model.l2g_t = None
    model.test_track_instances = None
    model.track_base.clear()


def export_checkpoint(model, wrapped_model, loader, dataset,
                      checkpoint_path: Path, split: str,
                      output_root: Path):
    _load_checkpoint(model, checkpoint_path)
    model.eval()
    _reset_test_state(model)
    epoch = _checkpoint_epoch(checkpoint_path)
    epoch_root = output_root / split / f'epoch_{epoch:03d}'
    references = _reference_indices(dataset)
    if len(references) != len(dataset):
        raise ValueError('Dataset reference mapping is inconsistent')
    output_paths = []
    for position, batch in enumerate(loader):
        reference_index = references[position]
        with torch.no_grad():
            results = wrapped_model(
                return_loss=False, rescale=True, **batch)
        if len(results) != 1 or 'occ' not in results[0]:
            raise RuntimeError(
                f'No OccWorld output for reference {reference_index}')
        occ = results[0]['occ']
        if ('world_pred' not in occ or
                'world_valid_probability' not in occ):
            raise RuntimeError(
                f'Incomplete OccWorld output for {reference_index}')
        prediction = occ['world_pred'].detach().cpu()
        valid_probability = occ[
            'world_valid_probability'].detach().cpu()
        future_change_logits = occ.get('future_change_logits')
        future_change_probability = None
        if future_change_logits is not None:
            future_change_probability = (
                future_change_logits.sigmoid().detach().cpu())
        future_flow = occ.get('future_flow')
        if future_flow is not None:
            future_flow = future_flow.detach().cpu()
        flow_change_prior = occ.get('flow_change_prior')
        if flow_change_prior is not None:
            flow_change_prior = flow_change_prior.detach().cpu()
        changed_class_logits = occ.get('future_changed_class_logits')
        changed_class_prediction = None
        if changed_class_logits is not None:
            changed_class_prediction = changed_class_logits.argmax(
                dim=2).detach().cpu()
        warped_instance_probability = occ.get(
            'warped_instance_probability')
        if warped_instance_probability is not None:
            warped_instance_probability = (
                warped_instance_probability.detach().cpu())
        physical_confidence_logits = occ.get(
            'physical_confidence_logits')
        physical_confidence_probability = None
        if physical_confidence_logits is not None:
            physical_confidence_probability = (
                physical_confidence_logits.sigmoid().detach().cpu())
        if prediction.ndim != 5 or prediction.shape[0] != 1:
            raise ValueError(
                f'Unexpected world prediction shape {prediction.shape}')
        prediction = prediction[0].numpy().astype(np.uint8, copy=False)
        valid_probability = valid_probability[0].numpy().astype(
            np.float32, copy=False)
        if future_change_probability is not None:
            if (future_change_probability.ndim != 5 or
                    future_change_probability.shape[0] != 1):
                raise ValueError(
                    'Unexpected future change probability shape '
                    f'{future_change_probability.shape}')
            future_change_probability = (
                future_change_probability[0].numpy().astype(
                    np.float16, copy=False))
        if future_flow is not None:
            if future_flow.ndim != 5 or future_flow.shape[0] != 1:
                raise ValueError(
                    f'Unexpected future flow shape {future_flow.shape}')
            future_flow = future_flow[0].numpy().astype(
                np.float16, copy=False)
        if flow_change_prior is not None:
            if (flow_change_prior.ndim != 5 or
                    flow_change_prior.shape[0] != 1):
                raise ValueError(
                    'Unexpected flow change prior shape '
                    f'{flow_change_prior.shape}')
            flow_change_prior = flow_change_prior[0].numpy().astype(
                np.float16, copy=False)
        if changed_class_prediction is not None:
            if (changed_class_prediction.ndim != 5 or
                    changed_class_prediction.shape[0] != 1):
                raise ValueError(
                    'Unexpected changed-class prediction shape '
                    f'{changed_class_prediction.shape}')
            changed_class_prediction = (
                changed_class_prediction[0].numpy().astype(
                    np.uint8, copy=False))
        if warped_instance_probability is not None:
            if (warped_instance_probability.ndim != 5 or
                    warped_instance_probability.shape[0] != 1):
                raise ValueError(
                    'Unexpected warped-instance shape '
                    f'{warped_instance_probability.shape}')
            warped_instance_probability = (
                warped_instance_probability[0].numpy().astype(
                    np.float16, copy=False))
        if physical_confidence_probability is not None:
            if (physical_confidence_probability.ndim != 5 or
                    physical_confidence_probability.shape[0] != 1):
                raise ValueError(
                    'Unexpected physical-confidence shape '
                    f'{physical_confidence_probability.shape}')
            physical_confidence_probability = (
                physical_confidence_probability[0].numpy().astype(
                    np.float16, copy=False))
        label_path = dataset.occworld_labels[reference_index]
        output_dir = epoch_root / f'{reference_index:06d}'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / _prediction_filename(label_path)
        payload = dict(
            reference_index=np.int64(reference_index),
            checkpoint_epoch=np.int64(epoch),
            world_pred_class_3d=prediction,
            world_valid_probability_3d=valid_probability)
        if future_change_probability is not None:
            payload['future_change_probability_3d'] = (
                future_change_probability)
        if future_flow is not None:
            payload['future_flow_2d'] = future_flow
        if flow_change_prior is not None:
            payload['flow_change_prior_3d'] = flow_change_prior
        if changed_class_prediction is not None:
            payload['future_changed_class_pred_3d'] = (
                changed_class_prediction)
        if warped_instance_probability is not None:
            payload['warped_instance_probability_3d'] = (
                warped_instance_probability)
        if physical_confidence_probability is not None:
            payload['physical_confidence_probability_3d'] = (
                physical_confidence_probability)
        np.savez_compressed(output_path, **payload)
        output_paths.append(str(output_path))
        del results, occ, prediction, valid_probability, payload
    return {
        'epoch': epoch,
        'checkpoint': str(checkpoint_path),
        'split': split,
        'reference_indices': references,
        'prediction_files': output_paths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path,
        default=Path(
            'projects/configs/stage2_e2e_lidar/'
            'base_e2e_lidar_occworld_split_v1_eval.py'))
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument('--checkpoint', type=Path)
    checkpoint_group.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--checkpoint-glob', default='epoch_*.pth')
    parser.add_argument(
        '--split', choices=(
            'validation', 'test', 'blind', 'final_holdout'),
        default='validation')
    parser.add_argument(
        '--output-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_world_only_v1'))
    parser.add_argument('--workers-per-gpu', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(str(args.config))
    dataset_key = 'test' if args.split == 'test' else 'val'
    dataset_cfg = cfg.data[dataset_key]
    dataset = build_dataset(dataset_cfg)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.CLASSES = dataset.CLASSES
    model = model.cuda()
    wrapped_model = MMDataParallel(model, device_ids=[0])
    summaries = []
    for checkpoint_path in _discover_checkpoints(args):
        summary = export_checkpoint(
            model=model,
            wrapped_model=wrapped_model,
            loader=loader,
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            split=args.split,
            output_root=args.output_root)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    print(json.dumps({
        'config': str(args.config),
        'split': args.split,
        'checkpoint_count': len(summaries),
        'dataset_length': len(dataset),
        'output_root': str(args.output_root),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
