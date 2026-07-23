#!/usr/bin/env python
"""Run one no-backward OccWorld training forward through UniADMotionLidar."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel

import projects.mmdet3d_plugin  # noqa: F401
from third_party.uniad_mmdet3d.datasets.builder import (
    build_dataloader,
    build_dataset,
)
from third_party.uniad_mmdet3d.models.builder import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default=(
            'projects/configs/stage2_e2e_lidar/'
            'base_e2e_lidar_occworld.py'))
    parser.add_argument('--checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint_path = args.checkpoint or cfg.load_from
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_result = model.load_state_dict(
        checkpoint['state_dict'], strict=False)
    del checkpoint
    non_world_missing = [
        key for key in load_result.missing_keys
        if key.startswith('occ_head.') and
        not key.startswith('occ_head.world_')]
    if non_world_missing:
        raise RuntimeError(
            f'Existing OccHead weights did not load: {non_world_missing}')
    model.CLASSES = dataset.CLASSES
    model.train()
    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    batch = next(iter(loader))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        losses = wrapped(return_loss=True, **batch)
    occ_losses = {
        key: float(value.detach().mean().cpu())
        for key, value in losses.items()
        if key.startswith('occ.')
    }
    required = {
        'occ.loss_dice', 'occ.loss_mask',
        'occ.loss_aux_dice', 'occ.loss_aux_mask',
        'occ.loss_world_current_ce',
        'occ.loss_world_future_ce',
        'occ.loss_world_visibility'}
    occ_head_cfg = cfg.model.get('occ_head', {})
    if occ_head_cfg.get('world_future_change_gate_loss_weight', 0) > 0:
        required.update({
            'occ.loss_world_future_change_gate',
            'occ.loss_world_future_changed_class_ce',
        })
    if occ_head_cfg.get('world_flow_loss_weight', 0) > 0:
        required.add('occ.loss_world_flow')
    if occ_head_cfg.get('world_physical_confidence_loss_weight', 0) > 0:
        required.add('occ.loss_world_physical_confidence')
    missing_losses = sorted(required.difference(occ_losses))
    if missing_losses:
        raise RuntimeError(
            f'OccWorld forward missed losses: {missing_losses}')
    if not all(torch.isfinite(torch.tensor(value))
               for value in occ_losses.values()):
        raise FloatingPointError(f'Non-finite losses: {occ_losses}')
    world_target = batch['gt_world_occ'].data[0]
    world_valid = batch['gt_world_valid'].data[0]
    result = {
        'config': args.config,
        'checkpoint': checkpoint_path,
        'dataset_type': type(dataset).__name__,
        'dataset_length': len(dataset),
        'reference_indices': dataset.valid_data_indices,
        'world_target_shape': list(world_target.shape),
        'world_valid_voxels': int(world_valid.sum()),
        'occ_losses': occ_losses,
        'peak_gpu_memory_mb': (
            torch.cuda.max_memory_allocated() / (1024 ** 2)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
