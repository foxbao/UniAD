#!/usr/bin/env python
"""Inspect learned lane-anchor gate/residual behavior on eval samples.

This script runs a small validation pass and summarizes whether the learned
lane-anchor path is active, how large its gate is, and how far the final plan
moves from the base planner toward the HD-map anchor.
"""

from __future__ import annotations

import argparse
import csv
import os
import os.path as osp
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from third_party.uniad_mmdet3d.datasets.builder import (  # noqa: E402
    build_dataloader, build_dataset)
from third_party.uniad_mmdet3d.models.builder import build_model  # noqa: E402
from projects.mmdet3d_plugin.uniad.apis.test import (  # noqa: E402
    _planning_tensor)


BUCKET_ORDER = ('all', 'static', 'slow', 'moving', 'turning')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Diagnose learned lane-anchor gate/residual statistics.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, default=512)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--static-thr', type=float, default=0.5)
    parser.add_argument('--slow-thr', type=float, default=2.0)
    parser.add_argument('--turn-lateral-thr', type=float, default=0.15)
    parser.add_argument('--turn-y-thr', type=float, default=1.0)
    return parser.parse_args()


def to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return _planning_tensor(value).detach().cpu().numpy()


def first_traj(value, channels=2):
    arr = to_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        arr = arr.reshape(-1, *arr.shape[-2:])
    return arr[:, :, :channels]


def scalar_array(value):
    arr = to_numpy(value)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def final_disp(traj: np.ndarray) -> np.ndarray:
    return np.linalg.norm(traj[:, -1, :2], axis=-1)


def classify_gt(gt_traj: np.ndarray, args: argparse.Namespace) -> str:
    if gt_traj is None or len(gt_traj) == 0:
        return 'all'
    traj = gt_traj[0]
    disp = float(np.linalg.norm(traj[-1, :2]))
    if disp < args.static_thr:
        return 'static'
    if disp < args.slow_thr:
        return 'slow'
    lateral = float(np.max(np.abs(traj[:, 1])))
    if lateral >= args.turn_y_thr:
        return 'turning'
    net = float(np.linalg.norm(traj[-1, :2]))
    if net > 1e-6:
        direction = traj[-1, :2] / net
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        ratio = float(np.max(np.abs(traj[:, :2] @ normal)) / net)
        if ratio >= args.turn_lateral_thr:
            return 'turning'
    return 'moving'


def add_stats(rows: List[Dict[str, float]], bucket: str, stats: Dict[str, list]):
    if not stats['gate']:
        return
    row = {'bucket': bucket, 'n': len(stats['gate'])}
    for key, values in stats.items():
        arr = np.asarray(values, dtype=np.float64)
        row[f'{key}_mean'] = float(np.mean(arr))
        row[f'{key}_p10'] = float(np.percentile(arr, 10))
        row[f'{key}_p50'] = float(np.percentile(arr, 50))
        row[f'{key}_p90'] = float(np.percentile(arr, 90))
    rows.append(row)


def write_csv(path: str, rows: Iterable[dict]):
    rows = list(rows)
    os.makedirs(osp.dirname(path), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as f:
            f.write('')
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_loaded_model(cfg: Config, checkpoint: str, device: str):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model.to(device), device_ids=[0])
    model.eval()
    return model


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = getattr(cfg, 'plugin_dir', None)
        if plugin_dir:
            module_path = '.'.join(osp.dirname(plugin_dir).split('/'))
            importlib.import_module(module_path)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    model = build_loaded_model(cfg, args.checkpoint, args.device)

    per_sample = []
    buckets = {
        name: {
            'gate': [], 'residual_l2': [], 'anchor_delta_l2': [],
            'final_delta_l2': [], 'base_final_disp': [], 'final_disp': [],
            'anchor_final_disp': [],
        } for name in BUCKET_ORDER
    }

    for idx, data in enumerate(loader):
        if idx >= args.max_samples:
            break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
        planning = result.get('planning', {})
        result_planning = planning.get('result_planning', {})
        planning_gt = planning.get('planning_gt', {})
        gate = scalar_array(result_planning.get('lane_anchor_gate'))
        base = first_traj(result_planning.get('sdc_traj_base'))
        final = first_traj(result_planning.get('sdc_traj'))
        anchor = first_traj(result_planning.get('lane_anchor'))
        residual = first_traj(result_planning.get('lane_anchor_residual'))
        gt = first_traj(planning_gt.get('sdc_planning'), channels=2)
        if gate is None or base is None or final is None or anchor is None:
            continue

        bucket = classify_gt(gt, args)
        residual_l2 = 0.0 if residual is None else float(
            np.linalg.norm(residual.reshape(-1, 2), axis=-1).mean())
        anchor_delta_l2 = float(
            np.linalg.norm((anchor - base).reshape(-1, 2), axis=-1).mean())
        final_delta_l2 = float(
            np.linalg.norm((final - base).reshape(-1, 2), axis=-1).mean())
        row = dict(
            index=idx,
            bucket=bucket,
            gate=float(gate[0]),
            residual_l2=residual_l2,
            anchor_delta_l2=anchor_delta_l2,
            final_delta_l2=final_delta_l2,
            base_final_disp=float(final_disp(base)[0]),
            final_disp=float(final_disp(final)[0]),
            anchor_final_disp=float(final_disp(anchor)[0]),
        )
        per_sample.append(row)
        for name in ('all', bucket):
            for key in buckets[name]:
                buckets[name][key].append(row[key])

    summary = []
    for bucket in BUCKET_ORDER:
        add_stats(summary, bucket, buckets[bucket])

    write_csv(osp.join(args.out_dir, 'lane_anchor_gate_samples.csv'),
              per_sample)
    write_csv(osp.join(args.out_dir, 'lane_anchor_gate_summary.csv'),
              summary)
    print(f'wrote {len(per_sample)} samples to {args.out_dir}')
    for row in summary:
        print(
            f"{row['bucket']}: n={row['n']} "
            f"gate_mean={row['gate_mean']:.4f} "
            f"gate_p50={row['gate_p50']:.4f} "
            f"final_delta_l2_mean={row['final_delta_l2_mean']:.4f} "
            f"residual_l2_mean={row['residual_l2_mean']:.4f}")


if __name__ == '__main__':
    main()
