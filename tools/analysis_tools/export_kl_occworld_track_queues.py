#!/usr/bin/env python
"""Export sequential TrackFormer predictions for complete OccWorld queues."""

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
from mmcv.parallel import MMDataParallel

import projects.mmdet3d_plugin  # noqa: F401
from third_party.uniad_mmdet3d.datasets.builder import (
    build_dataloader,
    build_dataset,
)
from third_party.uniad_mmdet3d.models.builder import build_model
from tools.analysis_tools.export_kl_occworld_predictions import (
    _load_checkpoint,
    _reference_indices,
    _reset_test_state,
    _shard_dataset_by_scene,
)
from tools.data_converter.kl_occworld_track_queue import (
    pack_track_queue_results,
)


def _select_references(dataset, requested):
    if requested is None:
        return
    requested = {int(value) for value in requested}
    raw_indices = list(dataset.valid_data_indices)
    positions = [
        position for position, raw_index in enumerate(raw_indices)
        if int(dataset.data_infos[raw_index].get(
            'sample_idx', raw_index)) in requested
    ]
    found = {
        int(dataset.data_infos[raw_indices[position]].get(
            'sample_idx', raw_indices[position]))
        for position in positions
    }
    missing = sorted(requested.difference(found))
    if missing:
        raise ValueError(f'Requested references are not in this split: {missing}')
    dataset.valid_data_indices = [raw_indices[position] for position in positions]
    if hasattr(dataset, 'flag'):
        dataset.flag = dataset.flag[np.asarray(positions, dtype=np.int64)]


def export_track_queues(model, wrapped_model, loader, dataset,
                        output_root: Path, score_threshold: float) -> dict:
    references = _reference_indices(dataset)
    if len(references) != len(dataset):
        raise ValueError('Dataset reference mapping is inconsistent')
    output_paths = []
    frame_counts = []
    kept_counts = []
    for position, batch in enumerate(loader):
        reference_index = references[position]
        _reset_test_state(model)
        with torch.no_grad():
            results = wrapped_model(
                return_loss=False,
                rescale=True,
                export_track_queue=True,
                **batch)
        if len(results) != 1 or 'pts_bbox' not in results[0]:
            raise RuntimeError(
                f'No TrackFormer queue output for {reference_index}')
        queue_results = results[0]['pts_bbox'].get('track_queue_results')
        if queue_results is None:
            raise RuntimeError(
                f'Incomplete TrackFormer queue output for {reference_index}')
        payload = pack_track_queue_results(
            queue_results,
            score_threshold=score_threshold,
            class_count=len(dataset.CLASSES))
        if int(payload['queue_frame_indices'][-1]) != reference_index:
            raise ValueError(
                f'Queue for {reference_index} ends at '
                f"{int(payload['queue_frame_indices'][-1])}")
        output_dir = output_root / f'{reference_index:06d}'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'occworld_track_queue.npz'
        np.savez_compressed(
            output_path,
            reference_index=np.int64(reference_index),
            **payload)
        output_paths.append(str(output_path))
        frame_counts.append(len(queue_results))
        kept_counts.append(payload['track_kept_counts'].tolist())
        print(json.dumps({
            'reference_index': reference_index,
            'queue_frame_indices': payload['queue_frame_indices'].tolist(),
            'track_kept_counts': payload['track_kept_counts'].tolist(),
            'output_path': str(output_path),
        }, ensure_ascii=False))
        del results, queue_results, payload
    return {
        'reference_indices': references,
        'reference_count': len(references),
        'queue_frame_counts': frame_counts,
        'track_kept_counts': kept_counts,
        'output_paths': output_paths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path,
        default=Path(
            'projects/configs/stage2_e2e_lidar/'
            'base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--output-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_track_queues_b15_validation_v1'))
    parser.add_argument('--reference-indices', type=int, nargs='*')
    parser.add_argument('--score-threshold', type=float, default=0.1)
    parser.add_argument('--workers-per-gpu', type=int, default=0)
    parser.add_argument('--summary-file', type=Path)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.score_threshold <= 1:
        raise ValueError('score-threshold must be within [0,1]')
    cfg = Config.fromfile(str(args.config))
    dataset = build_dataset(cfg.data.val)
    _select_references(dataset, args.reference_indices)
    shard_summary = _shard_dataset_by_scene(
        dataset, args.shard_count, args.shard_index)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.CLASSES = dataset.CLASSES
    _load_checkpoint(model, args.checkpoint)
    model = model.cuda().eval()
    wrapped_model = MMDataParallel(model, device_ids=[0])
    summary = export_track_queues(
        model=model,
        wrapped_model=wrapped_model,
        loader=loader,
        dataset=dataset,
        output_root=args.output_root,
        score_threshold=args.score_threshold)
    summary.update({
        'config': str(args.config),
        'checkpoint': str(args.checkpoint),
        'score_threshold': args.score_threshold,
        'output_root': str(args.output_root),
        'shard': shard_summary,
    })
    if args.summary_file is not None:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
