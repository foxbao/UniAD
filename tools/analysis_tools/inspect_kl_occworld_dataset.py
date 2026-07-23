#!/usr/bin/env python
"""Inspect the PyTorch OccWorld dataset contract and class balance."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.data_converter.kl_occworld_dataset import (
    FREE,
    IGNORE_INDEX,
    INSTANCE_OCCUPIED,
    KLOccWorldSequenceDataset,
    STATIC_OCCUPIED,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root', required=True)
    parser.add_argument('--history-root')
    parser.add_argument('--drop-missing-history', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--target-start', type=int, default=1)
    parser.add_argument('--target-end', type=int)
    parser.add_argument('--out-file', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('Invalid DataLoader parameters')
    dataset = KLOccWorldSequenceDataset(
        args.sequence_root,
        target_start=args.target_start,
        target_end=args.target_end,
        expected_shape=(10, 120, 160),
        history_root=args.history_root,
        drop_missing_history=args.drop_missing_history,
    )
    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(sample_count, max(0, args.max_samples))
        dataset.label_paths = dataset.label_paths[:sample_count]
        dataset.reference_indices = dataset.reference_indices[:sample_count]
        dataset.current_shapes = dataset.current_shapes[:sample_count]
        if dataset.history_paths is not None:
            dataset.history_paths = dataset.history_paths[:sample_count]
    if not sample_count:
        raise ValueError('No samples selected')

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)
    first_batch = next(iter(loader))
    target_count = first_batch['target_class_3d'].shape[1]
    input_one_hot_shape = list(first_batch['input_one_hot_3d'].shape)
    target_class_shape = list(first_batch['target_class_3d'].shape)
    first_target_times = first_batch['target_times_s'][0].tolist()
    history_one_hot_shape = None
    first_history_times = None
    if args.history_root:
        history_one_hot_shape = list(
            first_batch['history_one_hot_3d'].shape)
        first_history_times = first_batch['history_times_s'][0].tolist()
    del first_batch
    class_counts = np.zeros((target_count, 3), dtype=np.int64)
    valid_counts = np.zeros(target_count, dtype=np.int64)
    total_voxels = np.zeros(target_count, dtype=np.int64)
    max_one_hot_error = 0.0
    max_history_one_hot_error = 0.0
    history_valid_counts = None
    history_total_voxels = None
    input_history_current_mismatch = 0
    sequence_current_difference = 0
    for batch in loader:
        one_hot = batch['input_one_hot_3d']
        max_one_hot_error = max(
            max_one_hot_error,
            float(torch.abs(one_hot.sum(dim=1) - 1).max()))
        if args.history_root:
            history_one_hot = batch['history_one_hot_3d']
            max_history_one_hot_error = max(
                max_history_one_hot_error,
                float(torch.abs(history_one_hot.sum(dim=2) - 1).max()))
            history_valid = batch['history_valid_3d']
            per_history_valid = history_valid.sum(dim=(0, 2, 3, 4)).cpu().numpy()
            per_history_total = np.full(
                history_valid.shape[1],
                history_valid.shape[0] * np.prod(history_valid.shape[2:]),
                dtype=np.int64)
            if history_valid_counts is None:
                history_valid_counts = np.zeros_like(
                    per_history_valid, dtype=np.int64)
                history_total_voxels = np.zeros_like(
                    per_history_total, dtype=np.int64)
            history_valid_counts += per_history_valid.astype(np.int64)
            history_total_voxels += per_history_total
            input_history_current_mismatch += int(torch.count_nonzero(
                batch['input_state_3d'] - batch['history_state_3d'][:, -1]))
            sequence_current_difference += int(torch.count_nonzero(
                batch['input_state_3d'] -
                batch['sequence_current_state_3d']))
        target = batch['target_class_3d']
        for horizon in range(target_count):
            values = target[:, horizon]
            known = values != IGNORE_INDEX
            valid_counts[horizon] += int(known.sum())
            total_voxels[horizon] += int(values.numel())
            for class_id in range(3):
                class_counts[horizon, class_id] += int(
                    (values == class_id).sum())

    total_class_counts = class_counts.sum(axis=0)
    total_known = int(total_class_counts.sum())
    class_frequencies = total_class_counts / max(total_known, 1)
    inverse_sqrt_weights = 1.0 / np.sqrt(
        np.maximum(class_frequencies, np.finfo(np.float64).eps))
    inverse_sqrt_weights /= inverse_sqrt_weights.mean()

    summary = {
        'sample_count': sample_count,
        'batch_size': args.batch_size,
        'target_start': args.target_start,
        'target_end': args.target_end,
        'input_one_hot_shape': input_one_hot_shape,
        'target_class_shape': target_class_shape,
        'target_times_s_first_sample': first_target_times,
        'max_input_one_hot_sum_error': max_one_hot_error,
        'valid_ratio_by_horizon': [
            float(valid / total) if total else 0.0
            for valid, total in zip(valid_counts, total_voxels)],
        'class_counts_by_horizon': {
            'free': class_counts[:, 0].tolist(),
            'static_occupied': class_counts[:, 1].tolist(),
            'instance_occupied': class_counts[:, 2].tolist(),
        },
        'total_class_counts': {
            'free': int(total_class_counts[0]),
            'static_occupied': int(total_class_counts[1]),
            'instance_occupied': int(total_class_counts[2]),
        },
        'class_frequencies': {
            'free': float(class_frequencies[0]),
            'static_occupied': float(class_frequencies[1]),
            'instance_occupied': float(class_frequencies[2]),
        },
        'inverse_sqrt_class_weights_mean_1': {
            'free': float(inverse_sqrt_weights[0]),
            'static_occupied': float(inverse_sqrt_weights[1]),
            'instance_occupied': float(inverse_sqrt_weights[2]),
        },
        'loss_class_mapping': {
            'free': FREE - 1,
            'static_occupied': STATIC_OCCUPIED - 1,
            'instance_occupied': INSTANCE_OCCUPIED - 1,
            'unknown_ignore_index': IGNORE_INDEX,
        },
    }
    if args.history_root:
        summary['history'] = {
            'history_root': args.history_root,
            'history_one_hot_shape': history_one_hot_shape,
            'history_times_s_first_sample': first_history_times,
            'max_history_one_hot_sum_error': max_history_one_hot_error,
            'valid_ratio_by_history': [
                float(valid / total) if total else 0.0
                for valid, total in zip(
                    history_valid_counts, history_total_voxels)],
            'input_history_current_mismatch_voxels': (
                input_history_current_mismatch),
            'input_sequence_current_difference_voxels': (
                sequence_current_difference),
        }
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        with args.out_file.open('w') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
