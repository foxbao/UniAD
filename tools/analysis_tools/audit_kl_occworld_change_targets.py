#!/usr/bin/env python
"""Audit provenance and transitions of KL OccWorld change targets."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.data_converter.kl_occworld_dataset import (
    KLOccWorldSequenceDataset,
)


STATE_NAMES = ('unknown', 'free', 'static_occupied', 'instance_occupied')
SOURCE_NAMES = (
    'unknown', 'direct_observation',
    'future_repeated_free', 'future_repeated_static')


def _empty_counts(horizon_count: int):
    return {
        'changed': np.zeros(horizon_count, dtype=np.int64),
        'valid': np.zeros(horizon_count, dtype=np.int64),
        'current_visible': np.zeros((horizon_count, 2), dtype=np.int64),
        'reveal_detection': np.zeros(
            (horizon_count, 2), dtype=np.int64),
        'visible_transition_detection': np.zeros(
            (horizon_count, 2), dtype=np.int64),
        'reveal_target_state': np.zeros(
            (horizon_count, len(STATE_NAMES)), dtype=np.int64),
        'visible_transition_target_state': np.zeros(
            (horizon_count, len(STATE_NAMES)), dtype=np.int64),
        'completion_source': np.zeros(
            (horizon_count, len(SOURCE_NAMES)), dtype=np.int64),
        'target_state': np.zeros(
            (horizon_count, len(STATE_NAMES)), dtype=np.int64),
        'transition': np.zeros(
            (horizon_count, len(STATE_NAMES), len(STATE_NAMES)),
            dtype=np.int64),
    }


def _add_sample(counts, sample):
    current = sample['input_state_3d'].numpy()
    target = sample['target_state_3d'].numpy()
    valid = sample['target_valid_3d'].numpy()
    with np.load(sample['label_path'], allow_pickle=False) as label:
        source = label['completion_source_3d'][1:]
    persistence_class = np.zeros_like(current, dtype=np.int64)
    persistence_class[current != 0] = current[current != 0] - 1
    persistence_class = np.broadcast_to(
        persistence_class, target.shape)
    target_class = target.astype(np.int64) - 1
    changed = valid & (target_class != persistence_class)
    for horizon in range(target.shape[0]):
        horizon_changed = changed[horizon]
        counts['changed'][horizon] += int(horizon_changed.sum())
        counts['valid'][horizon] += int(valid[horizon].sum())
        counts['current_visible'][horizon, 0] += int(
            (horizon_changed & (current == 0)).sum())
        counts['current_visible'][horizon, 1] += int(
            (horizon_changed & (current != 0)).sum())
        current_unknown = current == 0
        reveal_positive = current_unknown & valid[horizon]
        counts['reveal_detection'][horizon] += np.asarray([
            int(reveal_positive.sum()),
            int((current_unknown & ~valid[horizon]).sum()),
        ], dtype=np.int64)
        visible_valid = (current != 0) & valid[horizon]
        visible_transition = visible_valid & (
            target_class[horizon] != persistence_class[horizon])
        counts['visible_transition_detection'][horizon] += np.asarray([
            int(visible_transition.sum()),
            int((visible_valid & ~visible_transition).sum()),
        ], dtype=np.int64)
        for target_state in range(len(STATE_NAMES)):
            target_is_state = target[horizon] == target_state
            counts['reveal_target_state'][horizon, target_state] += int(
                (reveal_positive & target_is_state).sum())
            counts['visible_transition_target_state'][
                horizon, target_state] += int(
                    (visible_transition & target_is_state).sum())
        for source_id in range(len(SOURCE_NAMES)):
            counts['completion_source'][horizon, source_id] += int(
                (horizon_changed & (source[horizon] == source_id)).sum())
        for target_state in range(len(STATE_NAMES)):
            counts['target_state'][horizon, target_state] += int(
                (horizon_changed &
                 (target[horizon] == target_state)).sum())
        for current_state in range(len(STATE_NAMES)):
            current_mask = current == current_state
            for target_state in range(len(STATE_NAMES)):
                counts['transition'][horizon, current_state, target_state] += int(
                    (horizon_changed & current_mask &
                     (target[horizon] == target_state)).sum())


def _ratio_mapping(values: np.ndarray, names, total: int):
    return {
        name: {
            'count': int(value),
            'ratio': float(value / total) if total else 0.0,
        }
        for name, value in zip(names, values)
    }


def _summarize(counts):
    total_changed = int(counts['changed'].sum())
    total_valid = int(counts['valid'].sum())
    transition = counts['transition'].sum(axis=0)
    transition_rows = {}
    for current_state, current_name in enumerate(STATE_NAMES):
        for target_state, target_name in enumerate(STATE_NAMES):
            value = int(transition[current_state, target_state])
            if value:
                transition_rows[f'{current_name}->{target_name}'] = {
                    'count': value,
                    'ratio': value / total_changed,
                }
    per_horizon = []
    for horizon in range(len(counts['changed'])):
        changed = int(counts['changed'][horizon])
        valid = int(counts['valid'][horizon])
        reveal = counts['reveal_detection'][horizon]
        reveal_total = int(reveal.sum())
        transition = counts['visible_transition_detection'][horizon]
        transition_total = int(transition.sum())
        per_horizon.append({
            'horizon_index': horizon,
            'changed_voxels': changed,
            'valid_voxels': valid,
            'change_ratio': changed / valid if valid else 0.0,
            'current_visibility': _ratio_mapping(
                counts['current_visible'][horizon],
                ('current_unknown', 'current_visible'), changed),
            'reveal_detection': _ratio_mapping(
                reveal, ('future_known', 'future_unknown'), reveal_total),
            'visible_transition_detection': _ratio_mapping(
                transition, ('changed', 'unchanged'), transition_total),
            'reveal_target_state': _ratio_mapping(
                counts['reveal_target_state'][horizon],
                STATE_NAMES, int(reveal[0])),
            'visible_transition_target_state': _ratio_mapping(
                counts['visible_transition_target_state'][horizon],
                STATE_NAMES, int(transition[0])),
            'completion_source': _ratio_mapping(
                counts['completion_source'][horizon],
                SOURCE_NAMES, changed),
            'target_state': _ratio_mapping(
                counts['target_state'][horizon],
                STATE_NAMES, changed),
        })
    reveal = counts['reveal_detection'].sum(axis=0)
    reveal_total = int(reveal.sum())
    transition = counts['visible_transition_detection'].sum(axis=0)
    transition_total = int(transition.sum())
    return {
        'changed_voxels': total_changed,
        'valid_voxels': total_valid,
        'change_ratio': (
            total_changed / total_valid if total_valid else 0.0),
        'current_visibility': _ratio_mapping(
            counts['current_visible'].sum(axis=0),
            ('current_unknown', 'current_visible'), total_changed),
        'reveal_detection': _ratio_mapping(
            reveal, ('future_known', 'future_unknown'), reveal_total),
        'visible_transition_detection': _ratio_mapping(
            transition, ('changed', 'unchanged'), transition_total),
        'reveal_target_state': _ratio_mapping(
            counts['reveal_target_state'].sum(axis=0),
            STATE_NAMES, int(reveal[0])),
        'visible_transition_target_state': _ratio_mapping(
            counts['visible_transition_target_state'].sum(axis=0),
            STATE_NAMES, int(transition[0])),
        'completion_source': _ratio_mapping(
            counts['completion_source'].sum(axis=0),
            SOURCE_NAMES, total_changed),
        'target_state': _ratio_mapping(
            counts['target_state'].sum(axis=0),
            STATE_NAMES, total_changed),
        'transitions': transition_rows,
        'per_horizon': per_horizon,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root', required=True)
    parser.add_argument('--history-root', required=True)
    parser.add_argument('--train-count', type=int, default=10)
    parser.add_argument('--validation-count', type=int, default=2)
    parser.add_argument('--out-file', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = KLOccWorldSequenceDataset(
        args.sequence_root,
        expected_shape=(10, 120, 160),
        history_root=args.history_root,
        drop_missing_history=True)
    test_start = args.train_count + args.validation_count
    if (args.train_count < 1 or args.validation_count < 1 or
            test_start >= len(dataset)):
        raise ValueError('Split must leave train, validation and test samples')
    split_indices = {
        'all': list(range(len(dataset))),
        'train': list(range(args.train_count)),
        'validation': list(range(args.train_count, test_start)),
        'test': list(range(test_start, len(dataset))),
    }
    result = {}
    for split, indices in split_indices.items():
        counts = _empty_counts(
            dataset[indices[0]]['target_state_3d'].shape[0])
        references = []
        for index in indices:
            sample = dataset[index]
            _add_sample(counts, sample)
            references.append(int(sample['reference_index']))
        result[split] = {
            'reference_indices': references,
            **_summarize(counts),
        }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
