#!/usr/bin/env python
"""Audit B24 event-reliability changes against paired raw predictions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.build_kl_occworld_scene_split import (  # noqa: E402
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (  # noqa: E402
    _load_manifest,
    _prediction_mapping,
    _split_references,
)


COUNT_KEYS = (
    'changed_voxels', 'improved_voxels', 'harmed_voxels',
    'still_wrong_voxels', 'unknown_target_voxels', 'net_correct_voxels')


def _empty_counts():
    return {key: 0 for key in COUNT_KEYS}


def _add_counts(destination, source):
    for key in COUNT_KEYS:
        destination[key] += int(source[key])


def _correction_counts(raw, final, target, target_known, selection=None):
    if selection is None:
        selection = np.ones_like(target_known, dtype=np.bool_)
    changed = (raw != final) & selection
    known_changed = changed & target_known
    improved = known_changed & (raw != target) & (final == target)
    harmed = known_changed & (raw == target) & (final != target)
    still_wrong = known_changed & (raw != target) & (final != target)
    result = {
        'changed_voxels': int(np.count_nonzero(changed)),
        'improved_voxels': int(np.count_nonzero(improved)),
        'harmed_voxels': int(np.count_nonzero(harmed)),
        'still_wrong_voxels': int(np.count_nonzero(still_wrong)),
        'unknown_target_voxels': int(np.count_nonzero(
            changed & ~target_known)),
    }
    result['net_correct_voxels'] = (
        result['improved_voxels'] - result['harmed_voxels'])
    return result


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {
            'count': 0, 'mean': None, 'p10': None, 'median': None,
            'p90': None}
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'p10': float(np.percentile(values, 10)),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
    }


def _target(path):
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive['world_target_state_3d'], dtype=np.uint8)
        valid = np.asarray(archive['world_target_valid_3d'], dtype=np.bool_)
        current_state = np.asarray(
            archive['current_observation_state_3d'], dtype=np.uint8)
        current_valid = np.asarray(
            archive['current_observation_valid_3d'], dtype=np.bool_)
    known = valid & (state != 0)
    target = np.zeros_like(state, dtype=np.uint8)
    target[known] = state[known] - 1
    current_known = current_valid & (current_state != 0)
    current_class = np.zeros_like(current_state, dtype=np.uint8)
    current_class[current_known] = current_state[current_known] - 1
    return target, known, current_class, current_known


def _prediction(path, reference, expected_epoch):
    required = (
        'reference_index', 'checkpoint_epoch', 'world_pred_class_3d',
        'raw_world_pred_class_3d', 'event_reliability_alpha_3d',
        'event_candidate_class_3d', 'event_candidate_mask_3d')
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required).difference(archive.files))
        if missing:
            raise ValueError(f'{path} lacks B24 diagnostics: {missing}')
        payload = {key: np.array(archive[key], copy=True) for key in required}
    if int(payload['reference_index']) != reference:
        raise ValueError(f'Prediction reference mismatch for {reference}')
    if int(payload['checkpoint_epoch']) != expected_epoch:
        raise ValueError(
            f'Expected epoch {expected_epoch}, got '
            f"{int(payload['checkpoint_epoch'])} for {reference}")
    return payload


def audit(manifest, sequences, predictions, split, expected_epoch):
    references = _split_references(manifest, split)
    if set(references) != set(predictions):
        raise ValueError('Predictions must exactly cover the requested split')
    if set(references).difference(sequences):
        raise ValueError('Sequence labels do not cover the requested split')
    scene_by_reference = {
        int(row['reference_index']): str(row['scene_token'])
        for row in manifest['splits'][split]
    }
    total = _empty_counts()
    by_event = {name: _empty_counts() for name in ('arrival', 'departure')}
    by_subset = {
        name: _empty_counts() for name in (
            'current_visible', 'reveal', 'state_change',
            'visible_transition', 'stable_current_visible')}
    by_horizon = [_empty_counts() for _ in range(4)]
    by_scene = {}
    alpha_chunks = {
        name: [] for name in (
            'all_candidates', 'known_candidates', 'candidate_should_win',
            'raw_should_win', 'both_wrong')}
    candidate_count = 0
    known_candidate_count = 0
    current_difference_voxels = 0
    non_event_difference_voxels = 0

    for reference in references:
        payload = _prediction(
            predictions[reference], reference, expected_epoch)
        target, known, current_class, current_known = _target(
            sequences[reference])
        raw = np.asarray(payload['raw_world_pred_class_3d'], dtype=np.uint8)
        final = np.asarray(payload['world_pred_class_3d'], dtype=np.uint8)
        alpha = np.asarray(
            payload['event_reliability_alpha_3d'], dtype=np.float32)
        candidate_class = np.asarray(
            payload['event_candidate_class_3d'], dtype=np.uint8)
        candidate_mask = np.asarray(
            payload['event_candidate_mask_3d'], dtype=np.bool_)
        expected_future = raw[1:].shape
        if (final.shape != raw.shape or target.shape != raw.shape or
                alpha.shape != expected_future or
                candidate_class.shape != expected_future or
                candidate_mask.shape != expected_future):
            raise ValueError(f'B24 diagnostic shape mismatch for {reference}')

        raw_future = raw[1:]
        final_future = final[1:]
        target_future = target[1:]
        known_future = known[1:]
        changed = raw_future != final_future
        current_difference_voxels += int(np.count_nonzero(raw[0] != final[0]))
        non_event_difference_voxels += int(np.count_nonzero(
            changed & ~candidate_mask))
        candidate_count += int(np.count_nonzero(candidate_mask))
        known_candidate = candidate_mask & known_future
        known_candidate_count += int(np.count_nonzero(known_candidate))

        local_counts = _correction_counts(
            raw_future, final_future, target_future, known_future)
        _add_counts(total, local_counts)
        scene = scene_by_reference[reference]
        by_scene.setdefault(scene, _empty_counts())
        _add_counts(by_scene[scene], local_counts)

        current_known_future = np.broadcast_to(
            current_known, target_future.shape)
        persistence = np.broadcast_to(current_class, target_future.shape)
        state_change = known_future & (target_future != persistence)
        subsets = {
            'current_visible': known_future & current_known_future,
            'reveal': known_future & ~current_known_future,
            'state_change': state_change,
            'visible_transition': state_change & current_known_future,
            'stable_current_visible': (
                known_future & current_known_future & ~state_change),
        }
        for name, selection in subsets.items():
            _add_counts(by_subset[name], _correction_counts(
                raw_future, final_future, target_future, known_future,
                selection))
        for horizon in range(4):
            selection = np.zeros_like(known_future)
            selection[horizon] = True
            _add_counts(by_horizon[horizon], _correction_counts(
                raw_future, final_future, target_future, known_future,
                selection))

        arrival = candidate_mask & (candidate_class == 2)
        departure = candidate_mask & (candidate_class == 0)
        for name, selection in (('arrival', arrival),
                                ('departure', departure)):
            _add_counts(by_event[name], _correction_counts(
                raw_future, final_future, target_future, known_future,
                selection))

        raw_correct = raw_future == target_future
        candidate_correct = candidate_class == target_future
        alpha_chunks['all_candidates'].append(alpha[candidate_mask])
        alpha_chunks['known_candidates'].append(alpha[known_candidate])
        alpha_chunks['candidate_should_win'].append(alpha[
            known_candidate & ~raw_correct & candidate_correct])
        alpha_chunks['raw_should_win'].append(alpha[
            known_candidate & raw_correct & ~candidate_correct])
        alpha_chunks['both_wrong'].append(alpha[
            known_candidate & ~raw_correct & ~candidate_correct])

    scene_rows = [
        {'scene_token': scene, **counts}
        for scene, counts in by_scene.items()
    ]
    scene_rows.sort(key=lambda row: (
        row['net_correct_voxels'], row['scene_token']))
    alpha_summary = {
        name: _distribution(np.concatenate(chunks) if chunks else [])
        for name, chunks in alpha_chunks.items()
    }
    return {
        'schema_version': 1,
        'purpose': 'B24 paired raw/final event-reliability audit',
        'split': split,
        'expected_checkpoint_epoch': int(expected_epoch),
        'reference_count': len(references),
        'scene_count': len(by_scene),
        'current_difference_voxels': current_difference_voxels,
        'non_event_difference_voxels': non_event_difference_voxels,
        'candidate_voxels': candidate_count,
        'known_candidate_voxels': known_candidate_count,
        'corrections': total,
        'corrections_by_event': by_event,
        'corrections_by_subset': by_subset,
        'corrections_by_horizon': by_horizon,
        'alpha_distribution': alpha_summary,
        'scene_outcome_counts': {
            'positive': sum(
                row['net_correct_voxels'] > 0 for row in scene_rows),
            'negative': sum(
                row['net_correct_voxels'] < 0 for row in scene_rows),
            'zero': sum(
                row['net_correct_voxels'] == 0 for row in scene_rows),
        },
        'rows_worst_first': scene_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sequence-root', type=Path, required=True)
    parser.add_argument('--prediction-root', type=Path, required=True)
    parser.add_argument('--split', default='internal_dev')
    parser.add_argument('--expected-checkpoint-epoch', type=int, required=True)
    parser.add_argument('--out-file', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = audit(
        manifest=_load_manifest(args.manifest),
        sequences=_sequence_mapping(args.sequence_root),
        predictions=_prediction_mapping(args.prediction_root),
        split=args.split,
        expected_epoch=args.expected_checkpoint_epoch)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
