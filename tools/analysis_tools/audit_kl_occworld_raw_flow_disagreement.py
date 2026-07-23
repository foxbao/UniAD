#!/usr/bin/env python
"""Audit raw/flow disagreements and causal flow-gating signals."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from projects.mmdet3d_plugin.uniad.dense_heads.occworld_head import (
    apply_physical_flow_fusion,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    IGNORE_INDEX,
    PREDICTION_CHANGE_PROBABILITY_KEY,
    PREDICTION_CLASS_KEY,
    PREDICTION_VALID_PROBABILITY_KEY,
    PREDICTION_WARPED_INSTANCE_KEY,
    _class_target,
    _load_manifest,
    _prediction_mapping,
    _persistence_prediction,
    _semantic_summary,
    _split_references,
    _update_confusion,
)


PREDICTION_FLOW_KEY = 'future_flow_2d'


def apply_flow_events_to_raw(
        raw_prediction: torch.Tensor,
        observation_class: torch.Tensor,
        observation_known: torch.Tensor,
        warped_instance_probability: torch.Tensor,
        event_threshold: float,
        change_probability: torch.Tensor = None,
        change_threshold: float = 0.0,
        flow_norm: torch.Tensor = None,
        flow_norm_threshold: float = 0.0) -> Tuple[
            torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply causal flow events while keeping raw semantics as the base."""
    if raw_prediction.ndim != 4 or warped_instance_probability.ndim != 4:
        raise ValueError('Expected raw [T,Z,H,W] and warped [T-1,Z,H,W]')
    horizon_count, z_count, height, width = raw_prediction.shape
    if tuple(warped_instance_probability.shape) != (
            horizon_count - 1, z_count, height, width):
        raise ValueError('Raw and warped shapes do not match')
    if (tuple(observation_class.shape) != (z_count, height, width) or
            tuple(observation_known.shape) != (z_count, height, width)):
        raise ValueError('Observation shapes do not match raw prediction')
    if not 0.0 < event_threshold <= 1.0:
        raise ValueError('Event threshold must be in (0, 1]')
    if not 0.0 <= change_threshold <= 1.0:
        raise ValueError('Change threshold must be in [0, 1]')
    if flow_norm_threshold < 0:
        raise ValueError('Flow-norm threshold must be non-negative')
    if change_probability is not None and tuple(change_probability.shape) != (
            horizon_count - 1, z_count, height, width):
        raise ValueError('Change probability shape does not match raw')
    if flow_norm is not None and tuple(flow_norm.shape) != (
            horizon_count - 1, height, width):
        raise ValueError('Flow norm shape does not match raw')

    current_instance = (
        observation_known.bool() & (observation_class == 2))
    current_instance = current_instance[None].to(
        warped_instance_probability.dtype)
    arrival_strength = (
        warped_instance_probability - current_instance).clamp(min=0)
    departure_strength = (
        current_instance - warped_instance_probability).clamp(min=0)
    if change_probability is None:
        change_gate = torch.ones_like(arrival_strength, dtype=torch.bool)
    else:
        change_gate = change_probability >= change_threshold
    if flow_norm is not None:
        change_gate = change_gate & (
            flow_norm[:, None] >= flow_norm_threshold)
    arrival = (arrival_strength >= event_threshold) & change_gate
    departure = (departure_strength >= event_threshold) & change_gate
    fused = raw_prediction.clone()
    fused[1:] = torch.where(
        arrival, torch.full_like(fused[1:], 2), fused[1:])
    fused[1:] = torch.where(
        departure, torch.zeros_like(fused[1:]), fused[1:])
    return fused, {
        'arrival': arrival,
        'departure': departure,
        'arrival_strength': arrival_strength,
        'departure_strength': departure_strength,
    }


def restore_raw_visible_changes(
        physical_prediction: torch.Tensor,
        raw_prediction: torch.Tensor,
        persistence: torch.Tensor,
        observation_known: torch.Tensor,
        change_probability: torch.Tensor,
        change_threshold: float,
        protected_event_mask: torch.Tensor = None) -> Tuple[
            torch.Tensor, torch.Tensor]:
    """Restore confident raw changes after persistence-based fusion."""
    if (physical_prediction.shape != raw_prediction.shape or
            physical_prediction.shape != persistence.shape or
            physical_prediction.ndim != 4):
        raise ValueError('Physical, raw and persistence shapes must match')
    future_shape = physical_prediction.shape[1:]
    expected_change_shape = (
        physical_prediction.shape[0] - 1, *future_shape)
    if tuple(change_probability.shape) != expected_change_shape:
        raise ValueError('Change probability shape does not match prediction')
    if tuple(observation_known.shape) != future_shape:
        raise ValueError('Observation-known shape does not match prediction')
    if (protected_event_mask is not None and
            tuple(protected_event_mask.shape) != expected_change_shape):
        raise ValueError('Protected event shape does not match prediction')
    if not 0.0 <= change_threshold <= 1.0:
        raise ValueError('Change threshold must be in [0, 1]')
    restore = (
        observation_known[None] &
        (raw_prediction[1:] != persistence[1:]) &
        (change_probability >= change_threshold))
    if protected_event_mask is not None:
        restore = restore & ~protected_event_mask.bool()
    hybrid = physical_prediction.clone()
    hybrid[1:] = torch.where(
        restore, raw_prediction[1:], hybrid[1:])
    return hybrid, restore


def _empty_confusions(horizon_count: int) -> Dict[str, np.ndarray]:
    shape = (horizon_count, 3, 3)
    return {
        'overall': np.zeros(shape, dtype=np.int64),
        'current_visible': np.zeros(shape, dtype=np.int64),
        'reveal_completion': np.zeros(shape, dtype=np.int64),
        'state_change': np.zeros(shape, dtype=np.int64),
        'visible_transition': np.zeros(shape, dtype=np.int64),
    }


def update_method_confusions(confusions: Dict[str, np.ndarray],
                             prediction: torch.Tensor, target: torch.Tensor,
                             persistence: torch.Tensor,
                             current_known: torch.Tensor):
    """Update semantic summaries for one prediction candidate."""
    target_known = target != IGNORE_INDEX
    current_visible = target_known & current_known
    reveal = target_known & ~current_known
    state_change = target_known & (target != persistence)
    visible_transition = state_change & current_known
    selections = {
        'overall': target_known,
        'current_visible': current_visible,
        'reveal_completion': reveal,
        'state_change': state_change,
        'visible_transition': visible_transition,
    }
    for horizon in range(target.shape[0]):
        for name, selection in selections.items():
            _update_confusion(
                confusions[name][horizon], prediction[horizon],
                target[horizon], selection[horizon])


def summarize_confusions(confusions: Dict[str, np.ndarray],
                         mean_target_times_s: np.ndarray) -> Dict[str, object]:
    return {
        name: _semantic_summary(values, mean_target_times_s)
        for name, values in confusions.items()
    }


def _empty_disagreement(horizon_count: int) -> Dict[str, np.ndarray]:
    fields = (
        'selected_voxels', 'disagreement_voxels', 'raw_only_correct',
        'alternative_only_correct', 'both_wrong', 'raw_correct',
        'alternative_correct')
    return {field: np.zeros(horizon_count, dtype=np.int64) for field in fields}


def accumulate_disagreement(acc: Dict[str, np.ndarray], raw: torch.Tensor,
                             alternative: torch.Tensor, target: torch.Tensor,
                             selection: torch.Tensor):
    """Count which side is correct, restricted to a semantic subset."""
    for horizon in range(target.shape[0]):
        selected = selection[horizon]
        raw_h = raw[horizon][selected]
        alternative_h = alternative[horizon][selected]
        target_h = target[horizon][selected]
        disagreement = raw_h != alternative_h
        raw_correct = raw_h == target_h
        alternative_correct = alternative_h == target_h
        acc['selected_voxels'][horizon] += int(selected.sum())
        acc['disagreement_voxels'][horizon] += int(disagreement.sum())
        acc['raw_only_correct'][horizon] += int(
            (disagreement & raw_correct & ~alternative_correct).sum())
        acc['alternative_only_correct'][horizon] += int(
            (disagreement & alternative_correct & ~raw_correct).sum())
        acc['both_wrong'][horizon] += int(
            (disagreement & ~raw_correct & ~alternative_correct).sum())
        acc['raw_correct'][horizon] += int(raw_correct.sum())
        acc['alternative_correct'][horizon] += int(
            alternative_correct.sum())


def _disagreement_summary(acc: Dict[str, np.ndarray]) -> list:
    rows = []
    for horizon in range(len(acc['selected_voxels'])):
        selected = int(acc['selected_voxels'][horizon])
        disagreement = int(acc['disagreement_voxels'][horizon])
        rows.append({
            'future_horizon_index': horizon + 1,
            'selected_voxels': selected,
            'disagreement_voxels': disagreement,
            'disagreement_ratio': (
                None if selected == 0 else disagreement / selected),
            'raw_only_correct': int(acc['raw_only_correct'][horizon]),
            'alternative_only_correct': int(
                acc['alternative_only_correct'][horizon]),
            'both_wrong': int(acc['both_wrong'][horizon]),
            'raw_correct': int(acc['raw_correct'][horizon]),
            'alternative_correct': int(
                acc['alternative_correct'][horizon]),
        })
    return rows


def _binary_counts(predicted: torch.Tensor, target: torch.Tensor) -> list:
    predicted = predicted.bool()
    target = target.bool()
    return [
        int((predicted & target).sum()),
        int((predicted & ~target).sum()),
        int((~predicted & target).sum()),
        int((~predicted & ~target).sum()),
    ]


def _sum_binary_counts(left: list, right: list) -> list:
    return [a + b for a, b in zip(left, right)]


def _binary_summary(counts: list) -> dict:
    tp, fp, fn, tn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,
        'true_negative': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def _empty_event_metrics(horizon_count: int) -> dict:
    return {
        event: [_empty_counts() for _ in range(horizon_count - 1)]
        for event in ('arrival', 'departure')
    }


def _empty_counts() -> list:
    return [0, 0, 0, 0]


def _update_event_metrics(metrics: dict, events: Dict[str, torch.Tensor],
                          target: torch.Tensor, current_instance: torch.Tensor):
    target_known = target[1:] != IGNORE_INDEX
    current_instance = current_instance[None].expand_as(target[1:])
    true_arrival = target[1:] == 2
    true_arrival = true_arrival & ~current_instance
    true_departure = target[1:] != 2
    true_departure = true_departure & current_instance
    selections = {
        'arrival': target_known & ~current_instance,
        'departure': target_known & current_instance,
    }
    truths = {
        'arrival': true_arrival,
        'departure': true_departure,
    }
    for event in ('arrival', 'departure'):
        for horizon in range(target.shape[0] - 1):
            counts = _binary_counts(
                events[event][horizon][selections[event][horizon]],
                truths[event][horizon][selections[event][horizon]])
            metrics[event][horizon] = _sum_binary_counts(
                metrics[event][horizon], counts)


def _event_summary(metrics: dict) -> dict:
    return {
        event: [
            dict(_binary_summary(counts), future_horizon_index=horizon + 1)
            for horizon, counts in enumerate(values)
        ]
        for event, values in metrics.items()
    }


def _empty_signal_stats() -> dict:
    return {
        'correct_count': 0,
        'incorrect_count': 0,
        'correct_event_strength_sum': 0.0,
        'incorrect_event_strength_sum': 0.0,
        'correct_change_probability_sum': 0.0,
        'incorrect_change_probability_sum': 0.0,
        'correct_flow_norm_sum': 0.0,
        'incorrect_flow_norm_sum': 0.0,
    }


def _update_signal_stats(stats: dict, events: Dict[str, torch.Tensor],
                         change_probability: torch.Tensor,
                         flow_norm: torch.Tensor, target: torch.Tensor,
                         current_instance: torch.Tensor):
    target_known = target[1:] != IGNORE_INDEX
    current_instance = current_instance[None].expand_as(target[1:])
    arrival = events['arrival']
    departure = events['departure']
    arrival_correct = arrival & (target[1:] == 2)
    departure_correct = departure & (target[1:] == 0)
    correct = (arrival_correct | departure_correct) & target_known
    candidate = (arrival | departure) & target_known
    incorrect = candidate & ~correct
    strength = torch.maximum(
        events['arrival_strength'], events['departure_strength'])
    flow_norm = flow_norm[:, None].expand_as(strength)
    for mask, prefix in ((correct, 'correct'), (incorrect, 'incorrect')):
        count = int(mask.sum())
        stats[f'{prefix}_count'] += count
        stats[f'{prefix}_event_strength_sum'] += float(strength[mask].sum())
        stats[f'{prefix}_change_probability_sum'] += float(
            change_probability[mask].sum())
        stats[f'{prefix}_flow_norm_sum'] += float(flow_norm[mask].sum())


def _signal_summary(stats: dict) -> dict:
    result = {'correct_candidate_count': stats['correct_count'],
              'incorrect_candidate_count': stats['incorrect_count']}
    for prefix in ('correct', 'incorrect'):
        count = stats[f'{prefix}_count']
        result[f'{prefix}_mean_event_strength'] = (
            None if count == 0 else stats[f'{prefix}_event_strength_sum'] / count)
        result[f'{prefix}_mean_change_probability'] = (
            None if count == 0
            else stats[f'{prefix}_change_probability_sum'] / count)
        result[f'{prefix}_mean_flow_norm'] = (
            None if count == 0 else stats[f'{prefix}_flow_norm_sum'] / count)
    return result


def _mean_iou(summary: dict, subset: str = 'overall') -> float:
    values = summary[subset]['by_horizon'][1:]
    return float(np.mean([item['mean_iou'] for item in values]))


def _method_row(name: str, summary: dict) -> dict:
    return {
        'name': name,
        'future_mean_iou': _mean_iou(summary),
        'future_reveal_mean_iou': _mean_iou(
            summary, 'reveal_completion'),
        'future_state_change_mean_iou': _mean_iou(
            summary, 'state_change'),
        'future_visible_transition_mean_iou': _mean_iou(
            summary, 'visible_transition'),
    }


def _oracle_prediction(raw: torch.Tensor, alternative: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    raw_correct = raw == target
    alternative_correct = alternative == target
    use_alternative = alternative_correct & ~raw_correct
    return torch.where(use_alternative, alternative, raw)


def _load_prediction(path: Path, expected_shape: Sequence[int]):
    with np.load(path, allow_pickle=False) as archive:
        required = {
            PREDICTION_CLASS_KEY,
            PREDICTION_CHANGE_PROBABILITY_KEY,
            PREDICTION_VALID_PROBABILITY_KEY,
            PREDICTION_WARPED_INSTANCE_KEY,
            PREDICTION_FLOW_KEY,
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f'{path} is missing {missing}')
        raw = torch.from_numpy(np.asarray(
            archive[PREDICTION_CLASS_KEY], dtype=np.int64))
        change = torch.from_numpy(np.asarray(
            archive[PREDICTION_CHANGE_PROBABILITY_KEY], dtype=np.float32))
        visibility = torch.from_numpy(np.asarray(
            archive[PREDICTION_VALID_PROBABILITY_KEY], dtype=np.float32))
        warped = torch.from_numpy(np.asarray(
            archive[PREDICTION_WARPED_INSTANCE_KEY], dtype=np.float32))
        flow = torch.from_numpy(np.asarray(
            archive[PREDICTION_FLOW_KEY], dtype=np.float32))
    expected_shape = tuple(int(value) for value in expected_shape)
    if tuple(raw.shape) != expected_shape or tuple(visibility.shape) != expected_shape:
        raise ValueError(f'Prediction shape mismatch in {path}')
    expected_future = (expected_shape[0] - 1, *expected_shape[1:])
    if (tuple(change.shape) != expected_future or
            tuple(warped.shape) != expected_future or
            tuple(flow.shape) != (expected_shape[0] - 1, 2,
                                  expected_shape[2], expected_shape[3])):
        raise ValueError(f'Future prediction shape mismatch in {path}')
    if not all(torch.isfinite(value).all() for value in
               (change, visibility, warped, flow)):
        raise ValueError(f'Non-finite prediction in {path}')
    return raw, change, visibility, warped, flow


def audit_split(manifest: Mapping[str, object], split: str,
                sequence_mapping: Mapping[int, Path],
                prediction_mapping: Mapping[int, Path],
                event_threshold: float = 0.5,
                change_threshold: float = 0.0,
                scan_event_thresholds: Sequence[float] = (
                    0.1, 0.3, 0.5, 0.7, 0.9),
                scan_change_thresholds: Sequence[float] = (
                    0.0, 0.05, 0.1, 0.2, 0.5, 0.8),
                scan_flow_norm_thresholds: Sequence[float] = (
                    0.0, 0.2, 0.5, 0.8, 1.0, 1.5),
                restore_change_thresholds: Sequence[float] = (
                    0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)) -> dict:
    references = _split_references(manifest, split)
    missing = sorted(set(references).difference(prediction_mapping))
    if missing:
        raise ValueError(f'Missing predictions for references {missing}')
    method_names = ('persistence', 'raw', 'raw_plus_flow', 'physical_flow',
                    'oracle_raw_plus_flow', 'oracle_physical_flow')
    confusions = None
    disagreements = None
    event_metrics = None
    signal_stats = _empty_signal_stats()
    target_time_sum = None
    grid_pairs = [
        (float(event_value), float(change_value), float(flow_norm_value))
        for event_value in scan_event_thresholds
        for change_value in scan_change_thresholds
        for flow_norm_value in scan_flow_norm_thresholds
    ]
    grid_confusions = None
    restore_confusions = None
    for reference in references:
        label_path = sequence_mapping[reference]
        with np.load(label_path, allow_pickle=False) as label:
            target_state = np.asarray(
                label['world_target_state_3d'], dtype=np.int64)
            target_valid = np.asarray(
                label['world_target_valid_3d'], dtype=np.bool_)
            current_state = np.asarray(
                label['current_observation_state_3d'], dtype=np.int64)
            current_valid = np.asarray(
                label['current_observation_valid_3d'], dtype=np.bool_)
            target_times_s = np.asarray(
                label['target_times_s'], dtype=np.float64)
        target = _class_target(target_state, target_valid)
        horizon_count = target.shape[0]
        raw, change, visibility, warped, flow = _load_prediction(
            prediction_mapping[reference], target.shape)
        persistence, _ = _persistence_prediction(
            current_state, current_valid, horizon_count)
        current_known = torch.from_numpy(
            current_valid & (current_state != 0))
        observation_class = persistence[0]
        current_instance = current_known & (observation_class == 2)
        raw_plus_flow, events = apply_flow_events_to_raw(
            raw, observation_class, current_known, warped,
            event_threshold=event_threshold,
            change_probability=change,
            change_threshold=change_threshold)
        physical = apply_physical_flow_fusion(
            raw[None], observation_class[None], current_known[None],
            warped[None], threshold=event_threshold)[0]
        _, physical_events = apply_flow_events_to_raw(
            raw, observation_class, current_known, warped,
            event_threshold=event_threshold)
        physical_event_mask = (
            physical_events['arrival'] | physical_events['departure'])
        oracle_raw_flow = _oracle_prediction(raw, raw_plus_flow, target)
        oracle_raw_physical = _oracle_prediction(raw, physical, target)
        predictions = {
            'persistence': persistence,
            'raw': raw,
            'raw_plus_flow': raw_plus_flow,
            'physical_flow': physical,
            'oracle_raw_plus_flow': oracle_raw_flow,
            'oracle_physical_flow': oracle_raw_physical,
        }
        if confusions is None:
            confusions = {
                name: _empty_confusions(horizon_count)
                for name in method_names
            }
            disagreements = {
                name: {
                    subset: _empty_disagreement(horizon_count)
                    for subset in ('overall', 'current_visible',
                                   'reveal_completion', 'visible_transition')
                }
                for name in ('raw_plus_flow', 'physical_flow')
            }
            event_metrics = _empty_event_metrics(horizon_count)
            target_time_sum = np.zeros(horizon_count, dtype=np.float64)
            grid_confusions = {
                pair: _empty_confusions(horizon_count)
                for pair in grid_pairs
            }
            restore_confusions = {
                float(threshold): _empty_confusions(horizon_count)
                for threshold in restore_change_thresholds
            }
        target_time_sum += target_times_s
        target_known = target != IGNORE_INDEX
        current_known_expanded = current_known[None].expand_as(target)
        reveal = target_known & ~current_known_expanded
        visible_transition = (
            target_known & current_known_expanded & (target != persistence))
        selections = {
            'overall': target_known,
            'current_visible': target_known & current_known_expanded,
            'reveal_completion': reveal,
            'visible_transition': visible_transition,
        }
        for name, prediction in predictions.items():
            update_method_confusions(
                confusions[name], prediction, target, persistence,
                current_known_expanded)
        for name, alternative in (
                ('raw_plus_flow', raw_plus_flow),
                ('physical_flow', physical)):
            for subset, selection in selections.items():
                accumulate_disagreement(
                    disagreements[name][subset], raw, alternative,
                    target, selection)
        _update_event_metrics(
            event_metrics, events, target, current_instance)
        flow_norm = torch.linalg.vector_norm(flow, dim=1)
        _update_signal_stats(
            signal_stats, events, change, flow_norm, target,
            current_instance)
        for pair in grid_pairs:
            grid_prediction, _ = apply_flow_events_to_raw(
                raw, observation_class, current_known, warped,
                event_threshold=pair[0], change_probability=change,
                change_threshold=pair[1], flow_norm=flow_norm,
                flow_norm_threshold=pair[2])
            update_method_confusions(
                grid_confusions[pair], grid_prediction, target,
                persistence, current_known_expanded)
        for restore_threshold in restore_confusions:
            restored_prediction, _ = restore_raw_visible_changes(
                physical, raw, persistence, current_known, change,
                change_threshold=restore_threshold,
                protected_event_mask=physical_event_mask)
            update_method_confusions(
                restore_confusions[restore_threshold], restored_prediction,
                target, persistence, current_known_expanded)

    mean_target_times_s = target_time_sum / len(references)
    summaries = {
        name: _method_row(
            name, summarize_confusions(values, mean_target_times_s))
        for name, values in confusions.items()
    }
    detailed_summaries = {
        name: summarize_confusions(values, mean_target_times_s)
        for name, values in confusions.items()
    }
    disagreement_summary = {
        name: {
            subset: _disagreement_summary(values)
            for subset, values in subsets.items()
        }
        for name, subsets in disagreements.items()
    }
    grid_rows = []
    for (grid_event_threshold, grid_change_threshold,
         grid_flow_norm_threshold), values in (
            grid_confusions.items()):
        row = _method_row(
            'raw_plus_causal_flow',
            summarize_confusions(values, mean_target_times_s))
        row.update({
            'event_threshold': grid_event_threshold,
            'change_threshold': grid_change_threshold,
            'flow_norm_threshold': grid_flow_norm_threshold,
        })
        grid_rows.append(row)
    raw_future = summaries['raw']['future_mean_iou']
    raw_transition = summaries['raw']['future_visible_transition_mean_iou']
    eligible_rows = [
        row for row in grid_rows
        if row['future_mean_iou'] >= raw_future and
        row['future_visible_transition_mean_iou'] > raw_transition
    ]
    best_transition_row = max(
        eligible_rows,
        key=lambda row: (
            row['future_visible_transition_mean_iou'],
            row['future_mean_iou'])) if eligible_rows else None
    best_future_row = max(
        grid_rows,
        key=lambda row: (
            row['future_mean_iou'],
            row['future_visible_transition_mean_iou']))
    restore_rows = []
    for restore_threshold, values in restore_confusions.items():
        row = _method_row(
            'physical_plus_restored_raw_change',
            summarize_confusions(values, mean_target_times_s))
        row['restore_change_threshold'] = restore_threshold
        restore_rows.append(row)
    physical_future = summaries['physical_flow']['future_mean_iou']
    restore_eligible_rows = [
        row for row in restore_rows
        if row['future_mean_iou'] >= physical_future and
        row['future_visible_transition_mean_iou'] > raw_transition
    ]
    best_restore_row = max(
        restore_eligible_rows,
        key=lambda row: (
            row['future_visible_transition_mean_iou'],
            row['future_mean_iou'])) if restore_eligible_rows else None
    return {
        'schema_version': 1,
        'split': split,
        'sample_count': len(references),
        'reference_indices': list(references),
        'event_threshold': event_threshold,
        'change_threshold': change_threshold,
        'method_rows': summaries,
        'method_semantic': detailed_summaries,
        'disagreement': disagreement_summary,
        'event_metrics': _event_summary(event_metrics),
        'event_signal_stats': _signal_summary(signal_stats),
        'causal_gate_grid': {
            'rows': grid_rows,
            'rows_meeting_raw_future_and_transition': eligible_rows,
            'best_transition_under_raw_future_constraint': (
                best_transition_row),
            'best_future_row': best_future_row,
        },
        'physical_restore_grid': {
            'rows': restore_rows,
            'rows_meeting_physical_future_and_raw_transition': (
                restore_eligible_rows),
            'best_transition_under_physical_future_constraint': (
                best_restore_row),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path('documents/patent_2026_occ/kl_occworld_scene_split_v2.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path('outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument(
        '--prediction-root', type=Path, required=True,
        help='Prediction directory containing reference prediction files')
    parser.add_argument(
        '--split', choices=('train', 'validation', 'test', 'blind'),
        default='validation')
    parser.add_argument('--event-threshold', type=float, default=0.5)
    parser.add_argument('--change-threshold', type=float, default=0.0)
    parser.add_argument(
        '--scan-event-thresholds', type=float, nargs='+',
        default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument(
        '--scan-change-thresholds', type=float, nargs='+',
        default=[0.0, 0.05, 0.1, 0.2, 0.5, 0.8])
    parser.add_argument(
        '--scan-flow-norm-thresholds', type=float, nargs='+',
        default=[0.0, 0.2, 0.5, 0.8, 1.0, 1.5])
    parser.add_argument(
        '--restore-change-thresholds', type=float, nargs='+',
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_raw_flow_disagreement_validation_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    # Labels are looked up by reference in the generated sequence root.
    from tools.analysis_tools.build_kl_occworld_scene_split import (
        _sequence_mapping,
    )
    sequence_mapping = _sequence_mapping(args.sequence_root)
    prediction_mapping = _prediction_mapping(args.prediction_root)
    result = audit_split(
        manifest, args.split, sequence_mapping, prediction_mapping,
        event_threshold=args.event_threshold,
        change_threshold=args.change_threshold,
        scan_event_thresholds=args.scan_event_thresholds,
        scan_change_thresholds=args.scan_change_thresholds,
        scan_flow_norm_thresholds=args.scan_flow_norm_thresholds,
        restore_change_thresholds=args.restore_change_thresholds)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    gate_grid = result['causal_gate_grid']
    restore_grid = result['physical_restore_grid']
    print(json.dumps({
        'method_rows': result['method_rows'],
        'event_metrics': result['event_metrics'],
        'event_signal_stats': result['event_signal_stats'],
        'causal_gate_grid': {
            'row_count': len(gate_grid['rows']),
            'eligible_row_count': len(
                gate_grid['rows_meeting_raw_future_and_transition']),
            'best_transition_under_raw_future_constraint': gate_grid[
                'best_transition_under_raw_future_constraint'],
            'best_future_row': gate_grid['best_future_row'],
        },
        'physical_restore_grid': {
            'row_count': len(restore_grid['rows']),
            'eligible_row_count': len(restore_grid[
                'rows_meeting_physical_future_and_raw_transition']),
            'best_transition_under_physical_future_constraint': restore_grid[
                'best_transition_under_physical_future_constraint'],
        },
    }, ensure_ascii=False, indent=2))
    print(f'out_file={args.out_file}')


if __name__ == '__main__':
    main()
