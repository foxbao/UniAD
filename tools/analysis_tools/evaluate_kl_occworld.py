#!/usr/bin/env python
"""Evaluate KL OccWorld predictions with a frozen scene-level manifest."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld_baselines import (
    CLASS_NAMES,
    _summarize_horizons,
    _update_confusion,
)
from tools.data_converter.kl_occworld_dataset import IGNORE_INDEX


PREDICTION_CLASS_KEY = 'world_pred_class_3d'
PREDICTION_VALID_PROBABILITY_KEY = 'world_valid_probability_3d'
PREDICTION_CHANGE_PROBABILITY_KEY = 'future_change_probability_3d'
PREDICTION_CHANGED_CLASS_KEY = 'future_changed_class_pred_3d'
PREDICTION_WARPED_INSTANCE_KEY = 'warped_instance_probability_3d'
PREDICTION_PHYSICAL_CONFIDENCE_KEY = (
    'physical_confidence_probability_3d')


def _load_manifest(path: Path) -> Dict[str, object]:
    with path.open() as source:
        manifest = json.load(source)
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported OccWorld split manifest version')
    if not isinstance(manifest.get('splits'), dict):
        raise ValueError('Manifest has no splits mapping')
    return manifest


def _split_references(manifest: Mapping[str, object],
                      split: str) -> Sequence[int]:
    splits = manifest['splits']
    if split not in splits:
        raise ValueError(f'Manifest has no {split} split')
    references = [
        int(record['reference_index']) for record in splits[split]
    ]
    if not references or len(references) != len(set(references)):
        raise ValueError(f'Invalid references in {split} split')
    return references


def _prediction_mapping(prediction_root: Path) -> Dict[int, Path]:
    mapping = {}
    paths = sorted(prediction_root.glob('*/*__occworld_prediction.npz'))
    if not paths:
        paths = sorted(prediction_root.glob('*__occworld_prediction.npz'))
    for path in paths:
        with np.load(path, allow_pickle=False) as prediction:
            if 'reference_index' not in prediction.files:
                raise ValueError(f'{path} has no reference_index')
            reference_index = int(prediction['reference_index'])
        if reference_index in mapping:
            raise ValueError(
                f'Duplicate prediction for reference {reference_index}')
        mapping[reference_index] = path
    if not mapping:
        raise FileNotFoundError(
            f'No OccWorld predictions found below {prediction_root}')
    return mapping


def _class_target(state: np.ndarray, valid: np.ndarray) -> torch.Tensor:
    if state.shape != valid.shape:
        raise ValueError('Target state and valid mask shapes must match')
    if np.any((state < 0) | (state > len(CLASS_NAMES))):
        raise ValueError('Target contains an invalid world state')
    known = valid.astype(bool) & (state != 0)
    target = np.full(state.shape, IGNORE_INDEX, dtype=np.int64)
    target[known] = state[known] - 1
    return torch.from_numpy(target)


def _persistence_prediction(
        current_state: np.ndarray,
        current_valid: np.ndarray,
        horizon_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if current_state.shape != current_valid.shape:
        raise ValueError('Current state and valid mask shapes must match')
    current_known = current_valid.astype(bool) & (current_state != 0)
    current_class = np.zeros(current_state.shape, dtype=np.int64)
    current_class[current_known] = current_state[current_known] - 1
    prediction = torch.from_numpy(current_class).unsqueeze(0).expand(
        horizon_count, *current_class.shape)
    visible = torch.from_numpy(current_known).unsqueeze(0).expand_as(
        prediction)
    return prediction, visible.to(torch.float32)


def _load_model_prediction(
        path: Path,
        expected_shape: Sequence[int]) -> Tuple[torch.Tensor,
                                                Optional[torch.Tensor],
                                                Optional[torch.Tensor],
                                                Optional[torch.Tensor],
                                                Optional[torch.Tensor],
                                                Optional[torch.Tensor]]:
    with np.load(path, allow_pickle=False) as archive:
        if PREDICTION_CLASS_KEY not in archive.files:
            raise ValueError(
                f'{path} has no {PREDICTION_CLASS_KEY}')
        prediction = np.asarray(
            archive[PREDICTION_CLASS_KEY], dtype=np.int64)
        valid_probability = None
        if PREDICTION_VALID_PROBABILITY_KEY in archive.files:
            valid_probability = np.asarray(
                archive[PREDICTION_VALID_PROBABILITY_KEY],
                dtype=np.float32)
        change_probability = None
        if PREDICTION_CHANGE_PROBABILITY_KEY in archive.files:
            change_probability = np.asarray(
                archive[PREDICTION_CHANGE_PROBABILITY_KEY],
                dtype=np.float32)
        changed_class_prediction = None
        if PREDICTION_CHANGED_CLASS_KEY in archive.files:
            changed_class_prediction = np.asarray(
                archive[PREDICTION_CHANGED_CLASS_KEY], dtype=np.int64)
        warped_instance_probability = None
        if PREDICTION_WARPED_INSTANCE_KEY in archive.files:
            warped_instance_probability = np.asarray(
                archive[PREDICTION_WARPED_INSTANCE_KEY], dtype=np.float32)
        physical_confidence_probability = None
        if PREDICTION_PHYSICAL_CONFIDENCE_KEY in archive.files:
            physical_confidence_probability = np.asarray(
                archive[PREDICTION_PHYSICAL_CONFIDENCE_KEY],
                dtype=np.float32)
    expected_shape = tuple(int(value) for value in expected_shape)
    if prediction.shape != expected_shape:
        raise ValueError(
            f'Prediction shape {prediction.shape} does not match '
            f'{expected_shape} in {path}')
    if np.any((prediction < 0) | (prediction >= len(CLASS_NAMES))):
        raise ValueError(f'Prediction class is outside [0, 2] in {path}')
    if (valid_probability is not None and
            valid_probability.shape != expected_shape):
        raise ValueError(
            f'Visibility shape mismatch in {path}')
    if (valid_probability is not None and
            (np.any(valid_probability < 0.0) or
             np.any(valid_probability > 1.0))):
        raise ValueError(
            f'Visibility probability is outside [0, 1] in {path}')
    expected_change_shape = (expected_shape[0] - 1, *expected_shape[1:])
    if (change_probability is not None and
            change_probability.shape != expected_change_shape):
        raise ValueError(
            f'Future change shape {change_probability.shape} does not '
            f'match {expected_change_shape} in {path}')
    if (change_probability is not None and
            (not np.all(np.isfinite(change_probability)) or
             np.any(change_probability < 0.0) or
             np.any(change_probability > 1.0))):
        raise ValueError(
            f'Future change probability is outside [0, 1] in {path}')
    if (changed_class_prediction is not None and
            changed_class_prediction.shape != expected_change_shape):
        raise ValueError(
            'Future changed-class shape '
            f'{changed_class_prediction.shape} does not match '
            f'{expected_change_shape} in {path}')
    if (changed_class_prediction is not None and
            np.any((changed_class_prediction < 0) |
                   (changed_class_prediction >= len(CLASS_NAMES)))):
        raise ValueError(
            f'Future changed class is outside [0, 2] in {path}')
    if (warped_instance_probability is not None and
            warped_instance_probability.shape != expected_change_shape):
        raise ValueError(
            'Warped instance shape '
            f'{warped_instance_probability.shape} does not match '
            f'{expected_change_shape} in {path}')
    if (warped_instance_probability is not None and
            (not np.all(np.isfinite(warped_instance_probability)) or
             np.any(warped_instance_probability < 0.0) or
             np.any(warped_instance_probability > 1.0))):
        raise ValueError(
            f'Warped instance probability is outside [0, 1] in {path}')
    if (physical_confidence_probability is not None and
            physical_confidence_probability.shape != expected_change_shape):
        raise ValueError(
            'Physical-confidence shape '
            f'{physical_confidence_probability.shape} does not match '
            f'{expected_change_shape} in {path}')
    if (physical_confidence_probability is not None and
            (not np.all(np.isfinite(physical_confidence_probability)) or
             np.any(physical_confidence_probability < 0.0) or
             np.any(physical_confidence_probability > 1.0))):
        raise ValueError(
            f'Physical confidence is outside [0, 1] in {path}')
    return (
        torch.from_numpy(prediction),
        None if valid_probability is None
        else torch.from_numpy(valid_probability),
        None if change_probability is None
        else torch.from_numpy(change_probability),
        None if changed_class_prediction is None
        else torch.from_numpy(changed_class_prediction),
        None if warped_instance_probability is None
        else torch.from_numpy(warped_instance_probability),
        None if physical_confidence_probability is None
        else torch.from_numpy(physical_confidence_probability))


def _binary_counts(predicted: torch.Tensor,
                   target: torch.Tensor) -> np.ndarray:
    predicted = predicted.to(torch.bool)
    target = target.to(torch.bool)
    return np.asarray([
        int((predicted & target).sum()),
        int((predicted & ~target).sum()),
        int((~predicted & target).sum()),
        int((~predicted & ~target).sum()),
    ], dtype=np.int64)


def _summarize_binary_counts(counts: np.ndarray) -> Dict[str, object]:
    true_positive, false_positive, false_negative, true_negative = (
        int(value) for value in counts)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    iou_denominator = (
        true_positive + false_positive + false_negative)
    total = int(counts.sum())
    precision = (
        true_positive / precision_denominator
        if precision_denominator else 0.0)
    recall = (
        true_positive / recall_denominator
        if recall_denominator else 0.0)
    f1_denominator = precision + recall
    return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_negative': true_negative,
        'precision': precision,
        'recall': recall,
        'f1': (
            2.0 * precision * recall / f1_denominator
            if f1_denominator else 0.0),
        'iou': (
            true_positive / iou_denominator
            if iou_denominator else 0.0),
        'accuracy': (
            (true_positive + true_negative) / total if total else 0.0),
    }


def _semantic_summary(confusions: np.ndarray,
                      mean_target_times_s: np.ndarray) -> Dict[str, object]:
    summary = _summarize_horizons(confusions)
    for horizon, metrics in enumerate(summary['by_horizon']):
        metrics['horizon_index'] = horizon
        metrics['mean_target_time_s'] = float(
            mean_target_times_s[horizon])
    return summary


def _visibility_summary(counts: np.ndarray,
                        mean_target_times_s: np.ndarray) -> Dict[str, object]:
    return {
        'overall': _summarize_binary_counts(counts.sum(axis=0)),
        'by_horizon': [
            dict(
                _summarize_binary_counts(horizon_counts),
                horizon_index=horizon,
                mean_target_time_s=float(mean_target_times_s[horizon]))
            for horizon, horizon_counts in enumerate(counts)
        ],
    }


def evaluate_split(
        manifest: Mapping[str, object],
        split: str,
        sequence_mapping: Mapping[int, Path],
        prediction_mapping: Optional[Mapping[int, Path]] = None,
        visibility_threshold: float = 0.5,
        change_gate_threshold: float = 0.5,
        apply_change_gate: bool = False,
        apply_completion_only: bool = False,
        apply_flow_only: bool = False,
        apply_flow_fusion: bool = False,
        apply_local_flow_overlay: bool = False,
        flow_fusion_threshold: float = 0.5,
        apply_physical_confidence: bool = False,
        physical_confidence_threshold: float = 0.5,
        physical_confidence_flow_threshold: float = 0.5) -> Dict[str, object]:
    """Evaluate persistence or pre-exported model predictions."""
    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError('Visibility threshold must be in [0, 1]')
    if not 0.0 <= change_gate_threshold <= 1.0:
        raise ValueError('Change-gate threshold must be in [0, 1]')
    if not 0.0 <= flow_fusion_threshold <= 1.0:
        raise ValueError('Flow-fusion threshold must be in [0, 1]')
    if not 0.0 <= physical_confidence_threshold <= 1.0:
        raise ValueError('Physical-confidence threshold must be in [0, 1]')
    if not 0.0 <= physical_confidence_flow_threshold <= 1.0:
        raise ValueError(
            'Physical-confidence flow threshold must be in [0, 1]')
    fusion_count = sum((
        bool(apply_change_gate), bool(apply_completion_only),
        bool(apply_flow_only), bool(apply_flow_fusion),
        bool(apply_local_flow_overlay), bool(apply_physical_confidence)))
    if fusion_count > 1:
        raise ValueError('Evaluation fusion modes are mutually exclusive')
    if prediction_mapping is None and fusion_count:
        raise ValueError('Evaluation fusion requires model predictions')
    references = _split_references(manifest, split)
    missing_labels = sorted(set(references).difference(sequence_mapping))
    if missing_labels:
        raise ValueError(f'Missing labels for references {missing_labels}')
    if prediction_mapping is not None:
        missing_predictions = sorted(
            set(references).difference(prediction_mapping))
        if missing_predictions:
            raise ValueError(
                f'Missing predictions for references {missing_predictions}')

    confusions = None
    visible_confusions = None
    reveal_confusions = None
    change_confusions = None
    transition_confusions = None
    instance_transition_confusions = None
    visibility_counts = None
    target_time_sum = None
    target_known_counts = None
    current_visible_counts = None
    reveal_counts = None
    state_change_counts = None
    visible_transition_counts = None
    instance_visible_transition_counts = None
    visibility_available = None
    change_gate_counts = None
    change_gate_available = None
    changed_class_confusions = None
    changed_class_available = None
    warped_instance_available = None
    physical_confidence_available = None

    for reference_index in references:
        label_path = sequence_mapping[reference_index]
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
        persistence, persistence_visibility = _persistence_prediction(
            current_state, current_valid, horizon_count)
        if prediction_mapping is None:
            prediction = persistence
            visibility_probability = persistence_visibility
            change_probability = None
            changed_class_prediction = None
            warped_instance_probability = None
            physical_confidence_probability = None
        else:
            (prediction, visibility_probability,
             change_probability,
             changed_class_prediction,
             warped_instance_probability,
             physical_confidence_probability) = _load_model_prediction(
                 prediction_mapping[reference_index], target.shape)
        sample_visibility_available = visibility_probability is not None
        if visibility_available is None:
            visibility_available = sample_visibility_available
        elif visibility_available != sample_visibility_available:
            raise ValueError(
                'Visibility prediction must be present for every sample '
                'or for none')
        sample_change_gate_available = change_probability is not None
        if change_gate_available is None:
            change_gate_available = sample_change_gate_available
        elif change_gate_available != sample_change_gate_available:
            raise ValueError(
                'Change-gate prediction must be present for every sample '
                'or for none')
        sample_changed_class_available = (
            changed_class_prediction is not None)
        if changed_class_available is None:
            changed_class_available = sample_changed_class_available
        elif changed_class_available != sample_changed_class_available:
            raise ValueError(
                'Changed-class prediction must be present for every sample '
                'or for none')
        sample_warped_instance_available = (
            warped_instance_probability is not None)
        if warped_instance_available is None:
            warped_instance_available = sample_warped_instance_available
        elif (warped_instance_available !=
              sample_warped_instance_available):
            raise ValueError(
                'Warped-instance prediction must be present for every '
                'sample or for none')
        sample_physical_confidence_available = (
            physical_confidence_probability is not None)
        if physical_confidence_available is None:
            physical_confidence_available = (
                sample_physical_confidence_available)
        elif (physical_confidence_available !=
              sample_physical_confidence_available):
            raise ValueError(
                'Physical-confidence prediction must be present for every '
                'sample or for none')
        if apply_change_gate:
            if (change_probability is None or
                    changed_class_prediction is None):
                raise ValueError(
                    'Hard change-gate fusion requires gate and class output')
            prediction = prediction.clone()
            current_known_for_future = torch.from_numpy(
                current_valid & (current_state != 0))[None].expand(
                    horizon_count - 1, *current_state.shape)
            prediction[1:] = torch.where(
                current_known_for_future,
                persistence[1:], prediction[1:])
            hard_change = (
                current_known_for_future &
                (change_probability >= change_gate_threshold))
            prediction[1:] = torch.where(
                hard_change, changed_class_prediction, prediction[1:])
        if apply_completion_only:
            prediction = prediction.clone()
            current_known_for_future = torch.from_numpy(
                current_valid & (current_state != 0))[None].expand(
                    horizon_count - 1, *current_state.shape)
            prediction[1:] = torch.where(
                current_known_for_future,
                persistence[1:], prediction[1:])
        if apply_flow_only:
            prediction = persistence.clone()
        if apply_flow_fusion or apply_flow_only:
            if warped_instance_probability is None:
                raise ValueError(
                    'Physical flow fusion requires warped instance output')
            prediction = prediction.clone()
            current_known_for_future = torch.from_numpy(
                current_valid & (current_state != 0))[None].expand(
                    horizon_count - 1, *current_state.shape)
            prediction[1:] = torch.where(
                current_known_for_future,
                persistence[1:], prediction[1:])
            current_instance = torch.from_numpy(
                current_valid & (current_state == 3))[None].expand(
                    horizon_count - 1, *current_state.shape).to(
                        warped_instance_probability.dtype)
            arrival = (
                warped_instance_probability - current_instance >=
                flow_fusion_threshold)
            departure = (
                current_instance - warped_instance_probability >=
                flow_fusion_threshold)
            prediction[1:] = torch.where(
                arrival, torch.full_like(prediction[1:], 2),
                prediction[1:])
            prediction[1:] = torch.where(
                departure, torch.zeros_like(prediction[1:]),
                prediction[1:])
        if apply_local_flow_overlay:
            if warped_instance_probability is None:
                raise ValueError(
                    'Local flow overlay requires warped instance output')
            prediction = prediction.clone()
            current_instance = torch.from_numpy(
                current_valid & (current_state == 3))[None].expand(
                    horizon_count - 1, *current_state.shape).to(
                        warped_instance_probability.dtype)
            arrival = (
                warped_instance_probability - current_instance >=
                flow_fusion_threshold)
            departure = (
                current_instance - warped_instance_probability >=
                flow_fusion_threshold)
            prediction[1:] = torch.where(
                arrival, torch.full_like(prediction[1:], 2),
                prediction[1:])
            prediction[1:] = torch.where(
                departure, torch.zeros_like(prediction[1:]),
                prediction[1:])
        if apply_physical_confidence:
            if (warped_instance_probability is None or
                    physical_confidence_probability is None):
                raise ValueError(
                    'Confidence fusion requires warped instance and '
                    'physical-confidence output')
            raw_prediction = prediction.clone()
            physical_prediction = raw_prediction.clone()
            current_known_for_future = torch.from_numpy(
                current_valid & (current_state != 0))[None].expand(
                    horizon_count - 1, *current_state.shape)
            physical_prediction[1:] = torch.where(
                current_known_for_future,
                persistence[1:], physical_prediction[1:])
            current_instance = torch.from_numpy(
                current_valid & (current_state == 3))[None].expand(
                    horizon_count - 1, *current_state.shape).to(
                        warped_instance_probability.dtype)
            arrival = (
                warped_instance_probability - current_instance >=
                physical_confidence_flow_threshold)
            departure = (
                current_instance - warped_instance_probability >=
                physical_confidence_flow_threshold)
            physical_prediction[1:] = torch.where(
                arrival, torch.full_like(physical_prediction[1:], 2),
                physical_prediction[1:])
            physical_prediction[1:] = torch.where(
                departure, torch.zeros_like(physical_prediction[1:]),
                physical_prediction[1:])
            use_physical = (
                physical_confidence_probability >=
                physical_confidence_threshold)
            prediction[1:] = torch.where(
                use_physical, physical_prediction[1:],
                raw_prediction[1:])

        if confusions is None:
            shape = (horizon_count, len(CLASS_NAMES), len(CLASS_NAMES))
            confusions = np.zeros(shape, dtype=np.int64)
            visible_confusions = np.zeros_like(confusions)
            reveal_confusions = np.zeros_like(confusions)
            change_confusions = np.zeros_like(confusions)
            transition_confusions = np.zeros_like(confusions)
            instance_transition_confusions = np.zeros_like(confusions)
            visibility_counts = np.zeros((horizon_count, 4), dtype=np.int64)
            change_gate_counts = np.zeros(
                (horizon_count - 1, 4), dtype=np.int64)
            changed_class_confusions = np.zeros(
                (horizon_count - 1, len(CLASS_NAMES), len(CLASS_NAMES)),
                dtype=np.int64)
            target_time_sum = np.zeros(horizon_count, dtype=np.float64)
            target_known_counts = np.zeros(horizon_count, dtype=np.int64)
            current_visible_counts = np.zeros(horizon_count, dtype=np.int64)
            reveal_counts = np.zeros(horizon_count, dtype=np.int64)
            state_change_counts = np.zeros(horizon_count, dtype=np.int64)
            visible_transition_counts = np.zeros(
                horizon_count, dtype=np.int64)
            instance_visible_transition_counts = np.zeros(
                horizon_count, dtype=np.int64)
        elif confusions.shape[0] != horizon_count:
            raise ValueError('Labels have inconsistent horizon counts')
        if target_times_s.shape != (horizon_count,):
            raise ValueError(f'Invalid target times in {label_path}')
        target_time_sum += target_times_s

        current_known = torch.from_numpy(
            current_valid & (current_state != 0))
        current_known = current_known.unsqueeze(0).expand_as(target)
        target_known = target != IGNORE_INDEX
        state_change = target_known & (target != persistence)
        reveal = target_known & ~current_known
        visible_transition = state_change & current_known
        current_instance = torch.from_numpy(
            current_valid & (current_state == 3))
        current_instance = current_instance.unsqueeze(0).expand_as(target)
        instance_visible_transition = (
            visible_transition &
            (current_instance | (target == 2)))
        for horizon in range(horizon_count):
            _update_confusion(
                confusions[horizon], prediction[horizon], target[horizon])
            _update_confusion(
                visible_confusions[horizon], prediction[horizon],
                target[horizon], current_known[horizon])
            _update_confusion(
                reveal_confusions[horizon], prediction[horizon],
                target[horizon], reveal[horizon])
            _update_confusion(
                change_confusions[horizon], prediction[horizon],
                target[horizon], state_change[horizon])
            _update_confusion(
                transition_confusions[horizon], prediction[horizon],
                target[horizon], visible_transition[horizon])
            _update_confusion(
                instance_transition_confusions[horizon],
                prediction[horizon], target[horizon],
                instance_visible_transition[horizon])
            if visibility_probability is not None:
                visibility_counts[horizon] += _binary_counts(
                    visibility_probability[horizon] >= visibility_threshold,
                    target_known[horizon])
            if horizon > 0 and change_probability is not None:
                gate_selection = (
                    current_known[horizon] & target_known[horizon])
                change_gate_counts[horizon - 1] += _binary_counts(
                    (change_probability[horizon - 1] >=
                     change_gate_threshold)[gate_selection],
                    visible_transition[horizon][gate_selection])
            if horizon > 0 and changed_class_prediction is not None:
                _update_confusion(
                    changed_class_confusions[horizon - 1],
                    changed_class_prediction[horizon - 1], target[horizon],
                    visible_transition[horizon])
            target_known_counts[horizon] += int(target_known[horizon].sum())
            current_visible_counts[horizon] += int(
                (target_known[horizon] & current_known[horizon]).sum())
            reveal_counts[horizon] += int(reveal[horizon].sum())
            state_change_counts[horizon] += int(state_change[horizon].sum())
            visible_transition_counts[horizon] += int(
                visible_transition[horizon].sum())
            instance_visible_transition_counts[horizon] += int(
                instance_visible_transition[horizon].sum())

    mean_target_times_s = target_time_sum / len(references)
    semantic = _semantic_summary(confusions, mean_target_times_s)
    semantic['current_visible_subset'] = _semantic_summary(
        visible_confusions, mean_target_times_s)
    semantic['reveal_completion_subset'] = _semantic_summary(
        reveal_confusions, mean_target_times_s)
    semantic['state_change_subset'] = _semantic_summary(
        change_confusions, mean_target_times_s)
    semantic['visible_transition_subset'] = _semantic_summary(
        transition_confusions, mean_target_times_s)
    semantic['instance_related_visible_transition_subset'] = (
        _semantic_summary(
            instance_transition_confusions, mean_target_times_s))
    return {
        'schema_version': 1,
        'manifest_name': manifest.get('name'),
        'split': split,
        'sample_count': len(references),
        'reference_indices': list(references),
        'prediction_source': (
            'constant_current_persistence'
            if prediction_mapping is None else 'exported_model_prediction'),
        'hard_change_gate_applied': bool(apply_change_gate),
        'completion_only_applied': bool(apply_completion_only),
        'flow_only_applied': bool(apply_flow_only),
        'physical_flow_fusion_applied': bool(apply_flow_fusion),
        'local_flow_overlay_applied': bool(apply_local_flow_overlay),
        'physical_confidence_applied': bool(apply_physical_confidence),
        'flow_warp_available': bool(warped_instance_available),
        'physical_confidence_available': bool(
            physical_confidence_available),
        'visibility_threshold': visibility_threshold,
        'change_gate_threshold': change_gate_threshold,
        'flow_fusion_threshold': flow_fusion_threshold,
        'physical_confidence_threshold': physical_confidence_threshold,
        'physical_confidence_flow_threshold': (
            physical_confidence_flow_threshold),
        'mean_target_times_s': mean_target_times_s.tolist(),
        'semantic': semantic,
        'visibility': (
            None if not visibility_available
            else _visibility_summary(
                visibility_counts, mean_target_times_s)),
        'future_change_gate': (
            None if not change_gate_available
            else _visibility_summary(
                change_gate_counts, mean_target_times_s[1:])),
        'future_changed_class_on_transition': (
            None if not changed_class_available
            else _semantic_summary(
                changed_class_confusions, mean_target_times_s[1:])),
        'subset_counts_by_horizon': {
            'target_known': target_known_counts.tolist(),
            'current_visible': current_visible_counts.tolist(),
            'reveal_completion': reveal_counts.tolist(),
            'state_change': state_change_counts.tolist(),
            'visible_transition': visible_transition_counts.tolist(),
            'instance_related_visible_transition': (
                instance_visible_transition_counts.tolist()),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_batch20'))
    parser.add_argument(
        '--split', choices=(
            'train', 'validation', 'test', 'blind', 'final_holdout'),
        default='test')
    parser.add_argument('--prediction-root', type=Path)
    parser.add_argument('--visibility-threshold', type=float, default=0.5)
    parser.add_argument('--change-gate-threshold', type=float, default=0.5)
    parser.add_argument('--apply-change-gate', action='store_true')
    parser.add_argument('--apply-completion-only', action='store_true')
    parser.add_argument('--apply-flow-only', action='store_true')
    parser.add_argument('--apply-flow-fusion', action='store_true')
    parser.add_argument('--apply-local-flow-overlay', action='store_true')
    parser.add_argument('--flow-fusion-threshold', type=float, default=0.5)
    parser.add_argument('--apply-physical-confidence', action='store_true')
    parser.add_argument(
        '--physical-confidence-threshold', type=float, default=0.5)
    parser.add_argument(
        '--physical-confidence-flow-threshold', type=float, default=0.5)
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_persistence_test_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    sequence_mapping = _sequence_mapping(args.sequence_root)
    predictions = (
        None if args.prediction_root is None
        else _prediction_mapping(args.prediction_root))
    summary = evaluate_split(
        manifest=manifest,
        split=args.split,
        sequence_mapping=sequence_mapping,
        prediction_mapping=predictions,
        visibility_threshold=args.visibility_threshold,
        change_gate_threshold=args.change_gate_threshold,
        apply_change_gate=args.apply_change_gate,
        apply_completion_only=args.apply_completion_only,
        apply_flow_only=args.apply_flow_only,
        apply_flow_fusion=args.apply_flow_fusion,
        apply_local_flow_overlay=args.apply_local_flow_overlay,
        flow_fusion_threshold=args.flow_fusion_threshold,
        apply_physical_confidence=args.apply_physical_confidence,
        physical_confidence_threshold=(
            args.physical_confidence_threshold),
        physical_confidence_flow_threshold=(
            args.physical_confidence_flow_threshold))
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
