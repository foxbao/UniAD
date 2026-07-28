#!/usr/bin/env python
"""Classify fixed-protocol local-flow overlay failures on validation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
    _split_references,
)


OUTCOMES = ('improved', 'harmed', 'still_wrong', 'unknown_target')
SUBSETS = (
    'future', 'current_visible', 'reveal', 'state_change',
    'visible_transition', 'stable_current_visible')
SIGNALS = (
    'event_strength', 'change_probability', 'visibility', 'flow_norm',
    'prior_history_known_count', 'prior_history_current_match_count',
    'prior_history_conflict_count')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def reconstruct_local_overlay(raw: np.ndarray, current_state: np.ndarray,
                              current_valid: np.ndarray,
                              warped: np.ndarray, threshold: float):
    """Rebuild the deployed hard arrival/departure overlay."""
    raw = np.asarray(raw, dtype=np.uint8)
    current_state = np.asarray(current_state, dtype=np.uint8)
    current_valid = np.asarray(current_valid, dtype=np.bool_)
    warped = np.asarray(warped, dtype=np.float32)
    expected = (raw.shape[0] - 1, *raw.shape[1:])
    if raw.ndim != 4 or warped.shape != expected:
        raise ValueError('Expected raw [T,Z,H,W] and warped [T-1,Z,H,W]')
    if current_state.shape != raw.shape[1:] or current_valid.shape != raw.shape[1:]:
        raise ValueError('Current online-input shape does not match prediction')
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('Threshold must be in [0, 1]')
    current_instance = current_valid & (current_state == 3)
    current_instance_float = current_instance[None].astype(np.float32)
    arrival_strength = np.maximum(warped - current_instance_float, 0.0)
    departure_strength = np.maximum(current_instance_float - warped, 0.0)
    arrival = arrival_strength >= threshold
    departure = departure_strength >= threshold
    candidate = raw.copy()
    candidate[1:][arrival] = 2
    candidate[1:][departure] = 0
    return candidate, {
        'arrival': arrival,
        'departure': departure,
        'arrival_strength': arrival_strength,
        'departure_strength': departure_strength,
    }


def classify_changes(raw: np.ndarray, candidate: np.ndarray,
                     target: np.ndarray, target_known: np.ndarray) -> dict:
    """Return disjoint correctness outcomes for effective overlay changes."""
    changed = raw != candidate
    known_changed = changed & target_known
    raw_correct = raw == target
    candidate_correct = candidate == target
    return {
        'changed': changed,
        'improved': known_changed & ~raw_correct & candidate_correct,
        'harmed': known_changed & raw_correct & ~candidate_correct,
        'still_wrong': known_changed & ~raw_correct & ~candidate_correct,
        'unknown_target': changed & ~target_known,
    }


def summarize_signal(values: np.ndarray) -> dict:
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


def _empty_counts() -> dict:
    return {
        'changed_voxels': 0, 'improved_voxels': 0, 'harmed_voxels': 0,
        'still_wrong_voxels': 0, 'unknown_target_voxels': 0,
        'net_correct_voxels': 0}


def _counts(outcomes: dict, selection=None) -> dict:
    if selection is None:
        selection = np.ones_like(outcomes['changed'], dtype=np.bool_)
    result = {
        'changed_voxels': int(np.count_nonzero(
            outcomes['changed'] & selection)),
        'improved_voxels': int(np.count_nonzero(
            outcomes['improved'] & selection)),
        'harmed_voxels': int(np.count_nonzero(
            outcomes['harmed'] & selection)),
        'still_wrong_voxels': int(np.count_nonzero(
            outcomes['still_wrong'] & selection)),
        'unknown_target_voxels': int(np.count_nonzero(
            outcomes['unknown_target'] & selection)),
    }
    result['net_correct_voxels'] = (
        result['improved_voxels'] - result['harmed_voxels'])
    return result


def _add_counts(destination: dict, source: dict):
    for key in destination:
        destination[key] += int(source[key])


def _online_input(path_value, prediction_path: Path, reference: int):
    path = Path(str(path_value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f'Online input recorded by {prediction_path} is missing: {path}')
    with np.load(path, allow_pickle=False) as payload:
        if int(payload['reference_index']) != reference:
            raise ValueError(f'Online input reference mismatch for {reference}')
        state = np.asarray(payload['current_world_state_3d'], dtype=np.uint8)
        valid = np.asarray(payload['current_world_valid_3d'], dtype=np.bool_)
        history_state = np.asarray(
            payload['history_world_state_3d'], dtype=np.uint8)
        history_valid = np.asarray(
            payload['history_world_valid_3d'], dtype=np.bool_)
    valid &= state != 0
    state = state.copy()
    state[~valid] = 0
    history_valid &= history_state != 0
    history_state = history_state.copy()
    history_state[~history_valid] = 0
    if history_state.shape[1:] != state.shape:
        raise ValueError(f'Online history shape mismatch for {reference}')
    return state, valid, history_state, history_valid, path


def _load_prediction(path: Path, reference: int) -> dict:
    required = (
        'world_pred_class_3d', 'raw_world_pred_class_3d',
        'world_valid_probability_3d', 'future_change_probability_3d',
        'future_flow_2d', 'warped_instance_probability_3d',
        'online_input_path')
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f'{path} lacks diagnostics: {missing}')
        result = {key: np.array(payload[key], copy=True) for key in required}
    result['online_input'] = _online_input(
        result['online_input_path'], path, reference)
    return result


def _target_payload(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        state = np.asarray(payload['world_target_state_3d'], dtype=np.uint8)
        valid = np.asarray(payload['world_target_valid_3d'], dtype=np.bool_)
        current_state = np.asarray(
            payload['current_observation_state_3d'], dtype=np.uint8)
        current_valid = np.asarray(
            payload['current_observation_valid_3d'], dtype=np.bool_)
    known = valid & (state != 0)
    target = np.zeros_like(state, dtype=np.uint8)
    target[known] = state[known] - 1
    direct_known = current_valid & (current_state != 0)
    direct_class = np.zeros_like(current_state, dtype=np.uint8)
    direct_class[direct_known] = current_state[direct_known] - 1
    return {
        'target': target, 'known': known,
        'direct_class': direct_class, 'direct_known': direct_known}


def _new_signal_store() -> dict:
    return {
        outcome: {signal: [] for signal in SIGNALS}
        for outcome in OUTCOMES}


def _append_signals(store: dict, outcomes: dict, signals: dict, selection):
    for outcome in OUTCOMES:
        mask = outcomes[outcome] & selection
        for name, values in signals.items():
            if np.any(mask):
                store[outcome][name].append(
                    np.asarray(values[mask], dtype=np.float32))


def _summarize_signal_store(store: dict) -> dict:
    result = {}
    for outcome, signals in store.items():
        result[outcome] = {}
        for name, chunks in signals.items():
            values = (np.concatenate(chunks) if chunks
                      else np.empty(0, dtype=np.float32))
            result[outcome][name] = summarize_signal(values)
    return result


def audit(manifest: dict, sequences: dict, predictions: dict, split: str,
          threshold: float) -> dict:
    references = _split_references(manifest, split)
    if set(references).difference(sequences):
        raise ValueError('Validation labels are incomplete')
    if set(predictions) != set(references):
        raise ValueError('Predictions must match the exact validation split')

    total = _empty_counts()
    by_event = {name: _empty_counts() for name in ('arrival', 'departure')}
    by_subset = {name: _empty_counts() for name in SUBSETS}
    by_horizon = [_empty_counts() for _ in range(4)]
    by_anchor_agreement = {
        name: _empty_counts() for name in ('agree', 'disagree')}
    by_raw_current_agreement = {
        name: _empty_counts()
        for name in ('raw_matches_current', 'raw_differs_current',
                     'current_unknown')}
    by_history_consistency = {
        name: _empty_counts()
        for name in ('all_known_prior_match_current',
                     'prior_conflicts_with_current',
                     'no_known_prior', 'current_unknown')}
    target_raw_matrices = {
        name: np.zeros((3, 3), dtype=np.int64)
        for name in ('arrival', 'departure')}
    event_binary = {
        name: {'tp': 0, 'fp': 0, 'fn': 0}
        for name in ('arrival', 'departure')}
    signal_store = _new_signal_store()
    rows = []
    parity_difference_voxels = 0
    parity_quantization_explained_voxels = 0
    parity_unexplained_voxels = 0
    online_paths = []

    for reference in references:
        prediction_path = predictions[reference]
        diagnostic = _load_prediction(prediction_path, reference)
        labels = _target_payload(sequences[reference])
        (online_state, online_valid, history_state, history_valid,
         online_path) = diagnostic['online_input']
        online_paths.append(_display_path(online_path))
        raw = np.asarray(diagnostic['raw_world_pred_class_3d'], dtype=np.uint8)
        saved = np.asarray(diagnostic['world_pred_class_3d'], dtype=np.uint8)
        warped = np.asarray(
            diagnostic['warped_instance_probability_3d'], dtype=np.float32)
        reconstructed, events = reconstruct_local_overlay(
            raw, online_state, online_valid, warped, threshold)
        parity_current = reconstructed[0] != saved[0]
        parity_future = reconstructed[1:] != saved[1:]
        threshold_distance = np.minimum(
            np.abs(events['arrival_strength'] - threshold),
            np.abs(events['departure_strength'] - threshold))
        quantization_tolerance = float(np.spacing(np.float16(threshold)))
        quantization_explained = (
            parity_future & (threshold_distance <= quantization_tolerance))
        parity_difference_voxels += int(np.count_nonzero(parity_current))
        parity_difference_voxels += int(np.count_nonzero(parity_future))
        parity_quantization_explained_voxels += int(
            np.count_nonzero(quantization_explained))
        parity_unexplained_voxels += int(np.count_nonzero(parity_current))
        parity_unexplained_voxels += int(np.count_nonzero(
            parity_future & ~quantization_explained))

        # The saved prediction is the actual model output. Recover effective
        # source-float32 events that disappeared when diagnostic probabilities
        # were quantized to float16 at the threshold boundary.
        effective_arrival = (saved[1:] == 2) & (raw[1:] != 2)
        effective_departure = (saved[1:] == 0) & (raw[1:] != 0)
        events['arrival'] |= effective_arrival
        events['departure'] |= effective_departure
        candidate = saved

        target = labels['target'][1:]
        target_known = labels['known'][1:]
        outcomes = classify_changes(
            raw[1:], candidate[1:], target, target_known)
        local_total = _counts(outcomes)
        _add_counts(total, local_total)

        direct_known = np.broadcast_to(
            labels['direct_known'], target.shape)
        current_visible = target_known & direct_known
        reveal = target_known & ~direct_known
        persistence = np.broadcast_to(
            labels['direct_class'], target.shape)
        state_change = target_known & (target != persistence)
        subsets = {
            'future': target_known,
            'current_visible': current_visible,
            'reveal': reveal,
            'state_change': state_change,
            'visible_transition': state_change & direct_known,
            'stable_current_visible': current_visible & ~state_change,
        }
        for name, selection in subsets.items():
            _add_counts(by_subset[name], _counts(outcomes, selection))
        for horizon in range(4):
            selection = np.zeros_like(target_known)
            selection[horizon] = True
            _add_counts(by_horizon[horizon], _counts(outcomes, selection))

        online_instance = online_valid & (online_state == 3)
        direct_instance = labels['direct_known'] & (labels['direct_class'] == 2)
        anchor_agree = np.broadcast_to(
            online_instance == direct_instance, target.shape)
        for name, selection in (('agree', anchor_agree),
                                ('disagree', ~anchor_agree)):
            _add_counts(
                by_anchor_agreement[name], _counts(outcomes, selection))

        online_class = np.zeros_like(online_state, dtype=np.uint8)
        online_class[online_valid] = online_state[online_valid] - 1
        online_class_future = np.broadcast_to(online_class, target.shape)
        online_known_future = np.broadcast_to(online_valid, target.shape)
        raw_current_selections = {
            'raw_matches_current': (
                online_known_future & (raw[1:] == online_class_future)),
            'raw_differs_current': (
                online_known_future & (raw[1:] != online_class_future)),
            'current_unknown': ~online_known_future,
        }
        for name, selection in raw_current_selections.items():
            _add_counts(
                by_raw_current_agreement[name], _counts(outcomes, selection))

        # The final history frame is the current observation, so temporal
        # consistency must be measured only from the earlier causal frames.
        prior_state = history_state[:-1]
        prior_valid = history_valid[:-1]
        prior_known_count = prior_valid.sum(axis=0).astype(np.float32)
        prior_match_count = (
            prior_valid & (prior_state == online_state[None])).sum(
                axis=0).astype(np.float32)
        prior_conflict_count = (
            prior_valid & (prior_state != online_state[None])).sum(
                axis=0).astype(np.float32)
        has_prior = prior_known_count > 0
        history_selections = {
            'all_known_prior_match_current': (
                online_valid & has_prior & (prior_conflict_count == 0)),
            'prior_conflicts_with_current': (
                online_valid & (prior_conflict_count > 0)),
            'no_known_prior': online_valid & ~has_prior,
            'current_unknown': ~online_valid,
        }
        for name, selection_3d in history_selections.items():
            _add_counts(
                by_history_consistency[name],
                _counts(outcomes, np.broadcast_to(selection_3d, target.shape)))

        flow_norm = np.linalg.norm(
            np.asarray(diagnostic['future_flow_2d'], dtype=np.float32),
            axis=1)
        flow_norm = np.broadcast_to(flow_norm[:, None], target.shape)
        signals = {
            'event_strength': np.maximum(
                events['arrival_strength'], events['departure_strength']),
            'change_probability': np.asarray(
                diagnostic['future_change_probability_3d'],
                dtype=np.float32),
            'visibility': np.asarray(
                diagnostic['world_valid_probability_3d'][1:],
                dtype=np.float32),
            'flow_norm': flow_norm,
            'prior_history_known_count': np.broadcast_to(
                prior_known_count, target.shape),
            'prior_history_current_match_count': np.broadcast_to(
                prior_match_count, target.shape),
            'prior_history_conflict_count': np.broadcast_to(
                prior_conflict_count, target.shape),
        }
        _append_signals(
            signal_store, outcomes, signals, outcomes['changed'])

        online_instance_future = np.broadcast_to(online_instance, target.shape)
        true_events = {
            'arrival': target_known & ~online_instance_future & (target == 2),
            'departure': target_known & online_instance_future & (target == 0),
        }
        for name in ('arrival', 'departure'):
            predicted = events[name] & target_known
            truth = true_events[name]
            event_binary[name]['tp'] += int(np.count_nonzero(predicted & truth))
            event_binary[name]['fp'] += int(np.count_nonzero(predicted & ~truth))
            event_binary[name]['fn'] += int(np.count_nonzero(~predicted & truth))
            _add_counts(by_event[name], _counts(outcomes, events[name]))
            changed = outcomes['changed'] & events[name] & target_known
            encoded = target[changed].astype(np.int64) * 3 + (
                raw[1:][changed].astype(np.int64))
            target_raw_matrices[name] += np.bincount(
                encoded, minlength=9).reshape(3, 3)

        rows.append({
            'reference_index': int(reference),
            'prediction_path': _display_path(prediction_path),
            'corrections': local_total,
        })

    event_summary = {}
    for name, values in event_binary.items():
        precision = values['tp'] / max(values['tp'] + values['fp'], 1)
        recall = values['tp'] / max(values['tp'] + values['fn'], 1)
        event_summary[name] = {
            **values, 'precision': float(precision), 'recall': float(recall),
            'f1': float(2 * precision * recall / max(precision + recall, 1e-12))}

    rows.sort(
        key=lambda row: row['corrections']['net_correct_voxels'])
    return {
        'schema_version': 1,
        'purpose': 'Fixed-protocol validation failure classification.',
        'split': split,
        'reference_count': len(references),
        'reference_indices': references,
        'event_threshold': float(threshold),
        'model_inference_performed': False,
        'threshold_scan_performed': False,
        'parity': {
            'candidate_vs_saved_difference_voxels': parity_difference_voxels,
            'float16_threshold_quantization_explained_voxels': (
                parity_quantization_explained_voxels),
            'unexplained_difference_voxels': parity_unexplained_voxels,
            'exact': parity_difference_voxels == 0,
            'accepted': parity_unexplained_voxels == 0,
            'note': (
                'Saved world_pred_class_3d is authoritative. Differences are '
                'accepted only when the exported float16 event strength lies '
                'within one float16 spacing of the frozen threshold.')},
        'corrections': total,
        'corrections_by_event': by_event,
        'corrections_by_subset': by_subset,
        'corrections_by_horizon': by_horizon,
        'corrections_by_online_vs_direct_instance_anchor': (
            by_anchor_agreement),
        'corrections_by_raw_vs_online_current': by_raw_current_agreement,
        'corrections_by_prior_history_consistency': by_history_consistency,
        'changed_voxel_target_rows_raw_columns': {
            name: matrix.tolist()
            for name, matrix in target_raw_matrices.items()},
        'event_detection_against_online_anchor_target': event_summary,
        'causal_signal_distributions_by_outcome': (
            _summarize_signal_store(signal_store)),
        'rows_worst_first': rows,
        'online_input_paths': online_paths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_full_train3_final30_manifest_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b17a_epoch3_full_history_'
            'validation_raw_diagnostic_v1/validation/epoch_003'))
    parser.add_argument('--split', choices=('validation',), default='validation')
    parser.add_argument('--event-threshold', type=float, default=0.9)
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b24_local_overlay_failure_audit_validation_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    report = audit(
        manifest, _sequence_mapping(args.sequence_root),
        _prediction_mapping(args.prediction_root), args.split,
        args.event_threshold)
    report['inputs'] = {
        'manifest': str(args.manifest),
        'manifest_sha256': _sha256(args.manifest),
        'sequence_root': str(args.sequence_root),
        'prediction_root': str(args.prediction_root),
    }
    if not report['parity']['accepted']:
        raise RuntimeError(
            'Reconstructed local overlay does not match saved model output')
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    print(json.dumps({
        'parity': report['parity'],
        'corrections': report['corrections'],
        'corrections_by_event': report['corrections_by_event'],
        'corrections_by_subset': report['corrections_by_subset'],
        'event_detection': report[
            'event_detection_against_online_anchor_target'],
        'signals': report['causal_signal_distributions_by_outcome'],
        'worst_scenes': report['rows_worst_first'][:5],
        'out_file': str(args.out_file),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
