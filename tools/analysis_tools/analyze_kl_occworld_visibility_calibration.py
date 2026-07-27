#!/usr/bin/env python
"""Audit OccWorld visibility ranking and calibration on validation only."""

import argparse
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
)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(probability) - np.log1p(-probability)


def apply_affine_logit(probability: np.ndarray, scale: float,
                       bias: float) -> np.ndarray:
    """Apply a monotonic two-parameter calibration in logit space."""
    if scale <= 0:
        raise ValueError('Affine-logit calibration scale must be positive')
    return _sigmoid(scale * _logit(probability) + bias).astype(
        np.float32, copy=False)


def _histogram_sufficient_statistics(probability: np.ndarray,
                                     target: np.ndarray,
                                     bin_count: int):
    if probability.shape != target.shape:
        raise ValueError('Calibration probability and target shapes differ')
    if bin_count < 16:
        raise ValueError('Calibration fit requires at least 16 bins')
    logits = _logit(probability).reshape(-1)
    target = target.astype(np.float64, copy=False).reshape(-1)
    low, high = -13.82, 13.82
    edges = np.linspace(low, high, bin_count + 1, dtype=np.float64)
    total, _ = np.histogram(logits, bins=edges)
    positive, _ = np.histogram(logits, bins=edges, weights=target)
    centers = (edges[:-1] + edges[1:]) * 0.5
    selected = total > 0
    return (centers[selected], total[selected].astype(np.float64),
            positive[selected].astype(np.float64))


def fit_affine_logit(probability: np.ndarray, target: np.ndarray,
                     bin_count: int = 4096,
                     max_iterations: int = 40) -> dict:
    """Fit positive-scale Platt calibration using binned Newton updates."""
    centers, total, positive = _histogram_sufficient_statistics(
        probability, target, bin_count)
    negative = total - positive
    sample_count = float(total.sum())

    def objective(scale, bias):
        calibrated = _sigmoid(scale * centers + bias)
        loss = -(
            positive * np.log(np.clip(calibrated, 1e-12, 1.0)) +
            negative * np.log(np.clip(1.0 - calibrated, 1e-12, 1.0))
        ).sum() / sample_count
        return float(loss)

    scale, bias = 1.0, 0.0
    loss = objective(scale, bias)
    completed = 0
    for iteration in range(max_iterations):
        calibrated = _sigmoid(scale * centers + bias)
        residual = calibrated * total - positive
        gradient = np.asarray([
            np.sum(residual * centers), np.sum(residual)
        ], dtype=np.float64) / sample_count
        weight = total * calibrated * (1.0 - calibrated)
        hessian = np.asarray([
            [np.sum(weight * centers * centers),
             np.sum(weight * centers)],
            [np.sum(weight * centers), np.sum(weight)],
        ], dtype=np.float64) / sample_count
        hessian += np.eye(2, dtype=np.float64) * 1e-8
        step = np.linalg.solve(hessian, gradient)
        accepted = False
        damping = 1.0
        for _ in range(20):
            candidate_scale = scale - damping * float(step[0])
            candidate_bias = bias - damping * float(step[1])
            if candidate_scale > 1e-4:
                candidate_loss = objective(
                    candidate_scale, candidate_bias)
                if candidate_loss <= loss:
                    scale, bias, loss = (
                        candidate_scale, candidate_bias, candidate_loss)
                    accepted = True
                    break
            damping *= 0.5
        completed = iteration + 1
        if not accepted or np.max(np.abs(damping * step)) < 1e-7:
            break
    return {
        'scale': float(scale),
        'bias': float(bias),
        'fit_negative_log_likelihood': float(loss),
        'iterations': int(completed),
        'histogram_bin_count': int(bin_count),
    }


def binary_metrics(target: np.ndarray, probability: np.ndarray,
                   threshold: float) -> dict:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('Visibility threshold must be in [0, 1]')
    target = target.astype(bool, copy=False)
    prediction = probability >= threshold
    true_positive = int(np.count_nonzero(prediction & target))
    false_positive = int(np.count_nonzero(prediction & ~target))
    false_negative = int(np.count_nonzero(~prediction & target))
    true_negative = int(np.count_nonzero(~prediction & ~target))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        'threshold': float(threshold),
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_negative': true_negative,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(
            true_positive /
            max(true_positive + false_positive + false_negative, 1)),
        'predicted_positive_ratio': float(prediction.mean()),
    }


def calibration_metrics(target: np.ndarray, probability: np.ndarray,
                        bin_count: int = 15) -> dict:
    if target.shape != probability.shape:
        raise ValueError('Calibration metric inputs must have matching shapes')
    target = target.astype(np.float64, copy=False).reshape(-1)
    probability = np.clip(
        probability.astype(np.float64, copy=False).reshape(-1),
        1e-7, 1.0 - 1e-7)
    brier = np.mean((probability - target) ** 2)
    nll = -np.mean(
        target * np.log(probability) +
        (1.0 - target) * np.log(1.0 - probability))
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    indices = np.minimum(
        np.searchsorted(edges, probability, side='right') - 1,
        bin_count - 1)
    rows = []
    expected_error = 0.0
    for index in range(bin_count):
        selected = indices == index
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        confidence = float(probability[selected].mean())
        observed = float(target[selected].mean())
        expected_error += count / len(target) * abs(confidence - observed)
        rows.append({
            'lower': float(edges[index]),
            'upper': float(edges[index + 1]),
            'count': count,
            'mean_probability': confidence,
            'observed_positive_ratio': observed,
        })
    return {
        'sample_count': int(len(target)),
        'positive_ratio': float(target.mean()),
        'brier_score': float(brier),
        'negative_log_likelihood': float(nll),
        'expected_calibration_error': float(expected_error),
        'bins': rows,
    }


def reference_folds(references) -> list:
    references = sorted(int(reference) for reference in references)
    if len(references) < 4:
        raise ValueError('Cross-scene calibration needs at least four scenes')
    folds = [references[::2], references[1::2]]
    if not folds[0] or not folds[1]:
        raise ValueError('Calibration folds must both be non-empty')
    return folds


def _concatenate(payloads: dict, references, key: str) -> np.ndarray:
    return np.concatenate([
        payloads[int(reference)][key].reshape(-1)
        for reference in references
    ])


def _probability_report(target, probability) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        'ranking': {
            'roc_auc': float(roc_auc_score(target, probability)),
            'average_precision': float(
                average_precision_score(target, probability)),
        },
        'calibration': calibration_metrics(target, probability),
        'binary_at_0_5': binary_metrics(target, probability, 0.5),
        'binary_at_0_7': binary_metrics(target, probability, 0.7),
    }


def analyze(payloads: dict, thresholds, bin_count: int) -> dict:
    references = sorted(payloads)
    all_target = _concatenate(payloads, references, 'target')
    all_probability = _concatenate(payloads, references, 'probability')
    folds = reference_folds(references)
    calibrated_by_reference = {}
    fold_rows = []
    for held_out_index, held_out_references in enumerate(folds):
        fit_references = folds[1 - held_out_index]
        fit_target = _concatenate(payloads, fit_references, 'target')
        fit_probability = _concatenate(
            payloads, fit_references, 'probability')
        parameters = fit_affine_logit(
            fit_probability, fit_target, bin_count=bin_count)
        for reference in held_out_references:
            calibrated_by_reference[reference] = apply_affine_logit(
                payloads[reference]['probability'],
                parameters['scale'], parameters['bias'])
        fold_rows.append({
            'held_out_fold': int(held_out_index),
            'fit_references': fit_references,
            'held_out_references': held_out_references,
            'parameters': parameters,
        })
    cross_validated_probability = np.concatenate([
        calibrated_by_reference[reference].reshape(-1)
        for reference in references
    ])
    all_fit = fit_affine_logit(
        all_probability, all_target, bin_count=bin_count)
    return {
        'reference_folds': folds,
        'raw': _probability_report(all_target, all_probability),
        'threshold_sweep': [
            binary_metrics(all_target, all_probability, threshold)
            for threshold in thresholds
        ],
        'cross_validated_affine_logit': {
            'folds': fold_rows,
            'report': _probability_report(
                all_target, cross_validated_probability),
        },
        'all_validation_fit_for_future_candidate_only': all_fit,
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
            'occworld_predictions_b17a_full_history_validation_v1/'
            'validation/epoch_003'))
    parser.add_argument('--split', default='validation')
    parser.add_argument(
        '--thresholds', type=float, nargs='+',
        default=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9))
    parser.add_argument('--fit-bin-count', type=int, default=4096)
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b17a_visibility_calibration_cv_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split != 'validation':
        raise ValueError('Visibility calibration development is validation-only')
    manifest = _load_manifest(args.manifest)
    records = manifest['splits'][args.split]
    references = [int(record['reference_index']) for record in records]
    sequences = _sequence_mapping(args.sequence_root)
    predictions = _prediction_mapping(args.prediction_root)
    if set(predictions) != set(references):
        raise ValueError('Calibration predictions are not the exact split')
    if not set(references).issubset(sequences):
        raise ValueError('Calibration labels are incomplete')
    payloads = {}
    for reference in references:
        with np.load(sequences[reference], allow_pickle=False) as label:
            target = np.asarray(
                label['world_target_valid_3d'], dtype=bool)
        with np.load(predictions[reference], allow_pickle=False) as prediction:
            probability = np.asarray(
                prediction['world_valid_probability_3d'],
                dtype=np.float32)
        if target.shape != probability.shape:
            raise ValueError(
                f'Visibility shape mismatch for reference {reference}')
        payloads[reference] = {
            'target': target,
            'probability': probability,
        }
    report = {
        'schema_version': 1,
        'name': args.out_file.stem,
        'split': args.split,
        'purpose': 'development_validation',
        'model_inference_performed': False,
        'network_training_performed': False,
        'holdout_accessed': False,
        'manifest': str(args.manifest),
        'sequence_root': str(args.sequence_root),
        'prediction_root': str(args.prediction_root),
        'reference_indices': references,
        **analyze(payloads, args.thresholds, args.fit_bin_count),
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'name': report['name'],
        'raw': report['raw'],
        'cross_validated_affine_logit': report[
            'cross_validated_affine_logit'],
        'all_validation_fit_for_future_candidate_only': report[
            'all_validation_fit_for_future_candidate_only'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
