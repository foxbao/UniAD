import numpy as np

from tools.analysis_tools.analyze_kl_occworld_visibility_calibration import (
    apply_affine_logit,
    binary_metrics,
    calibration_metrics,
    fit_affine_logit,
    reference_folds,
)


def test_affine_logit_fit_improves_overconfident_probabilities():
    target = np.asarray([0] * 800 + [1] * 200, dtype=bool)
    truthful = np.concatenate([
        np.linspace(0.01, 0.35, 800),
        np.linspace(0.65, 0.99, 200),
    ]).astype(np.float32)
    overconfident = apply_affine_logit(truthful, 1.8, 0.7)

    parameters = fit_affine_logit(
        overconfident, target, bin_count=128)
    calibrated = apply_affine_logit(
        overconfident, parameters['scale'], parameters['bias'])

    before = calibration_metrics(target, overconfident)
    after = calibration_metrics(target, calibrated)
    assert parameters['scale'] > 0
    assert after['negative_log_likelihood'] < (
        before['negative_log_likelihood'])
    assert after['brier_score'] < before['brier_score']


def test_binary_metrics_reports_precision_recall_and_f1():
    target = np.asarray([True, True, False, False])
    probability = np.asarray([0.9, 0.6, 0.8, 0.1])

    metrics = binary_metrics(target, probability, threshold=0.7)

    assert metrics['true_positive'] == 1
    assert metrics['false_positive'] == 1
    assert metrics['false_negative'] == 1
    assert metrics['true_negative'] == 1
    assert metrics['precision'] == 0.5
    assert metrics['recall'] == 0.5
    assert metrics['f1'] == 0.5


def test_reference_folds_are_disjoint_and_complete():
    folds = reference_folds([9, 2, 7, 4, 6, 1])

    assert set(folds[0]).isdisjoint(folds[1])
    assert set(folds[0]) | set(folds[1]) == {1, 2, 4, 6, 7, 9}
    assert folds == [[1, 4, 7], [2, 6, 9]]
