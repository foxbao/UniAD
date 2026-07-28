import numpy as np

from tools.analysis_tools.audit_kl_occworld_local_overlay_failures import (
    classify_changes,
    reconstruct_local_overlay,
    summarize_signal,
)


def test_reconstruct_local_overlay_keeps_raw_outside_fixed_events():
    raw = np.array([[[[0, 2, 1]]], [[[0, 2, 1]]]], dtype=np.uint8)
    state = np.array([[[1, 3, 2]]], dtype=np.uint8)
    valid = np.ones_like(state, dtype=np.bool_)
    warped = np.array([[[[0.95, 0.05, 0.4]]]], dtype=np.float32)

    candidate, events = reconstruct_local_overlay(
        raw, state, valid, warped, threshold=0.9)

    assert events['arrival'].tolist() == [[[[True, False, False]]]]
    assert events['departure'].tolist() == [[[[False, True, False]]]]
    assert candidate[1].tolist() == [[[2, 0, 1]]]
    assert candidate[0].tolist() == raw[0].tolist()


def test_classify_changes_produces_disjoint_failure_outcomes():
    raw = np.array([0, 0, 1, 2, 0], dtype=np.uint8)
    candidate = np.array([2, 2, 2, 0, 2], dtype=np.uint8)
    target = np.array([2, 0, 0, 2, 2], dtype=np.uint8)
    known = np.array([True, True, True, True, False])

    result = classify_changes(raw, candidate, target, known)

    assert np.flatnonzero(result['improved']).tolist() == [0]
    assert np.flatnonzero(result['harmed']).tolist() == [1, 3]
    assert np.flatnonzero(result['still_wrong']).tolist() == [2]
    assert np.flatnonzero(result['unknown_target']).tolist() == [4]
    total = sum(result[name].astype(np.int64) for name in (
        'improved', 'harmed', 'still_wrong', 'unknown_target'))
    assert np.array_equal(total.astype(np.bool_), result['changed'])


def test_summarize_signal_handles_empty_and_nonempty_values():
    assert summarize_signal(np.array([]))['mean'] is None
    summary = summarize_signal(np.array([0.1, 0.5, 0.9]))
    assert summary['count'] == 3
    assert summary['median'] == 0.5


def test_float16_spacing_covers_threshold_rounding_case():
    stored = np.float16(0.9).astype(np.float32)
    tolerance = float(np.spacing(np.float16(0.9)))

    assert stored < 0.9
    assert abs(float(stored) - 0.9) <= tolerance
