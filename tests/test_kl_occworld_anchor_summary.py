import pytest

from tools.analysis_tools.summarize_kl_occworld_anchor_substitution import (
    _delta,
    _summary_row,
)


def _report():
    def rows(values):
        return [{'mean_iou': value} for value in values]
    return {
        'semantic': {
            'overall': {'mean_iou': 0.8},
            'by_horizon': rows([0.9, 0.7, 0.5]),
            'reveal_completion_subset': {
                'by_horizon': rows([0.8, 0.6, 0.4])},
            'state_change_subset': {
                'by_horizon': rows([0.7, 0.3, 0.5])},
            'visible_transition_subset': {
                'by_horizon': rows([0.6, 0.2, 0.4])},
            'instance_related_visible_transition_subset': {
                'by_horizon': rows([0.0, 0.1, 0.3])},
        },
        'visibility': {'overall': {'f1': 0.75}},
    }


def test_anchor_summary_uses_only_future_horizons_and_reports_delta():
    baseline = _summary_row(_report())
    candidate = dict(baseline)
    candidate['future_mean_iou'] = 0.5

    assert baseline['future_mean_iou'] == pytest.approx(0.6)
    assert baseline['future_reveal_mean_iou'] == pytest.approx(0.5)
    assert _delta(candidate, baseline)['future_mean_iou'] == pytest.approx(
        -0.1)
