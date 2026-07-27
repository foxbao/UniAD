from tools.analysis_tools.analyze_kl_occworld_sealed_failures import (
    _distribution,
    _future_mean,
    _rank,
)


def test_failure_analysis_uses_future_horizons_only():
    summary = {
        'by_horizon': [
            {'mean_iou': 0.1},
            {'mean_iou': 0.4},
            {'mean_iou': 0.6},
        ],
    }

    assert _future_mean(summary) == 0.5


def test_failure_analysis_distribution_counts_signs():
    result = _distribution([-0.2, 0.0, 0.1, 0.3])

    assert result['count'] == 4
    assert result['positive_count'] == 2
    assert result['negative_count'] == 1


def test_failure_analysis_ranks_without_losing_scene_identity():
    rows = [
        {'reference_index': 1, 'scene_token': 'a', 'delta': 0.2},
        {'reference_index': 2, 'scene_token': 'b', 'delta': -0.1},
    ]

    result = _rank(rows, 'delta', count=1)

    assert result == [{
        'reference_index': 2,
        'scene_token': 'b',
        'delta': -0.1,
    }]
