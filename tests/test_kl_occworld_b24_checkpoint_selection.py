from tools.analysis_tools.select_kl_occworld_b24_checkpoint import select


def _evaluation(semantic, instance, transition, instance_transition):
    def section(value):
        return {'by_horizon': [
            {'mean_iou': value},
            *[{'mean_iou': value} for _ in range(4)],
        ]}

    return {'semantic': {
        'by_horizon': [
            {'mean_iou': semantic,
             'iou': {'instance_occupied': instance}},
            *[{'mean_iou': semantic,
               'iou': {'instance_occupied': instance}} for _ in range(4)],
        ],
        'visible_transition_subset': section(transition),
        'instance_related_visible_transition_subset': section(
            instance_transition),
    }}


def _audit(net, current=0, non_event=0):
    return {
        'current_difference_voxels': current,
        'non_event_difference_voxels': non_event,
        'corrections': {'net_correct_voxels': net},
    }


def test_b24_selector_uses_frozen_ranking_and_non_regression():
    baseline = _evaluation(0.80, 0.70, 0.20, 0.25)
    result = select([
        (1, baseline, _evaluation(0.81, 0.71, 0.21, 0.26), _audit(3)),
        (2, baseline, _evaluation(0.82, 0.72, 0.19, 0.27), _audit(8)),
        (3, baseline, _evaluation(0.81, 0.73, 0.21, 0.26), _audit(5)),
    ])

    assert result['selected_epoch'] == 3
    assert result['candidates_ranked'][0]['epoch'] == 2
    assert not result['candidates_ranked'][0]['eligible']
    assert result['exploratory_pass']


def test_b24_selector_rejects_spatial_invariant_violation():
    baseline = _evaluation(0.80, 0.70, 0.20, 0.25)
    result = select([
        (1, baseline, _evaluation(0.90, 0.90, 0.30, 0.35),
         _audit(100, non_event=1)),
    ])

    assert result['selected_epoch'] is None
    assert not result['exploratory_pass']
