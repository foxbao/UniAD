from tools.analysis_tools.run_kl_occworld_b17_fresh_holdout import (
    _generation_view,
    validate_fresh_records,
)


def _record(index):
    return {
        'reference_index': index,
        'scene_token': f'fresh-{index}',
        'sample_token': f'token-{index}',
        'timestamp': float(index),
    }


def test_b17_fresh_holdout_records_are_scene_disjoint():
    fresh = {
        'status': 'frozen_before_label_generation_and_inference',
        'selection_uses_occworld_labels': False,
        'selection_uses_model_predictions': False,
        'threshold_retuning_allowed': False,
        'splits': {'fresh_holdout': [_record(1), _record(2)]},
    }
    existing = {
        'splits': {'train': [{
            'reference_index': 10,
            'scene_token': 'old-10',
        }]},
    }

    records = validate_fresh_records(
        fresh, existing, expected_count=2)

    assert [record['reference_index'] for record in records] == [1, 2]


def test_b17_fresh_holdout_rejects_existing_scene_overlap():
    fresh = {
        'status': 'frozen_before_label_generation_and_inference',
        'selection_uses_occworld_labels': False,
        'selection_uses_model_predictions': False,
        'threshold_retuning_allowed': False,
        'splits': {'fresh_holdout': [_record(1)]},
    }
    existing = {
        'splits': {'train': [{
            'reference_index': 10,
            'scene_token': 'fresh-1',
        }]},
    }

    try:
        validate_fresh_records(fresh, existing, expected_count=1)
    except ValueError as error:
        assert 'overlaps existing scenes' in str(error)
    else:
        raise AssertionError('Expected scene overlap to fail')


def test_b17_generation_view_preserves_annotation_and_hides_holdout_name():
    manifest = {
        'status': 'frozen_before_fresh_holdout_gt_generation',
        'annotation_file': 'data/kl_8/kl_infos_val.pkl',
        'fresh_holdout_reference_count': 1,
        'splits': {'fresh_holdout': [_record(3)]},
    }

    view = _generation_view(manifest)

    assert view['status'] == 'frozen_before_full_gt_generation'
    assert view['annotation_file'] == 'data/kl_8/kl_infos_val.pkl'
    assert view['train_reference_count'] == 1
    assert view['splits'] == {'train': [_record(3)]}
