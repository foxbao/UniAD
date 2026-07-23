from tools.analysis_tools.build_kl_occworld_blind_holdout import (
    replace_pre_inference_invalid_record,
    select_blind_records,
)
from tools.analysis_tools.select_kl_occworld_expanded_scenes import (
    _evenly_spaced,
    audit_scene_references,
    select_split_records,
)
from tools.analysis_tools.select_kl_occworld_dense_train_references import (
    spread_scene_references,
)


def _toy_infos(scene_count=5, frames_per_scene=7):
    infos = []
    for scene_index in range(scene_count):
        for frame_index in range(frames_per_scene):
            infos.append({
                'scene_token': f'scene-{scene_index}',
                'token': f'{scene_index}-{frame_index}',
                'timestamp': frame_index * 0.5,
            })
    return infos


def test_even_selection_covers_both_ends_without_duplicates():
    items = [{'value': value} for value in range(10)]

    selected = _evenly_spaced(items, 4)

    assert [item['value'] for item in selected] == [0, 3, 6, 9]


def test_dense_reference_selection_keeps_base_and_spreads_new_samples():
    selected = spread_scene_references(
        candidates=list(range(10, 51)), required=[30],
        count=3, min_separation=8)

    assert selected == [10, 30, 50]


def test_dense_reference_selection_rejects_impossible_spacing():
    try:
        spread_scene_references(
            candidates=[10, 12, 14], required=[12],
            count=2, min_separation=5)
    except ValueError as error:
        assert 'satisfy separation' in str(error)
    else:
        raise AssertionError('Expected impossible dense selection to fail')


def test_scene_audit_selects_center_reference_with_full_time_window():
    eligible, rows = audit_scene_references(
        _toy_infos(scene_count=2, frames_per_scene=7),
        required_offsets=[-2, -1, 0, 1, 2],
        expected_step_s=0.5,
        max_time_error_s=0.01,
        check_lidar_files=False)

    assert [record['reference_index'] for record in eligible] == [3, 10]
    assert [row['status'] for row in rows] == ['eligible', 'eligible']


def test_scene_split_keeps_reused_scenes_only_in_train():
    eligible = [{
        'reference_index': index,
        'scene_token': f'scene-{index}',
        'sample_token': f'token-{index}',
        'timestamp': float(index),
    } for index in range(8)]
    reused = [{
        'reference_index': 2,
        'scene_token': 'scene-2',
        'sample_token': 'old-token-2',
        'timestamp': 2.0,
    }]

    splits = select_split_records(
        eligible, reused,
        train_scene_count=3,
        validation_scene_count=1,
        test_scene_count=1)

    assert [len(splits[name]) for name in (
        'train', 'validation', 'test')] == [3, 1, 1]
    assert any(record['scene_token'] == 'scene-2'
               for record in splits['train'])
    assert all(record['scene_token'] != 'scene-2'
               for name in ('validation', 'test')
               for record in splits[name])
    all_scenes = [
        record['scene_token']
        for records in splits.values()
        for record in records
    ]
    assert len(all_scenes) == len(set(all_scenes))


def test_blind_selection_excludes_every_prior_split_and_is_even():
    eligible = [{
        'reference_index': index,
        'scene_token': f'scene-{index}',
        'sample_token': f'token-{index}',
        'timestamp': float(index),
    } for index in range(10)]
    previous = {
        'splits': {
            'train': [eligible[0]],
            'validation': [eligible[4]],
            'test': [eligible[9]],
        },
    }

    selected = select_blind_records(eligible, previous, scene_count=3)

    assert [record['reference_index'] for record in selected] == [1, 5, 8]
    assert all(record['scene_token'] not in {
        'scene-0', 'scene-4', 'scene-9'} for record in selected)
    assert all(record['selection_source'] ==
               'fresh_evenly_spaced_before_blind_inference'
               for record in selected)


def test_blind_replacement_requires_fresh_complete_model_queue():
    infos = [{
        'scene_token': 'scene-invalid',
        'token': 'invalid',
        'prev': '',
        'timestamp': 0.0,
    }]
    for index in range(5):
        infos.append({
            'scene_token': 'scene-replacement',
            'token': f'replacement-{index}',
            'prev': '' if index == 0 else f'replacement-{index - 1}',
            'timestamp': index * 0.5,
        })
    manifest = {
        'splits': {'blind': [{
            'reference_index': 0,
            'scene_token': 'scene-invalid',
            'sample_token': 'invalid',
            'timestamp': 0.0,
        }]},
    }
    excluded_manifest = {
        'splits': {'train': [{
            'reference_index': 9,
            'scene_token': 'scene-old',
        }]},
    }

    revised = replace_pre_inference_invalid_record(
        manifest, infos, excluded_manifest,
        invalid_reference=0, replacement_reference=5,
        reason='missing prev chain')

    assert revised['selected_reference_indices'] == [5]
    assert revised['splits']['blind'][0]['scene_token'] == (
        'scene-replacement')
    assert revised['pre_inference_exclusions'][0][
        'model_predictions_inspected_before_replacement'] is False
