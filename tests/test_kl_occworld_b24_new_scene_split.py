from pathlib import Path

import numpy as np

from tools.analysis_tools.prepare_kl_occworld_b24_new_scene_split import (
    annotation_scene_manifest,
    build_b24_split_manifest,
    select_b24_new_scene_records,
    split_interleaved_records,
    validate_ego_pose_window,
)


def _records(count):
    return [{
        'reference_index': index,
        'scene_token': f'scene-{index}',
        'sample_token': f'token-{index}',
        'timestamp': float(index),
    } for index in range(count)]


def test_b24_selection_excludes_consumed_and_interleaves_timeline():
    records = _records(10)
    existing = {'splits': {'old': [records[0], records[2]]}}

    splits, diagnostics = select_b24_new_scene_records(
        records, existing,
        development_scene_count=2, final_scene_count=2,
        model_queue_valid=lambda index: index != 3)

    assert [record['reference_index'] for record in
            splits['development_validation']] == [1, 7]
    assert [record['reference_index'] for record in
            splits['final_holdout']] == [5, 9]
    assert diagnostics['fresh_before_model_queue_count'] == 8
    assert diagnostics['fresh_model_queue_rejected_count'] == 1
    assert diagnostics['development_validation_scene_count'] == 2
    assert diagnostics['final_holdout_scene_count'] == 2


def test_b24_canonical_annotation_excludes_even_unconsumed_scene():
    records = _records(3)
    canonical = annotation_scene_manifest([
        {'scene_token': 'scene-1'},
        {'scene_token': 'scene-1'},
    ])

    splits, _ = select_b24_new_scene_records(
        records, canonical,
        development_scene_count=1, final_scene_count=1,
        model_queue_valid=lambda _: True)

    selected_scenes = {
        record['scene_token']
        for records in splits.values() for record in records
    }
    assert selected_scenes == {'scene-0', 'scene-2'}


def test_b24_interleaving_rejects_duplicate_scenes():
    records = _records(4)
    records[3]['scene_token'] = records[0]['scene_token']

    try:
        split_interleaved_records(records, 2, 2)
    except ValueError as error:
        assert 'duplicate scene tokens' in str(error)
    else:
        raise AssertionError('Expected duplicate B24 scenes to fail')


def test_b24_pose_window_rejects_non_rigid_ego_transform():
    infos = [{'ego2global': np.eye(4)} for _ in range(3)]
    infos[2]['ego2global'][0, 0] = 2.0

    try:
        validate_ego_pose_window(infos, reference_index=1,
                                 required_offsets=[-1, 0, 1])
    except ValueError as error:
        assert 'rotation is not rigid' in str(error)
    else:
        raise AssertionError('Expected invalid B24 ego pose to fail')


def test_b24_selection_rejects_insufficient_fresh_scenes():
    records = _records(4)
    existing = {'splits': {'old': records[:2]}}

    try:
        select_b24_new_scene_records(
            records, existing,
            development_scene_count=2, final_scene_count=2,
            model_queue_valid=lambda _: True)
    except ValueError as error:
        assert 'Only 2 fresh scenes' in str(error)
    else:
        raise AssertionError('Expected insufficient B24 scenes to fail')


def test_b24_manifest_freezes_final_without_label_or_prediction_access():
    splits = split_interleaved_records(_records(4), 2, 2)

    manifest = build_b24_split_manifest(
        splits=splits,
        resolved_annotation_file=Path('/data/new_infos.pkl'),
        annotation_sha256='abc123',
        preflight_path=Path('/out/preflight.json'),
        existing_manifest_records=[{'path': 'old.json', 'sha256': 'old'}],
        existing_annotation_records=[{
            'path': 'old.pkl', 'sha256': 'canonical'}],
        required_offsets=range(-4, 9),
        expected_step_s=0.5,
        max_time_error_s=0.2,
        model_queue_length=5,
        model_queue_max_gap_s=1.0,
        expected_sensor_count=8,
        lidar_file_check=True)

    assert manifest['status'] == (
        'frozen_before_gt_generation_and_model_inference')
    assert manifest['final_holdout_status'] == 'frozen_not_evaluated'
    assert manifest['selection_uses_occworld_labels'] is False
    assert manifest['selection_uses_model_predictions'] is False
    assert manifest['threshold_retuning_on_final_holdout'] is False
    assert manifest['lidar_extrinsics_check'] is True
    assert manifest['ego_pose_check'] is True
    assert manifest['source_existing_annotations'][0]['path'] == 'old.pkl'
    assert set(manifest['split_scene_tokens']['development_validation']).isdisjoint(
        manifest['split_scene_tokens']['final_holdout'])
