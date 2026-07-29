from pathlib import Path

from tools.analysis_tools.prepare_kl_occworld_b24_exploratory_split import (
    build_exploratory_manifest,
    freeze_exploratory_splits,
)


def _records(scene_count=10, references_per_scene=2):
    records = []
    for scene_index in range(scene_count):
        for offset in range(references_per_scene):
            index = scene_index * references_per_scene + offset
            records.append({
                'scene_token': f'scene-{scene_index}',
                'reference_index': index,
                'sample_token': f'token-{index}',
            })
    return records


def test_b24_exploratory_split_is_scene_disjoint_and_preserves_rows():
    records = _records()

    splits = freeze_exploratory_splits(records)

    assert len(splits['internal_dev']) == 4
    assert len(splits['internal_train']) == 16
    train_scenes = {record['scene_token'] for record in
                    splits['internal_train']}
    dev_scenes = {record['scene_token'] for record in splits['internal_dev']}
    assert train_scenes.isdisjoint(dev_scenes)
    assert {record['reference_index'] for records in splits.values()
            for record in records} == set(range(20))


def test_b24_exploratory_manifest_marks_non_independent_evidence():
    train = _records()
    source = {
        'splits': {
            'train': train,
            'validation': [{
                'scene_token': 'validation', 'reference_index': 100,
            }],
            'test': [{
                'scene_token': 'test', 'reference_index': 101,
            }],
            'final_holdout': [{
                'scene_token': 'final', 'reference_index': 102,
            }],
        },
    }

    manifest = build_exploratory_manifest(
        source, Path('/source.json'), 'source-sha')

    assert manifest['status'] == 'frozen_before_b24_exploratory_head_training'
    assert manifest['is_independent_generalization_evidence'] is False
    assert manifest['formal_promotion_requires_new_scene_validation'] is True
    assert manifest['source_train_scene_count'] == 10
    assert manifest['source_train_reference_count'] == 20


def test_b24_exploratory_manifest_rejects_formal_split_overlap():
    train = _records(scene_count=5, references_per_scene=1)
    source = {
        'splits': {
            'train': train,
            'validation': [{
                'scene_token': 'scene-2', 'reference_index': 100,
            }],
            'test': [{
                'scene_token': 'test', 'reference_index': 101,
            }],
            'final_holdout': [{
                'scene_token': 'final', 'reference_index': 102,
            }],
        },
    }

    try:
        build_exploratory_manifest(source, Path('/source.json'), 'source-sha')
    except ValueError as error:
        assert 'overlaps a reserved formal split' in str(error)
    else:
        raise AssertionError('Expected formal-split overlap to fail')
