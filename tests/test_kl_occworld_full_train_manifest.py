from tools.analysis_tools import run_kl_occworld_full_train_data
from tools.analysis_tools.prepare_kl_occworld_full_train_manifest import (
    partition_scene_records,
)
from tools.analysis_tools.run_kl_occworld_full_train_data import (
    _annotation_args,
    _scene_shards,
)


def _records(start, count):
    return [{
        'reference_index': index,
        'scene_token': f'scene-{index}',
        'sample_token': f'token-{index}',
        'timestamp': float(index),
    } for index in range(start, start + count)]


def test_full_train_partition_freezes_fresh_final_holdout():
    eligible = _records(0, 12)
    v2 = {
        'splits': {
            'train': [eligible[0], eligible[1]],
            'validation': [eligible[2]],
            'test': [eligible[3]],
        },
    }
    blind = {'splits': {'blind': [eligible[4]]}}

    result = partition_scene_records(
        eligible, v2, blind, final_scene_count=2,
        final_reference_valid=lambda index: index != 6)

    final_indices = [
        record['reference_index'] for record in result['final_records']
    ]
    train_scenes = {
        record['scene_token'] for record in result['train_scene_records']
    }
    assert final_indices == [5, 11]
    assert train_scenes == {
        'scene-0', 'scene-1', 'scene-6', 'scene-7',
        'scene-8', 'scene-9', 'scene-10'}
    assert all(record['selection_source'] ==
               'fresh_evenly_spaced_before_b15_label_generation'
               for record in result['final_records'])


def test_full_train_partition_rejects_prior_holdout_overlap():
    eligible = _records(0, 6)
    v2 = {
        'splits': {
            'train': [eligible[0]],
            'validation': [eligible[1]],
            'test': [eligible[2]],
        },
    }
    blind = {'splits': {'blind': [eligible[2]]}}

    try:
        partition_scene_records(
            eligible, v2, blind, final_scene_count=1,
            final_reference_valid=lambda _: True)
    except ValueError as error:
        assert 'V2 and blind scene overlap' in str(error)
    else:
        raise AssertionError('Expected overlap validation to fail')


def test_full_train_shards_keep_scenes_together_and_balance_references():
    records = [
        {'scene_token': 'a', 'reference_index': 1},
        {'scene_token': 'a', 'reference_index': 2},
        {'scene_token': 'a', 'reference_index': 3},
        {'scene_token': 'b', 'reference_index': 10},
        {'scene_token': 'b', 'reference_index': 11},
        {'scene_token': 'c', 'reference_index': 20},
        {'scene_token': 'd', 'reference_index': 30},
    ]

    shards = _scene_shards(records, 3)

    assert sorted(value for shard in shards for value in shard) == [
        1, 2, 3, 10, 11, 20, 30]
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert sum(any(value in shard for value in (1, 2, 3))
               for shard in shards) == 1


def test_data_generation_requires_explicit_manifest_annotation():
    assert _annotation_args({
        'annotation_file': 'data/kl_8/kl_infos_val.pkl',
    }) == ['--ann-file', 'data/kl_8/kl_infos_val.pkl']

    try:
        _annotation_args({})
    except ValueError as error:
        assert 'no annotation_file' in str(error)
    else:
        raise AssertionError('Expected missing annotation file to fail')


def test_data_generation_passes_manifest_annotation_to_generators(
        tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        run_kl_occworld_full_train_data, '_run',
        lambda command, *_: commands.append(command))
    manifest = {
        'annotation_file': 'data/kl_8/kl_infos_val.pkl',
        'splits': {'train': [{
            'reference_index': 3,
            'scene_token': 'fresh-scene',
        }]},
        'roots': {
            'dual': 'unused/target-dual',
            'dual_cross_scene': 'unused/cross-scene',
        },
        'cross_scene_support': {
            'annotation_file': 'data/kl_8/kl_infos_train.pkl',
            'dual_dir': 'unused/support-dual',
            'reference_indices': [10],
        },
    }

    run_kl_occworld_full_train_data.run_base_labels(
        manifest, tmp_path, worker_count=1, dry_run=True)
    run_kl_occworld_full_train_data.run_sequence_history(
        manifest, tmp_path, worker_count=1, dry_run=True)
    run_kl_occworld_full_train_data.run_cross_scene(
        manifest, tmp_path, dry_run=True)

    generators_with_annotations = {
        'generate_kl_occworld_temporal_labels.py',
        'audit_kl_occworld_dual_batch.py',
        'generate_kl_occworld_observation_cache.py',
        'generate_kl_occworld_sequence_batch.py',
        'generate_kl_occworld_history_batch.py',
    }
    checked = set()
    for command in commands:
        script = command[1].split('/')[-1]
        if script not in generators_with_annotations:
            continue
        position = command.index('--ann-file')
        assert command[position + 1] == 'data/kl_8/kl_infos_val.pkl'
        checked.add(script)
    assert checked == generators_with_annotations
    cross_scene = next(
        command for command in commands
        if command[1].endswith(
            'audit_kl_occworld_dual_cross_scene.py'))
    target_position = cross_scene.index('--ann-file')
    support_position = cross_scene.index('--support-ann-file')
    assert cross_scene[target_position + 1] == (
        'data/kl_8/kl_infos_val.pkl')
    assert cross_scene[support_position + 1] == (
        'data/kl_8/kl_infos_train.pkl')
