import numpy as np

from tools.analysis_tools.run_kl_occworld_b17_fresh_holdout import (
    _generation_view,
    validate_online_artifacts,
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


def test_b17_online_artifacts_require_exact_causal_queue(tmp_path):
    track_root = tmp_path / 'track'
    online_root = tmp_path / 'online'
    (track_root / '000003').mkdir(parents=True)
    (online_root / '000003').mkdir(parents=True)
    np.savez_compressed(
        track_root / '000003/occworld_track_queue.npz',
        reference_index=np.int64(3),
        queue_frame_indices=np.arange(5, dtype=np.int64),
        queue_scene_tokens=np.asarray(['fresh-3'] * 5),
        track_box_offsets=np.asarray([0, 1, 1, 2, 2, 3]),
    )
    np.savez_compressed(
        online_root / '000003/occworld_online_input.npz',
        reference_index=np.int64(3),
        current_world_state_3d=np.zeros(
            (10, 120, 160), dtype=np.uint8),
        current_world_valid_3d=np.zeros(
            (10, 120, 160), dtype=np.bool_),
        history_world_state_3d=np.zeros(
            (5, 10, 120, 160), dtype=np.uint8),
        history_world_valid_3d=np.zeros(
            (5, 10, 120, 160), dtype=np.bool_),
        queue_frame_indices=np.arange(5, dtype=np.int64),
        input_contract=np.asarray(
            'five_frame_sequential_trackformer_predicted_boxes'),
    )
    manifest = {'splits': {'fresh_holdout': [_record(3)]}}
    audit = {
        'aggregate': {
            'reference_count': 1,
            'frame_count': 5,
            'instance_iou': 0.4,
            'instance_precision': 0.5,
            'instance_recall': 0.6,
            'mean_current_state_mismatch_ratio': 0.01,
            'mean_history_state_mismatch_ratio': 0.02,
        },
        'rows': [{'reference_index': 3}],
    }

    result = validate_online_artifacts(
        manifest, track_root, online_root, audit)

    assert result['track_queue_count'] == 1
    assert result['queue_frame_count'] == 5
    assert result['track_box_count'] == 3
