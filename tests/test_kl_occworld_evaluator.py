from pathlib import Path

import numpy as np
import torch

from tools.analysis_tools.build_kl_occworld_scene_split import (
    build_scene_split,
)
from tools.analysis_tools.audit_kl_occworld_predicted_flow import (
    _accumulate,
    _empty_accumulators,
    _summarize,
)
from tools.analysis_tools.audit_kl_occworld_raw_flow_disagreement import (
    _empty_disagreement,
    accumulate_disagreement,
    apply_flow_events_to_raw,
    restore_raw_visible_changes,
)
from tools.analysis_tools.evaluate_kl_occworld import evaluate_split
from tools.analysis_tools.select_kl_occworld_checkpoint import (
    select_checkpoint,
)


def _write_evaluation_label(root: Path) -> Path:
    output_dir = root / '000007'
    output_dir.mkdir(parents=True)
    current_state = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    current_valid = np.asarray([[[1, 1, 0]]], dtype=np.uint8)
    target_state = np.asarray([
        [[[1, 2, 3]]],
        [[[2, 2, 3]]],
    ], dtype=np.uint8)
    target_valid = np.asarray([
        [[[1, 1, 0]]],
        [[[1, 1, 1]]],
    ], dtype=np.uint8)
    path = output_dir / 'toy__occworld_sequence.npz'
    np.savez_compressed(
        path,
        reference_index=np.int64(7),
        current_observation_state_3d=current_state,
        current_observation_valid_3d=current_valid,
        world_target_state_3d=target_state,
        world_target_valid_3d=target_valid,
        target_times_s=np.asarray([0.0, 0.5], dtype=np.float32),
    )
    return path


def _manifest():
    return {
        'schema_version': 1,
        'name': 'toy_scene_split',
        'splits': {
            'train': [],
            'validation': [],
            'test': [{
                'reference_index': 7,
                'scene_token': 'scene-c',
            }],
        },
    }


def test_scene_split_keeps_all_references_from_one_scene_together():
    infos = [
        {'sample_idx': 0, 'scene_token': 'scene-a', 'token': 'a0'},
        {'sample_idx': 1, 'scene_token': 'scene-a', 'token': 'a1'},
        {'sample_idx': 2, 'scene_token': 'scene-b', 'token': 'b0'},
        {'sample_idx': 3, 'scene_token': 'scene-c', 'token': 'c0'},
    ]

    result = build_scene_split(
        infos, eligible_references=[0, 1, 2, 3],
        train_scene_count=1, validation_scene_count=1)

    assert [record['reference_index']
            for record in result['splits']['train']] == [0, 1]
    assert [record['reference_index']
            for record in result['splits']['validation']] == [2]
    assert [record['reference_index']
            for record in result['splits']['test']] == [3]
    scene_sets = [
        set(result['split_scene_tokens'][name])
        for name in ('train', 'validation', 'test')
    ]
    assert scene_sets[0].isdisjoint(scene_sets[1])
    assert scene_sets[0].isdisjoint(scene_sets[2])
    assert scene_sets[1].isdisjoint(scene_sets[2])


def test_predicted_flow_audit_reports_perfect_aligned_direction():
    target = np.asarray([[[
        [1.0, 255.0],
        [0.0, -1.0],
    ], [
        [0.0, 255.0],
        [2.0, 0.0],
    ]]], dtype=np.float32)
    target = torch.from_numpy(target)
    prediction = torch.where(
        target == 255.0, torch.zeros_like(target), target)
    acc = _empty_accumulators(horizon_count=1)

    _accumulate(acc, prediction, target)
    summary = _summarize(acc)[0]

    assert summary['valid_source_endpoints'] == 3
    assert summary['mean_endpoint_error_cells'] == 0.0
    assert summary['mean_direction_cosine'] == 1.0
    assert summary['nonzero_target_sign_accuracy_dy_dx'] == [1.0, 1.0]


def test_flow_events_keep_raw_base_and_use_causal_change_gate():
    raw = torch.tensor([
        [[[2, 0, 1]]],
        [[[2, 1, 1]]],
    ])
    observation_class = torch.tensor([[[2, 0, 1]]])
    observation_known = torch.ones_like(
        observation_class, dtype=torch.bool)
    warped = torch.tensor([[[[0.0, 1.0, 0.0]]]])
    change = torch.tensor([[[[0.2, 0.8, 0.9]]]])

    fused, events = apply_flow_events_to_raw(
        raw, observation_class, observation_known, warped,
        event_threshold=0.5,
        change_probability=change,
        change_threshold=0.5)

    assert events['departure'][0, 0, 0].tolist() == [False, False, False]
    assert events['arrival'][0, 0, 0].tolist() == [False, True, False]
    assert fused[1, 0, 0].tolist() == [2, 2, 1]

    norm_gated, _ = apply_flow_events_to_raw(
        raw, observation_class, observation_known, warped,
        event_threshold=0.5,
        change_probability=change,
        change_threshold=0.5,
        flow_norm=torch.tensor([[[0.1, 0.8, 0.8]]]),
        flow_norm_threshold=1.0)
    assert norm_gated[1, 0, 0].tolist() == [2, 1, 1]


def test_disagreement_audit_separates_raw_and_alternative_wins():
    raw = torch.tensor([[[[0, 1, 2]]]])
    alternative = torch.tensor([[[[1, 1, 0]]]])
    target = torch.tensor([[[[0, 1, 0]]]])
    selection = torch.ones_like(target, dtype=torch.bool)
    counts = _empty_disagreement(horizon_count=1)

    accumulate_disagreement(
        counts, raw, alternative, target, selection)

    assert counts['disagreement_voxels'].tolist() == [2]
    assert counts['raw_only_correct'].tolist() == [1]
    assert counts['alternative_only_correct'].tolist() == [1]
    assert counts['both_wrong'].tolist() == [0]


def test_restore_raw_visible_changes_keeps_unconfident_physical_result():
    raw = torch.tensor([
        [[[0, 1, 2]]],
        [[[1, 2, 2]]],
    ])
    persistence = torch.tensor([
        [[[0, 1, 2]]],
        [[[0, 1, 2]]],
    ])
    physical = persistence.clone()
    known = torch.ones((1, 1, 3), dtype=torch.bool)
    change = torch.tensor([[[[0.8, 0.2, 0.9]]]])

    hybrid, restore = restore_raw_visible_changes(
        physical, raw, persistence, known, change,
        change_threshold=0.5)

    assert restore[0, 0, 0].tolist() == [True, False, False]
    assert hybrid[1, 0, 0].tolist() == [1, 1, 2]

    protected, protected_restore = restore_raw_visible_changes(
        physical, raw, persistence, known, change,
        change_threshold=0.5,
        protected_event_mask=torch.tensor([[[[True, False, False]]]]))
    assert not protected_restore[0, 0, 0, 0]
    assert protected[1, 0, 0].tolist() == [0, 1, 2]


def test_persistence_evaluator_reports_horizon_and_subset_metrics(tmp_path):
    label_path = _write_evaluation_label(tmp_path)

    result = evaluate_split(
        manifest=_manifest(), split='test',
        sequence_mapping={7: label_path},
        flow_fusion_threshold=0.7)

    assert result['prediction_source'] == 'constant_current_persistence'
    assert result['flow_fusion_threshold'] == 0.7
    assert result['mean_target_times_s'] == [0.0, 0.5]
    assert result['semantic']['overall']['voxel_count'] == 5
    assert result['semantic']['overall']['accuracy'] == 0.6
    assert result['subset_counts_by_horizon'] == {
        'target_known': [2, 3],
        'current_visible': [2, 2],
        'reveal_completion': [0, 1],
        'state_change': [0, 2],
        'visible_transition': [0, 1],
        'instance_related_visible_transition': [0, 0],
    }
    assert result['visibility']['overall']['true_positive'] == 4
    assert result['visibility']['overall']['false_negative'] == 1
    assert result['visibility']['overall']['true_negative'] == 1
    assert result['visibility']['overall']['recall'] == 0.8


def test_evaluator_accepts_exported_semantic_and_visibility_predictions(
        tmp_path):
    label_path = _write_evaluation_label(tmp_path / 'labels')
    prediction_dir = tmp_path / 'predictions' / '000007'
    prediction_dir.mkdir(parents=True)
    prediction_path = prediction_dir / 'toy__occworld_prediction.npz'
    prediction = np.asarray([
        [[[0, 1, 2]]],
        [[[1, 1, 2]]],
    ], dtype=np.uint8)
    visibility = np.asarray([
        [[[1.0, 1.0, 0.0]]],
        [[[1.0, 1.0, 1.0]]],
    ], dtype=np.float32)
    change_probability = np.asarray([
        [[[0.9, 0.1, 0.9]]],
    ], dtype=np.float32)
    changed_class_prediction = np.asarray([
        [[[1, 0, 2]]],
    ], dtype=np.uint8)
    np.savez_compressed(
        prediction_path,
        reference_index=np.int64(7),
        world_pred_class_3d=prediction,
        world_valid_probability_3d=visibility,
        future_change_probability_3d=change_probability,
        future_changed_class_pred_3d=changed_class_prediction,
    )

    result = evaluate_split(
        manifest=_manifest(), split='test',
        sequence_mapping={7: label_path},
        prediction_mapping={7: prediction_path})

    assert result['prediction_source'] == 'exported_model_prediction'
    assert result['semantic']['overall']['accuracy'] == 1.0
    assert result['visibility']['overall']['f1'] == 1.0
    assert result['future_change_gate']['overall']['true_positive'] == 1
    assert result['future_change_gate']['overall']['true_negative'] == 1
    assert result['future_change_gate']['overall']['f1'] == 1.0
    assert result[
        'future_changed_class_on_transition']['overall']['accuracy'] == 1.0

    hard_result = evaluate_split(
        manifest=_manifest(), split='test',
        sequence_mapping={7: label_path},
        prediction_mapping={7: prediction_path},
        change_gate_threshold=0.5,
        apply_change_gate=True)
    assert hard_result['hard_change_gate_applied']
    assert hard_result['semantic']['overall']['accuracy'] == 1.0


def test_evaluator_separates_completion_only_and_flow_only(tmp_path):
    label_path = _write_evaluation_label(tmp_path / 'labels')
    prediction_path = tmp_path / 'toy__occworld_prediction.npz'
    prediction = np.asarray([
        [[[0, 1, 2]]],
        [[[1, 1, 2]]],
    ], dtype=np.uint8)
    warped_instance = np.asarray([
        [[[0.0, 0.0, 1.0]]],
    ], dtype=np.float32)
    np.savez_compressed(
        prediction_path,
        reference_index=np.int64(7),
        world_pred_class_3d=prediction,
        warped_instance_probability_3d=warped_instance)

    completion = evaluate_split(
        manifest=_manifest(), split='test',
        sequence_mapping={7: label_path},
        prediction_mapping={7: prediction_path},
        apply_completion_only=True)
    flow = evaluate_split(
        manifest=_manifest(), split='test',
        sequence_mapping={7: label_path},
        prediction_mapping={7: prediction_path},
        apply_flow_only=True,
        flow_fusion_threshold=0.7)
    local = evaluate_split(
        manifest=_manifest(), split='test',
        sequence_mapping={7: label_path},
        prediction_mapping={7: prediction_path},
        apply_local_flow_overlay=True,
        flow_fusion_threshold=0.7)

    assert completion['completion_only_applied']
    assert not completion['flow_only_applied']
    assert completion['semantic']['overall']['accuracy'] == 0.8
    assert completion['semantic'][
        'reveal_completion_subset']['overall']['accuracy'] == 1.0
    assert flow['flow_only_applied']
    assert not flow['physical_flow_fusion_applied']
    assert flow['semantic']['overall']['accuracy'] == 0.8
    assert flow['semantic'][
        'reveal_completion_subset']['overall']['accuracy'] == 1.0
    assert local['local_flow_overlay_applied']
    assert not local['physical_flow_fusion_applied']
    assert local['semantic']['overall']['accuracy'] == 1.0


def test_evaluator_applies_exported_physical_confidence(tmp_path):
    label_path = _write_evaluation_label(tmp_path / 'labels')
    prediction_path = tmp_path / 'toy__occworld_prediction.npz'
    prediction = np.asarray([
        [[[0, 1, 2]]],
        [[[1, 1, 2]]],
    ], dtype=np.uint8)
    np.savez_compressed(
        prediction_path,
        reference_index=np.int64(7),
        world_pred_class_3d=prediction,
        warped_instance_probability_3d=np.asarray([
            [[[0.0, 0.0, 1.0]]],
        ], dtype=np.float32),
        physical_confidence_probability_3d=np.zeros(
            (1, 1, 1, 3), dtype=np.float32))

    result = evaluate_split(
        manifest=_manifest(), split='test',
        sequence_mapping={7: label_path},
        prediction_mapping={7: prediction_path},
        apply_physical_confidence=True,
        physical_confidence_threshold=0.5,
        physical_confidence_flow_threshold=0.5)

    assert result['physical_confidence_applied']
    assert result['physical_confidence_available']
    assert result['semantic']['overall']['accuracy'] == 1.0


def test_evaluator_rejects_mixed_ablation_modes(tmp_path):
    label_path = _write_evaluation_label(tmp_path)

    try:
        evaluate_split(
            manifest=_manifest(), split='test',
            sequence_mapping={7: label_path},
            prediction_mapping={7: tmp_path / 'unused.npz'},
            apply_completion_only=True,
            apply_flow_only=True)
    except ValueError as error:
        assert 'mutually exclusive' in str(error)
    else:
        raise AssertionError('Expected mixed fusion modes to fail')


def test_validation_selector_uses_future_miou_then_visibility_f1(tmp_path):
    label_path = _write_evaluation_label(tmp_path / 'labels')
    manifest = _manifest()
    manifest['splits']['validation'] = manifest['splits']['test']
    manifest['splits']['test'] = []
    prediction_root = tmp_path / 'predictions'
    imperfect = np.zeros((2, 1, 1, 3), dtype=np.uint8)
    perfect = np.asarray([
        [[[0, 1, 2]]],
        [[[1, 1, 2]]],
    ], dtype=np.uint8)
    perfect_visibility = np.asarray([
        [[[0.8, 0.8, 0.2]]],
        [[[0.8, 0.8, 0.8]]],
    ], dtype=np.float32)
    for epoch, prediction in ((1, imperfect), (2, perfect)):
        output_dir = prediction_root / f'epoch_{epoch:03d}' / '000007'
        output_dir.mkdir(parents=True)
        np.savez_compressed(
            output_dir / 'toy__occworld_prediction.npz',
            reference_index=np.int64(7),
            world_pred_class_3d=prediction,
            world_valid_probability_3d=perfect_visibility)

    result = select_checkpoint(
        manifest=manifest,
        sequence_mapping={7: label_path},
        prediction_root=prediction_root,
        thresholds=[0.3, 0.5, 0.7])

    assert result['selected_epoch'] == 2
    assert result['selected_visibility_threshold'] == 0.5
