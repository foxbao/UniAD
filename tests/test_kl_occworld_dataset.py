from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.data_converter.kl_occworld_dataset import (
    IGNORE_INDEX,
    KLOccWorldSequenceDataset,
)
from tools.tutorials.occworld_toy.step01_current_to_future_baseline import (
    PersistenceResidualOccWorld,
    TinyOccWorld,
    change_aware_world_cross_entropy,
    masked_world_cross_entropy,
)
from tools.analysis_tools.evaluate_kl_occworld_baselines import (
    _constant_current_prediction,
    _summarize_confusion,
)
from tools.tutorials.occworld_toy.step02_temporal_fusion_baseline import (
    TemporalFusionOccWorld,
)
from tools.tutorials.occworld_toy.step03_change_gate_baseline import (
    GatedTemporalOccWorld,
    gated_occworld_loss,
)
from tools.tutorials.occworld_toy.step04_decomposed_change_baseline import (
    DecomposedTemporalOccWorld,
    decomposed_occworld_loss,
)


def _write_label(root: Path):
    out_dir = root / '000007'
    out_dir.mkdir(parents=True)
    current = np.zeros((2, 3, 4), dtype=np.uint8)
    current[0, 0, :4] = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    world = np.stack([current, current, current], axis=0)
    world[1, 1, 0, :4] = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    world[2, 1, 1, :4] = np.asarray([3, 2, 1, 0], dtype=np.uint8)
    path = out_dir / 'toy__occworld_sequence.npz'
    np.savez_compressed(
        path,
        current_observation_state_3d=current,
        current_observation_valid_3d=(current != 0).astype(np.uint8),
        # The dataset must not expose this future-only label-generation input.
        direct_observation_state_3d=world,
        world_target_state_3d=world,
        world_target_valid_3d=(world != 0).astype(np.uint8),
        world_target_state_bev=world.max(axis=1),
        traversability_state_bev=np.ones((3, 3, 4), dtype=np.uint8),
        traversability_valid_bev=np.ones((3, 3, 4), dtype=np.uint8),
        navigability_state_bev=np.ones((3, 3, 4), dtype=np.uint8),
        navigability_valid_bev=np.ones((3, 3, 4), dtype=np.uint8),
        target_offsets=np.asarray([0, 1, 2], dtype=np.int16),
        target_times_s=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        nominal_target_times_s=np.asarray(
            [0.0, 0.5, 1.0], dtype=np.float32),
        reference_index=np.int64(7),
        pc_range=np.asarray([-2, -3, -1, 2, 3, 1], dtype=np.float32),
        occ_size=np.asarray([4, 3, 2], dtype=np.int16),
    )
    return path


def _write_history(root: Path):
    out_dir = root / '000007'
    out_dir.mkdir(parents=True)
    history = np.zeros((3, 2, 3, 4), dtype=np.uint8)
    history[0, 0, 0, :4] = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    history[1, 0, 0, :4] = np.asarray([1, 2, 3, 0], dtype=np.uint8)
    history[2, 0, 0, :4] = np.asarray([3, 2, 1, 0], dtype=np.uint8)
    path = out_dir / 'toy__occworld_history.npz'
    np.savez_compressed(
        path,
        history_observation_state_3d=history,
        history_observation_valid_3d=(history != 0).astype(np.uint8),
        history_offsets=np.asarray([-2, -1, 0], dtype=np.int16),
        history_times_s=np.asarray([-1.0, -0.5, 0.0], dtype=np.float32),
        nominal_history_times_s=np.asarray(
            [-1.0, -0.5, 0.0], dtype=np.float32),
        reference_index=np.int64(7),
    )
    return path


def test_occworld_dataset_excludes_current_target_and_future_input(tmp_path):
    _write_label(tmp_path)
    dataset = KLOccWorldSequenceDataset(tmp_path)

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample['input_state_3d'].shape == (2, 3, 4)
    assert sample['input_one_hot_3d'].shape == (4, 2, 3, 4)
    assert sample['target_state_3d'].shape == (2, 2, 3, 4)
    assert sample['target_class_3d'].shape == (2, 2, 3, 4)
    torch.testing.assert_close(
        sample['target_times_s'], torch.tensor([0.5, 1.0]))
    assert 'direct_observation_state_3d' not in sample
    assert torch.all(sample['input_one_hot_3d'].sum(dim=0) == 1)


def test_occworld_dataset_maps_unknown_to_ignore_index(tmp_path):
    _write_label(tmp_path)
    target = KLOccWorldSequenceDataset(tmp_path)[0]['target_class_3d']

    assert target[0, 1, 0].tolist() == [IGNORE_INDEX, 0, 1, 2]
    assert target[1, 1, 1].tolist() == [2, 1, 0, IGNORE_INDEX]


def test_occworld_dataset_ignores_nonzero_state_marked_invalid(tmp_path):
    path = _write_label(tmp_path)
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True)
                  for key in source.files}
    arrays['world_target_state_3d'][1, 0, 0, 0] = 3
    arrays['world_target_valid_3d'][1, 0, 0, 0] = 0
    np.savez_compressed(path, **arrays)

    target = KLOccWorldSequenceDataset(tmp_path)[0]['target_class_3d']

    assert target[0, 0, 0, 0].item() == IGNORE_INDEX


def test_occworld_dataset_default_collate_builds_a_batch(tmp_path):
    _write_label(tmp_path)
    dataset = KLOccWorldSequenceDataset(tmp_path)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))

    assert batch['input_one_hot_3d'].shape == (1, 4, 2, 3, 4)
    assert batch['target_class_3d'].shape == (1, 2, 2, 3, 4)
    assert batch['reference_index'].tolist() == [7]
    assert batch['label_path'][0].endswith('toy__occworld_sequence.npz')


def test_occworld_dataset_uses_aligned_history_current(tmp_path):
    sequence_root = tmp_path / 'sequences'
    history_root = tmp_path / 'history'
    _write_label(sequence_root)
    _write_history(history_root)
    dataset = KLOccWorldSequenceDataset(
        sequence_root, history_root=history_root)

    sample = dataset[0]

    assert sample['history_state_3d'].shape == (3, 2, 3, 4)
    assert sample['history_one_hot_3d'].shape == (3, 4, 2, 3, 4)
    torch.testing.assert_close(
        sample['input_state_3d'], sample['history_state_3d'][-1])
    assert not torch.equal(
        sample['input_state_3d'], sample['sequence_current_state_3d'])
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    assert batch['history_one_hot_3d'].shape == (1, 3, 4, 2, 3, 4)


def test_tiny_occworld_emits_one_logit_volume_per_future_horizon():
    model = TinyOccWorld(
        in_channels=4, hidden_channels=2, horizon_count=2, class_count=3)
    logits = model(torch.zeros((1, 4, 2, 3, 4)))

    assert logits.shape == (1, 2, 3, 2, 3, 4)


def test_persistence_residual_starts_from_constant_current_prediction():
    states = torch.tensor([[[[0, 1, 2, 3]]]])
    inputs = torch.nn.functional.one_hot(
        states, num_classes=4).permute(0, 4, 1, 2, 3).float()
    model = PersistenceResidualOccWorld(
        in_channels=4, hidden_channels=2, horizon_count=2,
        class_count=3, persistence_logit_scale=2.0)

    prediction = model(inputs).argmax(dim=2)

    assert prediction.shape == (1, 2, 1, 1, 4)
    assert prediction[0, 0, 0, 0].tolist() == [0, 0, 1, 2]
    torch.testing.assert_close(prediction[:, 0], prediction[:, 1])


def test_temporal_fusion_starts_from_latest_history_persistence():
    states = torch.tensor([[
        [[[1, 2, 3, 0]]],
        [[[0, 1, 2, 3]]],
        [[[3, 2, 1, 0]]],
    ]])
    history = torch.nn.functional.one_hot(
        states, num_classes=4).permute(0, 1, 5, 2, 3, 4).float()
    model = TemporalFusionOccWorld(
        history_count=3, state_channels=4, hidden_channels=2,
        horizon_count=2, class_count=3,
        persistence_logit_scale=1.0,
        include_frame_differences=True)

    prediction = model(history).argmax(dim=2)

    assert model.temporal_encoder[0].in_channels == 20
    assert prediction.shape == (1, 2, 1, 1, 4)
    assert prediction[0, 0, 0, 0].tolist() == [2, 1, 0, 0]
    torch.testing.assert_close(prediction[:, 0], prediction[:, 1])


def test_change_gate_starts_closed_and_supports_separate_losses():
    states = torch.tensor([[
        [[[1, 2, 3, 0]]],
        [[[0, 1, 2, 3]]],
        [[[3, 2, 1, 0]]],
    ]])
    history = torch.nn.functional.one_hot(
        states, num_classes=4).permute(0, 1, 5, 2, 3, 4).float()
    model = GatedTemporalOccWorld(
        history_count=3, state_channels=4, hidden_channels=2,
        horizon_count=2, class_count=3, change_prior=0.05)

    prediction = model.predict(history, threshold=0.5)

    assert prediction['prediction_class'][0, 0, 0, 0].tolist() == [2, 1, 0, 0]
    torch.testing.assert_close(
        prediction['change_probability'],
        torch.full_like(prediction['change_probability'], 0.05))

    target = prediction['persistence_class'].clone()
    target[:, :, 0, 0, 0] = 0
    target[:, :, 0, 0, 3] = IGNORE_INDEX
    losses = gated_occworld_loss(
        model(history), target, states[:, -1],
        change_positive_weight=10.0)
    losses['loss'].backward()

    assert losses['changed_voxel_count'].item() == 2
    assert torch.isfinite(losses['loss'])
    assert model.change_head.bias.grad is not None
    assert model.changed_class_head.bias.grad is not None


def test_decomposed_heads_use_unknown_and_visible_supervision_separately():
    states = torch.tensor([[
        [[[1, 2, 3, 0]]],
        [[[0, 1, 2, 3]]],
        [[[0, 1, 2, 3]]],
    ]])
    history = torch.nn.functional.one_hot(
        states, num_classes=4).permute(0, 1, 5, 2, 3, 4).float()
    model = DecomposedTemporalOccWorld(
        history_count=3, state_channels=4, hidden_channels=2,
        horizon_count=2, class_count=3,
        reveal_prior=0.05, transition_prior=0.01)

    prediction = model.predict(history)
    probabilistic_prediction = model.predict(
        history, decision_mode='probabilistic')

    assert prediction['prediction_class'][0, 0, 0, 0].tolist() == [0, 0, 1, 2]
    torch.testing.assert_close(
        probabilistic_prediction['prediction_class'],
        prediction['prediction_class'])
    assert probabilistic_prediction['prediction_scores'].shape == (
        1, 2, 3, 1, 1, 4)
    torch.testing.assert_close(
        prediction['reveal_probability'],
        torch.full_like(prediction['reveal_probability'], 0.05))
    torch.testing.assert_close(
        prediction['transition_probability'],
        torch.full_like(prediction['transition_probability'], 0.01))

    target = torch.tensor([[[[[2, 1, 1, 2]]],
                            [[[IGNORE_INDEX, 0, 2, 2]]]]])
    losses = decomposed_occworld_loss(
        model(history), target, states[:, -1],
        reveal_positive_weight=2.0,
        transition_positive_weight=3.0)
    losses['loss'].backward()

    assert losses['reveal_voxel_count'].item() == 2
    assert losses['revealed_voxel_count'].item() == 1
    assert losses['transition_voxel_count'].item() == 6
    assert losses['changed_visible_voxel_count'].item() == 2
    assert torch.isfinite(losses['loss'])
    assert model.reveal_head.bias.grad is not None
    assert model.reveal_class_head.bias.grad is not None
    assert model.transition_head.bias.grad is not None
    assert model.transition_class_head.bias.grad is not None


def test_masked_world_cross_entropy_ignores_unknown_voxels():
    logits = torch.zeros((1, 1, 3, 1, 1, 2))
    target = torch.tensor([[[[[0, IGNORE_INDEX]]]]])
    baseline = masked_world_cross_entropy(logits, target)
    logits[..., 1] = 1000.0

    torch.testing.assert_close(
        masked_world_cross_entropy(logits, target), baseline)


def test_change_aware_loss_upweights_errors_against_persistence():
    logits = torch.zeros((1, 1, 3, 1, 1, 2))
    logits[:, :, 0] = 2.0
    target = torch.tensor([[[[[0, 1]]]]])
    current_state = torch.tensor([[[[1, 1]]]])

    regular = masked_world_cross_entropy(logits, target)
    change_aware = change_aware_world_cross_entropy(
        logits, target, current_state, change_voxel_weight=10.0)

    assert change_aware > regular


def test_constant_current_repeats_current_classes_across_horizons():
    batch = {
        'input_state_3d': torch.tensor([[[[0, 1, 2, 3]]]]),
        'target_class_3d': torch.zeros((1, 2, 1, 1, 4), dtype=torch.long),
    }

    prediction = _constant_current_prediction(batch)

    assert prediction.shape == (1, 2, 1, 1, 4)
    assert prediction[0, 0, 0, 0].tolist() == [0, 0, 1, 2]
    torch.testing.assert_close(prediction[:, 0], prediction[:, 1])


def test_confusion_summary_reports_accuracy_and_iou():
    confusion = np.asarray([
        [2, 0, 0],
        [0, 1, 1],
        [0, 0, 2],
    ], dtype=np.int64)

    summary = _summarize_confusion(confusion)

    assert summary['voxel_count'] == 6
    assert summary['accuracy'] == 5 / 6
    assert summary['iou']['free'] == 1.0
    assert summary['iou']['static_occupied'] == 0.5
    assert summary['iou']['instance_occupied'] == 2 / 3
