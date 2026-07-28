import sys
from types import SimpleNamespace

import numpy as np
import torch

from tools.analysis_tools.export_kl_occworld_predictions import (
    _motion_actor_diagnostic_payload,
    _override_current_anchor,
    _override_online_inputs,
    _shard_dataset_by_scene,
    parse_args,
)


def test_exporter_overrides_current_anchor_without_touching_history():
    state = np.full((10, 120, 160), 3, dtype=np.int64)
    valid = np.ones((10, 120, 160), dtype=np.bool_)
    current_state = torch.zeros((1, 10, 120, 160), dtype=torch.long)
    current_valid = torch.zeros((1, 10, 120, 160), dtype=torch.bool)
    history = torch.full((1, 5, 10, 120, 160), 7, dtype=torch.long)
    batch = {
        'current_world_state': SimpleNamespace(data=[current_state]),
        'current_world_valid': SimpleNamespace(data=[current_valid]),
        'history_world_state': SimpleNamespace(data=[history]),
    }

    _override_current_anchor(batch, state, valid)

    assert torch.equal(current_state[0], torch.from_numpy(state))
    assert torch.equal(current_valid[0], torch.from_numpy(valid))
    assert torch.all(history == 7)


def test_exporter_shards_complete_scenes_deterministically():
    dataset = SimpleNamespace(
        valid_data_indices=list(range(7)),
        data_infos=[
            {'scene_token': scene}
            for scene in ('a', 'a', 'a', 'b', 'b', 'c', 'd')
        ],
        flag=np.arange(7, dtype=np.uint8),
    )

    summary = _shard_dataset_by_scene(dataset, shard_count=2, shard_index=1)

    selected_scenes = {
        dataset.data_infos[index]['scene_token']
        for index in dataset.valid_data_indices
    }
    assert summary['all_shard_reference_counts'] == [4, 3]
    assert summary['reference_count'] == 3
    assert selected_scenes == {'b', 'c'}
    assert dataset.flag.tolist() == dataset.valid_data_indices


def test_exporter_overrides_complete_online_input_contract():
    shapes = {
        'current_world_state': (1, 10, 120, 160),
        'current_world_valid': (1, 10, 120, 160),
        'history_world_state': (1, 5, 10, 120, 160),
        'history_world_valid': (1, 5, 10, 120, 160),
    }
    batch = {
        key: SimpleNamespace(data=[torch.zeros(
            shape,
            dtype=torch.bool if 'valid' in key else torch.long)])
        for key, shape in shapes.items()
    }
    payload = {
        key: np.ones(
            shape[1:],
            dtype=np.bool_ if 'valid' in key else np.int64)
        for key, shape in shapes.items()
    }

    _override_online_inputs(batch, payload)

    for key in shapes:
        assert torch.all(batch[key].data[0])


def test_exporter_preserves_fresh_holdout_split_name(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'export_kl_occworld_predictions.py',
        '--checkpoint', 'candidate.pth',
        '--split', 'fresh_holdout',
    ])

    args = parse_args()

    assert args.split == 'fresh_holdout'


def test_exporter_exposes_opt_in_raw_prediction_diagnostic(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'export_kl_occworld_predictions.py',
        '--checkpoint', 'candidate.pth',
        '--save-raw-world-prediction',
    ])

    args = parse_args()

    assert args.save_raw_world_prediction


def test_exporter_exposes_opt_in_query_dynamic_diagnostic(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'export_kl_occworld_predictions.py',
        '--checkpoint', 'candidate.pth',
        '--save-query-dynamic-diagnostic',
    ])

    args = parse_args()

    assert args.save_query_dynamic_diagnostic


def test_exporter_exposes_opt_in_query_residual_ablation(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'export_kl_occworld_predictions.py',
        '--checkpoint', 'candidate.pth',
        '--save-query-residual-ablation',
    ])

    args = parse_args()

    assert args.save_query_residual_ablation


def test_exporter_exposes_opt_in_motion_actor_diagnostic(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'export_kl_occworld_predictions.py',
        '--checkpoint', 'candidate.pth',
        '--save-motion-actor-diagnostic',
    ])

    args = parse_args()

    assert args.save_motion_actor_diagnostic
    assert args.motion_actor_step_seconds == 0.5


def test_motion_actor_diagnostic_payload_preserves_geometry_and_time():
    future = torch.arange(48, dtype=torch.float32).reshape(1, 2, 12, 2)
    boxes = torch.arange(14, dtype=torch.float32).reshape(1, 2, 7)
    scores = torch.tensor([[0.8, 0.3]])
    valid = torch.tensor([[True, False]])

    payload = _motion_actor_diagnostic_payload(dict(
        planning_actor_future=future,
        planning_actor_boxes_3d=boxes,
        planning_actor_scores=scores,
        planning_actor_valid=valid))

    assert payload['motion_actor_future_xy'].shape == (2, 12, 2)
    assert np.array_equal(payload['motion_actor_boxes_3d'], boxes[0].numpy())
    assert np.array_equal(payload['motion_actor_valid'], valid[0].numpy())
    assert np.allclose(
        payload['motion_actor_step_times_s'],
        np.arange(1, 13, dtype=np.float32) * 0.5)
    assert payload['motion_actor_box_z_origin'].item() == 'bottom'
