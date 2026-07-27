import sys
from types import SimpleNamespace

import numpy as np
import torch

from tools.analysis_tools.export_kl_occworld_predictions import (
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
