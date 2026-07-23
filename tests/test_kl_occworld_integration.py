import json
from pathlib import Path

import numpy as np
import torch

from projects.mmdet3d_plugin.datasets.kl_occworld_dataset import (
    discover_occworld_current_anchors,
    discover_occworld_labels,
    load_occworld_current_anchor,
    load_occworld_current_observation,
    load_occworld_history,
    load_occworld_split_references,
    load_occworld_supervision,
    load_occworld_target,
)


def _write_label(root: Path, reference_index: int, state, valid):
    label_dir = root / f'{reference_index:06d}'
    label_dir.mkdir(parents=True)
    path = label_dir / 'toy__occworld_sequence.npz'
    np.savez_compressed(
        path,
        reference_index=np.int64(reference_index),
        current_observation_state_3d=np.asarray(
            state, dtype=np.uint8)[0],
        current_observation_valid_3d=np.asarray(
            valid, dtype=np.uint8)[0],
        world_target_state_3d=np.asarray(state, dtype=np.uint8),
        world_target_valid_3d=np.asarray(valid, dtype=np.uint8))
    return path


def test_occworld_label_discovery_and_state_mapping(tmp_path):
    state = np.asarray([[[[0, 1, 2, 3]]]], dtype=np.uint8)
    valid = np.asarray([[[[0, 1, 1, 1]]]], dtype=np.uint8)
    path = _write_label(tmp_path, 7, state, valid)

    mapping = discover_occworld_labels(str(tmp_path))
    target = load_occworld_target(path, state.shape)
    _, known = load_occworld_supervision(path, state.shape)

    assert mapping == {7: path}
    assert target.dtype == torch.long
    assert target.flatten().tolist() == [255, 0, 1, 2]
    assert known.dtype == torch.bool
    assert known.flatten().tolist() == [False, True, True, True]


def test_occworld_invalid_voxels_remain_ignored(tmp_path):
    state = np.asarray([[[[1, 2]]]], dtype=np.uint8)
    valid = np.asarray([[[[1, 0]]]], dtype=np.uint8)
    path = _write_label(tmp_path, 3, state, valid)

    target, known = load_occworld_supervision(
        path, state.shape, ignore_index=99)

    assert target.flatten().tolist() == [0, 99]
    assert known.flatten().tolist() == [True, False]


def test_current_observation_loader_masks_invalid_state_without_future(
        tmp_path):
    state = np.asarray([[[[1, 2, 3]]]], dtype=np.uint8)
    valid = np.asarray([[[[1, 0, 1]]]], dtype=np.uint8)
    path = _write_label(tmp_path, 5, state, valid)
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True)
                  for key in source.files}
    arrays['world_target_state_3d'][0, 0, 0] = [3, 3, 3]
    np.savez_compressed(path, **arrays)

    current_state, current_valid = load_occworld_current_observation(
        path, expected_shape=(1, 1, 3))

    assert current_state.flatten().tolist() == [1, 0, 3]
    assert current_valid.flatten().tolist() == [True, False, True]


def test_current_anchor_discovery_and_loader_validate_reference(tmp_path):
    anchor_dir = tmp_path / '000005'
    anchor_dir.mkdir(parents=True)
    anchor_path = anchor_dir / 'occworld_current_anchor.npz'
    np.savez_compressed(
        anchor_path,
        reference_index=np.int64(5),
        current_world_state_3d=np.asarray([[[1, 3, 2]]], dtype=np.uint8),
        current_world_valid_3d=np.asarray([[[1, 0, 1]]], dtype=np.uint8))

    mapping = discover_occworld_current_anchors(str(tmp_path))
    state, valid = load_occworld_current_anchor(
        anchor_path, expected_shape=(1, 1, 3),
        expected_reference_index=5)

    assert mapping == {5: anchor_path}
    assert state.flatten().tolist() == [1, 0, 2]
    assert valid.flatten().tolist() == [True, False, True]


def test_current_anchor_loader_rejects_wrong_reference(tmp_path):
    anchor_path = tmp_path / 'occworld_current_anchor.npz'
    np.savez_compressed(
        anchor_path,
        reference_index=np.int64(6),
        current_world_state_3d=np.ones((1, 1, 1), dtype=np.uint8),
        current_world_valid_3d=np.ones((1, 1, 1), dtype=np.uint8))

    try:
        load_occworld_current_anchor(
            anchor_path, expected_shape=(1, 1, 1),
            expected_reference_index=5)
    except ValueError as error:
        assert 'expected 5' in str(error)
    else:
        raise AssertionError('Expected reference mismatch to fail')


def test_history_loader_keeps_order_and_ends_at_current(tmp_path):
    path = tmp_path / 'toy__occworld_history.npz'
    state = np.asarray([
        [[[[1, 2, 0]]]],
        [[[[2, 3, 1]]]],
    ], dtype=np.uint8).reshape(2, 1, 1, 3)
    valid = np.asarray([
        [[[[1, 0, 0]]]],
        [[[[1, 1, 1]]]],
    ], dtype=np.uint8).reshape(2, 1, 1, 3)
    np.savez_compressed(
        path,
        history_observation_state_3d=state,
        history_observation_valid_3d=valid,
        history_offsets=np.asarray([-1, 0], dtype=np.int16))

    loaded_state, loaded_valid = load_occworld_history(
        path, expected_shape=(2, 1, 1, 3))

    assert loaded_state[0].flatten().tolist() == [1, 0, 0]
    assert loaded_state[-1].flatten().tolist() == [2, 3, 1]
    assert loaded_valid[-1].all()


def test_occworld_split_manifest_returns_only_requested_references(tmp_path):
    manifest_path = tmp_path / 'split.json'
    manifest_path.write_text(json.dumps({
        'schema_version': 1,
        'splits': {
            'train': [
                {'reference_index': 3},
                {'reference_index': 7},
            ],
            'validation': [{'reference_index': 11}],
            'test': [{'reference_index': 13}],
        },
    }))

    references = load_occworld_split_references(
        str(manifest_path), 'validation')

    assert references == (11,)
