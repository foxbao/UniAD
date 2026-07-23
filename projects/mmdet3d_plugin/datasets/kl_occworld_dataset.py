"""KL temporal dataset adapter for precomputed OccWorld supervision."""

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS

from .kl_dataset import KlTrackDataset


def discover_occworld_labels(label_root: str) -> Dict[int, Path]:
    """Map reference indices to generated OccWorld sequence files."""
    root = Path(label_root)
    mapping = {}
    for path in sorted(root.glob('*/*__occworld_sequence.npz')):
        with np.load(path, allow_pickle=False) as label:
            if 'reference_index' not in label.files:
                raise ValueError(f'{path} has no reference_index')
            reference_index = int(label['reference_index'])
        if reference_index in mapping:
            raise ValueError(
                f'Duplicate OccWorld label for reference {reference_index}')
        mapping[reference_index] = path
    if not mapping:
        raise FileNotFoundError(
            f'No OccWorld sequence labels found below {root}')
    return mapping


def discover_occworld_histories(history_root: str) -> Dict[int, Path]:
    """Map reference indices to causal OccWorld history files."""
    root = Path(history_root)
    mapping = {}
    for path in sorted(root.glob('*/*__occworld_history.npz')):
        with np.load(path, allow_pickle=False) as history:
            if 'reference_index' not in history.files:
                raise ValueError(f'{path} has no reference_index')
            reference_index = int(history['reference_index'])
        if reference_index in mapping:
            raise ValueError(
                f'Duplicate OccWorld history for {reference_index}')
        mapping[reference_index] = path
    if not mapping:
        raise FileNotFoundError(
            f'No OccWorld history files found below {root}')
    return mapping


def discover_occworld_current_anchors(anchor_root: str) -> Dict[int, Path]:
    """Map reference indices to precomputed online current anchors."""
    root = Path(anchor_root)
    mapping = {}
    for path in sorted(root.glob('*/occworld_current_anchor.npz')):
        with np.load(path, allow_pickle=False) as anchor:
            if 'reference_index' not in anchor.files:
                raise ValueError(f'{path} has no reference_index')
            reference_index = int(anchor['reference_index'])
        if reference_index in mapping:
            raise ValueError(
                f'Duplicate OccWorld current anchor for {reference_index}')
        mapping[reference_index] = path
    if not mapping:
        raise FileNotFoundError(
            f'No OccWorld current anchors found below {root}')
    return mapping


def load_occworld_split_references(
        manifest_path: str,
        split: str) -> Tuple[int, ...]:
    """Load one frozen reference split from an OccWorld manifest."""
    path = Path(manifest_path)
    with path.open() as source:
        manifest = json.load(source)
    if manifest.get('schema_version') != 1:
        raise ValueError(f'Unsupported OccWorld manifest version in {path}')
    splits = manifest.get('splits')
    if not isinstance(splits, dict) or split not in splits:
        raise ValueError(f'OccWorld manifest {path} has no {split} split')
    references = tuple(
        int(record['reference_index']) for record in splits[split])
    if not references or len(references) != len(set(references)):
        raise ValueError(f'Invalid {split} references in {path}')
    return references


def load_occworld_supervision(
        label_path: Path,
        expected_shape: Sequence[int],
        ignore_index: int = 255):
    """Return semantic class targets and known/unknown supervision."""
    with np.load(label_path, allow_pickle=False) as label:
        state = np.asarray(label['world_target_state_3d'], dtype=np.int64)
        valid = np.asarray(label['world_target_valid_3d'], dtype=np.bool_)
    expected_shape = tuple(int(value) for value in expected_shape)
    if state.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError(
            f'Unexpected OccWorld shape in {label_path}: '
            f'{state.shape}, {valid.shape}, expected {expected_shape}')
    if np.any((state < 0) | (state > 3)):
        raise ValueError(f'Invalid world state value in {label_path}')
    target = np.full(state.shape, int(ignore_index), dtype=np.int64)
    known = valid & (state != 0)
    target[known] = state[known] - 1
    return torch.from_numpy(target), torch.from_numpy(known)


def load_occworld_current_observation(
        label_path: Path,
        expected_shape: Sequence[int]):
    """Load only the causal current LiDAR observation, never world targets."""
    with np.load(label_path, allow_pickle=False) as label:
        state = np.asarray(
            label['current_observation_state_3d'], dtype=np.int64)
        valid = np.asarray(
            label['current_observation_valid_3d'], dtype=np.bool_)
    expected_shape = tuple(int(value) for value in expected_shape)
    if state.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError(
            f'Unexpected current observation shape in {label_path}: '
            f'{state.shape}, {valid.shape}, expected {expected_shape}')
    if np.any((state < 0) | (state > 3)):
        raise ValueError(
            f'Invalid current observation state in {label_path}')
    known = valid & (state != 0)
    state = np.array(state, copy=True)
    state[~known] = 0
    return torch.from_numpy(state), torch.from_numpy(known)


def load_occworld_current_anchor(
        anchor_path: Path,
        expected_shape: Sequence[int],
        expected_reference_index: int = None):
    """Load a causal online-anchor override without reading world targets."""
    with np.load(anchor_path, allow_pickle=False) as anchor:
        reference_index = int(anchor['reference_index'])
        state = np.asarray(
            anchor['current_world_state_3d'], dtype=np.int64)
        valid = np.asarray(
            anchor['current_world_valid_3d'], dtype=np.bool_)
    if (expected_reference_index is not None and
            reference_index != int(expected_reference_index)):
        raise ValueError(
            f'Current anchor {anchor_path} has reference {reference_index}, '
            f'expected {expected_reference_index}')
    expected_shape = tuple(int(value) for value in expected_shape)
    if state.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError(
            f'Unexpected current anchor shape in {anchor_path}: '
            f'{state.shape}, {valid.shape}, expected {expected_shape}')
    if np.any((state < 0) | (state > 3)):
        raise ValueError(f'Invalid current anchor state in {anchor_path}')
    known = valid & (state != 0)
    state = np.array(state, copy=True)
    state[~known] = 0
    return torch.from_numpy(state), torch.from_numpy(known)


def load_occworld_direct_current_observation(
        label_path: Path,
        expected_shape: Sequence[int]):
    """Load direct observation at t=0 only for history alignment checks."""
    with np.load(label_path, allow_pickle=False) as label:
        state = np.asarray(
            label['direct_observation_state_3d'][0], dtype=np.int64)
        valid = np.asarray(
            label['direct_observation_valid_3d'][0], dtype=np.bool_)
    expected_shape = tuple(int(value) for value in expected_shape)
    if state.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError(
            f'Unexpected direct-current shape in {label_path}: '
            f'{state.shape}, {valid.shape}, expected {expected_shape}')
    known = valid & (state != 0)
    state = np.array(state, copy=True)
    state[~known] = 0
    return torch.from_numpy(state), torch.from_numpy(known)


def load_occworld_history(history_path: Path,
                          expected_shape: Sequence[int]):
    """Load an aligned causal history ending at the reference frame."""
    with np.load(history_path, allow_pickle=False) as history:
        state = np.asarray(
            history['history_observation_state_3d'], dtype=np.int64)
        valid = np.asarray(
            history['history_observation_valid_3d'], dtype=np.bool_)
        offsets = np.asarray(history['history_offsets'], dtype=np.int64)
    expected_shape = tuple(int(value) for value in expected_shape)
    if state.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError(
            f'Unexpected history shape in {history_path}: '
            f'{state.shape}, {valid.shape}, expected {expected_shape}')
    if offsets.shape != (expected_shape[0],) or offsets[-1] != 0:
        raise ValueError(
            f'OccWorld history must end at offset 0 in {history_path}')
    if np.any(np.diff(offsets) <= 0):
        raise ValueError(
            f'OccWorld history offsets must increase in {history_path}')
    if np.any((state < 0) | (state > 3)):
        raise ValueError(f'Invalid history state in {history_path}')
    known = valid & (state != 0)
    state = np.array(state, copy=True)
    state[~known] = 0
    return torch.from_numpy(state), torch.from_numpy(known)


def load_occworld_target(
        label_path: Path,
        expected_shape: Sequence[int],
        ignore_index: int = 255) -> torch.Tensor:
    """Backward-compatible semantic target loader."""
    target, _ = load_occworld_supervision(
        label_path, expected_shape, ignore_index)
    return target


@DATASETS.register_module()
class KlOccWorldDataset(KlTrackDataset):
    """Restrict KL training to references with generated OccWorld labels."""

    def __init__(self,
                 *args,
                 occworld_label_root: str,
                 occworld_expected_shape: Tuple[int, ...] = (
                     5, 10, 120, 160),
                 occworld_ignore_index: int = 255,
                 occworld_manifest: str = None,
                 occworld_split: str = 'train',
                 occworld_history_root: str = None,
                 occworld_history_count: int = 5,
                 occworld_current_anchor_root: str = None,
                 **kwargs):
        self.occworld_labels = discover_occworld_labels(
            occworld_label_root)
        self.occworld_manifest = occworld_manifest
        self.occworld_split = str(occworld_split)
        if occworld_manifest is not None:
            split_references = load_occworld_split_references(
                occworld_manifest, self.occworld_split)
            missing = sorted(
                set(split_references).difference(self.occworld_labels))
            if missing:
                raise ValueError(
                    f'OccWorld {self.occworld_split} split has no labels '
                    f'for references {missing}')
            self.occworld_labels = {
                reference: self.occworld_labels[reference]
                for reference in split_references
            }
        self.occworld_expected_shape = tuple(
            int(value) for value in occworld_expected_shape)
        self.occworld_ignore_index = int(occworld_ignore_index)
        self.occworld_history_count = int(occworld_history_count)
        if self.occworld_history_count < 1:
            raise ValueError('occworld_history_count must be positive')
        self.occworld_histories = None
        if occworld_history_root is not None:
            histories = discover_occworld_histories(
                occworld_history_root)
            missing = sorted(
                set(self.occworld_labels).difference(histories))
            if missing:
                raise ValueError(
                    'OccWorld split has no histories for references '
                    f'{missing}')
            self.occworld_histories = {
                reference: histories[reference]
                for reference in self.occworld_labels
            }
        self.occworld_current_anchors = None
        if occworld_current_anchor_root is not None:
            anchors = discover_occworld_current_anchors(
                occworld_current_anchor_root)
            missing = sorted(
                set(self.occworld_labels).difference(anchors))
            if missing:
                raise ValueError(
                    'OccWorld split has no current anchors for references '
                    f'{missing}')
            self.occworld_current_anchors = {
                reference: anchors[reference]
                for reference in self.occworld_labels
            }
        super().__init__(*args, **kwargs)
        eligible = []
        for reference_index in sorted(self.occworld_labels):
            if not 0 <= reference_index < len(self.data_infos):
                continue
            sample_idx = int(
                self.data_infos[reference_index].get(
                    'sample_idx', reference_index))
            if sample_idx != reference_index:
                raise ValueError(
                    'OccWorld reference indices require sample_idx to equal '
                    f'the raw dataset index, got {sample_idx} and '
                    f'{reference_index}')
            if self._collect_queue_indices(reference_index) is not None:
                eligible.append(reference_index)
        if not eligible:
            raise ValueError(
                'No OccWorld labels have a valid temporal input queue')
        self.valid_data_indices = eligible
        if hasattr(self, 'flag'):
            self.flag = self.flag[np.asarray(eligible, dtype=np.int64)]

    def prepare_train_data(self, index):
        data = self._prepare_queue_data(self._to_raw_index(index))
        if data is not None:
            return data
        for _ in range(10):
            raw_index = int(np.random.choice(self.valid_data_indices))
            data = self._prepare_queue_data(raw_index)
            if data is not None:
                return data
        return None

    def _union2one(self, queue, raw_meta):
        sample = super()._union2one(queue, raw_meta)
        reference_index = int(raw_meta[-1]['sample_idx'])
        label_path = self.occworld_labels.get(reference_index)
        if label_path is None:
            raise KeyError(
                f'No OccWorld label for reference {reference_index}')
        target, valid = load_occworld_supervision(
            label_path,
            self.occworld_expected_shape,
            ignore_index=self.occworld_ignore_index)
        current_state, current_valid = load_occworld_current_observation(
            label_path, self.occworld_expected_shape[1:])
        current_anchor_path = None
        if self.occworld_current_anchors is not None:
            current_anchor_path = self.occworld_current_anchors[
                reference_index]
            current_state, current_valid = load_occworld_current_anchor(
                current_anchor_path,
                self.occworld_expected_shape[1:],
                expected_reference_index=reference_index)
        history_path = None
        history_state = None
        history_valid = None
        if self.occworld_histories is not None:
            history_path = self.occworld_histories[reference_index]
            history_state, history_valid = load_occworld_history(
                history_path,
                (self.occworld_history_count,
                 *self.occworld_expected_shape[1:]))
            direct_state, direct_valid = (
                load_occworld_direct_current_observation(
                    label_path, self.occworld_expected_shape[1:]))
            if (not torch.equal(history_state[-1], direct_state) or
                    not torch.equal(history_valid[-1], direct_valid)):
                raise ValueError(
                    'OccWorld history does not end at direct causal t=0 '
                    f'for reference {reference_index}')
        sample['gt_world_occ'] = DC(
            target, stack=True, pad_dims=None)
        sample['gt_world_valid'] = DC(
            valid, stack=True, pad_dims=None)
        sample['current_world_state'] = DC(
            current_state, stack=True, pad_dims=None)
        sample['current_world_valid'] = DC(
            current_valid, stack=True, pad_dims=None)
        if current_anchor_path is not None:
            sample['occworld_current_anchor_path'] = DC(
                str(current_anchor_path), cpu_only=True)
        if history_state is not None:
            sample['history_world_state'] = DC(
                history_state, stack=True, pad_dims=None)
            sample['history_world_valid'] = DC(
                history_valid, stack=True, pad_dims=None)
            sample['occworld_history_path'] = DC(
                str(history_path), cpu_only=True)
        sample['occworld_label_path'] = DC(
            str(label_path), cpu_only=True)
        return sample
