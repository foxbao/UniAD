#!/usr/bin/env python
"""PyTorch Dataset adapter for the KL OccWorld sequence prototype.

The generated NPZ contains both direct future observations and completed
world targets.  This adapter deliberately exposes only the current
observation as input; future direct observations stay inside the NPZ for
auditing and are never returned as model input.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


UNKNOWN = 0
FREE = 1
STATIC_OCCUPIED = 2
INSTANCE_OCCUPIED = 3
IGNORE_INDEX = 255
WORLD_CLASS_COUNT = 3
WORLD_STATE_COUNT = 4

REQUIRED_KEYS = frozenset({
    'current_observation_state_3d',
    'current_observation_valid_3d',
    'world_target_state_3d',
    'world_target_valid_3d',
    'target_offsets',
    'target_times_s',
    'nominal_target_times_s',
    'reference_index',
    'pc_range',
    'occ_size',
})
HISTORY_REQUIRED_KEYS = frozenset({
    'history_observation_state_3d',
    'history_observation_valid_3d',
    'history_offsets',
    'history_times_s',
    'nominal_history_times_s',
    'reference_index',
})


def _as_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Copy compressed-NPZ data before closing the archive."""
    return torch.as_tensor(np.array(array, copy=True), dtype=dtype)


def _discover_labels(sequence_root: Union[str, Path]) -> List[Path]:
    root = Path(sequence_root)
    paths = sorted(root.glob('*/*__occworld_sequence.npz'))
    if not paths:
        raise FileNotFoundError(
            f'No OccWorld sequence labels found below {root}')
    return paths


def _discover_histories(history_root: Union[str, Path]) -> Dict[int, Path]:
    root = Path(history_root)
    mapping = {}
    for path in sorted(root.glob('*/*__occworld_history.npz')):
        with np.load(path, allow_pickle=False) as history:
            if 'reference_index' not in history.files:
                raise ValueError(f'{path} has no reference_index')
            reference_index = int(history['reference_index'])
        if reference_index in mapping:
            raise ValueError(
                f'Duplicate history for reference {reference_index}')
        mapping[reference_index] = path
    if not mapping:
        raise FileNotFoundError(
            f'No OccWorld history labels found below {root}')
    return mapping


def _to_class_target(state: torch.Tensor,
                     valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Map raw world states [unknown, free, static, instance] to loss IDs."""
    target = torch.full_like(state, IGNORE_INDEX, dtype=torch.long)
    known = state != UNKNOWN
    if valid is not None:
        if valid.shape != state.shape:
            raise ValueError('World state and valid mask shapes must match')
        known &= valid
    target[known] = state[known] - 1
    if torch.any(target[known] >= WORLD_CLASS_COUNT):
        raise ValueError('World state contains an unknown class value')
    return target


class KLOccWorldSequenceDataset(Dataset):
    """Read generated KL OccWorld sequences as plain PyTorch tensors.

    By default, target index 0 (the reference/current frame) is excluded,
    so the sample predicts the four future horizons at approximately
    0.5, 1.0, 1.5 and 2.0 seconds.  Set ``target_start=0`` when a current
    frame reconstruction target is intentionally needed as an auxiliary loss.
    """

    def __init__(self,
                 sequence_root: Union[str, Path],
                 target_start: int = 1,
                 target_end: Optional[int] = None,
                 expected_shape: Optional[Sequence[int]] = None,
                 history_root: Optional[Union[str, Path]] = None,
                 drop_missing_history: bool = False):
        self.sequence_root = Path(sequence_root)
        self.label_paths = _discover_labels(self.sequence_root)
        self.target_start = int(target_start)
        self.target_end = None if target_end is None else int(target_end)
        if self.target_start < 0:
            raise ValueError('target_start must be non-negative')
        if self.target_end is not None and self.target_end <= self.target_start:
            raise ValueError('target_end must be greater than target_start')
        self.expected_shape = (
            None if expected_shape is None else tuple(int(x) for x in expected_shape))
        self._validate_index_metadata()
        self.history_paths = None
        if history_root is not None:
            history_mapping = _discover_histories(history_root)
            missing = [
                reference for reference in self.reference_indices
                if reference not in history_mapping]
            if missing and not drop_missing_history:
                raise ValueError(
                    f'Missing history for reference indices {missing}')
            if missing:
                keep = [
                    position for position, reference in enumerate(
                        self.reference_indices)
                    if reference in history_mapping]
                self.label_paths = [self.label_paths[position]
                                    for position in keep]
                self.reference_indices = [self.reference_indices[position]
                                          for position in keep]
                self.current_shapes = [self.current_shapes[position]
                                       for position in keep]
            self.history_paths = [
                history_mapping[reference]
                for reference in self.reference_indices]
            self._validate_history_metadata()

    def _validate_index_metadata(self):
        self.reference_indices = []
        self.current_shapes = []
        for path in self.label_paths:
            with np.load(path, allow_pickle=False) as label:
                missing = sorted(REQUIRED_KEYS.difference(label.files))
                if missing:
                    raise ValueError(
                        f'{path} is missing required dataset keys: {missing}')
                current_shape = tuple(label['current_observation_state_3d'].shape)
                target_shape = tuple(label['world_target_state_3d'].shape)
                if len(current_shape) != 3 or len(target_shape) != 4:
                    raise ValueError(f'Invalid OccWorld shape in {path}')
                if tuple(label['current_observation_valid_3d'].shape) != current_shape:
                    raise ValueError(
                        f'Current valid-mask shape mismatch in {path}')
                if tuple(label['world_target_valid_3d'].shape) != target_shape:
                    raise ValueError(
                        f'Target valid-mask shape mismatch in {path}')
                if target_shape[1:] != current_shape:
                    raise ValueError(
                        f'Current/target shape mismatch in {path}: '
                        f'{current_shape} vs {target_shape}')
                target_count = target_shape[0]
                for key in ('target_offsets', 'target_times_s',
                            'nominal_target_times_s'):
                    if tuple(label[key].shape) != (target_count,):
                        raise ValueError(
                            f'{key} shape mismatch in {path}')
                end = target_count if self.target_end is None else self.target_end
                if self.target_start >= target_count or end > target_count:
                    raise ValueError(
                        f'Target slice [{self.target_start}:{end}] is outside '
                        f'{target_count} frames in {path}')
                if (self.expected_shape is not None and
                        current_shape != self.expected_shape):
                    raise ValueError(
                        f'Unexpected current shape in {path}: {current_shape}')
                self.reference_indices.append(int(label['reference_index']))
                self.current_shapes.append(current_shape)

    def _validate_history_metadata(self):
        for path, reference_index, current_shape in zip(
                self.history_paths, self.reference_indices,
                self.current_shapes):
            with np.load(path, allow_pickle=False) as history:
                missing = sorted(
                    HISTORY_REQUIRED_KEYS.difference(history.files))
                if missing:
                    raise ValueError(
                        f'{path} is missing history keys: {missing}')
                if int(history['reference_index']) != reference_index:
                    raise ValueError(
                        f'History reference mismatch in {path}')
                state_shape = tuple(
                    history['history_observation_state_3d'].shape)
                valid_shape = tuple(
                    history['history_observation_valid_3d'].shape)
                if len(state_shape) != 4 or valid_shape != state_shape:
                    raise ValueError(f'Invalid history shape in {path}')
                if state_shape[1:] != current_shape:
                    raise ValueError(
                        f'History/current shape mismatch in {path}: '
                        f'{state_shape[1:]} vs {current_shape}')
                history_count = state_shape[0]
                for key in ('history_offsets', 'history_times_s',
                            'nominal_history_times_s'):
                    if tuple(history[key].shape) != (history_count,):
                        raise ValueError(
                            f'{key} shape mismatch in {path}')
                if int(history['history_offsets'][-1]) != 0:
                    raise ValueError(
                        f'History must end at offset 0 in {path}')

    def __len__(self) -> int:
        return len(self.label_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.label_paths[index]
        history_output = None
        if self.history_paths is not None:
            history_path = self.history_paths[index]
            with np.load(history_path, allow_pickle=False) as history:
                history_state = _as_tensor(
                    history['history_observation_state_3d'], torch.long)
                history_valid = _as_tensor(
                    history['history_observation_valid_3d'], torch.bool)
                history_output = {
                    'history_state_3d': history_state,
                    'history_valid_3d': history_valid,
                    'history_one_hot_3d': F.one_hot(
                        history_state, num_classes=WORLD_STATE_COUNT
                    ).permute(0, 4, 1, 2, 3).to(torch.float32),
                    'history_offsets': _as_tensor(
                        history['history_offsets'], torch.long),
                    'history_times_s': _as_tensor(
                        history['history_times_s'], torch.float32),
                    'nominal_history_times_s': _as_tensor(
                        history['nominal_history_times_s'], torch.float32),
                    'history_path': str(history_path),
                }
        with np.load(path, allow_pickle=False) as label:
            sequence_current_state = _as_tensor(
                label['current_observation_state_3d'], torch.long)
            sequence_current_valid = _as_tensor(
                label['current_observation_valid_3d'], torch.bool)
            if history_output is None:
                current_state = sequence_current_state
                current_valid = sequence_current_valid
            else:
                current_state = history_output['history_state_3d'][-1]
                current_valid = history_output['history_valid_3d'][-1]
            world_state = _as_tensor(
                label['world_target_state_3d'], torch.long)
            world_valid = _as_tensor(
                label['world_target_valid_3d'], torch.bool)
            start = self.target_start
            end = world_state.shape[0] if self.target_end is None else self.target_end
            world_state = world_state[start:end]
            world_valid = world_valid[start:end]
            target_times = _as_tensor(label['target_times_s'][start:end], torch.float32)
            nominal_times = _as_tensor(
                label['nominal_target_times_s'][start:end], torch.float32)
            target_offsets = _as_tensor(
                label['target_offsets'][start:end], torch.long)

            output: Dict[str, object] = {
                # Raw current state is useful for debugging and visualization.
                'input_state_3d': current_state,
                'input_valid_3d': current_valid,
                # Four channels preserve unknown/free/static/instance semantics.
                'input_one_hot_3d': F.one_hot(
                    current_state, num_classes=WORLD_STATE_COUNT
                ).permute(3, 0, 1, 2).to(torch.float32),
                # Raw labels retain the world-state convention for inspection.
                'target_state_3d': world_state,
                'target_valid_3d': world_valid,
                # Loss labels use free/static/instance = 0/1/2 and unknown=255.
                'target_class_3d': _to_class_target(
                    world_state, world_valid),
                'target_times_s': target_times,
                'nominal_target_times_s': nominal_times,
                'target_offsets': target_offsets,
                'reference_index': torch.as_tensor(
                    int(label['reference_index']), dtype=torch.long),
                'pc_range': _as_tensor(label['pc_range'], torch.float32),
                'occ_size': _as_tensor(label['occ_size'], torch.long),
                'label_path': str(path),
            }
            # These 2D targets are optional for the first 3D baseline, but
            # keeping them here avoids reopening the NPZ for a later auxiliary
            # traversability/navigability loss.
            if 'world_target_state_bev' in label.files:
                output['target_world_bev'] = _as_tensor(
                    label['world_target_state_bev'][start:end], torch.long)
            if 'traversability_state_bev' in label.files:
                output['target_traversability_bev'] = _as_tensor(
                    label['traversability_state_bev'][start:end], torch.long)
                output['target_traversability_valid_bev'] = _as_tensor(
                    label['traversability_valid_bev'][start:end], torch.bool)
            if 'navigability_state_bev' in label.files:
                output['target_navigability_bev'] = _as_tensor(
                    label['navigability_state_bev'][start:end], torch.long)
                output['target_navigability_valid_bev'] = _as_tensor(
                    label['navigability_valid_bev'][start:end], torch.bool)
            if history_output is not None:
                output.update(history_output)
                output['sequence_current_state_3d'] = sequence_current_state
                output['sequence_current_valid_3d'] = sequence_current_valid
        return output
