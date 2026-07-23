#!/usr/bin/env python
"""Validate generated OccWorld artifacts and finalize a planned split."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _resolve_path,
)


def _artifact_index(root: Path, suffix: str, key: str) -> Dict[int, Path]:
    mapping = {}
    for path in sorted(root.glob(f'*/*__{suffix}.npz')):
        with np.load(path, allow_pickle=False) as artifact:
            reference_index = int(artifact[key])
        if reference_index in mapping:
            raise ValueError(
                f'Duplicate {suffix} for reference {reference_index}')
        mapping[reference_index] = path
    return mapping


def _split_records(manifest: Mapping) -> Dict[str, Sequence[Mapping]]:
    splits = manifest.get('splits')
    required = ('train', 'validation', 'test')
    if not isinstance(splits, dict) or any(name not in splits for name in required):
        raise ValueError('Manifest must contain train/validation/test splits')
    all_scenes = []
    all_references = []
    for name in required:
        records = splits[name]
        if not records:
            raise ValueError(f'{name} split is empty')
        scenes = [str(record['scene_token']) for record in records]
        references = [int(record['reference_index']) for record in records]
        if len(scenes) != len(set(scenes)):
            raise ValueError(f'{name} contains duplicate scenes')
        if len(references) != len(set(references)):
            raise ValueError(f'{name} contains duplicate references')
        all_scenes.extend(scenes)
        all_references.extend(references)
    if len(all_scenes) != len(set(all_scenes)):
        raise ValueError('A scene is assigned to more than one split')
    if len(all_references) != len(set(all_references)):
        raise ValueError('A reference is assigned to more than one split')
    return {name: splits[name] for name in required}


def _class_statistics(states: np.ndarray, valid: np.ndarray) -> dict:
    counts = [
        int(np.count_nonzero(valid & (states == state_value)))
        for state_value in (1, 2, 3)
    ]
    known = sum(counts)
    return {
        'free_voxels': counts[0],
        'static_voxels': counts[1],
        'instance_voxels': counts[2],
        'known_voxels': known,
        'known_ratio': float(known / states.size),
        'class_fractions': [
            float(count / max(known, 1)) for count in counts
        ],
    }


def finalize_manifest(
        planned_manifest: Mapping, infos: Sequence[Mapping],
        sequence_root: Path, history_root: Path) -> dict:
    splits = _split_records(planned_manifest)
    sequence_mapping = _artifact_index(
        sequence_root, 'occworld_sequence', 'reference_index')
    history_mapping = _artifact_index(
        history_root, 'occworld_history', 'reference_index')
    expected_references = {
        int(record['reference_index'])
        for records in splits.values() for record in records
    }
    for name, mapping in (
            ('sequence', sequence_mapping), ('history', history_mapping)):
        actual = set(mapping)
        if actual != expected_references:
            raise ValueError(
                f'{name} reference mismatch: missing='
                f'{sorted(expected_references - actual)}, extra='
                f'{sorted(actual - expected_references)}')

    split_statistics = {}
    max_future_time_error = 0.0
    max_history_time_error = 0.0
    for split_name, records in splits.items():
        split_states = []
        split_valid = []
        for record in records:
            reference_index = int(record['reference_index'])
            expected_scene = str(record['scene_token'])
            if not 0 <= reference_index < len(infos):
                raise ValueError(f'Invalid reference {reference_index}')
            actual_scene = str(infos[reference_index].get('scene_token'))
            if actual_scene != expected_scene:
                raise ValueError(
                    f'Reference {reference_index} scene mismatch: '
                    f'{actual_scene} != {expected_scene}')

            with np.load(
                    sequence_mapping[reference_index],
                    allow_pickle=False) as sequence:
                state = sequence['world_target_state_3d']
                valid = sequence['world_target_valid_3d'] > 0
                if state.shape != (5, 10, 120, 160):
                    raise ValueError(
                        f'Reference {reference_index} sequence shape '
                        f'{state.shape}')
                if sequence['current_observation_state_3d'].shape != (
                        10, 120, 160):
                    raise ValueError(
                        f'Reference {reference_index} current shape invalid')
                np.testing.assert_array_equal(
                    sequence['target_offsets'], np.arange(5))
                target_indices = sequence['target_indices'].astype(int)
                target_times = sequence['target_times_s'].astype(float)
                for target_index in target_indices:
                    if str(infos[target_index].get('scene_token')) != expected_scene:
                        raise ValueError(
                            f'Reference {reference_index} target '
                            f'{target_index} crosses scene')
                max_future_time_error = max(
                    max_future_time_error,
                    float(np.max(np.abs(
                        target_times - np.arange(5) * 0.5))))
                direct_t0 = np.array(
                    sequence['direct_observation_state_3d'][0], copy=True)
                split_states.append(np.array(state, copy=True))
                split_valid.append(np.array(valid, copy=True))

            with np.load(
                    history_mapping[reference_index],
                    allow_pickle=False) as history:
                history_state = history['history_observation_state_3d']
                if history_state.shape != (5, 10, 120, 160):
                    raise ValueError(
                        f'Reference {reference_index} history shape '
                        f'{history_state.shape}')
                np.testing.assert_array_equal(
                    history['history_offsets'], np.arange(-4, 1))
                np.testing.assert_array_equal(history_state[-1], direct_t0)
                history_times = history['history_times_s'].astype(float)
                max_history_time_error = max(
                    max_history_time_error,
                    float(np.max(np.abs(
                        history_times - np.arange(-4, 1) * 0.5))))

        split_statistics[split_name] = _class_statistics(
            np.stack(split_states), np.stack(split_valid))

    result = dict(planned_manifest)
    result.update({
        'name': 'kl_occworld_scene_split_v2',
        'status': 'ready',
        'sequence_root': str(sequence_root),
        'history_root': str(history_root),
        'sequence_label_count': len(sequence_mapping),
        'history_queue_count': len(history_mapping),
        'artifact_validation': {
            'reference_sets_exactly_equal': True,
            'scene_sets_disjoint': True,
            'sequence_shape': [5, 10, 120, 160],
            'history_shape': [5, 10, 120, 160],
            'history_last_equals_direct_t0': True,
            'max_future_time_error_s': max_future_time_error,
            'max_history_time_error_s': max_history_time_error,
        },
        'split_statistics': split_statistics,
    })
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--planned-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v2_planned.json'))
    parser.add_argument(
        '--ann-file', type=Path,
        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument(
        '--history-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_history_expanded70'))
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v2.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    with args.planned_manifest.open() as source:
        planned_manifest = json.load(source)
    infos, _ = _load_infos(_resolve_path(str(args.ann_file)))
    manifest = finalize_manifest(
        planned_manifest, infos, args.sequence_root, args.history_root)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'name': manifest['name'],
        'status': manifest['status'],
        'split_counts': manifest['split_counts'],
        'sequence_label_count': manifest['sequence_label_count'],
        'history_queue_count': manifest['history_queue_count'],
        'artifact_validation': manifest['artifact_validation'],
        'split_statistics': manifest['split_statistics'],
    }, ensure_ascii=False, indent=2))
    print(f'manifest={args.out_file}')


if __name__ == '__main__':
    main()
