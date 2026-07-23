#!/usr/bin/env python
"""Build a deterministic scene-level split for generated KL OccWorld labels."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mmcv
import numpy as np

from tools.data_converter.kl_occworld_dataset import (
    _discover_histories,
    _discover_labels,
)


def _annotation_infos(annotation) -> Sequence[Mapping[str, object]]:
    if isinstance(annotation, dict):
        for key in ('data_list', 'infos'):
            if key in annotation:
                return annotation[key]
    if isinstance(annotation, list):
        return annotation
    raise ValueError('Annotation must contain data_list or infos')


def _sequence_mapping(sequence_root: Path) -> Dict[int, Path]:
    mapping = {}
    for path in _discover_labels(sequence_root):
        with np.load(path, allow_pickle=False) as label:
            reference_index = int(label['reference_index'])
        if reference_index in mapping:
            raise ValueError(
                f'Duplicate sequence label for reference {reference_index}')
        mapping[reference_index] = path
    return mapping


def _sample_record(infos: Sequence[Mapping[str, object]],
                   reference_index: int) -> Dict[str, object]:
    if not 0 <= reference_index < len(infos):
        raise ValueError(
            f'Reference {reference_index} is outside annotation range')
    info = infos[reference_index]
    sample_index = int(info.get('sample_idx', reference_index))
    if sample_index != reference_index:
        raise ValueError(
            f'Reference {reference_index} maps to sample_idx {sample_index}')
    scene_token = info.get('scene_token')
    if not scene_token:
        raise ValueError(f'Reference {reference_index} has no scene_token')
    record = {
        'reference_index': reference_index,
        'scene_token': str(scene_token),
    }
    if info.get('token') is not None:
        record['sample_token'] = str(info['token'])
    if info.get('timestamp') is not None:
        record['timestamp'] = float(info['timestamp'])
    return record


def build_scene_split(
        infos: Sequence[Mapping[str, object]],
        eligible_references: Iterable[int],
        train_scene_count: int,
        validation_scene_count: int) -> Dict[str, object]:
    """Group all eligible references by scene before assigning splits."""
    records = [
        _sample_record(infos, int(reference))
        for reference in sorted(set(eligible_references))
    ]
    scene_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        scene_groups[record['scene_token']].append(record)
    ordered_groups = sorted(
        scene_groups.items(),
        key=lambda item: min(
            record['reference_index'] for record in item[1]))
    if train_scene_count < 1 or validation_scene_count < 1:
        raise ValueError('Train and validation scene counts must be positive')
    test_start = train_scene_count + validation_scene_count
    if test_start >= len(ordered_groups):
        raise ValueError('Split must leave at least one scene for test')
    grouped_splits = {
        'train': ordered_groups[:train_scene_count],
        'validation': ordered_groups[
            train_scene_count:test_start],
        'test': ordered_groups[test_start:],
    }
    splits = {}
    split_scene_tokens = {}
    for name, groups in grouped_splits.items():
        split_scene_tokens[name] = [scene for scene, _ in groups]
        splits[name] = sorted(
            [record for _, group in groups for record in group],
            key=lambda record: record['reference_index'])
    all_scene_tokens = [
        scene for values in split_scene_tokens.values() for scene in values]
    if len(all_scene_tokens) != len(set(all_scene_tokens)):
        raise AssertionError('A scene was assigned to more than one split')
    return {
        'strategy': 'scene_groups_sorted_by_min_reference',
        'scene_count': len(ordered_groups),
        'reference_count': len(records),
        'split_scene_tokens': split_scene_tokens,
        'splits': splits,
    }


def _validate_target_scenes(
        infos: Sequence[Mapping[str, object]],
        sequence_mapping: Mapping[int, Path],
        eligible_references: Iterable[int]):
    for reference_index in eligible_references:
        reference_scene = str(infos[reference_index]['scene_token'])
        with np.load(
                sequence_mapping[reference_index],
                allow_pickle=False) as label:
            if 'target_indices' not in label.files:
                continue
            target_indices = np.asarray(
                label['target_indices'], dtype=np.int64).tolist()
        for target_index in target_indices:
            if not 0 <= target_index < len(infos):
                raise ValueError(
                    f'Target {target_index} is outside annotation range')
            target_scene = str(infos[target_index].get('scene_token'))
            if target_scene != reference_scene:
                raise ValueError(
                    f'Reference {reference_index} crosses from '
                    f'{reference_scene} into {target_scene}')


def build_manifest(annotation_file: Path,
                   sequence_root: Path,
                   history_root: Path,
                   train_scene_count: int,
                   validation_scene_count: int,
                   name: str) -> Dict[str, object]:
    annotation = mmcv.load(str(annotation_file))
    infos = _annotation_infos(annotation)
    sequence_mapping = _sequence_mapping(sequence_root)
    history_mapping = _discover_histories(history_root)
    eligible_references = sorted(
        set(sequence_mapping).intersection(history_mapping))
    if not eligible_references:
        raise ValueError('No sequence label has a matching history queue')
    _validate_target_scenes(
        infos, sequence_mapping, eligible_references)
    manifest = build_scene_split(
        infos, eligible_references,
        train_scene_count=train_scene_count,
        validation_scene_count=validation_scene_count)
    manifest.update({
        'schema_version': 1,
        'name': name,
        'annotation_file': str(annotation_file),
        'sequence_root': str(sequence_root),
        'history_root': str(history_root),
        'sequence_label_count': len(sequence_mapping),
        'history_queue_count': len(history_mapping),
        'excluded_sequence_references_missing_history': sorted(
            set(sequence_mapping).difference(history_mapping)),
    })
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', type=Path,
        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_batch20'))
    parser.add_argument(
        '--history-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_history_batch15'))
    parser.add_argument('--train-scene-count', type=int, default=10)
    parser.add_argument('--validation-scene-count', type=int, default=2)
    parser.add_argument('--name', default='kl_occworld_scene_split_v1')
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        annotation_file=args.ann_file,
        sequence_root=args.sequence_root,
        history_root=args.history_root,
        train_scene_count=args.train_scene_count,
        validation_scene_count=args.validation_scene_count,
        name=args.name)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
