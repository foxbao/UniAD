#!/usr/bin/env python
"""Freeze a scene-disjoint internal split for exploratory B24 training."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = Path(
    'documents/patent_2026_occ/'
    'kl_occworld_full_train3_final30_manifest_v1.json')
INTERNAL_DEV_FRACTION = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _evenly_spaced(items: Sequence, count: int) -> list:
    if not 1 <= count <= len(items):
        raise ValueError(f'Cannot select {count} items from {len(items)}')
    if count == len(items):
        return list(items)
    return [items[round(index * (len(items) - 1) / (count - 1))]
            for index in range(count)] if count > 1 else [items[len(items) // 2]]


def group_records_by_scene(records: Sequence[Mapping]) -> list:
    """Group a frozen reference split without changing its within-scene rows."""
    grouped = {}
    for record in records:
        scene = str(record['scene_token'])
        grouped.setdefault(scene, []).append(dict(record))
    if not grouped:
        raise ValueError('B24 source train split is empty')
    for scene_records in grouped.values():
        scene_records.sort(key=lambda record: int(record['reference_index']))
    return sorted(
        grouped.values(), key=lambda values: int(values[0]['reference_index']))


def freeze_exploratory_splits(train_records: Sequence[Mapping]) -> dict:
    """Reserve 20% of original train scenes for non-independent dev checks."""
    scene_groups = group_records_by_scene(train_records)
    scene_count = len(scene_groups)
    dev_scene_count = max(1, int(scene_count * INTERNAL_DEV_FRACTION))
    dev_groups = _evenly_spaced(scene_groups, dev_scene_count)
    dev_scenes = {str(group[0]['scene_token']) for group in dev_groups}
    internal_dev = [
        record for group in scene_groups
        if str(group[0]['scene_token']) in dev_scenes
        for record in group
    ]
    internal_train = [
        record for group in scene_groups
        if str(group[0]['scene_token']) not in dev_scenes
        for record in group
    ]
    if not internal_train or not internal_dev:
        raise ValueError('B24 internal split must keep train and dev rows')
    if len(internal_train) + len(internal_dev) != len(train_records):
        raise AssertionError('B24 internal split lost source records')
    train_scenes = {str(record['scene_token']) for record in internal_train}
    if train_scenes.intersection(dev_scenes):
        raise AssertionError('B24 internal train/dev scenes overlap')
    return {
        'internal_train': sorted(
            internal_train, key=lambda record: int(record['reference_index'])),
        'internal_dev': sorted(
            internal_dev, key=lambda record: int(record['reference_index'])),
    }


def _scene_set(records: Sequence[Mapping]) -> set:
    return {str(record['scene_token']) for record in records}


def build_exploratory_manifest(source_manifest: Mapping,
                               source_path: Path,
                               source_sha256: str) -> dict:
    required_splits = {'train', 'validation', 'test', 'final_holdout'}
    source_splits = source_manifest.get('splits')
    if not isinstance(source_splits, dict):
        raise ValueError('Source manifest has no splits')
    missing = sorted(required_splits.difference(source_splits))
    if missing:
        raise ValueError(f'Source manifest is missing {missing}')
    splits = freeze_exploratory_splits(source_splits['train'])
    internal_train_scenes = _scene_set(splits['internal_train'])
    internal_dev_scenes = _scene_set(splits['internal_dev'])
    reserved_scenes = set().union(*[
        _scene_set(source_splits[name])
        for name in ('validation', 'test', 'final_holdout')
    ])
    if internal_train_scenes.intersection(reserved_scenes):
        raise ValueError('B24 internal train overlaps a reserved formal split')
    if internal_dev_scenes.intersection(reserved_scenes):
        raise ValueError('B24 internal dev overlaps a reserved formal split')
    return {
        'schema_version': 1,
        'name': 'kl_occworld_b24_exploratory_internal_split_v1',
        'status': 'frozen_before_b24_exploratory_head_training',
        'strategy': 'scene_disjoint_temporally_even_20pct_internal_dev',
        'source_manifest': str(source_path),
        'source_manifest_sha256': str(source_sha256),
        'selection_uses_occworld_labels': False,
        'selection_uses_model_predictions': False,
        'is_independent_generalization_evidence': False,
        'checkpoint_selection_scope': 'exploratory_internal_dev_only',
        'formal_promotion_requires_new_scene_validation': True,
        'internal_dev_fraction': INTERNAL_DEV_FRACTION,
        'source_train_scene_count': len(
            internal_train_scenes | internal_dev_scenes),
        'source_train_reference_count': len(source_splits['train']),
        'excluded_source_splits': [
            'validation', 'test', 'final_holdout'],
        'splits': splits,
        'split_scene_tokens': {
            'internal_train': sorted(internal_train_scenes),
            'internal_dev': sorted(internal_dev_scenes),
        },
    }


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path,
                        default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b24_exploratory_internal_split_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_file.exists():
        raise FileExistsError(
            f'Refusing to overwrite frozen B24 internal split: '
            f'{args.out_file}')
    with args.source_manifest.open() as source:
        source_manifest = json.load(source)
    manifest = build_exploratory_manifest(
        source_manifest, args.source_manifest, _sha256(args.source_manifest))
    _write_json(args.out_file, manifest)
    print(json.dumps({
        'status': manifest['status'],
        'internal_train_scene_count': len(
            manifest['split_scene_tokens']['internal_train']),
        'internal_train_reference_count': len(
            manifest['splits']['internal_train']),
        'internal_dev_scene_count': len(
            manifest['split_scene_tokens']['internal_dev']),
        'internal_dev_reference_count': len(
            manifest['splits']['internal_dev']),
        'manifest': str(args.out_file),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
