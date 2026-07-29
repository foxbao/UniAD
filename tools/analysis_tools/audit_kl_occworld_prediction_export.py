#!/usr/bin/env python
"""Verify that an OccWorld prediction export covers one frozen split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--prediction-root', type=Path, required=True)
    parser.add_argument('--expected-checkpoint-epoch', type=int)
    parser.add_argument('--prediction-class-key',
                        default='world_pred_class_3d')
    parser.add_argument('--raw-class-key', default='raw_world_pred_class_3d')
    parser.add_argument('--require-raw-equals-prediction',
                        action='store_true')
    parser.add_argument('--out-file', type=Path)
    return parser.parse_args()


def _manifest_references(manifest_path: Path, split: str):
    with manifest_path.open(encoding='utf-8') as input_file:
        manifest = json.load(input_file)
    try:
        rows = manifest['splits'][split]
    except KeyError as error:
        raise KeyError(f'Manifest has no split {split!r}') from error
    references = [int(row['reference_index']) for row in rows]
    if len(references) != len(set(references)):
        raise ValueError(f'Manifest split {split!r} has duplicate references')
    return manifest, set(references)


def audit_export(manifest_path: Path, split: str, prediction_root: Path,
                 expected_checkpoint_epoch=None,
                 prediction_class_key='world_pred_class_3d',
                 raw_class_key='raw_world_pred_class_3d',
                 require_raw_equals_prediction=False):
    manifest, expected_references = _manifest_references(manifest_path, split)
    files = sorted(prediction_root.rglob('*__occworld_prediction.npz'))
    found_references = set()
    missing_keys = []
    wrong_epochs = []
    shapes = set()
    raw_mismatch_references = []
    duplicate_references = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            required = {'reference_index', prediction_class_key}
            if expected_checkpoint_epoch is not None:
                required.add('checkpoint_epoch')
            if require_raw_equals_prediction:
                required.add(raw_class_key)
            absent = sorted(required.difference(archive.files))
            if absent:
                missing_keys.append({'path': str(path), 'keys': absent})
                continue
            reference_index = int(archive['reference_index'])
            if reference_index in found_references:
                duplicate_references.append(reference_index)
            found_references.add(reference_index)
            prediction = np.asarray(archive[prediction_class_key])
            shapes.add(tuple(prediction.shape))
            if (expected_checkpoint_epoch is not None and
                    int(archive['checkpoint_epoch']) !=
                    expected_checkpoint_epoch):
                wrong_epochs.append({
                    'path': str(path),
                    'epoch': int(archive['checkpoint_epoch']),
                })
            if (require_raw_equals_prediction and not np.array_equal(
                    prediction, np.asarray(archive[raw_class_key]))):
                raw_mismatch_references.append(reference_index)
    summary = {
        'manifest': str(manifest_path),
        'manifest_sha256': hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
        'manifest_name': manifest.get('name'),
        'split': split,
        'prediction_root': str(prediction_root),
        'expected_reference_count': len(expected_references),
        'prediction_file_count': len(files),
        'found_reference_count': len(found_references),
        'missing_reference_indices': sorted(
            expected_references.difference(found_references)),
        'unexpected_reference_indices': sorted(
            found_references.difference(expected_references)),
        'duplicate_reference_indices': sorted(set(duplicate_references)),
        'missing_keys': missing_keys,
        'wrong_checkpoint_epochs': wrong_epochs,
        'prediction_shapes': [list(shape) for shape in sorted(shapes)],
        'raw_mismatch_reference_indices': sorted(raw_mismatch_references),
        'raw_equals_prediction_required': bool(
            require_raw_equals_prediction),
    }
    summary['passed'] = not any((
        summary['missing_reference_indices'],
        summary['unexpected_reference_indices'],
        summary['duplicate_reference_indices'],
        summary['missing_keys'],
        summary['wrong_checkpoint_epochs'],
        summary['raw_mismatch_reference_indices'],
    ))
    return summary


def main():
    args = parse_args()
    summary = audit_export(
        manifest_path=args.manifest,
        split=args.split,
        prediction_root=args.prediction_root,
        expected_checkpoint_epoch=args.expected_checkpoint_epoch,
        prediction_class_key=args.prediction_class_key,
        raw_class_key=args.raw_class_key,
        require_raw_equals_prediction=args.require_raw_equals_prediction)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + '\n'
    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    if not summary['passed']:
        raise SystemExit('OccWorld prediction export audit failed')


if __name__ == '__main__':
    main()
