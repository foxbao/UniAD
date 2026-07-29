#!/usr/bin/env python
"""Restore paired raw classes outside B24 candidate events."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def fix_export(input_root: Path, output_root: Path):
    input_root = input_root.resolve()
    if output_root.exists():
        raise FileExistsError(f'Refusing to overwrite {output_root}')
    files = sorted(input_root.rglob('*__occworld_prediction.npz'))
    if not files:
        raise FileNotFoundError(f'No predictions below {input_root}')
    output_root.mkdir(parents=True)
    changed_files = []
    changed_voxels = 0
    for source in files:
        destination = output_root / source.relative_to(input_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source, allow_pickle=False) as archive:
            required = {
                'world_pred_class_3d', 'raw_world_pred_class_3d',
                'event_candidate_mask_3d'}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f'{source} lacks keys: {missing}')
            final = np.asarray(archive['world_pred_class_3d'])
            raw = np.asarray(archive['raw_world_pred_class_3d'])
            event_mask = np.asarray(
                archive['event_candidate_mask_3d'], dtype=np.bool_)
            if (final.shape != raw.shape or
                    event_mask.shape != final[1:].shape):
                raise ValueError(f'Prediction shape mismatch in {source}')
            correction = np.zeros_like(final, dtype=np.bool_)
            correction[0] = final[0] != raw[0]
            correction[1:] = (final[1:] != raw[1:]) & ~event_mask
            count = int(np.count_nonzero(correction))
            if count == 0:
                os.link(source, destination)
                continue
            payload = {
                key: np.array(archive[key], copy=True)
                for key in archive.files
            }
        payload['world_pred_class_3d'][correction] = (
            payload['raw_world_pred_class_3d'][correction])
        np.savez_compressed(destination, **payload)
        changed_voxels += count
        changed_files.append({
            'reference_index': int(payload['reference_index']),
            'relative_path': str(source.relative_to(input_root)),
            'corrected_voxels': count,
        })
    return {
        'schema_version': 1,
        'purpose': (
            'Apply the source-level exact raw-logit invariant to artifacts '
            'exported before the numerical round-trip fix.'),
        'input_root': str(input_root),
        'output_root': str(output_root),
        'prediction_file_count': len(files),
        'hardlinked_unchanged_file_count': len(files) - len(changed_files),
        'rewritten_file_count': len(changed_files),
        'corrected_non_event_voxels': changed_voxels,
        'changed_files': changed_files,
        'uses_labels_or_ground_truth': False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--out-file', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = fix_export(args.input_root, args.output_root)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + '\n'
    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
