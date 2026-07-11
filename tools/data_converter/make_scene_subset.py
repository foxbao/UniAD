#!/usr/bin/env python
"""Build a deterministic, scene-complete subset for temporal training."""

import argparse
import pickle


def _records(data):
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list'], 'data_list'
    if isinstance(data, dict) and 'infos' in data:
        return data['infos'], 'infos'
    if isinstance(data, list):
        return data, None
    raise TypeError(f'Unsupported annotation container: {type(data)!r}')


def make_scene_subset(input_path, output_path, min_samples):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    records, key = _records(data)

    scene_sizes = {}
    scene_order = []
    for record in records:
        scene = record.get('scene_token')
        if not scene:
            raise ValueError('Every record must have a non-empty scene_token')
        if scene not in scene_sizes:
            scene_sizes[scene] = 0
            scene_order.append(scene)
        scene_sizes[scene] += 1

    selected = set()
    selected_samples = 0
    for scene in scene_order:
        selected.add(scene)
        selected_samples += scene_sizes[scene]
        if selected_samples >= min_samples:
            break

    subset_records = [
        record for record in records
        if record['scene_token'] in selected
    ]
    if isinstance(data, dict):
        subset = dict(data)
        subset[key] = subset_records
    else:
        subset = subset_records

    with open(output_path, 'wb') as f:
        pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f'total={len(records)} selected_scenes={len(selected)} '
        f'selected_samples={len(subset_records)} output={output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Create an ordered subset containing only complete scenes.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-samples', type=int, required=True)
    args = parser.parse_args()
    if args.min_samples <= 0:
        parser.error('--min-samples must be positive')
    make_scene_subset(args.input, args.output, args.min_samples)


if __name__ == '__main__':
    main()
