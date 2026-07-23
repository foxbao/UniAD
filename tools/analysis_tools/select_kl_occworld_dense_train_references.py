#!/usr/bin/env python
"""Select multiple valid references inside frozen OccWorld train scenes."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.select_kl_occworld_expanded_scenes import (
    _lidar_window_error,
    _scene_groups,
    _timing_error,
)
from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _resolve_path,
)


def spread_scene_references(candidates: Sequence[int], required: Sequence[int],
                            count: int, min_separation: int) -> List[int]:
    """Keep required references, then greedily maximize temporal coverage."""
    candidates = sorted(set(int(value) for value in candidates))
    selected = sorted(set(int(value) for value in required))
    if count < len(selected):
        raise ValueError('Requested count is smaller than required references')
    if min_separation < 0:
        raise ValueError('Minimum separation must be non-negative')
    missing = sorted(set(selected).difference(candidates))
    if missing:
        raise ValueError(f'Required references are not eligible: {missing}')
    while len(selected) < count:
        available = [
            candidate for candidate in candidates
            if candidate not in selected and all(
                abs(candidate - other) >= min_separation
                for other in selected)
        ]
        if not available:
            raise ValueError(
                f'Only {len(selected)} references satisfy separation '
                f'{min_separation}')
        candidate = max(
            available,
            key=lambda value: (
                min(abs(value - other) for other in selected), -value))
        selected.append(candidate)
    return sorted(selected)


def _load_manifest(path: Path) -> dict:
    with path.open() as source:
        manifest = json.load(source)
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported OccWorld manifest version')
    if not isinstance(manifest.get('splits'), dict):
        raise ValueError('OccWorld manifest has no splits')
    return manifest


def _sample_record(info: Mapping, reference_index: int,
                   selection_source: str) -> dict:
    return {
        'reference_index': int(reference_index),
        'scene_token': str(info['scene_token']),
        'sample_token': str(info.get('token', '')),
        'timestamp': float(info['timestamp']),
        'selection_source': selection_source,
    }


def select_dense_train_manifest(
        infos: Sequence[Mapping], base_manifest: Mapping,
        references_per_scene: int, min_separation: int,
        history_offsets: Sequence[int], future_offsets: Sequence[int],
        expected_step_s: float, max_time_error_s: float,
        expected_sensor_count: int, check_lidar_files: bool) -> dict:
    """Densify train references while leaving holdout records untouched."""
    if references_per_scene < 1:
        raise ValueError('references_per_scene must be positive')
    train_records = list(base_manifest['splits']['train'])
    base_by_scene: Dict[str, List[int]] = {}
    for record in train_records:
        base_by_scene.setdefault(str(record['scene_token']), []).append(
            int(record['reference_index']))
    holdout_scenes = {
        str(record['scene_token'])
        for split in ('validation', 'test')
        for record in base_manifest['splits'][split]
    }
    overlap = sorted(set(base_by_scene).intersection(holdout_scenes))
    if overlap:
        raise ValueError(f'Train and holdout scenes overlap: {overlap}')
    scene_indices = dict(_scene_groups(infos))
    required_offsets = sorted(set(
        int(value) for value in (*history_offsets, *future_offsets)))
    extrinsics_cache = {}
    dense_records = []
    scene_rows = []
    for scene in sorted(
            base_by_scene,
            key=lambda value: min(base_by_scene[value])):
        if scene not in scene_indices:
            raise ValueError(f'Train scene {scene} is absent from annotation')
        eligible = []
        timing_rejected = 0
        lidar_rejected = 0
        for reference_index in scene_indices[scene]:
            reason = _timing_error(
                infos, reference_index, required_offsets,
                expected_step_s, max_time_error_s)
            if reason:
                timing_rejected += 1
                continue
            if check_lidar_files:
                reason = _lidar_window_error(
                    infos, reference_index, required_offsets,
                    expected_sensor_count, extrinsics_cache)
                if reason:
                    lidar_rejected += 1
                    continue
            eligible.append(reference_index)
        selected_count = references_per_scene
        while True:
            try:
                selected = spread_scene_references(
                    eligible, base_by_scene[scene], selected_count,
                    min_separation)
                break
            except ValueError:
                selected_count -= 1
                if selected_count < len(base_by_scene[scene]):
                    raise ValueError(
                        f'Base references for scene {scene} are not eligible')
        required_set = set(base_by_scene[scene])
        for reference_index in selected:
            dense_records.append(_sample_record(
                infos[reference_index], reference_index,
                'base_v2_reference' if reference_index in required_set
                else 'dense_train_reference'))
        scene_rows.append({
            'scene_token': scene,
            'first_index': int(scene_indices[scene][0]),
            'last_index': int(scene_indices[scene][-1]),
            'frame_count': len(scene_indices[scene]),
            'eligible_reference_count': len(eligible),
            'timing_rejected_count': timing_rejected,
            'lidar_rejected_count': lidar_rejected,
            'base_reference_indices': sorted(base_by_scene[scene]),
            'selected_reference_indices': selected,
            'selected_reference_count': len(selected),
            'selection_shortfall': references_per_scene - len(selected),
        })
    dense_records.sort(key=lambda record: record['reference_index'])
    if any(record['scene_token'] in holdout_scenes for record in dense_records):
        raise AssertionError('Dense training selection entered a holdout scene')
    return {
        'schema_version': 1,
        'name': 'kl_occworld_dense_train_references_v1',
        'status': 'planned_before_label_generation',
        'strategy': 'fixed_scene_split_with_greedy_temporal_coverage',
        'base_manifest': str(base_manifest.get('name', '')),
        'maximum_references_per_train_scene': references_per_scene,
        'minimum_reference_separation_frames': min_separation,
        'required_offsets': required_offsets,
        'expected_step_s': expected_step_s,
        'max_time_error_s': max_time_error_s,
        'lidar_file_check': bool(check_lidar_files),
        'train_scene_count': len(base_by_scene),
        'train_reference_count': len(dense_records),
        'new_train_reference_count': (
            len(dense_records) - len(train_records)),
        'shortfall_scene_count': sum(
            row['selection_shortfall'] > 0 for row in scene_rows),
        'selected_references_per_scene': {
            str(count): sum(
                row['selected_reference_count'] == count
                for row in scene_rows)
            for count in sorted(set(
                row['selected_reference_count'] for row in scene_rows))
        },
        'split_scene_tokens': base_manifest['split_scene_tokens'],
        'splits': {
            'train': dense_records,
            'validation': base_manifest['splits']['validation'],
            'test': base_manifest['splits']['test'],
        },
        'scene_rows': scene_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', type=Path,
        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument(
        '--base-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/kl_occworld_scene_split_v2.json'))
    parser.add_argument('--references-per-scene', type=int, default=3)
    parser.add_argument('--min-separation-frames', type=int, default=8)
    parser.add_argument('--history-offsets', type=int, nargs='+',
                        default=[-4, -3, -2, -1, 0])
    parser.add_argument('--future-offsets', type=int, nargs='+',
                        default=list(range(9)))
    parser.add_argument('--expected-step-s', type=float, default=0.5)
    parser.add_argument('--max-time-error-s', type=float, default=0.2)
    parser.add_argument('--expected-sensor-count', type=int, default=8)
    parser.add_argument('--skip-lidar-file-check', action='store_true')
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_dense_train3_manifest_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    infos, _ = _load_infos(_resolve_path(str(args.ann_file)))
    base_manifest = _load_manifest(args.base_manifest)
    result = select_dense_train_manifest(
        infos, base_manifest,
        references_per_scene=args.references_per_scene,
        min_separation=args.min_separation_frames,
        history_offsets=args.history_offsets,
        future_offsets=args.future_offsets,
        expected_step_s=args.expected_step_s,
        max_time_error_s=args.max_time_error_s,
        expected_sensor_count=args.expected_sensor_count,
        check_lidar_files=not args.skip_lidar_file_check)
    result['base_manifest_path'] = str(args.base_manifest)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        key: result[key]
        for key in (
            'train_scene_count', 'train_reference_count',
            'new_train_reference_count',
            'maximum_references_per_train_scene',
            'minimum_reference_separation_frames', 'lidar_file_check')
    }, ensure_ascii=False, indent=2))
    print(f'out_file={args.out_file}')


if __name__ == '__main__':
    main()
