#!/usr/bin/env python
"""Select deterministic scene-isolated references for expanded OccWorld data."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data_converter.generate_kl_occworld_labels import (
    _find_extrinsics_path,
    _load_extrinsics,
    _load_infos,
    _resolve_path,
)


def _evenly_spaced(items: Sequence[dict], count: int) -> List[dict]:
    """Return ``count`` ordered items spread across the full input range."""
    if count < 0 or count > len(items):
        raise ValueError(
            f'Cannot select {count} items from a pool of {len(items)}')
    if count == 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    denominator = count - 1
    indices = [
        (position * (len(items) - 1) + denominator // 2) // denominator
        for position in range(count)
    ]
    if len(set(indices)) != count:
        raise AssertionError(f'Even selection produced duplicates: {indices}')
    return [items[index] for index in indices]


def _scene_groups(infos: Sequence[Mapping]) -> List[Tuple[str, List[int]]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, info in enumerate(infos):
        scene = str(info.get('scene_token', ''))
        if not scene:
            raise ValueError(f'Frame {index} has no scene_token')
        groups[scene].append(index)
    ordered = sorted(groups.items(), key=lambda item: item[1][0])
    for scene, indices in ordered:
        expected = list(range(indices[0], indices[-1] + 1))
        if indices != expected:
            raise ValueError(
                f'Scene {scene} is not contiguous in the annotation')
    return ordered


def _timing_error(
        infos: Sequence[Mapping], reference_index: int,
        required_offsets: Sequence[int], expected_step_s: float,
        max_time_error_s: float) -> str:
    scene = infos[reference_index].get('scene_token')
    reference_time = float(infos[reference_index]['timestamp'])
    for offset in required_offsets:
        source_index = reference_index + int(offset)
        if not 0 <= source_index < len(infos):
            return f'offset {offset} is outside the annotation'
        if infos[source_index].get('scene_token') != scene:
            return f'offset {offset} crosses the scene boundary'
        actual_time = float(infos[source_index]['timestamp']) - reference_time
        expected_time = float(offset) * expected_step_s
        if abs(actual_time - expected_time) > max_time_error_s:
            return (
                f'offset {offset} has time {actual_time:.3f}s, expected '
                f'{expected_time:.3f}s')
    return ''


def _lidar_window_error(
        infos: Sequence[Mapping], reference_index: int,
        required_offsets: Sequence[int], expected_sensor_count: int,
        extrinsics_cache: Dict[Path, set]) -> str:
    for offset in required_offsets:
        frame_index = reference_index + int(offset)
        entries = infos[frame_index].get('sync_info', {}).get('lidars', {})
        valid_entries = [
            (name, entry) for name, entry in sorted(entries.items())
            if entry.get('valid', False) and entry.get('path')
        ]
        if len(valid_entries) != expected_sensor_count:
            return (
                f'frame {frame_index} has {len(valid_entries)} valid LiDARs, '
                f'expected {expected_sensor_count}')
        for sensor_name, entry in valid_entries:
            try:
                point_path = _resolve_path(entry['path'])
                extrinsics_path = _find_extrinsics_path(point_path)
                if extrinsics_path not in extrinsics_cache:
                    extrinsics_cache[extrinsics_path] = set(
                        _load_extrinsics(extrinsics_path))
                if sensor_name not in extrinsics_cache[extrinsics_path]:
                    return (
                        f'{sensor_name} has no transform in '
                        f'{extrinsics_path}')
            except (FileNotFoundError, KeyError, ValueError) as error:
                return f'frame {frame_index} {sensor_name}: {error}'
    return ''


def audit_scene_references(
        infos: Sequence[Mapping], required_offsets: Sequence[int],
        expected_step_s: float, max_time_error_s: float,
        expected_sensor_count: int = 8,
        check_lidar_files: bool = True) -> Tuple[List[dict], List[dict]]:
    """Find one center-biased valid reference in every eligible scene."""
    required_offsets = sorted(set(int(value) for value in required_offsets))
    extrinsics_cache: Dict[Path, set] = {}
    eligible = []
    audit_rows = []
    for scene, scene_indices in _scene_groups(infos):
        midpoint = (scene_indices[0] + scene_indices[-1]) / 2.0
        candidates = sorted(
            scene_indices,
            key=lambda index: (abs(index - midpoint), index))
        selected = None
        timing_failures = 0
        lidar_failures = 0
        last_reason = ''
        for reference_index in candidates:
            reason = _timing_error(
                infos, reference_index, required_offsets,
                expected_step_s, max_time_error_s)
            if reason:
                timing_failures += 1
                last_reason = reason
                continue
            if check_lidar_files:
                reason = _lidar_window_error(
                    infos, reference_index, required_offsets,
                    expected_sensor_count, extrinsics_cache)
                if reason:
                    lidar_failures += 1
                    last_reason = reason
                    continue
            selected = reference_index
            break

        status = 'eligible' if selected is not None else 'ineligible'
        row = {
            'scene_token': scene,
            'first_index': scene_indices[0],
            'last_index': scene_indices[-1],
            'frame_count': len(scene_indices),
            'status': status,
            'selected_reference_index': (
                selected if selected is not None else ''),
            'timing_candidate_failures': timing_failures,
            'lidar_candidate_failures': lidar_failures,
            'reason': '' if selected is not None else last_reason,
        }
        audit_rows.append(row)
        if selected is not None:
            info = infos[selected]
            eligible.append({
                'reference_index': selected,
                'scene_token': scene,
                'sample_token': str(info.get('token', '')),
                'timestamp': float(info['timestamp']),
            })
    return eligible, audit_rows


def select_split_records(
        eligible_records: Sequence[dict], reused_records: Iterable[dict],
        train_scene_count: int, validation_scene_count: int,
        test_scene_count: int) -> Dict[str, List[dict]]:
    """Keep reused scenes in train and select all holdouts from fresh scenes."""
    reused_by_scene = {
        str(record['scene_token']): dict(record)
        for record in reused_records
    }
    if len(reused_by_scene) > train_scene_count:
        raise ValueError(
            f'{len(reused_by_scene)} reused scenes exceed the requested '
            f'{train_scene_count} training scenes')
    eligible_by_scene = {
        str(record['scene_token']): dict(record)
        for record in eligible_records
    }
    missing_reused = sorted(set(reused_by_scene).difference(eligible_by_scene))
    if missing_reused:
        raise ValueError(f'Reused scenes are no longer eligible: {missing_reused}')

    fresh = [
        record for record in eligible_records
        if str(record['scene_token']) not in reused_by_scene
    ]
    fresh_count = (
        train_scene_count - len(reused_by_scene) +
        validation_scene_count + test_scene_count)
    selected_fresh = _evenly_spaced(fresh, fresh_count)
    fresh_train_count = train_scene_count - len(reused_by_scene)
    validation_start = fresh_train_count
    test_start = validation_start + validation_scene_count

    reused_train = []
    for scene, old_record in reused_by_scene.items():
        current = dict(eligible_by_scene[scene])
        current['reference_index'] = int(old_record['reference_index'])
        current['sample_token'] = str(old_record.get(
            'sample_token', current.get('sample_token', '')))
        current['timestamp'] = float(old_record.get(
            'timestamp', current['timestamp']))
        current['selection_source'] = 'reused_v1_train'
        reused_train.append(current)
    fresh_train = [dict(record) for record in selected_fresh[:validation_start]]
    validation = [dict(record) for record in selected_fresh[
        validation_start:test_start]]
    test = [dict(record) for record in selected_fresh[test_start:]]
    for records in (fresh_train, validation, test):
        for record in records:
            record['selection_source'] = 'fresh_evenly_spaced'

    splits = {
        'train': sorted(
            reused_train + fresh_train,
            key=lambda record: int(record['reference_index'])),
        'validation': validation,
        'test': test,
    }
    scene_sets = {
        name: {str(record['scene_token']) for record in records}
        for name, records in splits.items()
    }
    names = tuple(splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            if not scene_sets[left].isdisjoint(scene_sets[right]):
                raise AssertionError(f'{left} and {right} share a scene')
    return splits


def _load_reused_records(path: Path) -> List[dict]:
    if path is None:
        return []
    with path.open() as source:
        manifest = json.load(source)
    return [
        record
        for split_records in manifest['splits'].values()
        for record in split_records
    ]


def _write_csv(path: Path, rows: List[dict]):
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', type=Path,
        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument(
        '--reuse-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/kl_occworld_scene_split_v1.json'))
    parser.add_argument('--train-scene-count', type=int, default=50)
    parser.add_argument('--validation-scene-count', type=int, default=10)
    parser.add_argument('--test-scene-count', type=int, default=10)
    parser.add_argument('--history-offsets', type=int, nargs='+',
                        default=[-4, -3, -2, -1, 0])
    parser.add_argument('--future-offsets', type=int, nargs='+',
                        default=list(range(9)))
    parser.add_argument('--expected-step-s', type=float, default=0.5)
    parser.add_argument('--max-time-error-s', type=float, default=0.2)
    parser.add_argument('--expected-sensor-count', type=int, default=8)
    parser.add_argument('--skip-lidar-file-check', action='store_true')
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_expanded70_selection'))
    parser.add_argument(
        '--out-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v2_planned.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    infos, _ = _load_infos(_resolve_path(str(args.ann_file)))
    required_offsets = sorted(set(
        args.history_offsets + args.future_offsets))
    eligible, audit_rows = audit_scene_references(
        infos, required_offsets,
        expected_step_s=args.expected_step_s,
        max_time_error_s=args.max_time_error_s,
        expected_sensor_count=args.expected_sensor_count,
        check_lidar_files=not args.skip_lidar_file_check)
    reused = _load_reused_records(args.reuse_manifest)
    splits = select_split_records(
        eligible, reused,
        train_scene_count=args.train_scene_count,
        validation_scene_count=args.validation_scene_count,
        test_scene_count=args.test_scene_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / 'scene_eligibility.csv', audit_rows)
    selected_records = [
        record for records in splits.values() for record in records
    ]
    selected_indices = sorted(
        int(record['reference_index']) for record in selected_records)
    summary = {
        'annotation_frame_count': len(infos),
        'annotation_scene_count': len(audit_rows),
        'eligible_scene_count': len(eligible),
        'ineligible_scene_count': len(audit_rows) - len(eligible),
        'selected_scene_count': len(selected_records),
        'split_counts': {name: len(records) for name, records in splits.items()},
        'reused_v1_train_scene_count': sum(
            record['selection_source'] == 'reused_v1_train'
            for record in splits['train']),
        'required_offsets': required_offsets,
        'expected_step_s': args.expected_step_s,
        'max_time_error_s': args.max_time_error_s,
        'lidar_file_check': not args.skip_lidar_file_check,
        'expected_sensor_count': args.expected_sensor_count,
        'selected_reference_indices': selected_indices,
    }
    with (args.out_dir / 'summary.json').open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')

    manifest = {
        'schema_version': 1,
        'name': 'kl_occworld_scene_split_v2_planned',
        'status': 'planned_before_label_generation',
        'strategy': (
            'reuse_v1_scenes_in_train_then_evenly_sample_fresh_scenes'),
        'annotation_file': str(args.ann_file),
        'selection_summary': str(args.out_dir / 'summary.json'),
        'reuse_manifest': str(args.reuse_manifest),
        'sequence_root': (
            'outputs/patent_2026_occ/occworld_sequence_expanded70'),
        'history_root': (
            'outputs/patent_2026_occ/occworld_history_expanded70'),
        'split_scene_tokens': {
            name: [record['scene_token'] for record in records]
            for name, records in splits.items()
        },
        'splits': splits,
        **summary,
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open('w') as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'manifest={args.out_manifest}')


if __name__ == '__main__':
    main()
