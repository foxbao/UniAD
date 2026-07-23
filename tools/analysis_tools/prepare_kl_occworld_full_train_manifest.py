#!/usr/bin/env python
"""Freeze a final holdout and prepare a scene-complete OccWorld train set."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.build_kl_occworld_blind_holdout import (
    load_eligible_records,
)
from tools.analysis_tools.select_kl_occworld_dense_train_references import (
    spread_scene_references,
)
from tools.analysis_tools.select_kl_occworld_expanded_scenes import (
    _evenly_spaced,
    _lidar_window_error,
    _scene_groups,
    _timing_error,
)
from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _resolve_path,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    _validate_fixed_frame_times,
)


def _manifest(path: Path) -> dict:
    with path.open() as source:
        manifest = json.load(source)
    if manifest.get('schema_version') != 1:
        raise ValueError(f'Unsupported manifest version: {path}')
    if not isinstance(manifest.get('splits'), dict):
        raise ValueError(f'Manifest has no splits: {path}')
    return manifest


def _scene_set(manifest: Mapping) -> set:
    return {
        str(record['scene_token'])
        for records in manifest['splits'].values()
        for record in records
    }


def _queue_checker(infos: Sequence[Mapping], queue_length: int,
                   max_time_gap_s: float):
    token_to_index = {
        str(info['token']): index
        for index, info in enumerate(infos) if info.get('token')
    }

    def is_valid(reference_index: int) -> bool:
        current = infos[reference_index]
        scene = current.get('scene_token')
        for _ in range(queue_length - 1):
            previous_index = token_to_index.get(str(current.get('prev', '')))
            if previous_index is None:
                return False
            previous = infos[previous_index]
            if previous.get('scene_token') != scene:
                return False
            if abs(float(current['timestamp']) -
                   float(previous['timestamp'])) > max_time_gap_s:
                return False
            current = previous
        return True

    return is_valid


def partition_scene_records(eligible_records: Sequence[Mapping],
                            v2_manifest: Mapping,
                            blind_manifest: Mapping,
                            final_scene_count: int,
                            final_reference_valid) -> Dict[str, object]:
    """Partition eligible scenes without inspecting labels or predictions."""
    if final_scene_count < 1:
        raise ValueError('final_scene_count must be positive')
    v2_scenes = _scene_set(v2_manifest)
    blind_scenes = _scene_set(blind_manifest)
    overlap = sorted(v2_scenes.intersection(blind_scenes))
    if overlap:
        raise ValueError(f'V2 and blind scene overlap: {overlap}')
    eligible_by_scene = {
        str(record['scene_token']): dict(record)
        for record in eligible_records
    }
    if len(eligible_by_scene) != len(eligible_records):
        raise ValueError('Eligibility records contain duplicate scenes')
    unknown_reserved = sorted(
        (v2_scenes | blind_scenes).difference(eligible_by_scene))
    if unknown_reserved:
        raise ValueError(
            f'Reserved scenes are absent from eligibility audit: '
            f'{unknown_reserved}')

    final_candidates = [
        dict(record) for record in eligible_records
        if str(record['scene_token']) not in v2_scenes | blind_scenes
        and final_reference_valid(int(record['reference_index']))
    ]
    final_records = _evenly_spaced(final_candidates, final_scene_count)
    for record in final_records:
        record['selection_source'] = (
            'fresh_evenly_spaced_before_b15_label_generation')
    final_scenes = {
        str(record['scene_token']) for record in final_records
    }
    reserved = v2_scenes | blind_scenes | final_scenes
    train_scene_records = [
        dict(record) for record in eligible_records
        if str(record['scene_token']) not in reserved
        or str(record['scene_token']) in {
            str(item['scene_token'])
            for item in v2_manifest['splits']['train']
        }
    ]
    train_scenes = {
        str(record['scene_token']) for record in train_scene_records
    }
    if train_scenes.intersection(
            blind_scenes | final_scenes |
            {str(record['scene_token'])
             for split in ('validation', 'test')
             for record in v2_manifest['splits'][split]}):
        raise AssertionError('Full train partition overlaps a holdout')
    return {
        'train_scene_records': train_scene_records,
        'final_records': final_records,
        'v2_scenes': v2_scenes,
        'blind_scenes': blind_scenes,
        'final_candidate_count': len(final_candidates),
    }


def _sample_record(info: Mapping, reference_index: int,
                   selection_source: str) -> dict:
    return {
        'reference_index': int(reference_index),
        'scene_token': str(info['scene_token']),
        'sample_token': str(info.get('token', '')),
        'timestamp': float(info['timestamp']),
        'selection_source': selection_source,
    }


def _existing_preparation_shards(manifest: Mapping,
                                 shard_count: int) -> dict:
    records = manifest.get('splits', {}).get('train', [])
    explicit = {
        str(record['scene_token']): int(record['preparation_shard_index'])
        for record in records
        if record.get('preparation_shard_index') is not None
    }
    if explicit:
        return explicit
    grouped = {}
    for record in records:
        grouped.setdefault(str(record['scene_token']), []).append(
            int(record['reference_index']))
    loads = [0] * shard_count
    result = {}
    for scene, values in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), min(item[1]))):
        shard_index = min(
            range(shard_count), key=lambda index: (loads[index], index))
        result[scene] = shard_index
        loads[shard_index] += len(values)
    return result


def select_full_train_references(
        infos: Sequence[Mapping], train_scene_records: Sequence[Mapping],
        v2_manifest: Mapping, references_per_scene: int,
        min_separation: int, required_offsets: Sequence[int],
        expected_step_s: float, max_time_error_s: float,
        expected_sensor_count: int, check_lidar_files: bool,
        model_queue_valid) -> Dict[str, object]:
    """Select temporally spread references from every usable train scene."""
    scene_indices = dict(_scene_groups(infos))
    v2_train_reference = {
        str(record['scene_token']): int(record['reference_index'])
        for record in v2_manifest['splits']['train']
    }
    extrinsics_cache = {}
    selected_records = []
    scene_rows = []
    excluded_scenes = []
    for scene_record in sorted(
            train_scene_records,
            key=lambda item: int(item['reference_index'])):
        scene = str(scene_record['scene_token'])
        timing_rejected = 0
        lidar_rejected = 0
        queue_rejected = 0
        sequence_timing_rejected = 0
        eligible = []
        for reference_index in scene_indices[scene]:
            if not model_queue_valid(reference_index):
                queue_rejected += 1
                continue
            reason = _timing_error(
                infos, reference_index, required_offsets,
                expected_step_s, max_time_error_s)
            if reason:
                timing_rejected += 1
                continue
            try:
                _validate_fixed_frame_times(
                    infos, reference_index,
                    target_offsets=[0, 1, 2, 3, 4],
                    reveal_offsets=[0, 1, 2, 3, 4],
                    expected_step_s=expected_step_s,
                    max_time_error_s=max_time_error_s)
            except ValueError:
                sequence_timing_rejected += 1
                continue
            if check_lidar_files:
                reason = _lidar_window_error(
                    infos, reference_index, required_offsets,
                    expected_sensor_count, extrinsics_cache)
                if reason:
                    lidar_rejected += 1
                    continue
            eligible.append(reference_index)
        if not eligible:
            excluded_scenes.append({
                'scene_token': scene,
                'audit_reference_index': int(
                    scene_record['reference_index']),
                'reason': 'no_reference_satisfies_full_model_input_contract',
                'timing_rejected_count': timing_rejected,
                'lidar_rejected_count': lidar_rejected,
                'model_queue_rejected_count': queue_rejected,
                'sequence_timing_rejected_count': (
                    sequence_timing_rejected),
            })
            continue

        preferred = v2_train_reference.get(
            scene, int(scene_record['reference_index']))
        preferred_adjusted = preferred not in eligible
        if preferred_adjusted:
            preferred = min(
                eligible,
                key=lambda value: (
                    abs(value - int(scene_record['reference_index'])),
                    value))
        selected_count = min(references_per_scene, len(eligible))
        while True:
            try:
                selected = spread_scene_references(
                    eligible, [preferred], selected_count,
                    min_separation)
                break
            except ValueError:
                selected_count -= 1
                if selected_count < 1:
                    raise AssertionError(
                        f'Unable to retain preferred reference for {scene}')
        for reference_index in selected:
            if reference_index == preferred:
                if scene in v2_train_reference and not preferred_adjusted:
                    source = 'base_v2_train_reference'
                elif preferred_adjusted:
                    source = 'nearest_full_input_contract_reference'
                else:
                    source = 'eligibility_audit_reference'
            else:
                source = 'full_train_temporal_coverage_reference'
            selected_records.append(_sample_record(
                infos[reference_index], reference_index, source))
        scene_rows.append({
            'scene_token': scene,
            'first_index': int(scene_indices[scene][0]),
            'last_index': int(scene_indices[scene][-1]),
            'frame_count': len(scene_indices[scene]),
            'eligible_reference_count': len(eligible),
            'timing_rejected_count': timing_rejected,
            'lidar_rejected_count': lidar_rejected,
            'model_queue_rejected_count': queue_rejected,
            'sequence_timing_rejected_count': sequence_timing_rejected,
            'audit_reference_index': int(
                scene_record['reference_index']),
            'preferred_reference_adjusted': preferred_adjusted,
            'selected_reference_indices': selected,
            'selected_reference_count': len(selected),
            'selection_shortfall': references_per_scene - len(selected),
        })
    selected_records.sort(key=lambda item: item['reference_index'])
    return {
        'records': selected_records,
        'scene_rows': scene_rows,
        'excluded_scenes': excluded_scenes,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', type=Path,
        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument(
        '--eligibility-audit', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_expanded70_selection/'
            'scene_eligibility.csv'))
    parser.add_argument(
        '--v2-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v2.json'))
    parser.add_argument(
        '--blind-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_blind_holdout_v1.json'))
    parser.add_argument('--final-scene-count', type=int, default=30)
    parser.add_argument('--references-per-scene', type=int, default=3)
    parser.add_argument('--min-separation-frames', type=int, default=8)
    parser.add_argument('--history-offsets', type=int, nargs='+',
                        default=[-4, -3, -2, -1, 0])
    parser.add_argument('--future-offsets', type=int, nargs='+',
                        default=list(range(9)))
    parser.add_argument('--expected-step-s', type=float, default=0.5)
    parser.add_argument('--max-time-error-s', type=float, default=0.2)
    parser.add_argument('--model-queue-length', type=int, default=5)
    parser.add_argument('--model-queue-max-gap-s', type=float, default=1.0)
    parser.add_argument('--expected-sensor-count', type=int, default=8)
    parser.add_argument('--skip-lidar-file-check', action='store_true')
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_full_train3_final30_manifest_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.references_per_scene < 1:
        raise ValueError('references-per-scene must be positive')
    prior_manifest = None
    prior_shards = {}
    if args.out_file.exists():
        prior_manifest = _manifest(args.out_file)
        prior_shards = _existing_preparation_shards(
            prior_manifest, shard_count=16)
    infos, _ = _load_infos(_resolve_path(str(args.ann_file)))
    v2_manifest = _manifest(args.v2_manifest)
    blind_manifest = _manifest(args.blind_manifest)
    eligible_records = load_eligible_records(
        args.eligibility_audit, infos)
    model_queue_valid = _queue_checker(
        infos, args.model_queue_length,
        args.model_queue_max_gap_s)
    partition = partition_scene_records(
        eligible_records, v2_manifest, blind_manifest,
        args.final_scene_count, model_queue_valid)
    required_offsets = sorted(set(
        int(value) for value in (
            *args.history_offsets, *args.future_offsets)))
    selection = select_full_train_references(
        infos, partition['train_scene_records'], v2_manifest,
        references_per_scene=args.references_per_scene,
        min_separation=args.min_separation_frames,
        required_offsets=required_offsets,
        expected_step_s=args.expected_step_s,
        max_time_error_s=args.max_time_error_s,
        expected_sensor_count=args.expected_sensor_count,
        check_lidar_files=not args.skip_lidar_file_check,
        model_queue_valid=model_queue_valid)

    train_records = selection['records']
    final_records = partition['final_records']
    if prior_shards:
        for record in train_records:
            record['preparation_shard_index'] = prior_shards[
                str(record['scene_token'])]
    for record in final_records:
        _validate_fixed_frame_times(
            infos, int(record['reference_index']),
            target_offsets=[0, 1, 2, 3, 4],
            reveal_offsets=[0, 1, 2, 3, 4],
            expected_step_s=args.expected_step_s,
            max_time_error_s=args.max_time_error_s)
    retained_train_scenes = {
        str(record['scene_token']) for record in train_records
    }
    splits = {
        'train': train_records,
        'validation': v2_manifest['splits']['validation'],
        'test': v2_manifest['splits']['test'],
        'blind_consumed': blind_manifest['splits']['blind'],
        'final_holdout': final_records,
    }
    scene_sets = {
        name: {str(record['scene_token']) for record in records}
        for name, records in splits.items()
    }
    names = list(scene_sets)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            overlap = scene_sets[name].intersection(scene_sets[other])
            if overlap:
                raise AssertionError(
                    f'Scene overlap between {name} and {other}: '
                    f'{sorted(overlap)}')

    selected_per_scene = {}
    for row in selection['scene_rows']:
        count = str(row['selected_reference_count'])
        selected_per_scene[count] = selected_per_scene.get(count, 0) + 1
    revisions = []
    if prior_manifest is not None:
        old_by_scene = {}
        new_by_scene = {}
        for record in prior_manifest['splits']['train']:
            old_by_scene.setdefault(str(record['scene_token']), []).append(
                int(record['reference_index']))
        for record in train_records:
            new_by_scene.setdefault(str(record['scene_token']), []).append(
                int(record['reference_index']))
        for scene in sorted(old_by_scene):
            old = sorted(old_by_scene[scene])
            new = sorted(new_by_scene[scene])
            if old != new:
                revisions.append({
                    'scene_token': scene,
                    'old_reference_indices': old,
                    'new_reference_indices': new,
                    'reason': (
                        'strict_target_reveal_interval_contract'),
                    'model_predictions_inspected': False,
                    'final_holdout_changed': False,
                })
    manifest = {
        'schema_version': 1,
        'name': 'kl_occworld_full_train3_final30_v1',
        'status': 'frozen_before_full_gt_generation',
        'strategy': (
            'all_eligible_train_scenes_with_three_spread_references_'
            'after_freezing_final_holdout'),
        'annotation_file': str(args.ann_file),
        'eligibility_audit': str(args.eligibility_audit),
        'v2_manifest': str(args.v2_manifest),
        'consumed_blind_manifest': str(args.blind_manifest),
        'eligible_scene_count': len(eligible_records),
        'train_candidate_scene_count': len(
            partition['train_scene_records']),
        'train_scene_count': len(retained_train_scenes),
        'train_reference_count': len(train_records),
        'train_input_contract_exclusion_count': len(
            selection['excluded_scenes']),
        'maximum_references_per_train_scene': (
            args.references_per_scene),
        'minimum_reference_separation_frames': (
            args.min_separation_frames),
        'selected_references_per_scene': selected_per_scene,
        'required_offsets': required_offsets,
        'expected_step_s': args.expected_step_s,
        'max_time_error_s': args.max_time_error_s,
        'model_queue_length': args.model_queue_length,
        'model_queue_max_gap_s': args.model_queue_max_gap_s,
        'strict_sequence_timing_contract': {
            'target_offsets': [0, 1, 2, 3, 4],
            'reveal_offsets': [0, 1, 2, 3, 4],
            'enabled': True,
        },
        'preparation_shard_count': 16,
        'sequence_contract_revisions': revisions,
        'lidar_file_check': not args.skip_lidar_file_check,
        'expected_sensor_count': args.expected_sensor_count,
        'final_holdout_scene_count': len(final_records),
        'final_holdout_candidate_count': partition[
            'final_candidate_count'],
        'final_holdout_protocol': {
            'selection_uses_model_output': False,
            'selection_uses_occworld_labels': False,
            'labels_generated_before_selection': False,
            'model_predictions_inspected': False,
            'threshold_retuning_allowed': False,
            'single_final_evaluation_only': True,
        },
        'roots': {
            'temporal': (
                'outputs/patent_2026_occ/'
                'occworld_temporal_full_train_v1'),
            'occlusion': (
                'outputs/patent_2026_occ/'
                'occworld_occlusion_full_train_v1'),
            'dual': (
                'outputs/patent_2026_occ/'
                'occworld_dual_full_train_v1'),
            'dual_cross_scene': (
                'outputs/patent_2026_occ/'
                'occworld_dual_cross_scene_full_train_v1'),
            'observation_cache': (
                'outputs/patent_2026_occ/'
                'occworld_observation_cache_full_train_v1'),
            'sequence': (
                'outputs/patent_2026_occ/'
                'occworld_sequence_full_train_v1'),
            'history': (
                'outputs/patent_2026_occ/'
                'occworld_history_full_train_v1'),
            'sequence_audit': (
                'outputs/patent_2026_occ/'
                'occworld_sequence_full_train_v1_audit'),
            'history_audit': (
                'outputs/patent_2026_occ/'
                'occworld_history_full_train_v1_audit'),
        },
        'split_scene_tokens': {
            name: [str(record['scene_token']) for record in records]
            for name, records in splits.items()
        },
        'splits': splits,
        'train_scene_rows': selection['scene_rows'],
        'train_input_contract_exclusions': selection[
            'excluded_scenes'],
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'status': manifest['status'],
        'eligible_scene_count': manifest['eligible_scene_count'],
        'train_candidate_scene_count': manifest[
            'train_candidate_scene_count'],
        'train_scene_count': manifest['train_scene_count'],
        'train_reference_count': manifest['train_reference_count'],
        'train_input_contract_exclusion_count': manifest[
            'train_input_contract_exclusion_count'],
        'selected_references_per_scene': manifest[
            'selected_references_per_scene'],
        'final_holdout_scene_count': manifest[
            'final_holdout_scene_count'],
        'out_file': str(args.out_file),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
