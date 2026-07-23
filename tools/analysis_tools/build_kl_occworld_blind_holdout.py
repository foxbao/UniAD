#!/usr/bin/env python
"""Freeze a fresh scene-level OccWorld blind holdout before inference."""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.select_kl_occworld_expanded_scenes import (
    _evenly_spaced,
)
from tools.analysis_tools.finalize_kl_occworld_scene_split import (
    _artifact_index,
    _class_statistics,
)
from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _resolve_path,
)


def _excluded_scenes(manifest: Mapping) -> set:
    splits = manifest.get('splits')
    if not isinstance(splits, dict) or not splits:
        raise ValueError('Excluded manifest has no splits')
    return {
        str(record['scene_token'])
        for records in splits.values() for record in records
    }


def load_eligible_records(audit_path: Path,
                          infos: Sequence[Mapping]) -> list:
    """Load the already completed LiDAR/timing audit as scene records."""
    records = []
    with audit_path.open(newline='') as source:
        for row in csv.DictReader(source):
            if row['status'] != 'eligible':
                continue
            reference_index = int(row['selected_reference_index'])
            info = infos[reference_index]
            scene_token = str(row['scene_token'])
            if str(info.get('scene_token')) != scene_token:
                raise ValueError(
                    f'Audit scene mismatch at reference {reference_index}')
            records.append({
                'reference_index': reference_index,
                'scene_token': scene_token,
                'sample_token': str(info.get('token', '')),
                'timestamp': float(info['timestamp']),
            })
    if not records:
        raise ValueError(f'No eligible scenes in {audit_path}')
    records.sort(key=lambda record: int(record['reference_index']))
    if len({record['scene_token'] for record in records}) != len(records):
        raise ValueError('Eligibility audit contains duplicate scenes')
    return records


def select_blind_records(eligible_records: Sequence[Mapping],
                         excluded_manifest: Mapping,
                         scene_count: int) -> list:
    """Select ordered, evenly spaced records outside every prior split."""
    excluded = _excluded_scenes(excluded_manifest)
    fresh = [
        dict(record) for record in eligible_records
        if str(record['scene_token']) not in excluded
    ]
    selected = _evenly_spaced(fresh, scene_count)
    for record in selected:
        record['selection_source'] = (
            'fresh_evenly_spaced_before_blind_inference')
    if not excluded.isdisjoint(
            {str(record['scene_token']) for record in selected}):
        raise AssertionError('Blind holdout overlaps a previous split')
    return selected


def _has_model_queue(infos: Sequence[Mapping], reference_index: int,
                     queue_length: int = 5,
                     max_time_gap: float = 1.0) -> bool:
    token_to_index = {
        info.get('token'): index
        for index, info in enumerate(infos) if info.get('token')
    }
    current = infos[reference_index]
    scene = current.get('scene_token')
    for _ in range(queue_length - 1):
        previous_index = token_to_index.get(current.get('prev'))
        if previous_index is None:
            return False
        previous = infos[previous_index]
        if previous.get('scene_token') != scene:
            return False
        if abs(float(current['timestamp']) -
               float(previous['timestamp'])) > max_time_gap:
            return False
        current = previous
    return True


def replace_pre_inference_invalid_record(
        manifest: Mapping, infos: Sequence[Mapping],
        excluded_manifest: Mapping, invalid_reference: int,
        replacement_reference: int, reason: str) -> dict:
    """Record a model-input exclusion and deterministic replacement."""
    result = dict(manifest)
    splits = dict(result['splits'])
    records = [dict(record) for record in splits['blind']]
    positions = [
        position for position, record in enumerate(records)
        if int(record['reference_index']) == invalid_reference
    ]
    if len(positions) != 1:
        raise ValueError(
            f'Expected one blind reference {invalid_reference}')
    if not _has_model_queue(infos, replacement_reference):
        raise ValueError(
            f'Replacement {replacement_reference} has no model queue')
    replacement_info = infos[replacement_reference]
    replacement_scene = str(replacement_info.get('scene_token'))
    unavailable_scenes = _excluded_scenes(excluded_manifest) | {
        str(record['scene_token']) for record in records
    }
    if replacement_scene in unavailable_scenes:
        raise ValueError(
            f'Replacement scene {replacement_scene} is not fresh')
    invalid_record = records[positions[0]]
    replacement_record = {
        'reference_index': int(replacement_reference),
        'scene_token': replacement_scene,
        'sample_token': str(replacement_info.get('token', '')),
        'timestamp': float(replacement_info['timestamp']),
        'selection_source': (
            'nearest_later_fresh_scene_with_valid_model_queue_'
            'before_inference'),
    }
    records[positions[0]] = replacement_record
    splits['blind'] = records
    result['splits'] = splits
    result['split_scene_tokens'] = {
        'blind': [record['scene_token'] for record in records],
    }
    result['selected_reference_indices'] = [
        int(record['reference_index']) for record in records
    ]
    result['status'] = 'frozen_after_input_contract_audit_before_inference'
    exclusions = list(result.get('pre_inference_exclusions', []))
    exclusions.append({
        'excluded_record': invalid_record,
        'reason': str(reason),
        'replacement_record': replacement_record,
        'model_predictions_inspected_before_replacement': False,
    })
    result['pre_inference_exclusions'] = exclusions
    for key in (
            'sequence_label_count', 'history_queue_count',
            'artifact_validation', 'blind_statistics'):
        result.pop(key, None)
    return result


def finalize_blind_manifest(manifest: Mapping,
                            sequence_root: Path,
                            history_root: Path) -> dict:
    """Validate exact blind artifacts without changing frozen selection."""
    records = manifest.get('splits', {}).get('blind')
    if not records:
        raise ValueError('Blind manifest has no blind records')
    expected = {int(record['reference_index']) for record in records}
    sequence_mapping = _artifact_index(
        sequence_root, 'occworld_sequence', 'reference_index')
    history_mapping = _artifact_index(
        history_root, 'occworld_history', 'reference_index')
    for name, mapping in (
            ('sequence', sequence_mapping), ('history', history_mapping)):
        actual = set(mapping)
        if actual != expected:
            raise ValueError(
                f'{name} reference mismatch: missing='
                f'{sorted(expected - actual)}, extra='
                f'{sorted(actual - expected)}')

    states = []
    valid = []
    max_future_time_error = 0.0
    max_history_time_error = 0.0
    for reference_index in sorted(expected):
        with np.load(
                sequence_mapping[reference_index],
                allow_pickle=False) as sequence:
            target_state = np.array(
                sequence['world_target_state_3d'], copy=True)
            target_valid = np.asarray(
                sequence['world_target_valid_3d'], dtype=np.bool_)
            if target_state.shape != (5, 10, 120, 160):
                raise ValueError(
                    f'Invalid sequence shape for {reference_index}: '
                    f'{target_state.shape}')
            direct_t0 = np.array(
                sequence['direct_observation_state_3d'][0], copy=True)
            target_times = np.asarray(
                sequence['target_times_s'], dtype=np.float64)
            max_future_time_error = max(
                max_future_time_error,
                float(np.max(np.abs(
                    target_times - np.arange(5) * 0.5))))
            states.append(target_state)
            valid.append(target_valid)
        with np.load(
                history_mapping[reference_index],
                allow_pickle=False) as history:
            history_state = history['history_observation_state_3d']
            if history_state.shape != (5, 10, 120, 160):
                raise ValueError(
                    f'Invalid history shape for {reference_index}: '
                    f'{history_state.shape}')
            np.testing.assert_array_equal(history_state[-1], direct_t0)
            history_times = np.asarray(
                history['history_times_s'], dtype=np.float64)
            max_history_time_error = max(
                max_history_time_error,
                float(np.max(np.abs(
                    history_times - np.arange(-4, 1) * 0.5))))

    result = dict(manifest)
    result.update({
        'status': 'ready_for_single_blind_inference',
        'sequence_root': str(sequence_root),
        'history_root': str(history_root),
        'sequence_label_count': len(sequence_mapping),
        'history_queue_count': len(history_mapping),
        'artifact_validation': {
            'reference_sets_exactly_equal': True,
            'sequence_shape': [5, 10, 120, 160],
            'history_shape': [5, 10, 120, 160],
            'history_last_equals_direct_t0': True,
            'max_future_time_error_s': max_future_time_error,
            'max_history_time_error_s': max_history_time_error,
        },
        'blind_statistics': _class_statistics(
            np.stack(states), np.stack(valid)),
    })
    return result


def mark_blind_evaluated(manifest: Mapping, prediction_root: Path,
                         raw_report_path: Path,
                         physical_report_path: Path) -> dict:
    """Seal one frozen blind evaluation and reject protocol drift."""
    expected = [
        int(record['reference_index'])
        for record in manifest['splits']['blind']
    ]
    prediction_epochs = {}
    for path in sorted(prediction_root.glob('*/*__occworld_prediction.npz')):
        with np.load(path, allow_pickle=False) as prediction:
            reference = int(prediction['reference_index'])
            prediction_epochs[reference] = int(
                prediction['checkpoint_epoch'])
    if sorted(prediction_epochs) != sorted(expected):
        raise ValueError('Blind predictions do not match frozen references')
    if set(prediction_epochs.values()) != {10}:
        raise ValueError('Blind predictions are not all from epoch 10')
    with raw_report_path.open() as source:
        raw_report = json.load(source)
    with physical_report_path.open() as source:
        physical_report = json.load(source)
    for name, report in (
            ('raw', raw_report), ('physical', physical_report)):
        if report.get('split') != 'blind':
            raise ValueError(f'{name} report is not a blind evaluation')
        if report.get('reference_indices') != expected:
            raise ValueError(
                f'{name} report references differ from the manifest')
        if float(report.get('visibility_threshold')) != 0.6:
            raise ValueError(
                f'{name} report did not use visibility threshold 0.6')
    if raw_report.get('physical_flow_fusion_applied'):
        raise ValueError('Raw report unexpectedly applies flow fusion')
    if not physical_report.get('physical_flow_fusion_applied'):
        raise ValueError('Physical report did not apply flow fusion')
    if float(physical_report.get('flow_fusion_threshold')) != 0.7:
        raise ValueError('Physical report did not use flow threshold 0.7')
    result = dict(manifest)
    result['status'] = 'blind_evaluated_once_no_retuning_allowed'
    result['blind_evaluation'] = {
        'completed_on': date.today().isoformat(),
        'checkpoint_epoch': 10,
        'visibility_threshold': 0.6,
        'physical_flow_fusion_threshold': 0.7,
        'prediction_root': str(prediction_root),
        'raw_report': str(raw_report_path),
        'physical_report': str(physical_report_path),
        'threshold_scan_performed': False,
        'further_tuning_on_this_holdout_allowed': False,
    }
    return result


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
        '--exclude-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_scene_split_v2.json'))
    parser.add_argument('--scene-count', type=int, default=20)
    parser.add_argument(
        '--sequence-root',
        default='outputs/patent_2026_occ/occworld_sequence_blind20')
    parser.add_argument(
        '--history-root',
        default='outputs/patent_2026_occ/occworld_history_blind20')
    parser.add_argument(
        '--out-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_blind_holdout_v1.json'))
    parser.add_argument(
        '--finalize', action='store_true',
        help='Validate generated sequence/history artifacts in-place.')
    parser.add_argument(
        '--replace-reference', type=int, nargs=2,
        metavar=('INVALID', 'REPLACEMENT'),
        help='Record one pre-inference input-contract replacement.')
    parser.add_argument(
        '--replacement-reason', default='',
        help='Required explanation used with --replace-reference.')
    parser.add_argument(
        '--mark-evaluated', action='store_true',
        help='Seal the single frozen blind inference and reports.')
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_world_only_anchor_history_flow_'
            'epoch10_blind20/blind/epoch_010'))
    parser.add_argument(
        '--raw-report', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_anchor_history_flow_epoch10_blind20_raw_v1.json'))
    parser.add_argument(
        '--physical-report', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_anchor_history_flow_epoch10_blind20_'
            'physical_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scene_count < 1:
        raise ValueError('scene-count must be positive')
    if args.finalize:
        with args.out_file.open() as source:
            manifest = json.load(source)
        manifest = finalize_blind_manifest(
            manifest, Path(args.sequence_root), Path(args.history_root))
        with args.out_file.open('w') as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write('\n')
        print(json.dumps({
            'name': manifest['name'],
            'status': manifest['status'],
            'blind_scene_count': manifest['blind_scene_count'],
            'sequence_label_count': manifest['sequence_label_count'],
            'history_queue_count': manifest['history_queue_count'],
            'artifact_validation': manifest['artifact_validation'],
            'blind_statistics': manifest['blind_statistics'],
            'out_file': str(args.out_file),
        }, ensure_ascii=False, indent=2))
        return
    if args.mark_evaluated:
        with args.out_file.open() as source:
            manifest = json.load(source)
        manifest = mark_blind_evaluated(
            manifest, args.prediction_root,
            args.raw_report, args.physical_report)
        with args.out_file.open('w') as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write('\n')
        print(json.dumps({
            'name': manifest['name'],
            'status': manifest['status'],
            'blind_evaluation': manifest['blind_evaluation'],
            'out_file': str(args.out_file),
        }, ensure_ascii=False, indent=2))
        return
    infos, _ = _load_infos(_resolve_path(str(args.ann_file)))
    if args.replace_reference:
        if not args.replacement_reason:
            raise ValueError(
                '--replacement-reason is required with '
                '--replace-reference')
        with args.out_file.open() as source:
            manifest = json.load(source)
        with args.exclude_manifest.open() as source:
            excluded_manifest = json.load(source)
        manifest = replace_pre_inference_invalid_record(
            manifest, infos, excluded_manifest,
            args.replace_reference[0], args.replace_reference[1],
            args.replacement_reason)
        with args.out_file.open('w') as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write('\n')
        print(json.dumps({
            'name': manifest['name'],
            'status': manifest['status'],
            'selected_reference_indices': (
                manifest['selected_reference_indices']),
            'pre_inference_exclusions': (
                manifest['pre_inference_exclusions']),
            'out_file': str(args.out_file),
        }, ensure_ascii=False, indent=2))
        return
    eligible = load_eligible_records(args.eligibility_audit, infos)
    with args.exclude_manifest.open() as source:
        excluded_manifest = json.load(source)
    records = select_blind_records(
        eligible, excluded_manifest, args.scene_count)
    excluded_count = len(_excluded_scenes(excluded_manifest))
    manifest = {
        'schema_version': 1,
        'name': 'kl_occworld_blind_holdout_v1',
        'status': 'frozen_before_label_generation_and_inference',
        'strategy': (
            'evenly_spaced_from_eligible_scenes_excluding_all_v2_splits'),
        'annotation_file': str(args.ann_file),
        'eligibility_audit': str(args.eligibility_audit),
        'exclude_manifest': str(args.exclude_manifest),
        'eligible_scene_count': len(eligible),
        'excluded_scene_count': excluded_count,
        'fresh_candidate_scene_count': len(eligible) - excluded_count,
        'blind_scene_count': len(records),
        'sequence_root': args.sequence_root,
        'history_root': args.history_root,
        'frozen_model_protocol': {
            'checkpoint_epoch': 10,
            'visibility_threshold': 0.6,
            'physical_flow_fusion_threshold': 0.7,
            'threshold_retuning_on_blind': False,
        },
        'split_scene_tokens': {
            'blind': [record['scene_token'] for record in records],
        },
        'splits': {'blind': records},
        'selected_reference_indices': [
            int(record['reference_index']) for record in records
        ],
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'name': manifest['name'],
        'status': manifest['status'],
        'eligible_scene_count': len(eligible),
        'excluded_scene_count': excluded_count,
        'fresh_candidate_scene_count': (
            manifest['fresh_candidate_scene_count']),
        'blind_scene_count': len(records),
        'selected_reference_indices': (
            manifest['selected_reference_indices']),
        'out_file': str(args.out_file),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
