#!/usr/bin/env python
"""Freeze a new scene-disjoint OccWorld holdout before label generation."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.prepare_kl_occworld_full_train_manifest import (
    _queue_checker,
)
from tools.analysis_tools.select_kl_occworld_expanded_scenes import (
    _evenly_spaced,
    audit_scene_references,
)
from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _resolve_path,
)


class InsufficientFreshScenesError(ValueError):
    """The annotation has fewer fresh eligible scenes than requested."""


def _load_manifest(path: Path) -> dict:
    with path.open() as source:
        result = json.load(source)
    if not isinstance(result.get('splits'), dict):
        raise ValueError(f'Manifest has no splits: {path}')
    return result


def _manifest_scene_set(manifest: Mapping) -> set:
    return {
        str(record['scene_token'])
        for records in manifest['splits'].values()
        for record in records
    }


def select_fresh_holdout_records(
        eligible_records: Sequence[Mapping], existing_manifest: Mapping,
        scene_count: int, model_queue_valid) -> tuple:
    """Return fresh records and queue-rejection diagnostics.

    ``eligible_records`` must already satisfy the timing and LiDAR contract.
    This function adds the UniAD five-frame queue condition and excludes every
    scene in the supplied manifest, including consumed splits.
    """
    if scene_count < 1:
        raise ValueError('scene_count must be positive')
    excluded_scenes = _manifest_scene_set(existing_manifest)
    by_scene = {
        str(record['scene_token']): dict(record)
        for record in eligible_records
    }
    if len(by_scene) != len(eligible_records):
        raise ValueError('Eligible records contain duplicate scene tokens')
    fresh_before_queue = [
        record for scene, record in by_scene.items()
        if scene not in excluded_scenes
    ]
    queue_rejected = [
        record for record in fresh_before_queue
        if not model_queue_valid(int(record['reference_index']))
    ]
    fresh = [
        record for record in fresh_before_queue
        if model_queue_valid(int(record['reference_index']))
    ]
    if len(fresh) < scene_count:
        raise InsufficientFreshScenesError(
            f'Only {len(fresh)} fresh scenes satisfy the complete contract; '
            f'{scene_count} requested')
    selected = _evenly_spaced(fresh, scene_count)
    for record in selected:
        record['selection_source'] = (
            'fresh_evenly_spaced_before_b17_holdout_label_generation')
    selected_scenes = {str(record['scene_token']) for record in selected}
    if selected_scenes.intersection(excluded_scenes):
        raise AssertionError('Fresh holdout overlaps an existing split')
    return selected, {
        'existing_scene_count': len(excluded_scenes),
        'fresh_before_model_queue_count': len(fresh_before_queue),
        'fresh_model_queue_rejected_count': len(queue_rejected),
        'fresh_complete_contract_count': len(fresh),
        'queue_rejected_scene_tokens': sorted(
            str(record['scene_token']) for record in queue_rejected),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
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
        '--existing-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_full_train3_final30_manifest_v1.json'))
    parser.add_argument('--scene-count', type=int, default=30)
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
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b17_fresh_holdout_selection'))
    parser.add_argument(
        '--out-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b17_fresh_holdout_manifest_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scene_count < 1:
        raise ValueError('--scene-count must be positive')
    existing_manifest = _load_manifest(args.existing_manifest)
    resolved_ann_file = Path(_resolve_path(str(args.ann_file)))
    annotation_sha256 = _sha256(resolved_ann_file)
    infos, _ = _load_infos(resolved_ann_file)
    required_offsets = sorted(set(
        int(value) for value in (
            *args.history_offsets, *args.future_offsets)))
    eligible, audit_rows = audit_scene_references(
        infos, required_offsets,
        expected_step_s=args.expected_step_s,
        max_time_error_s=args.max_time_error_s,
        expected_sensor_count=args.expected_sensor_count,
        check_lidar_files=not args.skip_lidar_file_check)
    queue_valid = _queue_checker(
        infos, args.model_queue_length, args.model_queue_max_gap_s)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.out_dir / 'scene_eligibility.csv'
    _write_csv(audit_csv, audit_rows)
    report = {
        'schema_version': 'kl_occworld_fresh_holdout_preflight_v1',
        'selection_uses_occworld_labels': False,
        'selection_uses_model_predictions': False,
        'annotation_file': str(args.ann_file),
        'resolved_annotation_file': str(resolved_ann_file),
        'annotation_sha256': annotation_sha256,
        'existing_manifest': str(args.existing_manifest),
        'existing_manifest_sha256': _sha256(args.existing_manifest),
        'annotation_frame_count': len(infos),
        'annotation_scene_count': len(audit_rows),
        'eligible_scene_count': len(eligible),
        'ineligible_scene_count': len(audit_rows) - len(eligible),
        'required_offsets': required_offsets,
        'expected_step_s': args.expected_step_s,
        'max_time_error_s': args.max_time_error_s,
        'model_queue_length': args.model_queue_length,
        'model_queue_max_gap_s': args.model_queue_max_gap_s,
        'expected_sensor_count': args.expected_sensor_count,
        'lidar_file_check': not args.skip_lidar_file_check,
        'eligibility_audit_csv': str(audit_csv),
        'requested_holdout_scene_count': args.scene_count,
    }
    try:
        selected, diagnostics = select_fresh_holdout_records(
            eligible, existing_manifest, args.scene_count, queue_valid)
    except InsufficientFreshScenesError as error:
        excluded = _manifest_scene_set(existing_manifest)
        fresh_before_queue = [
            record for record in eligible
            if str(record['scene_token']) not in excluded
        ]
        report.update({
            'status': 'insufficient_fresh_scenes_no_holdout_frozen',
            'reason': str(error),
            'existing_scene_count': len(excluded),
            'fresh_before_model_queue_count': len(fresh_before_queue),
            'fresh_complete_contract_count': sum(
                queue_valid(int(record['reference_index']))
                for record in fresh_before_queue),
        })
        report_path = args.out_dir / 'preflight.json'
        with report_path.open('w') as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write('\n')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    report.update(diagnostics)
    report.update({
        'status': 'fresh_holdout_frozen_before_label_generation',
        'selected_scene_count': len(selected),
        'selected_reference_indices': [
            int(record['reference_index']) for record in selected],
    })
    report_path = args.out_dir / 'preflight.json'
    with report_path.open('w') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    manifest = {
        'schema_version': 1,
        'name': 'kl_occworld_b17_fresh_holdout_v1',
        'status': 'frozen_before_label_generation_and_inference',
        'strategy': (
            'fresh_scene_disjoint_evenly_spaced_after_full_contract_audit'),
        'source_preflight': str(report_path),
        'source_annotation_file': str(resolved_ann_file),
        'source_annotation_sha256': annotation_sha256,
        'source_existing_manifest': str(args.existing_manifest),
        'source_existing_manifest_sha256': _sha256(args.existing_manifest),
        'selection_uses_occworld_labels': False,
        'selection_uses_model_predictions': False,
        'threshold_retuning_allowed': False,
        'single_final_evaluation_only': True,
        'required_offsets': required_offsets,
        'expected_step_s': args.expected_step_s,
        'max_time_error_s': args.max_time_error_s,
        'model_queue_length': args.model_queue_length,
        'model_queue_max_gap_s': args.model_queue_max_gap_s,
        'expected_sensor_count': args.expected_sensor_count,
        'splits': {'fresh_holdout': selected},
        'split_scene_tokens': {
            'fresh_holdout': [
                str(record['scene_token']) for record in selected]},
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open('w') as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'status': report['status'],
        'selected_scene_count': len(selected),
        'preflight': str(report_path),
        'manifest': str(args.out_manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
