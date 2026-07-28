#!/usr/bin/env python
"""Audit new annotations and freeze B24 development/final scene splits."""

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.prepare_kl_occworld_fresh_holdout import (
    InsufficientFreshScenesError,
    _combine_manifests,
    _load_manifest,
    _manifest_scene_set,
    _sha256,
    _write_csv,
    select_fresh_holdout_records,
)
from tools.analysis_tools.prepare_kl_occworld_full_train_manifest import (
    _queue_checker,
)
from tools.analysis_tools.select_kl_occworld_expanded_scenes import (
    audit_scene_references,
)
from tools.data_converter.generate_kl_occworld_labels import (
    _load_infos,
    _resolve_path,
)
from tools.data_converter.generate_kl_occworld_sequence_labels import (
    _validate_fixed_frame_times,
)


DEFAULT_EXISTING_MANIFESTS = (
    Path(
        'documents/patent_2026_occ/'
        'kl_occworld_full_train3_final30_manifest_v1.json'),
    Path(
        'documents/patent_2026_occ/'
        'kl_occworld_b17_fresh_holdout_val65_v1.json'),
    Path(
        'documents/patent_2026_occ/'
        'kl_occworld_b23_fresh_holdout_remaining_val31_v2.json'),
)
DEFAULT_EXISTING_ANNOTATIONS = (
    Path('data/kl_8/kl_infos_train.pkl'),
    Path('data/kl_8/kl_infos_val.pkl'),
)
MINIMUM_SCENES_PER_SPLIT = 30
HISTORY_OFFSETS = (-4, -3, -2, -1, 0)
FUTURE_OFFSETS = tuple(range(9))
EXPECTED_STEP_S = 0.5
MAX_TIME_ERROR_S = 0.2
MODEL_QUEUE_LENGTH = 5
MODEL_QUEUE_MAX_GAP_S = 1.0
EXPECTED_SENSOR_COUNT = 8


def annotation_scene_manifest(infos: Sequence[Mapping]) -> dict:
    """Represent every canonical annotation scene as an exclusion manifest."""
    records = {}
    for index, info in enumerate(infos):
        scene = str(info['scene_token'])
        records.setdefault(scene, {
            'scene_token': scene,
            'reference_index': index,
        })
    return {
        'splits': {'canonical_annotation_scenes': list(records.values())},
    }


def split_interleaved_records(
        selected_records: Sequence[Mapping], development_scene_count: int,
        final_scene_count: int) -> dict:
    """Split an already sampled timeline while interleaving both protocols."""
    if development_scene_count < 1 or final_scene_count < 1:
        raise ValueError('Both B24 split counts must be positive')
    expected_count = development_scene_count + final_scene_count
    if len(selected_records) != expected_count:
        raise ValueError(
            f'Expected {expected_count} selected records, got '
            f'{len(selected_records)}')
    scenes = [str(record['scene_token']) for record in selected_records]
    if len(scenes) != len(set(scenes)):
        raise ValueError('Selected records contain duplicate scene tokens')

    development = []
    final = []
    for record in selected_records:
        if len(development) == development_scene_count:
            destination = final
        elif len(final) == final_scene_count:
            destination = development
        elif (len(development) * final_scene_count <=
              len(final) * development_scene_count):
            destination = development
        else:
            destination = final
        destination.append(dict(record))

    for record in development:
        record['selection_source'] = (
            'b24_new_scene_interleaved_development_validation')
    for record in final:
        record['selection_source'] = (
            'b24_new_scene_interleaved_single_use_final_holdout')
    return {
        'development_validation': development,
        'final_holdout': final,
    }


def select_b24_new_scene_records(
        eligible_records: Sequence[Mapping], existing_manifest: Mapping,
        development_scene_count: int, final_scene_count: int,
        model_queue_valid) -> tuple:
    """Exclude every consumed scene, sample once, then freeze two splits."""
    total_scene_count = development_scene_count + final_scene_count
    selected, diagnostics = select_fresh_holdout_records(
        eligible_records, existing_manifest, total_scene_count,
        model_queue_valid)
    splits = split_interleaved_records(
        selected, development_scene_count, final_scene_count)
    development_scenes = {
        str(record['scene_token'])
        for record in splits['development_validation']
    }
    final_scenes = {
        str(record['scene_token']) for record in splits['final_holdout']
    }
    if development_scenes.intersection(final_scenes):
        raise AssertionError('B24 development and final scenes overlap')
    if ((development_scenes | final_scenes).intersection(
            _manifest_scene_set(existing_manifest))):
        raise AssertionError('B24 new-scene split overlaps consumed scenes')
    diagnostics.update({
        'selected_scene_count': total_scene_count,
        'development_validation_scene_count': len(development_scenes),
        'final_holdout_scene_count': len(final_scenes),
    })
    return splits, diagnostics


def build_b24_split_manifest(
        *, splits: Mapping, resolved_annotation_file: Path,
        annotation_sha256: str, preflight_path: Path,
        existing_manifest_records: Sequence[Mapping], required_offsets,
        existing_annotation_records: Sequence[Mapping] = (),
        expected_step_s: float, max_time_error_s: float,
        model_queue_length: int, model_queue_max_gap_s: float,
        expected_sensor_count: int, lidar_file_check: bool) -> dict:
    """Build the immutable pre-label, pre-inference protocol ledger."""
    development = list(splits['development_validation'])
    final = list(splits['final_holdout'])
    development_scenes = [str(record['scene_token'])
                          for record in development]
    final_scenes = [str(record['scene_token']) for record in final]
    if set(development_scenes).intersection(final_scenes):
        raise ValueError('B24 split manifest contains overlapping scenes')
    return {
        'schema_version': 'kl_occworld_b24_new_scene_split_v1',
        'name': 'kl_occworld_b24_new_scene_split_v1',
        'status': 'frozen_before_gt_generation_and_model_inference',
        'strategy': 'fresh_evenly_spaced_then_temporally_interleaved',
        'annotation_file': str(resolved_annotation_file),
        'source_annotation_sha256': str(annotation_sha256),
        'source_preflight': str(preflight_path),
        'source_existing_manifests': list(existing_manifest_records),
        'source_existing_annotations': list(existing_annotation_records),
        'selection_uses_occworld_labels': False,
        'selection_uses_model_predictions': False,
        'labels_inspected': False,
        'model_predictions_inspected': False,
        'development_validation_status': 'frozen_not_evaluated',
        'final_holdout_status': 'frozen_not_evaluated',
        'final_holdout_single_use_only': True,
        'checkpoint_selection_on_development_validation_only': True,
        'threshold_retuning_on_final_holdout': False,
        'required_offsets': list(required_offsets),
        'nested_sequence_target_offsets': list(range(5)),
        'nested_sequence_reveal_offsets': list(range(5)),
        'expected_step_s': float(expected_step_s),
        'max_time_error_s': float(max_time_error_s),
        'model_queue_length': int(model_queue_length),
        'model_queue_max_gap_s': float(model_queue_max_gap_s),
        'expected_sensor_count': int(expected_sensor_count),
        'lidar_file_check': bool(lidar_file_check),
        'splits': {
            'development_validation': development,
            'final_holdout': final,
        },
        'split_scene_tokens': {
            'development_validation': development_scenes,
            'final_holdout': final_scenes,
        },
    }


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', type=Path, required=True,
        help='New metadata-only annotation. Existing KL annotations are invalid.')
    parser.add_argument(
        '--existing-manifest', type=Path, action='append',
        help=(
            'Additional consumed manifest to exclude. Repeat as needed. The '
            'canonical B15/B17/B23 manifests are always included.'))
    parser.add_argument(
        '--existing-annotation', type=Path, action='append',
        help=(
            'Additional canonical annotation whose every scene must be '
            'excluded. The KL train/val annotations are always included.'))
    parser.add_argument('--development-scene-count', type=int, default=30)
    parser.add_argument('--final-scene-count', type=int, default=30)
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b24_new_scene_split_preflight_v1'))
    parser.add_argument(
        '--out-manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b24_new_scene_split_v1.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.development_scene_count < MINIMUM_SCENES_PER_SPLIT or
            args.final_scene_count < MINIMUM_SCENES_PER_SPLIT):
        raise ValueError(
            f'Both B24 splits require at least '
            f'{MINIMUM_SCENES_PER_SPLIT} scenes')
    if args.out_manifest.exists():
        raise FileExistsError(
            f'Refusing to overwrite frozen manifest: {args.out_manifest}')

    existing_manifest_paths = tuple(dict.fromkeys([
        *DEFAULT_EXISTING_MANIFESTS,
        *(args.existing_manifest or ()),
    ]))
    missing_manifests = [str(path) for path in existing_manifest_paths
                         if not path.is_file()]
    if missing_manifests:
        raise FileNotFoundError(
            f'Consumed manifests are missing: {missing_manifests}')
    existing_manifests = [
        _load_manifest(path) for path in existing_manifest_paths]
    existing_manifest_records = [
        {'path': str(path), 'sha256': _sha256(path)}
        for path in existing_manifest_paths
    ]
    existing_annotation_paths = tuple(dict.fromkeys([
        *DEFAULT_EXISTING_ANNOTATIONS,
        *(args.existing_annotation or ()),
    ]))
    resolved_existing_annotation_paths = [
        Path(_resolve_path(str(path))) for path in existing_annotation_paths]
    missing_annotations = [
        str(path) for path in resolved_existing_annotation_paths
        if not path.is_file()
    ]
    if missing_annotations:
        raise FileNotFoundError(
            f'Canonical annotations are missing: {missing_annotations}')
    existing_annotation_records = []
    existing_annotation_manifests = []
    for path, resolved_path in zip(
            existing_annotation_paths, resolved_existing_annotation_paths):
        canonical_infos, _ = _load_infos(resolved_path)
        scene_manifest = annotation_scene_manifest(canonical_infos)
        existing_annotation_manifests.append(scene_manifest)
        existing_annotation_records.append({
            'path': str(path),
            'resolved_path': str(resolved_path),
            'sha256': _sha256(resolved_path),
            'frame_count': len(canonical_infos),
            'scene_count': len(_manifest_scene_set(scene_manifest)),
        })
    existing_manifest = _combine_manifests(
        [*existing_manifests, *existing_annotation_manifests])

    resolved_annotation_file = Path(_resolve_path(str(args.ann_file)))
    annotation_sha256 = _sha256(resolved_annotation_file)
    infos, _ = _load_infos(resolved_annotation_file)
    required_offsets = sorted(set((*HISTORY_OFFSETS, *FUTURE_OFFSETS)))
    eligible, audit_rows = audit_scene_references(
        infos, required_offsets,
        expected_step_s=EXPECTED_STEP_S,
        max_time_error_s=MAX_TIME_ERROR_S,
        expected_sensor_count=EXPECTED_SENSOR_COUNT,
        check_lidar_files=True,
        reference_validator=lambda reference: _validate_fixed_frame_times(
            infos, reference,
            target_offsets=list(range(5)),
            reveal_offsets=list(range(5)),
            expected_step_s=EXPECTED_STEP_S,
            max_time_error_s=MAX_TIME_ERROR_S))
    model_queue_valid = _queue_checker(
        infos, MODEL_QUEUE_LENGTH, MODEL_QUEUE_MAX_GAP_S)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.out_dir / 'scene_eligibility.csv'
    preflight_path = args.out_dir / 'preflight.json'
    _write_csv(audit_csv, audit_rows)
    report = {
        'schema_version': 'kl_occworld_b24_new_scene_preflight_v1',
        'selection_uses_occworld_labels': False,
        'selection_uses_model_predictions': False,
        'annotation_file': str(args.ann_file),
        'resolved_annotation_file': str(resolved_annotation_file),
        'annotation_sha256': annotation_sha256,
        'existing_manifests': existing_manifest_records,
        'existing_annotations': existing_annotation_records,
        'annotation_frame_count': len(infos),
        'annotation_scene_count': len(audit_rows),
        'eligible_scene_count': len(eligible),
        'ineligible_scene_count': len(audit_rows) - len(eligible),
        'existing_scene_count': len(_manifest_scene_set(existing_manifest)),
        'required_offsets': required_offsets,
        'nested_sequence_target_offsets': list(range(5)),
        'nested_sequence_reveal_offsets': list(range(5)),
        'expected_step_s': EXPECTED_STEP_S,
        'max_time_error_s': MAX_TIME_ERROR_S,
        'model_queue_length': MODEL_QUEUE_LENGTH,
        'model_queue_max_gap_s': MODEL_QUEUE_MAX_GAP_S,
        'expected_sensor_count': EXPECTED_SENSOR_COUNT,
        'lidar_file_check': True,
        'eligibility_audit_csv': str(audit_csv),
        'requested_development_validation_scene_count': (
            args.development_scene_count),
        'requested_final_holdout_scene_count': args.final_scene_count,
    }
    try:
        splits, diagnostics = select_b24_new_scene_records(
            eligible, existing_manifest,
            args.development_scene_count, args.final_scene_count,
            model_queue_valid)
    except InsufficientFreshScenesError as error:
        excluded = _manifest_scene_set(existing_manifest)
        fresh_before_queue = [
            record for record in eligible
            if str(record['scene_token']) not in excluded
        ]
        report.update({
            'status': 'insufficient_fresh_scenes_no_split_frozen',
            'reason': str(error),
            'fresh_before_model_queue_count': len(fresh_before_queue),
            'fresh_complete_contract_count': sum(
                model_queue_valid(int(record['reference_index']))
                for record in fresh_before_queue),
        })
        _write_json(preflight_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    report.update(diagnostics)
    report['status'] = 'b24_new_scene_protocol_frozen'
    report['selected_reference_indices'] = {
        split_name: [int(record['reference_index'])
                     for record in records]
        for split_name, records in splits.items()
    }
    _write_json(preflight_path, report)
    manifest = build_b24_split_manifest(
        splits=splits,
        resolved_annotation_file=resolved_annotation_file,
        annotation_sha256=annotation_sha256,
        preflight_path=preflight_path,
        existing_manifest_records=existing_manifest_records,
        existing_annotation_records=existing_annotation_records,
        required_offsets=required_offsets,
        expected_step_s=EXPECTED_STEP_S,
        max_time_error_s=MAX_TIME_ERROR_S,
        model_queue_length=MODEL_QUEUE_LENGTH,
        model_queue_max_gap_s=MODEL_QUEUE_MAX_GAP_S,
        expected_sensor_count=EXPECTED_SENSOR_COUNT,
        lidar_file_check=True)
    _write_json(args.out_manifest, manifest)
    print(json.dumps({
        'status': manifest['status'],
        'development_validation_scene_count': len(
            splits['development_validation']),
        'final_holdout_scene_count': len(splits['final_holdout']),
        'preflight': str(preflight_path),
        'manifest': str(args.out_manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
