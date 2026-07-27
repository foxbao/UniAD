#!/usr/bin/env python
"""Prepare and audit B17 fresh-holdout GT before model inference."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.run_kl_occworld_full_train_data import (
    run_audit,
    run_base_labels,
    run_cross_scene,
    run_sequence_history,
    setup_paths,
)
from tools.analysis_tools.verify_kl_occworld_candidate_freeze import (
    verify_candidate_freeze,
)


FRESH_SELECTION_MANIFEST = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_b17_fresh_holdout_val65_v1.json')
EXISTING_MANIFEST = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_full_train3_final30_manifest_v1.json')
CANDIDATE_FREEZE = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_b17a_epoch3_candidate_freeze_v1.json')
EVALUATION_MANIFEST = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_b17a_fresh_holdout_val65_evaluation_v1.json')
TRACK_CONFIG = REPO_ROOT / (
    'projects/configs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b17a_fresh_holdout_val65_track_export.py')
EVAL_CONFIG = REPO_ROOT / (
    'projects/configs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b17a_fresh_holdout_val65_eval.py')
DISK_ROOT = Path(
    '/mnt/disk1/baojiali/UniAD_occworld_b17_fresh_holdout_val65_v1')
TRACK_QUEUE_ROOT = REPO_ROOT / (
    'outputs/patent_2026_occ/'
    'occworld_track_queues_b17_fresh_holdout_val65_v1')
ONLINE_INPUT_ROOT = REPO_ROOT / (
    'outputs/patent_2026_occ/'
    'occworld_online_inputs_b17_fresh_holdout_val65_v1')
ONLINE_INPUT_AUDIT = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_b17_fresh_holdout_val65_online_input_audit_v1.json')


def _read_json(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _record_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _root_mapping() -> dict:
    suffixes = {
        'temporal': 'occworld_temporal_b17_fresh_holdout_val65_v1',
        'occlusion': 'occworld_occlusion_b17_fresh_holdout_val65_v1',
        'dual': 'occworld_dual_b17_fresh_holdout_val65_v1',
        'dual_cross_scene': (
            'occworld_dual_cross_scene_b17_fresh_holdout_val65_v1'),
        'observation_cache': (
            'occworld_observation_cache_b17_fresh_holdout_val65_v1'),
        'sequence': 'occworld_sequence_b17_fresh_holdout_val65_v1',
        'history': 'occworld_history_b17_fresh_holdout_val65_v1',
        'sequence_audit': (
            'occworld_sequence_b17_fresh_holdout_val65_v1_audit'),
        'history_audit': (
            'occworld_history_b17_fresh_holdout_val65_v1_audit'),
    }
    return {
        key: f'outputs/patent_2026_occ/{value}'
        for key, value in suffixes.items()
    }


def validate_fresh_records(fresh_manifest: dict,
                           existing_manifest: dict,
                           expected_count: int = 30) -> list:
    """Validate frozen records without reading labels or predictions."""
    if fresh_manifest.get('status') != (
            'frozen_before_label_generation_and_inference'):
        raise ValueError('Fresh selection manifest is not frozen')
    if fresh_manifest.get('selection_uses_occworld_labels') is not False:
        raise ValueError('Fresh selection unexpectedly used OccWorld labels')
    if fresh_manifest.get('selection_uses_model_predictions') is not False:
        raise ValueError('Fresh selection unexpectedly used model predictions')
    if fresh_manifest.get('threshold_retuning_allowed') is not False:
        raise ValueError('Fresh selection permits threshold retuning')
    records = [
        dict(row) for row in
        fresh_manifest.get('splits', {}).get('fresh_holdout', [])
    ]
    if len(records) != expected_count:
        raise ValueError(
            f'Expected {expected_count} fresh scenes, got {len(records)}')
    references = [int(row['reference_index']) for row in records]
    scenes = [str(row['scene_token']) for row in records]
    if len(set(references)) != len(records):
        raise ValueError('Fresh holdout has duplicate references')
    if len(set(scenes)) != len(records):
        raise ValueError('Fresh holdout has duplicate scenes')
    old_scenes = {
        str(row['scene_token'])
        for rows in existing_manifest['splits'].values()
        for row in rows
    }
    overlap = sorted(set(scenes).intersection(old_scenes))
    if overlap:
        raise ValueError(f'Fresh holdout overlaps existing scenes: {overlap}')
    return records


def prepare_evaluation_manifest(
        fresh_path: Path, existing_path: Path, freeze_path: Path,
        track_config_path: Path, eval_config_path: Path) -> dict:
    fresh = _read_json(fresh_path)
    existing = _read_json(existing_path)
    candidate_summary = verify_candidate_freeze(freeze_path)
    records = validate_fresh_records(fresh, existing)
    annotation_file = Path(fresh['annotation_file'])
    if not annotation_file.is_file():
        raise FileNotFoundError(
            f'Fresh annotation is missing: {annotation_file}')
    annotation_sha256 = _sha256(annotation_file)
    if annotation_sha256 != fresh['source_annotation_sha256']:
        raise ValueError('Fresh annotation changed after scene selection')
    if existing.get('status') != 'ready_after_full_generation_and_audit':
        raise ValueError('Existing train manifest is not ready')
    freeze = _read_json(freeze_path)
    protocol = freeze['frozen_inference_protocol']
    checkpoint = freeze['selected_checkpoint']
    return {
        'schema_version': 1,
        'name': 'kl_occworld_b17a_fresh_holdout_val65_evaluation_v1',
        'status': 'frozen_before_fresh_holdout_gt_generation',
        'annotation_file': str(annotation_file),
        'annotation_sha256': annotation_sha256,
        'source_fresh_selection_manifest': str(
            fresh_path.relative_to(REPO_ROOT)),
        'source_fresh_selection_manifest_sha256': _sha256(fresh_path),
        'source_existing_manifest': str(
            existing_path.relative_to(REPO_ROOT)),
        'source_existing_manifest_sha256': _sha256(existing_path),
        'candidate_freeze': str(freeze_path.relative_to(REPO_ROOT)),
        'candidate_freeze_sha256': _sha256(freeze_path),
        'candidate_verification': candidate_summary,
        'track_export_config': str(
            track_config_path.relative_to(REPO_ROOT)),
        'track_export_config_sha256': _sha256(track_config_path),
        'evaluation_config': str(
            eval_config_path.relative_to(REPO_ROOT)),
        'evaluation_config_sha256': _sha256(eval_config_path),
        'fresh_holdout_reference_count': len(records),
        'split_scene_tokens': {
            'fresh_holdout': [
                str(row['scene_token']) for row in records]},
        'splits': {'fresh_holdout': records},
        'roots': _root_mapping(),
        'cross_scene_support': {
            'source_split': 'train',
            'annotation_file': existing['annotation_file'],
            'dual_dir': existing['roots']['dual'],
            'reference_indices': [
                int(row['reference_index'])
                for row in existing['splits']['train']],
            'target_holdout_used_as_mutual_support': False,
        },
        'frozen_model_protocol': {
            'candidate_name': freeze['candidate_name'],
            'checkpoint_path': checkpoint['path'],
            'checkpoint_sha256': checkpoint['sha256'],
            'checkpoint_epoch': checkpoint[
                'checkpoint_meta']['epoch'],
            'visibility_threshold': protocol['visibility_threshold'],
            'local_flow_overlay_threshold': protocol[
                'model_local_overlay_threshold'],
            'evaluator_post_overlay': protocol[
                'evaluator_post_overlay'],
            'threshold_retuning_allowed': False,
            'checkpoint_reselection_allowed': False,
            'single_model_inference_only': True,
        },
        'selection_uses_occworld_labels': False,
        'selection_uses_model_predictions': False,
        'model_predictions_inspected': False,
        'fresh_holdout_metrics_inspected': False,
    }


def _load_evaluation_manifest(path: Path) -> dict:
    manifest = _read_json(path)
    allowed = {
        'frozen_before_fresh_holdout_gt_generation',
        'ready_for_track_queue_export',
        'ready_for_single_model_inference',
    }
    if manifest.get('status') not in allowed:
        raise ValueError(
            f'Unexpected fresh-holdout status: {manifest.get("status")}')
    return manifest


def _generation_view(manifest: dict) -> dict:
    result = copy.deepcopy(manifest)
    result['status'] = 'frozen_before_full_gt_generation'
    result['splits'] = {
        'train': result['splits']['fresh_holdout']}
    result['train_reference_count'] = int(
        result['fresh_holdout_reference_count'])
    return result


def _single_artifact(root: Path, reference: int, name: str) -> Path:
    path = root / f'{reference:06d}' / name
    if not path.is_file():
        raise FileNotFoundError(f'Missing artifact: {path}')
    return path


def validate_online_artifacts(manifest: dict, track_queue_root: Path,
                              online_input_root: Path,
                              audit_report: dict) -> dict:
    """Validate exact causal online inputs without running the world model."""
    expected = {
        int(row['reference_index']): str(row['scene_token'])
        for row in manifest['splits']['fresh_holdout']
    }
    track_references = {
        int(path.parent.name)
        for path in track_queue_root.glob('*/occworld_track_queue.npz')
    }
    online_references = {
        int(path.parent.name)
        for path in online_input_root.glob('*/occworld_online_input.npz')
    }
    if track_references != set(expected):
        raise ValueError('Track queue references are not exact')
    if online_references != set(expected):
        raise ValueError('Online input references are not exact')
    total_frames = 0
    total_boxes = 0
    for reference, expected_scene in sorted(expected.items()):
        track_path = _single_artifact(
            track_queue_root, reference, 'occworld_track_queue.npz')
        online_path = _single_artifact(
            online_input_root, reference, 'occworld_online_input.npz')
        with np.load(track_path, allow_pickle=False) as track:
            if int(track['reference_index']) != reference:
                raise ValueError(f'Wrong track reference: {track_path}')
            queue_indices = np.asarray(
                track['queue_frame_indices'], dtype=np.int64)
            scenes = np.asarray(track['queue_scene_tokens']).astype(str)
            offsets = np.asarray(
                track['track_box_offsets'], dtype=np.int64)
            if queue_indices.shape != (5,) or scenes.shape != (5,):
                raise ValueError(f'Bad track queue shape for {reference}')
            if set(scenes.tolist()) != {expected_scene}:
                raise ValueError(
                    f'Track queue changed scene for {reference}')
            if offsets.shape != (6,) or np.any(np.diff(offsets) < 0):
                raise ValueError(f'Bad track box offsets for {reference}')
            total_frames += len(queue_indices)
            total_boxes += int(offsets[-1])
        with np.load(online_path, allow_pickle=False) as online:
            if int(online['reference_index']) != reference:
                raise ValueError(f'Wrong online reference: {online_path}')
            if online['current_world_state_3d'].shape != (10, 120, 160):
                raise ValueError(f'Bad current state for {reference}')
            if online['current_world_valid_3d'].shape != (10, 120, 160):
                raise ValueError(f'Bad current valid mask for {reference}')
            if online['history_world_state_3d'].shape != (
                    5, 10, 120, 160):
                raise ValueError(f'Bad history state for {reference}')
            if online['history_world_valid_3d'].shape != (
                    5, 10, 120, 160):
                raise ValueError(f'Bad history valid mask for {reference}')
            np.testing.assert_array_equal(
                online['queue_frame_indices'], queue_indices)
            if str(online['input_contract']) != (
                    'five_frame_sequential_trackformer_predicted_boxes'):
                raise ValueError(
                    f'Wrong online input contract for {reference}')
    aggregate = audit_report.get('aggregate', {})
    if int(aggregate.get('reference_count', -1)) != len(expected):
        raise ValueError('Online audit reference count changed')
    if int(aggregate.get('frame_count', -1)) != total_frames:
        raise ValueError('Online audit frame count changed')
    report_references = {
        int(row['reference_index'])
        for row in audit_report.get('rows', [])
    }
    if report_references != set(expected):
        raise ValueError('Online audit rows are not exact')
    return {
        'track_queue_count': len(track_references),
        'online_input_count': len(online_references),
        'queue_frame_count': total_frames,
        'track_box_count': total_boxes,
        'reference_sets_exactly_equal': True,
        'input_contract': (
            'five_frame_sequential_trackformer_predicted_boxes'),
        'instance_iou': aggregate['instance_iou'],
        'instance_precision': aggregate['instance_precision'],
        'instance_recall': aggregate['instance_recall'],
        'mean_current_state_mismatch_ratio': aggregate[
            'mean_current_state_mismatch_ratio'],
        'mean_history_state_mismatch_ratio': aggregate[
            'mean_history_state_mismatch_ratio'],
    }


def finalize_online_inputs(manifest: dict, track_queue_root: Path,
                           online_input_root: Path,
                           audit_path: Path) -> dict:
    if manifest.get('status') != 'ready_for_track_queue_export':
        raise ValueError('Fresh holdout is not ready for online inputs')
    for path_key, hash_key in (
            ('track_export_config', 'track_export_config_sha256'),
            ('evaluation_config', 'evaluation_config_sha256'),
            ('candidate_freeze', 'candidate_freeze_sha256')):
        path = REPO_ROOT / manifest[path_key]
        if _sha256(path) != manifest[hash_key]:
            raise ValueError(f'Frozen artifact changed: {path_key}')
    verify_candidate_freeze(REPO_ROOT / manifest['candidate_freeze'])
    audit = _read_json(audit_path)
    validation = validate_online_artifacts(
        manifest, track_queue_root, online_input_root, audit)
    result = copy.deepcopy(manifest)
    result['status'] = 'ready_for_single_model_inference'
    result['online_input_audit'] = {
        **validation,
        'track_queue_root': _record_path(track_queue_root),
        'online_input_root': _record_path(online_input_root),
        'audit_report': _record_path(audit_path),
        'audit_report_sha256': _sha256(audit_path),
        'model_inference_performed': False,
    }
    return result


def _artifact_mapping(root: Path, suffix: str) -> dict:
    mapping = {}
    for path in sorted(root.glob(f'*/*__{suffix}.npz')):
        with np.load(path, allow_pickle=False) as payload:
            reference = int(payload['reference_index'])
        if reference in mapping:
            raise ValueError(f'Duplicate artifact for reference {reference}')
        mapping[reference] = path
    return mapping


def finalize_gt(manifest: dict) -> dict:
    expected = {
        int(row['reference_index'])
        for row in manifest['splits']['fresh_holdout']
    }
    sequence_root = REPO_ROOT / manifest['roots']['sequence']
    history_root = REPO_ROOT / manifest['roots']['history']
    sequences = _artifact_mapping(sequence_root, 'occworld_sequence')
    histories = _artifact_mapping(history_root, 'occworld_history')
    if set(sequences) != expected or set(histories) != expected:
        raise ValueError('Fresh holdout artifact references are not exact')
    max_future_error = 0.0
    max_history_error = 0.0
    max_last_transform_error = 0.0
    valid_ratios = []
    for reference in sorted(expected):
        with np.load(sequences[reference], allow_pickle=False) as sequence:
            state = sequence['world_target_state_3d']
            valid = np.asarray(
                sequence['world_target_valid_3d'], dtype=np.bool_)
            if state.shape != (5, 10, 120, 160):
                raise ValueError(f'Bad sequence shape for {reference}')
            direct_t0 = np.asarray(
                sequence['direct_observation_state_3d'][0])
            times = np.asarray(
                sequence['target_times_s'], dtype=np.float64)
            max_future_error = max(
                max_future_error,
                float(np.max(np.abs(times - np.arange(5) * 0.5))))
            valid_ratios.append(float(valid.mean()))
        with np.load(histories[reference], allow_pickle=False) as history:
            history_state = history['history_observation_state_3d']
            if history_state.shape != (5, 10, 120, 160):
                raise ValueError(f'Bad history shape for {reference}')
            np.testing.assert_array_equal(history_state[-1], direct_t0)
            times = np.asarray(
                history['history_times_s'], dtype=np.float64)
            max_history_error = max(
                max_history_error,
                float(np.max(np.abs(
                    times - np.arange(-4, 1) * 0.5))))
            transforms = np.asarray(history['history_to_reference'])
            max_last_transform_error = max(
                max_last_transform_error,
                float(np.max(np.abs(transforms[-1] - np.eye(4)))))
    result = copy.deepcopy(manifest)
    result['status'] = 'ready_for_track_queue_export'
    result['fresh_holdout_generation_audit'] = {
        'sequence_count': len(sequences),
        'history_count': len(histories),
        'reference_sets_exactly_equal': True,
        'sequence_shape': [5, 10, 120, 160],
        'history_shape': [5, 10, 120, 160],
        'history_last_equals_direct_t0': True,
        'max_future_time_error_s': max_future_error,
        'max_history_time_error_s': max_history_error,
        'max_last_transform_error': max_last_transform_error,
        'mean_world_valid_ratio': float(np.mean(valid_ratios)),
        'model_inference_performed': False,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--phase', choices=(
            'prepare', 'setup', 'base-labels', 'cross-scene',
            'sequence-history', 'audit', 'all',
            'finalize-online-inputs'),
        default='prepare')
    parser.add_argument('--fresh-manifest', type=Path,
                        default=FRESH_SELECTION_MANIFEST)
    parser.add_argument('--existing-manifest', type=Path,
                        default=EXISTING_MANIFEST)
    parser.add_argument('--candidate-freeze', type=Path,
                        default=CANDIDATE_FREEZE)
    parser.add_argument('--evaluation-manifest', type=Path,
                        default=EVALUATION_MANIFEST)
    parser.add_argument('--track-config', type=Path,
                        default=TRACK_CONFIG)
    parser.add_argument('--eval-config', type=Path,
                        default=EVAL_CONFIG)
    parser.add_argument('--disk-root', type=Path, default=DISK_ROOT)
    parser.add_argument('--track-queue-root', type=Path,
                        default=TRACK_QUEUE_ROOT)
    parser.add_argument('--online-input-root', type=Path,
                        default=ONLINE_INPUT_ROOT)
    parser.add_argument('--online-input-audit', type=Path,
                        default=ONLINE_INPUT_AUDIT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.evaluation_manifest.resolve()
    if args.phase == 'prepare':
        if manifest_path.exists():
            raise FileExistsError(
                f'Refusing to replace frozen manifest: {manifest_path}')
        manifest = prepare_evaluation_manifest(
            args.fresh_manifest.resolve(),
            args.existing_manifest.resolve(),
            args.candidate_freeze.resolve(),
            args.track_config.resolve(),
            args.eval_config.resolve())
        _write_json(manifest_path, manifest)
    else:
        manifest = _load_evaluation_manifest(manifest_path)
        if args.phase == 'finalize-online-inputs':
            manifest = finalize_online_inputs(
                manifest,
                args.track_queue_root.resolve(),
                args.online_input_root.resolve(),
                args.online_input_audit.resolve())
            _write_json(manifest_path, manifest)
            print(json.dumps({
                'phase': args.phase,
                'status': manifest['status'],
                'fresh_holdout_reference_count': (
                    manifest['fresh_holdout_reference_count']),
                'evaluation_manifest': str(manifest_path),
                'disk_root': str(args.disk_root),
            }, ensure_ascii=False, indent=2))
            return
        view = _generation_view(manifest)
        setup_paths(view, args.disk_root)
        phases = (
            ['base-labels', 'cross-scene', 'sequence-history', 'audit']
            if args.phase == 'all' else [args.phase])
        for phase in phases:
            if phase == 'setup':
                continue
            if phase == 'base-labels':
                run_base_labels(
                    view, args.disk_root, args.workers, args.dry_run)
            elif phase == 'cross-scene':
                run_cross_scene(view, args.disk_root, args.dry_run)
            elif phase == 'sequence-history':
                run_sequence_history(
                    view, args.disk_root, args.workers, args.dry_run)
            elif phase == 'audit':
                run_audit(view, args.disk_root, args.dry_run)
                if not args.dry_run:
                    manifest = finalize_gt(manifest)
                    _write_json(manifest_path, manifest)
    print(json.dumps({
        'phase': args.phase,
        'status': manifest['status'],
        'fresh_holdout_reference_count': (
            manifest['fresh_holdout_reference_count']),
        'evaluation_manifest': str(manifest_path),
        'disk_root': str(args.disk_root),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
