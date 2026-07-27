#!/usr/bin/env python
"""Prepare, run and seal the one-shot B15 final-holdout evaluation."""

import argparse
import copy
import hashlib
import json
import sys
from datetime import date
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


SOURCE_MANIFEST = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_full_train3_final30_manifest_v1.json')
SELECTION_REPORT = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_b15_full_train_continuous10_validation_selection_v1.json')
EVALUATION_MANIFEST = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_b15_final_holdout30_evaluation_v1.json')
EVAL_CONFIG = REPO_ROOT / (
    'projects/configs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b15_final_holdout_eval.py')
DISK_ROOT = Path(
    '/mnt/disk1/baojiali/UniAD_occworld_b15_final_holdout30_v1')
PREDICTION_ROOT = REPO_ROOT / (
    'outputs/patent_2026_occ/'
    'occworld_predictions_b15_epoch7_final_holdout30_v1/'
    'final_holdout/epoch_007')
MODEL_REPORT = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_b15_epoch7_final_holdout30_v1.json')
PERSISTENCE_REPORT = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_b15_persistence_final_holdout30_v1.json')


def _read_json(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _record_path(path: Path) -> str:
    """Prefer repository-relative paths, retaining external disk locations."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _root_mapping() -> dict:
    names = {
        'temporal': 'occworld_temporal_b15_final_holdout30_v1',
        'occlusion': 'occworld_occlusion_b15_final_holdout30_v1',
        'dual': 'occworld_dual_b15_final_holdout30_v1',
        'dual_cross_scene': (
            'occworld_dual_cross_scene_b15_final_holdout30_v1'),
        'observation_cache': (
            'occworld_observation_cache_b15_final_holdout30_v1'),
        'sequence': 'occworld_sequence_b15_final_holdout30_v1',
        'history': 'occworld_history_b15_final_holdout30_v1',
        'sequence_audit': (
            'occworld_sequence_b15_final_holdout30_v1_audit'),
        'history_audit': (
            'occworld_history_b15_final_holdout30_v1_audit'),
    }
    return {
        key: f'outputs/patent_2026_occ/{name}'
        for key, name in names.items()
    }


def prepare_manifest(source_path: Path, selection_path: Path,
                     config_path: Path) -> dict:
    source = _read_json(source_path)
    selection = _read_json(selection_path)
    protocol = source.get('final_holdout_protocol', {})
    if source.get('status') != 'ready_after_full_generation_and_audit':
        raise ValueError('B15 source manifest is not ready')
    if not protocol.get('single_final_evaluation_only'):
        raise ValueError('Source manifest does not require one-shot evaluation')
    if protocol.get('threshold_retuning_allowed'):
        raise ValueError('Final holdout unexpectedly permits threshold tuning')
    records = [dict(row) for row in source['splits']['final_holdout']]
    if len(records) != 30:
        raise ValueError(f'Expected 30 final scenes, got {len(records)}')
    references = [int(row['reference_index']) for row in records]
    scenes = [str(row['scene_token']) for row in records]
    if len(set(references)) != len(records):
        raise ValueError('Final holdout has duplicate references')
    if len(set(scenes)) != len(records):
        raise ValueError('Final holdout has duplicate scenes')
    other_scenes = {
        str(row['scene_token'])
        for key, rows in source['splits'].items()
        if key != 'final_holdout' for row in rows
    }
    if other_scenes.intersection(scenes):
        raise ValueError('Final holdout overlaps another split')
    if selection.get('selection_split') != 'validation':
        raise ValueError('Checkpoint was not selected on validation')
    if int(selection.get('selected_epoch', -1)) != 7:
        raise ValueError('Frozen B15 checkpoint is not epoch 7')
    visibility = float(selection['selected_visibility_threshold'])
    overlay = selection.get('selected_local_flow_overlay_row')
    if visibility != 0.7 or overlay is None:
        raise ValueError('Frozen visibility/overlay selection is incomplete')
    overlay_threshold = float(overlay['flow_fusion_threshold'])
    if overlay_threshold != 0.9:
        raise ValueError('Frozen local-overlay threshold is not 0.9')
    return {
        'schema_version': 1,
        'name': 'kl_occworld_b15_final_holdout30_evaluation_v1',
        'status': 'frozen_before_final_holdout_gt_generation',
        'annotation_file': source['annotation_file'],
        'source_manifest': str(source_path.relative_to(REPO_ROOT)),
        'source_manifest_sha256': _sha256(source_path),
        'validation_selection_report': str(
            selection_path.relative_to(REPO_ROOT)),
        'validation_selection_report_sha256': _sha256(selection_path),
        'evaluation_config': str(config_path.relative_to(REPO_ROOT)),
        'evaluation_config_sha256': _sha256(config_path),
        'final_holdout_reference_count': len(records),
        'split_scene_tokens': {'final_holdout': scenes},
        'splits': {'final_holdout': records},
        'roots': _root_mapping(),
        'cross_scene_support': {
            'source_split': 'train',
            'dual_dir': source['roots']['dual'],
            'reference_indices': [
                int(row['reference_index'])
                for row in source['splits']['train']
            ],
            'target_holdout_used_as_mutual_support': False,
        },
        'frozen_model_protocol': {
            'checkpoint_epoch': 7,
            'visibility_threshold': visibility,
            'local_flow_overlay_threshold': overlay_threshold,
            'checkpoint_selection_metric': (
                selection['checkpoint_selection_metric']),
            'threshold_retuning_on_final_holdout': False,
            'single_model_inference_only': True,
            'deterministic_baseline': 'constant_current_persistence',
        },
        'model_predictions_inspected': False,
        'final_holdout_metrics_inspected': False,
    }


def _load_evaluation_manifest(path: Path) -> dict:
    manifest = _read_json(path)
    allowed = {
        'frozen_before_final_holdout_gt_generation',
        'ready_for_single_final_holdout_inference',
        'final_holdout_evaluated_once_no_retuning_allowed',
    }
    if manifest.get('status') not in allowed:
        raise ValueError(
            f'Unexpected evaluation status: {manifest.get("status")}')
    return manifest


def _generation_view(manifest: dict) -> dict:
    result = copy.deepcopy(manifest)
    result['status'] = 'frozen_before_full_gt_generation'
    result['splits'] = {'train': result['splits']['final_holdout']}
    result['train_reference_count'] = int(
        result['final_holdout_reference_count'])
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
        for row in manifest['splits']['final_holdout']
    }
    sequence_root = REPO_ROOT / manifest['roots']['sequence']
    history_root = REPO_ROOT / manifest['roots']['history']
    sequences = _artifact_mapping(sequence_root, 'occworld_sequence')
    histories = _artifact_mapping(history_root, 'occworld_history')
    if set(sequences) != expected or set(histories) != expected:
        raise ValueError('Final holdout artifact references are not exact')
    max_future_time_error = 0.0
    max_history_time_error = 0.0
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
            target_times = np.asarray(
                sequence['target_times_s'], dtype=np.float64)
            max_future_time_error = max(
                max_future_time_error,
                float(np.max(np.abs(
                    target_times - np.arange(5) * 0.5))))
            valid_ratios.append(float(valid.mean()))
        with np.load(histories[reference], allow_pickle=False) as history:
            history_state = history['history_observation_state_3d']
            if history_state.shape != (5, 10, 120, 160):
                raise ValueError(f'Bad history shape for {reference}')
            np.testing.assert_array_equal(history_state[-1], direct_t0)
            history_times = np.asarray(
                history['history_times_s'], dtype=np.float64)
            max_history_time_error = max(
                max_history_time_error,
                float(np.max(np.abs(
                    history_times - np.arange(-4, 1) * 0.5))))
            transforms = np.asarray(history['history_to_reference'])
            max_last_transform_error = max(
                max_last_transform_error,
                float(np.max(np.abs(transforms[-1] - np.eye(4)))))
    result = copy.deepcopy(manifest)
    result['status'] = 'ready_for_single_final_holdout_inference'
    result['final_holdout_generation_audit'] = {
        'sequence_count': len(sequences),
        'history_count': len(histories),
        'reference_sets_exactly_equal': True,
        'sequence_shape': [5, 10, 120, 160],
        'history_shape': [5, 10, 120, 160],
        'history_last_equals_direct_t0': True,
        'max_future_time_error_s': max_future_time_error,
        'max_history_time_error_s': max_history_time_error,
        'max_last_transform_error': max_last_transform_error,
        'mean_world_valid_ratio': float(np.mean(valid_ratios)),
    }
    return result


def _future_mean(report: dict) -> float:
    return float(np.mean([
        row['mean_iou'] for row in
        report['semantic']['by_horizon'][1:]
    ]))


def mark_evaluated(manifest: dict, prediction_root: Path,
                   model_report_path: Path,
                   persistence_report_path: Path) -> dict:
    if manifest.get('status') != 'ready_for_single_final_holdout_inference':
        raise ValueError('Final holdout is not ready for its single inference')
    if _sha256(REPO_ROOT / manifest['evaluation_config']) != (
            manifest['evaluation_config_sha256']):
        raise ValueError('Final evaluation config changed after freezing')
    expected = [
        int(row['reference_index'])
        for row in manifest['splits']['final_holdout']
    ]
    prediction_epochs = {}
    for path in sorted(prediction_root.glob('*/*__occworld_prediction.npz')):
        with np.load(path, allow_pickle=False) as payload:
            prediction_epochs[int(payload['reference_index'])] = int(
                payload['checkpoint_epoch'])
    if sorted(prediction_epochs) != sorted(expected):
        raise ValueError('Final predictions do not match frozen references')
    if set(prediction_epochs.values()) != {7}:
        raise ValueError('Final predictions are not all from epoch 7')
    model_report = _read_json(model_report_path)
    persistence_report = _read_json(persistence_report_path)
    for name, report in (
            ('model', model_report), ('persistence', persistence_report)):
        if report.get('split') != 'final_holdout':
            raise ValueError(f'{name} report is not final_holdout')
        if report.get('reference_indices') != expected:
            raise ValueError(f'{name} report reference order changed')
        if float(report['visibility_threshold']) != 0.7:
            raise ValueError(f'{name} report changed visibility threshold')
    if model_report.get('prediction_source') != 'exported_model_prediction':
        raise ValueError('Model report did not use exported predictions')
    if persistence_report.get('prediction_source') != (
            'constant_current_persistence'):
        raise ValueError('Persistence report is not the frozen baseline')
    result = copy.deepcopy(manifest)
    result['status'] = 'final_holdout_evaluated_once_no_retuning_allowed'
    result['model_predictions_inspected'] = True
    result['final_holdout_metrics_inspected'] = True
    result['final_holdout_evaluation'] = {
        'completed_on': date.today().isoformat(),
        'checkpoint_epoch': 7,
        'visibility_threshold': 0.7,
        'local_flow_overlay_threshold_in_forward_test': 0.9,
        'prediction_root': _record_path(prediction_root),
        'model_report': _record_path(model_report_path),
        'persistence_report': _record_path(persistence_report_path),
        'model_future_mean_iou': _future_mean(model_report),
        'persistence_future_mean_iou': _future_mean(persistence_report),
        'threshold_scan_performed': False,
        'additional_model_inference_allowed': False,
        'further_tuning_on_final_holdout_allowed': False,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--phase', choices=(
            'prepare', 'setup', 'base-labels', 'cross-scene',
            'sequence-history', 'audit', 'all', 'mark-evaluated'),
        default='prepare')
    parser.add_argument('--source-manifest', type=Path,
                        default=SOURCE_MANIFEST)
    parser.add_argument('--selection-report', type=Path,
                        default=SELECTION_REPORT)
    parser.add_argument('--evaluation-manifest', type=Path,
                        default=EVALUATION_MANIFEST)
    parser.add_argument('--evaluation-config', type=Path,
                        default=EVAL_CONFIG)
    parser.add_argument('--disk-root', type=Path, default=DISK_ROOT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prediction-root', type=Path,
                        default=PREDICTION_ROOT)
    parser.add_argument('--model-report', type=Path, default=MODEL_REPORT)
    parser.add_argument('--persistence-report', type=Path,
                        default=PERSISTENCE_REPORT)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.evaluation_manifest.resolve()
    if args.phase == 'prepare':
        if manifest_path.exists():
            raise FileExistsError(
                f'Refusing to replace frozen manifest: {manifest_path}')
        manifest = prepare_manifest(
            args.source_manifest.resolve(),
            args.selection_report.resolve(),
            args.evaluation_config.resolve())
        _write_json(manifest_path, manifest)
    else:
        manifest = _load_evaluation_manifest(manifest_path)
        if args.phase == 'mark-evaluated':
            manifest = mark_evaluated(
                manifest, args.prediction_root.resolve(),
                args.model_report.resolve(),
                args.persistence_report.resolve())
            _write_json(manifest_path, manifest)
        else:
            view = _generation_view(manifest)
            setup_paths(view, args.disk_root)
            phases = (
                ['base-labels', 'cross-scene',
                 'sequence-history', 'audit']
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
        'final_holdout_reference_count': (
            manifest['final_holdout_reference_count']),
        'evaluation_manifest': str(manifest_path),
        'disk_root': str(args.disk_root),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
