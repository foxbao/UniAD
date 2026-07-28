#!/usr/bin/env python
"""Freeze the validation-selected B23 inference-only candidate."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


BASE_FREEZE = Path(
    'documents/patent_2026_occ/'
    'kl_occworld_b17a_epoch3_candidate_freeze_v1.json')
VALIDATION_REPORT = Path(
    'documents/patent_2026_occ/'
    'kl_occworld_b23_raw_free_actor_arrival_validation_v1.json')
PARITY_REPORT = Path(
    'documents/patent_2026_occ/'
    'kl_occworld_b23_model_overlay_parity_v1.json')
OVERLAY_CONFIG = Path(
    'projects/configs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b23_raw_free_actor_arrival_eval.py')
HOLDOUT_MANIFEST = Path(
    'documents/patent_2026_occ/'
    'kl_occworld_b23_fresh_holdout_remaining_val31_v1.json')
HOLDOUT_PREFLIGHT = Path(
    'outputs/patent_2026_occ/'
    'occworld_b23_fresh_holdout_selection_remaining_val31_v1/preflight.json')
OUTPUT = Path(
    'documents/patent_2026_occ/'
    'kl_occworld_b23_candidate_freeze_v1.json')
IMPLEMENTATION_COMMIT = '4b269685c428d78061dc66f2a658accecad27f31'


def _read_json(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict:
    return {'path': str(path), 'sha256': _sha256(path)}


def build_freeze() -> dict:
    base = _read_json(BASE_FREEZE)
    validation = _read_json(VALIDATION_REPORT)
    parity = _read_json(PARITY_REPORT)
    holdout = _read_json(HOLDOUT_MANIFEST)
    preflight = _read_json(HOLDOUT_PREFLIGHT)
    if validation.get('qualification', {}).get('qualified') is not True:
        raise ValueError('B23 validation qualification is not true')
    if parity.get('passed') is not True:
        raise ValueError('B23 model/oracle parity did not pass')
    if int(parity.get('mask_difference_voxels', -1)) != 0:
        raise ValueError('B23 actor mask parity is not exact')
    if int(parity.get('prediction_difference_voxels', -1)) != 0:
        raise ValueError('B23 prediction parity is not exact')
    records = holdout.get('splits', {}).get('fresh_holdout', [])
    if len(records) != 30 or preflight.get('selected_scene_count') != 30:
        raise ValueError('B23 fresh holdout is not the frozen 30 scenes')

    freeze = copy.deepcopy(base)
    freeze['schema_version'] = 'kl_occworld_b23_candidate_freeze_v1'
    freeze['candidate_name'] = 'B23_raw_free_motion_actor_arrival_overlay'
    freeze['scope'] = (
        'Inference-only raw-free motion-actor arrival overlay on the frozen '
        'B17A epoch-3 world model; awaiting one fresh-scene evaluation.')
    freeze['base_model_candidate'] = _artifact(BASE_FREEZE)
    freeze['inference_only_overlay_selection'] = {
        'implementation_code_commit': IMPLEMENTATION_COMMIT,
        'validation_report': _artifact(VALIDATION_REPORT),
        'model_oracle_parity_report': _artifact(PARITY_REPORT),
        'evaluation_config_path': str(OVERLAY_CONFIG),
        'evaluation_config_sha256': _sha256(OVERLAY_CONFIG),
        'score_threshold': 0.1,
        'raw_class_gate': 0,
        'arrival_contract': (
            'motion_support AND NOT stationary_support AND raw==free'),
        'departure_update_enabled': False,
        'trainable_parameter_count_added': 0,
        'all_predeclared_validation_checks_passed': True,
        'mask_difference_voxels': 0,
        'prediction_difference_voxels': 0,
    }
    protocol = freeze['frozen_inference_protocol']
    protocol['evaluation_config'] = _artifact(OVERLAY_CONFIG)
    protocol['model_local_overlay_threshold'] = None
    protocol['motion_actor_arrival_overlay'] = True
    protocol['motion_actor_score_threshold'] = 0.1
    protocol['motion_actor_raw_class_gate'] = 0
    protocol['motion_actor_departure_update'] = False
    protocol['required_prediction_artifacts'] = {
        'motion_actor_arrival_mask_3d': [4, 10, 120, 160],
    }
    freeze['fresh_holdout_selection'] = {
        'source_annotation': holdout['annotation_file'],
        'source_annotation_sha256': holdout['source_annotation_sha256'],
        'annotation_scene_count': 65,
        'complete_contract_scene_count': 61,
        'previously_consumed_b17_scene_count': 30,
        'remaining_complete_contract_scene_count': 31,
        'selected_scene_count': 30,
        'reserved_unused_reference_index': 3000,
        'selection_uses_occworld_labels': False,
        'selection_uses_model_predictions': False,
        'manifest': _artifact(HOLDOUT_MANIFEST),
        'preflight': _artifact(HOLDOUT_PREFLIGHT),
    }
    freeze['holdout_rules'] = {
        'B15_final_holdout_may_be_reused': False,
        'B17_fresh_holdout_may_be_reused': False,
        'consumed_test_or_blind_may_be_reused': False,
        'threshold_retuning_on_fresh_holdout_allowed': False,
        'checkpoint_reselection_on_fresh_holdout_allowed': False,
        'single_inference_and_evaluation_only': True,
        'frozen_fresh_holdout_scene_count': 30,
        'reserved_scene_may_be_used_after_evaluation': False,
    }
    return freeze


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f'Refusing to replace frozen candidate: {args.output}')
    freeze = build_freeze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as output:
        json.dump(freeze, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'status': freeze['status'],
        'candidate_name': freeze['candidate_name'],
        'output': str(args.output),
        'sha256': _sha256(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
