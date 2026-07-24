#!/usr/bin/env python
"""Verify a frozen OccWorld candidate before fresh-holdout inference."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping


EXPECTED_STATUS = (
    'validation_selected_candidate_frozen_awaiting_fresh_scene_holdout')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _artifact_pairs(value) -> list:
    pairs = []
    if isinstance(value, dict):
        if (isinstance(value.get('path'), str)
                and isinstance(value.get('sha256'), str)):
            pairs.append((Path(value['path']), value['sha256']))
        for nested in value.values():
            pairs.extend(_artifact_pairs(nested))
    elif isinstance(value, list):
        for nested in value:
            pairs.extend(_artifact_pairs(nested))
    return pairs


def verify_artifact_hashes(freeze: Mapping) -> list:
    """Verify every nested ``path``/``sha256`` pair."""
    pairs = _artifact_pairs(freeze)
    if not pairs:
        raise ValueError('Candidate freeze contains no hashed artifacts')
    verified = []
    seen = set()
    for path, expected in pairs:
        key = str(path)
        if key in seen:
            raise ValueError(f'Duplicate hashed artifact path: {path}')
        seen.add(key)
        if not path.is_file():
            raise FileNotFoundError(f'Frozen artifact is missing: {path}')
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f'Frozen artifact hash mismatch for {path}: '
                f'expected={expected}, actual={actual}')
        verified.append({
            'path': key,
            'sha256': actual,
            'size_bytes': path.stat().st_size,
        })
    return verified


def verify_selection_metrics(freeze: Mapping) -> dict:
    """Require the frozen metric row to equal the source selection row."""
    selection = freeze.get('validation_selection', {})
    report_path = Path(selection.get(
        'selection_report', {}).get('path', ''))
    selected_model = selection.get('selected_model')
    metrics = selection.get('full_predicted_history_metrics')
    if not selected_model or not isinstance(metrics, dict):
        raise ValueError('Candidate freeze has incomplete selection fields')
    with report_path.open() as source:
        report = json.load(source)
    if report.get('selected_model') != selected_model:
        raise ValueError(
            f'Selection report chose {report.get("selected_model")}, '
            f'freeze chose {selected_model}')
    matches = [
        row for row in report.get('full_predicted_history_rows', [])
        if row.get('model') == selected_model
    ]
    if len(matches) != 1:
        raise ValueError(
            f'Expected one selection row for {selected_model}, '
            f'found {len(matches)}')
    expected = {
        'model': selected_model,
        'epoch': freeze['selected_checkpoint']['checkpoint_meta']['epoch'],
        **metrics,
    }
    if matches[0] != expected:
        raise ValueError('Frozen metrics differ from the selection report')
    return matches[0]


def verify_candidate_freeze(freeze_path: Path,
                            check_git_commit: bool = True) -> dict:
    """Run all candidate integrity checks and return a summary."""
    with freeze_path.open() as source:
        freeze = json.load(source)
    if freeze.get('status') != EXPECTED_STATUS:
        raise ValueError(
            f'Candidate is not frozen for a fresh holdout: '
            f'{freeze.get("status")}')
    checkpoint = freeze.get('selected_checkpoint', {})
    checkpoint_path = Path(checkpoint.get('path', ''))
    resolved_expected = checkpoint.get('resolved_path')
    if not resolved_expected:
        raise ValueError('Frozen checkpoint has no resolved_path')
    if str(checkpoint_path.resolve()) != resolved_expected:
        raise ValueError(
            f'Checkpoint resolves to {checkpoint_path.resolve()}, '
            f'expected {resolved_expected}')
    artifacts = verify_artifact_hashes(freeze)
    selection_row = verify_selection_metrics(freeze)
    training_commit = freeze.get('training', {}).get(
        'training_code_commit')
    if check_git_commit:
        if not training_commit:
            raise ValueError('Candidate freeze has no training commit')
        subprocess.run(
            ['git', 'cat-file', '-e', f'{training_commit}^{{commit}}'],
            check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
    return {
        'status': 'candidate_freeze_verified',
        'candidate_name': freeze.get('candidate_name'),
        'freeze_path': str(freeze_path),
        'freeze_sha256': _sha256(freeze_path),
        'verified_artifact_count': len(artifacts),
        'selected_checkpoint_path': str(checkpoint_path),
        'selected_checkpoint_sha256': checkpoint.get('sha256'),
        'training_code_commit': training_commit,
        'selected_model': selection_row['model'],
        'selected_future_mean_miou': selection_row['future_mean_miou'],
        'visibility_threshold': freeze[
            'frozen_inference_protocol']['visibility_threshold'],
        'model_local_overlay_threshold': freeze[
            'frozen_inference_protocol']['model_local_overlay_threshold'],
        'evaluator_post_overlay': freeze[
            'frozen_inference_protocol']['evaluator_post_overlay'],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--freeze-file', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b17a_epoch3_candidate_freeze_v1.json'))
    parser.add_argument('--skip-git-commit-check', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    summary = verify_candidate_freeze(
        args.freeze_file,
        check_git_commit=not args.skip_git_commit_check)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
