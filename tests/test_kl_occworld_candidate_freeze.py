import hashlib
import json

from tools.analysis_tools.verify_kl_occworld_candidate_freeze import (
    verify_artifact_hashes,
    verify_selection_metrics,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_freeze_verifies_hashed_artifacts(tmp_path):
    artifact = tmp_path / 'candidate.pth'
    artifact.write_bytes(b'occworld-candidate')
    freeze = {
        'selected_checkpoint': {
            'path': str(artifact),
            'sha256': _sha256(artifact),
        },
    }

    verified = verify_artifact_hashes(freeze)

    assert len(verified) == 1
    assert verified[0]['path'] == str(artifact)


def test_candidate_freeze_rejects_changed_artifact(tmp_path):
    artifact = tmp_path / 'candidate.pth'
    artifact.write_bytes(b'original')
    freeze = {
        'selected_checkpoint': {
            'path': str(artifact),
            'sha256': _sha256(artifact),
        },
    }
    artifact.write_bytes(b'changed')

    try:
        verify_artifact_hashes(freeze)
    except ValueError as error:
        assert 'hash mismatch' in str(error)
    else:
        raise AssertionError('Expected changed artifact to be rejected')


def test_candidate_freeze_metrics_must_match_selection_report(tmp_path):
    report_path = tmp_path / 'selection.json'
    metrics = {
        'overall_miou': 0.8,
        'future_mean_miou': 0.7,
    }
    report_path.write_text(json.dumps({
        'selected_model': 'B17A_epoch3',
        'full_predicted_history_rows': [{
            'model': 'B17A_epoch3',
            'epoch': 3,
            **metrics,
        }],
    }))
    freeze = {
        'selected_checkpoint': {'checkpoint_meta': {'epoch': 3}},
        'validation_selection': {
            'selected_model': 'B17A_epoch3',
            'selection_report': {'path': str(report_path)},
            'full_predicted_history_metrics': metrics,
        },
    }

    row = verify_selection_metrics(freeze)

    assert row['future_mean_miou'] == 0.7
