import json

import numpy as np

from tools.analysis_tools.audit_kl_occworld_prediction_export import (
    audit_export,
)


def _manifest(path):
    path.write_text(json.dumps({
        'name': 'toy',
        'splits': {'internal_dev': [
            {'reference_index': 3}, {'reference_index': 7},
        ]},
    }), encoding='utf-8')
    return path


def _prediction(path, reference_index, prediction, raw=None, epoch=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        reference_index=np.int64(reference_index),
        checkpoint_epoch=np.int64(epoch),
        world_pred_class_3d=np.asarray(prediction, dtype=np.uint8))
    if raw is not None:
        payload['raw_world_pred_class_3d'] = np.asarray(raw, dtype=np.uint8)
    np.savez_compressed(path, **payload)


def test_prediction_export_audit_accepts_complete_raw_baseline(tmp_path):
    manifest = _manifest(tmp_path / 'manifest.json')
    root = tmp_path / 'predictions'
    prediction = np.zeros((2, 1, 2, 3), dtype=np.uint8)
    _prediction(root / '000003' / 'a__occworld_prediction.npz', 3,
                prediction, raw=prediction)
    _prediction(root / '000007' / 'b__occworld_prediction.npz', 7,
                prediction, raw=prediction)

    summary = audit_export(
        manifest, 'internal_dev', root, expected_checkpoint_epoch=3,
        require_raw_equals_prediction=True)

    assert summary['passed']
    assert summary['prediction_shapes'] == [[2, 1, 2, 3]]


def test_prediction_export_audit_reports_coverage_and_raw_mismatch(tmp_path):
    manifest = _manifest(tmp_path / 'manifest.json')
    root = tmp_path / 'predictions'
    prediction = np.zeros((2, 1, 2, 3), dtype=np.uint8)
    raw = prediction.copy()
    raw[0, 0, 0, 0] = 1
    _prediction(root / '000003' / 'a__occworld_prediction.npz', 3,
                prediction, raw=raw, epoch=2)

    summary = audit_export(
        manifest, 'internal_dev', root, expected_checkpoint_epoch=3,
        require_raw_equals_prediction=True)

    assert not summary['passed']
    assert summary['missing_reference_indices'] == [7]
    assert summary['raw_mismatch_reference_indices'] == [3]
    assert summary['wrong_checkpoint_epochs'][0]['epoch'] == 2
