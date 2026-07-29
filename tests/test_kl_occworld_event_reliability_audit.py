import numpy as np
import pytest

from tools.analysis_tools.audit_kl_occworld_event_reliability import audit


def _write_label(path):
    state = np.ones((5, 1, 1, 4), dtype=np.uint8)
    state[1, 0, 0] = [1, 3, 1, 3]
    valid = np.ones_like(state, dtype=np.bool_)
    current_state = np.ones((1, 1, 4), dtype=np.uint8)
    current_valid = np.ones_like(current_state, dtype=np.bool_)
    np.savez_compressed(
        path,
        world_target_state_3d=state,
        world_target_valid_3d=valid,
        current_observation_state_3d=current_state,
        current_observation_valid_3d=current_valid)
    return path


def _write_prediction(path):
    raw = np.zeros((5, 1, 1, 4), dtype=np.uint8)
    final = raw.copy()
    final[1, 0, 0] = [0, 2, 2, 1]
    candidate = np.zeros((4, 1, 1, 4), dtype=np.uint8)
    candidate[0, 0, 0] = [0, 2, 2, 0]
    mask = np.zeros_like(candidate, dtype=np.bool_)
    mask[0, 0, 0, :3] = True
    alpha = np.zeros_like(candidate, dtype=np.float32)
    alpha[0, 0, 0, :3] = [0.1, 0.9, 0.8]
    np.savez_compressed(
        path,
        reference_index=np.int64(7),
        checkpoint_epoch=np.int64(2),
        world_pred_class_3d=final,
        raw_world_pred_class_3d=raw,
        event_reliability_alpha_3d=alpha,
        event_candidate_class_3d=candidate,
        event_candidate_mask_3d=mask)
    return path


def test_event_reliability_audit_reports_local_corrections(tmp_path):
    label = _write_label(tmp_path / 'label.npz')
    prediction = _write_prediction(tmp_path / 'prediction.npz')
    manifest = {
        'splits': {'internal_dev': [{
            'reference_index': 7, 'scene_token': 'scene-a'}]},
    }

    summary = audit(
        manifest, {7: label}, {7: prediction}, 'internal_dev', 2)

    assert summary['current_difference_voxels'] == 0
    assert summary['non_event_difference_voxels'] == 1
    assert summary['candidate_voxels'] == 3
    assert summary['corrections'] == {
        'changed_voxels': 3,
        'improved_voxels': 1,
        'harmed_voxels': 1,
        'still_wrong_voxels': 1,
        'unknown_target_voxels': 0,
        'net_correct_voxels': 0,
    }
    assert summary['alpha_distribution']['raw_should_win']['mean'] == (
        pytest.approx(0.8))
    assert summary['scene_outcome_counts'] == {
        'positive': 0, 'negative': 0, 'zero': 1}
