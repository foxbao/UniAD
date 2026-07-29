import numpy as np

from tools.analysis_tools.fix_kl_occworld_event_non_event_predictions import (
    fix_export,
)


def _write(path, changed):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.zeros((2, 1, 1, 3), dtype=np.uint8)
    final = raw.copy()
    final[0, 0, 0, 0] = int(changed)
    final[1, 0, 0] = [2, 2, 2]
    event = np.asarray([[[[True, False, True]]]])
    np.savez_compressed(
        path,
        reference_index=np.int64(7),
        world_pred_class_3d=final,
        raw_world_pred_class_3d=raw,
        event_candidate_mask_3d=event)
    return path


def test_event_non_event_fix_rewrites_only_spatial_invariant_violations(
        tmp_path):
    input_root = tmp_path / 'input'
    changed = _write(
        input_root / 'a' / 'changed__occworld_prediction.npz', True)
    unchanged = _write(
        input_root / 'b' / 'unchanged__occworld_prediction.npz', False)
    with np.load(unchanged, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    payload['world_pred_class_3d'][1, 0, 0, 1] = 0
    np.savez_compressed(unchanged, **payload)
    output_root = tmp_path / 'output'

    summary = fix_export(input_root, output_root)

    assert summary['prediction_file_count'] == 2
    assert summary['rewritten_file_count'] == 1
    assert summary['corrected_non_event_voxels'] == 2
    assert (output_root / 'b' / unchanged.name).stat().st_ino == (
        unchanged.stat().st_ino)
    with np.load(
            output_root / 'a' / changed.name,
            allow_pickle=False) as archive:
        final = archive['world_pred_class_3d']
        assert final[0, 0, 0, 0] == 0
        assert final[1, 0, 0].tolist() == [2, 0, 2]
