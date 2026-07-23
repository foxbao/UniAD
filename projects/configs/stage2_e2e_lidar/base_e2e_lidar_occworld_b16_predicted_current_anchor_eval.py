_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'
]

# Validation-only counterpart of the B16 current-anchor pilot. The fixed B15
# validation anchors were generated from TrackFormer predictions; history is
# intentionally unchanged so train and validation test the same substitution.
predicted_validation_anchor_root = (
    'outputs/patent_2026_occ/occworld_predicted_current_box_anchors_v1')

data = dict(
    val=dict(
        occworld_current_anchor_root=predicted_validation_anchor_root),
    test=dict(
        occworld_current_anchor_root=predicted_validation_anchor_root))
