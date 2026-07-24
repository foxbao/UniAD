_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'
]

# Validation-only deployment-input view. Both the current anchor and all five
# aligned history frames come from sequential TrackFormer predictions. This
# config must not be pointed at final holdout or used to select new thresholds.
predicted_validation_input_root = (
    'outputs/patent_2026_occ/'
    'occworld_online_inputs_full_predicted_history_validation_v1')

data = dict(
    val=dict(
        occworld_online_input_root=predicted_validation_input_root),
    test=dict(
        occworld_online_input_root=predicted_validation_input_root))
