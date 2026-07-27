_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'
]

# Evaluation keeps B17A's frozen inference graph and thresholds. The auxiliary
# weight is recorded for traceability but is inactive during forward_test.
model = dict(
    occ_head=dict(
        world_future_stability_loss_weight=0.1))
