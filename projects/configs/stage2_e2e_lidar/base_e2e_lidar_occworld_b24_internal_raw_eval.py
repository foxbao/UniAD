_base_ = [
    './base_e2e_lidar_occworld_b24_event_reliability_exploratory_eval.py'
]

# Exact B17A semantic baseline on the frozen B24 internal-dev references.
# This is an exploratory paired ablation, not an independent holdout result.
model = dict(
    occ_head=dict(
        world_use_event_reliability=False,
        world_event_reliability_loss_weight=0.0))
