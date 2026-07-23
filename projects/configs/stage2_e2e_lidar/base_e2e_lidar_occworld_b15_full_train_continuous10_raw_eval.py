_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'
]

# Checkpoint selection uses the unfused semantic output. The deployed B15
# evaluation config keeps the previously frozen B14 local-overlay threshold.
model = dict(
    occ_head=dict(
        world_local_flow_overlay_threshold=None))
