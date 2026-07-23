_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_pilot_eval.py'
]

# Frozen on validation: epoch 3, physical flow threshold 0.7 and visibility
# threshold 0.5. This config is only used for the one-time test export.
model = dict(
    occ_head=dict(
        world_physical_flow_fusion_threshold=0.7))
