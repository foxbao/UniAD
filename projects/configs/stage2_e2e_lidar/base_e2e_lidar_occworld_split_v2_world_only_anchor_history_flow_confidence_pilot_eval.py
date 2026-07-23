_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_aligned10_eval.py'
]

model = dict(
    occ_head=dict(
        world_use_physical_confidence=True,
        world_physical_confidence_prior=0.8,
        world_physical_confidence_flow_threshold=0.5))
