_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_aligned10_eval.py'
]

model = dict(
    occ_head=dict(
        world_flow_parameterization='cumulative_current',
        world_use_wide_history_context=False))
