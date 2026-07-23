_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_pilot_eval.py'
]

model = dict(
    occ_head=dict(
        world_align_bev_to_world_layout=True))
