_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_change_gate_pilot_eval.py'
]

model = dict(
    occ_head=dict(
        world_dynamic_change_logit_scale=10.0))
