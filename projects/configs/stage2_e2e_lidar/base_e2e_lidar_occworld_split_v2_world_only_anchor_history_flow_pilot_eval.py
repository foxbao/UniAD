_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_gate_pilot_eval.py'
]

model = dict(
    occ_head=dict(
        world_use_flow_warp=True,
        world_flow_loss_weight=1.0,
        world_flow_gate_logit_scale=10.0))
