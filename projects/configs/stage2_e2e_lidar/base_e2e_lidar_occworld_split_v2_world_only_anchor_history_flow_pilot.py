_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_history_gate_pilot.py'
]

# B8 predicts four incremental [dy, dx] BEV flows, forward-splats the causal
# current instance volume, and feeds physical motion changes to the gate.
model = dict(
    occ_head=dict(
        world_use_flow_warp=True,
        world_flow_loss_weight=1.0,
        world_flow_gate_logit_scale=10.0))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_history_flow_pilot')
