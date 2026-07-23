_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_change_gate_pilot.py'
]

# B6 reuses UniAD's frozen query-based future instance occupancy. Temporal
# probability differences provide a spatial prior to the learned 3D gate.
model = dict(
    occ_head=dict(
        world_dynamic_change_logit_scale=10.0))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_motion_gate_pilot')
