_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_change_gate_pilot.py'
]

occworld_history_root = (
    'outputs/patent_2026_occ/occworld_history_expanded70')

# B7 fuses the five aligned causal Occ volumes [-2s, -1.5s, -1s, -0.5s, 0s]
# with the existing BEV feature before predicting future world changes.
model = dict(
    occ_head=dict(
        world_history_count=5))

data = dict(
    train=dict(
        occworld_history_root=occworld_history_root,
        occworld_history_count=5))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_history_gate_pilot')
