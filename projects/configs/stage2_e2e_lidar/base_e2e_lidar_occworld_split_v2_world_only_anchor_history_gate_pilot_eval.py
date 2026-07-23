_base_ = [
    './base_e2e_lidar_occworld_split_v2_world_only_anchor_change_gate_pilot_eval.py'
]

occworld_history_root = (
    'outputs/patent_2026_occ/occworld_history_expanded70')

model = dict(
    occ_head=dict(
        world_history_count=5))

data = dict(
    val=dict(
        occworld_history_root=occworld_history_root,
        occworld_history_count=5),
    test=dict(
        occworld_history_root=occworld_history_root,
        occworld_history_count=5))
