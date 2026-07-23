_base_ = ['./base_e2e_lidar_occworld_split_v2_world_only_anchor_eval.py']

model = dict(
    occ_head=dict(
        world_future_transition_loss_weight=0.1))
