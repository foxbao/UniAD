_base_ = ['./base_e2e_lidar_occworld_split_v2_world_only_anchor_eval.py']

model = dict(
    occ_head=dict(
        world_use_future_change_gate=True,
        world_future_change_prior=0.01,
        world_future_change_gate_loss_weight=0.1,
        world_future_changed_class_loss_weight=0.1,
        world_future_change_positive_weight=10.0))
