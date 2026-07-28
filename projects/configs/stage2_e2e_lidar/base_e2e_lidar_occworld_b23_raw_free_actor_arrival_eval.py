_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_raw_eval.py'
]

# B23 is an inference-only raw-preserving actor-arrival overlay. It does not
# add trainable parameters and deliberately replaces, rather than combines
# with, the older 2D local-flow overlay for the paired validation test.
model = dict(
    occ_head=dict(
        world_use_motion_actor_arrival_overlay=True,
        world_motion_actor_score_threshold=0.1,
        world_motion_actor_raw_class_gate=0,
        world_motion_actor_pc_range=[-64.0, -48.0, -2.0,
                                     64.0, 48.0, 6.0]))
