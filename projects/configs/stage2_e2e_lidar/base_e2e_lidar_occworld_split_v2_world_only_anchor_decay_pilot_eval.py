_base_ = ['./base_e2e_lidar_occworld_split_v2_world_only_anchor_eval.py']

model = dict(
    occ_head=dict(
        world_observation_future_semantic_logit_scales=[
            1.5, 1.0, 0.75, 0.5,
        ]))
