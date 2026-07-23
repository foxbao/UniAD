_base_ = ['./base_e2e_lidar_occworld_split_v2_eval.py']

model = dict(
    occ_head=dict(
        world_class_weights=[
            0.382372,
            1.278013,
            1.339615,
        ],
        world_use_observation_anchor=True,
        world_observation_semantic_logit_scale=2.0,
        world_observation_valid_logit_scale=4.0))
