_base_ = ['./base_e2e_lidar_occworld_split_v2_world_only_anchor.py']

# B3 changes only the future persistence margin. All scales remain positive,
# so the untrained semantic argmax is still exactly persistence.
model = dict(
    occ_head=dict(
        world_observation_future_semantic_logit_scales=[
            1.5, 1.0, 0.75, 0.5,
        ]))

total_epochs = 5
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v2_world_only_anchor_decay_pilot')
