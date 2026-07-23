_base_ = ['./base_e2e_lidar_occworld_split_v1_world_only_balanced.py']

# B2 adds only the causal current-observation anchor. The observation comes
# from current LiDAR raycasting and is available at deployment time.
model = dict(
    occ_head=dict(
        world_use_observation_anchor=True,
        world_observation_semantic_logit_scale=2.0,
        world_observation_valid_logit_scale=4.0))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v1_world_only_anchor')
