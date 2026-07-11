_base_ = ['../base_e2e_lidar_plan_mapfuse_v3_b2_postattn_mapfusion_train.py']

# B2 ablation: load the B2 checkpoint, disable only post-attention map
# residual fusion, and retain the A-route learned lane-anchor path.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_position='post_bev',
        map_fusion_mode='force_residual',
        map_force_scale=0.0,
        lane_anchor_mode='learned_blend',
    ))

load_from = None
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v3_b2_postattn_mapoff_eval')
