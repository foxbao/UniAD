_base_ = ['../base_e2e_lidar_plan_mapfuse_v3_b1_mapfusion_train.py']

# B1 ablation: disable only latent map-query fusion while retaining the
# checkpoint's learned lane-anchor gate and residual correction.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='force_residual',
        map_force_scale=0.0,
        lane_anchor_mode='learned_blend',
    ))

load_from = None
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v3_b1_mapoff_eval')
