_base_ = ['../base_e2e_lidar_plan_mapfuse_v2.py']

# v3 local lane-anchor residual eval: add a local forward lane segment to the
# learned plan using the base planner's own displacement profile.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        lane_anchor_mode='residual',
        lane_anchor_scale=1.0,
        lane_anchor_offset_scale=0.0,
        lane_anchor_sample_mode='local_forward',
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_local_residual_eval'
